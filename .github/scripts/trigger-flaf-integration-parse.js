module.exports = async ({ github, context, core, process, require }) => {
  function parseGitHubVersionUrl(entry) {
    const normalizedEntry = entry.trim();
    const pullMatch = normalizedEntry.match(/^[-*]\s+https:\/\/github\.com\/[^/]+\/([^/]+)\/pull\/(\d+)\/?$/);
    if (pullMatch) {
      return {
        repo: pullMatch[1],
        version: `PR_${pullMatch[2]}`,
      };
    }

    const treeMatch = normalizedEntry.match(/^[-*]\s+https:\/\/github\.com\/[^/]+\/([^/]+)\/tree\/(.+?)\/?$/);
    if (treeMatch) {
      return {
        repo: treeMatch[1],
        version: decodeURIComponent(treeMatch[2].replace(/\/$/, '')),
      };
    }

    return null;
  }

  function normalizeVersionValue(key, value) {
    if (!key.endsWith('_version')) return value;

    const prValueMatch = value.match(/^PR_(\d+)$/);
    if (prValueMatch) return `PR_${prValueMatch[1]}`;

    const parsedUrl = parseGitHubVersionUrl(`- ${value}`);
    if (parsedUrl) return parsedUrl.version;

    return value;
  }

  const prNumber = context.payload.issue && context.payload.issue.pull_request ? context.payload.issue.number : null;
  if (!prNumber) {
    console.log('Not a PR event. Skipping.');
    return;
  }

  const owner = context.repo.owner;
  const repo = context.repo.repo;

  const commentBody = context.payload.comment.body;
  const commentAuthor = context.payload.comment.user.login;

  console.log(`Comment body: ${commentBody}`);
  console.log(`Comment author: ${commentAuthor}`);

  const commentBodyLines = commentBody.split('\n');
  const relevantLines = [];
  for (const lineRaw of commentBodyLines) {
    const line = lineRaw.trim().replaceAll('\\', '');
    if (!line || line.startsWith('#') || line.startsWith('<!--') || line.startsWith('```')) {
      continue;
    }
    relevantLines.push(line);
  }

  if (relevantLines.length === 0) {
    console.log('Comment is empty. Skipping.');
    return;
  }

  const fs = require('fs');
  const yaml = require('js-yaml');
  const cfgFile = fs.readFileSync(process.env.CONFIG_PATH, 'utf8');
  const cfg = yaml.load(cfgFile);
  console.log(`Loaded configuration: ${JSON.stringify(cfg)}`);

  const commentHeader = relevantLines[0];
  const expectedHeaders = cfg.expected_headers;
  let isCommandHeader = false;
  for (const expectedHeader of expectedHeaders) {
    if (commentHeader.startsWith(expectedHeader)) {
      isCommandHeader = true;
      break;
    }
  }

  if (!isCommandHeader) {
    console.log('Comment is not a cms-flaf-bot command. Skipping.');
    return;
  }

  const modifiedFiles = await github.paginate(github.rest.pulls.listFiles, {
    owner: context.repo.owner,
    repo: context.repo.repo,
    pull_number: prNumber,
  });

  let authorizedUsers = cfg.authorized_users;
  const githubAreaModified = modifiedFiles.some(file => file.filename.startsWith('.github/'));
  if (githubAreaModified) {
    const rootUser = 'kandrosov';
    authorizedUsers = [rootUser];
    console.log(`.github area is modified. Only ${rootUser} is authorized to trigger the pipeline.`);
  }
  if (!authorizedUsers.includes(commentAuthor)) {
    console.log(`User ${commentAuthor} is not authorized to trigger the pipeline. Skipping.`);
    return;
  }

  const variables = cfg.variables;
  const packages = Object.keys(variables).filter(key => key.endsWith('_version')).map(key => key.slice(0, -8));
  const rootPackages = Object.keys(variables).filter(key => key.endsWith('_active')).map(key => key.slice(0, -7));
  let gitlabBranch = cfg.gitlab_branch;

  if (repo !== 'FLAF' && !rootPackages.includes(repo) && !packages.includes(repo)) {
    console.log(`Repository ${repo} is neither FLAF, a root package, nor a known package. Skipping.`);
    return;
  }
  variables[`${repo}_version`] = `PR_${prNumber}`;

  let allOk = true;
  for (let i = 1; i < relevantLines.length; i++) {
    const entry = relevantLines[i];
    let parsed = false;
    if (entry.startsWith('- ') || entry.startsWith('* ')) {
      const parsedUrl = parseGitHubVersionUrl(entry);
      if (parsedUrl) {
        const key = `${parsedUrl.repo}_version`;
        if (key in variables) {
          variables[key] = parsedUrl.version;
          parsed = true;
        } else {
          console.log(`Unknown repository in GitHub URL: ${parsedUrl.repo}`);
        }
      }
    }
    if (!parsed && (entry.startsWith('- ') || entry.startsWith('* '))) {
      const parts = entry.substring(2).split('=');
      if (parts.length === 2) {
        const key = parts[0].trim();
        const value = parts[1].trim();
        if (key === 'gitlab_branch' || key === 'gitlab_ref') {
          gitlabBranch = value;
          parsed = true;
        } else if (key in variables) {
          variables[key] = normalizeVersionValue(key, value);
          parsed = true;
        } else {
          console.log(`Unknown variable: ${key}`);
        }
      }
    }
    if (!parsed) {
      console.log(`Invalid entry: ${entry}`);
      allOk = false;
      break;
    }
  }

  if (!allOk) {
    console.log('Invalid comment format. Skipping.');
    return;
  }

  console.log('Comment parsed.');

  const workflowNameItems = [];
  const sortedRootPackages = [...rootPackages].sort();
  for (const pkg of sortedRootPackages) {
    if (variables[`${pkg}_active`] === '1') {
      workflowNameItems.push(`${pkg}=${variables[`${pkg}_version`]}`);
    }
  }
  const sortedPackages = [...packages].sort();
  for (const pkg of sortedPackages) {
    if (!rootPackages.includes(pkg)) {
      workflowNameItems.push(`${pkg}=${variables[`${pkg}_version`]}`);
    }
  }
  variables.WORKFLOW_NAME = workflowNameItems.join(' ');
  variables.github_notify_url = `https://api.github.com/repos/${owner}/${repo}/issues/${prNumber}/comments`;

  core.setOutput('gitlab_url', cfg.gitlab_url);
  core.setOutput('gitlab_branch', gitlabBranch);
  core.setOutput('variables', JSON.stringify(variables));
  core.setOutput('trigger_pipeline', 'true');
};
