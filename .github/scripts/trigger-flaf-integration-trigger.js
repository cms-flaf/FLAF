module.exports = async ({ github, context, core, process, fetch, JSON, URLSearchParams, console }) => {
  const variables = JSON.parse(process.env.VARIABLES);
  const ciBackend = (process.env.CI_BACKEND || variables.ci_backend || 'gitlab').toLowerCase();

  if (ciBackend === 'github') {
    console.log('Triggering GitHub Actions integration test workflow...');
    const rootPackages = Object.keys(variables).filter(k => k.endsWith('_active')).map(k => k.slice(0, -7));
    let activeAnalysis = rootPackages.find(pkg => variables[`${pkg}_active`] === '1');
    if (!activeAnalysis) {
      activeAnalysis = 'HH_bbtautau';
    }

    const inputs = {
      analysis_name: activeAnalysis,
      analysis_version: variables[`${activeAnalysis}_version`] || 'main',
      flaf_version: variables['FLAF_version'] || 'default',
      plotkit_version: variables['PlotKit_version'] || 'default',
      corrections_version: variables['Corrections_version'] || 'default',
      statinference_version: variables['StatInference_version'] || 'default',
      eras: variables[`${activeAnalysis}_eras`] || 'Run3_2022EE',
      processes: variables[`${activeAnalysis}_processes`] || 'custom_CI_Signal custom_CI_Background custom_CI_Data',
      task: variables[`${activeAnalysis}_task`] || 'FLAF.Analysis.tasks.HistPlotTask',
      args: variables[`${activeAnalysis}_args`] || '--test 1000',
      github_notify_url: variables.github_notify_url || '',
    };

    console.log('Dispatching integration-test.yaml with inputs:');
    for (const [key, value] of Object.entries(inputs)) {
      console.log(`\t${key}: ${value}`);
    }

    const token = process.env.FLAF_GITHUB_TOKEN || process.env.GITHUB_TOKEN;
    const response = await fetch('https://api.github.com/repos/cms-flaf/FLAF/actions/workflows/integration-test.yaml/dispatches', {
      method: 'POST',
      headers: {
        'Accept': 'application/vnd.github+json',
        'Authorization': `Bearer ${token}`,
        'X-GitHub-Api-Version': '2022-11-28',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        ref: 'main',
        inputs: inputs,
      }),
    });

    if (response.status === 204) {
      console.log('GitHub Actions integration workflow dispatched successfully.');
      const workflowUrl = 'https://github.com/cms-flaf/FLAF/actions/workflows/integration-test.yaml';
      const message = `[GitHub Actions integration workflow](${workflowUrl}) dispatched for **${activeAnalysis}** (${inputs.analysis_version})`;
      core.setOutput('send_message', 'true');
      core.setOutput('message', message);
      return;
    }

    console.log(`Failed to dispatch GitHub workflow: ${response.status}`);
    const responseText = await response.text();
    console.log(responseText);
    throw new Error(`Failed to dispatch GitHub workflow: ${response.status} - ${responseText}`);
  }

  const data = {
    token: '****',
    ref: process.env.GITLAB_BRANCH,
  };
  for (const [key, value] of Object.entries(variables)) {
    data[`variables[${key}]`] = value;
  }

  console.log('Triggering the FLAF integration pipeline with the following data:');
  for (const [key, value] of Object.entries(data)) {
    console.log(`\t${key}: ${value}`);
  }

  data.token = process.env.FLAF_INTEGRATION_TOKEN;
  const formData = new URLSearchParams();
  for (const [key, value] of Object.entries(data)) {
    formData.append(key, value);
  }

  const response = await fetch(process.env.GITLAB_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: formData,
  });

  if (response.status === 201) {
    console.log('Pipeline triggered successfully.');
    const responseData = await response.json();
    console.log(responseData);

    const pipelineId = responseData.id;
    const pipelineUrl = responseData.web_url;
    const message = `[pipeline#${pipelineId}](${pipelineUrl}) started`;
    core.setOutput('send_message', 'true');
    core.setOutput('message', message);
    return;
  }

  console.log(`Failed to trigger pipeline: ${response.status}`);
  const responseText = await response.text();
  console.log(responseText);
  throw new Error(`Failed to trigger pipeline: ${response.status} - ${responseText}`);
};

