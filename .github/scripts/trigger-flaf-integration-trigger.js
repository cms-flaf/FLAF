module.exports = async ({ core, process, fetch, JSON, URLSearchParams, console }) => {
  const variables = JSON.parse(process.env.VARIABLES);

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
