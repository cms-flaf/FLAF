#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=.github/scripts/ci/common.sh
source "${SCRIPT_DIR}/common.sh"

require_var ANALYSIS_NAME

ANALYSIS_VERSION="${ANALYSIS_VERSION:-main}"
FLAF_version="${FLAF_version:-default}"
PlotKit_version="${PlotKit_version:-default}"
Corrections_version="${Corrections_version:-default}"
StatInference_version="${StatInference_version:-default}"
ERAS="${ERAS:-Run3_2022EE}"
PROCESSES="${PROCESSES:-}"
ANALYSIS_TASK="${ANALYSIS_TASK:-FLAF.Analysis.tasks.HistPlotTask}"
ANALYSIS_ARGS="${ANALYSIS_ARGS:---test 1000}"
GITHUB_NOTIFY_URL="${GITHUB_NOTIFY_URL:-}"
FLAF_GITHUB_TOKEN="${FLAF_GITHUB_TOKEN:-}"

post_github_notification() {
  local status_text=$1
  if [[ -n "${GITHUB_NOTIFY_URL}" && -n "${FLAF_GITHUB_TOKEN}" ]]; then
    local run_url="https://github.com/cms-flaf/FLAF/actions/runs/${GITHUB_RUN_ID:-}"
    local msg
    if [[ "${status_text}" == "success" ]]; then
      msg="✅ [GitHub Actions CI Run](${run_url}) passed for **${ANALYSIS_NAME}** (${ANALYSIS_VERSION})."
    else
      msg="❌ [GitHub Actions CI Run](${run_url}) failed for **${ANALYSIS_NAME}** (${ANALYSIS_VERSION})."
    fi
    curl -s -X POST \
      -H "Accept: application/vnd.github+json" \
      -H "Authorization: Bearer ${FLAF_GITHUB_TOKEN}" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "${GITHUB_NOTIFY_URL}" \
      -d "{\"body\": \"${msg}\"}" || true
  fi
}

trap_handler() {
  local exit_code=$?
  if [[ ${exit_code} -ne 0 ]]; then
    echo "=== Integration Test Failed with exit code ${exit_code} ==="
    post_github_notification "failure"
  fi
  exit "${exit_code}"
}
trap trap_handler EXIT

echo "=== Starting FLAF GitHub Actions Integration Test ==="
echo "Analysis: ${ANALYSIS_NAME} (${ANALYSIS_VERSION})"
echo "FLAF: ${FLAF_version}, PlotKit: ${PlotKit_version}, Corrections: ${Corrections_version}, StatInference: ${StatInference_version}"
echo "Eras: ${ERAS}"
echo "Processes: ${PROCESSES}"
echo "Task: ${ANALYSIS_TASK} ${ANALYSIS_ARGS}"

init_env
init_ssh
init_voms
init_gfal

WORK_DIR="/tmp/flaf_run_${GITHUB_RUN_ID:-$$}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

echo "Cloning ${ANALYSIS_NAME}..."
retry 3 5 git clone "https://github.com/cms-flaf/${ANALYSIS_NAME}.git" "${ANALYSIS_NAME}"

cd "${ANALYSIS_NAME}"
echo "Updating submodules..."
retry 3 5 git submodule update --init --recursive

echo "Applying test versions..."
switch_root_repo "${ANALYSIS_VERSION}"
switch_submodule_repo FLAF "${FLAF_version}"
switch_submodule_repo FLAF/PlotKit "${PlotKit_version}"
switch_submodule_repo Corrections "${Corrections_version}"
switch_submodule_repo StatInference "${StatInference_version}"

echo "Fetching Git LFS objects..."
retry 3 5 git lfs pull || true

if [[ -f config/ci_custom.yaml ]]; then
  cp config/ci_custom.yaml config/user_custom.yaml
fi

source_analysis_env

# Handle ERA expansion if "ALL" is specified
if [[ "${ERAS}" == "ALL" ]]; then
  ERAS="Run3_2022 Run3_2022EE Run3_2023 Run3_2023BPix"
fi

for era in ${ERAS}; do
  echo "========================================================="
  echo "=== Running era: ${era} ==="
  echo "========================================================="
  
  if [[ -n "${PROCESSES}" ]]; then
    for proc in ${PROCESSES}; do
      echo "--- Running process: ${proc} (era: ${era}) ---"
      # shellcheck disable=SC2086
      law run "${ANALYSIS_TASK}" --version CI --period "${era}" --workflow local --workers 4 ${ANALYSIS_ARGS} --process "${proc}"
    done
  else
    # shellcheck disable=SC2086
    law run "${ANALYSIS_TASK}" --version CI --period "${era}" --workflow local --workers 4 ${ANALYSIS_ARGS}
  fi
done

echo "========================================================="
echo "=== All integration test tasks completed successfully! ==="
echo "========================================================="
post_github_notification "success"
