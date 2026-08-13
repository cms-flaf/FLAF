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

if [[ -n "${VARIABLES:-}" ]]; then
  ANALYSIS_VERSION=$(python3 -c "import json, os; vars=json.loads(os.environ.get('VARIABLES', '{}')); print(vars.get('${ANALYSIS_NAME}_version', '${ANALYSIS_VERSION}'))")
  FLAF_version=$(python3 -c "import json, os; vars=json.loads(os.environ.get('VARIABLES', '{}')); print(vars.get('FLAF_version', '${FLAF_version}'))")
  PlotKit_version=$(python3 -c "import json, os; vars=json.loads(os.environ.get('VARIABLES', '{}')); print(vars.get('PlotKit_version', '${PlotKit_version}'))")
  Corrections_version=$(python3 -c "import json, os; vars=json.loads(os.environ.get('VARIABLES', '{}')); print(vars.get('Corrections_version', '${Corrections_version}'))")
  StatInference_version=$(python3 -c "import json, os; vars=json.loads(os.environ.get('VARIABLES', '{}')); print(vars.get('StatInference_version', '${StatInference_version}'))")
fi

echo "=== Building ${ANALYSIS_NAME} (${ANALYSIS_VERSION}) ==="
echo "FLAF: ${FLAF_version}, PlotKit: ${PlotKit_version}, Corrections: ${Corrections_version}, StatInference: ${StatInference_version}"

init_env
init_ssh
init_voms
init_gfal

BUILD_DIR="/tmp/build_${ANALYSIS_NAME}_${GITHUB_RUN_ID:-$$}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Cloning ${ANALYSIS_NAME}..."
retry 3 5 git clone "https://github.com/cms-flaf/${ANALYSIS_NAME}.git" "${ANALYSIS_NAME}"

cd "${ANALYSIS_NAME}"
echo "Updating submodules..."
retry 3 5 git submodule update --init --recursive

echo "Applying requested versions..."
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

cd "${BUILD_DIR}"
echo "Creating build tarball ${ANALYSIS_NAME}.tar.bz2..."
tar cjf "/workspace/${ANALYSIS_NAME}.tar.bz2" "${ANALYSIS_NAME}"

echo "Build complete: /workspace/${ANALYSIS_NAME}.tar.bz2"
