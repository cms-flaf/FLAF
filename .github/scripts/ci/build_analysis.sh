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

rm -rf "${FLAF_CI_ROOT:?}/${ANALYSIS_NAME}"
mkdir -p "${FLAF_CI_ROOT}"
cd "${FLAF_CI_ROOT}"

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

echo "Resolved revisions:"
git --no-pager log -1 --format="  ${ANALYSIS_NAME}: %H %s"
for submodule_name in FLAF FLAF/PlotKit Corrections StatInference; do
  if [[ -d ${submodule_name} ]]; then
    (
      cd "${submodule_name}"
      git --no-pager log -1 --format="  ${submodule_name}: %H %s"
    )
  fi
done

echo "Fetching Git LFS objects..."
retry 3 5 git lfs pull || true

if [[ -f config/ci_custom.yaml ]]; then
  cp config/ci_custom.yaml config/user_custom.yaml
fi

# Install the analysis environment (flaf_env, CMSSW, combine) here, once per analysis.
# It is shipped to the test jobs inside the build archive, which is the reason for
# having a separate build stage: env.sh installs into $ANALYSIS_PATH/soft, i.e. inside
# the checkout that is archived below, and the test jobs run with FLAF_NO_INSTALL=1.
echo "Installing the analysis environment..."
(
  source_analysis_env
  echo "Analysis environment installed in ${FLAF_ENVIRONMENT_PATH}"
)
if [[ ! -d soft/flaf_env ]]; then
  echo "Error: the analysis environment was not installed in $(pwd)/soft." >&2
  exit 1
fi
du -sh soft/*

cd "${FLAF_CI_ROOT}"
create_build_archive "${ANALYSIS_NAME}"
df -h "${FLAF_CI_ROOT}" | tail -1
