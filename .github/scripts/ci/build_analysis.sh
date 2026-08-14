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
REBUILD_CACHE="${rebuild_cache:-0}"

if [[ -n "${VARIABLES:-}" ]]; then
  ANALYSIS_VERSION=$(python3 -c "import json, os; vars=json.loads(os.environ.get('VARIABLES', '{}')); print(vars.get('${ANALYSIS_NAME}_version', '${ANALYSIS_VERSION}'))")
  FLAF_version=$(python3 -c "import json, os; vars=json.loads(os.environ.get('VARIABLES', '{}')); print(vars.get('FLAF_version', '${FLAF_version}'))")
  PlotKit_version=$(python3 -c "import json, os; vars=json.loads(os.environ.get('VARIABLES', '{}')); print(vars.get('PlotKit_version', '${PlotKit_version}'))")
  Corrections_version=$(python3 -c "import json, os; vars=json.loads(os.environ.get('VARIABLES', '{}')); print(vars.get('Corrections_version', '${Corrections_version}'))")
  StatInference_version=$(python3 -c "import json, os; vars=json.loads(os.environ.get('VARIABLES', '{}')); print(vars.get('StatInference_version', '${StatInference_version}'))")
  REBUILD_CACHE=$(python3 -c "import json, os; vars=json.loads(os.environ.get('VARIABLES', '{}')); print(vars.get('rebuild_cache', '${REBUILD_CACHE}'))")
fi

echo "=== Building ${ANALYSIS_NAME} (${ANALYSIS_VERSION}) ==="
echo "FLAF: ${FLAF_version}, PlotKit: ${PlotKit_version}, Corrections: ${Corrections_version}, StatInference: ${StatInference_version}"

init_env
init_ssh
init_voms
init_gfal

mkdir -p "${FLAF_CI_ROOT}"
cd "${FLAF_CI_ROOT}"

# Reference cache: a pristine checkout of the default branches with the environment
# already installed, restored by the workflow from the GitHub Actions cache (see
# reference_archive_path). It plays the same role as the install cache on EOS used by the
# GitLab pipeline: the requested revisions are applied on top of it, so a run only pays
# for the delta instead of a full clone + ~15 min environment install.
REFERENCE_ARCHIVE="$(reference_archive_path "${ANALYSIS_NAME}")"

restore_reference() {
  [[ -f ${REFERENCE_ARCHIVE} ]] || return 1
  echo "Restoring the reference checkout from ${REFERENCE_ARCHIVE}..."
  rm -rf "${FLAF_CI_ROOT:?}/${ANALYSIS_NAME}"
  tar -xf "${REFERENCE_ARCHIVE}" -C "${FLAF_CI_ROOT}" || return 1
  return 0
}

clone_pristine() {
  echo "Cloning ${ANALYSIS_NAME}..."
  rm -rf "${FLAF_CI_ROOT:?}/${ANALYSIS_NAME}"
  retry 3 5 git clone "https://github.com/cms-flaf/${ANALYSIS_NAME}.git" "${ANALYSIS_NAME}"
  (
    cd "${ANALYSIS_NAME}"
    echo "Updating submodules..."
    retry 3 5 git submodule update --init --recursive
    retry 3 5 git lfs pull || true
  )
}

if [[ ${REBUILD_CACHE} == "1" ]]; then
  echo "rebuild_cache requested; rebuilding the ${ANALYSIS_NAME} reference from scratch."
  clone_pristine
  BUILD_CACHE_REFRESH=1
elif ! restore_reference; then
  echo "No usable reference cache for ${ANALYSIS_NAME}; building it from scratch."
  clone_pristine
  BUILD_CACHE_REFRESH=1
fi

cd "${ANALYSIS_NAME}"

# Install (or, for a restored reference, validate) the analysis environment before the
# requested revisions are applied, so that the reference archive published below carries
# it. env.sh is a no-op when the installation flags of the current LCG/CMSSW versions are
# already there, and reinstalls only what those versions changed.
echo "Preparing the analysis environment..."
(
  source_analysis_env
  echo "Analysis environment ready in ${FLAF_ENVIRONMENT_PATH}"
)

# Publish the reference for the next runs while the checkout is still pristine: it must
# not carry the revisions of the PR under test. The workflow stores it in the Actions
# cache when its (weekly) key was not an exact hit.
if [[ ${BUILD_CACHE_REFRESH:-0} == "1" ]]; then
  cd "${FLAF_CI_ROOT}"
  create_reference_archive "${ANALYSIS_NAME}"
  cd "${ANALYSIS_NAME}"
fi

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

# Re-run env.sh for the requested revisions: it is a no-op unless they changed the
# environment recipe (e.g. a new LCG or CMSSW version), in which case the test jobs -- which
# run with FLAF_NO_INSTALL=1 -- would otherwise abort. The environment lives in
# $ANALYSIS_PATH/soft, inside the checkout archived below.
echo "Finalizing the analysis environment..."
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
