#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=.github/scripts/ci/common.sh
source "${SCRIPT_DIR}/common.sh"

require_var ANALYSIS_NAME
require_var ANALYSIS_TASK
require_var ANALYSIS_ARGS
require_var ERA

init_env
init_ssh
init_voms
init_gfal

RUN_DIR="/tmp/test_${ANALYSIS_NAME}_${GITHUB_RUN_ID:-$$}"
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"

if [[ ! -f "/workspace/${ANALYSIS_NAME}.tar.bz2" ]]; then
  echo "Error: /workspace/${ANALYSIS_NAME}.tar.bz2 not found." >&2
  exit 1
fi

echo "Unpacking /workspace/${ANALYSIS_NAME}.tar.bz2..."
tar xjf "/workspace/${ANALYSIS_NAME}.tar.bz2"

cd "${ANALYSIS_NAME}"

# Restore previous stage outputs if passed via /workspace/inputs
if [[ -d "/workspace/inputs" ]]; then
  echo "Restoring previous stage artifacts from /workspace/inputs..."
  mkdir -p data/CI output
  cp -rn /workspace/inputs/* data/CI/ 2>/dev/null || true
  cp -rn /workspace/inputs/* output/ 2>/dev/null || true
fi

source_analysis_env
init_voms

echo "Running task: ${ANALYSIS_TASK} ${ANALYSIS_ARGS} for era: ${ERA}"
# shellcheck disable=SC2086
law run "${ANALYSIS_TASK}" --version CI --period "${ERA}" --workflow local --workers 4 ${ANALYSIS_ARGS}

# Stage outputs for upload
mkdir -p /workspace/outputs
if [[ -d "data/CI" ]]; then
  cp -rn data/CI/* /workspace/outputs/ 2>/dev/null || true
fi
if [[ -d "output" ]]; then
  cp -rn output/* /workspace/outputs/ 2>/dev/null || true
fi

echo "=== Task ${ANALYSIS_TASK} (${ERA}) completed successfully ==="
