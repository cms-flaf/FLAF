#!/bin/bash

# Installs the python packages that CWS (HH_CWS_PATH, cms-analysis/general/cws) needs beyond
# flaf_env into the directory given, which cwsEnv puts on PYTHONPATH for the CWS commands alone.
# hydra-core 1.3 needs the antlr4 runtime 4.9, while flaf_env carries the 4.13 of its LCG release,
# so they cannot go into flaf_env. Executed, never sourced, with flaf_env active; every package is
# pinned, and env.sh identifies the installation by a hash of this file.
run_cmd() {
  "$@"
  RESULT=$?
  if (( $RESULT != 0 )); then
    echo "Error while running '$@'"
    exit $RESULT
  fi
}

target="$1"
if [[ -z "$target" ]]; then
  echo "usage: $0 TARGET_DIR"
  exit 1
fi

run_cmd rm -rf "$target"
run_cmd mkdir -p "$target"
run_cmd pip install --no-cache-dir --no-deps --target "$target" \
  hydra-core==1.3.2 omegaconf==2.3.1 antlr4-python3-runtime==4.9.3

exit 0
