#!/usr/bin/env bash

# Directory in which the analysis checkout produced by the build job lives. Every job
# must see it under exactly the same absolute path: the flaf_env virtualenv and the
# CMSSW/combine areas installed by the build job record their own location in activate
# scripts, script shebangs and SCRAM configuration, so an installation unpacked
# somewhere else would not work. It is a bind mount of a runner-workspace directory, so
# the path inside the container is identical in build and test jobs
# (see .github/actions/flaf-ci-run).
FLAF_CI_ROOT="${FLAF_CI_ROOT:-/flaf_ci}"

# Runner workspace: where the build archive is written by the build job and picked up
# from by the test jobs (through GitHub artifacts), and where the outputs of previous
# stages are staged for the current one.
FLAF_CI_WORKSPACE="${FLAF_CI_WORKSPACE:-/workspace}"
STAGE_INPUTS_DIR="${STAGE_INPUTS_DIR:-${FLAF_CI_WORKSPACE}/ci_inputs}"
# Workspace subdirectory saved to / restored from the GitHub Actions cache by the build
# job; keep it in sync with the `path:` of the cache steps in integration-test.yaml.
BUILD_CACHE_DIR_NAME="${BUILD_CACHE_DIR_NAME:-build_cache}"

require_var() {
  local var_name=$1
  if [[ -z ${!var_name:-} ]]; then
    echo "Required variable '${var_name}' is not set." >&2
    exit 1
  fi
}

retry() {
  local max_attempts=$1
  local delay=$2
  shift 2
  local attempt=1
  local rc
  while true; do
    rc=0
    "$@" || rc=$?
    if ((rc == 0)); then
      return 0
    fi
    if ((attempt >= max_attempts)); then
      echo "retry: '$*' still failing (exit ${rc}) after ${attempt} attempts; giving up." >&2
      return "${rc}"
    fi
    echo "retry: '$*' failed (exit ${rc}); attempt ${attempt}/${max_attempts}, retrying in ${delay}s..." >&2
    sleep "${delay}"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
  done
}

init_env() {
  export LC_ALL=C
}

init_ssh() {
  mkdir -p "${HOME}/.ssh"
  chmod 700 "${HOME}/.ssh"

  cat << 'EOF' > "${HOME}/.ssh/config"
Host github.com
    HostName github.com
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null

Host gitlab.cern.ch
    HostName gitlab.cern.ch
    Port 7999
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
  chmod 600 "${HOME}/.ssh/config"

  if [[ -n ${SSH_PRIVATE_KEY:-} ]]; then
    printf '%s' "${SSH_PRIVATE_KEY}" | base64 -d > "${HOME}/.ssh/id_ed25519" 2>/dev/null || printf '%s' "${SSH_PRIVATE_KEY}" > "${HOME}/.ssh/id_ed25519"
    chmod 400 "${HOME}/.ssh/id_ed25519"
    eval "$(ssh-agent -s)" 2>/dev/null || true
    ssh-add "${HOME}/.ssh/id_ed25519" 2>/dev/null || true
  else
    echo "WARNING: SSH_PRIVATE_KEY is not set in GitHub Secrets. GitLab SSH submodules (e.g. inference) require SSH_PRIVATE_KEY."
    git config --global url."https://github.com/".insteadOf "git@github.com:"
  fi

  retry 3 3 _scan_known_hosts
  git config --global user.email "cms-flaf@proton.me"
  git config --global user.name "CMS FLAF Integration Bot"
  git config --global fetch.recurseSubmodules no
}

_scan_known_hosts() {
  ssh-keyscan github.com >> "${HOME}/.ssh/known_hosts" 2>/dev/null || true
  ssh-keyscan -p 7999 gitlab.cern.ch >> "${HOME}/.ssh/known_hosts" 2>/dev/null || true
}

init_voms() {
  if [[ -z ${GRID_USERCERT:-} || -z ${GRID_USERKEY:-} || -z ${GRID_PASSWORD:-} ]]; then
    echo "Grid certificate secrets not fully provided. Skipping VOMS proxy initialization."
    return 0
  fi

  export X509_VOMS_DIR=/cvmfs/grid.cern.ch/etc/grid-security/vomsdir/
  export VOMS_USERCONF=/cvmfs/grid.cern.ch/etc/grid-security/vomses/
  export X509_CERT_DIR=/cvmfs/grid.cern.ch/etc/grid-security/certificates/
  mkdir -p "${HOME}/.globus"
  chmod 700 "${HOME}/.globus"
  printf '%s' "${GRID_USERCERT}" | base64 -d > "${HOME}/.globus/usercert.pem" 2>/dev/null || printf '%s' "${GRID_USERCERT}" > "${HOME}/.globus/usercert.pem"
  printf '%s' "${GRID_USERKEY}" | base64 -d > "${HOME}/.globus/userkey.pem" 2>/dev/null || printf '%s' "${GRID_USERKEY}" > "${HOME}/.globus/userkey.pem"
  chmod 400 "${HOME}/.globus/usercert.pem"
  chmod 400 "${HOME}/.globus/userkey.pem"

  export X509_USER_PROXY="${X509_USER_PROXY:-/tmp/x509up_u$(id -u)_${GITHUB_RUN_ID:-$$}}"
  retry 3 5 _create_voms_proxy
  voms-proxy-info --all || true
}

_create_voms_proxy() {
  local pw
  pw=$(printf '%s' "${GRID_PASSWORD}" | base64 -d 2>/dev/null || printf '%s' "${GRID_PASSWORD}")
  printf '%s' "${pw}" \
    | voms-proxy-init -voms cms -rfc -valid 192:00 -pwstdin -out "${X509_USER_PROXY}" \
        -cert "${HOME}/.globus/usercert.pem" -key "${HOME}/.globus/userkey.pem" \
    && voms-proxy-info -exists -valid 1:00 -file "${X509_USER_PROXY}"
}

GFAL_LCG_SETUP="${GFAL_LCG_SETUP:-/cvmfs/sft.cern.ch/lcg/views/LCG_108a/x86_64-el9-gcc15-opt/setup.sh}"

init_gfal() {
  if ! command -v gfal-copy >/dev/null 2>&1; then
    if [[ -f ${GFAL_LCG_SETUP} ]]; then
      local had_nounset=0
      if [[ $- == *u* ]]; then
        had_nounset=1
        set +u
      fi
      # shellcheck disable=SC1090
      source "${GFAL_LCG_SETUP}"
      if [[ ${had_nounset} -eq 1 ]]; then
        set -u
      fi
    fi
  fi
}

source_analysis_env() {
  local had_nounset=0
  if [[ $- == *u* ]]; then
    had_nounset=1
    set +u
  fi

  source env.sh

  if [[ ${had_nounset} -eq 1 ]]; then
    set -u
  fi
}

# The build archive carries the installed analysis environment (flaf_env + CMSSW +
# combine), so it is several GiB and is unpacked by every test job. bzip2 would spend
# minutes of every job on decompression alone, hence the fastest available compressor.
_archive_compressor() {
  if command -v zstd > /dev/null 2>&1; then
    echo "zstd -T0 -3"
  elif command -v pigz > /dev/null 2>&1; then
    echo "pigz -3"
  else
    echo "gzip -3"
  fi
}

_archive_ext() {
  if command -v zstd > /dev/null 2>&1; then
    echo "tar.zst"
  else
    echo "tar.gz"
  fi
}

build_archive_path() {
  echo "${FLAF_CI_WORKSPACE}/${1}.$(_archive_ext)"
}

# Reference checkout (pristine, environment installed) kept across runs in the GitHub
# Actions cache. It lives in the runner workspace because that is what the cache action
# can store; the build job unpacks it into FLAF_CI_ROOT.
reference_archive_path() {
  echo "${FLAF_CI_WORKSPACE}/${BUILD_CACHE_DIR_NAME}/${1}.$(_archive_ext)"
}

# Unlike the build archive, this one keeps .git: the next run switches the checkout to the
# revisions it has to test, which needs the repository history.
create_reference_archive() {
  local analysis=$1
  local archive
  archive="$(reference_archive_path "${analysis}")"
  echo "Creating reference archive ${archive}..."
  mkdir -p "$(dirname "${archive}")"
  tar -I "$(_archive_compressor)" -cf "${archive}" -C "${FLAF_CI_ROOT}" "${analysis}"
  echo "Reference archive created: $(du -h "${archive}" | cut -f1)"
}

create_build_archive() {
  local analysis=$1
  local archive
  archive="$(build_archive_path "${analysis}")"
  echo "Creating build archive ${archive}..."
  # Git metadata is not needed downstream (nothing in FLAF reads it at run time) and
  # would nearly double the archive size because of the LFS payloads.
  tar --exclude=.git -I "$(_archive_compressor)" \
    -cf "${archive}" -C "${FLAF_CI_ROOT}" "${analysis}"
  echo "Build archive created: $(du -h "${archive}" | cut -f1)"
}

extract_build_archive() {
  local analysis=$1
  local archive
  archive="$(build_archive_path "${analysis}")"
  if [[ ! -f ${archive} ]]; then
    echo "Error: build archive ${archive} not found." >&2
    return 1
  fi
  echo "Unpacking ${archive} into ${FLAF_CI_ROOT}..."
  mkdir -p "${FLAF_CI_ROOT}"
  tar -xf "${archive}" -C "${FLAF_CI_ROOT}"
  # Reclaim the runner disk: the unpacked checkout plus its environment is large.
  rm -f "${archive}"
}

sync_and_update_submodules() {
  local recurse=${1:-}

  git submodule init
  git submodule sync
  while read -r key url; do
    [[ -z ${key} ]] && continue
    local name=${key#submodule.}
    name=${name%.url}
    local gitdir
    gitdir=$(git rev-parse --git-path "modules/${name}")
    if [[ -d ${gitdir} ]]; then
      git --git-dir="${gitdir}" remote set-url origin "${url}" 2>/dev/null || true
    fi
  done < <(git config -f .gitmodules --get-regexp '^submodule\..*\.url$' 2>/dev/null || true)
  git submodule update --init

  if [[ ${recurse} == "--recursive" ]]; then
    local sub_path
    while read -r sub_path; do
      [[ -n ${sub_path} && -e ${sub_path}/.git ]] || continue
      ( cd "${sub_path}" && sync_and_update_submodules --recursive )
    done < <(git config -f .gitmodules --get-regexp '^submodule\..*\.path$' 2>/dev/null | awk '{print $2}')
  fi
}

switch_current_repo() {
  local version=$1

  if [[ ${version} == "default" ]]; then
    return 0
  fi

  if [[ ${version:0:3} == "PR_" ]]; then
    git fetch --no-recurse-submodules origin "pull/${version:3}/head:${version}"
    git submodule deinit -f --all 2>/dev/null || true
    git merge "${version}"
    sync_and_update_submodules --recursive
    return 0
  fi

  if [[ $(git rev-parse --abbrev-ref HEAD) == "${version}" ]]; then
    git pull --no-recurse-submodules origin "${version}"
  else
    git fetch --no-recurse-submodules origin "${version}:${version}"
    git switch "${version}"
  fi
}

switch_root_repo() {
  local version=$1
  switch_current_repo "${version}"
  sync_and_update_submodules --recursive
}

switch_submodule_repo() {
  local submodule_name=$1
  local version=$2

  if [[ ! -d ${submodule_name} ]]; then
    echo "Submodule ${submodule_name} not found. Skipping."
    return 0
  fi

  if [[ ${version} == "default" ]]; then
    return 0
  fi

  (
    cd "${submodule_name}"
    switch_current_repo "${version}"
    sync_and_update_submodules
  )
}
