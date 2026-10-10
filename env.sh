
run_cmd() {
  echo "> $@"
  "$@"
  RESULT=$?
  if (( $RESULT != 0 )); then
    echo "Error while running '$@'"
    kill -INT $$
  fi
}

get_os_prefix() {
  local os_version=$1
  local for_global_tag=$2
  if (( $os_version >= 8 )); then
    echo el
  elif (( $os_version < 6 )); then
    echo error
  else
    if [[ $for_global_tag == 1 || $os_version == 6 ]]; then
      echo slc
    else
      echo cc
    fi
  fi
}

do_install_cmssw() {
  export SCRAM_ARCH=$1
  local CMSSW_VER=$2
  local cmb_version=$3
  if ! [ -f "$ANALYSIS_SOFT_PATH/$CMSSW_VER/.installed" ]; then
    run_cmd mkdir -p "$ANALYSIS_SOFT_PATH"
    run_cmd cd "$ANALYSIS_SOFT_PATH"
    run_cmd source /cvmfs/cms.cern.ch/cmsset_default.sh
    if [ -d $CMSSW_VER ]; then
      echo "Removing incomplete $CMSSW_VER installation..."
      run_cmd rm -rf $CMSSW_VER
    fi
    echo "Creating $CMSSW_VER area for in $PWD ..."
    run_cmd scramv1 project CMSSW $CMSSW_VER
    run_cmd cd $CMSSW_VER/src
    eval `scramv1 runtime -sh`
    if [[ $(type -t apply_cmssw_customization_steps) == function ]] ; then
      run_cmd apply_cmssw_customization_steps
    fi
    if [ "$cmb_version" != "none" ]; then
      run_cmd git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
      run_cmd cd HiggsAnalysis/CombinedLimit
      run_cmd git checkout $cmb_version
      run_cmd cd ../..
      run_cmd git clone https://github.com/cms-flaf/CombineHarvester.git CombineHarvester
    fi
    run_cmd scram b -j8
    if [ "$cmb_version" != "none" ]; then
      run_cmd touch "$ANALYSIS_SOFT_PATH/$CMSSW_VER/.installed_combine_$cmb_version"
    fi
    run_cmd touch "$ANALYSIS_SOFT_PATH/$CMSSW_VER/.installed"
  fi
}

do_install_cmssw_combine() {
  export SCRAM_ARCH=$1
  local CMSSW_VER=$2
  local cmb_version=$3
  local installed_flag=$4

  echo "Moving the combine of $CMSSW_VER to $cmb_version..."
  run_cmd rm -f "$ANALYSIS_SOFT_PATH/$CMSSW_VER"/.installed_combine_*
  run_cmd source /cvmfs/cms.cern.ch/cmsset_default.sh
  run_cmd cd "$ANALYSIS_SOFT_PATH/$CMSSW_VER/src"
  eval `scramv1 runtime -sh`
  if [ ! -d HiggsAnalysis/CombinedLimit ]; then
    run_cmd git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
  fi
  if [ ! -d CombineHarvester ]; then
    run_cmd git clone https://github.com/cms-flaf/CombineHarvester.git CombineHarvester
  fi
  # the standalone build that earlier FLAF versions made inside this checkout
  if [ -d HiggsAnalysis/CombinedLimit/build ]; then
    run_cmd rm -rf HiggsAnalysis/CombinedLimit/build
  fi
  run_cmd git -C HiggsAnalysis/CombinedLimit fetch --tags origin
  run_cmd git -C HiggsAnalysis/CombinedLimit checkout -f "$cmb_version"
  run_cmd cd HiggsAnalysis/CombinedLimit
  run_cmd scram b clean
  run_cmd cd ../..
  run_cmd scram b -j8
  run_cmd touch "$installed_flag"
}

do_install_combine() {
  local cmb_path="$1"
  local cmb_version="$2"
  local cmb_patch="$3"
  local installed_flag="$4"

  echo "Compiling standalone combine $cmb_version..."
  source "$FLAF_ENVIRONMENT_PATH/bin/activate"
  if [ ! -d "$cmb_path" ]; then
    run_cmd git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git "$cmb_path"
  fi
  run_cmd cd "$cmb_path"
  run_cmd git fetch --tags origin
  run_cmd git checkout -f "$cmb_version"
  run_cmd git apply "$cmb_patch"
  if [ -d build ]; then
    echo "Removing incomplete or outdated combine build..."
    run_cmd rm -rf build
  fi
  run_cmd mkdir build
  run_cmd cd build
  run_cmd cmake ..
  run_cmd cmake --build . -j8
  run_cmd touch "$installed_flag"
}

do_install_inference() {
  local cmb_version=$1

  local installed_flag=$2

  local setups_dir="$HH_INFERENCE_PATH/.setups"
  local setup_file="$setups_dir/flaf.sh"

  run_cmd rm -f "$HH_INFERENCE_PATH"/data/.installed*
  run_cmd mkdir -p "$setups_dir"
  cat > "$setup_file" <<EOF
export DHI_USER="$(whoami)"
export DHI_USER_FIRSTCHAR="\${DHI_USER:0:1}"
export DHI_DATA="\$ANALYSIS_PATH/inference/data"
export DHI_STORE="\$DHI_DATA/store"
export DHI_STORE_JOBS="\$DHI_STORE"
export DHI_STORE_BUNDLES="\$DHI_STORE"
export DHI_STORE_EOSUSER="/eos/user/\$DHI_USER_FIRSTCHAR/\$DHI_USER/dhi/store"
export DHI_SOFTWARE="\$DHI_DATA/software"
export DHI_WLCG_USE_CACHE="false"
export DHI_LOCAL_SCHEDULER="True"
export DHI_SCHEDULER_HOST="hh:cmshhcombr2@hh-scheduler1.cern.ch"
export DHI_SCHEDULER_PORT="80"
export DHI_COMBINE_VERSION="$cmb_version"
EOF

  # run_cmd mkdir -p "$ANALYSIS_SOFT_PATH/bin"
  # if ! [ -f "$ANALYSIS_SOFT_PATH/bin/combine" ]; then
  #   run_cmd ln -s "$FLAF_PATH/run_tools/cmsExe.sh" "$ANALYSIS_SOFT_PATH/bin/combine"
  # fi
  # if ! [ -f "$ANALYSIS_SOFT_PATH/bin/text2workspace.py" ]; then
  #   run_cmd ln -s "$FLAF_PATH/run_tools/cmsExe.sh" "$ANALYSIS_SOFT_PATH/bin/text2workspace.py"
  # fi

  run_cmd mkdir -p "$HH_INFERENCE_PATH/data"
  run_cmd touch "$installed_flag"
}

install() {
  local env_file="$1"
  local node_os=$2
  local target_os=$3
  local cmd_to_run=$4
  local installed_flag=$5

  if [ -f "$installed_flag" ]; then
    return 0
  fi

  if [[ "${FLAF_NO_INSTALL:-0}" == "1" ]]; then
    echo "ERROR: $installed_flag not found and FLAF_NO_INSTALL=1"
    kill -INT $$
    # the signal does not stop a shell that sourced this file in a subshell
    return 1
  fi

  if [[ $node_os == $target_os ]]; then
    local env_cmd=""
    local env_cmd_args=""
  else
    local env_cmd="cmssw-$target_os"
    if ! command -v $env_cmd &> /dev/null; then
      echo "Unable to do a cross-platform installation. $env_cmd is not available."
      return 1
    fi
    local env_cmd_args="--command-to-run"
  fi

  run_cmd $env_cmd $env_cmd_args /usr/bin/env -i HOME=$HOME bash "$env_file" $cmd_to_run "${@:6}"
}

install_cmssw() {
  local env_file="$1"
  local node_os=$2
  local target_os=$3
  local scram_arch=$4
  local cmssw_version=$5
  local cmb_ver=$6
  install "$env_file" $node_os $target_os install_cmssw "$ANALYSIS_SOFT_PATH/$cmssw_version/.installed" "$scram_arch" "$cmssw_version" "$cmb_ver"
}

install_cmssw_combine() {
  local env_file="$1"
  local node_os=$2
  local target_os=$3
  local scram_arch=$4
  local cmssw_version=$5
  local cmb_ver=$6
  local installed_flag="$ANALYSIS_SOFT_PATH/$cmssw_version/.installed_combine_$cmb_ver"
  install "$env_file" $node_os $target_os install_cmssw_combine "$installed_flag" "$scram_arch" "$cmssw_version" \
    "$cmb_ver" "$installed_flag"
}

install_combine() {
  local env_file="$1"
  local node_os=$2
  local target_os=$3
  local cmb_path=$4
  local cmb_version=$5
  local cmb_patch=$6
  local installed_flag=$7
  install "$env_file" $node_os $target_os install_combine "$installed_flag" "$cmb_path" "$cmb_version" \
    "$cmb_patch" "$installed_flag"
}


install_inference() {
  local env_file="$1"
  local node_os=$2
  local target_os=$3
  local cmb_ver=$4
  local installed_flag="$HH_INFERENCE_PATH/data/.installed_$cmb_ver"
  install "$env_file" $node_os $target_os install_inference "$installed_flag" $cmb_ver "$installed_flag"
}

load_flaf_env() {

  [ -z "$LAW_HOME" ] && export LAW_HOME="$ANALYSIS_PATH/.law"
  [ -z "$LAW_CONFIG_FILE" ] && export LAW_CONFIG_FILE="$ANALYSIS_PATH/config/law.cfg"
  [ -z "$ANALYSIS_DATA_PATH" ] && export ANALYSIS_DATA_PATH="$ANALYSIS_PATH/data"
  [ -z "$X509_USER_PROXY" ] && export X509_USER_PROXY="$ANALYSIS_DATA_PATH/voms.proxy"

  if [[ ! -d "$ANALYSIS_DATA_PATH" ]]; then
    run_cmd mkdir -p "$ANALYSIS_DATA_PATH"
  fi

  local FLAF_LCG_VERSION="LCG_110a"
  local FLAF_LCG_ARCH="x86_64-el9-gcc15-opt"
  # flaf_env is identified by the LCG release and by the script that installs it, pins included;
  # its marker carries that identity, so a change to either builds a new environment.
  local flaf_env_script="$FLAF_PATH/run_tools/mk_flaf_env.sh"
  local flaf_env_script_hash
  # hashed from stdin: for a file name with a backslash, sha256sum prefixes the digest with one
  if ! flaf_env_script_hash="$(sha256sum < "$flaf_env_script")"; then
    echo "ERROR: cannot identify the FLAF environment: hashing $flaf_env_script failed"
    kill -INT $$
    return 1
  fi
  export FLAF_ENVIRONMENT_ID="${FLAF_LCG_VERSION}_${FLAF_LCG_ARCH}_${flaf_env_script_hash:0:12}"
  local flaf_env_marker="$FLAF_ENVIRONMENT_PATH/.$FLAF_ENVIRONMENT_ID"
  if [[ ! -f "$flaf_env_marker" ]]; then
    # A law batch job (law exports LAW_JOB_HOME before the bootstrap sources this file) runs from
    # an environment that other jobs share, so it never builds one.
    if [[ "${FLAF_NO_INSTALL:-0}" == "1" || -n "$LAW_JOB_HOME" ]]; then
      echo "ERROR: $FLAF_ENVIRONMENT_PATH was not built by the current $flaf_env_script (no marker .$FLAF_ENVIRONMENT_ID), and nothing is built from a law batch job or with FLAF_NO_INSTALL=1. Source env.sh on the submitting machine, which rebuilds it, and resubmit."
      kill -INT $$
      return 1
    fi
    # Whether the environment exists, and whether its marker is missing, is taken from listings,
    # and a stat that disagrees with them stops: neither a stat that failed nor a listing that is
    # stale (the storage blinking) may remove a working environment.
    local flaf_env_parent flaf_env_name flaf_env_listed
    flaf_env_parent="$(dirname "$FLAF_ENVIRONMENT_PATH")"
    flaf_env_name="$(basename "$FLAF_ENVIRONMENT_PATH")"
    if ! mkdir -p "$flaf_env_parent" \
        || ! flaf_env_listed="$(find -H "$flaf_env_parent" -mindepth 1 -maxdepth 1 -name "$flaf_env_name")"; then
      echo "ERROR: cannot tell whether $FLAF_ENVIRONMENT_PATH exists ($flaf_env_parent cannot be listed), so it is neither removed nor rebuilt"
      kill -INT $$
      return 1
    fi
    if [[ -n "$flaf_env_listed" ]]; then
      if [[ ! -d "$FLAF_ENVIRONMENT_PATH" ]]; then
        echo "ERROR: cannot tell whether $FLAF_ENVIRONMENT_PATH is an environment: it is listed in $flaf_env_parent but not seen as a directory, so it is neither removed nor rebuilt"
        kill -INT $$
        return 1
      fi
      if ! flaf_env_listed="$(find -H "$FLAF_ENVIRONMENT_PATH" -mindepth 1 -maxdepth 1 -name ".$FLAF_ENVIRONMENT_ID")" \
          || [[ -n "$flaf_env_listed" ]]; then
        echo "ERROR: cannot tell whether $FLAF_ENVIRONMENT_PATH has its marker .$FLAF_ENVIRONMENT_ID, so it is neither removed nor rebuilt"
        kill -INT $$
        return 1
      fi
      echo "Removing old FLAF environment installation in $FLAF_ENVIRONMENT_PATH ..."
    elif [[ -e "$FLAF_ENVIRONMENT_PATH" || -L "$FLAF_ENVIRONMENT_PATH" ]]; then
      echo "ERROR: cannot tell whether $FLAF_ENVIRONMENT_PATH exists: it is seen but not listed in $flaf_env_parent, so it is neither removed nor rebuilt"
      kill -INT $$
      return 1
    fi
    echo "Creating FLAF environment in $FLAF_ENVIRONMENT_PATH ..."
    # Each status is checked here rather than by run_cmd, whose SIGINT a shell that ignores it
    # carries past: the marker must follow a complete build only.
    if ! rm -rf "$FLAF_ENVIRONMENT_PATH" \
        || ! "$flaf_env_script" "$FLAF_ENVIRONMENT_PATH" "$FLAF_LCG_VERSION" "$FLAF_LCG_ARCH" \
        || ! touch "$flaf_env_marker"; then
      echo "ERROR: building the FLAF environment in $FLAF_ENVIRONMENT_PATH failed"
      kill -INT $$
      return 1
    fi
  fi
  source "$FLAF_ENVIRONMENT_PATH/bin/activate"

  local os_version=$(cat /etc/os-release | grep VERSION_ID | sed -E 's/VERSION_ID="([0-9]+).*"/\1/')
  local os_prefix=$(get_os_prefix $os_version)
  local node_os=$os_prefix$os_version

  [ -z "$FLAF_CMSSW_VERSION" ] && export FLAF_CMSSW_VERSION="CMSSW_16_0_6"
  [ -z "$FLAF_CMSSW_COMPILER" ] && export FLAF_CMSSW_COMPILER="gcc13"
  [ -z "$FLAF_CMSSW_OS_VERSION" ] && export FLAF_CMSSW_OS_VERSION="9"
  local target_os_prefix=$(get_os_prefix $FLAF_CMSSW_OS_VERSION)
  local target_os_gt_prefix=$(get_os_prefix $FLAF_CMSSW_OS_VERSION 1)
  local target_os=$target_os_prefix$FLAF_CMSSW_OS_VERSION
  [ -z "$FLAF_COMBINE_VERSION" ] && export FLAF_COMBINE_VERSION="v11.1.0"
  export FLAF_CMSSW_BASE="$ANALYSIS_SOFT_PATH/$FLAF_CMSSW_VERSION"
  export FLAF_CMSSW_ARCH="${target_os_gt_prefix}${FLAF_CMSSW_OS_VERSION}_amd64_${FLAF_CMSSW_COMPILER}"
  export PYTHONPATH="$ANALYSIS_PATH:$PYTHONPATH"
  # Make `import FLAF` / `import Corrections` resolve to the configured locations.
  # In production FLAF_PATH/CORRECTIONS_PATH equal $ANALYSIS_PATH/FLAF(/Corrections),
  # so this is a no-op. When overridden (dev overlay) prepend their parent dirs so the
  # edited copies win, and enable PYTHONSAFEPATH so an implicit cwd/script-dir entry
  # cannot pull the submodule copy ahead of the overlay for `python -m` / `python -c`
  # (FLAF and Corrections are namespace packages, merged across all sys.path entries).
  if [ "$FLAF_PATH" != "$ANALYSIS_PATH/FLAF" ]; then
    export PYTHONPATH="$(dirname "$FLAF_PATH"):$PYTHONPATH"
    export PYTHONSAFEPATH=1
  fi
  if [ "$CORRECTIONS_PATH" != "$ANALYSIS_PATH/Corrections" ]; then
    export PYTHONPATH="$(dirname "$CORRECTIONS_PATH"):$PYTHONPATH"
    export PYTHONSAFEPATH=1
  fi
  # When FLAF_NO_INSTALL=1 (bundle mode) skip CMSSW/combine/inference entirely if
  # CMSSW is not present in the bundle; tasks that need it declare the cmssw bundle flavour.
  if [[ "${FLAF_NO_INSTALL:-0}" == "0" ]] || [[ -d "$FLAF_CMSSW_BASE" ]]; then
    install_cmssw "$env_file" $node_os $target_os $FLAF_CMSSW_ARCH $FLAF_CMSSW_VERSION $FLAF_COMBINE_VERSION
    if [ "$FLAF_COMBINE_VERSION" != "none" ]; then
      # The same combine version is built in the CMSSW area (CombineHarvester, tasks run in CMSSW,
      # bundle jobs) and on its own against the ROOT of flaf_env (dhi); both follow its changes.
      install_cmssw_combine "$env_file" $node_os $target_os $FLAF_CMSSW_ARCH $FLAF_CMSSW_VERSION \
        $FLAF_COMBINE_VERSION
      local cmb_os_version=9
      local cmb_os_prefix=$(get_os_prefix $cmb_os_version)
      local cmb_os=$cmb_os_prefix$cmb_os_version
      # The combine of flaf_env (dhi) is a checkout of its own, built against the ROOT of flaf_env
      # and rebuilt with every new LCG release or combine version. It is in no bundle: bundle jobs
      # use the combine of the CMSSW area.
      export FLAF_COMBINE_PATH="$ANALYSIS_SOFT_PATH/HiggsAnalysis-CombinedLimit"
      if [[ "${FLAF_NO_INSTALL:-0}" == "0" ]] || [[ -d "$FLAF_COMBINE_PATH" ]]; then
        install_combine "$env_file" $node_os $cmb_os "$FLAF_COMBINE_PATH" "$FLAF_COMBINE_VERSION" \
          "$FLAF_PATH/run_tools/combine_root638_clipping.patch" \
          "$FLAF_COMBINE_PATH/build/.installed_${FLAF_COMBINE_VERSION}_${FLAF_LCG_VERSION}_${FLAF_LCG_ARCH}"
        export PATH="$FLAF_COMBINE_PATH/build/bin:$PATH"
        export LD_LIBRARY_PATH="$FLAF_COMBINE_PATH/build/lib:$LD_LIBRARY_PATH"
        export PYTHONPATH="$FLAF_COMBINE_PATH/build/python:$PYTHONPATH"
      fi
      if [ -d "$HH_INFERENCE_PATH" ]; then
        install_inference "$env_file" $node_os $cmb_os $FLAF_COMBINE_VERSION
        export PYTHONPATH="$HH_INFERENCE_PATH:$PYTHONPATH"
        source $HH_INFERENCE_PATH/.setups/flaf.sh
      fi
    fi
  fi

  if [ ! -z $ZSH_VERSION ]; then
    autoload bashcompinit
    bashcompinit
  fi

  source "$( law completion )" 2>/dev/null
  current_args=( "$@" )
  set --
  # Pin the Rucio version instead of following the volatile 'current' symlink (issue #146),
  # which occasionally breaks. Override with FLAF_RUCIO_VERSION; if the pinned version is not
  # on cvmfs, fall back to the standard setup script.
  export FLAF_RUCIO_VERSION="${FLAF_RUCIO_VERSION:-39.2.0}"
  local rucio_arch=$(uname -m)/$(/cvmfs/cms.cern.ch/common/cmsos | cut -d_ -f1 | sed 's|^[a-z]*|rhel|')
  local rucio_root=/cvmfs/cms.cern.ch/rucio/${rucio_arch}/py3
  local rucio_dir=${rucio_root}/${FLAF_RUCIO_VERSION}
  if [ -e "${rucio_dir}/bin/rucio" ]; then
    local rucio_pydir="$(grep '#!/' ${rucio_dir}/bin/rucio | sed 's|^#!||;s|/bin/python[^/]*$||')"
    [ -e "${rucio_pydir}/etc/profile.d/init.sh" ] && source "${rucio_pydir}/etc/profile.d/init.sh"
    export PATH="${rucio_dir}/bin${PATH:+:$PATH}"
    export PYTHONPATH="$(ls -d ${rucio_dir}/lib/python*/site-packages)${PYTHONPATH:+:$PYTHONPATH}"
    export RUCIO_HOME="${rucio_dir}"
    # Warn if cvmfs offers a newer Rucio than the pinned one, so the pin can be refreshed.
    local rucio_latest=$(ls -1 "${rucio_root}" 2>/dev/null | grep -E '^[0-9]' | sort -V | tail -1)
    if [ -n "$rucio_latest" ] && [ "$rucio_latest" != "$FLAF_RUCIO_VERSION" ] \
        && [ "$(printf '%s\n%s\n' "$FLAF_RUCIO_VERSION" "$rucio_latest" | sort -V | tail -1)" = "$rucio_latest" ]; then
      echo "Warning: a newer Rucio version '$rucio_latest' is available on cvmfs; FLAF is pinned to '$FLAF_RUCIO_VERSION' (set FLAF_RUCIO_VERSION to override). See FLAF issue #146." >&2
    fi
  else
    echo "Warning: pinned Rucio version '$FLAF_RUCIO_VERSION' not found on cvmfs; falling back to 'current'." >&2
    source /cvmfs/cms.cern.ch/rucio/setup-py3.sh &> /dev/null
  fi
  set -- "${current_args[@]}"
  #export PATH="$ANALYSIS_SOFT_PATH/bin:$PATH"
  alias cmsEnv="env -i HOME=$HOME ANALYSIS_PATH=$ANALYSIS_PATH ANALYSIS_DATA_PATH=$ANALYSIS_DATA_PATH X509_USER_PROXY=$X509_USER_PROXY FLAF_CMSSW_BASE=$FLAF_CMSSW_BASE FLAF_CMSSW_ARCH=$FLAF_CMSSW_ARCH $FLAF_PATH/cmsEnv.sh"

  ulimit -n 4096
}

source_env_fn() {
  local env_file="$1"
  local cmd="$2"

  local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
  local this_dir="$( cd "$( dirname "$this_file" )" && pwd )"

  if [ -z "$ANALYSIS_PATH" ]; then
    echo "ANALYSIS_PATH is not set. Exiting..."
    kill -INT $$
  fi

  [ -z "$ANALYSIS_SOFT_PATH" ] && export ANALYSIS_SOFT_PATH="$ANALYSIS_PATH/soft"
  [ -z "$FLAF_ENVIRONMENT_PATH" ] && export FLAF_ENVIRONMENT_PATH="$ANALYSIS_SOFT_PATH/flaf_env"
  # FLAF_PATH and CORRECTIONS_PATH are *inputs* to env.sh: if already set (e.g. by
  # flaf_dev.sh pointing at the edited top-level copies in a FLAF_all workspace) they
  # are respected as-is; otherwise they default to the submodule copies under
  # ANALYSIS_PATH. Everything downstream (PYTHONPATH, bundling, job bootstrap) derives
  # from these, so the dev overlay is just "set FLAF_PATH before sourcing env.sh".
  [ -z "$FLAF_PATH" ] && export FLAF_PATH="$this_dir"
  [ -z "$CORRECTIONS_PATH" ] && export CORRECTIONS_PATH="$ANALYSIS_PATH/Corrections"

  if [ "$cmd" = "install_cmssw" ]; then
    do_install_cmssw "${@:3}"
  elif [ "$cmd" = "install_cmssw_combine" ]; then
    do_install_cmssw_combine "${@:3}"
  elif [ "$cmd" = "install_combine" ]; then
    do_install_combine "${@:3}"
  elif [ "$cmd" = "install_inference" ]; then
    do_install_inference "${@:3}"
  else
    load_flaf_env "$env_file"
  fi
}

source_env_fn "$@"

unset -f run_cmd
unset -f get_os_prefix
unset -f do_install_cmssw
unset -f do_install_cmssw_combine
unset -f do_install_combine
unset -f do_install_inference
unset -f install
unset -f install_cmssw
unset -f install_cmssw_combine
unset -f install_combine
unset -f install_inference
unset -f load_flaf_env
unset -f source_env_fn
