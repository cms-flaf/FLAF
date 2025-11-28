
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
      run_cmd git clone https://github.com/cms-analysis/CombineHarvester.git CombineHarvester
    fi
    run_cmd scram b -j8
    run_cmd touch "$ANALYSIS_SOFT_PATH/$CMSSW_VER/.installed"
  fi
}

do_install_combine() {
  local cmb_version=$1
  local cmb_path="$1"

  echo "Compiling standalone combine..."
  source "$FLAF_ENVIRONMENT_PATH/bin/activate"
  run_cmd cd "$cmb_path"
  run_cmd make -j8
  run_cmd touch "$cmb_path/build/.installed"
}

do_install_inference() {
  local cmb_version=$1

  local setups_dir="$HH_INFERENCE_PATH/.setups"
  local setup_file="$setups_dir/flaf.sh"

  if ! [ -f "$setup_file" ]; then
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
  fi

  # run_cmd mkdir -p "$ANALYSIS_SOFT_PATH/bin"
  # if ! [ -f "$ANALYSIS_SOFT_PATH/bin/combine" ]; then
  #   run_cmd ln -s "$FLAF_PATH/run_tools/cmsExe.sh" "$ANALYSIS_SOFT_PATH/bin/combine"
  # fi
  # if ! [ -f "$ANALYSIS_SOFT_PATH/bin/text2workspace.py" ]; then
  #   run_cmd ln -s "$FLAF_PATH/run_tools/cmsExe.sh" "$ANALYSIS_SOFT_PATH/bin/text2workspace.py"
  # fi

  run_cmd mkdir -p "$HH_INFERENCE_PATH/data"
  run_cmd touch "$HH_INFERENCE_PATH/data/.installed"
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

install_combine() {
  local env_file="$1"
  local node_os=$2
  local target_os=$3
  local cmb_path=$4
  install "$env_file" $node_os $target_os install_combine "$cmb_path/build/.installed" "$cmb_path"
}


install_inference() {
  local env_file="$1"
  local node_os=$2
  local target_os=$3
  local cmb_ver=$4
  install "$env_file" $node_os $target_os install_inference "$HH_INFERENCE_PATH/data/.installed" $cmb_ver
}

load_flaf_env() {

  [ -z "$LAW_HOME" ] && export LAW_HOME="$ANALYSIS_PATH/.law"
  [ -z "$LAW_CONFIG_FILE" ] && export LAW_CONFIG_FILE="$ANALYSIS_PATH/config/law.cfg"
  [ -z "$ANALYSIS_DATA_PATH" ] && export ANALYSIS_DATA_PATH="$ANALYSIS_PATH/data"
  [ -z "$X509_USER_PROXY" ] && export X509_USER_PROXY="$ANALYSIS_DATA_PATH/voms.proxy"

  if [[ ! -d "$ANALYSIS_DATA_PATH" ]]; then
    run_cmd mkdir -p "$ANALYSIS_DATA_PATH"
  fi

  local FLAF_LCG_VERSION="LCG_108a"
  local FLAF_LCG_ARCH="x86_64-el9-clang19-opt"
  if [[ ! -f "$FLAF_ENVIRONMENT_PATH/.${FLAF_LCG_VERSION}_${FLAF_LCG_ARCH}" ]]; then
    if [[ -d "$FLAF_ENVIRONMENT_PATH" ]]; then
      echo "Removing old FLAF environment installation in $FLAF_ENVIRONMENT_PATH ..."
      run_cmd rm -rf "$FLAF_ENVIRONMENT_PATH"
    fi
    echo "Creating FLAF environment in $FLAF_ENVIRONMENT_PATH ..."
    run_cmd $FLAF_PATH/run_tools/mk_flaf_env.sh "$FLAF_ENVIRONMENT_PATH" "$FLAF_LCG_VERSION" "$FLAF_LCG_ARCH"
  fi
  source "$FLAF_ENVIRONMENT_PATH/bin/activate"

  local os_version=$(cat /etc/os-release | grep VERSION_ID | sed -E 's/VERSION_ID="([0-9]+).*"/\1/')
  local os_prefix=$(get_os_prefix $os_version)
  local node_os=$os_prefix$os_version

  local flaf_cmssw_ver=CMSSW_14_1_7
  local target_os_version=9
  local target_os_prefix=$(get_os_prefix $target_os_version)
  local target_os_gt_prefix=$(get_os_prefix $target_os_version 1)
  local target_os=$target_os_prefix$target_os_version
  [ -z "$FLAF_COMBINE_VERSION" ] && export FLAF_COMBINE_VERSION="v10.2.0"
  export FLAF_CMSSW_BASE="$ANALYSIS_SOFT_PATH/$flaf_cmssw_ver"
  export FLAF_CMSSW_ARCH="${target_os_gt_prefix}${target_os_version}_amd64_gcc12"
  install_cmssw "$env_file" $node_os $target_os $FLAF_CMSSW_ARCH $flaf_cmssw_ver $FLAF_COMBINE_VERSION

  export PYTHONPATH="$ANALYSIS_PATH:$PYTHONPATH"
  if [ "$FLAF_COMBINE_VERSION" != "none" ]; then
    local cmb_os_version=9
    local cmb_os_prefix=$(get_os_prefix $cmb_os_version)
    local cmb_os=$cmb_os_prefix$cmb_os_version
    export FLAF_COMBINE_PATH="$FLAF_CMSSW_BASE/src/HiggsAnalysis/CombinedLimit"
    # install_combine "$env_file" $node_os $cmb_os "$FLAF_COMBINE_PATH"

    export PATH="$FLAF_COMBINE_PATH/build/bin:$PATH"
    export LD_LIBRARY_PATH="$FLAF_COMBINE_PATH/build/lib:$LD_LIBRARY_PATH"
    export PYTHONPATH="$FLAF_COMBINE_PATH/build/lib/python:$PYTHONPATH"
    if [ -d "$HH_INFERENCE_PATH" ]; then
      install_inference "$env_file" $node_os $cmb_os $FLAF_COMBINE_VERSION
      export PYTHONPATH="$HH_INFERENCE_PATH:$PYTHONPATH"
      source $HH_INFERENCE_PATH/.setups/flaf.sh
    fi
  fi

  if [ ! -z $ZSH_VERSION ]; then
    autoload bashcompinit
    bashcompinit
  fi

  source "$( law completion )"
  current_args=( "$@" )
  set --
  source /cvmfs/cms.cern.ch/rucio/setup-py3.sh &> /dev/null
  set -- "${current_args[@]}"
  #export PATH="$ANALYSIS_SOFT_PATH/bin:$PATH"
  alias cmsEnv="env -i HOME=$HOME ANALYSIS_PATH=$ANALYSIS_PATH ANALYSIS_DATA_PATH=$ANALYSIS_DATA_PATH X509_USER_PROXY=$X509_USER_PROXY FLAF_CMSSW_BASE=$FLAF_CMSSW_BASE FLAF_CMSSW_ARCH=$FLAF_CMSSW_ARCH $FLAF_PATH/cmsEnv.sh"
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
  export FLAF_PATH="$this_dir"

  if [ "$cmd" = "install_cmssw" ]; then
    do_install_cmssw "${@:3}"
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
unset -f do_install_inference
unset -f install
unset -f install_cmssw
unset -f install_inference
unset -f load_flaf_env
unset -f source_env_fn
