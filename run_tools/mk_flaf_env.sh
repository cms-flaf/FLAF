#!/bin/bash

run_cmd() {
  "$@"
  RESULT=$?
  if (( $RESULT != 0 )); then
    echo "Error while running '$@'"
    kill -INT $$
  fi
}

link_all() {
    local in_dir="$1"
    local out_dir="$2"
    local exceptions="${@:3}"
    echo "Linking files from $in_dir into $out_dir"
    cd "$out_dir"
    for f in $(ls $in_dir); do
        if ! [[ $exceptions =~ (^|[[:space:]])"$f"($|[[:space:]]) ]]; then
            ln -s "$in_dir/$f"
        fi
    done
}

install() {
    local env_base=$1

    echo "Installing packages in $env_base"
    run_cmd source $env_base/bin/activate
    run_cmd pip install --upgrade pip
    run_cmd pip install law scinum
    run_cmd pip install https://github.com/riga/plotlib/archive/refs/heads/master.zip
    run_cmd pip install fastcrc
    run_cmd pip install bayesian-optimization
    run_cmd pip install yamllint
}

create() {
    local env_base=$1
    local lcg_view=$2
    local lcg_arch=$3

    local lcg_base=/cvmfs/sft.cern.ch/lcg/views/$lcg_view/$lcg_arch

    echo "Loading $lcg_view for $lcg_arch"
    run_cmd source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh $lcg_view $lcg_arch
    echo "Creating virtual environment in $env_base"
    run_cmd python3 -m venv $env_base --prompt flaf_env
    local root_path=$(realpath $(which root))
    local root_dir="$( cd "$( dirname "$root_path" )/.." && pwd )"
    cat >> $env_base/bin/activate <<EOF

export ROOTSYS=${root_dir}
export ROOT_INCLUDE_PATH=${env_base}/include
export LD_LIBRARY_PATH=${env_base}/lib/python3.11/site-packages

EOF
    link_all $lcg_base/bin $env_base/bin pip pip3 pip3.11 python python3 python3.11 gosam2herwig gosam-config.py gosam.py git java
    link_all $lcg_base/lib $env_base/lib/python3.11/site-packages python3.11
    link_all $lcg_base/lib/python3.11/site-packages $env_base/lib/python3.11/site-packages _distutils_hack distutils-precedence.pth pip pkg_resources setuptools graphviz py __pycache__ gosam-2.1.1_4b98559-py3.11.egg-info
    link_all $lcg_base/lib64 $env_base/lib/python3.11/site-packages cmake libonnx_proto.a libsvm.so.2 pkgconfig ThePEG libavh_olo.a libff.a libqcdloop.a
    link_all $lcg_base/include $env_base/include python3.11 gosam-contrib
}

action() {
    local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local env_base="$1"
    run_cmd "$this_file" create "$env_base" LCG_107_cuda x86_64-el9-gcc11-opt
    run_cmd "$this_file" install "$env_base"
}

if [[ "$1" == "create" ]]; then
    create "${@:2}"
elif [[ "$1" == "install" ]]; then
    install "${@:2}"
else
    action "${@:1}"
fi

exit 0
