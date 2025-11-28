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
    local lcg_version=$2
    local lcg_arch=$3

    local lcg_base=/cvmfs/sft.cern.ch/lcg/views/$lcg_version/$lcg_arch

    echo "Loading $lcg_version for $lcg_arch"
    run_cmd source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh $lcg_version $lcg_arch
    echo "Creating virtual environment in $env_base"
    run_cmd python3 -m venv $env_base --prompt flaf_env
    local root_path=$(realpath $(which root))
    local root_dir="$( cd "$( dirname "$root_path" )/.." && pwd )"
    cat >> $env_base/bin/activate <<EOF

export ROOTSYS=${root_dir}
export ROOT_INCLUDE_PATH=${env_base}/include
export LD_LIBRARY_PATH=${env_base}/lib/python3.12/site-packages:${env_base}/lib/python3.12/site-packages/torch/lib:${env_base}/lib/python3.12/site-packages/tensorflow:${env_base}/lib/python3.12/site-packages/tensorflow/contrib/tensor_forest:${env_base}/lib/python3.12/site-packages/tensorflow/python/framework:${env_base}/lib:/cvmfs/sft.cern.ch/lcg/releases/clang/19.1.3-e838d/x86_64-el9/lib:/cvmfs/sft.cern.ch/lcg/releases/clang/19.1.3-e838d/x86_64-el9/lib/clang/19/lib/x86_64-unknown-linux-gnu:/cvmfs/sft.cern.ch/lcg/releases/clang/19.1.3-e838d/x86_64-el9/lib/x86_64-unknown-linux-gnu:/cvmfs/sft.cern.ch/lcg/releases/gcc/14.2.0-2f0a0/x86_64-el9/lib:/cvmfs/sft.cern.ch/lcg/releases/gcc/14.2.0-2f0a0/x86_64-el9/lib64:/cvmfs/sft.cern.ch/lcg/releases/binutils/2.40-acaab/x86_64-el9/lib:/cvmfs/sft.cern.ch/lcg/releases/binutils/2.40-acaab/x86_64-el9/lib

EOF

    link_all $lcg_base/bin $env_base/bin pip pip3 pip3.12 python python3 python3.12 gosam2herwig gosam-config.py gosam.py git java
    link_all $lcg_base/lib $env_base/lib/python3.12/site-packages python3.12
    link_all $lcg_base/lib/python3.12/site-packages $env_base/lib/python3.12/site-packages _distutils_hack distutils-precedence.pth pip pkg_resources setuptools graphviz py __pycache__ gosam-2.1.1_4b98559-py3.12.egg-info tenacity tenacity-9.0.0.dist-info servicex servicex-3.1.0.dist-info paramiko paramiko-2.9.2-py3.12.egg-info
    link_all $lcg_base/lib64 $env_base/lib/python3.12/site-packages cairo cmake libonnx_proto.a libsvm.so.2 pkgconfig ThePEG libavh_olo.a libff.a libqcdloop.a python3.12
    link_all $lcg_base/include $env_base/include python3.12 gosam-contrib
}

action() {
    local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local env_base="$1"
    local lcg_version="$2"
    local lcg_arch="$3"
    # currently tuned for LCG_108a x86_64-el9-clang19-opt
    run_cmd "$this_file" create "$env_base" "$lcg_version" "$lcg_arch"
    run_cmd "$this_file" install "$env_base"
    run_cmd touch "$env_base/.${lcg_version}_${lcg_arch}"
}

if [[ "$1" == "create" ]]; then
    create "${@:2}"
elif [[ "$1" == "install" ]]; then
    install "${@:2}"
else
    action "${@:1}"
fi

exit 0
