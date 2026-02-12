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
    run_cmd pip install black
}

join_by() {
    local IFS="$1"
    shift
    echo "$*"
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
    local ld_lib_path=$(join_by : \
        ${env_base}/lib/python3.12/site-packages \
        ${env_base}/lib/python3.12/site-packages/torch/lib \
        ${env_base}/lib/python3.12/site-packages/tensorflow \
        ${env_base}/lib/python3.12/site-packages/tensorflow/contrib/tensor_forest \
        ${env_base}/lib/python3.12/site-packages/tensorflow/python/framework \
        ${env_base}/lib/ \
    )
    cat >> $env_base/bin/activate <<EOF

export ROOTSYS=${root_dir}
export ROOT_INCLUDE_PATH=${env_base}/include
export LD_LIBRARY_PATH=${ld_lib_path}
export CC=${env_base}/bin/gcc
export CXX=${env_base}/bin/g++
export C_INCLUDE_PATH=${env_base}/include
export CPLUS_INCLUDE_PATH=${env_base}/include
export CMAKE_PREFIX_PATH="${env_base}"
EOF


    run_cmd ln -s $lcg_base/cmake $env_base/cmake
    link_all $lcg_base/bin $env_base/bin pip pip3 pip3.12 python python3 python3.12 gosam2herwig gosam-config.py gosam.py git java black blackd
    link_all /cvmfs/sft.cern.ch/lcg/releases/gcc/15.2.0-35657/x86_64-el9/bin $env_base/bin go gofmt
    link_all $lcg_base/lib $env_base/lib/python3.12/site-packages python3.12
    link_all $lcg_base/lib $env_base/lib python3.12
    link_all $lcg_base/lib/python3.12/site-packages $env_base/lib/python3.12/site-packages _distutils_hack distutils-precedence.pth pip pkg_resources setuptools black blackd black-24.10.0.dist-info blib2to3 pathspec pathspec-0.11.1.dist-info graphviz py __pycache__ gosam-2.1.1_4b98559-py3.12.egg-info tenacity tenacity-9.0.0.dist-info servicex servicex-3.1.0.dist-info paramiko paramiko-2.9.2-py3.12.egg-info
    link_all $lcg_base/lib64 $env_base/lib/python3.12/site-packages cairo cmake libonnx_proto.a libsvm.so.2 pkgconfig ThePEG libavh_olo.a libff.a libqcdloop.a python3.12
    link_all /cvmfs/sft.cern.ch/lcg/releases/gcc/15.2.0-35657/x86_64-el9/lib $env_base/lib
    link_all /cvmfs/sft.cern.ch/lcg/releases/gcc/15.2.0-35657/x86_64-el9/lib64 $env_base/lib
    link_all /cvmfs/sft.cern.ch/lcg/releases/binutils/2.40-acaab/x86_64-el9/lib $env_base/lib libbfd.a libbfd.la libctf-nobfd.a libctf-nobfd.la libctf.a libctf.la libopcodes.a libopcodes.la libsframe.a libsframe.la
    link_all $lcg_base/include $env_base/include python3.12 gosam-contrib
}

action() {
    local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local env_base="$1"
    local lcg_version="$2"
    local lcg_arch="$3"
    # currently tuned for LCG_108a x86_64-el9-gcc15-opt
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
