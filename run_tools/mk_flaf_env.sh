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
    if [[ ! -d "$in_dir" ]] || ! cd "$out_dir"; then
        echo "ERROR: cannot link $in_dir into $out_dir"
        kill -INT $$
    fi
    for f in $(ls $in_dir); do
        if ! [[ $exceptions =~ (^|[[:space:]])"$f"($|[[:space:]]) ]]; then
            ln -s "$in_dir/$f"
        fi
    done
}

install() {
    local env_base=$1
    local law_version=$2

    echo "Installing packages in $env_base"
    run_cmd source $env_base/bin/activate
    run_cmd pip install --upgrade pip
    run_cmd pip install luigi==3.8.1 "law==$law_version" scinum
    run_cmd pip install https://github.com/riga/plotlib/archive/refs/heads/master.zip
    run_cmd pip install fastcrc
    run_cmd pip install bayesian-optimization
    run_cmd pip install yamllint
    run_cmd pip install black
    run_cmd pip install cmsstyle  # optional PlotKit backend (ROOT/cmsstyle)
    # the LCG_110a mplhep 1.1.0 passes a 2-D y to the text of hep.cms.label
    run_cmd pip install mplhep==1.1.3
}

install_gh_cli() {
    local env_base=$1
    local gh_version="2.93.0"
    local gh_tarball="gh_${gh_version}_linux_amd64.tar.gz"
    local gh_url="https://github.com/cli/cli/releases/download/v${gh_version}/${gh_tarball}"
    local tmp_dir
    tmp_dir=$(mktemp -d)
    echo "Downloading GitHub CLI v${gh_version}..."
    run_cmd curl -fsSL -o "${tmp_dir}/${gh_tarball}" "${gh_url}"
    run_cmd tar -xzf "${tmp_dir}/${gh_tarball}" -C "${tmp_dir}"
    run_cmd cp "${tmp_dir}/gh_${gh_version}_linux_amd64/bin/gh" "${env_base}/bin/gh"
    run_cmd chmod +x "${env_base}/bin/gh"
    run_cmd rm -rf "${tmp_dir}"
    echo "GitHub CLI installed: $(${env_base}/bin/gh --version)"
}

join_by() {
    local IFS="$1"
    shift
    echo "$*"
}

dist_info_of() {
    # Metadata directories (dist-info / egg-info) of the given distributions in a site-packages dir.
    local site_dir="$1"
    shift
    local name
    for name in "$@"; do
        (cd "$site_dir" && ls -d ${name}-*.dist-info ${name}-*.egg-info 2>/dev/null)
    done
}

create() {
    local env_base=$1
    local lcg_version=$2
    local lcg_arch=$3

    local lcg_base=/cvmfs/sft.cern.ch/lcg/views/$lcg_version/$lcg_arch

    echo "Loading $lcg_version for $lcg_arch"
    run_cmd source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh $lcg_version $lcg_arch

    # Everything that differs between LCG releases is read from the loaded view.
    local py_ver=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    local py=python${py_ver}
    local gcc_base="$(cd "$(dirname "$(realpath "$(command -v gcc)")")/.." && pwd)"
    local binutils_base="$(cd "$(dirname "$(realpath "$(command -v ld)")")/.." && pwd)"
    local lcg_site=$lcg_base/lib/$py/site-packages
    for dir in "$lcg_site" "$gcc_base/lib64" "$binutils_base/lib"; do
        if [[ ! -d "$dir" ]]; then
            echo "ERROR: $dir not found for $lcg_version $lcg_arch"
            kill -INT $$
        fi
    done
    if [[ "$gcc_base" != /cvmfs/sft.cern.ch/lcg/releases/gcc/* || "$binutils_base" != /cvmfs/sft.cern.ch/lcg/releases/binutils/* ]]; then
        echo "ERROR: gcc ($gcc_base) or binutils ($binutils_base) do not come from the LCG release area"
        kill -INT $$
    fi
    echo "Using $py, gcc from $gcc_base, binutils from $binutils_base"

    echo "Creating virtual environment in $env_base"
    run_cmd python3 -m venv $env_base --prompt flaf_env
    local env_site=$env_base/lib/$py/site-packages
    local root_path=$(realpath $(which root))
    local root_dir="$( cd "$( dirname "$root_path" )/.." && pwd )"
    local ld_lib_path=$(join_by : \
        ${env_site} \
        ${env_site}/torch/lib \
        ${env_site}/tensorflow \
        ${env_site}/tensorflow/contrib/tensor_forest \
        ${env_site}/tensorflow/python/framework \
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
    link_all $lcg_base/bin $env_base/bin pip pip3 pip${py_ver} python python3 ${py} gosam2herwig gosam-config.py gosam.py git java black blackd
    link_all $gcc_base/bin $env_base/bin go gofmt
    link_all $lcg_base/lib $env_site ${py}
    link_all $lcg_base/lib $env_base/lib ${py}
    link_all $lcg_site $env_site _distutils_hack distutils-precedence.pth pip pkg_resources setuptools black blackd blib2to3 pathspec graphviz py __pycache__ tenacity servicex paramiko mplhep $(dist_info_of $lcg_site black pathspec tenacity servicex paramiko mplhep)
    link_all $lcg_base/lib64 $env_site cairo cmake libonnx_proto.a libsvm.so.2 pkgconfig ThePEG libavh_olo.a libff.a libqcdloop.a ${py} libzip.so src
    link_all $gcc_base/lib $env_base/lib
    link_all $gcc_base/lib64 $env_base/lib
    link_all $binutils_base/lib $env_base/lib libbfd.a libbfd.la libctf-nobfd.a libctf-nobfd.la libctf.a libctf.la libopcodes.a libopcodes.la libsframe.a libsframe.la
    link_all $lcg_base/include $env_base/include ${py} gosam-contrib
}

action() {
    local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local env_base="$1"
    local lcg_version="$2"
    local lcg_arch="$3"
    local law_version="$4"
    # currently tuned for LCG_110a x86_64-el9-gcc15-opt
    run_cmd "$this_file" create "$env_base" "$lcg_version" "$lcg_arch"
    run_cmd "$this_file" install "$env_base" "$law_version"
    run_cmd "$this_file" install_gh_cli "$env_base"
    run_cmd touch "$env_base/.${lcg_version}_${lcg_arch}"
}

if [[ "$1" == "create" ]]; then
    create "${@:2}"
elif [[ "$1" == "install" ]]; then
    install "${@:2}"
elif [[ "$1" == "install_gh_cli" ]]; then
    install_gh_cli "${@:2}"
else
    action "${@:1}"
fi

exit 0
