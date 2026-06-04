#!/usr/bin/env bash

action() {
    local run_token_server_host="{{run_token_server_host}}"
    local run_token_server_port="{{run_token_server_port}}"
    local analysis_path="{{analysis_path}}"
    local bundle_list="{{bundle_list}}"

    if [ -n "${run_token_server_host}" ] && [ -n "${run_token_server_port}" ]; then
        local get_run_token_script
        get_run_token_script=$(ls "${LAW_JOB_INIT_DIR}"/get_run_token*.py 2>/dev/null | head -1)
        if [ -z "${get_run_token_script}" ]; then
            echo "ERROR: get_run_token.py not found in ${LAW_JOB_INIT_DIR}"
            return 1
        fi
        python3 "${get_run_token_script}" \
            --server "${run_token_server_host}" \
            --port "${run_token_server_port}" \
            --path "${analysis_path}"
        local rc=$?
        if [ ${rc} -ne 0 ]; then
            echo "ERROR: failed to obtain run token (exit code ${rc}), aborting."
            return ${rc}
        fi
    fi

    if [ -f "${LAW_JOB_INIT_DIR}/voms.proxy" ] && [ -z "${X509_USER_PROXY:-}" ]; then
        chmod 600 "${LAW_JOB_INIT_DIR}/voms.proxy"
        export X509_USER_PROXY="${LAW_JOB_INIT_DIR}/voms.proxy"
    fi

    if [ -n "${bundle_list}" ]; then
        local lcg_setup="/cvmfs/sft.cern.ch/lcg/views/LCG_108a/x86_64-el9-gcc15-opt/setup.sh"
        if [ -f "${lcg_setup}" ]; then
            source "${lcg_setup}" 2>/dev/null
        fi

        local gfal_copy_bin
        gfal_copy_bin=$(which gfal-copy 2>/dev/null)
        if [ -z "${gfal_copy_bin}" ]; then
            echo "ERROR: gfal-copy not found; cannot download bundles"
            return 1
        fi

        local bundle_dir="${LAW_JOB_HOME}/bundle"
        mkdir -p "${bundle_dir}"

        local has_cmssw=0

        for entry in ${bundle_list}; do
            local flavour="${entry%%:*}"
            local url="${entry#*:}"

            echo "bootstrap: downloading ${flavour} bundle from ${url}"
            local bundle_tar="${LAW_JOB_HOME}/${flavour}.tar.bz2"
            env -i X509_USER_PROXY="${X509_USER_PROXY:-}" \
                "${gfal_copy_bin}" -f "${url}" "file://${bundle_tar}"
            local rc=$?
            if [ ${rc} -ne 0 ]; then
                echo "ERROR: ${flavour} bundle download failed (exit code ${rc})"
                return ${rc}
            fi

            echo "bootstrap: unpacking ${flavour} bundle"
            tar -xjf "${bundle_tar}" -C "${bundle_dir}"
            rm -f "${bundle_tar}"

            if [ "${flavour}" = "cmssw" ]; then
                has_cmssw=1
            fi
        done

        echo "bootstrap: patching bundle paths"
        local real_analysis_path
        real_analysis_path=$(head -1 "${bundle_dir}/soft/flaf_env/bin/pip" 2>/dev/null \
            | sed 's|^#!||; s|/soft/flaf_env.*||')
        if [ -n "${real_analysis_path}" ]; then
            grep -rl "${real_analysis_path}" "${bundle_dir}/soft/flaf_env/bin/" 2>/dev/null \
                | xargs -r sed -i "s|${real_analysis_path}|${bundle_dir}|g"
        fi

        if [ ${has_cmssw} -eq 1 ]; then
            local flaf_cmssw_version
            flaf_cmssw_version=$(sed -n 's/.*FLAF_CMSSW_VERSION="\([^"]*\)".*/\1/p' \
                "${bundle_dir}/env.sh" 2>/dev/null | head -1)
            if [ -n "${flaf_cmssw_version}" ]; then
                local cmssw_dir="${bundle_dir}/soft/${flaf_cmssw_version}"
                if [ -d "${cmssw_dir}/src" ]; then
                    echo "bootstrap: relocating CMSSW ${flaf_cmssw_version}"
                    source /cvmfs/cms.cern.ch/cmsset_default.sh 2>/dev/null || true
                    local prev_dir="${PWD}"
                    cd "${cmssw_dir}/src"
                    eval "$(scramv1 runtime -sh 2>/dev/null)" || true
                    scramv1 b ProjectRename "${cmssw_dir}" 2>/dev/null || true
                    cd "${prev_dir}"
                fi
            fi
        fi

        echo "bootstrap: sourcing env.sh from bundle"
        export FLAF_NO_INSTALL=1
        source "${bundle_dir}/env.sh"
    else
        if [ "${analysis_path}" = "NONE" ]; then
            echo "ERROR: analysis_path is NONE but no bundle_list was provided"
            return 1
        fi
        source "${analysis_path}/env.sh"
    fi
}
action
