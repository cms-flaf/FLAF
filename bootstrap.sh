#!/usr/bin/env bash

action() {
    local run_token_server_host="{{run_token_server_host}}"
    local run_token_server_port="{{run_token_server_port}}"
    local analysis_path="{{analysis_path}}"

    if [ -n "${run_token_server_host}" ] && [ -n "${run_token_server_port}" ]; then
        local get_run_token_script
        get_run_token_script=$(ls "${LAW_JOB_INIT_DIR}"/get_token*.py 2>/dev/null | head -1)
        if [ -z "${get_run_token_script}" ]; then
            echo "ERROR: get_token.py not found in ${LAW_JOB_INIT_DIR}"
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

    source "${analysis_path}/env.sh"
}
action
