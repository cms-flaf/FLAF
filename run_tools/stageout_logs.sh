#!/usr/bin/env bash

log_remote_base_url="{{log_remote_base_url}}"

if [ -z "${log_remote_base_url}" ]; then
    echo "stageout_logs: no remote log URL configured, skipping"
    exit 0
fi

postfix="${LAW_HTCONDOR_JOB_POSTFIX}"
if [ -n "${postfix}" ]; then
    log_file="stdall${postfix}.txt"
else
    cluster="${LAW_HTCONDOR_JOB_CLUSTER}"
    process="${LAW_HTCONDOR_JOB_PROCESS}"
    if [ -n "${cluster}" ] && [ -n "${process}" ]; then
        log_file="stdall_${cluster}_${process}.txt"
    else
        log_file="stdall.txt"
    fi
fi

if [ -n "${LAW_JOB_INIT_DIR}" ]; then
    log_path="${LAW_JOB_INIT_DIR}/${log_file}"
else
    log_path="${log_file}"
fi

if [ ! -f "${log_path}" ]; then
    echo "stageout_logs: log file '${log_path}' not found, skipping"
    exit 0
fi

log_remote_url="${log_remote_base_url%/}/${log_file}"

GFAL_COPY=$(which gfal-copy 2>/dev/null)
if [ -z "${GFAL_COPY}" ]; then
    echo "stageout_logs: gfal-copy not found in PATH, skipping"
    exit 0
fi

if [ -z "${X509_USER_PROXY:-}" ] && [ -f "${LAW_JOB_INIT_DIR}/voms.proxy" ]; then
    chmod 600 "${LAW_JOB_INIT_DIR}/voms.proxy"
    export X509_USER_PROXY="${LAW_JOB_INIT_DIR}/voms.proxy"
fi

# Pre-create the remote parent directory.  Without this, gfal-copy to a path whose
# parent does not yet exist can create the *leaf* (the log filename) as a directory
# and place the file inside it, producing ".../stdall_0To1.txt/stdall_0To1.txt".
# That mismatches the single-file URL recorded for failure reports, so the log
# appears "missing" at the reported location.  Creating the parent first makes
# gfal-copy write the file at exactly the intended path.
GFAL_MKDIR=$(which gfal-mkdir 2>/dev/null)
if [ -n "${GFAL_MKDIR}" ]; then
    env -i X509_USER_PROXY="${X509_USER_PROXY}" "${GFAL_MKDIR}" -p "${log_remote_base_url%/}" >/dev/null 2>&1
fi

local_url="file://$(realpath "${log_path}")"
echo "stageout_logs: uploading '${log_path}' to '${log_remote_url}'"
env -i X509_USER_PROXY="${X509_USER_PROXY}" "${GFAL_COPY}" -p -f "${local_url}" "${log_remote_url}"
ret=$?
if [ "${ret}" != "0" ]; then
    echo "stageout_logs: upload failed with exit code ${ret}, continuing"
fi
exit 0
