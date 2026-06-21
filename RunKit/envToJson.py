import json
import os
import sys

if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(file_dir))
    __package__ = "RunKit"

from .run_tools import ps_call


def get_env(script, python_cmd="python3", singularity_cmd=None):
    magic_str = "---envToJson.py---"

    cmd = f'{script}; echo {magic_str}; {python_cmd} -c "import json, os; print(json.dumps(dict(os.environ)))"'
    if singularity_cmd:
        cmd = f"{singularity_cmd} --command-to-run '{cmd}'"
    returncode, output, err = ps_call(
        [cmd],
        shell=True,
        env={},
        catch_stdout=True,
        catch_stderr=True,
        split="\n",
        verbose=0,
    )
    for n, line in enumerate(output):
        if line == magic_str:
            break
    env = json.loads(output[n + 1])
    to_del = ["_"]
    if singularity_cmd:
        to_del.extend(
            [
                "APPTAINER_COMMAND",
                "APPTAINER_CONTAINER",
                "SINGULARITY_CONTAINER",
                "APPTAINER_BIND",
                "APPTAINER_NAME",
                "SINGULARITY_ENVIRONMENT",
                "SINGULARITY_NAME",
                "APPTAINER_APPNAME",
                "APPTAINER_ENVIRONMENT",
                "SINGULARITY_BIND",
            ]
        )
    for key in to_del:
        if key in env:
            del env[key]
    return env


def get_cmsenv(
    cmssw_path, python_cmd=None, crab_env=False, crab_type="", singularity_cmd=None
):
    script = f'source /cvmfs/cms.cern.ch/cmsset_default.sh; cd "{cmssw_path}"; eval `scramv1 runtime -sh`'
    if not python_cmd:
        python_cmd = "python3"
        _, cmssw_release_str = os.path.split(os.path.abspath(cmssw_path))
        cmssw_id_parts = cmssw_release_str.split("_")
        if len(cmssw_id_parts) > 1 and int(cmssw_id_parts[1]) < 10:
            python_cmd = "python"
    if crab_env:
        if python_cmd != "python":
            script += "; alias python=$(which python3)"
        script += f"; source /cvmfs/cms.cern.ch/common/crab-setup.sh {crab_type}"
    return get_env(script, python_cmd=python_cmd, singularity_cmd=singularity_cmd)


if __name__ == "__main__":
    script = " ".join(sys.argv[1:])
    # env = get_env(script)
    env = get_cmsenv(sys.argv[1])
    print(json.dumps(env, indent=2))
