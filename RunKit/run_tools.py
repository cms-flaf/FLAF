import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import zlib
from threading import Timer


class PsCallError(RuntimeError):
    def __init__(self, cmd_str, return_code, additional_message=None):
        msg = f'Error while running "{cmd_str}".'
        if return_code is not None:
            msg += f" Error code: {return_code}"
        if additional_message is not None:
            msg += f" {additional_message}"
        super(PsCallError, self).__init__(msg)
        self.cmd_str = cmd_str
        self.return_code = return_code
        self.message = additional_message


def ps_call(
    cmd,
    shell=False,
    catch_stdout=False,
    catch_stderr=False,
    decode=True,
    split=None,
    print_output=False,
    expected_return_codes=[0],
    env=None,
    cwd=None,
    timeout=None,
    singularity_cmd=None,
    verbose=0,
):
    if isinstance(cmd, str):
        cmd = [cmd]
    if shell:
        if not (isinstance(cmd, list) and len(cmd) == 1):
            raise ValueError(
                "cmd must be a string or a list with a single element when shell=True"
            )
    if singularity_cmd is not None:
        env_list = []
        if env is not None:
            for key in ["PATH", "LD_LIBRARY_PATH"]:
                if key in env:
                    env_list.append(f'{key}="{env[key]}"')
        if shell:
            env_str = " ".join(env_list)
            if len(env_str) > 0:
                env_str += " "
            full_cmd = [f"{singularity_cmd} --command-to-run '{env_str}{cmd[0]}'"]
        else:
            full_cmd = [singularity_cmd, "--command-to-run", "env"] + env_list + cmd
        if env is not None:
            env_str = ""
            for key in ["PATH", "LD_LIBRARY_PATH"]:
                if key in env:
                    env_str += f'{key}="{env[key]}" '

    else:
        full_cmd = cmd
    cmd_str = []
    for s in cmd:
        if " " in s and not shell:
            s = f"'{s}'"
        cmd_str.append(s)
    cmd_str = " ".join(cmd_str)
    if verbose > 0:
        if singularity_cmd is not None:
            print(f"Entering {singularity_cmd} ...", file=sys.stderr)
        print(f">> {cmd_str}", file=sys.stderr)
    kwargs = {
        "shell": shell,
    }
    if catch_stdout:
        kwargs["stdout"] = subprocess.PIPE
    if catch_stderr:
        if print_output:
            kwargs["stderr"] = subprocess.STDOUT
        else:
            kwargs["stderr"] = subprocess.PIPE
    if env is not None:
        kwargs["env"] = env
    if cwd is not None:
        kwargs["cwd"] = cwd

    # psutil.Process.children does not work.
    def kill_proc(pid):
        child_list = subprocess.run(
            ["ps", "h", "--ppid", str(pid)], capture_output=True, encoding="utf-8"
        )
        for line in child_list.stdout.split("\n"):
            child_info = line.split(" ")
            child_info = [s for s in child_info if len(s) > 0]
            if len(child_info) > 0:
                child_pid = child_info[0]
                kill_proc(child_pid)
        subprocess.run(["kill", "-9", str(pid)], capture_output=True)

    proc = subprocess.Popen(full_cmd, **kwargs)

    def kill_main_proc():
        print(f"\nTimeout is reached while running:\n\t{cmd_str}", file=sys.stderr)
        print(f"Killing process tree...", file=sys.stderr)
        print(f"Main process PID = {proc.pid}", file=sys.stderr)
        kill_proc(proc.pid)

    timer = Timer(timeout, kill_main_proc) if timeout is not None else None
    try:
        if timer is not None:
            timer.start()
        if catch_stdout and print_output:
            output = b""
            err = b""
            for line in proc.stdout:
                output += line
                print(line.decode("utf-8"), end="")
            proc.stdout.close()
            proc.wait()
        else:
            output, err = proc.communicate()
    finally:
        if timer is not None:
            timer.cancel()
    if (
        expected_return_codes is not None
        and proc.returncode not in expected_return_codes
    ):
        raise PsCallError(cmd_str, proc.returncode)
    if decode:
        if catch_stdout:
            output_decoded = output.decode("utf-8")
            if split is None:
                output = output_decoded
            else:
                output = output_decoded.split(split)
        if catch_stderr:
            err_decoded = err.decode("utf-8")
            if split is None:
                err = err_decoded
            else:
                err = err_decoded.split(split)

    return proc.returncode, output, err


def timestamp_str(timestamp=None):
    t = timestamp or datetime.datetime.now()
    return t.strftime("%Y-%m-%d %H:%M:%S")


def print_ts(msg, prefix="", timestamp=None, *args, **kwargs):
    print(f"{prefix}[{timestamp_str(timestamp=timestamp)}] {msg}", *args, **kwargs)


class PrintOpt:
    def __init__(self, min_interval=60):
        self.last_print = None
        self.min_interval = min_interval

    def __call__(self, msg, force=False, prefix="", *args, **kwargs):
        now = datetime.datetime.now()
        if force or (
            self.last_print is None
            or (now - self.last_print).total_seconds() >= self.min_interval
        ):
            print_ts(msg, prefix=prefix, timestamp=now, *args, **kwargs)
            self.last_print = now


def update_kerberos_ticket(verbose=1):
    ps_call(["kinit", "-R"], verbose=verbose)


def timed_call_wrapper(fn, update_interval, verbose=0):
    last_update = None

    def update(*args, **kwargs):
        nonlocal last_update
        now = datetime.datetime.now()
        delta_t = (
            (now - last_update).total_seconds()
            if last_update is not None
            else float("inf")
        )
        if verbose > 0:
            print(f"timed_call for {fn.__name__}: delta_t = {delta_t} seconds")
        if delta_t >= update_interval:
            fn(*args, **kwargs)
            last_update = now

    return update


def adler32sum(file_name):
    block_size = 256 * 1024 * 1024
    asum = 1
    with open(file_name, "rb") as f:
        while data := f.read(block_size):
            asum = zlib.adler32(data, asum)
    return asum


def repeat_until_success(
    fn, opt_list=([],), exception=None, n_retries=4, retry_sleep_interval=10, verbose=1
):
    for n in range(n_retries):
        for opt in opt_list:
            try:
                fn(*opt)
                return True
            except:
                if verbose > 0:
                    print(traceback.format_exc())
        if n != n_retries - 1:
            sleep_interval = retry_sleep_interval ** (n + 1)
            if verbose > 0:
                print(f"Waiting for {sleep_interval} seconds before the next try.")
            time.sleep(sleep_interval)

    if exception is not None:
        raise exception
    return False


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def check_root_file_integrity(file_name, tmp_file=None, verbose=1):
    if tmp_file is None:
        tmp_file_desc, tmp_file = tempfile.mkstemp()
        os.close(tmp_file_desc)
    try:
        ret_code, std_out, std_err = ps_call(
            ["hadd", "-f", "-O", tmp_file, file_name],
            catch_stderr=True,
            catch_stdout=True,
            decode=True,
            split="\n",
            verbose=verbose,
        )
        all_ok = True
        for line in std_err:
            if line.lower().startswith("error"):
                all_ok = False
                if verbose > 0:
                    print(line, file=sys.stderr)
                else:
                    break
        return all_ok
    except PsCallError as e:
        if verbose > 0:
            print(
                f"Error while checking integrity of {file_name}: {e}", file=sys.stderr
            )
        return False
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1]
    out = getattr(sys.modules[__name__], cmd)(*sys.argv[2:])
    if out is not None:
        out_t = type(out)
        if out_t in [list, dict]:
            print(json.dumps(out, indent=2))
        else:
            print(out)
