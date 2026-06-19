import datetime
import json
import os
import re
import sys

if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(file_dir))
    __package__ = "RunKit"

from .run_tools import ps_call, repeat_until_success, adler32sum, PsCallError

COPY_TMP_SUFFIX = ".tmp"
COPY_TMP_LOCAL_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".gfal_copy_safe_tmp"
)
CHECK_WRITE_SUFFIX = ".check"


class FileInfo:
    def __init__(self, name=None, path=None, size=None, date=None, is_dir=None):
        self.name = name
        self.path = path
        self.size = size
        self.date = date
        self.is_dir = is_dir

    @property
    def full_name(self):
        return os.path.join(self.path, self.name)

    def __str__(self):
        date_str = (
            self.date.strftime("%Y-%m-%dT%H:%M") if self.date is not None else None
        )
        return f'name="{self.name}", path="{self.path}", size={self.size}, date={date_str}, is_dir={self.is_dir}'

    def __repr__(self):
        return self.__str__()


class GfalError(RuntimeError):
    def __init__(self, msg):
        super(GfalError, self).__init__(msg)


def get_voms_proxy_info():
    _, output, _ = ps_call(["voms-proxy-info"], catch_stdout=True, split="\n")
    info = {}
    for line in output:
        if len(line) == 0:
            continue
        match = re.match(r"^(.+) : (.+)", line)
        key = match.group(1).strip()
        info[key] = match.group(2)
    if "timeleft" in info:
        h, m, s = info["timeleft"].split(":")
        info["timeleft"] = float(h) + (float(m) + float(s) / 60.0) / 60.0
    return info


def get_voms_proxy_token(voms_token=None):
    if voms_token is None:
        return get_voms_proxy_info()["path"]
    return voms_token


def check_download(
    local_file,
    expected_adler32sum=None,
    raise_error=False,
    remote_file=None,
    remove_bad_file=False,
):
    if expected_adler32sum is not None:
        asum = adler32sum(local_file)
        if asum != expected_adler32sum:
            if remove_bad_file:
                os.remove(local_file)
            if raise_error:
                remote_name = remote_file if remote_file is not None else "file"
                raise RuntimeError(
                    f"Unable to copy {remote_name} from remote. Failed adler32sum check."
                    + f" {asum:x} != {expected_adler32sum:x}."
                )
            return False
    return True


def xrd_copy(
    input_remote_file,
    output_local_file,
    n_retries=4,
    n_retries_xrdcp=4,
    n_streams=1,
    retry_sleep_interval=10,
    expected_adler32sum=None,
    verbose=1,
    prefixes=[
        "root://cms-xrd-global.cern.ch/",
        "root://xrootd-cms.infn.it/",
        "root://cmsxrootd.fnal.gov/",
    ],
):
    def download(prefix):
        xrdcp_args = [
            "xrdcp",
            "--retry",
            str(n_retries_xrdcp),
            "--streams",
            str(n_streams),
        ]
        if os.path.exists(output_local_file):
            xrdcp_args.append("--continue")
        if verbose == 0:
            xrdcp_args.append("--silent")
        xrdcp_args.extend([f"{prefix}{input_remote_file}", output_local_file])
        ps_call(xrdcp_args, verbose=1)

        check_download(
            output_local_file,
            expected_adler32sum=expected_adler32sum,
            remove_bad_file=True,
            raise_error=True,
            remote_file=input_remote_file,
        )

    if os.path.exists(output_local_file):
        os.remove(output_local_file)

    if input_remote_file.startswith("/store/"):
        optlist = [(prefix,) for prefix in prefixes]
    else:
        optlist = [("",)]

    repeat_until_success(
        download,
        opt_list=optlist,
        n_retries=n_retries,
        retry_sleep_interval=retry_sleep_interval,
        exception=GfalError(f"Unable to copy {input_remote_file} from remote."),
        verbose=verbose,
    )


def create_tmp_local_file():
    if not os.path.exists(COPY_TMP_LOCAL_FILE):
        with open(COPY_TMP_LOCAL_FILE, "w") as f:
            f.write("0")
    return COPY_TMP_LOCAL_FILE


def gfal_env(voms_token):
    return {"X509_USER_PROXY": voms_token, "GFAL_PYTHONBIN": "/usr/bin/python3"}


def gfal_copy_safe(
    input_file,
    output_file,
    voms_token=None,
    number_of_streams=2,
    timeout=7200,
    expected_adler32sum=None,
    n_retries=4,
    retry_sleep_interval=10,
    copy_mode="copy_flag",
    verbose=1,
):
    voms_token = get_voms_proxy_token(voms_token)
    if expected_adler32sum is None:
        try:
            stat = gfal_stat(input_file, voms_token=voms_token)
            if stat["type"] == "regular file":
                expected_adler32sum = gfal_sum(
                    input_file, voms_token=voms_token, sum_type="adler32"
                )
        except GfalError as e:
            if verbose > 0:
                print(f'WARNING: gfal_sum failed for "{input_file}".\n{e}')
    if copy_mode not in ["copy_rename", "copy_flag"]:
        raise RuntimeError(f'gfal_copy_safe: unknown copy mode "{copy_mode}".')
    if copy_mode == "copy_flag":
        tmp_local_file = create_tmp_local_file()
    output_file_tmp = output_file + COPY_TMP_SUFFIX
    output_file_sum_target = (
        output_file if copy_mode == "copy_flag" else output_file_tmp
    )
    attempt = -1

    def download():
        nonlocal attempt
        attempt += 1
        active_verbose = min(verbose + attempt if verbose > 0 else 0, 2)
        if gfal_exists(output_file, voms_token=voms_token):
            gfal_rm(output_file, voms_token=voms_token, recursive=False)
        if gfal_exists(output_file_tmp, voms_token=voms_token):
            gfal_rm(output_file_tmp, voms_token=voms_token, recursive=False)
        if copy_mode == "copy_flag":
            gfal_copy(
                tmp_local_file,
                output_file_tmp,
                voms_token=voms_token,
                number_of_streams=number_of_streams,
                timeout=timeout,
                verbose=active_verbose,
            )
            gfal_copy(
                input_file,
                output_file,
                voms_token=voms_token,
                number_of_streams=number_of_streams,
                timeout=timeout,
                verbose=active_verbose,
            )
        elif copy_mode == "copy_rename":
            gfal_copy(
                input_file,
                output_file_tmp,
                voms_token=voms_token,
                number_of_streams=number_of_streams,
                timeout=timeout,
                verbose=active_verbose,
            )
        if expected_adler32sum is not None:
            output_adler32sum = gfal_sum(
                output_file_sum_target, voms_token=voms_token, sum_type="adler32"
            )
            if output_adler32sum != expected_adler32sum:
                raise GfalError(
                    f'Failed adler32sum check for "{output_file_sum_target}".'
                    f" {output_adler32sum:x} != {expected_adler32sum:x}."
                )
        if copy_mode == "copy_flag":
            gfal_rm(output_file_tmp, voms_token=voms_token, recursive=False)
        elif copy_mode == "copy_rename":
            gfal_rename(output_file_tmp, output_file, voms_token=voms_token)
            if not gfal_exists(output_file, voms_token=voms_token):
                raise GfalError(
                    f'Failed to rename "{output_file_tmp}" to "{output_file}".'
                )

    repeat_until_success(
        download,
        n_retries=n_retries,
        retry_sleep_interval=retry_sleep_interval,
        verbose=verbose,
        exception=GfalError(f'Unable to copy "{input_file}" to "{output_file}".'),
    )


def gfal_copy(
    input_file,
    output_file,
    voms_token=None,
    number_of_streams=2,
    timeout=7200,
    verbose=1,
):
    voms_token = get_voms_proxy_token(voms_token)
    try:
        catch_output = verbose == 0
        cmd = [
            "gfal-copy",
            "--parent",
            "--recursive",
            "--nbstreams",
            str(number_of_streams),
            "--timeout",
            str(timeout),
        ]
        if verbose > 1:
            n_v = min(3, verbose - 1)
            cmd.append("-" + "v" * n_v)
        cmd.extend([input_file, output_file])
        ps_call(
            cmd,
            shell=False,
            env=gfal_env(voms_token),
            verbose=verbose,
            catch_stdout=catch_output,
            catch_stderr=catch_output,
        )
    except PsCallError as e:
        raise GfalError(
            f'gfal_copy: unable to copy "{input_file}" to "{output_file}"\n{e}'
        ) from None


def gfal_ls(path, voms_token=None, catch_stderr=False, verbose=1):
    voms_token = get_voms_proxy_token(voms_token)
    try:
        _, output, _ = ps_call(
            ["gfal-ls", "--long", "--all", "--time-style", "long-iso", path],
            shell=False,
            env=gfal_env(voms_token),
            catch_stdout=True,
            catch_stderr=catch_stderr,
            split="\n",
            verbose=verbose,
        )
    except PsCallError as e:
        raise GfalError(f'gfal_ls: unable to list "{path}"\n{e}') from None
    files = []
    for line in output:
        if len(line) == 0:
            continue
        items = re.match(
            r"^([rwx\-d]+) +[0-9]+ +[0-9]+ +[0-9]+ +([0-9]+) +([0-9\-]+ [0-9:]+) +(.*)$",
            line,
        )
        if items is None:
            raise GfalError(f'gfal_ls: unable to parse "{line}"')
        file = FileInfo()
        file.name = items.group(4).strip()
        if file.name in [".", ".."]:
            continue
        if file.name == path:
            file.path, file.name = os.path.split(path)
        else:
            file.path = path
        file.size = int(items.group(2))
        file.date = datetime.datetime.strptime(items.group(3), "%Y-%m-%d %H:%M")
        file.is_dir = items.group(1).startswith("d")
        files.append(file)
    return files


def gfal_ls_recursive(path, voms_token=None, verbose=1):
    voms_token = get_voms_proxy_token(voms_token)
    all_files = []
    path_files = gfal_ls(path, voms_token=voms_token, verbose=verbose)
    for file in path_files:
        all_files.append(file)
        if file.is_dir:
            all_files.extend(
                gfal_ls_recursive(
                    file.full_name, voms_token=voms_token, verbose=verbose
                )
            )
    return sorted(set(all_files), key=lambda f: f.full_name)


def gfal_ls_safe(path, voms_token=None, catch_stderr=False, verbose=1):
    try:
        return gfal_ls(
            path, voms_token=voms_token, catch_stderr=catch_stderr, verbose=verbose
        )
    except GfalError:
        return None


def gfal_stat(path, voms_token=None):
    voms_token = get_voms_proxy_token(voms_token)
    result = {"size": None, "type": None}
    try:
        _, stdout, _ = ps_call(
            ["gfal-stat", path],
            shell=False,
            env=gfal_env(voms_token),
            catch_stdout=True,
            catch_stderr=True,
            decode=True,
            split="\n",
        )

        if len(stdout) > 1:
            match = re.match(r"  Size: ([0-9]+) *(.+)", stdout[1])
            if match is not None:
                result["size"] = int(match.group(1))
                result["type"] = match.group(2).strip()
    except PsCallError as e:
        pass
    return result


def gfal_exists(path, voms_token=None):
    voms_token = get_voms_proxy_token(voms_token)
    try:
        ps_call(
            ["gfal-stat", path],
            shell=False,
            env=gfal_env(voms_token),
            catch_stdout=True,
            catch_stderr=True,
        )
    except PsCallError as e:
        return False
    return True


def gfal_check_write(path, return_exception=False, voms_token=None, verbose=0):
    voms_token = get_voms_proxy_token(voms_token)
    target_path = path + CHECK_WRITE_SUFFIX
    tmp_local_file = create_tmp_local_file()
    result = (True, None)
    try:
        if gfal_exists(target_path, voms_token=voms_token):
            gfal_rm(target_path, voms_token=voms_token, recursive=False)
        gfal_copy(tmp_local_file, target_path, voms_token=voms_token, verbose=verbose)
        gfal_rm(target_path, voms_token=voms_token, verbose=verbose)
    except GfalError as e:
        result = (False, e)
    if return_exception:
        return result
    return result[0]


def gfal_sum(path, voms_token=None, sum_type="adler32"):
    voms_token = get_voms_proxy_token(voms_token)
    try:
        _, output, _ = ps_call(
            ["gfal-sum", path, sum_type],
            shell=False,
            env=gfal_env(voms_token),
            catch_stdout=True,
        )
        sum_str = output.split(" ")[-1]
        sum_int = int(sum_str, 16)
    except PsCallError as e:
        raise GfalError(
            f'gfal_sum: unable to get {sum_type} for "{path}"\n{e}'
        ) from None
    except ValueError as e:
        raise GfalError(
            f'gfal_sum: unable to parse {sum_type} for "{path}".'
            f"\ngfal-sum output:\n--------\n{output}--------\n{e}"
        ) from None
    return sum_int


def gfal_rm(path, voms_token=None, recursive=False, verbose=0, timeout=1800):
    voms_token = get_voms_proxy_token(voms_token)
    cmd = ["gfal-rm", "-t", str(timeout)]
    if recursive:
        cmd.append("-r")
    cmd.append(path)
    try:
        ps_call(
            cmd,
            shell=False,
            env=gfal_env(voms_token),
            catch_stdout=(verbose == 0),
            verbose=verbose,
        )
    except PsCallError as e:
        raise GfalError(f'gfal_rm: unable to remove "{path}"\n{e}') from None


def gfal_rm_recursive(path, voms_token=None, timeout=86400):
    gfal_rm(path, voms_token=voms_token, recursive=True, verbose=1, timeout=timeout)


def gfal_rename(path, new_path, voms_token=None):
    voms_token = get_voms_proxy_token(voms_token)
    try:
        ps_call(
            ["gfal-rename", path, new_path],
            shell=False,
            env=gfal_env(voms_token),
            catch_stdout=True,
        )
    except PsCallError as e:
        raise GfalError(
            f'gfal_rename: unable to rename "{path}" to "{new_path}"\n{e}'
        ) from None


def lfn_to_pfn(server, lfn):
    from rucio.client import Client

    client = Client()
    key = f"user.jdoe:{lfn}"
    result = client.lfns2pfns(server, [key])
    return result[key]


def path_to_pfn(path, *sub_paths):
    if path.startswith("T"):
        server, lfn = path.split(":")
        pfn = lfn_to_pfn(server, lfn)
    else:
        pfn = path
    return os.path.join(pfn, *sub_paths)


def get_local_site():
    local_conf = "/cvmfs/cms.cern.ch/SITECONF/local"
    if os.path.exists(local_conf) and os.path.islink(local_conf):
        return os.readlink(local_conf)
    return None


def get_distances(local_site, sites):
    distances = {}
    try:
        from rucio.client import Client
    except ImportError:
        try:
            _, out, _ = ps_call(
                """
        ARCH=$(uname -m)/$(/cvmfs/cms.cern.ch/common/cmsos | cut -d_ -f1 | sed 's|^[a-z]*|rhel|');
        echo /cvmfs/cms.cern.ch/rucio/$ARCH/py3/current;
        echo /cvmfs/cms.cern.ch/rucio/$ARCH/py3/current/lib/python*/site-packages""",
                shell=True,
                catch_stdout=True,
                split="\n",
            )
            sys.path.append(out[1])
            os.environ["RUCIO_HOME"] = out[0]
            from rucio.client import Client
        except:

            class Client:
                def get_distance(self, site1, site2):
                    return [{"distance": 1}]

    client = Client()
    for site in sites:
        if local_site is None or site == local_site:
            distances[site] = 0
        else:
            try:
                dist = client.get_distance(site, local_site)
            except:
                dist = []
            if len(dist) > 0:
                distances[site] = dist[0]["distance"]
            else:
                distances[site] = float("inf")
    return distances


def run_dasgoclient(
    query, inputDBS="global", json_output=False, timeout=None, verbose=0
):
    if inputDBS != "global":
        query += f" instance=prod/{inputDBS}"
    cmd = ["/cvmfs/cms.cern.ch/common/dasgoclient", "--query", query]
    if json_output:
        cmd.append("--json")
    env = {
        "PATH": "/usr/bin",
        "X509_USER_PROXY": os.environ["X509_USER_PROXY"],
        "HOME": os.environ.get("HOME", os.getcwd()),
    }
    split = None if json_output else "\n"
    _, output, _ = ps_call(
        cmd, catch_stdout=True, split=split, timeout=timeout, verbose=verbose, env=env
    )
    if json_output:
        return json.loads(output)
    return [line.strip() for line in output if len(line.strip()) > 0]


def das_file_site_info(file, inputDBS="global", verbose=0):
    return run_dasgoclient(
        f"site file={file}", inputDBS=inputDBS, json_output=True, verbose=verbose
    )


def das_file_pfns(
    file,
    disk_only=True,
    return_adler32=False,
    inputDBS="global",
    keep_rse=False,
    verbose=0,
):
    site_info = das_file_site_info(file, inputDBS=inputDBS, verbose=verbose)
    pfns_all = {}
    adler32 = None
    for entry in site_info:
        if "site" not in entry:
            continue
        for site in entry["site"]:
            if "pfns" not in site:
                continue
            for pfns_link, pfns_info in site["pfns"].items():
                pnfs_type = pfns_info.get("type", "UNKNOWN")
                if pnfs_type not in pfns_all:
                    pfns_all[pnfs_type] = set()
                entry = (pfns_link, pfns_info["rse"]) if keep_rse else pfns_link
                pfns_all[pnfs_type].add(entry)
            if "adler32" in site:
                site_adler32 = int(site["adler32"], 16)
                if adler32 is not None and adler32 != site_adler32:
                    raise RuntimeError(f"Inconsistent adler32 sum for {file}")
                adler32 = site_adler32
    if disk_only:
        pfns = pfns_all.get("DISK", set())
    else:
        pfns = pfns_all
    if return_adler32:
        return pfns, adler32
    return pfns


def copy_remote_file(
    input_remote_file,
    output_local_file,
    inputDBS="global",
    n_retries=4,
    retry_sleep_interval=10,
    custom_pfns_prefix="",
    voms_token=None,
    verbose=1,
):
    voms_token = get_voms_proxy_token(voms_token)
    from_das = input_remote_file.startswith("/store/")
    if from_das:
        pfns_info, adler32 = das_file_pfns(
            input_remote_file,
            disk_only=True,
            return_adler32=True,
            inputDBS=inputDBS,
            keep_rse=True,
            verbose=verbose,
        )
        sites = [rse for _, rse in pfns_info]
        local_site = get_local_site()
        distances = get_distances(local_site, sites)
        pfns_info = [(pfns, rse, distances[rse]) for pfns, rse in pfns_info]
        pfns_info = sorted(pfns_info, key=lambda x: (x[2], x[1]))
        if verbose > 0:
            print("Avaliable pfns:")
            for pfns, rse, dist in pfns_info:
                print(f"  {rse} (distance={dist}): {pfns}")
        pfns_list = [pfns for pfns, _, _ in pfns_info]
    else:
        if len(custom_pfns_prefix) > 0:
            file_pfns = custom_pfns_prefix + input_remote_file
        else:
            file_pfns = input_remote_file
        adler32 = gfal_sum(file_pfns, voms_token=voms_token, sum_type="adler32")
        pfns_list = [file_pfns]
    if os.path.exists(output_local_file):
        if adler32 is not None and check_download(
            output_local_file, expected_adler32sum=adler32
        ):
            return
        os.remove(output_local_file)

    if len(pfns_list) == 0:
        raise RuntimeError(
            f'Unable to find any remote location for "{input_remote_file}".'
        )

    def download(pfns):
        if verbose > 0:
            print(f"Trying to copy file from {pfns}")
        if pfns.startswith("root:") or pfns.startswith("/store/"):
            xrd_copy(
                pfns,
                output_local_file,
                expected_adler32sum=adler32,
                n_retries=1,
                prefixes=[""],
                verbose=verbose,
            )
        elif (
            pfns.startswith("srm:")
            or pfns.startswith("gsiftp")
            or pfns.startswith("davs:")
        ):
            gfal_copy_safe(
                pfns,
                output_local_file,
                voms_token,
                expected_adler32sum=adler32,
                n_retries=1,
            )
        else:
            raise RuntimeError('Skipping an unknown remote source "{pfns}".')

    repeat_until_success(
        download,
        opt_list=[(pfns,) for pfns in pfns_list],
        n_retries=n_retries,
        exception=GfalError(f"Unable to copy {input_remote_file} from remote."),
        retry_sleep_interval=retry_sleep_interval,
        verbose=verbose,
    )


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1]
    cmd_args = [f'"{arg}"' for arg in sys.argv[2:]]
    cmd_str = cmd + "(" + ",".join(cmd_args) + ")"
    print(f"> {cmd_str}")
    try:
        out = getattr(sys.modules[__name__], cmd)(*sys.argv[2:])
        if out is not None:
            try:
                out_str = json.dumps(out, indent=2)
            except TypeError:
                if type(out) == list:
                    out_str = "\n".join([str(o) for o in out])
                else:
                    out_str = out
            print(out_str)
    except RuntimeError as e:
        print(f"ERROR: {type(e).__name__} -- {e}")
        sys.exit(1)
