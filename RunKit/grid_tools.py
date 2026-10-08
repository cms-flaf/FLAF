import datetime
import errno
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import time
import uuid

if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(file_dir))
    __package__ = "RunKit"

from .run_tools import ps_call, repeat_until_success, adler32sum, PsCallError

COPY_TMP_SUFFIX = ".tmp"

# Marker of a `copy_rename` upload in progress. Unique per writer, because two jobs publishing
# the same target (a resubmission racing the job it replaced, or a duplicate) would otherwise
# share one tmp path and remove each other's upload. It is appended after the target's own
# extension so that nothing globbing `*.root` picks it up.
COPY_RENAME_TMP_PREFIX = ".flaf-tmp-"


def copy_rename_tmp_suffix():
    """A fresh, writer-unique suffix for a `copy_rename` upload."""
    return f"{COPY_RENAME_TMP_PREFIX}{os.getpid()}-{uuid.uuid4().hex[:12]}"


def is_copy_rename_tmp(name):
    """Whether `name` is an in-progress or orphaned `copy_rename` upload."""
    return COPY_RENAME_TMP_PREFIX in os.path.basename(name)


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
    """Path and remaining lifetime of the current proxy.

    `-dont-verify-ac` skips the attribute-certificate check: with a stale CRL for the VOMS
    server, plain `voms-proxy-info` exits 1 while the proxy is usable, and a job that reads
    its proxy while building targets dies before doing any work (309 of 1400 DSProd CRAB
    jobs, CERN batch nodes included). A missing or unreadable proxy still exits non-zero
    and raises, with the command's stderr in the exception.
    """
    _, output, _ = ps_call(
        ["voms-proxy-info", "-dont-verify-ac"],
        catch_stdout=True,
        catch_stderr=True,
        split="\n",
    )
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


def create_tmp_local_file():
    if not os.path.exists(COPY_TMP_LOCAL_FILE):
        with open(COPY_TMP_LOCAL_FILE, "w") as f:
            f.write("0")
    return COPY_TMP_LOCAL_FILE


def gfal_env(voms_token):
    # `gfal-ls --time-style long-iso` prints times in the client's timezone; UTC keeps a
    # listing's dates independent of the host and of daylight-saving changes.
    return {
        "X509_USER_PROXY": voms_token,
        "GFAL_PYTHONBIN": "/usr/bin/python3",
        "TZ": "UTC",
    }


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
    else:
        output_file_tmp = output_file + copy_rename_tmp_suffix()
    output_file_sum_target = (
        output_file if copy_mode == "copy_flag" else output_file_tmp
    )
    attempt = -1

    def download():
        nonlocal attempt
        attempt += 1
        active_verbose = min(verbose + attempt if verbose > 0 else 0, 2)
        # Only `copy_flag` copies onto the destination itself and has to clear it first. In
        # `copy_rename` mode the destination is created only by the rename below; removing it
        # here would leave nothing published for the whole duration of the upload.
        if copy_mode == "copy_flag" and gfal_exists(output_file, voms_token=voms_token):
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
            _rename_onto(output_file_tmp, output_file, voms_token=voms_token)
            if not gfal_exists(output_file, voms_token=voms_token):
                raise GfalError(
                    f'Failed to rename "{output_file_tmp}" to "{output_file}".'
                )

    try:
        repeat_until_success(
            download,
            n_retries=n_retries,
            retry_sleep_interval=retry_sleep_interval,
            verbose=verbose,
            exception=GfalError(f'Unable to copy "{input_file}" to "{output_file}".'),
        )
    except GfalError:
        if copy_mode == "copy_rename":
            # The tmp name is unique to this call, so no later attempt would ever remove
            # it: a full-size orphan next to the target. Best effort -- the storage may be
            # what failed.
            try:
                if gfal_exists(output_file_tmp, voms_token=voms_token):
                    gfal_rm(output_file_tmp, voms_token=voms_token, recursive=False)
            except GfalError:
                pass
        raise


def _rename_onto(tmp_file, output_file, voms_token):
    """Publish the verified upload `tmp_file` as `output_file`.

    On EOS over xrootd, gfal-rename replaces an existing target, and over davs davix sends
    MOVE without an Overwrite header, which RFC 4918 treats as "Overwrite: T". For a
    storage that refuses to rename onto an existing name anyway: an identical target is
    kept and the upload dropped; a different one is removed and the rename repeated, which
    leaves the target absent for the duration of those two namespace operations.
    """
    try:
        gfal_rename(tmp_file, output_file, voms_token=voms_token)
        return
    except GfalError:
        if not gfal_exists(output_file, voms_token=voms_token):
            raise
    tmp_sum = gfal_sum(tmp_file, voms_token=voms_token, sum_type="adler32")
    if gfal_sum(output_file, voms_token=voms_token, sum_type="adler32") == tmp_sum:
        gfal_rm(tmp_file, voms_token=voms_token, recursive=False)
        return
    gfal_rm(output_file, voms_token=voms_token, recursive=False)
    gfal_rename(tmp_file, output_file, voms_token=voms_token)


def gfal_copy(
    input_file,
    output_file,
    voms_token=None,
    number_of_streams=2,
    timeout=7200,
    force=False,
    verbose=1,
):
    """Copy `input_file` to `output_file`.

    gfal-copy refuses an existing destination unless `force` is set, which overwrites it.
    """
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
        if force:
            cmd.append("--force")
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


def gfal_ls(path, voms_token=None, catch_stderr=False, verbose=1, timeout=None):
    voms_token = get_voms_proxy_token(voms_token)
    cmd = ["gfal-ls", "--long", "--all", "--time-style", "long-iso"]
    if timeout is not None:
        cmd += ["--timeout", str(int(timeout))]
    try:
        _, output, _ = ps_call(
            cmd + [path],
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
    """List `path`, or None if that did not work for any reason.

    Only for best-effort callers; anything that decides whether a path exists needs
    `gfal_ls_checked`, which tells an absent path from a listing that failed.
    """
    try:
        return gfal_ls(
            path, voms_token=voms_token, catch_stderr=catch_stderr, verbose=verbose
        )
    except GfalError:
        return None


# The last line a gfal CLI writes when it fails: `gfal-<cmd> error: <errno> (<strerror>) - ...`
_GFAL_ERROR_RE = re.compile(r"gfal-[a-z-]+ error: (\d+) \(")


def is_absent_error(err):
    """Whether a gfal error says that the path does not exist.

    Decided by the errno that gfal reports (ENOENT), which it does for xrootd ("Failed to
    stat file") and for davs ("HTTP 404 : File not found"). The text "No such file or
    directory" alone is not evidence: davix prints it when the proxy file is missing, before
    failing with "gfal-ls error: 1 (Operation not permitted)" for every path.
    """
    codes = _GFAL_ERROR_RE.findall(str(err))
    return len(codes) > 0 and int(codes[-1]) == errno.ENOENT


def gfal_ls_checked(path, voms_token=None, attempts=3, delay=2.0, timeout=300):
    """List `path`; return None only when gfal says that it is not there.

    Any other failure (a timeout, an SSL error, a missing credential, an endpoint under
    load) is retried with a growing delay and then raised as GfalError. A caller that reads
    "could not list" as "not there" concludes that a product is missing: in DSProd one
    failed listing per job turned into 1400 failed CRAB jobs. Each attempt is bounded by
    `timeout` seconds; gfal's own default is half an hour against an endpoint that hangs.
    """
    for attempt in range(1, attempts + 1):
        try:
            return gfal_ls(
                path,
                voms_token=voms_token,
                catch_stderr=True,
                verbose=0,
                timeout=timeout,
            )
        except GfalError as e:
            if is_absent_error(e):
                return None
            if attempt == attempts:
                raise
            time.sleep(delay * attempt)
    # Reached only with attempts < 1: nothing was listed, which must not read as "absent".
    raise GfalError(f"gfal_ls_checked: {attempts} attempts is not a listing of {path}")


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


def gfal_sum(path, voms_token=None, sum_type="adler32", timeout=None):
    voms_token = get_voms_proxy_token(voms_token)
    try:
        _, output, _ = ps_call(
            ["gfal-sum", path, sum_type],
            shell=False,
            env=gfal_env(voms_token),
            catch_stdout=True,
            timeout=timeout,
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


# Persistent (server, lfn) -> pfn cache. The mapping is determined by an RSE's
# protocol configuration and changes very rarely, so caching it lets law commands keep
# working through transient Rucio outages (issue #115): once a base path has been
# resolved, it is reused without contacting Rucio again.
_lfn_pfn_cache = None


def _lfn_pfn_cache_path():
    override = os.environ.get("FLAF_LFN_PFN_CACHE")
    if override:
        return override
    base = os.environ.get("ANALYSIS_DATA_PATH") or os.path.join(
        os.path.expanduser("~"), ".flaf"
    )
    return os.path.join(base, "lfn_pfn_cache.json")


def _load_lfn_pfn_cache():
    try:
        with open(_lfn_pfn_cache_path(), "r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _store_lfn_pfn_cache(cache):
    # Best-effort persistence: an atomic rename keeps concurrent writers from
    # corrupting the file, and any I/O failure is ignored (the cache is an optimisation).
    path = _lfn_pfn_cache_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(cache, f)
        os.replace(tmp, path)
    except OSError:
        pass


def lfn_to_pfn(server, lfn):
    global _lfn_pfn_cache
    if _lfn_pfn_cache is None:
        _lfn_pfn_cache = _load_lfn_pfn_cache()
    key = f"{server}\t{lfn}"
    if key in _lfn_pfn_cache:
        return _lfn_pfn_cache[key]

    rucio_key = f"user.jdoe:{lfn}"
    try:
        client = get_rucio_client()
        pfn = client.lfns2pfns(server, [rucio_key])[rucio_key]
    except Exception as e:
        raise RuntimeError(
            f"lfn_to_pfn: unable to resolve PFN for {server}:{lfn} and no cached value "
            f"is available. Rucio may be unavailable ({type(e).__name__}: {e})."
        ) from None

    _lfn_pfn_cache[key] = pfn
    _store_lfn_pfn_cache(_lfn_pfn_cache)
    return pfn


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


_rucio_client = None


def get_rucio_client():
    """Return a cached Rucio client, setting up the client library from cvmfs if needed."""
    global _rucio_client
    if _rucio_client is not None:
        return _rucio_client
    try:
        from rucio.client import Client
    except ImportError:
        # Pin the Rucio version instead of the volatile 'current' symlink (issue #146),
        # matching env.sh; fall back to 'current' if the pinned version is unavailable.
        _, out, _ = ps_call(
            """
        VER=${FLAF_RUCIO_VERSION:-39.2.0};
        ARCH=$(uname -m)/$(/cvmfs/cms.cern.ch/common/cmsos | cut -d_ -f1 | sed 's|^[a-z]*|rhel|');
        RUCIO_DIR=/cvmfs/cms.cern.ch/rucio/$ARCH/py3/$VER;
        [ -e $RUCIO_DIR/bin/rucio ] || RUCIO_DIR=/cvmfs/cms.cern.ch/rucio/$ARCH/py3/current;
        echo $RUCIO_DIR;
        echo $RUCIO_DIR/lib/python*/site-packages""",
            shell=True,
            catch_stdout=True,
            split="\n",
        )
        sys.path.append(out[1])
        os.environ["RUCIO_HOME"] = out[0]
        from rucio.client import Client
    if "RUCIO_ACCOUNT" not in os.environ and "USER" in os.environ:
        os.environ["RUCIO_ACCOUNT"] = os.environ["USER"]
    _rucio_client = Client()
    return _rucio_client


def get_distances(local_site, sites):
    distances = {}
    try:
        client = get_rucio_client()
    except Exception:
        client = None
    for site in sites:
        if local_site is None or site == local_site:
            distances[site] = 0
        elif client is None:
            distances[site] = 1
        else:
            try:
                dist = client.get_distance(site, local_site)
            except Exception:
                dist = []
            if len(dist) > 0:
                distances[site] = dist[0]["distance"]
            else:
                distances[site] = float("inf")
    return distances


def rucio_list_files(dataset, scope="cms"):
    """List the files (LFNs) of a CMS dataset with their size and adler32 checksum.

    A CMS "dataset" path (e.g. /A/B/NANOAODSIM) is a Rucio container; list_files
    traverses it down to the individual file DIDs.
    """
    client = get_rucio_client()
    files = []
    for entry in client.list_files(scope, dataset):
        files.append(
            {
                "name": entry["name"],
                "bytes": entry.get("bytes"),
                "adler32": entry.get("adler32"),
            }
        )
    return files


def rucio_list_replicas(files, scope="cms", schemes=("root", "davs", "gsiftp")):
    """Return replica information for a list of LFNs.

    Result maps each LFN to a dict with:
      "pfns"      : {"DISK": [(pfn, rse), ...], "TAPE": [...]}
      "available" : True if at least one DISK replica is in state AVAILABLE
      "adler32"   : checksum string (or None)
      "bytes"     : file size (or None)
    """
    if isinstance(files, str):
        files = [files]
    client = get_rucio_client()
    dids = [{"scope": scope, "name": f} for f in files]
    result = {}
    for rep in client.list_replicas(
        dids, schemes=list(schemes), ignore_availability=False
    ):
        states = rep.get("states", {})
        pfns = {}
        available = False
        for pfns_link, pfns_info in rep.get("pfns", {}).items():
            pfns_type = pfns_info.get("type", "UNKNOWN")
            rse = pfns_info.get("rse")
            pfns.setdefault(pfns_type, []).append((pfns_link, rse))
            if pfns_type == "DISK" and states.get(rse) == "AVAILABLE":
                available = True
        result[rep["name"]] = {
            "pfns": pfns,
            "available": available,
            "adler32": rep.get("adler32"),
            "bytes": rep.get("bytes"),
        }
    return result


# DAS (dasgoclient) query helpers. No longer used by the default file-discovery path
# (which goes through Rucio, above), but retained for future use cases that Rucio does
# not cover -- e.g. per-file event counts, or the phys03 instance for USER datasets.
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


def das_dataset_file_info(dataset, inputDBS="global", timeout=600, verbose=0):
    """Per-file event counts and sizes of a CMS dataset, ``{lfn: {n_events, size}}``.

    Rucio reports file sizes but leaves the CMS ``events`` field empty, so this is the
    only cheap source of event counts for centrally produced NanoAOD.  One query covers
    a whole dataset (~1 s for 3k files).  Returns an empty dict on any failure: event
    counts are an optimisation for job-cost estimation, never a requirement.
    """
    try:
        entries = run_dasgoclient(
            f"file dataset={dataset}",
            inputDBS=inputDBS,
            json_output=True,
            timeout=timeout,
            verbose=verbose,
        )
    except Exception as e:
        print(f"das_dataset_file_info: query for {dataset} failed: {e}")
        return {}
    info = {}
    for entry in entries or []:
        for file_entry in entry.get("file", []):
            name = file_entry.get("name")
            if not name:
                continue
            n_events = file_entry.get("nevents")
            size = file_entry.get("size")
            info[name] = {
                "n_events": int(n_events) if n_events else None,
                "size": int(size) if size else None,
            }
    return info


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
    n_retries=4,
    retry_sleep_interval=10,
    custom_pfns_prefix="",
    voms_token=None,
    verbose=1,
):
    """Copy a remote file to `output_local_file` (see copy_with_failover).

    A `/store/...` LFN is read from its Rucio disk replicas, with the CMS xrootd federation
    as the last source, and verified against the size and adler32 that Rucio records;
    anything else is a single URL (prefixed by `custom_pfns_prefix`), verified against its
    adler32. An intact `output_local_file` is kept as it is.
    """
    voms_token = get_voms_proxy_token(voms_token)
    size = None
    if input_remote_file.startswith("/store/"):
        info = rucio_replica_info(
            input_remote_file,
            retry_sleep_interval=retry_sleep_interval,
            verbose=verbose,
        )
        replicas = info["pfns"].get("DISK", [])
        if len(replicas) == 0:
            raise GfalError(f'No disk replica of "{input_remote_file}" in Rucio.')
        adler32 = int(info["adler32"], 16) if info.get("adler32") else None
        size = info.get("bytes")
        distances = get_distances(get_local_site(), {rse for _, rse in replicas})
        # A site without a known distance (inf) still comes before the federation.
        sources = [
            (pfns, rse, min(distances[rse], sys.maxsize)) for pfns, rse in replicas
        ]
        sources.append(
            (COPY_FEDERATION_PREFIX + input_remote_file, "xrootd federation", math.inf)
        )
    else:
        file_pfns = custom_pfns_prefix + input_remote_file
        adler32 = gfal_sum(file_pfns, voms_token=voms_token, timeout=COPY_FIRST_TIMEOUT)
        sources = [(file_pfns, None, 0)]
    if os.path.exists(output_local_file):
        if adler32 is not None and check_download(
            output_local_file, expected_adler32sum=adler32
        ):
            return
        os.remove(output_local_file)

    copy_with_failover(
        sources,
        output_local_file,
        size=size,
        expected_adler32sum=adler32,
        voms_token=voms_token,
        n_rounds=n_retries,
        round_sleep_interval=retry_sleep_interval,
        verbose=verbose,
    )


def rucio_replica_info(lfn, n_tries=3, retry_sleep_interval=10, verbose=1):
    """rucio_list_replicas entry of one LFN, retried: a transient Rucio error (e.g. HTTP 503)
    should not fail a job. Without it the copy cannot be verified, so it is not skipped.
    """
    for attempt in range(n_tries):
        try:
            info = rucio_list_replicas([lfn]).get(lfn)
            if info is None:
                raise GfalError(f'"{lfn}" is not known to Rucio.')
            return info
        except Exception as e:
            if attempt == n_tries - 1:
                raise GfalError(f'Unable to query Rucio for "{lfn}": {e}') from None
            if verbose > 0:
                print(f"Rucio query for {lfn} failed ({e}), retrying.")
            time.sleep(retry_sleep_interval * 3**attempt)


# Limits of one copy attempt in the first round (round 0):
# - timeout: the attempt is stopped; it gives the file at least COPY_MIN_RATE. Each later round
#   multiplies it by COPY_TIMEOUT_GROWTH, so a slow but working connection still gets through.
# - expected: past it, an attempt that is not about to finish is joined by one on the next
#   source. It grows with the timeout.
# - stall: the attempt is stopped when its partial file has not changed for this long. The copy
#   tools write the data as it arrives, in steps of 8-13 MB (xrdcp 6, gfal-copy over davs), so
#   this stops only copies slower than ~30 kB/s. It does not grow: a source that never
#   delivers costs this much in every round.
COPY_FIRST_TIMEOUT = 300
COPY_MIN_RATE = 1e6  # bytes/s
COPY_UNKNOWN_SIZE_TIMEOUT = 1800
COPY_EXPECTED_OVERHEAD = 60
COPY_EXPECTED_RATE = 10e6  # bytes/s
COPY_UNKNOWN_SIZE_EXPECTED = 300
COPY_STALL_TIMEOUT = 300
COPY_TIMEOUT_GROWTH = 3
COPY_MAX_TIMEOUT = 4 * 3600
COPY_MAX_PARALLEL = 2
COPY_MAX_TOTAL_TIME = 6 * 3600
COPY_FEDERATION_PREFIX = "root://cms-xrd-global.cern.ch/"
_COPY_SCHEME_ORDER = ("root:", "davs:", "https:", "gsiftp:", "srm:")

# Site -> number of failed or overtaken attempts in this process (a job copies the inputs of
# all its branches); such sites are tried after the others.
_copy_failed_rses = {}


def _note_failed_rse(rse):
    if rse is not None:
        _copy_failed_rses[rse] = _copy_failed_rses.get(rse, 0) + 1


def copy_attempt_limits(size, round_idx):
    """(timeout, expected, stall) in seconds of an attempt in round `round_idx`."""
    if size:
        timeout = max(COPY_FIRST_TIMEOUT, size / COPY_MIN_RATE)
        expected = COPY_EXPECTED_OVERHEAD + size / COPY_EXPECTED_RATE
    else:
        timeout = COPY_UNKNOWN_SIZE_TIMEOUT
        expected = COPY_UNKNOWN_SIZE_EXPECTED
    scale = COPY_TIMEOUT_GROWTH**round_idx
    timeout = min(COPY_MAX_TIMEOUT, timeout * scale)
    return timeout, min(expected * scale, timeout / 2), min(COPY_STALL_TIMEOUT, timeout)


def copy_source_order(sources):
    """Sources of one round: the first endpoint of every site, nearest site first, before a
    second endpoint of a site already tried (a site that is down usually takes all its
    endpoints with it). Sites that failed earlier in this process come after the others, and
    sites at the same distance are drawn in random order, so that the jobs of a dataset do
    not all start on the same site."""

    def scheme_rank(pfns):
        for idx, prefix in enumerate(_COPY_SCHEME_ORDER):
            if pfns.startswith(prefix):
                return idx
        return len(_COPY_SCHEME_ORDER)

    draw = {rse: random.random() for rse in {s[1] for s in sources}}
    ranked = sorted(
        sources,
        key=lambda s: (
            _copy_failed_rses.get(s[1], 0) > 0,
            s[2],
            draw[s[1]],
            scheme_rank(s[0]),
        ),
    )
    first, rest, seen = [], [], set()
    for source in ranked:
        (rest if source[1] in seen else first).append(source)
        seen.add(source[1])
    return first + rest


def _copy_command(source, output_file, timeout, voms_token):
    if source.startswith("root:"):
        cmd = ["xrdcp", "--force", "--nopbar", "--streams", "1", source, output_file]
        return cmd, dict(os.environ, X509_USER_PROXY=voms_token)
    if source.startswith(("davs:", "https:", "gsiftp:", "srm:")):
        cmd = [
            "gfal-copy",
            "--force",
            "--timeout",
            str(int(timeout)),
            source,
            "file://" + os.path.abspath(output_file),
        ]
        return cmd, gfal_env(voms_token)
    raise GfalError(f'Unknown remote source "{source}".')


class _CopyAttempt:
    """One xrdcp/gfal-copy process writing `output_file`. It runs in its own process group,
    which is killed as a whole, so that a copy run by a wrapper script is stopped too.
    """

    def __init__(self, source, rse, output_file, limits, voms_token):
        self.source = source
        self.rse = rse
        self.output_file = output_file
        self.log_file = output_file + ".log"
        self.timeout, self.expected, self.stall = limits
        cmd, env = _copy_command(source, output_file, self.timeout, voms_token)
        with open(self.log_file, "w") as log:
            self.proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        self.start = self.last_growth = time.monotonic()
        self.size = -1

    def elapsed(self):
        return time.monotonic() - self.start

    def check_progress(self):
        """None while the attempt may continue, else why it has to stop."""
        if self.elapsed() >= self.timeout:
            return f"no result after {self.timeout:.0f} s"
        try:
            size = os.path.getsize(self.output_file)
        except FileNotFoundError:
            size = -1
        # Any change counts: `gfal-copy --force` first removes a file left at the same path.
        if size != self.size:
            self.size, self.last_growth = size, time.monotonic()
        elif time.monotonic() - self.last_growth >= self.stall:
            return f"no data for {self.stall:.0f} s"
        return None

    def is_late(self, file_size):
        """Past its expected time and either without data for that long or, at its average
        rate so far, not done within another expected time, so that a new attempt is likely
        to finish first."""
        if self.elapsed() < self.expected:
            return False
        if time.monotonic() - self.last_growth >= self.expected:
            return True
        if not file_size or self.size <= 0:
            return True
        return (file_size - self.size) * self.elapsed() / self.size > self.expected

    def kill(self):
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.proc.wait()

    def log_tail(self, n_lines=5):
        try:
            with open(self.log_file, "r", errors="replace") as log:
                return "".join(log.readlines()[-n_lines:]).strip()
        except OSError:
            return ""

    def cleanup(self):
        for path in (self.output_file, self.log_file):
            if os.path.exists(path):
                os.remove(path)


def copy_with_failover(
    sources,
    output_local_file,
    size=None,
    expected_adler32sum=None,
    voms_token=None,
    n_rounds=4,
    round_sleep_interval=10,
    poll_interval=1,
    verbose=1,
):
    """Copy a file to `output_local_file` from the first of `sources` [(pfns, site, distance)]
    that delivers it intact (exit code 0, `size` and `expected_adler32sum` when known).

    Every attempt is one xrdcp/gfal-copy, stopped when it fails, stalls or runs out of time
    (copy_attempt_limits); the next source is then tried. An attempt that is late
    (_CopyAttempt.is_late) is joined by one on the next source, at most COPY_MAX_PARALLEL at
    a time, and the first intact copy wins. Every source is tried in up to `n_rounds` rounds
    with growing limits, each `round_sleep_interval` x round seconds after its previous
    attempt ended, and the whole copy gives up after COPY_MAX_TOTAL_TIME.
    """
    voms_token = get_voms_proxy_token(voms_token)
    pending = [
        (round_idx, source, copy_attempt_limits(size, round_idx))
        for round_idx in range(n_rounds)
        for source in copy_source_order(sources)
    ]
    running, failures, n_started = [], [], 0
    last_end = {}  # source -> end of its last attempt
    start = time.monotonic()

    def fail(source, rse, reason):
        _note_failed_rse(rse)
        last_end[source] = time.monotonic()
        failures.append(f"{source}: {reason}")
        if verbose > 0:
            print(f"Copy attempt from {source} failed: {reason}")

    def give_up(reason):
        raise GfalError(
            f"Unable to copy {output_local_file}: {reason}\n  " + "\n  ".join(failures)
        )

    def ready(entry, now):
        """Not being copied from, and the pause after the source's last attempt is over."""
        round_idx, (pfns, _, _), _ = entry
        if any(attempt.source == pfns for attempt in running):
            return False
        return now >= last_end.get(pfns, -math.inf) + round_sleep_interval * round_idx

    try:
        while True:
            for attempt in list(running):
                return_code = attempt.proc.poll()
                if return_code is None:
                    reason = attempt.check_progress()
                    if reason is None:
                        continue
                    attempt.kill()
                elif return_code != 0:
                    reason = f"exit code {return_code}: {attempt.log_tail()}"
                elif not os.path.exists(attempt.output_file):
                    reason = "no file written"
                elif size is not None and os.path.getsize(attempt.output_file) != size:
                    reason = (
                        f"{os.path.getsize(attempt.output_file)} bytes, expected {size}"
                    )
                elif not check_download(attempt.output_file, expected_adler32sum):
                    reason = "adler32 mismatch"
                else:
                    os.replace(attempt.output_file, output_local_file)
                    # A site that was overtaken after its expected time counts as failed.
                    for other in running:
                        if (
                            other.rse != attempt.rse
                            and other.elapsed() >= other.expected
                        ):
                            _note_failed_rse(other.rse)
                    if verbose > 0:
                        print(
                            f"Copied from {attempt.source} in {attempt.elapsed():.0f} s"
                        )
                    return
                running.remove(attempt)
                attempt.cleanup()
                fail(attempt.source, attempt.rse, reason)

            if time.monotonic() - start >= COPY_MAX_TOTAL_TIME:
                give_up(f"no copy within {COPY_MAX_TOTAL_TIME:.0f} s")

            now = time.monotonic()
            entry = next((e for e in pending if ready(e, now)), None)
            if (
                entry is not None
                and len(running) < COPY_MAX_PARALLEL
                and all(attempt.is_late(size) for attempt in running)
            ):
                pending.remove(entry)
                round_idx, (pfns, rse, _), limits = entry
                if verbose > 0:
                    print(
                        f"{'Joining with' if running else 'Trying'} {pfns} (round"
                        f" {round_idx + 1}, timeout {limits[0]:.0f} s, expected"
                        f" {limits[1]:.0f} s, stall {limits[2]:.0f} s)"
                    )
                part_file = f"{output_local_file}.part{n_started}"
                n_started += 1
                try:
                    running.append(
                        _CopyAttempt(pfns, rse, part_file, limits, voms_token)
                    )
                except (OSError, GfalError) as e:
                    for path in (part_file, part_file + ".log"):
                        if os.path.exists(path):
                            os.remove(path)
                    fail(pfns, rse, f"unable to start the copy: {e}")
                continue
            if not running and not pending:
                give_up("every source failed")
            time.sleep(poll_interval)
    finally:
        for attempt in running:
            attempt.kill()
            attempt.cleanup()


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
