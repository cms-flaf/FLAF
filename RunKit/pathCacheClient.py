"""Client of the path-cache server (pathCacheServer.py) of this user on this machine.

The server listens on a Unix socket in a directory only its user can enter, by default
/tmp/flaf-path-cache-<uid>/ (FLAF_PATH_CACHE_DIR overrides it). ensure_local_server() starts it
when it is not running, so at most one runs per user and machine; its state survives restarts
in a file next to the socket.
"""

import fcntl
import os
import socket
import stat
import subprocess
import sys
import threading
import time

LOCAL_SERVER_DIR_ENV = "FLAF_PATH_CACHE_DIR"
# Named after the protocol version: a server speaking another protocol listens elsewhere.
SOCKET_NAME = "server_v1.sock"
STATE_NAME = "cache.json"
# sun_path holds 108 bytes on Linux, the terminating null included
MAX_SOCKET_PATH = 107
SERVER_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pathCacheServer.py"
)


class ServerUnavailable(OSError):
    """The server could not be reached, or stopped answering in the middle of a request."""


def _connect(address, timeout):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(address)
    except OSError:
        sock.close()
        raise
    return sock


def _send_all(sock, data, failure):
    try:
        sock.sendall(data)
    except OSError as e:
        failure.append(e)


def _read_line(file):
    line = file.readline()
    if not line:
        raise ServerUnavailable("the path-cache server closed the connection")
    return line.decode("utf-8").strip()


def _set_value(path, status):
    if status is None or status == "u":
        return "u"
    if status == "U":
        return "U"
    if isinstance(status, bool):
        return "1" if status else "0"
    raise ValueError(f"Invalid status value: {status!r} for path: {path!r}")


def _pipelined(address, timeout, lines):
    """Send `lines` on one connection while their answers are read; returns the answers.

    The server stops reading once its answers fill the socket buffer (on a Unix socket every
    short answer takes ~1 kB of it), so a client that read only after sending all would wait
    for ever; sendall, since a single write may send only part of a large request.
    """
    request = "".join(lines).encode("utf-8")
    with _connect(address, timeout) as sock:
        file = sock.makefile(mode="rb")
        failure = []
        sender = threading.Thread(
            target=_send_all, args=(sock, request, failure), daemon=True
        )
        sender.start()
        try:
            answers = [_read_line(file) for _ in lines]
        finally:
            sender.join()
        if failure:
            raise failure[0]
    return answers


def set_status(entries, address, timeout, verbose=0, batch_size=1000):
    """Publish (path, status) pairs: True/False, None or "u" to forget a path, "U" to forget
    every path matching a regex. Raises ServerUnavailable."""
    lines = [f"SET {path} {_set_value(path, status)}\n" for path, status in entries]
    try:
        for i in range(0, len(lines), batch_size):
            for line in _pipelined(address, timeout, lines[i : i + batch_size]):
                if line != "OK" and verbose > 0:
                    print(
                        f"pathCacheClient.set_status: SET failed: {line!r}",
                        file=sys.stderr,
                    )
    except ServerUnavailable:
        raise
    except OSError as e:
        raise ServerUnavailable(f"path-cache server at {address}: {e}") from e


def _decode_status(line):
    if line == "1":
        return True
    if line == "0":
        return False
    return None


def get_status_many(paths, address, timeout, verbose=0, batch_size=1000):
    """Query existence for many paths over as few connections as possible (pipelined:
    all GETs of a batch are sent on one connection, then all responses are read). Returns
    {path: True|False|None}. The server processes one request line at a time, so responses
    line up with the request order. Raises ServerUnavailable."""
    results = {}
    try:
        for i in range(0, len(paths), batch_size):
            batch = paths[i : i + batch_size]
            answers = _pipelined(address, timeout, [f"GET {p}\n" for p in batch])
            results.update(zip(batch, map(_decode_status, answers)))
    except ServerUnavailable:
        raise
    except OSError as e:
        raise ServerUnavailable(f"path-cache server at {address}: {e}") from e
    return results


def get_status(path, address, timeout, verbose=0):
    """True, False or None (unknown) for `path`. Raises ServerUnavailable."""
    try:
        with _connect(address, timeout) as sock:
            file = sock.makefile(mode="rwb", buffering=0)
            sock.sendall(f"GET {path}\n".encode("utf-8"))
            line = _read_line(file)
    except ServerUnavailable:
        raise
    except OSError as e:
        raise ServerUnavailable(f"path-cache server at {address}: {e}") from e
    if line not in ("1", "0", "u") and verbose > 0:
        print(
            f"pathCacheClient.get_status: unexpected response: {line!r}",
            file=sys.stderr,
        )
    return _decode_status(line)


def local_server_dir():
    # absolute: the server runs in it, and resolves the paths it is given from there
    return os.path.abspath(
        os.environ.get(LOCAL_SERVER_DIR_ENV)
        or os.path.join("/tmp", f"flaf-path-cache-{os.getuid()}")
    )


def local_server_address(directory=None):
    return os.path.join(directory or local_server_dir(), SOCKET_NAME)


def _private_dir(directory):
    try:
        os.mkdir(directory, 0o700)
    except FileExistsError:
        pass
    st = os.lstat(directory)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise ServerUnavailable(
            f"{directory} is not a directory of this user that nobody else can enter "
            "(mode 0700), so no path-cache server is run from it"
        )
    return directory


def server_answers(address, timeout=5.0):
    try:
        get_status("/", address, timeout)
    except ServerUnavailable:
        return False
    return True


def ensure_local_server(directory=None, startup_timeout=30.0):
    """The socket address of this user's path-cache server on this machine, started first
    if it is not running. Raises ServerUnavailable when it cannot be started."""
    try:
        return _ensure_local_server(directory, startup_timeout)
    except ServerUnavailable:
        raise
    except OSError as e:
        raise ServerUnavailable(f"the path-cache server cannot be started: {e}") from e


def _ensure_local_server(directory, startup_timeout):
    directory = _private_dir(os.path.abspath(directory or local_server_dir()))
    address = local_server_address(directory)
    if len(os.fsencode(address)) > MAX_SOCKET_PATH:
        raise ServerUnavailable(
            f"{address} is longer than a Unix socket path may be ({MAX_SOCKET_PATH} bytes); "
            f"set {LOCAL_SERVER_DIR_ENV} to a shorter directory"
        )
    if server_answers(address):
        return address
    # One process starts it; the others wait for the lock and then find it running.
    with open(os.path.join(directory, "server.lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if server_answers(address):
            return address
        log_path = os.path.join(directory, "server.log")
        with open(log_path, "a") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    SERVER_SCRIPT,
                    "--socket",
                    address,
                    "--cache-file",
                    os.path.join(directory, STATE_NAME),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=directory,
                # not stopped by a Ctrl-C meant for the law process that starts it
                start_new_session=True,
                close_fds=True,
            )
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if server_answers(address, timeout=1.0):
                return address
            if process.poll() is not None:
                break
            time.sleep(0.1)
    raise ServerUnavailable(
        f"the path-cache server did not start (exit code {process.poll()}); see {log_path}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Client of this user's path-cache server on this machine."
    )
    parser.add_argument(
        "--socket",
        default=None,
        type=str,
        help=f"server socket (default: {local_server_address()})",
    )
    parser.add_argument("--command", required=True, type=str, help="command to execute")
    parser.add_argument("--path", required=True, type=str, help="path to query/set")
    parser.add_argument(
        "--path-exists",
        required=False,
        default=None,
        type=int,
        help="exists flag when setting",
    )
    parser.add_argument(
        "--timeout",
        required=False,
        default=5,
        type=float,
        help="connection timeout in seconds",
    )
    args = parser.parse_args()
    address = args.socket or local_server_address()

    try:
        if args.command == "get":
            status = get_status(args.path, address, args.timeout, verbose=1)
            print(f"Result: {status}")
        elif args.command == "set":
            if args.path_exists is None or args.path_exists not in [0, 1]:
                print(
                    "Error: --path-exists must be provided for set command and its values should be 0 or 1",
                    file=sys.stderr,
                )
                sys.exit(1)
            exists = True if args.path_exists == 1 else False
            set_status([(args.path, exists)], address, args.timeout, verbose=1)
        elif args.command == "invalidate":
            set_status([(args.path, None)], address, args.timeout, verbose=1)
        elif args.command == "invalidate_regex":
            set_status([(args.path, "U")], address, args.timeout, verbose=1)
        else:
            print(f"Error: unknown command {args.command!r}", file=sys.stderr)
            sys.exit(1)
    except ServerUnavailable as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
