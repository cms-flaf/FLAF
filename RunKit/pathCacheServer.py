"""The path-cache server of one user on one machine.

It is started on demand by the first law process of the user on the machine that needs it
(pathCacheClient.ensure_local_server) and listens on a Unix socket in a directory only that
user can enter, so other users of a shared machine can neither read nor alter the cache. Its
state is written to a file next to the socket and read back when it is started again, e.g.
after the session that started it ended. Jobs never reach it: they work from the snapshot
their submission ships.
"""

import copy
import fcntl
import socket
import signal
import stat
import threading
import datetime
import json
import time
import sys
import multiprocessing
import os
import yaml
import re
from concurrent.futures import ThreadPoolExecutor
from queue import Empty

# The key law_gfal publishes after listing a directory (law_gfal.LISTING_MARKER): it implies
# "absent" for every file of the directory not listed, so it lives as long as a negative.
LISTING_MARKER = ".flaf_listed"

# How long a client may keep a connection open without sending a request.
CLIENT_TIMEOUT = 30

# What ends a server: logind sends SIGTERM and then SIGHUP to a session's processes.
EXIT_SIGNALS = (signal.SIGTERM, signal.SIGHUP)

MSG_UNKNOWN = 0
MSG_FORMAT_ERROR = 1
MSG_GET = 2
MSG_SET = 3
MSG_INVALIDATE = 4


def timestamp_str(timestamp=None):
    t = timestamp or datetime.datetime.now()
    return t.strftime("%Y-%m-%d %H:%M:%S")


def print_ts(msg, prefix="", timestamp=None, *args, **kwargs):
    print(f"{prefix}[{timestamp_str(timestamp=timestamp)}] {msg}", *args, **kwargs)


class Logger:
    message_texts = {
        MSG_UNKNOWN: "Unknown message: {entry}",
        MSG_FORMAT_ERROR: "Format error for message: {entry}. Error: {error}.",
        MSG_GET: "({n_calls}) get: {path} exists={exists} addr={addr_str}",
        MSG_SET: "({n_calls}) set: {path} exists={exists_bool} addr={addr_str}",
        MSG_INVALIDATE: "({n_calls}) invalidate: {path} n_invalidated_entries={n_invalidated_entries} addr={addr_str}",
    }

    def __init__(self, verbose=False):
        # whether every request is logged, not only errors and the server's own events
        self.verbose = verbose
        self.queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(target=self._logger_process, daemon=True)

    def __enter__(self):
        self.process.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.join()
        return False

    def log(self, msg, **kwargs):
        entry = msg if len(kwargs) == 0 else (msg, kwargs)
        self.queue.put(entry)

    def join(self):
        # A dead logger reads nothing: exiting must then not wait for the queue's feeder.
        self.queue.cancel_join_thread()
        self.queue.put(None)
        self.process.join()

    def get_text(self, entry):
        if isinstance(entry, tuple):
            entry = list(entry)
        if isinstance(entry, int):
            entry = [entry, {}]
        if isinstance(entry, str):
            return entry
        if not isinstance(entry, list) or len(entry) != 2:
            return None
        msg_id, format_args = entry
        if msg_id not in Logger.message_texts:
            return None
        msg_format = Logger.message_texts[msg_id]
        try:
            msg = msg_format.format(**format_args)
            return msg
        except Exception as e:
            return Logger.message_texts[MSG_FORMAT_ERROR].format(
                entry=entry, error=str(e)
            )

    def _logger_process(self):
        # the server stops its logger once it has logged its shutdown
        for signum in EXIT_SIGNALS:
            signal.signal(signum, signal.SIG_IGN)
        parent = os.getppid()
        while True:
            try:
                try:
                    entry = self.queue.get(timeout=60)
                except Empty:
                    # a server killed outright cannot stop its logger
                    if os.getppid() != parent:
                        break
                    continue
                if entry is None:
                    break
                message = self.get_text(entry)
                if message is None:
                    message = Logger.message_texts[MSG_UNKNOWN].format(entry=entry)
                print_ts(message, file=sys.stderr)
            except KeyboardInterrupt as e:
                pass
            except Exception:
                # e.g. a full disk: a message is lost, not the logger
                pass


def parse_seconds(value, what):
    if isinstance(value, str):
        value = eval(value, {}, {})
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"{what} must be a number of seconds, got: {value}")
    return value


class ValidityPeriodDB:
    def __init__(self, periods, negative_validity=None):
        self.periods = {}
        for prefix, validity in periods.items():
            if not isinstance(prefix, str):
                raise RuntimeError(
                    f"Validity period prefix must be a string, got: {prefix}"
                )
            self.periods[prefix] = parse_seconds(
                validity, f"Validity period for prefix {prefix}"
            )
        if "default" not in self.periods:
            raise RuntimeError("Validity periods must include a 'default' entry")
        # Jobs do not publish what they write, so an "absent" here is only as fresh as the
        # last listing a driver on this machine took; it is kept for a shorter time.
        self.negative_validity = (
            None
            if negative_validity is None
            else parse_seconds(negative_validity, "negative_validity")
        )

    def get_validity(self, path, exists=True):
        validity = self.periods["default"]
        for prefix, prefix_validity in self.periods.items():
            if path.startswith(prefix):
                validity = prefix_validity
                break
        negative = not exists or os.path.basename(path) == LISTING_MARKER
        if negative and self.negative_validity is not None:
            return min(validity, self.negative_validity)
        return validity


def standardize_path_name(path):
    parts = path.split("/")
    standardized_parts = [p for p in parts if len(p) > 0]
    if path.startswith("/"):
        standardized_parts.insert(0, "")
    return "/".join(standardized_parts)


def iter_parent_paths(path):
    """Yield the ancestor paths of a standardized path, from the immediate parent up to
    (but not including) the root, e.g. "/a/b/c" -> "/a/b", "/a"."""
    while True:
        idx = path.rfind("/")
        if idx <= 0:
            break
        path = path[:idx]
        yield path


class PathCache:
    def __init__(self, validity_db, logger, cache_file=None, cache_write_interval=-1):
        self.validity_db = validity_db
        self.logger = logger
        self.lock = threading.Lock()
        self.cache_file = cache_file
        self.cache_write_interval = (
            cache_write_interval if cache_file is not None else -1
        )
        self.cache = {}
        self.n_calls = 0
        self.last_write_time = 0
        # A write runs in a thread of its own: a later copy of the cache must never be
        # replaced by an earlier one that took longer to write (the last, at shutdown).
        self.write_lock = threading.Lock()
        self.n_copies = 0
        self.n_copy_written = 0
        if cache_file is not None and os.path.exists(cache_file):
            try:
                self._load(cache_file)
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
                # A cache that cannot be read is started afresh: it holds nothing that a
                # listing cannot find again.
                self.cache = {}
                self.n_calls = 0
                logger.log(
                    f"Cache file {cache_file} could not be read ({e}); starting empty"
                )
        self.last_request_time = time.time()

    def _load(self, cache_file):
        with open(cache_file, "r") as f:
            data = json.load(f)
        cache = {}
        current_time = time.time()
        for path, entry in data["cache"].items():
            fixed_path = standardize_path_name(path)
            fixed_entry = {}
            entry_valid = True
            for key, type in (("exists", bool), ("expiration_time", int)):
                if key not in entry:
                    entry_valid = False
                    break
                value = entry[key]
                if not isinstance(value, type):
                    entry_valid = False
                    break
                fixed_entry[key] = value
            entry_valid = entry_valid and fixed_entry["expiration_time"] >= current_time
            if entry_valid:
                cache[fixed_path] = fixed_entry
        self.cache = cache
        self.n_calls = data["n_calls"]
        self.last_write_time = data["last_write_time"]
        self.logger.log(
            f"Cache loaded from {cache_file} with {len(self.cache)} entries and n_calls={self.n_calls}"
        )

    def touch(self):
        self.last_request_time = time.time()

    def write(self, current_time=None, force=False, as_thread=True):
        if self.cache_file is not None:
            current_time = current_time or time.time()
            need_write = force or (
                self.cache_write_interval >= 0
                and (current_time - self.last_write_time) >= self.cache_write_interval
            )
            if need_write:
                self.last_write_time = current_time
                if as_thread:
                    threading.Thread(
                        target=self._write, args=(True,), daemon=True
                    ).start()
                else:
                    self._write(need_lock=False)

    def _write(self, need_lock):
        if need_lock:
            with self.lock:
                n_calls = self.n_calls
                cache = copy.deepcopy(self.cache)
                last_write_time = self.last_write_time
                self.n_copies += 1
                n_copy = self.n_copies
        else:
            n_calls = self.n_calls
            cache = self.cache
            last_write_time = self.last_write_time
            self.n_copies += 1
            n_copy = self.n_copies
        # Expired entries are otherwise dropped only when they are asked for, so the state
        # would keep everything ever learned.
        current_time = time.time()
        cache = {
            path: entry
            for path, entry in cache.items()
            if entry["expiration_time"] >= current_time
        }
        data = {"n_calls": n_calls, "last_write_time": last_write_time, "cache": cache}
        tmp_file = f"{self.cache_file}.{threading.get_ident()}.tmp"
        with open(tmp_file, "w") as f:
            json.dump(data, f)
        with self.write_lock:
            if n_copy < self.n_copy_written:
                os.remove(tmp_file)
                return
            os.replace(tmp_file, self.cache_file)
            self.n_copy_written = n_copy
        self.logger.log(
            f"Cache written to {self.cache_file} with {len(cache)} entries and n_calls={n_calls}"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self.lock:
            self.write(force=True, as_thread=False)
        return False

    def set(self, path, exists):
        path = standardize_path_name(path)
        with self.lock:
            current_time = time.time()
            expiration_time = int(
                current_time + self.validity_db.get_validity(path, exists=exists)
            )
            self.cache[path] = {"exists": exists, "expiration_time": expiration_time}
            # If a path exists, every ancestor directory exists too. Drop any stale
            # "does not exist" ancestor entries that would otherwise (via directory-negative
            # inference in get()) wrongly imply this path is absent.
            if exists:
                for parent in iter_parent_paths(path):
                    pentry = self.cache.get(parent)
                    if pentry is not None and not pentry["exists"]:
                        del self.cache[parent]
            self.n_calls += 1
            self.write(current_time=current_time)
            return path, self.n_calls

    def get(self, path):
        path = standardize_path_name(path)
        with self.lock:
            current_time = time.time()
            result = None
            if path in self.cache:
                entry = self.cache[path]
                if entry["expiration_time"] >= current_time:
                    result = entry["exists"]
                else:
                    del self.cache[path]
                    self.write(current_time=current_time)
            if result is None:
                # Directory-negative inference: if the nearest cached ancestor directory
                # does not exist, then this path cannot exist either. This lets a single
                # "directory absent" entry answer existence for an entire missing subtree
                # without any gfal calls from the clients.
                for parent in iter_parent_paths(path):
                    pentry = self.cache.get(parent)
                    if pentry is None:
                        continue
                    if pentry["expiration_time"] < current_time:
                        del self.cache[parent]
                        continue
                    if not pentry["exists"]:
                        result = False
                    break
            self.n_calls += 1
            return path, result, self.n_calls

    def invalidate(self, path):
        path = standardize_path_name(path)
        with self.lock:
            n_removed = 0
            if path in self.cache:
                del self.cache[path]
                n_removed = 1
            if n_removed > 0:
                self.write()
            return path, n_removed, self.n_calls

    def invalidate_all(self, path_pattern):
        pattern = re.compile(path_pattern)
        with self.lock:
            to_remove = []
            for path in self.cache.keys():
                if pattern.match(path):
                    to_remove.append(path)
            for path in to_remove:
                del self.cache[path]
            n_removed = len(to_remove)
            if n_removed > 0:
                self.write()
            return n_removed, self.n_calls


def result_to_string(result):
    if result is None:
        return "u"
    return "1" if result else "0"


def handle_client(conn, addr, cache, logger, connections=None):
    # a Unix socket's peer has no address
    addr_str = f"{addr[0]}:{addr[1]}" if isinstance(addr, tuple) else "local"
    try:
        # a client that stops talking must not keep a worker thread for ever
        conn.settimeout(CLIENT_TIMEOUT)
        with conn:
            # buffered: an unbuffered socket file reads a request one byte per system call
            file = conn.makefile(mode="rwb")

            for line in file:
                line = line.decode("utf-8").strip()
                if not line:
                    break
                cache.touch()

                parts = line.split(" ", 2)
                cmd = parts[0].upper()

                if cmd == "GET":
                    # GET <path>
                    if len(parts) != 2:
                        file.write(b"ERROR invalid GET command\n")
                        file.flush()
                        logger.log(
                            f'ERROR: invalid GET command: "{line}". addr={addr_str}'
                        )
                        continue

                    path = parts[1]
                    path, exists, n_calls = cache.get(path)
                    response = f"{result_to_string(exists)}\n"
                    file.write(response.encode("utf-8"))
                    file.flush()
                    if logger.verbose:
                        logger.log(
                            MSG_GET,
                            n_calls=n_calls,
                            path=path,
                            exists=exists,
                            addr_str=addr_str,
                        )
                elif cmd == "SET":
                    # SET <path> <0|1|u|U>
                    if len(parts) != 3:
                        file.write(b"ERROR invalid SET command\n")
                        file.flush()
                        logger.log(
                            f'ERROR: invalid SET command: "{line}". addr={addr_str}'
                        )
                        continue

                    path = parts[1]
                    val_str = parts[2]
                    if val_str not in ("0", "1", "u", "U"):
                        file.write(b"ERROR SET value must be 0, 1, u or U\n")
                        file.flush()
                        logger.log(
                            f'ERROR: invalid SET value: "{val_str}" in line: "{line}". addr={addr_str}'
                        )
                        continue

                    if val_str == "u":
                        path, n_invalidated_entries, n_calls = cache.invalidate(path)
                    elif val_str == "U":
                        n_invalidated_entries, n_calls = cache.invalidate_all(path)
                    else:
                        exists_bool = val_str == "1"
                        path, n_calls = cache.set(path, exists_bool)

                    file.write(b"OK\n")
                    file.flush()
                    if logger.verbose and val_str.lower() == "u":
                        logger.log(
                            MSG_INVALIDATE,
                            n_calls=n_calls,
                            path=path,
                            n_invalidated_entries=n_invalidated_entries,
                            addr_str=addr_str,
                        )
                    elif logger.verbose:
                        logger.log(
                            MSG_SET,
                            n_calls=n_calls,
                            path=path,
                            exists_bool=exists_bool,
                            addr_str=addr_str,
                        )
                else:
                    file.write(b"ERROR unknown command\n")
                    file.flush()
                    logger.log(f'ERROR: unknown command: "{line}". addr={addr_str}')

    except Exception as e:
        logger.log(f"Error handling client {addr_str}: {e}")
    finally:
        if connections is not None:
            connections.discard(conn)


def _socket_inode(path):
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    return st.st_ino if stat.S_ISSOCK(st.st_mode) else None


def _exit_on_signal(signum, frame):
    # Unwinds start_server, which writes the state before the process ends. Once: a second
    # signal would cut that short.
    for other in EXIT_SIGNALS:
        signal.signal(other, signal.SIG_IGN)
    raise SystemExit(0)


def _server_answers(socket_path):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        probe.settimeout(5)
        try:
            probe.connect(socket_path)
        except OSError:
            return False
    return True


# How long a server waits for one that is shutting down to release the socket.
LOCK_WAIT = 60


def start_server(cfg, socket_path, cache_file, n_threads, verbose=False):
    max_connections = cfg["max_connections"]
    validity_db = ValidityPeriodDB(
        cfg["validity_periods"], cfg.get("negative_validity")
    )
    idle_timeout = parse_seconds(cfg.get("idle_timeout", 24 * 60 * 60), "idle_timeout")
    for signum in EXIT_SIGNALS:
        signal.signal(signum, _exit_on_signal)
    # Delivered to the main thread only, and only while it waits in accept(): a thread that
    # took one would leave the main thread asleep there. Every thread inherits the block.
    signal.pthread_sigmask(signal.SIG_BLOCK, EXIT_SIGNALS)

    with Logger(verbose=verbose) as logger:
        # Held for the server's whole life: one server per socket, so the socket found at
        # the path is never one a live server listens on. Taken after the logger is forked,
        # which would otherwise hold it on after the server is killed. A server shutting
        # down holds it until its state is written: waited for, unless another one answers.
        run_lock = open(socket_path + ".lock", "a")
        deadline = time.time() + LOCK_WAIT
        while True:
            try:
                fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if _server_answers(socket_path) or time.time() > deadline:
                    run_lock.close()
                    logger.log(
                        f"A cache server already runs for {socket_path}; exiting"
                    )
                    return 0
                time.sleep(0.2)
        connections = set()
        with PathCache(
            validity_db,
            logger,
            cache_file=cache_file,
            cache_write_interval=cfg.get("cache_write_interval", 60),
        ) as cache:
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                bound_inode = None
                status = 0
                try:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                        if _socket_inode(socket_path) is not None:
                            # left behind by a server that did not exit cleanly
                            os.unlink(socket_path)
                        old_umask = os.umask(0o177)
                        try:
                            s.bind(socket_path)
                        finally:
                            os.umask(old_umask)
                        bound_inode = _socket_inode(socket_path)
                        s.listen(max_connections)
                        # wake up now and then to see whether the server is still used
                        s.settimeout(min(60.0, idle_timeout))
                        logger.log(
                            f"Cache server listening on {socket_path} (pid {os.getpid()})"
                        )

                        while True:
                            signal.pthread_sigmask(signal.SIG_UNBLOCK, EXIT_SIGNALS)
                            try:
                                conn, addr = s.accept()
                            except socket.timeout:
                                idle = time.time() - cache.last_request_time
                                if idle >= idle_timeout:
                                    logger.log(
                                        f"No request for {idle:.0f} s; shutting down."
                                    )
                                    break
                                continue
                            finally:
                                signal.pthread_sigmask(signal.SIG_BLOCK, EXIT_SIGNALS)
                            connections.add(conn)
                            executor.submit(
                                handle_client, conn, addr, cache, logger, connections
                            )
                except KeyboardInterrupt:
                    logger.log("Server shutting down due to keyboard interrupt.")
                except SystemExit:
                    logger.log("Server shutting down on a signal.")
                except Exception as e:
                    logger.log(f"Server error: {e}")
                    status = 1
                finally:
                    # only the socket this server bound: a later server may own the path
                    if (
                        bound_inode is not None
                        and _socket_inode(socket_path) == bound_inode
                    ):
                        os.unlink(socket_path)
                    # a client that keeps a connection open does not delay the shutdown
                    for conn in list(connections):
                        try:
                            conn.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
        run_lock.close()
    return status


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Path cache server.")
    parser.add_argument(
        "--cfg",
        required=False,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "pathCacheServerCfg.yaml"
        ),
        type=str,
        help="configuration file",
    )
    parser.add_argument(
        "--socket", required=True, type=str, help="Unix socket to listen on"
    )
    parser.add_argument(
        "--cache-file", required=False, default=None, type=str, help="cache file"
    )
    parser.add_argument(
        "--n-threads",
        required=False,
        default=16,
        type=int,
        help="number of threads for handling clients",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="log every request, not only errors"
    )
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    sys.exit(
        start_server(
            cfg, args.socket, args.cache_file, args.n_threads, verbose=args.verbose
        )
    )
