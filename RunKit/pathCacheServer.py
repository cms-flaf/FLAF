import copy
import socket
import threading
import datetime
import json
import time
import sys
import multiprocessing
import os
import yaml
import re
import shutil
from concurrent.futures import ThreadPoolExecutor

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

    def __init__(self):
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
        while True:
            try:
                entry = self.queue.get()
                if entry is None:
                    break
                message = self.get_text(entry)
                if message is None:
                    message = Logger.message_texts[MSG_UNKNOWN].format(entry=entry)
                print_ts(message, file=sys.stderr)
            except KeyboardInterrupt as e:
                pass


class ValidityPeriodDB:
    def __init__(self, periods):
        self.periods = {}
        for prefix, validity in periods.items():
            if not isinstance(prefix, str):
                raise RuntimeError(
                    f"Validity period prefix must be a string, got: {prefix}"
                )
            if isinstance(validity, str):
                validity = eval(validity, {}, {})
            if not isinstance(validity, (int, float)):
                raise RuntimeError(
                    f"Validity period must be a number of seconds, got: {validity} for prefix: {prefix}"
                )
            self.periods[prefix] = validity
        if "default" not in self.periods:
            raise RuntimeError("Validity periods must include a 'default' entry")

    def get_validity(self, path):
        for prefix, validity in self.periods.items():
            if path.startswith(prefix):
                return validity
        return self.periods["default"]


def standardize_path_name(path):
    parts = path.split("/")
    standardized_parts = [p for p in parts if len(p) > 0]
    if path.startswith("/"):
        standardized_parts.insert(0, "")
    return "/".join(standardized_parts)


class PathCache:
    def __init__(self, validity_db, logger, cache_file=None, cache_write_interval=-1):
        self.validity_db = validity_db
        self.logger = logger
        self.lock = threading.Lock()
        self.cache_file = cache_file
        self.cache_write_interval = (
            cache_write_interval if cache_file is not None else -1
        )
        if cache_file is not None and os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                data = json.load(f)
            self.cache = {}
            for path, entry in data["cache"].items():
                fixed_path = standardize_path_name(path)
                fixed_entry = {}
                entry_valid = True
                current_time = time.time()
                for key, type in (("exists", bool), ("expiration_time", int)):
                    if key not in entry:
                        entry_valid = False
                        break
                    value = entry[key]
                    if not isinstance(value, type):
                        entry_valid = False
                        break
                    fixed_entry[key] = value
                entry_valid = (
                    entry_valid and fixed_entry["expiration_time"] >= current_time
                )
                if entry_valid:
                    self.cache[fixed_path] = fixed_entry

            self.n_calls = data["n_calls"]
            self.last_write_time = data["last_write_time"]
            self.cache_write_interval = cache_write_interval
            logger.log(
                f"Cache loaded from {cache_file} with {len(self.cache)} entries and n_calls={self.n_calls}"
            )
        else:
            self.cache = {}
            self.n_calls = 0
            self.last_write_time = 0

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
        else:
            n_calls = self.n_calls
            cache = self.cache
            last_write_time = self.last_write_time
        data = {"n_calls": n_calls, "last_write_time": last_write_time, "cache": cache}
        tmp_file = self.cache_file + ".tmp"
        backup_file = self.cache_file + ".bak"
        with open(tmp_file, "w") as f:
            json.dump(data, f, indent=2)
        if os.path.exists(self.cache_file):
            shutil.move(self.cache_file, backup_file)
        shutil.move(tmp_file, self.cache_file)
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
            expiration_time = int(current_time + self.validity_db.get_validity(path))
            self.cache[path] = {"exists": exists, "expiration_time": expiration_time}
            self.n_calls += 1
            self.write(current_time=current_time)
            return path, self.n_calls

    def get(self, path):
        path = standardize_path_name(path)
        with self.lock:
            result = None
            if path in self.cache:
                entry = self.cache[path]
                current_time = time.time()
                if entry["expiration_time"] >= current_time:
                    result = entry["exists"]
                else:
                    del self.cache[path]
                    self.write(current_time=current_time)
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


def handle_client(conn, addr, cache, logger):
    try:
        addr_str = f"{addr[0]}:{addr[1]}"
        with conn:
            file = conn.makefile(mode="rwb", buffering=0)

            for line in file:
                line = line.decode("utf-8").strip()
                if not line:
                    break

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
                    if val_str.lower() == "u":
                        logger.log(
                            MSG_INVALIDATE,
                            n_calls=n_calls,
                            path=path,
                            n_invalidated_entries=n_invalidated_entries,
                            addr_str=addr_str,
                        )
                    else:
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


def start_server(cfg, cache_file, n_threads):
    host = cfg["host"]
    port = cfg["port"]
    max_connections = cfg["max_connections"]
    validity_db = ValidityPeriodDB(cfg["validity_periods"])

    with Logger() as logger:
        with PathCache(
            validity_db,
            logger,
            cache_file=cache_file,
            cache_write_interval=cfg.get("cache_write_interval", 60),
        ) as cache:
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        s.bind((host, port))
                        s.listen(max_connections)
                        logger.log(f"Cache server listening on {host}:{port}")

                        while True:
                            conn, addr = s.accept()
                            executor.submit(handle_client, conn, addr, cache, logger)
                except KeyboardInterrupt:
                    logger.log("Server shutting down due to keyboard interrupt.")
                except Exception as e:
                    logger.log(f"Server error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Path cache server.")
    parser.add_argument("--cfg", required=True, type=str, help="configuration file")
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
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    start_server(cfg, args.cache_file, args.n_threads)
