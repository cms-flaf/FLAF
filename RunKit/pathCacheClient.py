import socket
import sys


def set_status(entries, host, port, timeout, verbose=0):
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            file = sock.makefile(mode="rwb", buffering=0)
            for path, status in entries:
                if status is None or status == "u":
                    val = "u"
                elif status == "U":
                    val = "U"
                elif isinstance(status, bool):
                    val = "1" if status else "0"
                else:
                    raise ValueError(
                        f"Invalid status value: {status!r} for path: {path!r}"
                    )
                request = f"SET {path} {val}\n".encode("utf-8")
                file.write(request)
                file.flush()
                line = file.readline()

                if verbose > 0:
                    if line:
                        line = line.decode("utf-8").strip()
                        if line != "OK":
                            print(
                                f"pathCacheClient.set_status: SET failed: {line!r}",
                                file=sys.stderr,
                            )
                    else:
                        print(
                            "pathCacheClient.set_status: no response from server",
                            file=sys.stderr,
                        )

    except TimeoutError as e:
        if verbose > 0:
            print(
                f"pathCacheClient.set_status: connection to server timed out: {e}",
                file=sys.stderr,
            )
    except Exception as e:
        if verbose > 0:
            print(f"pathCacheClient.set_status: unexpected error: {e}", file=sys.stderr)


def get_status(path, host, port, timeout, verbose=0):
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            file = sock.makefile(mode="rwb", buffering=0)
            request = f"GET {path}\n".encode("utf-8")
            file.write(request)
            file.flush()

            line = file.readline()
            if not line:
                if verbose > 0:
                    print(
                        "pathCacheClient.get_status: no response from server",
                        file=sys.stderr,
                    )
                return None
            line = line.decode("utf-8").strip()

            if line == "1":
                return True
            elif line == "0":
                return False
            elif line == "u":
                return None
            else:
                if verbose > 0:
                    print(
                        f"pathCacheClient.get_status: unexpected response: {line!r}",
                        file=sys.stderr,
                    )
                return None
    except TimeoutError as e:
        if verbose > 0:
            print(
                f"pathCacheClient.get_status: connection to server timed out: {e}",
                file=sys.stderr,
            )
    except Exception as e:
        if verbose > 0:
            print(f"pathCacheClient.get_status: unexpected error: {e}", file=sys.stderr)
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Path cache client.")
    parser.add_argument("--host", required=True, type=str, help="host")
    parser.add_argument("--port", required=True, type=str, help="port")
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

    if args.command == "get":
        status = get_status(args.path, args.host, args.port, args.timeout, verbose=1)
        print(f"Result: {status}")
    elif args.command == "set":
        if args.path_exists is None or args.path_exists not in [0, 1]:
            print(
                "Error: --path-exists must be provided for set command and its values should be 0 or 1",
                file=sys.stderr,
            )
            sys.exit(1)
        exists = True if args.path_exists == 1 else False
        set_status([(args.path, exists)], args.host, args.port, args.timeout, verbose=1)
    elif args.command == "invalidate":
        set_status([(args.path, None)], args.host, args.port, args.timeout, verbose=1)
    elif args.command == "invalidate_regex":
        set_status([(args.path, "U")], args.host, args.port, args.timeout, verbose=1)
    else:
        print(f"Error: unknown command {args.command!r}", file=sys.stderr)
        sys.exit(1)
