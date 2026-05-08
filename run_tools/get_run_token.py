"""Client that requests a run token from the token server.

Server responses:
  OK            — token granted, proceed
  WAIT <secs>   — server queue is busy; disconnect, sleep <secs>, reconnect
  ERROR <msg>   — server error; retry after retry_interval

Usage (called from bootstrap.sh):
    python3 get_token.py --server HOST --port PORT --path PATH
"""

import socket
import sys
import time


def recv_line(s):
    """Read until newline from socket s, return stripped string."""
    data = b''
    while b'\n' not in data:
        chunk = s.recv(1024)
        if not chunk:
            break
        data += chunk
    return data.decode().strip()


def get_token(host, port, path, timeout=3600, retry_interval=10, connection_timeout=60):
    deadline = time.monotonic() + timeout
    attempt = 0

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f'Timeout waiting for token from {host}:{port}', file=sys.stderr)
            return False

        attempt += 1
        sleep_time = min(retry_interval, remaining)
        try:
            with socket.create_connection((host, port), timeout=connection_timeout) as s:
                s.sendall((path + '\n').encode())
                response = recv_line(s)

            if response == 'OK':
                return True

            if response.startswith('WAIT '):
                try:
                    wait_secs = float(response.split()[1])
                except (IndexError, ValueError):
                    wait_secs = retry_interval
                sleep_time = min(wait_secs, remaining)
                print(
                    f'[attempt {attempt}] Server queue busy, reconnecting in {wait_secs:.0f}s',
                    file=sys.stderr,
                )
            else:
                # ERROR or unexpected response — log and retry after a short pause
                print(
                    f'[attempt {attempt}] Server response: {response!r}, '
                    f'retrying in {sleep_time}s',
                    file=sys.stderr,
                )

        except OSError as e:
            print(
                f'[attempt {attempt}] Connection to {host}:{port} failed: {e}. '
                f'Retrying in {sleep_time}s',
                file=sys.stderr,
            )
        time.sleep(sleep_time)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Request a start token from the token server')
    parser.add_argument('--server', required=True, help='server hostname')
    parser.add_argument('--port', type=int, required=True, help='server port')
    parser.add_argument('--path', required=True, help='path identifying this job group')
    parser.add_argument('--timeout', type=float, default=3600, help='max wait time in seconds')
    parser.add_argument('--retry-interval', type=float, default=10, help='seconds to wait between retries')
    args = parser.parse_args()

    print(
        f'Requesting run token from {args.server}:{args.port} for {args.path}',
        file=sys.stderr,
    )
    if not get_token(args.server, args.port, args.path, args.timeout, args.retry_interval):
        sys.exit(1)
    print('Run token received.', file=sys.stderr)
