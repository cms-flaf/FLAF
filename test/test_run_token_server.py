"""Quick test for the token server.

Usage:
    python3 test_run_token_server.py --server HOST --port PORT [--path PATH] [--count N]
"""

import socket
import sys
import time

def request_token(host, port, analysis_path):
    t0 = time.monotonic()
    with socket.create_connection((host, port), timeout=30) as s:
        s.sendall((analysis_path + '\n').encode())
        response = b''
        while b'\n' not in response:
            chunk = s.recv(1024)
            if not chunk:
                break
            response += chunk
    elapsed = time.monotonic() - t0
    return response.decode().strip(), elapsed


def test_token_server(server, port, path, count):
    print(f'Testing token server at {server}:{port}')
    print(f'Path: {path}')
    print(f'Requests: {count}')
    print()

    ok = True
    for i in range(1, count + 1):
        try:
            response, elapsed = request_token(server, port, path)
            status = 'OK' if response == 'OK' else f'UNEXPECTED: {response!r}'
            print(f'  request {i}/{count}: response={response!r}  elapsed={elapsed:.2f}s  [{status}]')
            if response != 'OK':
                ok = False
        except Exception as e:
            print(f'  request {i}/{count}: FAILED — {e}')
            ok = False

    print()
    if ok:
        print('All requests succeeded.')
    else:
        print('Some requests failed.')
        sys.exit(1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Smoke-test the token server')
    parser.add_argument('--server', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--path', default='/test/analysis/path',
                        help='path to send (default: /test/analysis/path)')
    parser.add_argument('--count', type=int, default=3,
                        help='number of sequential token requests (default: 3)')
    args = parser.parse_args()

    test_token_server(args.server, args.port, args.path, args.count)
