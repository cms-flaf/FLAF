"""TCP-based run token server that rate-limits job starts per a given path.

Usage:
    python3 run_token_server.py --port PORT [--interval SECONDS] [--host HOST]
                                [--max-connections N] [--wait-threshold N]

Each unique path receives at most one token per --interval seconds.
Clients connect, send their path terminated by a newline, and receive
either:
  OK              — token granted, job may start
  WAIT <seconds>  — queue too long; client should disconnect, sleep, reconnect
  ERROR <msg>     — unrecoverable error
"""

import socket
import threading
import time

def timestamp_str():
    return time.strftime('%Y-%m-%d %H:%M:%S')


class PathState:
    __slots__ = ('lock', 'last_token_time', 'queue_size')

    def __init__(self):
        self.lock = threading.Lock()
        self.last_token_time = 0.0
        self.queue_size = 0


class TokenServer:
    def __init__(self, interval, max_connections, wait_threshold, min_sleep_interval=0.2, recv_chunk_size=4096,
                 max_request_length=1024*1024, backlog=512):
        self.interval = interval
        self.max_connections = max_connections
        self.wait_threshold = wait_threshold
        self.min_sleep_interval = min_sleep_interval
        self.recv_chunk_size = recv_chunk_size
        self.max_request_length = max_request_length
        self.backlog = backlog

        self.conn_lock = threading.Lock()
        self.active_connections = 0

        self.paths_lock = threading.Lock()
        self.path_states = {}

    def _get_path_state(self, path):
        with self.paths_lock:
            if path not in self.path_states:
                self.path_states[path] = PathState()
            return self.path_states[path]

    def _acquire_token(self, state):
        """Block until a token is issued for *state*.

        Returns 0 on success, or a positive int (expected wait seconds) when
        the queue is too long and the caller should disconnect and retry later.
        """
        queued = False
        try:
            with state.lock:
                if state.queue_size > self.wait_threshold:
                    exp_wait = int(state.queue_size * self.interval) + 1
                    return exp_wait
                state.queue_size += 1
                queued = True

            while True:
                with state.lock:
                    now = time.monotonic()
                    if now - state.last_token_time >= self.interval:
                        state.last_token_time = now
                        return 0
                    sleep_for = self.interval - (now - state.last_token_time)
                time.sleep(min(sleep_for, self.min_sleep_interval))
        finally:
            if queued:
                with state.lock:
                    state.queue_size -= 1

    def _handle_client(self, conn, addr):
        try:
            data = b''
            while b'\n' not in data and len(data) < self.max_request_length:
                chunk = conn.recv(self.recv_chunk_size)
                if not chunk:
                    conn.sendall(b'ERROR: no data received\n')
                    return
                data += chunk
            if len(data) >= self.max_request_length:
                conn.sendall(b'ERROR: request too long\n')
                return
            ref_path = data.decode().strip()
            if not ref_path:
                conn.sendall(b'ERROR: empty path\n')
                return

            ts = timestamp_str()
            print(f'[{ts}] request  {addr[0]}:{addr[1]}  {ref_path}')

            state = self._get_path_state(ref_path)
            result = self._acquire_token(state)

            if result > 0:
                conn.sendall(f'WAIT {result}\n'.encode())
                ts = timestamp_str()
                print(
                    f'[{ts}] redirect {addr[0]}:{addr[1]}  {ref_path}  '
                    f'queue={state.queue_size}  wait={result}s'
                )
            else:
                conn.sendall(b'OK\n')
                ts = timestamp_str()
                print(f'[{ts}] issued   {addr[0]}:{addr[1]}  {ref_path}')

        except Exception as e:
            print(f'Error handling {addr}: {e}')
        finally:
            with self.conn_lock:
                self.active_connections -= 1
            conn.close()

    def run(self, host, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((host, port))
            server.listen(self.backlog)
            ts = timestamp_str()
            print(
                f'[{ts}] Token server listening on {host or "*"}:{port}  '
                f'interval={self.interval}s  '
                f'max_connections={self.max_connections}  '
                f'wait_threshold={self.wait_threshold}',
            )
            while True:
                conn, addr = server.accept()
                with self.conn_lock:
                    if self.active_connections >= self.max_connections:
                        try:
                            conn.sendall(b'ERROR: server at max capacity\n')
                        except Exception:
                            pass
                        conn.close()
                        ts = timestamp_str()
                        print(
                            f'[{ts}] rejected {addr[0]}:{addr[1]} '
                            f'(connections={self.active_connections}/{self.max_connections})',
                        )
                        continue
                    self.active_connections += 1

                t = threading.Thread(
                    target=self.handle_client, args=(conn, addr), daemon=True
                )
                t.start()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Token server for rate-limiting distributed job starts per a reference path'
    )
    parser.add_argument('--host', default='',
                        help='interface to bind to (default: all interfaces)')
    parser.add_argument('--port', type=int, default=5007,
                        help='TCP port to listen on')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='minimum seconds between tokens per a reference path')
    parser.add_argument('--max-connections', type=int, default=10000,
                        help='maximum simultaneous open connections')
    parser.add_argument('--wait-threshold', type=int, default=100,
                        help='queue depth above which a WAIT redirect is sent instead of '
                             'holding the connection open')
    args = parser.parse_args()

    server = TokenServer(args.interval, args.max_connections, args.wait_threshold)
    server.run(args.host, args.port)
