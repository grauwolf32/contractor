#!/usr/bin/env python3
"""Bounded TCP passthrough: mesh-only ingress to the existing loopback mTLS API."""

import argparse
import contextlib
import ipaddress
import select
import signal
import socket
import socketserver
import threading


class Relay(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 64

    def __init__(self, listen, upstream):
        self.upstream = upstream
        self.slots = threading.BoundedSemaphore(64)
        self.connections = set()
        self.lock = threading.Lock()
        super().__init__(listen, Handler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def handle_error(self, request, client_address):
        # No payloads, URLs, request headers or TLS content in diagnostics.
        pass


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        with socket.create_connection(self.server.upstream, timeout=10) as upstream:
            pair = (self.request, upstream)
            with self.server.lock:
                self.server.connections.update(pair)
            try:
                while True:
                    readable, _, _ = select.select(pair, [], [], 900)
                    if not readable:
                        return
                    for source in readable:
                        data = source.recv(65536)
                        if not data:
                            return
                        target = upstream if source is self.request else self.request
                        target.sendall(data)
            except (OSError, ValueError):
                return
            finally:
                with self.server.lock:
                    self.server.connections.difference_update(pair)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", required=True)
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--upstream-port", type=int, default=8443)
    args = parser.parse_args()
    if not ipaddress.ip_address(args.listen).is_private:
        raise ValueError("ingress must bind to a private mesh address")
    if not 1024 <= args.port <= 65535 or not 1024 <= args.upstream_port <= 65535:
        raise ValueError("invalid port")
    server = Relay((args.listen, args.port), ("127.0.0.1", args.upstream_port))

    def stop(*unused):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever()
    finally:
        with server.lock:
            for connection in list(server.connections):
                with contextlib.suppress(OSError):
                    connection.shutdown(socket.SHUT_RDWR)
                connection.close()
        server.server_close()


if __name__ == "__main__":
    main()
