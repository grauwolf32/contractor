"""Loopback-only genuine SQLite injection target and parameter-bound control."""

import argparse
import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

parser = argparse.ArgumentParser()
parser.add_argument("--listen", required=True)
arguments = parser.parse_args()
host, port = arguments.listen.rsplit(":", 1)
if host != "127.0.0.1":
    raise SystemExit("fixture requires loopback")
lock = threading.Lock()
requests = 0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_arguments):
        pass

    def do_GET(self):
        global requests
        parts = urlsplit(self.path)
        status = 200
        if parts.path == "/readyz":
            body = b"ready"
        elif parts.path == "/requests":
            with lock:
                body = json.dumps({"requests": requests}).encode()
        else:
            with lock:
                requests += 1
            if parts.path == "/.git/config":
                body = b"[core]\nrepositoryformatversion = 0\n[remote origin]\nurl = https://example.test/fixture.git\n"
            elif parts.path == "/tech":
                body = b"nginx/1.24.0"
            elif parts.path == "/fixed/.git/config":
                status, body = 404, b"missing"
            elif parts.path in {"/vulnerable", "/fixed"}:
                parameter = parse_qs(parts.query, keep_blank_values=True).get("id", ["1"])[0]
                connection = sqlite3.connect(":memory:")
                try:
                    connection.execute("CREATE TABLE items (id INTEGER, content TEXT)")
                    connection.executemany("INSERT INTO items VALUES (?,?)", [(1, "alpha"), (2, "beta")])
                    # Bound query work independently of the HTTP deadline.
                    connection.set_progress_handler(lambda: 1, 100000)
                    if parts.path == "/fixed":
                        result = connection.execute("SELECT content FROM items WHERE id = ?", (parameter,))
                    else:
                        result = connection.execute("SELECT content FROM items WHERE id = " + parameter)
                    body = ("Items: " + ", ".join(row[0] for row in result)).encode()
                except sqlite3.Error as error:
                    body = ("SQLite3 error: " + str(error)).encode()
                finally:
                    connection.close()
            else:
                body = b"fixture"
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


ThreadingHTTPServer((host, int(port)), Handler).serve_forever()
