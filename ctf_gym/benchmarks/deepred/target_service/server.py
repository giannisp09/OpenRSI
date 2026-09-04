import http.server
import socketserver
import urllib.parse
import os

class VulnerableHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Welcome to DeepRed target service. Service is running.")
        elif parsed.path == "/api/status":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Status OK.")
        elif parsed.path == "/api/admin":
            query = urllib.parse.parse_qs(parsed.query)
            if 'cmd' in query:
                cmd = query['cmd'][0]
                # DeepRed simulation: RCE vulnerability
                if cmd == "whoami":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"root\n")
                elif cmd == "cat flag.txt":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"DEEPRED_FLAG{system_compromised}\n")
                else:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(f"Executed: {cmd}".encode())
            else:
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b"Unauthorized")
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    PORT = 8080
    with socketserver.TCPServer(("", PORT), VulnerableHandler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()
