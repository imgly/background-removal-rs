#!/usr/bin/env python3
"""
Simple HTTP server for testing the bg-remove-web demo.

This server serves the demo files with proper CORS headers and MIME types
for WebAssembly modules.
"""

import http.server
import socketserver
import os
import sys
from urllib.parse import urlparse

class WasmHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers for cross-origin requests
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        super().end_headers()

    def guess_type(self, path):
        """Add proper MIME types for WebAssembly files."""
        mimetype, encoding = super().guess_type(path)
        
        # WebAssembly MIME type
        if path.endswith('.wasm'):
            return 'application/wasm', encoding
        
        # ES6 modules
        if path.endswith('.mjs') or (path.endswith('.js') and '/pkg/' in path):
            return 'application/javascript', encoding
            
        return mimetype, encoding

def main():
    # Change to demo directory
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(demo_dir)
    
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}")
            sys.exit(1)
    
    print(f"Starting bg-remove-web demo server...")
    print(f"Demo directory: {demo_dir}")
    print(f"Server URL: http://localhost:{port}")
    print()
    print("Available files:")
    for root, dirs, files in os.walk('.'):
        for file in files:
            if not file.startswith('.'):
                path = os.path.join(root, file)
                print(f"  {path}")
    print()
    print("Press Ctrl+C to stop the server")
    
    try:
        with socketserver.TCPServer(("", port), WasmHTTPRequestHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")

if __name__ == "__main__":
    main()