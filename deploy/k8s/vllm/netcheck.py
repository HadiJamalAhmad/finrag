"""Can this pod reach the Internet? Run inside a pod; prints one JSON line.

Three independent tries, so a block is not mistaken for one broken path:
DNS resolution, HTTPS to Hugging Face (where weights would come from), and a raw TCP
connection to a public IP (1.1.1.1:443), which needs no DNS.
"""

import json
import socket
import urllib.request


def attempt(fn):
    try:
        fn()
        return {"reached": True}
    except Exception as e:
        return {"reached": False, "error": type(e).__name__}


print(json.dumps({
    "dns": attempt(lambda: socket.getaddrinfo("huggingface.co", 443)),
    "https_huggingface": attempt(
        lambda: urllib.request.urlopen("https://huggingface.co", timeout=8).read(1)),
    "tcp_1.1.1.1": attempt(
        lambda: socket.create_connection(("1.1.1.1", 443), timeout=8).close()),
}))
