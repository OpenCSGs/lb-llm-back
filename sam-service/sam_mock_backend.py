import argparse
import json
import os
import random
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _load_env(path):
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


MODEL_VERSION = "sam-random-mask-v1"
MASK_WIDTH = 100
MASK_HEIGHT = 100


def _encode_rle(values):
    """Encode byte values using Label Studio's brush-mask RLE format."""
    base = f"{len(values):032b}{7:05b}" + "".join(f"{size - 1:04b}" for size in (3, 4, 8, 16))
    encoded = ""
    start = 0
    while start < len(values):
        value = values[start]
        end = start + 1
        while end < len(values) and values[end] == value:
            end += 1
        run_length = end - start
        while run_length:
            chunk = min(run_length, 65536)
            if chunk == 1:
                encoded += f"000000{value:08b}"
            elif chunk <= 8:
                encoded += f"100{chunk - 1:03b}{value:08b}"
            elif chunk <= 16:
                encoded += f"101{chunk - 1:04b}{value:08b}"
            elif chunk <= 256:
                encoded += f"110{chunk - 1:08b}{value:08b}"
            else:
                encoded += f"111{chunk - 1:016b}{value:08b}"
            run_length -= chunk
        start = end

    bits = base + encoded
    bits += "0" * ((8 - len(bits) % 8) % 8)
    return [int(bits[index : index + 8], 2) for index in range(0, len(bits), 8)]


def _random_mask_rle():
    center_x = random.randint(20, 80)
    center_y = random.randint(20, 80)
    radius_x = random.randint(8, 25)
    radius_y = random.randint(8, 25)
    rgba = []
    for y in range(MASK_HEIGHT):
        for x in range(MASK_WIDTH):
            inside = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2 <= 1
            pixel = 255 if inside else 0
            rgba.extend((pixel, pixel, pixel, pixel))
    return _encode_rle(rgba)


def _random_prediction(task):
    return {
        "result": [
            {
                "id": f"mask-{random.randint(100000, 999999)}",
                "from_name": "label",
                "to_name": "image",
                "type": "brushlabels",
                "origin": "prediction",
                "original_width": MASK_WIDTH,
                "original_height": MASK_HEIGHT,
                "image_rotation": 0,
                "value": {
                    "format": "rle",
                    "rle": _random_mask_rle(),
                    "brushlabels": ["object"],
                },
            }
        ],
        "score": random.random(),
        "model_version": MODEL_VERSION,
        "task_id": task.get("id"),
    }


class SAMHandler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "UP", "model_version": MODEL_VERSION})
        else:
            self._send_json(404, {"detail": "Not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"detail": "Invalid JSON"})
            return

        if self.path == "/setup":
            self._send_json(200, {"model_version": MODEL_VERSION})
            return
        if self.path != "/predict":
            self._send_json(404, {"detail": "Not found"})
            return

        delay = float(os.getenv("SAM_SERVICE_DELAY_SECONDS", "0.5"))
        if delay > 0:
            time.sleep(delay)
        predictions = [_random_prediction(task) for task in payload.get("tasks") or []]
        self._send_json(200, {"results": predictions, "model_version": MODEL_VERSION})

    def log_message(self, message, *args):
        print(f"[SAM Service] {self.address_string()} - {message % args}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random-mask SAM service")
    parser.add_argument("--host", default=os.getenv("SAM_SERVICE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SAM_SERVICE_PORT", "9091")))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), SAMHandler)
    print(f"[SAM Service] Listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
