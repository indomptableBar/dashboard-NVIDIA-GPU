#!/usr/bin/env python3
"""NVIDIA GPU Dashboard Backend - serves GPU metrics via HTTP API."""

import json
import subprocess
import xml.etree.ElementTree as ET
from http.server import HTTPServer, SimpleHTTPRequestHandler
import time


def safe_int(text, default=0):
    """Safely parse int from text like '45 C' or 'N/A'."""
    if text is None:
        return default
    text = text.strip()
    if text == 'N/A':
        return default
    try:
        return int(text.split()[0])
    except (ValueError, IndexError):
        return default


def safe_float(text, default=0.0):
    """Safely parse float from text like '5.42 W' or 'N/A'."""
    if text is None:
        return default
    text = text.strip()
    if text == 'N/A':
        return default
    try:
        return float(text.split()[0])
    except (ValueError, IndexError):
        return default


def find_text(parent, path):
    """Find element text safely."""
    elem = parent.find(path)
    return elem.text if elem is not None else None


def get_gpu_data():
    """Query nvidia-smi and return parsed GPU data."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '-q', '-x'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return {'error': 'nvidia-smi command failed', 'gpus': []}

        root = ET.fromstring(result.stdout)
        gpus = []

        for gpu in root.findall('.//gpu'):
            gpu_id = gpu.get('id', 'N/A')
            name = find_text(gpu, 'product_name') or 'N/A'
            uuid = find_text(gpu, 'uuid') or 'N/A'

            temp_text = find_text(gpu, 'temperature/gpu_temp')
            temp_value = safe_int(temp_text)

            power_current = safe_float(find_text(gpu, 'gpu_power_readings/instant_power_draw'))
            power_limit = safe_float(find_text(gpu, 'gpu_power_readings/current_power_limit'))

            mem_used = safe_float(find_text(gpu, 'fb_memory_usage/used'))
            mem_total = safe_float(find_text(gpu, 'fb_memory_usage/total'))

            gpu_util = safe_int(find_text(gpu, 'utilization/gpu_util'))
            mem_util = safe_int(find_text(gpu, 'utilization/memory_util'))

            fan_text = find_text(gpu, 'fan_speed')
            fan_speed = safe_int(fan_text)

            processes = []
            for proc in gpu.findall('.//process_info'):
                processes.append({
                    'pid': find_text(proc, 'pid') or 'N/A',
                    'name': find_text(proc, 'process_name') or 'N/A',
                    'memory': find_text(proc, 'used_memory') or 'N/A'
                })

            gpu_info = {
                'id': gpu_id,
                'name': name,
                'uuid': uuid,
                'temperature': {'value': temp_value, 'unit': 'C'},
                'power': {'current': power_current, 'limit': power_limit, 'unit': 'W'},
                'memory': {'used': mem_used, 'total': mem_total, 'unit': 'MiB'},
                'utilization': {'gpu': gpu_util, 'memory': mem_util, 'unit': '%'},
                'fan_speed': fan_speed,
                'processes': processes
            }

            gpus.append(gpu_info)

        return {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'driver_version': find_text(root, 'driver_version') or 'N/A',
            'cuda_version': find_text(root, 'cuda_version') or 'N/A',
            'gpus': gpus
        }
    except Exception as e:
        return {'error': str(e), 'gpus': []}


class GPUHandler(SimpleHTTPRequestHandler):
    """HTTP handler for GPU dashboard."""

    def do_GET(self):
        if self.path == '/api/gpu':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(get_gpu_data()).encode())
        elif self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            with open('dashboard.html', 'rb') as f:
                self.wfile.write(f.read())
        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass


def main():
    port = 8888
    server = HTTPServer(('0.0.0.0', port), GPUHandler)
    print(f"GPU Dashboard running at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
