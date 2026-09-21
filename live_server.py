"""
AirLoop Live Continuous Server with Auto-Refreshing Simulation Daemon.
Serves the interactive Leaflet application, provides on-demand /api/refresh endpoints,
and continuously synchronizes with NASA FIRMS and Open-Meteo in the background.
"""

import os
import sys
import json
import time
import logging
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

# Setup root directory
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from run_airloop import run_coupled_system

logger = logging.getLogger("airloop.live_server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Global synchronization lock to avoid simultaneous overlapping runs
simulation_lock = threading.Lock()
last_update_time = time.time()
update_in_progress = False


def execute_background_refresh(force_live: bool = True) -> bool:
    """Run the physical simulation and rebuild the web dashboard."""
    global last_update_time, update_in_progress
    if simulation_lock.locked():
        logger.warning("Simulation update already in progress. Skipping.")
        return False
        
    with simulation_lock:
        update_in_progress = True
        try:
            logger.info("Executing scheduled/on-demand live simulation update...")
            run_coupled_system(
                forecast_hours=72,
                use_cache=not force_live,
                serve=False
            )
            last_update_time = time.time()
            logger.info("Live simulation update successfully completed.")
            return True
        except Exception as e:
            logger.error(f"Error during simulation update: {e}", exc_info=True)
            return False
        finally:
            update_in_progress = False


def continuous_auto_sync_daemon(interval_seconds: int = 600):
    """Background daemon that periodically pulls new satellite & weather runs."""
    logger.info(f"Starting continuous background synchronization daemon (every {interval_seconds // 60} minutes)...")
    while True:
        time.sleep(interval_seconds)
        logger.info("Daemon interval reached. Triggering automatic satellite & weather sync...")
        execute_background_refresh(force_live=True)


class LiveAirLoopHandler(SimpleHTTPRequestHandler):
    """HTTP handler with API endpoints for live status and refresh triggers."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "/index.html"
            return super().do_GET()
        elif self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            status_data = {
                "service": "AirLoop Live Continuous Engine",
                "status": "operational",
                "last_update_timestamp": last_update_time,
                "last_update_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(last_update_time)),
                "is_updating": update_in_progress,
                "auto_sync_interval_seconds": 600
            }
            self.wfile.write(json.dumps(status_data).encode("utf-8"))
            return
        elif self.path == "/api/refresh":
            # Allow GET trigger as well for easy testing
            return self.handle_refresh()
            
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/refresh":
            return self.handle_refresh()
        self.send_error(404, "Not Found")

    def handle_refresh(self):
        """Handle on-demand data refresh trigger."""
        logger.info("Received on-demand refresh request from web client.")
        success = execute_background_refresh(force_live=True)
        
        self.send_response(200 if success else 500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        
        resp = {
            "success": success,
            "refreshed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(last_update_time)),
            "message": "Live satellite and weather data synchronized."
        }
        self.wfile.write(json.dumps(resp).encode("utf-8"))

    def log_message(self, format, *args):
        # Quiet standard static asset logging
        if args and isinstance(args[0], str) and "/api/" in args[0]:
            super().log_message(format, *args)


def start_live_server(port: int = 8080, auto_sync_mins: int = 10, open_browser: bool = True):
    """Start the live server and background sync thread."""
    # 1. Start background auto-sync thread
    daemon_thread = threading.Thread(
        target=continuous_auto_sync_daemon,
        args=(auto_sync_mins * 60,),
        daemon=True
    )
    daemon_thread.start()

    # 2. Start HTTP server
    server = ThreadingHTTPServer(("0.0.0.0", port), LiveAirLoopHandler)
    url = f"http://localhost:{port}"
    print("\n" + "=" * 76)
    print(f"  AIRLOOP LIVE CONTINUOUS SERVER ACTIVE")
    print(f"  Live Dashboard: {url}")
    print(f"  API Endpoint:   {url}/api/status")
    print(f"  Auto-Sync:      Every {auto_sync_mins} minutes in background")
    print("=" * 76 + "\n")

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Live Server...")
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AirLoop Live Auto-Refreshing Web Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    parser.add_argument("--interval", type=int, default=10, help="Auto-sync interval in minutes (default: 10)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    
    args = parser.parse_args()
    start_live_server(port=args.port, auto_sync_mins=args.interval, open_browser=not args.no_browser)
