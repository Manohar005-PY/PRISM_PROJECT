"""PRISM Live Telemetry Streaming Client.

Streams real-time per-frame perception, tracking, TTC, risk, and decision JSON
from an Android device running PRISM directly to your laptop over Wi-Fi or USB ADB.

Usage:
  # Via USB cable (automatic ADB port forwarding):
  python scripts/view_live_telemetry.py --adb

  # Via Wi-Fi (using phone's local IP displayed on phone HUD):
  python scripts/view_live_telemetry.py --url http://192.168.1.15:8080/stream

  # Print raw JSON strings directly:
  python scripts/view_live_telemetry.py --adb --raw
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
import urllib.request


def find_adb_executable() -> str:
    """Find the path to the adb executable."""
    # 1. Check if adb is in system PATH
    import shutil
    adb_path = shutil.which("adb")
    if adb_path:
        return adb_path

    # 2. Check standard Android SDK locations on Windows / Linux / Mac
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, "AppData", "Local", "Android", "Sdk", "platform-tools", "adb.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Android", "Sdk", "platform-tools", "adb.exe"),
        os.path.join(os.environ.get("ANDROID_HOME", ""), "platform-tools", "adb.exe"),
        os.path.join(os.environ.get("ANDROID_SDK_ROOT", ""), "platform-tools", "adb.exe"),
        os.path.join(home, "Android", "Sdk", "platform-tools", "adb"),
        os.path.join(home, "Library", "Android", "sdk", "platform-tools", "adb"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p

    return "adb"


def setup_adb_forward(port: int = 8080) -> bool:
    """Forward phone's telemetry port to localhost via ADB."""
    adb_bin = find_adb_executable()
    try:
        cmd = [adb_bin, "forward", f"tcp:{port}", f"tcp:{port}"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"[ADB] Port forwarded: localhost:{port} -> device:{port} (using {adb_bin})")
            return True
        else:
            print(f"[ADB Warning] 'adb forward' returned: {res.stderr.strip()}")
            return False
    except Exception as ex:
        print(f"[ADB Error] Could not run adb at '{adb_bin}': {ex}")
        return False


def stream_telemetry(url: str, output_file: str | None = None, raw_json: bool = False) -> None:
    """Connect to PRISM Server-Sent Events (SSE) telemetry stream and render live data."""
    print(f"\n=======================================================")
    print(f"  PRISM Live Telemetry Studio Client")
    print(f"  Streaming from: {url}")
    if output_file:
        print(f"  Saving JSONL session to: {output_file}")
    print(f"=======================================================\n")

    f_out = None
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        f_out = open(output_file, "a", encoding="utf-8")

    total_frames = 0
    t_start = time.time()

    req = urllib.request.Request(
        url,
        headers={"Accept": "text/event-stream", "User-Agent": "PRISM-Laptop-Client/1.0"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            print("[Connected] Receiving real-time frames from phone...\n")
            for line in resp:
                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str.startswith("data:"):
                    continue

                json_payload = line_str[5:].strip()
                if not json_payload:
                    continue

                try:
                    data = json.loads(json_payload)
                    total_frames += 1

                    if f_out:
                        f_out.write(json.dumps(data) + "\n")
                        f_out.flush()

                    if raw_json:
                        print(json.dumps(data))
                        continue

                    # Render Formatted Console Frame
                    frame_id = data.get("frame_id", 0)
                    state = data.get("runtime_state", "RUNNING")
                    timing = data.get("timing", {})
                    fps = timing.get("fps", 0.0)
                    lat = timing.get("total_ms", 0.0)

                    dec = data.get("decision", {})
                    action = dec.get("action", "continue").upper()
                    heading_deg = dec.get("recommended_heading_deg")
                    heading_str = f"({heading_deg:+.1f}°)" if heading_deg is not None else "(straight)"

                    threat = data.get("primary_threat", {})
                    risk_lvl = threat.get("risk_level", "SAFE").upper()
                    risk_score = threat.get("risk_score", 0.0)
                    ttc = threat.get("ttc_s")
                    ttc_str = f"{ttc:.2f}s" if ttc is not None else "N/A"
                    cpa = threat.get("cpa_m")
                    cpa_str = f"{cpa:.2f}m" if cpa is not None else "N/A"

                    entities = data.get("entities", [])

                    # Color coding for terminal
                    action_color = "\033[92m" if "CONTINUE" in action else "\033[93m" if ("TURN" in action or "SLOW" in action) else "\033[91m"
                    risk_color = "\033[92m" if "SAFE" in risk_lvl else "\033[93m" if ("LOW" in risk_lvl or "MEDIUM" in risk_lvl) else "\033[91m"
                    reset_color = "\033[0m"

                    print(f"--- [Frame #{frame_id:05d}] {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} | FPS: {fps:4.1f} | Latency: {lat:4.2f}ms ---")
                    print(f"  Decision: {action_color}{action:<14}{reset_color} Heading: {heading_str:<10} Reason: {dec.get('reason')}")
                    print(f"  Risk:     {risk_color}{risk_lvl:<10}{reset_color} Score: {risk_score:4.1f}/100 | TTC: {ttc_str:<6} | CPA: {cpa_str}")
                    
                    if entities:
                        print(f"  Entities ({len(entities)} active):")
                        for ent in entities:
                            e_id = ent.get("id")
                            is_threat = (e_id == threat.get("id"))
                            prefix = "  -> [*THREAT*]" if is_threat else "     "
                            print(f"{prefix} ID#{e_id} [{ent.get('class')}]: Pos=(X:{ent.get('x'):+5.2f}m, Z:{ent.get('z'):4.2f}m) | Vel=(Vx:{ent.get('vx'):+4.2f}, Vz:{ent.get('vz'):+4.2f}) | Dist:{ent.get('distance_m'):4.2f}m | TTC:{ent.get('ttc_s'):4.2f}s | Risk:{ent.get('risk_score'):4.1f}")
                    else:
                        print("  Entities: [Corridor Clear - No Obstacles]")

                    cues = data.get("presentation_cues", [])
                    if cues:
                        cue_summaries = [f"[{c.get('channel')}: {c.get('text') or c.get('haptic_pattern') or c.get('direction')}]" for c in cues if c.get('channel') != 'visual']
                        if cue_summaries:
                            print(f"  Cues:     {' '.join(cue_summaries)}")

                    print()

                except json.JSONDecodeError:
                    continue

    except KeyboardInterrupt:
        print("\n[Stopped] Streaming paused by user.")
    except Exception as ex:
        print(f"\n[Connection Error] {ex}")
        print("\nTip: Make sure the PRISM Android app is running on your phone.")
        print("  - If using USB: run 'adb forward tcp:8080 tcp:8080' and pass '--adb'")
        print("  - If using Wi-Fi: pass '--url http://<phone-ip>:8080/stream'")
    finally:
        if f_out:
            f_out.close()
            print(f"[Export] Saved {total_frames} frames to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PRISM Live Telemetry Streaming Client")
    parser.add_argument("--url", type=str, default="http://localhost:8080/stream", help="Telemetry stream URL")
    parser.add_argument("--adb", action="store_true", help="Automatically set up ADB port forwarding (localhost:8080)")
    parser.add_argument("--raw", action="store_true", help="Print raw JSON lines")
    parser.add_argument("--output", type=str, default=None, help="Path to save output .jsonl session recording")

    args = parser.parse_args()

    if args.adb:
        setup_adb_forward(8080)
        url = "http://localhost:8080/stream"
    else:
        url = args.url

    output_file = args.output
    if output_file is None:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join("data", "telemetry_recordings", f"prism_session_{timestamp_str}.jsonl")

    stream_telemetry(url, output_file=output_file, raw_json=args.raw)


if __name__ == "__main__":
    main()
