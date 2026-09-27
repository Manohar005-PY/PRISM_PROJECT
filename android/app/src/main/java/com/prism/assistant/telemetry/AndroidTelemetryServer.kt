package com.prism.assistant.telemetry

import android.content.Context
import android.net.wifi.WifiManager
import android.util.Log
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader
import java.io.OutputStream
import java.io.PrintWriter
import java.net.InetAddress
import java.net.NetworkInterface
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Embedded HTTP / Server-Sent Events (SSE) Telemetry Server on Android.
 * Streams real-time detection & risk JSON to laptops over Wi-Fi or USB (via adb forward).
 */
class AndroidTelemetryServer(
    private val context: Context,
    private val telemetrySink: AndroidTelemetrySink,
    val port: Int = 8080
) {
    private var serverSocket: ServerSocket? = null
    private val isRunning = AtomicBoolean(false)
    private val executor = Executors.newCachedThreadPool()
    private val sseClients = CopyOnWriteArrayList<PrintWriter>()

    init {
        telemetrySink.addListener { jsonStr ->
            broadcastJsonToSse(jsonStr)
        }
    }

    fun start() {
        if (isRunning.getAndSet(true)) return

        executor.submit {
            try {
                serverSocket = ServerSocket(port)
                Log.i("PRISM_SERVER", "PRISM Telemetry Server listening on port $port")

                while (isRunning.get() && serverSocket != null && !serverSocket!!.isClosed) {
                    val clientSocket = serverSocket!!.accept()
                    executor.submit {
                        handleClient(clientSocket)
                    }
                }
            } catch (ex: Exception) {
                if (isRunning.get()) {
                    Log.e("PRISM_SERVER", "Server error: ${ex.message}")
                }
            }
        }
    }

    fun stop() {
        if (!isRunning.getAndSet(false)) return
        try {
            for (client in sseClients) {
                try { client.close() } catch (_: Exception) {}
            }
            sseClients.clear()
            serverSocket?.close()
            serverSocket = null
        } catch (ex: Exception) {
            Log.e("PRISM_SERVER", "Stop error: ${ex.message}")
        }
    }

    private fun handleClient(socket: Socket) {
        try {
            socket.soTimeout = 0 // Keep alive for streaming
            val reader = BufferedReader(InputStreamReader(socket.getInputStream()))
            val out = socket.getOutputStream()
            val writer = PrintWriter(out, true)

            val requestLine = reader.readLine() ?: return
            val tokens = requestLine.split(" ")
            val path = if (tokens.size > 1) tokens[1] else "/"

            when {
                path == "/stream" || path == "/telemetry" -> {
                    // Server-Sent Events (SSE) Stream
                    writer.print("HTTP/1.1 200 OK\r\n")
                    writer.print("Content-Type: text/event-stream\r\n")
                    writer.print("Cache-Control: no-cache\r\n")
                    writer.print("Connection: keep-alive\r\n")
                    writer.print("Access-Control-Allow-Origin: *\r\n\r\n")
                    writer.flush()

                    sseClients.add(writer)

                    // Keep socket open until client disconnects
                    while (isRunning.get() && !socket.isClosed) {
                        Thread.sleep(1000)
                    }
                }

                path == "/latest" -> {
                    val json = telemetrySink.getLatestJson()
                    val bytes = json.toByteArray(Charsets.UTF_8)
                    writer.print("HTTP/1.1 200 OK\r\n")
                    writer.print("Content-Type: application/json\r\n")
                    writer.print("Content-Length: ${bytes.size}\r\n")
                    writer.print("Access-Control-Allow-Origin: *\r\n\r\n")
                    writer.flush()
                    out.write(bytes)
                    out.flush()
                    socket.close()
                }

                path == "/export" || path == "/export_jsonl" -> {
                    val sessionFile = telemetrySink.getSessionFile()
                    val bytes = if (sessionFile != null && sessionFile.exists()) {
                        sessionFile.readBytes()
                    } else {
                        telemetrySink.getBufferedLines().joinToString("\n").toByteArray(Charsets.UTF_8)
                    }
                    val fileName = sessionFile?.name ?: "prism_session.jsonl"

                    writer.print("HTTP/1.1 200 OK\r\n")
                    writer.print("Content-Type: application/x-jsonlines\r\n")
                    writer.print("Content-Disposition: attachment; filename=\"$fileName\"\r\n")
                    writer.print("Content-Length: ${bytes.size}\r\n")
                    writer.print("Access-Control-Allow-Origin: *\r\n\r\n")
                    writer.flush()
                    out.write(bytes)
                    out.flush()
                    socket.close()
                }

                path == "/export_json" -> {
                    val jsonArrayStr = telemetrySink.getBufferedJsonArrayString()
                    val bytes = jsonArrayStr.toByteArray(Charsets.UTF_8)
                    val fileName = "prism_session_recording.json"

                    writer.print("HTTP/1.1 200 OK\r\n")
                    writer.print("Content-Type: application/json\r\n")
                    writer.print("Content-Disposition: attachment; filename=\"$fileName\"\r\n")
                    writer.print("Content-Length: ${bytes.size}\r\n")
                    writer.print("Access-Control-Allow-Origin: *\r\n\r\n")
                    writer.flush()
                    out.write(bytes)
                    out.flush()
                    socket.close()
                }

                else -> {
                    // HTML Dashboard
                    val html = getDashboardHtml()
                    val bytes = html.toByteArray(Charsets.UTF_8)
                    writer.print("HTTP/1.1 200 OK\r\n")
                    writer.print("Content-Type: text/html; charset=utf-8\r\n")
                    writer.print("Content-Length: ${bytes.size}\r\n")
                    writer.print("Access-Control-Allow-Origin: *\r\n\r\n")
                    writer.flush()
                    out.write(bytes)
                    out.flush()
                    socket.close()
                }
            }
        } catch (ex: Exception) {
            // Client disconnected
        }
    }

    private fun broadcastJsonToSse(jsonStr: String) {
        val deadClients = mutableListOf<PrintWriter>()
        for (client in sseClients) {
            try {
                client.print("data: $jsonStr\n\n")
                client.flush()
                if (client.checkError()) {
                    deadClients.add(client)
                }
            } catch (ex: Exception) {
                deadClients.add(client)
            }
        }
        sseClients.removeAll(deadClients)
    }

    fun getDeviceIpAddress(): String {
        try {
            val interfaces = NetworkInterface.getNetworkInterfaces()
            while (interfaces.hasMoreElements()) {
                val iface = interfaces.nextElement()
                if (iface.isLoopback || !iface.isUp) continue
                val addresses = iface.inetAddresses
                while (addresses.hasMoreElements()) {
                    val addr = addresses.nextElement()
                    if (!addr.isLoopbackAddress && addr.hostAddress?.contains(":") == false) {
                        return addr.hostAddress ?: "127.0.0.1"
                    }
                }
            }
        } catch (ex: Exception) {
            // Fallback
        }
        return "127.0.0.1"
    }

    private fun getDashboardHtml(): String {
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRISM Real-Time Telemetry & Corridor Validation Studio</title>
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: #131b2e;
            --card-header: #1c2742;
            --border-color: #243456;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --green: #10b981;
            --yellow: #f59e0b;
            --orange: #f97316;
            --red: #ef4444;
            --blue: #0ea5e9;
            --purple: #a855f7;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
        body { background: var(--bg-color); color: var(--text-main); padding: 16px; font-size: 14px; }
        header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 14px; border-bottom: 1px solid var(--border-color); margin-bottom: 16px; flex-wrap: wrap; gap: 12px; }
        .logo-box { display: flex; align-items: center; gap: 12px; }
        .logo { font-size: 22px; font-weight: 900; letter-spacing: 2px; color: var(--green); }
        .status-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 700; text-transform: uppercase; background: #10b98120; color: var(--green); border: 1px solid var(--green); }
        .status-badge.disconnected { background: #ef444420; color: var(--red); border-color: var(--red); }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; box-shadow: 0 0 8px currentColor; }
        
        .btn-group { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
        .btn { background: #2563eb; color: #fff; border: 1px solid transparent; padding: 7px 14px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 13px; text-decoration: none; display: inline-flex; align-items: center; gap: 6px; transition: all 0.15s ease; }
        .btn:hover { background: #1d4ed8; }
        .btn-secondary { background: #1e293b; color: var(--text-main); border-color: var(--border-color); }
        .btn-secondary:hover { background: #334155; }
        .btn-success { background: #059669; }
        .btn-success:hover { background: #047857; }
        .btn-warning { background: #d97706; }
        .btn-warning:hover { background: #b45309; }

        .nav-tabs { display: flex; gap: 8px; margin-bottom: 16px; border-bottom: 1px solid var(--border-color); padding-bottom: 8px; }
        .tab-btn { background: transparent; border: none; color: var(--text-muted); font-size: 14px; font-weight: 700; padding: 8px 16px; border-radius: 6px; cursor: pointer; }
        .tab-btn.active { background: var(--card-bg); color: var(--text-main); border: 1px solid var(--border-color); }

        .grid { display: grid; grid-template-columns: 340px 1fr; gap: 16px; }
        @media (max-width: 960px) { .grid { grid-template-columns: 1fr; } }
        
        .card { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 14px; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.25); }
        .card-title { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); margin-bottom: 10px; font-weight: 700; display: flex; justify-content: space-between; align-items: center; }
        
        .metric-val { font-size: 26px; font-weight: 800; margin-bottom: 2px; }
        .metric-sub { font-size: 12px; color: var(--text-muted); }
        
        .radar-container { position: relative; }
        .radar-canvas { background: #070b12; border: 1px solid var(--border-color); border-radius: 8px; width: 100%; height: 320px; display: block; }
        
        .stats-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-bottom: 16px; }
        .stat-box { background: #0e1526; border: 1px solid var(--border-color); border-radius: 6px; padding: 8px 10px; text-align: center; }
        .stat-box .count { font-size: 18px; font-weight: 800; }
        .stat-box .label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; }

        .table-custom { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 4px; font-family: monospace; }
        .table-custom th, .table-custom td { padding: 7px 9px; text-align: left; border-bottom: 1px solid var(--border-color); }
        .table-custom th { color: var(--text-muted); font-size: 11px; text-transform: uppercase; background: #0f172a; }
        
        .stream-box { background: #060911; border: 1px solid var(--border-color); border-radius: 6px; padding: 10px; height: 360px; overflow-y: auto; font-family: monospace; font-size: 11px; color: #38bdf8; line-height: 1.45; white-space: pre-wrap; word-break: break-all; }
        
        .scenario-card { background: #0e1526; border: 1px solid var(--border-color); border-radius: 8px; padding: 12px; margin-bottom: 12px; border-left: 4px solid var(--blue); }
        .scenario-card.crit { border-left-color: var(--red); }
        .scenario-card.warn { border-left-color: var(--yellow); }
        .scenario-card.pass { border-left-color: var(--green); }
        .scenario-title { font-size: 14px; font-weight: 700; color: #fff; margin-bottom: 4px; display: flex; justify-content: space-between; }
        .scenario-desc { font-size: 12px; color: var(--text-muted); line-height: 1.4; margin-bottom: 6px; }
        .scenario-metrics { display: flex; gap: 12px; font-size: 11px; background: #131b2e; padding: 6px 10px; border-radius: 4px; color: #cbd5e1; }
    </style>
</head>
<body>
    <header>
        <div class="logo-box">
            <div class="logo">PRISM</div>
            <div id="connBadge" class="status-badge"><span class="status-dot"></span><span id="connText">STREAMING LIVE</span></div>
            <span id="headerMetrics" style="font-family: monospace; font-size: 13px; color: var(--text-muted);">FPS: 0.0 | Latency: 0.0ms | Frames: 0</span>
        </div>
        <div class="btn-group">
            <button id="btnPause" class="btn btn-secondary" onclick="togglePauseStream()">⏸ Pause View</button>
            <button class="btn btn-secondary" onclick="downloadCurrentFrameJson()">⬇ Current Frame (.json)</button>
            <a href="/export" class="btn btn-success">⬇ Full Session (.jsonl)</a>
            <a href="/export_json" class="btn btn-warning">⬇ Full Session (.json)</a>
        </div>
    </header>

    <div class="nav-tabs">
        <button class="tab-btn active" onclick="switchTab('tabLive')">📡 Live Detection & Radar</button>
        <button class="tab-btn" onclick="switchTab('tabScenarios')">📋 Real Corridor Test Protocols (6 Scenarios)</button>
    </div>

    <!-- TAB 1: LIVE STUDIO -->
    <div id="tabLive">
        <!-- Event Statistics Counters -->
        <div class="stats-row">
            <div class="stat-box">
                <div id="cntSafe" class="count" style="color: var(--green);">0</div>
                <div class="label">Safe Cruises</div>
            </div>
            <div class="stat-box">
                <div id="cntSlow" class="count" style="color: var(--yellow);">0</div>
                <div class="label">Slow Downs</div>
            </div>
            <div class="stat-box">
                <div id="cntTurnL" class="count" style="color: var(--orange);">0</div>
                <div class="label">Turn Left</div>
            </div>
            <div class="stat-box">
                <div id="cntTurnR" class="count" style="color: var(--orange);">0</div>
                <div class="label">Turn Right</div>
            </div>
            <div class="stat-box">
                <div id="cntCrit" class="count" style="color: var(--red);">0</div>
                <div class="label">Emergency Stops</div>
            </div>
        </div>

        <div class="grid">
            <!-- Left Column: Decision & Radar -->
            <div>
                <div class="card">
                    <div class="card-title">Decision Arbitration</div>
                    <div id="actionVal" class="metric-val" style="color: var(--green);">CONTINUE</div>
                    <div id="reasonVal" class="metric-sub">no_hazard</div>
                    <div id="headingVal" style="margin-top: 6px; font-size: 13px; font-weight: 700; color: var(--blue);">Heading: Straight (0.0°)</div>
                </div>

                <div class="card">
                    <div class="card-title">Primary Threat & Risk</div>
                    <div id="riskVal" class="metric-val" style="color: var(--green);">SAFE (0.0)</div>
                    <div style="display: flex; justify-content: space-between; margin-top: 6px; font-size: 13px;">
                        <div><span style="color: var(--text-muted);">TTC:</span> <strong id="ttcVal">N/A</strong></div>
                        <div><span style="color: var(--text-muted);">CPA:</span> <strong id="cpaVal">N/A</strong></div>
                        <div><span style="color: var(--text-muted);">Range:</span> <strong id="distVal">N/A</strong></div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-title">
                        <span>2D Ground-Plane Radar</span>
                        <span style="font-size: 10px; color: var(--text-muted);">Max 5.5m</span>
                    </div>
                    <div class="radar-container">
                        <canvas id="radarCanvas" class="radar-canvas" width="308" height="320"></canvas>
                    </div>
                    <div style="text-align: center; font-size: 11px; color: var(--text-muted); margin-top: 6px;">
                        User at bottom center (0, 0) | Dashed box: Safety Corridor (|X| &le; 0.85m)
                    </div>
                </div>
            </div>

            <!-- Right Column: Table & JSON Stream -->
            <div>
                <div class="card">
                    <div class="card-title">
                        <span>Active Tracked Entities (<span id="activeCount">0</span>)</span>
                        <span id="sessionTimer" style="font-family: monospace; font-size: 12px; color: var(--text-muted);">Session: 00:00</span>
                    </div>
                    <table class="table-custom">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Status</th>
                                <th>Class</th>
                                <th>Position (X, Z)</th>
                                <th>Height</th>
                                <th>Geom Conf</th>
                                <th>Temp Conf</th>
                                <th>Phys Conf</th>
                                <th>TTC / Risk</th>
                            </tr>
                        </thead>
                        <tbody id="trackTableBody">
                            <tr><td colspan="9" style="text-align: center; color: var(--text-muted);">Corridor clear &mdash; no obstacles detected</td></tr>
                        </tbody>
                    </table>
                </div>

                <div class="card">
                    <div class="card-title">
                        <span>Per-Frame Telemetry Stream</span>
                        <div style="display: flex; gap: 6px;">
                            <button class="btn btn-secondary" style="padding: 3px 8px; font-size: 11px;" onclick="copyLatestJson()">📋 Copy JSON</button>
                            <button class="btn btn-secondary" style="padding: 3px 8px; font-size: 11px;" onclick="toggleAutoScroll()">Auto-Scroll: <span id="scrollState">ON</span></button>
                            <button class="btn btn-secondary" style="padding: 3px 8px; font-size: 11px;" onclick="clearStream()">Clear</button>
                        </div>
                    </div>
                    <div id="jsonStream" class="stream-box">Connecting to PRISM real-time stream...</div>
                </div>
            </div>
        </div>
    </div>

    <!-- TAB 2: CORRIDOR TEST PROTOCOLS -->
    <div id="tabScenarios" style="display: none;">
        <div class="card">
            <div class="card-title">PRISM Real-World Corridor Evaluation Suite</div>
            <p style="color: var(--text-muted); margin-bottom: 14px; font-size: 13px;">
                Execute these 6 standard test scenarios in your test corridor. Monitor live detections on this dashboard and verify that PRISM generates appropriate multimodal alerts with zero false stops.
            </p>

            <!-- Scenario 1 -->
            <div class="scenario-card pass">
                <div class="scenario-title">
                    <span>Scenario 1: Clear Corridor Baseline Walk (False-Positive Check)</span>
                    <span style="color: var(--green);">PASS CRITERIA: 0 False Stops</span>
                </div>
                <div class="scenario-desc">
                    <strong>Procedure:</strong> Walk at normal pace (~1.0 m/s) down an empty hallway with standard floor tiles, reflections, and side doors for 15-20 meters.
                </div>
                <div class="scenario-metrics">
                    <span><strong>Expected Decision:</strong> CONTINUE</span>
                    <span><strong>Risk:</strong> SAFE (0.0)</span>
                    <span><strong>Active Tracks:</strong> 0</span>
                    <span><strong>Audio:</strong> Silent (Green banner only)</span>
                </div>
            </div>

            <!-- Scenario 2 -->
            <div class="scenario-card warn">
                <div class="scenario-title">
                    <span>Scenario 2: Static Obstacle Evasion (Chair / Box in Mid-Corridor)</span>
                    <span style="color: var(--yellow);">PASS CRITERIA: Directional Turn Warning</span>
                </div>
                <div class="scenario-desc">
                    <strong>Procedure:</strong> Place an upright obstacle (e.g. office chair or box) on the right side of the corridor (X &approx; +0.3m, Z = 3.0m). Walk toward it.
                </div>
                <div class="scenario-metrics">
                    <span><strong>Expected Decision:</strong> TURN_LEFT (Recommended Heading: -0.42 rad / -24&deg;)</span>
                    <span><strong>Trigger Range:</strong> Z &le; 2.6m</span>
                    <span><strong>Audio:</strong> "Turn left"</span>
                </div>
            </div>

            <!-- Scenario 3 -->
            <div class="scenario-card warn">
                <div class="scenario-title">
                    <span>Scenario 3: Head-On Walking Pedestrian (Dynamic TTC Collision)</span>
                    <span style="color: var(--orange);">PASS CRITERIA: Dynamic Closing Velocity Vz &lt; -0.5 m/s</span>
                </div>
                <div class="scenario-desc">
                    <strong>Procedure:</strong> A volunteer walks toward the user at normal walking pace (~1.0 m/s) from 5 meters away in the same hallway lane.
                </div>
                <div class="scenario-metrics">
                    <span><strong>Expected Decision:</strong> SLOW_DOWN &rarr; TURN_RIGHT / EMERGENCY_STOP</span>
                    <span><strong>TTC Calculation:</strong> Active TTC &lt; 2.0s</span>
                    <span><strong>Audio:</strong> "Slow down" &rarr; "Turn right"</span>
                </div>
            </div>

            <!-- Scenario 4 -->
            <div class="scenario-card warn">
                <div class="scenario-title">
                    <span>Scenario 4: Lateral Crossing Pedestrian (Cross-Corridor Evasion)</span>
                    <span style="color: var(--blue);">PASS CRITERIA: Velocity Tracking Vx != 0</span>
                </div>
                <div class="scenario-desc">
                    <strong>Procedure:</strong> A person crosses the corridor from a side room doorway (moving laterally from left X = -1.5m to right X = +1.5m at Z = 2.5m).
                </div>
                <div class="scenario-metrics">
                    <span><strong>Expected Decision:</strong> Safe tracking with lateral clearance &rarr; SLOW_DOWN if intersecting path</span>
                    <span><strong>Audio:</strong> "Caution" / "Slow down"</span>
                </div>
            </div>

            <!-- Scenario 5 -->
            <div class="scenario-card crit">
                <div class="scenario-title">
                    <span>Scenario 5: Multi-Entity Bilateral Corridor Block (Corridor Congestion)</span>
                    <span style="color: var(--yellow);">PASS CRITERIA: Dual Track Tracking & Clearance Selection</span>
                </div>
                <div class="scenario-desc">
                    <strong>Procedure:</strong> Place two obstacles simultaneously on both sides of the corridor (Left X = -0.4m, Right X = +0.4m at Z = 2.5m).
                </div>
                <div class="scenario-metrics">
                    <span><strong>Expected Decision:</strong> SLOW_DOWN (Both corridors blocked) &rarr; STOP if distance &lt; 1.0m</span>
                    <span><strong>Active Tracks:</strong> 2</span>
                </div>
            </div>

            <!-- Scenario 6 -->
            <div class="scenario-card crit">
                <div class="scenario-title">
                    <span>Scenario 6: Sudden Emergency Proximity Obstacle (Critical Distance Stop)</span>
                    <span style="color: var(--red);">PASS CRITERIA: Immediate Preemption Emergency Stop</span>
                </div>
                <div class="scenario-desc">
                    <strong>Procedure:</strong> User approaches within 0.8 meters of an obstacle, or a person steps directly in front of the camera (Z &lt; 1.0m).
                </div>
                <div class="scenario-metrics">
                    <span><strong>Expected Decision:</strong> EMERGENCY_STOP</span>
                    <span><strong>Risk Level:</strong> CRITICAL</span>
                    <span><strong>Audio:</strong> "Stop" (Flush Preemption) + Emergency Haptic Pulse</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        let autoScroll = true;
        let isPaused = false;
        let lastFrameData = null;
        let sessionStartTime = Date.now();
        let counts = { safe: 0, slow: 0, turnL: 0, turnR: 0, crit: 0 };

        const streamBox = document.getElementById('jsonStream');
        const radar = document.getElementById('radarCanvas');
        const rCtx = radar.getContext('2d');

        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('tabLive').style.display = (tabId === 'tabLive') ? 'block' : 'none';
            document.getElementById('tabScenarios').style.display = (tabId === 'tabScenarios') ? 'block' : 'none';
        }

        function toggleAutoScroll() {
            autoScroll = !autoScroll;
            document.getElementById('scrollState').innerText = autoScroll ? 'ON' : 'OFF';
        }

        function togglePauseStream() {
            isPaused = !isPaused;
            document.getElementById('btnPause').innerText = isPaused ? '▶ Resume View' : '⏸ Pause View';
            document.getElementById('btnPause').classList.toggle('btn-warning', isPaused);
        }

        function clearStream() {
            streamBox.innerText = '';
        }

        function copyLatestJson() {
            if (!lastFrameData) return;
            navigator.clipboard.writeText(JSON.stringify(lastFrameData, null, 2)).then(() => {
                alert('Copied latest frame telemetry JSON to clipboard!');
            });
        }

        function downloadCurrentFrameJson() {
            if (!lastFrameData) {
                alert('No frame telemetry received yet.');
                return;
            }
            const blob = new Blob([JSON.stringify(lastFrameData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'prism_frame_' + (lastFrameData.frame_id || 'snap') + '.json';
            a.click();
            URL.revokeObjectURL(url);
        }

        // Update Session Timer
        setInterval(() => {
            const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
            const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
            const secs = String(elapsed % 60).padStart(2, '0');
            document.getElementById('sessionTimer').innerText = 'Session: ' + mins + ':' + secs;
        }, 1000);

        function drawRadar(entities, primaryThreatId) {
            const w = radar.width;
            const h = radar.height;
            rCtx.clearRect(0, 0, w, h);

            const originX = w / 2;
            const originY = h - 25;
            const scaleZ = (h - 40) / 5.5; // 5.5m max range
            const scaleX = scaleZ;

            // Distance concentric range arcs
            rCtx.strokeStyle = '#1e293b';
            rCtx.lineWidth = 1;
            for (let d = 1; d <= 5; d++) {
                rCtx.beginPath();
                rCtx.arc(originX, originY, d * scaleZ, Math.PI, 0);
                rCtx.stroke();
                rCtx.fillStyle = '#475569';
                rCtx.font = '10px monospace';
                rCtx.fillText(d + 'm', originX + 5, originY - (d * scaleZ) + 10);
            }

            // Danger Stop Zone (Z < 1.0m, |X| < 0.65m)
            rCtx.fillStyle = 'rgba(239, 68, 68, 0.08)';
            rCtx.fillRect(originX - (0.65 * scaleX), originY - (1.0 * scaleZ), 1.3 * scaleX, 1.0 * scaleZ);

            // Safety Corridor Boundaries (|X| <= 0.85m)
            rCtx.setLineDash([4, 4]);
            rCtx.strokeStyle = '#334155';
            rCtx.beginPath();
            rCtx.moveTo(originX - (0.85 * scaleX), originY);
            rCtx.lineTo(originX - (0.85 * scaleX), 10);
            rCtx.moveTo(originX + (0.85 * scaleX), originY);
            rCtx.lineTo(originX + (0.85 * scaleX), 10);
            rCtx.stroke();
            rCtx.setLineDash([]);

            // User Origin Avatar
            rCtx.fillStyle = '#10b981';
            rCtx.beginPath();
            rCtx.arc(originX, originY, 6, 0, Math.PI * 2);
            rCtx.fill();

            // Render Entities
            if (entities && entities.length > 0) {
                entities.forEach(ent => {
                    const eX = originX + (ent.x * scaleX);
                    const eY = originY - (ent.z * scaleZ);

                    const isPrimary = (ent.id === primaryThreatId);
                    const isPerson = (ent['class'] === 'person');
                    
                    rCtx.fillStyle = isPrimary ? '#ef4444' : isPerson ? '#a855f7' : '#f59e0b';
                    rCtx.beginPath();
                    rCtx.arc(eX, eY, isPrimary ? 8 : 6, 0, Math.PI * 2);
                    rCtx.fill();

                    // Velocity Vector Arrow
                    if (ent.vx !== 0 || ent.vz !== 0) {
                        rCtx.strokeStyle = isPrimary ? '#ef4444' : '#94a3b8';
                        rCtx.lineWidth = 2;
                        rCtx.beginPath();
                        rCtx.moveTo(eX, eY);
                        rCtx.lineTo(eX + (ent.vx * 16), eY - (ent.vz * 16));
                        rCtx.stroke();
                    }

                    // Entity Tag
                    rCtx.fillStyle = '#ffffff';
                    rCtx.font = 'bold 10px monospace';
                    rCtx.fillText('#' + ent.id + ' (' + ent.z.toFixed(1) + 'm)', eX + 8, eY + 3);
                });
            }
        }

        // Connect Server-Sent Events (SSE)
        function connectSSE() {
            const eventSource = new EventSource('/stream');

            eventSource.onopen = function() {
                document.getElementById('connBadge').classList.remove('disconnected');
                document.getElementById('connText').innerText = 'STREAMING LIVE';
            };

            eventSource.onerror = function() {
                document.getElementById('connBadge').classList.add('disconnected');
                document.getElementById('connText').innerText = 'RECONNECTING...';
            };

            eventSource.onmessage = function(e) {
                try {
                    const data = JSON.parse(e.data);
                    lastFrameData = data;

                    if (isPaused) return;

                    // Update Counters
                    const actLower = (data.decision.action || '').toLowerCase();
                    if (actLower === 'continue') counts.safe++;
                    else if (actLower === 'slow_down') counts.slow++;
                    else if (actLower === 'turn_left') counts.turnL++;
                    else if (actLower === 'turn_right') counts.turnR++;
                    else if (actLower === 'emergency_stop' || actLower === 'stop') counts.crit++;

                    document.getElementById('cntSafe').innerText = counts.safe;
                    document.getElementById('cntSlow').innerText = counts.slow;
                    document.getElementById('cntTurnL').innerText = counts.turnL;
                    document.getElementById('cntTurnR').innerText = counts.turnR;
                    document.getElementById('cntCrit').innerText = counts.crit;

                    // Header HUD Metrics
                    document.getElementById('headerMetrics').innerText = 
                        'FPS: ' + data.timing.fps.toFixed(1) + 
                        ' | Latency: ' + data.timing.total_ms.toFixed(1) + 'ms' +
                        ' | Frame #' + data.frame_id;

                    // Decision Banner
                    const actEl = document.getElementById('actionVal');
                    actEl.innerText = data.decision.action.toUpperCase();
                    actEl.style.color = (actLower === 'continue') ? 'var(--green)' : 
                                        (actLower.includes('turn') || actLower.includes('slow')) ? 'var(--yellow)' : 'var(--red)';
                    
                    document.getElementById('reasonVal').innerText = data.decision.reason;
                    const hDeg = data.decision.recommended_heading_deg;
                    document.getElementById('headingVal').innerText = 'Heading: ' + 
                        (hDeg !== null && hDeg !== undefined ? (hDeg > 0 ? '+' : '') + hDeg.toFixed(1) + '°' : 'Straight (0.0°)');

                    // Risk & Threat
                    const threat = data.primary_threat || {};
                    const riskEl = document.getElementById('riskVal');
                    const rLvl = (threat.risk_level || 'safe').toLowerCase();
                    riskEl.innerText = rLvl.toUpperCase() + ' (' + (threat.risk_score || 0).toFixed(1) + ')';
                    riskEl.style.color = (rLvl === 'safe') ? 'var(--green)' : 
                                         (rLvl === 'low' || rLvl === 'medium') ? 'var(--yellow)' : 
                                         (rLvl === 'high') ? 'var(--orange)' : 'var(--red)';

                    document.getElementById('ttcVal').innerText = threat.ttc_s ? threat.ttc_s.toFixed(2) + 's' : 'N/A';
                    document.getElementById('cpaVal').innerText = threat.cpa_m ? threat.cpa_m.toFixed(2) + 'm' : 'N/A';
                    
                    const entList = data.entities || [];
                    document.getElementById('activeCount').innerText = entList.length;
                    const primaryEnt = entList.find(e => e.id === threat.id);
                    document.getElementById('distVal').innerText = primaryEnt ? primaryEnt.distance_m.toFixed(2) + 'm' : 'N/A';

                    // Update Entities Table
                    const tbody = document.getElementById('trackTableBody');
                    if (entList.length > 0) {
                        tbody.innerHTML = entList.map(ent => {
                            const isThreat = (ent.id === threat.id);
                            const idColor = isThreat ? '#ef4444' : '#ffffff';
                            const signX = ent.x > 0 ? '+' : '';
                            const ttcText = ent.ttc_s ? ent.ttc_s.toFixed(2) + 's' : 'N/A';
                            const statusBadge = '<span style="padding:2px 6px; border-radius:4px; font-size:11px; font-weight:bold; background:' + 
                                (ent.status === 'CONFIRMED' ? 'rgba(16, 185, 129, 0.2); color:#10b981;' : 'rgba(245, 158, 11, 0.2); color:#f59e0b;') + '">' + (ent.status || 'TRK') + '</span>';
                            const geomConf = ent.geometric_confidence !== undefined ? ent.geometric_confidence.toFixed(2) : '1.00';
                            const tempConf = ent.temporal_confidence !== undefined ? ent.temporal_confidence.toFixed(2) : '1.00';
                            const physConf = ent.physical_confidence !== undefined ? ent.physical_confidence.toFixed(2) : '1.00';
                            const heightText = ent.height_m !== undefined ? ent.height_m.toFixed(2) + 'm' : 'N/A';

                            return '<tr>' +
                                '<td style="font-weight:bold; color:' + idColor + ';">#' + ent.id + '</td>' +
                                '<td>' + statusBadge + '</td>' +
                                '<td>' + ent['class'] + '</td>' +
                                '<td>(' + signX + ent.x.toFixed(2) + ', ' + ent.z.toFixed(2) + ')</td>' +
                                '<td>' + heightText + '</td>' +
                                '<td>' + geomConf + '</td>' +
                                '<td>' + tempConf + '</td>' +
                                '<td><strong style="color:' + (physConf >= 0.7 ? '#10b981' : physConf >= 0.4 ? '#f59e0b' : '#ef4444') + ';">' + physConf + '</strong></td>' +
                                '<td>' + ttcText + ' / <strong>' + ent.risk_score.toFixed(1) + '</strong></td>' +
                                '</tr>';
                        }).join('');
                    } else {
                        tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: var(--text-muted);">Corridor clear &mdash; no obstacles detected</td></tr>';
                    }

                    // Render 2D Radar
                    drawRadar(entList, threat.id);

                    // Append JSON Log
                    const formattedJson = JSON.stringify(data, null, 2);
                    if (streamBox.innerText.length > 35000) {
                        streamBox.innerText = streamBox.innerText.substring(18000);
                    }
                    streamBox.innerText += '\n--- Frame #' + data.frame_id + ' [' + new Date().toLocaleTimeString() + '] ---\n' + formattedJson;
                    if (autoScroll) {
                        streamBox.scrollTop = streamBox.scrollHeight;
                    }
                } catch (err) {
                    console.error(err);
                }
            };
        }

        connectSSE();
        drawRadar([], null);
    </script>
</body>
</html>
        """.trimIndent()
    }
}
