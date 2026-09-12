# pyright: reportMissingImports=false
"""
Automated Natural Language Incident Report Generator.

Primary  : Ollama local LLM (phi3:mini) — runs fully offline, no API key.
Fallback : Rule-based NLG — only used if Ollama fails or times out.

Flow per report cycle:
    1. Previous report stays visible while new one is being generated.
    2. Background thread tries Ollama first (OLLAMA_TIMEOUT seconds).
    3. If Ollama succeeds  → display LLM-generated report.
    4. If Ollama fails     → display rule-based report.
    No flickering. No switching mid-display.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from typing import List, Optional

logger = logging.getLogger(__name__)

OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_MODEL   = "phi3:mini"
OLLAMA_TIMEOUT = 10          # seconds to wait for LLM before falling back

_GRID = {
    (0, 0): "northwest", (0, 1): "northern",  (0, 2): "northeast",
    (1, 0): "western",   (1, 1): "central",   (1, 2): "eastern",
    (2, 0): "southwest", (2, 1): "southern",  (2, 2): "southeast",
}


# ── Ollama ────────────────────────────────────────────────────────────────────

def _ollama_running() -> bool:
    try:
        with urllib.request.urlopen("http://localhost:11434", timeout=2):
            return True
    except Exception:
        return False


def _call_ollama(prompt: str) -> Optional[str]:
    payload = json.dumps({
        "model"  : OLLAMA_MODEL,
        "prompt" : prompt,
        "stream" : False,
        "options": {"temperature": 0.3, "num_predict": 120},
    }).encode()
    try:
        req = urllib.request.Request(
            OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            return json.loads(resp.read().decode()).get("response", "").strip()
    except Exception as e:
        logger.debug("Ollama unavailable: %s", e)
        return None


# ── Rule-based fallback ───────────────────────────────────────────────────────

def _confidence_word(avg_conf: float) -> str:
    if avg_conf >= 0.75: return "High-confidence"
    if avg_conf >= 0.50: return "Moderate-confidence"
    return "Possible"


def _rule_description(fire_count, smoke_count, avg_conf, region, trend, consecutive, prediction):
    conf = _confidence_word(avg_conf)
    what = ("fire and smoke" if fire_count > 0 and smoke_count > 0
            else "fire" if fire_count > 0 else "smoke")
    parts = [f"{conf} {what} detected in the {region} of the monitored area."]

    if smoke_count > 0 and fire_count > 0:
        if trend == "increasing":
            parts.append(
                f"Smoke density is increasing and the fire region has expanded"
                f" across {consecutive} consecutive frames."
            )
        else:
            parts.append(
                f"Smoke presence observed. Detection sustained across"
                f" {consecutive} consecutive frames."
            )
    elif consecutive >= 5:
        parts.append(f"Detection sustained across {consecutive} consecutive frames.")

    if prediction and prediction.get("available"):
        growth = prediction.get("growth_label", "stable")
        risk   = int(prediction.get("risk_score", 0))
        if growth == "critical":
            parts.append(
                f"AI progression analysis indicates rapidly escalating fire behavior"
                f" (risk score: {risk}/100)."
            )
        elif growth == "growing":
            parts.append(
                f"AI progression analysis indicates growing fire activity"
                f" (risk score: {risk}/100)."
            )
    return " ".join(parts)


def _rule_recommendation(severity, fire_count):
    if severity == "critical" or fire_count >= 2:
        return ("Possible rapidly developing fire. Immediate evacuation of the"
                " affected zone is recommended. Contact emergency services without delay.")
    if severity == "high" or fire_count >= 1:
        return ("Alert emergency services immediately."
                " Begin evacuation procedures for the affected zone.")
    if severity == "medium":
        return ("Closely monitor the situation. Prepare evacuation protocols"
                " and alert facility management.")
    return "Continue monitoring. Verify detection with additional sensors if available."


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(fire_count, smoke_count, avg_conf, region, trend, consecutive, severity, prediction):
    pred_line = ""
    if prediction and prediction.get("available"):
        pred_line = (f"- AI fire progression: {prediction.get('growth_label','stable')}"
                     f" (risk: {int(prediction.get('risk_score',0))}/100)\n")
    return (
        "You are a fire safety monitoring AI. Write exactly 2 sentences based on the data below.\n"
        "Sentence 1: describe the incident (what, where, how long, severity).\n"
        "Sentence 2: give a safety recommendation.\n"
        "Be concise and professional. No extra text.\n\n"
        f"- Fire detections: {fire_count}\n"
        f"- Smoke detections: {smoke_count}\n"
        f"- Confidence: {avg_conf * 100:.0f}%\n"
        f"- Location: {region}\n"
        f"- Trend: {trend}\n"
        f"- Consecutive frames: {consecutive}\n"
        f"- Severity: {severity}\n"
        f"{pred_line}"
        "\nResponse:"
    )


# ── Reporter ──────────────────────────────────────────────────────────────────

class IncidentReporter:
    """
    Call update() every frame. Report refreshes every `report_interval` frames.
    Previous report stays on screen while the next one is being generated.
    Ollama is always tried first; rule-based only runs if Ollama fails.
    """

    def __init__(self, report_interval: int = 30, area_history_len: int = 60):
        self._interval         = report_interval
        self._area_history: List[float] = []
        self._area_history_len = area_history_len
        self._consecutive_frames = 0
        self._last_report: dict  = {}
        self._ready_report: dict = {}   # finished background result, waiting to be swapped
        self._last_report_frame  = -(report_interval + 1)
        self._generating         = False
        self._lock               = threading.Lock()

        self._ollama_ok = _ollama_running()
        if self._ollama_ok:
            logger.info("Ollama online — LLM reports active (%s)", OLLAMA_MODEL)
        else:
            logger.info("Ollama offline — rule-based NLG active")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _region(self, detections, frame_shape) -> str:
        if not detections:
            return "monitored area"
        h, w = frame_shape[:2]
        fire  = [d for d in detections if d.label == "fire"]
        ref   = fire[0] if fire else detections[0]
        cx    = (ref.bbox[0] + ref.bbox[2]) / 2 / max(w, 1)
        cy    = (ref.bbox[1] + ref.bbox[3]) / 2 / max(h, 1)
        return _GRID.get((min(int(cy * 3), 2), min(int(cx * 3), 2)), "central") + " region"

    def _trend(self) -> str:
        if len(self._area_history) < 10:
            return "stable"
        w = self._area_history[-10:]
        s = (w[-1] - w[0]) / len(w)
        return "increasing" if s > 0.005 else "decreasing" if s < -0.005 else "stable"

    # ── Background worker ─────────────────────────────────────────────────────

    def _worker(self, prompt: str, ctx: dict):
        """
        Try Ollama first. If it fails or times out, use rule-based.
        Result is stored in _ready_report and swapped on the next update() call.
        """
        response = _call_ollama(prompt) if self._ollama_ok else None

        if response:
            lines  = [l.strip() for l in response.splitlines() if l.strip()]
            desc   = lines[0] if lines else ctx["description"]
            rec    = lines[1] if len(lines) > 1 else ctx["recommendation"]
            source = "llm"
        else:
            # Ollama failed — generate rule-based inside the thread
            desc   = _rule_description(
                ctx["fire_count"], ctx["smoke_count"], ctx["avg_conf"],
                ctx["region"], ctx["trend"], ctx["consecutive"], ctx["prediction"],
            )
            rec    = _rule_recommendation(ctx["severity"], ctx["fire_count"])
            source = "rule-based"

        with self._lock:
            self._ready_report = {
                "active"            : True,
                "description"       : desc,
                "recommendation"    : rec,
                "severity"          : ctx["severity"],
                "region"            : ctx["region"],
                "consecutive_frames": ctx["consecutive"],
                "trend"             : ctx["trend"],
                "timestamp"         : time.strftime("%H:%M:%S"),
                "source"            : source,
            }
            self._generating = False

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, detections, stats: dict, frame_idx: int, frame_shape,
               prediction: Optional[dict] = None) -> dict:

        fire_count  = stats.get("fire_count",  0)
        smoke_count = stats.get("smoke_count", 0)
        total       = fire_count + smoke_count

        # Track fire area for trend
        h, w        = frame_shape[:2]
        fire_px     = sum(
            (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
            for d in detections if d.label == "fire"
        )
        self._area_history.append(fire_px / max(h * w, 1))
        if len(self._area_history) > self._area_history_len:
            self._area_history.pop(0)

        if total > 0:
            self._consecutive_frames += 1
        else:
            self._consecutive_frames = 0
            self._last_report = {"active": False, "description": "", "recommendation": ""}
            return self._last_report

        # Swap in any finished background result
        with self._lock:
            if self._ready_report:
                self._last_report  = self._ready_report
                self._ready_report = {}

        # Check if it's time to start a new generation cycle
        if (frame_idx - self._last_report_frame) < self._interval:
            return self._last_report

        # Don't start another if one is already running
        if self._generating:
            return self._last_report

        self._last_report_frame = frame_idx
        self._generating        = True

        region   = self._region(detections, frame_shape)
        trend    = self._trend()
        avg_conf = stats.get("avg_confidence", 0.5)
        severity = stats.get("severity_label", "low")

        ctx = {
            "fire_count" : fire_count,
            "smoke_count": smoke_count,
            "avg_conf"   : avg_conf,
            "region"     : region,
            "trend"      : trend,
            "consecutive": self._consecutive_frames,
            "severity"   : severity,
            "prediction" : prediction,
        }
        prompt = _build_prompt(
            fire_count, smoke_count, avg_conf, region,
            trend, self._consecutive_frames, severity, prediction,
        )
        threading.Thread(target=self._worker, args=(prompt, ctx), daemon=True).start()

        # Return previous report while new one generates (no flicker)
        return self._last_report

    def reset(self):
        self._area_history.clear()
        self._consecutive_frames = 0
        self._last_report        = {}
        self._ready_report       = {}
        self._generating         = False
        self._last_report_frame  = -(self._interval + 1)
