import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from backend.app.utils.logger import logger

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
METRICS_FILE = DATA_DIR / "operational_metrics.json"

class OperationalMetricsTracker:
    """
    Thread-safe operational telemetry tracker for VoiceAffect translation pipeline.
    Tracks live execution stats without storing sensitive raw audio:
      - Total, successful, and failed requests
      - Stage latency breakdowns (ASR, SER, NMT, TTS, Total)
      - Translation direction counts (Hindi -> English, English -> Hindi)
      - Recent request timeline for research dashboard analytics
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or METRICS_FILE
        self._lock = threading.Lock()
        self._total_requests: int = 0
        self._successful_requests: int = 0
        self._failed_requests: int = 0
        self._direction_counts: Dict[str, int] = {
            "hi_to_en": 0,
            "en_to_hi": 0,
            "auto": 0
        }
        self._latency_sums: Dict[str, float] = {
            "asr_ms": 0.0,
            "ser_ms": 0.0,
            "nmt_ms": 0.0,
            "tts_ms": 0.0,
            "total_ms": 0.0
        }
        self._latency_counts: Dict[str, int] = {
            "asr_ms": 0,
            "ser_ms": 0,
            "nmt_ms": 0,
            "tts_ms": 0,
            "total_ms": 0
        }
        self._recent_requests: List[Dict[str, Any]] = []
        self._max_recent: int = 50

        # Load persisted metrics if available
        self._load()

    def record_request(
        self,
        request_id: str,
        source_lang: str,
        target_lang: str,
        success: bool,
        latencies: Dict[str, float],
        detected_emotion: Optional[str] = None,
        tts_engine: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> None:
        """Records a completed or failed translation request."""
        with self._lock:
            self._total_requests += 1
            if success:
                self._successful_requests += 1
            else:
                self._failed_requests += 1

            # Direction tracking
            norm_src = "hi" if (source_lang and source_lang.startswith("hi")) else ("en" if (source_lang and source_lang.startswith("en")) else "unknown")
            norm_tgt = "en" if (target_lang and target_lang.startswith("en")) else ("hi" if (target_lang and target_lang.startswith("hi")) else "unknown")

            if norm_src == "hi" and norm_tgt == "en":
                dir_key = "hi_to_en"
            elif norm_src == "en" and norm_tgt == "hi":
                dir_key = "en_to_hi"
            else:
                dir_key = f"{norm_src}_to_{norm_tgt}"

            self._direction_counts[dir_key] = self._direction_counts.get(dir_key, 0) + 1

            # Accumulate latencies
            for k in ("asr_ms", "ser_ms", "nmt_ms", "tts_ms", "total_ms"):
                val = latencies.get(k)
                if val is not None and val > 0:
                    self._latency_sums[k] += float(val)
                    self._latency_counts[k] += 1

            # Sanitized entry without raw audio
            entry = {
                "request_id": request_id,
                "timestamp": round(time.time(), 3),
                "direction": dir_key,
                "source_lang": norm_src,
                "target_lang": norm_tgt,
                "success": bool(success),
                "detected_emotion": detected_emotion or "neutral",
                "tts_engine": tts_engine or "macOS say",
                "latencies": {
                    "asr_ms": round(latencies.get("asr_ms", 0.0), 1),
                    "ser_ms": round(latencies.get("ser_ms", 0.0), 1),
                    "nmt_ms": round(latencies.get("nmt_ms", 0.0), 1),
                    "tts_ms": round(latencies.get("tts_ms", 0.0), 1),
                    "total_ms": round(latencies.get("total_ms", 0.0), 1),
                },
                "error": error_message
            }

            self._recent_requests.append(entry)
            if len(self._recent_requests) > self._max_recent:
                self._recent_requests = self._recent_requests[-self._max_recent:]

            self._save()

    def get_overview(self) -> Dict[str, Any]:
        """Returns high-level operational research overview using genuine collected data."""
        with self._lock:
            # Average pipeline latency
            total_cnt = self._latency_counts.get("total_ms", 0)
            avg_pipeline_ms = round(self._latency_sums["total_ms"] / total_cnt, 2) if total_cnt > 0 else 0.0

            # Most frequently used translation direction
            most_freq_dir = "None"
            max_dir_count = 0
            for d, cnt in self._direction_counts.items():
                if cnt > max_dir_count:
                    max_dir_count = cnt
                    most_freq_dir = d

            # Stage averages
            stage_averages = {}
            for k in ("asr_ms", "ser_ms", "nmt_ms", "tts_ms", "total_ms"):
                cnt = self._latency_counts.get(k, 0)
                stage_averages[k] = round(self._latency_sums[k] / cnt, 2) if cnt > 0 else 0.0

            return {
                "total_requests": self._total_requests,
                "successful_requests": self._successful_requests,
                "failed_requests": self._failed_requests,
                "success_rate_percent": round((self._successful_requests / self._total_requests) * 100, 1) if self._total_requests > 0 else 0.0,
                "average_pipeline_latency_ms": avg_pipeline_ms,
                "most_frequently_used_direction": most_freq_dir,
                "direction_distribution": dict(self._direction_counts),
                "stage_average_latencies": stage_averages,
                "recent_requests_count": len(self._recent_requests),
                "recent_requests": list(reversed(self._recent_requests[-20:]))
            }

    def reset_metrics(self) -> None:
        """Resets all collected operational metrics."""
        with self._lock:
            self._total_requests = 0
            self._successful_requests = 0
            self._failed_requests = 0
            self._direction_counts = {"hi_to_en": 0, "en_to_hi": 0, "auto": 0}
            self._latency_sums = {k: 0.0 for k in self._latency_sums}
            self._latency_counts = {k: 0 for k in self._latency_counts}
            self._recent_requests = []
            if self.storage_path.exists():
                try:
                    self.storage_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove metrics file: {e}")

    def _save(self) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "total_requests": self._total_requests,
                "successful_requests": self._successful_requests,
                "failed_requests": self._failed_requests,
                "direction_counts": self._direction_counts,
                "latency_sums": self._latency_sums,
                "latency_counts": self._latency_counts,
                "recent_requests": self._recent_requests
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not persist operational metrics: {e}")

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._total_requests = data.get("total_requests", 0)
            self._successful_requests = data.get("successful_requests", 0)
            self._failed_requests = data.get("failed_requests", 0)
            self._direction_counts = data.get("direction_counts", {"hi_to_en": 0, "en_to_hi": 0, "auto": 0})
            self._latency_sums = data.get("latency_sums", {k: 0.0 for k in self._latency_sums})
            self._latency_counts = data.get("latency_counts", {k: 0 for k in self._latency_counts})
            self._recent_requests = data.get("recent_requests", [])
        except Exception as e:
            logger.warning(f"Could not load operational metrics from disk: {e}")

# Global singleton instance
metrics_tracker = OperationalMetricsTracker()

