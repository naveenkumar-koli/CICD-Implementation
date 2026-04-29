"""
drift_monitor.py
Evidently AI-powered drift detection for Sales Sentiment Analysis.
Logs predictions and generates HTML reports for data/model drift.
"""

import os
import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path

# Evidently AI — optional, graceful fallback
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import ColumnDriftMetric
    import pandas as pd
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

import logging
logger = logging.getLogger(__name__)


class DriftMonitor:
    """
    Tracks prediction statistics and detects drift using Evidently AI.
    """

    def __init__(
        self,
        log_path: str = "app/models/prediction_log.json",
        reference_path: str = "app/models/reference_data.json",
        reports_dir: str = "app/reports/drift",
    ):
        self.log_path = log_path
        self.reference_path = reference_path
        self.reports_dir = reports_dir
        Path(reports_dir).mkdir(parents=True, exist_ok=True)

    # ── Prediction Logging ────────────────────────────────────────────────

    def log_prediction(self, text: str, sentiment: str, confidence: float, method: str):
        """Append one prediction record to the rolling log (max 2000 entries)."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "sentiment": sentiment,
            "confidence": round(float(confidence), 4),
            "method": method,
            "text_length": len(text),
        }
        logs = self._load_json(self.log_path)
        logs.append(entry)
        self._save_json(self.log_path, logs[-2000:])   # keep last 2000

    # ── Simple Statistics Drift Check (no Evidently required) ────────────

    def check_drift(self, window_hours: int = 24) -> dict:
        """
        Returns a health dict based on recent predictions.
        Works even if Evidently is not installed.
        """
        logs = self._load_json(self.log_path)
        if not logs:
            return {"status": "no_data", "alerts": [], "sample_size": 0}

        cutoff = datetime.now() - timedelta(hours=window_hours)
        recent = [
            l for l in logs
            if datetime.fromisoformat(l["timestamp"]) > cutoff
        ]

        if not recent:
            return {"status": "no_recent_data", "alerts": [], "sample_size": 0}

        alerts = []
        confidences   = [l["confidence"] for l in recent]
        avg_conf      = statistics.mean(confidences)
        neg_count     = sum(1 for l in recent if l["sentiment"] == "negative")
        neg_rate      = neg_count / len(recent)
        low_conf_rate = sum(1 for c in confidences if c < 0.60) / len(recent)

        if avg_conf < 0.65:
            alerts.append(f"Low average confidence: {avg_conf:.3f} (threshold 0.65)")
        if neg_rate > 0.70:
            alerts.append(f"High negative rate: {neg_rate:.1%} (threshold 70%)")
        if low_conf_rate > 0.40:
            alerts.append(f"High low-confidence rate: {low_conf_rate:.1%} (threshold 40%)")

        return {
            "status": "alert" if alerts else "healthy",
            "alerts": alerts,
            "avg_confidence": round(avg_conf, 3),
            "negative_rate": round(neg_rate, 3),
            "low_confidence_rate": round(low_conf_rate, 3),
            "sample_size": len(recent),
            "window_hours": window_hours,
        }

    # ── Evidently HTML Report ─────────────────────────────────────────────

    def generate_evidently_report(self, current_df) -> dict:
        """
        Compare current batch against the saved reference dataset.
        Returns the path to the generated HTML report.
        Requires: evidently, pandas
        """
        if not EVIDENTLY_AVAILABLE:
            return {"status": "evidently_not_installed", "report_path": None}

        import pandas as pd

        # Save current batch as reference if none exists
        if not os.path.exists(self.reference_path):
            ref_data = current_df[["confidence_numeric"]].rename(
                columns={"confidence_numeric": "confidence"}
            ).to_dict(orient="records") if "confidence_numeric" in current_df.columns \
                else current_df[["Confidence"]].rename(
                columns={"Confidence": "confidence"}
            ).to_dict(orient="records")
            self._save_json(self.reference_path, ref_data)
            logger.info("[DRIFT] No reference found — saved current batch as reference.")
            return {"status": "reference_created", "report_path": None}

        # Load reference
        ref_data = self._load_json(self.reference_path)
        ref_df   = pd.DataFrame(ref_data)

        # Prepare current
        if "Confidence" in current_df.columns:
            cur_df = current_df[["Confidence"]].rename(columns={"Confidence": "confidence"})
        else:
            logger.warning("[DRIFT] No Confidence column found in current data.")
            return {"status": "missing_confidence_column", "report_path": None}

        try:
            report = Report(metrics=[
                DataDriftPreset(),
                ColumnDriftMetric(column_name="confidence"),
            ])
            report.run(reference_data=ref_df, current_data=cur_df)

            timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(self.reports_dir, f"drift_report_{timestamp}.html")
            report.save_html(report_path)

            result      = report.as_dict()
            drift_flag  = result["metrics"][0]["result"].get("dataset_drift", False)
            logger.info(f"[DRIFT] Report saved: {report_path} | drift={drift_flag}")

            return {
                "status": "report_generated",
                "drift_detected": drift_flag,
                "report_path": report_path,
                "timestamp": timestamp,
            }
        except Exception as e:
            logger.error(f"[DRIFT] Evidently report failed: {e}")
            return {"status": "error", "error": str(e), "report_path": None}

    def latest_report_path(self) -> str | None:
        """Return path to the most recent drift report, or None."""
        reports = sorted(Path(self.reports_dir).glob("*.html"), reverse=True)
        return str(reports[0]) if reports else None

    # ── Helpers ───────────────────────────────────────────────────────────

    def _load_json(self, path: str) -> list:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_json(self, path: str, data):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
