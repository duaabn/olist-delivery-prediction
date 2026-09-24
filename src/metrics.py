import threading


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.requests = 0
        self.errors = 0
        self.total_latency_seconds = 0.0
        self.predictions = {"late": 0, "on_time": 0}
        self.status_codes = {}

    def record(self, latency_seconds: float, prediction: int | None = None, error: bool = False):
        with self._lock:
            self.requests += 1
            self.total_latency_seconds += latency_seconds
            self.errors += int(error)
            if prediction is not None:
                self.predictions["late" if prediction == 1 else "on_time"] += 1

    def snapshot(self) -> dict:
        with self._lock:
            average = self.total_latency_seconds / self.requests if self.requests else 0.0
            return {
                "requests": self.requests,
                "errors": self.errors,
                "error_rate": self.errors / self.requests if self.requests else 0.0,
                "average_latency_seconds": average,
                "prediction_counts": dict(self.predictions),
                "status_codes": dict(self.status_codes),
            }

    def record_http(self, status_code: int):
        with self._lock:
            key = str(status_code)
            self.status_codes[key] = self.status_codes.get(key, 0) + 1

    def prometheus(self) -> str:
        snapshot = self.snapshot()
        lines = [
            "# HELP olist_http_requests_total Total HTTP requests.",
            "# TYPE olist_http_requests_total counter",
            f"olist_http_requests_total {snapshot['requests']}",
            "# HELP olist_http_errors_total Total prediction errors.",
            "# TYPE olist_http_errors_total counter",
            f"olist_http_errors_total {snapshot['errors']}",
            "# HELP olist_prediction_latency_seconds Average prediction latency.",
            "# TYPE olist_prediction_latency_seconds gauge",
            f"olist_prediction_latency_seconds {snapshot['average_latency_seconds']}",
        ]
        for status_code, count in snapshot["status_codes"].items():
            lines.append(f'olist_http_responses_total{{status_code="{status_code}"}} {count}')
        for label, count in snapshot["prediction_counts"].items():
            lines.append(f'olist_predictions_total{{result="{label}"}} {count}')
        return "\n".join(lines) + "\n"


metrics = Metrics()
