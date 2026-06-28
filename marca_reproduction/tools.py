from __future__ import annotations

from math import sqrt
from typing import Dict, Iterable, List, Mapping, Sequence

from marca_reproduction.schemas import (
    FunctionCall,
    ServiceSnapshot,
    ToolObservation,
    clamp01,
)


class InMemoryObservability:
    """Local implementation of the paper's function-call tool layer."""

    def __init__(
        self,
        topology: Mapping[str, Iterable[str]],
        snapshots: Mapping[str, ServiceSnapshot],
    ) -> None:
        self.topology = {node: list(neighbors) for node, neighbors in topology.items()}
        self.snapshots = dict(snapshots)

    def neighbors(self, service: str) -> List[str]:
        return self.topology.get(service, [])

    def services(self) -> List[str]:
        names = set(self.snapshots)
        names.update(self.topology)
        for neighbors in self.topology.values():
            names.update(neighbors)
        return sorted(names)

    def snapshot(self, service: str) -> ServiceSnapshot:
        return self.snapshots.get(service, ServiceSnapshot())

    def query_metrics(
        self,
        service: str,
        metric_names: Sequence[str],
        window: str = "incident",
    ) -> ToolObservation:
        snapshot = self.snapshot(service)
        selected = {
            name: list(snapshot.current_metrics.get(name, []))
            for name in metric_names
            if name in snapshot.current_metrics
        }
        score = metric_anomaly_score(snapshot)
        return ToolObservation(
            call=FunctionCall(
                "query_metrics",
                {"service": service, "metric_names": list(metric_names), "window": window},
            ),
            service=service,
            modality="metrics",
            score=score,
            summary=f"metric anomaly score={score:.3f}",
            payload={"series": selected},
        )

    def filter_logs(
        self,
        service: str,
        severity: str = "warning+",
        keywords: Sequence[str] = (
            "timeout",
            "refused",
            "error",
            "503",
            "retry",
            "cpu",
            "memory",
            "oom",
            "loss",
            "partition",
            "unreachable",
            "killed",
            "fatal",
            "panic",
        ),
        window: str = "incident",
    ) -> ToolObservation:
        snapshot = self.snapshot(service)
        lowered_keywords = [keyword.lower() for keyword in keywords]
        matched = [
            line
            for line in snapshot.logs
            if any(keyword in line.lower() for keyword in lowered_keywords)
        ]
        score = log_anomaly_score(snapshot.logs)
        return ToolObservation(
            call=FunctionCall(
                "filter_logs",
                {
                    "service": service,
                    "severity": severity,
                    "keywords": list(keywords),
                    "window": window,
                },
            ),
            service=service,
            modality="logs",
            score=score,
            summary=f"{len(matched)} relevant log lines, score={score:.3f}",
            payload={"logs": matched},
        )

    def analyze_traces(
        self,
        service: str,
        max_depth: int = 2,
        metrics: Sequence[str] = ("latency", "error_rate"),
    ) -> ToolObservation:
        snapshot = self.snapshot(service)
        score = trace_anomaly_score(snapshot.trace_error_rate, snapshot.trace_latency_ms)
        return ToolObservation(
            call=FunctionCall(
                "analyze_traces",
                {"service": service, "max_depth": max_depth, "metrics": list(metrics)},
            ),
            service=service,
            modality="traces",
            score=score,
            summary=(
                f"trace error_rate={snapshot.trace_error_rate:.3f}, "
                f"latency_ms={snapshot.trace_latency_ms:.1f}, score={score:.3f}"
            ),
            payload={
                "error_rate": snapshot.trace_error_rate,
                "latency_ms": snapshot.trace_latency_ms,
            },
        )

    def query_retcodes(self, service: str, window: str = "incident") -> ToolObservation:
        snapshot = self.snapshot(service)
        score = retcode_anomaly_score(snapshot.retcodes)
        return ToolObservation(
            call=FunctionCall("query_retcodes", {"service": service, "window": window}),
            service=service,
            modality="retcodes",
            score=score,
            summary=f"retcodes={list(snapshot.retcodes)}, score={score:.3f}",
            payload={"retcodes": list(snapshot.retcodes)},
        )


def metric_anomaly_score(snapshot: ServiceSnapshot) -> float:
    values = []
    for name, current in snapshot.current_metrics.items():
        baseline = snapshot.baseline_metrics.get(name, [])
        if not baseline or not current:
            continue
        mean = sum(baseline) / len(baseline)
        variance = sum((x - mean) ** 2 for x in baseline) / max(1, len(baseline) - 1)
        std = sqrt(variance) or 1e-9
        z = abs((sum(current) / len(current) - mean) / std)
        values.append(clamp01(z / 4.0))
    return max(values, default=0.0)


def log_anomaly_score(logs: Sequence[str]) -> float:
    if not logs:
        return 0.0
    critical = (
        "fatal",
        "panic",
        "connection refused",
        "timeout",
        "unavailable",
        "partition",
        "unreachable",
        "packet loss",
        "oom",
        "killed",
    )
    warning = ("error", "exception", "failed", "retry", "slow", "cpu", "memory", "stress")
    score = 0.0
    for line in logs:
        lowered = line.lower()
        if any(term in lowered for term in critical):
            score = max(score, 1.0)
        elif any(term in lowered for term in warning):
            score = max(score, 0.65)
    return score


def trace_anomaly_score(error_rate: float, latency_ms: float) -> float:
    error_component = clamp01(error_rate)
    latency_component = clamp01((latency_ms - 200.0) / 1800.0)
    return max(error_component, latency_component)


def retcode_anomaly_score(retcodes: Sequence[str]) -> float:
    if not retcodes:
        return 0.0
    max_score = 0.0
    for code in retcodes:
        text = str(code).upper()
        if text.startswith("5") or "503" in text or "TIMEOUT" in text:
            max_score = max(max_score, 1.0)
        elif text.startswith("4") or "FAILED" in text or "REFUSED" in text:
            max_score = max(max_score, 0.75)
    return max_score


def metric_correlation(a: ServiceSnapshot, b: ServiceSnapshot) -> float:
    scores = []
    for name, series_a in a.current_metrics.items():
        series_b = b.current_metrics.get(name)
        if not series_b:
            continue
        scores.append(abs(pearson(series_a, series_b)))
    return max(scores, default=0.0)


def code_similarity(a_codes: Sequence[str], b_codes: Sequence[str]) -> float:
    a = {normalize_code(code) for code in a_codes if code}
    b = {normalize_code(code) for code in b_codes if code}
    if not a or not b:
        return 0.0
    if a & b:
        return 1.0
    if {code[:1] for code in a} & {code[:1] for code in b}:
        return 0.55
    return 0.0


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    xs = list(a)[-n:]
    ys = list(b)[-n:]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = sqrt(sum((y - mean_y) ** 2 for y in ys))
    denom = den_x * den_y
    return num / denom if denom else 0.0


def normalize_code(code: str) -> str:
    text = str(code).strip().upper()
    if "503" in text:
        return "503"
    if "500" in text:
        return "500"
    if "TIMEOUT" in text:
        return "TIMEOUT"
    if "REFUSED" in text:
        return "REFUSED"
    return text


def evidence_scores_from_observations(
    observations: Sequence[ToolObservation],
) -> Dict[str, float]:
    return {observation.modality: observation.score for observation in observations}
