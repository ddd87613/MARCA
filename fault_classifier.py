from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from schemas import Candidate, DiagnosisResult


@dataclass(frozen=True)
class FaultClassification:
    label: str
    confidence: float
    evidence: List[str]


def classify_result(result: DiagnosisResult) -> Optional[FaultClassification]:
    if not result.ranking:
        return None
    return classify_candidate(result.ranking[0])


def classify_candidate(candidate: Candidate) -> FaultClassification:
    semantic_parts = [candidate.service]
    cpu_values: List[float] = []
    memory_values: List[float] = []
    for observation in candidate.evidence:
        semantic_parts.append(observation.summary)
        if observation.modality == "logs":
            semantic_parts.extend(str(line) for line in observation.payload.get("logs", []))
        if observation.modality == "retcodes":
            semantic_parts.extend(str(code) for code in observation.payload.get("retcodes", []))
        if observation.modality == "metrics":
            series = observation.payload.get("series", {})
            if isinstance(series, dict):
                cpu_values.extend(float(value) for value in series.get("cpu", []))
                memory_values.extend(float(value) for value in series.get("memory", []))

    text = " ".join(semantic_parts).lower()

    if any(term in text for term in ("process kill", "killed", "panic", "fatal")):
        return FaultClassification("Process Kill", 0.86, ["fatal/process termination semantics"])
    if memory_values and sum(memory_values) / len(memory_values) >= 80:
        return FaultClassification("Memory Stress", 0.86, ["high memory metric values"])
    if any(term in text for term in ("oom", "out of memory")):
        return FaultClassification("Memory Stress", 0.82, ["memory pressure semantics"])
    if cpu_values and sum(cpu_values) / len(cpu_values) >= 80:
        return FaultClassification("CPU Stress", 0.86, ["high CPU metric values"])
    if "cpu" in text and any(term in text for term in ("stress", "saturation", "high")):
        return FaultClassification("CPU Stress", 0.82, ["CPU saturation semantics"])
    if any(term in text for term in ("partition", "unreachable", "no route")):
        return FaultClassification("Network Partition", 0.84, ["partition/unreachable semantics"])
    if any(term in text for term in ("packet loss", "network loss", "loss")):
        return FaultClassification("Network Loss", 0.80, ["packet loss semantics"])
    if any(term in text for term in ("timeout", "503", "unavailable", "refused")):
        return FaultClassification("Network Delay", 0.76, ["timeout/unavailable/refused semantics"])
    if candidate.modality_scores.get("metrics", 0.0) >= 0.85:
        return FaultClassification("Network Delay", 0.55, ["strong metric anomaly without specific semantics"])
    return FaultClassification("Unknown", 0.20, ["insufficient fault-type evidence"])
