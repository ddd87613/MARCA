from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


FAULT_TYPES: Sequence[str] = (
    "CPU Stress",
    "Memory Stress",
    "Network Delay",
    "Network Loss",
    "Network Partition",
    "Process Kill",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    services: str
    fault_types: Sequence[str]


@dataclass(frozen=True)
class MetricRow:
    method: str
    dataset: str
    acc_at_1: float
    acc_at_3: float
    f1: float


DATASETS: Sequence[DatasetSpec] = (
    DatasetSpec(
        name="Self-built Payment System",
        description="Chaos Mesh injected e-commerce payment platform with 61 validated cases.",
        services="payment microservice system",
        fault_types=FAULT_TYPES,
    ),
    DatasetSpec(
        name="TrainTicket",
        description="Benchmark train ticket booking microservice system.",
        services="41 services",
        fault_types=("public benchmark faults",),
    ),
    DatasetSpec(
        name="MicroSS",
        description="Large-scale e-commerce observability benchmark.",
        services="large-scale microservice system",
        fault_types=("public benchmark faults",),
    ),
)


TABLE_1_RESULTS: Sequence[MetricRow] = (
    MetricRow("Eadro", "Self-built Payment", 56.34, 68.21, 0.523),
    MetricRow("DejaVu", "Self-built Payment", 53.28, 65.47, 0.498),
    MetricRow("HolisticRCA", "Self-built Payment", 62.15, 73.84, 0.587),
    MetricRow("RCACopilot", "Self-built Payment", 64.82, 76.35, 0.612),
    MetricRow("Fine-tuned GPT-4o", "Self-built Payment", 66.45, 77.92, 0.635),
    MetricRow("DeepSeek-V2 Single", "Self-built Payment", 68.73, 79.56, 0.658),
    MetricRow("MARCA Llama-2", "Self-built Payment", 76.87, 85.42, 0.790),
    MetricRow("MARCA DeepSeek-V2", "Self-built Payment", 79.25, 88.10, 0.815),
    MetricRow("Eadro", "TrainTicket", 52.18, 64.35, 0.485),
    MetricRow("DejaVu", "TrainTicket", 49.67, 61.23, 0.452),
    MetricRow("HolisticRCA", "TrainTicket", 58.34, 69.78, 0.541),
    MetricRow("RCACopilot", "TrainTicket", 55.26, 67.41, 0.518),
    MetricRow("Fine-tuned GPT-4o", "TrainTicket", 57.89, 69.15, 0.536),
    MetricRow("DeepSeek-V2 Single", "TrainTicket", 60.42, 71.83, 0.562),
    MetricRow("MARCA Llama-2", "TrainTicket", 72.34, 79.81, 0.723),
    MetricRow("MARCA DeepSeek-V2", "TrainTicket", 74.56, 81.23, 0.769),
    MetricRow("Eadro", "MicroSS", 50.23, 62.18, 0.468),
    MetricRow("DejaVu", "MicroSS", 47.85, 59.64, 0.441),
    MetricRow("HolisticRCA", "MicroSS", 56.72, 68.35, 0.524),
    MetricRow("RCACopilot", "MicroSS", 53.41, 65.28, 0.497),
    MetricRow("Fine-tuned GPT-4o", "MicroSS", 55.68, 67.42, 0.513),
    MetricRow("DeepSeek-V2 Single", "MicroSS", 58.94, 70.16, 0.541),
    MetricRow("MARCA Llama-2", "MicroSS", 69.18, 76.42, 0.710),
    MetricRow("MARCA DeepSeek-V2", "MicroSS", 71.30, 78.95, 0.741),
)


SENSITIVITY_TABLE: Dict[str, Dict[str, float]] = {
    "Uniform": {"acc_at_1": 74.2, "f1": 0.762},
    "Retcode-Priority": {"acc_at_1": 76.9, "f1": 0.790},
    "Log-Priority": {"acc_at_1": 71.3, "f1": 0.731},
    "Metric-Priority": {"acc_at_1": 73.8, "f1": 0.755},
}


ABLATION_FINDINGS: Dict[str, str] = {
    "w/o Multi-modal": "Remove logs and retcodes; F1 drops strongly because error codes carry semantic cues.",
    "w/o Voting": "Replace weighted consensus with majority voting; robustness drops under conflicting evidence.",
    "w/o Topology": "Disable graph constraints; Acc@1 drops from 76.87% to 52.18% in the paper.",
}


TOKEN_EFFICIENCY = {
    "marca_tokens": 1842,
    "holistic_rca_tokens": 4521,
    "reported_reduction": 0.60,
}


FAILURE_BOUNDARIES: Sequence[str] = (
    "Incomplete or stale topology can stop traversal at an intermediate gateway.",
    "Serverless or ephemeral invocations may be missing from service mesh topology.",
    "Concurrent faults can produce low-margin rankings and incorrect Top-1 output.",
    "Generic HTTP 500 without stack traces weakens semantic code similarity.",
    "Degraded log pipelines should reduce log weight during online adaptation.",
)


RELATED_WORK_POSITIONING: Dict[str, str] = {
    "Traditional RCA": "Rules, Bayesian networks, LightGBM, Random Forests; interpretable but brittle under dynamic topology.",
    "Multi-modal RCA": "CNN-LSTM or Transformer-GNN fusion; stronger accuracy but high alignment overhead and lower transparency.",
    "LLM RCA": "Tool-augmented agents such as RCAgent; powerful but often single-pass and weak in hypothesis arbitration.",
    "MARCA": "Closed-loop Controller-Executor-Voter workflow with topology constraints and modality-aware weighted voting.",
}


def format_table_1() -> str:
    rows = [
        "Method | Dataset | Acc@1 | Acc@3 | F1",
        "--- | --- | ---: | ---: | ---:",
    ]
    for row in TABLE_1_RESULTS:
        rows.append(
            f"{row.method} | {row.dataset} | {row.acc_at_1:.2f}% | "
            f"{row.acc_at_3:.2f}% | {row.f1:.3f}"
        )
    return "\n".join(rows)


def token_reduction() -> float:
    marca = TOKEN_EFFICIENCY["marca_tokens"]
    baseline = TOKEN_EFFICIENCY["holistic_rca_tokens"]
    return 1.0 - marca / baseline
