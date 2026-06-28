from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from evaluation.fault_classifier import classify_result
from marca_reproduction.agents import Marca
from marca_reproduction.llm_client import BaseLLMClient
from marca_reproduction.schemas import DiagnosisResult, MarcaConfig
from marca_reproduction.tools import InMemoryObservability


@dataclass(frozen=True)
class FaultCase:
    name: str
    observability: InMemoryObservability
    entry_service: str
    root_cause: str
    fault_type: str


@dataclass(frozen=True)
class EvaluationResult:
    acc_at_1: float
    acc_at_3: float
    macro_f1: float
    predictions: Mapping[str, DiagnosisResult]


def evaluate_cases(
    cases: Sequence[FaultCase],
    config: MarcaConfig,
    llm_client: Optional[BaseLLMClient] = None,
) -> EvaluationResult:
    predictions: Dict[str, DiagnosisResult] = {}
    predicted_roots: List[str] = []
    true_roots: List[str] = []
    predicted_types: List[str] = []
    true_types: List[str] = []

    for case in cases:
        result = Marca(case.observability, config, llm_client=llm_client).diagnose(
            case.entry_service
        )
        predictions[case.name] = result
        predicted_roots.append(result.root_cause or "")
        true_roots.append(case.root_cause)
        classification = classify_result(result)
        predicted_types.append(classification.label if classification else "Unknown")
        true_types.append(case.fault_type)

    return EvaluationResult(
        acc_at_1=_acc_at_k(predictions, cases, 1),
        acc_at_3=_acc_at_k(predictions, cases, 3),
        macro_f1=macro_f1(true_types, predicted_types),
        predictions=predictions,
    )


def run_ablation_suite(cases: Sequence[FaultCase]) -> Mapping[str, EvaluationResult]:
    base = MarcaConfig(confidence_threshold=0.97, corr_threshold=0.40)
    configs = {
        "full": base,
        "w/o Multi-modal": MarcaConfig(
            confidence_threshold=0.97,
            corr_threshold=0.40,
            enabled_modalities=("metrics", "traces"),
        ),
        "w/o Voting": MarcaConfig(
            confidence_threshold=0.97,
            corr_threshold=0.40,
            voting_strategy="majority",
        ),
        "w/o Topology": MarcaConfig(
            confidence_threshold=0.97,
            corr_threshold=0.40,
            use_topology_guidance=False,
        ),
    }
    return {name: evaluate_cases(cases, config) for name, config in configs.items()}


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def _acc_at_k(
    predictions: Mapping[str, DiagnosisResult],
    cases: Sequence[FaultCase],
    k: int,
) -> float:
    if not cases:
        return 0.0
    correct = 0
    for case in cases:
        ranking = predictions[case.name].ranking[:k]
        correct += int(any(candidate.service == case.root_cause for candidate in ranking))
    return correct / len(cases)


def estimate_token_reduction(config: MarcaConfig) -> float:
    return 1.0 - config.target_prompt_tokens / config.holistic_rca_prompt_tokens
