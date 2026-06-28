from __future__ import annotations

from typing import Dict, Sequence, Tuple

from agents import Marca
from evaluation import EvaluationResult, FaultCase, evaluate_cases, run_ablation_suite
from fault_classifier import FaultClassification, classify_candidate, classify_result
from llm_client import BaseLLMClient, NoOpLLMClient, OpenAICompatibleLLMClient
from paper_details import (
    ABLATION_FINDINGS,
    DATASETS,
    FAILURE_BOUNDARIES,
    FAULT_TYPES,
    SENSITIVITY_TABLE,
    TABLE_1_RESULTS,
    TOKEN_EFFICIENCY,
)
from reporting import build_diagnosis_report
from schemas import (
    Candidate,
    DiagnosisResult,
    FunctionCall,
    LLMCallRecord,
    MarcaConfig,
    ServiceSnapshot,
    ToolObservation,
    normalize_weights,
)
from tools import InMemoryObservability


def update_modality_weights_ema(
    old_weights: Dict[str, float],
    contribution_indicator: Dict[str, float],
    alpha: float = 0.8,
) -> Dict[str, float]:
    updated = {
        modality: alpha * old_weights.get(modality, 0.0)
        + (1.0 - alpha) * contribution_indicator.get(modality, 0.0)
        for modality in set(old_weights) | set(contribution_indicator)
    }
    return normalize_weights(updated)


def grid_search_parameters(
    cases: Sequence[Tuple[InMemoryObservability, str, str]],
    lambda_values: Sequence[float] = (0.3, 0.4, 0.5, 0.6, 0.7),
    retcode_weights: Sequence[float] = (0.25, 0.35, 0.4, 0.5),
) -> MarcaConfig:
    """Tiny offline optimizer mirroring the paper's parameter-learning phase."""

    best_config = MarcaConfig()
    best_accuracy = -1.0
    for lambda_metric in lambda_values:
        lambda_code = 1.0 - lambda_metric
        for ret_w in retcode_weights:
            rest = max(0.0, 1.0 - ret_w)
            config = MarcaConfig(
                lambda_metric=lambda_metric,
                lambda_code=lambda_code,
                modality_weights={
                    "metrics": rest * 0.5,
                    "logs": rest * 0.3,
                    "traces": rest * 0.2,
                    "retcodes": ret_w,
                },
            )
            correct = 0
            for observability, entry, expected_root in cases:
                result = Marca(observability, config).diagnose(entry)
                correct += int(result.root_cause == expected_root)
            accuracy = correct / max(1, len(cases))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = config.normalized()
    return best_config


__all__ = [
    "Candidate",
    "BaseLLMClient",
    "DiagnosisResult",
    "EvaluationResult",
    "FaultCase",
    "FaultClassification",
    "FunctionCall",
    "InMemoryObservability",
    "LLMCallRecord",
    "Marca",
    "MarcaConfig",
    "NoOpLLMClient",
    "OpenAICompatibleLLMClient",
    "ServiceSnapshot",
    "ToolObservation",
    "ABLATION_FINDINGS",
    "DATASETS",
    "FAILURE_BOUNDARIES",
    "FAULT_TYPES",
    "SENSITIVITY_TABLE",
    "TABLE_1_RESULTS",
    "TOKEN_EFFICIENCY",
    "build_diagnosis_report",
    "classify_candidate",
    "classify_result",
    "evaluate_cases",
    "grid_search_parameters",
    "run_ablation_suite",
    "update_modality_weights_ema",
]
