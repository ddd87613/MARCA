from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


ModalityScores = Dict[str, float]


@dataclass(frozen=True)
class ServiceSnapshot:
    """Observability summary for one service."""

    baseline_metrics: Mapping[str, Sequence[float]] = field(default_factory=dict)
    current_metrics: Mapping[str, Sequence[float]] = field(default_factory=dict)
    logs: Sequence[str] = field(default_factory=list)
    retcodes: Sequence[str] = field(default_factory=list)
    trace_error_rate: float = 0.0
    trace_latency_ms: float = 0.0


@dataclass(frozen=True)
class MarcaConfig:
    model_name: str = "local-llm"
    llm_enabled: bool = False
    llm_api_base: str = "http://localhost:8000/v1"
    llm_api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.1
    max_tokens: int = 2048
    lambda_metric: float = 0.6
    lambda_code: float = 0.4
    corr_threshold: float = 0.45
    confidence_threshold: float = 0.75
    marginal_gain_epsilon: float = 0.03
    max_depth: int = 4
    ema_alpha: float = 0.8
    stable_weight_epsilon: float = 0.01
    offline_iterations: int = 30
    few_shot_k: int = 2
    enabled_modalities: Sequence[str] = ("metrics", "logs", "traces", "retcodes")
    use_topology_guidance: bool = True
    voting_strategy: str = "weighted"
    score_margin_warning: float = 0.05
    target_prompt_tokens: int = 1842
    holistic_rca_prompt_tokens: int = 4521
    modality_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "metrics": 0.30,
            "logs": 0.20,
            "traces": 0.10,
            "retcodes": 0.40,
        }
    )

    def normalized(self) -> "MarcaConfig":
        lm = max(0.0, self.lambda_metric)
        lc = max(0.0, self.lambda_code)
        denom = lm + lc or 1.0
        weights = normalize_weights(self.modality_weights)
        return MarcaConfig(
            model_name=self.model_name,
            llm_enabled=self.llm_enabled,
            llm_api_base=self.llm_api_base,
            llm_api_key_env=self.llm_api_key_env,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            lambda_metric=lm / denom,
            lambda_code=lc / denom,
            corr_threshold=self.corr_threshold,
            confidence_threshold=self.confidence_threshold,
            marginal_gain_epsilon=self.marginal_gain_epsilon,
            max_depth=self.max_depth,
            ema_alpha=self.ema_alpha,
            stable_weight_epsilon=self.stable_weight_epsilon,
            offline_iterations=self.offline_iterations,
            few_shot_k=self.few_shot_k,
            enabled_modalities=tuple(self.enabled_modalities),
            use_topology_guidance=self.use_topology_guidance,
            voting_strategy=self.voting_strategy,
            score_margin_warning=self.score_margin_warning,
            target_prompt_tokens=self.target_prompt_tokens,
            holistic_rca_prompt_tokens=self.holistic_rca_prompt_tokens,
            modality_weights=weights,
        )


@dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True)
class ToolObservation:
    call: FunctionCall
    service: str
    modality: str
    score: float
    summary: str
    payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceBundle:
    service: str
    observations: List[ToolObservation]
    modality_scores: ModalityScores


@dataclass(frozen=True)
class TraversalTask:
    service: str
    parent: Optional[str]
    depth: int
    instruction: str


@dataclass(frozen=True)
class Candidate:
    service: str
    score: float
    modality_scores: ModalityScores
    depth: int
    parent: Optional[str]
    evidence: List[ToolObservation] = field(default_factory=list)


@dataclass(frozen=True)
class LLMCallRecord:
    role: str
    prompt: str
    response_text: str
    parsed_json: Optional[Mapping[str, object]] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class DiagnosisResult:
    root_cause: Optional[str]
    ranking: List[Candidate]
    traversal_path: List[str]
    visited_edges: List[Tuple[str, str, float]]
    function_calls: List[ToolObservation]
    llm_calls: List[LLMCallRecord] = field(default_factory=list)


def normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    cleaned = {key: max(0.0, value) for key, value in weights.items()}
    total = sum(cleaned.values()) or 1.0
    return {key: value / total for key, value in cleaned.items()}


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
