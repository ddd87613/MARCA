from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from llm_client import BaseLLMClient, NoOpLLMClient
from prompts import build_controller_prompt, build_executor_prompt, build_voter_prompt
from schemas import (
    Candidate,
    DiagnosisResult,
    EvidenceBundle,
    LLMCallRecord,
    MarcaConfig,
    ToolObservation,
    TraversalTask,
    clamp01,
)
from tools import (
    InMemoryObservability,
    code_similarity,
    evidence_scores_from_observations,
    metric_correlation,
)


class ExecutorAgent:
    """Executes the paper's function-call interface for one service."""

    def __init__(
        self,
        observability: InMemoryObservability,
        config: MarcaConfig,
        llm_client: BaseLLMClient,
    ) -> None:
        self.observability = observability
        self.config = config.normalized()
        self.llm_client = llm_client
        self.prompt_log: List[str] = []
        self.llm_call_log: List[LLMCallRecord] = []

    def collect_evidence(self, task: TraversalTask) -> EvidenceBundle:
        prompt = build_executor_prompt(task, self.config)
        self.prompt_log.append(prompt)
        self._maybe_call_llm(prompt)
        observations: List[ToolObservation] = []
        enabled = set(self.config.enabled_modalities)
        if "metrics" in enabled:
            observations.append(
                self.observability.query_metrics(task.service, ["latency", "cpu", "memory"])
            )
        if "logs" in enabled:
            observations.append(self.observability.filter_logs(task.service))
        if "traces" in enabled:
            observations.append(self.observability.analyze_traces(task.service))
        if "retcodes" in enabled:
            observations.append(self.observability.query_retcodes(task.service))
        return EvidenceBundle(
            service=task.service,
            observations=observations,
            modality_scores=evidence_scores_from_observations(observations),
        )

    def _maybe_call_llm(self, prompt: str) -> None:
        if self.config.llm_enabled:
            self.llm_call_log.append(
                self.llm_client.complete_json("executor", prompt, self.config)
            )


class VotingAgent:
    """Weighted consensus over modality evidence."""

    def __init__(self, config: MarcaConfig, llm_client: BaseLLMClient) -> None:
        self.config = config.normalized()
        self.llm_client = llm_client
        self.prompt_log: List[str] = []
        self.llm_call_log: List[LLMCallRecord] = []

    def score(
        self,
        service: str,
        depth: int,
        parent: Optional[str],
        evidence: EvidenceBundle,
    ) -> Candidate:
        if self.config.voting_strategy == "majority":
            values = list(evidence.modality_scores.values())
            weighted = (
                sum(1.0 for value in values if value >= 0.5) / len(values)
                if values
                else 0.0
            )
        else:
            weighted = sum(
                self.config.modality_weights.get(modality, 0.0) * value
                for modality, value in evidence.modality_scores.items()
            )
        return Candidate(
            service=service,
            score=clamp01(weighted),
            modality_scores=evidence.modality_scores,
            depth=depth,
            parent=parent,
            evidence=evidence.observations,
        )

    def rank(self, candidates: Mapping[str, Candidate]) -> List[Candidate]:
        ranking = sorted(candidates.values(), key=lambda c: c.score, reverse=True)
        prompt = build_voter_prompt(ranking, self.config)
        self.prompt_log.append(prompt)
        if self.config.llm_enabled:
            self.llm_call_log.append(
                self.llm_client.complete_json("voter", prompt, self.config)
            )
        return ranking


class ControllerAgent:
    """Controller loop: topology-constrained traversal plus termination."""

    def __init__(
        self,
        observability: InMemoryObservability,
        executor: ExecutorAgent,
        voter: VotingAgent,
        config: MarcaConfig,
        llm_client: BaseLLMClient,
    ) -> None:
        self.observability = observability
        self.executor = executor
        self.voter = voter
        self.config = config.normalized()
        self.llm_client = llm_client
        self.prompt_log: List[str] = []
        self.llm_call_log: List[LLMCallRecord] = []

    def diagnose(self, entry_service: str) -> DiagnosisResult:
        queue: Deque[Tuple[str, int, Optional[str]]] = deque([(entry_service, 0, None)])
        seen = {entry_service}
        candidates: Dict[str, Candidate] = {}
        traversal_path: List[str] = []
        visited_edges: List[Tuple[str, str, float]] = []
        function_calls: List[ToolObservation] = []
        previous_top_score = 0.0

        while queue:
            service, depth, parent = queue.popleft()
            task = TraversalTask(
                service=service,
                parent=parent,
                depth=depth,
                instruction=f"Analyze {service} and decide downstream traversal.",
            )
            prompt = build_controller_prompt(task, candidates.values(), self.config)
            self.prompt_log.append(prompt)
            if self.config.llm_enabled:
                self.llm_call_log.append(
                    self.llm_client.complete_json("controller", prompt, self.config)
                )
            traversal_path.append(service)

            evidence = self.executor.collect_evidence(task)
            function_calls.extend(evidence.observations)
            candidates[service] = self.voter.score(service, depth, parent, evidence)
            ranking = self.voter.rank(candidates)
            top_score = ranking[0].score if ranking else 0.0
            marginal_gain = top_score - previous_top_score

            if ranking and top_score >= self.config.confidence_threshold:
                return DiagnosisResult(
                    root_cause=ranking[0].service,
                    ranking=ranking,
                    traversal_path=traversal_path,
                    visited_edges=visited_edges,
                    function_calls=function_calls,
                    llm_calls=self._all_llm_calls(),
                )

            if depth >= self.config.max_depth:
                continue

            for neighbor in self._next_services(service):
                if neighbor in seen:
                    continue
                corr = causal_correlation(
                    self.observability,
                    service,
                    neighbor,
                    self.config.lambda_metric,
                    self.config.lambda_code,
                )
                visited_edges.append((service, neighbor, corr))
                if not self.config.use_topology_guidance or corr >= self.config.corr_threshold:
                    seen.add(neighbor)
                    queue.append((neighbor, depth + 1, service))

            if (
                depth > 0
                and not queue
                and ranking
                and abs(marginal_gain) < self.config.marginal_gain_epsilon
            ):
                return DiagnosisResult(
                    root_cause=ranking[0].service,
                    ranking=ranking,
                    traversal_path=traversal_path,
                    visited_edges=visited_edges,
                    function_calls=function_calls,
                    llm_calls=self._all_llm_calls(),
                )
            previous_top_score = max(previous_top_score, top_score)

        ranking = self.voter.rank(candidates)
        return DiagnosisResult(
            root_cause=ranking[0].service if ranking else None,
            ranking=ranking,
            traversal_path=traversal_path,
            visited_edges=visited_edges,
            function_calls=function_calls,
            llm_calls=self._all_llm_calls(),
        )

    def _all_llm_calls(self) -> List[LLMCallRecord]:
        return [
            *self.llm_call_log,
            *self.executor.llm_call_log,
            *self.voter.llm_call_log,
        ]

    def _next_services(self, service: str) -> List[str]:
        if self.config.use_topology_guidance:
            return self.observability.neighbors(service)
        return [name for name in self.observability.services() if name != service]


def causal_correlation(
    observability: InMemoryObservability,
    src: str,
    dst: str,
    lambda_metric: float,
    lambda_code: float,
) -> float:
    src_snapshot = observability.snapshot(src)
    dst_snapshot = observability.snapshot(dst)
    metric_corr = metric_correlation(src_snapshot, dst_snapshot)
    code_sim = code_similarity(src_snapshot.retcodes, dst_snapshot.retcodes)
    return clamp01(lambda_metric * metric_corr + lambda_code * code_sim)


class Marca:
    """Convenience facade wiring Controller, Executor, and Voter."""

    def __init__(
        self,
        observability: InMemoryObservability,
        config: MarcaConfig,
        llm_client: Optional[BaseLLMClient] = None,
    ) -> None:
        self.observability = observability
        self.config = config.normalized()
        self.llm_client = llm_client or NoOpLLMClient()
        self.executor = ExecutorAgent(observability, self.config, self.llm_client)
        self.voter = VotingAgent(self.config, self.llm_client)
        self.controller = ControllerAgent(
            observability=observability,
            executor=self.executor,
            voter=self.voter,
            config=self.config,
            llm_client=self.llm_client,
        )

    def diagnose(self, entry_service: str) -> DiagnosisResult:
        return self.controller.diagnose(entry_service)

    def prompt_history(self) -> Dict[str, Sequence[str]]:
        return {
            "controller": self.controller.prompt_log,
            "executor": self.executor.prompt_log,
            "voter": self.voter.prompt_log,
        }

    def llm_history(self) -> Dict[str, Sequence[LLMCallRecord]]:
        return {
            "controller": self.controller.llm_call_log,
            "executor": self.executor.llm_call_log,
            "voter": self.voter.llm_call_log,
        }
