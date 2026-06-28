from __future__ import annotations

from fault_classifier import classify_result
from paper_details import FAILURE_BOUNDARIES, TOKEN_EFFICIENCY, token_reduction
from schemas import DiagnosisResult, MarcaConfig


def build_diagnosis_report(result: DiagnosisResult, config: MarcaConfig) -> str:
    classification = classify_result(result)
    lines = [
        "# MARCA Diagnosis Report",
        "",
        f"Root cause: {result.root_cause}",
        f"Traversal path: {' -> '.join(result.traversal_path)}",
        f"Function calls: {len(result.function_calls)}",
        f"LLM calls: {len(result.llm_calls)}",
        "",
        "## Ranking",
    ]
    for index, candidate in enumerate(result.ranking, 1):
        lines.append(
            f"{index}. {candidate.service}: score={candidate.score:.3f}, "
            f"modalities={candidate.modality_scores}"
        )
    lines.extend(
        [
            "",
            "## Fault Classification",
            (
                f"{classification.label} "
                f"(confidence={classification.confidence:.2f})"
                if classification
                else "Unknown"
            ),
            "",
            "## Token Efficiency",
            (
                f"Paper target: MARCA {TOKEN_EFFICIENCY['marca_tokens']} tokens vs "
                f"HolisticRCA {TOKEN_EFFICIENCY['holistic_rca_tokens']} tokens; "
                f"reduction={token_reduction():.1%}."
            ),
            f"Current config target tokens: {config.target_prompt_tokens}",
            "",
            "## Known Failure Boundaries",
        ]
    )
    lines.extend(f"- {item}" for item in FAILURE_BOUNDARIES)
    return "\n".join(lines)
