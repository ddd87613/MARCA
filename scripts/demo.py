from _bootstrap import PROJECT_ROOT  # noqa: F401

from marca_reproduction.core import (
    InMemoryObservability,
    Marca,
    MarcaConfig,
    ServiceSnapshot,
    classify_result,
)


def main() -> None:
    topology = {
        "frontend": ["order-service"],
        "order-service": ["payment-gateway", "database"],
        "payment-gateway": [],
        "database": [],
    }

    baseline_latency = [95, 100, 102, 98, 101, 99]
    snapshots = {
        "frontend": ServiceSnapshot(
            baseline_metrics={"latency": baseline_latency},
            current_metrics={"latency": [900, 980, 1050, 1100]},
            logs=["payment timeout returned to user"],
            retcodes=["TIMEOUT"],
            trace_error_rate=0.45,
            trace_latency_ms=1000,
        ),
        "order-service": ServiceSnapshot(
            baseline_metrics={"latency": baseline_latency},
            current_metrics={"latency": [820, 930, 990, 1070]},
            logs=["retrying downstream payment request"],
            retcodes=["TIMEOUT"],
            trace_error_rate=0.35,
            trace_latency_ms=900,
        ),
        "payment-gateway": ServiceSnapshot(
            baseline_metrics={"latency": baseline_latency},
            current_metrics={"latency": [760, 900, 970, 1120]},
            logs=["upstream unavailable: HTTP 503"],
            retcodes=["503"],
            trace_error_rate=0.75,
            trace_latency_ms=1200,
        ),
        "database": ServiceSnapshot(
            baseline_metrics={"latency": baseline_latency},
            current_metrics={"latency": [250, 300, 320, 330]},
            logs=["slow query observed after retry storm"],
            retcodes=["200"],
            trace_error_rate=0.08,
            trace_latency_ms=320,
        ),
    }

    observability = InMemoryObservability(topology, snapshots)
    config = MarcaConfig(confidence_threshold=0.97, corr_threshold=0.40)
    result = Marca(observability, config).diagnose("frontend")

    print("Root cause:", result.root_cause)
    print("LLM enabled:", config.llm_enabled)
    classification = classify_result(result)
    if classification:
        print(
            "Fault type:",
            f"{classification.label} (confidence={classification.confidence:.2f})",
        )
    print("Traversal path:", " -> ".join(result.traversal_path))
    print("\nFunction calls:")
    for observation in result.function_calls:
        args = ", ".join(f"{key}={value}" for key, value in observation.call.arguments.items())
        print(
            f"  {observation.call.name}({args}) -> "
            f"{observation.modality} score={observation.score:.3f}"
        )
    print("\nVisited edges:")
    for src, dst, corr in result.visited_edges:
        print(f"  {src:16s} -> {dst:16s} xi={corr:.3f}")

    print("\nRanking:")
    for rank, candidate in enumerate(result.ranking, 1):
        parts = ", ".join(
            f"{name}={score:.2f}"
            for name, score in sorted(candidate.modality_scores.items())
        )
        print(f"  {rank}. {candidate.service:16s} score={candidate.score:.3f} [{parts}]")

    print("\nPrompt templates are defined in prompts.py.")
    print("Run scripts/inspect_prompts.py to print the full Controller/Executor/Voter prompts.")
    print("See scripts/llm_wiring_example.py for the explicit LLM integration point.")
    print("Run scripts/paper_suite.py to inspect paper tables, sensitivity, and ablation scaffolding.")


if __name__ == "__main__":
    main()
