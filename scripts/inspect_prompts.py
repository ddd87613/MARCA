from _bootstrap import PROJECT_ROOT  # noqa: F401

from marca_reproduction.core import InMemoryObservability, Marca, MarcaConfig, ServiceSnapshot


def build_demo_marca() -> Marca:
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
    }
    return Marca(
        InMemoryObservability(topology, snapshots),
        MarcaConfig(confidence_threshold=0.97, corr_threshold=0.40),
    )


def main() -> None:
    marca = build_demo_marca()
    marca.diagnose("frontend")
    history = marca.prompt_history()
    for role, prompts in history.items():
        print("=" * 80)
        print(role.upper())
        print("=" * 80)
        print(prompts[0] if prompts else "No prompt generated.")


if __name__ == "__main__":
    main()
