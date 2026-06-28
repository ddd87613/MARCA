from marca_core import (
    InMemoryObservability,
    Marca,
    MarcaConfig,
    OpenAICompatibleLLMClient,
    ServiceSnapshot,
)


def main() -> None:
    topology = {"frontend": ["order-service"], "order-service": ["payment-gateway"]}
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

    # LLM 接口位置:
    # 1. 启动一个 OpenAI-compatible 服务，例如本地 vLLM:
    #    vllm serve <your-model> --host 0.0.0.0 --port 8000
    # 2. 将 llm_enabled=True。
    # 3. 把 OpenAICompatibleLLMClient 传给 Marca。
    #
    # 真实 LLM 的输出会记录在 DiagnosisResult.llm_calls 中。
    config = MarcaConfig(
        llm_enabled=True,
        model_name="your-local-vllm-model",
        llm_api_base="http://localhost:8000/v1",
        confidence_threshold=0.97,
        corr_threshold=0.40,
    )
    llm_client = OpenAICompatibleLLMClient()

    marca = Marca(
        InMemoryObservability(topology, snapshots),
        config,
        llm_client=llm_client,
    )
    result = marca.diagnose("frontend")

    print("Root cause:", result.root_cause)
    print("LLM calls:", len(result.llm_calls))
    for record in result.llm_calls:
        print(record.role, "error=", record.error)


if __name__ == "__main__":
    main()
