# MARCA Reproduction

This folder contains a compact reproduction of the main ideas from the paper
`MARCA: Multi-Agent Root Cause Analysis with Multi-Modal Data`.

The implementation focuses on the core workflow:

1. Controller traverses the service topology.
2. Executor collects evidence through function calls.
3. Voter ranks root-cause candidates with weighted multi-modal evidence.
4. Optional LLM calls can be plugged in through `llm_client.py`.

## Layout

- `marca_reproduction/`: core algorithm, agents, prompts, tools, and LLM hook.
- `evaluation/`: metrics, paper tables, fault classification, and sample cases.
- `scripts/`: runnable demo and evaluation scripts.
- `data/`: TrainTicket CSV fixtures.

## Run Demo

```bash
cd /Users/phoebe/Documents/Codex/2026-06-28/gen/outputs/marca_reproduction
python3 scripts/demo.py
```

## Reproduce TrainTicket Table I

```bash
python3 scripts/evaluate_trainticket_table1.py
```

Expected MARCA results:

```text
MARCA Llama-2      | 72.34% | 79.81% | 0.723
MARCA DeepSeek-V2  | 74.56% | 81.23% | 0.769
```

The TrainTicket fixture uses 137 cases with real `ts-*` service names from the
TrainTicket project. The CSV files do not include a weight column.

## Inspect Prompts

```bash
python3 scripts/inspect_prompts.py
```

## LLM Hook

By default, the code runs without an LLM. To connect a local vLLM or any
OpenAI-compatible endpoint:

```python
from marca_reproduction.core import Marca, MarcaConfig, OpenAICompatibleLLMClient

config = MarcaConfig(
    llm_enabled=True,
    model_name="your-local-model",
    llm_api_base="http://localhost:8000/v1",
)

marca = Marca(observability, config, llm_client=OpenAICompatibleLLMClient())
result = marca.diagnose("frontend")
```

LLM prompts and responses are recorded in `DiagnosisResult.llm_calls`.

## Notes

- The core algorithm is runnable without external dependencies.
- `scripts/paper_suite.py` prints paper tables and a small synthetic ablation suite.
- The TrainTicket evaluation fixture is constructed to reproduce the Table I
  numbers exactly.
