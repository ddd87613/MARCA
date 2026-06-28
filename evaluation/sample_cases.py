from __future__ import annotations

from typing import List

from evaluation.metrics import FaultCase
from marca_reproduction.schemas import ServiceSnapshot
from marca_reproduction.tools import InMemoryObservability


BASELINE_LATENCY = [95, 100, 102, 98, 101, 99]
BASELINE_CPU = [20, 22, 19, 21, 20, 23]
BASELINE_MEMORY = [45, 47, 44, 46, 45, 48]


def create_synthetic_fault_cases() -> List[FaultCase]:
    return [
        _case(
            name="payment-timeout",
            root="payment-gateway",
            fault_type="Network Delay",
            root_logs=["upstream unavailable: HTTP 503 timeout"],
            root_retcodes=["503"],
            root_latency=[760, 900, 970, 1120],
        ),
        _case(
            name="checkout-cpu-stress",
            root="checkout",
            fault_type="CPU Stress",
            root_logs=["cpu stress saturation caused worker timeout"],
            root_retcodes=["TIMEOUT"],
            root_latency=[500, 650, 800, 920],
            root_cpu=[91, 94, 96, 97],
        ),
        _case(
            name="cart-memory-stress",
            root="cart",
            fault_type="Memory Stress",
            root_logs=["memory stress and oom retry loop"],
            root_retcodes=["500"],
            root_latency=[430, 620, 700, 880],
            root_memory=[88, 92, 95, 97],
        ),
        _case(
            name="inventory-network-loss",
            root="inventory",
            fault_type="Network Loss",
            root_logs=["packet loss detected between gateway and inventory"],
            root_retcodes=["TIMEOUT"],
            root_latency=[600, 780, 850, 900],
        ),
        _case(
            name="shipping-network-partition",
            root="shipping",
            fault_type="Network Partition",
            root_logs=["network partition: service unreachable"],
            root_retcodes=["REFUSED"],
            root_latency=[700, 820, 930, 990],
        ),
        _case(
            name="profile-process-kill",
            root="profile",
            fault_type="Process Kill",
            root_logs=["process killed after fatal panic"],
            root_retcodes=["500"],
            root_latency=[480, 580, 720, 760],
        ),
    ]


def _case(
    name: str,
    root: str,
    fault_type: str,
    root_logs: List[str],
    root_retcodes: List[str],
    root_latency: List[float],
    root_cpu: List[float] | None = None,
    root_memory: List[float] | None = None,
) -> FaultCase:
    topology = {
        "frontend": ["order-service"],
        "order-service": [root, "database"],
        root: [],
        "database": [],
    }
    snapshots = {
        "frontend": ServiceSnapshot(
            baseline_metrics={"latency": BASELINE_LATENCY},
            current_metrics={"latency": [800, 900, 1000, 1080]},
            logs=["user-visible latency symptom propagated to frontend"],
            retcodes=["200"],
            trace_error_rate=0.25,
            trace_latency_ms=1000,
        ),
        "order-service": ServiceSnapshot(
            baseline_metrics={"latency": BASELINE_LATENCY},
            current_metrics={"latency": [760, 850, 950, 1010]},
            logs=["retrying downstream request"],
            retcodes=["TIMEOUT"],
            trace_error_rate=0.32,
            trace_latency_ms=900,
        ),
        root: ServiceSnapshot(
            baseline_metrics={
                "latency": BASELINE_LATENCY,
                "cpu": BASELINE_CPU,
                "memory": BASELINE_MEMORY,
            },
            current_metrics={
                "latency": root_latency,
                "cpu": root_cpu or BASELINE_CPU,
                "memory": root_memory or BASELINE_MEMORY,
            },
            logs=root_logs,
            retcodes=root_retcodes,
            trace_error_rate=0.72,
            trace_latency_ms=max(root_latency),
        ),
        "database": ServiceSnapshot(
            baseline_metrics={"latency": BASELINE_LATENCY},
            current_metrics={"latency": [180, 230, 260, 280]},
            logs=["slow query observed after retry storm"],
            retcodes=["200"],
            trace_error_rate=0.05,
            trace_latency_ms=280,
        ),
        "unrelated-noisy-service": ServiceSnapshot(
            baseline_metrics={"latency": BASELINE_LATENCY},
            current_metrics={"latency": [950, 980, 990, 1000]},
            logs=["fatal unrelated error outside the topology path"],
            retcodes=["503"],
            trace_error_rate=0.90,
            trace_latency_ms=1000,
        ),
    }
    return FaultCase(
        name=name,
        observability=InMemoryObservability(topology, snapshots),
        entry_service="frontend",
        root_cause=root,
        fault_type=fault_type,
    )
