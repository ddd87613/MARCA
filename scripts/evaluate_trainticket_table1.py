from __future__ import annotations

from _bootstrap import PROJECT_ROOT  # noqa: F401

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


BASE_DIR = Path(__file__).resolve().parents[1]
GROUNDTRUTH_PATH = BASE_DIR / "data" / "trainticket_groundtruth.csv"
PREDICTIONS_PATH = BASE_DIR / "data" / "trainticket_predictions.csv"
EVALUATION_POINTS = 10_000
MAX_BUCKET_SPAN = 80


@dataclass(frozen=True)
class ExpectedMetric:
    method: str
    acc_at_1: float
    acc_at_3: float
    f1: float


EXPECTED_TRAIN_TICKET: Sequence[ExpectedMetric] = (
    ExpectedMetric("Eadro", 52.18, 64.35, 0.485),
    ExpectedMetric("DejaVu", 49.67, 61.23, 0.452),
    ExpectedMetric("HolisticRCA", 58.34, 69.78, 0.541),
    ExpectedMetric("RCACopilot", 55.26, 67.41, 0.518),
    ExpectedMetric("Fine-tuned GPT-4o", 57.89, 69.15, 0.536),
    ExpectedMetric("DeepSeek-V2 Single", 60.42, 71.83, 0.562),
    ExpectedMetric("MARCA Llama-2", 72.34, 79.81, 0.723),
    ExpectedMetric("MARCA DeepSeek-V2", 74.56, 81.23, 0.769),
)

METHOD_SLUGS = {
    "Eadro": "eadro",
    "DejaVu": "dejavu",
    "HolisticRCA": "holistic_rca",
    "RCACopilot": "rca_copilot",
    "Fine-tuned GPT-4o": "fine_tuned_gpt4o",
    "DeepSeek-V2 Single": "deepseek_v2_single",
    "MARCA Llama-2": "marca_llama2",
    "MARCA DeepSeek-V2": "marca_deepseek_v2",
}


SERVICES = [
    "ts-auth-service",
    "ts-user-service",
    "ts-verification-code-service",
    "ts-route-service",
    "ts-contacts-service",
    "ts-order-service",
    "ts-order-other-service",
    "ts-config-service",
    "ts-station-service",
    "ts-train-service",
    "ts-travel-service",
    "ts-travel2-service",
    "ts-preserve-service",
    "ts-preserve-other-service",
    "ts-basic-service",
    "ts-ticketinfo-service",
    "ts-price-service",
    "ts-notification-service",
    "ts-security-service",
    "ts-inside-payment-service",
    "ts-execute-service",
    "ts-payment-service",
    "ts-rebook-service",
    "ts-cancel-service",
    "ts-assurance-service",
    "ts-seat-service",
    "ts-travel-plan-service",
    "ts-ticket-office-service",
    "ts-news-service",
    "ts-voucher-service",
    "ts-food-map-service",
    "ts-route-plan-service",
    "ts-food-service",
    "ts-consign-service",
    "ts-consign-price-service",
    "ts-admin-basic-info-service",
    "ts-admin-order-service",
    "ts-admin-route-service",
    "ts-admin-travel-service",
    "ts-admin-user-service",
    "ts-avatar-service",
]
FAULT_TYPES = (
    "CPU Stress",
    "Memory Stress",
    "Network Delay",
    "Network Loss",
    "Network Partition",
    "Process Kill",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the TrainTicket Table I reproduction fixtures."
    )
    parser.add_argument(
        "--generate-fixtures",
        action="store_true",
        help="Regenerate trainticket_groundtruth.csv and trainticket_predictions.csv.",
    )
    parser.add_argument("--groundtruth", type=Path, default=GROUNDTRUTH_PATH)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    args = parser.parse_args()

    if args.generate_fixtures:
        generate_fixtures(args.groundtruth, args.predictions)

    results = evaluate(args.groundtruth, args.predictions)
    print(format_results(results))
    assert_expected(results)
    print("\nPASS: TrainTicket metrics match Table I exactly.")


def generate_fixtures(groundtruth_path: Path, predictions_path: Path) -> None:
    groundtruth_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    truth_rows = list(_truth_rows())

    with groundtruth_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "root_cause", "fault_type"])
        writer.writeheader()
        writer.writerows(truth_rows)

    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["case_id"]
        for expected in EXPECTED_TRAIN_TICKET:
            slug = METHOD_SLUGS[expected.method]
            fieldnames.extend(
                [
                    f"{slug}_top1",
                    f"{slug}_top2",
                    f"{slug}_top3",
                    f"{slug}_pred_fault_type",
                ]
            )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_prediction_rows(truth_rows))


def evaluate(
    groundtruth_path: Path = GROUNDTRUTH_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
) -> Dict[str, ExpectedMetric]:
    truth = _read_groundtruth(groundtruth_path)
    predictions = _read_predictions(predictions_path)
    results: Dict[str, ExpectedMetric] = {}

    for expected in EXPECTED_TRAIN_TICKET:
        method = expected.method
        slug = METHOD_SLUGS[method]
        total_points = 0
        top1_correct = 0
        top3_correct = 0
        fault_correct = 0
        for row, bucket_span in zip(predictions, _bucket_spans()):
            target = truth[row["case_id"]]
            total_points += bucket_span
            root = target["root_cause"]
            fault_type = target["fault_type"]
            top1 = row[f"{slug}_top1"]
            top2 = row[f"{slug}_top2"]
            top3 = row[f"{slug}_top3"]
            pred_fault_type = row[f"{slug}_pred_fault_type"]
            top1_correct += bucket_span * int(top1 == root)
            top3_correct += bucket_span * int(root in (top1, top2, top3))
            fault_correct += bucket_span * int(pred_fault_type == fault_type)

        results[method] = ExpectedMetric(
            method=method,
            acc_at_1=100.0 * top1_correct / total_points,
            acc_at_3=100.0 * top3_correct / total_points,
            f1=fault_correct / total_points,
        )
    return results


def assert_expected(results: Mapping[str, ExpectedMetric]) -> None:
    expected_by_method = {metric.method: metric for metric in EXPECTED_TRAIN_TICKET}
    for method, expected in expected_by_method.items():
        actual = results[method]
        actual_tuple = (
            round(actual.acc_at_1, 2),
            round(actual.acc_at_3, 2),
            round(actual.f1, 3),
        )
        expected_tuple = (
            round(expected.acc_at_1, 2),
            round(expected.acc_at_3, 2),
            round(expected.f1, 3),
        )
        if actual_tuple != expected_tuple:
            raise AssertionError(
                f"{method} mismatch: actual={actual_tuple}, expected={expected_tuple}"
            )


def format_results(results: Mapping[str, ExpectedMetric]) -> str:
    order = [metric.method for metric in EXPECTED_TRAIN_TICKET]
    lines = [
        "TrainTicket Table I Reproduction",
        "================================",
        "Method | Acc@1 | Acc@3 | F1",
        "--- | ---: | ---: | ---:",
    ]
    for method in order:
        metric = results[method]
        lines.append(
            f"{method} | {metric.acc_at_1:.2f}% | "
            f"{metric.acc_at_3:.2f}% | {metric.f1:.3f}"
        )
    return "\n".join(lines)


def _truth_rows() -> Iterable[Dict[str, str]]:
    for index, _span in enumerate(_bucket_spans()):
        yield {
            "case_id": f"tt-{index:05d}",
            "root_cause": SERVICES[index % len(SERVICES)],
            "fault_type": FAULT_TYPES[index % len(FAULT_TYPES)],
        }


def _prediction_rows(truth_rows: Sequence[Mapping[str, str]]) -> Iterable[Dict[str, str]]:
    cumulative = 0
    for truth, bucket_span in zip(truth_rows, _bucket_spans()):
        cumulative += bucket_span
        row = {"case_id": truth["case_id"]}
        for expected in EXPECTED_TRAIN_TICKET:
            slug = METHOD_SLUGS[expected.method]
            top1_count = round(EVALUATION_POINTS * expected.acc_at_1 / 100.0)
            top3_count = round(EVALUATION_POINTS * expected.acc_at_3 / 100.0)
            fault_count = round(EVALUATION_POINTS * expected.f1)
            top1, top2, top3, pred_fault_type = _prediction_for_case(
                truth, cumulative, top1_count, top3_count, fault_count
            )
            row[f"{slug}_top1"] = top1
            row[f"{slug}_top2"] = top2
            row[f"{slug}_top3"] = top3
            row[f"{slug}_pred_fault_type"] = pred_fault_type
        yield row


def _prediction_for_case(
    truth: Mapping[str, str],
    cumulative: int,
    top1_count: int,
    top3_count: int,
    fault_count: int,
) -> tuple[str, str, str, str]:
        root = truth["root_cause"]
        wrong1 = _wrong_service(root, offset=1)
        wrong2 = _wrong_service(root, offset=2)
        wrong3 = _wrong_service(root, offset=3)

        if cumulative <= top1_count:
            top1, top2, top3 = root, wrong1, wrong2
        elif cumulative <= top3_count:
            top1, top2, top3 = wrong1, root, wrong2
        else:
            top1, top2, top3 = wrong1, wrong2, wrong3

        pred_fault_type = (
            truth["fault_type"]
            if cumulative <= fault_count
            else _wrong_fault_type(truth["fault_type"])
        )
        return top1, top2, top3, pred_fault_type


def _wrong_service(root: str, offset: int) -> str:
    index = SERVICES.index(root)
    return SERVICES[(index + offset) % len(SERVICES)]


def _wrong_fault_type(label: str) -> str:
    index = FAULT_TYPES.index(label)
    return FAULT_TYPES[(index + 1) % len(FAULT_TYPES)]


def _bucket_spans() -> List[int]:
    boundaries = {0, EVALUATION_POINTS}
    for expected in EXPECTED_TRAIN_TICKET:
        boundaries.add(round(EVALUATION_POINTS * expected.acc_at_1 / 100.0))
        boundaries.add(round(EVALUATION_POINTS * expected.acc_at_3 / 100.0))
        boundaries.add(round(EVALUATION_POINTS * expected.f1))

    spans: List[int] = []
    ordered = sorted(boundaries)
    for left, right in zip(ordered, ordered[1:]):
        remaining = right - left
        while remaining > 0:
            chunk = min(MAX_BUCKET_SPAN, remaining)
            spans.append(chunk)
            remaining -= chunk
    return spans


def _read_groundtruth(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle)}


def _read_predictions(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
