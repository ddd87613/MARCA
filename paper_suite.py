from evaluation import estimate_token_reduction, run_ablation_suite
from paper_details import (
    ABLATION_FINDINGS,
    DATASETS,
    SENSITIVITY_TABLE,
    format_table_1,
)
from sample_cases import create_synthetic_fault_cases
from schemas import MarcaConfig


def main() -> None:
    print("Datasets")
    print("========")
    for dataset in DATASETS:
        print(f"- {dataset.name}: {dataset.description} ({dataset.services})")

    print("\nTable I")
    print("=======")
    print(format_table_1())

    print("\nSensitivity Table")
    print("=================")
    for name, metrics in SENSITIVITY_TABLE.items():
        print(f"- {name}: Acc@1={metrics['acc_at_1']}%, F1={metrics['f1']}")

    print("\nAblation Findings")
    print("=================")
    for name, finding in ABLATION_FINDINGS.items():
        print(f"- {name}: {finding}")

    print("\nSynthetic Reproduction Suite")
    print("============================")
    cases = create_synthetic_fault_cases()
    results = run_ablation_suite(cases)
    for name, result in results.items():
        print(
            f"- {name}: Acc@1={result.acc_at_1:.3f}, "
            f"Acc@3={result.acc_at_3:.3f}, macro-F1={result.macro_f1:.3f}"
        )

    config = MarcaConfig()
    print("\nToken reduction target")
    print("======================")
    print(f"{estimate_token_reduction(config):.1%}")


if __name__ == "__main__":
    main()
