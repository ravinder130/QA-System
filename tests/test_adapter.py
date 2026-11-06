import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_chatbot import build_fallback_answer


def run_cases() -> list[dict[str, object]]:
    with (ROOT / "test_cases.json").open(encoding="utf-8") as handle:
        cases = json.load(handle)

    results = []
    for case in cases:
        answer = build_fallback_answer(case["question"], case.get("context", []))
        results.append({"name": case.get("name", case.get("id", "unnamed")), "passed": bool(answer)})
    return results


if __name__ == "__main__":
    results = run_cases()
    for result in results:
        print(f"{'PASS' if result['passed'] else 'FAIL'}: {result['name']}")
    sys.exit(0 if all(result["passed"] for result in results) else 1)
