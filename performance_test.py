import json
import statistics
import time
from pathlib import Path

from test_adapter import ask_chatbot

REPRESENTATIVE_CASE_IDS = (
    "supported_boundary_01",
    "ambiguous_scope_01",
    "unsupported_01",
)

if __name__ == "__main__":
    cases = json.loads(Path("test_cases.json").read_text(encoding="utf-8"))
    selected_cases = [
        next(case for case in cases if case["id"] == case_id)
        for case_id in REPRESENTATIVE_CASE_IDS
    ]

    latencies = []
    failures = 0
    for case in selected_cases:
        start = time.perf_counter()
        try:
            answer = ask_chatbot(case["question"])
            if not str(answer).strip():
                failures += 1
        except Exception as exc:
            failures += 1
            print(f"{case['id']} ERROR: {exc}")
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        print(f"{case['id']}: {elapsed:.3f} seconds")

    if latencies:
        print("\n--- Performance Summary ---")
        print(f"Requests: {len(latencies)}")
        print(f"Failures: {failures}")
        print(f"Minimum: {min(latencies):.2f}s")
        print(f"Average: {statistics.mean(latencies):.2f}s")
        print(f"Median:  {statistics.median(latencies):.2f}s")
        print(f"Maximum: {max(latencies):.2f}s")
