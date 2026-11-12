import json
import math
from pathlib import Path

import pytest

from rag_chatbot import answer_question


PROJECT_ROOT = Path(__file__).resolve().parents[1]
with (PROJECT_ROOT / "test_cases.json").open(encoding="utf-8") as handle:
    QA_CASES = json.load(handle)


REFUSAL_PHRASES = (
    "not explicitly mentioned",
    "could not find",
    "not found",
    "not provided",
    "unable to",
    "do not know",
    "don't know",
    "not in the documents",
    "not in the pdf",
)

CLARIFICATION_PHRASES = (
    "clarify",
    "which organization",
    "which tool",
    "please specify",
    "could you specify",
    "more information",
)


@pytest.mark.parametrize("case", QA_CASES, ids=[case["id"] for case in QA_CASES])
def test_chatbot_against_json_case(case):
    answer = answer_question(case["question"])["answer"]
    assert isinstance(answer, str) and answer.strip(), (
        f"{case['id']}: chatbot returned an empty answer"
    )

    normalized_answer = answer.casefold()
    case_type = case["type"]

    if case_type == "unsupported" or case.get("expected_response") == "not_found":
        assert any(phrase in normalized_answer for phrase in REFUSAL_PHRASES), (
            f"{case['id']}: expected an explicit unavailable-information response; "
            f"got: {answer}"
        )
        return

    if case_type == "ambiguous" and case["id"] == "ambiguous_scope_01":
        assert any(phrase in normalized_answer for phrase in CLARIFICATION_PHRASES), (
            f"{case['id']}: expected a clarification request; got: {answer}"
        )
        return

    expected_keywords = case.get("expected_keywords", [])
    assert expected_keywords, f"{case['id']}: expected_keywords must not be empty"
    matched_keywords = [
        keyword
        for keyword in expected_keywords
        if keyword.casefold() in normalized_answer
    ]
    required_matches = max(1, math.ceil(len(expected_keywords) / 2))
    assert len(matched_keywords) >= required_matches, (
        f"{case['id']}: expected at least {required_matches} of "
        f"{len(expected_keywords)} keywords, matched {matched_keywords}; "
        f"answer: {answer}"
    )