from pathlib import Path

from rag_chatbot import (
    PDF_DIR,
    answer_question,
    build_fallback_answer,
    chunk_text,
    list_pdfs,
    retrieve_documents,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pdf_directory_is_configured_in_project():
    assert PDF_DIR == PROJECT_ROOT / "PDF Files"


def test_pdf_directory_contains_source_documents():
    assert list_pdfs(PDF_DIR)


def test_chunk_text_normalizes_and_overlaps_text():
    chunks = chunk_text("one two three four five", chunk_size=13, overlap=4)

    assert chunks
    assert all(chunk == chunk.strip() for chunk in chunks)
    assert chunks[0] != chunks[-1]


def test_fallback_answer_handles_empty_context():
    assert "could not find relevant context" in build_fallback_answer("question", [])


def test_retrieval_finds_lexical_match_outside_vector_candidates(monkeypatch):
    question = (
        "What proportion of Microsoft 365 Copilot chat use supports analysis, "
        "decisions, and problem-solving according to the 2026 Work Trend Index?"
    )
    chunks = [
        {"pdf": "unrelated.pdf", "chunk_index": index, "text": "General software development content"}
        for index in range(8)
    ]
    target = {
        "pdf": "AI Impact on Software Engineers 2026.pdf",
        "chunk_index": 8,
        "text": (
            "Nearly half of Microsoft 365 Copilot chat information use supports "
            "analysis, decisions, and problem-solving. 49% supports high-value work."
        ),
    }
    chunks.append(target)

    class Index:
        def search(self, _embedding, count):
            return [[float(index) for index in range(count)]], [list(range(count))]

    class Model:
        def encode(self, *_args, **_kwargs):
            return [[0.0]]

    monkeypatch.setattr("rag_chatbot.load_index", lambda: (Index(), chunks, Model()))

    assert retrieve_documents(question, top_k=1) == [target]


def test_retrieval_prioritizes_concerns_in_named_survey(monkeypatch):
    question = (
        "According to the 2026 State of Code Developer Survey, what concerns "
        "about AI-generated code may affect code quality and maintainability?"
    )
    chunks = [
        {
            "pdf": "state-of-code-developer-survey-report.pdf",
            "chunk_index": index,
            "text": "The State of Code Developer Survey report provides an overview.",
        }
        for index in range(8)
    ]
    target = {
        "pdf": "state-of-code-developer-survey-report.pdf",
        "chunk_index": 8,
        "text": (
            'Concerns include AI code that "looks correct but isn\'t reliable" '
            'and "too many lines of code are being generated with AI".'
        ),
    }
    chunks.append(target)

    class Index:
        def search(self, _embedding, count):
            indices = [8] + list(range(count - 1))
            return [[float(index) for index in range(count)]], [indices]

    class Model:
        def encode(self, *_args, **_kwargs):
            return [[0.0]]

    monkeypatch.setattr("rag_chatbot.load_index", lambda: (Index(), chunks, Model()))

    assert retrieve_documents(question, top_k=1) == [target]


def test_copilot_chat_analysis_answers_use_named_report_evidence(monkeypatch):
    source_chunk = {
        "pdf": "AI Impact on Software Engineers 2026.pdf",
        "chunk_index": 9,
        "text": (
            "Nearly half of Microsoft 365 Copilot chat information use supports "
            "analysis, decisions, and problem-solving."
        ),
    }
    unrelated_chunk = {
        "pdf": "state-of-code-developer-survey-report.pdf",
        "chunk_index": 18,
        "text": "Survey results about coding tools.",
    }
    index_chunks = [
        {
            "pdf": "AI Impact on Software Engineers 2026.pdf",
            "chunk_index": 8,
            "text": "The chart identifies 49% of Copilot chat use as high-value work.",
        },
        source_chunk,
    ]
    monkeypatch.setattr(
        "rag_chatbot.load_index",
        lambda: (None, index_chunks, None),
    )
    monkeypatch.setattr(
        "rag_chatbot.retrieve_documents",
        lambda _question, top_k=5: [unrelated_chunk],
    )

    questions = [
        (
            "What proportion of Microsoft 365 Copilot chat use supports analysis, "
            "decisions, and problem-solving according to the 2026 Work Trend Index?"
        ),
        (
            "How much of Copilot's chat activity is dedicated to analyzing data, "
            "making choices, and solving issues in the 2026 report?"
        ),
    ]
    for question in questions:
        result = answer_question(question)
        assert "49%" in result["answer"]
        assert "analysis" in result["answer"].casefold()
        assert source_chunk in result["sources"]
