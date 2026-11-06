from pathlib import Path

from rag_chatbot import PDF_DIR, build_fallback_answer, chunk_text, list_pdfs


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
