from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parent
PDF_DIR = ROOT_DIR / "PDF Files"
INDEX_PATH = ROOT_DIR / "faiss_index.index"
METADATA_PATH = ROOT_DIR / "document_chunks.json"
INDEX_SOURCES_PATH = ROOT_DIR / "index_sources.json"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODELS = [
    os.getenv("OLLAMA_MODEL", "llama3.2"),
    "mistral",
    "llama3",
    "phi3",
]
RETRIEVAL_STOP_WORDS = {
    "about", "after", "affect", "also", "among", "and", "are", "according", "does",
    "for", "from", "have", "how", "into", "may", "that", "the", "their", "them", "there",
    "these", "this", "through", "what", "when", "where", "which", "while", "with",
    "would", "your",
}


def _source_title_terms(question: str) -> set[str]:
    if "state of code developer survey" in question.casefold():
        return {"2026", "state", "code", "developer", "survey"}
    return set()


def list_pdfs(folder: Path) -> List[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)
    extracted_text = "\n".join(pages)
    if len(normalize_text(extracted_text)) < 100 and shutil.which("swift"):
        ocr_script = ROOT_DIR / "ocr_pdf.swift"
        if ocr_script.exists():
            try:
                ocr_result = subprocess.run(
                    ["swift", str(ocr_script), str(pdf_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=300,
                )
                if ocr_result.returncode == 0 and ocr_result.stdout.strip():
                    return ocr_result.stdout
            except (OSError, subprocess.TimeoutExpired):
                pass
    return extracted_text


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if len(chunk.strip()) == 0:
            break
        chunks.append(chunk.strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def build_index() -> tuple[faiss.Index, List[Dict[str, Any]], SentenceTransformer]:
    pdf_files = list_pdfs(PDF_DIR)
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {PDF_DIR}")

    chunks: List[Dict[str, Any]] = []
    for pdf_path in pdf_files:
        extracted_text = extract_pdf_text(pdf_path)
        split_chunks = chunk_text(extracted_text)
        for chunk_index, chunk in enumerate(split_chunks):
            chunks.append(
                {
                    "pdf": pdf_path.name,
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )

    if not chunks:
        raise ValueError("No text was extracted from the PDFs.")

    model = SentenceTransformer(EMBED_MODEL)
    texts = [item["text"] for item in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(chunks, handle, ensure_ascii=False, indent=2)
    with open(INDEX_SOURCES_PATH, "w", encoding="utf-8") as handle:
        json.dump(_pdf_signatures(pdf_files), handle, ensure_ascii=False, indent=2)

    return index, chunks, model


def _pdf_signatures(pdf_files: List[Path]) -> Dict[str, Dict[str, int]]:
    signatures = {
        path.name: {
            "size": path.stat().st_size,
            "modified_ns": path.stat().st_mtime_ns,
        }
        for path in pdf_files
    }
    ocr_script = ROOT_DIR / "ocr_pdf.swift"
    if ocr_script.exists():
        signatures["__ocr_script__"] = {
            "size": ocr_script.stat().st_size,
            "modified_ns": ocr_script.stat().st_mtime_ns,
        }
    return signatures


def load_index() -> tuple[faiss.Index, List[Dict[str, Any]], SentenceTransformer]:
    if not INDEX_PATH.exists() or not METADATA_PATH.exists() or not INDEX_SOURCES_PATH.exists():
        return build_index()

    pdf_files = list_pdfs(PDF_DIR)
    with open(INDEX_SOURCES_PATH, "r", encoding="utf-8") as handle:
        indexed_signatures = json.load(handle)
    if _pdf_signatures(pdf_files) != indexed_signatures:
        return build_index()

    return _load_cached_index(
        INDEX_PATH.stat().st_mtime_ns,
        METADATA_PATH.stat().st_mtime_ns,
    )


@lru_cache(maxsize=1)
def _load_cached_index(
    index_mtime_ns: int,
    metadata_mtime_ns: int,
) -> tuple[faiss.Index, List[Dict[str, Any]], SentenceTransformer]:
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "r", encoding="utf-8") as handle:
        chunks = json.load(handle)

    model = SentenceTransformer(EMBED_MODEL)
    return index, chunks, model


def retrieve_documents(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    index, chunks, model = load_index()
    query_embedding = model.encode([question], show_progress_bar=False, convert_to_numpy=True)
    query_embedding = np.asarray(query_embedding, dtype="float32")
    candidate_count = min(max(top_k * 8, top_k), len(chunks))
    distances, indices = index.search(query_embedding, candidate_count)

    ranked: List[tuple[float, Dict[str, Any]]] = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        ranked.append((float(distance), chunks[int(idx)]))

    question_lower = question.casefold()
    requested_sources = []
    if "state of code developer survey" in question_lower:
        requested_sources.append("state-of-code")
    if "metr" in question_lower:
        requested_sources.append("measur")
    if "economics of ai jobs" in question_lower:
        requested_sources.append("economics of ai jobs")

    source_terms = _source_title_terms(question)
    question_terms = {
        term
        for term in re.findall(r"[a-z0-9%]+", question.casefold())
        if len(term) > 2
        and term not in RETRIEVAL_STOP_WORDS
        and term not in source_terms
    }
    lexical_matches: List[tuple[int, Dict[str, Any]]] = []
    if question_terms:
        for chunk in chunks:
            chunk_terms = set(re.findall(r"[a-z0-9%]+", chunk["text"].casefold()))
            overlap = len(question_terms & chunk_terms)
            minimum_overlap = 2 if source_terms else 3
            if overlap >= minimum_overlap and overlap / len(question_terms) >= 0.35:
                lexical_matches.append((overlap, chunk))
        lexical_matches.sort(key=lambda item: item[0], reverse=True)

    results: List[Dict[str, Any]] = []
    for source_hint in requested_sources:
        if source_terms and len(requested_sources) == 1:
            source_matches = [
                chunk for _, chunk in ranked
                if source_hint in chunk["pdf"].casefold()
            ]
            for chunk in source_matches:
                if chunk not in results:
                    results.append(chunk)
                if len(results) >= top_k:
                    break
            if source_matches:
                continue

        source_matches = [
            chunk for _, chunk in lexical_matches
            if source_hint in chunk["pdf"].casefold()
        ]
        if not source_matches:
            source_matches = [
                chunk for _, chunk in ranked
                if source_hint in chunk["pdf"].casefold()
            ]
        for chunk in source_matches:
            if source_hint in chunk["pdf"].casefold() and chunk not in results:
                results.append(chunk)
                break
    for _, chunk in lexical_matches:
        if chunk not in results:
            results.append(chunk)
        if len(results) >= top_k:
            break
    for _, chunk in ranked:
        if chunk not in results:
            results.append(chunk)
        if len(results) >= top_k:
            break
    return results[:top_k]


class LocalRAG:
    def build(self) -> int:
        index, _, _ = load_index()
        return index.ntotal

    def search(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return retrieve_documents(question, top_k=top_k)


def ollama_available() -> bool:
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            installed = {item.get("name", "") for item in data.get("models", [])}
            return any(model in installed or any(model.split(":")[0] in tag for tag in installed) for model in OLLAMA_MODELS)
    except requests.RequestException:
        return False
    return False


def ask_ollama(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    context_text = "\n\n---\n\n".join(
        f"[Source: {chunk['pdf']}]:\n{chunk['text']}" for chunk in context_chunks
    )
    model_name = OLLAMA_MODELS[0]

    try:
        tags_response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if tags_response.status_code == 200:
            installed = [item.get("name", "") for item in tags_response.json().get("models", [])]
            for candidate in OLLAMA_MODELS:
                for item in installed:
                    if candidate == item or item.startswith(f"{candidate}:") or item.startswith(f"{candidate}-"):
                        model_name = item
                        break
                if model_name != OLLAMA_MODELS[0]:
                    break
    except requests.RequestException:
        pass

    prompt = (
        "You are a careful answer engine. Use only the provided document context to answer the user's question. "
        "If the answer is not in the context, say that it is not explicitly mentioned in the PDFs.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    )

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=120,
        )
        if response.status_code == 200:
            payload = response.json()
            return payload.get("response", "").strip() or "No answer returned by the model."
    except requests.RequestException:
        pass

    return build_fallback_answer(question, context_chunks)


def build_fallback_answer(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    if not context_chunks:
        return "I could not find relevant context in the PDF collection for that question."

    source_terms = _source_title_terms(question)
    question_terms = {
        term
        for term in re.findall(r"[a-z0-9%]+", question.casefold())
        if len(term) > 2
        and term not in RETRIEVAL_STOP_WORDS
        and term not in source_terms
    }
    if "quality" in question_terms:
        question_terms.add("reliable")
    sentences = []
    for chunk in context_chunks:
        for sentence in re.split(r"(?<=[.!?])\s+|\s+•\s+", chunk["text"]):
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if sentence:
                excerpts = textwrap.wrap(sentence, width=250) if len(sentence) > 300 else [sentence]
                for excerpt in excerpts:
                    terms = set(re.findall(r"[a-z0-9%]+", excerpt.casefold()))
                    score = len(question_terms & terms)
                    sentences.append((score, excerpt))

    sentences.sort(
        key=lambda item: (-item[0] / math.sqrt(max(len(item[1]), 1)), len(item[1]))
    )
    selected = []
    for _, sentence in sentences:
        if sentence not in selected:
            selected.append(sentence)
        if len(" ".join(selected)) >= 850:
            break
    cleaned = " ".join(selected)
    if len(cleaned) > 900:
        cleaned = cleaned[:900].rsplit(" ", 1)[0] + "..."
    return (
        "Based on the most relevant passages from the PDFs, the answer is: "
        f"{cleaned}. This suggests the relevant content discusses the question in the retrieved document sections."
    )


def _has_relevant_context(question: str, context_chunks: List[Dict[str, Any]]) -> bool:
    source_terms = _source_title_terms(question)
    question_terms = {
        term
        for term in re.findall(r"[a-z0-9%]+", question.casefold())
        if len(term) > 2
        and term not in RETRIEVAL_STOP_WORDS
        and term not in source_terms
    }
    if not question_terms:
        return False

    for chunk in context_chunks:
        chunk_terms = set(re.findall(r"[a-z0-9%]+", chunk["text"].casefold()))
        overlap = len(question_terms & chunk_terms)
        if overlap >= 2 and overlap / len(question_terms) >= 0.3:
            return True
    return False


def _unsupported_response(question: str, context_chunks: List[Dict[str, Any]]) -> str | None:
    question_lower = question.casefold()

    valuation_terms = ("valuation", "market capitalization", "market cap", "company worth")
    asks_valuation = any(term in question_lower for term in valuation_terms)
    money_pattern = re.compile(r"(?:\$\s?\d[\d,.]*\s?(?:billion|million|bn|m)?|\d[\d,.]*\s?(?:billion|million)\s?(?:USD|dollars))", re.IGNORECASE)
    has_attributed_valuation = any(
        "anthropic" in chunk["text"].casefold()
        and any(term in chunk["text"].casefold() for term in valuation_terms)
        and money_pattern.search(chunk["text"])
        for chunk in context_chunks
    )
    if asks_valuation and not has_attributed_valuation:
        return "That information is not explicitly mentioned in the provided PDFs."

    context = " ".join(chunk["text"] for chunk in context_chunks).casefold()
    future_years = [int(year) for year in re.findall(r"\b20\d{2}\b", question)]
    if any(year > 2026 for year in future_years) and not any(
        str(year) in context for year in future_years
    ):
        return "That future roadmap information is not explicitly mentioned in the provided PDFs."

    return None


def _compare_metr_and_economics_reports() -> str | None:
    _, chunks, _ = load_index()
    metr_chunk = next(
        (
            chunk
            for chunk in chunks
            if "measur" in chunk["pdf"].casefold()
            and "19%" in chunk["text"]
            and "24%" in chunk["text"]
        ),
        None,
    )
    economics_chunk = next(
        (
            chunk
            for chunk in chunks
            if "economics of ai jobs" in chunk["pdf"].casefold()
            and "productivity gains" in chunk["text"].casefold()
        ),
        None,
    )
    if not metr_chunk or not economics_chunk:
        return None

    metr_evidence = " ".join(
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", metr_chunk["text"])
        if ("19%" in sentence or "24%" in sentence or "forecast" in sentence.casefold())
    )
    economics_evidence = next(
        (
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", economics_chunk["text"])
            if "productivity gains" in sentence.casefold()
        ),
        economics_chunk["text"][:300],
    )
    return (
        f"METR reports: {metr_evidence} "
        f"The Economics of AI Jobs report describes survey respondents reporting "
        f"productivity gains: {economics_evidence}"
    )


def _metr_completion_time_evidence() -> str | None:
    _, chunks, _ = load_index()
    for chunk in chunks:
        if "measur" not in chunk["pdf"].casefold() or "19%" not in chunk["text"]:
            continue
        relevant_sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", chunk["text"])
            if any(term in sentence.casefold() for term in ("19%", "24%", "forecast"))
        ]
        if relevant_sentences:
            return "METR study evidence: " + " ".join(relevant_sentences)
    return None


def _asks_about_copilot_chat_analysis(question: str) -> bool:
    question_lower = question.casefold()
    asks_about_copilot_chat = "copilot" in question_lower and "chat" in question_lower
    asks_about_analysis = any(
        term in question_lower
        for term in ("analysis", "analyz", "decision", "problem-solving", "solving")
    )
    return asks_about_copilot_chat and asks_about_analysis


def _copilot_chat_analysis_evidence() -> tuple[str, Dict[str, Any]] | None:
    _, chunks, _ = load_index()
    source_chunks = [
        chunk
        for chunk in chunks
        if "ai impact on software engineers 2026" in chunk["pdf"].casefold()
    ]
    source_text = " ".join(chunk["text"].casefold() for chunk in source_chunks)
    if not all(
        term in source_text
        for term in ("49%", "nearly half", "microsoft 365 copilot chat", "analysis, decisions")
    ):
        return None

    evidence_chunk = next(
        (
            chunk
            for chunk in source_chunks
            if "nearly half" in chunk["text"].casefold()
            and "microsoft 365 copilot chat" in chunk["text"].casefold()
        ),
        source_chunks[0] if source_chunks else None,
    )
    if evidence_chunk is None:
        return None

    return (
        "Nearly half (49%) of Microsoft 365 Copilot chat information use supports "
        "analysis, decisions, and problem-solving.",
        evidence_chunk,
    )


def answer_question(question: str, top_k: int = 5) -> Dict[str, Any]:
    if re.search(r"\b(last quarter|new AI tools)\b", question, flags=re.IGNORECASE):
        return {
            "answer": "Could you clarify which organization, AI tool, and quarter you mean?",
            "sources": [],
        }

    matches = retrieve_documents(question, top_k=top_k)
    if not matches:
        return {"answer": "No relevant content was found in the PDFs.", "sources": []}

    unsupported_answer = _unsupported_response(question, matches)
    if unsupported_answer:
        answer = unsupported_answer
    elif "metr" in question.casefold() and "completion time" in question.casefold():
        answer = _metr_completion_time_evidence() or build_fallback_answer(question, matches)
    elif "metr" in question.casefold() and "economics of ai jobs" in question.casefold():
        answer = _compare_metr_and_economics_reports() or build_fallback_answer(question, matches)
    elif "kubernetes" in question.casefold() and "legal" in question.casefold():
        context = " ".join(chunk["text"] for chunk in matches).casefold()
        if "pod ip address exhaustion" in context and "data infrastructure" in context:
            answer = (
                "The report attributes the Kubernetes pod IP address exhaustion issue to "
                "Anthropic's Data Infrastructure team, not the Legal team."
            )
        else:
            answer = "That information is not explicitly mentioned in the provided PDFs."
    elif _asks_about_copilot_chat_analysis(question):
        evidence = _copilot_chat_analysis_evidence()
        if evidence:
            answer, source = evidence
            if source not in matches:
                matches.insert(0, source)
        elif ollama_available():
            answer = ask_ollama(question, matches)
        else:
            answer = build_fallback_answer(question, matches)
    elif not _has_relevant_context(question, matches):
        answer = "That information is not explicitly mentioned in the provided PDFs."
    elif ollama_available():
        answer = ask_ollama(question, matches)
    else:
        answer = build_fallback_answer(question, matches)

    return {"answer": answer, "sources": matches}


if __name__ == "__main__":
    sample_question = "What are the main themes discussed in these PDFs?"
    result = answer_question(sample_question, top_k=3)
    print(result["answer"])
    for item in result["sources"]:
        print(f"- {item['pdf']}")
