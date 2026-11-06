# QA System Test Report

**Date:** October 1, 2026  
**Project:** PDF RAG Chatbot  
**Environment:** macOS, Python 3.9.6, pytest 8.4.2, `.venv`

## Summary

The latest pytest run executed the 4 unit tests and all 10 JSON QA scenarios: **14 passed, 0 failed**, with process exit code `0`. Four non-fatal dependency warnings were reported.

## Automated tests executed

Command:

```bash
.venv/bin/python -m pytest -q
```

| Test | Result | What it checks |
|---|---|---|
| `test_pdf_directory_is_configured_in_project` | Passed | The RAG system points to the project’s `PDF Files/` directory. |
| `test_pdf_directory_contains_source_documents` | Passed | PDF source documents are present. |
| `test_chunk_text_normalizes_and_overlaps_text` | Passed | Sample text is chunked into trimmed chunks. |
| `test_fallback_answer_handles_empty_context` | Passed | The fallback helper returns a response for empty context. |

**Duration:** 4.20 seconds for the latest complete run.

## JSON QA scenarios executed

The tests in `tests/test_json_qa_cases.py` load `test_cases.json`, send each question to the chatbot, and check the response against the case expectations.

| Case ID | Result |
|---|---|
| `supported_01` | Passed |
| `supported_paraphrase_01` | Passed |
| `supported_boundary_01` | Passed |
| `supported_multi_doc_01` | Passed |
| `ambiguous_01` | Passed |
| `ambiguous_scope_01` | Passed |
| `retrieval_failure_01` | Passed |
| `hallucination_01` | Passed |
| `unsupported_01` | Passed |
| `unsupported_temporal_01` | Passed |

All 10 JSON scenarios passed, along with the 4 structural/unit tests.

## Findings and changes

- The FAISS index tracks source PDF and OCR-helper signatures, preventing needless repeated rebuilds while refreshing when sources or OCR logic change.
- Added an optional macOS Vision OCR helper in [ocr_pdf.swift](ocr_pdf.swift) for PDFs that produce almost no text through normal extraction.
- The fallback now selects question-relevant evidence sentences, asks for clarification on the underspecified adoption query, corrects the Legal/Data Infrastructure attribution, and refuses unsupported valuation and future-roadmap claims.
- Explicit METR and Economics of AI Jobs comparisons now include evidence from both reports.

## Warnings

The pytest run reported three SWIG/FAISS deprecation warnings and one `urllib3` warning related to the Python build using LibreSSL. They did not cause test failures.

## Performance test

Command:

```bash
.venv/bin/python performance_test.py
```

Three representative questions were measured:

| Case ID | Scenario | Response latency |
|---|---|---:|
| `supported_boundary_01` | Supported METR factual question | 1.859 s |
| `ambiguous_scope_01` | Clarification-needed question | <0.001 s (printed as 0.000 s) |
| `unsupported_01` | Unsupported valuation question | 0.023 s |

| Metric | Result |
|---|---:|
| Minimum latency | 0.000 s |
| Average latency | 0.63 s |
| Median latency | 0.02 s |
| Maximum latency | 1.86 s |
| Failed requests | 0 of 3 |

The supported question was the slowest and may include initialization overhead on the first call.

## Stress test

Command:

```bash
.venv/bin/python stress_test.py
```

| Concurrent requests | Successful | Failed | Average response latency | Total wall time |
|---:|---:|---:|---:|---:|
| 1 | 1/1 | 0 | 2.01 s | 2.01 s |
| 3 | 3/3 | 0 | 0.04 s | 0.04 s |
| 5 | 5/5 | 0 | 0.07 s | 0.07 s |

**Observations:** All 9 requests completed. No request/API/rate-limit errors or crashes were printed, and the application remained responsive during this small run. Latency did not increase monotonically as concurrency increased; the concurrency-1 request likely included first-request initialization overhead. This is a small local check, not a production-scale capacity or rate-limit test.

## Not covered by the recorded run

- Higher concurrency, longer-duration load, and remote API rate limits were not tested.

## Conclusion

All 14 automated tests passed, including all 10 JSON-backed QA scenarios. The representative performance run and small concurrency stress run both recorded 0 failed requests. Four dependency warnings remain non-fatal.
