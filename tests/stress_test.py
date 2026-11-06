import argparse
import time

from rag_chatbot import chunk_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exercise chunking repeatedly.")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    sample = "Stress test document text. " * 1_000
    started = time.perf_counter()
    total_chunks = sum(len(chunk_text(sample)) for _ in range(args.iterations))
    elapsed = time.perf_counter() - started
    print(f"Processed {args.iterations} iterations and {total_chunks} chunks in {elapsed:.4f}s")
