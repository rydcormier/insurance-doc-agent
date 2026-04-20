"""
scripts/ingest.py

Ingest one or more PDF documents into the vector store.

Usage:
    python scripts/ingest.py --file data/raw/sample_policy.pdf
    python scripts/ingest.py --dir data/raw/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.parser import DocumentParser
from embeddings.store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF documents into the vector store.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a single PDF file.")
    group.add_argument("--dir", type=str, help="Path to a directory containing PDF files.")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()
    
    doc_parser = DocumentParser(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    store = VectorStore()
    
    if args.file:
        print(f"Ingesting {args.file}...")
        chunks = doc_parser.parse(args.file)
    else:
        print(f"Ingesting directory {args.dir}...")
        chunks = doc_parser.parse_directory(args.dir)
        
    store.add_chunks(chunks)
    print("Done. Total chunks in store: {store.count}")
    
    
if __name__ == "__main__":
    main()
    