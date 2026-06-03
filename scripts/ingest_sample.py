from __future__ import annotations

import argparse

from api.main import SERVICES


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a sample document into ResearchGraph-RAG")
    parser.add_argument("--pdf", type=str, default=None)
    parser.add_argument("--docx", type=str, default=None)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--arxiv", type=str, default=None)
    args = parser.parse_args()

    SERVICES.initialize()
    assert SERVICES.pipeline is not None

    if args.pdf:
        with open(args.pdf, "rb") as f:
            result = SERVICES.pipeline.ingest_pdf(file_bytes=f.read(), source_name=args.pdf)
    elif args.docx:
        with open(args.docx, "rb") as f:
            result = SERVICES.pipeline.ingest_docx(file_bytes=f.read(), source_name=args.docx)
    elif args.url:
        result = SERVICES.pipeline.ingest_url(args.url)
    elif args.arxiv:
        result = SERVICES.pipeline.ingest_arxiv(args.arxiv)
    else:
        raise ValueError("Provide one source: --pdf/--docx/--url/--arxiv")

    print({"source_doc": result.source_doc, "chunks_created": result.chunks_created})


if __name__ == "__main__":
    main()
