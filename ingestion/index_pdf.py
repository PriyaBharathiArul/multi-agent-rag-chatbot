"""
Ingestion pipeline: loads a PDF, chunks the text, embeds with OpenAI,
and upserts into a Pinecone index.

Usage:
    cd multi-agent-rag-chatbot
    python -m ingestion.index_pdf --pdf path/to/machine-learning.pdf [--chunk-size 500]
"""

import argparse
import os
import string

import pandas as pd
import tiktoken
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

INDEX_NAME = "mp2-machine-learning-pdf"
EMBEDDING_MODEL = "text-embedding-3-small"

enc = tiktoken.get_encoding("cl100k_base")


def num_tokens(text: str) -> int:
    return len(enc.encode(text))


def get_embedding(client: OpenAI, text: str) -> list[float]:
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=EMBEDDING_MODEL).data[0].embedding


def load_pdf(pdf_path: str) -> tuple[list[str], list[int]]:
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    page_texts = [doc.page_content for doc in docs]
    page_numbers = [doc.metadata["page"] for doc in docs]
    print(f"Loaded {len(page_texts)} pages from {pdf_path}")
    return page_texts, page_numbers


def chunk_and_embed(
    client: OpenAI,
    page_texts: list[str],
    page_numbers: list[int],
    chunk_size: int = 500,
    overlap: int = 50,
) -> pd.DataFrame:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunked_texts = []
    chunk_page_nums = []
    previous_page_tail = ""

    for page_num, page_text in zip(page_numbers, page_texts):
        combined_text = previous_page_tail + page_text
        chunks = splitter.split_text(combined_text)

        for ch in chunks:
            chunked_texts.append(ch)
            chunk_page_nums.append(page_num)

        previous_page_tail = combined_text[-overlap:] if len(combined_text) >= overlap else combined_text

    df = pd.DataFrame({"text": chunked_texts, "page_number": chunk_page_nums})
    df["text"] = df["text"].str.replace("\n", " ", regex=False)
    df["text"] = df["text"].apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))

    embeds = []
    for text in tqdm(df["text"], desc="Embedding chunks"):
        embeds.append(get_embedding(client, text))
    df["embeddings"] = embeds

    print(f"Created {len(df)} chunks (chunk_size={chunk_size})")
    return df


def upsert_to_pinecone(
    index,
    df: pd.DataFrame,
    namespace: str,
    chunk_size: int,
    batch_size: int = 100,
):
    batch = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Upserting to {namespace}"):
        batch.append({
            "id": f"{namespace}_ml_{i}",
            "values": row["embeddings"],
            "metadata": {
                "text": row["text"],
                "page_number": int(row["page_number"]),
                "num_tokens": num_tokens(row["text"]),
                "chunk_size": chunk_size,
            },
        })
        if len(batch) == batch_size:
            index.upsert(vectors=batch, namespace=namespace)
            batch = []

    if batch:
        index.upsert(vectors=batch, namespace=namespace)

    stats = index.describe_index_stats()
    count = stats["namespaces"].get(namespace, {}).get("vector_count", 0)
    print(f"Namespace {namespace} vector count: {count}")


def main():
    parser = argparse.ArgumentParser(description="Index a PDF into Pinecone")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--chunk-size", type=int, default=500, help="Text chunk size (default: 500)")
    parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap (default: 50)")
    args = parser.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not openai_key or not pinecone_key:
        raise RuntimeError("Set OPENAI_API_KEY and PINECONE_API_KEY environment variables.")

    client = OpenAI(api_key=openai_key)

    pc = Pinecone(api_key=pinecone_key)
    existing = [x["name"] for x in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Created Pinecone index: {INDEX_NAME}")

    index = pc.Index(INDEX_NAME)

    page_texts, page_numbers = load_pdf(args.pdf)

    namespace = f"ns{args.chunk_size}"
    df = chunk_and_embed(client, page_texts, page_numbers, args.chunk_size, args.overlap)
    upsert_to_pinecone(index, df, namespace, args.chunk_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
