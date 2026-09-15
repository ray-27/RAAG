def chunk_text(text: str, chunk_size: int | None = None, overlap: int = 0) -> list[str]:
    text = (text or "").strip()
    if not text:
        return [""]
    if not chunk_size or chunk_size <= 0 or len(text) <= chunk_size:
        return [text]

    overlap = max(0, min(overlap, chunk_size - 1))
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        if 0 < next_start < n:
            snapped = next_start
            while snapped > 0 and not text[snapped - 1].isspace():
                snapped -= 1
            next_start = snapped if snapped > start else next_start
        start = next_start
    return chunks or [text]


def chunk_document(
    title: str = "",
    text: str = "",
    chunk_size: int | None = None,
    overlap: int = 0,
) -> list[str]:
    full = f"{title or ''} {text or ''}".strip()
    return chunk_text(full, chunk_size=chunk_size, overlap=overlap)


def iter_corpus_chunks(
    corpus: dict[str, dict[str, str]],
    chunk_size: int | None = None,
    overlap: int = 0,
):
    for doc_id, doc in corpus.items():
        for chunk in chunk_document(
            doc.get("title", ""),
            doc.get("text", ""),
            chunk_size=chunk_size,
            overlap=overlap,
        ):
            yield doc_id, chunk
