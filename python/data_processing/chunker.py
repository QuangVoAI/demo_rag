"""
Phân mảnh văn bản ArXiv papers thành các chunks phù hợp cho embedding.
Sử dụng Semantic Chunking: split theo section boundaries của paper.

Tính năng:
- Nhận diện cấu trúc paper: Abstract, Introduction, Method, Results, Conclusion
- Semantic boundaries (NOT fixed-size chunking)
- Context Injection: ghép [Paper Title (Year)] Section: vào mỗi chunk
- Token-based overlap cho sub-sections quá dài
"""
import re
import tiktoken
from dataclasses import dataclass, field
from rich.console import Console

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

console = Console()

# Tokenizer để đếm tokens chính xác
_ENCODER = tiktoken.get_encoding("cl100k_base")

# ─── Configuration ───────────────────────────────────────────────────
CHUNK_MAX_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 100


@dataclass
class TextChunk:
    """Một mảnh văn bản với metadata."""
    text: str
    doc_id: int
    chunk_id: int
    doc_title: str = ""
    level: int = 0  # 0 = leaf node (chunk gốc)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "doc_title": self.doc_title,
            "level": self.level,
            "metadata": self.metadata,
        }


def count_tokens(text: str) -> int:
    """Đếm số tokens trong text."""
    return len(_ENCODER.encode(text, disallowed_special=()))


def get_overlap_text(text: str, overlap_tokens: int) -> str:
    """Lấy phần cuối của text theo đúng số tokens."""
    tokens = _ENCODER.encode(text, disallowed_special=())
    if len(tokens) <= overlap_tokens:
        return text
    overlap_token_ids = tokens[-overlap_tokens:]
    return _ENCODER.decode(overlap_token_ids)


# ─── Section detection patterns ─────────────────────────────────────
SECTION_PATTERNS = [
    # Numbered sections: "1 Introduction", "2. Related Work", "3.1 Method"
    re.compile(r'^\s*(\d+\.?\d*\.?)\s+([A-Z][^\n]{2,80})\s*$', re.MULTILINE),
    # Unnumbered major sections
    re.compile(
        r'^\s*(Abstract|Introduction|Related Work|Background|'
        r'Methodology|Method|Methods|Approach|Model|Architecture|'
        r'Experiment|Experiments|Results|Evaluation|'
        r'Discussion|Analysis|Ablation|'
        r'Conclusion|Conclusions|Future Work|Acknowledgment|Acknowledgments)\s*$',
        re.MULTILINE | re.IGNORECASE
    ),
]


def detect_sections(text: str) -> list[dict]:
    """
    Chia paper thành các sections dựa trên headings.

    Returns:
        List[dict] với keys: 'text', 'section_name'
    """
    # Tìm tất cả section headings
    breaks = []
    for pattern in SECTION_PATTERNS:
        for match in pattern.finditer(text):
            heading = match.group(0).strip()
            breaks.append({
                "pos": match.start(),
                "heading": heading,
            })

    # Sort by position
    breaks.sort(key=lambda x: x["pos"])

    if not breaks:
        # Không tìm thấy sections → trả về nguyên paper
        return [{"text": text.strip(), "section_name": "Full Paper"}]

    sections = []

    # Content trước section đầu tiên (thường là metadata/abstract inline)
    if breaks[0]["pos"] > 100:
        preamble = text[:breaks[0]["pos"]].strip()
        if preamble:
            sections.append({"text": preamble, "section_name": "Preamble"})

    # Các sections
    for i, brk in enumerate(breaks):
        start = brk["pos"]
        end = breaks[i + 1]["pos"] if i + 1 < len(breaks) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append({
                "text": section_text,
                "section_name": brk["heading"],
            })

    return sections


def chunk_section(section_text: str, max_tokens: int = CHUNK_MAX_TOKENS) -> list[str]:
    """
    Chia 1 section thành chunks nếu quá dài.
    Ưu tiên chia theo paragraph boundaries.
    """
    if count_tokens(section_text) <= max_tokens:
        return [section_text]

    # Chia theo paragraphs
    paragraphs = re.split(r'\n\s*\n', section_text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if len(paragraphs) <= 1:
        # Chỉ 1 paragraph dài → chia theo sentences
        return chunk_by_sentences(section_text, max_tokens)

    chunks = []
    current = ""

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if para_tokens > max_tokens:
            # Paragraph quá dài → chia theo sentences trước
            if current:
                chunks.append(current.strip())
                current = ""
            sub_chunks = chunk_by_sentences(para, max_tokens)
            chunks.extend(sub_chunks)
            continue

        combined_tokens = count_tokens(current + "\n\n" + para) if current else para_tokens

        if combined_tokens <= max_tokens:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current.strip())
            # Overlap
            overlap = get_overlap_text(current, CHUNK_OVERLAP_TOKENS) if current else ""
            current = (overlap + "\n\n" + para).strip() if overlap else para

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_by_sentences(text: str, max_tokens: int = CHUNK_MAX_TOKENS) -> list[str]:
    """Chia text theo câu khi paragraph quá dài."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for sent in sentences:
        if count_tokens(current + " " + sent) <= max_tokens:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            if count_tokens(sent) > max_tokens:
                # Câu đơn quá dài → cắt theo tokens
                tokens = _ENCODER.encode(sent, disallowed_special=())
                current = _ENCODER.decode(tokens[:max_tokens])
            else:
                overlap = get_overlap_text(current, CHUNK_OVERLAP_TOKENS) if current else ""
                current = (overlap + " " + sent).strip() if overlap else sent

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_documents(documents: list[dict]) -> list[TextChunk]:
    """
    Chunking toàn bộ danh sách papers.

    Args:
        documents: List[dict] với 'text', 'title', 'id', 'authors', 'year', 'arxiv_id'

    Returns:
        List[TextChunk]
    """
    all_chunks = []
    chunk_counter = 0

    for doc in documents:
        text = doc.get("text", "")
        doc_id = doc.get("id", 0)
        doc_title = doc.get("title", "")
        authors = doc.get("authors", "")
        year = doc.get("year", "")
        arxiv_id = doc.get("arxiv_id", "")

        if not text:
            continue

        # Step 1: Detect sections
        sections = detect_sections(text)

        # Step 2: Chunk each section
        for section in sections:
            section_name = section["section_name"]
            section_text = section["text"]
            sub_chunks = chunk_section(section_text)

            for chunk_text in sub_chunks:
                # Context Injection: [Paper Title (Year)] Section:
                context_prefix = f"[{doc_title} ({year})] {section_name}:"
                contextualized_text = f"{context_prefix}\n{chunk_text}"

                chunk = TextChunk(
                    text=contextualized_text,
                    doc_id=doc_id,
                    chunk_id=chunk_counter,
                    doc_title=doc_title,
                    level=0,
                    metadata={
                        "tokens": count_tokens(contextualized_text),
                        "section": section_name,
                        "authors": authors,
                        "year": year,
                        "arxiv_id": arxiv_id,
                    }
                )
                all_chunks.append(chunk)
                chunk_counter += 1

    console.print(
        f"[green]✂️ Đã tạo {len(all_chunks)} chunks từ {len(documents)} papers[/]"
    )

    # Thống kê
    token_counts = [c.metadata["tokens"] for c in all_chunks]
    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
        console.print(
            f"[dim]Tokens/chunk: min={min(token_counts)}, "
            f"max={max(token_counts)}, avg={avg_tokens:.0f}[/]"
        )

    return all_chunks


if __name__ == "__main__":
    sample_doc = {
        "id": 0,
        "title": "Attention Is All You Need",
        "authors": "Vaswani et al.",
        "year": 2017,
        "arxiv_id": "1706.03762",
        "text": """
Abstract

We propose a new simple network architecture, the Transformer,
based solely on attention mechanisms.

1 Introduction

The dominant sequence transduction models are based on complex
recurrent or convolutional neural networks that include an encoder
and a decoder. The best performing models also connect the encoder
and decoder through an attention mechanism.

2 Background

Neural sequence transduction models have a long history.

3 Model Architecture

Most competitive neural sequence transduction models have an
encoder-decoder structure. The encoder maps an input sequence
to a sequence of continuous representations.

3.1 Multi-Head Attention

Multi-head attention allows the model to jointly attend to
information from different representation subspaces at different
positions. With a single attention head, averaging inhibits this.

4 Experiments

We trained on the WMT 2014 English-German dataset.

5 Conclusion

In this work, we presented the Transformer, the first sequence
transduction model based entirely on attention.
"""
    }

    chunks = chunk_documents([sample_doc])
    for c in chunks:
        console.print(f"\n[bold]Chunk {c.chunk_id}[/] ({c.metadata['tokens']} tokens):")
        console.print(f"[dim]{c.text[:200]}...[/]")
