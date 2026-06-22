"""
Policy Loader — Load & index chính sách MyKingdom vào Qdrant.

Đọc trực tiếp từ mykingdom_policies.json (dữ liệu thật từ website MyKingdom).
Chunking theo TỪNG SECTION để retrieval chính xác hơn.

Chiến lược:
  policies[i].sections[j] → 1 chunk
  Mỗi chunk chứa: [policy_title] > [section_heading] + content + contact info
"""
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console
from config import DATA_DIR
from data_processing.chunker import TextChunk

console = Console()

# Contact info sẽ được inject vào cuối mỗi chunk liên quan
CONTACT_FOOTER = ""


def load_mykingdom_policies(path=None):
    """Load policies từ mykingdom_policies.json."""
    if path is None:
        path = DATA_DIR / "mykingdom_policies.json"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {path}\n"
            f"Hãy đảm bảo file mykingdom_policies.json nằm trong thư mục data/"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    policies = data.get("policies", [])

    # Extract contact info
    contact = metadata.get("contact", {})
    global CONTACT_FOOTER
    CONTACT_FOOTER = (
        f"\n---\n"
        f"Liên hệ hỗ trợ: Hotline {contact.get('hotline', '1900 1208')} | "
        f"Email {contact.get('email', 'hotro@mykingdom.com.vn')} | "
        f"Giờ làm việc: {contact.get('working_hours', '')}"
    )

    console.print(f"[green]✅ Loaded {len(policies)} policies from {path.name}[/]")
    console.print(f"[dim]   Brand: {metadata.get('brand', 'N/A')} | "
                  f"Company: {metadata.get('company', 'N/A')}[/]")

    return data


def chunk_policies_by_section(data):
    """
    Chunk policies theo TỪNG SECTION (thay vì toàn bộ policy).
    
    Mỗi section → 1 chunk riêng biệt, chứa:
    - Tiêu đề policy cha
    - Heading của section
    - Nội dung section
    - Metadata: policy_id, keywords, url
    - Contact footer
    """
    policies = data.get("policies", [])
    metadata = data.get("metadata", {})
    chunks = []
    chunk_idx = 0

    for policy_idx, policy in enumerate(policies):
        policy_id = policy.get("id", f"policy_{policy_idx}")
        policy_title = policy.get("title", "")
        policy_url = policy.get("url", "")
        policy_summary = policy.get("summary", "")
        policy_keywords = policy.get("keywords", [])
        sections = policy.get("sections", [])

        if not sections:
            # Policy không có sections → dùng summary làm chunk
            text = (
                f"📋 {policy_title}\n"
                f"Tóm tắt: {policy_summary}\n"
                f"Từ khóa: {', '.join(policy_keywords)}"
                f"{CONTACT_FOOTER}"
            )
            chunk = TextChunk(
                text=text,
                doc_id=policy_idx,
                chunk_id=chunk_idx,
                doc_title=policy_title,
                level=0,
                metadata={
                    "policy_id": policy_id,
                    "keywords": policy_keywords,
                    "url": policy_url,
                    "section_heading": "Tổng quan",
                    "brand": metadata.get("brand", "MyKingdom"),
                },
            )
            chunks.append(chunk)
            chunk_idx += 1
            continue

        for section in sections:
            heading = section.get("heading", "")
            content = section.get("content", "")

            # Build chunk text: policy context + section detail + contact
            text = (
                f"📋 {policy_title} > {heading}\n\n"
                f"{content}"
                f"{CONTACT_FOOTER}"
            )

            # Derive category from keywords
            category = _derive_category(policy_keywords)

            chunk = TextChunk(
                text=text,
                doc_id=policy_idx,
                chunk_id=chunk_idx,
                doc_title=f"{policy_title} — {heading}",
                level=0,
                metadata={
                    "policy_id": policy_id,
                    "category": category,
                    "keywords": policy_keywords,
                    "url": policy_url,
                    "section_heading": heading,
                    "brand": metadata.get("brand", "MyKingdom"),
                },
            )
            chunks.append(chunk)
            chunk_idx += 1

    console.print(f"[green]✅ Created {len(chunks)} section-level chunks "
                  f"from {len(policies)} policies[/]")
    return chunks


def _derive_category(keywords):
    """Derive category từ keywords của policy."""
    kw_lower = [k.lower() for k in keywords]
    kw_text = " ".join(kw_lower)

    if any(k in kw_text for k in ["bảo hành", "đổi trả", "hoàn tiền", "trả hàng"]):
        return "exchange_refund"
    elif any(k in kw_text for k in ["thành viên", "mypoints", "tích điểm", "voucher"]):
        return "loyalty"
    elif any(k in kw_text for k in ["giao hàng", "ship", "vận chuyển", "freeship"]):
        return "shipping"
    elif any(k in kw_text for k in ["đóng gói", "kiểm hàng", "unbox", "khiếu nại"]):
        return "packaging"
    elif any(k in kw_text for k in ["thanh toán", "cod", "visa", "zalopay"]):
        return "payment"
    elif any(k in kw_text for k in ["bảo mật", "dữ liệu", "quyền riêng tư"]):
        return "privacy"
    elif any(k in kw_text for k in ["cửa hàng", "store", "chi nhánh"]):
        return "store_info"
    else:
        return "general"


def index_policies(recreate=True):
    """Full pipeline: Load → Section-level Chunk → Embed → Index vào Qdrant."""
    import numpy as np
    from retrieval.qdrant_client import QdrantWrapper
    from agents.model_registry import get_embed_model

    # 1. Load MyKingdom policies
    data = load_mykingdom_policies()

    # 2. Section-level chunking
    chunks = chunk_policies_by_section(data)

    # 3. Connect to Qdrant
    qdrant = QdrantWrapper()
    qdrant.create_collection(recreate=recreate)

    # 4. Embed & Index
    model = get_embed_model()
    texts = [c.text for c in chunks]

    console.print(f"[cyan]🔄 Embedding {len(texts)} policy section chunks...[/]")
    embeddings = model.encode(
        texts, normalize_embeddings=True,
        show_progress_bar=True, batch_size=32,
    )

    nodes = []
    for chunk in chunks:
        nodes.append({
            "text": chunk.text,
            "node_id": chunk.chunk_id,
            "level": 0,
            "doc_title": chunk.doc_title,
            "doc_id": chunk.doc_id,
            "metadata": chunk.metadata,
        })

    qdrant.upsert_nodes(
        nodes=nodes,
        embeddings=np.array(embeddings),
        batch_size=100,
    )

    console.print(f"[bold green]✅ Indexed {len(nodes)} section-level chunks into Qdrant[/]")

    # Print summary
    console.print("\n[bold]📊 Chunking Summary:[/]")
    categories = {}
    for chunk in chunks:
        cat = chunk.metadata.get("category", "general")
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        console.print(f"  [{cat}]: {count} chunks")

    return len(nodes)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true", help="Recreate collection")
    args = parser.parse_args()

    count = index_policies(recreate=args.recreate)
    console.print(f"\n[bold]Total indexed: {count} chunks[/]")
