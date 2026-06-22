"""
Shared Model Registry — Singleton cho tất cả AI models.

Tránh load model nhiều lần (BGE-M3 ~2.3GB fp32 / ~1.15GB fp16, Reranker ~1GB fp32 / ~0.5GB fp16).
Trên Q/P1000 4GB VRAM: fp16 tổng ~1.65GB → an toàn, fp32 ~3.3GB → OOM.
Device selection: GPU fp16 nếu đủ VRAM, fallback CPU fp32.
"""
import torch
from utils.console import console

_embed_model = None
_reranker_model = None


def _select_device(min_free_gb: float = 1.3) -> str:
    """
    Chọn device: CUDA nếu còn đủ VRAM, ngược lại CPU.
    min_free_gb: VRAM tối thiểu cần có trước khi load model tiếp theo.
    """
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / 1024 ** 3
        total_gb = total_bytes / 1024 ** 3
        if free_gb >= min_free_gb:
            console.print(
                f"[dim]  VRAM: {free_gb:.1f}/{total_gb:.1f}GB free → CUDA[/]"
            )
            return "cuda"
        console.print(
            f"[yellow]  VRAM thấp: {free_gb:.1f}/{total_gb:.1f}GB free "
            f"(cần {min_free_gb}GB) → CPU[/]"
        )
    return "cpu"


def get_embed_model():
    """Singleton embedding model — shared giữa router, query_engine, indexer."""
    global _embed_model
    if _embed_model is None:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))

        from sentence_transformers import SentenceTransformer
        from config import EMBEDDING_MODEL

        device = _select_device(min_free_gb=1.3)
        dtype = torch.float16 if device == "cuda" else torch.float32
        precision = "fp16" if dtype == torch.float16 else "fp32"

        console.print(
            f"[cyan]🔄 Loading embedding model: {EMBEDDING_MODEL} "
            f"({device}, {precision})...[/]"
        )
        _embed_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=device,
            model_kwargs={"torch_dtype": dtype},
        )
        console.print(
            f"[green]✅ Embedding model ready "
            f"({_embed_model.get_sentence_embedding_dimension()}D)[/]"
        )
    return _embed_model


def get_reranker_model():
    """Singleton reranker model (CrossEncoder)."""
    global _reranker_model
    if _reranker_model is None:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))

        from sentence_transformers import CrossEncoder
        from config import RERANKER_MODEL

        device = _select_device(min_free_gb=0.6)
        dtype = torch.float16 if device == "cuda" else torch.float32
        precision = "fp16" if dtype == torch.float16 else "fp32"

        console.print(
            f"[cyan]🔄 Loading reranker model: {RERANKER_MODEL} "
            f"({device}, {precision})...[/]"
        )
        _reranker_model = CrossEncoder(
            RERANKER_MODEL,
            max_length=512,
            device=device,
            automodel_args={"torch_dtype": dtype},
        )
        console.print("[green]✅ Reranker model ready[/]")
    return _reranker_model


def warmup():
    """Pre-load tất cả models lúc startup thay vì lúc query đầu tiên."""
    console.print("[bold cyan]🔥 Warming up models...[/]")
    get_embed_model()
    get_reranker_model()
    if torch.cuda.is_available():
        used_bytes = torch.cuda.memory_allocated()
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        console.print(
            f"[dim]  VRAM sau warmup: "
            f"{used_bytes/1024**3:.2f}/{total_bytes/1024**3:.1f}GB used[/]"
        )
    console.print("[bold green]✅ All models ready![/]")
