"""Day-1 validation: verify all components work before building app."""
import sys


def check_mongodb():
    from pymongo import MongoClient
    client = MongoClient("mongodb://localhost:27018/?directConnection=true", serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    print("✓ MongoDB connected")


def check_mlx_embeddings():
    from mlx_embedding_models.embedding import EmbeddingModel
    model = EmbeddingModel.from_registry("nomic-text-v1.5")
    result = model.encode(["test embedding"])
    assert result.shape == (1, 768), f"Expected (1, 768), got {result.shape}"
    print("✓ MLX embeddings working (768-dim)")


def check_ocrmac():
    from ocrmac import ocrmac  # noqa: F401
    print("✓ ocrmac importable")


def check_pymupdf():
    import fitz
    print(f"✓ PyMuPDF {fitz.version[0]}")


def check_mlx_llm():
    import httpx
    try:
        r = httpx.get("http://localhost:8080/v1/models", timeout=5)
        r.raise_for_status()
        print("✓ MLX LLM server running")
    except Exception:
        print("⚠ MLX LLM server not running (start with: mlx_lm.server --model mlx-community/Qwen3-14B-4bit --port 8080)")


if __name__ == "__main__":
    checks = [check_mongodb, check_mlx_embeddings, check_ocrmac, check_pymupdf, check_mlx_llm]
    failures = 0
    for check in checks:
        try:
            check()
        except Exception as e:
            print(f"✗ {check.__name__}: {e}")
            failures += 1
    sys.exit(failures)
