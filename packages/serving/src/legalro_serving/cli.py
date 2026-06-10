"""CLI entry point."""
import typer
from legalro_core.config import _load_dotenv

# Load .env before anything reads os.environ — covers LEGALRO_API_URL,
# LEGALRO_API_TOKEN, and other secrets used by the remote client checks.
_load_dotenv()

app = typer.Typer(name="legalro", help="Romanian Legal RAG System")

def _project_root():
    from pathlib import Path
    return Path(__file__).parent.parent.parent.parent


def _process_alive(pid: int) -> bool:
    import os
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@app.command()
def start():
    """Start MongoDB and Ollama (local mode only)."""
    import subprocess
    import time
    import httpx
    from rich.console import Console
    from legalro_core.config import load_settings

    console = Console()
    settings = load_settings()

    if settings.llm.provider not in ("ollama",):
        console.print("[yellow]Cloud mode — no local services to start.[/yellow]")
        console.print("Set MONGODB_URI and GEMINI_API_KEY env vars.")
        return

    root = _project_root()

    # ── MongoDB ──────────────────────────────────────────────────────────────
    console.print("[bold]Starting MongoDB...[/bold]")
    r = subprocess.run(
        ["docker", "compose", "up", "-d"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        console.print(f"[red]MongoDB failed:[/red] {r.stderr.strip()}")
        raise typer.Exit(1)
    console.print("  [green]✓[/green] MongoDB running")

    # ── Ollama ───────────────────────────────────────────────────────────────
    # Check if already running
    try:
        httpx.get("http://localhost:11434/api/tags", timeout=2)
        console.print(f"  [yellow]⚠[/yellow] Ollama already running — using model [bold]{settings.llm.model}[/bold]")
    except Exception:
        console.print("[bold]Starting Ollama...[/bold]")
        subprocess.Popen(
            ["ollama", "serve"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        console.print("  Waiting for Ollama to be ready", end="")
        for _ in range(30):
            try:
                httpx.get("http://localhost:11434/api/tags", timeout=2)
                console.print("\n  [green]✓[/green] Ollama ready")
                break
            except Exception:
                console.print(".", end="", highlight=False)
                time.sleep(1)
        else:
            console.print("\n  [yellow]⚠[/yellow] Ollama not responding — run [bold]ollama serve[/bold] manually")

    console.print("\n[bold green]System is up.[/bold green] Run [bold]legalro status[/bold] to verify.")


@app.command()
def stop():
    """Stop MongoDB (local mode only). Ollama manages its own lifecycle."""
    import subprocess
    from rich.console import Console
    from legalro_core.config import load_settings

    console = Console()
    settings = load_settings()

    if settings.llm.provider not in ("ollama",):
        console.print("[yellow]Cloud mode — no local services to stop.[/yellow]")
        return

    root = _project_root()

    console.print("[bold]Stopping MongoDB...[/bold]")
    r = subprocess.run(
        ["docker", "compose", "stop"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        console.print("  [green]✓[/green] MongoDB stopped")
    else:
        console.print(f"  [red]✗[/red] {r.stderr.strip()}")

    console.print("  Ollama continues running in the background (stop manually with [bold]pkill ollama[/bold] if needed)")
    console.print("\n[bold]System is down.[/bold]")


@app.command()
def query(
    question: str,
    act_type: str = "",
    year: int = 0,
    no_agentic: bool = False,
    local: bool = typer.Option(False, "--local", help="Run in-process instead of calling the remote API"),
):
    """Ask a single question (remote by default if LEGALRO_API_URL is set)."""
    import os
    if not local and os.environ.get("LEGALRO_API_URL"):
        from legalro_serving.client import query as remote_query
        typer.echo(remote_query(question, act_type))
        return

    from legalro_core.config import load_settings
    from legalro_serving.generation import run_query_hybrid, run_query
    settings = load_settings()
    if no_agentic:
        typer.echo(run_query(question, settings))
    else:
        typer.echo(run_query_hybrid(question, settings))


@app.command()
def ingest(
    path: str,
    batch: bool = False,
    local: bool = typer.Option(False, "--local", help="Run in-process instead of calling the remote API"),
):
    """Ingest PDFs from a directory or single file (remote by default if LEGALRO_API_URL is set)."""
    import os
    from pathlib import Path

    pdf_path = Path(path)
    pdfs = [pdf_path] if pdf_path.is_file() else sorted(pdf_path.rglob("*.pdf"))

    if not local and os.environ.get("LEGALRO_API_URL"):
        from legalro_serving.client import ingest as remote_ingest
        for pdf in pdfs:
            typer.echo(f"  Uploading {pdf.name}…")
            job = remote_ingest(str(pdf))
            status = "✓" if job["status"] == "done" else "✗"
            typer.echo(f"  {status} {pdf.name}: {job.get('chunks_created', 0)} chunks — {job.get('detail', '')}")
        return

    from rich.progress import Progress
    from legalro_core.config import load_settings
    from legalro_processing.pipeline import process_gazette

    settings = load_settings()
    with Progress() as progress:
        task = progress.add_task("Ingesting...", total=len(pdfs))
        for pdf in pdfs:
            result = process_gazette(str(pdf), settings)
            status = "✓" if result.status == "completed" else "⚠"
            progress.console.print(
                f"  {status} {pdf.name}: {result.acts_segmented} acts, {result.chunks_created} chunks"
            )
            if result.warnings:
                for w in result.warnings:
                    progress.console.print(f"    ⚠ {w}", style="yellow")
            progress.advance(task)


@app.command()
def chat(
    local: bool = typer.Option(False, "--local", help="Run in-process instead of calling the remote API"),
):
    """Interactive chat mode (remote by default if LEGALRO_API_URL is set)."""
    import os
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()
    console.print("[bold]LegalRo Chat[/bold] (type 'exit' to quit)\n")

    if not local and os.environ.get("LEGALRO_API_URL"):
        from legalro_serving.client import query as remote_query
        while True:
            question = console.input("[bold green]> [/bold green]")
            if question.lower() in ("exit", "quit", "q"):
                break
            answer = remote_query(question)
            console.print(Markdown(answer))
            console.print()
        return

    from legalro_core.config import load_settings
    from legalro_serving.generation import create_agent
    settings = load_settings()
    agent = create_agent(settings)

    while True:
        question = console.input("[bold green]> [/bold green]")
        if question.lower() in ("exit", "quit", "q"):
            break
        result = agent.run_sync(question)
        console.print(Markdown(result.output))
        console.print()


@app.command()
def status():
    """Show system status."""
    from rich.table import Table
    from rich.console import Console
    from legalro_core.config import load_settings
    from legalro_core.store import get_db

    settings = load_settings()
    console = Console()
    table = Table(title="LegalRo Status")
    table.add_column("Component")
    table.add_column("Status")
    table.add_column("Details")

    try:
        db = get_db(settings)
        gazette_count = db.gazettes.count_documents({})
        act_count = db.acts.count_documents({})
        chunk_count = db.chunks.count_documents({})
        table.add_row("MongoDB", "✓ Connected", f"{gazette_count} gazettes, {act_count} acts, {chunk_count} chunks")
    except Exception as e:
        table.add_row("MongoDB", "✗ Error", str(e))

    if settings.llm.provider == "ollama":
        import httpx
        try:
            resp = httpx.get("http://localhost:11434/api/tags", timeout=3)
            loaded = [m["name"] for m in resp.json().get("models", [])]
            model_ok = any(settings.llm.model in m for m in loaded)
            if model_ok:
                table.add_row("LLM (Ollama)", "✓ Running", settings.llm.model)
            else:
                pulled = ", ".join(loaded) or "none"
                table.add_row("LLM (Ollama)", "⚠ Model missing",
                              f"{settings.llm.model} not pulled — run: ollama pull {settings.llm.model}\n(pulled: {pulled})")
        except Exception:
            table.add_row("LLM (Ollama)", "✗ Offline", "Run: ollama serve")
    else:
        table.add_row("LLM", "✓ Cloud", f"{settings.llm.model} via {settings.llm.provider}")

    console.print(table)
