"""CLI entry point."""
import typer
from legalro_core.config import _load_dotenv

# Load .env before anything reads os.environ — covers LEGALRO_API_URL,
# LEGALRO_API_TOKEN, and other secrets used by the remote client checks.
_load_dotenv()

app = typer.Typer(name="legalro", help="Romanian Legal RAG System")

# PID file for the background MLX LLM server process
_PID_FILE = "/tmp/legalro_mlx.pid"


def _project_root():
    from pathlib import Path
    return Path(__file__).parent.parent.parent.parent


def _mlx_pid() -> int | None:
    from pathlib import Path
    p = Path(_PID_FILE)
    if not p.exists():
        return None
    try:
        return int(p.read_text().strip())
    except ValueError:
        return None


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
    """Start MongoDB and the LLM server (local MLX mode only)."""
    import subprocess
    import time
    import httpx
    from pathlib import Path
    from rich.console import Console
    from legalro_core.config import load_settings

    console = Console()
    settings = load_settings()

    if settings.llm.provider != "mlx":
        console.print("[yellow]Cloud mode — no local services to start.[/yellow]")
        console.print("Set MONGODB_URI, QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY env vars.")
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
    console.print("  [green]✓[/green] MongoDB running on port 27018")

    # ── LLM server ───────────────────────────────────────────────────────────
    existing_pid = _mlx_pid()
    if existing_pid and _process_alive(existing_pid):
        console.print(f"  [yellow]⚠[/yellow] LLM server already running (pid {existing_pid})")
    else:
        console.print(f"[bold]Starting LLM server[/bold] ({settings.llm.model})...")
        log_path = Path("/tmp/legalro_mlx.log")
        with open(log_path, "w") as log:
            proc = subprocess.Popen(
                ["mlx_lm.server", "--model", settings.llm.model, "--port", "8080",
                 "--max-tokens", str(settings.llm.max_tokens),
                 "--chat-template-args", '{"enable_thinking": true, "thinking_budget": 2048}'],
                stdout=log,
                stderr=log,
                start_new_session=True,  # detach from current terminal
            )
        Path(_PID_FILE).write_text(str(proc.pid))
        console.print(f"  pid {proc.pid} · log: {log_path}")

        # Wait up to 60 s for the server to become ready
        console.print("  Waiting for LLM server to be ready", end="")
        for _ in range(60):
            try:
                httpx.get(f"{settings.llm.base_url}/models", timeout=2)
                console.print("\n  [green]✓[/green] LLM server ready")
                break
            except Exception:
                console.print(".", end="", highlight=False)
                time.sleep(1)
        else:
            console.print("\n  [yellow]⚠[/yellow] LLM server not responding yet — check /tmp/legalro_mlx.log")

    console.print("\n[bold green]System is up.[/bold green] Run [bold]legalro status[/bold] to verify.")


@app.command()
def stop():
    """Stop the LLM server and MongoDB (local MLX mode only)."""
    import os
    import signal
    import subprocess
    from pathlib import Path
    from rich.console import Console
    from legalro_core.config import load_settings

    console = Console()
    settings = load_settings()

    if settings.llm.provider != "mlx":
        console.print("[yellow]Cloud mode — no local services to stop.[/yellow]")
        return

    root = _project_root()

    # ── LLM server ───────────────────────────────────────────────────────────
    pid = _mlx_pid()
    if pid and _process_alive(pid):
        console.print(f"[bold]Stopping LLM server[/bold] (pid {pid})...")
        try:
            os.kill(pid, signal.SIGTERM)
            console.print("  [green]✓[/green] LLM server stopped")
        except Exception as e:
            console.print(f"  [red]✗[/red] Could not stop LLM server: {e}")
    else:
        console.print("  LLM server was not running")
    Path(_PID_FILE).unlink(missing_ok=True)

    # ── MongoDB ──────────────────────────────────────────────────────────────
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

    if settings.llm.provider == "mlx":
        import httpx
        pid = _mlx_pid()
        if pid and _process_alive(pid):
            try:
                httpx.get(f"{settings.llm.base_url}/models", timeout=3)
                table.add_row("LLM", "✓ Running", f"{settings.llm.model}  (pid {pid})")
            except Exception:
                table.add_row("LLM", "⚠ Starting", f"pid {pid} alive but not responding yet")
        else:
            table.add_row("LLM", "✗ Offline", "Run: legalro start")
    else:
        table.add_row("LLM", "✓ Cloud", f"{settings.llm.model} via {settings.llm.provider}")

    console.print(table)
