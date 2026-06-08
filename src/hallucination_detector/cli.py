"""CLI entrypoint for the hallucination detection pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="Detect hallucinations in LLM-generated text.")
console = Console()


@app.command()
def detect(
    source: Path = typer.Option(..., help="Path to the source document (ground truth)."),
    generated: Path = typer.Option(..., help="Path to the LLM-generated text to evaluate."),
    output: Optional[Path] = typer.Option(None, help="Path to save the JSON report."),
    html: Optional[Path] = typer.Option(None, help="Path to save an HTML report."),
    llm_judge: bool = typer.Option(False, "--llm-judge", help="Enable LLM escalation for uncertain claims."),
    nli_model: str = typer.Option("cross-encoder/nli-deberta-v3-small", help="HuggingFace NLI model name."),
) -> None:
    """Run the hallucination detection pipeline end-to-end."""
    from .extractor import ClaimExtractor
    from .nli_scorer import NLIScorer
    from .llm_judge import LLMJudge
    from .reporter import build_report, render_console, save_json, save_html

    source_text = source.read_text(encoding="utf-8")
    generated_text = generated.read_text(encoding="utf-8")

    console.print("[bold cyan]Step 1/3:[/bold cyan] Extracting claims…")
    extractor = ClaimExtractor()
    claims = extractor.extract(generated_text)
    console.print(f"  → {len(claims)} claims extracted")

    console.print("[bold cyan]Step 2/3:[/bold cyan] Running NLI scoring…")
    scorer = NLIScorer(model_name=nli_model)
    verdicts = scorer.score_all(claims, source_text)

    if llm_judge:
        console.print("[bold cyan]Step 2b:[/bold cyan] Escalating to LLM judge for uncertain claims…")
        judge = LLMJudge()
        verdicts = judge.escalate(verdicts, source_text)

    console.print("[bold cyan]Step 3/3:[/bold cyan] Building report…")
    report = build_report(verdicts, source_text, generated_text)
    render_console(report)

    if output:
        save_json(report, output)
    if html:
        save_html(report, html)


if __name__ == "__main__":
    app()
