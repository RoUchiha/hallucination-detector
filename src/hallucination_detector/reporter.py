"""Report generation — console (rich), JSON, and HTML."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from rich.console import Console
from rich.table import Table
from rich import box

from .models import ClaimVerdict, HallucinationReport

console = Console()

_VERDICT_COLOR = {
    "SUPPORTED": "green",
    "HALLUCINATED": "red",
    "UNCERTAIN": "yellow",
}

_TIER_COLOR = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}


def build_report(
    verdicts: List[ClaimVerdict],
    source_doc: str,
    generated_text: str,
) -> HallucinationReport:
    total = len(verdicts)
    hallucinated = sum(1 for v in verdicts if v.final_verdict == "HALLUCINATED")
    rate = hallucinated / total if total > 0 else 0.0
    tier = "LOW" if rate < 0.2 else "MEDIUM" if rate < 0.5 else "HIGH"

    return HallucinationReport(
        source_doc_hash=HallucinationReport.compute_hash(source_doc),
        generated_text_hash=HallucinationReport.compute_hash(generated_text),
        total_claims=total,
        hallucination_rate=round(rate, 4),
        risk_tier=tier,
        verdicts=verdicts,
        generated_at=datetime.now(timezone.utc),
    )


def render_console(report: HallucinationReport) -> None:
    tier_color = _TIER_COLOR[report.risk_tier]
    console.print(f"\n[bold]Hallucination Detection Report[/bold]  "
                  f"[{tier_color}]Risk: {report.risk_tier}[/{tier_color}]  "
                  f"Rate: [bold]{report.hallucination_rate:.1%}[/bold] "
                  f"({sum(1 for v in report.verdicts if v.final_verdict == 'HALLUCINATED')} / {report.total_claims} claims)\n")

    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Claim", ratio=4)
    table.add_column("NLI", justify="center", width=14)
    table.add_column("Conf", justify="right", width=6)
    table.add_column("LLM", justify="center", width=12)
    table.add_column("Final", justify="center", width=13)

    for v in report.verdicts:
        color = _VERDICT_COLOR[v.final_verdict]
        table.add_row(
            str(v.claim.id),
            v.claim.text[:100],
            v.nli_label,
            f"{v.nli_confidence:.2f}",
            v.llm_verdict or "—",
            f"[{color}]{v.final_verdict}[/{color}]",
        )
    console.print(table)


def save_json(report: HallucinationReport, path: str | Path) -> None:
    Path(path).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"\n[dim]Report saved → {path}[/dim]")


def save_html(report: HallucinationReport, path: str | Path) -> None:
    rows = ""
    for v in report.verdicts:
        color = {"SUPPORTED": "#22c55e", "HALLUCINATED": "#ef4444", "UNCERTAIN": "#eab308"}[v.final_verdict]
        rows += (
            f"<tr><td>{v.claim.id}</td><td>{v.claim.text}</td>"
            f"<td>{v.nli_label}</td><td>{v.nli_confidence:.2f}</td>"
            f"<td>{v.llm_verdict or '—'}</td>"
            f"<td style='color:{color};font-weight:bold'>{v.final_verdict}</td></tr>\n"
        )
    html = f"""<!DOCTYPE html><html><head><title>Hallucination Report</title>
<style>body{{font-family:sans-serif;padding:2rem}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ddd;padding:8px;text-align:left}}th{{background:#f5f5f5}}</style></head>
<body><h1>Hallucination Detection Report</h1>
<p>Risk Tier: <strong>{report.risk_tier}</strong> &bull; Rate: {report.hallucination_rate:.1%} &bull; Generated: {report.generated_at}</p>
<table><thead><tr><th>#</th><th>Claim</th><th>NLI</th><th>Conf</th><th>LLM</th><th>Final</th></tr></thead>
<tbody>{rows}</tbody></table></body></html>"""
    Path(path).write_text(html, encoding="utf-8")
    console.print(f"[dim]HTML report saved → {path}[/dim]")
