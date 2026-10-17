"""`tausft`: the commands the Makefile wraps. M0: smoke, split-check, leak-check."""

from __future__ import annotations

import tempfile
from pathlib import Path

import typer
from rich.console import Console

from tau_telecom_sft import config, leak, split

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console(soft_wrap=True)

FIXTURE = config.FIXTURES_DIR / "board_mini.json"


@app.command("split-check")
def split_check() -> None:
    """The split's sha256 matches the manifest and tau2-loop's split v3, and it is leak-free."""
    c = split.check()
    console.print(f"data/splits/telecom.json  sha256 {c.sha256}")
    console.print(f"manifest                  sha256 {c.recorded}")
    console.print(
        f"tau2-loop split v3        sha256 {c.sibling}"
        if c.sibling
        else "tau2-loop split v3        not checked out here; manifest only"
    )
    issues = split.problems(split.current())
    for p in issues:
        console.print(f"[red]problem[/red] {p}")
    if not c.ok or issues:
        raise typer.Exit(1)
    s = split.current()
    console.print(f"[green]ok[/green] split v{s.version}: {len(s.train)} train, {len(s.test)} test")


@app.command("leak-check")
def leak_check(paths: list[Path] = typer.Argument(None)) -> None:  # noqa: B008 - typer idiom
    """No test task, and no test task's twin, in anything the trainer reads."""
    roots = paths or list(config.TRAINER_INPUTS)
    leaks = leak.scan(roots)
    for lk in leaks:
        console.print(f"[red]leak[/red] {lk}")
    if leaks:
        raise typer.Exit(1)
    shown = ", ".join(
        str(r.relative_to(config.ROOT)) if r.is_relative_to(config.ROOT) else str(r) for r in roots
    )
    console.print(
        f"[green]ok[/green] {leak.scanned_files(roots)} file(s) under {shown}: no test task or twin"
    )


@app.command()
def smoke(no_mlflow: bool = typer.Option(False, help="build and check only; log nothing")) -> None:
    """Zero-cost: build a dataset from the fixture conversations, check it, log it to MLflow."""
    from tau_telecom_sft.build.dataset import Source, build

    with tempfile.TemporaryDirectory(prefix="tausft-smoke-") as tmp:
        built = build([Source("fixture", "tests/fixtures/board_mini", FIXTURE)], Path(tmp), "smoke")
        st = built.stats
        counted = sum(f["dropped"] for f in st["filters"]) + st["n_kept"]
        if counted != st["n_loaded"]:
            console.print(f"[red]counts do not add up[/red]: {counted} != {st['n_loaded']}")
            raise typer.Exit(1)
        leaks = leak.scan([Path(tmp)])
        if leaks:
            for lk in leaks:
                console.print(f"[red]leak[/red] {lk}")
            raise typer.Exit(1)
        for f in st["filters"]:
            console.print(
                f"  {f['rule']:<4} in {f['in']:>3}  dropped {f['dropped']:>2}  {f['keep_if']}"
            )
        console.print(
            f"kept {st['n_kept']} of {st['n_loaded']}; train.jsonl sha256 {st['train_sha256'][:16]}…"
        )
        if no_mlflow:
            return
        from tau_telecom_sft import track

        rid = track.log_build(built, kind="smoke", log_rows=True)
        console.print(f"run_id {rid}  ({config.EXPERIMENT_DATA})")


if __name__ == "__main__":
    app()
