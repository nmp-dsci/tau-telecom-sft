"""`tausft`: the commands the Makefile wraps. M0: smoke, split-check, leak-check. M1: download,
sources-check, build, review-sample."""

from __future__ import annotations

import json
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


@app.command()
def download() -> None:
    """Fetch the board's telecom files (S1, S2) into raw/board/, once, and verify their bytes."""
    from tau_telecom_sft.sources import board

    for bf in board.BOARD:
        path = board.download(bf)
        console.print(
            f"[green]ok[/green] {bf.id} {bf.file}  {bf.bytes:,} bytes  sha256 {bf.sha256[:16]}…"
        )
        console.print(f"   {path.relative_to(config.ROOT)}")


@app.command("sources-check")
def sources_check() -> None:
    """Load S1, S2 and one tau2-loop telecom run with the same loader; their shapes must match."""
    from tau_telecom_sft.sources import board, compare

    paths = [(f"{b.id} {b.label}", b.path) for b in board.BOARD]
    loop = compare.tau2_loop_results()
    if loop.exists():
        paths.append((f"S5 tau2-loop/{compare.TAU2_LOOP_RUN}", loop))
    else:
        console.print(f"[yellow]tau2-loop run not here[/yellow] ({loop}); checking the board only")
    profiles = [compare.profile(p, name) for name, p in paths]
    for p in profiles:
        console.print(
            f"  {p['source']:<48} {p['conversations']:>4} conversations  roles {p['roles']}  "
            f"tool results {p['tool_results_by_requestor']}  agent view removed {p['agent_view_removed']}"
        )
    for d in compare.differences(profiles):
        console.print(f"[yellow]differs[/yellow] {d} (added to tau2 after February; not needed)")
    issues = compare.problems(profiles)
    for i in issues:
        console.print(f"[red]problem[/red] {i}")
    if issues:
        raise typer.Exit(1)
    console.print(f"[green]ok[/green] {len(profiles)} files, one loader, one shape")


@app.command()
def build(
    version: str = typer.Option("v1", help="the dataset version folder under data/sft/"),
    max_tokens: int = typer.Option(config.MAX_TOKENS, help="F10's limit, in the student's tokens"),
    no_mlflow: bool = typer.Option(False, help="build and check only; log nothing"),
    force: bool = typer.Option(False, help="rebuild an existing version (it must not change)"),
) -> None:
    """Build a dataset version from the board files: agent view, F1 to F10, rows, card."""
    from tau_telecom_sft.build import review, tokens
    from tau_telecom_sft.build.dataset import Source
    from tau_telecom_sft.build.dataset import build as run_build
    from tau_telecom_sft.sources import board

    out = config.SFT_DIR / version
    if (out / "card.md").exists() and not force:
        console.print(f"[red]{out.relative_to(config.ROOT)} exists[/red]: a version never changes")
        raise typer.Exit(1)
    for bf in board.BOARD:
        if not board.verify(bf):
            console.print(f"[red]{bf.id} missing[/red]: run `make download`")
            raise typer.Exit(1)
    before = (
        json.loads((out / "stats.json").read_text())["train_sha256"]
        if force and (out / "stats.json").exists()
        else None
    )
    built = run_build(
        [Source(bf.id, bf.label, bf.path) for bf in board.BOARD],
        out,
        version,
        count_tokens=tokens.count,
        token_counter=tokens.label(),
        max_tokens=max_tokens,
        source_checks=[board.checks(bf) for bf in board.BOARD],
    )
    st = built.stats
    counted = sum(f["dropped"] for f in st["filters"]) + st["n_kept"]
    if counted != st["n_loaded"]:
        console.print(f"[red]counts do not add up[/red]: {counted} != {st['n_loaded']}")
        raise typer.Exit(1)
    if before and before != st["train_sha256"]:
        console.print(
            f"[red]rebuild changed train.jsonl[/red]: {before[:16]}… → {st['train_sha256'][:16]}…"
        )
        raise typer.Exit(1)
    leaks = leak.scan([out])
    for lk in leaks:
        console.print(f"[red]leak[/red] {lk}")
    if leaks:
        raise typer.Exit(1)
    for f in st["filters"]:
        console.print(
            f"  {f['rule']:<4} in {f['in']:>4}  dropped {f['dropped']:>4}  {f['keep_if']}"
        )
    t = st["tokens"]
    console.print(
        f"kept {st['n_kept']} of {st['n_loaded']}, {st['training_tasks_covered']}/{st['training_tasks_total']} "
        f"training tasks; tokens median {t['median']:,}, max {t['max']:,} (limit {st['max_tokens']:,})"
    )
    sample = review.draw(out)
    console.print(
        f"review sample: {len(sample['kept'])} kept + {len(sample['dropped'])} dropped, seed {sample['seed']}"
    )
    console.print(f"train.jsonl sha256 {st['train_sha256']}")
    if no_mlflow:
        return
    from tau_telecom_sft import track

    rid = track.log_build(built, kind="build", log_rows=True)
    console.print(f"run_id {rid}  ({config.EXPERIMENT_DATA})")


@app.command()
def train(
    config_file: Path = typer.Option(  # noqa: B008 - typer idiom
        Path("configs/smoke.yaml"), "--config", help="a configs/*.yaml"
    ),
    max_steps: int = typer.Option(
        None, help="stop after this many optimiser steps (a measured step)"
    ),
    no_mlflow: bool = typer.Option(False, help="write the run folder only"),
) -> None:
    """LoRA SFT in TRL with the loss on the agent's turns; checks the loss mask before step 1."""
    from tau_telecom_sft.train.sft import train as run_train

    run_dir = run_train(config_file, max_steps=max_steps, use_mlflow=not no_mlflow, console=console)
    console.print(f"run folder {run_dir.relative_to(config.ROOT)}")


@app.command()
def mask(
    config_file: Path = typer.Option(Path("configs/smoke.yaml"), "--config"),  # noqa: B008
    row: int = typer.Option(0, help="which selected row"),
) -> None:
    """Print one row's loss mask as the trainer will apply it, without training."""
    from transformers import AutoTokenizer
    from trl.chat_template_utils import get_training_chat_template

    from tau_telecom_sft.train import mask as mask_mod
    from tau_telecom_sft.train.sft import load_config, print_mask, select_rows

    cfg = load_config(config_file)
    rows, _ = select_rows(cfg)
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    ex = mask_mod.tokenize(tok, rows[row], get_training_chat_template(tok))
    segs = mask_mod.segments(tok, ex["input_ids"], ex["assistant_masks"])
    chk = mask_mod.check(rows[row], segs)
    print_mask(segs, chk, console.print)
    if not chk["ok"]:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
