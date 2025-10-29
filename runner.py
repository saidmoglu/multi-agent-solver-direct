from __future__ import annotations

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TaskID,
)
from datetime import datetime as dt

from concurrent.futures import ThreadPoolExecutor

import asyncio
import base64
import io
import json
import os
import traceback
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

from agents import GeneratorAgent, InstructionAgent
from graph import build_solver_graph, initialise_state
from logging_utils import LogPrint, generate_run_id
from representation_utils import GRID, grid_to_base64_png
from state import SampleGrids, SolverConfig, SolverState, TokenUsage


# -------------------------------
# Configuration (adjust as needed)
# -------------------------------
CONFIG = SolverConfig(
    task_file="data/arc-prize-2025/arc-agi_evaluation_challenges.json",
    task_ids=None,
    num_of_tasks=None,
    offset_tasks=0,
    reasoning_level="medium",
    max_steps=10,
    max_concurrency=120,
    use_images=True,
    tracing=False,
)


@dataclass
class TaskOutcome:
    task_id: str
    graph_status: str
    train_solved: bool
    test_correct: bool | None
    summary_image_path: str
    instructions: list[str]
    token_usage: TokenUsage


@dataclass
class TaskRunResult:
    task_id: str
    outcome: TaskOutcome | None
    submission_entries: dict[str, list[dict[str, GRID]]] | None
    token_usage: TokenUsage
    solved: int
    failed: int
    test_correct: int
    test_incorrect: int
    error: str | None = None
    final_state: SolverState | None = None


def _usage_snapshot(usage: TokenUsage) -> TokenUsage:
    snapshot = TokenUsage()
    snapshot.merge(usage)
    return snapshot


async def _run_single_task(
    *,
    index: int,
    total: int,
    task_id: str,
    app,
    run_id: str,
    samples: SampleGrids,
    initial_state: SolverState,
    outputs_dir: Path,
    solutions_data: dict[str, Any] | None,
    executor: ThreadPoolExecutor,
    semaphore: asyncio.Semaphore,
    progress_bar: Progress | None = None,
    progress_task_id: TaskID | None = None,
    overall_progress_task_id: TaskID | None = None,
) -> TaskRunResult:
    async with semaphore:
        loop = asyncio.get_running_loop()
        initial_state.progress_callback(
            initial_state.step_index,
            initial_state.config.max_steps,
            "in progress",
            None,
        )
        initial_state.logs.append(f"[Task {task_id}] ({index}/{total}) Starting task")
        try:
            final_state_raw = await loop.run_in_executor(
                executor, app.invoke, initial_state
            )
        except Exception as exc:  # noqa: BLE001
            error_message = f"Task {task_id} failed with runtime error: {exc}\n{traceback.format_exc()}"
            initial_state.logs.append(error_message)
            return TaskRunResult(
                task_id=task_id,
                outcome=None,
                submission_entries=None,
                token_usage=TokenUsage(),
                solved=0,
                failed=1,
                test_correct=0,
                test_incorrect=0,
                error=error_message,
            )

        final_state = (
            final_state_raw
            if isinstance(final_state_raw, SolverState)
            else SolverState(**final_state_raw)
        )
        usage_snapshot = _usage_snapshot(final_state.token_usage)
        train_solved = final_state.graph_status == "completed"
        solved_count = 1 if train_solved else 0
        failed_count = 0 if train_solved else 1

        final_train_outputs = _ensure_outputs(
            final_state.train_partial_outputs, samples.train_outputs
        )
        final_test_outputs = _ensure_outputs(
            final_state.test_partial_outputs, samples.test_inputs
        )
        summary_path = _save_summary_image(
            run_id,
            task_id,
            samples,
            final_train_outputs,
            final_test_outputs,
            outputs_dir,
        )
        final_state.logs.append(f"Saved summary image: {summary_path}")

        test_correct: bool | None = None
        test_correct_count = 0
        test_incorrect_count = 0
        submission_entries: dict[str, list[dict[str, GRID]]] | None = None
        if solutions_data:
            expected_tests = solutions_data.get(task_id)
            if expected_tests is not None:
                test_correct = _compare_test_outputs(expected_tests, final_test_outputs)
                if test_correct:
                    test_correct_count = 1
                    final_state.logs.append("Test outputs match solutions.")
                    loop.call_soon_threadsafe(
                        partial(
                            progress_bar.update,
                            progress_task_id,
                            status=f"{final_state.graph_status} - test match",
                            result_emoji="[green]:heavy_check_mark:[/]",
                        )
                    )
                    loop.call_soon_threadsafe(
                        partial(
                            progress_bar.update, overall_progress_task_id, advance=1
                        )
                    )
                else:
                    test_incorrect_count = 1
                    final_state.logs.append("Test outputs differ from solutions.")
                    loop.call_soon_threadsafe(
                        partial(
                            progress_bar.update,
                            progress_task_id,
                            status=f"{final_state.graph_status} - test mismatch",
                            result_emoji="[red]:x:[/]",
                        )
                    )
            else:
                final_state.logs.append(
                    f"[{task_id}] No solution entry available; skipped solution check."
                )
        else:
            submission_entries = {
                task_id: _build_submission_entries(samples, final_test_outputs)
            }

        final_state.logs.append(f"[{task_id}] Instruction history:")
        for idx, instruction in enumerate(final_state.instructions, start=1):
            final_state.logs.append(f"[{task_id}]   {idx}. {instruction}")
        final_state.logs.append(
            f"[{task_id}] Generator status and findings: {final_state.generator_status_and_findings}"
        )
        final_state.logs.append(f"[{task_id}] Graph status: {final_state.graph_status}")

        outcome = TaskOutcome(
            task_id=task_id,
            graph_status=final_state.graph_status,
            train_solved=train_solved,
            test_correct=test_correct,
            summary_image_path=summary_path,
            instructions=final_state.instructions,
            token_usage=usage_snapshot,
        )

        return TaskRunResult(
            task_id=task_id,
            outcome=outcome,
            submission_entries=submission_entries,
            token_usage=usage_snapshot,
            solved=solved_count,
            failed=failed_count,
            test_correct=test_correct_count,
            test_incorrect=test_incorrect_count,
            final_state=final_state,
        )


def _calculate_max_concurrency(task_count: int) -> int:
    configured = CONFIG.max_concurrency
    if configured is not None and configured > 0:
        return max(1, min(configured, task_count or 1))
    return max(1, min(10, task_count or 1))


async def _async_main() -> None:
    start_time = dt.now()
    run_id = generate_run_id()
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    log_filename = outputs_dir / f"run_{run_id}.log"
    LogPrint.init_file(log_filename.as_posix())
    LogPrint.log(f"Run ID: {run_id}")
    LogPrint.log(f"Logging to: {log_filename.as_posix()}")

    tasks_data = _load_json(CONFIG.task_file)
    task_ids = _select_task_ids(tasks_data, CONFIG)
    LogPrint.log(
        f"Total tasks available: {len(tasks_data)} | Selected to run: {len(task_ids)}",
    )

    solutions_data, solutions_path = _load_solutions(CONFIG.task_file)
    if solutions_data:
        LogPrint.log(f"Loaded solutions file: {solutions_path}")
    else:
        LogPrint.log("No solutions file detected; will build submission JSON.")

    OPENAI_CLIENT_ARGS = {
        "max_retries": 5,
        "timeout": 1200,
    }
    client: OpenAI = OpenAI(**OPENAI_CLIENT_ARGS)
    callbacks: list[Any] = []
    if CONFIG.tracing:
        LogPrint.log("Initializing Langfuse tracing...")
        Langfuse(
            public_key="pk-lf-08d20d92-afc2-447c-88b0-ab8213a03be2",
            secret_key="sk-lf-89754f76-1300-456f-af59-cc4224990d76",
            host="http://localhost:3000",
        )
        callbacks.append(CallbackHandler())
        from langfuse.openai import OpenAI as LangfuseOpenAI

        client = LangfuseOpenAI(**OPENAI_CLIENT_ARGS)

    instruction_agent = InstructionAgent(client)
    generator_agent = GeneratorAgent(client)
    workflow = build_solver_graph(instruction_agent, generator_agent)
    app = workflow.compile().with_config(
        {
            "recursion_limit": CONFIG.max_steps * 4,
            "callbacks": callbacks,
        },
    )

    aggregate_usage = TokenUsage()
    outcomes: list[TaskOutcome] = []
    submission_payload: dict[str, list[dict[str, GRID]]] = {}

    solved_count = 0
    failed_count = 0
    test_correct_count = 0
    test_incorrect_count = 0

    max_concurrency = _calculate_max_concurrency(len(task_ids))
    semaphore = asyncio.Semaphore(max_concurrency)
    task_coroutines: list[asyncio.Task[TaskRunResult]] = []
    task_id_to_progress: dict[str, TaskID] = {}

    # Define columns for custom progress info
    columns = [
        TextColumn("{task.fields[task_id]}", style="bold"),
        BarColumn(),
        TaskProgressColumn(text_format="{task.fields[progress_type]} {task.completed}"),
        TimeElapsedColumn(),
        TextColumn("{task.fields[status]}", style="yellow"),
        TextColumn("{task.fields[result_emoji]}"),
    ]

    progress = Progress(*columns)
    overall_progress_task_id = progress.add_task(
        description="Test pass rate",
        total=len(task_ids),
        completed=0,
        task_id="Overall",
        status="in progress",
        result_emoji="",
        progress_type="Tests passed ",
    )
    loop = asyncio.get_running_loop()

    # Create per-task progress callbacks that marshal updates onto the event loop thread
    def make_progress_callback(task_id, progress_task_id):
        def _cb(step, max_steps, status, result_ok: bool | None):
            if result_ok is None:
                result_emoji = ""
            else:
                result_emoji = (
                    "[green]:heavy_check_mark:[/]" if result_ok else "[red]:x:[/]"
                )
            update_call = partial(
                progress.update,
                progress_task_id,
                completed=step,
                total=max_steps,
                status=status,
                visible=True,
                result_emoji=result_emoji,
            )
            loop.call_soon_threadsafe(update_call)

        return _cb

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        with progress:
            for index, task_id in enumerate(task_ids, start=1):
                progress_task_id = progress.add_task(
                    description="",
                    total=CONFIG.max_steps,
                    step=0,
                    task_id=task_id,
                    status="not started yet",
                    result_emoji="",
                    visible=False,
                    progress_type="Step ",
                )
                task_id_to_progress[task_id] = progress_task_id

                progress_callback = make_progress_callback(task_id, progress_task_id)
                samples = _samples_from_task(tasks_data[task_id])
                initial_state = initialise_state(CONFIG, task_id, samples)
                initial_state.progress_callback = progress_callback
                task_coroutines.append(
                    asyncio.create_task(
                        _run_single_task(
                            index=index,
                            total=len(task_ids),
                            task_id=task_id,
                            app=app,
                            run_id=run_id,
                            samples=samples,
                            initial_state=initial_state,
                            outputs_dir=outputs_dir,
                            solutions_data=solutions_data,
                            executor=executor,
                            semaphore=semaphore,
                            progress_bar=progress,
                            progress_task_id=progress_task_id,
                            overall_progress_task_id=overall_progress_task_id,
                        )
                    )
                )
            task_results = await asyncio.gather(*task_coroutines, return_exceptions=True)

    for result in task_results:
        if isinstance(result, Exception):
            LogPrint.log(
                f"Unexpected error during task execution: {result}\n{traceback.format_exc()}"
            )
            failed_count += 1
            continue

        if result.error:
            LogPrint.log(f"Task {result.task_id} failed: {result.error}")
            progress.update(
                task_id_to_progress[result.task_id],
                status="execution error",
                result_emoji="[red]:x:[/]",
            )
            continue
        LogPrint.log(f"Completed Task {result.task_id}. Logs are written to file.")
        LogPrint.log(
            "\n########################################"
            f"Task {result.task_id} logs:\n" + "\n".join(result.final_state.logs),
            console_print=False,
        )
        aggregate_usage.merge(result.token_usage)
        solved_count += result.solved
        failed_count += result.failed
        test_correct_count += result.test_correct
        test_incorrect_count += result.test_incorrect

        if result.outcome:
            outcomes.append(result.outcome)
        if result.submission_entries:
            submission_payload.update(result.submission_entries)

    if submission_payload:
        submission_path = outputs_dir / f"submission_{run_id}.json"
        with submission_path.open("w", encoding="utf-8") as fh:
            json.dump(submission_payload, fh, indent=2)
        LogPrint.log(
            f"Wrote submission file: {submission_path}",
        )

    LogPrint.log("------ Run Summary ------")
    LogPrint.log(f"Tasks attempted: {len(task_ids)}")
    LogPrint.log(f"Tasks solved (train): {solved_count}")
    LogPrint.log(f"Tasks failed (train): {failed_count}")

    if solutions_data:
        total_evaluated = test_correct_count + test_incorrect_count
        if total_evaluated:
            accuracy = (test_correct_count / total_evaluated) * 100
            LogPrint.log(
                f"Test outputs correct: {test_correct_count}, incorrect: {test_incorrect_count}, percentage: {accuracy:.2f}%"
            )
        else:
            LogPrint.log("No evaluable test outputs were produced.")
    LogPrint.log(
        (
            f"Token usage - input: {aggregate_usage.input_tokens}, "
            f"cached: {aggregate_usage.cached_tokens}, "
            f"output: {aggregate_usage.output_tokens}, "
            f"total: {aggregate_usage.total_tokens}\n"
            f"Estimated GPT-5 cost: ${aggregate_usage.cost_usd():.4f}, "
            f"Average per completed task: ${aggregate_usage.cost_usd() / max(len(outcomes), 1):.4f}"
        ),
    )
    LogPrint.log("Run complete.")
    end_time = dt.now()
    elapsed = end_time - start_time
    LogPrint.log(
        f"Took: {elapsed.seconds // 3600}h {elapsed.seconds // 60 % 60}m {elapsed.seconds % 60}s"
    )


def main() -> None:
    asyncio.run(_async_main())


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _select_task_ids(data: dict[str, Any], config: SolverConfig) -> list[str]:
    if config.task_ids:
        return [task_id for task_id in config.task_ids if task_id in data]
    all_ids = sorted(data.keys())
    start = min(config.offset_tasks, len(all_ids))
    if config.num_of_tasks is None:
        return all_ids[start:]
    return all_ids[start : start + config.num_of_tasks]


def _load_solutions(task_file: str) -> tuple[dict[str, Any] | None, str | None]:
    candidate = task_file.replace("challenges", "solutions")
    if not os.path.exists(candidate):
        return None, None
    return _load_json(candidate), candidate


def _samples_from_task(data: dict[str, Any]) -> SampleGrids:
    train_inputs: list[GRID] = []
    train_outputs: list[GRID] = []
    for sample in data.get("train", []):
        train_inputs.append(sample["input"])
        train_outputs.append(sample["output"])
    test_inputs = [sample["input"] for sample in data.get("test", [])]
    return SampleGrids(
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        test_inputs=test_inputs,
    )


def _ensure_outputs(partials: list[GRID | None], references: list[GRID]) -> list[GRID]:
    ensured: list[GRID] = []
    for partial, reference in zip(partials, references, strict=True):
        if partial is not None:
            ensured.append(partial)
        else:
            ensured.append(_blank_grid_like(reference))
    return ensured


def _blank_grid_like(grid: GRID) -> GRID:
    rows = len(grid)
    cols = len(grid[0]) if rows else 1
    return [[0 for _ in range(cols)] for _ in range(rows or 1)]


def _save_summary_image(
    run_id: str,
    task_id: str,
    samples: SampleGrids,
    train_outputs: list[GRID],
    test_outputs: list[GRID],
    outputs_dir: Path,
) -> str:
    rows: list[list[tuple[str, Image.Image]]] = []
    for idx in range(samples.train_count()):
        rows.append(
            [
                (f"Train {idx} Input", _grid_to_image(samples.train_inputs[idx])),
                (f"Train {idx} Expected", _grid_to_image(samples.train_outputs[idx])),
                (f"Train {idx} Predicted", _grid_to_image(train_outputs[idx])),
            ]
        )
    for idx in range(samples.test_count()):
        rows.append(
            [
                (f"Test {idx} Input", _grid_to_image(samples.test_inputs[idx])),
                (f"Test {idx} Predicted", _grid_to_image(test_outputs[idx])),
            ]
        )
    summary = _compose_labelled_grid(rows)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    image_path = outputs_dir / f"{run_id}_{task_id}_summary.png"
    summary.save(image_path)
    return image_path.as_posix()


def _grid_to_image(grid: GRID) -> Image.Image:
    b64 = grid_to_base64_png(grid)
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data))


def _compose_labelled_grid(rows: list[list[tuple[str, Image.Image]]]) -> Image.Image:
    padding = 12
    background = (24, 24, 24)
    text_color = (235, 235, 235)
    font = ImageFont.load_default()
    label_height = font.getbbox("Sample")[3] + 4

    row_images: list[Image.Image] = []
    max_width = 0
    total_height = padding
    for row in rows:
        max_tile_height = max(image.height for _, image in row)
        row_height = label_height + padding + max_tile_height + padding
        row_width = padding * (len(row) + 1) + sum(image.width for _, image in row)
        row_canvas = Image.new("RGB", (row_width, row_height), background)
        draw = ImageDraw.Draw(row_canvas)
        x_offset = padding
        for label, image in row:
            draw.text((x_offset, padding // 2), label, fill=text_color, font=font)
            row_canvas.paste(image, (x_offset, padding + label_height))
            x_offset += image.width + padding
        row_images.append(row_canvas)
        max_width = max(max_width, row_width)
        total_height += row_height + padding

    canvas = Image.new("RGB", (max_width + padding * 2, total_height), background)
    y_offset = padding
    for row_image in row_images:
        canvas.paste(row_image, (padding, y_offset))
        y_offset += row_image.height + padding
    return canvas


def _compare_test_outputs(expected_tests: list[Any], predicted: list[GRID]) -> bool:
    if len(expected_tests) != len(predicted):
        return False
    for expected_grid, predicted_grid in zip(expected_tests, predicted, strict=True):
        if expected_grid != predicted_grid:
            return False
    return True


def _build_submission_entries(
    samples: SampleGrids,
    predicted: list[GRID],
) -> list[dict[str, GRID]]:
    entries: list[dict[str, GRID]] = []
    for idx, grid in enumerate(predicted):
        entries.append(
            {
                "attempt_1": grid,
                "attempt_2": grid,
            }
        )
    return entries


if __name__ == "__main__":
    main()
