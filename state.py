from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Callable, Iterable, Literal, Sequence

from representation_utils import (
    GRID,
    grid_to_base64_png,
    grid_to_text,
)

ReasoningLevel = Literal["minimal", "low", "medium", "high"]
ConfidenceLevel = Literal["low", "medium", "high"]


@dataclass(slots=True)
class SolverConfig:
    """Static configuration for a LangGraph ARC run."""

    task_file: str = "arc-agi_training_challenges.json"
    task_ids: list[str] | None = None
    num_of_tasks: int | None = None
    offset_tasks: int = 0
    reasoning_level: ReasoningLevel = "minimal"
    max_steps: int = 10
    max_concurrency: int | None = None
    use_images: bool = True
    tracing: bool = False

    def reasoning_effort(self) -> ReasoningLevel:
        """Expose reasoning level in the exact literal form required by GPT-5."""
        return self.reasoning_level

    def should_use_images(self) -> bool:
        return self.use_images


@dataclass(slots=True)
class TokenUsage:
    """Accumulates token counts reported by OpenAI responses."""

    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: dict[str, int] | None) -> None:
        """Add usage counts returned by the API (if any)."""
        if not usage:
            return
        self.input_tokens += int(usage.get("input_tokens", 0))
        self.cached_tokens += int(usage.get("cached_tokens", 0))
        self.output_tokens += int(usage.get("output_tokens", 0))
        self.total_tokens += int(usage.get("total_tokens", 0))

    def merge(self, other: TokenUsage) -> None:
        """Accumulate counts from another TokenUsage instance."""
        self.input_tokens += other.input_tokens
        self.cached_tokens += other.cached_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens

    def cost_usd(self) -> float:
        """Compute USD cost using GPT-5 pricing."""
        billable_input = max(self.input_tokens - self.cached_tokens, 0)
        cost = (
            billable_input * 2.5 + self.cached_tokens * 0.25 + self.output_tokens * 20
        )
        return cost / 1_000_000


@dataclass(slots=True)
class SampleGrids:
    """All grids associated with a single ARC task."""

    train_inputs: list[GRID]
    train_outputs: list[GRID]
    test_inputs: list[GRID]

    def train_count(self) -> int:
        return len(self.train_inputs)

    def test_count(self) -> int:
        return len(self.test_inputs)

    def iter_train(self) -> Iterable[tuple[int, GRID, GRID]]:
        return (
            (index, inp, out)
            for index, (inp, out) in enumerate(
                zip(self.train_inputs, self.train_outputs, strict=True)
            )
        )

    def iter_test(self) -> Iterable[tuple[int, GRID]]:
        return enumerate(self.test_inputs)


@dataclass(slots=True)
class SolverState:
    """Mutable state propagated through the LangGraph nodes."""

    config: SolverConfig
    task_id: str
    samples: SampleGrids
    step_index: int = 0
    instructions: list[str] = field(default_factory=list)
    train_partial_outputs: list[GRID] = field(default_factory=list)
    test_partial_outputs: list[GRID] = field(default_factory=list)
    generator_status_and_findings: str | None = None
    pending_instruction: str | None = None
    pending_instruction_rationale: str | None = None
    pending_instruction_confidence: ConfidenceLevel | None = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    graph_status: Literal[
        "running train", "running test", "completed", "failed_max_steps", "failed_error"
    ] = "running train"
    last_verifier_passed: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    progress_callback: Callable | None = None
    logs: list[str] = field(default_factory=list)
    status_from_instr_node: str | None = None

    def clone_with_updates(
        self,
        *,
        instructions: list[str] | None = None,
        train_partial_outputs: list[GRID | None] | None = None,
        test_partial_outputs: list[GRID | None] | None = None,
        generator_status_and_findings: str | None = None,
        graph_status: str | None = None,
        last_verifier_passed: bool | None = None,
        step_index: int | None = None,
        pending_instruction: str | None = None,
        pending_instruction_rationale: str | None = None,
        pending_instruction_confidence: ConfidenceLevel | None = None,
        status_from_instr_node: str | None = None,
    ) -> SolverState:
        state = replace(self)
        if instructions is not None:
            state.instructions = instructions
        if train_partial_outputs is not None:
            state.train_partial_outputs = train_partial_outputs
        if test_partial_outputs is not None:
            state.test_partial_outputs = test_partial_outputs
        if generator_status_and_findings is not None:
            state.generator_status_and_findings = generator_status_and_findings
        if graph_status is not None:
            state.graph_status = graph_status  # type: ignore[assignment]
        if last_verifier_passed is not None:
            state.last_verifier_passed = last_verifier_passed
        if step_index is not None:
            state.step_index = step_index
        if pending_instruction is not None:
            state.pending_instruction = pending_instruction
        if pending_instruction_rationale is not None:
            state.pending_instruction_rationale = pending_instruction_rationale
        if pending_instruction_confidence is not None:
            state.pending_instruction_confidence = pending_instruction_confidence
        if status_from_instr_node is not None:
            state.status_from_instr_node = status_from_instr_node
        return state

    def advance_step(
        self,
        *,
        new_instruction: str,
        new_train_outputs: Sequence[GRID | None],
        new_test_outputs: Sequence[GRID | None],
        status_and_findings: str,
    ) -> SolverState:
        merged_train = merge_partial_outputs(
            self.train_partial_outputs, new_train_outputs
        )
        merged_test = merge_partial_outputs(self.test_partial_outputs, new_test_outputs)
        updated_instructions = [*self.instructions, new_instruction]
        return self.clone_with_updates(
            instructions=updated_instructions,
            train_partial_outputs=merged_train,
            test_partial_outputs=merged_test,
            generator_status_and_findings=status_and_findings,
            step_index=self.step_index + 1,
            pending_instruction=None,
            # pending_instruction_rationale=None,
            # pending_instruction_confidence=None,
        )

    def is_max_steps_reached(self) -> bool:
        return self.step_index >= self.config.max_steps

    def attach_token_usage(self, usage: dict[str, int] | None) -> None:
        self.token_usage.add(usage)


def create_initial_state(
    config: SolverConfig,
    task_id: str,
    samples: SampleGrids,
) -> SolverState:
    train_partials: list[GRID] = [[[1]]] * samples.train_count()
    test_partials: list[GRID] = [[[1]]] * samples.test_count()
    return SolverState(
        config=config,
        task_id=task_id,
        samples=samples,
        train_partial_outputs=train_partials,
        test_partial_outputs=test_partials,
    )


def merge_partial_outputs(
    existing: Sequence[GRID | None],
    updates: Sequence[GRID | None],
) -> list[GRID | None]:
    if len(existing) != len(updates):
        updates = copy.deepcopy(existing)
    merged: list[GRID | None] = []
    for original, newer in zip(existing, updates, strict=True):
        merged.append(newer if newer is not None else original)
    return merged


def ensure_partial_lengths(
    samples: SampleGrids,
) -> tuple[list[GRID | None], list[GRID | None]]:
    return (
        [None] * samples.train_count(),
        [None] * samples.test_count(),
    )


def build_user_content_samples(
    title: str,
    entries: Sequence[dict[str, Any]],
    *,
    use_images: bool = True,
) -> list[dict[str, str]]:
    """
    Helper for constructing OpenAI Responses API `input` segments.

    Parameters
    ----------
    title:
        Section heading prepended to each entry label.
    entries:
        Iterable of dictionaries containing:
            - "label": str  -> descriptor appended to the title.
            - "grid": GRID  -> grid to render.
            - "text": str | None -> optional textual summary override.
            - "diff": str | None -> optional diff string appended as text.
    use_images:
        Whether to include base64 image segments.

    Returns
    -------
    list[dict[str, str]]
        Sequence ready to append into the `input` parameter for the Responses API.
    """
    parts: list[dict[str, str]] = []
    for index, item in enumerate(entries):
        label = f"{title} {item.get('label', index)}"
        grid: GRID = item["grid"]
        grid_text = item.get("text") or grid_to_text(grid)
        parts.append({"type": "input_text", "text": label})
        parts.append({"type": "input_text", "text": grid_text})
        diff_text = item.get("diff")

        if use_images:
            parts.append(
                {
                    "type": "input_text",
                    "text": f"{label} (image below)",
                }
            )
            parts.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{grid_to_base64_png(grid)}",
                }
            )
        if diff_text:
            parts.append(
                {
                    "type": "input_text",
                    "text": "Difference notation (actual→expected):",
                }
            )
            parts.append({"type": "input_text", "text": diff_text})
    return parts


def flatten_optional_grids(grids: Sequence[GRID | None]) -> list[GRID | None]:
    return list(grids)


def iter_with_previous(values: Sequence[Any]) -> Iterable[tuple[int, Any, Any | None]]:
    """
    Utility for agents to compare current vs previous values.

    Returns
    -------
    Iterator yielding `(index, current, previous)` triples.
    """
    previous_iter = itertools.chain([None], values[:-1])
    return (
        (idx, current, prev)
        for idx, (current, prev) in enumerate(zip(values, previous_iter))
    )
