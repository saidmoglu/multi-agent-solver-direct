#!/usr/bin/env python3
"""Simplified Kaggle-compatible ARC solver powered by Qwen3 VL."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

GRID = List[List[int]]
Color = Tuple[int, int, int]

DEFAULT_COLOR_MAP: Dict[int, Color] = {
    0: (0, 0, 0),
    1: (0, 0, 255),
    2: (255, 0, 0),
    3: (0, 255, 0),
    4: (255, 255, 0),
    5: (192, 192, 192),
    6: (255, 0, 255),
    7: (255, 165, 0),
    8: (0, 255, 255),
    9: (128, 255, 128),
}

MODEL_DIR = os.environ.get(
    "QWEN3_VL_MODEL_DIR",
    "/kaggle/input/qwen-3-vl/transformers/8b-thinking/1/",
)

_MODEL: Optional[Qwen3VLForConditionalGeneration] = None
_PROCESSOR: Optional[AutoProcessor] = None

torch.set_grad_enabled(False)


def load_model() -> tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    global _MODEL, _PROCESSOR
    if _MODEL is None or _PROCESSOR is None:
        _MODEL = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_DIR, dtype="auto", device_map="auto"
        )
        _MODEL.eval()
        _PROCESSOR = AutoProcessor.from_pretrained(MODEL_DIR)
    return _MODEL, _PROCESSOR


def get_model_response(
    messages: List[Dict[str, Any]], max_thinking_length: int = 1024
) -> str:
    model, processor = load_model()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Step 1: Generate with max_new_tokens = max_thinking_length
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_thinking_length)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    decoded = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    text = decoded[0] if decoded else ""

    # Step 2: Check if </think> is present
    if "</think>" in text:
        messages.append(
            {
                "role": "assistant",
                "content": text,
            }
        )
    else:
        # If </think> is not present, append the forced ending and continue
        forced_bit = "Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n"
        text_with_forced = text + forced_bit
        messages.append(
            {
                "role": "assistant",
                "content": text_with_forced,
            }
        )

    new_input = processor.apply_chat_template(
        messages,
        tokenize=True,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    new_input = new_input.to(model.device)
    max_tokens = 262100 - new_input.input_ids.shape[1]
    with torch.no_grad():
        generated_ids = model.generate(**new_input, max_new_tokens=max_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(new_input.input_ids, generated_ids)
    ]
    decoded = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    text = decoded[0] if decoded else ""
    if "</think>" in text:
        text = text.split("</think>", 1)[1].lstrip()
    return text


def _fallback_color(value: int) -> Color:
    normalized = abs(int(value)) % 256
    return (normalized, normalized, normalized)


def _validate_rectangular_grid(grid: GRID) -> Tuple[int, int]:
    if not grid:
        raise ValueError("Grid must have at least one row.")
    width = len(grid[0])
    if width == 0:
        raise ValueError("Grid rows must have at least one column.")
    for idx, row in enumerate(grid):
        if len(row) != width:
            raise ValueError(
                f"Grid is not rectangular: row {idx} has length {len(row)} instead of {width}."
            )
    return len(grid), width


def grid_to_text(grid: GRID) -> str:
    _validate_rectangular_grid(grid)
    return "\n".join(" ".join(str(int(cell)) for cell in row) for row in grid)


def grid_to_image(
    grid: GRID,
    *,
    cell_size: int = 24,
    draw_grid_lines: bool = True,
    grid_line_width: int = 1,
) -> Image.Image:
    rows, cols = _validate_rectangular_grid(grid)
    image = Image.new("RGB", (cols * cell_size, rows * cell_size), color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    for row_index, row in enumerate(grid):
        for col_index, value in enumerate(row):
            color = DEFAULT_COLOR_MAP.get(int(value), _fallback_color(value))
            top_left = (col_index * cell_size, row_index * cell_size)
            bottom_right = (
                top_left[0] + cell_size,
                top_left[1] + cell_size,
            )
            draw.rectangle([top_left, bottom_right], fill=color)
    if draw_grid_lines and grid_line_width > 0:
        for row_index in range(1, rows):
            y = row_index * cell_size
            draw.line(
                [(0, y), (cols * cell_size, y)],
                fill=(255, 255, 255),
                width=grid_line_width,
            )
        for col_index in range(1, cols):
            x = col_index * cell_size
            draw.line(
                [(x, 0), (x, rows * cell_size)],
                fill=(255, 255, 255),
                width=grid_line_width,
            )
    return image


def blank_grid_like(grid: GRID) -> GRID:
    rows, cols = _validate_rectangular_grid(grid)
    return [[0 for _ in range(cols)] for _ in range(rows)]


def generate_grid_diff(expected: GRID, actual: GRID) -> str:
    try:
        _validate_rectangular_grid(expected)
        _validate_rectangular_grid(actual)
    except ValueError:
        return "Diff unavailable (invalid grid shapes)."
    if len(expected) != len(actual):
        return "Diff unavailable (row count mismatch)."
    lines: List[str] = []
    for exp_row, act_row in zip(expected, actual):
        if len(exp_row) != len(act_row):
            return "Diff unavailable (column count mismatch)."
        row_entries = []
        for exp_val, act_val in zip(exp_row, act_row):
            if exp_val == act_val:
                row_entries.append(f"✓{exp_val}")
            else:
                row_entries.append(f"{act_val}→{exp_val}")
        lines.append(" | ".join(row_entries))
    return "\n".join(lines)


def _extract_first_json_block(text: str) -> Optional[str]:
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def call_model_json(
    messages: List[Dict[str, Any]], *, max_retries: int = 2
) -> Dict[str, Any]:
    history = list(messages)
    last_text = ""
    for attempt in range(max_retries + 1):
        last_text = get_model_response(history)
        json_block = _extract_first_json_block(last_text.strip())
        if json_block is not None:
            try:
                return json.loads(json_block)
            except json.JSONDecodeError:
                pass
        history = history + [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "The previous reply was not valid JSON. Respond again with ONLY a valid JSON object that matches the requested schema without any extra commentary.",
                    }
                ],
            }
        ]
    raise ValueError(f"Model failed to produce valid JSON. Last response: {last_text}")


def ensure_optional_grid_list(
    candidate: Optional[Sequence[Optional[GRID]]],
    previous: Sequence[Optional[GRID]],
    shape_references: Sequence[GRID],
) -> List[Optional[GRID]]:
    normalized: List[Optional[GRID]] = []
    total = len(shape_references)
    for index in range(total):
        grid_data: Optional[GRID] = None
        if candidate and index < len(candidate):
            grid_data = candidate[index]
        converted: Optional[GRID] = None
        if grid_data is not None:
            try:
                converted = [[int(cell) for cell in row] for row in grid_data]
                _validate_rectangular_grid(converted)
            except Exception:  # noqa: BLE001
                converted = None
        if converted is None:
            fallback = previous[index] if index < len(previous) else None
            normalized.append(fallback)
        else:
            normalized.append(converted)
    return normalized


def ensure_submission_grid_list(
    grids: Sequence[Optional[GRID]],
    shape_references: Sequence[GRID],
) -> List[GRID]:
    outputs: List[GRID] = []
    for grid, reference in zip(grids, shape_references):
        if grid is None:
            outputs.append(blank_grid_like(reference))
        else:
            outputs.append([[int(cell) for cell in row] for row in grid])
    return outputs


@dataclass
class InstructionResult:
    instruction: str
    status: str
    summary: str


@dataclass
class GeneratorResult:
    status_and_findings: str
    train_outputs: List[Optional[GRID]]
    test_outputs: List[Optional[GRID]]


INSTRUCTION_SYSTEM_PROMPT = """
You are an expert assistant for solving Abstraction and Reasoning Corpus (ARC) puzzles.
You are collaborating with a separate generator that executes your instructions on the grids.
Carefully study the training and test samples, then propose the next high-impact instruction.
Return a single JSON object with the following fields:
  - "instruction": string containing one actionable transformation to apply now.
  - "status": either "in_progress" if more instructions will be needed, or "complete" if the task is ready for finalisation.
  - "summary": short natural-language description of your reasoning for this step.
Do not include any additional text outside of the JSON object.
"""

GENERATOR_SYSTEM_PROMPT = """
You transform ARC grids exactly according to the provided instruction.
Always re-emit full grids for every train and test sample.
Return a JSON object with fields:
  - "status_and_findings": concise summary of what changed.
  - "train_outputs": list of 2D integer arrays for each train sample (use null to keep previous output).
  - "test_outputs": list of 2D integer arrays for each test sample (use null to keep previous output).
Do not include explanations outside the JSON object.
"""


@dataclass
class SolverConfig:
    task_file: str
    task_ids: Optional[List[str]] = None
    num_of_tasks: Optional[int] = None
    offset_tasks: int = 0
    max_steps: int = 12
    use_images: bool = False


@dataclass
class SampleGrids:
    train_inputs: List[GRID]
    train_outputs: List[GRID]
    test_inputs: List[GRID]


@dataclass
class SolverState:
    config: SolverConfig
    task_id: str
    samples: SampleGrids
    instructions: List[str] = field(default_factory=list)
    summaries: List[str] = field(default_factory=list)
    train_partial_outputs: List[Optional[GRID]] = field(default_factory=list)
    test_partial_outputs: List[Optional[GRID]] = field(default_factory=list)
    generator_status_and_findings: str = ""
    graph_status: str = "running train"
    step_index: int = 0

    def train_shapes(self) -> List[GRID]:
        return self.samples.train_outputs

    def test_shapes(self) -> List[GRID]:
        return self.samples.test_inputs


def merge_partial_outputs(
    previous: Sequence[Optional[GRID]],
    updates: Sequence[Optional[GRID]],
) -> List[Optional[GRID]]:
    merged: List[Optional[GRID]] = []
    for older, newer in zip(previous, updates):
        merged.append(newer if newer is not None else older)
    return merged


def train_outputs_match(
    expected: Sequence[GRID],
    actual: Sequence[Optional[GRID]],
) -> bool:
    for exp, act in zip(expected, actual):
        if act is None:
            return False
        if exp != act:
            return False
    return True


class InstructionAgent:
    def __init__(self, config: SolverConfig) -> None:
        self.config = config

    def __call__(self, state: SolverState) -> InstructionResult:
        phase = "training" if state.graph_status == "running train" else "test"
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": INSTRUCTION_SYSTEM_PROMPT,
                    }
                ],
            },
            {
                "role": "user",
                "content": self._build_user_content(state, phase),
            },
        ]
        data = call_model_json(messages)
        instruction = str(data.get("instruction", "")).strip()
        status = str(data.get("status", "in_progress")).strip().lower()
        summary = str(data.get("summary", "")).strip()
        if status not in {"in_progress", "complete"}:
            status = "in_progress"
        if not instruction:
            raise ValueError("Instruction agent returned an empty instruction.")
        return InstructionResult(
            instruction=instruction, status=status, summary=summary
        )

    def _build_user_content(
        self, state: SolverState, phase: str
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        content.append(
            {
                "type": "text",
                "text": (
                    f"Task ID: {state.task_id}\n"
                    f"Phase: {phase}\n"
                    f"Step index: {state.step_index}\n"
                    f"Instructions so far: {len(state.instructions)}"
                ),
            }
        )
        if state.instructions:
            history = "\n".join(
                f"{idx + 1}. {inst}" for idx, inst in enumerate(state.instructions)
            )
            content.append({"type": "text", "text": f"Instruction history:\n{history}"})
        else:
            content.append({"type": "text", "text": "Instruction history: (none)"})
        if state.generator_status_and_findings:
            content.append(
                {
                    "type": "text",
                    "text": f"Generator findings so far:\n{state.generator_status_and_findings}",
                }
            )
        for index, (train_input, expected_output) in enumerate(
            zip(state.samples.train_inputs, state.samples.train_outputs)
        ):
            content.append(
                {
                    "type": "text",
                    "text": f"Train sample {index} - INPUT:\n{grid_to_text(train_input)}",
                }
            )
            if self.config.use_images:
                content.append(
                    {
                        "type": "image",
                        "image": grid_to_image(train_input),
                    }
                )
            content.append(
                {
                    "type": "text",
                    "text": f"Train sample {index} - EXPECTED OUTPUT:\n{grid_to_text(expected_output)}",
                }
            )
            if self.config.use_images:
                content.append(
                    {
                        "type": "image",
                        "image": grid_to_image(expected_output),
                    }
                )
            current = state.train_partial_outputs[index]
            if current is not None:
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"Train sample {index} - CURRENT OUTPUT:\n{grid_to_text(current)}"
                        ),
                    }
                )
                if self.config.use_images:
                    content.append(
                        {
                            "type": "image",
                            "image": grid_to_image(current),
                        }
                    )
                diff = generate_grid_diff(expected_output, current)
                content.append(
                    {
                        "type": "text",
                        "text": f"Train sample {index} diff (actual vs expected):\n{diff}",
                    }
                )
        for index, test_input in enumerate(state.samples.test_inputs):
            content.append(
                {
                    "type": "text",
                    "text": f"Test sample {index} - INPUT:\n{grid_to_text(test_input)}",
                }
            )
            if self.config.use_images:
                content.append(
                    {
                        "type": "image",
                        "image": grid_to_image(test_input),
                    }
                )
            current = state.test_partial_outputs[index]
            if current is not None:
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"Test sample {index} - CURRENT OUTPUT:\n{grid_to_text(current)}"
                        ),
                    }
                )
                if self.config.use_images:
                    content.append(
                        {
                            "type": "image",
                            "image": grid_to_image(current),
                        }
                    )
        return content


class GeneratorAgent:
    def __init__(self, config: SolverConfig) -> None:
        self.config = config

    def __call__(self, state: SolverState, instruction: str) -> GeneratorResult:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": GENERATOR_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": self._build_user_content(state, instruction),
            },
        ]
        data = call_model_json(messages)
        status_and_findings = str(data.get("status_and_findings", "")).strip()
        train_outputs_raw = data.get("train_outputs")
        test_outputs_raw = data.get("test_outputs")
        train_outputs = ensure_optional_grid_list(
            train_outputs_raw,
            state.train_partial_outputs,
            state.samples.train_outputs,
        )
        test_outputs = ensure_optional_grid_list(
            test_outputs_raw,
            state.test_partial_outputs,
            state.samples.test_inputs,
        )
        return GeneratorResult(
            status_and_findings=status_and_findings,
            train_outputs=train_outputs,
            test_outputs=test_outputs,
        )

    def _build_user_content(
        self,
        state: SolverState,
        instruction: str,
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        content.append(
            {
                "type": "text",
                "text": (
                    f"Task ID: {state.task_id}\n"
                    f"Step index: {state.step_index}\n"
                    f"Graph status: {state.graph_status}\n"
                    f"Apply instruction: {instruction}"
                ),
            }
        )
        for index, train_input in enumerate(state.samples.train_inputs):
            content.append(
                {
                    "type": "text",
                    "text": f"Train sample {index} - INPUT:\n{grid_to_text(train_input)}",
                }
            )
            if self.config.use_images:
                content.append(
                    {
                        "type": "image",
                        "image": grid_to_image(train_input),
                    }
                )
            previous = state.train_partial_outputs[index]
            if previous is not None:
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"Train sample {index} - PREVIOUS OUTPUT:\n{grid_to_text(previous)}"
                        ),
                    }
                )
                if self.config.use_images:
                    content.append(
                        {
                            "type": "image",
                            "image": grid_to_image(previous),
                        }
                    )
        for index, test_input in enumerate(state.samples.test_inputs):
            content.append(
                {
                    "type": "text",
                    "text": f"Test sample {index} - INPUT:\n{grid_to_text(test_input)}",
                }
            )
            if self.config.use_images:
                content.append(
                    {
                        "type": "image",
                        "image": grid_to_image(test_input),
                    }
                )
            previous = state.test_partial_outputs[index]
            if previous is not None:
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"Test sample {index} - PREVIOUS OUTPUT:\n{grid_to_text(previous)}"
                        ),
                    }
                )
                if self.config.use_images:
                    content.append(
                        {
                            "type": "image",
                            "image": grid_to_image(previous),
                        }
                    )
        return content


@dataclass
class TaskResult:
    task_id: str
    train_solved: bool
    graph_status: str
    train_outputs: List[GRID]
    test_outputs: List[GRID]
    instructions: List[str]
    summaries: List[str]
    generator_status: str


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def select_task_ids(data: Dict[str, Any], config: SolverConfig) -> List[str]:
    all_ids = sorted(data.keys())
    if config.task_ids:
        filtered = [task_id for task_id in config.task_ids if task_id in data]
    else:
        filtered = all_ids
    start = min(config.offset_tasks, len(filtered))
    if config.num_of_tasks is None:
        return filtered[start:]
    return filtered[start : start + config.num_of_tasks]


def load_solutions(task_file: str) -> Optional[Dict[str, Any]]:
    candidate = task_file.replace("challenges", "solutions")
    if os.path.exists(candidate):
        return load_json(candidate)
    return None


def prepare_samples(task_data: Dict[str, Any]) -> SampleGrids:
    train_inputs: List[GRID] = []
    train_outputs: List[GRID] = []
    for sample in task_data.get("train", []):
        train_inputs.append(sample["input"])
        train_outputs.append(sample["output"])
    test_inputs = [sample["input"] for sample in task_data.get("test", [])]
    return SampleGrids(
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        test_inputs=test_inputs,
    )


def build_submission_entries(predicted: Sequence[GRID]) -> List[Dict[str, GRID]]:
    entries: List[Dict[str, GRID]] = []
    for grid in predicted:
        entries.append({"attempt_1": grid, "attempt_2": grid})
    return entries


class Solver:
    def __init__(self, config: SolverConfig) -> None:
        self.config = config
        self.instruction_agent = InstructionAgent(config)
        self.generator_agent = GeneratorAgent(config)

    def solve_task(
        self,
        task_id: str,
        task_data: Dict[str, Any],
    ) -> TaskResult:
        samples = prepare_samples(task_data)
        state = SolverState(
            config=self.config,
            task_id=task_id,
            samples=samples,
            train_partial_outputs=[None] * len(samples.train_inputs),
            test_partial_outputs=[None] * len(samples.test_inputs),
        )
        for step in range(self.config.max_steps):
            print(f"[{task_id}] Step {step + 1} / {self.config.max_steps}")
            instruction_result = self.instruction_agent(state)
            generator_result = self.generator_agent(
                state, instruction_result.instruction
            )
            state.instructions.append(instruction_result.instruction)
            state.summaries.append(instruction_result.summary)
            state.train_partial_outputs = merge_partial_outputs(
                state.train_partial_outputs, generator_result.train_outputs
            )
            state.test_partial_outputs = merge_partial_outputs(
                state.test_partial_outputs, generator_result.test_outputs
            )
            state.generator_status_and_findings = generator_result.status_and_findings
            state.step_index += 1
            if state.graph_status == "running train":
                if train_outputs_match(
                    samples.train_outputs, state.train_partial_outputs
                ):
                    state.graph_status = "running test"
                    print(f"[{task_id}] Training phase solved. Moving to test phase.")
            if state.graph_status == "running test":
                if instruction_result.status == "complete" and all(
                    output is not None for output in state.test_partial_outputs
                ):
                    state.graph_status = "completed"
                    print(f"[{task_id}] Declared complete by instruction agent.")
                    break
        else:
            if state.graph_status != "completed":
                state.graph_status = "failed_max_steps"
                print(f"[{task_id}] Reached maximum step count without completion.")
        final_train = ensure_submission_grid_list(
            state.train_partial_outputs, samples.train_outputs
        )
        final_test = ensure_submission_grid_list(
            state.test_partial_outputs, samples.test_inputs
        )
        train_solved = train_outputs_match(
            samples.train_outputs, state.train_partial_outputs
        )
        return TaskResult(
            task_id=task_id,
            train_solved=train_solved,
            graph_status=state.graph_status,
            train_outputs=final_train,
            test_outputs=final_test,
            instructions=state.instructions,
            summaries=state.summaries,
            generator_status=state.generator_status_and_findings,
        )


def run_solver(config: SolverConfig) -> None:
    tasks_data = load_json(config.task_file)
    task_ids = select_task_ids(tasks_data, config)
    print(f"Loaded {len(tasks_data)} tasks. Running {len(task_ids)} task(s).")
    solutions_data = load_solutions(config.task_file)
    if solutions_data:
        print("Solutions file detected. Will evaluate test accuracy.")
    else:
        print("No solutions file detected. Will generate submission payload.")
    solver = Solver(config)
    submission: Dict[str, List[Dict[str, GRID]]] = {}
    solved_train = 0
    completed = 0
    test_correct = 0
    test_total = 0
    for index, task_id in enumerate(task_ids, start=1):
        print(f"==== Task {index}/{len(task_ids)} :: {task_id} ====")
        result = solver.solve_task(task_id, tasks_data[task_id])
        solved_train += int(result.train_solved)
        if result.graph_status == "completed":
            completed += 1
        if solutions_data and task_id in solutions_data:
            expected_tests = solutions_data[task_id]
            if len(expected_tests) == len(result.test_outputs) and all(
                exp == pred for exp, pred in zip(expected_tests, result.test_outputs)
            ):
                test_correct += 1
                print(f"[{task_id}] Test outputs match ground truth.")
            else:
                print(f"[{task_id}] Test outputs differ from ground truth.")
            test_total += 1
        else:
            submission[task_id] = build_submission_entries(result.test_outputs)
        print(f"[{task_id}] Final status: {result.graph_status}")
        print(f"[{task_id}] Instruction count: {len(result.instructions)}")
        if result.instructions:
            print(f"[{task_id}] Last instruction: {result.instructions[-1]}")
        if result.generator_status:
            print(f"[{task_id}] Generator summary: {result.generator_status}")
    print("==== Run summary ====")
    print(f"Train puzzles solved: {solved_train} / {len(task_ids)}")
    print(f"Tasks completed (train + test): {completed} / {len(task_ids)}")
    if test_total:
        accuracy = (test_correct / test_total) * 100.0
        print(f"Test accuracy: {test_correct}/{test_total} ({accuracy:.2f}%)")
    if submission:
        output_path = "submission.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(submission, handle, indent=2)
        print(f"Wrote submission file to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simplified ARC solver using Qwen3 VL."
    )
    parser.add_argument(
        "--task-file",
        default="data/arc-prize-2025/arc-agi_evaluation_challenges.json",
        help="Path to the ARC challenges JSON file.",
    )
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Comma-separated list of task IDs to run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tasks to run.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of tasks to skip before starting.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=12,
        help="Maximum number of instruction/generator iterations per task.",
    )
    parser.add_argument(
        "--use-images",
        action="store_true",
        help="Include rendered grid images in LLM prompts (higher cost).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_ids = args.task_ids.split(",") if args.task_ids else None
    config = SolverConfig(
        task_file=args.task_file,
        task_ids=task_ids,
        num_of_tasks=args.limit,
        offset_tasks=args.offset,
        max_steps=args.max_steps,
        use_images=args.use_images,
    )
    run_solver(config)


if __name__ == "__main__":
    main()
