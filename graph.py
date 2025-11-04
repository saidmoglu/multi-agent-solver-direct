from __future__ import annotations

from typing import Any, Callable, Sequence

from langgraph.graph import END, StateGraph


from agents import GeneratorAgent, InstructionAgent
from representation_utils import GRID, generate_grid_diff
from state import (
    SampleGrids,
    SolverState,
    create_initial_state,
)
import state


def build_solver_graph_old(
    instruction_agent: InstructionAgent,
    generator_agent: GeneratorAgent,
) -> StateGraph:
    """
    Construct the LangGraph with Instruction → Generator → Verifier loop.

    Nodes:
        instruction_node: Calls instruction agent, updates pending instruction fields.
        generator_node: Calls generator agent with pending instruction and merges outputs.
        verifier_node: Compares train outputs to expected ones and controls loop.

    Returns
    -------
    StateGraph
        Configured LangGraph graph ready for compilation.
    """
    graph = StateGraph(SolverState)

    graph.add_node("instruction", _make_instruction_node(instruction_agent))
    graph.add_node("generator", _make_generator_node(generator_agent))
    graph.add_node("verifier", verifier_node)

    graph.set_entry_point("instruction")
    graph.add_edge("instruction", "generator")
    graph.add_edge("generator", "verifier")

    def route_from_verifier_node(state: SolverState) -> str:
        if state.graph_status in {"completed", "failed_max_steps", "failed_error"}:
            return END
        return "instruction"

    graph.add_conditional_edges(
        "verifier",
        route_from_verifier_node,
    )

    return graph


def build_solver_graph(
    instruction_agent: InstructionAgent,
    generator_agent: GeneratorAgent,
) -> StateGraph:
    graph = StateGraph(SolverState)

    graph.add_node("instruction", _make_instruction_node(instruction_agent))
    graph.add_node("generator", _make_generator_node(generator_agent))
    graph.add_node("verifier", verifier_node)
    graph.set_entry_point("instruction")

    def route_from_instruction_node(state: SolverState) -> str:
        if state.status_from_instr_node == "complete":
            return END
        return "generator"

    graph.add_conditional_edges(
        "instruction",
        route_from_instruction_node,
    )

    def route_from_generator_node(state: SolverState) -> str:
        if state.is_max_steps_reached():
            state.logs.append("Maximum steps reached without solving the task.")
            state.progress_callback(
                state.step_index, state.config.max_steps, "reached_max_steps", False
            )
            return END
        if state.graph_status == "running train":
            state.progress_callback(
                state.step_index, state.config.max_steps, state.graph_status, None
            )
            return "verifier"
        state.progress_callback(
            state.step_index, state.config.max_steps, state.graph_status, None
        )
        return "instruction"

    graph.add_conditional_edges(
        "generator",
        route_from_generator_node,
    )

    graph.add_edge("verifier", "instruction")

    return graph


def initialise_state(config, task_id: str, samples) -> SolverState:
    """Convenience wrapper for callers to build initial state."""
    return create_initial_state(config, task_id, samples)


def _make_instruction_node(
    agent: InstructionAgent,
) -> Callable[[SolverState], SolverState]:
    def run(state: SolverState) -> SolverState:
        result = agent(state)
        state.attach_token_usage(result.usage)
        state.logs.append(
            f"\nInstruction node output:\nInstruction: {result.instruction}"
        )
        return state.clone_with_updates(
            pending_instruction=result.instruction,
            status_from_instr_node=result.status,
            # pending_instruction_rationale=result.rationale,
            # pending_instruction_confidence=result.confidence,
        )

    return run


def _summarise_diffs(state: SolverState, samples: SampleGrids) -> str:
    lines: list[str] = []
    for idx, expected in enumerate(samples.train_outputs):
        actual = state.train_partial_outputs[idx]
        if actual is None:
            lines.append(f"Train sample {idx}: no output yet.")
            continue
        if actual != expected:
            diff = generate_grid_diff(expected, actual)
            lines.append(f"Train sample {idx} diff:\n{diff}")
    return "\n".join(lines)


def _make_generator_node(
    agent: GeneratorAgent,
) -> Callable[[SolverState], SolverState]:
    def run(state: SolverState) -> SolverState:
        if not state.pending_instruction:
            raise ValueError("Generator node invoked without pending instruction.")
        result = agent(state, state.pending_instruction)
        state.attach_token_usage(result.usage)
        state.logs.append(
            f"\nGenerator node produced {len(result.train_outputs)} train and "
            f"{len(result.test_outputs)} test outputs."
            f" Status and Findings: {result.status_and_findings}"
        )
        new_state = state.advance_step(
            new_instruction=state.pending_instruction,
            new_train_outputs=[
                item.grid if item else None for item in result.train_outputs
            ],
            new_test_outputs=[
                item.grid if item else None for item in result.test_outputs
            ],
            status_and_findings=result.status_and_findings,
        )
        diffs_summary = _summarise_diffs(new_state, new_state.samples)
        if diffs_summary:
            state.logs.append(f"Output diffs:\n{diffs_summary}")
        return new_state

    return run


def verifier_node(state: SolverState) -> SolverState:
    """Verify train outputs, update graph status, and optionally attach diffs."""
    all_match = True
    for index, expected in enumerate(state.samples.train_outputs):
        actual = state.train_partial_outputs[index]
        if actual is None or actual != expected:
            all_match = False
            break

    if all_match:
        state.logs.append("All train outputs match expected outputs.")
        return state.clone_with_updates(
            graph_status="running test",
            last_verifier_passed=True,
        )
    state.logs.append("Train outputs do not match expected outputs; continuing.")
    return state.clone_with_updates(
        last_verifier_passed=False,
    )
