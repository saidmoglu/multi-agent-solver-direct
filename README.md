# Multi-Agent ARC Solver (LangGraph)

A modular, multi-agent system for solving Abstraction and Reasoning Corpus (ARC) puzzles using LLM-driven instruction planning and grid transformation. This project leverages [LangGraph](https://github.com/langchain-ai/langgraph) to orchestrate a loop of instruction and generator agents, with robust logging, visualization, and support for both local and Kaggle-style evaluation.

---

## Features

- **Instruction Agent**: LLM-based planner that generates minimal, incremental transformation steps for ARC puzzles.
- **Generator Agent**: Applies instructions to grid samples, producing updated outputs and status summaries.
- **Verifier Node**: Compares generated outputs to expected results, controlling the solve loop.
- **Rich Logging**: Step-by-step logs, token usage, and cost tracking.
- **Visualization**: Generates base64 PNGs and combined summary images for all grid states.
- **Kaggle Compatibility**: Includes a standalone script for Kaggle ARC competitions.
- **Highly Configurable**: Control task selection, step limits, reasoning level, and more via config.

---

## Project Structure

- [`representation_utils.py`](representation_utils.py): Grid-to-text, diff, and image utilities.
- [`state.py`](state.py): Dataclasses for solver state, configuration, and token usage.
- [`agents.py`](agents.py): Instruction and Generator agent implementations, prompt templates, and response parsing.
- [`graph.py`](graph.py): LangGraph graph definition and node logic.
- [`runner.py`](runner.py): Orchestrates task loading, execution, logging, and output artifact generation.
- [`logging_utils.py`](logging_utils.py): File logger and print wrapper.
- [`kaggle_solver.py`](kaggle_solver.py): Standalone for Kaggle notebook.
- `data/`: Contains ARC challenge and solution JSON files.
- `outputs/`: Logs and generated artifacts (created at runtime).

---

## Installation

**Requirements:**
- Python 3.13+
- See [`pyproject.toml`](pyproject.toml) for dependencies:
  - `langchain`, `langfuse`, `langgraph`, `openai`, `pillow`, `rich`

**Install dependencies:**
```bash
uv sync
```

---

## Usage

### 1. Standard Run

```bash
uv run runner.py
```

- By default, runs on `data/arc-prize-2025/arc-agi_evaluation_challenges.json`.
- Logs and output images are saved in `outputs/`.

### 2. Configuration

Edit the `CONFIG` object in [`runner.py`](runner.py:41):

- `task_file`: Path to ARC challenge JSON.
- `task_ids`: Comma-separated list of task IDs.
- `num_of_tasks`: Max number of tasks.
- `offset_tasks`: Number of tasks to skip.
- `max_steps`: Max instruction/generator iterations per task.
- `use_images`: Include rendered grid images in LLM prompts.

---

## How It Works

1. **State Initialization**: Loads ARC tasks and initializes solver state for each.
2. **LangGraph Loop**:
   - **Instruction Node**: LLM proposes the next transformation step.
   - **Generator Node**: Applies instruction, updates outputs, and summarizes changes.
   - **Verifier Node**: Checks if outputs match expected; loops or completes.
3. **Logging & Visualization**: Logs all steps, token usage, and generates summary images.
4. **Evaluation**: If solutions are available, compares outputs and reports accuracy; otherwise, builds Kaggle submission.

---

## Customization & Extensibility

- **Prompts**: Modify system/user prompts in [`agents.py`](agents.py) for different LLMs or strategies.
- **Grid Rendering**: Adjust color maps and rendering in [`representation_utils.py`](representation_utils.py).
- **Graph Logic**: Extend or modify the solve loop in [`graph.py`](graph.py).
- **Logging**: Customize log formatting or output destinations in [`logging_utils.py`](logging_utils.py).

---

## References

- [ARC Prize](https://arcprize.org/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangChain](https://github.com/langchain-ai/langchain)
- [OpenAI API](https://platform.openai.com/docs/)
- [Kaggle ARC Competition](https://www.kaggle.com/competitions/abstraction-and-reasoning-challenge)

---

## Acknowledgments

- Inspired by the ARC Prize and the broader ARC research community.
- Built with LangGraph, LangChain, and OpenAI APIs.
