"""Mapping between Skill Markdown and an optimizable DSPy program.

The Skill Markdown becomes the instructions of the single predictor component
that GEPA optimizes. Exporting a candidate reads the optimized component's
instructions back out, so the digest of the exported Markdown and the content
the runtime delivers are the same bytes.
"""

from __future__ import annotations

from typing import Any

import dspy


def build_skill_program(skill_markdown: str) -> dspy.Predict:
    """Build a Predict whose instructions are exactly the Skill Markdown."""
    predictor = dspy.Predict("task -> answer")
    predictor.signature.instructions = skill_markdown
    return predictor


def export_skill_markdown(program: dspy.Predict) -> str:
    """Export the optimized component back to Skill Markdown bytes."""
    instructions = program.signature.instructions
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("Optimized Skill component has no instructions to export.")
    return instructions


def skill_examples(records: list[dict[str, Any]]) -> list[dspy.Example]:
    """Convert eval JSONL records (with keyword answers) to DSPy examples."""
    examples: list[dspy.Example] = []
    for record in records:
        task = record.get("input")
        if not isinstance(task, str):
            continue
        gold = record.get("target")
        examples.append(
            dspy.Example(
                task=task, answer=str(gold if gold is not None else "")
            ).with_inputs("task")
        )
    return examples
