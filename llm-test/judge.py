#!/usr/bin/env python3
"""Blindly score saved candidate-model responses with DeepSeek."""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from string import Template

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).parent
DEFAULT_RESULTS_DIR = BASE_DIR / "results"
DEFAULT_OUTPUT_DIR = BASE_DIR / "judge-results"
JUDGE_PROMPT_PATH = BASE_DIR / "prompts" / "judge.txt"


def load_judge_prompt() -> Template:
    return Template(JUDGE_PROMPT_PATH.read_text(encoding="utf-8"))


def parse_result(path: Path) -> tuple[str, str]:
    """Extract the original prompt and candidate response from a result file."""
    text = path.read_text(encoding="utf-8")
    prompt_match = re.search(r"## Prompt\s*\n\s*(.*?)\s*\n## Response", text, re.DOTALL)
    response_match = re.search(r"## Response\s*\n\s*(.*?)\s*\n---", text, re.DOTALL)
    if not prompt_match or not response_match:
        raise ValueError(f"Could not parse prompt/response sections in {path}")
    return prompt_match.group(1).strip(), response_match.group(1).strip()


def parse_json_response(content: str) -> dict:
    """Parse JSON, allowing a model to wrap it in a Markdown code fence."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned).strip()
    return json.loads(cleaned)


def judge_result(client: OpenAI, model: str, prompt: str, response: str) -> dict:
    judge_prompt = load_judge_prompt().substitute(prompt=prompt, response=response)
    result = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,
        max_tokens=1200,
    )
    content = result.choices[0].message.content or ""
    return parse_json_response(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidate LLM results with DeepSeek")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="deepseek-chat", help="DeepSeek judge model ID")
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing from the environment or .env")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_files = sorted(args.results_dir.glob("*.md"))
    if not result_files:
        raise RuntimeError(f"No candidate result files found in {args.results_dir}")

    for result_file in result_files:
        if result_file.name.endswith("_ERROR.md"):
            logger.warning("Skipping failed candidate result: %s", result_file.name)
            continue

        prompt, response = parse_result(result_file)
        scores = judge_result(client, args.model, prompt, response)
        output = {
            "source_file": result_file.name,
            "judge_model": args.model,
            "scores": scores,
        }
        output_path = args.output_dir / f"{result_file.stem}_judge.json"
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        logger.info("Saved evaluation: %s", output_path)


if __name__ == "__main__":
    main()
