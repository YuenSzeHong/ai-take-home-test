#!/usr/bin/env python3
"""
Pantheon Lab LLM Evaluation Script
Auto-runs all prompts for a single model, saves responses as markdown files.
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from string import Template

from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

BASE_DIR = Path(__file__).parent
PROMPTS_DIR = BASE_DIR / "prompts"
TEMPLATES_DIR = BASE_DIR / "templates"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


def load_template() -> Template:
    """Load markdown template."""
    template_path = TEMPLATES_DIR / "eval_template.md"
    if not template_path.exists():
        logger.error("Template not found: %s", template_path)
        raise FileNotFoundError(f"Template missing: {template_path}")
    return Template(template_path.read_text(encoding="utf-8"))


def load_prompts() -> dict[str, str]:
    """Load all prompt files from prompts/ directory."""
    prompts = {}
    for f in sorted(PROMPTS_DIR.glob("*.txt")):
        prompts[f.stem] = f.read_text(encoding="utf-8").strip()
    if not prompts:
        logger.warning("No prompt files found in %s", PROMPTS_DIR)
    return prompts


def test_prompt(model_name: str, model_id: str, prompt_name: str, prompt_text: str) -> dict:
    """Test a single prompt."""
    logger.info("-" * 50)
    logger.info("Prompt: %s", prompt_name)

    start = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.7,
            max_tokens=2048
        )

        elapsed = time.perf_counter() - start

        return {
            "model_name": model_name,
            "model_id": model_id,
            "prompt_name": prompt_name,
            "timestamp": datetime.now().isoformat(),
            "response_time": round(elapsed, 2),
            "tokens_prompt": response.usage.prompt_tokens,
            "tokens_completion": response.usage.completion_tokens,
            "tokens_total": response.usage.total_tokens,
            "prompt": prompt_text,
            "response": response.choices[0].message.content
        }

    except Exception as e:
        logger.error("Failed: %s", e)
        return {
            "model_name": model_name,
            "model_id": model_id,
            "prompt_name": prompt_name,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def save_markdown(result: dict, template: Template, quantization: str, params: str) -> Path:
    """Save result as markdown using template."""
    safe_model = result["model_name"].replace(" ", "_")
    safe_prompt = result["prompt_name"]

    if "error" in result:
        filepath = RESULTS_DIR / f"{safe_model}_{safe_prompt}_ERROR.md"
        content = f"""# {result['model_name']} — {result['prompt_name']} — ERROR

**Timestamp:** {result['timestamp']}
**Error:** {result['error']}
"""
    else:
        filepath = RESULTS_DIR / f"{safe_model}_{safe_prompt}.md"
        content = template.substitute(
            model_name=result["model_name"],
            prompt_name=result["prompt_name"],
            model_id=result["model_id"],
            params=params,
            quantization=quantization,
            timestamp=result["timestamp"],
            response_time=result["response_time"],
            tokens_prompt=result["tokens_prompt"],
            tokens_completion=result["tokens_completion"],
            tokens_total=result["tokens_total"],
            prompt=result["prompt"],
            response=result["response"]
        )

    filepath.write_text(content, encoding="utf-8")
    logger.info("Saved: %s", filepath)
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM models via LM Studio")
    parser.add_argument("--name", required=True, help="Model display name")
    parser.add_argument("--model-id", required=True, help="Model ID in LM Studio")
    parser.add_argument("--quantization", default="Q4_K_M")
    parser.add_argument("--params", required=True, help="e.g., 9B")

    args = parser.parse_args()

    template = load_template()
    prompts = load_prompts()

    if not prompts:
        logger.error("No prompts to test. Create .txt files in %s/", PROMPTS_DIR)
        return

    logger.info("=" * 60)
    logger.info("Testing: %s", args.name)
    logger.info("Prompts: %d", len(prompts))
    logger.info("=" * 60)

    for prompt_name, prompt_text in prompts.items():
        result = test_prompt(args.name, args.model_id, prompt_name, prompt_text)
        save_markdown(result, template, args.quantization, args.params)

    logger.info("=" * 60)
    logger.info("All prompts complete for %s", args.name)
    logger.info("Results: %s", RESULTS_DIR.resolve())
    logger.info("Next: Switch model in LM Studio, then run again.")


if __name__ == "__main__":
    main()
