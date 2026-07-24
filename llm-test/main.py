#!/usr/bin/env python3
"""
Pantheon Lab LLM Evaluation Script
Runs all prompts against pre-configured models in LM Studio, then judges results with DeepSeek.
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from string import Template

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration – edit the models list to change which candidate models are tested
# ---------------------------------------------------------------------------
MODELS = [
    {
        "name": "Qwen3.5 9B",
        "model_id": "qwen/qwen3.5-9b",
        "quantization": "Q4_K_M",
        "params": "9B"
    },
    {
        "name": "Qwen3.5 4B",
        "model_id": "qwen/qwen3.5-4b",
        "quantization": "Q4_K_M",
        "params": "4B"
    },
    {
        "name": "GPT-OSS 20B",
        "model_id": "openai/gpt-oss-20b",
        "quantization": "MXFP4",
        "params": "20B"
    },
]

JUDGE_MODEL = "deepseek-v4-pro"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
PROMPTS_DIR = BASE_DIR / "prompts"
TEMPLATES_DIR = BASE_DIR / "templates"
RESULTS_DIR = BASE_DIR / "results"
JUDGE_RESULTS_DIR = BASE_DIR / "judge-results"

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
candidate_client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
)

judge_key = os.getenv("DEEPSEEK_API_KEY")
if not judge_key:
    raise RuntimeError("DEEPSEEK_API_KEY missing – set it in .env or as an environment variable")

judge_client = OpenAI(
    api_key=judge_key,
    base_url="https://api.deepseek.com",
)

# ---------------------------------------------------------------------------
# Load prompts and template
# ---------------------------------------------------------------------------
def load_template() -> Template:
    return Template((TEMPLATES_DIR / "eval_template.md").read_text(encoding="utf-8"))


def load_prompts() -> dict[str, str]:
    prompts = {}
    for f in sorted(PROMPTS_DIR.glob("*.txt")):
        if f.stem == "judge":
            continue  # skip the judge prompt, it is not a candidate prompt
        prompts[f.stem] = f.read_text(encoding="utf-8").strip()
    return prompts


def load_judge_prompt() -> Template:
    return Template((PROMPTS_DIR / "judge.txt").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------
def run_candidate(model: dict, prompts: dict[str, str], template: Template) -> list[dict]:
    logger.info("=" * 60)
    logger.info("Testing candidate: %s (%s)", model["name"], model["model_id"])
    logger.info("Prompts: %d", len(prompts))
    logger.info("=" * 60)

    results = []
    for prompt_name, prompt_text in prompts.items():
        logger.info("-" * 40)
        logger.info("Prompt: %s", prompt_name)
        start = time.perf_counter()

        try:
            resp = candidate_client.chat.completions.create(
                model=model["model_id"],
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.7,
                max_tokens=2048,
            )
            elapsed = time.perf_counter() - start
            result = {
                "model_name": model["name"],
                "model_id": model["model_id"],
                "prompt_name": prompt_name,
                "timestamp": datetime.now().isoformat(),
                "response_time": round(elapsed, 2),
                "tokens_prompt": resp.usage.prompt_tokens,
                "tokens_completion": resp.usage.completion_tokens,
                "tokens_total": resp.usage.total_tokens,
                "prompt": prompt_text,
                "response": resp.choices[0].message.content,
            }
        except Exception as e:
            logger.error("Failed: %s", e)
            result = {
                "model_name": model["name"],
                "model_id": model["model_id"],
                "prompt_name": prompt_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

        save_candidate_result(result, template, model)
        results.append(result)

    logger.info("=" * 60)
    logger.info("Candidate complete: %s", model["name"])
    return results


def save_candidate_result(result: dict, template: Template, model: dict) -> Path:
    safe_model = result["model_name"].replace(" ", "_")
    safe_prompt = result["prompt_name"]

    if "error" in result:
        path = RESULTS_DIR / f"{safe_model}_{safe_prompt}_ERROR.md"
        content = f"""# {result['model_name']} — {result['prompt_name']} — ERROR
**Timestamp:** {result['timestamp']}
**Error:** {result['error']}
"""
    else:
        path = RESULTS_DIR / f"{safe_model}_{safe_prompt}.md"
        content = template.substitute(
            model_name=result["model_name"],
            prompt_name=result["prompt_name"],
            model_id=result["model_id"],
            params=model["params"],
            quantization=model["quantization"],
            timestamp=result["timestamp"],
            response_time=result["response_time"],
            tokens_prompt=result["tokens_prompt"],
            tokens_completion=result["tokens_completion"],
            tokens_total=result["tokens_total"],
            prompt=result["prompt"],
            response=result["response"],
        )

    path.write_text(content, encoding="utf-8")
    logger.info("Saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Independent judging
# ---------------------------------------------------------------------------
def parse_response_from_md(path: Path) -> tuple[str, str] | None:
    """Extract prompt and response from a saved candidate Markdown file."""
    text = path.read_text(encoding="utf-8")
    prompt_match = re.search(r"## Prompt\s*\n\s*(.*?)\s*\n## Response", text, re.DOTALL)
    resp_match = re.search(r"## Response\s*\n\s*(.*?)\s*\n---", text, re.DOTALL)
    if not prompt_match or not resp_match:
        return None
    return prompt_match.group(1).strip(), resp_match.group(1).strip()


def run_judge(result_path: Path) -> dict | None:
    parsed = parse_response_from_md(result_path)
    if not parsed:
        logger.warning("Skipping unparseable result: %s", result_path.name)
        return None

    prompt_text, response_text = parsed
    judge_prompt = load_judge_prompt().substitute(prompt=prompt_text, response=response_text)

    resp = judge_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,
        max_tokens=4096,
    )
    content = resp.choices[0].message.content or ""
    if not content.strip():
        logger.error("Judge returned empty response for %s", result_path.name)
        return None
    content = content.strip()
    # Replace em-dashes that can break JSON or downstream display
    content = content.replace("\u2014", "--")
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.error("Judge returned invalid JSON. Raw response (%d chars): %s", len(content), content[:500])
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    JUDGE_RESULTS_DIR.mkdir(exist_ok=True)

    template = load_template()
    prompts = load_prompts()
    if not prompts:
        logger.error("No prompt files found in %s", PROMPTS_DIR)
        return

    for model in MODELS:
        run_candidate(model, prompts, template)

        # Judge the newly generated results for this candidate
        safe_name = model["name"].replace(" ", "_")
        for prompt_name in prompts:
            result_path = RESULTS_DIR / f"{safe_name}_{prompt_name}.md"
            if not result_path.exists():
                continue

            logger.info("Judging: %s — %s", model["name"], prompt_name)
            try:
                scores = run_judge(result_path)
            except Exception as e:
                logger.error("Judge failed for %s: %s", result_path.name, e)
                continue

            output = {
                "candidate_model": model["name"],
                "candidate_model_id": model["model_id"],
                "judge_model": JUDGE_MODEL,
                "prompt_name": prompt_name,
                "scores": scores,
            }
            out_path = JUDGE_RESULTS_DIR / f"{safe_name}_{prompt_name}_judge.json"
            out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Saved evaluation: %s", out_path)

    logger.info("=" * 60)
    logger.info("All models and prompts complete.")
    logger.info("Results:       %s", RESULTS_DIR.resolve())
    logger.info("Judge results: %s", JUDGE_RESULTS_DIR.resolve())


if __name__ == "__main__":
    main()
