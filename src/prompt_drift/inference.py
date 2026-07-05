"""Model inference for Gemini API and local HuggingFace models."""

import time
import json
from pathlib import Path


def run_gemini_inference(prompts: list[dict], api_key: str, model_name: str, max_tokens: int = 256) -> list[dict]:
    """Run inference on all prompts using Gemini API.

    Args:
        prompts: List of prompt dicts from build_prompt_matrix
        api_key: Gemini API key
        model_name: Gemini model identifier
        max_tokens: Maximum tokens to generate

    Returns:
        List of result dicts with prompt info and model response
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    results = []

    for prompt in prompts:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt["text"],
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.0,
                ),
            )
            result_text = response.text if response.text else ""
        except Exception as e:
            result_text = f"[ERROR: {str(e)}]"

        results.append({
            **prompt,
            "model": model_name,
            "response": result_text,
        })
        time.sleep(1)

    return results


def run_local_inference(prompts: list[dict], model_name: str, max_tokens: int = 256) -> list[dict]:
    """Run inference on all prompts using a local HuggingFace model.

    Args:
        prompts: List of prompt dicts from build_prompt_matrix
        model_name: HuggingFace model identifier
        max_tokens: Maximum tokens to generate

    Returns:
        List of result dicts with prompt info and model response
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    results = []

    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt["text"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            temperature=0.0,
            do_sample=False,
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        results.append({
            **prompt,
            "model": model_name,
            "response": response,
        })

    return results


def save_results(results: list[dict], output_path: str) -> None:
    """Save inference results to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(path: str) -> list[dict]:
    """Load inference results from JSON."""
    with open(path, "r") as f:
        return json.load(f)
