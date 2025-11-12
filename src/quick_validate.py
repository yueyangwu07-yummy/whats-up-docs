"""
Quick validation script for model evaluation on small sample.

This script evaluates the model on a small validation set (100 samples)
and computes ROUGE metrics with detailed statistics.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from rouge_score import rouge_scorer, scoring

# Import from train_led_large for compatibility
try:
    from .train_led_large import (
        clean_text,
        load_config,
        select_prompt_by_length,
    )
except ImportError:
    # Fallback for direct execution
    from train_led_large import (
        clean_text,
        load_config,
        select_prompt_by_length,
    )

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
    from peft import AutoPeftModelForSeq2SeqLM
except ImportError:
    AutoPeftModelForSeq2SeqLM = None  # type: ignore

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def load_model_and_tokenizer(config: Dict[str, Any]):
    """
    Load model and tokenizer from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_source = config.get("model_dir") or config.get("output_dir") or config.get("model_name")
    if not model_source:
        raise ValueError("Config must specify 'model_dir', 'output_dir', or 'model_name'")
    
    model_dir = Path(model_source)
    if model_dir.exists():
        LOGGER.info("Loading model from directory: %s", model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Check for PEFT adapter
        adapter_config = model_dir / "adapter_config.json"
        if adapter_config.exists() and AutoPeftModelForSeq2SeqLM is not None:
            LOGGER.info("Detected PEFT adapter")
            model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_dir)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    else:
        # Load from HuggingFace model name
        LOGGER.info("Loading model from HuggingFace: %s", model_source)
        tokenizer = AutoTokenizer.from_pretrained(model_source)
        
        quant_config = None
        if config.get("use_quantization", False):
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        
        model_kwargs: Dict[str, Any] = {}
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = config.get("device_map", "auto")
        
        model = AutoModelForSeq2SeqLM.from_pretrained(model_source, **model_kwargs)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if "device_map" not in model_kwargs:
        model.to(device)
    model.eval()
    
    return model, tokenizer


def generate_summary(
    model,
    tokenizer,
    text: str,
    config: Dict[str, Any],
    prompt_fn,
) -> str:
    """
    Generate summary for given text.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        text: Input text
        config: Configuration dictionary
        prompt_fn: Prompt function
        
    Returns:
        Generated summary
    """
    cleaned_text = clean_text(text)
    prompted = prompt_fn(cleaned_text)
    
    max_length = config.get("max_input_length", tokenizer.model_max_length)
    inputs = tokenizer(
        prompted,
        max_length=max_length,
        truncation=True,
        padding="longest",
        return_tensors="pt",
    )
    
    if config.get("use_global_attention", True):
        global_attention_mask = torch.zeros_like(inputs["input_ids"])
        global_attention_mask[:, 0] = 1
        inputs["global_attention_mask"] = global_attention_mask
    
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    generation_kwargs = {
        "max_length": config.get("generation_max_length", config.get("max_target_length", 256)),
        "num_beams": config.get("num_beams", 4),
        "do_sample": config.get("do_sample", False),
    }
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    
    summary = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]
    
    return summary.strip()


def compute_rouge_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute ROUGE metrics.
    
    Args:
        predictions: List of predicted summaries
        references: List of reference summaries
        
    Returns:
        Dictionary of ROUGE metrics
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    
    aggregator = scoring.BootstrapAggregator()
    for pred, ref in zip(predictions, references):
        scores = scorer.score(target=ref, prediction=pred)
        aggregator.add_scores(scores)
    
    result = aggregator.aggregate()
    metrics = {
        metric: {
            "precision": value.mid.precision,
            "recall": value.mid.recall,
            "fmeasure": value.mid.fmeasure,
        }
        for metric, value in result.items()
    }
    
    return metrics


def quick_validate(
    config_path: Path,
    data_path: Path,
    sample_size: int = 100,
    output_path: Optional[Path] = None,
) -> None:
    """
    Run quick validation on small sample.
    
    Args:
        config_path: Path to model configuration file
        data_path: Path to validation data CSV
        sample_size: Number of samples to evaluate (default: 100)
        output_path: Optional path to save detailed results
    """
    LOGGER.info("Loading configuration from %s", config_path)
    config = load_config(config_path)
    
    LOGGER.info("Loading validation data from %s", data_path)
    df = pd.read_csv(data_path)
    
    if "text" not in df.columns or "summary" not in df.columns:
        raise ValueError("Data must contain 'text' and 'summary' columns")
    
    # Clean text
    df["text"] = df["text"].apply(clean_text)
    df["summary"] = df["summary"].apply(clean_text)
    
    # Sample data
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        LOGGER.info("Sampled %d samples for validation", sample_size)
    else:
        LOGGER.info("Using all %d samples for validation", len(df))
    
    # Load model
    LOGGER.info("Loading model and tokenizer")
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Setup prompt function
    prompt_style = config.get("prompt_style", "auto")
    if prompt_style in ("auto", "academic", "structured", "concise", "detailed"):
        # For auto, we'll select per text
        if prompt_style == "auto":
            def prompt_fn(text: str):
                selected_fn = select_prompt_by_length(text, prompt_style="auto")
                return selected_fn(text)
        else:
            selected_fn = select_prompt_by_length("", prompt_style=prompt_style)
            prompt_fn = selected_fn
    else:
        prompt_fn = lambda x: x
    
    # Generate predictions
    LOGGER.info("Generating predictions...")
    predictions = []
    references = df["summary"].tolist()
    
    for idx, row in df.iterrows():
        try:
            pred = generate_summary(model, tokenizer, row["text"], config, prompt_fn)
            predictions.append(pred)
        except Exception as exc:
            LOGGER.exception("Error generating summary for sample %d: %s", idx, exc)
            predictions.append("")
        
        if (idx + 1) % 10 == 0:
            LOGGER.info("Processed %d/%d samples", idx + 1, len(df))
    
    # Compute metrics
    LOGGER.info("Computing ROUGE metrics...")
    metrics = compute_rouge_metrics(predictions, references)
    
    # Calculate statistics
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    for pred, ref in zip(predictions, references):
        scores = scorer.score(target=ref, prediction=pred)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)
    
    # Print results
    print("\n" + "=" * 80)
    print("QUICK VALIDATION RESULTS")
    print("=" * 80)
    print(f"\nNumber of samples: {len(df)}")
    print(f"\nROUGE Metrics (F-measure):")
    print(f"  ROUGE-1: {metrics['rouge1']['fmeasure']:.4f}")
    print(f"  ROUGE-2: {metrics['rouge2']['fmeasure']:.4f}")
    print(f"  ROUGE-L: {metrics['rougeL']['fmeasure']:.4f}")
    
    print(f"\nStatistics:")
    for metric_name, scores in [
        ("ROUGE-1", rouge1_scores),
        ("ROUGE-2", rouge2_scores),
        ("ROUGE-L", rougeL_scores),
    ]:
        print(f"  {metric_name}:")
        print(f"    Mean:   {np.mean(scores):.4f}")
        print(f"    Std:    {np.std(scores):.4f}")
        print(f"    Min:    {np.min(scores):.4f}")
        print(f"    Max:    {np.max(scores):.4f}")
    
    # Find best and worst examples
    print(f"\nTop 3 Best Examples (by ROUGE-2):")
    best_indices = np.argsort(rouge2_scores)[-3:][::-1]
    for rank, idx in enumerate(best_indices, 1):
        print(f"\n  Rank {rank} (ROUGE-2: {rouge2_scores[idx]:.4f}):")
        print(f"    Reference: {references[idx][:200]}...")
        print(f"    Prediction: {predictions[idx][:200]}...")
    
    print(f"\nTop 3 Worst Examples (by ROUGE-2):")
    worst_indices = np.argsort(rouge2_scores)[:3]
    for rank, idx in enumerate(worst_indices, 1):
        print(f"\n  Rank {rank} (ROUGE-2: {rouge2_scores[idx]:.4f}):")
        print(f"    Reference: {references[idx][:200]}...")
        print(f"    Prediction: {predictions[idx][:200]}...")
    
    print("\n" + "=" * 80)
    
    # Save detailed results
    if output_path:
        results = {
            "config": str(config_path),
            "data_path": str(data_path),
            "sample_size": len(df),
            "metrics": metrics,
            "statistics": {
                "rouge1": {
                    "mean": float(np.mean(rouge1_scores)),
                    "std": float(np.std(rouge1_scores)),
                    "min": float(np.min(rouge1_scores)),
                    "max": float(np.max(rouge1_scores)),
                },
                "rouge2": {
                    "mean": float(np.mean(rouge2_scores)),
                    "std": float(np.std(rouge2_scores)),
                    "min": float(np.min(rouge2_scores)),
                    "max": float(np.max(rouge2_scores)),
                },
                "rougeL": {
                    "mean": float(np.mean(rougeL_scores)),
                    "std": float(np.std(rougeL_scores)),
                    "min": float(np.min(rougeL_scores)),
                    "max": float(np.max(rougeL_scores)),
                },
            },
            "examples": [
                {
                    "index": int(idx),
                    "rouge1": float(rouge1_scores[idx]),
                    "rouge2": float(rouge2_scores[idx]),
                    "rougeL": float(rougeL_scores[idx]),
                    "reference": ref,
                    "prediction": pred,
                }
                for idx, (ref, pred) in enumerate(zip(references, predictions))
            ],
        }
        
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        LOGGER.info("Saved detailed results to %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Quick validation on small sample of validation data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to model configuration YAML file.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to validation data CSV (must contain 'text' and 'summary' columns).",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save detailed results JSON.",
    )
    
    args = parser.parse_args()
    
    try:
        quick_validate(
            config_path=args.config,
            data_path=args.data,
            sample_size=args.sample_size,
            output_path=args.output,
        )
    except Exception as exc:
        LOGGER.exception("Validation failed: %s", exc)
        raise


if __name__ == "__main__":
    main()

