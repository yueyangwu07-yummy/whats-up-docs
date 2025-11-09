import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from rouge_score import rouge_scorer, scoring
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_data(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=seed, shuffle=True)
    LOGGER.info("Dataset split into %d train rows and %d validation rows", len(train_df), len(val_df))
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def chunk_tokens(token_ids: List[int], chunk_size: int, overlap: int) -> List[List[int]]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap.")
    stride = chunk_size - overlap
    chunks: List[List[int]] = []
    for start in range(0, len(token_ids), stride):
        end = start + chunk_size
        chunk = token_ids[start:end]
        if not chunk:
            continue
        chunks.append(chunk)
        if end >= len(token_ids):
            break
    return chunks


def generate_summary(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
    max_input_length: int,
    generation_max_length: int,
) -> str:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_length=generation_max_length,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def generate_chunked_summary(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
    chunk_size: int,
    chunk_overlap: int,
    max_target_length: int,
    generation_max_length: int,
    final_max_input_length: int,
) -> str:
    token_ids = tokenizer.encode(text, truncation=False)
    if len(token_ids) <= chunk_size:
        return generate_summary(
            model=model,
            tokenizer=tokenizer,
            text=text,
            device=device,
            max_input_length=final_max_input_length,
            generation_max_length=generation_max_length,
        )

    chunk_ids = chunk_tokens(token_ids, chunk_size=chunk_size, overlap=chunk_overlap)
    chunk_summaries: List[str] = []
    for ids in chunk_ids:
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, device=device)
        with torch.no_grad():
            summary_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_target_length,
                num_beams=4,
                early_stopping=True,
            )
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        chunk_summaries.append(summary_text.strip())

    combined_summary_input = " ".join(chunk_summaries)
    return generate_summary(
        model=model,
        tokenizer=tokenizer,
        text=combined_summary_input,
        device=device,
        max_input_length=final_max_input_length,
        generation_max_length=generation_max_length,
    )


def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()
    for pred, ref in zip(predictions, references):
        aggregator.add_scores(scorer.score(target=ref, prediction=pred))
    result = aggregator.aggregate()
    return {metric: scores.mid.fmeasure for metric, scores in result.items()}


def main(config_path: Path):
    config = load_config(config_path)

    model_dir = Path(config["model_dir"])
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory {model_dir} does not exist.")

    data_path = Path(config["data_path"])
    df = pd.read_csv(data_path)

    val_ratio = float(config.get("val_ratio", 0.1))
    seed = int(config.get("seed", 42))
    set_seed(seed)

    _, val_df = split_data(df, val_ratio=val_ratio, seed=seed)
    eval_limit = config.get("eval_sample_limit")
    if eval_limit:
        val_df = val_df.head(int(eval_limit))
        LOGGER.info("Restricting evaluation to first %d samples", len(val_df))

    LOGGER.info("Loading model from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    chunk_size = int(config.get("chunk_size", 400))
    chunk_overlap = int(config.get("chunk_overlap", 100))
    max_input_length = int(config.get("max_input_length", 512))
    max_target_length = int(config.get("max_target_length", 150))
    generation_max_length = int(config.get("generation_max_length", 150))

    direct_predictions: List[str] = []
    chunk_predictions: List[str] = []
    references: List[str] = []

    LOGGER.info("Running direct summarization baseline")
    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Direct"):
        direct_summary = generate_summary(
            model=model,
            tokenizer=tokenizer,
            text=row["text"],
            device=device,
            max_input_length=max_input_length,
            generation_max_length=generation_max_length,
        )
        direct_predictions.append(direct_summary)
        references.append(row["summary"])

    LOGGER.info("Running chunked summarization (two-stage)")
    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Chunked"):
        chunk_summary = generate_chunked_summary(
            model=model,
            tokenizer=tokenizer,
            text=row["text"],
            device=device,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_target_length=max_target_length,
            generation_max_length=generation_max_length,
            final_max_input_length=max_input_length,
        )
        chunk_predictions.append(chunk_summary)

    LOGGER.info("Computing ROUGE metrics")
    direct_metrics = compute_rouge(direct_predictions, references)
    chunk_metrics = compute_rouge(chunk_predictions, references)

    output_path = Path(config.get("output_path", "experiments/chunking_results.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "model_dir": str(model_dir),
        "val_samples": len(val_df),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "direct_metrics": direct_metrics,
        "chunked_metrics": chunk_metrics,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    LOGGER.info("Saved results to %s", output_path)
    LOGGER.info("Direct summary ROUGE: %s", direct_metrics)
    LOGGER.info("Chunked summary ROUGE: %s", chunk_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate chunk-based summarization strategy.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/chunking_config.yaml"),
        help="Path to chunking evaluation configuration file.",
    )
    args = parser.parse_args()
    main(args.config)

