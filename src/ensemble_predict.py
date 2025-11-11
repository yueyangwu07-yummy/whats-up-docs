import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

from train_led_large import clean_text, create_academic_prompt, load_config

try:
    from peft import AutoPeftModelForSeq2SeqLM
except ImportError:  # pragma: no cover
    AutoPeftModelForSeq2SeqLM = None  # type: ignore

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_prompt_fn(prompt_style: str):
    if prompt_style == "academic":
        return create_academic_prompt
    if prompt_style == "concise":
        return lambda text: (
            "Summarize the following paper concisely in 3-4 sentences.\n\n"
            f"{text}\n\n"
            "Summary:"
        )
    return lambda text: text


def build_inputs(
    tokenizer,
    text: str,
    max_length: int,
    padding: str,
    use_global_attention: bool,
) -> Dict[str, torch.Tensor]:
    inputs = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=padding,
        return_tensors="pt",
    )
    if use_global_attention and "input_ids" in inputs:
        global_attention_mask = torch.zeros_like(inputs["input_ids"])
        global_attention_mask[:, 0] = 1
        inputs["global_attention_mask"] = global_attention_mask
    return inputs


def load_model_and_tokenizer(config: Dict[str, Any]):
    model_dir = config.get("model_dir") or config.get("output_dir")
    if not model_dir:
        raise ValueError("Each config must specify 'model_dir' or 'output_dir'.")
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    quant_config = None
    if config.get("use_quantization", False):
        LOGGER.info("Loading %s with 4-bit quantization", model_dir)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model_kwargs: Dict[str, Any] = {}
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = config.get("device_map", "auto")

    adapter_config = model_dir / "adapter_config.json"
    if adapter_config.exists():
        if AutoPeftModelForSeq2SeqLM is None:
            raise ImportError("peft is required to load LoRA adapters.")
        LOGGER.info("Detected PEFT adapter in %s", model_dir)
        model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_dir, **model_kwargs)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, **model_kwargs)

    target_device = config.get(
        "device",
        "cuda" if torch.cuda.is_available() and "device_map" not in model_kwargs else "cpu",
    )
    if hasattr(model, "device") and str(model.device) == "meta":
        target_device = "cpu"
    if "device_map" not in model_kwargs:
        model.to(target_device)
    model.eval()

    if hasattr(model, "config") and getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def generate_summary(
    model,
    tokenizer,
    text: str,
    config: Dict[str, Any],
    prompt_fn,
) -> str:
    cleaned_text = clean_text(text)
    prompted = prompt_fn(cleaned_text)
    inputs = build_inputs(
        tokenizer=tokenizer,
        text=prompted,
        max_length=config.get("max_input_length", tokenizer.model_max_length),
        padding=config.get("tokenizer_padding", "longest"),
        use_global_attention=config.get("use_global_attention", True),
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    generation_kwargs = {
        "max_length": config.get(
            "generation_max_length",
            config.get("max_target_length", 256),
        ),
        "num_beams": config.get("num_beams", 4),
        "do_sample": config.get("do_sample", False),
        "temperature": config.get("temperature"),
        "top_p": config.get("top_p"),
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    summary = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]
    return summary.strip()


def select_best_summary(
    summaries: Sequence[str],
    reference: Optional[str],
    scorer: rouge_scorer.RougeScorer,
) -> str:
    normalized = [s.strip() for s in summaries if s.strip()]
    if not normalized:
        return ""
    freq = Counter(normalized)
    top_summary, top_count = freq.most_common(1)[0]
    if top_count > 1:
        return top_summary

    if reference:
        scores = [
            scorer.score(target=reference, prediction=summary)["rouge2"].fmeasure
            for summary in normalized
        ]
        best_idx = int(np.argmax(scores))
        return normalized[best_idx]

    if len(normalized) == 1:
        return normalized[0]

    consensus_scores = []
    for idx, candidate in enumerate(normalized):
        score_accum = 0.0
        for jdx, other in enumerate(normalized):
            if idx == jdx:
                continue
            score_accum += scorer.score(target=other, prediction=candidate)["rougeL"].fmeasure
        consensus_scores.append(score_accum / max(len(normalized) - 1, 1))
    best_idx = int(np.argmax(consensus_scores))
    return normalized[best_idx]


def ensemble_predict(
    configs: Sequence[Path],
    data_path: Path,
    output_path: Path,
    reference_column: Optional[str],
) -> None:
    df = pd.read_csv(data_path)
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")
    df["text"] = df["text"].apply(clean_text)
    paper_ids = df["paper_id"].tolist() if "paper_id" in df.columns else list(range(len(df)))

    models: List[Tuple[Any, Any, Dict[str, Any], Any]] = []
    for config_path in configs:
        try:
            config = load_config(config_path)
        except Exception as exc:
            LOGGER.exception("Failed to load config %s: %s", config_path, exc)
            continue
        try:
            model, tokenizer = load_model_and_tokenizer(config)
        except Exception as exc:
            LOGGER.exception("Failed to load model for %s: %s", config_path, exc)
            empty_cuda_cache()
            continue
        prompt_fn = resolve_prompt_fn(config.get("prompt_style", "none"))
        models.append((model, tokenizer, config, prompt_fn))

    if not models:
        raise RuntimeError("No models could be loaded for ensemble prediction.")

    scorer = rouge_scorer.RougeScorer(["rouge2", "rougeL"], use_stemmer=True)

    ensembled_summaries: List[str] = []
    all_predictions: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        texts: List[str] = []
        model_outputs: List[str] = []
        for model, tokenizer, config, prompt_fn in models:
            try:
                summary = generate_summary(model, tokenizer, row["text"], config, prompt_fn)
            except Exception as exc:
                LOGGER.exception(
                    "Generation failed on sample %s with model %s: %s",
                    idx,
                    config.get("output_dir") or config.get("model_dir"),
                    exc,
                )
                summary = ""
            model_outputs.append(summary)
            texts.append(summary)
        reference = None
        if reference_column and reference_column in row and pd.notna(row[reference_column]):
            reference = clean_text(row[reference_column])
        best_summary = select_best_summary(texts, reference, scorer)
        ensembled_summaries.append(best_summary)
        all_predictions.append(
            {
                "paper_id": paper_ids[idx],
                "reference": reference,
                "selected_summary": best_summary,
                "model_summaries": model_outputs,
            }
        )
        if (idx + 1) % 10 == 0:
            LOGGER.info("Processed %d samples", idx + 1)

    submission_df = pd.DataFrame(
        {
            "paper_id": paper_ids,
            "summary": ensembled_summaries,
        }
    )
    submission_df.to_csv(output_path, index=False)
    LOGGER.info("Saved ensemble summaries to %s", output_path)

    details_path = output_path.parent / f"{output_path.stem}_details.jsonl"
    with details_path.open("w", encoding="utf-8") as handle:
        for record in all_predictions:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    LOGGER.info("Saved detailed prediction traces to %s", details_path)

    for model, _, _, _ in models:
        del model
    empty_cuda_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Run ensemble summarization across multiple trained models.")
    parser.add_argument(
        "--config_paths",
        type=Path,
        nargs="+",
        required=True,
        help="List of configuration YAML files associated with trained models.",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="CSV file containing at least 'text' column and optional 'paper_id'.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("ensemble_submission.csv"),
        help="Destination CSV path for ensemble predictions.",
    )
    parser.add_argument(
        "--reference_column",
        type=str,
        default=None,
        help="Optional column name containing reference summaries for validation-time ROUGE selection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        ensemble_predict(
            configs=args.config_paths,
            data_path=args.data_path,
            output_path=args.output_path,
            reference_column=args.reference_column,
        )
    except Exception as exc:
        LOGGER.exception("Ensemble prediction failed: %s", exc)
        raise


if __name__ == "__main__":
    main()

