import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from rouge_score import rouge_scorer, scoring
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_data(
    train_path: Path,
    seed: int,
    train_subset_size: Optional[int],
    eval_subset_size: Optional[int],
    val_ratio: Optional[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    full_df = pd.read_csv(train_path)
    LOGGER.info("Loaded %d training rows", len(full_df))

    if train_subset_size is not None:
        subset_df = full_df.sample(
            n=min(train_subset_size, len(full_df)),
            random_state=seed,
        )
        LOGGER.info("Subsampled training data to %d rows", len(subset_df))
    else:
        subset_df = full_df

    if val_ratio is not None and 0.0 < val_ratio < 1.0:
        train_subset, eval_subset = train_test_split(
            subset_df,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
        )
    elif eval_subset_size is not None and eval_subset_size > 0:
        eval_subset = subset_df.sample(
            n=min(eval_subset_size, len(subset_df)),
            random_state=seed,
        )
        train_subset = subset_df.drop(eval_subset.index)
        if len(train_subset) == 0:
            raise ValueError(
                "Evaluation subset size consumes all training samples. "
                "Reduce eval_subset_size or provide a val_ratio instead."
            )
    else:
        train_subset = subset_df
        eval_subset = pd.DataFrame()

    LOGGER.info(
        "Using %d samples for training and %d samples for validation",
        len(train_subset),
        len(eval_subset),
    )

    return train_subset.reset_index(drop=True), eval_subset.reset_index(drop=True)


def prepare_dataset(df: pd.DataFrame) -> Dataset:
    dataset = Dataset.from_pandas(
        df[["text", "summary"]],
        preserve_index=False,
    )
    return dataset


def tokenize_function_builder(
    tokenizer: AutoTokenizer,
    max_input_length: int,
    max_target_length: int,
    use_global_attention: bool = False,
):
    def preprocess_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        inputs = tokenizer(
            examples["text"],
            max_length=max_input_length,
            truncation=True,
            padding="max_length" if use_global_attention else False,
        )
        targets = tokenizer(
            text_target=examples["summary"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length" if use_global_attention else False,
        )
        if use_global_attention:
            global_attention_mask = []
            for input_ids in inputs["input_ids"]:
                mask = [0] * len(input_ids)
                if mask:
                    mask[0] = 1
                global_attention_mask.append(mask)
            inputs["global_attention_mask"] = global_attention_mask
        inputs["labels"] = targets["input_ids"]
        return inputs

    return preprocess_function


def build_rouge_metric(tokenizer: AutoTokenizer):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace ignored index in labels with pad token id so they can be decoded
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        aggregator = scoring.BootstrapAggregator()
        for pred, label in zip(decoded_preds, decoded_labels):
            aggregator.add_scores(scorer.score(target=label, prediction=pred))

        result = aggregator.aggregate()
        metrics = {
            key: value.mid.fmeasure for key, value in result.items()
        }
        # Also report average generated length for visibility
        metrics["gen_len"] = np.mean(
            [
                len(tokenizer.encode(pred, add_special_tokens=False))
                for pred in decoded_preds
            ]
        )
        return metrics

    return compute_metrics


def main(config_path: Path):
    config = load_config(config_path)

    data_dir = Path(config.get("data_dir", "data"))
    train_path = data_dir / config.get("train_filename", "train.csv")

    output_dir = Path(config.get("output_dir", "models/baseline_t5"))
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = config.get("seed", 42)
    set_seed(seed)

    train_subset_size = config.get("train_subset_size")
    if train_subset_size in (0, "full", "Full", "FULL"):
        train_subset_size = None
    eval_subset_size = config.get("eval_subset_size")
    if eval_subset_size in (0, "full", "Full", "FULL"):
        eval_subset_size = None
    val_ratio = config.get("val_ratio")
    if isinstance(val_ratio, str):
        try:
            val_ratio = float(val_ratio)
        except ValueError:
            val_ratio = None

    train_df, eval_df = load_data(
        train_path=train_path,
        seed=seed,
        train_subset_size=train_subset_size,
        eval_subset_size=eval_subset_size,
        val_ratio=val_ratio,
    )

    model_name = config.get("model_name", "t5-small")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    max_input_length = config.get("max_input_length", 512)
    max_target_length = config.get("max_target_length", 150)

    use_global_attention = config.get("use_global_attention", False)
    preprocess_function = tokenize_function_builder(
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        use_global_attention=use_global_attention,
    )

    train_dataset = prepare_dataset(train_df)
    eval_dataset = prepare_dataset(eval_df) if len(eval_df) > 0 else None

    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    if eval_dataset is not None:
        tokenized_eval = eval_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=eval_dataset.column_names,
        )
    else:
        tokenized_eval = None

    batch_size = config.get("batch_size", 2)
    eval_batch_size = config.get("eval_batch_size", batch_size)
    logging_steps = config.get("logging_steps", 25)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch" if tokenized_eval is not None else "no",
        save_strategy="epoch" if tokenized_eval is not None else "no",
        logging_strategy="steps",
        logging_steps=logging_steps,
        learning_rate=config.get("learning_rate", 5e-4),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        weight_decay=config.get("weight_decay", 0.01),
        save_total_limit=config.get("save_total_limit", 2),
        num_train_epochs=config.get("epochs", 3),
        predict_with_generate=True,
        generation_max_length=config.get("generation_max_length", max_target_length),
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        warmup_steps=config.get("warmup_steps", 0),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        seed=seed,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    compute_metrics = build_rouge_metric(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if tokenized_eval is not None else None,
    )

    LOGGER.info("Starting training")
    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if tokenized_eval is not None:
        LOGGER.info("Running evaluation on validation subset")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    else:
        eval_metrics = {}

    metrics_path = output_dir / "metrics_summary.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train_metrics": metrics,
                "eval_metrics": eval_metrics,
            },
            f,
            indent=2,
        )
    LOGGER.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune T5-small baseline for document summarization.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline_config.yaml"),
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()
    main(args.config)

