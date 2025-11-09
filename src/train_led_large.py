import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
    EarlyStoppingCallback,
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
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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
        train_subset = full_df.sample(
            n=min(train_subset_size, len(full_df)),
            random_state=seed,
        )
        LOGGER.info("Subsampled train set to %d rows", len(train_subset))
    else:
        train_subset = full_df

    if val_ratio is not None and 0.0 < val_ratio < 1.0:
        train_df, eval_df = train_test_split(
            train_subset,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
        )
    elif eval_subset_size is not None and eval_subset_size > 0:
        eval_df = train_subset.sample(
            n=min(eval_subset_size, len(train_subset)),
            random_state=seed,
        )
        train_df = train_subset.drop(eval_df.index)
        if len(train_df) == 0:
            raise ValueError(
                "Evaluation subset size consumes all training samples. "
                "Reduce eval_subset_size or specify val_ratio."
            )
    else:
        raise ValueError(
            "Validation split not provided. Please set val_ratio or eval_subset_size."
        )

    LOGGER.info(
        "Using %d samples for training and %d samples for validation",
        len(train_df),
        len(eval_df),
    )
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


def prepare_dataset(df: pd.DataFrame, text_column: str, summary_column: Optional[str] = None) -> Dataset:
    if summary_column:
        return Dataset.from_pandas(
            df[[text_column, summary_column]],
            preserve_index=False,
        )
    return Dataset.from_pandas(
        df[[text_column]],
        preserve_index=False,
    )


def tokenize_builder(
    tokenizer: AutoTokenizer,
    max_input_length: int,
    max_target_length: Optional[int],
    use_global_attention: bool,
    padding: str,
):
    def preprocess_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        model_inputs = tokenizer(
            examples["text"],
            max_length=max_input_length,
            truncation=True,
            padding=padding,
        )

        if use_global_attention:
            global_attention_mask = []
            input_ids = model_inputs["input_ids"]
            for ids in input_ids:
                mask = [0] * len(ids)
                if mask:
                    mask[0] = 1
                global_attention_mask.append(mask)
            model_inputs["global_attention_mask"] = global_attention_mask

        if max_target_length is not None and "summary" in examples:
            labels = tokenizer(
                text_target=examples["summary"],
                max_length=max_target_length,
                truncation=True,
                padding=padding,
            )
            model_inputs["labels"] = labels["input_ids"]

        return model_inputs

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

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        aggregator = scoring.BootstrapAggregator()
        for pred, label in zip(decoded_preds, decoded_labels):
            aggregator.add_scores(scorer.score(target=label, prediction=pred))

        result = aggregator.aggregate()
        metrics = {metric: value.mid.fmeasure for metric, value in result.items()}
        metrics["gen_len"] = np.mean(
            [
                len(tokenizer.encode(text, add_special_tokens=False))
                for text in decoded_preds
            ]
        )
        return metrics

    return compute_metrics


def cleanup_checkpoints(output_dir: Path):
    for path in output_dir.glob("checkpoint-*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def main(config_path: Path):
    config = load_config(config_path)

    data_dir = Path(config.get("data_dir", "data"))
    train_path = data_dir / config.get("train_filename", "train.csv")
    test_path = data_dir / config.get("test_filename", "test_features.csv")
    output_dir = Path(config.get("output_dir", "models/led_large_best"))
    results_path = Path(config.get("results_path", output_dir / "final_metrics.json"))
    submission_path = Path(config.get("submission_path", "submission.csv"))

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.parent.mkdir(parents=True, exist_ok=True)

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

    model_name = config.get("model_name", "allenai/led-large-16384")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.model_max_length = config.get("max_input_length", tokenizer.model_max_length)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    max_input_length = config.get("max_input_length", 16384)
    max_target_length = config.get("max_target_length", 256)
    generation_max_length = config.get("generation_max_length", max_target_length)
    padding = config.get("tokenizer_padding", "longest")
    use_global_attention = config.get("use_global_attention", True)

    train_dataset = prepare_dataset(train_df, text_column="text", summary_column="summary")
    eval_dataset = prepare_dataset(eval_df, text_column="text", summary_column="summary")

    preprocess_train = tokenize_builder(
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        use_global_attention=use_global_attention,
        padding=padding,
    )

    tokenized_train = train_dataset.map(
        preprocess_train,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train set",
    )

    tokenized_eval = eval_dataset.map(
        preprocess_train,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing validation set",
    )

    batch_size = config.get("batch_size", 1)
    eval_batch_size = config.get("eval_batch_size", batch_size)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4)
    warmup_ratio = config.get("warmup_ratio")
    warmup_steps = config.get("warmup_steps")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=config.get("logging_steps", 50),
        learning_rate=config.get("learning_rate", 3e-5),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=config.get("weight_decay", 0.01),
        num_train_epochs=config.get("epochs", 3),
        predict_with_generate=True,
        generation_max_length=generation_max_length,
        generation_num_beams=config.get("num_beams", 4),
        fp16=config.get("fp16", torch.cuda.is_available()),
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_rougeL",
        greater_is_better=True,
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        seed=seed,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        label_smoothing_factor=config.get("label_smoothing_factor", 0.0),
        dataloader_num_workers=config.get("dataloader_num_workers", 2),
        report_to=config.get("report_to", []),
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    compute_metrics = build_rouge_metric(tokenizer)

    early_stopping_patience = config.get("early_stopping_patience", 2)
    early_stopping_threshold = config.get("early_stopping_threshold", 0.0)

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )
    ]

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    LOGGER.info("Starting LED training with %s", model_name)
    train_result = trainer.train()
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    LOGGER.info("Evaluating best checkpoint")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    LOGGER.info("Saving best model to %s", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    cleanup_checkpoints(output_dir)

    results_payload = {
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
    }

    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results_payload, handle, indent=2)
        LOGGER.info("Saved final metrics to %s", results_path)

    LOGGER.info("Preparing test dataset")
    test_df = pd.read_csv(test_path)
    paper_ids = test_df["paper_id"].tolist()
    test_dataset = prepare_dataset(test_df, text_column="text")

    preprocess_test = tokenize_builder(
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_target_length=None,
        use_global_attention=use_global_attention,
        padding=padding,
    )

    tokenized_test = test_dataset.map(
        preprocess_test,
        batched=True,
        remove_columns=test_dataset.column_names,
        desc="Tokenizing test set",
    )

    LOGGER.info("Generating predictions for submission")
    predictions = trainer.predict(
        tokenized_test,
        max_length=generation_max_length,
        num_beams=config.get("num_beams", 4),
    )

    decoded_predictions = tokenizer.batch_decode(
        predictions.predictions,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    submission_df = pd.DataFrame(
        {
            "paper_id": paper_ids,
            "summary": [prediction.strip() for prediction in decoded_predictions],
        }
    )
    submission_df.to_csv(submission_path, index=False)
    LOGGER.info("Saved submission file to %s", submission_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LED-large summarization model and generate submission."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/led_large_config.yaml"),
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()
    main(args.config)



