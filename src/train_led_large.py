import argparse
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rouge_score import rouge_scorer, scoring
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def clean_text(text: Any) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_data(
    train_path: Path,
    seed: int,
    train_subset_size: Optional[int],
    eval_subset_size: Optional[int],
    val_ratio: Optional[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    full_df = pd.read_csv(train_path)
    LOGGER.info("Loaded %d training rows", len(full_df))

    if "text" in full_df.columns:
        full_df["text"] = full_df["text"].apply(clean_text)
    if "summary" in full_df.columns:
        full_df["summary"] = full_df["summary"].apply(clean_text)

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


def create_academic_prompt_structured(text: str) -> str:
    """
    Create a structured academic prompt template.
    Best for medium-length documents (5000-10000 words).
    
    Args:
        text: Input paper text
        
    Returns:
        Formatted prompt string
    """
    return (
        "Summarize the following academic research paper. Focus on:\n"
        "Main research question or objective\n"
        "Methodology used\n"
        "Key findings\n"
        "Conclusions and implications\n\n"
        f"Paper content:\n{text}\n\nSummary:"
    )


def create_academic_prompt_concise(text: str) -> str:
    """
    Create a concise academic prompt template.
    Best for short documents (<5000 words).
    
    Args:
        text: Input paper text
        
    Returns:
        Formatted prompt string
    """
    return (
        "Summarize the following academic paper concisely. "
        "Focus on the main contribution and key results.\n\n"
        f"{text}\n\n"
        "Summary:"
    )


def create_academic_prompt_detailed(text: str) -> str:
    """
    Create a detailed academic prompt template.
    Best for long documents (>10000 words).
    
    Args:
        text: Input paper text
        
    Returns:
        Formatted prompt string
    """
    return (
        "Provide a comprehensive summary of the following academic research paper. "
        "Include:\n"
        "1. Research background and motivation\n"
        "2. Research questions and objectives\n"
        "3. Methodology and experimental setup\n"
        "4. Main findings and results\n"
        "5. Discussion and implications\n"
        "6. Limitations and future work\n\n"
        f"Paper content:\n{text}\n\n"
        "Comprehensive Summary:"
    )


def select_prompt_by_length(text: str, prompt_style: str = "auto") -> Callable[[str], str]:
    """
    Select the appropriate prompt function based on document length or style preference.
    
    Args:
        text: Input paper text
        prompt_style: Prompt style preference. Options:
            - "auto": Automatically select based on document length
            - "structured": Use structured prompt (medium documents)
            - "concise": Use concise prompt (short documents)
            - "detailed": Use detailed prompt (long documents)
    
    Returns:
        Prompt function to use
    """
    if prompt_style == "auto":
        # Count words (approximate)
        word_count = len(text.split())
        
        if word_count < 5000:
            LOGGER.debug("Document length: %d words, selecting concise prompt", word_count)
            return create_academic_prompt_concise
        elif word_count <= 10000:
            LOGGER.debug("Document length: %d words, selecting structured prompt", word_count)
            return create_academic_prompt_structured
        else:
            LOGGER.debug("Document length: %d words, selecting detailed prompt", word_count)
            return create_academic_prompt_detailed
    elif prompt_style == "structured":
        return create_academic_prompt_structured
    elif prompt_style == "concise":
        return create_academic_prompt_concise
    elif prompt_style == "detailed":
        return create_academic_prompt_detailed
    else:
        LOGGER.warning("Unknown prompt_style '%s', defaulting to structured", prompt_style)
        return create_academic_prompt_structured


def create_academic_prompt(text: str) -> str:
    """
    Legacy function for backward compatibility.
    Uses structured prompt by default.
    
    Args:
        text: Input paper text
        
    Returns:
        Formatted prompt string
    """
    return create_academic_prompt_structured(text)


def tokenize_builder(
    tokenizer: AutoTokenizer,
    max_input_length: int,
    max_target_length: Optional[int],
    use_global_attention: bool,
    padding: str,
    prompt_fn: Optional[Callable[[str], str]] = None,
):
    def preprocess_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        inputs = examples["text"]
        if prompt_fn is not None:
            inputs = [prompt_fn(text) for text in inputs]

        model_inputs = tokenizer(
            inputs,
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


def empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_learning_rate(config: Dict[str, Any], use_lora: bool) -> float:
    if "learning_rate" in config and config["learning_rate"] is not None:
        return float(config["learning_rate"])
    return 5e-4 if use_lora else 3e-5


def main(config_path: Path):
    config = load_config(config_path)

    do_train = bool(config.get("do_train", True))
    do_eval = bool(config.get("do_eval", do_train))
    do_predict = bool(config.get("do_predict", True))
    cleanup_after_training = bool(config.get("cleanup_checkpoints", True))

    data_dir = Path(config.get("data_dir", "data"))
    train_path = data_dir / config.get("train_filename", "train.csv")
    test_path = data_dir / config.get("test_filename", "test_features.csv")
    output_dir = Path(config.get("output_dir", "models/led_large_best"))
    results_path = Path(config.get("results_path", output_dir / "final_metrics.json"))
    submission_path = Path(config.get("submission_path", "submission.csv"))

    output_dir.mkdir(parents=True, exist_ok=True)
    if do_train or do_eval:
        results_path.parent.mkdir(parents=True, exist_ok=True)
    if do_predict:
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

    train_df: Optional[pd.DataFrame] = None
    eval_df: Optional[pd.DataFrame] = None
    if do_train or do_eval:
        train_df, eval_df = load_data(
            train_path=train_path,
            seed=seed,
            train_subset_size=train_subset_size,
            eval_subset_size=eval_subset_size,
            val_ratio=val_ratio,
        )

    model_source = config.get("model_name", "allenai/led-large-16384")
    if not do_train:
        model_source = config.get("model_dir", model_source)

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.model_max_length = config.get("max_input_length", tokenizer.model_max_length)

    use_quantization = bool(config.get("use_quantization", False))
    quant_config = None
    if use_quantization:
        LOGGER.info("Initializing 4-bit quantization with BitsAndBytesConfig")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model_load_kwargs: Dict[str, Any] = {}
    if quant_config is not None:
        model_load_kwargs["quantization_config"] = quant_config
        model_load_kwargs["device_map"] = config.get("device_map", "auto")
    torch_dtype = config.get("torch_dtype")
    if torch_dtype is not None:
        dtype = getattr(torch, str(torch_dtype))
        model_load_kwargs["torch_dtype"] = dtype
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_source, **model_load_kwargs)
    except Exception as exc:
        LOGGER.exception("Failed to load model %s: %s", model_source, exc)
        empty_cuda_cache()
        raise

    max_input_length = config.get("max_input_length", 16384)
    max_target_length = config.get("max_target_length", 256)
    generation_max_length = config.get("generation_max_length", max_target_length)
    padding = config.get("tokenizer_padding", "longest")
    use_global_attention = config.get("use_global_attention", True)

    prompt_style = config.get("prompt_style", "none")
    prompt_fn: Optional[Callable[[str], str]] = None
    if prompt_style in ("academic", "auto", "structured", "concise", "detailed"):
        if prompt_style == "auto":
            LOGGER.info("Using auto prompt selection based on document length")
            # Create a wrapper that selects prompt based on text length
            def auto_prompt_wrapper(text: str) -> str:
                selected_fn = select_prompt_by_length(text, prompt_style="auto")
                return selected_fn(text)
            prompt_fn = auto_prompt_wrapper
        elif prompt_style == "academic":
            LOGGER.info("Using academic prompt (structured) for inputs")
            prompt_fn = create_academic_prompt_structured
        else:
            LOGGER.info("Using %s prompt style for inputs", prompt_style)
            # Get a sample text to determine the function (we'll override in tokenize_builder)
            prompt_fn = select_prompt_by_length("", prompt_style=prompt_style)
            # Create wrapper to handle per-text selection
            if prompt_style in ("structured", "concise", "detailed"):
                def style_prompt_wrapper(text: str) -> str:
                    selected_fn = select_prompt_by_length(text, prompt_style=prompt_style)
                    return selected_fn(text)
                prompt_fn = style_prompt_wrapper

    preprocess_train = tokenize_builder(
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        use_global_attention=use_global_attention,
        padding=padding,
        prompt_fn=prompt_fn,
    )

    tokenized_train = None
    tokenized_eval = None

    if do_train and train_df is not None:
        train_dataset = prepare_dataset(train_df, text_column="text", summary_column="summary")
        tokenized_train = train_dataset.map(
            preprocess_train,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train set",
        )

    if do_eval and eval_df is not None and not eval_df.empty:
        eval_dataset = prepare_dataset(eval_df, text_column="text", summary_column="summary")
        tokenized_eval = eval_dataset.map(
            preprocess_train,
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing validation set",
        )

    batch_size = config.get("batch_size", 1)
    eval_batch_size = config.get("eval_batch_size", batch_size)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 16)
    warmup_ratio = config.get("warmup_ratio", 0.1)
    warmup_steps = config.get("warmup_steps")
    epochs = config.get("epochs", 5)
    lr_scheduler_type = config.get("lr_scheduler_type", "cosine")

    use_lora = bool(config.get("use_lora", False))
    learning_rate = resolve_learning_rate(config, use_lora=use_lora)

    if use_quantization or use_lora:
        LOGGER.info("Preparing model for parameter-efficient fine-tuning")
        model = prepare_model_for_kbit_training(model)

    if use_lora:
        LOGGER.info("Enabling LoRA with r=%s alpha=%s", config.get("lora_r", 16), config.get("lora_alpha", 32))
        lora_config = LoraConfig(
            r=int(config.get("lora_r", 16)),
            lora_alpha=int(config.get("lora_alpha", 32)),
            target_modules=config.get(
                "lora_target_modules",
                ["q_proj", "v_proj", "k_proj", "o_proj"],
            ),
            lora_dropout=float(config.get("lora_dropout", 0.05)),
            bias=config.get("lora_bias", "none"),
            task_type="SEQ_2_SEQ_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if config.get("gradient_checkpointing", False):
        LOGGER.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch" if do_eval else "no",
        save_strategy="epoch" if do_train else "no",
        logging_strategy="steps",
        logging_steps=config.get("logging_steps", 50),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=config.get("weight_decay", 0.01),
        num_train_epochs=epochs,
        predict_with_generate=do_eval or do_predict,
        generation_max_length=generation_max_length,
        generation_num_beams=config.get("num_beams", 4),
        fp16=config.get("fp16", torch.cuda.is_available()),
        save_total_limit=config.get("save_total_limit", 1),
        load_best_model_at_end=do_train and do_eval,
        metric_for_best_model="eval_rouge2",
        greater_is_better=True,
        lr_scheduler_type=lr_scheduler_type,
        seed=seed,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        label_smoothing_factor=config.get("label_smoothing_factor", 0.0),
        dataloader_num_workers=config.get("dataloader_num_workers", 2),
        report_to=config.get("report_to", []),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    compute_metrics = build_rouge_metric(tokenizer) if do_eval else None

    callbacks = []
    if do_train and do_eval:
        early_stopping_patience = config.get("early_stopping_patience", 3)
        early_stopping_threshold = config.get("early_stopping_threshold", 0.0)
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            )
        )

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

    train_metrics = None
    eval_metrics = None

    resume_option = config.get("resume_from_checkpoint", True)
    resume_path: Optional[str] = None
    if do_train and resume_option:
        if isinstance(resume_option, str) and Path(resume_option).expanduser().exists():
            resume_path = str(Path(resume_option).expanduser())
        else:
            try:
                resume_path = get_last_checkpoint(str(output_dir))
            except OSError:
                resume_path = None
        if resume_path:
            LOGGER.info("Resuming training from checkpoint %s", resume_path)
        else:
            LOGGER.info("No checkpoint found in %s. Starting from scratch.", output_dir)

    if do_train:
        LOGGER.info("Starting LED training with %s", model_source)
        train_result = trainer.train(resume_from_checkpoint=resume_path)
        train_metrics = train_result.metrics
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()
        empty_cuda_cache()

        if do_eval and tokenized_eval is not None:
            LOGGER.info("Evaluating best checkpoint")
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
            empty_cuda_cache()

        LOGGER.info("Saving model to %s", output_dir)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        if cleanup_after_training:
            cleanup_checkpoints(output_dir)
        empty_cuda_cache()
    elif do_eval and tokenized_eval is not None:
        LOGGER.info("Running evaluation")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        empty_cuda_cache()

    results_payload: Dict[str, Any] = {}
    if train_metrics is not None:
        results_payload["train_metrics"] = train_metrics
    if eval_metrics is not None:
        results_payload["eval_metrics"] = eval_metrics
    if do_train and trainer.state.best_model_checkpoint:
        results_payload["best_checkpoint"] = trainer.state.best_model_checkpoint

    if results_payload:
        with results_path.open("w", encoding="utf-8") as handle:
            json.dump(results_payload, handle, indent=2)
            LOGGER.info("Saved metrics to %s", results_path)

    if not do_predict:
        return

    LOGGER.info("Preparing test dataset")
    test_df = pd.read_csv(test_path)
    if "text" in test_df.columns:
        test_df["text"] = test_df["text"].apply(clean_text)
    paper_ids = test_df["paper_id"].tolist()
    test_dataset = prepare_dataset(test_df, text_column="text")

    preprocess_test = tokenize_builder(
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_target_length=None,
        use_global_attention=use_global_attention,
        padding=padding,
        prompt_fn=prompt_fn,
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
    empty_cuda_cache()


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



