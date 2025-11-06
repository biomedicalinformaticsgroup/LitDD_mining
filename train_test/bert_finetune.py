#!/usr/bin/env python3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
import argparse
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
import evaluate
import torch

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def main(args):
    ds_bert = load_from_disk(args.train_ds_dir)  
    ds_test = load_from_disk(args.test_ds_dir)   

    # create a small validation split (stratified by label)
    split = ds_bert.train_test_split(
        test_size=args.val_size,
        seed=args.seed,
        stratify_by_column="label"
    )
    ds_train = split["train"]
    ds_val   = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.input_model)

    # tokenize: truncation only; dynamic padding via collator
    def preprocess_function(examples):
        return tokenizer(
            examples["tiab"],
            truncation=True,
            max_length=512,
        )

    keep_cols = {"tiab", "label"}
    tokenized_train = ds_train.map(
        preprocess_function,
        batched=True,
        remove_columns=[c for c in ds_train.column_names if c not in keep_cols]
    )
    tokenized_val = ds_val.map(
        preprocess_function,
        batched=True,
        remove_columns=[c for c in ds_val.column_names if c not in keep_cols]
    )
    tokenized_test = ds_test.map(
        preprocess_function,
        batched=True,
        remove_columns=[c for c in ds_test.column_names if c not in keep_cols]
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    acc_metric = evaluate.load("accuracy")
    prec_metric = evaluate.load("precision")
    rec_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "eval_accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
            "eval_precision": prec_metric.compute(predictions=preds, references=labels)["precision"],
            "eval_recall": rec_metric.compute(predictions=preds, references=labels)["recall"],
            "eval_f1": f1_metric.compute(predictions=preds, references=labels)["f1"],
        }

    model = AutoModelForSequenceClassification.from_pretrained(args.input_model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,               
        per_device_train_batch_size=args.train_bs,      
        per_device_eval_batch_size=args.eval_bs,        
        num_train_epochs=args.epochs,                   
        weight_decay=args.weight_decay,                 
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        metric_for_best_model="eval_f1",
        load_best_model_at_end=True,
        greater_is_better=True,
        seed=args.seed,
        report_to=[],
        logging_steps=100,
        dataloader_num_workers=max(1, os.cpu_count() // 2),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,  
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))

    print(f"[Info] Train={len(ds_train)} rows, Val={len(ds_val)} rows, Test={len(ds_test)} rows")

    trainer.train()
    trainer.save_model(args.best_model_dir)
    tokenizer.save_pretrained(args.best_model_dir)

    val_metrics = trainer.evaluate(tokenized_val)
    print("[VAL metrics]", val_metrics)
    trainer.save_metrics("val", val_metrics)

    test_metrics = trainer.evaluate(tokenized_test)
    print("[TEST metrics]", test_metrics)
    trainer.save_metrics("test", test_metrics)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_ds_dir", type=str, default="path_to_train_ds", help="Path to training dataset (save_to_disk)")
    p.add_argument("--test_ds_dir", type=str, default="path_to_test_ds", help="Path to test dataset (save_to_disk)")
    p.add_argument("--input_model", type=str, default="answerdotai/ModernBERT-large")
    # Numeric hyperparameters (kept identical)
    p.add_argument("--learning_rate", type=float, default=1.736e-5)
    p.add_argument("--train_bs", type=int, default=32)
    p.add_argument("--eval_bs", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--weight_decay", type=float, default=0.3)
    p.add_argument("--val_size", type=float, default=0.1, help="Fraction of train used for validation")
    p.add_argument("--output_dir", type=str, default="path_to_output_dir")
    p.add_argument("--best_model_dir", type=str, default="path_to_best_model_dir")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
