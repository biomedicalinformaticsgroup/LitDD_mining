from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainingArguments, CrossEncoderTrainer
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator
from datasets import load_from_disk
import torch


if torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability(0)
    if major >= 8:
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True

ds_bert = load_from_disk('ds_bert_train')
hard_negatives_ds = load_from_disk('hard_negatives_dataset')
ds_test = load_from_disk('ds_test')

def finetune_crossencoder(model_name):
    model = CrossEncoder(model_name)

    loss = BinaryCrossEntropyLoss(model)
    args = CrossEncoderTrainingArguments(
    output_dir="finetuned_cross_encoders",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3.879271032713091e-05,
    warmup_ratio=0.1,
    eval_strategy="no",
    save_strategy="no"
    )

    class_eval = CrossEncoderClassificationEvaluator(
        sentence_pairs = list(zip(ds_test["g2p_lgmde"], ds_test["tiab"])),
        labels=list(ds_test["label"]),
        name="anno_test",
    )

    results = class_eval(model)

    print(f'{model_name} baseline results')
    print(results)
    print(class_eval.primary_metric)
    print(results[class_eval.primary_metric])
    print('  ')

    trainer = CrossEncoderTrainer(
    model=model,
    args = args,
    train_dataset=hard_negatives_ds,
    loss=loss,
    )

    trainer.train()

    finetuned_results = class_eval(model)
    print(f'{model_name} finetuned crossencoder results')
    print(finetuned_results)
    print('')

    trainer.save_model(f'finetuned_ncbi_medcpt_cross')

    print(f'crossencoder finetuned model saved: ncbi_medcpt_cross')

finetune_crossencoder("ncbi/MedCPT-Cross-Encoder")
