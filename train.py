import os
import argparse
from typing import Optional, Dict, Tuple, List, Any, TYPE_CHECKING
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchao
from torchao.quantization import quantize_
from torchao.prototype.quantized_training import Int8MixedPrecisionTrainingConfig
import pandas as pd
from pandarallel import pandarallel
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from modules import consts
from modules.model import MultiTaskDeberta
from modules.loss import compute_multitask_loss
from modules.dataset import TrainDataset
from modules.preprocess import preprocess_for_train
from modules.utils import make_tokenizer, device_to_normalized_form, get_device, str_to_dtype


def preprocess_data(x, features_scaled: np.ndarray) -> Dict[str, Any]:
    return {
        'main': float(x['label']),
        'aux': [float(feature) for feature in features_scaled[x.name]],
    }

# CSV 파일을 읽고, 정규화까지 처리하는 함수
def load_and_preprocess_data(path: os.PathLike, scaler: MinMaxScaler, fit_scaler=False) -> Tuple[List[str], List[Dict[str, Any]]]:
    df = pd.read_csv(path)
    texts = df['text'].tolist()
    features = df[['aux1', 'aux2']].copy()

    if fit_scaler:
        scaler.fit(features)
    features_scaled = scaler.transform(features)

    labels = []
    if not TYPE_CHECKING:
        labels = df.parallel_apply(lambda x: preprocess_data(x, features_scaled), axis=1)

    return texts, labels

# 모델이 한 번 훈련 혹은 검증을 수행하는 함수
def run_epoch(model: MultiTaskDeberta, dataloader: DataLoader, optimizer: Optional[torch.optim.Optimizer] = None, weights: List[float] = [1.0, *([0.0] * MultiTaskDeberta.NUM_AUXILIARY_TASKS)], accumulation_steps = 1) -> Tuple[float, List[float]]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    loss = torch.tensor([0.0, *([0.0] * MultiTaskDeberta.NUM_AUXILIARY_TASKS)], dtype=torch.float64, device=model.device)
    weights_tensor = torch.tensor(weights, dtype=model.dtype, device=model.device)
    for i, batch in enumerate(tqdm.tqdm(dataloader, leave=False)):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = {k: v.to(model.device) for k, v in batch['labels'].items()}

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss_per_tasks = compute_multitask_loss(outputs, labels)
            weighted_loss = loss_per_tasks * weights_tensor
            loss_sum = torch.sum(weighted_loss)

            if is_train:
                loss_sum = loss_sum / accumulation_steps
                loss_sum.backward()
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

        loss += weighted_loss

    total_loss = torch.sum(loss)
    total_loss = total_loss.item() / len(dataloader)

    loss = loss / len(dataloader)
    loss = loss.detach().cpu()
    loss = loss.tolist()

    return total_loss, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AI Text Detection Model")
    parser.add_argument("--ckpt", type=str, help="Path to checkpoint file", required=False)
    parser.add_argument("--prefix", type=str, required=False, default="model_epoch")
    parser.add_argument("--train", type=str, default="train_dataset.csv", help="Path to train CSV file")
    parser.add_argument("--val", type=str, help="Path to validate CSV file", required=False)
    parser.add_argument("--weights", type=str, required=False, default="1.0")
    parser.add_argument("--epoch", type=int, help="Epoch", required=True)
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--device", type=str, default=device_to_normalized_form(get_device()))
    parser.add_argument("--dtype", type=str, default="fp32", help="DataType")
    parser.add_argument("--quant", action='store_true', default=False)
    parser.add_argument("--save-loss", action='store_true', default=False)
    args = parser.parse_args()

    pandarallel.initialize(progress_bar=True, nb_workers=consts.PARALLELISM)

    # 디바이스 설정
    device = torch.device(args.device)
    dtype = str_to_dtype(args.dtype)

    if device.type == "cuda":
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        torch.cuda.tunable.set_filename("tunableop.csv")
    elif device.type == "cpu":
        torch.set_num_threads(consts.PARALLELISM)
        torch.set_num_interop_threads(consts.PARALLELISM)

    # 정규화 도구 생성
    scaler = MinMaxScaler()

    tokenizer = make_tokenizer()

    # 훈련 데이터 전처리
    train_texts, train_labels = load_and_preprocess_data(args.train, scaler, fit_scaler=True)
    # 텍스트 전처리
    train_encodings = preprocess_for_train(tokenizer, train_texts, train_labels, None, dtype)
    # Dataset 구성
    train_dataset = TrainDataset(train_encodings)
    # DataLoader 설정
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=consts.PARALLELISM, pin_memory=device.type == "cuda", shuffle=True)

    weights = [None if len(weight) == 0 else float(weight) for weight in args.weights.split(",")]
    for i in range(len(weights), MultiTaskDeberta.NUM_AUXILIARY_TASKS + 1):
        weights.append(None)

    # 모델 및 옵티마이저 초기화
    if args.ckpt is None:
        model = MultiTaskDeberta(auxiliary_tasks=[weight is not None for weight in weights[1:]])
        model = model.to(device=device, dtype=dtype)
    else:
        model = MultiTaskDeberta.from_pretrained(args.ckpt, device, dtype)
        for i in range(1, len(weights)):
            if weights[i] is not None and not model.has_auxiliary_task(i - 1):
                raise ValueError(f"Checkpoint does not contain auxiliary task #{i}.")

    if args.quant:
        config = Int8MixedPrecisionTrainingConfig(
            output=True,
            grad_input=True,
            grad_weight=False,
        )
        quantize_(model, config)

    model_runtime = model
    if device.type == "cuda" and not args.quant and not TYPE_CHECKING:
        model_runtime = model.to_compiled()

    optimizer = torch.optim.AdamW(model_runtime.parameters(), lr=args.learning_rate)
    if args.quant:
        optimizer = torchao.optim._AdamW(model_runtime.parameters(), lr=args.learning_rate)

    # 체크포인트 디렉토리 생성
    os.makedirs("checkpoints", exist_ok=True)

    print(f"PyTorch: device={device}, dtype={dtype}")
    print(f"Auxiliary tasks enabled: {', '.join([f'aux{i}={weight}' for i, weight in enumerate(weights) if i != 0])}")

    progress = tqdm.trange(args.epoch, desc=f"Epoch 0/{args.epoch}")
    record_total = []
    record_per_tasks = [None if weight is None else [] for weight in weights]

    weights = [0.0 if weight is None else weight for weight in weights]

    if args.val is None:
        # 학습 루프
        progress.set_postfix(loss='?')
        for epoch in progress:
            loss, loss_per_tasks = run_epoch(model_runtime, train_loader, optimizer, weights, args.accumulation_steps)
            record_total.append(loss)
            postfix = {'loss': loss, 'main': loss_per_tasks[0]}
            for i, aux_loss in enumerate(loss_per_tasks):
                record = record_per_tasks[i]
                if record is None:
                    continue
                record.append(aux_loss)
                if i != 0:
                    postfix[f'aux{i}'] = aux_loss
            progress.set_description(f"Epoch {epoch + 1}/{args.epoch}")
            progress.set_postfix(postfix)
            torch.save(model.state_dict(), f"checkpoints/{args.prefix}_epoch{epoch}.pth")
    else:
        val_texts, val_labels = load_and_preprocess_data(args.val, scaler)
        val_encodings = preprocess_for_train(tokenizer, val_texts, val_labels, None, dtype)
        val_dataset = TrainDataset(val_encodings)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        # 학습 루프
        progress.set_postfix(train_loss='?', val_loss='?')
        for epoch in progress:
            train_loss, loss_per_tasks = run_epoch(model_runtime, train_loader, optimizer, weights, args.accumulation_steps)
            val_loss, _ = run_epoch(model_runtime, val_loader)
            record_total.append(train_loss)
            postfix = {'train_loss': train_loss, 'val_loss': val_loss, 'main': loss_per_tasks[0]}
            for i, aux_loss in enumerate(loss_per_tasks):
                record = record_per_tasks[i]
                if record is None:
                    continue
                record.append(aux_loss)
                if i != 0:
                    postfix[f'aux{i}'] = aux_loss
            progress.set_description(f"Epoch {epoch + 1}/{args.epoch}")
            progress.set_postfix(postfix)
            torch.save(model.state_dict(), f"checkpoints/{args.prefix}_epoch{epoch}.pth")
    print(record_total)
    print(record_per_tasks)
    if args.save_loss:
        plt.plot(record_total)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(f"checkpoints/{args.prefix}_loss.png")
