import glob
import argparse
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from modules import consts
from modules.utils import device_to_normalized_form, get_device, str_to_dtype
from modules.evaluate import make_dataset, evaluate


# ----------------------------------------
# 메인 실행
# ----------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AI Text Detection Model")
    parser.add_argument("--model", type=str, nargs='+', required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=device_to_normalized_form(get_device()))
    parser.add_argument("--dtype", type=str, default="fp32")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default=0.5)")
    args = parser.parse_args()

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

    df = pd.read_csv(args.dataset)
    labels = df['label'].tolist()
    dataset = make_dataset(df)

    models = []
    for model in args.model:
        models.extend(glob.glob(model))

    for model in models:
        preds = evaluate(model, dataset, args.threshold, args.batch_size, device, dtype)

        # 지표 계산
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds)
        rec = recall_score(labels, preds)
        f1 = f1_score(labels, preds)

        # 결과 출력
        print(f"Model: {model}")
        print(f"Evaluation Results (threshold={args.threshold})")
        print(f" - Accuracy : {acc:.4f}")
        print(f" - Precision: {prec:.4f}")
        print(f" - Recall   : {rec:.4f}")
        print(f" - F1 Score : {f1:.4f}")

        torch.cuda.empty_cache()
