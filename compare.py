import argparse
import pandas as pd
import torch
from modules.utils import get_device, str_to_dtype
from modules.evaluate import make_dataset, evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs='+', required=True)
    parser.add_argument("--dataset", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--result", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp32")
    args = parser.parse_args()

    # 디바이스 설정
    device = get_device()
    dtype = str_to_dtype(args.dtype)

    if device.type == "cuda":
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        torch.cuda.tunable.set_filename("tunableop.csv")

    df = pd.read_csv(args.dataset)
    labels = df['label'].tolist()
    dataset = make_dataset(df)

    results = []
    for model in args.model:
        preds = evaluate(model, dataset, args.threshold, args.batch_size, device, dtype)
        results.append(preds)
        torch.cuda.empty_cache()

    df = pd.DataFrame(results).T
    df.columns = args.models
    df['label'] = labels
    df.to_csv(args.result, index=False)
