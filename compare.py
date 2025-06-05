import glob
import argparse
import pandas as pd
import torch
from modules import consts
from modules.utils import device_to_normalized_form, get_device, str_to_dtype
from modules.evaluate import make_dataset, evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs='+', required=True)
    parser.add_argument("--dataset", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--result", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=device_to_normalized_form(get_device()))
    parser.add_argument("--dtype", type=str, default="fp32")
    parser.add_argument("--with-label", action='store_true', default=False, help="Include labels in the output CSV")
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
    for model in args.models:
        models.extend(glob.glob(model))

    results = []
    for model in models:
        result = evaluate(model, dataset, None, args.batch_size, device, dtype)
        results.append(result)
        torch.cuda.empty_cache()

    df = pd.DataFrame(results).T
    df.columns = models
    if args.with_label:
        df['label'] = labels
    df.to_csv(args.result, index=False)
