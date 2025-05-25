import sys
import argparse
from typing import TYPE_CHECKING
import torch
from transformers import DebertaV2Tokenizer
from modules.model import AuxiliaryDeberta
from modules.utils import get_device


# 전역 tokenizer (train 시와 동일한 사전 학습 모델 사용)
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large", use_fast=True)

# 예측 함수: 외부에서 함수로도 호출 가능
def predict(text: str, model: AuxiliaryDeberta) -> float:
    # 토큰화
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 예측 수행
    with torch.no_grad():
        outputs = model(inputs["input_ids"], inputs["attention_mask"])
        prob = outputs["main"].item()
        return prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Text Detection Prediction CLI")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    args = parser.parse_args()

    # 디바이스 설정
    device = get_device()

    if device.type == "cuda":
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        torch.cuda.tunable.set_filename("tunableop.csv")

    model = AuxiliaryDeberta()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    if device.type == "cuda" and not TYPE_CHECKING:
        model = torch.compile(model)
    model.eval()

    try:
        while True:
            print("ready.")
            text = sys.stdin.read()
            prob = predict(text, model)
            print(f"AI-generated probability: {prob:.4f}")
            if prob >= 0.5:
                print("→ This text is likely AI-generated.")
            else:
                print("→ This text is likely human-written.")
    except KeyboardInterrupt:
        pass
