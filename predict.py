import sys
import argparse
from typing import TYPE_CHECKING
import torch
from modules.model import MultiTaskDeberta
from modules.utils import make_tokenizer, get_device


# 전역 tokenizer (train 시와 동일한 사전 학습 모델 사용)
tokenizer = make_tokenizer()

# 예측 함수: 외부에서 함수로도 호출 가능
def predict(text: str, model: MultiTaskDeberta) -> float:
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

    model = MultiTaskDeberta.from_pretrained(args.model, device)
    model = model.to(device)
    model.eval()
    if device.type == "cuda" and not TYPE_CHECKING:
        model = model.to_compiled()

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
