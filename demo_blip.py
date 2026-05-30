"""
BLIP 아키텍처 데모

1. 구현한 모델 구조 세 가지 목적함수 forward pass 확인
2. HuggingFace 사전학습 BLIP 모델로 Image Captioning 예시

사전학습 모델 사용 방법:
    pip install transformers Pillow
"""
import torch

from blip import BLIP, BLIPConfig


# ──────────────────────────────────────────────
# 1. 구현한 아키텍처로 세 가지 목적함수 확인
# ──────────────────────────────────────────────

def demo_architecture():
    print("=" * 50)
    print("[1] 구현 아키텍처 — ITC / ITM / LM forward pass")
    print("=" * 50)

    config = BLIPConfig()
    model = BLIP(config)
    model.eval()

    B, L = 2, 20
    dummy_image = torch.randn(B, 3, 224, 224)
    input_ids = torch.randint(0, config.vocab_size, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)

    # ITM 레이블: 0=unmatched, 1=matched
    itm_labels = torch.tensor([1, 0])

    # LM 레이블: -100은 loss 계산에서 무시 (패딩)
    lm_labels = input_ids.clone()
    lm_labels[:, -5:] = -100

    with torch.no_grad():
        total_loss, losses = model(
            image=dummy_image,
            input_ids=input_ids,
            attention_mask=attention_mask,
            itm_labels=itm_labels,
            lm_labels=lm_labels,
        )

    print(f"Image input  : {dummy_image.shape}")
    print(f"Text input   : {input_ids.shape}")
    print(f"ITC loss     : {losses['itc'].item():.4f}")
    print(f"ITM loss     : {losses['itm'].item():.4f}")
    print(f"LM  loss     : {losses['lm'].item():.4f}")
    print(f"Total loss   : {total_loss.item():.4f}\n")

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"총 파라미터 수: {total_params:.1f}M\n")


# ──────────────────────────────────────────────
# 2. 사전학습 모델 Image Captioning (optional)
# ──────────────────────────────────────────────

def demo_pretrained_captioning():
    """
    HuggingFace BLIP 사전학습 모델로 이미지 캡션 생성.
    `pip install transformers Pillow` 필요.
    """
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from PIL import Image
    except ImportError:
        print("[2] 사전학습 데모 skip: `pip install transformers Pillow`")
        return

    print("=" * 50)
    print("[2] HuggingFace BLIP — Image Captioning")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Salesforce/blip-image-captioning-base"

    print(f"모델 로딩: {model_name}")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    # 임의 이미지 생성 (실제 사용 시 Image.open("path/to/image.jpg") 사용)
    dummy_pil = Image.fromarray((torch.rand(224, 224, 3).numpy() * 255).astype("uint8"))

    inputs = processor(images=dummy_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)

    caption = processor.decode(out[0], skip_special_tokens=True)
    print(f"생성 캡션 (임의 이미지): {caption}\n")


if __name__ == "__main__":
    demo_architecture()
    demo_pretrained_captioning()
