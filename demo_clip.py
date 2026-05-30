"""
CLIP 아키텍처 데모

1. 구현한 모델 구조 forward pass 확인
2. 사전학습 모델(OpenAI CLIP) 로드 후 zero-shot 분류 예시

사전학습 모델 사용 방법:
    pip install git+https://github.com/openai/CLIP.git Pillow
"""
import torch
import torch.nn.functional as F

from clip import CLIP, CLIPConfig, CLIP_CONFIGS


# ──────────────────────────────────────────────
# 1. 구현한 아키텍처로 forward pass 확인
# ──────────────────────────────────────────────

def demo_architecture():
    print("=" * 50)
    print("[1] 구현 아키텍처 forward pass")
    print("=" * 50)

    config = CLIP_CONFIGS["ViT-B/32"]
    model = CLIP(config)
    model.eval()

    B = 2
    dummy_image = torch.randn(B, 3, 224, 224)
    dummy_text = torch.zeros(B, 77, dtype=torch.long)
    # EOS token (49407) 위치 설정
    dummy_text[:, -1] = 49407

    with torch.no_grad():
        logits_per_image, logits_per_text = model(dummy_image, dummy_text)

    print(f"Config          : ViT-B/32")
    print(f"Image input     : {dummy_image.shape}")
    print(f"Text input      : {dummy_text.shape}")
    print(f"logits_per_image: {logits_per_image.shape}  (B x B 유사도 행렬)")
    print(f"logits_per_text : {logits_per_text.shape}")

    probs = logits_per_image.softmax(dim=-1)
    print(f"확률 (image→text): {probs}")

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"총 파라미터 수   : {total_params:.1f}M\n")


# ──────────────────────────────────────────────
# 2. 사전학습 모델 zero-shot 분류 (optional)
# ──────────────────────────────────────────────

def demo_pretrained_zeroshot():
    """
    OpenAI 공식 CLIP으로 zero-shot 이미지 분류.

    주의: OpenAI clip 패키지는 로컬 clip/ 디렉토리와 이름이 충돌합니다.
    사전학습 모델을 사용하려면 별도 스크립트에서:

        import sys, importlib
        sys.path = [p for p in sys.path if 'YBIGTA' not in p]
        import clip
        model, preprocess = clip.load("ViT-B/32")

    또는 HuggingFace openai/clip-vit-base-patch32 를 사용하세요:

        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    """
    print("=" * 50)
    print("[2] 사전학습 모델 사용 안내")
    print("=" * 50)
    print("  HuggingFace: pip install transformers")
    print("  from transformers import CLIPModel, CLIPProcessor")
    print("  model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')")
    print()


if __name__ == "__main__":
    demo_architecture()
    demo_pretrained_zeroshot()
