# YBIGTA DS Mini Project
26-1 YBIGTA DS mini project :: CLIP / BLIP 논문 코드 구현

---

## 개요

CLIP과 BLIP 두 논문의 아키텍처를 PyTorch로 직접 구현합니다.  
외부 라이브러리 없이 논문의 핵심 구조를 이해하고 코드로 옮기는 것을 목표로 합니다.

| 논문 | 출처 | 핵심 아이디어 |
|------|------|--------------|
| CLIP | Radford et al., OpenAI 2021 | 이미지-텍스트 대조 학습으로 zero-shot 전이 |
| BLIP | Li et al., Salesforce 2022 | ITC + ITM + LM 세 가지 목적함수로 이해/생성 통합 |

---

## 프로젝트 구조

```
YBIGTA-DS-Mini-Project/
│
├── clip/
│   ├── image_encoder.py   # Vision Transformer (ViT): PatchEmbed → Attention → CLS 투영
│   ├── text_encoder.py    # Causal Transformer: EOS 토큰 기반 텍스트 인코더
│   ├── model.py           # CLIP 메인 모델 + CLIPConfig (ViT-B/32, B/16, L/14)
│   ├── loss.py            # InfoNCE (대칭 cross-entropy)
│   └── __init__.py
│
├── blip/
│   ├── vit.py             # Vision Transformer: 전체 패치 토큰 반환 (cross-attention용)
│   ├── med.py             # BERT 기반 멀티모달 인코더-디코더 (ITC/ITM/LM 세 모드)
│   ├── model.py           # BLIP 메인 모델 + 세 가지 forward pass
│   ├── loss.py            # ITCLoss / ITMLoss / LMLoss
│   └── __init__.py
│
├── demo_clip.py           # CLIP 아키텍처 forward pass 확인
├── demo_blip.py           # BLIP ITC / ITM / LM loss 확인
└── requirements.txt
```

---

## CLIP 구현

**논문:** *Learning Transferable Visual Models From Natural Language Supervision* (Radford et al., 2021)

### 아키텍처

```
이미지 → VisionTransformer → CLS 토큰 → Linear proj → L2 norm → image_feat
텍스트 → TextTransformer  → EOS 토큰 → Linear proj → L2 norm → text_feat

유사도 행렬 = logit_scale × image_feat @ text_feat.T   # (B, B)
```

### 핵심 구성 요소

- **PatchEmbed**: Conv2d로 이미지를 패치 시퀀스로 변환
- **VisionTransformer**: CLS 토큰 + 위치 임베딩 + Transformer blocks → 투영
- **TextTransformer**: Causal(GPT-style) attention, EOS 토큰에서 특징 추출
- **CLIPLoss**: 양방향 InfoNCE — 배치 내 매칭 쌍이 정답

### 사전 정의된 설정

| Config | patch_size | vision_dim | text_dim | output_dim |
|--------|-----------|-----------|---------|-----------|
| ViT-B/32 | 32 | 768 | 512 | 512 |
| ViT-B/16 | 16 | 768 | 512 | 512 |
| ViT-L/14 | 14 | 1024 | 768 | 768 |

---

## BLIP 구현

**논문:** *BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation* (Li et al., 2022)

### 아키텍처

```
이미지 → VisionTransformer → (B, N+1, 768)  ← 전체 패치 토큰

[ITC]  텍스트 → BertModel(cross-attn=None)  → CLS → proj(256) → InfoNCE
[ITM]  텍스트 → BertModel(cross-attn=ViT)   → CLS → Linear(2) → BCE
[LM]   텍스트 → BertModel(causal+cross-attn) → 전체 hidden → vocab logits
```

### 세 가지 목적함수

| 목적함수 | 설명 | 텍스트 인코더 모드 |
|---------|------|-----------------|
| **ITC** | 이미지-텍스트 대조 학습 (InfoNCE) | Self-attention only (unimodal) |
| **ITM** | 매칭/비매칭 이진 분류 | Self-attn + Cross-attn to ViT |
| **LM**  | 이미지 조건부 캡션 생성 | Causal self-attn + Cross-attn to ViT |

### 노트북 구현과의 차이

| 항목 | BLIP.ipynb (단순화) | 현재 구현 (논문 충실) |
|------|-------------------|-------------------|
| ITM cross-attn | ITC feature 위에 별도 모듈 | BERT 각 레이어 내부에 내장 |
| LM decoder 깊이 | 1 layer | 12 layer BertModel |
| Cross-attn 대상 | CLS 1개 토큰 | 전체 패치 토큰 (N+1개) |
| 이미지 인코더 | timm ViT-Tiny (pretrained) | ViT-B/16 직접 구현 |

---

## 실행

```bash
pip install torch torchvision

# CLIP 아키텍처 확인
python demo_clip.py

# BLIP 아키텍처 확인 (ITC + ITM + LM)
python demo_blip.py
```

사전학습 모델 사용 시 (선택):
```bash
pip install transformers

# BLIP captioning
from transformers import BlipProcessor, BlipForConditionalGeneration
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# CLIP zero-shot
from transformers import CLIPModel, CLIPProcessor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
```

---

## 참고 논문

- [CLIP — Radford et al., 2021](https://arxiv.org/abs/2103.00020)
- [BLIP — Li et al., 2022](https://arxiv.org/abs/2201.12086)
- [ViT — Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929)
