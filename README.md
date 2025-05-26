# Multimodal-Emotion-Recognition-using-Disjoint-Datasets
This repository contains the implementation of EmoXFormer, a context-aware multimodal emotion recognition model that leverages disjoint datasets (MELD for text, RAVDESS for audio, FER2013 for vision). It uses a two-stage training strategy with context-gated cross-attention to fuse representations from language, audio, and vision modalities.
## 🧠 Key Features

- ✅ **Disjoint Modality Learning** — works without aligned data
- 🧩 **Cross-Modality Attention Fusion** using Context-Gated Cross-Attention
- 🧪 **Two-Stage Training**: frozen backbone → full fine-tuning
- 💬 Text via RoBERTa
- 🔊 Audio via Wav2Vec2
- 🖼 Visual via ViT (Vision Transformer)
- 📊 Balanced training with custom weighted sampling
- 🎯 7-class emotion classification: `anger`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`

---

## 🗂 Dataset Setup

You need to manually download and arrange the following datasets:
datasets/
├── meld/
│ └── data/
│ ├── train_sent_emo.csv
│ ├── dev_sent_emo.csv
│ └── test_sent_emo.csv
├── ravdess/
│ └── Actor_*/ (all RAVDESS .wav files)
└── fer2013/
└── fer2013.csv
```bash
pip install -r requirements.txt
Required Libraries:

torch, torchaudio, transformers, timm

pandas, opencv-python, matplotlib, tqdm
Training the Model
python 2-stage-emoTransformer.py
Inside the script:

train_emotion_model_multistage(
    root_dir="/path/to/datasets",
    stage1_epochs=50,
    stage2_epochs=40,
    batch_size=32,
    lr_stage1=1e-4,
    lr_stage2=5e-5,
    weight_decay=1e-4,
    max_seq_len=128,
    early_stop_patience=20
)


Model Architecture
Backbones:

RoBERTa → text encoding

Wav2Vec2 → audio encoding

ViT (Vision Transformer) → image encoding

Fusion: 4 layers of Context-Gated Cross-Attention

Classification: Aggregated features → MLP → 7-class prediction

 Evaluation
After training, the model:

Saves the best model from Stage 1 and Stage 2

Reports loss and accuracy for each stage

Performs random test inference:

Sample 1: True = Happy, Pred = Happy
Sample 2: True = Sad,   Pred = Sad
Citation: 
@misc{emoxformer2025,
  title={EmoXFormer: Human Cognition Inspired Multimodal Emotion Recognition using Disjoint Modality Datasets},
  author={Qurat ul ain Aisha and Byung Gyu Kim},
  year={2025},
  institution={Sookmyung Women's University}
}
Contact
Qurat ul ain Aisha: aishanazar65@sookmyung.ac.kr

Prof. Byung Gyu Kim: bg.kim@sookmyung.ac.kr


