import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
import torchaudio
import timm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset, random_split, DataLoader, WeightedRandomSampler
from transformers import RobertaModel, Wav2Vec2Model, Wav2Vec2Config, AutoTokenizer
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

################################################################################
# LIGHTER AUGMENTATION FUNCTIONS
################################################################################
import torchvision.transforms as T
from PIL import Image

def random_word_dropout(text, p=0.05):
    """
    Randomly drop words from a sentence with probability `p`.
    E.g., "I am very happy" -> "am very" with p=0.3
    """
    words = text.split()
    new_words = [w for w in words if random.random() > p]
    return " ".join(new_words)

def random_gain(waveform, gain_range=(0.9, 1.1)):
    """
    Slight random gain (amplitude change) to the waveform.
    """
    gain = random.uniform(*gain_range)
    return waveform * gain

def random_time_shift(waveform, max_shift=400):
    """
    Shift the waveform left or right by up to `max_shift` samples.
    """
    shift_amount = random.randint(0, max_shift)
    if random.random() < 0.5:
        shifted = torch.roll(waveform, shifts=shift_amount, dims=-1)
    else:
        shifted = torch.roll(waveform, shifts=-shift_amount, dims=-1)
    return shifted

# Minimal image augment: random crop, horizontal flip, slight color jitter
train_image_transform = T.Compose([
    T.ToPILImage(),
    T.RandomResizedCrop((224, 224), scale=(0.9, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.1, contrast=0.1),
    T.ToTensor()
])

val_image_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor()
])

################################################################################
# LABEL MAPPINGS
################################################################################
def map_meld_label(label_str):
    mapping = {
        "anger": 0,
        "disgust": 1,
        "fear": 2,
        "joy": 3,       # map joy -> happy
        "sadness": 4,
        "surprise": 5,
        "neutral": 6
    }
    return mapping.get(label_str.strip().lower(), 6)

def map_ravdess_label(emotion_id):
    # RAVDESS: 1=neutral, 2=calm, 3=happy, 4=sad, 5=angry, 6=fearful, 7=disgust, 8=surprised
    mapping = {
        1: 6,  # neutral
        2: 6,  # calm => neutral
        3: 3,  # happy
        4: 4,  # sad
        5: 0,  # angry
        6: 2,  # fearful
        7: 1,  # disgust
        8: 5   # surprised
    }
    return mapping.get(emotion_id, 6)

################################################################################
# DATASETS
################################################################################
class MELDDataset(Dataset):
    def __init__(self, meld_data_dir, split='train', tokenizer=None, max_seq_len=128, augment=False):
        file_map = {
            "train": "train_sent_emo.csv",
            "val":   "dev_sent_emo.csv",
            "test":  "test_sent_emo.csv"
        }
        csv_file = os.path.join(meld_data_dir, file_map[split])
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"MELD CSV not found at {csv_file}")
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["Utterance"]
        label = map_meld_label(row["Emotion"])

        # LIGHT TEXT AUGMENT
        if self.augment:
            text = random_word_dropout(text, p=0.05)

        input_ids, attention_mask = None, None
        if self.tokenizer is not None:
            encoded = self.tokenizer(text, truncation=True, max_length=self.max_seq_len, return_tensors="pt")
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio": None,
            "image": None,
            "label": label
        }

class RAVDESSDataset(Dataset):
    def __init__(self, ravdess_root, augment=False):
        self.wav_files = glob.glob(os.path.join(ravdess_root, "*.wav"))
        self.resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)
        self.augment = augment

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        file_path = self.wav_files[idx]
        basename = os.path.basename(file_path)
        parts = basename.split("-")
        emotion_id = int(parts[2])
        label = map_ravdess_label(emotion_id)

        waveform, sr = torchaudio.load(file_path)
        if sr != 16000:
            waveform = self.resampler(waveform)
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # LIGHT AUDIO AUGMENT
        if self.augment:
            if random.random() < 0.5:
                waveform = random_gain(waveform, gain_range=(0.9, 1.1))
            if random.random() < 0.5:
                waveform = random_time_shift(waveform, max_shift=400)

        return {
            "input_ids": None,
            "attention_mask": None,
            "audio": waveform,
            "image": None,
            "label": label
        }

class FER2013Dataset(Dataset):
    def __init__(self, fer_root, split='train', augment=False):
        self.split = split.lower()
        csv_file = os.path.join(fer_root, "fer2013.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"FER2013 CSV not found at {csv_file}")
        self.data = pd.read_csv(csv_file)

        usage_map = {"train": "Training", "val": "PublicTest", "test": "PrivateTest"}
        self.data = self.data[self.data["Usage"] == usage_map[self.split]]

        self.augment = augment
        self.img_transform = train_image_transform if self.augment else val_image_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = int(row["emotion"])
        pixels = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8)
        if pixels.size != 48 * 48:
            raise ValueError("Invalid pixel count")
        img = pixels.reshape((48, 48))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_tensor = self.img_transform(img_rgb)

        return {
            "input_ids": None,
            "attention_mask": None,
            "audio": None,
            "image": img_tensor,
            "label": label
        }

################################################################################
# SPLIT HELPERS & COLLATE
################################################################################
def get_ravdess_splits(ravdess_dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    total_size = len(ravdess_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    return random_split(ravdess_dataset, [train_size, val_size, test_size])

def multimodal_collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    MIN_AUDIO_LEN = 400

    input_ids_list, attention_mask_list = [], []
    audio_list, image_list, label_list = [], [], []

    for sample in batch:
        # Text
        if sample["input_ids"] is None:
            ids = torch.zeros(1, dtype=torch.long)
            amask = torch.zeros(1, dtype=torch.long)
        else:
            ids = sample["input_ids"]
            amask = sample["attention_mask"]
        input_ids_list.append(ids)
        attention_mask_list.append(amask)

        # Audio
        if sample["audio"] is None:
            audio_tensor = torch.zeros((1, MIN_AUDIO_LEN), dtype=torch.float32)
        else:
            audio_tensor = sample["audio"]
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            if audio_tensor.shape[-1] < MIN_AUDIO_LEN:
                pad_size = MIN_AUDIO_LEN - audio_tensor.shape[-1]
                audio_tensor = torch.cat([audio_tensor, torch.zeros((audio_tensor.shape[0], pad_size))], dim=-1)
        audio_list.append(audio_tensor)

        # Image
        if sample["image"] is None:
            img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
        else:
            img_tensor = sample["image"]
        image_list.append(img_tensor)

        label_list.append(sample["label"])

    # Pad text sequences
    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

    # Pad audio to max in batch
    max_audio_len = max(a.size(-1) for a in audio_list)
    padded_audio_list = []
    for a in audio_list:
        time_dim = a.size(-1)
        if time_dim < max_audio_len:
            pad_size = max_audio_len - time_dim
            a = torch.cat([a, torch.zeros((a.size(0), pad_size))], dim=-1)
        padded_audio_list.append(a)
    audio_stacked = torch.stack(padded_audio_list, dim=0)

    image_stacked = torch.stack(image_list, dim=0)
    labels_tensor = torch.LongTensor(label_list)

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "audio": audio_stacked,
        "image": image_stacked,
        "label": labels_tensor
    }

################################################################################
# MODEL DEFINITION
################################################################################
class ContextGatedCrossAttention(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=4):
        super().__init__()
        self.cross_attn_t2a = nn.MultiheadAttention(hidden_dim, num_heads)
        self.cross_attn_t2v = nn.MultiheadAttention(hidden_dim, num_heads)
        self.cross_attn_a2v = nn.MultiheadAttention(hidden_dim, num_heads)

        self.gate_t = nn.Linear(hidden_dim, hidden_dim)
        self.gate_a = nn.Linear(hidden_dim, hidden_dim)
        self.gate_v = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, text_seq, audio_seq, vision_seq):
        t = text_seq.transpose(0, 1)   # [T, B, D]
        a = audio_seq.transpose(0, 1)
        v = vision_seq.transpose(0, 1)

        # Text attended by (audio, vision)
        t2a_out, _ = self.cross_attn_t2a(t, a, a)
        t2v_out, _ = self.cross_attn_t2v(t, v, v)
        t_combined = t + 0.5 * (t2a_out + t2v_out)
        t_gate = torch.sigmoid(self.gate_t(t_combined))
        t_out = t_combined * t_gate

        # Audio attended by (text, vision)
        a2t_out, _ = self.cross_attn_t2a(a, t, t)
        a2v_out, _ = self.cross_attn_a2v(a, v, v)
        a_combined = a + 0.5 * (a2t_out + a2v_out)
        a_gate = torch.sigmoid(self.gate_a(a_combined))
        a_out = a_combined * a_gate

        # Vision attended by (text, audio)
        v2t_out, _ = self.cross_attn_t2v(v, t, t)
        v2a_out, _ = self.cross_attn_a2v(v, a, a)
        v_combined = v + 0.5 * (v2t_out + v2a_out)
        v_gate = torch.sigmoid(self.gate_v(v_combined))
        v_out = v_combined * v_gate

        return t_out.transpose(0, 1), a_out.transpose(0, 1), v_out.transpose(0, 1)

class EmoXFormers(nn.Module):
    def __init__(self, num_classes=7, hidden_dim=512):
        super().__init__()
        # --- Text model ---
        self.text_model = RobertaModel.from_pretrained("roberta-base")
        self.text_model.gradient_checkpointing_enable()

        # --- Audio model ---
        config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
        config.apply_spec_augment = False  # turn off time masking altogether
        config.gradient_checkpointing = True
        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", config=config)

        # --- Vision model ---
        self.vision_model = timm.create_model("vit_base_patch16_224", pretrained=True)
        vit_in_features = self.vision_model.head.in_features
        self.vision_model.head = nn.Identity()  # remove original head

        # --- Projection layers ---
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_model.config.hidden_size, hidden_dim),
            nn.Dropout(0.3)
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(self.audio_model.config.hidden_size, hidden_dim),
            nn.Dropout(0.3)
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(vit_in_features, hidden_dim),
            nn.Dropout(0.3)
        )

        # --- Four cross-attention layers (example) ---
        self.fusion_block1 = ContextGatedCrossAttention(hidden_dim=hidden_dim)
        self.fusion_block2 = ContextGatedCrossAttention(hidden_dim=hidden_dim)
        self.fusion_block3 = ContextGatedCrossAttention(hidden_dim=hidden_dim)
        self.fusion_block4 = ContextGatedCrossAttention(hidden_dim=hidden_dim)

        # --- Aggregation + classifier ---
        self.agg_linear = nn.Linear(hidden_dim * 3, hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask, audio, image):
        device = next(self.parameters()).device

        # Text
        if input_ids is not None and input_ids.dim() > 1 and input_ids.size(1) > 1:
            text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_seq = self.text_proj(text_out.last_hidden_state)
        else:
            batch_size = audio.size(0) if audio is not None else image.size(0)
            text_seq = torch.zeros(batch_size, 1, 512, device=device)

        # Audio
        if audio is not None and audio.dim() > 2:
            audio_mono = audio.mean(dim=1)
            audio_out = self.audio_model(audio_mono, return_dict=True)
            audio_seq = self.audio_proj(audio_out.last_hidden_state)
        else:
            batch_size = image.size(0) if image is not None else text_seq.size(0)
            audio_seq = torch.zeros(batch_size, 1, 512, device=device)

        # Image
        if image is not None and image.dim() > 2:
            vision_out = self.vision_model(image)
            vision_out = vision_out.unsqueeze(1)
            vision_seq = self.vision_proj(vision_out)
        else:
            batch_size = text_seq.size(0)
            vision_seq = torch.zeros(batch_size, 1, 512, device=device)

        # Fusion
        t_fused, a_fused, v_fused = self.fusion_block1(text_seq, audio_seq, vision_seq)
        t_fused, a_fused, v_fused = self.fusion_block2(t_fused, a_fused, v_fused)
        # You can use fusion_block3/fusion_block4 if you want more layers:
        t_fused, a_fused, v_fused = self.fusion_block3(t_fused, a_fused, v_fused)
        t_fused, a_fused, v_fused = self.fusion_block4(t_fused, a_fused, v_fused)

        # Aggregation
        t_mean = torch.mean(t_fused, dim=1)
        a_mean = torch.mean(a_fused, dim=1)
        v_mean = torch.mean(v_fused, dim=1)

        combined = torch.cat([t_mean, a_mean, v_mean], dim=1)
        combined = F.relu(self.agg_linear(combined))
        logits = self.classifier(combined)
        return logits

################################################################################
# TRAINING FUNCTIONS
################################################################################
class FocalLoss(nn.Module):
    """
    Focal Loss helps with class imbalance by down-weighting easy examples.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, preds, targets):
        ce_loss = self.ce(preds, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        audio = batch["audio"].to(device)
        image = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask, audio, image)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(dataloader)
    train_acc = correct / total
    return train_loss, train_acc

def evaluate_model(model, dataloader, criterion, device, desc="Validation"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    progress_bar = tqdm(dataloader, desc=desc, leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            audio = batch["audio"].to(device)
            image = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask, audio, image)
            loss = criterion(logits, labels)
            running_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    eval_loss = running_loss / len(dataloader)
    eval_acc = correct / total
    return eval_loss, eval_acc

################################################################################
# TWO-STAGE TRAINING
################################################################################
def train_emotion_model_multistage(
    root_dir="/home/aisha/navi-data/EmoTransformer/datasets",
    stage1_epochs=5,
    stage2_epochs=20,
    batch_size=32,
    lr_stage1=1e-4,
    lr_stage2=5e-5,
    weight_decay=1e-4,
    max_seq_len=128,
    early_stop_patience=5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # A. Build MELD & FER2013
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    meld_train = MELDDataset(
        os.path.join(root_dir, "meld", "data"), split="train",
        tokenizer=tokenizer, max_seq_len=max_seq_len, augment=True
    )
    meld_val   = MELDDataset(
        os.path.join(root_dir, "meld", "data"), split="val",
        tokenizer=tokenizer, max_seq_len=max_seq_len, augment=False
    )
    meld_test  = MELDDataset(
        os.path.join(root_dir, "meld", "data"), split="test",
        tokenizer=tokenizer, max_seq_len=max_seq_len, augment=False
    )

    fer_train = FER2013Dataset(os.path.join(root_dir, "fer2013"), split="train", augment=True)
    fer_val   = FER2013Dataset(os.path.join(root_dir, "fer2013"), split="val",   augment=False)
    fer_test  = FER2013Dataset(os.path.join(root_dir, "fer2013"), split="test",  augment=False)

    # B. Build RAVDESS
    ravdess_full = RAVDESSDataset(os.path.join(root_dir, "ravdess/Actor_*"), augment=False)
    print("Full Length RAVDESS:", len(ravdess_full))
    rav_train_split, rav_val_split, rav_test_split = get_ravdess_splits(ravdess_full, 0.8, 0.1, 0.1)
    rav_train_split.dataset.augment = True

    # replicate factor for RAVDESS
    replicate_factor = 7
    ravdess_train = ConcatDataset([rav_train_split for _ in range(replicate_factor)])
    ravdess_val   = rav_val_split
    ravdess_test  = rav_test_split
    print(f"RAVDESS TRAIN (replicated): {len(ravdess_train)}")
    print(f"RAVDESS VAL: {len(ravdess_val)}")
    print(f"RAVDESS TEST: {len(ravdess_test)}")

    # C. Combine train/val/test
    train_ds = ConcatDataset([meld_train, fer_train, ravdess_train])
    val_ds   = ConcatDataset([meld_val,   fer_val,   ravdess_val])
    test_ds  = ConcatDataset([meld_test,  fer_test,  ravdess_test])

    print(f"TRAIN SAMPLES: {len(train_ds)} | VAL SAMPLES: {len(val_ds)} | TEST SAMPLES: {len(test_ds)}")

    # D. DATASET-BASED Weighted Sampler:
    #    We want exactly 1/3 from MELD, 1/3 from FER, 1/3 from RAVDESS in each epoch
    meld_len = len(meld_train)
    fer_len  = len(fer_train)
    rav_len  = len(ravdess_train)
    total_len = meld_len + fer_len + rav_len

    # Starting indices for each dataset slice within train_ds
    start_fer = meld_len
    start_rav = meld_len + fer_len

    # Each dataset gets 1/3 of the total "weight"
    weights = [0.0] * total_len

    # MELD
    for i in range(meld_len):
        weights[i] = 1.0 / (3.0 * meld_len)

    # FER
    for i in range(fer_len):
        weights[start_fer + i] = 1.0 / (3.0 * fer_len)

    # RAVDESS
    for i in range(rav_len):
        weights[start_rav + i] = 1.0 / (3.0 * rav_len)

    sampler = WeightedRandomSampler(weights, num_samples=total_len, replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        collate_fn=multimodal_collate_fn, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=multimodal_collate_fn, num_workers=2
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=multimodal_collate_fn, num_workers=2
    )

    # E. Initialize model
    model = EmoXFormers(num_classes=7, hidden_dim=512).to(device)

    # F. Stage 1: Freeze big backbones
    for param in model.text_model.parameters():
        param.requires_grad = False
    for param in model.audio_model.parameters():
        param.requires_grad = False
    for param in model.vision_model.parameters():
        param.requires_grad = False

    stage1_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_stage1 = optim.AdamW(stage1_params, lr=lr_stage1, weight_decay=weight_decay)
    scheduler_stage1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_stage1, mode='min', factor=0.2, patience=2, verbose=True
    )
    criterion = FocalLoss(alpha=1.0, gamma=2.0)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    no_improve_count = 0
    best_ckpt_path = "best_checkpoint_stage1.pth"

    print("\n=========== STAGE 1: Training new layers (frozen backbones) ===========")
    for epoch in range(stage1_epochs):
        print(f"\n[Epoch {epoch+1}/{stage1_epochs} - Stage 1]")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer_stage1, criterion, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device, desc="Validation")

        print(f"Train => Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   => Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        scheduler_stage1.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  New best val acc: {best_val_acc:.4f} -> Saved checkpoint")

        if no_improve_count >= early_stop_patience:
            print("Early stopping triggered in Stage 1.")
            break

    # Load best from Stage 1
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    # G. Stage 2: Unfreeze & fine-tune
    for param in model.text_model.parameters():
        param.requires_grad = True
    for param in model.audio_model.parameters():
        param.requires_grad = True
    for param in model.vision_model.parameters():
        param.requires_grad = True

    optimizer_stage2 = optim.AdamW(model.parameters(), lr=lr_stage2, weight_decay=weight_decay)
    scheduler_stage2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_stage2, mode='min', factor=0.2, patience=2, verbose=True
    )

    best_val_acc = 0.0
    best_val_loss = float("inf")
    no_improve_count = 0
    best_ckpt_path = "best_checkpoint_stage2.pth"

    print("\n=========== STAGE 2: Fine-tuning all layers ===========")
    for epoch in range(stage2_epochs):
        print(f"\n[Epoch {epoch+1}/{stage2_epochs} - Stage 2]")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer_stage2, criterion, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device, desc="Validation")

        print(f"Train => Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   => Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        scheduler_stage2.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  New best val acc: {best_val_acc:.4f} -> Saved checkpoint")

        if no_improve_count >= early_stop_patience:
            print("Early stopping triggered in Stage 2.")
            break

    # H. Final evaluation
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device, desc="Test")
    print(f"\n================= Final Test => Loss: {test_loss:.4f}, Acc: {test_acc:.4f} =================")

    print("\nInference on 5 random test samples:")
    model.eval()
    label_map = {0:"Angry",1:"Disgust",2:"Fear",3:"Happy",4:"Sad",5:"Surprise",6:"Neutral"}

    indices = random.sample(range(len(test_ds)), 5)
    inference_samples = [test_ds[i] for i in indices]
    inference_batch = multimodal_collate_fn(inference_samples)
    for key in ["input_ids","attention_mask","audio","image"]:
        inference_batch[key] = inference_batch[key].to(device)
    true_labels = inference_batch["label"].to(device)

    with torch.no_grad():
        logits = model(
            inference_batch["input_ids"],
            inference_batch["attention_mask"],
            inference_batch["audio"],
            inference_batch["image"]
        )
        preds = torch.argmax(logits, dim=1)

    for i in range(5):
        print(f" Sample {i+1}: True = {label_map[true_labels[i].item()]}, "
              f"Pred = {label_map[preds[i].item()]}")

if __name__ == "__main__":
    train_emotion_model_multistage(
        root_dir="/home/aisha/navi-data/EmoTransformer/datasets",
        stage1_epochs=50,    # freeze backbone
        stage2_epochs=40,    # fine-tune all
        batch_size=32,
        lr_stage1=1e-4,
        lr_stage2=5e-5,
        weight_decay=1e-4,
        max_seq_len=128,
        early_stop_patience=20
    )
