import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import gc
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from collections import Counter
from sklearn.svm import SVC
import cv2
import mediapipe as mp

# moviepy is used to safely extract audio from mp4 files
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip # type: ignore

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import copy
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# REQUIRED INSTALLATIONS:
# pip install numpy librosa torch torchvision scikit-learn seaborn matplotlib opencv-python mediapipe moviepy shap
# ==========================================

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
DATA_PATH = r"D:\CSE427_end_game\RAVDESS"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# We process .mp4 files (Audio+Video). .wav duplicate files removed to force true multimodal learning
FILE_EXTENSIONS = ["*.mp4"]

# Audio Settings
SAMPLE_RATE = 22050
DIR_DURATION = 3.0  # seconds
SAMPLES = int(SAMPLE_RATE * DIR_DURATION)

# Video Settings
MAX_VIDEO_FRAMES = 30 # Number of frames to uniform-sample per video

# Training Config
BATCH_SIZE = 32 
EPOCHS = 60 # Increased epochs slightly to allow the lower LR to converge
LEARNING_RATE = 3e-4 # Reduced heavily to stabilize training curves

# RAVDESS Emotion mapping (01 to 08)
EMOTION_DICT = {
    "01": "Neutral", "02": "Calm", "03": "Happy", "04": "Sad",
    "05": "Angry", "06": "Fearful", "07": "Disgust", "08": "Surprised"
}

# Hardware Acceleration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# 1. DATA LOADING (Split Modalities)
# ==========================================

# ==========================================
# UTILITY: COUNT VIDEO FILES
# ==========================================
def count_video_files(data_path):
    all_videos = glob.glob(os.path.join(data_path, "**", "*.mp4"), recursive=True)
    total_video = len(all_videos)
    print(f"[INFO] Video (.mp4) files found: {total_video}")
    return total_video

def load_data(data_path):
    """
    Search recursively for .mp4 files in the data_path.
    Extract the emotion label and actor ID from the filename.
    """
    print(f"[*] Searching for {FILE_EXTENSIONS} files in: {data_path}")
    file_paths = []
    for ext in FILE_EXTENSIONS:
        search_pattern = os.path.join(data_path, "**", ext)
        file_paths.extend(glob.glob(search_pattern, recursive=True))
    
    file_paths = sorted(file_paths)
    
    if not file_paths:
        print(f"[!] No matching media files found! Check your dataset path.")
        return [], [], []

    files = []
    labels = []
    actors = []
    for fp in file_paths:
        file_name = os.path.basename(fp)
        parts = file_name.split('-')
        # RAVDESS filename format: 01-02-01-01-01-01-02.mp4
        # The 3rd part (index 2) is the emotion ID.
        # The 7th part (index 6) is the actor ID.
        if len(parts) >= 7:
            emotion_id = parts[2]
            actor_id = parts[6].split('.')[0]
            if emotion_id in EMOTION_DICT:
                files.append(fp)
                labels.append(int(emotion_id) - 1)  # 0-indexed (0 to 7)
                actors.append(int(actor_id))
    
    print(f"[*] Successfully found {len(files)} valid multimodal files.")
    return files, labels, actors


# ==========================================
# 2. PREPROCESSING & FEATURE EXTRACTION
# ==========================================

# Use Mediapipe FaceMesh to extract 468 3D landmarks
mp_face_mesh = mp.solutions.face_mesh # type: ignore

def extract_features_from_file(file_path):
    """
    Extracts TWO sets of features from a single file:
    1. Audio: MFCCs, Delta, Delta-Delta, ZCR
    2. Video: 468 Facial Landmarks per sampled frame
    """
    audio_features = None
    video_features = None
    
    # Initialize extractor inside function for thread/process safety
    face_mesh_extractor = mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1, 
        min_detection_confidence=0.5
    )
    
    try:
        # ---------------------------------------------
        # 2a. Audio Preprocessing & Extraction
        # ---------------------------------------------
        clip = VideoFileClip(file_path)
        if clip.audio is not None:
            # Extract raw audio array
            audio_signal = clip.audio.to_soundarray(fps=SAMPLE_RATE)
            if audio_signal.ndim == 2:
                audio_signal = audio_signal.mean(axis=1) # Convert stereo to mono
        else:
            audio_signal = np.zeros(SAMPLES)
        clip.close()
        
        # Temporal encoding: Fixed-length windows
        if len(audio_signal) > SAMPLES:
            # Center-cropping strategy to capture the most active parts of the speech
            start = (len(audio_signal) - SAMPLES) // 2
            audio_signal = audio_signal[start : start + SAMPLES]
        else:
            padding = SAMPLES - len(audio_signal)
            audio_signal = np.pad(audio_signal, (0, padding), 'constant')
            
        # Audio Feature extraction
        mfccs = librosa.feature.mfcc(y=audio_signal, sr=SAMPLE_RATE, n_mfcc=60)
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)
        zcr = librosa.feature.zero_crossing_rate(y=audio_signal)
        
        # Audio Shape: (Time, Features)
        audio_features = np.concatenate((mfccs, delta, delta2, zcr), axis=0).T
        
        # ---------------------------------------------
        # 2b. Video Preprocessing & Extraction
        # ---------------------------------------------
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        raw_frames_data = []
        if total_frames > 0:
            # DYNAMIC FRAME SAMPLING:
            # Sample 3x the target frames to find the most active moments.
            candidate_indices = np.linspace(0, total_frames - 1, min(total_frames, MAX_VIDEO_FRAMES * 3), dtype=int)
            
            prev_landmarks = None
            
            for idx in candidate_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                frame_features = np.zeros(468 * 3)
                movement_score = 0.0
                
                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh_extractor.process(rgb_frame)
                    
                    if results.multi_face_landmarks:
                        face = results.multi_face_landmarks[0] # Grab first face detected
                        
                        # Center landmarks to the nose tip (landmark 1)
                        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in face.landmark])
                        nose_tip = landmarks_array[1].copy()
                        landmarks_array = landmarks_array - nose_tip
                        frame_features = landmarks_array.flatten()
                        
                        # Calculate movement (L2 norm) relative to previous candidate frame
                        if prev_landmarks is not None:
                            movement_score = np.linalg.norm(frame_features - prev_landmarks)
                        
                        prev_landmarks = frame_features.copy()
                        
                raw_frames_data.append({
                    'idx': idx,
                    'features': frame_features,
                    'movement': movement_score
                })
        
        cap.release()
        
        # Select the top MAX_VIDEO_FRAMES based on movement score
        if len(raw_frames_data) > MAX_VIDEO_FRAMES:
            # Sort by highest movement
            raw_frames_data.sort(key=lambda x: x['movement'], reverse=True)
            selected_frames = raw_frames_data[:MAX_VIDEO_FRAMES]
            # CRITICAL: Resort chronologically to maintain temporal sequence for LSTM/Transformer
            selected_frames.sort(key=lambda x: x['idx'])
        else:
            selected_frames = raw_frames_data
            
        frames_array = [item['features'] for item in selected_frames]
        
        # Sequence Padding logic just in case the file could not yield enough frames
        while len(frames_array) < MAX_VIDEO_FRAMES:
            frames_array.append(np.zeros(468 * 3))
            
        video_features = np.array(frames_array)  # Shape: (MAX_VIDEO_FRAMES, 1404)
        
        return audio_features, video_features
        
    except Exception as e:
        print(f"[!] Error processing {file_path}: {e}")
        return None, None


# ==========================================
# 3. EDA (Exploratory Data Analysis)
# ==========================================
def plot_modalities_distribution(file_paths, labels):
    """
    Visualizes the breakdown of .mp4 files across all emotion classes.
    """
    print("[*] Plotting Modalities Distribution (.mp4)...")
    mp4_counts = {e: 0 for e in EMOTION_DICT.values()}
    
    for fp, lbl in zip(file_paths, labels):
        emotion = EMOTION_DICT[f"{lbl+1:02d}"]
        ext = os.path.splitext(fp)[1].lower()
        if ext == '.mp4':
            mp4_counts[emotion] += 1
            
    emotions = list(EMOTION_DICT.values())
    mp4_values = [mp4_counts[e] for e in emotions]
    
    x = np.arange(len(emotions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, mp4_values, width, label='Audio-Video (.mp4)', color='lightcoral')
    ax.set_title('Class Balance of RAVDESS Dataset (.mp4)')
    
    ax.set_ylabel('Total File Count')
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "modalities_distribution.png"))
    plt.close()
    print(f"[*] Modalities distribution saved in {OUTPUT_DIR}/modalities_distribution.png")

def perform_eda(labels, file_paths):
    """
    Perform class balance analysis and save to disk.
    """
    print("[*] Performing EDA (Class Balance)...")
    
    # 1. Combined Class Balance
    plt.figure(figsize=(10, 5))
    emotion_names = [EMOTION_DICT[f"{idx+1:02d}"] for idx in labels]
    sns.countplot(x=emotion_names, order=EMOTION_DICT.values(), palette="magma")
    plt.title("Class Balance of RAVDESS Dataset (.mp4)")
    plt.xlabel("Emotions")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_class_balance.png"))
    plt.close()
        
    print(f"[*] EDA complete. Plotted to {OUTPUT_DIR}/eda_class_balance*.png.")


# ==========================================
# UTILITY: OVERSAMPLING & DATASET SETUP
# ==========================================
def manual_random_over_sample(XA, XV, y):
    """
    Manually implements Random Over-Sampling to balance classes by duplicating 
    samples from minority classes until they match the majority class count.
    """
    counts = Counter(y)
    max_count = max(counts.values())
    
    XA_resampled = []
    XV_resampled = []
    y_resampled = []
    
    rng = np.random.default_rng(seed=42)
    for label in sorted(counts.keys()):
        indices = np.where(y == label)[0]
        # Randomly sample with replacement to match max_count
        if len(indices) >= max_count:
            resampled_indices = indices  # majority class — keep as-is, no duplication
        else:
            resampled_indices = rng.choice(indices, size=max_count, replace=True)
        
        XA_resampled.append(XA[resampled_indices])
        XV_resampled.append(XV[resampled_indices])
        y_resampled.append(y[resampled_indices])
        
    return np.concatenate(XA_resampled, axis=0), np.concatenate(XV_resampled, axis=0), np.concatenate(y_resampled, axis=0)

class RAVDESSMultimodalDataset(Dataset):
    def __init__(self, X_A, X_V, y, augment=False):
        self.X_A = torch.tensor(X_A, dtype=torch.float32)
        self.X_V = torch.tensor(X_V, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        a = self.X_A[idx].clone()
        v = self.X_V[idx].clone()
        
        # Data Augmentation to prevent overfitting
        if self.augment:
            # SpecAugment: time masking (mask up to 20% of time steps)
            T = a.shape[0]
            t_mask = max(1, int(T * 0.20))
            t0 = torch.randint(0, T - t_mask + 1, (1,)).item()
            a[t0:t0 + t_mask, :] = 0.0

            # SpecAugment: frequency masking (mask up to 15% of feature bands)
            F = a.shape[1]
            f_mask = max(1, int(F * 0.15))
            f0 = torch.randint(0, F - f_mask + 1, (1,)).item()
            a[:, f0:f0 + f_mask] = 0.0

            # Mild noise on video (landmarks are already quite noisy)
            v = v + torch.randn_like(v) * 0.02
            
        return a, v, self.y[idx]


# ==========================================
# 4 & 5. MODEL ARCHITECTURE & MULTIMODAL FUSION
# Option A: CNN + LSTM with Late Fusion Layer
# Option B: Cross-Modal Transformer (Arch 3)
# ==========================================
class CrossModalTransformer(nn.Module):
    def __init__(self, audio_dim, video_dim, d_model=128, nhead=4, num_layers=2, num_classes=8):
        super(CrossModalTransformer, self).__init__()
        
        # 1. Project both modalities to same dimension (Tokenization)
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.video_proj = nn.Linear(video_dim, d_model)
        
        # Positional embeddings (generous max sequence lengths) scaled down
        self.audio_pos = nn.Parameter(torch.randn(1, 400, d_model) * 0.02)
        self.video_pos = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        
        # 2. Independent Transformer Encoders (norm_first=True for training stability)
        audio_enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.4, norm_first=True)
        self.audio_encoder = nn.TransformerEncoder(audio_enc_layer, num_layers=num_layers)
        
        video_enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.4, norm_first=True)
        self.video_encoder = nn.TransformerEncoder(video_enc_layer, num_layers=num_layers)
        
        # 3. Cross-modal attention mapping
        self.cross_attn_A_V = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, dropout=0.3)
        self.cross_attn_V_A = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, dropout=0.3)
        
        # 4. MLP Head
        self.layer_norm = nn.LayerNorm(d_model * 2)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Increased Dropout
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x_audio, x_video):
        # Token sequence projection
        a = self.audio_proj(x_audio)
        v = self.video_proj(x_video)
        
        # Add positional embeddings safely based on exact sequence length
        a = a + self.audio_pos[:, :a.size(1), :]
        v = v + self.video_pos[:, :v.size(1), :]
        
        # Encode independent modalities
        a_enc = self.audio_encoder(a)
        v_enc = self.video_encoder(v)
        
        # Cross-modal Attention
        # Audio Query, Video Key/Value
        a_cross, _ = self.cross_attn_A_V(query=a_enc, key=v_enc, value=v_enc)
        # Video Query, Audio Key/Value
        v_cross, _ = self.cross_attn_V_A(query=v_enc, key=a_enc, value=a_enc)
        
        # Global Average Pooling (proxy for CLS token)
        a_pool = a_cross.mean(dim=1)
        v_pool = v_cross.mean(dim=1)
        
        # Fusion + MLP
        fused = torch.cat((a_pool, v_pool), dim=1)
        fused = self.layer_norm(fused)
        
        out = self.mlp(fused)
        return out


class Arch2_Attention_Gated_Fusion(nn.Module):
    def __init__(self, audio_dim, video_dim, num_classes=8):
        super(Arch2_Attention_Gated_Fusion, self).__init__()
        # --- AUDIO BRANCH ---
        # Original: VGGish -> GRU (128) -> 256-d
        # Adapted: GRU -> 256-d output
        self.audio_gru = nn.GRU(audio_dim, 128, bidirectional=True, batch_first=True) # 128 * 2 = 256
        
        # --- VIDEO BRANCH ---
        # Original: ResNet-50 -> Temporal Attention pool -> 256-d
        # Adapted: Linear proj -> Temporal Attention pool -> 256-d
        self.video_proj = nn.Linear(video_dim, 256)
        self.video_attn = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # --- GATED ATTENTION FUSION ---
        # g = sigmoid(W*[a;v])
        self.fuse_proj = nn.Linear(512, 256)
        self.fuse_gate = nn.Linear(512, 256)
        
        # --- CLASSIFIER ---
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), # Increased Dropout
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x_audio, x_video):
        # Audio
        a_out, _ = self.audio_gru(x_audio) # (B, T, 256)
        a_feat = a_out.mean(dim=1) # Average mean pooling over time (safest for bidirectional)
        
        # Video
        v_proj = self.video_proj(x_video) # (B, T, 256)
        attn_weights = torch.softmax(self.video_attn(v_proj), dim=1) # (B, T, 1)
        v_feat = torch.sum(v_proj * attn_weights, dim=1) # Weighted sum -> (B, 256)
        
        # Gated Fusion
        concat = torch.cat([a_feat, v_feat], dim=-1) # (B, 512)
        f = self.fuse_proj(concat) # Feature projection (B, 256)
        g = torch.sigmoid(self.fuse_gate(concat)) # Learned gate (B, 256)
        fused = f * g # Gated representation
        
        # Classification
        out = self.classifier(fused)
        return out


class Arch4_Bottleneck_Fusion(nn.Module):
    def __init__(self, audio_dim, video_dim, num_classes=8):
        super(Arch4_Bottleneck_Fusion, self).__init__()
        # --- AUDIO BRANCH ---
        # Original: wav2vec2 -> weighted mean pool -> linear to 256
        # Adapted: Sequence proj to 256 -> temporal softmax weighting
        self.audio_proj = nn.Linear(audio_dim, 256)
        self.audio_time_weights = nn.Linear(256, 1)
        self.audio_down_proj = nn.Identity()
        
        # --- VIDEO BRANCH ---
        # Original: VGGFace2 -> GRU aggregation -> 256-d linear
        # Adapted: Linear proj -> GRU -> 256-d linear
        self.video_proj = nn.Linear(video_dim, 512)
        self.video_gru = nn.GRU(512, 256, batch_first=True)
        
        # --- BOTTLENECK FUSION ---
        # 512-d concat -> 256-d -> Dropout
        self.bottleneck = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.4) # Increased Dropout
        )
        
        # --- CLASSIFIER ---
        # FC 256 -> FC 128 -> Softmax 8 (BatchNorm indicated)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5), # Increased Dropout
            nn.Linear(128, num_classes)
        )

    def forward(self, x_audio, x_video):
        # Audio aggregation (Weighted mean pool)
        a_proj = self.audio_proj(x_audio) # (B, T, 256)
        a_weights = torch.softmax(self.audio_time_weights(a_proj), dim=1) # (B, T, 1)
        a_feat = self.audio_down_proj(torch.sum(a_proj * a_weights, dim=1)) # (B, 256)
        
        # Video aggregation (GRU)
        v_512 = self.video_proj(x_video) # (B, T, 512)
        v_out, _ = self.video_gru(v_512) # (B, T, 256)
        v_feat = v_out.mean(dim=1) # Mean pooling over time sequence
        
        # Bottleneck Fusion
        concat = torch.cat([a_feat, v_feat], dim=-1) # (B, 512)
        fused = self.bottleneck(concat) # (B, 256)
        
        # Classification
        out = self.classifier(fused)
        return out


class Multimodal_CNN_LSTM(nn.Module):
    def __init__(self, audio_dim, video_dim, hidden_dim, num_classes):
        super(Multimodal_CNN_LSTM, self).__init__()
        
        # --- AUDIO BRANCH ---
        self.audio_conv = nn.Conv1d(in_channels=audio_dim, out_channels=128, kernel_size=3, padding=1)
        self.audio_bn = nn.BatchNorm1d(128)
        self.audio_pool = nn.MaxPool1d(kernel_size=2)
        self.audio_lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, num_layers=2, 
                                  batch_first=True, dropout=0.4, bidirectional=True)
        
        # --- VIDEO BRANCH ---
        self.video_conv = nn.Conv1d(in_channels=video_dim, out_channels=128, kernel_size=3, padding=1)
        self.video_bn = nn.BatchNorm1d(128)
        self.video_pool = nn.MaxPool1d(kernel_size=2)
        self.video_lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, num_layers=2, 
                                  batch_first=True, dropout=0.4, bidirectional=True)
        
        self.relu = nn.ReLU()
        
        # --- LATE FUSION HEAD ---
        # Each bidirectional LSTM outputs: hidden_dim * 2
        # Fusion concatenates both outputs: (hidden_dim*2) + (hidden_dim*2) = hidden_dim * 4
        self.fc1 = nn.Linear(hidden_dim * 4, 128)
        self.dropout = nn.Dropout(0.5) # Increased dropout
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x_audio, x_video):
        # 1. Processing Audio
        # Conv1d expects (Batch, Features, Time)
        a = x_audio.permute(0, 2, 1)
        a = self.audio_conv(a)
        a = self.audio_bn(a)
        a = self.relu(a)
        a = self.audio_pool(a)
        a = a.permute(0, 2, 1) # Back to (Batch, Time, Features) for LSTM
        a_out, _ = self.audio_lstm(a)
        a_embed = a_out.mean(dim=1) # Mean pooling over time sequence (crucial for bidirectional)
        
        # 2. Processing Video
        v = x_video.permute(0, 2, 1)
        v = self.video_conv(v)
        v = self.video_bn(v)
        v = self.relu(v)
        v = self.video_pool(v)
        v = v.permute(0, 2, 1)
        v_out, _ = self.video_lstm(v)
        v_embed = v_out.mean(dim=1) # Mean pooling over time sequence
        
        # 3. Fusion (Late Fusion / Concat strategy)
        fused = torch.cat((a_embed, v_embed), dim=1)
        
        # 4. Classification
        out = self.fc1(fused)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


# ==========================================
# 6. TRAINING SCRIPT
# ==========================================
def train_multimodal_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    print("[*] Starting Multimodal Training Loop...")
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    
    train_losses, val_losses = [], []
    
    patience = 3 # Reduced patience to stop sooner when val starts rising
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_a, batch_v, labels in train_loader:
            batch_a, batch_v, labels = batch_a.to(DEVICE), batch_v.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass with both modalities
            outputs = model(batch_a, batch_v)
            loss = criterion(outputs, labels)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Prevent exploding gradients
            optimizer.step()
            running_loss += loss.item() * batch_a.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        corrects = 0
        with torch.no_grad():
            for batch_a, batch_v, labels in val_loader:
                batch_a, batch_v, labels = batch_a.to(DEVICE), batch_v.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(batch_a, batch_v)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * batch_a.size(0)
                
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_acc = float(corrects) / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # Learning Rate Scheduler step
        scheduler.step(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early Stopping check
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[*] Early stopping triggered at epoch {epoch+1}.")
                break
            
    # Load best model weights for final evaluation
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses


# ==========================================
# 7. EVALUATION
# ==========================================
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_a, batch_v, labels in test_loader:
            batch_a, batch_v, labels = batch_a.to(DEVICE), batch_v.to(DEVICE), labels.to(DEVICE)
            outputs = model(batch_a, batch_v)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return all_labels, all_preds


def generate_performance_report(all_labels, all_preds, model_name):
    # Dynamically find the unique classes present in this test run 
    # to prevent ValueError when the test split doesn't contain all 8 classes
    unique_classes = sorted(list(set(all_labels) | set(all_preds)))
    target_names = [EMOTION_DICT[f"{i+1:02d}"] for i in unique_classes]
    
    acc = accuracy_score(all_labels, all_preds)
    ua = balanced_accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, labels=unique_classes, target_names=target_names, zero_division=0)
    report_dict = classification_report(all_labels, all_preds, labels=unique_classes, target_names=target_names, zero_division=0, output_dict=True)
    assert isinstance(report_dict, dict)
    print(f"[{model_name}] Final Cross-Validated Classification Report:\n", report)
    print(f"[{model_name}] UA (Unweighted Accuracy): {ua*100:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=unique_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Final Confusion Matrix ({model_name})')
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace(":", "").replace("+", "_")
    plt.savefig(os.path.join(OUTPUT_DIR, f'{safe_name}_final_confusion_matrix.png'))
    plt.close()
    print(f"[*] Final confusion matrix saved as {OUTPUT_DIR}/{safe_name}_final_confusion_matrix.png")
    
    return {
        'accuracy': acc,
        'ua': ua,
        'precision': report_dict['weighted avg']['precision'], # type: ignore
        'recall': report_dict['weighted avg']['recall'], # type: ignore
        'f1': report_dict['weighted avg']['f1-score'] # type: ignore
    }

# ==========================================
# 8. BASELINE COMPARISON (SVM Fusion)
# ==========================================
def run_baseline_svm_fold(XA_train, XV_train, y_train, XA_test, XV_test):
    """
    Trains an SVM on unified, flattened mean-features mapping to provide 
    a non-deep-learning multimodal baseline comparison.
    """
    print("[*] Running Baseline Multimodal SVM Model...")
    # Average across time dimensions
    XA_train_mean = np.mean(XA_train, axis=1)
    XV_train_mean = np.mean(XV_train, axis=1)
    XA_test_mean = np.mean(XA_test, axis=1)
    XV_test_mean = np.mean(XV_test, axis=1)
    
    # Concatenate features (Early Fusion approach for SVM)
    X_train_fused = np.concatenate((XA_train_mean, XV_train_mean), axis=1)
    X_test_fused = np.concatenate((XA_test_mean, XV_test_mean), axis=1)
    
    svm = SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced')
    svm.fit(X_train_fused, y_train)
    
    predictions = svm.predict(X_test_fused)
    
    return predictions


def run_baseline_majority_class_fold(y_train, y_test):
    majority_class = np.bincount(y_train).argmax()
    print(f"[*] Baseline Majority Class: always predicts '{EMOTION_DICT[f'{majority_class+1:02d}']}'")
    return np.full(shape=len(y_test), fill_value=majority_class)

# ==========================================
# 9. VISUALIZATION
# ==========================================
def plot_extracted_features(X_A, X_V, y):
    print("\n[*] Generating visualization of extracted features for a sample...")
    # Get a sample (first one)
    audio_feat = X_A[0]  # Example shape: (130, 121)
    video_feat = X_V[0]  # Example shape: (30, 1404)
    emotion_label = EMOTION_DICT[f"{int(y[0])+1:02d}"]
    
    fig = plt.figure(figsize=(16, 5))
    plt.suptitle(f"Extracted Features Representation (Class: {emotion_label})", fontsize=16)
    
    # 1. Audio Features Heatmap
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(audio_feat.T, aspect='auto', origin='lower', cmap='viridis')
    ax1.set_title("Audio Features over Time\n(MFCC, Delta, ZCR)")
    ax1.set_xlabel("Time Frames")
    ax1.set_ylabel("Feature Index")
    plt.colorbar(im1, ax=ax1)
    
    # 2. Video Features Heatmap
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(video_feat.T, aspect='auto', origin='lower', cmap='plasma')
    ax2.set_title("Video Features over Time\n(Flattened 3D Landmarks)")
    ax2.set_xlabel("Time Frames (sampled)")
    ax2.set_ylabel("Landmark Vector Index")
    plt.colorbar(im2, ax=ax2)
    
    # 3. Reconstructed 2D Face from a single frame (middle frame)
    ax3 = plt.subplot(1, 3, 3)
    mid_frame_idx = video_feat.shape[0] // 2
    
    # Check if the video feature vector is not entirely zeros
    if np.any(video_feat[mid_frame_idx]):
        mid_frame_landmarks = video_feat[mid_frame_idx].reshape(-1, 3) # (468, 3)
        xs = mid_frame_landmarks[:, 0]
        ys = -mid_frame_landmarks[:, 1] # Invert Y for correct rendering orientation
        ax3.scatter(xs, ys, s=5, color='coral')
        ax3.set_title(f"Reconstructed Face Landmarks\n(Frame {mid_frame_idx})")
    else:
        ax3.text(0.5, 0.5, "No Video Features\n(Audio-Only File)", 
                 horizontalalignment='center', verticalalignment='center')
        ax3.set_title("Reconstructed Face Landmarks")
        
    ax3.set_xlabel("X Coordinate")
    ax3.set_ylabel("Y Coordinate")
    ax3.set_aspect('equal', adjustable='datalim')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    save_path = os.path.join(OUTPUT_DIR, "sample_extracted_features.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[*] Saved sample feature visualization to {save_path}")

def plot_balanced_distribution(y_balanced):
    """
    Visualizes the class balance after oversampling.
    """
    print("[*] Plotting Balanced Distribution (Oversampled)...")
    plt.figure(figsize=(10, 5))
    emotion_names = [EMOTION_DICT[f"{idx+1:02d}"] for idx in y_balanced]
    sns.countplot(x=emotion_names, order=EMOTION_DICT.values(), palette="viridis")
    plt.title("Balanced Class Distribution (After Random Over-Sampling)")
    plt.xlabel("Emotions")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_balanced_distribution.png"))
    plt.close()
    print(f"[*] Balanced distribution chart saved in {OUTPUT_DIR}/eda_balanced_distribution.png")

def plot_comparative_metrics(results_dict):
    print("\n[*] Plotting Comparative Performance Metrics...")
    models = list(results_dict.keys())
    display_names = [m.replace("Arch_", "").replace("_", " ") for m in models]
    
    accs = [results_dict[m]['accuracy'] for m in models]
    precs = [results_dict[m]['precision'] for m in models]
    recs = [results_dict[m]['recall'] for m in models]
    f1s = [results_dict[m]['f1'] for m in models]
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5*width, accs, width, label='Accuracy', color='#4c72b0')
    ax.bar(x - 0.5*width, precs, width, label='Precision', color='#dd8452')
    ax.bar(x + 0.5*width, recs, width, label='Recall', color='#55a868')
    ax.bar(x + 1.5*width, f1s, width, label='F1-Score', color='#c44e52')
    
    ax.set_ylabel('Score')
    ax.set_title('Comparative Performance Metrics across Models')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparative_metrics.png"))
    plt.close()
    print(f"[*] Comparative metrics saved in {OUTPUT_DIR}/comparative_metrics.png")

def plot_loss_curves(train_losses, val_losses, model_name="Multimodal"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title(f'{model_name} Training Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace(":", "").replace("+", "_")
    plt.savefig(os.path.join(OUTPUT_DIR, f'{safe_name}_loss_curves.png'))
    plt.close()
    print(f"[*] Loss curves saved as {OUTPUT_DIR}/{safe_name}_loss_curves.png")


# ==========================================
# 10. EXPLAINABILITY (LIME / SHAP)
# ==========================================
def explain_best_model(best_model_name, best_model, train_loader, test_loader):
    print(f"\n" + "="*50)
    print(f" EXPLAINABILITY (SHAP/LIME): {best_model_name}")
    print("="*50)
    
    # We implement a LIME-style Feature Occlusion to guarantee robust insights 
    # into which Modality AND which Timesteps contribute most.
    print("[*] Performing LIME-style Temporal Occlusion Analysis...")
    best_model.eval()
    
    # 1. Grab a single test sample that was predicted correctly (if possible)
    test_a, test_v, true_label = None, None, None
    pred_label = 0
    for batch_a, batch_v, labels in test_loader:
        batch_a, batch_v, labels = batch_a.to(DEVICE), batch_v.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            outputs = best_model(batch_a, batch_v)
            preds = torch.argmax(outputs, dim=1)
        
        # Find a correct prediction
        correct_idx = (preds == labels).nonzero(as_tuple=True)[0]
        if len(correct_idx) > 0:
            idx = int(correct_idx[0].item())
            test_a = batch_a[idx:idx+1]
            test_v = batch_v[idx:idx+1]
            true_label = int(labels[idx].item())
            pred_label = int(preds[idx].item())
            break
            
    if test_a is None or test_v is None or true_label is None:
        print("[!] Could not find a correct prediction in test set to explain.")
        return
        
    assert test_a is not None and test_v is not None and true_label is not None
    pred_label_idx = int(pred_label)

    with torch.no_grad():
        base_outputs = best_model(test_a, test_v)
        base_probs = torch.softmax(base_outputs, dim=1)[0]
        base_confidence = base_probs[pred_label_idx].item()
        
    emotion_names = list(EMOTION_DICT.values())
    print(f"    -> Sample True Emotion: {emotion_names[true_label]}")
    print(f"    -> Sample Pred Emotion: {emotion_names[pred_label_idx]} (Confidence: {base_confidence*100:.2f}%)")
    
    # 2. Modality Level Importance (Drop in Confidence when zeroed)
    with torch.no_grad():
        out_no_audio = best_model(torch.zeros_like(test_a), test_v)
        prob_no_audio = torch.softmax(out_no_audio, dim=1)[0][pred_label_idx].item()
        
        out_no_video = best_model(test_a, torch.zeros_like(test_v))
        prob_no_video = torch.softmax(out_no_video, dim=1)[0][pred_label_idx].item()
        
    audio_importance = max(0, base_confidence - prob_no_audio)
    video_importance = max(0, base_confidence - prob_no_video)
    
    # Plot Modality Importance
    plt.figure(figsize=(8, 5))
    plt.bar(["Audio Features", "Video Features"], [audio_importance, video_importance], color=['skyblue', 'lightcoral'])
    plt.title(f"Modality Importance for Predicting '{emotion_names[pred_label_idx]}'\n(Drop in Confidence when occluded)")
    plt.ylabel("Confidence Drop")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{best_model_name}_explainer_modality.png"))
    plt.close()
    print(f"[*] Saved Modality Importance plot to {OUTPUT_DIR}/{best_model_name}_explainer_modality.png")
    
    # 3. Temporal Level Importance (LIME Sliding Window)
    time_steps_a = test_a.size(1)
    audio_time_importance = []
    window_a = max(1, time_steps_a // 10)
    
    for i in range(0, time_steps_a, window_a):
        occluded_a = test_a.clone()
        end_idx = min(i + window_a, time_steps_a)
        occluded_a[:, i:end_idx, :] = 0 
        with torch.no_grad():
            prob = torch.softmax(best_model(occluded_a, test_v), dim=1)[0][pred_label_idx].item()
        audio_time_importance.append(base_confidence - prob)
        
    time_steps_v = test_v.size(1)
    video_time_importance = []
    window_v = max(1, time_steps_v // 10)
    
    for i in range(0, time_steps_v, window_v):
        occluded_v = test_v.clone()
        end_idx = min(i + window_v, time_steps_v)
        occluded_v[:, i:end_idx, :] = 0
        with torch.no_grad():
            prob = torch.softmax(best_model(test_a, occluded_v), dim=1)[0][pred_label_idx].item()
        video_time_importance.append(base_confidence - prob)
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(np.linspace(0, 3.0, len(audio_time_importance)), audio_time_importance, marker='o', color='blue')
    ax1.set_title("Audio Temporal Feature Importance (LIME Occlusion)")
    ax1.set_ylabel("Confidence Drop")
    ax1.set_xlabel("Time (seconds)")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(np.linspace(0, 3.0, len(video_time_importance)), video_time_importance, marker='o', color='red')
    ax2.set_title("Video Temporal Feature Importance (LIME Occlusion)")
    ax2.set_ylabel("Confidence Drop")
    ax2.set_xlabel("Time (seconds)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{best_model_name}_explainer_temporal.png"))
    plt.close()
    print(f"[*] Saved Temporal Importance plot to {OUTPUT_DIR}/{best_model_name}_explainer_temporal.png")
    
    # 4. Deep SHAP Explainer (if supported by architecture)
    try:
        import shap  # type: ignore
        print("\n[*] SHAP library detected. Attempting Deep Gradient Integration...")
        bg_a, bg_v, _ = next(iter(train_loader))
        bg_a, bg_v = bg_a[:8].to(DEVICE), bg_v[:8].to(DEVICE)
        
        explainer = shap.GradientExplainer(best_model, [bg_a, bg_v])
        shap_values = explainer.shap_values([test_a, test_v])
        print("    -> Successfully computed theoretical SHAP Gradient values for this sample!")
    except ImportError:
        print("\n[!] Note: `shap` package not installed. Skipping PyTorch deep gradient SHAP analysis.")
    except Exception as e:
        print(f"\n[!] Built-in SHAP gradient failed for this complex multi-head architecture: {e}")


# ==========================================
# MAIN ROUTINE
# ==========================================
if __name__ == "__main__":
    # Ensure GPU is available
    if not torch.cuda.is_available():
        print("[!] Warning: CUDA not available. Running on CPU — training will be slow.")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()} | Using device: {DEVICE}")
    print("=" * 60)
    print(" RAVDESS MULTIMODAL (Audio + Video) Emotion Pipeline")
    print("=" * 60)
    
    # Run the count utility
    count_video_files(DATA_PATH)
    
    # 1. Loading data
    file_paths, labels, actors = load_data(DATA_PATH)
    if not file_paths:
        exit("[!] Fatal Error: Dataset not found or empty.")
        
    # Plot dataset distribution BEFORE limiting it, to see the true scale
    plot_modalities_distribution(file_paths, labels)
        
    # Limit to first 100 files for quick testing (user can change later)
    # Set N_LIMIT = None to process the entire dataset
    N_LIMIT = None
    if N_LIMIT:
        import random
        # Combine, shuffle, and unzip to keep them synced
        combined = list(zip(file_paths, labels, actors))
        random.seed(42) # For reproducibility 
        random.shuffle(combined)
        file_paths, labels, actors = zip(*combined)
        
        file_paths = list(file_paths[:N_LIMIT])
        labels = list(labels[:N_LIMIT])
        actors = list(actors[:N_LIMIT])
        print(f"[*] Limiting dataset to {N_LIMIT} random files for testing.")
        
    labels = np.array(labels)
    actors = np.array(actors)
    
    # 2. EDA
    perform_eda(labels, file_paths)
    
    # 3. Extracting Features (Audio + Visual Landmarks)
    CACHE_FILE = os.path.join(OUTPUT_DIR, f"ravdess_features_dynamic_{len(file_paths)}.npz")
    if os.path.exists(CACHE_FILE):
        print(f"[*] Found cached features at {CACHE_FILE}. Loading...")
        data = np.load(CACHE_FILE)
        X_A = data['X_A'].astype(np.float32)
        X_V = data['X_V'].astype(np.float32)
        y = data['y']
        
        if 'actors' in data:
            actors = data['actors']
        else:
            if len(actors) != len(y):
                print("[!] Warning: Cached labels count differs from file paths. Padding/truncating actors. Best to delete cache if issues arise.")
                if len(actors) > len(y):
                    actors = actors[:len(y)]
                else:
                    actors = np.pad(actors, (0, len(y) - len(actors)), mode='edge')
        
        print(f"[*] Loaded cached data.")
        print(f"    - Audio sequence shape: {X_A.shape}")
        print(f"    - Video sequence shape: {X_V.shape}")
        print(f"    - Labels shape: {y.shape}")
    else:
        print(f"[*] Starting Dual-Feature Extraction for {len(file_paths)} files.")
        print("    This will engage both Audio processing and FaceMesh processing.")
        print("    (Depending on dataset size, grabbing a coffee is advised \u2615)")
        
        X_audio_list = []
        X_video_list = []
        y_list = []
        actors_list = []
        
        for i, fp in enumerate(file_paths):
            a_feat, v_feat = extract_features_from_file(fp)
            if a_feat is not None and v_feat is not None:
                X_audio_list.append(a_feat)
                X_video_list.append(v_feat)
                y_list.append(labels[i])
                actors_list.append(actors[i])
                
            if (i+1) % 50 == 0:
                print(f"    -> Processed {i+1} / {len(file_paths)} files...")
                
        X_A = np.array(X_audio_list, dtype=np.float32)
        X_V = np.array(X_video_list, dtype=np.float32)
        y = np.array(y_list)
        actors = np.array(actors_list)
        
        print(f"\n[*] Extraction complete.")
        print(f"    - Audio sequence shape: {X_A.shape}")
        print(f"    - Video sequence shape: {X_V.shape}")
        print(f"    - Labels shape: {y.shape}")
        
        np.savez_compressed(CACHE_FILE, X_A=X_A, X_V=X_V, y=y, actors=actors)
        print(f"[*] Features safely cached to {CACHE_FILE} for future use.")
    
    num_audio_features = X_A.shape[2]
    num_video_features = X_V.shape[2]

    # Visualize a sample of the extracted features before proceeding
    if len(X_A) > 0:
        plot_extracted_features(X_A, X_V, y)

    # K-Fold Cross-Validation Setup (Grouped by Actor to avoid overfit/leakage)
    K_FOLDS = 5
    skf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    def get_fresh_models():
        return {
            "Arch_1_CNN_LSTM": Multimodal_CNN_LSTM(
                audio_dim=num_audio_features, 
                video_dim=num_video_features, 
                hidden_dim=128, 
                num_classes=8
            ).to(DEVICE),
            "Arch_2_Attention_Gated": Arch2_Attention_Gated_Fusion(
                audio_dim=num_audio_features,
                video_dim=num_video_features,
                num_classes=8
            ).to(DEVICE),
            "Arch_3_Cross_Modal_Transformer": CrossModalTransformer(
                audio_dim=num_audio_features, 
                video_dim=num_video_features, 
                d_model=128, 
                nhead=4, 
                num_layers=1, 
                num_classes=8
            ).to(DEVICE),
            "Arch_4_Bottleneck_Fusion": Arch4_Bottleneck_Fusion(
                audio_dim=num_audio_features,
                video_dim=num_video_features,
                num_classes=8
            ).to(DEVICE)
        }

    model_names = list(get_fresh_models().keys()) + ["Baseline_SVM", "Baseline_Majority_Class"]
    results_leaderboard = {}
    
    # Track models for explainability
    best_trained_models = {}
    best_dataloaders = {}

    print(f"\n[*] Starting {K_FOLDS}-Fold Cross Validation...")

    for model_name in model_names:
        print(f"\n" + "="*50)
        print(f" CROSS-VALIDATING: {model_name}")
        print("="*50)
        
        best_fold_acc = -1
        
        model_all_preds = []
        model_all_labels = []
        
        for fold, (train_val_idx, test_idx) in enumerate(skf.split(X_A, y, groups=actors), 1):
            print(f"\n--- Fold {fold}/{K_FOLDS} ({model_name}) ---")
            
            # Further split train_val into train and val for early stopping safely by actor group
            gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
            
            # Use next() to get the single split from GroupShuffleSplit
            train_sub_idx, val_sub_idx = next(gss.split(
                train_val_idx, 
                y[train_val_idx], 
                groups=actors[train_val_idx]
            ))
            
            train_idx = train_val_idx[train_sub_idx]
            val_idx = train_val_idx[val_sub_idx]

            # Slicing datasets using matched indices
            XA_train_raw, XV_train_raw, y_train = X_A[train_idx], X_V[train_idx], y[train_idx]
            XA_val_raw, XV_val_raw, y_val = X_A[val_idx], X_V[val_idx], y[val_idx]
            XA_test_raw, XV_test_raw, y_test = X_A[test_idx], X_V[test_idx], y[test_idx]

            # Fit Scalers strictly on training data to prevent Data Leakage
            scaler_A = StandardScaler()
            b, t, f = XA_train_raw.shape
            XA_train = scaler_A.fit_transform(XA_train_raw.reshape(-1, f)).astype(np.float32).reshape(b, t, f)
            b, t, f = XA_val_raw.shape
            XA_val = scaler_A.transform(XA_val_raw.reshape(-1, f)).astype(np.float32).reshape(b, t, f)
            b, t, f = XA_test_raw.shape
            XA_test = scaler_A.transform(XA_test_raw.reshape(-1, f)).astype(np.float32).reshape(b, t, f)

            scaler_V = StandardScaler()
            b, t, f = XV_train_raw.shape
            XV_train = scaler_V.fit_transform(XV_train_raw.reshape(-1, f)).astype(np.float32).reshape(b, t, f)
            b, t, f = XV_val_raw.shape
            XV_val = scaler_V.transform(XV_val_raw.reshape(-1, f)).astype(np.float32).reshape(b, t, f)
            b, t, f = XV_test_raw.shape
            XV_test = scaler_V.transform(XV_test_raw.reshape(-1, f)).astype(np.float32).reshape(b, t, f)

            if model_name == "Baseline_SVM":
                # For Baseline_SVM we need the full train_val set merged, correctly scaled
                XA_train_full_raw, XV_train_full_raw, y_train_full = X_A[train_val_idx], X_V[train_val_idx], y[train_val_idx]
                
                scaler_A_full = StandardScaler()
                b, t, f = XA_train_full_raw.shape
                XA_train_full = scaler_A_full.fit_transform(XA_train_full_raw.reshape(-1, f)).astype(np.float32).reshape(b, t, f)
                b, t, f = XA_test_raw.shape
                XA_test_svm = scaler_A_full.transform(XA_test_raw.reshape(-1, f)).astype(np.float32).reshape(b, t, f)

                scaler_V_full = StandardScaler()
                b, t, f = XV_train_full_raw.shape
                XV_train_full = scaler_V_full.fit_transform(XV_train_full_raw.reshape(-1, f)).astype(np.float32).reshape(b, t, f)
                b, t, f = XV_test_raw.shape
                XV_test_svm = scaler_V_full.transform(XV_test_raw.reshape(-1, f)).astype(np.float32).reshape(b, t, f)

                predictions = run_baseline_svm_fold(XA_train_full, XV_train_full, y_train_full, XA_test_svm, XV_test_svm)
                
                model_all_preds.extend(predictions)
                model_all_labels.extend(y_test)
                
                fold_acc = accuracy_score(y_test, predictions)
                if fold_acc > best_fold_acc:
                    best_fold_acc = fold_acc

            elif model_name == "Baseline_Majority_Class":
                predictions = run_baseline_majority_class_fold(y[train_val_idx], y_test)
                model_all_preds.extend(predictions)
                model_all_labels.extend(y_test)
                fold_acc = accuracy_score(y_test, predictions)
                if fold_acc > best_fold_acc:
                    best_fold_acc = fold_acc
                    
            else:
                # APPLY OVERSAMPLING to handle class imbalance before training
                XA_train_balanced, XV_train_balanced, y_train_balanced = manual_random_over_sample(XA_train, XV_train, y_train)
                
                # Plot balanced distribution chart once for the first fold of the first deep model
                if fold == 1 and not os.path.exists(os.path.join(OUTPUT_DIR, "eda_balanced_distribution.png")):
                    plot_balanced_distribution(y_train_balanced)

                train_dataset = RAVDESSMultimodalDataset(XA_train_balanced, XV_train_balanced, y_train_balanced, augment=True)
                val_dataset = RAVDESSMultimodalDataset(XA_val, XV_val, y_val, augment=False)
                test_dataset = RAVDESSMultimodalDataset(XA_test, XV_test, y_test, augment=False)
                
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
                val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
                
                # Re-initialize the model to avoid data leakage
                model = get_fresh_models()[model_name]
                
                # Handle class imbalance for the current fold safely (ensuring 8 classes)
                # Note: Oversampling is already applied, so class_weights are set to uniform
                criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Adjusted label smoothing
                optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3) # Increased weight decay
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-5) # Slower learning rate decay
                
                model_trained, train_loss, val_loss = train_multimodal_model(
                    model=model, 
                    train_loader=train_loader, 
                    val_loader=val_loader, 
                    criterion=criterion, 
                    optimizer=optimizer, 
                    scheduler=scheduler, 
                    epochs=EPOCHS
                )
                
                all_labels_fold, all_preds_fold = evaluate_model(model_trained, test_loader)
                model_all_labels.extend(all_labels_fold)
                model_all_preds.extend(all_preds_fold)
                
                fold_acc = accuracy_score(all_labels_fold, all_preds_fold)
                    
                if fold_acc > best_fold_acc:
                    best_fold_acc = fold_acc
                    best_trained_models[model_name] = copy.deepcopy(model_trained.state_dict())
                    best_dataloaders[model_name] = (train_loader, test_loader)
                    # Only plot loss curve for the best fold
                    plot_loss_curves(train_loss, val_loss, model_name=f"{model_name}")
                    
            # Memory cleanup
            del XA_train_raw, XV_train_raw, XA_val_raw, XV_val_raw, XA_test_raw, XV_test_raw
            del XA_train, XV_train, XA_val, XV_val, XA_test, XV_test
            gc.collect()
        
        # Now generated final aggregated report for the whole dataset across all folds
        metrics = generate_performance_report(model_all_labels, model_all_preds, model_name)
        results_leaderboard[model_name] = metrics
        
        print(f"[*] {model_name} Overall CV Performance: " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
        
        if model_name not in ["Baseline_SVM", "Baseline_Majority_Class"] and model_name in best_trained_models:
            torch.save(best_trained_models[model_name], os.path.join(OUTPUT_DIR, f"ravdess_{model_name}_best.pth"))
            print(f"[*] Saved best performing fold for {model_name} in {OUTPUT_DIR}/.")
        elif model_name not in ["Baseline_SVM", "Baseline_Majority_Class"]:
            print(f"[!] Warning: No best weights saved for {model_name} — skipping save.")

    # Final Leaderboard Output
    print("\n" + "🌟"*25)
    print("    FINAL CROSS-VALIDATED LEADERBOARD    ")
    print("🌟"*25)
    best_model_name = None
    best_acc = -1
    for m_name, metrics in sorted(results_leaderboard.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        acc = metrics['accuracy']
        ua_val = metrics.get('ua', 0.0)
        if ua_val > 0:
            print(f" - {m_name:<35}: {acc * 100:>6.2f}% (Acc) | {ua_val * 100:>6.2f}% (UA)")
        else:
            print(f" - {m_name:<35}: {acc * 100:>6.2f}% (Acc)")
        if acc > best_acc and m_name not in ["Baseline_SVM", "Baseline_Majority_Class"]:
            best_acc = acc
            best_model_name = m_name
    print("🌟"*25)
    
    # Render comparative graph
    plot_comparative_metrics(results_leaderboard)
    
    if best_model_name:
        # Re-load the best fold state for testing with LIME/SHAP
        best_model_instance = get_fresh_models()[best_model_name]
        best_model_instance.load_state_dict(best_trained_models[best_model_name])
        t_loader, ts_loader = best_dataloaders[best_model_name]
        explain_best_model(best_model_name, best_model_instance, t_loader, ts_loader)

    print("\n[*] Pipeline successfully achieved! All models cross-validated and saved.")
