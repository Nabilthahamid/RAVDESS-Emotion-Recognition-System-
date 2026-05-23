# ESRA-RELA++: Emotion-Saliency and Reliability-Aware Multimodal Fusion for RAVDESS Emotion Recognition

A publication-oriented, reproducible multimodal emotion-recognition project using the **RAVDESS** audio-visual dataset.  
The project focuses on **subject-wise actor-independent emotion recognition**, where test actors are unseen during training.

The final selected configuration uses **calibrated simple-average fusion** over four active branches:

1. Hybrid audio branch  
2. OpenFace-guided facial video branch  
3. OpenFace Action Unit statistical branch  
4. OpenFace Action Unit dynamic/FACS trajectory branch  

The optional LoRA audio branch, meta-fusion, pair specialists, gender specialists, and Sad-vs-rest specialist were also implemented and evaluated as ablations.

---

## Table of Contents

- [Project Motivation](#project-motivation)
- [Main Contributions](#main-contributions)
- [Final Result Summary](#final-result-summary)
- [Dataset Used](#dataset-used)
- [Why Only 1440 Videos Are Used](#why-only-1440-videos-are-used)
- [Architecture Overview](#architecture-overview)
- [Full Workflow](#full-workflow)
- [Active Branches in the Best Model](#active-branches-in-the-best-model)
- [Experimental Versions](#experimental-versions)
- [LoRA Experiment: Why It Was Not Selected](#lora-experiment-why-it-was-not-selected)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [OpenFace Setup](#openface-setup)
- [How to Run](#how-to-run)
- [Output Files](#output-files)
- [How to Interpret Results](#how-to-interpret-results)
- [Comparison With Prior Work](#comparison-with-prior-work)
- [Challenges Faced](#challenges-faced)
- [Troubleshooting](#troubleshooting)
- [Recommended Paper Framing](#recommended-paper-framing)
- [Future Improvements](#future-improvements)
- [Citation / Dataset Attribution](#citation--dataset-attribution)

---

## Project Motivation

Emotion recognition from audio-visual data is challenging because emotions are expressed through multiple signals:

- voice tone,
- facial expression,
- facial muscle movement,
- temporal changes,
- actor-specific variation,
- recording quality,
- and modality reliability.

Many existing RAVDESS papers report high accuracy, but results are often difficult to compare because different works use different splitting protocols. This project focuses on a stricter and more defensible setup:

> **Subject-wise 5-fold cross-validation**, where the actors in the test fold are never seen during training.

This makes the task harder than random or trial-based splitting, but it better evaluates generalization to unseen speakers/actors.

---

## Main Contributions

This project proposes and evaluates **ESRA-RELA++**, an emotion-saliency and reliability-aware multimodal fusion framework.

Key contributions:

1. **Hybrid audio feature extraction**
   - Frozen SSL audio embedding from Wav2Vec2.
   - Handcrafted acoustic and prosodic features such as MFCC, delta MFCC, chroma, mel spectrogram, spectral features, pitch/F0, RMS, and ZCR.

2. **OpenFace-guided video saliency**
   - Emotion-salient frames are selected using OpenFace Action Unit activity and confidence.
   - Faces are cropped using 68 OpenFace landmarks.
   - MobileNetV3 is used for facial frame embedding.
   - Temporal pooling is applied using mean, standard deviation, max, frame difference, first frame, and last frame.

3. **OpenFace AU statistical branch**
   - Extracts global statistics from AU intensity trajectories.

4. **OpenFace AU-dynamic/FACS branch**
   - Extracts segment-wise AU behavior, AU velocity, peak timing, and FACS group trajectory features.

5. **Leakage-safe subject-wise evaluation**
   - Uses actor-independent 5-fold splitting.
   - Test actors are never used during training.

6. **Out-of-fold modality probability generation**
   - Each modality produces out-of-fold probabilities for fair fusion.

7. **Temperature calibration**
   - Modality posteriors are calibrated before fusion.

8. **Reliability feature logging**
   - The system records audio, video, and AU reliability indicators.

9. **Optional LoRA audio branch**
   - Implements fold-safe parameter-efficient audio adaptation.
   - Evaluated as an ablation.

10. **Specialist correction modules**
    - Weak-pair specialists and Sad-vs-rest specialist were implemented and tested.
    - They were not selected in the final best-performing configuration because they did not improve the final score.

11. **Detailed diagnostics and paper-ready outputs**
    - Confusion matrices,
    - classification reports,
    - modality ablations,
    - uncertainty logs,
    - fold summaries,
    - sequential stage ablations,
    - and summary JSON files.

---

## Final Result Summary

The best-performing configuration was:

> **ESRA-RELA++ Calibrated Simple-Average Fusion without LoRA and without specialist correction**

Final subject-wise 5-fold result:

| Metric | Result |
|---|---:|
| Accuracy | **72.40% ± 3.81** |
| Macro-F1 | **71.94% ± 4.12** |
| UAR / Balanced Accuracy | **72.67% ± 3.82** |

Fold-wise result:

| Fold | Test Actors | Accuracy | Macro-F1 | UAR | Uncertain Samples |
|---:|---|---:|---:|---:|---:|
| 0 | 2, 5, 14, 15, 16 | 73.00% | 72.58% | 74.06% | 58 |
| 1 | 3, 6, 7, 13, 18 | 77.33% | 76.14% | 77.81% | 60 |
| 2 | 10, 11, 12, 19, 20 | 70.33% | 69.43% | 70.00% | 64 |
| 3 | 8, 17, 21, 23, 24 | 66.33% | 65.40% | 66.88% | 66 |
| 4 | 1, 4, 9, 22 | 75.00% | 76.13% | 74.61% | 42 |

Class-wise summary from the best run:

| Emotion | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Neutral | 71.15% | 77.08% | 74.00% |
| Calm | 75.24% | 80.73% | 77.89% |
| Happy | 83.50% | 86.98% | 85.20% |
| Sad | 61.49% | 51.56% | 56.09% |
| Angry | 77.04% | 78.65% | 77.84% |
| Fearful | 62.24% | 63.54% | 62.89% |
| Disgust | 80.19% | 86.46% | 83.21% |
| Surprised | 62.94% | 55.73% | 59.12% |

Strong classes:

- Happy
- Disgust
- Calm
- Angry
- Neutral

Weak classes:

- Sad
- Surprised
- Fearful

---

## Dataset Used

Dataset: **RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song**

This project uses the **full audio-visual speech** subset only.

RAVDESS filename format:

```text
Modality - Vocal Channel - Emotion - Intensity - Statement - Repetition - Actor
```

Example:

```text
01-01-03-01-01-01-01.mp4
```

Meaning:

| Code | Meaning |
|---|---|
| 01 | Full audio-video |
| 01 | Speech |
| 03 | Happy |
| 01 | Normal intensity |
| 01 | Statement 1 |
| 01 | First repetition |
| 01 | Actor 1 |

---

## Why Only 1440 Videos Are Used

The full folder may contain about **4904 `.mp4` files**, but those include multiple subsets:

| Subset | Approx. Count | Used in Main Experiment? |
|---|---:|---|
| Full-AV Speech | 1440 | Yes |
| Video-only Speech | 1440 | No |
| Full-AV Song | 1012 | No |
| Video-only Song | 1012 | No |
| Total MP4 | 4904 | Mixed dataset |

The code filters only:

```python
modality == 1 and channel == 1
```

That means:

```text
01 = full audio-video
01 = speech
```

So the final number becomes:

```text
1440 full-AV speech videos
```

This is intentional because:

1. The project is audio-visual emotion recognition.
2. Video-only files do not contain audio.
3. Song and speech are different domains.
4. The selected setup is cleaner and easier to compare with subject-wise speech-based RAVDESS baselines.

---

## Architecture Overview

Final selected architecture:

```text
RAVDESS full-AV speech video
        │
        ├── Audio waveform
        │       ├── frozen Wav2Vec2 SSL embedding
        │       └── handcrafted acoustic/prosody features
        │
        ├── Video frames
        │       ├── OpenFace saliency frame selection
        │       ├── face crop using 68 facial landmarks
        │       └── MobileNetV3 temporal frame features
        │
        └── OpenFace AU CSV
                ├── AU statistical features
                └── AU-dynamic/FACS trajectory features

Modality classifiers
        ↓
Temperature calibration
        ↓
Calibrated simple-average fusion
        ↓
Final emotion prediction
```

The optional/ablated architecture also includes:

```text
Optional LoRA audio branch
Reliability-aware meta-fusion
Weak-pair specialists
Gender specialists
Sad-vs-rest specialist
Uncertainty logging
```

---

## Full Workflow

### Step 1: Discover videos

The code recursively searches the dataset directory for video files:

```python
*.mp4, *.avi, *.mov, *.mkv
```

Then it parses RAVDESS filenames and keeps only full-AV speech clips.

---

### Step 2: Load OpenFace CSV files

Each video is matched with a corresponding OpenFace CSV.

OpenFace provides:

- frame confidence,
- success score,
- 68 facial landmarks,
- Action Unit intensity columns,
- Action Unit presence columns.

These CSV files are used in three places:

1. saliency-based frame selection,
2. face cropping,
3. AU feature extraction.

---

### Step 3: Extract audio features

The audio branch extracts:

#### SSL features

From:

```text
facebook/wav2vec2-base-960h
```

Pooling strategy:

```text
mean pooling + standard deviation pooling + max pooling
```

#### Handcrafted acoustic/prosodic features

- MFCC mean/std,
- delta MFCC mean/std,
- chroma mean/std,
- log-mel mean/std,
- spectral centroid mean/std,
- spectral bandwidth mean/std,
- spectral rolloff mean/std,
- spectral contrast mean/std,
- F0/pitch statistics,
- RMS,
- zero crossing rate.

Final audio vector:

```text
SSL embedding + handcrafted acoustic features
```

---

### Step 4: Extract video features

Video workflow:

1. Read frames from video.
2. Use OpenFace AU activity and confidence to score frames.
3. Select emotion-salient frames.
4. Crop face using 68 landmark coordinates.
5. Extract MobileNetV3 embeddings.
6. Apply temporal pooling.

Temporal pooling:

```text
mean
std
max
delta mean
first frame
last frame
```

---

### Step 5: Extract AU statistical features

For each AU, the code extracts:

```text
mean
standard deviation
maximum
minimum
median
IQR
mean absolute velocity
start-to-end change
```

This captures global facial muscle behavior.

---

### Step 6: Extract AU-dynamic/FACS trajectory features

This branch captures temporal AU movement.

Features include:

- segment-wise AU means,
- segment-wise AU maxima,
- AU velocity,
- AU max velocity,
- AU peak timing,
- FACS group trajectory statistics.

FACS groups:

| Group | Example AUs |
|---|---|
| Brow | AU01, AU02, AU04 |
| Eye | AU05, AU06, AU07, AU45 |
| Nose | AU09, AU10 |
| Mouth | AU12, AU14, AU15, AU17, AU20, AU23, AU25, AU26 |

---

### Step 7: Build reliability features

Reliability vector includes:

- audio RMS,
- audio ZCR,
- audio feature norm,
- number of video frames used,
- video saliency availability,
- video feature norm,
- AU found flag,
- AU confidence,
- AU success rate,
- AU-dynamic found flag,
- AU-dynamic confidence,
- AU-dynamic success rate.

In the final best run, reliability is mostly used for diagnostics/alternate fusion experiments.

---

### Step 8: Train modality classifiers

Each modality is trained separately using actor-group-aware out-of-fold logic.

Active modalities in best run:

```text
audio
video
AU
AU-dynamic
```

Each branch produces an 8-class probability vector.

---

### Step 9: Temperature calibration

Each modality probability is calibrated using a scalar temperature.

This reduces overconfident probability estimates.

---

### Step 10: Final fusion

The best configuration uses unweighted calibrated simple-average fusion:

```text
final_probability =
(audio_probability + video_probability + AU_probability + AU_dynamic_probability) / 4
```

Then:

```text
final_prediction = argmax(final_probability)
```

---

### Step 11: Evaluate subject-wise folds

The model uses fixed subject-wise 5-fold splits.

No test actor appears in training for that fold.

---

### Step 12: Save results

The code saves:

- fold summary,
- final summary JSON,
- confusion matrix,
- normalized confusion matrix,
- classification report,
- uncertainty log,
- ablation metrics,
- specialist configuration,
- sequential stage ablation,
- diagnostics.

---

## Active Branches in the Best Model

Best model configuration:

| Component | Status |
|---|---|
| Hybrid audio branch | ON |
| Video branch | ON |
| AU statistical branch | ON |
| AU-dynamic branch | ON |
| LoRA audio branch | OFF |
| Meta-fusion | Not used as final decision |
| Weighted dual fusion | OFF |
| Pair specialists | OFF |
| Gender specialists | No effect because pair specialists are OFF |
| Sad-vs-rest specialist | OFF |
| Temperature calibration | ON |
| Uncertainty logging | ON |
| Metadata covariates | OFF |

Important:

> The best result is not from the most complex model.  
> The best result comes from the cleaner calibrated average-fusion setting.

---

## Experimental Versions

The project went through many improvements.

### Early deep fusion version

Initial architecture used heavy deep components:

- WavLM / Wav2Vec2 branch,
- VideoMAE / visual branch,
- AU transformer branch,
- gated fusion,
- actor adversarial ideas,
- specialist losses.

Problem:

- Too complex for only 1440 RAVDESS full-AV speech samples.
- Overfitting risk was high.
- Accuracy stayed low around the mid-50% range.

---

### RELA-HLF direction

Next idea:

- extract stable modality features,
- train light classifiers,
- use OOF stacking,
- add reliability features,
- add weak-pair specialists.

This improved stability and made the pipeline easier to defend academically.

---

### ESRA-RELA++ direction

Final improved idea:

- hybrid audio,
- OpenFace saliency video,
- AU statistics,
- AU-dynamic trajectory,
- optional LoRA audio,
- calibrated OOF probability fusion,
- ablations,
- diagnostics.

This became the final project direction.

---

## LoRA Experiment: Why It Was Not Selected

LoRA was implemented as a fold-safe optional audio branch.

It was trained only inside train folds and never on outer-test actors.

### With LoRA

| Metric | Result |
|---|---:|
| Accuracy | 70.07% ± 3.81 |
| Macro-F1 | 68.56% ± 5.55 |
| UAR | 69.95% ± 4.08 |

### Without LoRA

| Metric | Result |
|---|---:|
| Accuracy | 71.48% ± 3.65 |
| Macro-F1 | 71.16% ± 3.84 |
| UAR | 71.77% ± 3.42 |

### Best simple-average no-LoRA

| Metric | Result |
|---|---:|
| Accuracy | 72.40% ± 3.81 |
| Macro-F1 | 71.94% ± 4.12 |
| UAR | 72.67% ± 3.82 |

Conclusion:

> LoRA was working technically, but it did not improve subject-wise performance under the current configuration.

Why LoRA may have underperformed:

1. RAVDESS is small.
2. LoRA may overfit actor-specific vocal patterns.
3. The selected base model was not emotion-pretrained.
4. The LoRA branch produced weaker modality-level performance than AU/AU-dynamic.
5. Fusion assigned low trust to LoRA probabilities.
6. The strong AU and AU-dynamic branches already captured useful emotion information.

Paper statement:

```text
The LoRA audio branch was evaluated as a parameter-efficient adaptation module. Although technically successful and leakage-safe, it did not improve subject-wise generalization under the current setting. Therefore, the final selected configuration excludes LoRA and reports it as an ablation.
```

---

## Repository Structure

Suggested GitHub structure:

```text
ESRA-RELA-PlusPlus/
│
├── README.md
├── requirements.txt
├── esra_rela_pp_v15_1_lora_pylance_clean.py
│
├── docs/
│   ├── ESRA_RELA_PP_Detailed_Experiment_Challenges_and_Paper_Guide.md
│   └── methodology_notes.md
│
├── scripts/
│   ├── run_best_no_lora_simpleavg.bat
│   ├── run_lora_ablation.bat
│   └── run_no_lora_full.bat
│
├── results/
│   └── README_results.md
│
└── .gitignore
```

Do not upload:

- RAVDESS dataset,
- OpenFace extracted CSVs if too large,
- cache folders,
- model checkpoints,
- huge result directories.

---

## Environment Setup

Recommended Python:

```text
Python 3.10 or 3.11
```

Install core dependencies:

```bash
pip install torch torchvision torchaudio
pip install transformers librosa opencv-python pandas scikit-learn scipy tqdm matplotlib pillow
```

Optional LoRA dependencies:

```bash
pip install peft accelerate
```

Recommended `requirements.txt`:

```text
torch
torchvision
torchaudio
transformers
librosa
opencv-python
pandas
scikit-learn
scipy
tqdm
matplotlib
pillow
peft
accelerate
```

---

## OpenFace Setup

This project requires OpenFace CSV files for the best setup.

OpenFace is used for:

- AU intensity extraction,
- facial landmarks,
- frame saliency,
- face cropping,
- confidence and success rates.

Example OpenFace command for one video:

```powershell
.\FeatureExtraction.exe -f "D:\CSE427_end_game\RAVDESS\Video_Speech_Actor_01\Actor_01\01-01-03-01-01-01-01.mp4" -out_dir "D:\CSE427_end_game\OpenFace_AU_CSV"
```

For all videos, use a PowerShell loop.

Expected OpenFace folder:

```text
D:\CSE427_end_game\OpenFace_AU_CSV
```

Expected coverage:

```text
1440/1440 = 100%
```

---

## How to Run

### Best final model: no LoRA, calibrated simple average

```bat
cd /d D:\CSE427_end_game

python "D:/CSE427_end_game/esra_rela_pp_v15_1_lora_pylance_clean.py" ^
--data_dir "D:/CSE427_end_game/RAVDESS" ^
--openface_dir "D:/CSE427_end_game/OpenFace_AU_CSV" ^
--cache_dir "D:/CSE427_end_game/esra_rela_pp_v15_1_simpleavg_no_lora_cache" ^
--results_dir "D:/CSE427_end_game/esra_rela_pp_v15_1_simpleavg_no_lora_results" ^
--run_all_folds ^
--base_model logreg ^
--meta_model logreg ^
--dual_fusion_blend 1.0 ^
--no_dual_fusion_weighted ^
--no_specialists ^
--no_sad_specialist ^
--specialist_pca 128
```

### Full no-LoRA model with specialists

```bat
cd /d D:\CSE427_end_game

python "D:/CSE427_end_game/esra_rela_pp_v15_1_lora_pylance_clean.py" ^
--data_dir "D:/CSE427_end_game/RAVDESS" ^
--openface_dir "D:/CSE427_end_game/OpenFace_AU_CSV" ^
--cache_dir "D:/CSE427_end_game/esra_rela_pp_v15_1_no_lora_cache" ^
--results_dir "D:/CSE427_end_game/esra_rela_pp_v15_1_no_lora_results" ^
--run_all_folds ^
--base_model logreg ^
--meta_model logreg ^
--specialist_blend 0.20 ^
--dual_fusion_blend 0.25 ^
--sad_threshold 0.42 ^
--sad_blend 0.18 ^
--sad_min_base_prob 0.05 ^
--specialist_pca 128
```

### LoRA ablation

```bat
cd /d D:\CSE427_end_game

python "D:/CSE427_end_game/esra_rela_pp_v15_1_lora_pylance_clean.py" ^
--data_dir "D:/CSE427_end_game/RAVDESS" ^
--openface_dir "D:/CSE427_end_game/OpenFace_AU_CSV" ^
--cache_dir "D:/CSE427_end_game/esra_rela_pp_v15_1_lora_cache" ^
--results_dir "D:/CSE427_end_game/esra_rela_pp_v15_1_lora_results" ^
--run_all_folds ^
--base_model logreg ^
--meta_model logreg ^
--specialist_blend 0.20 ^
--dual_fusion_blend 0.25 ^
--sad_threshold 0.42 ^
--sad_blend 0.18 ^
--sad_min_base_prob 0.05 ^
--specialist_pca 128 ^
--use_lora_audio ^
--lora_audio_model "facebook/wav2vec2-base-960h" ^
--lora_epochs 8 ^
--lora_batch_size 2 ^
--lora_grad_accum 2 ^
--lora_lr 0.0001 ^
--lora_r 8 ^
--lora_alpha 16 ^
--lora_dropout 0.10
```

---

## Output Files

Important output files:

```text
summary.json
fold_summary.csv
final_results/r01_cm.csv
final_results/r02_cm_norm.csv
final_results/r03_classification_report.csv
final_results/r04_fold_metrics.csv
final_results/r06_ablation_metrics.csv
final_results/r07_uncertain_predictions.csv
final_results/r08_modality_oof_uar.csv
final_results/r09_specialist_config.csv
final_results/r10_sequential_stage_ablation.csv
diagnostics/
```

### `summary.json`

Contains:

- configuration,
- mean accuracy,
- macro-F1,
- UAR,
- class-wise report,
- method notes.

### `fold_summary.csv`

Contains:

- fold ID,
- test actors,
- fold accuracy,
- macro-F1,
- UAR,
- number of uncertain predictions.

### `r03_classification_report.csv`

Contains class-wise precision, recall, F1-score, and support.

### `r06_ablation_metrics.csv`

Contains modality and fusion ablation results.

### `r08_modality_oof_uar.csv`

Contains modality-level OOF UAR values.

### `r10_sequential_stage_ablation.csv`

Shows how the result changes after different post-processing stages.

---

## How to Interpret Results

Important metrics:

| Metric | Meaning |
|---|---|
| Accuracy | Overall correct predictions |
| Macro-F1 | Average F1 across classes |
| UAR / Balanced Accuracy | Average recall across classes |
| Per-class recall | How well each emotion is recognized |
| Fold std | Stability across actor groups |
| Uncertain samples | Low-confidence cases |

For RAVDESS subject-wise evaluation, **UAR and Macro-F1 are very important** because class-wise balance matters.

---

## Comparison With Prior Work

### Paper 1

Paper 1 reports around **86.70%** under subject-wise 5-fold.

This project does not beat that accuracy.

Main reason:

- Paper 1 uses a very strong fine-tuned audio model.
- Their audio-only performance is already high.
- This project focuses more on explainable multimodal architecture and ablation analysis.

Fair statement:

```text
Although the proposed method does not surpass the strongest audio-dominant RAVDESS baseline, it provides a more explainable multimodal framework with OpenFace-guided saliency, AU-dynamic modeling, calibrated fusion, uncertainty logging, and detailed ablation analysis.
```

### Paper 2

Paper 2 reports very high RAVDESS accuracy, but the split is not directly comparable if it uses trial-based or non-subject-independent evaluation.

Fair statement:

```text
The proposed method uses subject-wise actor-independent evaluation, which is stricter than trial-based splits where the same actor may appear in both training and testing.
```

### Paper 3

If the referenced third paper is retracted, do not use it as a serious benchmark.

Fair statement:

```text
Retracted work is excluded from quantitative benchmark comparison and is not used as methodological support.
```

---

## Challenges Faced

### 1. Dataset structure confusion

The folder contained about 4904 `.mp4` files, but the correct main setup uses only 1440 full-AV speech samples.

### 2. OpenFace setup

OpenFace CSVs were not included in the dataset. They had to be generated separately.

### 3. FFmpeg and audio loading

Earlier versions faced FFmpeg/audio-reading issues. The final code uses stable `librosa` loading.

### 4. Heavy end-to-end models overfit

Large models such as VideoMAE-style or heavy transformer fusion were too complex for the small 1440-sample dataset.

### 5. LoRA did not improve

LoRA worked technically but did not improve subject-wise performance.

### 6. Meta-fusion over-correction

Meta-fusion and specialist correction reduced performance compared with calibrated average fusion.

### 7. Weak classes remained difficult

Sad, Surprised, and Fearful remained the most difficult classes.

### 8. Accuracy did not beat Paper 1

The strongest prior paper used a stronger audio model, so the final result was lower. The project is better framed as architectural novelty and explainability, not state-of-the-art accuracy.

---

## Troubleshooting

### `ffmpeg was not found`

Install FFmpeg or use the final code that relies mainly on `librosa`.

### OpenFace CSV not found

Check that CSV names match video stems:

```text
01-01-03-01-01-01-01.csv
```

Expected folder:

```text
D:\CSE427_end_game\OpenFace_AU_CSV
```

### LoRA is too slow

Use the no-LoRA final model. It performs better in the current experiments.

### CUDA out of memory

Try:

```bat
--lora_batch_size 1
--lora_grad_accum 4
```

Or disable LoRA.

### Result lower than expected

Check:

- OpenFace coverage,
- correct 1440 full-AV speech filtering,
- no random split,
- subject-wise folds,
- branch ablation results,
- whether specialists are enabled.

---

## Recommended Paper Framing

Suggested title:

```text
ESRA-RELA++: Emotion-Saliency and Reliability-Aware Multimodal Fusion for Subject-Independent Audio-Visual Emotion Recognition
```

Main contribution paragraph:

```text
This work proposes ESRA-RELA++, a multimodal emotion recognition framework for subject-independent RAVDESS evaluation. The proposed method integrates hybrid acoustic features, OpenFace-guided saliency-based facial video representation, AU statistical descriptors, and AU-dynamic/FACS trajectory features. A calibrated fusion strategy is used to combine modality-level predictions, and additional modules such as LoRA audio adaptation, weak-pair specialists, and Sad-vs-rest correction are evaluated through ablation. The final selected configuration achieves the best subject-wise performance using calibrated average fusion over four active modalities.
```

Limitation paragraph:

```text
The proposed method does not exceed the highest reported accuracy on RAVDESS subject-wise evaluation. This is mainly due to the relatively weaker audio branch compared with highly optimized fine-tuned speech models used in prior work. However, the proposed framework contributes a more explainable and modular multimodal pipeline, including OpenFace-based visual saliency, AU-dynamic modeling, uncertainty logging, and detailed ablation analysis.
```

---

## Future Improvements

Possible next steps:

1. Use stronger audio models such as XLSR or WavLM.
2. Fine-tune an emotion-pretrained speech model.
3. Try LoRA with `facebook/wav2vec2-large-xlsr-53`.
4. Add temporal neural modeling for video features.
5. Add an AU sequence model using BiLSTM or a lightweight Transformer.
6. Improve Sad and Surprised recognition.
7. Add cross-corpus evaluation.
8. Tune fusion weights using only inner validation.
9. Report statistical significance testing.
10. Test on CREMA-D, SAVEE, or IEMOCAP.

---

## Citation / Dataset Attribution

If using RAVDESS, cite:

```text
Livingstone, S. R., & Russo, F. A. (2018).
The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS):
A dynamic, multimodal set of facial and vocal expressions in North American English.
PLOS ONE, 13(5), e0196391.
```

Also cite OpenFace if using AU extraction:

```text
Baltrušaitis, T., Zadeh, A., Lim, Y. C., & Morency, L.-P. (2018).
OpenFace 2.0: Facial Behavior Analysis Toolkit.
IEEE International Conference on Automatic Face and Gesture Recognition.
```

---

## Short Project Summary

ESRA-RELA++ is a subject-wise multimodal RAVDESS emotion recognition framework. It combines hybrid audio features, OpenFace-guided video saliency, AU statistical descriptors, and AU-dynamic/FACS trajectory features. The final best-performing setup uses calibrated simple-average fusion without LoRA or specialist correction, reaching **72.40% accuracy**, **71.94% macro-F1**, and **72.67% UAR** under subject-wise 5-fold evaluation.
