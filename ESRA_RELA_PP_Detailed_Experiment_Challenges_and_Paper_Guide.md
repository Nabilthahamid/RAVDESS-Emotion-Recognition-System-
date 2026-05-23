# ESRA-RELA++ V15.1 Complete Detailed Project Guide

**Project:** Subject-wise audio-visual emotion recognition on RAVDESS  
**Final selected configuration:** ESRA-RELA++ V15.1 calibrated simple-average fusion without LoRA  
**Best confirmed result:** Accuracy = **72.40% ± 3.81**, Macro-F1 = **71.94% ± 4.12**, UAR = **72.67% ± 3.82**  
**Dataset setting:** RAVDESS full audio-visual speech subset, 1440 samples, 8 emotion classes, 24 actors, subject-wise 5-fold evaluation  

---

## 1. Executive Summary

This project started as an attempt to build a stronger multimodal emotion-recognition model on the RAVDESS dataset using audio, video, and facial Action Unit information. The goal was not only to reproduce existing papers, but to design a more explainable and publication-defensible architecture that can show modality behavior, reliability, weak-class errors, and ablation evidence.

The final practical result is that the **best-performing version is not the most complex version**. The best confirmed result came from a simplified but stable variant:

```text
Audio branch + Video branch + AU branch + AU-dynamic branch
        ↓
Temperature-calibrated modality probabilities
        ↓
Simple average fusion
        ↓
Final emotion prediction
```

In this best result, LoRA, pair specialists, gender specialists, Sad-vs-rest specialist, and the heavy meta-correction pathway were not used in the final decision. They were still important experiments because they showed which components helped and which components introduced noise.

The most important lesson from the experiments is:

> More architectural components do not always improve accuracy on a small subject-wise dataset like RAVDESS. The best configuration was the one that preserved strong modality diversity but avoided over-correction.

---

## 2. Dataset Explanation

### 2.1 Why 4904 video files become 1440 samples

The full RAVDESS folder contains many `.mp4` files. Your script detected approximately:

```text
Video files before filtering: 4904
Samples after full-AV speech filtering: 1440
```

This is correct. The code intentionally uses only full audio-video speech samples.

RAVDESS files are encoded by filename. Example:

```text
01-01-03-01-01-01-01.mp4
```

The identifiers mean:

```text
01 = modality = full audio-video
01 = vocal channel = speech
03 = emotion = happy
01 = emotional intensity = normal
01 = statement
01 = repetition
01 = actor ID
```

The code keeps only:

```text
modality = 01 → full audio-video
channel  = 01 → speech
```

Therefore:

```text
Full-AV Speech      = 1440 files  → used
Video-only Speech   = 1440 files  → not used
Full-AV Song        = 1012 files  → not used
Video-only Song     = 1012 files  → not used
Total MP4           ≈ 4904 files
```

### 2.2 Why using only 1440 files is correct

Using all 4904 files would mix different tasks:

- speech emotion recognition,
- song emotion recognition,
- full audio-video recognition,
- video-only recognition.

That would make the experiment unfair and difficult to compare with prior work. The final experiment uses the clean **full audio-visual speech** subset because it matches the subject-wise RAVDESS setup used by the strongest comparison paper.

### 2.3 Class distribution

The 1440 full-AV speech samples contain 8 emotion classes:

| Emotion | Samples |
|---|---:|
| Neutral | 96 |
| Calm | 192 |
| Happy | 192 |
| Sad | 192 |
| Angry | 192 |
| Fearful | 192 |
| Disgust | 192 |
| Surprised | 192 |

Neutral has fewer samples because RAVDESS does not have a strong-intensity neutral class.

---

## 3. Evaluation Protocol

### 3.1 Subject-wise 5-fold evaluation

The project uses subject-wise 5-fold cross-validation. This means actors in the test fold are never seen during training.

This is much harder than a random split because the model must generalize to unseen speakers/faces.

The fold structure is:

| Fold | Test actors |
|---|---|
| Fold 0 | 2, 5, 14, 15, 16 |
| Fold 1 | 3, 6, 7, 13, 18 |
| Fold 2 | 10, 11, 12, 19, 20 |
| Fold 3 | 8, 17, 21, 23, 24 |
| Fold 4 | 1, 4, 9, 22 |

Four folds use 5 actors and one fold uses 4 actors. This follows the Paper-1 actor protocol.

### 3.2 Why subject-wise evaluation matters

Some papers report very high RAVDESS accuracy using trial-based or random splits. In those setups, the same actor may appear in both train and test. That makes the task easier because the model can learn actor-specific vocal or facial patterns.

Your result is more realistic because:

```text
Train actors ≠ Test actors
```

This should be clearly explained in the paper.

---

## 4. Final Architecture Overview

The final code is **ESRA-RELA++ V15.1 Pylance-clean**. It supports several branches and experimental options.

The full available architecture contains:

```text
1. Hybrid audio branch
2. Optional LoRA audio branch
3. Video branch
4. AU statistical branch
5. AU-dynamic branch
6. Reliability feature vector
7. Out-of-fold stacking
8. Temperature calibration
9. Dual fusion
10. Pair specialists
11. Gender specialists
12. Sad-vs-rest specialist
13. Uncertainty logging
14. Result and diagnostic saving
```

However, the **best-performing final setting** uses:

```text
Hybrid audio branch       ON
Video branch              ON
AU statistical branch     ON
AU-dynamic branch         ON
Temperature calibration   ON
Simple average fusion     ON
LoRA branch               OFF
Pair specialists          OFF
Gender specialists        OFF/effectively unused
Sad specialist            OFF
```

---

## 5. Detailed Branch Explanation

## 5.1 Hybrid Audio Branch

The normal audio branch extracts two categories of audio features.

### 5.1.1 SSL audio embedding

The code loads:

```text
facebook/wav2vec2-base-960h
```

It extracts hidden-state embeddings and pools them using:

```text
mean pooling
standard deviation pooling
max pooling
```

This creates a high-level speech representation.

### 5.1.2 Handcrafted acoustic and prosodic features

The code also extracts traditional audio features:

```text
MFCC
Delta MFCC
Chroma
Mel spectrogram statistics
Spectral centroid
Spectral bandwidth
Spectral rolloff
Spectral contrast
F0/pitch statistics
RMS energy
Zero crossing rate
```

These handcrafted features help capture emotion-related acoustic properties such as energy, pitch, spectral sharpness, and voice dynamics.

### 5.1.3 Why this branch matters

Emotion in speech often appears through:

- pitch change,
- intensity change,
- speaking rhythm,
- voice energy,
- spectral tone.

Therefore, using both SSL and handcrafted audio features is more robust than using only one type.

---

## 5.2 Optional LoRA Audio Branch

### 5.2.1 What LoRA was supposed to do

LoRA was added to adapt Wav2Vec2 to emotion recognition while updating only a small number of parameters. This was designed as a parameter-efficient alternative to full fine-tuning.

When LoRA is enabled, the code trains a separate audio classification model inside each fold.

### 5.2.2 LoRA safety

The LoRA branch is fold-safe:

```text
Outer test actors are never used for LoRA training.
Inner OOF models train only on inner train actors.
Final LoRA model trains only on outer train actors.
```

This avoids data leakage.

### 5.2.3 LoRA experiment result

LoRA did run correctly, but it did not improve the final result.

| Version | Accuracy | Macro-F1 | UAR |
|---|---:|---:|---:|
| Without LoRA | 71.48% ± 3.65 | 71.16% ± 3.84 | 71.77% ± 3.42 |
| With LoRA | 70.07% ± 3.81 | 68.56% ± 5.55 | 69.95% ± 4.08 |

LoRA decreased:

```text
Accuracy  : -1.42 percentage points
Macro-F1  : -2.60 percentage points
UAR       : -1.82 percentage points
```

The likely reason is that RAVDESS is small, and LoRA audio adaptation did not learn a strong enough emotion-specific representation. The LoRA branch had low modality usefulness compared with AU and AU-dynamic branches.

### 5.2.4 Final decision about LoRA

LoRA should be reported as an ablation, not as the final main model.

Suggested paper wording:

> LoRA-based audio adaptation was evaluated as an optional parameter-efficient module. Under the current configuration, it did not improve subject-wise performance and was therefore not selected for the final model.

---

## 5.3 Video Branch

The video branch is not a simple frame sampler. It uses OpenFace information to select more emotionally important frames.

Workflow:

```text
Video file
  ↓
OpenFace AU/confidence scoring
  ↓
Emotion-saliency frame selection
  ↓
Face crop using 68 facial landmarks
  ↓
MobileNetV3 feature extraction
  ↓
Temporal pooling
```

### 5.3.1 Frame selection

Instead of selecting frames uniformly only, the code uses OpenFace AU intensity, AU changes, confidence, and success rate to choose important frames.

This means the model focuses on frames where the face is likely showing stronger emotion.

### 5.3.2 Face crop

The code uses OpenFace landmark points `x_0...x_67` and `y_0...y_67` to crop the face region. This fixed an earlier issue where only a few landmark points were accidentally treated like a bounding box.

### 5.3.3 MobileNetV3 features

Each selected cropped face frame is passed through MobileNetV3-Small.

Then the code creates a temporal video feature using:

```text
mean embedding
standard deviation embedding
max embedding
frame-difference/delta embedding
first-frame embedding
last-frame embedding
```

This gives the video branch both appearance and simple temporal change information.

---

## 5.4 AU Statistical Branch

The AU statistical branch uses OpenFace Action Unit intensity features.

For each AU, the code calculates:

```text
mean
standard deviation
maximum
minimum
median
interquartile range
average absolute velocity
start-to-end change
```

This branch captures the overall facial muscle activation pattern during the clip.

---

## 5.5 AU-Dynamic Branch

The AU-dynamic branch was added because emotions are temporal. It captures how Action Units evolve over time.

It extracts:

```text
segment-wise AU means
segment-wise AU maxima
mean AU velocity
maximum AU velocity
normalized AU peak timing
FACS-group trajectory features
```

FACS groups include:

```text
brow
eye
nose
mouth
```

This branch is a strong novelty point because many simple RAVDESS methods use OpenFace AUs only as static/global statistics.

---

## 5.6 Reliability Features

The code creates a reliability vector for every sample.

It includes:

```text
audio RMS
audio ZCR
audio feature norm
video frames used
video saliency availability
video feature norm
AU found
AU confidence mean
AU success rate
AU-dynamic found
AU-dynamic confidence mean
AU-dynamic success rate
```

These values help analyze whether a modality is trustworthy for a sample.

In the final best simple-average variant, these reliability features are less central than in the meta-fusion version, but they remain useful for diagnostics and paper explanation.

---

## 6. Fusion Strategy Evolution

## 6.1 Initial idea: reliability-aware meta-fusion

The original idea was to train a meta-classifier on:

```text
modality probabilities
entropy values
margin values
reliability features
```

This created a reliability-aware stacking system.

The idea was academically strong, but the result showed that meta-fusion and specialist correction could overfit or over-correct on this small dataset.

## 6.2 Dual fusion

The code then added dual fusion:

```text
final probability = meta-fusion probability + calibrated average probability
```

This was useful because simple average fusion was already strong.

## 6.3 Final best fusion: calibrated simple average

The best current result came when the final decision was set close to simple average:

```text
dual_fusion_blend = 1.0
weighted fusion = false
specialists = off
sad specialist = off
LoRA = off
```

This means final prediction is approximately:

```text
Final probability = average(audio_prob, video_prob, au_prob, au_dynamic_prob)
```

This produced:

```text
Accuracy = 72.40% ± 3.81
Macro-F1 = 71.94% ± 4.12
UAR = 72.67% ± 3.82
```

---

## 7. Specialist Modules and Why They Were Disabled

## 7.1 Pair specialists

The code supports binary specialist classifiers for common confusion pairs:

```text
Fearful vs Surprised
Sad vs Disgust
Sad vs Fearful
Angry vs Disgust
Neutral vs Calm
```

These specialists were designed to fix common emotion confusions.

## 7.2 Gender specialists

The code also supports gender-stratified pair specialists because RAVDESS actor IDs encode male/female actors.

This was added cautiously as an exploratory component, not as a demographic claim.

## 7.3 Sad-vs-rest specialist

Sad was consistently one of the weakest classes, so the code added a binary:

```text
Sad vs Not-Sad
```

specialist.

## 7.4 Why specialists were not used in the final best result

Although these modules are explainable, they did not improve the best result. On small subject-wise folds, specialists can over-correct predictions.

Therefore, the best final model disables:

```text
pair specialists
Sad-vs-rest specialist
LoRA branch
```

This should be presented honestly as an ablation-driven decision.

---

## 8. Experiment History and Modifications

## 8.1 Early deep-fusion versions

At the beginning, the project attempted heavier deep fusion using audio/video/AU components with more complex training.

Problems:

```text
Low accuracy around the mid-50% range
Overfitting risk
Heavy GPU use
Small dataset mismatch
Complex losses did not help enough
```

Lesson:

> A very large end-to-end fusion model is not ideal for only 1440 subject-wise RAVDESS samples.

## 8.2 Move to RELA-HLF / ESRA-RELA style

The next improvement was to use stable feature extraction and lightweight classifiers.

Key changes:

```text
frozen SSL audio embeddings
handcrafted audio features
OpenFace AU statistics
MobileNetV3 video features
OOF stacking
subject-wise evaluation
```

This improved the result to around the high-60% range.

## 8.3 OpenFace integration

OpenFace was added to extract AU CSVs and landmarks.

Challenges:

```text
OpenFace did not exist in the RAVDESS dataset by default
OpenFace had to be downloaded separately
CSV files had to be generated from videos
Wrong file path initially caused video open errors
OpenFace output folder had to be checked
```

Once OpenFace was generated correctly, coverage reached:

```text
1440/1440 = 100%
```

## 8.4 Diagnostic outputs added

The code was expanded to save:

```text
preprocessing graphs
class distribution matrices
OpenFace coverage
confusion matrices
classification reports
ablation metrics
modality OOF-UAR
specialist configuration
uncertainty logs
sequential stage ablation
```

This made the work more publication-friendly.

## 8.5 V10/V11 improvements

Key additions:

```text
AU-dynamic trajectory branch
richer acoustic/prosody features
OpenFace-guided saliency video
Sad-vs-rest specialist
weighted dual fusion
LoRA optional branch
```

## 8.6 V15.1 Pylance-clean fixes

The final code also fixed practical issues:

```text
Pylance errors
librosa.pyin fmin/fmax type issue
PyTorch Dataset type issue
DataLoader label type issue
OpenFace cache invalidation
face crop from 68 landmarks
LoRA memory cleanup
fold-safe LoRA training
prediction CSV details
specialist config saving
sequential ablation saving
```

---

## 9. Final Experiment Results

## 9.1 Final selected result: no-LoRA simple average

| Fold | Test actors | Accuracy | Macro-F1 | UAR | Uncertain |
|---|---|---:|---:|---:|---:|
| 0 | 2,5,14,15,16 | 73.00% | 72.58% | 74.06% | 58 |
| 1 | 3,6,7,13,18 | 77.33% | 76.14% | 77.81% | 60 |
| 2 | 10,11,12,19,20 | 70.33% | 69.43% | 70.00% | 64 |
| 3 | 8,17,21,23,24 | 66.33% | 65.40% | 66.88% | 66 |
| 4 | 1,4,9,22 | 75.00% | 76.13% | 74.61% | 42 |

Final summary:

```text
Accuracy  = 72.40% ± 3.81
Macro-F1  = 71.94% ± 4.12
UAR       = 72.67% ± 3.82
```

## 9.2 Class-wise performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Neutral | 71.15% | 77.08% | 74.00% |
| Calm | 75.24% | 80.73% | 77.89% |
| Happy | 83.50% | 86.98% | 85.20% |
| Sad | 61.49% | 51.56% | 56.09% |
| Angry | 77.04% | 78.65% | 77.84% |
| Fearful | 62.24% | 63.54% | 62.89% |
| Disgust | 80.19% | 86.46% | 83.21% |
| Surprised | 62.94% | 55.73% | 59.12% |

Best classes:

```text
Happy
Disgust
Calm
Angry
Neutral
```

Weak classes:

```text
Sad
Surprised
Fearful
```

## 9.3 Interpretation of weak classes

Sad remains difficult because it is a low-arousal emotion and can overlap with Neutral, Calm, Fearful, or Disgust depending on actor expression.

Surprised and Fearful are also confused because both can involve widened eyes, mouth opening, and high facial activation.

This supports the need for better temporal modeling or stronger audio models in future work.

---

## 10. LoRA vs No-LoRA Experiment

## 10.1 Why LoRA was tested

LoRA was tested because Paper 1 has a very strong fine-tuned audio branch. The purpose was to see whether parameter-efficient audio adaptation could improve your audio branch without full fine-tuning.

## 10.2 Results

| Version | Accuracy | Macro-F1 | UAR |
|---|---:|---:|---:|
| No-LoRA final model | 71.48% ± 3.65 | 71.16% ± 3.84 | 71.77% ± 3.42 |
| LoRA final model | 70.07% ± 3.81 | 68.56% ± 5.55 | 69.95% ± 4.08 |
| No-LoRA simple-average selected | **72.40% ± 3.81** | **71.94% ± 4.12** | **72.67% ± 3.82** |

## 10.3 Conclusion from LoRA experiment

LoRA did not help in the current configuration. It is likely that:

```text
RAVDESS is too small for stable LoRA adaptation
LoRA learned actor/training-specific patterns
The LoRA branch produced weak probability estimates
Adding weak LoRA probabilities made fusion noisier
```

Therefore:

```text
LoRA should be reported as an ablation, not used in the selected final model.
```

---

## 11. Comparison with Three Papers

## 11.1 Paper 1

Paper 1 reports approximately:

```text
86.70% accuracy under subject-wise 5-fold
```

Your final selected result is:

```text
72.40% accuracy under subject-wise 5-fold
```

Therefore, your model does not beat Paper 1 in raw accuracy.

However, your work has architectural novelty:

| Paper 1 | Your ESRA-RELA++ |
|---|---|
| Strong fine-tuned audio model | Hybrid audio + handcrafted prosody |
| OpenFace AU visual model | OpenFace AU + AU-dynamic trajectory |
| Late fusion | Calibrated average / OOF fusion framework |
| Less detailed uncertainty analysis | Uncertainty logging |
| Limited branch diagnostics | Full ablation and modality analysis |
| No saliency-guided frame selection emphasis | OpenFace-guided video saliency |

Suggested explanation:

> Although the proposed method does not exceed the strongest audio-dominant prior result, it introduces a more explainable and modular multimodal architecture with saliency-guided video features, AU dynamic trajectory modeling, calibrated fusion, uncertainty analysis, and detailed ablation evidence.

## 11.2 Paper 2

Paper 2 reports a very high RAVDESS score, but its RAVDESS split is not directly comparable because it uses a trial-based setting rather than strict subject-wise unseen-actor testing.

Your defense:

> The proposed method uses subject-wise actor-independent evaluation, which is stricter and more realistic than trial-based evaluation where actor identity may be shared between train and test.

## 11.3 Paper 3

If the third paper is the retracted one, do not use it as a serious benchmark.

Your code and paper should state that retracted work is excluded from benchmark claims.

---

## 12. Challenging Parts of the Project

## 12.1 Dataset complexity

The dataset contains 4904 videos, but not all are useful for the same task. Filtering to 1440 full-AV speech samples was necessary for a clean experiment.

Challenge:

```text
Avoid mixing speech/song and audio-video/video-only subsets.
```

## 12.2 OpenFace setup

OpenFace CSVs were not included in the dataset. They had to be generated separately.

Challenges:

```text
Downloading OpenFace
Running FeatureExtraction.exe
Fixing wrong paths
Generating CSVs for all speech videos
Ensuring 100% CSV coverage
```

## 12.3 Small dataset problem

RAVDESS full-AV speech has only 1440 samples. Under subject-wise evaluation, each fold trains on about 19 or 20 actors and tests on unseen actors.

This makes deep model fine-tuning difficult.

## 12.4 Audio branch weakness

The strongest paper uses a highly optimized fine-tuned audio model. Your audio branch is more modular but not as strong.

This is the biggest reason your accuracy does not reach Paper 1.

## 12.5 LoRA instability

LoRA was logically useful, but under current settings it did not improve performance.

Challenge:

```text
Parameter-efficient fine-tuning still needs enough data and careful validation.
```

## 12.6 Fusion over-correction

Meta-fusion and specialists were explainable but reduced accuracy.

Lesson:

```text
On small datasets, a simpler calibrated average can generalize better than a complex correction pipeline.
```

## 12.7 Weak classes

Sad, Surprised, and Fearful remain difficult.

Reasons:

```text
Sad overlaps with Neutral/Calm/Fearful
Fearful and Surprised share high facial activation
Actor-wise expression variation is high
```

---

## 13. Commands Used

## 13.1 Best final selected command

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

## 13.2 Standard no-LoRA command

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

## 13.3 LoRA command

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

## 14. What to Report in the Paper

## 14.1 Main result table

| Method | Split | Accuracy | Macro-F1 | UAR |
|---|---|---:|---:|---:|
| ESRA-RELA++ selected simple-average fusion | Subject-wise 5-fold | 72.40% | 71.94% | 72.67% |
| ESRA-RELA++ no-LoRA full fusion | Subject-wise 5-fold | 71.48% | 71.16% | 71.77% |
| ESRA-RELA++ with LoRA | Subject-wise 5-fold | 70.07% | 68.56% | 69.95% |
| Paper 1 | Subject-wise 5-fold | 86.70% | not same | not same |
| Paper 2 | Trial split | 93.59% | not directly comparable | not directly comparable |

## 14.2 Best contribution statement

> This work proposes ESRA-RELA++, a modular and explainable multimodal framework for subject-wise audio-visual emotion recognition. The framework integrates hybrid audio features, OpenFace-guided saliency video features, AU statistical descriptors, AU-dynamic/FACS trajectory descriptors, calibrated fusion, uncertainty logging, and extensive ablation analysis. Although it does not exceed the strongest audio-dominant baseline, it provides a reproducible and interpretable evaluation framework under strict actor-independent testing.

## 14.3 Limitation statement

> The proposed method achieves lower accuracy than the strongest reported RAVDESS subject-wise result. This is mainly due to the weaker audio branch compared with fully optimized fine-tuned XLSR/Wav2Vec2 models used in prior work. In addition, RAVDESS is a small dataset, and complex correction modules such as LoRA and pair specialists can overfit or introduce noise under subject-wise folds. However, the proposed framework provides detailed modality analysis, uncertainty logging, and interpretable AU-dynamic modeling, which are valuable for understanding multimodal emotion recognition behavior.

## 14.4 Why lower accuracy is acceptable

You should explain:

```text
The work is not only about beating accuracy.
The work provides architecture novelty, explainability, ablation, and strict evaluation.
The evaluation is actor-independent.
The final model is selected based on evidence from ablation.
```

---

## 15. Future Improvement Plan

To improve beyond 72–73%, the most important path is audio.

Recommended future work:

```text
1. Use XLSR-Wav2Vec2 or WavLM emotion-pretrained audio models.
2. Fine-tune audio more carefully with actor-wise validation.
3. Try LoRA with XLSR-large using lower learning rate and longer training.
4. Add temporal video model such as BiLSTM/temporal attention over frame embeddings.
5. Add AU sequence model instead of only fixed-vector AU dynamics.
6. Improve Sad and Surprised using class-specific augmentation.
7. Test automatic fusion-rule selection using OOF validation only.
8. Report statistical significance between variants.
```

Potential XLSR-LoRA experiment:

```bat
--audio_model "facebook/wav2vec2-large-xlsr-53" ^
--use_lora_audio ^
--lora_audio_model "facebook/wav2vec2-large-xlsr-53" ^
--lora_batch_size 1 ^
--lora_grad_accum 4 ^
--lora_lr 0.00005
```

---

## 16. Final Decision

The current stage is enough to start paper writing if the paper is framed correctly.

Use this as the final selected model:

```text
ESRA-RELA++ calibrated simple-average fusion without LoRA
```

Final result:

```text
Accuracy  = 72.40% ± 3.81
Macro-F1  = 71.94% ± 4.12
UAR       = 72.67% ± 3.82
```

Do not claim state-of-the-art accuracy against Paper 1.

Claim:

```text
The proposed method is a more explainable, modular, and ablation-rich multimodal framework under strict subject-wise evaluation.
```

This is a defensible and honest paper direction.

---

## 17. One-Paragraph Paper-Ready Summary

This study proposes ESRA-RELA++, an explainable multimodal framework for subject-wise emotion recognition on the RAVDESS full audio-visual speech subset. The framework combines hybrid speech features, OpenFace-guided saliency-based facial video representations, global AU statistics, and AU-dynamic/FACS trajectory descriptors. Several fusion and correction strategies were evaluated, including reliability-aware meta-fusion, LoRA-based audio adaptation, weak-pair specialists, and Sad-vs-rest correction. Experimental results showed that the best-performing configuration was a calibrated simple-average fusion over the four non-LoRA modalities, achieving 72.40% accuracy, 71.94% macro-F1, and 72.67% UAR under subject-wise 5-fold evaluation. Although the method does not surpass the strongest audio-dominant prior work, it provides a more interpretable and reproducible analysis framework with modality-specific ablations, uncertainty logging, and detailed weak-class investigation.

