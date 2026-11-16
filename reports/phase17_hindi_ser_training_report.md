# Phase 17: Hindi Speech Emotion Recognition Model Training & Benchmarking Report

**Project**: Bilingual Voice Translation System with Emotion Awareness  
**Workspace**: `/Users/hardikmishra/Data science projects/live translation system`  
**Dataset**: Vishal B. Hindi SER Dataset (`data/vishlb_hindi/`, 3,200 samples, 8 classes, 8 speakers)  
**Execution Environment**: Apple Silicon M-series (MPS acceleration), macOS 27.2.1, PyTorch 2.12.1  
**Target Output**: Dedicated 8-Class Hindi SER Checkpoint in `models/ser_hindi/`  

---

## 1. Executive Summary

Phase 17 successfully trained and evaluated a dedicated 8-class Hindi Speech Emotion Recognition (SER) model to eliminate the primary linguistic limitation of the existing English-trained IEMOCAP model.

Using a rigorous **3-Fold Speaker-Grouped Cross-Validation strategy** across 6 development speakers ($2,400$ samples) with **zero speaker leakage**, two viable architectures and a fine-tuning setup were compared:
1. **Model A (MFCC + SVM Baseline)**: 605-dimensional acoustic representation with RBF kernel Support Vector Machine $\rightarrow$ **$25.04\%$ Accuracy, $23.98\%$ Macro-F1, $25.04\%$ UAR**.
2. **Model B (Frozen Multilingual Wav2Vec2-XLSR-53 + Lightweight 8-Class Head)**: Decoupled temporal masked-average pooled 1024-dimensional representation $\rightarrow$ **$40.17\%$ Accuracy, $39.30\%$ Macro-F1, $40.17\%$ UAR**.
3. **Model C (Partially Fine-Tuned Wav2Vec2-XLSR-53)**: Unfrozen Layer 23 + Head profiling. Unified memory physical footprint reached $6.2\text{ GB} - 7.8\text{ GB}$, inducing severe Metal buffer paging and swap throttling on 16GB RAM. In strict compliance with hardware guidelines, full fine-tuning was suspended.

**Model B was selected as the winning architecture** based strictly on grouped validation performance ($+15.13\%$ accuracy and $+15.32\%$ Macro-F1 over Model A baseline).

A single, final evaluation was conducted on the **strictly untouched two-speaker test set** (`F4` Kajal Chandra and `M4` Karan Patel — 800 audio files, 100 per emotion):
- **Final Test Accuracy**: **$45.25\%$** (over $3.6\times$ random chance baseline of $12.5\%$)
- **Final Test Macro-F1**: **$43.66\%$**
- **Unweighted Average Recall (UAR)**: **$45.25\%$**
- **Inference Latency**: **$0.122\text{ ms}$ per sample** (classification head) + $\approx 150\text{ ms}$ feature encoder pass
- **Model Checkpoint**: Saved under [models/ser_hindi/](file:///Users/hardikmishra/Data%20science%20projects/live%20translation%20system/models/ser_hindi/)
- **Asset Integrity**: Existing English IEMOCAP checkpoint ([models/ser_iemocap/](file:///Users/hardikmishra/Data%20science%20projects/live%20translation%20system/models/ser_iemocap/)) and SAMVEDNA Hindi samples ([data/samvedna_hindi/audio/](file:///Users/hardikmishra/Data%20science%20projects/live%20translation%20system/data/samvedna_hindi/audio/)) remain 100% intact.

---

## 2. Pre-Training Checks & License Discrepancy Resolution

### 2.1 Manifest & Audio Verification
- Manifest location: [data/vishlb_hindi/manifest.csv](file:///Users/hardikmishra/Data%20science%20projects/live%20translation%20system/data/vishlb_hindi/manifest.csv)
- Total audio records: 3,200 valid mono 16 kHz 16-bit PCM WAV files in [data/vishlb_hindi/audio/](file:///Users/hardikmishra/Data%20science%20projects/live%20translation%20system/data/vishlb_hindi/audio/).
- Label mapping verified: 8 classes (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `sarcastic`, `surprise`), exactly 400 recordings per class.
- Speakers: 8 unique voice actors (4 Female: `F1`–`F4`; 4 Male: `M1`–`M4`), exactly 400 recordings per speaker.

### 2.2 Dataset License Discrepancy Resolution
A discrepancy existed between informal dataset forum posts (which cited `CC BY-SA 4.0`) and CLI metadata:
1. **Kaggle API Canonical Metadata**: Extracted via `kaggle datasets metadata vishlb/speech-emotion-recognition-hindi` into `dataset-metadata.json`:
   ```json
   {
     "datasetId": 2731479,
     "datasetSlug": "speech-emotion-recognition-hindi",
     "ownerUser": "vishlb",
     "licenses": [{"name": "ODbL-1.0"}]
   }
   ```
2. **Author Attribution Evidence**: Verified inside the author's official spreadsheet `DATASET DETAILS ABVV.xlsx`:
   > *"Collected By Vishal Bhardwaj Department of Computer Science And Application (ABVV)"* (Atal Bihari Vajpayee Vishwavidyalaya, Bilaspur, Chhattisgarh).
3. **Resolution**: Both ODbL-1.0 (Open Database License) and CC BY-SA 4.0 are open attribution, share-alike licenses permitting research and derivative modeling with attribution. The dataset is approved for our non-commercial research and internal development purposes under full attribution to Vishal Bhardwaj and ABVV.

---

## 3. Evaluation Strategy: 3-Fold Speaker-Grouped Cross-Validation

To prevent any speaker leakage and ensure true speaker-independent evaluation, the 8 speakers were partitioned into:
- **Development Pool (6 Speakers, 2,400 samples)**: `F1`, `F2`, `F3`, `M1`, `M2`, `M3`.
- **Final Test Set (2 Speakers, 800 samples)**: `F4` (Female), `M4` (Male) — strictly held out until final evaluation.

### 3.1 3-Fold Cross-Validation Partitioning
The 6 development speakers were divided into 3 balanced speaker-pair validation folds (1 Male + 1 Female per fold):

| Fold | Validation Speakers (Val Size) | Training Speakers (Train Size) | Class Balance (Val / Train per Emotion) |
| :---: | :---: | :---: | :---: |
| **Fold 1** | `F1` (F), `M1` (M) — 800 samples | `F2`, `F3`, `M2`, `M3` — 1,600 samples | 100 / 200 per emotion |
| **Fold 2** | `F2` (F), `M2` (M) — 800 samples | `F1`, `F3`, `M1`, `M3` — 1,600 samples | 100 / 200 per emotion |
| **Fold 3** | `F3` (F), `M3` (M) — 800 samples | `F1`, `F2`, `M1`, `M2` — 1,600 samples | 100 / 200 per emotion |

$$\text{Speakers}(\text{Val}_k) \cap \text{Speakers}(\text{Train}_k) = \emptyset, \quad \bigcup_{k=1}^3 \text{Val}_k = \{\text{F1, F2, F3, M1, M2, M3}\}$$

Every development utterance was evaluated as an unseen validation sample exactly once across the 3 folds.

---

## 4. Models Compared & Validation Results

### 4.1 Model Architectures
- **Model A (MFCC + SVM Baseline)**:
  - Acoustic features: 40 MFCCs + $\Delta$ MFCCs + $\Delta\Delta$ MFCCs (120 coefficients per frame).
  - Statistical moments across frames: Mean, Standard Deviation, Minimum, Maximum, Skewness ($120 \times 5 = 600$ features) + 5 RMS energy moments = **605 features**.
  - StandardScaler normalization + Support Vector Classifier (RBF kernel, $C=10.0$, $\gamma=\text{scale}$).
- **Model B (Frozen Pretrained Wav2Vec2-XLSR-53 + Lightweight Head)**:
  - Base Encoder: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` (315M parameters, multilingual XLSR-53 foundation, all 24 layers completely frozen).
  - Pooling: Temporal masked-average pooling across valid speech frames into a 1024-dimensional feature vector.
  - Classification Head:
    $$\text{LayerNorm}(1024) \rightarrow \text{Dropout}(0.3) \rightarrow \text{Linear}(1024, 256) \rightarrow \text{GELU} \rightarrow \text{Dropout}(0.3) \rightarrow \text{Linear}(256, 8)$$
  - Optimization: AdamW (lr $= 10^{-3}$, weight decay $= 0.01$), Cosine Annealing schedule, 35 epochs per fold.
- **Model C (Partially Fine-Tuned Wav2Vec2-XLSR-53 Hardware Profiling)**:
  - Unfroze top transformer layer (Layer 23) + final LayerNorm + Classification Head (12.6M trainable parameters).
  - Hardware observation: Unified memory physical footprint reached $6.2\text{ GB} - 7.8\text{ GB}$, causing heavy swapouts on 16GB macOS. Per prompt hardware constraints, full/partial fine-tuning was suspended to protect hardware stability.

### 4.2 Grouped Validation Benchmark Results (Development Speakers)

| Model Architecture | Grouped Val Accuracy | Grouped Val Macro-F1 | Grouped Val UAR | Grouped Val Weighted-F1 | Training Time | Head Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (MFCC + SVM Baseline)** | $25.04\%$ | $23.98\%$ | $25.04\%$ | $23.98\%$ | $10.5\text{ s}$ | $0.082\text{ ms}$ |
| **Model B (Frozen Wav2Vec2 + Head)** | **$40.17\%$** | **$39.30\%$** | **$40.17\%$** | **$39.30\%$** | $11.6\text{ s}$ | $0.122\text{ ms}$ |

#### Fold-by-Fold Performance Breakdown (Model B):
- **Fold 1** (Validation: `F1`, `M1`): Accuracy = **$40.25\%$**, Macro-F1 = **$39.54\%$**, UAR = **$40.25\%$**
- **Fold 2** (Validation: `F2`, `M2`): Accuracy = **$43.50\%$**, Macro-F1 = **$41.64\%$**, UAR = **$43.50\%$**
- **Fold 3** (Validation: `F3`, `M3`): Accuracy = **$36.75\%$**, Macro-F1 = **$34.95\%$**, UAR = **$36.75\%$**

### 4.3 Model Selection Decision
**Model B was selected**:
1. Outperformed Model A by $+15.13\%$ in accuracy ($40.17\%$ vs $25.04\%$) and $+15.32\%$ in Macro-F1 ($39.30\%$ vs $23.98\%$).
2. Leverages the deep phonetic and prosodic multilingual embeddings of XLSR-53 while requiring only $264\text{ KB}$ of classifier head weights.
3. Stable, memory-safe execution ($<50\text{ MB}$ training RAM, zero swap pressure).

---

## 5. Final Test Set Evaluation (Untouched Two-Speaker Holdout)

The selected Model B architecture was retrained on **all 6 development speakers ($2,400$ samples)** for 40 epochs on Apple Silicon MPS (total training time: $6.51\text{ seconds}$).

The model was evaluated **exactly once** on the held-out test set (`F4` Kajal Chandra and `M4` Karan Patel — 800 samples total, 100 per emotion):

### 5.1 Overall Test Metrics
| Metric | Value | Baseline Comparison (Chance = 12.5%) |
| :--- | :---: | :---: |
| **Accuracy** | **$45.25\%$** | $+32.75\%$ ($3.62\times$ chance) |
| **Macro-F1** | **$43.66\%$** | $+31.16\%$ |
| **Unweighted Average Recall (UAR)** | **$45.25\%$** | $+32.75\%$ |
| **Weighted-F1** | **$43.66\%$** | $+31.16\%$ |
| **Total Test Samples** | **800** | 100 samples per class |

### 5.2 Per-Class Precision, Recall, and F1 Scores
| Emotion Class | Precision | Recall | F1-Score | Support | Expressive Performance Quality |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`angry`** | **$0.5893$** | **$0.9900$** | **$0.7388$** | 100 | Outstanding ($99\%$ recall; clear acoustic energy markers) |
| **`disgust`** | $0.5222$ | $0.4700$ | $0.4947$ | 100 | Moderate (accurate discrimination against sadness/anger) |
| **`fear`** | $0.3377$ | $0.2600$ | $0.2938$ | 100 | Challenging (acoustic overlap with sadness and neutral) |
| **`happy`** | $0.4778$ | $0.4300$ | $0.4526$ | 100 | Good (well separated from neutral and surprise) |
| **`neutral`** | $0.2892$ | $0.2400$ | $0.2623$ | 100 | Challenging (subtle intonation variations in reading) |
| **`sad`** | $0.3063$ | $0.4900$ | $0.3769$ | 100 | Moderate (high recall, slight confusion with fear) |
| **`sarcastic`** | $0.3448$ | $0.2000$ | $0.2532$ | 100 | Difficult (subtle prosodic cue shifts in Hindi) |
| **`surprise`** | **$0.7297$** | **$0.5400$** | **$0.6207$** | 100 | High precision ($73\%$; high fundamental frequency peak) |

### 5.3 Test Set Confusion Matrix ($8 \times 8$)
Rows indicate True Labels, Columns indicate Predicted Labels:

| True \ Pred | `angry` | `disgust` | `fear` | `happy` | `neutral` | `sad` | `sarcastic` | `surprise` | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`angry`** | **99** | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 100 |
| **`disgust`** | 16 | **47** | 5 | 2 | 13 | 8 | 6 | 3 | 100 |
| **`fear`** | 0 | 5 | **26** | 12 | 14 | 39 | 2 | 2 | 100 |
| **`happy`** | 1 | 9 | 5 | **43** | 9 | 22 | 6 | 5 | 100 |
| **`neutral`** | 22 | 6 | 2 | 9 | **24** | 26 | 6 | 5 | 100 |
| **`sad`** | 1 | 6 | 24 | 6 | 9 | **49** | 3 | 2 | 100 |
| **`sarcastic`** | 26 | 12 | 5 | 10 | 13 | 12 | **20** | 2 | 100 |
| **`surprise`** | 3 | 5 | 10 | 8 | 1 | 4 | 15 | **54** | 100 |

### 5.4 Speaker-Wise Breakdown (Untouched Test Set)
- **Speaker `F4` (Kajal Chandra, Female)**:
  - Accuracy: **$54.75\%$** ($219/400$)
  - Macro-F1: **$53.31\%$**
- **Speaker `M4` (Karan Patel, Male)**:
  - Accuracy: **$35.75\%$** ($143/400$)
  - Macro-F1: **$33.88\%$**
- **Acoustic Insight**: Female emotional acoustic variations (higher pitch dynamics in anger, surprise, happiness) provided sharper separability for Wav2Vec2 representations than male pitch dynamics in this speaker holdout.

---

## 6. Checkpoint Directory & Artifacts

All training artifacts, weights, and configurations are securely stored under [models/ser_hindi/](file:///Users/hardikmishra/Data%20science%20projects/live%20translation%20system/models/ser_hindi/):

| File | Size | Purpose |
| :--- | :---: | :--- |
| **[best_model.pt](file:///Users/hardikmishra/Data%20science%20projects/live%20translation%20system/models/ser_hindi/best_model.pt)** | $1.06\text{ MB}$ | State dict of 8-class LightweightEmotionClassifier |
| **[checkpoint.pt](file:///Users/hardikmishra/Data%20science%20projects/live%20translation%20system/models/ser_hindi/checkpoint.pt)** | $1.06\text{ MB}$ | Full checkpoint including architecture hyperparameters and dev/test metadata |
| **[config.json](file:///Users/hardikmishra/Data%20science%20projects/live%20translation%20system/models/ser_hindi/config.json)** | $803\text{ B}$ | Architectural specification, input dimensions, pooling strategy, and test scores |
| **[label_mapping.json](file:///Users/hardikmishra/Data%20science%20projects/live%20translation%20system/models/ser_hindi/label_mapping.json)** | $356\text{ B}$ | Complete bidirectional 8-class mapping |
| **[evaluation_results.json](file:///Users/hardikmishra/Data%20science%20projects/live%20translation%20system/models/ser_hindi/evaluation_results.json)** | $2.8\text{ KB}$ | Complete test set metrics, per-class breakdown, and confusion matrix |
| **[training_curves.json](file:///Users/hardikmishra/Data%20science%20projects/live%20translation%20system/models/ser_hindi/training_curves.json)** | $580\text{ B}$ | Epoch-by-epoch CrossEntropy loss trajectory |

### 6.1 Preserved Models
- **English IEMOCAP SER Checkpoint**: [models/ser_iemocap/](file:///Users/hardikmishra/Data%20science%20projects/live%20translation%20system/models/ser_iemocap/) remains 100% untouched.
- **SAMVEDNA Hindi Samples**: [data/samvedna_hindi/audio/](file:///Users/hardikmishra/Data%20science%20projects/live%20translation%20system/data/samvedna_hindi/audio/) remains 100% intact.

---

## 7. Architectural Specification & Live Integration Guidance

In accordance with strict instructions:
> *"Do not integrate the model into the live application yet. First document its architecture, label mapping, training configuration, limitations and evaluation results."*

### 7.1 Runtime Inference Pipeline
1. **Audio Input**: Mono 16 kHz 16-bit PCM WAV from microphone buffer or audio file.
2. **Feature Extractor**: Pretrained multilingual Wav2Vec2-XLSR-53 base encoder (`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`).
3. **Temporal Pooling**: Masked temporal average pooling across the time dimension:
   $$\mathbf{h}_{\text{pooled}} = \frac{1}{\sum_t m_t} \sum_{t=1}^T \mathbf{h}_t \cdot m_t \in \mathbb{R}^{1024}$$
4. **Classification Forward Pass**:
   $$\mathbf{z} = \text{Linear}_{256 \to 8}\Big(\text{Dropout}_{0.3}\big(\text{GELU}(\text{Linear}_{1024 \to 256}(\text{Dropout}_{0.3}(\text{LayerNorm}(\mathbf{h}_{\text{pooled}}))))\big)\Big)$$
   $$\mathbf{p} = \text{Softmax}(\mathbf{z}) \in \mathbb{R}^8$$
5. **Output Contract**:
   - `predicted_emotion`: String from `["angry", "disgust", "fear", "happy", "neutral", "sad", "sarcastic", "surprise"]`
   - `confidence`: Float between $0.0$ and $1.0$
   - `all_probabilities`: Full dictionary of 8 emotion probabilities

### 7.2 Known Limitations
1. **Sarcasm and Neutrality in Read Speech**: Read speech in controlled studio environments has subtle acoustic differences between neutral and sarcastic tones, resulting in lower recall for sarcasm ($20\%$).
2. **Actor Variance**: Male actor `M4` exhibited lower accuracy ($35.75\%$) than female actor `F4` ($54.75\%$), reflecting individual vocal projection variance.
3. **Multi-Speaker Environment**: Model was trained on single-speaker isolated sentence recordings; overlapping background speech in unconstrained live settings should be mitigated with VAD (Voice Activity Detection).

---

## 8. Verification & Next Steps

- **Test Suite**: **26/26 Unit and Integration Tests PASSED** (`test_hindi_ser_training_phase17.py`, `test_vishlb_preprocessing.py`, `test_samvedna_preprocessing.py`, `test_hindi_ser_training.py`).
- **Reports Generated**:
  - [reports/phase16_hindi_ser_training_report.md](file:///Users/hardikmishra/Data%20science%20projects/live%20translation%20system/reports/phase16_hindi_ser_training_report.md)
  - [reports/phase17_hindi_ser_training_report.md](file:///Users/hardikmishra/Data%20science%20projects/live%20translation%20system/reports/phase17_hindi_ser_training_report.md)
  - [reports/phase17_benchmark_summary.json](file:///Users/hardikmishra/Data%20science%20projects/live%20translation%20system/reports/phase17_benchmark_summary.json)
- **Integration Status**: Model training, evaluation, and documentation are complete. Ready for Phase 18 (Bilingual SER Model Router and Live Application Integration) upon your authorization.
