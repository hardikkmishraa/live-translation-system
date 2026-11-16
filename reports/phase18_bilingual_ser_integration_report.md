# Phase 18: Bilingual SER Router and Live Application Integration Report

**Date**: 2026-11-15  
**Project**: Bilingual Voice Translation System with Emotion Awareness (`live translation system`)  
**Status**: Completed & Verified  

---

## 1. Executive Summary

Phase 18 successfully unifies our two separately trained Speech Emotion Recognition (SER) models into a cohesive, production-grade bilingual inference architecture:

1. **English SER**: IEMOCAP 4-class model (`angry`, `happy`, `sad`, `neutral`) checkpoint located at `models/ser_iemocap/` (1.26 GB).
2. **Hindi SER**: Vishal B. Hindi dataset 8-class model (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `sarcastic`, `surprise`) checkpoint located at `models/ser_hindi/` (1.06 MB head weights on top of frozen `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` backbone).

Crucially, **neither model nor label space was merged or diluted**. English inputs retain their native 4-class IEMOCAP distribution, while Hindi inputs retain their native 8-class distribution. The models load once at application startup into unified memory, are safely reused across concurrent and sequential inference requests, and dynamically gate downstream expressive speech synthesis through latent-depth prosody modulation.

---

## 2. Architecture & Component Design

```
                                  +-----------------------+
                                  |   Input Audio (PCM)   |
                                  +-----------+-----------+
                                              |
                                     (Preprocessing / VAD)
                                              |
                                              v
                              +-------------------------------+
                              |    ASR / Language Detection   |
                              |     (openai/whisper-tiny)     |
                              +---------------+---------------+
                                              |
                                   Detected Language (hi / en)
                                              |
                                              v
                             +---------------------------------+
                             |      Bilingual SER Router       |
                             |   (BilingualSERRouter Class)    |
                             +----------------+----------------+
                                              |
                     +------------------------+------------------------+
                     | (hi / hin / hindi)                              | (en / eng / english / fallback)
                     v                                                 v
    +----------------------------------+              +----------------------------------+
    |    HindiSpeechEmotionRecognizer  |              |  IEMOCAPSpeechEmotionRecognizer  |
    |      (8 Native Hindi Classes)    |              |     (4 Native English Classes)   |
    | - angry, disgust, fear, happy,   |              | - angry, happy, sad, neutral     |
    |   neutral, sad, sarcastic,       |              |                                  |
    |   surprise                       |              |                                  |
    +----------------+-----------------+              +----------------+-----------------+
                     |                                                 |
                     +------------------------+------------------------+
                                              |
                                  SERResult + Telemetry
                                  (Model ID, Latency, Conf)
                                              |
                                              v
                         +-----------------------------------------+
                         |      Latent-Depth Emotion Gating        |
                         |   (LatentDepthEmotionPreserver Class)   |
                         +--------------------+--------------------+
                                              |
                           Confidence-Gated Prosody Multipliers
                           - < 0.35: Neutral Baseline (1.0x)
                           - 0.35 - 0.50: Soft Linear Blending
                           - >= 0.50: Full Expressive Synthesis
                                              |
                                              v
                         +-----------------------------------------+
                         |       Neural TTS Synthesis Engine       |
                         |      (Sarvam AI bulbul:v4-flash)        |
                         |   Emotion-Conditioned Persona Mappings  |
                         +--------------------+--------------------+
                                              |
                                              v
                         +-----------------------------------------+
                         |   Vanilla HTML/CSS/JS Frontend UI       |
                         |   (Dual Audio, Dynamic Posteriors)      |
                         +-----------------------------------------+
```

### 2.1 Router Design (`backend/app/models/ser_router.py`)
- **Language Normalization**: Evaluates input `source_language` strings (e.g., `hi-IN`, `hin`, `Hindi`, `en-US`, `English`) to canonical two-letter codes (`hi` or `en`).
- **Dynamic Dispatch**:
  - `hi` $\to$ Dispatches to `HindiSpeechEmotionRecognizer` (8 classes).
  - `en` $\to$ Dispatches to `IEMOCAPSpeechEmotionRecognizer` (4 classes).
  - Unsupported / Uncertain Language (e.g. `fr`, `unknown`, `None`) $\to$ Triggers documented fallback to English IEMOCAP baseline with `is_routing_fallback=True` and explanatory metadata.
- **Warmup Forward Pass**: Executes a 1-second dummy waveform on both models during `_ensure_loaded()` at startup, completely eliminating cold Metal JIT compilation latency spikes on the first user request.

### 2.2 Hindi SER Adapter (`backend/app/models/ser_hindi.py`)
- Encapsulates the frozen `Wav2Vec2Model` backbone and the trained `LightweightEmotionClassifier` head (`LayerNorm(1024) -> Dropout(0.3) -> Linear(1024, 256) -> GELU -> Dropout(0.3) -> Linear(256, 8)`).
- Locates the local HuggingFace cache snapshot (`.cache/huggingface/hub/models--ehcalabres--wav2vec2-lg-xlsr-en-speech-emotion-recognition/`) to ensure deterministic offline execution without network dependencies.
- Applies zero-mean unit-variance audio normalization, attention-masked temporal pooling, and outputs full 8-class posterior probabilities and 3D VAD (Valence, Arousal, Dominance) coordinates.

---

## 3. Inference Optimization & Memory Lifecycle

### 3.1 Model Lifecycle Persistence (`backend/app/services/model_lifecycle.py`)
- Configured default SER provider: `SER_MODEL_PROVIDER="router"`.
- Resolved instance-reset edge case by establishing canonical provider mapping (`router`), ensuring the router instance is initialized **once** as a thread-safe singleton.
- **Latency Benchmarks**:
  - Cold startup & warmup on Apple Silicon MPS: ~3.06 s total (both 1.26 GB IEMOCAP + Wav2Vec2 backbone loaded).
  - Subsequent router retrieval (`model_lifecycle.get_ser()`): **0.01 ms** (cached singleton).
  - Single-utterance SER forward pass on MPS:
    - Hindi SER: **183.6 ms**
    - English SER: **189.7 ms**
    - Router dispatch overhead: **< 0.1 ms**
- **Memory Footprint**: Total pipeline memory usage remains under 2.8 GB unified memory on Apple Silicon, well below the 16 GB hardware threshold.

---

## 4. Confidence Gating & Downstream TTS Modulation

### 4.1 Latent-Depth Emotion Gating (`backend/app/models/emotion_preservation.py`)
To prevent erratic or jarring vocal synthesis when SER confidence is weak, a 3-tier confidence gating protocol is enforced:
1. **Low Confidence ($< 0.35$)**: Strict neutral fallback. Pitch, rate, and volume multipliers are locked to `1.00x`. Applied emotion reported as `neutral`.
2. **Medium Confidence ($0.35 \le c < 0.50$)**: Soft blending. Multipliers are linearly interpolated between baseline ($1.00x$) and target emotional modulation using factor $\alpha = \frac{c - 0.35}{0.50 - 0.35}$.
3. **High Confidence ($\ge 0.50$)**: Full expressive prosody plan applied.

### 4.2 Support for All 8 Hindi Emotion Classes
Prosodic profiles and Sarvam TTS speaker personas were expanded to support all 8 emotion classes:

| Emotion | Target Pitch | Target Rate | Target Vol | Prosody Characterization | Sarvam Persona (Hindi F/M) | Sarvam Persona (English F/M) |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| **Neutral** | 1.00 | 1.00 | 1.00 | Conversational baseline | `kavya` / `aditya` | `amelia` / `dev` |
| **Happy** | 1.15 | 1.12 | 1.10 | Upbeat cadence, elevated pitch | `simran` / `amit` | `sophia` / `ashwin` |
| **Sad** | 0.88 | 0.85 | 0.85 | Subdued pitch, slower tempo | `roopa` / `ratan` | `chandrika` / `rohan` |
| **Angry** | 1.18 | 1.18 | 1.20 | Sharp emphasis, high intensity | `ritu` / `shubh` | `zarina` / `rustom` |
| **Disgust** | 0.95 | 0.92 | 0.95 | Guttural pitch, deliberate pause | `ritu` / `shubh` | `zarina` / `rustom` |
| **Fear** | 1.15 | 1.12 | 0.90 | Tense tract, hurried cadence, low vol | `roopa` / `ratan` | `chandrika` / `rohan` |
| **Surprise**| 1.25 | 1.10 | 1.15 | Upward pitch inflection | `simran` / `amit` | `sophia` / `ashwin` |
| **Sarcastic**| 1.05 | 0.94 | 1.02 | Drawn-out cadence, dynamic glide | `ritu` / `shubh` | `zarina` / `rustom` |

*Truthful Fidelity Reporting*: The pipeline explicitly discloses that while neural prosodic modulations and expressive speaker personas are applied, speaker identity cloning remains deactivated to protect user privacy.

---

## 5. Frontend UI Integration (`frontend/`)

In adherence to system requirements, the frontend remains in vanilla HTML5, CSS3, and modern modular ES6 JavaScript without introducing React or heavy frontend frameworks:

1. **Emotion Icons & Badges (`frontend/js/results.js` & `frontend/css/`)**:
   - Added icon glyphs: Fear (`😨`), Surprise (`😲`), Sarcastic (`😏`), Disgust (`🤢`).
   - Added CSS badge pill styles (`.emotion-badge-pill.fear`, `.surprise`, `.sarcastic`) in `frontend/css/components.css`.
   - Added design token `--emo-sarcastic: #f59e0b;` in `frontend/css/design-system.css`.
2. **Dynamic Posterior Distribution**:
   - Added dedicated progress bar for `Sarcastic` in `frontend/index.html`.
   - Updated distribution rendering loop to seamlessly hide irrelevant classes depending on whether the active inference result is 4-class IEMOCAP, 8-class RAVDESS, or 8-class Vishal B. Hindi.
3. **Telemetry & Status Cards**:
   - Displays detected language, routed model ID (`bilingual-ser-router-v1 -> hindi-wav2vec2-xlsr-8class`), routing status, and latency breakdown.

---

## 6. Comprehensive Verification & Live Benchmark Results

### 6.1 Automated Test Suites (All Passed)
- **`tests/test_bilingual_ser_router.py`**: **13/13 PASSED** (Language normalization, model dispatch, fallback telemetry, singleton lifecycle, confidence gating, Sarvam persona mapping, full pipeline routing integration).
- **`tests/test_hindi_ser_training_phase17.py`**: **6/6 PASSED** (Cross-validation integrity, head forward pass, speaker disjointness).
- **`tests/test_ser_iemocap_integration.py`**: **13/13 PASSED** (English IEMOCAP regression, fallback behavior, real audio inference).
- **`tests/test_tts_emotion.py`**: **10/10 PASSED** (Prosody planning, gating thresholds, cascade fallback).
- **`tests/test_api_v1.py`**: **9/9 PASSED** (Health, capabilities, audio upload, text translate, WebSocket handshake).
- **`tests/test_frontend.py`**: **4/4 PASSED** (Asset serving, accessibility, HTML structure).

### 6.2 Live Pipeline Audio Validation
Evaluated using actual WAV audio files through `LiveTranslationPipeline`:

```
================================================================================
Test Utterance 1: Hindi Speech (data/vishlb_hindi/audio/M4_fear_8.2.fear-03.wav)
================================================================================
  - Source Language:        hi
  - Target Language:        en
  - Ground Truth Emotion:   fear
  - Routed Model ID:        bilingual-ser-router-v1 -> hindi-wav2vec2-xlsr-8class
  - Detected Emotion:       fear (Ground truth matched!)
  - Emotion Confidence:     0.9412 (94.1%)
  - Top 3 Probabilities:    [('fear', 0.9412), ('neutral', 0.0245), ('sad', 0.0152)]
  - SER Latency (MPS):      183.6 ms
  - Downstream Prosody:     Tense vocal tract, hurried cadence (Fearful); Full expressive modulation

================================================================================
Test Utterance 2: English Speech (tests/test_audio/test_speech.wav)
================================================================================
  - Source Language:        en
  - Target Language:        hi
  - Routed Model ID:        bilingual-ser-router-v1 -> iemocap-wav2vec2-xlsr-4class
  - Detected Emotion:       happy
  - Emotion Confidence:     0.4208 (42.1%)
  - Native Probabilities:   [('happy', 0.4208), ('sad', 0.2525), ('angry', 0.1686), ('neutral', 0.1580)]
  - SER Latency (MPS):      189.7 ms
  - Downstream Prosody:     Elevated pitch contour (Happy); Soft blended: scaled with factor 0.47
```

---

## 7. Modified & Created Artifacts

| File | Change Type | Description |
| :--- | :---: | :--- |
| `backend/app/models/ser_hindi.py` | **Created** | Dedicated 8-class Hindi SER model adapter using trained weights and frozen XLSR encoder |
| `backend/app/models/ser_router.py` | **Created** | Dedicated bilingual SER router with language normalization, dynamic dispatch, and fallback |
| `backend/app/models/emotion_preservation.py` | **Updated** | Added target prosodic modulations and notes for `disgust` and `sarcastic` |
| `backend/app/models/tts_sarvam.py` | **Updated** | Added persona mappings for `disgust`, `sarcastic`, `fear`, and `surprise` |
| `backend/app/services/pipeline.py` | **Updated** | Fed detected ASR source language to router in auto-detect mode |
| `backend/app/services/model_lifecycle.py` | **Updated** | Enforced router provider canonicalization and singleton persistence |
| `backend/app/config.py` | **Updated** | Added `SER_MODEL_PROVIDER="router"` and `SER_HINDI_CHECKPOINT_DIR` |
| `frontend/index.html` | **Updated** | Added posterior distribution bar row for `Sarcastic` |
| `frontend/js/results.js` | **Updated** | Added emotion icons, distribution rendering for 8-class Hindi, and telemetry labels |
| `frontend/css/design-system.css` | **Updated** | Added `--emo-sarcastic` color variable |
| `frontend/css/components.css` | **Updated** | Added styling for `.dist-bar.sarcastic` and `.emotion-badge-pill.sarcastic` |
| `tests/test_bilingual_ser_router.py` | **Created** | 13 comprehensive unit and integration tests covering routing and lifecycle |

---

## 8. Limitations & Future Roadmap

1. **Acoustic Domain Variance**: The Vishal B. Hindi dataset consists of scripted/acted recordings. In conversational real-world scenarios with spontaneous disfluencies or varying background noise, additional fine-tuning or acoustic data augmentation (e.g., SpecAugment, room impulse response simulation) will further enhance robustness.
2. **End-to-End Expressive Synthesis**: While persona selection and prosody multipliers provide significant emotional expressiveness, next-generation neural TTS architectures (e.g., zero-shot emotion prompt conditioning) can further elevate voice nuance.
