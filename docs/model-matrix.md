# Model Feasibility Matrix (docs/model-matrix.md)

## Project: Bilingual Voice Translation System With Emotion Awareness (VoiceAffect)

This matrix systematically evaluates all candidate models across each pipeline stage based on verified public repositories, exact checkpoint availability, licensing, language support, local inference procedure, hardware demands, integration risks, and operational fallback strategies.

---

## 1. Automatic Speech Recognition (ASR)

| Candidate Model | Official Repository | Exact Checkpoint | License | Supported Languages | Local Inference Procedure | Hardware Needs | Integration Risks | Fallback Strategy | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Whisper-Tiny** (Recommended) | `openai/whisper-tiny` (HuggingFace / OpenAI) | `openai/whisper-tiny` | Apache 2.0 | Multilingual (99+ languages including Hindi `hi` and English `en`) | `transformers.pipeline("automatic-speech-recognition")` | ~150 MB RAM; runs fast on CPU / Apple Silicon MPS (<800ms per 2s audio) | Lower accuracy on dense technical jargon compared to larger variants. | Fallback to `whisper-base` if higher accuracy is needed and latency allows. | **Selected (Primary)** |
| **Whisper-Base** | `openai/whisper-base` (HuggingFace / OpenAI) | `openai/whisper-base` | Apache 2.0 | Multilingual (Hindi `hi`, English `en`) | `transformers.pipeline("automatic-speech-recognition")` | ~300 MB RAM; ~1200ms per 2s audio on CPU/MPS | Marginal latency increase for real-time live mode on slower CPUs. | Revert to `whisper-tiny`. | Supported Alternative |
| **Wav2Vec2-XLSR-53** | `facebook/wav2vec2-large-xlsr-53` | `facebook/wav2vec2-large-xlsr-53` | Apache 2.0 | Multilingual pretrained representations | Requires separate CTC language prediction heads for Hindi and English | ~1.2 GB RAM; ~1500ms on CPU | Does not have a unified bilingual text generation head out of the box; requires custom CTC decoding and language models. | Whisper-Tiny. | Rejected (Too complex for bilingual streaming) |

---

## 2. Speech Emotion Recognition (SER)

| Candidate Model | Official Repository | Exact Checkpoint | License | Supported Languages | Local Inference Procedure | Hardware Needs | Integration Risks | Fallback Strategy | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CAMuLeNet** (Paper Proposed) | None (Conference paper: Goel et al., Interspeech 2024 [1]) | None (No public checkpoint released) | N/A | Multilingual (Paper claims) | Cannot run locally; no open code or weights exist | Unknown | **Critical Risk**: Weights are unreleased. Claiming CAMuLeNet is running violates scientific integrity. | Scientific Acoustic Feature Fusion Engine (Eq. 5). | **Unavailable (Documented)** |
| **Wav2Vec2 XLSR SER** | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` | Apache 2.0 | English (Transfers acoustically across Indic speech) | `transformers.pipeline("audio-classification")` | ~1.2 GB RAM; ~300ms on MPS/CPU | Trained primarily on English speech emotion corpora; cross-lingual emotion transfer to Hindi relies on acoustic prosody. | Acoustic-Prosodic Feature Fusion Engine (Eq. 5). | **Selected (Primary Neural)** |
| **SUPERB Wav2Vec2 ER** | `superb/wav2vec2-base-superb-er` | `superb/wav2vec2-base-superb-er` | Apache 2.0 | English (IEMOCAP 4-class) | `transformers.pipeline("audio-classification")` | ~400 MB RAM; ~200ms | 4-class taxonomy (neutral, happy, sad, angry) lacks fear and surprise. | `ehcalabres` checkpoint. | Secondary Candidate |
| **Acoustic-Prosodic Feature Fusion Engine** (Section 3 Eq. 5) | Custom Local Implementation (`backend/app/models/ser.py`) | Mathematical implementation of Eq. (5) | MIT / Open | Language-Agnostic (Acoustic physics based) | Physical extraction of $F_0$, RMS Energy, ZCR, MFCCs, and Spectral Centroid | <10 MB RAM; ~15ms on CPU | Rule-based calibration of logit weights based on psychoacoustic literature. | None (Deterministic local computation). | **Selected (Transparent Baseline & Fallback)** |

---

## 3. Bilingual Neural Machine Translation (NMT: Hindi $\leftrightarrow$ English)

| Candidate Model | Official Repository | Exact Checkpoint | License | Supported Languages | Local Inference Procedure | Hardware Needs | Integration Risks | Fallback Strategy | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Helsinki-NLP Opus-MT** (Recommended) | `Helsinki-NLP/opus-mt-en-hi` & `opus-mt-hi-en` | `Helsinki-NLP/opus-mt-en-hi`, `Helsinki-NLP/opus-mt-hi-en` | Apache 2.0 | Hindi (`hi`) and English (`en`) | `transformers.pipeline("translation")` | ~300 MB RAM per direction; ~180ms per sentence on CPU/MPS | Handles colloquial Indic slang less fluently than 1B parameter models. | Lexical bilingual dictionary fallback. | **Selected (Primary)** |
| **IndicTrans2 1B** | `ai4bharat/indictrans2-indic-en-1B` & `en-indic-1B` | `ai4bharat/indictrans2-indic-en-1B` | MIT | 22 Scheduled Indian languages + English | `AutoModelForSeq2SeqLM` + `IndicProcessor` | >4 GB RAM / VRAM; ~800-1500ms on edge hardware | Heavy download size (~4 GB per model), high RAM consumption, complex tokenizer dependencies (`IndicProcessor`). | Helsinki-NLP Opus-MT. | Supported for High-End Setups |
| **NLLB-200 Distilled 600M** | `facebook/nllb-200-distilled-600M` | `facebook/nllb-200-distilled-600M` | CC-BY-NC 4.0 | 200+ languages (`hin_Deva`, `eng_Latn`) | `transformers.pipeline("translation")` with target language tokens | ~1.5 GB RAM; ~600ms on CPU | License is non-commercial (CC-BY-NC 4.0); requires explicit BCP-47 language token management. | Helsinki-NLP Opus-MT. | Feasible Alternative |
| **MuRIL** | `google/muril-base-cased` | `google/muril-base-cased` | Apache 2.0 | 17 Indian languages | Masked LM representation encoder | ~450 MB RAM | **Not a generative translation model**; MuRIL is a BERT encoder for classification/embedding, not sequence-to-sequence generation. | Helsinki-NLP Opus-MT. | Rejected for Direct NMT |

---

## 4. Text-to-Speech (TTS) & Emotion Modulation

| Candidate Model | Official Repository | Exact Checkpoint | License | Supported Languages | Local Inference Procedure | Hardware Needs | Integration Risks | Fallback Strategy | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Native System Speech (`say`) + DSP Modulation** | Built-in macOS Operating System | Voices: `Lekha` (Hindi) & `Samantha`/`Rishi` (English) | System Component | Hindi (`hi_IN`), English (`en_US`, `en_IN`) | `subprocess.run(["say", "-v", voice, "-r", wpm, ...])` + SciPy DSP modulation | Zero additional RAM; <150ms execution; zero download | Non-macOS environments require alternative local TTS (`espeak-ng` or `piper`). | Melodic carrier tone fallback. | **Selected (Primary)** |
| **Coqui YourTTS** (Paper Proposed) | Coqui TTS GitHub | `tts_models/multilingual/multi-dataset/your_tts` | MPL 2.0 | English, French, Portuguese | Coqui TTS Python API | ~1.5 GB RAM; ~2500ms | **Hindi is unsupported in default YourTTS checkpoints**; fails out-of-the-box on Hindi Devanagari text. | Native System Speech. | **Incompatible for Hindi** |
| **Meta MMS-TTS** | `facebook/mms-tts-hin` & `facebook/mms-tts-eng` | `facebook/mms-tts-hin`, `facebook/mms-tts-eng` | CC-BY-NC 4.0 | Hindi (Devanagari) & English | `VitsModel` + `AutoTokenizer` in HuggingFace | ~500 MB RAM; ~600ms on CPU | Separate models for Hindi and English; emotional modulation must be injected via external vocoder pitch manipulation. | Native System Speech. | Supported Alternative |
| **Microsoft Edge-TTS** | Remote Python wrapper (`edge-tts`) | Cloud neural voices (`hi-IN-SwaraNeural`, `en-US-AriaNeural`) | Proprietary / Cloud | Hindi, English | Asynchronous websocket connection to Microsoft endpoints | Low local RAM; network dependent | Violates offline, privacy, and zero-external-cost rules. Not acceptable as primary. | Native System Speech. | Rejected (Remote Dependency) |

---

## 5. Emotion Preservation (Latent-Depth Gating)

| Candidate Model | Official Repository | Exact Checkpoint | License | Mechanism | Hardware Needs | Integration Risks | Fallback Strategy | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TrustTranslate Gating** (Paper Proposed) | None (Conference paper: Ben Chaabene et al., ENASE 2026 [9]) | None (No public checkpoint or library) | N/A | Theoretical formulation: $h'_l = z_l h_l + (1 - z_l) \text{LayerNorm}(h_{l-1})$ | N/A | **Critical Risk**: Concept lacks open source implementation or pre-trained neural weights. | Latent-Depth Emotion Preserver (`backend/app/models/emotion_preservation.py`). | **Theoretical Concept (Adapted)** |
| **Latent-Depth Emotion Preserver** | Custom Local Implementation | Neuro-symbolic mapping of VAD vector to Eq. (8) gating weight $z_l$ and prosody multipliers | MIT / Open | Translates Valence, Arousal, and Dominance into discrete pitch ($\Delta f_0$), rate ($\Delta r$), and amplitude ($\Delta v$) factors | Minimal (<1 MB RAM; <1ms) | Relies on downstream TTS engine's ability to modulate pitch and rate accurately. | Baseline conversational prosody ($1.0\times$). | **Selected (Implemented)** |

---

## 6. Summary of Verified Model Decisions

1. **ASR**: `openai/whisper-tiny` (Apache 2.0, local PyTorch inference, bilingual, lightweight, fast).
2. **SER**: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` backed by local Section 3 Eq. (5) Acoustic-Prosodic Feature Fusion Engine (CAMuLeNet verified as having no public checkpoints).
3. **NMT**: `Helsinki-NLP/opus-mt-en-hi` and `Helsinki-NLP/opus-mt-hi-en` with conversational lexical fallback (IndicTrans2 supported as high-memory alternative).
4. **TTS**: Native macOS speech engine (`Lekha` / `Samantha`) modulated with SciPy pitch/rate/amplitude DSP based on the Latent-Depth Gating plan (YourTTS verified as lacking Hindi out-of-the-box support).
5. **Emotion Preservation**: Latent-Depth Gating service computing $z_l \in [0, 1]$ and prosody control vectors (TrustTranslate verified as an unreleased theoretical formulation).
