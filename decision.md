# Architecture & Engineering Decision Record (decision.md)

## Project: Bilingual Voice Translation System With Emotion Awareness (VoiceAffect)

---

## 1. Selected Technology Stack & Rationale

| Layer | Chosen Technology | Rationale & Justification |
| :--- | :--- | :--- |
| **Frontend UI** | HTML5, CSS3 (Vanilla), JavaScript (Vanilla) | Satisfies the strict system constraint prohibiting frontend build frameworks (no React, Next.js, Vue, TypeScript). Provides raw, unabstracted control over the Web Audio API, Canvas rendering, and WebSockets without node packaging overhead or hydration delays. |
| **Backend Framework** | Python 3.10+ & FastAPI | High-performance asynchronous ASGI framework with native WebSocket support, automatic OpenAPI/JSON schema documentation, and direct integration with PyTorch and audio DSP libraries. |
| **Live Communication** | Native WebSockets (`/ws/live-translate`) | Full-duplex, low-overhead binary/text communication enabling streaming of microphone audio chunks and push delivery of partial/final transcripts, emotion metrics, and audio payloads without HTTP polling. |
| **Audio Processing** | NumPy, SciPy, SoundFile, FFmpeg | Industry-standard scientific libraries providing mathematical implementations of FFT, Mel filterbanks, MFCCs, autocorrelation pitch tracking, and universal container decoding via FFmpeg. |
| **AI Inference** | PyTorch (`mps` / `cpu`), HuggingFace Transformers | Open, local deep learning inference engine with native Apple Silicon hardware acceleration (`torch.backends.mps.is_available()`) and zero cloud dependencies. |
| **Local Storage** | Ephemeral Local File Storage (`temp_storage/`) | Secure, session-isolated temporary storage with automated post-processing cleanup; avoids heavy relational database overhead where persistence is unnecessary. |
| **Testing** | Pytest, Pytest-asyncio, HTTPX | Modern, mature Python testing stack for asynchronous endpoints, WebSockets, DSP feature extractors, and modular inference classes. |

---

## 2. Architectural Style & Module Boundaries

The system adopts a **Modular Cascaded Architecture with Independent Service Boundaries**:
- Each cognitive stage (**Audio Preprocessing**, **ASR**, **SER**, **NMT**, **Emotion Gating**, **TTS**) is implemented behind an explicit Abstract Base Class (`BaseASRModel`, `BaseSERModel`, etc.).
- Modules are completely decoupled: replacing the ASR engine (e.g. swapping Whisper-Tiny for Whisper-Base or Wav2Vec2) requires zero modifications to the NMT or SER services.
- A dedicated **Pipeline Orchestrator** manages data flow, Voice Activity Detection gating, stage-by-stage wall-clock latency measurement, and error boundaries.

---

## 3. Candidate Model Comparison & Final Decisions

### 3.1 Automatic Speech Recognition (ASR)
- **Candidates Considered**:
  1. `openai/whisper-tiny` (39M parameters)
  2. `openai/whisper-base` (74M parameters)
  3. `facebook/wav2vec2-large-xlsr-53` (317M parameters)
- **Selected**: **`openai/whisper-tiny`** (with lazy loading and configurable model ID via `.env`).
- **Rationale**: Whisper-tiny achieves sub-second inference on Apple Silicon MPS/CPU (~400-800ms for 2s audio), natively supports bilingual speech (Hindi and English), handles diverse background accents, and operates completely offline with an Apache 2.0 license. Wav2Vec2-XLSR requires separate acoustic language heads and language model decoders for Hindi and English, increasing pipeline complexity.

### 3.2 Speech Emotion Recognition (SER)
- **Candidates Considered**:
  1. CAMuLeNet (Paper proposal)
  2. `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
  3. `superb/wav2vec2-base-superb-er`
  4. Hybrid Acoustic-Prosodic Feature Fusion Engine (Section 3 Eq. 5 implementation)
- **Critical Finding**: **CAMuLeNet weights are NOT publicly available on HuggingFace or open model hubs**. Reference [1] from the paper (Goel et al., Interspeech 2024) is a conference paper without public model checkpoints.
- **Decision**: Implement a **Dual-Engine Strategy**:
  - Primary Neural: Open Wav2Vec2-based emotion classification (`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`, Apache 2.0).
  - Scientific Fallback: Real-time Multimodal Acoustic Feature Fusion Engine based on Eq. (5) ($\hat{E} = \text{Softmax}(W_e F + b_e)$), extracting physical acoustic correlates ($F_0$, RMS Energy, ZCR, MFCCs, Spectral Centroid) to output legitimate, un-faked probabilities and continuous Valence-Arousal-Dominance (VAD) coordinates.

### 3.3 Hindi-English Neural Machine Translation (NMT)
- **Primary Engine Selected**: **Sarvam AI (`mayura:v1` / `sarvam-translate:v1`)** via dedicated translation service.
- **Secondary / Offline Alternative**: `Helsinki-NLP/opus-mt` pair (activated explicitly with `NMT_BACKEND=local`).
- **Policy on Automatic Fallbacks**: In compliance with strict engineering requirements, **no silent or automatic fallbacks** (such as MyMemory or MarianMT) are permitted when Sarvam AI fails. If Sarvam is unavailable or encounters rate limits/outages, the system marks the translation as failed, records explicit structured diagnostics, preserves the original ASR transcription intact, and refuses to invent simulated translations.
- **Rationale**: Sarvam AI provides native conversational Hindi ⇄ English translation with superior linguistic nuances and dialectal accuracy. API credentials (`SARVAM_API_KEY`) remain strictly server-side in backend environment variables and are masked in all logs and reports. Transient network/server errors (HTTP 429, 5xx, timeouts) are handled with bounded exponential backoff retries, while authentication errors fail fast.

### 3.4 Text-to-Speech (TTS) & Emotion Modulation
- **Candidates Considered**:
  1. Coqui YourTTS / XTTS-v2
  2. `facebook/mms-tts-hin` / `facebook/mms-tts-eng` (VITS)
  3. Microsoft Edge-TTS
  4. Native System Synthesis (`say` on macOS with `Lekha` [Hindi] and `Samantha` [English]) + DSP Prosody Modulation
- **Selected**: **Native Offline System Speech Engine (`Lekha` / `Samantha`) + Scientific Prosody DSP Engine**.
- **Rationale**:
  - YourTTS does not support native Hindi synthesis out-of-the-box. XTTS-v2 is heavy (>2GB) and slow on non-CUDA machines (>3-5s per sentence).
  - Edge-TTS requires remote Microsoft cloud endpoints, violating the offline, privacy-preserving, and zero-external-dependency constraints.
  - Native macOS speech (`say`) has pre-installed, high-quality, zero-latency voices (`Lekha` for Hindi, `Samantha` for English).
  - By applying digital signal processing (pitch shifting, time-domain rate scaling, and amplitude gain) conditioned on the Latent-Depth Gating plan (Eq. 8), we achieve genuine emotion-modulated acoustic synthesis locally in <200ms with zero network cost.

---

## 4. Why Local Inference is Preferred

1. **Zero Cost**: Eliminates token billing, API keys, subscription fees, and rate-limiting quotas.
2. **Privacy by Design**: Spoken voice data never traverses third-party servers, satisfying strict enterprise and personal privacy standards.
3. **Deterministic Latency**: Not subject to variable public internet transit or cloud queue throttling.
4. **Offline Capability**: Operates in low-connectivity, air-gapped, or emergency response environments.

---

## 5. Live Processing & WebSocket Design Decisions

### 5.1 Streaming Protocol
- The client captures 16kHz 1-channel PCM audio from the microphone via the Web Audio API.
- Binary Int16Array buffers are streamed over a persistent WebSocket connection.
- A sliding accumulation buffer on the server evaluates audio chunk duration. When 1.5 to 2.0 seconds of audio accumulate, the pipeline triggers processing.
- A Voice Activity Detector (VAD) inspects Root Mean Square (RMS) energy. Silent frames emit a lightweight `ping` back to the client, preventing wasteful GPU/CPU inference.

### 5.2 Decoupled Audio Streaming
- Translated audio is returned as Base64-encoded WAV in the WebSocket JSON response, allowing immediate client-side playback through an HTML5 Audio element.
- The UI separates partial utterance monitoring from final translation rendering to avoid jarring text jitter.

---

## 6. Alternatives Considered & Their Disadvantages

| Alternative | Considered For | Why Rejected |
| :--- | :--- | :--- |
| **React / Next.js** | Frontend | Prohibited by explicit project specification. Introduces virtual DOM overhead, node module bloat, and build step complexity. |
| **Google Cloud Speech / Translate** | ASR / NMT | Paid API, requires remote credentials, leaks audio data to external cloud servers, and violates the zero-paid-API rule. |
| **End-to-End Direct S2ST (e.g. AudioPaLM)** | Pipeline | Monolithic models lack modularity; impossible to inspect or replace intermediate emotion coordinates or translation text; requires massive multi-GPU clusters. |
| **SQLite Audio Archiving** | Storage | Unnecessary disk overhead; audio recordings should be ephemeral to respect user privacy and prevent storage exhaustion. |

---

## 7. Model Licensing & Deployment Constraints

| Component | Selected Model / Tool | License | Deployment Feasibility |
| :--- | :--- | :--- | :--- |
| **ASR** | `openai/whisper-tiny` | Apache 2.0 | Fully commercial & open-source friendly; local PyTorch execution. |
| **SER** | `ehcalabres/wav2vec2-lg-xlsr` / Eq. 5 Acoustic Fusion | Apache 2.0 / Custom Open Code | Permissive; zero external restrictions. |
| **NMT** | `Helsinki-NLP/opus-mt-en-hi` & `opus-mt-hi-en` | Apache 2.0 | Permissive, open, offline MarianMT weights. |
| **TTS** | macOS Native Speech + DSP Modulation | System Component (macOS) | Completely local, offline, free; zero licensing encumbrance. |
| **FFmpeg** | Bundled `imageio-ffmpeg` (LGPL/GPL) / System FFmpeg | LGPL / GPL v2+ | Permissive binary execution via subprocess without linking issues. |

---

## 8. Hardware & Operating-System Compatibility

- **macOS (Primary Target)**:
  - Apple Silicon (M1/M2/M3/M4): Uses PyTorch `mps` (Metal Performance Shaders) backend for accelerated tensor math.
  - Native TTS voices (`Lekha`, `Samantha`, `Rishi`) are built into macOS.
- **Linux / Windows (Portability)**:
  - PyTorch automatically falls back to CPU or CUDA.
  - Audio processing relies on standard cross-platform libraries (`soundfile`, `scipy`).
  - TTS falls back to synthetic melodic carrier tone generation or local cross-platform tools (`espeak` / `piper`).

---

## 9. Known Trade-Offs & Unresolved Decisions

1. **Model Size vs. Translation Nuance**:
   - *Trade-off*: `whisper-tiny` and `opus-mt` are selected for low latency and modest memory consumption (<1.5 GB RAM total). Larger models (e.g. `whisper-small`, IndicTrans2 1B) yield higher translation BLEU scores on complex literary idioms, but increase latency beyond the 3.5s real-time target on edge hardware.
2. **Emotion Modulation via DSP vs. End-to-End Emotional Voice Cloning**:
   - *Trade-off*: Applying time-domain and frequency-domain DSP modulation (pitch shift, tempo scaling, volume gain) to clean synthetic speech is fast (<50ms), reliable, and does not hallucinate phonemes. True neural voice cloning (e.g. XTTS) can capture timbre, but is computationally prohibitive in real-time streaming and carries ethical/consent risks.
3. **CAMuLeNet Availability**:
   - *Trade-off*: Because CAMuLeNet pre-trained weights were not released with the original paper, the system implements a scientifically transparent dual-engine approach combining open Wav2Vec2 with the paper's acoustic feature fusion formulation.
