# Software Requirements Specification (requirement.md)

## Project: Bilingual Voice Translation System With Emotion Awareness (VoiceAffect)

---

## 1. Functional Requirements (FR)

| Identifier | Requirement Statement | Verification Method |
| :--- | :--- | :--- |
| **FR-001** | The system **shall** capture live microphone audio from the user's browser via the HTML5 Web Audio API at a sample rate of 16 kHz mono. | Automated Test & Browser Inspection |
| **FR-002** | The system **shall** render a real-time animated frequency visualizer on an HTML5 Canvas during active microphone streaming. | UI Visual Inspection |
| **FR-003** | The system **shall** establish a persistent WebSocket connection (`/ws/live-translate`) to stream binary PCM audio chunks from client to server. | Unit Test & Integration Test |
| **FR-004** | The system **shall** accept uploaded audio and video recordings in WAV, MP3, M4A, OGG, FLAC, WebM, and MP4 formats up to 25 MB in size. | API Integration Test |
| **FR-005** | The system **shall** decode, extract, and convert uploaded audio containers into standard 16 kHz mono 16-bit PCM WAV using FFmpeg without path traversal risks. | Automated Test |
| **FR-006** | The system **shall** perform Voice Activity Detection (VAD) on incoming audio chunks to filter out silence before triggering downstream inference. | Unit Test |
| **FR-007** | The system **shall** compute Fast Fourier Transform (FFT) spectrograms and extract 13 Mel-Frequency Cepstral Coefficients (MFCCs) in accordance with Section 3, Eq. (2) of the source paper. | Mathematical Unit Test |
| **FR-008** | The system **shall** transcribe spoken speech using a local Transformer ASR model (Whisper architecture) with per-utterance confidence estimation (Eq. 4). | Automated Model Test |
| **FR-009** | The system **shall** perform Speech Emotion Recognition (SER), computing probability distributions over 6 emotion classes (*neutral, happy, sad, angry, fearful, surprised*) adhering to Eq. (5). | Mathematical Unit Test |
| **FR-010** | The system **shall** extract continuous paralinguistic coordinates: Valence ($V$), Arousal ($A$), and Dominance ($D$) from acoustic prosodic features. | Unit Test |
| **FR-011** | The system **shall** translate text bidirectionally between Hindi (Devanagari script) and English using an open Seq2Seq Transformer model (Eq. 6 & 7). | Unit Test |
| **FR-012** | The system **shall** implement Latent-Depth Emotion Gating (Eq. 8) to calculate an adaptive gating factor $z_l$ and physical prosody modulation multipliers (Pitch $\Delta f_0$, Rate $\Delta r$, Gain $\Delta v$). | Mathematical Unit Test |
| **FR-013** | The system **shall** synthesize emotion-modulated speech audio in the target language (Hindi using `Lekha`, English using `Samantha`/`Rishi`). | Integration Test |
| **FR-014** | The system **shall** display synchronized dual-language captions (original spoken transcript and target translation) in a side-by-side card layout. | UI Inspection |
| **FR-015** | The system **shall** instrument and report genuine wall-clock processing latency broken down into: ASR latency, SER latency, NMT latency, TTS latency, and Total pipeline latency. | API & WebSocket Test |
| **FR-016** | The system **shall** provide an explicit `/api/health` diagnostic endpoint indicating device acceleration status (MPS/CPU) and model readiness. | API Unit Test |
| **FR-017** | The system **shall** immediately clean up all temporary input audio and video files from `temp_storage/` post-processing. | Filesystem Audit Test |

---

## 2. Non-Functional Requirements (NFR)

| Identifier | Requirement Statement | Target Metric |
| :--- | :--- | :--- |
| **NFR-001** | **Total Processing Latency**: End-to-end processing time for a 2-second conversational speech chunk on Apple Silicon / modern CPU. | $\le 3500$ ms |
| **NFR-002** | **ASR Latency**: Monotonic execution time of speech transcription. | $\le 1200$ ms |
| **NFR-003** | **SER Latency**: Monotonic execution time of acoustic feature extraction and emotion classification. | $\le 300$ ms |
| **NFR-004** | **NMT Latency**: Monotonic execution time of sequence-to-sequence translation. | $\le 500$ ms |
| **NFR-005** | **TTS Latency**: Monotonic execution time of expressive speech synthesis. | $\le 800$ ms |
| **NFR-006** | **Scientific Integrity**: No inference results, emotion probabilities, translation text, or latency values shall be simulated or faked. | 100% Truthful Diagnostics |
| **NFR-007** | **Zero External Cost**: The system must run without requiring paid API keys, tokens, or subscription endpoints. | $0.00 Running Cost |
| **NFR-008** | **Memory Footprint**: Total host RAM consumed during inference with default models. | $\le 3.5$ GB |
| **NFR-009** | **Frontend Portability**: The UI must function in standard modern browsers (Chrome, Edge, Safari, Firefox) without external build tooling. | 100% Vanilla HTML5/CSS3/JS |
| **NFR-010** | **Input Security**: Uploaded files must be validated against allowed extensions and restricted to 25 MB. | Strict HTTP 400/413 Guards |

---

## 3. Software Prerequisites & Environment

### 3.1 Python Version
- **Selected Version**: **Python 3.10 to 3.13** (Active host runtime: Python 3.13.9).
- **Justification**: Compatible with PyTorch 2.12+, TorchAudio 2.11+, HuggingFace Transformers 5.17+, FastAPI 0.115+, and standard scientific libraries (`numpy`, `scipy`, `soundfile`).

### 3.2 System Packages & Audio Tools
- **FFmpeg**: Required for audio/video container demuxing and PCM resampling.
  - Automatically detected from system `PATH` (e.g. `/opt/homebrew/bin/ffmpeg`) or via bundled Python binary `imageio_ffmpeg` (`ffmpeg-macos-aarch64-v7.1`).
- **macOS Speech Synthesizer (`say`)**: Built into macOS; requires installed voices:
  - Hindi: `Lekha` (`hi_IN`)
  - English: `Samantha` (`en_US`) or `Rishi` (`en_IN`)

---

## 4. Dependencies & Model Files

### 4.1 Python Dependencies (`requirements.txt`)
```text
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.7.0
python-multipart>=0.0.9
websockets>=12.0
torch>=2.2.0
torchaudio>=2.2.0
transformers>=4.40.0
numpy>=1.26.0
scipy>=1.12.0
soundfile>=0.12.1
pydub>=0.25.1
imageio-ffmpeg>=0.4.9
pytest>=8.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0
```

### 4.2 Model Identifiers & Download Instructions
All models download automatically to the local HuggingFace cache upon first invocation:
1. **ASR**: `openai/whisper-tiny` (~150 MB)
2. **SER**: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` (~1.2 GB) or instant local Eq. (5) Acoustic Feature Fusion fallback.
3. **NMT (En $\rightarrow$ Hi)**: `Helsinki-NLP/opus-mt-en-hi` (~300 MB)
4. **NMT (Hi $\rightarrow$ En)**: `Helsinki-NLP/opus-mt-hi-en` (~300 MB)
5. **TTS**: Native macOS speech engine (zero download required).

---

## 5. API & WebSocket Specifications

### 5.1 REST Endpoints
- `GET /api/health`: Diagnostic health check returning device type (`mps` / `cpu`), FFmpeg path, and model statuses.
- `POST /api/translate-audio`: Multipart file upload (`file`, `source_lang`, `target_lang`, `synthesize_speech`).
- `POST /api/text-translate`: JSON body (`text`, `source_lang`, `target_lang`, `emotion`, `synthesize_speech`).
- `GET /api/audio/{filename}`: Safe temporary audio streaming endpoint with path traversal validation.

### 5.2 WebSocket Endpoint
- `WS /ws/live-translate`:
  - Client sends binary PCM chunks (16kHz, 16-bit, mono) or JSON actions (`{"action": "flush"}`).
  - Server sends JSON events (`translation_chunk`, `translation_final`, `ping`, `error`).

---

## 6. Input Validation & Restrictions
- Maximum Upload File Size: **25 MB** (`MAX_UPLOAD_SIZE_MB = 25`).
- Allowed Extensions: `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, `.webm`, `.mp4`.
- Path Traversal Guard: All file access must strictly resolve within `TEMP_STORAGE_DIR`.

---

## 7. Testing Requirements & Acceptance Criteria
- Full test coverage across audio feature extraction, MFCC computation, ASR confidence calculation, SER emotion distribution, NMT bidirectional translation, Latent-depth gating, and TTS synthesis.
- Minimum 10 automated test cases executed via `pytest tests -v` passing with zero errors.

---

## 8. Explicit Hardware Limitations
- **Apple Silicon (M1/M2/M3/M4 with MPS)**: Achieves expected real-time latency ($\tau_{\text{Total}} \approx 1.5 - 2.8$ seconds).
- **CPU-Only Environments**: End-to-end latency may increase to $4.0 - 6.5$ seconds depending on CPU core frequency and thermal throttling.
- **Initial Model Cold-Start**: First invocation requires weights download and memory caching; subsequent invocations execute at warm-state speeds.
