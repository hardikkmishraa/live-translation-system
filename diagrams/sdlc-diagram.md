# Software Development Life Cycle (SDLC) Specification
## Project: Bilingual Voice Translation System With Emotion Awareness (VoiceAffect)

---

## 1. SDLC Methodology Overview

VoiceAffect is engineered using an **Agile-V-Model Hybrid SDLC** tailored for Deep Learning Speech Processing and Paralinguistic Systems. This methodology combines the rapid iterative sprints of Agile with the rigorous verification and validation checkpoints of the V-Model.

Because neural speech systems (ASR, SER, NMT, Latent Gating, TTS) suffer from **cascaded error propagation** where upstream acoustic or phonetic errors multiply downstream, every development phase is strictly paired with a corresponding empirical verification gate.

```mermaid
flowchart TD
    subgraph Requirements & Design [Specification & Architecture Phase]
        REQ["1. Requirements & PRD<br/>• Problem Definition<br/>• Target Languages (Hindi ⇄ English)<br/>• Non-Functional Constraints"]
        SYS_DESIGN["2. System Architecture Design<br/>• Decoupled 6-Stage Pipeline<br/>• WebSocket Streaming Protocol<br/>• Local Offline Execution (MPS/CPU)"]
        MODEL_DESIGN["3. Neural Model & Mathematical Design<br/>• Acoustic Physics (Eq. 1 & 2)<br/>• Transformer ASR & NMT (Eq. 3, 4, 6, 7)<br/>• Neuro-Symbolic Latent Gating (Eq. 8, 9)"]
    end

    subgraph Implementation [Core Implementation Phase]
        IMPL["4. Pipeline & Module Implementation<br/>• Audio Preprocessor & VAD<br/>• Whisper & MarianMT Integration<br/>• Wav2Vec2 SER & Latent Gating<br/>• Local Expressive TTS Engine<br/>• Progressive Web Audio Frontend"]
    end

    subgraph Verification & Validation [Empirical Testing & Auditing Phase]
        UNIT_TEST["5. Unit & Component Verification<br/>• Audio Preprocessing Tests<br/>• Energy-based VAD Cutoffs<br/>• Model Interface Contracts"]
        INT_TEST["6. Integration & Latency Benchmarks<br/>• Full Pipeline Roundtrip (≤ 3.5s)<br/>• Real-Time WebSocket Streaming<br/>• Memory & Ephemeral Cache Leaks"]
        SCIENTIFIC_AUDIT["7. Scientific Validation & Paper Auditing<br/>• ASR Confidence Verification (C ≥ 0.85)<br/>• Emotion Retention & Consistency (COS Sim)<br/>• Acoustic SNR & MOS Proxies<br/>• Truthful Methodology Disclosures"]
        DEPLOY["8. Deployment & Operational Lifecycle<br/>• Local Offline MPS/Apple Silicon Execution<br/>• Health Probes & Lifecycle Manager<br/>• Ephemeral Storage Garbage Collection"]
    end

    REQ -.->|Validated By| SCIENTIFIC_AUDIT
    SYS_DESIGN -.->|Verified By| INT_TEST
    MODEL_DESIGN -.->|Verified By| UNIT_TEST
    REQ --> SYS_DESIGN
    SYS_DESIGN --> MODEL_DESIGN
    MODEL_DESIGN --> IMPL
    IMPL --> UNIT_TEST
    UNIT_TEST --> INT_TEST
    INT_TEST --> SCIENTIFIC_AUDIT
    SCIENTIFIC_AUDIT --> DEPLOY
```

---

## 2. Phase-by-Phase SDLC Breakdown

### Phase 1: Requirements Engineering & Literature Grounding
* **Objective**: Formulate the problem statement based on peer-reviewed Cross-Language Speech Translation (CLST) literature and define actionable project goals.
* **Key Inputs**:
  * Research Paper: *"Bilingual Voice Translation System With Emotion Awareness"* (Ayush Singh Mehta, Hardik Mishra, Hardik Sishodia, Amit Sharma — Sharda University).
  * Limitations of conventional CLST: Affective flattening, vocal identity dissociation, and lack of Hindi parallel emotional datasets.
* **Key Deliverables**:
  * Product Requirements Document (`prd.md`).
  * Requirement Traceability Matrix (`requirement.md`).
  * Architectural Decisions & Methodology Disclosures (`decision.md`).
* **Exit Criteria**:
  * Clear functional requirements for Live Speech, File Upload, and Direct Text translation.
  * Explicit latency target ($\le 3.5$s end-to-end) and local offline inference constraints (Zero paid cloud APIs, complete data privacy).

---

### Phase 2: System Architecture & Mathematical Specification
* **Objective**: Design a decoupled, non-blocking modular architecture ensuring real-time responsiveness and scientific rigor.
* **Key Tasks**:
  * Design decoupled client-server topology (FastAPI backend + Vanilla HTML5/CSS/JS frontend).
  * Define bidirectional WebSocket protocol (`/ws/live-translate`) with sliding-window chunking.
  * Formalize mathematical pipeline equations:
    * **Eq. (1)**: Audio acquisition waveform modeling $x(t) = A(t) \cdot \sin(2\pi f t + \phi)$.
    * **Eq. (2)**: Mel Filterbank feature representation $M(f) = 2595 \cdot \log_{10}(1 + f / 700)$.
    * **Eq. (3, 4)**: Transformer ASR posterior decoding and utterance confidence $C = \frac{1}{T}\sum_{t=1}^T \log P(w_t | w_{<t}, X)$.
    * **Eq. (5)**: Acoustic-prosodic multimodal feature fusion for SER.
    * **Eq. (6, 7)**: Bidirectional encoder-decoder self-attention for Hindi $\leftrightarrow$ English NMT.
    * **Eq. (8)**: Neuro-symbolic latent-depth prosody gating $z_l = \sigma(W_v \cdot V + W_a \cdot A + W_d \cdot D + b)$.
    * **Eq. (9)**: Target speech synthesis prosodic scaling $(\Delta f_0, \Delta r, \Delta v)$.
* **Key Deliverables**:
  * Architecture Document (`docs/ARCHITECTURE.md`).
  * OpenAPI REST & WebSocket Contract (`docs/api-contract.md`).
* **Exit Criteria**:
  * Module contracts approved with explicit error codes and latency breakdown schemas.

---

### Phase 3: Neural Model Selection, Adaptation & Prototyping
* **Objective**: Evaluate, select, and integrate local open-source neural models meeting license, memory, and Apple Silicon/MPS acceleration constraints.
* **Component Matrix**:
  | Module | Selected Checkpoint / Engine | Device Support | Role & Scientific Justification |
  | :--- | :--- | :--- | :--- |
  | **ASR** | `openai/whisper-tiny` | MPS / CPU | Multilingual Hindi/English speech transcription; robust to acoustic noise |
  | **SER** | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` | MPS / CPU | 8-class $\to$ 6-class emotion mapping + continuous VAD coordinate extraction |
  | **NMT** | `Helsinki-NLP/opus-mt-hi-en` & `opus-mt-en-hi` | MPS / CPU | Bilingual sequence-to-sequence translation with subword tokenization |
  | **Gating** | Custom Neuro-Symbolic Latent Engine | CPU | Computes latent weight $z_l$ (Eq. 8) and maps VAD to prosody multipliers |
  | **TTS** | Native OS Speech Engine + DSP Prosody Vocoder | System / Native | Offline synthesis supporting Hindi & English with pitch & tempo scaling |
* **Key Deliverables**:
  * Model Matrix Document (`docs/model-matrix.md`).
  * Model lifecycle manager with lazy loading and memory conservation (`backend/app/services/model_lifecycle.py`).
* **Exit Criteria**:
  * All models pass initialization tests without external network dependencies.

---

### Phase 4: Core Implementation & Progressive Streaming
* **Objective**: Implement the full pipeline, WebSocket streaming handler, and frontend user interface.
* **Backend Implementation**:
  * `backend/app/core/audio_processor.py`: Audio normalization, resampling to 16 kHz Mono PCM, energy-based VAD.
  * `backend/app/services/pipeline.py`: Asynchronous pipeline orchestrator tying together ASR, SER, NMT, Latent Gating, and TTS.
  * `backend/app/api/endpoints/websocket.py`: Full-duplex session manager, sliding-window buffer, utterance boundary detector.
* **Frontend Implementation**:
  * `frontend/js/audio.js`: Web Audio API capture, downsampling, and real-time visualizer.
  * `frontend/js/playback_queue.js`: Progressive audio queue ensuring synchronized, jitter-free playback.
  * `frontend/index.html` & CSS: Clean, modern, distraction-free interface.
* **Key Deliverables**:
  * End-to-end runnable application on `localhost:8000`.
* **Exit Criteria**:
  * Zero browser console errors; successful audio recording, translation, and voice playback roundtrip.

---

### Phase 5: Verification, Validation & Benchmarking
* **Objective**: Conduct rigorous empirical testing across functionality, latency, and scientific claims.
* **Testing Dimensions**:
  * **Unit Testing**: Pytest suite for individual modules (FFmpeg extraction, audio resampling, VAD energy cutoff).
  * **Integration Testing**: WebSocket session handshake, cancellation, flush, and timeout handling.
  * **Latency Benchmarks**: Automated measuring of each stage's latency against the $\le 3.5$s requirement.
  * **Scientific Fidelity Audit**: Evaluation of emotion retention consistency between input speech and synthesized audio.
* **Key Deliverables**:
  * Test Suite (`tests/test_api.py`, `tests/test_pipeline.py`, etc.).
  * Scientific Audit Report (`docs/phase7-audit-report.md`).
* **Exit Criteria**:
  * 100% of integration test suites pass; verified offline operation without internet connectivity.

---

### Phase 6: Deployment, Optimization & Maintenance
* **Objective**: Ensure long-running operational stability, memory reclamation, and graceful error handling.
* **Key Operational Tasks**:
  * PyTorch device allocation (`mps` on macOS Apple Silicon, `cuda` on Nvidia Linux, `cpu` fallback).
  * Ephemeral audio garbage collection cleaning `temp_storage/` to prevent disk bloat.
  * Idle session cleanup in WebSocket gateway (120-second inactivity timeout).
  * Health check probes (`/api/health`) tracking uptime and module readiness.
* **Key Deliverables**:
  * Production-ready deployment scripts and documentation (`README.md`).

---

## 3. SDLC Stage Traceability Matrix

```mermaid
gantt
    title VoiceAffect SDLC Implementation Timeline & Verification Gates
    dateFormat  YYYY-MM-DD
    section Phase 1: Planning
    Literature Review & PRD Formulation       :done,    p1_1, 2026-09-01, 2026-09-10
    Technical Feasibility & Gap Analysis     :done,    p1_2, 2026-09-11, 2026-09-15
    section Phase 2: Architecture
    Decoupled System Architecture Design      :done,    p2_1, 2026-09-16, 2026-09-22
    Mathematical Modeling (Eq. 1-9)          :done,    p2_2, 2026-09-23, 2026-09-28
    section Phase 3: Model Engineering
    Model Selection (Whisper, MarianMT, SER) :done,    p3_1, 2026-09-29, 2026-10-04
    Acoustic Feature Extraction & VAD Setup  :done,    p3_2, 2026-10-05, 2026-10-08
    section Phase 4: Implementation
    Backend Orchestrator & WS Endpoints      :done,    p4_1, 2026-10-09, 2026-10-12
    Client Web Audio & Visualizer            :done,    p4_2, 2026-10-12, 2026-10-14
    section Phase 5: Verification & Audit
    Empirical Latency & MOS Benchmarking     :done,    p5_1, 2026-10-15, 2026-10-16
    Phase 7 Scientific Auditing & Disclosures:done,    p5_2, 2026-10-16, 2026-10-17
    section Phase 6: UX Polish & Release
    Frontend UI Simplification & Cleanup     :done,    p6_1, 2026-10-17, 2026-10-17
    Operational Deployment & Docs Delivery   :active,  p6_2, 2026-10-17, 2026-10-18
```
