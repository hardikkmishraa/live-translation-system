# Project Workflow Specification & Process Architecture
## Project: Bilingual Voice Translation System With Emotion Awareness (VoiceAffect)

---

## 1. High-Level Process Architecture

The VoiceAffect system coordinates three primary operational workflows:
1. **Real-Time Live Speech Streaming Workflow** (WebSocket full-duplex session).
2. **Batch Audio File Ingestion Workflow** (REST multipart upload).
3. **Direct Text Translation & Expressive Synthesis Workflow** (REST JSON payload).

```mermaid
flowchart TD
    subgraph Client Layer [User Interfaces & Web Audio Client]
        MIC["Microphone Audio Stream (16kHz PCM)"]
        FILE["File Upload (WAV, MP3, MP4, WebM)"]
        TEXT["Direct Text Input (Hindi or English)"]
        QUEUE["Progressive Playback Queue & Canvas Waveform"]
    end

    subgraph Ingestion & Gateway [FastAPI Server / Local Gateway]
        WS_GW["WebSocket Gateway (/ws/live-translate)<br/>• Session Manager<br/>• Rolling Buffer (1.5s - 2.0s)"]
        REST_GW["REST API Gateway (/api/translate/*)<br/>• Input Validation<br/>• Rate Limiting & Auth"]
    end

    subgraph Audio Preprocessing [Core Audio Service]
        FFMPEG["FFmpeg Transcoder & Demuxer<br/>Extract to 16kHz Mono WAV"]
        NORM["Normalizer & Physics Modeler<br/>Peak = 0.95, Eq. (1)"]
        VAD["Voice Activity Detector (VAD)<br/>Energy Threshold RMS ≥ 0.008"]
    end

    subgraph Neural Processing Pipeline [Decoupled Cognitive Models]
        direction TB
        subgraph StageA [Parallel Speech & Emotion Recognition]
            FEAT["Feature Extractor<br/>Mel Filterbank Eq. (2)"]
            ASR["Transformer ASR (Whisper)<br/>Text + Confidence C Eq. (3, 4)"]
            SER["Emotion Recognition (Wav2Vec2)<br/>6-Class + VAD Vector Eq. (5)"]
        end

        subgraph StageB [Bilingual Neural Machine Translation]
            NMT["MarianMT Encoder-Decoder<br/>Hindi ⇄ English Eq. (6, 7)"]
        end

        subgraph StageC [Prosodic Preserving Voice Generation]
            GATING["Neuro-Symbolic Latent Gating<br/>Gating Weight z_l Eq. (8)<br/>Pitch Δf0, Tempo Δr, Volume Δv"]
            TTS["Expressive Voice Synthesizer<br/>DSP Prosody Vocoder Eq. (9)"]
        end
    end

    subgraph Ephemeral Output [Storage & Delivery]
        RESP["Structured JSON Result<br/>• Transcripts & Confidence<br/>• Emotion Radar & VAD<br/>• Latency Breakdown"]
        AUDIO_OUT["Base64 / Temp WAV Audio File"]
    end

    MIC -->|Binary PCM Chunks| WS_GW
    FILE -->|Multipart Form| REST_GW
    TEXT -->|JSON Request| REST_GW

    WS_GW -->|Rolling Window| VAD
    REST_GW -->|Media File| FFMPEG
    REST_GW -->|Text String| NMT

    FFMPEG --> NORM
    NORM --> VAD

    VAD -->|Active Speech| FEAT
    VAD -->|Silence / Noise| WS_GW

    FEAT --> ASR
    FEAT --> SER

    ASR -->|Source Text| NMT
    SER -->|VAD Vector| GATING
    NMT -->|Translated Text| TTS
    GATING -->|Prosody Multipliers| TTS

    TTS --> AUDIO_OUT
    ASR & SER & NMT & GATING & TTS --> RESP

    RESP & AUDIO_OUT --> WS_GW
    RESP & AUDIO_OUT --> REST_GW

    WS_GW -->|Streaming Events| QUEUE
    REST_GW -->|Synchronous Response| QUEUE
```

---

## 2. Sequence Workflow 1: Real-Time Live Speech Streaming (WebSocket)

This is the primary user interaction workflow. The browser captures microphone audio via Web Audio API, resamples it to 16 kHz Mono PCM, and streams binary chunks across a WebSocket connection.

```mermaid
sequenceDiagram
    autonumber
    actor User as Speaker
    participant UI as Browser UI / Canvas
    participant WA as Web Audio API Recorder
    participant Queue as Playback Queue
    participant WS as WebSocket Endpoint (/ws/live-translate)
    participant VAD as Energy VAD Gate
    participant ASR as Whisper ASR
    participant SER as Wav2Vec2 SER
    participant NMT as MarianMT NMT
    participant Gate as Latent Gating
    participant TTS as Expressive Synthesizer

    User->>UI: Clicks "Start Speaking"
    UI->>WA: Initialize AudioContext (16kHz Mono PCM)
    UI->>WS: Connect WebSocket & send {"action": "session_start"}
    WS-->>UI: {"type": "session_acknowledged", "session_id": "8f3a12b4"}
    
    loop Real-Time Audio Streaming (Every ~100ms - 200ms)
        User->>WA: Spoken Audio Input
        WA->>UI: Draw real-time waveform on Canvas
        WA->>WS: Send Binary PCM Chunks
        WS->>WS: Accumulate in Rolling Buffer (1.5s - 2.0s window)
        
        alt Buffer window ready or Utterance Boundary reached
            WS->>VAD: Evaluate RMS Energy
            alt Audio is Silence (RMS < 0.008)
                VAD-->>WS: Below Speech Threshold
                WS-->>UI: {"type": "ping", "status": "listening"}
            else Audible Speech Detected (RMS >= 0.008)
                VAD-->>WS: Speech Active
                
                par Speech Recognition & Emotion Detection
                    WS->>ASR: Process 16kHz PCM Buffer
                    ASR-->>WS: Source Transcript ("नमस्ते आप कैसे हैं") + Conf C = 0.94
                    WS-->>UI: {"type": "partial_transcript", "text": "नमस्ते आप कैसे हैं"}
                and Emotion Classification
                    WS->>SER: Extract MFCC + Wav2Vec2 Latents
                    SER-->>WS: Emotion: "Happy" (0.78), VAD: [V=0.72, A=0.65, D=0.60]
                    WS-->>UI: {"type": "emotion_update", "emotion": "Happy", "vad": [0.72, 0.65, 0.60]}
                end

                WS->>NMT: Translate ("नमस्ते आप कैसे हैं", hi -> en)
                NMT-->>WS: Translated Text: "Hello, how are you?"

                WS->>Gate: Compute Latent Gating Weight z_l (Eq. 8)
                Gate-->>WS: Pitch Δf0 = 1.15x, Rate Δr = 1.05x, Volume Δv = 1.05x

                WS->>TTS: Synthesize "Hello, how are you?" with Prosody Multipliers
                TTS-->>WS: Generated WAV Bytes (16kHz) + Latency Breakdown

                WS-->>UI: {"type": "translation_chunk", "segment_id": 1, "text": "Hello, how are you?", "audio_base64": "..."}
                UI->>Queue: Enqueue Segment #1
                Queue->>User: Play emotion-modulated synthetic speech
            end
        end
    end

    User->>UI: Clicks "Stop"
    UI->>WS: {"action": "flush"}
    WS->>WS: Process any remaining audio in buffer
    WS-->>UI: {"type": "session_closed"}
    WA->>WA: Disconnect microphone tracks
```

---

## 3. Sequence Workflow 2: Batch Audio File Upload & Processing

Used when translating pre-recorded interviews, voice notes, or video clips (WAV, MP3, M4A, FLAC, WebM, MP4 up to 25 MB).

```mermaid
sequenceDiagram
    autonumber
    actor User as User
    participant UI as Web Frontend
    participant API as REST Endpoint (/api/translate/audio)
    participant FF as FFmpeg Transcoder
    participant Core as Pipeline Orchestrator
    participant Disk as Ephemeral Storage (/temp_storage)

    User->>UI: Drops file "interview.mp4" into dropzone
    UI->>UI: Validate file extension & size (<= 25MB)
    User->>UI: Clicks "Translate Audio File"
    UI->>API: HTTP POST /api/translate/audio (Multipart FormData)
    
    API->>Disk: Write uploaded payload to temp file
    API->>FF: Execute FFmpeg subprocess to extract audio
    FF->>Disk: Transcode to 16kHz Mono 16-bit PCM WAV
    
    API->>Core: Invoke pipeline orchestrator on clean WAV
    Core->>Core: Run VAD, Whisper ASR, Wav2Vec2 SER
    Core->>Core: Run MarianMT NMT & Latent-Depth Gating
    Core->>Core: Synthesize expressive output WAV
    Core->>Disk: Save generated speech audio
    
    Core-->>API: Aggregate Result (Transcripts, Emotion, VAD, Latency)
    API-->>UI: HTTP 200 OK (JSON with original_audio_url, audio_url, telemetry)
    
    UI->>UI: Render Dual Transcripts & Emotion Probability Bars
    UI->>UI: Enable Audio Playback Controls & Download Button
    UI->>User: Auto-plays translated audio (if toggle enabled)
```

---

## 4. Sequence Workflow 3: Direct Text Translation & Expressive Synthesis

Used for testing prosodic modulation directly by providing source text and selecting a desired emotional disposition.

```mermaid
sequenceDiagram
    autonumber
    actor User as Researcher / User
    participant UI as Web Frontend
    participant API as REST Endpoint (/api/translate/text)
    participant NMT as MarianMT Service
    participant Gate as Latent Gating Engine
    participant TTS as Expressive TTS Service

    User->>UI: Enters "मुझे आपकी सहायता की आवश्यकता है", selects Target Emotion: "Urgent/Angry"
    User->>UI: Clicks "Translate Text"
    UI->>API: HTTP POST /api/translate/text {"text": "...", "source_lang": "hi", "target_lang": "en", "emotion": "angry"}
    
    API->>NMT: Translate text to English
    NMT-->>API: "I need your assistance immediately."
    
    API->>Gate: Compute heuristic prosody for emotion "angry"
    Gate-->>API: Pitch Δf0 = 1.20x, Rate Δr = 1.15x, Volume Δv = 1.10x
    
    API->>TTS: Synthesize English voice with elevated pitch & tempo
    TTS-->>API: 16kHz WAV Audio + Latency (NMT + TTS ms)
    
    API-->>UI: HTTP 200 OK {"translated_text": "...", "audio_base64": "...", "latency_ms": 310.5}
    UI->>UI: Display translation & play expressive audio
```

---

## 5. Pipeline Mathematical Transformation Signatures

| Stage | Input Domain | Output Domain | Governing Mathematical Formula |
| :--- | :--- | :--- | :--- |
| **1. Acquisition & VAD** | Raw Audio Signal $s(t)$ | Normalized Mono 16kHz $x(t) \in [-1.0, 1.0]$ | **Eq. (1)**: $x(t) = A(t) \cdot \sin(2\pi f t + \phi)$; $E_{\text{RMS}} = \sqrt{\frac{1}{N}\sum x[n]^2} \ge 0.008$ |
| **2. Feature Extraction** | PCM Array $x[n]$ | Mel Frequency Log Energies $M \in \mathbb{R}^{T \times 80}$ | **Eq. (2)**: $M(f) = 2595 \cdot \log_{10}(1 + f / 700)$ via 25ms STFT window |
| **3. Whisper ASR** | Mel Spectrogram $M$ | Sequence of Tokens $W = \{w_1, \dots, w_T\}$ | **Eq. (3, 4)**: $P(W\|X) = \prod_{t=1}^T P(w_t \| w_{<t}, X)$; Conf $C = \frac{1}{T}\sum \log P(w_t)$ |
| **4. Wav2Vec2 SER** | Raw Audio $x(t)$ + MFCC | 6-Class Probs $P(e)$ + VAD Vector $\vec{v} \in [0, 1]^3$ | **Eq. (5)**: $H_{\text{SER}} = \text{Softmax}(W_a \cdot F_{\text{wav2vec}} + W_p \cdot F_{\text{acoustic}} + b)$ |
| **5. MarianMT NMT** | Source Text Tokens $W_{\text{src}}$ | Target Text Tokens $Y = \{y_1, \dots, y_U\}$ | **Eq. (6, 7)**: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ |
| **6. Latent Gating** | VAD Vector $(V, A, D)$ | Latent Weight $z_l \in [0, 1]$ & Multipliers $(\Delta f_0, \Delta r, \Delta v)$ | **Eq. (8)**: $z_l = \sigma(W_v \cdot V + W_a \cdot A + W_d \cdot D + b)$ |
| **7. Expressive TTS** | Target Text $Y$ + $(\Delta f_0, \Delta r)$ | Target Audio Waveform $y_{\text{synth}}(t)$ | **Eq. (9)**: $y(t) = \text{Vocoder}\left(Y, f_0 \cdot \Delta f_0, r \cdot \Delta r, v \cdot \Delta v\right)$ |

---

## 6. Error Handling & Edge Case Workflows

```mermaid
flowchart TD
    START([Incoming Audio Chunk / Event]) --> CHECK_CONN{WebSocket Active?}
    
    CHECK_CONN -->|No| DISCONNECT[Trigger onDeviceDisconnected -> Clean up AudioContext & UI Notice]
    CHECK_CONN -->|Yes| CHECK_SILENCE{RMS Energy >= 0.008?}
    
    CHECK_SILENCE -->|Silence/Noise| PING[Send Heartbeat Ping -> Skip Heavy Inference -> Conserve Hardware]
    CHECK_SILENCE -->|Audible Speech| CHECK_BUFFER{Buffer Size > Limit?}
    
    CHECK_BUFFER -->|Overflow > 80000 samples| FORCE_BOUNDARY[Enforce Immediate Utterance Boundary -> Prevent Latency Lag]
    CHECK_BUFFER -->|Normal Buffer| INFER[Run ASR + SER + NMT Pipeline]
    FORCE_BOUNDARY --> INFER
    
    INFER --> CHECK_MODEL_ERR{Model Execution Error?}
    CHECK_MODEL_ERR -->|Yes| FALLBACK_ERR[Emit Error Notification Banner -> Gracefully Return Partial Transcript]
    CHECK_MODEL_ERR -->|No| CHECK_TTS_ERR{TTS Audio Synthesis Failed?}
    
    CHECK_TTS_ERR -->|Yes| TEXT_ONLY[Return Valid Translated Text -> Mark Expressive Audio Unsupported]
    CHECK_TTS_ERR -->|No| STREAM_CLIENT[Send Translation Event + Base64 WAV to Client Playback Queue]
```
