# API & WebSocket Interface Contract (docs/api-contract.md)

## Project: Bilingual Voice Translation System With Emotion Awareness (VoiceAffect)

---

## 1. Protocol Overview

The application provides two complementary communication interfaces:
1. **REST API**: For operational health inspection, media file uploading, text translation, evaluation reports, and audio streaming.
2. **WebSocket API**: For full-duplex, low-latency live microphone streaming and real-time bidirectional translation with monotonic segment ordering.

Both `/api/...` (legacy convenience) and `/api/v1/...` (standard versioned) routes are supported.

---

## 2. REST Endpoints

### 2.1 System Health & Diagnostics
- **Method**: `GET`
- **Routes**: `/api/health`, `/api/v1/health`
- **Description**: Returns hardware acceleration status, FFmpeg availability, and cognitive model readiness.
- **Response Format**: `application/json`

#### Response Body Example (`200 OK`):
```json
{
  "status": "healthy",
  "device": "mps",
  "ffmpeg_available": true,
  "ffmpeg_path": "/opt/anaconda3/lib/python3.13/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1",
  "modules": {
    "asr": {
      "module": "ASR (Automatic Speech Recognition)",
      "model_name": "openai/whisper-tiny",
      "is_loaded": true,
      "device": "mps",
      "status_message": "Loaded & Ready"
    },
    "ser": {
      "module": "SER (Speech Emotion Recognition)",
      "model_name": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
      "is_loaded": true,
      "device": "mps",
      "status_message": "Acoustic-Prosodic Fusion Engine Ready (Eq. 5)"
    },
    "nmt": {
      "module": "NMT (Neural Machine Translation)",
      "model_name": "Helsinki-NLP/opus-mt-hi-en / Helsinki-NLP/opus-mt-en-hi",
      "is_loaded": true,
      "device": "mps",
      "status_message": "Transformer Models Loaded"
    },
    "tts": {
      "module": "TTS (Text-to-Speech)",
      "model_name": "Native Emotion-Modulated TTS (Lekha / Samantha)",
      "is_loaded": true,
      "device": "cpu",
      "status_message": "Offline Native Synthesis Ready with Emotion Prosody Modulation"
    }
  },
  "supported_pairs": ["Hindi -> English", "English -> Hindi"],
  "timestamp": 1776360000.123
}
```

---

### 2.2 Model Status & Readiness
- **Method**: `GET`
- **Route**: `/api/v1/models/status`
- **Description**: Returns detailed operational status for each pipeline model adapter.

---

### 2.3 Upload Audio / Video Recording
- **Method**: `POST`
- **Routes**: `/api/v1/audio/upload`, `/api/translate-audio`
- **Content-Type**: `multipart/form-data`
- **Description**: Uploads an audio or video file, resamples to 16kHz mono WAV, and runs the complete ASR $\rightarrow$ SER $\rightarrow$ NMT $\rightarrow$ Gating $\rightarrow$ TTS pipeline.
- **Request Parameters**:
  - `file` (Binary, Required): Audio or video file (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, `.webm`, `.mp4`). Max 25 MB.
  - `source_lang` (String, Optional): Source language (`hi`, `en`, or null for auto-detect).
  - `target_lang` (String, Optional): Target language (`hi` or `en`).
  - `synthesize_speech` (Boolean, Optional): Default `true`.
  - `consent_voice_cloning` (Boolean, Optional): Default `false`.

#### Response Body Example (`200 OK`):
```json
{
  "success": true,
  "source_language": "hi",
  "target_language": "en",
  "source_transcript": "नमस्ते, आप कैसे हैं?",
  "translated_text": "Hello, how are you?",
  "asr_confidence": 0.945,
  "detected_emotion": "happy",
  "emotion_probabilities": {
    "neutral": 0.08,
    "happy": 0.72,
    "sad": 0.02,
    "angry": 0.05,
    "fearful": 0.03,
    "surprised": 0.10
  },
  "valence": 0.68,
  "arousal": 0.65,
  "dominance": 0.58,
  "gating_weight_z": 0.612,
  "preservation_plan": {
    "emotion": "happy",
    "pitch_multiplier": 1.18,
    "rate_multiplier": 1.10,
    "volume_multiplier": 1.10,
    "gating_weight_z": 0.612,
    "applied_to_tts": true,
    "prosody_notes": "Elevated pitch contour with upbeat speaking cadence",
    "expressive_synthesis_supported": false,
    "emotional_fidelity_status": "Heuristic Approximation Only",
    "speaker_similarity_retained": false
  },
  "audio_url": "/api/audio/tts_7d9e4a3b.wav",
  "audio_base64": "UklGRi...",
  "latency": {
    "asr_ms": 482.3,
    "ser_ms": 45.1,
    "nmt_ms": 112.4,
    "tts_ms": 195.8,
    "total_pipeline_ms": 864.1
  },
  "speech_detected": true
}
```

---

### 2.4 Direct Text Translation
- **Method**: `POST`
- **Routes**: `/api/v1/translate/text`, `/api/text-translate`
- **Content-Type**: `application/json`

---

### 2.5 Empirical Evaluation Report
- **Method**: `GET`
- **Route**: `/api/v1/evaluation/report`
- **Description**: Returns empirical scientific benchmark results calculated from local model runs (WER, sentence BLEU, emotion retention, acoustic SNR/MOS proxy, latency breakdown, memory RSS).

---

### 2.6 Stream Generated Audio File
- **Method**: `GET`
- **Route**: `/api/audio/{filename}`
- **Response**: `audio/wav` stream
- **Security**: Validates that resolved file path is strictly within `temp_storage/` via `validate_safe_path` to prevent directory traversal.

---

## 3. WebSocket Live Protocol (`/ws/live-translate`)

### 3.1 Handshake & Session Protocol
- Client initiates standard WebSocket connection to `/ws/live-translate` or `/api/v1/ws/live-translate`.
- Handshake initiation message:
  ```json
  {
    "action": "session_start",
    "session_id": "sess-live-001",
    "source_lang": "en",
    "target_lang": "hi",
    "user_consent": false
  }
  ```
- Server Acknowledgement:
  ```json
  {
    "type": "session_acknowledged",
    "session_id": "sess-live-001",
    "source_lang": "en",
    "target_lang": "hi",
    "status": "ready"
  }
  ```

### 3.2 Client $\rightarrow$ Server Messages

1. **Binary PCM Audio Chunks**:
   - Format: Raw 16-bit signed integer linear PCM (`Int16Array`), 1-channel, 16,000 Hz.
   - Frequency: Streamed every 128ms to 256ms from the browser.

2. **Control Actions**:
   - **Heartbeat**: `{"action": "ping"}`
   - **Cancel in-flight speech**: `{"action": "cancel"}`
   - **Flush speech boundary**: `{"action": "flush"}`
   - **Stop session**: `{"action": "stop"}`

### 3.3 Server $\rightarrow$ Client Messages

1. **Partial Transcript Hypothesis**:
   ```json
   {
     "type": "partial_transcript",
     "session_id": "sess-live-001",
     "segment_id": 1,
     "text": "Hello how are you"
   }
   ```

2. **Finalized Utterance Translation Chunk**:
   ```json
   {
     "type": "translation_chunk",
     "session_id": "sess-live-001",
     "segment_id": 1,
     "source_transcript": "Hello, how are you today?",
     "translated_text": "नमस्ते, आप आज कैसे हैं?",
     "source_lang": "en",
     "target_lang": "hi",
     "detected_emotion": "happy",
     "confidence": 0.94,
     "audio_base64": "UklGRi...",
     "latency": {
       "asr_ms": 232.6,
       "ser_ms": 50.8,
       "nmt_ms": 221.6,
       "tts_ms": 732.6,
       "total_pipeline_ms": 1237.6
     }
   }
   ```

3. **Utterance Cancelled Event**:
   ```json
   {
     "type": "cancelled",
     "session_id": "sess-live-001",
     "segment_id": 1,
     "message": "Current in-flight utterance cancelled."
   }
   ```

4. **Session Closed Event**:
   ```json
   {
     "type": "session_closed",
     "session_id": "sess-live-001",
     "message": "Live translation session closed successfully."
   }
   ```

5. **Error Event**:
   ```json
   {
     "type": "error",
     "error_code": "UNKNOWN_ACTION",
     "message": "Action 'invalid' is unrecognized.",
     "recoverable": true
   }
   ```
