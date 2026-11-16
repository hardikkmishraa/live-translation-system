# API Configuration & Environment Audit Guide

**Project**: Bilingual Voice Translation System With Emotion Awareness  
**Audit Type**: Inspection-Only API Key & Environment Configuration Audit  
**Date**: November 2026  
**Auditor**: Antigravity AI  

---

## 1. Executive Summary

This document provides a comprehensive audit of all external service integrations, API credentials, and environment configuration parameters within the **Bilingual Voice Translation System With Emotion Awareness**.

### Key Findings
1. **Single Cloud Credential Requirement**: Only **one** external API key is utilized across the entire platform: `SARVAM_API_KEY`.
2. **Key Reuse Across Multiple Pipelines**: The `SARVAM_API_KEY` is shared by two distinct cloud endpoints:
   - **Machine Translation**: Sarvam AI Mayura Translation API (`https://api.sarvam.ai/translate`)
   - **Text-to-Speech Synthesis**: Sarvam AI Bulbul TTS API (`https://api.sarvam.ai/text-to-speech`)
3. **Zero-Key Offline Capable**: The application is architected to function entirely **without any API keys or internet connection** if configured to use local/offline models:
   - **ASR**: Whisper-Tiny runs locally via PyTorch / Metal Performance Shaders (`mps`).
   - **SER**: Dedicated IEMOCAP 4-class CNN-BiLSTM model runs locally via PyTorch (`models/ser_iemocap/`).
   - **NMT**: Helsinki-NLP MarianMT (`Helsinki-NLP/opus-mt-hi-en` / `opus-mt-en-hi`) runs locally when `NMT_BACKEND=local`.
   - **TTS**: Microsoft Edge-TTS (free online WebSocket, zero API key) or macOS native system synthesizer (`/usr/bin/say` with `Lekha`/`Samantha`).
4. **Current Status in `.env`**:
   - `SARVAM_API_KEY` is **currently configured** in `.env` (Valid 36-character key present).
5. **`.env.example` Discrepancy**:
   - The `.env.example` template file is out of date: it is missing `SARVAM_API_KEY`, `NMT_BACKEND`, `SARVAM_TTS_MODEL`, and modern voice selection parameters.

---

## 2. Configuration Files Inventory

| Configuration File | Path | Status | Purpose |
| :--- | :--- | :---: | :--- |
| **`.env`** | `/.env` | Active (Ignored in Git) | Primary runtime environment file loaded by `dotenv` in `config.py`. |
| **`.env.example`** | `/.env.example` | Committed in Git | Template reference for developers copying to `.env`. |
| **`config.py`** | `/backend/app/config.py` | Core Source Code | Central configuration loader reading variables via `os.getenv()`. |
| **`pipeline.py`** | `/backend/app/services/pipeline.py` | Core Service | Reads `CONCURRENT_ASR_SER` environment flag. |
| **`requirements.txt`**| `/requirements.txt` | Dependency Spec | Specifies 19 core Python package requirements. |
| **`pytest.ini`** | `/pytest.ini` | Test Runner Config | Specifies pytest root options, asyncio loop scope, and warning filters. |
| **`label_mapping.json`** | `/data/iemocap/label_mapping.json` | Dataset Asset | Defines IEMOCAP 4-class SER mapping (`angry`, `happy`, `sad`, `neutral`). |

---

## 3. External Services & API Configuration Table

The table below catalogs every external service and model provider present in the codebase, with verified file references:

| Service Name | Purpose / Pipeline Role | Environment Variable | File Where Read | File Where Used | API Key Required? | Key Reused? | Currently Configured? |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Sarvam AI Translation API** | Hindi $\leftrightarrow$ English neural machine translation (`mayura:v1`) | `SARVAM_API_KEY`<br>`NMT_BACKEND` | `backend/app/config.py`<br>(lines 42, 46) | `backend/app/models/nmt.py`<br>`backend/app/services/sarvam_translation.py`<br>`backend/app/services/model_lifecycle.py` | **Yes**<br>(Header `api-subscription-key`) | Yes (shared with Sarvam TTS) | **Yes**<br>(Set in `.env`) |
| **Sarvam AI Text-to-Speech API** | Neural Indic expressive speech synthesis (`bulbul:v4-flash` / `bulbul:v3`) | `SARVAM_API_KEY`<br>`TTS_ENGINE`<br>`SARVAM_TTS_MODEL` | `backend/app/config.py`<br>(lines 46, 53, 54) | `backend/app/models/tts_sarvam.py`<br>`backend/app/models/tts.py` | **Yes**<br>(Header `api-subscription-key`) | Yes (shared with Sarvam NMT) | **Yes**<br>(Set in `.env`) |
| **Microsoft Edge-TTS** | Secondary zero-cost high-fidelity neural speech synthesis (`Swara`, `Madhur`, `Jenny`, `Guy`) | `EDGE_TTS_VOICE_HI_*`<br>`EDGE_TTS_VOICE_EN_*` | `backend/app/config.py`<br>(lines 62–65) | `backend/app/models/tts_edge.py` | **No**<br>(Free public WebSocket API) | N/A | **Yes**<br>(Defaults in `config.py`) |
| **macOS Native Speech (`say`)** | Tertiary offline local speech fallback (`Lekha`, `Samantha`, `Rishi`, `Daniel`) | `TTS_ENGINE` | `backend/app/config.py`<br>(lines 53, 68–71) | `backend/app/models/tts_macos.py` | **No**<br>(Local OS binary `/usr/bin/say`) | N/A | **Yes**<br>(Offline available on macOS) |
| **OpenAI Whisper ASR** | Local speech recognition / transcription (`openai/whisper-tiny`) | `ASR_MODEL_ID`<br>`DEVICE` | `backend/app/config.py`<br>(lines 16, 34) | `backend/app/models/asr.py`<br>`backend/app/services/model_lifecycle.py` | **No**<br>(Downloaded from Hugging Face Hub) | N/A | **Yes**<br>(Cached in `~/.cache/huggingface`) |
| **IEMOCAP 4-Class SER** | Primary speech emotion recognition (`angry`, `happy`, `sad`, `neutral`) | `SER_MODEL_PROVIDER`<br>`SER_IEMOCAP_CHECKPOINT_DIR` | `backend/app/config.py`<br>(lines 36, 37) | `backend/app/models/ser_iemocap.py` | **No**<br>(Local custom checkpoint) | N/A | **Yes**<br>(Present in `models/ser_iemocap/`) |
| **Helsinki-NLP MarianMT** | Local offline translation fallback (`opus-mt-hi-en` & `opus-mt-en-hi`) | `NMT_BACKEND`<br>`NMT_HI_TO_EN_MODEL`<br>`NMT_EN_TO_HI_MODEL` | `backend/app/config.py`<br>(lines 42, 49, 50) | `backend/app/models/nmt.py`<br>`backend/app/services/model_lifecycle.py` | **No**<br>(Downloaded from Hugging Face Hub) | N/A | **Yes**<br>(Configured when `NMT_BACKEND=local`) |
| **Wav2Vec2-XLSR SER (RAVDESS)** | Alternative 8-class emotion classifier (`ehcalabres/...`) | `SER_MODEL_ID`<br>`SER_RAVDESS_MODEL_ID` | `backend/app/config.py`<br>(lines 35, 38) | `backend/app/models/ser.py` | **No**<br>(Downloaded from Hugging Face Hub) | N/A | **Yes**<br>(Legacy fallback option) |

---

## 4. Environment Variables Audit: `.env` vs `.env.example` vs `config.py`

### 4.1 Variables in `.env` That Are MISSING in `.env.example`
The following 11 variables are defined in the active `.env` file but omitted from `.env.example`:

1. **`SARVAM_API_KEY`** *(CRITICAL)*: The subscription key for Sarvam AI cloud NMT & TTS.
2. **`NMT_BACKEND`**: Selects between cloud (`sarvam`) and offline local (`local`).
3. **`SARVAM_TTS_MODEL`**: Target Sarvam model (`bulbul:v4-flash`).
4. **`TTS_VOICE_HI_FEMALE`**: Hindi female persona (e.g., `kavya_hi_conversational`).
5. **`TTS_VOICE_HI_MALE`**: Hindi male persona (e.g., `aditya_hi_conversational`).
6. **`TTS_VOICE_EN_FEMALE`**: English female persona (e.g., `amelia_en_conversational`).
7. **`TTS_VOICE_EN_MALE`**: English male persona (e.g., `dev_en_conversational`).
8. **`MACOS_VOICE_HI_FEMALE`**: macOS voice (e.g., `Lekha`).
9. **`MACOS_VOICE_HI_MALE`**: macOS voice (e.g., `Rishi`).
10. **`MACOS_VOICE_EN_FEMALE`**: macOS voice (e.g., `Samantha`).
11. **`MACOS_VOICE_EN_MALE`**: macOS voice (e.g., `Daniel`).

### 4.2 Variables Present in `.env` but UNUSED by the Code
The following variables are defined in `.env` but are **not consumed** by `backend/app/config.py` or the application runtime:

1. **`ASR_DEVICE` & `SER_DEVICE`**:
   - *Reason*: Legacy per-model device flags. Device allocation is now unified under `DEVICE` (`mps`, `cuda`, or `cpu`) via `get_optimal_device()` in `config.py`.
2. **`DEBUG` & `ENVIRONMENT`**:
   - *Reason*: Not read by `config.py` or `main.py`. The `/api/v1/health` endpoint currently hardcodes `environment="development"`.
3. **`HOST` & `PORT`**:
   - *Reason*: Not loaded by `config.py`. These parameters are passed directly via the Uvicorn command-line interface (`uvicorn backend.app.main:app --host ... --port ...`).
4. **`MACOS_VOICE_HI_FEMALE`, `MACOS_VOICE_HI_MALE`, `MACOS_VOICE_EN_FEMALE`, `MACOS_VOICE_EN_MALE`**:
   - *Reason*: In `backend/app/config.py` (lines 68–71), these values are defined as literal constant strings (`"Lekha"`, `"Rishi"`, `"Samantha"`, `"Daniel"`) rather than wrapped in `os.getenv()`.

### 4.3 Supported Optional Variables Missing from `.env`
The application supports these variables in `config.py` and `pipeline.py`, which use built-in defaults when omitted from `.env`:

| Variable | Default Value | Purpose |
| :--- | :--- | :--- |
| **`DEVICE`** | Auto-detected (`mps` $\rightarrow$ `cuda` $\rightarrow$ `cpu`) | Force PyTorch device override (`cpu`, `mps`, or `cuda`). |
| **`SER_MODEL_PROVIDER`** | `"iemocap"` | Select emotion model: `"iemocap"` (4-class) or `"ehcalabres"` (8-class). |
| **`SER_IEMOCAP_CHECKPOINT_DIR`**| `"models/ser_iemocap"` | Custom filesystem path for IEMOCAP model checkpoint. |
| **`TTS_CONFIDENCE_THRESHOLD`** | `0.50` | Minimum emotion confidence required for full prosody modulation. |
| **`TTS_MIN_CONFIDENCE_THRESHOLD`**| `0.35` | Threshold below which emotion is strictly clamped to `neutral`. |
| **`EDGE_TTS_VOICE_HI_FEMALE`** | `"hi-IN-SwaraNeural"` | Default Edge-TTS female voice for Hindi. |
| **`EDGE_TTS_VOICE_HI_MALE`** | `"hi-IN-MadhurNeural"` | Default Edge-TTS male voice for Hindi. |
| **`EDGE_TTS_VOICE_EN_FEMALE`** | `"en-US-JennyNeural"` | Default Edge-TTS female voice for English. |
| **`EDGE_TTS_VOICE_EN_MALE`** | `"en-US-GuyNeural"` | Default Edge-TTS male voice for English. |
| **`CONCURRENT_ASR_SER`** | `"false"` | Flag in `pipeline.py` enabling concurrent thread-pool execution for ASR & SER. |

---

## 5. Official Provider Links & Credential Acquisition

### 5.1 Sarvam AI (`SARVAM_API_KEY`)
- **Required For**:
  - Cloud NMT translation (`mayura:v1`) when `NMT_BACKEND=sarvam`
  - Cloud TTS speech synthesis (`bulbul:v4-flash`) when `TTS_ENGINE=sarvam`
- **Official Portal**: [https://www.sarvam.ai](https://www.sarvam.ai)
- **Developer Documentation**: [https://docs.sarvam.ai](https://docs.sarvam.ai)
- **Dashboard / API Keys**: [https://dashboard.sarvam.ai](https://dashboard.sarvam.ai)
- **Billing Tier**: Free trial credits upon signup; pay-as-you-go thereafter.
- **Header Format**: `api-subscription-key: <YOUR_KEY>`
- **Format**: 36-character hexadecimal string or alphanumeric UUID format.

### 5.2 Hugging Face Hub (`HF_TOKEN` — Optional)
- **Required For**: **Not strictly required.** Public model downloads (`openai/whisper-tiny`, `Helsinki-NLP/opus-mt-*`) do not require authentication.
- **When Needed**: Only if experiencing IP rate limits during automated CI/CD pipeline builds.
- **Official Portal**: [https://huggingface.co](https://huggingface.co)
- **Token Generation**: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Read-only token is sufficient).

### 5.3 Microsoft Edge-TTS (Zero Key)
- **Required For**: High-fidelity neural TTS fallback.
- **Official Documentation**: [https://github.com/rany2/edge-tts](https://github.com/rany2/edge-tts)
- **Authentication**: **None.** Uses public Microsoft Edge Read Aloud WebSocket service.

---

## 6. Beginner's Step-by-Step Configuration Guide

Follow these instructions to configure the application from scratch on any new machine.

### Step 1: Copy the Environment Template
Create your local `.env` file from the repository root:
```bash
cp .env.example .env
```

### Step 2: Choose Your Operating Mode

#### Mode A: Cloud Accelerated (Recommended — Best Quality)
Requires an active Sarvam AI API key.
Add or verify the following lines in your `.env`:
```ini
# Core Sarvam Cloud Configuration
NMT_BACKEND=sarvam
SARVAM_API_KEY=your_actual_sarvam_api_key_here

# Speech Synthesis Engine
TTS_ENGINE=sarvam
SARVAM_TTS_MODEL=bulbul:v4-flash

# Emotion Recognition Engine (IEMOCAP 4-Class)
SER_MODEL_PROVIDER=iemocap

# Hardware Device Selection (auto, mps, cuda, cpu)
DEVICE=auto
```

#### Mode B: 100% Offline / Zero-Cost (No API Keys Required)
Completely offline, air-gapped, zero external charges:
```ini
# Completely Offline Translation via MarianMT
NMT_BACKEND=local

# Completely Offline Speech Synthesis via Edge-TTS or macOS
# On macOS:
TTS_ENGINE=macos_native
# On Linux / Windows:
TTS_ENGINE=edge_tts

# Emotion Recognition Engine (IEMOCAP 4-Class)
SER_MODEL_PROVIDER=iemocap

# Leave API key empty
SARVAM_API_KEY=
```

### Step 3: Verify Configuration Without Starting Server
Run this verification one-liner in your terminal:
```bash
python3 -c "
from backend.app.config import NMT_BACKEND, TTS_ENGINE, SARVAM_API_KEY, DEVICE, SER_MODEL_PROVIDER
print('--- Configuration Verification ---')
print(f'Device Acceleration : {DEVICE}')
print(f'Translation Backend : {NMT_BACKEND}')
print(f'TTS Engine          : {TTS_ENGINE}')
print(f'Emotion Provider    : {SER_MODEL_PROVIDER}')
print(f'Sarvam Key Present  : {bool(SARVAM_API_KEY and not SARVAM_API_KEY.startswith(\"your_\"))}')
"
```

### Step 4: Run the Application
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Navigate to `http://127.0.0.1:8000` to access the live translation studio.

---

## 7. Security Best Practices & Secret Hygiene

> [!CAUTION]
> **Never commit your `.env` file to version control.**
> The `.env` file is listed in `.gitignore`. Ensure you never force-add (`git add -f .env`) or log raw API keys to stdout or disk.

1. **Client-Side Isolation**: The `SARVAM_API_KEY` is loaded exclusively inside the server process. It is **never** sent to the browser or embedded in client-side JavaScript.
2. **Log Redaction**: The system's `PrivacyPreservingLoggingMiddleware` and `backend/app/services/sarvam_translation.py` automatically redact API keys matching `sk_...` or matching the configured key value with `[REDACTED_API_KEY]`.
3. **Key Masking on Health Endpoints**: Neither `/api/v1/health` nor `/api/v1/models/status` returns the key value; they only return a boolean flag indicating whether the key is configured.
