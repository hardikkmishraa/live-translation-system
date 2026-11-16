import time
import threading
from typing import Dict, Any, Optional, Union

from backend.app.config import (
    DEVICE, ASR_MODEL_ID, SER_MODEL_ID, SER_MODEL_PROVIDER,
    SER_IEMOCAP_CHECKPOINT_DIR, SER_HINDI_CHECKPOINT_DIR, SER_RAVDESS_MODEL_ID,
    NMT_HI_TO_EN_MODEL, NMT_EN_TO_HI_MODEL, NMT_BACKEND, SARVAM_API_KEY,
)
from backend.app.schemas.models import ModelStatusEnum, ModelCapabilityInfo, ModelsStatusResponse
from backend.app.models.base import BaseSERModel
from backend.app.models.asr import WhisperASRModel
from backend.app.models.ser import SpeechEmotionRecognizer
from backend.app.models.ser_iemocap import IEMOCAPSpeechEmotionRecognizer
from backend.app.models.ser_hindi import HindiSpeechEmotionRecognizer
from backend.app.models.ser_router import BilingualSERRouter
from backend.app.models.nmt import SarvamNMTModel, BilingualNMTModel
from backend.app.models.emotion_preservation import LatentDepthEmotionPreserver
from backend.app.models.tts import EmotionPreservingTTS
from backend.app.utils.logger import logger


class ModelLifecycleManager:
    """Owns singleton adapters and reports adapter readiness separately from loaded weights."""

    def __init__(self):
        self._lock = threading.RLock()
        self._asr_instance: Optional[WhisperASRModel] = None
        self._ser_provider = SER_MODEL_PROVIDER
        self._ser_instance: Optional[BaseSERModel] = None
        self._nmt_instance: Optional[Union[SarvamNMTModel, BilingualNMTModel]] = None
        self._gating_instance: Optional[LatentDepthEmotionPreserver] = None
        self._tts_instance: Optional[EmotionPreservingTTS] = None

        if self._ser_provider in ("router", "bilingual", "bilingual_router"):
            ser_model_name = "bilingual-ser-router-v1"
            ser_basis = "Bilingual SER: English (IEMOCAP 4-class) + Hindi (Vishal B. 8-class)"
            ser_languages = ["hi (native 8-class)", "en (native 4-class)"]
        elif self._ser_provider == "hindi":
            ser_model_name = "hindi-wav2vec2-xlsr-8class"
            ser_basis = "Wav2Vec2 XLSR-53 (Vishal B. Hindi 8-class fine-tuned)"
            ser_languages = ["hi (native 8-class)"]
        elif self._ser_provider == "iemocap":
            ser_model_name = "iemocap-wav2vec2-xlsr-4class"
            ser_basis = "Wav2Vec2 XLSR-53 (IEMOCAP 4-class fine-tuned)"
            ser_languages = ["en (native 4-class)", "hi (fallback)"]
        else:
            ser_model_name = SER_RAVDESS_MODEL_ID
            ser_basis = "Wav2Vec2 XLSR-53 (RAVDESS English fine-tuned)"
            ser_languages = ["en (native)", "hi (prosodic transfer with documented limits)"]

        self._states: Dict[str, Dict[str, Any]] = {
            "asr": {
                "status": ModelStatusEnum.AVAILABLE_AND_READY,
                "is_loaded": False,
                "model_id": ASR_MODEL_ID,
                "device": DEVICE,
                "supported_languages": ["hi", "en"],
                "scientific_basis": "Whisper Transformer (Section 3, Eq. 3 & 4)",
                "status_message": "ASR adapter available; Whisper weights load on first inference.",
                "failure_reason": None,
            },
            "ser": {
                "status": ModelStatusEnum.AVAILABLE_AND_READY,
                "is_loaded": False,
                "model_id": ser_model_name,
                "provider": self._ser_provider,
                "device": DEVICE,
                "supported_languages": ser_languages,
                "scientific_basis": ser_basis,
                "status_message": f"SER adapter available ({self._ser_provider}); model weights validate on first inference.",
                "failure_reason": None,
            },
            "nmt": {
                "status": ModelStatusEnum.AVAILABLE_AND_READY,
                "is_loaded": NMT_BACKEND == "sarvam",  # Sarvam client is always "ready"; MarianMT weights are lazy
                "model_id": (
                    "sarvam_ai / mayura:v1"
                    if NMT_BACKEND == "sarvam"
                    else f"{NMT_HI_TO_EN_MODEL} / {NMT_EN_TO_HI_MODEL}"
                ),
                "device": "sarvam_cloud" if NMT_BACKEND == "sarvam" else DEVICE,
                "supported_languages": ["hi", "en"],
                "scientific_basis": "Encoder-Decoder Transformer Self-Attention (Section 3, Eq. 6 & 7)",
                "status_message": (
                    "Sarvam AI API client ready. Endpoint: POST https://api.sarvam.ai/translate"
                    if NMT_BACKEND == "sarvam"
                    else "MarianMT adapter available; directional model weights load on first translation."
                ),
                "failure_reason": None,
            },
            "gating": {
                "status": ModelStatusEnum.AVAILABLE_AND_READY,
                "is_loaded": False,
                "model_id": "Heuristic Emotion Prosody Planner",
                "device": "cpu",
                "supported_languages": ["hi", "en"],
                "scientific_basis": "Heuristic prosody calculation (Section 3, Eq. 8)",
                "status_message": "Rule-based prosody planner; no neural gating weights are loaded.",
                "failure_reason": None,
            },
            "tts": {
                "status": ModelStatusEnum.UNSUPPORTED_FOR_FEATURE,
                "is_loaded": False,
                "model_id": "macOS native say speech synthesis",
                "device": "system",
                "supported_languages": ["hi", "en"],
                "scientific_basis": "Native speech synthesis with heuristic prosody controls (Section 3, Eq. 9)",
                "status_message": "TTS availability has not yet been checked.",
                "failure_reason": None,
            },
        }

    def get_asr(self) -> WhisperASRModel:
        with self._lock:
            if self._asr_instance is None:
                self._asr_instance = WhisperASRModel(model_id=ASR_MODEL_ID, device=DEVICE)
            self._states["asr"]["is_loaded"] = bool(self._asr_instance._is_loaded)
            return self._asr_instance

    def get_ser(self, provider: Optional[str] = None) -> BaseSERModel:
        with self._lock:
            if provider is not None:
                self._ser_provider = provider.lower()
            chosen_provider = self._ser_provider.lower()
            chosen_canonical = "router" if chosen_provider in ("router", "bilingual", "bilingual_router") else chosen_provider

            # If switching provider, reset instance
            current_provider = getattr(self._ser_instance, "_provider_name", None)
            current_canonical = "router" if current_provider in ("router", "bilingual", "bilingual_router") else current_provider

            if self._ser_instance is not None and current_canonical != chosen_canonical:
                self._ser_instance = None

            if self._ser_instance is None:
                if chosen_canonical == "router":
                    logger.info(f"[SER Lifecycle] Initializing Bilingual SER Router on device: {DEVICE}")
                    self._ser_instance = BilingualSERRouter(
                        device=DEVICE,
                        iemocap_dir=SER_IEMOCAP_CHECKPOINT_DIR,
                        hindi_dir=SER_HINDI_CHECKPOINT_DIR,
                        preload_at_startup=True
                    )
                    self._ser_instance._provider_name = "router"
                elif chosen_provider == "hindi":
                    config_file = SER_HINDI_CHECKPOINT_DIR / "config.json"
                    weights_file = SER_HINDI_CHECKPOINT_DIR / "best_model.pt"
                    if not (config_file.exists() and weights_file.exists()):
                        err_msg = (
                            f"Configured SER provider 'hindi' checkpoint not found at: {SER_HINDI_CHECKPOINT_DIR}. "
                            "Verify Phase 17 artifacts in models/ser_hindi/."
                        )
                        logger.error(f"[SER Lifecycle] {err_msg}")
                        raise FileNotFoundError(err_msg)

                    logger.info(f"[SER Lifecycle] Initializing Hindi SER adapter from: {SER_HINDI_CHECKPOINT_DIR}")
                    self._ser_instance = HindiSpeechEmotionRecognizer(
                        checkpoint_dir=SER_HINDI_CHECKPOINT_DIR,
                        device=DEVICE
                    )
                    self._ser_instance._provider_name = "hindi"
                elif chosen_provider == "iemocap":
                    config_file = SER_IEMOCAP_CHECKPOINT_DIR / "config.json"
                    weights_file = SER_IEMOCAP_CHECKPOINT_DIR / "pytorch_model.bin"
                    if not (config_file.exists() and weights_file.exists()):
                        err_msg = (
                            f"Configured SER provider 'iemocap' checkpoint not found at: {SER_IEMOCAP_CHECKPOINT_DIR}. "
                            "Verify Phase 11 artifacts in models/ser_iemocap/."
                        )
                        logger.error(f"[SER Lifecycle] {err_msg}")
                        raise FileNotFoundError(err_msg)

                    logger.info(f"[SER Lifecycle] Initializing IEMOCAP SER adapter from: {SER_IEMOCAP_CHECKPOINT_DIR}")
                    self._ser_instance = IEMOCAPSpeechEmotionRecognizer(
                        checkpoint_dir=SER_IEMOCAP_CHECKPOINT_DIR,
                        device=DEVICE
                    )
                    self._ser_instance._provider_name = "iemocap"
                elif chosen_provider == "ravdess":
                    logger.info(f"[SER Lifecycle] Initializing RAVDESS SER adapter with model: {SER_RAVDESS_MODEL_ID}")
                    self._ser_instance = SpeechEmotionRecognizer(
                        model_id=SER_RAVDESS_MODEL_ID,
                        device=DEVICE
                    )
                    self._ser_instance._provider_name = "ravdess"
                else:
                    raise ValueError(f"Unsupported SER provider '{chosen_provider}'. Allowed: 'router', 'hindi', 'iemocap', 'ravdess'.")

            self._states["ser"]["is_loaded"] = bool(getattr(self._ser_instance, "_is_loaded", False))
            self._states["ser"]["model_id"] = getattr(self._ser_instance, "model_name", SER_MODEL_ID)
            return self._ser_instance

    def get_nmt(self) -> Union[SarvamNMTModel, BilingualNMTModel]:
        with self._lock:
            if self._nmt_instance is None:
                if NMT_BACKEND == "sarvam":
                    logger.info("[NMT] Initialising Sarvam AI translation client.")
                    self._nmt_instance = SarvamNMTModel(api_key=SARVAM_API_KEY)
                else:
                    # NMT_BACKEND=local — load MarianMT weights on first translate() call
                    logger.info("[NMT] Initialising local MarianMT (NMT_BACKEND=local).")
                    self._nmt_instance = BilingualNMTModel(
                        hi_to_en_model=NMT_HI_TO_EN_MODEL,
                        en_to_hi_model=NMT_EN_TO_HI_MODEL,
                        device=DEVICE,
                    )
            return self._nmt_instance

    def get_gating(self) -> LatentDepthEmotionPreserver:
        with self._lock:
            if self._gating_instance is None:
                self._gating_instance = LatentDepthEmotionPreserver()
            return self._gating_instance

    def get_tts(self) -> EmotionPreservingTTS:
        with self._lock:
            if self._tts_instance is None:
                self._tts_instance = EmotionPreservingTTS()
            health = self._tts_instance.get_health()
            state = self._states["tts"]
            state["status"] = (
                ModelStatusEnum.AVAILABLE_AND_READY
                if health.is_loaded or self._tts_instance._has_native else ModelStatusEnum.UNSUPPORTED_FOR_FEATURE
            )
            state["is_loaded"] = bool(health.is_loaded)
            state["model_id"] = health.model_name
            state["status_message"] = health.status_message
            state["failure_reason"] = None
            state["details"] = health.details
            return self._tts_instance

    def _refresh_instance_states(self) -> None:
        if self._asr_instance is not None:
            state = self._states["asr"]
            state["is_loaded"] = bool(self._asr_instance._is_loaded)
            error = self._asr_instance._load_error
            if self._asr_instance._is_loading:
                state.update(
                    status=ModelStatusEnum.STILL_LOADING,
                    status_message="Whisper weights are currently loading.",
                    failure_reason=None,
                )
            elif error:
                state.update(
                    status=ModelStatusEnum.FAILED_TO_LOAD,
                    status_message="Whisper failed to load.",
                    failure_reason=error,
                )
            elif self._asr_instance._is_loaded or state["status"] != ModelStatusEnum.FAILED_TO_LOAD:
                state.update(
                    status=ModelStatusEnum.AVAILABLE_AND_READY,
                    status_message=(
                        "Whisper weights loaded and ready."
                        if state["is_loaded"] else
                        "ASR adapter available; Whisper weights load on first inference."
                    ),
                    failure_reason=None,
                )

        if self._ser_instance is not None:
            state = self._states["ser"]
            state["is_loaded"] = bool(self._ser_instance._is_loaded)
            error = self._ser_instance._load_error
            if self._ser_instance._is_loading:
                state.update(
                    status=ModelStatusEnum.STILL_LOADING,
                    status_message="Wav2Vec2 backbone and trained classifier checkpoint are loading.",
                    failure_reason=None,
                )
            elif error:
                state.update(
                    status=ModelStatusEnum.FAILED_TO_LOAD,
                    status_message="SER checkpoint or trained classifier failed validation/loading.",
                    failure_reason=error,
                )
            elif self._ser_instance._is_loaded or state["status"] != ModelStatusEnum.FAILED_TO_LOAD:
                state.update(
                    status=ModelStatusEnum.AVAILABLE_AND_READY,
                    status_message=(
                        "Wav2Vec2 backbone and trained classifier weights loaded."
                        if state["is_loaded"] else
                        "SER adapter available; checkpoint weights load and validate on first inference."
                    ),
                    failure_reason=None,
                )

        if self._nmt_instance is not None:
            state = self._states["nmt"]
            if isinstance(self._nmt_instance, SarvamNMTModel):
                # API client — always ready, no weights to load
                state["is_loaded"] = True
                state["status"] = ModelStatusEnum.AVAILABLE_AND_READY
                state["status_message"] = (
                    "Sarvam AI API client ready. "
                    "Endpoint: POST https://api.sarvam.ai/translate"
                )
                state["failure_reason"] = None
            elif isinstance(self._nmt_instance, BilingualNMTModel):
                model = self._nmt_instance
                loaded_directions = []
                if model._hi_to_en_model is not None:
                    loaded_directions.append("hi_to_en")
                if model._en_to_hi_model is not None:
                    loaded_directions.append("en_to_hi")
                state["is_loaded"] = bool(loaded_directions)
                if model._is_loading:
                    state.update(
                        status=ModelStatusEnum.STILL_LOADING,
                        status_message="MarianMT tokenizer/model weights are loading.",
                        failure_reason=None,
                    )
                elif model._load_error:
                    state.update(
                        status=ModelStatusEnum.FAILED_TO_LOAD,
                        status_message="MarianMT failed to load.",
                        failure_reason=model._load_error,
                    )
                else:
                    state.update(
                        status=ModelStatusEnum.AVAILABLE_AND_READY,
                        status_message=(
                            "MarianMT direction(s) loaded: " + ", ".join(loaded_directions)
                            if loaded_directions else
                            "MarianMT adapter available; direction-specific weights load on first translation."
                        ),
                        failure_reason=None,
                    )
                state["details"] = {"loaded_directions": loaded_directions}

        if self._tts_instance is not None:
            self.get_tts()

    def get_detailed_status(self) -> ModelsStatusResponse:
        """Return live adapter/weight status without claiming lazy weights are already loaded."""
        with self._lock:
            # TTS adapter construction is lightweight and required to determine
            # whether a real speech engine exists on this host.
            if self._tts_instance is None:
                self.get_tts()
            self._refresh_instance_states()
            module_reports = {}
            for key, state in self._states.items():
                module_reports[key] = ModelCapabilityInfo(
                    module=key.upper(),
                    model_identifier=state["model_id"],
                    status=state["status"],
                    is_loaded=state["is_loaded"],
                    device=state["device"],
                    supported_languages=state["supported_languages"],
                    scientific_basis=state["scientific_basis"],
                    status_message=state["status_message"],
                    failure_reason=state["failure_reason"],
                    details=state.get("details", {}),
                )

            nmt_operational = self._nmt_instance is not None
            if isinstance(self._nmt_instance, BilingualNMTModel):
                nmt_operational = (
                    self._nmt_instance._hi_to_en_model is not None
                    and self._nmt_instance._en_to_hi_model is not None
                )

            all_ready = (
                all(
                    module.status == ModelStatusEnum.AVAILABLE_AND_READY
                    for module in module_reports.values()
                )
                and self._asr_instance is not None
                and bool(self._asr_instance._is_loaded)
                and self._ser_instance is not None
                and bool(self._ser_instance._is_loaded)
                and nmt_operational
            )
            return ModelsStatusResponse(
                modules=module_reports,
                all_ready=all_ready,
                timestamp=time.time(),
            )


model_lifecycle = ModelLifecycleManager()
