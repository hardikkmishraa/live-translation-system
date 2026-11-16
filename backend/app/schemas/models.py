from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class ModelStatusEnum(str, Enum):
    AVAILABLE_AND_READY = "available_and_ready"
    NOT_INSTALLED = "not_installed"
    STILL_LOADING = "still_loading"
    FAILED_TO_LOAD = "failed_to_load"
    UNSUPPORTED_FOR_FEATURE = "unsupported_for_feature"

class ModelCapabilityInfo(BaseModel):
    model_config = {"protected_namespaces": ()}

    module: str = Field(..., description="Cognitive module identifier (ASR, SER, NMT, TTS, GATING)")
    model_identifier: str = Field(..., description="Model ID or checkpoint name")
    status: ModelStatusEnum = Field(..., description="Granular readiness state")
    is_loaded: bool = Field(..., description="Whether weights are cached in memory")
    device: str = Field(..., description="Target compute device")
    supported_languages: List[str] = Field(..., description="Languages genuinely supported by this module")
    scientific_basis: str = Field(..., description="Source research paper equation or citation mapping")
    status_message: str = Field(..., description="Truthful explanation of state")
    failure_reason: Optional[str] = Field(None, description="Diagnostic error details if failed")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ModelsStatusResponse(BaseModel):
    modules: Dict[str, ModelCapabilityInfo] = Field(..., description="Map of all cognitive pipeline modules")
    all_ready: bool = Field(
        ...,
        description="True only when required modules are available, ASR/SER weights are loaded, NMT is configured/loaded, and TTS is supported."
    )
    timestamp: float = Field(..., description="Timestamp of status evaluation")
