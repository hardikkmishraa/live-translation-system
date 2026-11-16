import time
from typing import Optional, Any
from pydantic import BaseModel, Field

class BaseResponse(BaseModel):
    success: bool = Field(True, description="Indicates whether the request was successful")
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp of response")

class ErrorDetail(BaseModel):
    code: str = Field(..., description="Machine-readable error identifier")
    message: str = Field(..., description="Human-readable explanation of the error")
    details: Optional[Any] = Field(None, description="Optional diagnostic context")

class StandardErrorResponse(BaseModel):
    success: bool = Field(False, description="Always false for error responses")
    error: ErrorDetail = Field(..., description="Error descriptor")
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp of response")
