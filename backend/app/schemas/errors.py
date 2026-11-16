import time
from typing import Optional, Any
from pydantic import BaseModel, Field

class APIErrorDetail(BaseModel):
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable explanation of failure")
    field: Optional[str] = Field(None, description="Invalid field name if validation error")
    diagnostics: Optional[Any] = Field(None, description="Optional diagnostic details")

class APIErrorResponse(BaseModel):
    success: bool = Field(False, description="Always false for error responses")
    error: APIErrorDetail = Field(..., description="Detailed error information")
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp of error")
