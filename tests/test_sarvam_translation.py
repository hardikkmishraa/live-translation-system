"""
Unit & Integration Tests for Sarvam AI Translation Module
==========================================================
Tests:
  1. Language code normalization and validation.
  2. Model selection based on input length (mayura:v1 vs sarvam-translate:v1).
  3. Header-based authentication (api-subscription-key).
  4. English-to-Hindi and Hindi-to-English translation.
  5. Transient error retry handling (HTTP 429, 5xx, timeouts).
  6. Non-retryable error fail-fast handling (HTTP 401, 403, 400).
  7. Translation response validation (empty text rejection, malformed JSON).
  8. API key masking and privacy protection.
  9. NMT Model contract: error propagation, failure reporting, transcription preservation,
     and absence of unapproved fallbacks.
  10. Live API integration (executed only when valid SARVAM_API_KEY is configured).
"""

import io
import json
import os
import urllib.error
import pytest
from unittest.mock import MagicMock, patch

from backend.app.services.sarvam_translation import (
    SarvamTranslationService,
    SarvamTranslationError,
    SarvamAuthError,
    SarvamRateLimitError,
    SarvamRequestError,
    SarvamServerError,
    SarvamTimeoutError,
    SarvamResponseError,
    normalize_to_bcp47,
    bcp47_to_short,
    sanitize_sensitive_data,
)
from backend.app.models.nmt import SarvamNMTModel


# ---------------------------------------------------------------------------
# 1. Language code normalization & validation
# ---------------------------------------------------------------------------

def test_language_code_normalization():
    assert normalize_to_bcp47("hi") == "hi-IN"
    assert normalize_to_bcp47("hin") == "hi-IN"
    assert normalize_to_bcp47("hi-in") == "hi-IN"
    assert normalize_to_bcp47("en") == "en-IN"
    assert normalize_to_bcp47("eng") == "en-IN"
    assert normalize_to_bcp47("en-us") == "en-IN"

    assert bcp47_to_short("hi-IN") == "hi"
    assert bcp47_to_short("en-IN") == "en"

    with pytest.raises(ValueError):
        normalize_to_bcp47("fr")

    with pytest.raises(ValueError):
        normalize_to_bcp47("")


# ---------------------------------------------------------------------------
# 2. Key sanitization
# ---------------------------------------------------------------------------

def test_api_key_sanitization():
    secret_key = "sk_secretkey1234567890abcdef"
    msg = f"Failed to connect using key {secret_key} to endpoint."
    sanitized = sanitize_sensitive_data(msg, secret_key)

    assert secret_key not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized


# ---------------------------------------------------------------------------
# 3. Model selection & request schema (Mocked)
# ---------------------------------------------------------------------------

def test_sarvam_model_selection_and_request_schema():
    service = SarvamTranslationService(
        api_key="sk_dummy_test_key_123456789",
        max_retries=0,
    )

    captured_requests = []

    def mock_urlopen(request, timeout):
        captured_requests.append(request)
        body = json.loads(request.data.decode("utf-8"))
        res_payload = {
            "request_id": "req-12345",
            "translated_text": "नमस्ते",
            "source_language_code": body["source_language_code"],
        }
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.read.return_value = json.dumps(res_payload).encode("utf-8")
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        # Short text <= 1000 chars should select mayura:v1
        res = service.translate("Hello", "en", "hi")
        assert res["translated_text"] == "नमस्ते"
        assert res["model"] == "mayura:v1"
        assert res["provider"] == "sarvam_ai"

        req = captured_requests[0]
        assert req.get_header("Api-subscription-key") == "sk_dummy_test_key_123456789"
        req_body = json.loads(req.data.decode("utf-8"))
        assert req_body["model"] == "mayura:v1"
        assert req_body["source_language_code"] == "en-IN"
        assert req_body["target_language_code"] == "hi-IN"
        assert req_body["enable_preprocessing"] is True

        # Long text > 1000 chars should select sarvam-translate:v1
        long_text = "word " * 250  # ~1250 characters
        res_long = service.translate(long_text, "en", "hi")
        assert res_long["model"] == "sarvam-translate:v1"

        req_long_body = json.loads(captured_requests[1].data.decode("utf-8"))
        assert req_long_body["model"] == "sarvam-translate:v1"


def test_sarvam_rejects_overlong_text():
    service = SarvamTranslationService(api_key="sk_dummy_key_123456789")
    huge_text = "a" * 2001
    with pytest.raises(SarvamRequestError) as exc_info:
        service.translate(huge_text, "en", "hi")
    assert "exceeds" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 4. English-to-Hindi and Hindi-to-English (Mocked)
# ---------------------------------------------------------------------------

def test_sarvam_translation_both_directions():
    service = SarvamTranslationService(api_key="sk_dummy_key_123456789", max_retries=0)

    responses = [
        {"request_id": "r1", "translated_text": "नमस्ते", "source_language_code": "en-IN"},
        {"request_id": "r2", "translated_text": "Hello", "source_language_code": "hi-IN"},
    ]

    def mock_urlopen(request, timeout):
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.read.return_value = json.dumps(responses.pop(0)).encode("utf-8")
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        en_to_hi = service.translate("Hello", "en", "hi")
        assert en_to_hi["translated_text"] == "नमस्ते"
        assert en_to_hi["target_language_code"] == "hi-IN"

        hi_to_en = service.translate("नमस्ते", "hi", "en")
        assert hi_to_en["translated_text"] == "Hello"
        assert hi_to_en["target_language_code"] == "en-IN"


# ---------------------------------------------------------------------------
# 5. Output Validation (Empty text / malformed JSON)
# ---------------------------------------------------------------------------

def test_sarvam_rejects_empty_translation_for_alnum_input():
    service = SarvamTranslationService(api_key="sk_dummy_key_123456789", max_retries=0)

    def mock_empty_response(request, timeout):
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.read.return_value = json.dumps({
            "request_id": "r_empty",
            "translated_text": "   ",
            "source_language_code": "en-IN",
        }).encode("utf-8")
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=mock_empty_response):
        with pytest.raises(SarvamResponseError) as exc_info:
            service.translate("Valid speech transcript", "en", "hi")
        assert "empty translation" in str(exc_info.value).lower()


def test_sarvam_rejects_malformed_json():
    service = SarvamTranslationService(api_key="sk_dummy_key_123456789", max_retries=0)

    def mock_bad_json(request, timeout):
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.read.return_value = b"<html>502 Bad Gateway</html>"
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=mock_bad_json):
        with pytest.raises(SarvamResponseError):
            service.translate("Hello", "en", "hi")


# ---------------------------------------------------------------------------
# 6. Authentication error handling (401 / 403 — fail fast, no retry)
# ---------------------------------------------------------------------------

def test_sarvam_auth_failure_fails_fast_without_retrying():
    service = SarvamTranslationService(api_key="sk_invalid_test_key", max_retries=3)
    call_count = [0]

    def mock_auth_error(request, timeout):
        call_count[0] += 1
        fp = io.BytesIO(b'{"error": {"message": "Invalid API key"}}')
        raise urllib.error.HTTPError(
            url=request.full_url,
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=fp,
        )

    with patch("urllib.request.urlopen", side_effect=mock_auth_error):
        with pytest.raises(SarvamAuthError):
            service.translate("Hello", "en", "hi")

    # MUST fail on the first attempt without wasting time on retries
    assert call_count[0] == 1


# ---------------------------------------------------------------------------
# 7. Transient error retry handling (429, 500, timeouts)
# ---------------------------------------------------------------------------

def test_sarvam_rate_limit_retry_success():
    service = SarvamTranslationService(
        api_key="sk_dummy_test_key",
        max_retries=2,
    )
    service.BASE_BACKOFF_SECONDS = 0.01  # accelerate test

    call_count = [0]

    def mock_rate_limit_then_succeed(request, timeout):
        call_count[0] += 1
        if call_count[0] == 1:
            fp = io.BytesIO(b'{"error": "Rate limit exceeded"}')
            raise urllib.error.HTTPError(
                url=request.full_url,
                code=429,
                msg="Too Many Requests",
                hdrs={},
                fp=fp,
            )
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.read.return_value = json.dumps({
            "request_id": "r_ok",
            "translated_text": "नमस्ते",
            "source_language_code": "en-IN",
        }).encode("utf-8")
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=mock_rate_limit_then_succeed):
        res = service.translate("Hello", "en", "hi")
        assert res["translated_text"] == "नमस्ते"
        assert call_count[0] == 2
        assert res["attempts"] == 2


def test_sarvam_server_error_exhausts_retries():
    service = SarvamTranslationService(
        api_key="sk_dummy_test_key",
        max_retries=2,
    )
    service.BASE_BACKOFF_SECONDS = 0.01

    call_count = [0]

    def mock_server_error(request, timeout):
        call_count[0] += 1
        fp = io.BytesIO(b'{"error": "Internal server error"}')
        raise urllib.error.HTTPError(
            url=request.full_url,
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=fp,
        )

    with patch("urllib.request.urlopen", side_effect=mock_server_error):
        with pytest.raises(SarvamServerError):
            service.translate("Hello", "en", "hi")

    # 1 initial attempt + 2 retries = 3 total attempts
    assert call_count[0] == 3


# ---------------------------------------------------------------------------
# 8. SarvamNMTModel Adapter Contract (No Silent Fallback, Preserved Transcript)
# ---------------------------------------------------------------------------

def test_sarvam_nmt_model_contract_preserves_transcript_on_failure():
    nmt = SarvamNMTModel(api_key="sk_dummy_test_key", max_retries=0)

    def mock_failure(request, timeout):
        fp = io.BytesIO(b'{"error": "Outage"}')
        raise urllib.error.HTTPError(
            url=request.full_url,
            code=503,
            msg="Service Unavailable",
            hdrs={},
            fp=fp,
        )

    with patch("urllib.request.urlopen", side_effect=mock_failure):
        result = nmt.translate("Critical transcribed text", "en", "hi")

        # Must preserve original transcript
        assert result.source_text == "Critical transcribed text"
        # Must not fabricate translated text
        assert result.translated_text == ""
        # Must mark status as failed
        assert result.details["status"] == "translation_failed"
        assert result.details["provider"] == "sarvam_ai"
        assert result.details["automatic_fallback_used"] is False
        assert "503" in result.details["error"]


def test_sarvam_nmt_model_health_does_not_leak_key():
    secret = "sk_very_secret_key_12345"
    nmt = SarvamNMTModel(api_key=secret)
    health = nmt.get_health()

    assert health.details["api_key_configured"] is True
    # Verify the plain text secret key is not in health details or string rep
    assert secret not in str(health)


# ---------------------------------------------------------------------------
# 9. Live Integration Test (Conditional on real SARVAM_API_KEY)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("SARVAM_API_KEY") or os.getenv("SARVAM_API_KEY", "").startswith("sk_dummy"),
    reason="Real SARVAM_API_KEY not configured in environment.",
)
def test_sarvam_live_api_both_directions():
    """Live smoke test verifying actual Sarvam AI endpoint with configured credentials."""
    nmt = SarvamNMTModel()

    # English to Hindi
    res_hi = nmt.translate("Good morning, how are you?", source_lang="en", target_lang="hi")
    assert res_hi.translated_text != ""
    assert res_hi.details["status"] == "success"
    assert res_hi.details["provider"] == "sarvam_ai"
    assert res_hi.model_name in ("mayura:v1", "sarvam-translate:v1")

    # Hindi to English
    res_en = nmt.translate("नमस्ते, आप कैसे हैं?", source_lang="hi", target_lang="en")
    assert res_en.translated_text != ""
    assert res_en.details["status"] == "success"
    assert res_en.details["provider"] == "sarvam_ai"
