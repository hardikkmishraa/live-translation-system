import re
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

def test_index_html_served():
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    # Verify key sections
    assert "VoiceAffect" in html
    assert "sectionLanding" in html
    assert "sectionWorkspace" in html
    assert "sectionResearch" in html
    assert "liveVisualizerCanvas" in html

def test_static_css_assets_served():
    for css_file in ["style.css", "design-system.css", "components.css", "layout.css"]:
        res = client.get(f"/static/css/{css_file}")
        assert res.status_code == 200
        assert len(res.text) > 50

def test_static_js_modules_served():
    for js_file in ["main.js", "api.js", "websocket.js", "audio.js", "results.js", "research.js"]:
        res = client.get(f"/static/js/{js_file}")
        assert res.status_code == 200
        assert len(res.text) > 50

def test_html_accessibility_and_controls():
    index_path = FRONTEND_DIR / "index.html"
    content = index_path.read_text(encoding="utf-8")
    
    # Check accessibility elements
    assert 'lang="en"' in content
    assert 'aria-label' in content
    assert 'aria-selected' in content
    assert 'role="tab"' in content
    
    # Check all key interactive button IDs exist in HTML
    expected_ids = [
      "btnHeroStartLive",
      "btnHeroUpload",
      "btnHeroResearch",
      "tabWorkspaceLive",
      "tabWorkspaceUpload",
      "selectLanguageDirection",
      "btnSwapLanguage",
      "btnStartLiveSpeech",
      "btnPauseLiveSpeech",
      "btnStopLiveSpeech",
      "btnFlushLiveSpeech",
      "uploadDropzone",
      "audioFileInput",
      "btnProcessUploadedAudio",
      "btnDownloadAudio",
      "btnLoadDemoSample"
    ]
    for btn_id in expected_ids:
        assert f'id="{btn_id}"' in content, f"Missing control id: {btn_id}"
