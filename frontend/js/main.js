/**
 * VoiceAffect Main Bootstrap & Event Controller
 * Integrates ApiClient, WebSocket, AudioCapture, Results, Research Insights,
 * History drawer, Settings modal, and Premium Voice workspace controls.
 */

import { ApiClient } from './api.js';
import { LiveWebSocketClient } from './websocket.js';
import { AudioCaptureManager, WaveformVisualizer } from './audio.js';
import { ResultsRenderer } from './results.js';
import { ResearchInsightsRenderer } from './research.js';
import { ProgressivePlaybackQueue } from './playback_queue.js';

document.addEventListener('DOMContentLoaded', () => {
  // --- Service Instances ---
  const api = new ApiClient();
  const wsClient = new LiveWebSocketClient();
  const results = new ResultsRenderer();
  const research = new ResearchInsightsRenderer(api);
  research.init();

  // Progressive Audio Playback Queue (Phase 6)
  const toggleAutoPlay = document.getElementById('toggleAutoPlaySpeech');
  const playbackQueue = new ProgressivePlaybackQueue({
    maxQueueSize: 10,
    autoPlay: toggleAutoPlay ? toggleAutoPlay.checked : true,
    onStatusChange: (status) => {
      results.renderPlaybackQueueStatus(status);
    }
  });

  if (toggleAutoPlay) {
    toggleAutoPlay.addEventListener('change', (e) => {
      playbackQueue.setAutoPlay(e.target.checked);
    });
  }

  // Canvas & Visualizer
  const canvasEl = document.getElementById('liveVisualizerCanvas');
  const visualizer = new WaveformVisualizer(canvasEl);

  // Audio Capture with Resampling & Device Disconnection Listener
  let recordedBlob = null;
  const audioManager = new AudioCaptureManager(
    (pcmBuffer) => {
      // Dispatch streaming 16kHz PCM chunk to active WebSocket
      if (wsClient.isConnected) {
        wsClient.sendAudioChunk(pcmBuffer);
        setPipelineStage('recording');
      }
    },
    () => {
      // Graceful device disconnection handler
      const btnStopLive = document.getElementById('btnStopLiveSpeech');
      if (btnStopLive) btnStopLive.click();
      results.showError('Microphone hardware disconnected. Please reconnect your input device.');
    },
    (formattedDuration, totalSeconds) => {
      if (audioManager.isRecording && !audioManager.isPaused) {
        const sub = document.getElementById('voiceStatusSubtitle');
        if (sub) {
          sub.textContent = `${formattedDuration} · Speak now in Hindi`;
        }
      }
    }
  );

  // --- Translation History Storage (Session) ---
  const historyListEl = document.getElementById('historyList');
  let translationHistory = [];

  function addHistoryItem(sourceText, translatedText, emotion = 'neutral', timestamp = new Date()) {
    if (!sourceText && !translatedText) return;
    const item = {
      id: Date.now(),
      source: sourceText,
      translated: translatedText,
      emotion: emotion || 'neutral',
      time: timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    translationHistory.unshift(item);
    renderHistory();
  }

  function renderHistory() {
    if (!historyListEl) return;
    if (translationHistory.length === 0) {
      historyListEl.innerHTML = `
        <div style="color: var(--color-text-tertiary); font-size: 13px; text-align: center; margin-top: 30px;">
          No translations in this session yet.
        </div>`;
      return;
    }

    historyListEl.innerHTML = translationHistory.map(h => `
      <div class="history-item-card" data-id="${h.id}">
        <div class="history-meta-row">
          <span>${h.time}</span>
          <span class="tag-badge">${h.emotion}</span>
        </div>
        <div class="history-text-source">${escapeHtml(h.source)}</div>
        <div class="history-text-translated">${escapeHtml(h.translated)}</div>
      </div>
    `).join('');
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/[&<>'"]/g, tag => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
    }[tag] || tag));
  }

  // --- Navigation & Sections ---
  const navLinks = document.querySelectorAll('.nav-link');
  const sections = {
    landing: document.getElementById('sectionLanding'),
    workspace: document.getElementById('sectionWorkspace'),
    research: document.getElementById('sectionResearch')
  };

  function showSection(sectionKey) {
    Object.keys(sections).forEach((key) => {
      if (sections[key]) {
        sections[key].classList.toggle('active', key === sectionKey);
      }
    });
    navLinks.forEach((link) => {
      link.classList.toggle('active', link.dataset.section === sectionKey);
    });
    const sidebarItems = document.querySelectorAll('.sidebar-nav-item');
    sidebarItems.forEach(item => {
      if (item.dataset.section) {
        item.classList.toggle('active', item.dataset.section === sectionKey);
      }
    });
    if (sectionKey === 'research') {
      research.refreshAll();
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  navLinks.forEach((link) => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const sec = link.dataset.section;
      if (sec) showSection(sec);
    });
  });

  const brandLink = document.getElementById('brandLink');
  if (brandLink) {
    brandLink.addEventListener('click', (e) => {
      e.preventDefault();
      showSection('workspace');
    });
  }

  // Landing Page CTAs (compatibility)
  const btnHeroStartLive = document.getElementById('btnHeroStartLive');
  if (btnHeroStartLive) {
    btnHeroStartLive.addEventListener('click', () => {
      showSection('workspace');
      switchWorkspaceTab('live');
    });
  }

  const btnHeroUpload = document.getElementById('btnHeroUpload');
  if (btnHeroUpload) {
    btnHeroUpload.addEventListener('click', () => {
      showSection('workspace');
      switchWorkspaceTab('upload');
    });
  }

  const btnHeroResearch = document.getElementById('btnHeroResearch');
  if (btnHeroResearch) {
    btnHeroResearch.addEventListener('click', () => {
      showSection('research');
    });
  }

  // --- Backend Health Check ---
  const backendPill = document.getElementById('backendStatusPill');
  const backendText = document.getElementById('backendStatusText');
  const deviceText = document.getElementById('deviceStatusText');

  async function checkHealth() {
    try {
      const health = await api.checkHealth();
      if (health.status === 'healthy') {
        backendPill.className = 'status-pill-green';
        backendText.textContent = 'Backend connected';
        if (health.device) {
          deviceText.textContent = health.device.toUpperCase() === 'MPS' ? 'Apple Silicon (MPS)' : health.device.toUpperCase();
        }
        research.updateModelStatus(health);
      } else {
        backendPill.className = 'status-indicator-pill';
        backendText.textContent = 'Backend Offline';
        deviceText.textContent = 'Unavailable';
      }
    } catch {
      backendPill.className = 'status-indicator-pill';
      backendText.textContent = 'Connecting...';
    }
  }
  checkHealth();
  setInterval(checkHealth, 20000);

  // --- Workspace Mode Tabs ---
  const tabLive = document.getElementById('tabWorkspaceLive');
  const tabUpload = document.getElementById('tabWorkspaceUpload');
  const viewLive = document.getElementById('viewWorkspaceLive');
  const viewUpload = document.getElementById('viewWorkspaceUpload');

  function switchWorkspaceTab(mode) {
    tabLive.classList.toggle('active', mode === 'live');
    tabUpload.classList.toggle('active', mode === 'upload');
    tabLive.setAttribute('aria-selected', mode === 'live');
    tabUpload.setAttribute('aria-selected', mode === 'upload');

    viewLive.style.display = mode === 'live' ? 'inline-flex' : 'none';
    viewUpload.style.display = mode === 'upload' ? 'block' : 'none';
  }

  tabLive.addEventListener('click', () => switchWorkspaceTab('live'));
  tabUpload.addEventListener('click', () => switchWorkspaceTab('upload'));

  // --- Language Direction & Swap ---
  const languageSelect = document.getElementById('selectLanguageDirection');
  const btnSwapLang = document.getElementById('btnSwapLanguage');
  const pillLangSource = document.getElementById('pillLangSource');
  const pillLangTarget = document.getElementById('pillLangTarget');

  function updateLanguagePills() {
    if (!languageSelect) return;
    const val = languageSelect.value;
    if (val === 'en_to_hi') {
      if (pillLangSource) pillLangSource.textContent = 'English';
      if (pillLangTarget) pillLangTarget.textContent = 'Hindi';
    } else {
      if (pillLangSource) pillLangSource.textContent = 'Hindi';
      if (pillLangTarget) pillLangTarget.textContent = 'English';
    }
  }

  if (btnSwapLang) {
    btnSwapLang.addEventListener('click', () => {
      if (!languageSelect) return;
      const current = languageSelect.value;
      if (current === 'hi_to_en' || current === 'auto') {
        languageSelect.value = 'en_to_hi';
      } else {
        languageSelect.value = 'hi_to_en';
      }
      updateLanguagePills();
    });
  }

  // --- Pipeline Stage Tracker ---
  const stageSteps = {
    idle: document.getElementById('stageIdle'),
    recording: document.getElementById('stageRecording'),
    asr: document.getElementById('stageAsr'),
    ser: document.getElementById('stageSer'),
    nmt: document.getElementById('stageNmt'),
    tts: document.getElementById('stageTts'),
    done: document.getElementById('stageDone')
  };

  function setPipelineStage(activeStage) {
    Object.keys(stageSteps).forEach((k) => {
      const el = stageSteps[k];
      if (!el) return;
      el.classList.remove('active', 'completed');
    });

    const order = ['idle', 'recording', 'asr', 'ser', 'nmt', 'tts', 'done'];
    const activeIdx = order.indexOf(activeStage);

    order.forEach((k, idx) => {
      const el = stageSteps[k];
      if (!el) return;
      if (idx < activeIdx) el.classList.add('completed');
      else if (idx === activeIdx) el.classList.add('active');
    });
  }

  // --- Dynamic Voice Workspace Status ---
  const voiceStatusTitle = document.getElementById('voiceStatusTitle');
  const voiceStatusSubtitle = document.getElementById('voiceStatusSubtitle');
  const mainMicButton = document.getElementById('mainMicButton');
  const visualizerOverlay = document.getElementById('visualizerStatusOverlay');

  // --- Live Microphone & WebSocket Integration ---
  const btnStartLive = document.getElementById('btnStartLiveSpeech');
  const btnPauseLive = document.getElementById('btnPauseLiveSpeech');
  const btnStopLive = document.getElementById('btnStopLiveSpeech');
  const btnFlushLive = document.getElementById('btnFlushLiveSpeech');
  const recIndicator = document.getElementById('recPulseIndicator');

  // Wire Central Big Mic Button
  if (mainMicButton) {
    mainMicButton.addEventListener('click', () => {
      if (!audioManager.isRecording) {
        btnStartLive.click();
      } else {
        btnStopLive.click();
      }
    });
  }

  // Wire WebSocket Event Callbacks
  wsClient.onStatusChangeCallback = (status) => {
    results.renderWsStatus(status);
    if (status === 'connected') {
      btnFlushLive.disabled = false;
    } else if (status === 'disconnected' || status === 'error') {
      btnFlushLive.disabled = true;
    }
  };

  wsClient.onMessageCallback = (msg) => {
    if (msg.type === 'recording_started' || msg.type === 'session_acknowledged') {
      console.log(`[LiveTranslate] Recording/Session initialized: ${msg.session_id}`);
      if (voiceStatusSubtitle) voiceStatusSubtitle.textContent = 'Streaming speech... Speak naturally';
    } else if (msg.type === 'audio_chunk_received') {
      // Chunk telemetry receipt
      console.debug(`[LiveTranslate] Chunk #${msg.chunk_index} received (RMS: ${msg.rms})`);
    } else if (msg.type === 'partial_transcript') {
      results.renderPartialTranscript(msg.source_transcript);
      setPipelineStage('asr');
    } else if (msg.type === 'final_transcript') {
      results.hidePartialTranscript();
      setPipelineStage('ser');
      if (voiceStatusTitle) voiceStatusTitle.textContent = 'Finalizing translation...';
    } else if (msg.type === 'emotion_updated' || msg.type === 'emotion_update') {
      setPipelineStage('nmt');
      console.log(`[LiveTranslate] Emotion smoothed: ${msg.detected_emotion} (Valence: ${msg.valence}, Arousal: ${msg.arousal})`);
    } else if (msg.type === 'translation_completed') {
      setPipelineStage('tts');
      console.log(`[LiveTranslate] Translation complete: "${msg.translated_text}" via ${msg.provider} (${msg.latency_ms}ms)`);
    } else if (msg.type === 'audio_ready') {
      if (msg.audio_base64) {
        playbackQueue.enqueueSegment(msg.segment_id, msg.audio_base64, true);
      }
    } else if (msg.type === 'translation_chunk') {
      results.hidePartialTranscript();
      setPipelineStage('done');
      results.renderSuccess(msg, recordedBlob, false);

      if (voiceStatusTitle) voiceStatusTitle.textContent = 'Translation ready';
      if (voiceStatusSubtitle) voiceStatusSubtitle.textContent = 'Click microphone to continue conversation';
      if (mainMicButton) mainMicButton.classList.remove('recording', 'processing');

      // Add to session history
      addHistoryItem(msg.source_transcript, msg.translated_text, msg.detected_emotion);

      if (msg.audio_base64 && !playbackQueue.isPlaying) {
        playbackQueue.enqueueSegment(msg.segment_id, msg.audio_base64, true);
      }
    } else if (msg.type === 'cancelled') {
      results.hidePartialTranscript();
      setPipelineStage('idle');
    } else if (msg.type === 'processing_error' || msg.type === 'error') {
      results.showError(`Live Streaming Error (${msg.error_code || 'WS'}): ${msg.message}`);
      if (mainMicButton) mainMicButton.classList.remove('recording', 'processing');
    }
  };

  // Start Live Speech
  btnStartLive.addEventListener('click', async () => {
    try {
      const langDir = languageSelect ? languageSelect.value : 'hi_to_en';
      let srcLang = 'hi';
      let tgtLang = 'en';
      if (langDir === 'en_to_hi') {
        srcLang = 'en';
        tgtLang = 'hi';
      }

      await wsClient.connect(null, srcLang, tgtLang, false);

      const analyser = await audioManager.start();
      visualizer.attachAnalyser(analyser);
      visualizer.start();

      btnStartLive.disabled = true;
      btnPauseLive.disabled = false;
      btnStopLive.disabled = false;
      btnFlushLive.disabled = false;

      if (mainMicButton) {
        mainMicButton.classList.add('recording');
        mainMicButton.classList.remove('processing');
      }
      if (voiceStatusTitle) voiceStatusTitle.textContent = 'Listening...';
      if (voiceStatusSubtitle) {
        voiceStatusSubtitle.textContent = srcLang === 'hi' ? 'Speak naturally in Hindi' : 'Speak naturally in English';
      }

      recIndicator.style.display = 'inline-flex';
      if (visualizerOverlay) visualizerOverlay.style.display = 'none';
      setPipelineStage('recording');

    } catch (err) {
      wsClient.disconnect();
      if (audioManager.isRecording) audioManager.stop();
      alert(`Could not start live translation: ${err.message}`);
      btnStartLive.disabled = false;
      results.showError(`Live Translation Startup Failed: ${err.message}`);
    }
  });

  // Pause / Resume Live Speech
  btnPauseLive.addEventListener('click', () => {
    if (!audioManager.isPaused) {
      audioManager.pause();
      btnPauseLive.innerHTML = `
        <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg> Resume`;
      recIndicator.style.opacity = '0.5';
      if (voiceStatusSubtitle) voiceStatusSubtitle.textContent = 'Recording paused';
    } else {
      audioManager.resume();
      btnPauseLive.innerHTML = `
        <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
          <rect x="6" y="4" width="4" height="16"></rect>
          <rect x="14" y="4" width="4" height="16"></rect>
        </svg> Pause`;
      recIndicator.style.opacity = '1.0';
      if (voiceStatusSubtitle) voiceStatusSubtitle.textContent = 'Listening... Speak now';
    }
  });

  // Stop Live Speech
  btnStopLive.addEventListener('click', () => {
    recordedBlob = audioManager.stop();
    visualizer.stop();
    wsClient.stopSession();
    playbackQueue.stopAndClear();
    results.hidePartialTranscript();

    btnStartLive.disabled = false;
    btnPauseLive.disabled = true;
    btnPauseLive.innerHTML = `
      <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
        <rect x="6" y="4" width="4" height="16"></rect>
        <rect x="14" y="4" width="4" height="16"></rect>
      </svg> Pause`;
    btnStopLive.disabled = true;
    btnFlushLive.disabled = true;

    if (mainMicButton) {
      mainMicButton.classList.remove('recording');
      mainMicButton.classList.add('processing');
    }
    if (voiceStatusTitle) voiceStatusTitle.textContent = 'Translating...';
    if (voiceStatusSubtitle) voiceStatusSubtitle.textContent = 'Processing speech with Whisper & Sarvam AI...';

    recIndicator.style.display = 'none';
    if (visualizerOverlay) visualizerOverlay.style.display = 'block';
    setPipelineStage('asr');
  });

  // Flush / Force-Process Current Utterance
  btnFlushLive.addEventListener('click', () => {
    wsClient.flush();
    setPipelineStage('asr');
  });

  // --- New Translation Action Button ---
  const btnNewTranslation = document.getElementById('btnNewTranslation');
  if (btnNewTranslation) {
    btnNewTranslation.addEventListener('click', () => {
      if (audioManager.isRecording) {
        btnStopLive.click();
      }
      showSection('workspace');
      switchWorkspaceTab('live');

      // Reset displayed texts
      document.getElementById('sourceTranscriptText').innerHTML = '<span class="panel-transcript-placeholder">Spoken words will appear here...</span>';
      document.getElementById('translatedTranscriptText').innerHTML = '<span class="panel-transcript-placeholder">Translated speech will appear here...</span>';
      if (voiceStatusTitle) voiceStatusTitle.textContent = 'Start a conversation';
      if (voiceStatusSubtitle) voiceStatusSubtitle.textContent = 'Your microphone is ready';
      if (mainMicButton) mainMicButton.classList.remove('recording', 'processing');

      const origPlayer = document.getElementById('originalAudioPlayer');
      const genPlayer = document.getElementById('generatedAudioPlayer');
      if (origPlayer) { origPlayer.pause(); origPlayer.style.display = 'none'; }
      if (genPlayer) { genPlayer.pause(); genPlayer.style.display = 'none'; }

      setPipelineStage('idle');
    });
  }

  // --- Copy Text Buttons ---
  function setupCopyButton(btnId, targetElId) {
    const btn = document.getElementById(btnId);
    if (!btn) return;
    btn.addEventListener('click', async () => {
      const target = document.getElementById(targetElId);
      if (!target) return;
      const text = target.textContent.trim();
      if (!text || text.includes('will appear here')) return;

      try {
        await navigator.clipboard.writeText(text);
        btn.classList.add('copied');
        btn.innerHTML = `
          <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.5">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
          Copied!`;
        setTimeout(() => {
          btn.classList.remove('copied');
          btn.innerHTML = `
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
            Copy`;
        }, 1500);
      } catch (err) {
        console.error('Clipboard copy failed:', err);
      }
    });
  }

  setupCopyButton('btnCopySource', 'sourceTranscriptText');
  setupCopyButton('btnCopyTranslated', 'translatedTranscriptText');

  // --- Drawer & Modal Controls ---
  const backdrop = document.getElementById('drawerBackdrop');
  const historyDrawer = document.getElementById('historyDrawer');
  const settingsModal = document.getElementById('settingsModal');
  const appSidebar = document.getElementById('appSidebar');

  function closeAllOverlays() {
    if (historyDrawer) historyDrawer.classList.remove('open');
    if (settingsModal) settingsModal.classList.remove('open');
    if (appSidebar) appSidebar.classList.remove('open');
    if (backdrop) backdrop.classList.remove('open');
  }

  if (backdrop) backdrop.addEventListener('click', closeAllOverlays);

  // History Drawer
  const navItemHistory = document.getElementById('navItemHistory');
  const btnCloseHistory = document.getElementById('btnCloseHistory');
  const btnClearHistory = document.getElementById('btnClearHistory');

  if (navItemHistory) {
    navItemHistory.addEventListener('click', () => {
      renderHistory();
      if (historyDrawer) historyDrawer.classList.add('open');
      if (backdrop) backdrop.classList.add('open');
    });
  }

  if (btnCloseHistory) btnCloseHistory.addEventListener('click', closeAllOverlays);

  if (btnClearHistory) {
    btnClearHistory.addEventListener('click', () => {
      translationHistory = [];
      renderHistory();
    });
  }

  // Settings Modal
  const navItemSettings = document.getElementById('navItemSettings');
  const btnHeaderSettings = document.getElementById('btnHeaderSettings');
  const btnCloseSettings = document.getElementById('btnCloseSettings');
  const btnSaveSettings = document.getElementById('btnSaveSettings');
  const modalToggleAutoPlay = document.getElementById('modalToggleAutoPlay');

  function openSettings() {
    if (modalToggleAutoPlay && toggleAutoPlay) {
      modalToggleAutoPlay.checked = toggleAutoPlay.checked;
    }
    if (settingsModal) settingsModal.classList.add('open');
    if (backdrop) backdrop.classList.add('open');
  }

  if (navItemSettings) navItemSettings.addEventListener('click', openSettings);
  if (btnHeaderSettings) btnHeaderSettings.addEventListener('click', openSettings);
  if (btnCloseSettings) btnCloseSettings.addEventListener('click', closeAllOverlays);
  if (btnSaveSettings) {
    btnSaveSettings.addEventListener('click', () => {
      if (modalToggleAutoPlay && toggleAutoPlay) {
        toggleAutoPlay.checked = modalToggleAutoPlay.checked;
        playbackQueue.setAutoPlay(modalToggleAutoPlay.checked);
      }
      closeAllOverlays();
    });
  }

  // Mobile Sidebar Toggle
  const btnToggleSidebar = document.getElementById('btnToggleSidebar');
  if (btnToggleSidebar) {
    btnToggleSidebar.addEventListener('click', () => {
      if (appSidebar) {
        appSidebar.classList.toggle('open');
        if (backdrop) backdrop.classList.toggle('open', appSidebar.classList.contains('open'));
      }
    });
  }

  // --- Upload Recording Mode ---
  const dropzoneEl = document.getElementById('uploadDropzone');
  const fileInputEl = document.getElementById('audioFileInput');
  const fileSelectedBox = document.getElementById('fileSelectedBox');
  const selectedFileName = document.getElementById('selectedFileName');
  const btnClearSelectedFile = document.getElementById('btnClearSelectedFile');
  const btnProcessUploadedAudio = document.getElementById('btnProcessUploadedAudio');
  let currentUploadedFile = null;

  dropzoneEl.addEventListener('click', () => fileInputEl.click());

  dropzoneEl.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzoneEl.classList.add('dragover');
  });

  dropzoneEl.addEventListener('dragleave', () => {
    dropzoneEl.classList.remove('dragover');
  });

  dropzoneEl.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzoneEl.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  });

  fileInputEl.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  });

  function handleFileSelection(file) {
    const validExtensions = ['.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm', '.mp4'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!validExtensions.includes(ext)) {
      results.showError(`Invalid file format '${ext}'. Supported: ${validExtensions.join(', ')}`);
      return;
    }

    if (file.size > 25 * 1024 * 1024) {
      results.showError('File exceeds 25MB maximum upload limit.');
      return;
    }

    currentUploadedFile = file;
    selectedFileName.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
    fileSelectedBox.style.display = 'flex';
    btnProcessUploadedAudio.disabled = false;
  }

  if (btnClearSelectedFile) {
    btnClearSelectedFile.addEventListener('click', () => {
      currentUploadedFile = null;
      fileInputEl.value = '';
      fileSelectedBox.style.display = 'none';
      btnProcessUploadedAudio.disabled = true;
    });
  }

  btnProcessUploadedAudio.addEventListener('click', async () => {
    if (!currentUploadedFile) return;

    btnProcessUploadedAudio.disabled = true;
    btnProcessUploadedAudio.textContent = 'Processing Pipeline...';
    setPipelineStage('asr');

    const dir = languageSelect ? languageSelect.value : 'hi_to_en';
    let sLang = 'hi';
    let tLang = 'en';
    if (dir === 'en_to_hi') {
      sLang = 'en';
      tLang = 'hi';
    }

    try {
      setPipelineStage('ser');
      const response = await api.translateAudioFile(
        currentUploadedFile,
        sLang,
        tLang,
        true
      );
      setPipelineStage('done');
      results.renderSuccess(response, currentUploadedFile, toggleAutoPlay ? toggleAutoPlay.checked : true);

      // Add to session history
      addHistoryItem(response.source_transcript, response.translated_text, response.detected_emotion);

    } catch (err) {
      setPipelineStage('idle');
      results.showError(`Translation Error: ${err.message}`, () => {
        btnProcessUploadedAudio.click();
      });
    } finally {
      btnProcessUploadedAudio.disabled = false;
      btnProcessUploadedAudio.textContent = 'Translate Audio File';
    }
  });

  // --- Demonstration Sample Loader ---
  const btnLoadDemoSample = document.getElementById('btnLoadDemoSample');
  if (btnLoadDemoSample) {
    btnLoadDemoSample.addEventListener('click', async () => {
      btnLoadDemoSample.disabled = true;
      btnLoadDemoSample.textContent = 'Translating Sample...';
      setPipelineStage('asr');

      try {
        setPipelineStage('nmt');
        const textResponse = await api.translateText(
          'नमस्ते, आप कैसे हैं?',
          'hi',
          'en',
          'neutral',
          true
        );

        setPipelineStage('done');
        results.renderSuccess({
          source_language: textResponse.source_lang,
          target_language: textResponse.target_lang,
          source_transcript: textResponse.source_text,
          translated_text: textResponse.translated_text,
          asr_confidence: 1.0,
          detected_emotion: textResponse.emotion,
          emotion_probabilities: null,
          valence: null,
          arousal: null,
          dominance: null,
          gating_weight_z: textResponse.preservation_plan ? textResponse.preservation_plan.gating_weight_z : null,
          preservation_plan: textResponse.preservation_plan,
          expressive_synthesis_supported: textResponse.expressive_synthesis_supported,
          emotional_fidelity_status: textResponse.emotional_fidelity_status,
          speaker_similarity_retained: textResponse.speaker_similarity_retained,
          synthesis_voice_used: textResponse.synthesis_voice_used,
          audio_base64: textResponse.audio_base64,
          audio_url: textResponse.audio_url,
          latency: {
            asr_ms: 0.0,
            ser_ms: 0.0,
            nmt_ms: textResponse.latency_ms,
            tts_ms: 0.0,
            total_pipeline_ms: textResponse.latency_ms
          }
        }, null, toggleAutoPlay ? toggleAutoPlay.checked : true);

        // Add to history
        addHistoryItem(textResponse.source_text, textResponse.translated_text, textResponse.emotion);

      } catch (err) {
        setPipelineStage('idle');
        results.showError(`Sample Execution Error: ${err.message}`, () => {
          btnLoadDemoSample.click();
        });
      } finally {
        btnLoadDemoSample.disabled = false;
        btnLoadDemoSample.textContent = 'Demo Sample';
      }
    });
  }
});
