/**
 * VoiceAffect - Bilingual Voice Translation System With Emotion Awareness
 * Vanilla JavaScript Frontend Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  // --- DOM Elements ---
  const backendStatusPill = document.getElementById('backendStatusPill');
  const backendStatusText = document.getElementById('backendStatusText');
  const deviceText = document.getElementById('deviceText');

  // Tabs
  const tabLive = document.getElementById('tabLive');
  const tabUpload = document.getElementById('tabUpload');
  const tabText = document.getElementById('tabText');
  const viewLive = document.getElementById('viewLive');
  const viewUpload = document.getElementById('viewUpload');
  const viewText = document.getElementById('viewText');

  // Controls
  const languageDirection = document.getElementById('languageDirection');
  const autoPlayAudio = document.getElementById('autoPlayAudio');

  // Live Speech
  const btnToggleRecording = document.getElementById('btnToggleRecording');
  const btnRecordingText = document.getElementById('btnRecordingText');
  const btnFlushChunk = document.getElementById('btnFlushChunk');
  const liveRecordingIndicator = document.getElementById('liveRecordingIndicator');
  const visualizerCanvas = document.getElementById('audioVisualizer');
  const visualizerOverlay = document.getElementById('visualizerOverlay');

  // Upload Recording
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const filePreview = document.getElementById('filePreview');
  const previewFileName = document.getElementById('previewFileName');
  const btnClearFile = document.getElementById('btnClearFile');
  const btnUploadTranslate = document.getElementById('btnUploadTranslate');

  // Text Mode
  const directTextInput = document.getElementById('directTextInput');
  const textEmotionSelect = document.getElementById('textEmotionSelect');
  const btnTranslateText = document.getElementById('btnTranslateText');

  // Captions
  const sourceLangBadge = document.getElementById('sourceLangBadge');
  const targetLangBadge = document.getElementById('targetLangBadge');
  const asrConfidenceBadge = document.getElementById('asrConfidenceBadge');
  const sourceTranscriptText = document.getElementById('sourceTranscriptText');
  const translatedTranscriptText = document.getElementById('translatedTranscriptText');
  const btnReplayAudio = document.getElementById('btnReplayAudio');
  const audioPlayback = document.getElementById('audioPlayback');

  // Emotion Telemetry
  const dominantEmotionBadge = document.getElementById('dominantEmotionBadge');
  const probBars = {
    neutral: { fill: document.getElementById('fillNeutral'), text: document.getElementById('probNeutral') },
    happy: { fill: document.getElementById('fillHappy'), text: document.getElementById('probHappy') },
    sad: { fill: document.getElementById('fillSad'), text: document.getElementById('probSad') },
    angry: { fill: document.getElementById('fillAngry'), text: document.getElementById('probAngry') },
    fearful: { fill: document.getElementById('fillFearful'), text: document.getElementById('probFearful') },
    surprised: { fill: document.getElementById('fillSurprised'), text: document.getElementById('probSurprised') },
  };
  const valValence = document.getElementById('valValence');
  const fillValence = document.getElementById('fillValence');
  const valArousal = document.getElementById('valArousal');
  const fillArousal = document.getElementById('fillArousal');
  const valDominance = document.getElementById('valDominance');
  const fillDominance = document.getElementById('fillDominance');
  const gatingWeightVal = document.getElementById('gatingWeightVal');
  const pitchMultVal = document.getElementById('pitchMultVal');
  const rateMultVal = document.getElementById('rateMultVal');
  const prosodyNotesText = document.getElementById('prosodyNotesText');

  // Latency Dashboard
  const latAsr = document.getElementById('latAsr');
  const latSer = document.getElementById('latSer');
  const latNmt = document.getElementById('latNmt');
  const latTts = document.getElementById('latTts');
  const totalLatencyPill = document.getElementById('totalLatencyPill');

  // --- State Variables ---
  let isRecording = false;
  let audioContext = null;
  let mediaStream = null;
  let scriptProcessor = null;
  let analyserNode = null;
  let visualizerAnimId = null;
  let ws = null;
  let selectedFile = null;
  let lastAudioBase64 = null;

  // --- Health Check Initialization ---
  async function checkBackendHealth() {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        backendStatusText.textContent = 'System Online';
        backendStatusPill.querySelector('.status-dot').style.backgroundColor = '#10b981';
        if (data.device) {
          deviceText.textContent = data.device.toUpperCase() === 'MPS' ? 'Apple Silicon (MPS)' : data.device.toUpperCase();
        }
      } else {
        throw new Error('Health check non-200');
      }
    } catch (err) {
      backendStatusText.textContent = 'Backend Offline';
      backendStatusPill.querySelector('.status-dot').style.backgroundColor = '#f43f5e';
    }
  }
  checkBackendHealth();
  setInterval(checkBackendHealth, 15000);

  // --- Tab Switching ---
  function switchTab(activeTab, activeView) {
    [tabLive, tabUpload, tabText].forEach(t => t.classList.remove('active'));
    [viewLive, viewUpload, viewText].forEach(v => (v.style.display = 'none'));
    activeTab.classList.add('active');
    activeView.style.display = 'block';
  }
  tabLive.addEventListener('click', () => switchTab(tabLive, viewLive));
  tabUpload.addEventListener('click', () => switchTab(tabUpload, viewUpload));
  tabText.addEventListener('click', () => switchTab(tabText, viewText));

  // --- Canvas Audio Visualizer ---
  const canvasCtx = visualizerCanvas.getContext('2d');
  function drawVisualizer() {
    if (!analyserNode) return;
    visualizerAnimId = requestAnimationFrame(drawVisualizer);

    const bufferLength = analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyserNode.getByteFrequencyData(dataArray);

    canvasCtx.fillStyle = '#0a0d14';
    canvasCtx.fillRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);

    const barWidth = (visualizerCanvas.width / bufferLength) * 2.5;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * visualizerCanvas.height;
      const gradient = canvasCtx.createLinearGradient(0, visualizerCanvas.height, 0, 0);
      gradient.addColorStop(0, '#6366f1');
      gradient.addColorStop(1, '#06b6d4');

      canvasCtx.fillStyle = gradient;
      canvasCtx.fillRect(x, visualizerCanvas.height - barHeight, barWidth, barHeight);
      x += barWidth + 1;
    }
  }

  function clearVisualizer() {
    if (visualizerAnimId) cancelAnimationFrame(visualizerAnimId);
    canvasCtx.fillStyle = '#0a0d14';
    canvasCtx.fillRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
  }

  // --- WebSocket Connection ---
  function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/live-translate`;
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      console.log('Live Translation WebSocket connected.');
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'translation_chunk' || msg.type === 'translation_final') {
          handlePipelineResponse(msg);
        }
      } catch (e) {
        console.error('Error parsing WS message:', e);
      }
    };

    ws.onerror = (err) => {
      console.warn('WebSocket error:', err);
    };

    ws.onclose = () => {
      if (isRecording) {
        setTimeout(connectWebSocket, 1000);
      }
    };
  }

  // --- Microphone Capture ---
  async function startRecording() {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      const sourceNode = audioContext.createMediaStreamSource(mediaStream);

      analyserNode = audioContext.createAnalyser();
      analyserNode.fftSize = 128;
      sourceNode.connect(analyserNode);

      // Downsample / PCM conversion buffer
      scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
      analyserNode.connect(scriptProcessor);
      scriptProcessor.connect(audioContext.destination);

      connectWebSocket();

      scriptProcessor.onaudioprocess = (e) => {
        if (!isRecording || !ws || ws.readyState !== WebSocket.OPEN) return;
        const inputBuffer = e.inputBuffer.getChannelData(0);
        // Convert Float32Array to 16-bit PCM Int16Array
        const pcm16 = new Int16Array(inputBuffer.length);
        for (let i = 0; i < inputBuffer.length; i++) {
          const s = Math.max(-1, Math.min(1, inputBuffer[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        ws.send(pcm16.buffer);
      };

      isRecording = true;
      btnToggleRecording.classList.add('recording');
      btnRecordingText.textContent = 'Stop Live Translation';
      btnFlushChunk.disabled = false;
      liveRecordingIndicator.style.display = 'flex';
      visualizerOverlay.style.display = 'none';
      drawVisualizer();

    } catch (err) {
      alert(`Microphone permission error: ${err.message}. Please allow microphone access.`);
      console.error(err);
    }
  }

  function stopRecording() {
    isRecording = false;
    btnToggleRecording.classList.remove('recording');
    btnRecordingText.textContent = 'Start Live Translation';
    btnFlushChunk.disabled = true;
    liveRecordingIndicator.style.display = 'none';
    visualizerOverlay.style.display = 'block';

    if (scriptProcessor) {
      scriptProcessor.disconnect();
      scriptProcessor = null;
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop());
      mediaStream = null;
    }
    if (audioContext && audioContext.state !== 'closed') {
      audioContext.close();
      audioContext = null;
    }
    clearVisualizer();

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'flush' }));
    }
  }

  btnToggleRecording.addEventListener('click', () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  });

  btnFlushChunk.addEventListener('click', () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'flush' }));
    }
  });

  // --- Pipeline Response Renderer ---
  function handlePipelineResponse(data) {
    if (!data.speech_detected && !data.source_transcript) {
      return;
    }

    // 1. Captions
    if (data.source_transcript) {
      sourceTranscriptText.textContent = data.source_transcript;
      sourceLangBadge.textContent = data.source_lang === 'hi' ? 'Hindi (हिंदी)' : 'English';
    }
    if (data.confidence !== undefined) {
      asrConfidenceBadge.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
    }

    if (data.translated_text) {
      translatedTranscriptText.textContent = data.translated_text;
      targetLangBadge.textContent = data.target_lang === 'hi' ? 'Hindi Translation' : 'English Translation';
    }

    // 2. Emotion Telemetry
    if (data.detected_emotion) {
      dominantEmotionBadge.textContent = data.detected_emotion;
      dominantEmotionBadge.className = `emotion-pill ${data.detected_emotion.toLowerCase()}`;
    }

    if (data.emotion_probabilities) {
      for (const [emo, prob] of Object.entries(data.emotion_probabilities)) {
        if (probBars[emo]) {
          const pct = Math.round(prob * 100);
          probBars[emo].fill.style.width = `${pct}%`;
          probBars[emo].text.textContent = `${pct}%`;
        }
      }
    }

    // VAD Vectors
    if (data.valence !== undefined) {
      valValence.textContent = data.valence.toFixed(2);
      fillValence.style.width = `${Math.round(data.valence * 100)}%`;
    }
    if (data.arousal !== undefined) {
      valArousal.textContent = data.arousal.toFixed(2);
      fillArousal.style.width = `${Math.round(data.arousal * 100)}%`;
    }
    if (data.dominance !== undefined) {
      valDominance.textContent = data.dominance.toFixed(2);
      fillDominance.style.width = `${Math.round(data.dominance * 100)}%`;
    }

    // Gating and Prosody
    if (data.gating_weight_z !== undefined) {
      gatingWeightVal.textContent = data.gating_weight_z.toFixed(3);
    }
    if (data.preservation_plan) {
      pitchMultVal.textContent = `${data.preservation_plan.pitch_multiplier || 1.0}x`;
      rateMultVal.textContent = `${data.preservation_plan.rate_multiplier || 1.0}x`;
      prosodyNotesText.textContent = data.preservation_plan.prosody_notes || 'Emotion prosody applied';
    }

    // 3. Latency Dashboard
    if (data.latency) {
      latAsr.textContent = `${Math.round(data.latency.asr_ms)} ms`;
      latSer.textContent = `${Math.round(data.latency.ser_ms)} ms`;
      latNmt.textContent = `${Math.round(data.latency.nmt_ms)} ms`;
      latTts.textContent = `${Math.round(data.latency.tts_ms)} ms`;
      totalLatencyPill.textContent = `Total: ${Math.round(data.latency.total_pipeline_ms)} ms`;
    }

    // 4. Audio Playback
    if (data.audio_base64) {
      lastAudioBase64 = data.audio_base64;
      btnReplayAudio.disabled = false;
      if (autoPlayAudio.checked) {
        playBase64Audio(data.audio_base64);
      }
    } else if (data.audio_url) {
      btnReplayAudio.disabled = false;
      if (autoPlayAudio.checked) {
        audioPlayback.src = data.audio_url;
        audioPlayback.play().catch(e => console.warn('Audio auto-play blocked by browser:', e));
      }
    }
  }

  function playBase64Audio(b64) {
    const audioBlob = b64toBlob(b64, 'audio/wav');
    const audioUrl = URL.createObjectURL(audioBlob);
    audioPlayback.src = audioUrl;
    audioPlayback.play().catch(e => console.warn('Audio playback error:', e));
  }

  function b64toBlob(b64Data, contentType = '', sliceSize = 512) {
    const byteCharacters = atob(b64Data);
    const byteArrays = [];
    for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
      const slice = byteCharacters.slice(offset, offset + sliceSize);
      const byteNumbers = new Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        byteNumbers[i] = slice.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      byteArrays.push(byteArray);
    }
    return new Blob(byteArrays, { type: contentType });
  }

  btnReplayAudio.addEventListener('click', () => {
    if (lastAudioBase64) {
      playBase64Audio(lastAudioBase64);
    } else if (audioPlayback.src) {
      audioPlayback.currentTime = 0;
      audioPlayback.play();
    }
  });

  // --- File Upload Handling ---
  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    selectedFile = file;
    previewFileName.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
    filePreview.style.display = 'flex';
    dropzone.querySelector('.dropzone-content').style.display = 'none';
    btnUploadTranslate.disabled = false;
  }

  btnClearFile.addEventListener('click', (e) => {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = '';
    filePreview.style.display = 'none';
    dropzone.querySelector('.dropzone-content').style.display = 'block';
    btnUploadTranslate.disabled = true;
  });

  btnUploadTranslate.addEventListener('click', async () => {
    if (!selectedFile) return;

    btnUploadTranslate.disabled = true;
    btnUploadTranslate.innerHTML = `<span class="rec-dot"></span> Processing Pipeline...`;

    const formData = new FormData();
    formData.append('file', selectedFile);

    const dir = languageDirection.value;
    if (dir === 'hi_to_en') {
      formData.append('source_lang', 'hi');
      formData.append('target_lang', 'en');
    } else if (dir === 'en_to_hi') {
      formData.append('source_lang', 'en');
      formData.append('target_lang', 'hi');
    }

    try {
      const res = await fetch('/api/translate-audio', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Upload translation failed');
      }
      handlePipelineResponse(data);
    } catch (err) {
      alert(`Translation Error: ${err.message}`);
    } finally {
      btnUploadTranslate.disabled = false;
      btnUploadTranslate.innerHTML = `
        <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg> Process & Translate File`;
    }
  });

  // --- Text Translation Handling ---
  btnTranslateText.addEventListener('click', async () => {
    const text = directTextInput.value.trim();
    if (!text) return;

    btnTranslateText.disabled = true;
    btnTranslateText.textContent = 'Translating...';

    const dir = languageDirection.value;
    const sourceLang = dir === 'hi_to_en' ? 'hi' : (dir === 'en_to_hi' ? 'en' : 'en');
    const targetLang = sourceLang === 'en' ? 'hi' : 'en';

    try {
      const res = await fetch('/api/text-translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text,
          source_lang: sourceLang,
          target_lang: targetLang,
          emotion: textEmotionSelect.value,
          synthesize_speech: true
        })
      });
      const data = await res.json();
      if (res.ok) {
        sourceTranscriptText.textContent = data.source_text;
        sourceLangBadge.textContent = data.source_lang === 'hi' ? 'Hindi' : 'English';
        translatedTranscriptText.textContent = data.translated_text;
        targetLangBadge.textContent = data.target_lang === 'hi' ? 'Hindi Translation' : 'English Translation';

        if (data.emotion) {
          dominantEmotionBadge.textContent = data.emotion;
          dominantEmotionBadge.className = `emotion-pill ${data.emotion}`;
        }
        if (data.preservation_plan) {
          pitchMultVal.textContent = `${data.preservation_plan.pitch_multiplier}x`;
          rateMultVal.textContent = `${data.preservation_plan.rate_multiplier}x`;
          gatingWeightVal.textContent = data.preservation_plan.gating_weight_z.toFixed(3);
          prosodyNotesText.textContent = data.preservation_plan.prosody_notes;
        }
        if (data.audio_base64) {
          lastAudioBase64 = data.audio_base64;
          btnReplayAudio.disabled = false;
          if (autoPlayAudio.checked) {
            playBase64Audio(data.audio_base64);
          }
        }
        totalLatencyPill.textContent = `Total: ${Math.round(data.latency_ms)} ms`;
      }
    } catch (err) {
      alert(`Error translating text: ${err.message}`);
    } finally {
      btnTranslateText.disabled = false;
      btnTranslateText.textContent = 'Translate & Speak';
    }
  });
});
