/**
 * VoiceAffect Audio Manager & Waveform Visualizer (Phase 4 & Phase 6)
 * Real-time microphone capture, 16,000 Hz resampling, AnalyserNode audio visualization,
 * duration timer, device enumeration, and graceful error handling.
 */

export class AudioCaptureManager {
  constructor(onChunkReady = null, onDeviceDisconnected = null, onDurationUpdate = null) {
    this.onChunkReady = onChunkReady;
    this.onDeviceDisconnected = onDeviceDisconnected;
    this.onDurationUpdate = onDurationUpdate;

    this.audioContext = null;
    this.mediaStream = null;
    this.sourceNode = null;
    this.analyserNode = null;
    this.scriptProcessor = null;

    this.isRecording = false;
    this.isPaused = false;
    this.recordedChunks = [];
    this.currentRms = 0.0;

    this.startTime = null;
    this.pausedDuration = 0;
    this.pauseStartTime = null;
    this.timerInterval = null;
  }

  /**
   * Checks whether any audio input (microphone) device is available in the browser.
   */
  static async checkDeviceAvailability() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      return { available: false, reason: 'MediaDevices API is unsupported in this browser.' };
    }
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = devices.filter(d => d.kind === 'audioinput');
      return {
        available: audioInputs.length > 0,
        count: audioInputs.length,
        devices: audioInputs
      };
    } catch (err) {
      return { available: false, reason: err.message };
    }
  }

  async start() {
    this.recordedChunks = [];
    this.currentRms = 0.0;
    this.pausedDuration = 0;
    this.pauseStartTime = null;

    // 1. Device availability check
    const check = await AudioCaptureManager.checkDeviceAvailability();
    if (!check.available && check.reason) {
      throw new Error(check.reason);
    }

    // 2. Request microphone permission
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
    } catch (err) {
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        throw new Error('Microphone permission was denied. Please allow microphone access in your browser settings.');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        throw new Error('No microphone device found. Please connect an audio input device.');
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        throw new Error('Microphone hardware is busy or locked by another application.');
      } else {
        throw new Error(`Microphone access error (${err.name}): ${err.message}`);
      }
    }

    // 3. Attach track-ended disconnection listener
    const audioTracks = this.mediaStream.getAudioTracks();
    if (audioTracks.length > 0) {
      audioTracks[0].onended = () => {
        console.warn('Microphone hardware track disconnected.');
        this.stop();
        if (this.onDeviceDisconnected) {
          this.onDeviceDisconnected();
        }
      };
    }

    // 4. Initialize Web Audio Context
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    this.audioContext = new AudioContextClass();
    const inputSampleRate = this.audioContext.sampleRate;
    console.log(`[AudioCapture] Initialized: Native rate = ${inputSampleRate}Hz -> Target = 16000Hz Mono PCM`);

    this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);
    this.analyserNode = this.audioContext.createAnalyser();
    this.analyserNode.fftSize = 256;
    this.analyserNode.smoothingTimeConstant = 0.8;

    this.sourceNode.connect(this.analyserNode);

    // Audio processor buffer (4096 frames)
    this.scriptProcessor = this.audioContext.createScriptProcessor(4096, 1, 1);
    this.analyserNode.connect(this.scriptProcessor);
    this.scriptProcessor.connect(this.audioContext.destination);

    this.scriptProcessor.onaudioprocess = (event) => {
      if (!this.isRecording || this.isPaused) return;

      const rawSamples = event.inputBuffer.getChannelData(0);

      // Calculate Root-Mean-Square (RMS) amplitude
      let sumSq = 0.0;
      for (let i = 0; i < rawSamples.length; i++) {
        sumSq += rawSamples[i] * rawSamples[i];
      }
      this.currentRms = Math.sqrt(sumSq / rawSamples.length);

      // Resample to exact 16,000 Hz if native rate differs
      const resampled = this._resampleTo16k(rawSamples, inputSampleRate);

      // Quantize float32 (-1.0 to 1.0) into signed 16-bit PCM integer
      const pcm16 = new Int16Array(resampled.length);
      for (let i = 0; i < resampled.length; i++) {
        const s = Math.max(-1.0, Math.min(1.0, resampled[i]));
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }

      this.recordedChunks.push(new Int16Array(pcm16));

      if (this.onChunkReady) {
        this.onChunkReady(pcm16.buffer, this.currentRms);
      }
    };

    this.isRecording = true;
    this.isPaused = false;
    this.startTime = Date.now();

    // Start duration timer
    this._startDurationTimer();

    return this.analyserNode;
  }

  _startDurationTimer() {
    this._stopDurationTimer();
    this.timerInterval = setInterval(() => {
      if (!this.isRecording || this.isPaused) return;
      const elapsedMs = (Date.now() - this.startTime) - this.pausedDuration;
      const totalSec = Math.floor(elapsedMs / 1000);
      const mins = Math.floor(totalSec / 60);
      const secs = totalSec % 60;
      const formatted = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
      if (this.onDurationUpdate) {
        this.onDurationUpdate(formatted, totalSec);
      }
    }, 500);
  }

  _stopDurationTimer() {
    if (this.timerInterval) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }
  }

  pause() {
    if (!this.isRecording || this.isPaused) return;
    this.isPaused = true;
    this.pauseStartTime = Date.now();
  }

  resume() {
    if (!this.isRecording || !this.isPaused) return;
    if (this.pauseStartTime) {
      this.pausedDuration += (Date.now() - this.pauseStartTime);
      this.pauseStartTime = null;
    }
    this.isPaused = false;
  }

  stop() {
    this.isRecording = false;
    this.isPaused = false;
    this.currentRms = 0.0;
    this._stopDurationTimer();

    if (this.scriptProcessor) {
      try { this.scriptProcessor.disconnect(); } catch (_) {}
      this.scriptProcessor = null;
    }
    if (this.sourceNode) {
      try { this.sourceNode.disconnect(); } catch (_) {}
      this.sourceNode = null;
    }
    if (this.analyserNode) {
      try { this.analyserNode.disconnect(); } catch (_) {}
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => {
        try { track.stop(); } catch (_) {}
      });
      this.mediaStream = null;
    }
    if (this.audioContext && this.audioContext.state !== 'closed') {
      try { this.audioContext.close(); } catch (_) {}
      this.audioContext = null;
    }

    return this.getRecordedWavBlob();
  }

  getAudioLevel() {
    return this.currentRms;
  }

  _resampleTo16k(samples, originRate) {
    if (originRate === 16000) return samples;

    const ratio = originRate / 16000;
    const newLength = Math.round(samples.length / ratio);
    const result = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
      const originIndex = i * ratio;
      const indexFloor = Math.floor(originIndex);
      const indexCeil = Math.min(indexFloor + 1, samples.length - 1);
      const weight = originIndex - indexFloor;
      result[i] = samples[indexFloor] * (1.0 - weight) + samples[indexCeil] * weight;
    }
    return result;
  }

  getRecordedWavBlob() {
    if (this.recordedChunks.length === 0) return null;

    let totalSamples = 0;
    for (const chunk of this.recordedChunks) {
      totalSamples += chunk.length;
    }

    const merged = new Int16Array(totalSamples);
    let offset = 0;
    for (const chunk of this.recordedChunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }

    return this._encodeWav(merged, 16000);
  }

  _encodeWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    // RIFF chunk descriptor
    this._writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    this._writeString(view, 8, 'WAVE');

    // fmt sub-chunk
    this._writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // Linear PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // Byte rate
    view.setUint16(32, 2, true); // Block align
    view.setUint16(34, 16, true); // 16-bit

    // data sub-chunk
    this._writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    // Write PCM samples
    let index = 44;
    for (let i = 0; i < samples.length; i++) {
      view.setInt16(index, samples[i], true);
      index += 2;
    }

    return new Blob([view], { type: 'audio/wav' });
  }

  _writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }
}

export class WaveformVisualizer {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
    this.analyser = null;
    this.animId = null;
    this.isRunning = false;
    this._phase = 0;
  }

  attachAnalyser(analyserNode) {
    this.analyser = analyserNode;
  }

  start() {
    this.isRunning = true;
    this._render();
  }

  _render() {
    if (!this.isRunning) return;
    this.animId = requestAnimationFrame(() => this._render());

    const width = this.canvas.width;
    const height = this.canvas.height;
    const centerY = height / 2;
    this.ctx.clearRect(0, 0, width, height);

    if (!this.analyser) {
      // Draw elegant quiescent resting wave
      this._phase += 0.04;
      this.ctx.save();
      this.ctx.strokeStyle = 'rgba(59, 130, 246, 0.22)';
      this.ctx.lineWidth = 1.5;
      this.ctx.beginPath();
      for (let x = 0; x < width; x += 4) {
        const y = centerY + Math.sin(x * 0.02 + this._phase) * 2;
        if (x === 0) {
          this.ctx.moveTo(x, y);
        } else {
          this.ctx.lineTo(x, y);
        }
      }
      this.ctx.stroke();
      this.ctx.restore();
      return;
    }

    // Render genuine acoustic frequency data from microphone
    const bufferLength = this.analyser.frequencyBinCount;
    const freqData = new Uint8Array(bufferLength);
    this.analyser.getByteFrequencyData(freqData);

    const numBars = 48; // Crisp, balanced bar distribution
    const step = Math.floor(bufferLength / numBars);
    const barWidth = Math.max(3, (width / numBars) - 2.5);
    const totalContentWidth = numBars * (barWidth + 2.5);
    let x = (width - totalContentWidth) / 2;

    this.ctx.save();
    for (let i = 0; i < numBars; i++) {
      let sum = 0;
      for (let j = 0; j < step; j++) {
        sum += freqData[i * step + j] || 0;
      }
      const val = sum / step;
      const normalized = Math.min(1.0, val / 220);
      const barHeight = Math.max(3, normalized * (height * 0.42));

      // Dual-tone vertical gradient
      const grad = this.ctx.createLinearGradient(0, centerY - barHeight, 0, centerY + barHeight);
      grad.addColorStop(0, '#60a5fa');
      grad.addColorStop(0.5, '#3b82f6');
      grad.addColorStop(1, '#818cf8');

      this.ctx.fillStyle = grad;
      this._drawPill(x, centerY - barHeight, barWidth, barHeight * 2);
      x += barWidth + 2.5;
    }
    this.ctx.restore();
  }

  _drawPill(x, y, w, h) {
    const r = Math.min(w / 2, h / 2);
    this.ctx.beginPath();
    this.ctx.moveTo(x + r, y);
    this.ctx.arcTo(x + w, y, x + w, y + h, r);
    this.ctx.arcTo(x + w, y + h, x, y + h, r);
    this.ctx.arcTo(x, y + h, x, y, r);
    this.ctx.arcTo(x, y, x + w, y, r);
    this.ctx.closePath();
    this.ctx.fill();
  }

  stop() {
    this.isRunning = false;
    if (this.animId) {
      cancelAnimationFrame(this.animId);
      this.animId = null;
    }
    const width = this.canvas.width;
    const height = this.canvas.height;
    this.ctx.clearRect(0, 0, width, height);
  }
}
