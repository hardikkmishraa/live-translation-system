/**
 * VoiceAffect API Client
 * Clean interface to backend REST endpoints with validation and error formatting
 */

export class ApiClient {
  constructor(baseUrl = '') {
    this.baseUrl = baseUrl;
  }

  async checkHealth() {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 6000);
      const res = await fetch(`${this.baseUrl}/api/health`, {
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: Failed to reach health service`);
      }
      return await res.json();
    } catch (err) {
      return {
        status: 'offline',
        error: err.name === 'AbortError' ? 'Health check timed out' : err.message
      };
    }
  }

  async translateAudioFile(file, sourceLang = null, targetLang = null, synthesizeSpeech = true) {
    if (!file) {
      throw new Error('No audio file provided.');
    }

    const maxBytes = 25 * 1024 * 1024;
    if (file.size > maxBytes) {
      throw new Error(`File size ${(file.size / (1024 * 1024)).toFixed(1)}MB exceeds maximum 25MB limit.`);
    }

    const formData = new FormData();
    formData.append('file', file);
    if (sourceLang && sourceLang !== 'auto') {
      formData.append('source_lang', sourceLang);
    }
    if (targetLang && targetLang !== 'auto') {
      formData.append('target_lang', targetLang);
    }
    formData.append('synthesize_speech', synthesizeSpeech.toString());

    const res = await fetch(`${this.baseUrl}/api/translate-audio`, {
      method: 'POST',
      body: formData
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || `Upload translation failed with HTTP ${res.status}`);
    }
    return data;
  }

  async translateText(text, sourceLang = 'en', targetLang = 'hi', emotion = 'neutral', synthesizeSpeech = true) {
    if (!text || !text.trim()) {
      throw new Error('Please enter text to translate.');
    }

    const res = await fetch(`${this.baseUrl}/api/text-translate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text.trim(),
        source_lang: sourceLang,
        target_lang: targetLang,
        emotion: emotion,
        synthesize_speech: synthesizeSpeech
      })
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || `Text translation failed with HTTP ${res.status}`);
    }
    return data;
  }

  async getResearchOverview() {
    const res = await fetch(`${this.baseUrl}/api/v1/research/overview`);
    if (!res.ok) throw new Error(`Failed to fetch research overview: HTTP ${res.status}`);
    return await res.json();
  }

  async getModelHealth() {
    const res = await fetch(`${this.baseUrl}/api/v1/research/model-health`);
    if (!res.ok) throw new Error(`Failed to fetch model health: HTTP ${res.status}`);
    return await res.json();
  }

  async getPerformanceAnalytics() {
    const res = await fetch(`${this.baseUrl}/api/v1/research/performance`);
    if (!res.ok) throw new Error(`Failed to fetch performance analytics: HTTP ${res.status}`);
    return await res.json();
  }

  async getEvaluationMetrics() {
    const res = await fetch(`${this.baseUrl}/api/v1/research/evaluation-metrics`);
    if (!res.ok) throw new Error(`Failed to fetch evaluation metrics: HTTP ${res.status}`);
    return await res.json();
  }
}
