/**
 * VoiceAffect Results Renderer
 * Renders transcripts, dual audio players, emotion telemetry, VAD gauges, and latency
 */

const EMOTION_ICONS = {
  neutral: '😐',
  calm: '😌',
  happy: '😊',
  sad: '😔',
  angry: '😠',
  fearful: '😨',
  fear: '😨',
  disgust: '🤢',
  surprised: '😲',
  surprise: '😲',
  sarcastic: '😏',
  unavailable: '⚠️'
};

export class ResultsRenderer {
  constructor() {
    // Transcripts
    this.sourceTextEl = document.getElementById('sourceTranscriptText');
    this.translatedTextEl = document.getElementById('translatedTranscriptText');
    this.sourceLangTag = document.getElementById('sourceLangBadge');
    this.targetLangTag = document.getElementById('targetLangBadge');
    this.confidenceTag = document.getElementById('asrConfidenceBadge');

    // Audio Players
    this.originalAudioPlayer = document.getElementById('originalAudioPlayer');
    this.generatedAudioPlayer = document.getElementById('generatedAudioPlayer');
    this.btnDownloadAudio = document.getElementById('btnDownloadAudio');

    // Emotion Telemetry
    this.dominantEmotionBadge = document.getElementById('dominantEmotionBadge');
    this.emotionDistRows = {
      neutral: { bar: document.getElementById('fillNeutral'), label: document.getElementById('probNeutral') },
      calm: { bar: document.getElementById('fillCalm'), label: document.getElementById('probCalm') },
      happy: { bar: document.getElementById('fillHappy'), label: document.getElementById('probHappy') },
      sad: { bar: document.getElementById('fillSad'), label: document.getElementById('probSad') },
      angry: { bar: document.getElementById('fillAngry'), label: document.getElementById('probAngry') },
      fearful: { bar: document.getElementById('fillFearful'), label: document.getElementById('probFearful') },
      disgust: { bar: document.getElementById('fillDisgust'), label: document.getElementById('probDisgust') },
      surprised: { bar: document.getElementById('fillSurprised'), label: document.getElementById('probSurprised') },
      sarcastic: { bar: document.getElementById('fillSarcastic'), label: document.getElementById('probSarcastic') }
    };

    // 4-Stage Emotion & Voice Pipeline Grid (Phase 5)
    this.stageDetectedEmotionText = document.getElementById('stageDetectedEmotionText');
    this.stageEmotionMetaText = document.getElementById('stageEmotionMetaText');
    this.stageTranslationStatusText = document.getElementById('stageTranslationStatusText');
    this.stageTranslationMetaText = document.getElementById('stageTranslationMetaText');
    this.stageProsodyStatusText = document.getElementById('stageProsodyStatusText');
    this.stageProsodyMetaText = document.getElementById('stageProsodyMetaText');
    this.stageTtsStatusText = document.getElementById('stageTtsStatusText');
    this.stageTtsMetaText = document.getElementById('stageTtsMetaText');

    // VAD Gauges
    this.valValence = document.getElementById('valValence');
    this.fillValence = document.getElementById('fillValence');
    this.valArousal = document.getElementById('valArousal');
    this.fillArousal = document.getElementById('fillArousal');
    this.valDominance = document.getElementById('valDominance');
    this.fillDominance = document.getElementById('fillDominance');

    // Gating Plan
    this.gatingWeightVal = document.getElementById('gatingWeightVal');
    this.pitchMultVal = document.getElementById('pitchMultVal');
    this.rateMultVal = document.getElementById('rateMultVal');
    this.prosodyNotesText = document.getElementById('prosodyNotesText');

    // Phase 5 Emotion & Fidelity Audit
    this.emotionConfidenceBadge = document.getElementById('emotionConfidenceBadge');
    this.expressiveSynthesisBadge = document.getElementById('expressiveSynthesisBadge');
    this.synthesisVoiceUsedBadge = document.getElementById('synthesisVoiceUsedBadge');
    this.detectedGenderBadge = document.getElementById('detectedGenderBadge');
    this.speakerSimilarityBadge = document.getElementById('speakerSimilarityBadge');
    this.emotionalFidelityStatusText = document.getElementById('emotionalFidelityStatusText');
    this.languageWarningBox = document.getElementById('languageWarningBox');
    this.languageWarningText = document.getElementById('languageWarningText');
    this.emotionModelStatusBadge = document.getElementById('emotionModelStatusBadge');

    // Latency Dashboard
    this.latAsr = document.getElementById('latAsr');
    this.latSer = document.getElementById('latSer');
    this.latNmt = document.getElementById('latNmt');
    this.latTts = document.getElementById('latTts');
    this.totalLatencyPill = document.getElementById('totalLatencyPill');

    // Phase 6 Live UI & Playback Queue
    this.wsStatusPill = document.getElementById('wsStatusPill');
    this.wsStatusDot = document.getElementById('wsStatusDot');
    this.wsStatusText = document.getElementById('wsStatusText');
    this.playbackQueuePill = document.getElementById('playbackQueuePill');
    this.playbackQueueDot = document.getElementById('playbackQueueDot');
    this.playbackQueueText = document.getElementById('playbackQueueText');
    this.livePartialCaptionBox = document.getElementById('livePartialCaptionBox');
    this.livePartialCaptionText = document.getElementById('livePartialCaptionText');

    // Error Box
    this.errorBanner = document.getElementById('errorBanner');
    this.errorMessageText = document.getElementById('errorMessageText');

    // Cached audio data
    this.lastGeneratedBlobUrl = null;
    this.lastOriginalBlobUrl = null;
  }

  renderSuccess(data, originalAudioBlob = null, autoPlay = true) {
    this.hideError();

    // 1. Render Transcripts
    if (data.source_transcript !== undefined) {
      this.sourceTextEl.textContent = data.source_transcript || '(No audible speech detected)';
      this.sourceLangTag.textContent = data.source_language === 'hi' ? 'Hindi (हिंदी)' : (data.source_language === 'en' ? 'English' : data.source_language);
    }
    if (data.asr_confidence !== undefined && data.asr_confidence !== null) {
      this.confidenceTag.textContent = `Confidence: ${(data.asr_confidence * 100).toFixed(1)}%`;
    }

    if (data.translated_text !== undefined) {
      const isFailed = !data.translated_text && (
        data.translation_status === 'failed' ||
        data.diagnostics?.nmt_details?.status === 'translation_failed'
      );
      if (isFailed) {
        const errorMsg = data.translation_error || data.diagnostics?.nmt_details?.error || 'Translation failed';
        this.translatedTextEl.textContent = `[Translation Failed: ${errorMsg}]`;
      } else {
        this.translatedTextEl.textContent = data.translated_text || '(No translation output)';
      }
      this.targetLangTag.textContent = data.target_language === 'hi' ? 'Hindi Translation' : (data.target_language === 'en' ? 'English Translation' : data.target_language);
    }

    // 2. Original Audio Player
    if (originalAudioBlob) {
      if (this.lastOriginalBlobUrl) URL.revokeObjectURL(this.lastOriginalBlobUrl);
      this.lastOriginalBlobUrl = URL.createObjectURL(originalAudioBlob);
      this.originalAudioPlayer.src = this.lastOriginalBlobUrl;
      this.originalAudioPlayer.style.display = 'block';
    } else if (data.original_audio_url) {
      this.originalAudioPlayer.src = data.original_audio_url;
      this.originalAudioPlayer.style.display = 'block';
    }

    // 3. Generated Audio Player & Download
    let audioBlob = null;
    if (data.audio_base64) {
      audioBlob = this._base64ToBlob(data.audio_base64, 'audio/wav');
    }

    if (audioBlob) {
      if (this.lastGeneratedBlobUrl) URL.revokeObjectURL(this.lastGeneratedBlobUrl);
      this.lastGeneratedBlobUrl = URL.createObjectURL(audioBlob);
      this.generatedAudioPlayer.src = this.lastGeneratedBlobUrl;
      this.generatedAudioPlayer.style.display = 'block';

      this.btnDownloadAudio.disabled = false;
      this.btnDownloadAudio.onclick = () => {
        const a = document.createElement('a');
        a.href = this.lastGeneratedBlobUrl;
        a.download = `voiceaffect_translated_${data.target_language || 'speech'}.wav`;
        a.click();
      };

      if (autoPlay) {
        this.generatedAudioPlayer.currentTime = 0;
        this.generatedAudioPlayer.play().catch((err) => {
          console.warn('Browser policy prevented autoplay:', err);
        });
      }
    } else if (data.audio_url) {
      this.generatedAudioPlayer.src = data.audio_url;
      this.generatedAudioPlayer.style.display = 'block';
      this.btnDownloadAudio.disabled = false;
      this.btnDownloadAudio.onclick = () => {
        const a = document.createElement('a');
        a.href = data.audio_url;
        a.download = `voiceaffect_translated_${data.target_language || 'speech'}.wav`;
        a.click();
      };
      if (autoPlay) {
        this.generatedAudioPlayer.play().catch(() => {});
      }
    } else {
      this.generatedAudioPlayer.style.display = 'none';
      this.btnDownloadAudio.disabled = true;
    }

    // 4. Emotion Telemetry & Confidence (Phase 12: IEMOCAP & RAVDESS Dynamic Integration)
    const rawEmotion = (data.detected_emotion || 'neutral').toLowerCase();
    const isSerUnavailable = rawEmotion === 'unavailable' || data.ser_status === 'failed';
    const isInsufficientAudio = data.ser_status === 'skipped_silence' ||
                                data.diagnostics?.insufficient_audio ||
                                data.diagnostics?.status === 'silence_or_below_vad_threshold';
    const emotionIcon = EMOTION_ICONS[rawEmotion] || (isSerUnavailable ? '⚠️' : '😐');
    const emotionLabel = rawEmotion.charAt(0).toUpperCase() + rawEmotion.slice(1);

    const modelName = data.ser_model_id || (data.ser_model_provider === 'ravdess' ? 'Wav2Vec2-XLSR (RAVDESS)' : 'IEMOCAP Wav2Vec2-XLSR (4-class)');
    const serLatencyStr = (data.ser_latency_ms !== undefined && data.ser_latency_ms !== null)
      ? `${data.ser_latency_ms} ms`
      : (data.latency?.ser_ms ? `${data.latency.ser_ms} ms` : null);

    if (isSerUnavailable) {
      this.dominantEmotionBadge.textContent = 'Emotion detection unavailable';
      this.dominantEmotionBadge.className = 'emotion-badge-pill unavailable';
    } else if (isInsufficientAudio) {
      this.dominantEmotionBadge.textContent = 'Insufficient audio for emotion detection';
      this.dominantEmotionBadge.className = 'emotion-badge-pill neutral';
    } else {
      this.dominantEmotionBadge.textContent = `${emotionIcon} ${emotionLabel}`;
      this.dominantEmotionBadge.className = `emotion-badge-pill ${rawEmotion}`;
    }

    if (this.emotionModelStatusBadge) {
      if (isSerUnavailable) {
        this.emotionModelStatusBadge.textContent = 'SER Unavailable';
        this.emotionModelStatusBadge.className = 'tag-badge failed';
      } else {
        const prov = data.ser_model_provider ? data.ser_model_provider.toUpperCase() : 'IEMOCAP';
        this.emotionModelStatusBadge.textContent = `${prov} Ready`;
        this.emotionModelStatusBadge.className = 'tag-badge ready';
      }
    }

    if (this.emotionConfidenceBadge) {
      if (!isSerUnavailable && !isInsufficientAudio && data.emotion_confidence !== undefined && data.emotion_confidence !== null) {
        this.emotionConfidenceBadge.textContent = `Conf: ${(data.emotion_confidence * 100).toFixed(1)}%`;
      } else if (isInsufficientAudio) {
        this.emotionConfidenceBadge.textContent = 'Conf: N/A (Silence)';
      } else if (isSerUnavailable) {
        this.emotionConfidenceBadge.textContent = 'Conf: N/A';
      } else {
        this.emotionConfidenceBadge.textContent = 'Conf: --%';
      }
    }

    // Populate Native 4-class (IEMOCAP) or 8-class (Hindi / RAVDESS) Distribution
    const distScores = data.emotion_probabilities || data.native_emotion_scores || null;
    const is4Class = distScores && ('angry' in distScores) && !('calm' in distScores) && !('sarcastic' in distScores) && !('disgust' in distScores);

    for (const [emo, row] of Object.entries(this.emotionDistRows)) {
      if (!row || !row.bar || !row.label) continue;
      const parentRow = row.bar.closest('.emotion-dist-row');

      let val = null;
      if (distScores) {
        if (distScores[emo] !== undefined && distScores[emo] !== null) {
          val = distScores[emo];
        } else if (emo === 'fearful' && distScores['fear'] !== undefined) {
          val = distScores['fear'];
        } else if (emo === 'surprised' && distScores['surprise'] !== undefined) {
          val = distScores['surprise'];
        }
      }

      if (val !== null) {
        if (parentRow) parentRow.style.display = 'block';
        const pct = Math.round(val * 100);
        row.bar.style.width = `${pct}%`;
        row.label.textContent = `${pct}%`;
      } else if (is4Class && ['calm', 'fearful', 'disgust', 'surprised', 'sarcastic'].includes(emo)) {
        // Gracefully hide classes not in 4-class IEMOCAP configuration
        if (parentRow) parentRow.style.display = 'none';
        row.bar.style.width = '0%';
        row.label.textContent = '--';
      } else if (distScores && ['calm', 'sarcastic'].includes(emo)) {
        // Gracefully hide classes absent in the active 8-class configuration (e.g. calm for Hindi, sarcastic for RAVDESS)
        if (parentRow) parentRow.style.display = 'none';
        row.bar.style.width = '0%';
        row.label.textContent = '--';
      } else {
        if (parentRow) parentRow.style.display = 'block';
        row.bar.style.width = '0%';
        row.label.textContent = '--';
      }
    }

    // 5. Populate 4-Stage Pipeline Experience Grid
    // Stage 1: Detected Emotion
    if (this.stageDetectedEmotionText) {
      if (isSerUnavailable) {
        this.stageDetectedEmotionText.textContent = 'Emotion detection unavailable';
        if (this.stageEmotionMetaText) {
          this.stageEmotionMetaText.textContent = 'Translation continued without emotion information (Graceful degradation)';
        }
      } else if (isInsufficientAudio) {
        this.stageDetectedEmotionText.textContent = 'Insufficient audio for emotion detection';
        if (this.stageEmotionMetaText) {
          this.stageEmotionMetaText.textContent = 'Audio energy below VAD threshold; defaulted to neutral.';
        }
      } else {
        const confStr = (data.emotion_confidence !== undefined && data.emotion_confidence !== null)
          ? `${(data.emotion_confidence * 100).toFixed(1)}%`
          : '--%';
        this.stageDetectedEmotionText.textContent = `${emotionIcon} ${emotionLabel}`;
        if (this.stageEmotionMetaText) {
          const latPart = serLatencyStr ? ` · Time: ${serLatencyStr}` : '';
          this.stageEmotionMetaText.textContent = `Model: ${modelName} · Conf: ${confStr}${latPart} · Status: Active`;
        }
      }
    }

    // Stage 2: Sarvam Translation
    if (this.stageTranslationStatusText) {
      const isNmtFailed = !data.translated_text && (
        data.translation_status === 'failed' ||
        data.diagnostics?.nmt_details?.status === 'translation_failed'
      );
      if (isNmtFailed) {
        const errMsg = data.translation_error || data.diagnostics?.nmt_details?.error || 'Sarvam AI returned error';
        this.stageTranslationStatusText.textContent = '⚠️ Translation Failed';
        if (this.stageTranslationMetaText) {
          this.stageTranslationMetaText.textContent = `${errMsg} (Original transcript preserved)`;
        }
      } else {
        const providerName = (data.translation_provider || 'Sarvam AI').toUpperCase();
        const modelId = data.translation_model || 'mayura:v1';
        const sLangName = data.source_language === 'hi' ? 'Hindi' : 'English';
        const tLangName = data.target_language === 'hi' ? 'Hindi' : 'English';
        this.stageTranslationStatusText.textContent = `${providerName} (${modelId})`;
        if (this.stageTranslationMetaText) {
          this.stageTranslationMetaText.textContent = `${sLangName} ➔ ${tLangName} · Status: Verified`;
        }
      }
    }

    // Stage 3: Heuristic Prosody Adaptation
    if (this.stageProsodyStatusText) {
      const plan = data.preservation_plan;
      const rate = plan?.rate_multiplier ?? 1.0;
      const pitch = plan?.pitch_multiplier ?? 1.0;
      const zl = data.gating_weight_z ?? plan?.gating_weight_z;
      const zlStr = (zl !== undefined && zl !== null) ? Number(zl).toFixed(3) : '0.500';
      const isAdapted = (rate !== 1.0 || pitch !== 1.0);

      this.stageProsodyStatusText.textContent = isAdapted ? 'Heuristic Prosody Applied' : 'Baseline Prosody (Neutral)';
      if (this.stageProsodyMetaText) {
        this.stageProsodyMetaText.textContent = `Rate: ${rate}x · Pitch: ${pitch}x · Gating (zₗ): ${zlStr}`;
      }
    }

    // Stage 4: TTS Speech Output
    if (this.stageTtsStatusText) {
      const engineName = data.tts_engine || 'macOS say';
      const voiceUsed = data.synthesis_voice_used || (data.target_language === 'hi' ? 'Lekha (hi_IN)' : 'Samantha (en_US)');
      const isExpressive = Boolean(data.expressive_synthesis_supported);
      if (audioBlob || data.audio_url || data.audio_base64) {
        this.stageTtsStatusText.textContent = `${engineName.toUpperCase()} · ${voiceUsed}`;
        if (this.stageTtsMetaText) {
          this.stageTtsMetaText.textContent = isExpressive
            ? `Voice: ${voiceUsed} · Expressive neural synthesis active (${engineName})`
            : `Voice: ${voiceUsed} · Offline fallback (Heuristic tempo/pitch only)`;
        }
      } else if (data.tts_status === 'failed' || data.tts_status === 'unavailable') {
        this.stageTtsStatusText.textContent = '⚠️ TTS Unavailable';
        if (this.stageTtsMetaText) {
          this.stageTtsMetaText.textContent = 'No audio generated (Translation text preserved)';
        }
      } else {
        this.stageTtsStatusText.textContent = 'TTS Skipped';
        if (this.stageTtsMetaText) {
          this.stageTtsMetaText.textContent = 'Speech synthesis was not requested';
        }
      }
    }

    // 6. VAD Continuous Meters
    if (data.valence !== undefined && data.valence !== null) {
      this.valValence.textContent = data.valence.toFixed(2);
      this.fillValence.style.width = `${Math.round(data.valence * 100)}%`;
    } else {
      this.valValence.textContent = '--';
      this.fillValence.style.width = '0%';
    }

    if (data.arousal !== undefined && data.arousal !== null) {
      this.valArousal.textContent = data.arousal.toFixed(2);
      this.fillArousal.style.width = `${Math.round(data.arousal * 100)}%`;
    } else {
      this.valArousal.textContent = '--';
      this.fillArousal.style.width = '0%';
    }

    if (data.dominance !== undefined && data.dominance !== null) {
      this.valDominance.textContent = data.dominance.toFixed(2);
      this.fillDominance.style.width = `${Math.round(data.dominance * 100)}%`;
    } else {
      this.valDominance.textContent = '--';
      this.fillDominance.style.width = '0%';
    }

    // 6. Latent-Depth Gating & Prosody Plan
    if (data.gating_weight_z !== undefined && data.gating_weight_z !== null) {
      this.gatingWeightVal.textContent = data.gating_weight_z.toFixed(3);
    } else {
      this.gatingWeightVal.textContent = '--';
    }

    if (data.preservation_plan) {
      this.pitchMultVal.textContent = `${data.preservation_plan.pitch_multiplier || 1.0}x`;
      this.rateMultVal.textContent = `${data.preservation_plan.rate_multiplier || 1.0}x`;
      this.prosodyNotesText.textContent = data.preservation_plan.prosody_notes || 'Standard baseline prosody applied';
    } else {
      this.pitchMultVal.textContent = '1.0x';
      this.rateMultVal.textContent = '1.0x';
      this.prosodyNotesText.textContent = 'Standard baseline prosody applied';
    }

    // 7. Emotional Fidelity vs Synthesis Capability Audit (Phase 5)
    if (this.expressiveSynthesisBadge) {
      const isExpressive = Boolean(data.expressive_synthesis_supported);
      this.expressiveSynthesisBadge.textContent = isExpressive ? 'Supported' : 'Unsupported (Baseline Speech)';
      this.expressiveSynthesisBadge.style.background = isExpressive ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)';
      this.expressiveSynthesisBadge.style.color = isExpressive ? '#34d399' : '#f87171';
    }

    if (this.detectedGenderBadge) {
      const gender = data.detected_speaker_gender || (data.preservation_plan && data.preservation_plan.detected_gender);
      if (gender === 'male') {
        const confPct = Math.round((data.gender_confidence || 0.85) * 100);
        this.detectedGenderBadge.textContent = `👨 Male (${confPct}%)`;
        this.detectedGenderBadge.style.color = '#38bdf8';
      } else if (gender === 'female') {
        const confPct = Math.round((data.gender_confidence || 0.85) * 100);
        this.detectedGenderBadge.textContent = `👩 Female (${confPct}%)`;
        this.detectedGenderBadge.style.color = '#f472b6';
      } else {
        this.detectedGenderBadge.textContent = 'Auto-Detect';
        this.detectedGenderBadge.style.color = 'var(--color-accent-blue)';
      }
    }

    if (this.synthesisVoiceUsedBadge) {
      this.synthesisVoiceUsedBadge.textContent = data.synthesis_voice_used || (data.target_language === 'hi' ? 'Lekha (hi_IN)' : 'Samantha (en_US)');
    }

    if (this.speakerSimilarityBadge) {
      const retained = Boolean(data.speaker_similarity_retained);
      this.speakerSimilarityBadge.textContent = retained ? 'Retained (Authorized)' : 'Not Retained (Consent Gated)';
      this.speakerSimilarityBadge.style.color = retained ? 'var(--color-teal)' : 'var(--color-text-tertiary)';
    }

    if (this.emotionalFidelityStatusText) {
      const statusText = data.emotional_fidelity_status || (data.preservation_plan && data.preservation_plan.emotional_fidelity_status) || 'Heuristic prosody applied (Eq. 8 z_l). Emotional fidelity is NOT claimed without an end-to-end emotion vocoder.';
      this.emotionalFidelityStatusText.textContent = statusText;
    }

    if (this.languageWarningBox && this.languageWarningText) {
      if (data.language_limitation_warning) {
        this.languageWarningBox.style.display = 'flex';
        this.languageWarningText.textContent = data.language_limitation_warning;
      } else {
        this.languageWarningBox.style.display = 'none';
      }
    }

    // 8. Measured Latencies
    if (data.latency) {
      this.latAsr.textContent = `${Math.round(data.latency.asr_ms || 0)} ms`;
      this.latSer.textContent = data.latency.ser_ms > 0 ? `${Math.round(data.latency.ser_ms)} ms` : `${Math.round(data.latency.ser_ms || 0)} ms`;
      this.latNmt.textContent = `${Math.round(data.latency.nmt_ms || 0)} ms`;
      this.latTts.textContent = `${Math.round(data.latency.tts_ms || 0)} ms`;
      this.totalLatencyPill.textContent = `Total: ${Math.round(data.latency.total_pipeline_ms || 0)} ms`;
    }

  }

  showError(message, retryCallback = null) {
    this.errorBanner.style.display = 'flex';
    this.errorMessageText.textContent = message;

    const btnRetry = document.getElementById('btnRetryAction');
    if (retryCallback && btnRetry) {
      btnRetry.style.display = 'inline-flex';
      btnRetry.onclick = () => {
        this.hideError();
        retryCallback();
      };
    } else if (btnRetry) {
      btnRetry.style.display = 'none';
    }
  }

  renderPartialTranscript(text) {
    if (this.livePartialCaptionBox && this.livePartialCaptionText) {
      this.livePartialCaptionText.textContent = text;
      this.livePartialCaptionBox.style.display = 'block';
    }
  }

  hidePartialTranscript() {
    if (this.livePartialCaptionBox) {
      this.livePartialCaptionBox.style.display = 'none';
      if (this.livePartialCaptionText) this.livePartialCaptionText.textContent = '';
    }
  }

  renderWsStatus(status) {
    if (!this.wsStatusPill || !this.wsStatusDot || !this.wsStatusText) return;

    if (status === 'connected') {
      this.wsStatusDot.style.background = '#10b981'; // Green
      this.wsStatusText.textContent = 'WS: Connected';
      this.wsStatusPill.style.color = '#34d399';
    } else if (status === 'connecting') {
      this.wsStatusDot.style.background = '#f59e0b'; // Amber
      this.wsStatusText.textContent = 'WS: Connecting';
      this.wsStatusPill.style.color = '#fbbf24';
    } else if (status === 'error') {
      this.wsStatusDot.style.background = '#ef4444'; // Red
      this.wsStatusText.textContent = 'WS: Error';
      this.wsStatusPill.style.color = '#f87171';
    } else {
      this.wsStatusDot.style.background = '#94a3b8'; // Gray
      this.wsStatusText.textContent = 'WS: Idle';
      this.wsStatusPill.style.color = 'var(--color-text-secondary)';
    }
  }

  renderPlaybackQueueStatus(status) {
    if (!this.playbackQueuePill || !this.playbackQueueDot || !this.playbackQueueText) return;

    if (status.isPlaying) {
      this.playbackQueueDot.style.background = '#06b6d4'; // Cyan
      const queueSuffix = status.queueLength > 0 ? ` (+${status.queueLength} waiting)` : '';
      this.playbackQueueText.textContent = `Playing #${status.currentPlayingSegmentId}${queueSuffix}`;
      this.playbackQueuePill.style.color = 'var(--color-cyan)';
    } else if (status.queueLength > 0) {
      this.playbackQueueDot.style.background = '#f59e0b'; // Amber
      this.playbackQueueText.textContent = `Queue: ${status.queueLength} pending`;
      this.playbackQueuePill.style.color = '#fbbf24';
    } else {
      this.playbackQueueDot.style.background = '#94a3b8'; // Gray
      this.playbackQueueText.textContent = 'Queue: Idle';
      this.playbackQueuePill.style.color = 'var(--color-text-secondary)';
    }
  }

  hideError() {
    this.errorBanner.style.display = 'none';
  }

  _base64ToBlob(base64Data, contentType = 'audio/wav') {
    const byteCharacters = atob(base64Data);
    const byteArrays = [];
    const sliceSize = 512;
    for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
      const slice = byteCharacters.slice(offset, offset + sliceSize);
      const byteNumbers = new Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        byteNumbers[i] = slice.charCodeAt(i);
      }
      byteArrays.push(new Uint8Array(byteNumbers));
    }
    return new Blob(byteArrays, { type: contentType });
  }
}
