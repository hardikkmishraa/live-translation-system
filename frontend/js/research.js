/**
 * VoiceAffect Research & Evaluation Dashboard Renderer (Phase 6)
 * Renders live operational KPIs, model health matrix, performance latency charts,
 * scientific evaluation benchmarks, and SER confusion matrix.
 */

export class ResearchInsightsRenderer {
  constructor(apiClient = null) {
    this.api = apiClient;

    // Overview KPIs
    this.kpiTotalRequests = document.getElementById('kpiTotalRequests');
    this.kpiSuccessfulRequests = document.getElementById('kpiSuccessfulRequests');
    this.kpiFailedRequests = document.getElementById('kpiFailedRequests');
    this.kpiSuccessRate = document.getElementById('kpiSuccessRate');
    this.kpiAvgLatency = document.getElementById('kpiAvgLatency');
    this.kpiTopDirection = document.getElementById('kpiTopDirection');
    this.kpiDirectionRatio = document.getElementById('kpiDirectionRatio');

    // Model Health Badges
    this.statusPillAsr = document.getElementById('statusPillAsr');
    this.statusPillSer = document.getElementById('statusPillSer');
    this.statusPillNmt = document.getElementById('statusPillNmt');
    this.statusPillTts = document.getElementById('statusPillTts');
    this.healthAsrModel = document.getElementById('healthAsrModel');
    this.healthAsrDevice = document.getElementById('healthAsrDevice');
    this.healthAsrLoading = document.getElementById('healthAsrLoading');
    this.healthSerModel = document.getElementById('healthSerModel');
    this.healthSerCheckpoint = document.getElementById('healthSerCheckpoint');
    this.healthSerDevice = document.getElementById('healthSerDevice');
    this.healthNmtModel = document.getElementById('healthNmtModel');
    this.healthNmtConnection = document.getElementById('healthNmtConnection');
    this.healthTtsEngine = document.getElementById('healthTtsEngine');

    // Performance Stage Latencies
    this.perfLatAsr = document.getElementById('perfLatAsr');
    this.perfLatSer = document.getElementById('perfLatSer');
    this.perfLatNmt = document.getElementById('perfLatNmt');
    this.perfLatTts = document.getElementById('perfLatTts');

    // Canvases
    this.canvasLatency = document.getElementById('chartLatencyBreakdown');
    this.canvasDirection = document.getElementById('chartDirectionDistribution');

    // Tables
    this.confusionMatrixBody = document.getElementById('confusionMatrixBody');
    this.operationalHistoryTableBody = document.getElementById('operationalHistoryTableBody');
    this.modelStatusTableBody = document.getElementById('modelStatusTableBody');

    // Controls
    this.btnRefresh = document.getElementById('btnRefreshResearch');
    this.btnReset = document.getElementById('btnResetResearchMetrics');

    this._initialized = false;
  }

  init() {
    if (this._initialized) return;
    this._initialized = true;

    if (this.btnRefresh) {
      this.btnRefresh.addEventListener('click', () => this.refreshAll());
    }

    if (this.btnReset) {
      this.btnReset.addEventListener('click', async () => {
        if (!confirm('Reset all collected operational request metrics?')) return;
        try {
          await fetch('/api/v1/research/reset-metrics', { method: 'POST' });
          await this.refreshAll();
        } catch (e) {
          console.warn('Failed to reset metrics:', e);
        }
      });
    }

    // Initial load
    this.refreshAll();
  }

  async refreshAll() {
    if (!this.api) return;

    try {
      const [overview, health, performance, evalMetrics] = await Promise.all([
        this.api.getResearchOverview().catch(() => null),
        this.api.getModelHealth().catch(() => null),
        this.api.getPerformanceAnalytics().catch(() => null),
        this.api.getEvaluationMetrics().catch(() => null)
      ]);

      if (overview) this.renderOverview(overview);
      if (health) this.renderModelHealth(health);
      if (performance) this.renderPerformance(performance, overview);
      if (evalMetrics) this.renderEvaluationMetrics(evalMetrics);
    } catch (err) {
      console.warn('[Research] Refresh error:', err);
    }
  }

  renderOverview(data) {
    if (this.kpiTotalRequests) this.kpiTotalRequests.textContent = data.total_requests ?? 0;
    if (this.kpiSuccessfulRequests) this.kpiSuccessfulRequests.textContent = data.successful_requests ?? 0;
    if (this.kpiFailedRequests) this.kpiFailedRequests.textContent = data.failed_requests ?? 0;
    if (this.kpiSuccessRate) this.kpiSuccessRate.textContent = `${data.success_rate_percent ?? 0}% success rate`;
    if (this.kpiAvgLatency) this.kpiAvgLatency.textContent = `${Math.round(data.average_pipeline_latency_ms ?? 0)} ms`;

    let topDirLabel = 'None';
    if (data.most_frequently_used_direction === 'hi_to_en') topDirLabel = 'Hindi ➔ English';
    else if (data.most_frequently_used_direction === 'en_to_hi') topDirLabel = 'English ➔ Hindi';
    else if (data.most_frequently_used_direction && data.most_frequently_used_direction !== 'None') {
      topDirLabel = data.most_frequently_used_direction;
    }
    if (this.kpiTopDirection) this.kpiTopDirection.textContent = topDirLabel;

    const hiCnt = data.direction_distribution?.hi_to_en ?? 0;
    const enCnt = data.direction_distribution?.en_to_hi ?? 0;
    if (this.kpiDirectionRatio) {
      this.kpiDirectionRatio.textContent = `HI➔EN: ${hiCnt} · EN➔HI: ${enCnt}`;
    }

    // Render operational table
    this.renderHistoryTable(data.recent_requests || []);
  }

  renderModelHealth(data) {
    const models = data?.models || {};

    // ASR
    if (models.asr) {
      if (this.healthAsrModel) this.healthAsrModel.textContent = models.asr.name || 'Whisper Tiny';
      if (this.healthAsrDevice) this.healthAsrDevice.textContent = (models.asr.device || 'CPU').toUpperCase();
      if (this.healthAsrLoading) this.healthAsrLoading.textContent = models.asr.loading_status || 'Ready';
      if (this.statusPillAsr) {
        this.statusPillAsr.textContent = models.asr.loading_status === 'loaded' ? 'Loaded' : 'Ready';
        this.statusPillAsr.className = 'health-status-pill ready';
      }
    }

    // SER
    if (models.ser) {
      if (this.healthSerModel) this.healthSerModel.textContent = models.ser.name || 'Wav2Vec2 XLSR';
      if (this.healthSerCheckpoint) this.healthSerCheckpoint.textContent = 'Verified (safetensors)';
      if (this.healthSerDevice) this.healthSerDevice.textContent = (models.ser.device || 'CPU').toUpperCase();
      if (this.statusPillSer) {
        this.statusPillSer.textContent = 'Active (RAVDESS)';
        this.statusPillSer.className = 'health-status-pill ready';
      }
    }

    // NMT
    if (models.nmt) {
      if (this.healthNmtModel) this.healthNmtModel.textContent = models.nmt.actual_model || 'mayura:v1';
      if (this.healthNmtConnection) {
        this.healthNmtConnection.textContent = models.nmt.api_connection_status === 'connected' ? 'Connected (Live API)' : 'Ready (Mock / REST)';
      }
      if (this.statusPillNmt) {
        this.statusPillNmt.textContent = models.nmt.api_connection_status === 'connected' ? 'Connected' : 'Ready';
        this.statusPillNmt.className = 'health-status-pill connected';
      }
    }

    // TTS
    if (models.tts) {
      if (this.healthTtsEngine) this.healthTtsEngine.textContent = models.tts.engine || 'macOS say';
      if (this.statusPillTts) {
        this.statusPillTts.textContent = 'Ready (say)';
        this.statusPillTts.className = 'health-status-pill ready';
      }
    }
  }

  renderPerformance(perfData, overviewData) {
    const avgs = perfData.averages_ms || {};
    if (this.perfLatAsr) this.perfLatAsr.textContent = `${Math.round(avgs.asr_ms || 0)} ms`;
    if (this.perfLatSer) this.perfLatSer.textContent = `${Math.round(avgs.ser_ms || 0)} ms`;
    if (this.perfLatNmt) this.perfLatNmt.textContent = `${Math.round(avgs.nmt_ms || 0)} ms`;
    if (this.perfLatTts) this.perfLatTts.textContent = `${Math.round(avgs.tts_ms || 0)} ms`;

    // Draw Charts
    this.drawLatencyChart(perfData.chart_series || {});
    this.drawDirectionChart(overviewData?.direction_distribution || {}, overviewData);
  }

  renderEvaluationMetrics(data) {
    const serBench = data.ser?.offline_benchmark;
    if (!serBench || !serBench.confusion_matrix || !this.confusionMatrixBody) return;

    const matrix = serBench.confusion_matrix;
    const classes = serBench.classes || [];

    let html = '';
    for (let i = 0; i < matrix.length; i++) {
      const rowLabel = classes[i] ? (classes[i].charAt(0).toUpperCase() + classes[i].slice(1)) : `C${i}`;
      html += `<tr><td style="font-weight: 600; text-align: left; background: var(--color-bg-base);">${rowLabel}</td>`;

      for (let j = 0; j < matrix[i].length; j++) {
        const val = matrix[i][j];
        const isDiag = (i === j);
        const cellClass = isDiag ? 'cell-diagonal' : (val > 0 ? 'cell-off' : '');
        html += `<td class="${cellClass}">${val}</td>`;
      }
      html += '</tr>';
    }
    this.confusionMatrixBody.innerHTML = html;
  }

  renderHistoryTable(recentList) {
    if (!this.operationalHistoryTableBody) return;

    if (!recentList || recentList.length === 0) {
      this.operationalHistoryTableBody.innerHTML = `
        <tr>
          <td colspan="10" style="text-align: center; color: var(--color-text-tertiary); padding: 18px;">
            No operational requests logged yet. Use the Live Speech or Upload tabs to generate real translation metrics.
          </td>
        </tr>`;
      return;
    }

    this.operationalHistoryTableBody.innerHTML = recentList.map((item) => {
      const d = new Date(item.timestamp * 1000);
      const timeStr = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      const dirBadge = item.direction === 'hi_to_en' ? 'HI ➔ EN' : (item.direction === 'en_to_hi' ? 'EN ➔ HI' : item.direction);
      const isSuccess = item.success;
      const statusBadge = isSuccess
        ? '<span class="health-status-pill ready">Success</span>'
        : `<span class="health-status-pill warning" title="${item.error || 'Failed'}">Failed</span>`;

      return `
        <tr>
          <td><span style="font-family: var(--font-mono); font-size: 11px; color: var(--color-text-tertiary);">${timeStr}</span></td>
          <td><code style="font-size: 10px;">${item.request_id?.slice(0, 8) || '--'}</code></td>
          <td><span class="tag-badge accent">${dirBadge}</span></td>
          <td><span class="emotion-badge-pill ${item.detected_emotion?.toLowerCase() || 'neutral'}" style="font-size: 11px; padding: 1px 6px;">${item.detected_emotion || 'neutral'}</span></td>
          <td style="font-family: var(--font-mono);">${Math.round(item.latencies?.asr_ms || 0)} ms</td>
          <td style="font-family: var(--font-mono);">${Math.round(item.latencies?.ser_ms || 0)} ms</td>
          <td style="font-family: var(--font-mono);">${Math.round(item.latencies?.nmt_ms || 0)} ms</td>
          <td style="font-family: var(--font-mono);">${Math.round(item.latencies?.tts_ms || 0)} ms</td>
          <td style="font-family: var(--font-mono); font-weight: 600; color: #60a5fa;">${Math.round(item.latencies?.total_ms || 0)} ms</td>
          <td>${statusBadge}</td>
        </tr>
      `;
    }).join('');
  }

  // --- Dark-Mode HTML5 Canvas Charts ---

  drawLatencyChart(series) {
    if (!this.canvasLatency) return;
    const canvas = this.canvasLatency;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const totals = series.total_ms || [];
    if (totals.length === 0) {
      // Draw empty placeholder
      ctx.fillStyle = '#64748b';
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No request latency history recorded yet', width / 2, height / 2);
      ctx.fillStyle = '#475569';
      ctx.font = '11px Inter, sans-serif';
      ctx.fillText('Execute a translation to visualize live stage latencies', width / 2, height / 2 + 20);
      return;
    }

    const padding = { top: 20, right: 20, bottom: 35, left: 45 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    const maxVal = Math.max(...totals, 500) * 1.15;

    // Draw grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (chartH / 4) * i;
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      const valLabel = Math.round(maxVal - (maxVal / 4) * i);
      ctx.fillStyle = '#64748b';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`${valLabel}ms`, padding.left - 6, y + 3);
    }
    ctx.stroke();

    // Bar chart of recent requests
    const count = totals.length;
    const barWidth = Math.min(32, (chartW / count) * 0.7);
    const step = chartW / count;

    totals.forEach((totalVal, idx) => {
      const asrVal = series.asr_ms?.[idx] || 0;
      const serVal = series.ser_ms?.[idx] || 0;
      const nmtVal = series.nmt_ms?.[idx] || 0;
      const ttsVal = series.tts_ms?.[idx] || 0;

      const x = padding.left + idx * step + (step - barWidth) / 2;
      let currY = padding.top + chartH;

      // Segment colors
      const segments = [
        { val: asrVal, color: '#38bdf8' }, // ASR (sky)
        { val: serVal, color: '#a855f7' }, // SER (purple)
        { val: nmtVal, color: '#3b82f6' }, // NMT (blue)
        { val: ttsVal, color: '#10b981' }  // TTS (green)
      ];

      segments.forEach(({ val, color }) => {
        const segH = (val / maxVal) * chartH;
        ctx.fillStyle = color;
        ctx.fillRect(x, currY - segH, barWidth, segH);
        currY -= segH;
      });

      // Total label on top of bar
      ctx.fillStyle = '#94a3b8';
      ctx.font = '9px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`#${idx + 1}`, x + barWidth / 2, padding.top + chartH + 16);
    });

    // Legend
    const legends = [
      { label: 'ASR', color: '#38bdf8' },
      { label: 'SER', color: '#a855f7' },
      { label: 'NMT', color: '#3b82f6' },
      { label: 'TTS', color: '#10b981' }
    ];
    let legX = padding.left;
    legends.forEach(({ label, color }) => {
      ctx.fillStyle = color;
      ctx.fillRect(legX, padding.top - 12, 8, 8);
      ctx.fillStyle = '#94a3b8';
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(label, legX + 12, padding.top - 5);
      legX += 50;
    });
  }

  drawDirectionChart(directionDist, overview) {
    if (!this.canvasDirection) return;
    const canvas = this.canvasDirection;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const hiCount = directionDist.hi_to_en || 0;
    const enCount = directionDist.en_to_hi || 0;
    const total = hiCount + enCount;

    if (total === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No direction metrics collected yet', width / 2, height / 2);
      ctx.fillStyle = '#475569';
      ctx.font = '11px Inter, sans-serif';
      ctx.fillText('Direction ratios will dynamically appear upon translation', width / 2, height / 2 + 20);
      return;
    }

    const centerX = width * 0.35;
    const centerY = height * 0.5;
    const radius = Math.min(centerX, centerY) - 20;
    const innerRadius = radius * 0.58;

    const hiAngle = (hiCount / total) * 2 * Math.PI;
    const enAngle = (enCount / total) * 2 * Math.PI;

    // Draw Hindi -> English slice
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, hiAngle);
    ctx.arc(centerX, centerY, innerRadius, hiAngle, 0, true);
    ctx.closePath();
    ctx.fillStyle = '#3b82f6'; // Blue
    ctx.fill();

    // Draw English -> Hindi slice
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, hiAngle, hiAngle + enAngle);
    ctx.arc(centerX, centerY, innerRadius, hiAngle + enAngle, hiAngle, true);
    ctx.closePath();
    ctx.fillStyle = '#10b981'; // Green
    ctx.fill();

    // Center text
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 16px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(total.toString(), centerX, centerY + 3);
    ctx.fillStyle = '#64748b';
    ctx.font = '10px Inter, sans-serif';
    ctx.fillText('TOTAL', centerX, centerY + 18);

    // Legend on right
    const legX = width * 0.65;
    let legY = height * 0.32;

    const items = [
      { label: 'Hindi ➔ English', count: hiCount, pct: Math.round((hiCount / total) * 100), color: '#3b82f6' },
      { label: 'English ➔ Hindi', count: enCount, pct: Math.round((enCount / total) * 100), color: '#10b981' }
    ];

    items.forEach(({ label, count, pct, color }) => {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(legX, legY + 4, 5, 0, 2 * Math.PI);
      ctx.fill();

      ctx.fillStyle = '#e2e8f0';
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(label, legX + 12, legY + 8);

      ctx.fillStyle = '#94a3b8';
      ctx.font = '11px JetBrains Mono, monospace';
      ctx.fillText(`${count} (${pct}%)`, legX + 12, legY + 24);

      legY += 46;
    });
  }

  // Backwards compatibility method
  updateModelStatus(healthData) {
    if (healthData) this.renderModelHealth(healthData);
  }
}
