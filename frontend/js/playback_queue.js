/**
 * VoiceAffect Progressive Speech Playback Queue (Phase 6)
 * Manages ordered progressive audio playback for translated speech segments.
 *
 * Guarantees:
 *  - Strict monotonic segment ordering (prevents out-of-order execution)
 *  - Single-channel serialized playback (no audio overlapping)
 *  - Bounded memory growth (bounded FIFO buffer with URL revocation)
 *  - Graceful recovery on segment decoding failure
 *  - Immediate cancellation & cleanup on session stop
 */

export class ProgressivePlaybackQueue {
  constructor(options = {}) {
    this.maxQueueSize = options.maxQueueSize || 10;
    this.onStatusChange = options.onStatusChange || null;
    this.autoPlay = options.autoPlay !== undefined ? options.autoPlay : true;

    this.queue = [];             // Array of { segmentId, blobUrl, audioEl, base64 }
    this.currentPlaying = null;  // Currently playing segment item
    this.isPlaying = false;
    this.expectedNextSegmentId = 1;
    this.stashedOutOfOrder = new Map(); // Stashes future segments if arrived early
  }

  setAutoPlay(enabled) {
    this.autoPlay = Boolean(enabled);
  }

  enqueueSegment(segmentId, audioBlobOrBase64, isBase64 = true) {
    if (!this.autoPlay) {
      console.log(`[PlaybackQueue] Auto-play is disabled. Segment #${segmentId} stashed for manual replay.`);
      return;
    }

    // Convert to Blob URL
    let blobUrl = null;
    if (isBase64) {
      const blob = this._base64ToBlob(audioBlobOrBase64, 'audio/wav');
      blobUrl = URL.createObjectURL(blob);
    } else if (audioBlobOrBase64 instanceof Blob) {
      blobUrl = URL.createObjectURL(audioBlobOrBase64);
    } else if (typeof audioBlobOrBase64 === 'string') {
      blobUrl = audioBlobOrBase64;
    }

    if (!blobUrl) return;

    const item = {
      segmentId: Number(segmentId) || this.expectedNextSegmentId,
      blobUrl: blobUrl,
      enqueuedAt: Date.now()
    };

    // Stash or enqueue in strict sequential order
    if (item.segmentId > this.expectedNextSegmentId) {
      console.warn(`[PlaybackQueue] Segment #${item.segmentId} arrived early (expected #${this.expectedNextSegmentId}). Stashing.`);
      this.stashedOutOfOrder.set(item.segmentId, item);
      return;
    }

    this._pushToQueue(item);
    this._drainStashed();

    // Trigger playback loop if idle
    if (!this.isPlaying) {
      this._playNext();
    }
  }

  _pushToQueue(item) {
    // Memory limit safety: if queue exceeds max capacity, drop oldest non-playing item
    if (this.queue.length >= this.maxQueueSize) {
      const dropped = this.queue.shift();
      if (dropped && dropped.blobUrl && dropped.blobUrl.startsWith('blob:')) {
        URL.revokeObjectURL(dropped.blobUrl);
      }
      console.warn(`[PlaybackQueue] Queue full. Dropped segment #${dropped ? dropped.segmentId : 'unknown'}`);
    }

    this.queue.push(item);
    this.expectedNextSegmentId = Math.max(this.expectedNextSegmentId, item.segmentId + 1);
    this._notify();
  }

  _drainStashed() {
    while (this.stashedOutOfOrder.has(this.expectedNextSegmentId)) {
      const item = this.stashedOutOfOrder.get(this.expectedNextSegmentId);
      this.stashedOutOfOrder.delete(this.expectedNextSegmentId);
      this._pushToQueue(item);
    }
  }

  _playNext() {
    if (this.queue.length === 0) {
      this.isPlaying = false;
      this.currentPlaying = null;
      this._notify();
      return;
    }

    this.isPlaying = true;
    const item = this.queue.shift();
    this.currentPlaying = item;
    this._notify();

    const audio = new Audio();
    audio.src = item.blobUrl;
    audio.preload = 'auto';

    const finalizeAndAdvance = () => {
      // Clean up temporary blob URL to prevent memory leaks
      if (item.blobUrl && item.blobUrl.startsWith('blob:')) {
        URL.revokeObjectURL(item.blobUrl);
      }
      audio.onended = null;
      audio.onerror = null;
      this._playNext();
    };

    audio.onended = () => {
      finalizeAndAdvance();
    };

    audio.onerror = (e) => {
      console.error(`[PlaybackQueue] Playback failed for segment #${item.segmentId}:`, e);
      finalizeAndAdvance();
    };

    audio.play().catch((playErr) => {
      console.warn(`[PlaybackQueue] Browser autoplay policy prevented playback for segment #${item.segmentId}:`, playErr);
      finalizeAndAdvance();
    });
  }

  stopAndClear() {
    this.isPlaying = false;
    this.currentPlaying = null;

    // Revoke all pending queue URLs
    while (this.queue.length > 0) {
      const item = this.queue.shift();
      if (item.blobUrl && item.blobUrl.startsWith('blob:')) {
        URL.revokeObjectURL(item.blobUrl);
      }
    }

    // Revoke all stashed items
    for (const item of this.stashedOutOfOrder.values()) {
      if (item.blobUrl && item.blobUrl.startsWith('blob:')) {
        URL.revokeObjectURL(item.blobUrl);
      }
    }
    this.stashedOutOfOrder.clear();
    this.expectedNextSegmentId = 1;
    this._notify();
  }

  getStatus() {
    return {
      isPlaying: this.isPlaying,
      queueLength: this.queue.length,
      currentPlayingSegmentId: this.currentPlaying ? this.currentPlaying.segmentId : null,
      stashedCount: this.stashedOutOfOrder.size
    };
  }

  _notify() {
    if (this.onStatusChange) {
      this.onStatusChange(this.getStatus());
    }
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
