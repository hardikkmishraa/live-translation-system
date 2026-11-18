/**
 * VoiceAffect WebSocket Manager (Phase 6)
 * Manages live full-duplex speech translation streaming over /ws/live-translate
 * Implementing complete session protocol, acknowledgements, partial captions,
 * segment sequencing, heartbeat, and graceful error handling.
 */

export class LiveWebSocketClient {
  constructor(endpoint = '/ws/live-translate') {
    this.endpoint = endpoint;
    this.ws = null;
    this.sessionId = null;
    this.isConnected = false;
    this.isConnecting = false;
    this.connectionPromise = null;

    // Callbacks
    this.onMessageCallback = null;
    this.onStatusChangeCallback = null;
    this.onErrorCallback = null;

    // Reconnection & Heartbeat
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 3;
    this.pingInterval = null;
  }

  connect(sessionId = null, sourceLang = 'auto', targetLang = 'auto', userConsent = false) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return Promise.resolve();
    }
    if (this.ws && this.ws.readyState === WebSocket.CONNECTING) {
      return this.connectionPromise;
    }

    this.sessionId = sessionId || Math.random().toString(36).substring(2, 10);
    this.isConnecting = true;
    this._notifyStatus('connecting');

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let host = window.location.host;
    if (!host || window.location.hostname === 'localhost') {
      const port = window.location.port || '8000';
      host = `127.0.0.1:${port}`;
    }
    const wsUrl = `${protocol}//${host}${this.endpoint}`;

    this.connectionPromise = new Promise((resolve, reject) => {
      let settled = false;
      const finish = (error = null) => {
        if (settled) return;
        settled = true;
        clearTimeout(connectionTimeout);
        this.connectionPromise = null;
        if (error) reject(error);
        else resolve();
      };

      const connectionTimeout = setTimeout(() => {
        this.isConnecting = false;
        this._notifyStatus('error');
        const error = new Error(`WebSocket connection timed out: ${wsUrl}`);
        if (this.onErrorCallback) {
          this.onErrorCallback({
            type: 'error',
            error_code: 'CONNECTION_TIMEOUT',
            message: error.message,
            recoverable: true
          });
        }
        finish(error);
        if (this.ws) this.ws.close();
      }, 10000);

      try {
        this.ws = new WebSocket(wsUrl);
        this.ws.binaryType = 'arraybuffer';

        this.ws.onopen = () => {
          this.isConnected = true;
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this._notifyStatus('connected');

          this.sendControl({
            action: 'session_start',
            session_id: this.sessionId,
            source_lang: sourceLang,
            target_lang: targetLang,
            user_consent: Boolean(userConsent)
          });

          this._startHeartbeat();
          finish();
        };

        this.ws.onmessage = (event) => {
          if (typeof event.data === 'string') {
            try {
              const parsed = JSON.parse(event.data);
              if (this.onMessageCallback) {
                this.onMessageCallback(parsed);
              }
            } catch (e) {
              console.error('[WebSocket] Failed to parse text message:', e);
            }
          }
        };

        this.ws.onerror = (err) => {
          console.error('[WebSocket] Connection error:', err);
          this.isConnecting = false;
          this._notifyStatus('error');
          const error = new Error(`WebSocket failed to connect to ${wsUrl}. Verify the server is running on http://${host}.`);
          if (this.onErrorCallback) {
            this.onErrorCallback({
              type: 'error',
              error_code: 'CONNECTION_ERROR',
              message: error.message,
              recoverable: true
            });
          }
          finish(error);
        };

        this.ws.onclose = (event) => {
          this.isConnected = false;
          this.isConnecting = false;
          this._stopHeartbeat();
          this._notifyStatus('disconnected');
          if (!settled) {
            finish(new Error(`WebSocket closed before connecting (code ${event.code}).`));
          }
          console.log(`[WebSocket] Closed (Code: ${event.code}, Clean: ${event.wasClean})`);
        };

      } catch (err) {
        this.isConnecting = false;
        this._notifyStatus('error');
        if (this.onErrorCallback) {
          this.onErrorCallback({
            type: 'error',
            error_code: 'INIT_FAILED',
            message: err.message,
            recoverable: true
          });
        }
        finish(err);
      }
    });

    return this.connectionPromise;
  }

  sendAudioChunk(pcmArrayBuffer) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(pcmArrayBuffer);
      return true;
    }
    return false;
  }

  sendControl(controlObj) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(controlObj));
      return true;
    }
    return false;
  }

  flush() {
    return this.sendControl({ action: 'flush' });
  }

  cancel() {
    return this.sendControl({ action: 'cancel' });
  }

  stopSession() {
    return this.sendControl({ action: 'stop' });
  }

  disconnect() {
    this._stopHeartbeat();
    if (this.ws) {
      try {
        this.ws.close();
      } catch (_) {}
      this.ws = null;
    }
    this.isConnected = false;
    this.isConnecting = false;
    this._notifyStatus('disconnected');
  }

  _startHeartbeat() {
    this._stopHeartbeat();
    this.pingInterval = setInterval(() => {
      if (this.isConnected) {
        this.sendControl({ action: 'heartbeat', timestamp: Date.now() });
      }
    }, 15000);
  }

  _stopHeartbeat() {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  _notifyStatus(status) {
    if (this.onStatusChangeCallback) {
      this.onStatusChangeCallback(status);
    }
  }
}
