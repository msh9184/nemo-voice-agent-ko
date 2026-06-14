/**
 * Copyright (c) 2024–2025, Daily
 *
 * SPDX-License-Identifier: BSD 2-Clause License
 */

/**
 * NeMo Voice Agent - Premium STT Client with Runtime Configuration
 *
 * Features:
 * - Real-time speech-to-text with flowing partial-to-final paradigm
 * - Enhanced audio visualization with Web Audio API
 * - Runtime configuration panel for VAD, Aggregator, STT, and Diarization
 * - VAD (Voice Activity Detection) status display
 * - Professional waveform and frequency visualization
 * - Speaker diarization with smooth speaker transitions
 * - Premium dark theme UI inspired by ElevenLabs/Soniox
 */

import {
  RTVIClient,
  RTVIClientOptions,
  RTVIEvent,
} from '@pipecat-ai/client-js';
import {
  WebSocketTransport,
  ProtobufFrameSerializer
} from "@pipecat-ai/websocket-transport";

// ============================================================================
// GLOBAL WEBSOCKET INTERCEPTOR
// Patches WebSocket to capture messages BEFORE pipecat's Protobuf deserializer
// This is necessary because pipecat rejects plain JSON messages with "Unknown data type"
// ============================================================================
type WebSocketMessageHandler = (data: any) => void;

const globalWebSocketHandlers: WebSocketMessageHandler[] = [];
let interceptedWebSocket: WebSocket | null = null;

/**
 * Register a handler for intercepted WebSocket messages
 */
function registerWebSocketHandler(handler: WebSocketMessageHandler): void {
  if (!globalWebSocketHandlers.includes(handler)) {
    globalWebSocketHandlers.push(handler);
    console.log('[GlobalWS] Handler registered, total:', globalWebSocketHandlers.length);
  }
}

/**
 * Get the intercepted WebSocket instance
 */
function getInterceptedWebSocket(): WebSocket | null {
  return interceptedWebSocket;
}

/**
 * Extract JSON from potentially binary/Protobuf-framed data
 */
function extractJsonFromData(data: string | ArrayBuffer | Blob): any | null {
  let text: string;

  if (typeof data === 'string') {
    text = data;
  } else if (data instanceof ArrayBuffer) {
    text = new TextDecoder('utf-8').decode(data);
  } else {
    // Blob - can't extract synchronously
    return null;
  }

  // Try direct JSON parse
  try {
    return JSON.parse(text);
  } catch (e) {
    // Not valid JSON directly, try to find JSON in binary data
  }

  // Search for JSON object in binary data (Protobuf frame may have JSON embedded)
  let i = 0;
  while (i < text.length) {
    if (text[i] === '{') {
      let braceCount = 1;
      let j = i + 1;
      while (j < text.length && braceCount > 0) {
        if (text[j] === '{') braceCount++;
        else if (text[j] === '}') braceCount--;
        j++;
      }
      if (braceCount === 0) {
        try {
          const jsonStr = text.substring(i, j);
          return JSON.parse(jsonStr);
        } catch (e) {
          // Continue searching
        }
      }
    }
    i++;
  }

  return null;
}

// Patch WebSocket constructor to intercept messages
const OriginalWebSocket = window.WebSocket;

class InterceptedWebSocket extends OriginalWebSocket {
  constructor(url: string | URL, protocols?: string | string[]) {
    super(url, protocols);

    console.log('[GlobalWS] New WebSocket created:', url);
    interceptedWebSocket = this;

    // Add message listener to intercept messages
    this.addEventListener('message', async (event: MessageEvent) => {
      try {
        let data: any = null;

        // Handle different data types
        if (typeof event.data === 'string') {
          data = extractJsonFromData(event.data);
        } else if (event.data instanceof ArrayBuffer) {
          data = extractJsonFromData(event.data);
        } else if (event.data instanceof Blob) {
          // Handle Blob asynchronously
          const text = await event.data.text();
          data = extractJsonFromData(text);
        }

        // If we extracted a JSON object with a type, call handlers
        if (data && data.type) {
          const customTypes = [
            'server-config', 'speaker-status', 'speaker-transcription',
            'config-available', 'config-update-result', 'config-reset-result',
            'bot-llm-text', 'bot-llm-stream', 'bot-tts-text', 'bot-status',
            'bot-audio-level'
          ];

          if (customTypes.includes(data.type)) {
            console.log('[GlobalWS] Intercepted custom message:', data.type);
            for (const handler of globalWebSocketHandlers) {
              try {
                handler(data);
              } catch (e) {
                console.error('[GlobalWS] Handler error:', e);
              }
            }
          }
        }
      } catch (e) {
        // Ignore parse errors - let pipecat handle its own messages
      }
    });
  }
}

// Replace global WebSocket with our intercepted version
(window as any).WebSocket = InterceptedWebSocket;
console.log('[GlobalWS] WebSocket constructor patched');

// ============================================================================
// END GLOBAL WEBSOCKET INTERCEPTOR
// ============================================================================

// ============================================================================
// AUDIO FILE UPLOADER
// Enables uploading audio files for STT processing (alternative to microphone)
// Uses same streaming approach as audio_file_client.py for consistency
// ============================================================================

/**
 * Audio input mode enumeration
 */
enum AudioInputMode {
  MICROPHONE = 'microphone',
  FILE = 'file'
}

/**
 * Audio file upload progress callback type
 */
type AudioUploadProgressCallback = (
  progress: number,      // 0-100 percentage
  elapsedSecs: number,   // Seconds elapsed
  totalSecs: number,     // Total duration
  status: string         // Status message
) => void;

/**
 * Audio file upload complete callback type
 */
type AudioUploadCompleteCallback = (
  success: boolean,
  message: string,
  stats: {
    filename: string;
    duration: number;
    chunks: number;
    bytesSent: number;
  }
) => void;

/**
 * AudioFileUploader - Streams audio files to server via WebSocket
 *
 * This class handles:
 * - Decoding various audio formats (WAV, MP3, OGG, M4A, etc.)
 * - Resampling to 16kHz mono (server requirement)
 * - Real-time streaming at configurable playback speed
 * - Progress tracking and cancellation
 *
 * Uses the same Protobuf serialization as pipecat's WebSocket transport
 * to ensure compatibility with the server's audio processing pipeline.
 */
class AudioFileUploader {
  private audioContext: AudioContext | null = null;
  private serializer: ProtobufFrameSerializer;
  private isUploading: boolean = false;
  private isCancelled: boolean = false;
  private uploadSpeed: number = 1.0;

  // Progress tracking
  public onProgress: AudioUploadProgressCallback | null = null;
  public onComplete: AudioUploadCompleteCallback | null = null;

  // Statistics
  private filename: string = '';
  private totalChunks: number = 0;
  private currentChunk: number = 0;
  private bytesSent: number = 0;
  private startTime: number = 0;

  // Synchronized Audio Playback
  private playbackEnabled: boolean = true;        // Default: enabled
  private playbackVolume: number = 0.8;           // 0.0 to 1.0
  private sourceNode: AudioBufferSourceNode | null = null;
  private gainNode: GainNode | null = null;
  private currentAudioBuffer: AudioBuffer | null = null;
  private isPlaying: boolean = false;
  private playbackStartOffset: number = 0;        // For resume after pause

  // Playback callbacks
  public onPlaybackStateChange: ((playing: boolean, position: number) => void) | null = null;

  // Waveform data for visualization
  private waveformData: number[] = [];
  public onWaveformReady: ((data: number[]) => void) | null = null;

  // Audio parameters (matching server expectations)
  private readonly TARGET_SAMPLE_RATE = 16000;  // Server expects 16kHz
  private readonly NUM_CHANNELS = 1;             // Mono audio
  private readonly CHUNK_DURATION_MS = 16;       // 16ms chunks (same as mic input)

  // Supported file formats
  private readonly SUPPORTED_FORMATS = [
    'audio/wav', 'audio/wave', 'audio/x-wav',
    'audio/mpeg', 'audio/mp3',
    'audio/ogg', 'audio/vorbis',
    'audio/mp4', 'audio/m4a', 'audio/x-m4a',
    'audio/flac', 'audio/x-flac',
    'audio/webm'
  ];

  constructor() {
    this.serializer = new ProtobufFrameSerializer();
  }

  /**
   * Set progress callback
   */
  setProgressCallback(callback: AudioUploadProgressCallback): void {
    this.onProgress = callback;
  }

  /**
   * Set completion callback
   */
  setCompleteCallback(callback: AudioUploadCompleteCallback): void {
    this.onComplete = callback;
  }

  /**
   * Set upload speed multiplier (1.0 = real-time, 2.0 = 2x speed)
   * Also synchronizes playback rate when playback is active
   */
  setUploadSpeed(speed: number): void {
    const previousSpeed = this.uploadSpeed;
    this.uploadSpeed = speed === 0 ? 0 : Math.max(0.5, Math.min(10, speed));

    // Max speed (0) is incompatible with synchronized playback
    if (speed === 0 && this.playbackEnabled) {
      console.log('[AudioUploader] Max speed selected - playback auto-disabled');
      this.setPlaybackEnabled(false);
    }

    // Sync playback rate if currently playing
    if (this.sourceNode && this.isPlaying && this.uploadSpeed > 0) {
      this.sourceNode.playbackRate.value = this.uploadSpeed;
      console.log(`[AudioUploader] Playback rate synced to ${this.uploadSpeed}x`);
    }
  }

  // ============================================
  // SYNCHRONIZED AUDIO PLAYBACK METHODS
  // ============================================

  /**
   * Enable or disable synchronized playback
   */
  setPlaybackEnabled(enabled: boolean): void {
    this.playbackEnabled = enabled;
    console.log(`[AudioUploader] Playback ${enabled ? 'enabled' : 'disabled'}`);

    if (!enabled && this.isPlaying) {
      this.stopPlayback();
    }
  }

  /**
   * Check if playback is enabled
   */
  get playbackIsEnabled(): boolean {
    return this.playbackEnabled;
  }

  /**
   * Check if currently playing
   */
  get playing(): boolean {
    return this.isPlaying;
  }

  /**
   * Set playback volume (0.0 to 1.0)
   */
  setPlaybackVolume(volume: number): void {
    this.playbackVolume = Math.max(0, Math.min(1, volume));
    if (this.gainNode) {
      this.gainNode.gain.value = this.playbackVolume;
    }
    console.log(`[AudioUploader] Playback volume set to ${Math.round(this.playbackVolume * 100)}%`);
  }

  /**
   * Get current playback volume
   */
  get volume(): number {
    return this.playbackVolume;
  }

  /**
   * Get waveform data for visualization
   */
  getWaveformData(): number[] {
    return this.waveformData;
  }

  /**
   * Start synchronized playback from the beginning or resume from paused position
   */
  private startSynchronizedPlayback(): void {
    if (!this.audioContext || !this.currentAudioBuffer || !this.playbackEnabled) {
      return;
    }

    // Can't play at max speed (0) - no timing reference
    if (this.uploadSpeed === 0) {
      console.log('[AudioUploader] Cannot play at max speed - no timing sync possible');
      return;
    }

    try {
      // Ensure AudioContext is running (might be suspended due to autoplay policy)
      if (this.audioContext.state === 'suspended') {
        this.audioContext.resume();
      }

      // Create nodes
      this.sourceNode = this.audioContext.createBufferSource();
      this.sourceNode.buffer = this.currentAudioBuffer;
      this.sourceNode.playbackRate.value = this.uploadSpeed;

      // Create gain node for volume control
      if (!this.gainNode) {
        this.gainNode = this.audioContext.createGain();
        this.gainNode.connect(this.audioContext.destination);
      }
      this.gainNode.gain.value = this.playbackVolume;

      // Connect source -> gain -> destination
      this.sourceNode.connect(this.gainNode);

      // Handle playback end
      this.sourceNode.onended = () => {
        if (this.isPlaying) {
          this.isPlaying = false;
          this.playbackStartOffset = 0;
          this.onPlaybackStateChange?.(false, this.currentAudioBuffer?.duration || 0);
          console.log('[AudioUploader] Playback ended');
        }
      };

      // Start playback
      this.sourceNode.start(0, this.playbackStartOffset);
      this.isPlaying = true;

      console.log(`[AudioUploader] Playback started at ${this.playbackStartOffset.toFixed(2)}s, ` +
                  `speed: ${this.uploadSpeed}x, volume: ${Math.round(this.playbackVolume * 100)}%`);

      this.onPlaybackStateChange?.(true, this.playbackStartOffset);
    } catch (error) {
      console.error('[AudioUploader] Failed to start playback:', error);
    }
  }

  /**
   * Stop playback (can be resumed)
   */
  private stopPlayback(): void {
    if (this.sourceNode) {
      try {
        this.sourceNode.stop();
        this.sourceNode.disconnect();
      } catch (e) {
        // Ignore errors if already stopped
      }
      this.sourceNode = null;
    }

    if (this.isPlaying) {
      this.isPlaying = false;
      this.onPlaybackStateChange?.(false, this.playbackStartOffset);
      console.log('[AudioUploader] Playback stopped');
    }
  }

  /**
   * Generate waveform data from AudioBuffer for visualization
   * Returns normalized amplitude values (0-1) sampled across the audio duration
   */
  private generateWaveformData(audioBuffer: AudioBuffer, numSamples: number = 100): number[] {
    const channelData = audioBuffer.getChannelData(0);
    const samplesPerBucket = Math.floor(channelData.length / numSamples);
    const waveform: number[] = [];

    for (let i = 0; i < numSamples; i++) {
      const start = i * samplesPerBucket;
      const end = start + samplesPerBucket;

      // Calculate RMS (root mean square) for this bucket
      let sum = 0;
      for (let j = start; j < end && j < channelData.length; j++) {
        sum += channelData[j] * channelData[j];
      }
      const rms = Math.sqrt(sum / samplesPerBucket);

      // Normalize to 0-1 range (with some headroom)
      waveform.push(Math.min(1, rms * 3));
    }

    return waveform;
  }

  /**
   * Check if a file format is supported
   */
  isFormatSupported(file: File): boolean {
    const type = file.type.toLowerCase();
    const ext = file.name.toLowerCase().split('.').pop() || '';

    // Check MIME type
    if (this.SUPPORTED_FORMATS.includes(type)) return true;

    // Fallback to extension check
    const supportedExts = ['wav', 'mp3', 'ogg', 'oga', 'm4a', 'mp4', 'flac', 'webm'];
    return supportedExts.includes(ext);
  }

  /**
   * Get human-readable file info
   */
  getFileInfo(file: File): string {
    const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
    const ext = file.name.split('.').pop()?.toUpperCase() || 'Unknown';
    return `${ext} file, ${sizeMB} MB`;
  }

  /**
   * Check if currently uploading
   */
  get uploading(): boolean {
    return this.isUploading;
  }

  /**
   * Upload an audio file to the server
   *
   * @param file - The audio file to upload
   * @returns Promise that resolves when upload completes or is cancelled
   */
  async uploadFile(file: File): Promise<void> {
    const ws = getInterceptedWebSocket();

    if (!ws || ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected. Please connect to the server first.');
    }

    if (!this.isFormatSupported(file)) {
      throw new Error(`Unsupported audio format: ${file.type || file.name}`);
    }

    if (this.isUploading) {
      throw new Error('Upload already in progress. Cancel it first.');
    }

    this.isUploading = true;
    this.isCancelled = false;
    this.filename = file.name;
    this.bytesSent = 0;
    this.startTime = performance.now();

    try {
      // Report starting
      this.reportProgress(0, 0, 0, 'Decoding audio file...');

      // 1. Decode audio file to AudioBuffer
      const arrayBuffer = await file.arrayBuffer();

      // Create AudioContext with target sample rate for automatic resampling
      if (this.audioContext) {
        await this.audioContext.close();
      }
      this.audioContext = new AudioContext({ sampleRate: this.TARGET_SAMPLE_RATE });

      const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);

      console.log(`[AudioUploader] Decoded: ${audioBuffer.duration.toFixed(2)}s, ` +
                  `${audioBuffer.sampleRate}Hz, ${audioBuffer.numberOfChannels}ch`);

      // Store AudioBuffer for synchronized playback
      this.currentAudioBuffer = audioBuffer;
      this.playbackStartOffset = 0;

      // Generate waveform data for visualization
      this.reportProgress(0, 0, audioBuffer.duration, 'Generating waveform...');
      this.waveformData = this.generateWaveformData(audioBuffer, 100);
      this.onWaveformReady?.(this.waveformData);
      console.log(`[AudioUploader] Waveform generated: ${this.waveformData.length} samples`);

      // 2. Convert to 16-bit PCM mono
      this.reportProgress(0, 0, audioBuffer.duration, 'Converting to PCM...');
      const pcmData = this.audioBufferToPCM16(audioBuffer);

      console.log(`[AudioUploader] PCM data: ${pcmData.length} samples, ` +
                  `${(pcmData.length / this.TARGET_SAMPLE_RATE).toFixed(2)}s`);

      // 3. Start synchronized playback (if enabled)
      if (this.playbackEnabled && this.uploadSpeed > 0) {
        this.startSynchronizedPlayback();
      }

      // 4. Stream to server at real-time rate
      this.reportProgress(0, 0, audioBuffer.duration, 'Streaming to server...');
      await this.streamPCMData(ws, pcmData, audioBuffer.duration);

      // Stop playback when streaming completes
      this.stopPlayback();

      // Report completion
      const stats = {
        filename: this.filename,
        duration: audioBuffer.duration,
        chunks: this.totalChunks,
        bytesSent: this.bytesSent
      };

      if (this.isCancelled) {
        this.onComplete?.(false, 'Upload cancelled by user', stats);
      } else {
        this.onComplete?.(true, 'Upload completed successfully', stats);
      }

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      console.error('[AudioUploader] Error:', errorMsg);

      this.onComplete?.(false, `Upload failed: ${errorMsg}`, {
        filename: this.filename,
        duration: 0,
        chunks: this.currentChunk,
        bytesSent: this.bytesSent
      });

      throw error;
    } finally {
      this.isUploading = false;
    }
  }

  /**
   * Cancel ongoing upload and stop playback
   */
  cancel(): void {
    if (this.isUploading) {
      this.isCancelled = true;
      this.stopPlayback();
      console.log('[AudioUploader] Upload cancelled');
    }
  }

  /**
   * Convert AudioBuffer to 16-bit PCM Int16Array (mono)
   */
  private audioBufferToPCM16(buffer: AudioBuffer): Int16Array {
    let channelData: Float32Array;

    // Mix to mono if stereo
    if (buffer.numberOfChannels > 1) {
      channelData = this.mixToMono(buffer);
    } else {
      channelData = buffer.getChannelData(0);
    }

    // Convert Float32 (-1.0 to 1.0) to Int16 (-32768 to 32767)
    const pcm = new Int16Array(channelData.length);
    for (let i = 0; i < channelData.length; i++) {
      // Clamp to valid range and convert
      const sample = Math.max(-1, Math.min(1, channelData[i]));
      pcm[i] = Math.round(sample * 32767);
    }

    return pcm;
  }

  /**
   * Mix multi-channel audio to mono
   */
  private mixToMono(buffer: AudioBuffer): Float32Array {
    const numChannels = buffer.numberOfChannels;
    const length = buffer.length;
    const mono = new Float32Array(length);

    // Sum all channels
    for (let ch = 0; ch < numChannels; ch++) {
      const channelData = buffer.getChannelData(ch);
      for (let i = 0; i < length; i++) {
        mono[i] += channelData[i];
      }
    }

    // Average
    const scale = 1 / numChannels;
    for (let i = 0; i < length; i++) {
      mono[i] *= scale;
    }

    return mono;
  }

  /**
   * Stream PCM data to server in real-time chunks
   */
  private async streamPCMData(
    ws: WebSocket,
    pcmData: Int16Array,
    totalDurationSecs: number
  ): Promise<void> {
    const samplesPerChunk = Math.floor(
      this.TARGET_SAMPLE_RATE * this.CHUNK_DURATION_MS / 1000
    );
    this.totalChunks = Math.ceil(pcmData.length / samplesPerChunk);
    this.currentChunk = 0;

    const delayMs = this.CHUNK_DURATION_MS / this.uploadSpeed;

    console.log(`[AudioUploader] Starting stream: ${this.totalChunks} chunks, ` +
                `${samplesPerChunk} samples/chunk, ${delayMs.toFixed(1)}ms delay`);

    for (let i = 0; i < pcmData.length && !this.isCancelled; i += samplesPerChunk) {
      // Extract chunk
      const end = Math.min(i + samplesPerChunk, pcmData.length);
      const chunkData = pcmData.slice(i, end);

      // Convert Int16Array to ArrayBuffer for serialization
      const arrayBuffer = chunkData.buffer.slice(
        chunkData.byteOffset,
        chunkData.byteOffset + chunkData.byteLength
      );

      // Serialize using pipecat's Protobuf format
      const serializedFrame = this.serializer.serializeAudio(
        arrayBuffer,
        this.TARGET_SAMPLE_RATE,
        this.NUM_CHANNELS
      );

      // Send to server
      ws.send(serializedFrame);

      this.bytesSent += chunkData.byteLength;
      this.currentChunk++;

      // Report progress
      const progress = (this.currentChunk / this.totalChunks) * 100;
      const elapsedSecs = (i + samplesPerChunk) / this.TARGET_SAMPLE_RATE;
      this.reportProgress(progress, elapsedSecs, totalDurationSecs, 'Streaming...');

      // Wait for real-time playback rate
      await this.sleep(delayMs);
    }

    console.log(`[AudioUploader] Stream complete: ${this.currentChunk} chunks sent, ` +
                `${(this.bytesSent / 1024).toFixed(1)} KB`);
  }

  /**
   * Report progress to callback
   */
  private reportProgress(
    progress: number,
    elapsedSecs: number,
    totalSecs: number,
    status: string
  ): void {
    this.onProgress?.(progress, elapsedSecs, totalSecs, status);
  }

  /**
   * Sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    this.cancel();
    this.stopPlayback();

    // Clean up gain node
    if (this.gainNode) {
      this.gainNode.disconnect();
      this.gainNode = null;
    }

    // Clean up audio buffer
    this.currentAudioBuffer = null;
    this.waveformData = [];

    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }
  }
}

// ============================================================================
// END AUDIO FILE UPLOADER
// ============================================================================

// Configuration parameter mapping
interface ConfigSpec {
  name: string;
  category: string;
  type: string;
  default: any;
  description: string;
  min?: number;
  max?: number;
  options?: any[];
  unit?: string;
  current?: any;
}

interface ConfigState {
  [key: string]: any;
}

// Transcript segment for flowing paradigm
interface TranscriptSegment {
  id: string;
  text: string;
  speakerId: number | null;
  speakerName: string;
  speakerColor: string;
  isFinal: boolean;
  reason: string;
  timestamp: string;
  wordCount: number;
}

// Conversation message for Voice Agent mode
interface ConversationMessage {
  id: string;
  role: 'user' | 'bot';
  text: string;
  timestamp: Date;
  status: 'speaking' | 'thinking' | 'complete';
  ttsChunkId?: number;
  speakerId?: number | null;
  source?: 'stt' | 'text-input';  // Input source: STT (voice) or text input
}

// Bot status for Voice Agent mode
interface BotStatus {
  status: 'idle' | 'thinking' | 'speaking';
  ttsPlaying: boolean;
  currentText: string;
}

// TTS text chunk for synchronization
interface TTSChunk {
  id: number;
  text: string;
  isFinal: boolean;
}

/**
 * Professional Audio Visualizer using Web Audio API
 * Provides multiple visualization modes with smooth animations
 */
class AudioVisualizer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private analyser: AnalyserNode | null = null;
  private dataArray: Uint8Array | null = null;
  private frequencyData: Uint8Array | null = null;
  private animationId: number | null = null;
  private isActive: boolean = false;

  // Visualization settings
  private mode: 'bars' | 'wave' | 'circular' = 'bars';
  private smoothingFactor: number = 0.8;
  private barCount: number = 64;
  private previousBarHeights: number[] = [];

  // Color settings - theme-aware
  private primaryColor: string = '#76b900';
  private secondaryColor: string = '#06b6d4';
  private isDarkTheme: boolean = true;
  private colorScheme: 'green' | 'blue' = 'green';  // Input=green, Output=blue

  // Theme-specific color palettes for GREEN (input) scheme
  private readonly darkThemeColorsGreen = {
    bgPrimary: '#0c0c0d',
    bgSecondary: '#111113',
    primaryFull: '#76b900',
    primaryHigh: 'rgba(118, 185, 0, 0.5)',
    primaryMed: 'rgba(118, 185, 0, 0.3)',
    primaryLow: 'rgba(118, 185, 0, 0.2)',
    primaryVeryLow: 'rgba(118, 185, 0, 0.1)',
    secondaryMed: 'rgba(6, 182, 212, 0.3)',
    idleLine: 'rgba(255, 255, 255, 0.03)',
    centerGradientStart: 'rgba(118, 185, 0, 0.1)',
    centerGradientEnd: 'rgba(118, 185, 0, 0)',
  };

  private readonly lightThemeColorsGreen = {
    bgPrimary: '#f8fafc',
    bgSecondary: '#f1f5f9',
    primaryFull: '#65a30d',
    primaryHigh: 'rgba(101, 163, 13, 0.7)',
    primaryMed: 'rgba(101, 163, 13, 0.5)',
    primaryLow: 'rgba(101, 163, 13, 0.35)',
    primaryVeryLow: 'rgba(101, 163, 13, 0.2)',
    secondaryMed: 'rgba(14, 165, 233, 0.4)',
    idleLine: 'rgba(0, 0, 0, 0.06)',
    centerGradientStart: 'rgba(101, 163, 13, 0.15)',
    centerGradientEnd: 'rgba(101, 163, 13, 0)',
  };

  // Theme-specific color palettes for BLUE (output) scheme
  private readonly darkThemeColorsBlue = {
    bgPrimary: '#0c0c0d',
    bgSecondary: '#111113',
    primaryFull: '#3b82f6',
    primaryHigh: 'rgba(59, 130, 246, 0.5)',
    primaryMed: 'rgba(59, 130, 246, 0.3)',
    primaryLow: 'rgba(59, 130, 246, 0.2)',
    primaryVeryLow: 'rgba(59, 130, 246, 0.1)',
    secondaryMed: 'rgba(139, 92, 246, 0.3)',
    idleLine: 'rgba(255, 255, 255, 0.03)',
    centerGradientStart: 'rgba(59, 130, 246, 0.1)',
    centerGradientEnd: 'rgba(59, 130, 246, 0)',
  };

  private readonly lightThemeColorsBlue = {
    bgPrimary: '#f8fafc',
    bgSecondary: '#f1f5f9',
    primaryFull: '#2563eb',
    primaryHigh: 'rgba(37, 99, 235, 0.7)',
    primaryMed: 'rgba(37, 99, 235, 0.5)',
    primaryLow: 'rgba(37, 99, 235, 0.35)',
    primaryVeryLow: 'rgba(37, 99, 235, 0.2)',
    secondaryMed: 'rgba(124, 58, 237, 0.4)',
    idleLine: 'rgba(0, 0, 0, 0.06)',
    centerGradientStart: 'rgba(37, 99, 235, 0.15)',
    centerGradientEnd: 'rgba(37, 99, 235, 0)',
  };

  // Animation state
  private phase: number = 0;
  private isSpeaking: boolean = false;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
    this.previousBarHeights = new Array(this.barCount).fill(0);
    this.isDarkTheme = !document.documentElement.hasAttribute('data-theme') ||
      document.documentElement.getAttribute('data-theme') !== 'light';
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  /**
   * Update theme colors based on current theme setting
   */
  setTheme(isDark: boolean): void {
    this.isDarkTheme = isDark;
    // Update primary color based on theme and color scheme
    const colors = this.getColors();
    this.primaryColor = colors.primaryFull;
  }

  /**
   * Set the color scheme (green for input, blue for output)
   */
  setColorScheme(scheme: 'green' | 'blue'): void {
    this.colorScheme = scheme;
    const colors = this.getColors();
    this.primaryColor = colors.primaryFull;
    this.secondaryColor = scheme === 'green' ? '#06b6d4' : '#8b5cf6';
  }

  /**
   * Get theme-aware colors based on current theme and color scheme
   */
  private getColors() {
    if (this.colorScheme === 'blue') {
      return this.isDarkTheme ? this.darkThemeColorsBlue : this.lightThemeColorsBlue;
    }
    return this.isDarkTheme ? this.darkThemeColorsGreen : this.lightThemeColorsGreen;
  }

  resize(): void {
    const container = this.canvas.parentElement;
    if (container) {
      const dpr = window.devicePixelRatio || 1;
      const rect = container.getBoundingClientRect();
      this.canvas.width = rect.width * dpr;
      this.canvas.height = rect.height * dpr;
      this.ctx.scale(dpr, dpr);
      this.canvas.style.width = `${rect.width}px`;
      this.canvas.style.height = `${rect.height}px`;
    }
  }

  setAnalyser(analyser: AnalyserNode): void {
    this.analyser = analyser;
    this.analyser.fftSize = 256;
    this.analyser.smoothingTimeConstant = this.smoothingFactor;
    const bufferLength = this.analyser.frequencyBinCount;
    this.dataArray = new Uint8Array(bufferLength);
    this.frequencyData = new Uint8Array(bufferLength);
  }

  setSpeaking(speaking: boolean): void {
    this.isSpeaking = speaking;
  }

  setMode(mode: 'bars' | 'wave' | 'circular'): void {
    this.mode = mode;
  }

  start(): void {
    if (this.isActive) return;
    this.isActive = true;
    this.draw();
  }

  stop(): void {
    this.isActive = false;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
    this.clearCanvas();
  }

  private clearCanvas(): void {
    const width = this.canvas.width / (window.devicePixelRatio || 1);
    const height = this.canvas.height / (window.devicePixelRatio || 1);
    const colors = this.getColors();

    // Draw gradient background (theme-aware)
    const gradient = this.ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, colors.bgPrimary);
    gradient.addColorStop(1, colors.bgSecondary);
    this.ctx.fillStyle = gradient;
    this.ctx.fillRect(0, 0, width, height);

    // Draw idle state indicator (theme-aware)
    this.ctx.fillStyle = colors.idleLine;
    const centerY = height / 2;
    this.ctx.fillRect(0, centerY - 1, width, 2);
  }

  private draw(): void {
    if (!this.isActive) return;

    const width = this.canvas.width / (window.devicePixelRatio || 1);
    const height = this.canvas.height / (window.devicePixelRatio || 1);
    const colors = this.getColors();

    // Clear with gradient background (theme-aware)
    const bgGradient = this.ctx.createLinearGradient(0, 0, 0, height);
    bgGradient.addColorStop(0, colors.bgPrimary);
    bgGradient.addColorStop(1, colors.bgSecondary);
    this.ctx.fillStyle = bgGradient;
    this.ctx.fillRect(0, 0, width, height);

    // Get frequency data
    if (this.analyser && this.frequencyData) {
      this.analyser.getByteFrequencyData(this.frequencyData);
    }

    this.phase += 0.02;

    switch (this.mode) {
      case 'bars':
        this.drawBars(width, height);
        break;
      case 'wave':
        this.drawWave(width, height);
        break;
      case 'circular':
        this.drawCircular(width, height);
        break;
    }

    this.animationId = requestAnimationFrame(() => this.draw());
  }

  private drawBars(width: number, height: number): void {
    if (!this.frequencyData) return;

    const colors = this.getColors();
    const barWidth = width / this.barCount;
    const gap = 2;
    const centerY = height / 2;

    // Create gradient for bars
    const gradient = this.ctx.createLinearGradient(0, 0, width, 0);
    gradient.addColorStop(0, this.primaryColor);
    gradient.addColorStop(0.5, this.secondaryColor);
    gradient.addColorStop(1, this.primaryColor);

    for (let i = 0; i < this.barCount; i++) {
      // Sample from frequency data
      const dataIndex = Math.floor((i / this.barCount) * this.frequencyData.length);
      const value = this.frequencyData[dataIndex] / 255;

      // Apply smoothing
      const targetHeight = value * (height * 0.4);
      this.previousBarHeights[i] += (targetHeight - this.previousBarHeights[i]) * 0.15;
      const barHeight = this.previousBarHeights[i];

      // Add subtle wave animation
      const waveOffset = Math.sin(this.phase + i * 0.1) * 2;
      const animatedHeight = Math.max(2, barHeight + (this.isSpeaking ? waveOffset : 0));

      // Calculate position for mirrored bars
      const x = i * barWidth + gap / 2;
      const actualBarWidth = barWidth - gap;

      // Draw top bar (theme-aware colors)
      const topGradient = this.ctx.createLinearGradient(0, centerY - animatedHeight, 0, centerY);
      topGradient.addColorStop(0, this.isSpeaking ? colors.primaryFull : colors.primaryMed);
      topGradient.addColorStop(1, colors.primaryVeryLow);
      this.ctx.fillStyle = topGradient;
      this.ctx.beginPath();
      this.ctx.roundRect(x, centerY - animatedHeight, actualBarWidth, animatedHeight, 2);
      this.ctx.fill();

      // Draw bottom bar (mirror, theme-aware colors)
      const bottomGradient = this.ctx.createLinearGradient(0, centerY, 0, centerY + animatedHeight);
      bottomGradient.addColorStop(0, colors.primaryVeryLow);
      bottomGradient.addColorStop(1, this.isSpeaking ? colors.primaryFull : colors.primaryMed);
      this.ctx.fillStyle = bottomGradient;
      this.ctx.beginPath();
      this.ctx.roundRect(x, centerY, actualBarWidth, animatedHeight, 2);
      this.ctx.fill();
    }

    // Draw center line glow
    if (this.isSpeaking) {
      this.ctx.shadowColor = colors.primaryFull;
      this.ctx.shadowBlur = 10;
      this.ctx.strokeStyle = colors.primaryFull;
      this.ctx.lineWidth = 1;
      this.ctx.beginPath();
      this.ctx.moveTo(0, centerY);
      this.ctx.lineTo(width, centerY);
      this.ctx.stroke();
      this.ctx.shadowBlur = 0;
    }
  }

  private drawWave(width: number, height: number): void {
    if (!this.dataArray || !this.analyser) return;

    const colors = this.getColors();
    this.analyser.getByteTimeDomainData(this.dataArray);

    const centerY = height / 2;
    const sliceWidth = width / this.dataArray.length;

    // Draw wave fill
    this.ctx.beginPath();
    this.ctx.moveTo(0, centerY);

    let x = 0;
    for (let i = 0; i < this.dataArray.length; i++) {
      const v = this.dataArray[i] / 128.0;
      const y = (v * height) / 2;

      if (i === 0) {
        this.ctx.moveTo(x, y);
      } else {
        // Use quadratic curves for smoother lines
        const prevX = (i - 1) * sliceWidth;
        const prevV = this.dataArray[i - 1] / 128.0;
        const prevY = (prevV * height) / 2;
        const cpX = (prevX + x) / 2;
        this.ctx.quadraticCurveTo(prevX, prevY, cpX, (prevY + y) / 2);
      }
      x += sliceWidth;
    }

    this.ctx.lineTo(width, centerY);
    this.ctx.closePath();

    // Fill gradient (theme-aware)
    const fillGradient = this.ctx.createLinearGradient(0, 0, 0, height);
    fillGradient.addColorStop(0, colors.primaryLow);
    fillGradient.addColorStop(0.5, colors.secondaryMed);
    fillGradient.addColorStop(1, colors.primaryLow);
    this.ctx.fillStyle = fillGradient;
    this.ctx.fill();

    // Draw wave line
    this.ctx.beginPath();
    x = 0;
    for (let i = 0; i < this.dataArray.length; i++) {
      const v = this.dataArray[i] / 128.0;
      const y = (v * height) / 2;

      if (i === 0) {
        this.ctx.moveTo(x, y);
      } else {
        this.ctx.lineTo(x, y);
      }
      x += sliceWidth;
    }

    // Theme-aware stroke style
    this.ctx.strokeStyle = this.isSpeaking ? colors.primaryFull : colors.primaryHigh;
    this.ctx.lineWidth = 2;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';

    if (this.isSpeaking) {
      this.ctx.shadowColor = colors.primaryFull;
      this.ctx.shadowBlur = 8;
    }
    this.ctx.stroke();
    this.ctx.shadowBlur = 0;
  }

  private drawCircular(width: number, height: number): void {
    if (!this.frequencyData) return;

    const colors = this.getColors();
    const centerX = width / 2;
    const centerY = height / 2;
    const baseRadius = Math.min(width, height) * 0.25;

    // Theme-aware HSL adjustments
    // Light mode: higher saturation, adjusted lightness for visibility
    const speakingSat = this.isDarkTheme ? 80 : 70;
    const speakingLight = this.isDarkTheme ? 50 : 40;
    const idleSat = this.isDarkTheme ? 40 : 50;
    const idleLight = this.isDarkTheme ? 40 : 35;
    const idleAlpha = this.isDarkTheme ? 0.3 : 0.5;

    // Color scheme-aware hue ranges
    // Green scheme: 80-140 (green to cyan)
    // Blue scheme: 200-280 (blue to purple)
    const hueBase = this.colorScheme === 'green' ? 80 : 200;
    const hueRange = this.colorScheme === 'green' ? 60 : 80;

    // Draw circular bars
    const bars = 32;
    for (let i = 0; i < bars; i++) {
      const dataIndex = Math.floor((i / bars) * this.frequencyData.length);
      const value = this.frequencyData[dataIndex] / 255;

      const angle = (i / bars) * Math.PI * 2 - Math.PI / 2;
      const barLength = value * baseRadius * 0.8 + 5;

      const x1 = centerX + Math.cos(angle) * baseRadius;
      const y1 = centerY + Math.sin(angle) * baseRadius;
      const x2 = centerX + Math.cos(angle) * (baseRadius + barLength);
      const y2 = centerY + Math.sin(angle) * (baseRadius + barLength);

      const hue = (i / bars) * hueRange + hueBase; // Color scheme aware hue
      this.ctx.strokeStyle = this.isSpeaking
        ? `hsla(${hue}, ${speakingSat}%, ${speakingLight}%, ${0.5 + value * 0.5})`
        : `hsla(${hue}, ${idleSat}%, ${idleLight}%, ${idleAlpha})`;
      this.ctx.lineWidth = 3;
      this.ctx.lineCap = 'round';

      this.ctx.beginPath();
      this.ctx.moveTo(x1, y1);
      this.ctx.lineTo(x2, y2);
      this.ctx.stroke();
    }

    // Draw center circle (theme-aware)
    const gradient = this.ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, baseRadius);
    gradient.addColorStop(0, colors.centerGradientStart);
    gradient.addColorStop(1, colors.centerGradientEnd);
    this.ctx.fillStyle = gradient;
    this.ctx.beginPath();
    this.ctx.arc(centerX, centerY, baseRadius, 0, Math.PI * 2);
    this.ctx.fill();

    // Draw inner glow when speaking
    if (this.isSpeaking) {
      this.ctx.strokeStyle = colors.primaryFull;
      this.ctx.lineWidth = 2;
      this.ctx.shadowColor = colors.primaryFull;
      this.ctx.shadowBlur = 15;
      this.ctx.beginPath();
      this.ctx.arc(centerX, centerY, baseRadius * 0.4, 0, Math.PI * 2);
      this.ctx.stroke();
      this.ctx.shadowBlur = 0;
    }
  }
}

class WebsocketClientApp {
  private rtviClient: RTVIClient | null = null;

  // Control elements
  private connectBtn: HTMLButtonElement | null = null;
  private disconnectBtn: HTMLButtonElement | null = null;
  private muteBtn: HTMLButtonElement | null = null;
  private clearStreamBtn: HTMLButtonElement | null = null;
  private clearTranscriptBtn: HTMLButtonElement | null = null;
  private exportBtn: HTMLButtonElement | null = null;
  private toggleSidebarBtn: HTMLButtonElement | null = null;
  private applyConfigBtn: HTMLButtonElement | null = null;
  private resetConfigBtn: HTMLButtonElement | null = null;
  private vizModeBtn: HTMLButtonElement | null = null;
  private themeToggleBtn: HTMLButtonElement | null = null;

  // Status elements
  private connectionIndicator: HTMLElement | null = null;
  private statusText: HTMLElement | null = null;
  private footerStats: HTMLElement | null = null;

  // Module badges (STT, LLM, TTS, DIAR)
  private sttBadge: HTMLElement | null = null;
  private llmBadge: HTMLElement | null = null;
  private ttsBadge: HTMLElement | null = null;
  private diarBadge: HTMLElement | null = null;

  // Model info panel elements
  private modelInfoPanel: HTMLElement | null = null;
  private sttModelName: HTMLElement | null = null;
  private sttModelParams: HTMLElement | null = null;
  private llmModelName: HTMLElement | null = null;
  private llmModelParams: HTMLElement | null = null;
  private ttsModelName: HTMLElement | null = null;
  private ttsModelParams: HTMLElement | null = null;
  private diarModelName: HTMLElement | null = null;
  private diarModelParams: HTMLElement | null = null;

  // Server config storage
  private serverModelConfig: {
    stt?: { model?: string; device?: string; params?: any };
    llm?: { model?: string; type?: string };
    tts?: { type?: string; model?: string };
    diar?: { enabled?: boolean; model?: string; threshold?: number };
  } | null = null;

  // Real-time stream panels (legacy)
  private realtimeStreamsZone: HTMLElement | null = null;
  private sttStreamPanel: HTMLElement | null = null;
  private llmStreamPanel: HTMLElement | null = null;
  private sttStreamContent: HTMLElement | null = null;
  private llmStreamContent: HTMLElement | null = null;
  private streamResizeHandle: HTMLElement | null = null;

  // NEW: IO Zone elements
  private inputZone: HTMLElement | null = null;
  private outputZone: HTMLElement | null = null;
  private ioResizeHandle: HTMLElement | null = null;
  private inputStatusBadge: HTMLElement | null = null;
  private outputStatusBadge: HTMLElement | null = null;
  private sttLiveIndicator: HTMLElement | null = null;
  private ttsSyncIndicator: HTMLElement | null = null;
  private speakingIndicator: HTMLElement | null = null;
  private speakingStatusLabel: HTMLElement | null = null;

  // Text Input Zone elements (for text input modes: LLM-Only, TTS-Only, LLM+TTS)
  private textInputZone: HTMLElement | null = null;
  private textInputTextarea: HTMLTextAreaElement | null = null;
  private textSendBtn: HTMLButtonElement | null = null;
  private textInputStatusBadge: HTMLElement | null = null;
  private isTextInputMode: boolean = false;

  // STT-only mode tracking
  private transcriptArea: HTMLElement | null = null;
  private isSTTOnlyMode: boolean = false;

  // Volume elements - Input
  private volumeBar: HTMLElement | null = null;
  private volumeText: HTMLElement | null = null;
  private inputVolumeBar: HTMLElement | null = null;
  private inputVolumeText: HTMLElement | null = null;

  // Volume elements - Output (TTS)
  private outputVolumeBar: HTMLElement | null = null;
  private outputVolumeText: HTMLElement | null = null;

  // Waveform elements - Input
  private waveformCanvas: HTMLCanvasElement | null = null;
  private audioVisualizer: AudioVisualizer | null = null;
  private inputWaveformCanvas: HTMLCanvasElement | null = null;
  private inputAudioVisualizer: AudioVisualizer | null = null;

  // Waveform elements - Output (TTS)
  private outputWaveformCanvas: HTMLCanvasElement | null = null;
  private outputAudioVisualizer: AudioVisualizer | null = null;
  private outputAudioContext: AudioContext | null = null;
  private outputAnalyser: AnalyserNode | null = null;

  // Circular visualizer for TTS output (ChatGPT-style)
  private circularVisualizer: HTMLCanvasElement | null = null;
  private circularVisualizerCtx: CanvasRenderingContext2D | null = null;
  private circularAnimationId: number | null = null;
  private currentAudioLevel: number = 0;
  private targetAudioLevel: number = 0;
  private currentPeakLevel: number = 0;
  private isBotSpeaking: boolean = false;

  // Pipeline stage elements
  private pipelineStatus: HTMLElement | null = null;
  private pipelineSTT: HTMLElement | null = null;
  private pipelineLLM: HTMLElement | null = null;
  private pipelineTTS: HTMLElement | null = null;

  // History panel elements
  private historyPanel: HTMLElement | null = null;
  private historyList: HTMLElement | null = null;
  private clearHistoryBtn: HTMLButtonElement | null = null;
  private historyCollapseBtn: HTMLButtonElement | null = null;

  // Debug panel elements
  private debugPanel: HTMLElement | null = null;
  private debugLog: HTMLElement | null = null;
  private clearDebugBtn: HTMLButtonElement | null = null;
  private debugCollapseBtn: HTMLButtonElement | null = null;
  private autoScrollToggle: HTMLInputElement | null = null;
  private debugResizeHandle: HTMLElement | null = null;

  // VAD elements
  private vadIndicator: HTMLElement | null = null;
  private vadLabel: HTMLElement | null = null;
  private vadTimer: HTMLElement | null = null;

  // NEW: Realtime STT Flow elements
  private sttFlow: HTMLElement | null = null;
  private sttCursor: HTMLElement | null = null;
  private liveBadge: HTMLElement | null = null;
  private historyContent: HTMLElement | null = null;
  private historyEmpty: HTMLElement | null = null;
  private clearFlowBtn: HTMLButtonElement | null = null;
  private maxWordsDisplayInput: HTMLInputElement | null = null;

  // Resize handles
  private resizeLeftSidebar: HTMLElement | null = null;
  private resizeRightSidebar: HTMLElement | null = null;
  private resizeBottomPanel: HTMLElement | null = null;
  private bottomPanel: HTMLElement | null = null;
  private monitorsSidebar: HTMLElement | null = null;

  // Font and language settings
  private fontStyleSelect: HTMLSelectElement | null = null;
  private fontSizeSlider: HTMLInputElement | null = null;
  private fontSizeValue: HTMLElement | null = null;
  private defaultLanguageSelect: HTMLSelectElement | null = null;
  private defaultLanguage: string = 'en';

  // Legacy transcript elements (for backward compatibility)
  private partialTranscript: HTMLElement | null = null;
  private finalTranscripts: HTMLElement | null = null;
  private transcriptCount: HTMLElement | null = null;
  private streamLog: HTMLElement | null = null;
  private liveTranscriptArea: HTMLElement | null = null;

  // Sidebar elements
  private settingsSidebar: HTMLElement | null = null;
  private settingsContent: HTMLElement | null = null;
  private diarSettings: HTMLElement | null = null;

  // Speaker diarization elements
  private speakerIndicators: HTMLElement[] = [];
  private speakerPanel: HTMLElement | null = null;

  // Audio elements
  private botAudio: HTMLAudioElement;

  // Theme state
  private isDarkMode: boolean = true;

  // State
  private isConnecting: boolean = false;
  private isDisconnecting: boolean = false;
  private isMuted: boolean = false;
  private transcriptCounter: number = 0;

  // Audio monitoring
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private microphone: MediaStreamAudioSourceNode | null = null;
  private volumeUpdateInterval: number | null = null;

  // VAD tracking
  private vadStartTime: number | null = null;
  private vadTimerInterval: number | null = null;

  // Speaker tracking
  private activeSpeakerId: number | null = null;
  private speakerColors: string[] = ['#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6'];
  private speakerNames: string[] = ['Speaker 1', 'Speaker 2', 'Speaker 3', 'Speaker 4'];
  private diarizationEnabled: boolean = false;

  // Configuration state
  private configSpecs: { [key: string]: ConfigSpec } = {};
  private pendingConfig: ConfigState = {};
  private serverConfig: ConfigState = {};

  // Flowing transcript state - continuous accumulation paradigm
  private currentSegment: TranscriptSegment | null = null;
  private accumulatedText: string = '';  // Current partial being typed
  private flowLines: string[] = [];       // Accumulated finalized lines in the flow
  private currentSpeakerId: number | null = null;
  private maxWordsPerLine: number = 15;   // Words per line before wrapping
  private segmentIdCounter: number = 0;
  private currentLineWordCount: number = 0;

  // STT-only mode stability: Windowed buffer for flowLines to prevent OOM
  private readonly MAX_FLOW_LINES: number = 100;  // Maximum lines to keep in memory

  // Visualization mode
  private vizMode: 'bars' | 'wave' | 'circular' = 'bars';

  // Voice Agent Conversation state
  private conversationMessages: ConversationMessage[] = [];
  private currentBotMessage: ConversationMessage | null = null;
  private currentUserMessage: ConversationMessage | null = null;
  private botStatus: BotStatus = { status: 'idle', ttsPlaying: false, currentText: '' };
  private llmAccumulatedText: string = '';
  private ttsChunks: TTSChunk[] = [];
  private currentTTSChunkIndex: number = 0;
  private messageIdCounter: number = 0;

  // Flag to prevent duplicate message handler registration
  private messageHandlersSetup: boolean = false;

  // Conversation View elements
  private conversationView: HTMLElement | null = null;
  private botStatusIndicator: HTMLElement | null = null;
  private botThinkingIndicator: HTMLElement | null = null;
  private clearConversationBtn: HTMLButtonElement | null = null;
  private botStatusText: HTMLElement | null = null;

  // Audio File Upload elements
  private audioInputMode: AudioInputMode = AudioInputMode.MICROPHONE;
  private audioFileUploader: AudioFileUploader | null = null;
  private inputModeToggle: HTMLElement | null = null;
  private modeMicBtn: HTMLButtonElement | null = null;
  private modeFileBtn: HTMLButtonElement | null = null;
  private inputModeLabel: HTMLElement | null = null;
  private fileUploadSection: HTMLElement | null = null;
  private fileDropzone: HTMLElement | null = null;
  private fileInput: HTMLInputElement | null = null;
  private fileInfoPanel: HTMLElement | null = null;
  private uploadFileName: HTMLElement | null = null;
  private uploadFileMeta: HTMLElement | null = null;
  private fileRemoveBtn: HTMLButtonElement | null = null;
  private uploadProgressFill: HTMLElement | null = null;
  private uploadProgressText: HTMLElement | null = null;
  private uploadProgressTime: HTMLElement | null = null;
  private uploadSpeedSelect: HTMLSelectElement | null = null;
  private uploadStartBtn: HTMLButtonElement | null = null;
  private uploadCancelBtn: HTMLButtonElement | null = null;
  private selectedFile: File | null = null;

  // Audio Playback UI Elements
  private playbackEnableCheckbox: HTMLInputElement | null = null;
  private playbackVolumeSlider: HTMLInputElement | null = null;
  private playbackVolumeValue: HTMLElement | null = null;
  private playbackStatus: HTMLElement | null = null;
  private playbackStatusText: HTMLElement | null = null;
  private playbackControls: HTMLElement | null = null;
  private playbackPositionMarker: HTMLElement | null = null;
  private playbackWaveformCanvas: HTMLCanvasElement | null = null;
  private playbackWaveformPosition: HTMLElement | null = null;
  private waveformPlayed: HTMLElement | null = null;

  // Server connection info - Voice Agent only (7860 HTTP + 8765 WebSocket)
  private readonly voiceAgentServer = {
    name: 'NeMo Voice Agent',
    baseUrl: `http://${window.location.hostname}:7860`,
    wsPort: 8765
  };

  constructor() {
    console.log("NeMo Voice Agent Client initialized");
    this.botAudio = document.createElement('audio');
    this.botAudio.autoplay = true;
    document.body.appendChild(this.botAudio);

    // Register global WebSocket message handler EARLY (before any connection)
    // This ensures we capture server-config messages before pipecat rejects them
    this.registerGlobalMessageHandler();

    this.setupDOMElements();
    this.setupEventListeners();
    this.setupSettingsPanel();
    this.updateFooterStats('Ready');
  }

  /**
   * Register global WebSocket message handler for custom message types
   * Called early in constructor to ensure handler is ready before connection
   */
  private registerGlobalMessageHandler(): void {
    registerWebSocketHandler((message: any) => {
      console.log('[GlobalHandler] Received message:', message.type);
      this.handleCustomMessage(message);
    });
    console.log('[GlobalHandler] Early registration complete');
  }

  /**
   * Set up references to DOM elements
   */
  private setupDOMElements(): void {
    // Controls
    this.connectBtn = document.getElementById('connect-btn') as HTMLButtonElement;
    this.disconnectBtn = document.getElementById('disconnect-btn') as HTMLButtonElement;
    this.muteBtn = document.getElementById('mute-btn') as HTMLButtonElement;
    this.clearStreamBtn = document.getElementById('clear-stream-btn') as HTMLButtonElement;
    this.clearTranscriptBtn = document.getElementById('clear-transcript-btn') as HTMLButtonElement;
    this.exportBtn = document.getElementById('export-btn') as HTMLButtonElement;
    this.toggleSidebarBtn = document.getElementById('toggle-sidebar-btn') as HTMLButtonElement;
    this.applyConfigBtn = document.getElementById('apply-config-btn') as HTMLButtonElement;
    this.resetConfigBtn = document.getElementById('reset-config-btn') as HTMLButtonElement;
    this.themeToggleBtn = document.getElementById('theme-toggle') as HTMLButtonElement;

    // Status
    this.connectionIndicator = document.getElementById('connection-indicator');
    this.statusText = document.getElementById('connection-status');
    this.footerStats = document.getElementById('footer-stats');

    // Module badges
    this.sttBadge = document.getElementById('stt-badge');
    this.llmBadge = document.getElementById('llm-badge');
    this.ttsBadge = document.getElementById('tts-badge');
    this.diarBadge = document.getElementById('diar-badge');

    // Model info panel
    this.modelInfoPanel = document.getElementById('model-info-panel');
    this.sttModelName = document.getElementById('stt-model-name');
    this.sttModelParams = document.getElementById('stt-model-params');
    this.llmModelName = document.getElementById('llm-model-name');
    this.llmModelParams = document.getElementById('llm-model-params');
    this.ttsModelName = document.getElementById('tts-model-name');
    this.ttsModelParams = document.getElementById('tts-model-params');
    this.diarModelName = document.getElementById('diar-model-name');
    this.diarModelParams = document.getElementById('diar-model-params');

    // Real-time stream panels (legacy)
    this.realtimeStreamsZone = document.getElementById('realtime-streams-zone');
    this.sttStreamPanel = document.getElementById('realtime-stt-panel');
    this.llmStreamPanel = document.getElementById('realtime-llm-panel');
    this.sttStreamContent = document.getElementById('stt-stream-content');
    this.llmStreamContent = document.getElementById('llm-stream-content');
    this.streamResizeHandle = document.getElementById('stream-resize-handle');

    // NEW: IO Zone elements
    this.inputZone = document.getElementById('input-zone');
    this.outputZone = document.getElementById('output-zone');
    this.ioResizeHandle = document.getElementById('io-resize-handle');
    this.inputStatusBadge = document.getElementById('input-status-badge');
    this.outputStatusBadge = document.getElementById('output-status-badge');
    this.sttLiveIndicator = document.getElementById('stt-live-indicator');
    this.ttsSyncIndicator = document.getElementById('tts-sync-indicator');
    this.speakingIndicator = document.getElementById('speaking-indicator');
    this.speakingStatusLabel = document.getElementById('speaking-status-label');

    // Text Input Zone elements (for text input modes)
    this.textInputZone = document.getElementById('text-input-zone');
    this.textInputTextarea = document.getElementById('text-input-textarea') as HTMLTextAreaElement;
    this.textSendBtn = document.getElementById('text-send-btn') as HTMLButtonElement;
    this.textInputStatusBadge = document.getElementById('text-input-status-badge');

    // Transcript area for STT-only mode layout control
    this.transcriptArea = document.querySelector('.transcript-area') as HTMLElement;

    // Fallback: Create text input zone dynamically if not found in HTML
    // This handles cases where user is running from outdated dist build
    if (!this.textInputZone) {
      console.warn('[DOM] text-input-zone not found in HTML! Creating dynamically...');
      console.warn('[DOM] TIP: Run "npm run build" to update dist, or use "npm run dev" for development.');
      this.createTextInputZoneFallback();
    }

    // Audio File Upload elements
    this.inputModeToggle = document.getElementById('input-mode-toggle');
    this.modeMicBtn = document.getElementById('mode-mic-btn') as HTMLButtonElement;
    this.modeFileBtn = document.getElementById('mode-file-btn') as HTMLButtonElement;
    this.inputModeLabel = document.getElementById('input-mode-label');
    this.fileUploadSection = document.getElementById('file-upload-section');
    this.fileDropzone = document.getElementById('file-dropzone');
    this.fileInput = document.getElementById('file-input') as HTMLInputElement;
    this.fileInfoPanel = document.getElementById('file-info-panel');
    this.uploadFileName = document.getElementById('upload-file-name');
    this.uploadFileMeta = document.getElementById('upload-file-meta');
    this.fileRemoveBtn = document.getElementById('file-remove-btn') as HTMLButtonElement;
    this.uploadProgressFill = document.getElementById('upload-progress-fill');
    this.uploadProgressText = document.getElementById('upload-progress-text');
    this.uploadProgressTime = document.getElementById('upload-progress-time');
    this.uploadSpeedSelect = document.getElementById('upload-speed-select') as HTMLSelectElement;
    this.uploadStartBtn = document.getElementById('upload-start-btn') as HTMLButtonElement;
    this.uploadCancelBtn = document.getElementById('upload-cancel-btn') as HTMLButtonElement;

    // Audio Playback UI Elements
    this.playbackEnableCheckbox = document.getElementById('playback-enable') as HTMLInputElement;
    this.playbackVolumeSlider = document.getElementById('playback-volume') as HTMLInputElement;
    this.playbackVolumeValue = document.getElementById('playback-volume-value');
    this.playbackStatus = document.getElementById('playback-status');
    this.playbackStatusText = this.playbackStatus?.querySelector('.playback-status-text') || null;
    this.playbackControls = document.getElementById('playback-controls');
    this.playbackPositionMarker = document.getElementById('playback-position-marker');
    this.playbackWaveformCanvas = document.getElementById('playback-waveform-canvas') as HTMLCanvasElement;
    this.playbackWaveformPosition = document.getElementById('waveform-cursor');
    this.waveformPlayed = document.getElementById('waveform-played');

    // Volume - Input
    this.volumeBar = document.getElementById('volume-bar');
    this.volumeText = document.getElementById('volume-text');
    this.inputVolumeBar = document.getElementById('input-volume-bar');
    this.inputVolumeText = document.getElementById('input-volume-text');

    // Volume - Output (TTS)
    this.outputVolumeBar = document.getElementById('output-volume-bar');
    this.outputVolumeText = document.getElementById('output-volume-text');

    // Waveform with enhanced visualizer - Input
    this.waveformCanvas = document.getElementById('waveform-canvas') as HTMLCanvasElement;
    if (this.waveformCanvas) {
      this.audioVisualizer = new AudioVisualizer(this.waveformCanvas);
    }
    this.inputWaveformCanvas = document.getElementById('input-waveform-canvas') as HTMLCanvasElement;
    if (this.inputWaveformCanvas) {
      this.inputAudioVisualizer = new AudioVisualizer(this.inputWaveformCanvas);
    }

    // Waveform - Output (TTS)
    this.outputWaveformCanvas = document.getElementById('output-waveform-canvas') as HTMLCanvasElement;
    if (this.outputWaveformCanvas) {
      this.outputAudioVisualizer = new AudioVisualizer(this.outputWaveformCanvas);
      // Set output visualizer to blue color scheme (theme-aware)
      this.outputAudioVisualizer.setColorScheme('blue');
    }

    // Circular visualizer for TTS output (ChatGPT-style)
    this.circularVisualizer = document.getElementById('circular-visualizer') as HTMLCanvasElement;
    if (this.circularVisualizer) {
      this.circularVisualizerCtx = this.circularVisualizer.getContext('2d');
      // Start the circular visualization animation loop
      this.startCircularVisualization();
    }

    // Pipeline stage elements
    this.pipelineStatus = document.getElementById('pipeline-status');
    this.pipelineSTT = document.getElementById('pipeline-stt');
    this.pipelineLLM = document.getElementById('pipeline-llm');
    this.pipelineTTS = document.getElementById('pipeline-tts');

    // History panel elements
    this.historyPanel = document.getElementById('history-panel');
    this.historyList = document.getElementById('history-list');
    this.clearHistoryBtn = document.getElementById('clear-history-btn') as HTMLButtonElement;
    this.historyCollapseBtn = document.getElementById('history-collapse-btn') as HTMLButtonElement;

    // Debug panel elements
    this.debugPanel = document.getElementById('debug-panel');
    this.debugLog = document.getElementById('debug-log');
    this.clearDebugBtn = document.getElementById('clear-debug-btn') as HTMLButtonElement;
    this.debugCollapseBtn = document.getElementById('debug-collapse-btn') as HTMLButtonElement;
    this.debugResizeHandle = document.getElementById('debug-resize-handle');
    this.autoScrollToggle = document.getElementById('auto-scroll-toggle') as HTMLInputElement;

    // Visualization mode button
    this.vizModeBtn = document.getElementById('viz-mode-btn') as HTMLButtonElement;

    // VAD
    this.vadIndicator = document.getElementById('vad-indicator');
    this.vadLabel = document.getElementById('vad-label');
    this.vadTimer = document.getElementById('vad-timer');

    // NEW: Realtime STT Flow elements
    this.sttFlow = document.getElementById('stt-flow');
    this.sttCursor = document.getElementById('stt-cursor');
    this.liveBadge = document.getElementById('live-badge');
    this.historyContent = document.getElementById('history-content');
    this.historyEmpty = document.getElementById('history-empty');
    this.clearFlowBtn = document.getElementById('clear-flow-btn') as HTMLButtonElement;
    this.maxWordsDisplayInput = document.getElementById('max-words-display') as HTMLInputElement;

    // Resize handles and panels
    this.resizeLeftSidebar = document.getElementById('resize-left-sidebar');
    this.resizeRightSidebar = document.getElementById('resize-right-sidebar');
    this.resizeBottomPanel = document.getElementById('resize-bottom-panel');
    this.bottomPanel = document.getElementById('bottom-panel');
    this.monitorsSidebar = document.getElementById('monitors-sidebar');

    // Font and language settings
    this.fontStyleSelect = document.getElementById('font-style-select') as HTMLSelectElement;
    this.fontSizeSlider = document.getElementById('font-size-slider') as HTMLInputElement;
    this.fontSizeValue = document.getElementById('font-size-value');
    this.defaultLanguageSelect = document.getElementById('default-language-select') as HTMLSelectElement;

    // Legacy transcript elements (for backward compatibility)
    this.partialTranscript = document.getElementById('partial-transcript');
    this.finalTranscripts = document.getElementById('final-transcripts');
    this.transcriptCount = document.getElementById('transcript-count');
    this.streamLog = document.getElementById('stream-log');
    this.liveTranscriptArea = document.getElementById('live-transcript');

    // Sidebar
    this.settingsSidebar = document.getElementById('settings-sidebar');
    this.settingsContent = document.getElementById('settings-content');
    this.diarSettings = document.getElementById('diar-settings');

    // Speaker diarization
    this.speakerPanel = document.getElementById('speaker-panel');
    for (let i = 0; i < 4; i++) {
      const indicator = document.getElementById(`speaker-${i}`);
      if (indicator) {
        this.speakerIndicators.push(indicator);
      }
    }

    // Voice Agent Conversation View elements
    this.conversationView = document.getElementById('conversation-view');
    this.botStatusIndicator = document.getElementById('bot-status-indicator');
    this.botThinkingIndicator = document.getElementById('bot-thinking');
    this.clearConversationBtn = document.getElementById('clear-conversation-btn') as HTMLButtonElement;
    this.botStatusText = document.getElementById('bot-status-text');

    // Initialize theme from localStorage or default to dark
    this.initializeTheme();

    // Initialize language from localStorage or env config
    this.initializeLanguage();

    // Initialize display settings (font, size, words per line) from localStorage or env config
    this.initializeDisplaySettings();
  }

  /**
   * Create text input zone dynamically when HTML element is not found.
   * This is a fallback for outdated dist builds that don't have the element.
   */
  private createTextInputZoneFallback(): void {
    // Find the input zone (voice) to insert text input zone before it
    const inputZone = document.getElementById('input-zone');
    const transcriptArea = document.querySelector('.transcript-area');

    if (!transcriptArea) {
      console.error('[DOM] Cannot create text-input-zone: transcript-area not found');
      return;
    }

    // Create the text input zone HTML structure
    const textInputZone = document.createElement('div');
    textInputZone.className = 'io-zone text-input-zone';
    textInputZone.id = 'text-input-zone';
    textInputZone.style.display = 'none';

    textInputZone.innerHTML = `
      <div class="io-zone-header">
        <div class="io-zone-title">
          <svg class="io-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
            <line x1="3" y1="9" x2="21" y2="9"/>
            <line x1="9" y1="21" x2="9" y2="9"/>
          </svg>
          <span class="io-label">Input</span>
          <span class="io-sublabel">Text</span>
        </div>
        <div class="io-zone-controls">
          <div class="io-status-badge text-input-status" id="text-input-status-badge">
            <span class="status-dot"></span>
            <span class="status-label">Ready</span>
          </div>
        </div>
      </div>
      <div class="io-zone-body text-input-body">
        <div class="text-input-wrapper">
          <textarea
            id="text-input-textarea"
            class="text-input-textarea"
            placeholder="메시지를 입력하세요... (Enter로 전송, Shift+Enter로 줄바꿈)"
            rows="3"
          ></textarea>
          <div class="text-input-actions">
            <span class="text-input-hint">Enter to send | Shift+Enter for new line</span>
            <button class="btn btn-primary text-send-btn" id="text-send-btn">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="22" y1="2" x2="11" y2="13"/>
                <polygon points="22 2 15 22 11 13 2 9 22 2"/>
              </svg>
              Send
            </button>
          </div>
        </div>
      </div>
    `;

    // Insert before input zone or append to transcript area
    if (inputZone) {
      inputZone.parentNode?.insertBefore(textInputZone, inputZone);
    } else {
      transcriptArea.appendChild(textInputZone);
    }

    // Update references
    this.textInputZone = textInputZone;
    this.textInputTextarea = document.getElementById('text-input-textarea') as HTMLTextAreaElement;
    this.textSendBtn = document.getElementById('text-send-btn') as HTMLButtonElement;
    this.textInputStatusBadge = document.getElementById('text-input-status-badge');

    console.log('[DOM] Text input zone created dynamically');
  }

  /**
   * Initialize theme from localStorage, env config, or system preference
   */
  private initializeTheme(): void {
    const savedTheme = localStorage.getItem('nemo-theme');
    if (savedTheme) {
      this.isDarkMode = savedTheme === 'dark';
    } else if (import.meta.env.VITE_DEFAULT_THEME) {
      // Use environment config
      this.isDarkMode = import.meta.env.VITE_DEFAULT_THEME === 'dark';
    } else {
      // Check system preference
      this.isDarkMode = !window.matchMedia('(prefers-color-scheme: light)').matches;
    }
    this.applyTheme();
  }

  /**
   * Apply the current theme to the document
   */
  private applyTheme(): void {
    if (this.isDarkMode) {
      document.documentElement.removeAttribute('data-theme');
    } else {
      document.documentElement.setAttribute('data-theme', 'light');
    }
    localStorage.setItem('nemo-theme', this.isDarkMode ? 'dark' : 'light');

    // Update audio visualizers with new theme
    this.audioVisualizer?.setTheme(this.isDarkMode);
    this.inputAudioVisualizer?.setTheme(this.isDarkMode);
    this.outputAudioVisualizer?.setTheme(this.isDarkMode);
  }

  /**
   * Toggle between dark and light themes
   */
  private toggleTheme(): void {
    this.isDarkMode = !this.isDarkMode;
    this.applyTheme();
    this.addStreamEntry('system', `Theme: ${this.isDarkMode ? 'Dark' : 'Light'} mode`);
  }

  /**
   * Clear the flowing STT area only (keep history)
   */
  private clearFlowArea(): void {
    this.flowLines = [];
    this.accumulatedText = '';
    this.currentLineWordCount = 0;
    if (this.sttFlow) {
      this.sttFlow.innerHTML = '';
    }
    if (this.sttCursor) {
      this.sttCursor.style.display = 'none';
    }
    this.addStreamEntry('system', 'STT flow cleared');
  }

  /**
   * Update font style for ALL text display areas
   * Applies to: STT flow, Conversation view, LLM stream panel, STT stream panel
   */
  private updateFont(): void {
    const fontMap: { [key: string]: string } = {
      // Premium Korean fonts (ordered by premium feel)
      'default': "'SUIT Variable', 'SUIT', 'Pretendard Variable', 'Pretendard', sans-serif",
      'premium': "'Gothic A1', 'Spoqa Han Sans Neo', 'Pretendard', sans-serif",
      'modern': "'Pretendard Variable', 'Pretendard', 'Apple SD Gothic Neo', sans-serif",
      'noto-sans': "'Noto Sans KR', sans-serif",
      'casual': "'Gowun Dodum', sans-serif",
      'elegant': "'Noto Serif KR', 'Nanum Myeongjo', serif",
      'nanum': "'Nanum Gothic', sans-serif",
      'handwriting': "'Nanum Pen Script', cursive",
      'playful': "'Gaegu', cursive",
      // English fonts
      'courier': "'Courier New', Courier, monospace",
      'consolas': "Consolas, 'Courier New', monospace",
      'mono': "'JetBrains Mono', 'Fira Code', Consolas, monospace"
    };

    const selectedFont = this.fontStyleSelect?.value || 'default';
    const fontFamily = fontMap[selectedFont] || fontMap['default'];

    // Apply font to STT flow
    if (this.sttFlow) {
      this.sttFlow.style.fontFamily = fontFamily;
    }

    // Apply font to Conversation view (VOICE AGENT window)
    if (this.conversationView) {
      this.conversationView.style.fontFamily = fontFamily;
    }

    // Apply font to LLM stream panel (OUTPUT section)
    if (this.llmStreamContent) {
      this.llmStreamContent.style.fontFamily = fontFamily;
    }

    // Apply font to STT stream panel (OUTPUT section)
    if (this.sttStreamContent) {
      this.sttStreamContent.style.fontFamily = fontFamily;
    }

    // Apply font to message bubbles via CSS custom property
    document.documentElement.style.setProperty('--font-korean-display', fontFamily);

    // Save to localStorage for persistence
    localStorage.setItem('nemo-font-style', selectedFont);
    console.log(`[Font] Applied font: ${selectedFont} → ${fontFamily}`);
  }

  /**
   * Update font size for STT display
   */
  private updateFontSize(): void {
    const size = this.fontSizeSlider?.value || '20';
    const sizeNum = parseInt(size, 10);

    if (this.sttFlow) {
      this.sttFlow.style.fontSize = `${sizeNum}px`;
    }

    if (this.fontSizeValue) {
      this.fontSizeValue.textContent = `${sizeNum}px`;
    }

    // Save to localStorage for persistence
    localStorage.setItem('nemo-font-size', size);
  }

  /**
   * Update default language setting
   */
  private updateLanguage(): void {
    const lang = this.defaultLanguageSelect?.value || 'en';
    this.defaultLanguage = lang;

    // Store in localStorage for persistence
    localStorage.setItem('nemo-default-language', lang);

    this.addStreamEntry('system', `Language: ${lang === 'en' ? 'English' : lang === 'ko' ? 'Korean' : 'Auto Detect'}`);
  }

  /**
   * Initialize language from localStorage or env config
   */
  private initializeLanguage(): void {
    const savedLang = localStorage.getItem('nemo-default-language');
    const envLang = import.meta.env.VITE_DEFAULT_LANGUAGE;

    if (savedLang && this.defaultLanguageSelect) {
      this.defaultLanguage = savedLang;
      this.defaultLanguageSelect.value = savedLang;
    } else if (envLang && this.defaultLanguageSelect) {
      this.defaultLanguage = envLang;
      this.defaultLanguageSelect.value = envLang;
    }
  }

  /**
   * Initialize all display settings from localStorage or env config
   * Called after DOM elements are set up
   */
  private initializeDisplaySettings(): void {
    // Font style
    const savedFont = localStorage.getItem('nemo-font-style');
    const envFont = import.meta.env.VITE_DEFAULT_FONT;
    if (savedFont && this.fontStyleSelect) {
      this.fontStyleSelect.value = savedFont;
    } else if (envFont && this.fontStyleSelect) {
      this.fontStyleSelect.value = envFont;
    }
    this.updateFont();

    // Font size
    const savedSize = localStorage.getItem('nemo-font-size');
    const envSize = import.meta.env.VITE_DEFAULT_FONT_SIZE;
    if (savedSize && this.fontSizeSlider) {
      this.fontSizeSlider.value = savedSize;
    } else if (envSize && this.fontSizeSlider) {
      this.fontSizeSlider.value = envSize;
    }
    this.updateFontSize();

    // Words per line
    const savedWords = localStorage.getItem('nemo-max-words-per-line');
    const envWords = import.meta.env.VITE_MAX_WORDS_PER_LINE;
    if (savedWords && this.maxWordsDisplayInput) {
      const value = parseInt(savedWords, 10);
      this.maxWordsPerLine = Math.max(5, Math.min(50, value));
      this.maxWordsDisplayInput.value = this.maxWordsPerLine.toString();
    } else if (envWords && this.maxWordsDisplayInput) {
      const value = parseInt(envWords, 10);
      this.maxWordsPerLine = Math.max(5, Math.min(50, value));
      this.maxWordsDisplayInput.value = this.maxWordsPerLine.toString();
    }
  }

  /**
   * Setup resize handles for sidebars and bottom panel
   */
  private setupResizeHandles(): void {
    // Left sidebar resize
    if (this.resizeLeftSidebar && this.settingsSidebar) {
      this.setupHorizontalResize(this.resizeLeftSidebar, this.settingsSidebar, 'right');
    }

    // Right sidebar resize
    if (this.resizeRightSidebar && this.monitorsSidebar) {
      this.setupHorizontalResize(this.resizeRightSidebar, this.monitorsSidebar, 'left');
    }

    // Bottom panel resize
    if (this.resizeBottomPanel && this.bottomPanel) {
      this.setupVerticalResize(this.resizeBottomPanel, this.bottomPanel);
    }
  }

  /**
   * Setup horizontal resize for sidebars
   */
  private setupHorizontalResize(handle: HTMLElement, panel: HTMLElement, direction: 'left' | 'right'): void {
    let isResizing = false;
    let startX = 0;
    let startWidth = 0;

    const onMouseDown = (e: MouseEvent) => {
      isResizing = true;
      startX = e.clientX;
      startWidth = panel.offsetWidth;
      handle.classList.add('dragging');
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      e.preventDefault();
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;

      const delta = direction === 'right'
        ? e.clientX - startX
        : startX - e.clientX;

      const newWidth = Math.max(200, Math.min(500, startWidth + delta));
      panel.style.width = `${newWidth}px`;
      panel.style.minWidth = `${newWidth}px`;
      panel.style.maxWidth = `${newWidth}px`;
    };

    const onMouseUp = () => {
      if (!isResizing) return;
      isResizing = false;
      handle.classList.remove('dragging');
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    handle.addEventListener('mousedown', onMouseDown);
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }

  /**
   * Setup vertical resize for bottom panel
   */
  private setupVerticalResize(handle: HTMLElement, panel: HTMLElement): void {
    let isResizing = false;
    let startY = 0;
    let startHeight = 0;

    const onMouseDown = (e: MouseEvent) => {
      isResizing = true;
      startY = e.clientY;
      startHeight = panel.offsetHeight;
      handle.classList.add('dragging');
      document.body.style.cursor = 'row-resize';
      document.body.style.userSelect = 'none';
      e.preventDefault();
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;

      // Dragging up increases height
      const delta = startY - e.clientY;
      const newHeight = Math.max(150, Math.min(500, startHeight + delta));
      panel.style.height = `${newHeight}px`;
      panel.style.minHeight = `${newHeight}px`;
      panel.style.maxHeight = `${newHeight}px`;
    };

    const onMouseUp = () => {
      if (!isResizing) return;
      isResizing = false;
      handle.classList.remove('dragging');
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    handle.addEventListener('mousedown', onMouseDown);
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }

  /**
   * Set up event listeners
   */
  private setupEventListeners(): void {
    this.connectBtn?.addEventListener('click', () => this.connect());
    this.disconnectBtn?.addEventListener('click', () => this.disconnect());
    this.muteBtn?.addEventListener('click', () => this.toggleMute());
    this.clearStreamBtn?.addEventListener('click', () => this.clearStreamLog());
    this.clearTranscriptBtn?.addEventListener('click', () => this.clearTranscripts());
    this.exportBtn?.addEventListener('click', () => this.exportTranscript());
    this.toggleSidebarBtn?.addEventListener('click', () => this.toggleSidebar());
    this.applyConfigBtn?.addEventListener('click', () => this.applyConfiguration());
    this.resetConfigBtn?.addEventListener('click', () => this.resetConfiguration());
    this.vizModeBtn?.addEventListener('click', () => this.cycleVisualizationMode());
    this.themeToggleBtn?.addEventListener('click', () => this.toggleTheme());

    // Text Input event handlers (for text input modes)
    this.textSendBtn?.addEventListener('click', () => this.sendTextInput());
    this.textInputTextarea?.addEventListener('keydown', (e: KeyboardEvent) => {
      // Enter to send (without Shift), Shift+Enter for new line
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendTextInput();
      }
    });
    // Focus/blur styling for text input
    this.textInputTextarea?.addEventListener('focus', () => {
      this.textInputZone?.classList.add('active');
    });
    this.textInputTextarea?.addEventListener('blur', () => {
      this.textInputZone?.classList.remove('active');
    });

    // Audio File Upload event handlers
    this.setupFileUploadEventListeners();

    // Clear flow button
    this.clearFlowBtn?.addEventListener('click', () => this.clearFlowArea());

    // Clear conversation button
    this.clearConversationBtn?.addEventListener('click', () => this.clearConversation());

    // STT panel collapse toggle
    const collapseSTTBtn = document.getElementById('collapse-stt-btn');
    const sttZone = document.getElementById('realtime-stt-zone');
    collapseSTTBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      sttZone?.classList.toggle('collapsed');
    });
    // Also allow clicking the header to collapse
    const sttHeader = document.getElementById('stt-header');
    sttHeader?.addEventListener('click', () => {
      sttZone?.classList.toggle('collapsed');
    });

    // Words per line setting
    this.maxWordsDisplayInput?.addEventListener('change', () => {
      const value = parseInt(this.maxWordsDisplayInput?.value || '15', 10);
      this.maxWordsPerLine = Math.max(5, Math.min(50, value));
      // Save to localStorage for persistence
      localStorage.setItem('nemo-max-words-per-line', this.maxWordsPerLine.toString());
      this.addStreamEntry('system', `Words/Line set to: ${this.maxWordsPerLine}`);
    });

    // Setup resize handles
    this.setupResizeHandles();

    // Font and language settings
    this.fontStyleSelect?.addEventListener('change', () => this.updateFont());
    this.fontSizeSlider?.addEventListener('input', () => this.updateFontSize());
    this.defaultLanguageSelect?.addEventListener('change', () => this.updateLanguage());

    // Stream header collapse toggle
    const streamHeader = document.getElementById('stream-header');
    streamHeader?.addEventListener('click', (e) => {
      if (!(e.target as HTMLElement).closest('.btn-icon')) {
        const streamCard = streamHeader.closest('.stream-card');
        streamCard?.classList.toggle('collapsed');
      }
    });

    // NEW: IO Zone stream clear buttons
    const clearSttStreamBtn = document.getElementById('clear-stt-stream-btn');
    clearSttStreamBtn?.addEventListener('click', () => this.clearSTTStream());
    const clearLlmStreamBtn = document.getElementById('clear-llm-stream-btn');
    clearLlmStreamBtn?.addEventListener('click', () => this.clearLLMStream());

    // NEW: IO Zone resize handle
    this.setupIOZoneResize();

    // NEW: History panel buttons
    this.clearHistoryBtn?.addEventListener('click', () => this.clearHistory());
    this.historyCollapseBtn?.addEventListener('click', () => this.toggleHistoryPanel());

    // NEW: Debug panel buttons
    this.clearDebugBtn?.addEventListener('click', () => this.clearDebugLog());
    this.debugCollapseBtn?.addEventListener('click', () => this.toggleDebugPanel());

    // NEW: Debug panel resize handle
    this.setupDebugPanelResize();
  }

  /**
   * Setup resize handle for IO zones (vertical resize between input and output)
   */
  private setupIOZoneResize(): void {
    if (!this.ioResizeHandle || !this.inputZone || !this.outputZone) return;

    let isResizing = false;
    let startY = 0;
    let startInputHeight = 0;
    let startOutputHeight = 0;

    this.ioResizeHandle.addEventListener('mousedown', (e: MouseEvent) => {
      isResizing = true;
      startY = e.clientY;
      startInputHeight = this.inputZone!.offsetHeight;
      startOutputHeight = this.outputZone!.offsetHeight;
      document.body.style.cursor = 'row-resize';
      document.body.style.userSelect = 'none';
      this.ioResizeHandle!.classList.add('dragging');
    });

    document.addEventListener('mousemove', (e: MouseEvent) => {
      if (!isResizing) return;

      const diff = e.clientY - startY;
      const totalHeight = startInputHeight + startOutputHeight;
      const newInputHeight = Math.max(150, Math.min(totalHeight - 150, startInputHeight + diff));
      const newOutputHeight = totalHeight - newInputHeight;

      this.inputZone!.style.flex = `0 0 ${newInputHeight}px`;
      this.outputZone!.style.flex = `0 0 ${newOutputHeight}px`;
    });

    document.addEventListener('mouseup', () => {
      if (isResizing) {
        isResizing = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        this.ioResizeHandle!.classList.remove('dragging');
      }
    });
  }

  /**
   * Clear history panel
   */
  private clearHistory(): void {
    if (this.historyList) {
      this.historyList.innerHTML = '<div class="history-empty"><span class="empty-text">대화 기록이 여기에 표시됩니다</span></div>';
    }
    this.addDebugEntry('system', 'History cleared');
  }

  /**
   * Toggle history panel collapse state
   */
  private toggleHistoryPanel(): void {
    this.historyPanel?.classList.toggle('collapsed');
  }

  /**
   * Clear debug log
   */
  private clearDebugLog(): void {
    if (this.debugLog) {
      this.debugLog.innerHTML = '<div class="debug-entry debug-info"><span class="debug-time">--:--:--</span><span class="debug-msg">Log cleared</span></div>';
    }
  }

  /**
   * Toggle debug panel collapse state
   */
  private toggleDebugPanel(): void {
    this.debugPanel?.classList.toggle('collapsed');
  }

  /**
   * Setup resize handle for debug panel (vertical resize)
   */
  private setupDebugPanelResize(): void {
    if (!this.debugResizeHandle || !this.debugPanel) return;

    let isResizing = false;
    let startY = 0;
    let startHeight = 0;

    this.debugResizeHandle.addEventListener('mousedown', (e: MouseEvent) => {
      isResizing = true;
      startY = e.clientY;
      startHeight = this.debugPanel!.offsetHeight;
      document.body.style.cursor = 'row-resize';
      document.body.style.userSelect = 'none';
      this.debugResizeHandle!.classList.add('active');
    });

    document.addEventListener('mousemove', (e: MouseEvent) => {
      if (!isResizing) return;

      const diff = startY - e.clientY;  // Inverted because we're resizing from top
      const newHeight = Math.max(80, Math.min(400, startHeight + diff));

      this.debugPanel!.style.flex = `0 0 ${newHeight}px`;
      this.debugPanel!.style.minHeight = `${newHeight}px`;
    });

    document.addEventListener('mouseup', () => {
      if (isResizing) {
        isResizing = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        this.debugResizeHandle!.classList.remove('active');
      }
    });
  }

  // ============================================================================
  // AUDIO FILE UPLOAD METHODS
  // ============================================================================

  /**
   * Set up event listeners for file upload functionality
   */
  private setupFileUploadEventListeners(): void {
    // Input mode toggle buttons
    this.modeMicBtn?.addEventListener('click', () => this.setAudioInputMode(AudioInputMode.MICROPHONE));
    this.modeFileBtn?.addEventListener('click', () => this.setAudioInputMode(AudioInputMode.FILE));

    // File dropzone click to open file dialog
    this.fileDropzone?.addEventListener('click', () => this.fileInput?.click());

    // File input change handler
    this.fileInput?.addEventListener('change', (e: Event) => {
      const target = e.target as HTMLInputElement;
      if (target.files && target.files.length > 0) {
        this.handleFileSelection(target.files[0]);
      }
    });

    // Drag and drop handlers
    this.fileDropzone?.addEventListener('dragover', (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      this.fileDropzone?.classList.add('drag-over');
    });

    this.fileDropzone?.addEventListener('dragleave', (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      this.fileDropzone?.classList.remove('drag-over');
    });

    this.fileDropzone?.addEventListener('drop', (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      this.fileDropzone?.classList.remove('drag-over');

      const files = e.dataTransfer?.files;
      if (files && files.length > 0) {
        this.handleFileSelection(files[0]);
      }
    });

    // File remove button
    this.fileRemoveBtn?.addEventListener('click', () => this.removeSelectedFile());

    // Upload start button
    this.uploadStartBtn?.addEventListener('click', () => this.startFileUpload());

    // Upload cancel button
    this.uploadCancelBtn?.addEventListener('click', () => this.cancelFileUpload());

    // Speed select change
    this.uploadSpeedSelect?.addEventListener('change', () => {
      if (this.audioFileUploader) {
        const speed = parseFloat(this.uploadSpeedSelect?.value || '1');
        this.audioFileUploader.setUploadSpeed(speed);
        this.addDebugEntry('info', `Upload speed set to ${speed === 0 ? 'Max' : speed + 'x'}`);

        // Update playback controls state (disabled at max speed)
        this.updatePlaybackControlsState(speed);
      }
    });

    // Playback enable checkbox change
    this.playbackEnableCheckbox?.addEventListener('change', () => {
      if (this.audioFileUploader) {
        const enabled = this.playbackEnableCheckbox?.checked || false;
        this.audioFileUploader.setPlaybackEnabled(enabled);
        this.updatePlaybackStatusUI(enabled ? 'Ready' : 'Disabled');
        this.addDebugEntry('info', `Playback ${enabled ? 'enabled' : 'disabled'}`);
      }
    });

    // Playback volume slider change
    this.playbackVolumeSlider?.addEventListener('input', () => {
      const volume = parseInt(this.playbackVolumeSlider?.value || '80', 10) / 100;
      if (this.audioFileUploader) {
        this.audioFileUploader.setPlaybackVolume(volume);
      }
      if (this.playbackVolumeValue) {
        this.playbackVolumeValue.textContent = `${Math.round(volume * 100)}%`;
      }
    });
  }

  /**
   * Update playback controls UI state based on speed setting
   */
  private updatePlaybackControlsState(speed: number): void {
    const isMaxSpeed = speed === 0;

    if (this.playbackControls) {
      this.playbackControls.classList.toggle('disabled', isMaxSpeed);
    }

    if (this.playbackEnableCheckbox) {
      this.playbackEnableCheckbox.disabled = isMaxSpeed;
      if (isMaxSpeed) {
        this.playbackEnableCheckbox.checked = false;
      }
    }

    this.updatePlaybackStatusUI(isMaxSpeed ? 'Max speed' : 'Ready');
  }

  /**
   * Update playback status UI
   */
  private updatePlaybackStatusUI(status: string, isPlaying: boolean = false): void {
    if (this.playbackStatus) {
      this.playbackStatus.classList.toggle('playing', isPlaying);
      this.playbackStatus.classList.toggle('disabled', status === 'Disabled' || status === 'Max speed');
    }
    if (this.playbackStatusText) {
      this.playbackStatusText.textContent = status;
    }
    // Toggle waveform cursor visibility
    if (this.playbackWaveformPosition) {
      this.playbackWaveformPosition.classList.toggle('active', isPlaying);
    }
  }

  // Store waveform data for redraws
  private currentWaveformData: number[] = [];

  /**
   * Render ultra-premium waveform visualization on canvas
   * Creates a smooth, professional-grade audio waveform with bezier curves and reflections
   */
  private renderWaveform(waveformData: number[], playedPercent: number = 0): void {
    // Try to acquire canvas reference if not available
    if (!this.playbackWaveformCanvas) {
      this.playbackWaveformCanvas = document.getElementById('playback-waveform-canvas') as HTMLCanvasElement;
      if (!this.playbackWaveformCanvas) {
        console.warn('[Waveform] Canvas element not found in DOM');
        return;
      }
      console.log('[Waveform] Canvas reference acquired on demand');
    }

    const canvas = this.playbackWaveformCanvas;
    const ctx = canvas.getContext('2d', { alpha: true });
    if (!ctx) {
      console.warn('[Waveform] Could not get 2D context');
      return;
    }

    // Store for position updates
    this.currentWaveformData = waveformData;

    // Set canvas size with high DPI support for crisp rendering
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    if (rect.width === 0 || rect.height === 0) {
      console.warn('[Waveform] Canvas has zero dimensions');
      return;
    }

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    // Clear with transparency
    ctx.clearRect(0, 0, rect.width, rect.height);

    // Theme detection
    const isDarkMode = document.documentElement.getAttribute('data-theme') !== 'light';

    // Premium color palette
    const colors = isDarkMode ? {
      unplayedStart: 'rgba(71, 85, 105, 0.8)',      // Slate-600
      unplayedEnd: 'rgba(51, 65, 85, 0.4)',         // Slate-700
      playedStart: 'rgba(59, 130, 246, 1)',         // Blue-500
      playedEnd: 'rgba(37, 99, 235, 0.9)',          // Blue-600
      glowPrimary: 'rgba(59, 130, 246, 0.6)',       // Blue glow
      glowSecondary: 'rgba(147, 51, 234, 0.4)',     // Purple accent
      reflection: 'rgba(255, 255, 255, 0.08)',
      cursorGlow: 'rgba(59, 130, 246, 0.9)'
    } : {
      unplayedStart: 'rgba(148, 163, 184, 0.9)',    // Slate-400
      unplayedEnd: 'rgba(203, 213, 225, 0.5)',      // Slate-300
      playedStart: 'rgba(37, 99, 235, 1)',          // Blue-600
      playedEnd: 'rgba(29, 78, 216, 0.85)',         // Blue-700
      glowPrimary: 'rgba(59, 130, 246, 0.5)',
      glowSecondary: 'rgba(124, 58, 237, 0.3)',
      reflection: 'rgba(0, 0, 0, 0.05)',
      cursorGlow: 'rgba(37, 99, 235, 0.9)'
    };

    const width = rect.width;
    const height = rect.height;
    const centerY = height * 0.45;  // Slightly above center for reflection space
    const maxAmplitude = height * 0.38;
    const reflectionHeight = height * 0.18;
    const playedX = (playedPercent / 100) * width;

    // Upsample waveform data for smoother curves
    const numPoints = Math.max(waveformData.length * 2, 200);
    const smoothedData: number[] = [];
    for (let i = 0; i < numPoints; i++) {
      const t = i / (numPoints - 1);
      const sourceIndex = t * (waveformData.length - 1);
      const i0 = Math.floor(sourceIndex);
      const i1 = Math.min(i0 + 1, waveformData.length - 1);
      const frac = sourceIndex - i0;
      // Smooth interpolation (cosine)
      const smoothFrac = (1 - Math.cos(frac * Math.PI)) / 2;
      smoothedData.push(waveformData[i0] * (1 - smoothFrac) + waveformData[i1] * smoothFrac);
    }

    // Helper function to draw smooth waveform path
    const drawWaveformPath = (data: number[], yCenter: number, amplitude: number, flip: boolean = false): void => {
      ctx.beginPath();
      const sign = flip ? 1 : -1;

      for (let i = 0; i < data.length; i++) {
        const x = (i / (data.length - 1)) * width;
        const y = yCenter + sign * data[i] * amplitude;

        if (i === 0) {
          ctx.moveTo(x, yCenter);
          ctx.lineTo(x, y);
        } else {
          // Use bezier curves for smooth connections
          const prevX = ((i - 1) / (data.length - 1)) * width;
          const prevY = yCenter + sign * data[i - 1] * amplitude;
          const cpx = (prevX + x) / 2;
          ctx.bezierCurveTo(cpx, prevY, cpx, y, x, y);
        }
      }

      // Close path back to center
      ctx.lineTo(width, yCenter);
      ctx.lineTo(0, yCenter);
      ctx.closePath();
    };

    // Draw unplayed waveform (full width)
    ctx.save();
    const unplayedGradient = ctx.createLinearGradient(0, centerY - maxAmplitude, 0, centerY + maxAmplitude);
    unplayedGradient.addColorStop(0, colors.unplayedStart);
    unplayedGradient.addColorStop(0.5, colors.unplayedEnd);
    unplayedGradient.addColorStop(1, colors.unplayedStart);

    // Draw upper half
    ctx.fillStyle = unplayedGradient;
    drawWaveformPath(smoothedData, centerY, maxAmplitude, false);
    ctx.fill();

    // Draw lower half (mirror)
    drawWaveformPath(smoothedData, centerY, maxAmplitude, true);
    ctx.fill();
    ctx.restore();

    // Draw played portion with premium gradient and glow
    if (playedPercent > 0) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, playedX, height);
      ctx.clip();

      // Multi-stop gradient for played section
      const playedGradient = ctx.createLinearGradient(0, centerY - maxAmplitude, 0, centerY + maxAmplitude);
      playedGradient.addColorStop(0, colors.playedStart);
      playedGradient.addColorStop(0.3, colors.glowSecondary);
      playedGradient.addColorStop(0.5, colors.playedEnd);
      playedGradient.addColorStop(0.7, colors.glowSecondary);
      playedGradient.addColorStop(1, colors.playedStart);

      // Add glow effect
      ctx.shadowColor = colors.glowPrimary;
      ctx.shadowBlur = 8;
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 0;

      ctx.fillStyle = playedGradient;
      drawWaveformPath(smoothedData, centerY, maxAmplitude, false);
      ctx.fill();
      drawWaveformPath(smoothedData, centerY, maxAmplitude, true);
      ctx.fill();

      ctx.shadowBlur = 0;
      ctx.restore();
    }

    // Draw reflection (subtle mirror effect)
    ctx.save();
    const reflectionY = centerY + maxAmplitude + 2;
    const reflectionGradient = ctx.createLinearGradient(0, reflectionY, 0, reflectionY + reflectionHeight);
    reflectionGradient.addColorStop(0, colors.reflection);
    reflectionGradient.addColorStop(1, 'transparent');

    ctx.globalAlpha = 0.3;
    ctx.fillStyle = reflectionGradient;

    ctx.beginPath();
    for (let i = 0; i < smoothedData.length; i++) {
      const x = (i / (smoothedData.length - 1)) * width;
      const y = reflectionY + smoothedData[i] * reflectionHeight * 0.6;
      if (i === 0) {
        ctx.moveTo(x, reflectionY);
        ctx.lineTo(x, y);
      } else {
        const prevX = ((i - 1) / (smoothedData.length - 1)) * width;
        const prevY = reflectionY + smoothedData[i - 1] * reflectionHeight * 0.6;
        const cpx = (prevX + x) / 2;
        ctx.bezierCurveTo(cpx, prevY, cpx, y, x, y);
      }
    }
    ctx.lineTo(width, reflectionY);
    ctx.lineTo(0, reflectionY);
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    // Draw playback cursor line
    if (playedPercent > 0 && playedPercent < 100) {
      ctx.save();
      ctx.strokeStyle = colors.cursorGlow;
      ctx.lineWidth = 2;
      ctx.shadowColor = colors.cursorGlow;
      ctx.shadowBlur = 6;

      ctx.beginPath();
      ctx.moveTo(playedX, centerY - maxAmplitude - 4);
      ctx.lineTo(playedX, centerY + maxAmplitude + 4);
      ctx.stroke();

      // Cursor head (diamond shape)
      ctx.fillStyle = colors.cursorGlow;
      ctx.beginPath();
      ctx.moveTo(playedX, centerY - maxAmplitude - 6);
      ctx.lineTo(playedX + 4, centerY - maxAmplitude - 2);
      ctx.lineTo(playedX, centerY - maxAmplitude + 2);
      ctx.lineTo(playedX - 4, centerY - maxAmplitude - 2);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    console.log(`[Waveform] Premium render: ${smoothedData.length} points, ${playedPercent.toFixed(1)}% played`);
  }

  /**
   * Update waveform position indicator during playback
   * Redraws waveform with played/unplayed sections
   */
  private updateWaveformPosition(progress: number): void {
    // Update the waveform overlay for played section
    if (this.waveformPlayed) {
      this.waveformPlayed.style.width = `${progress}%`;
    }

    // Update cursor position
    if (this.playbackWaveformPosition) {
      this.playbackWaveformPosition.style.left = `${progress}%`;
    }

    // Update progress bar marker
    if (this.playbackPositionMarker) {
      this.playbackPositionMarker.style.left = `${progress}%`;
      this.playbackPositionMarker.classList.toggle('active', progress > 0 && progress < 100);
    }

    // Redraw waveform with updated played position for real-time color update
    if (this.currentWaveformData.length > 0) {
      this.renderWaveform(this.currentWaveformData, progress);
    }
  }

  /**
   * Set audio input mode (microphone or file)
   */
  private setAudioInputMode(mode: AudioInputMode): void {
    if (this.audioInputMode === mode) return;

    this.audioInputMode = mode;
    this.addDebugEntry('info', `Audio input mode: ${mode}`);

    // Update toggle button states
    if (mode === AudioInputMode.MICROPHONE) {
      this.modeMicBtn?.classList.add('active');
      this.modeFileBtn?.classList.remove('active');
      this.inputZone?.classList.remove('file-mode');
      if (this.inputModeLabel) this.inputModeLabel.textContent = 'Microphone';
    } else {
      this.modeMicBtn?.classList.remove('active');
      this.modeFileBtn?.classList.add('active');
      this.inputZone?.classList.add('file-mode');
      if (this.inputModeLabel) this.inputModeLabel.textContent = 'File Upload';
    }

    // Show/hide file upload section
    if (this.fileUploadSection) {
      this.fileUploadSection.style.display = mode === AudioInputMode.FILE ? 'flex' : 'none';
    }
  }

  /**
   * Handle file selection (from input or drag-drop)
   */
  private handleFileSelection(file: File): void {
    // Validate file type
    const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/ogg', 'audio/m4a',
                        'audio/flac', 'audio/webm', 'audio/x-wav', 'audio/x-m4a'];
    const validExtensions = ['.wav', '.mp3', '.ogg', '.m4a', '.flac', '.webm'];

    const hasValidType = validTypes.some(type => file.type === type || file.type.startsWith('audio/'));
    const hasValidExtension = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));

    if (!hasValidType && !hasValidExtension) {
      this.addDebugEntry('error', `Invalid file type: ${file.type || 'unknown'}`);
      alert('Please select an audio file (WAV, MP3, OGG, M4A, FLAC, or WebM)');
      return;
    }

    // Validate file size (max 500MB)
    const maxSize = 500 * 1024 * 1024; // 500MB
    if (file.size > maxSize) {
      this.addDebugEntry('error', `File too large: ${(file.size / 1024 / 1024).toFixed(1)} MB`);
      alert('File size exceeds 500MB limit');
      return;
    }

    this.selectedFile = file;
    this.addDebugEntry('info', `File selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`);

    // Show file info panel
    this.showFileInfoPanel(file);
  }

  /**
   * Show file info panel with selected file details
   */
  private showFileInfoPanel(file: File): void {
    if (!this.fileInfoPanel) return;

    // Hide dropzone, show info panel
    if (this.fileDropzone) this.fileDropzone.style.display = 'none';
    this.fileInfoPanel.style.display = 'flex';

    // Re-acquire canvas reference now that panel is visible
    // This ensures the canvas is properly initialized
    if (!this.playbackWaveformCanvas) {
      this.playbackWaveformCanvas = document.getElementById('playback-waveform-canvas') as HTMLCanvasElement;
      console.log('[Waveform] Re-acquired canvas reference:', !!this.playbackWaveformCanvas);
    }
    if (!this.playbackWaveformPosition) {
      this.playbackWaveformPosition = document.getElementById('waveform-cursor');
    }
    if (!this.waveformPlayed) {
      this.waveformPlayed = document.getElementById('waveform-played');
    }

    // Update file info
    if (this.uploadFileName) this.uploadFileName.textContent = file.name;
    if (this.uploadFileMeta) {
      const sizeStr = file.size < 1024 * 1024
        ? `${(file.size / 1024).toFixed(1)} KB`
        : `${(file.size / 1024 / 1024).toFixed(1)} MB`;
      this.uploadFileMeta.textContent = `${sizeStr} • ${file.type || 'audio file'}`;
    }

    // Reset progress
    this.resetUploadProgress();

    // Show start button, hide cancel button
    if (this.uploadStartBtn) this.uploadStartBtn.style.display = 'inline-flex';
    if (this.uploadCancelBtn) this.uploadCancelBtn.style.display = 'none';
  }

  /**
   * Remove selected file and reset UI
   */
  private removeSelectedFile(): void {
    this.selectedFile = null;

    // Cancel any ongoing upload
    if (this.audioFileUploader) {
      this.audioFileUploader.cancel();
      this.audioFileUploader = null;
    }

    // Reset file input
    if (this.fileInput) this.fileInput.value = '';

    // Show dropzone, hide info panel
    if (this.fileDropzone) this.fileDropzone.style.display = 'flex';
    if (this.fileInfoPanel) this.fileInfoPanel.style.display = 'none';

    // Reset progress
    this.resetUploadProgress();

    this.addDebugEntry('info', 'File removed');
  }

  /**
   * Reset upload progress UI
   */
  private resetUploadProgress(): void {
    if (this.uploadProgressFill) this.uploadProgressFill.style.width = '0%';
    if (this.uploadProgressText) this.uploadProgressText.textContent = '0%';
    if (this.uploadProgressTime) this.uploadProgressTime.textContent = '0:00 / 0:00';
    this.fileInfoPanel?.classList.remove('uploading', 'completed', 'error');
  }

  /**
   * Start file upload (streaming to server)
   */
  private async startFileUpload(): Promise<void> {
    if (!this.selectedFile) {
      this.addDebugEntry('error', 'No file selected');
      return;
    }

    const ws = getInterceptedWebSocket();
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      this.addDebugEntry('error', 'WebSocket not connected. Please connect first.');
      alert('Please connect to the server first before uploading audio.');
      return;
    }

    // Create audio file uploader
    this.audioFileUploader = new AudioFileUploader();

    // Set upload speed
    const speed = parseFloat(this.uploadSpeedSelect?.value || '1');
    this.audioFileUploader.setUploadSpeed(speed);

    // Set progress callback
    this.audioFileUploader.onProgress = (progress, elapsed, total, status) => {
      this.updateUploadProgress(progress, elapsed, total, status);
    };

    // Set complete callback
    this.audioFileUploader.onComplete = (success, message, stats) => {
      this.handleUploadComplete(success, message, stats);
    };

    // Set waveform ready callback
    this.audioFileUploader.onWaveformReady = (waveformData) => {
      this.renderWaveform(waveformData);
    };

    // Set playback state change callback
    this.audioFileUploader.onPlaybackStateChange = (isPlaying, position) => {
      this.updatePlaybackStatusUI(isPlaying ? 'Playing' : 'Stopped', isPlaying);

      // Update waveform position based on progress percentage
      if (this.audioFileUploader && isPlaying) {
        // Position is in seconds, we'll update based on progress in updateUploadProgress
      }
    };

    // Set initial playback volume from UI
    const initialVolume = parseInt(this.playbackVolumeSlider?.value || '80', 10) / 100;
    this.audioFileUploader.setPlaybackVolume(initialVolume);

    // Set initial playback enabled state from UI
    const playbackEnabled = this.playbackEnableCheckbox?.checked ?? true;
    this.audioFileUploader.setPlaybackEnabled(playbackEnabled);

    // Update UI for uploading state
    this.fileInfoPanel?.classList.add('uploading');
    if (this.uploadStartBtn) this.uploadStartBtn.style.display = 'none';
    if (this.uploadCancelBtn) this.uploadCancelBtn.style.display = 'inline-flex';

    this.addDebugEntry('info', `Starting upload: ${this.selectedFile.name}`);

    try {
      await this.audioFileUploader.uploadFile(this.selectedFile);
    } catch (error) {
      this.addDebugEntry('error', `Upload failed: ${error}`);
      this.handleUploadComplete(false, String(error), {
        filename: this.selectedFile?.name || 'unknown',
        duration: 0,
        chunks: 0,
        bytesSent: 0
      });
    }
  }

  /**
   * Cancel ongoing file upload
   */
  private cancelFileUpload(): void {
    if (this.audioFileUploader) {
      this.audioFileUploader.cancel();
      this.addDebugEntry('info', 'Upload cancelled');
    }

    // Reset UI
    this.fileInfoPanel?.classList.remove('uploading');
    if (this.uploadStartBtn) this.uploadStartBtn.style.display = 'inline-flex';
    if (this.uploadCancelBtn) this.uploadCancelBtn.style.display = 'none';
  }

  /**
   * Update upload progress UI
   */
  private updateUploadProgress(progress: number, elapsed: number, total: number, status: string): void {
    if (this.uploadProgressFill) {
      this.uploadProgressFill.style.width = `${progress.toFixed(1)}%`;
    }
    if (this.uploadProgressText) {
      this.uploadProgressText.textContent = `${progress.toFixed(1)}%`;
    }
    if (this.uploadProgressTime) {
      const formatTime = (secs: number) => {
        const mins = Math.floor(secs / 60);
        const s = Math.floor(secs % 60);
        return `${mins}:${s.toString().padStart(2, '0')}`;
      };
      this.uploadProgressTime.textContent = `${formatTime(elapsed)} / ${formatTime(total)}`;
    }

    // Update waveform position indicator during playback
    if (this.audioFileUploader?.playing) {
      this.updateWaveformPosition(progress);
    }
  }

  /**
   * Handle upload completion
   */
  private handleUploadComplete(
    success: boolean,
    message: string,
    stats: { filename: string; duration: number; chunks: number; bytesSent: number }
  ): void {
    this.fileInfoPanel?.classList.remove('uploading');

    if (success) {
      this.fileInfoPanel?.classList.add('completed');
      this.addDebugEntry('info', `Upload complete: ${stats.filename} (${stats.chunks} chunks, ${(stats.bytesSent / 1024).toFixed(1)} KB)`);
    } else {
      this.fileInfoPanel?.classList.add('error');
      this.addDebugEntry('error', `Upload failed: ${message}`);
    }

    // Reset buttons
    if (this.uploadStartBtn) this.uploadStartBtn.style.display = 'inline-flex';
    if (this.uploadCancelBtn) this.uploadCancelBtn.style.display = 'none';

    // Reset playback status UI
    this.updatePlaybackStatusUI(success ? 'Complete' : 'Stopped', false);
    this.updateWaveformPosition(success ? 100 : 0);

    // Cleanup uploader
    this.audioFileUploader = null;
  }

  // ============================================================================
  // END AUDIO FILE UPLOAD METHODS
  // ============================================================================

  /**
   * Add entry to debug log
   */
  private addDebugEntry(type: 'info' | 'warn' | 'error' | 'system' | 'stt' | 'tts' | 'llm', message: string): void {
    if (!this.debugLog) return;

    const now = new Date();
    const time = now.toTimeString().slice(0, 8);

    const entry = document.createElement('div');
    entry.className = `debug-entry debug-${type}`;
    entry.innerHTML = `<span class="debug-time">${time}</span><span class="debug-msg">${message}</span>`;

    this.debugLog.appendChild(entry);

    // Keep only last 150 entries
    while (this.debugLog.children.length > 150) {
      this.debugLog.removeChild(this.debugLog.firstChild!);
    }

    // Auto-scroll if enabled (always show latest)
    if (this.autoScrollToggle?.checked) {
      this.debugLog.scrollTop = this.debugLog.scrollHeight;
    }
  }

  /**
   * Add entry to history panel
   */
  private addHistoryEntry(role: 'user' | 'bot', text: string): void {
    if (!this.historyList) return;

    // Remove empty message if present
    const emptyMsg = this.historyList.querySelector('.history-empty');
    if (emptyMsg) emptyMsg.remove();

    const now = new Date();
    const time = now.toTimeString().slice(0, 5);

    const entry = document.createElement('div');
    entry.className = `history-item history-${role}`;
    entry.innerHTML = `
      <div class="history-header">
        <span class="history-role">${role === 'user' ? '<span class="role-icon user-icon"></span> User' : '<span class="role-icon bot-icon"></span> Assistant'}</span>
        <span class="history-time">${time}</span>
      </div>
      <div class="history-text">${text}</div>
    `;

    this.historyList.appendChild(entry);
    this.historyList.scrollTop = this.historyList.scrollHeight;
  }

  /**
   * Update pipeline stage status
   */
  private updatePipelineStage(stage: 'stt' | 'llm' | 'tts', active: boolean): void {
    const stageElement = stage === 'stt' ? this.pipelineSTT : stage === 'llm' ? this.pipelineLLM : this.pipelineTTS;
    if (stageElement) {
      stageElement.classList.toggle('active', active);
    }
  }

  /**
   * Update pipeline overall status
   */
  private updatePipelineStatus(status: 'standby' | 'listening' | 'processing' | 'speaking'): void {
    if (!this.pipelineStatus) return;

    const dot = this.pipelineStatus.querySelector('.pipeline-status-dot');
    const text = this.pipelineStatus.querySelector('.pipeline-status-text');

    if (dot) {
      dot.className = 'pipeline-status-dot';
      dot.classList.add(status);
    }
    if (text) {
      const statusMap: Record<string, string> = {
        'standby': 'Standby',
        'listening': 'Listening',
        'processing': 'Processing',
        'speaking': 'Speaking'
      };
      text.textContent = statusMap[status] || 'Standby';
    }
  }

  /**
   * Update input status badge
   */
  private updateInputStatus(status: 'ready' | 'listening' | 'processing'): void {
    if (!this.inputStatusBadge) return;

    const dot = this.inputStatusBadge.querySelector('.status-dot');
    const label = this.inputStatusBadge.querySelector('.status-label');

    this.inputStatusBadge.className = 'io-status-badge';
    this.inputStatusBadge.classList.add(status);

    if (label) {
      const statusMap: Record<string, string> = {
        'ready': 'Ready',
        'listening': 'Listening',
        'processing': 'Processing'
      };
      label.textContent = statusMap[status] || 'Ready';
    }
  }

  /**
   * Update output status badge and active state
   */
  private updateOutputStatus(status: 'idle' | 'thinking' | 'speaking'): void {
    if (!this.outputStatusBadge) return;

    const label = this.outputStatusBadge.querySelector('.status-label');
    const isSpeaking = status === 'speaking';

    this.outputStatusBadge.className = 'io-status-badge output-status';
    this.outputStatusBadge.classList.add(status);

    if (label) {
      const statusMap: Record<string, string> = {
        'idle': 'Idle',
        'thinking': 'Thinking',
        'speaking': 'Speaking'
      };
      label.textContent = statusMap[status] || 'Idle';
    }

    // Toggle output zone active state (glowing border)
    if (this.outputZone) {
      this.outputZone.classList.toggle('active', isSpeaking);
    }

    // Toggle speaking indicator animation
    if (this.speakingIndicator) {
      this.speakingIndicator.classList.toggle('active', isSpeaking);
    }

    // Update speaking status label (Korean)
    if (this.speakingStatusLabel) {
      this.speakingStatusLabel.textContent = isSpeaking ? '응답 중' : '대기 중';
    }
  }

  /**
   * Update STT live indicator
   */
  private updateSTTLiveIndicator(active: boolean): void {
    if (this.sttLiveIndicator) {
      this.sttLiveIndicator.classList.toggle('active', active);
    }
  }

  /**
   * Update TTS sync indicator
   */
  private updateTTSSyncIndicator(active: boolean): void {
    if (this.ttsSyncIndicator) {
      this.ttsSyncIndicator.classList.toggle('active', active);
    }
  }

  /**
   * Cycle through visualization modes
   */
  private cycleVisualizationMode(): void {
    const modes: Array<'bars' | 'wave' | 'circular'> = ['bars', 'wave', 'circular'];
    const currentIndex = modes.indexOf(this.vizMode);
    this.vizMode = modes[(currentIndex + 1) % modes.length];

    // Update all visualizers
    if (this.audioVisualizer) {
      this.audioVisualizer.setMode(this.vizMode);
    }
    if (this.inputAudioVisualizer) {
      this.inputAudioVisualizer.setMode(this.vizMode);
    }
    if (this.outputAudioVisualizer) {
      this.outputAudioVisualizer.setMode(this.vizMode);
    }

    // Update button label
    if (this.vizModeBtn) {
      const label = this.vizModeBtn.querySelector('.btn-text');
      if (label) {
        label.textContent = this.vizMode.charAt(0).toUpperCase() + this.vizMode.slice(1);
      }
    }

    this.addStreamEntry('system', `Visualization mode: ${this.vizMode}`);
  }

  /**
   * Set up settings panel interactions
   */
  private setupSettingsPanel(): void {
    // Section collapse toggles
    const sectionHeaders = document.querySelectorAll('.section-header');
    sectionHeaders.forEach(header => {
      header.addEventListener('click', () => {
        const section = header.closest('.settings-section');
        section?.classList.toggle('collapsed');
      });
    });

    // Slider value updates
    this.setupSlider('vad-confidence', 'vad.confidence', (v) => v.toFixed(2));
    this.setupSlider('vad-stop-secs', 'vad.stop_secs', (v) => `${v.toFixed(1)}s`);
    this.setupSlider('vad-start-secs', 'vad.start_secs', (v) => `${v.toFixed(2)}s`);
    this.setupSlider('vad-min-volume', 'vad.min_volume', (v) => v.toFixed(2));
    this.setupSlider('aggregator-max-words', 'aggregator.max_words', (v) => v.toString());
    this.setupSlider('aggregator-min-words', 'aggregator.min_words', (v) => v.toString());
    this.setupSlider('aggregator-silence-timeout', 'aggregator.silence_timeout', (v) => `${v.toFixed(1)}s`);
    this.setupSlider('diar-max-speakers', 'diar.max_speakers', (v) => v.toString());
    this.setupSlider('diar-threshold', 'diar.threshold', (v) => v.toFixed(2));

    // Toggle setup
    const consoleToggle = document.getElementById('aggregator-console-display') as HTMLInputElement;
    if (consoleToggle) {
      consoleToggle.addEventListener('change', () => {
        this.pendingConfig['aggregator.console_display'] = consoleToggle.checked;
      });
    }

    // STT attention context select
    const attContextSelect = document.getElementById('stt-att-context') as HTMLSelectElement;
    if (attContextSelect) {
      attContextSelect.addEventListener('change', () => {
        const values = attContextSelect.value.split(',').map(v => parseInt(v.trim()));
        this.pendingConfig['stt.att_context'] = values;
      });
    }
  }

  /**
   * Setup a slider with value display
   */
  private setupSlider(elementId: string, configKey: string, formatter: (v: number) => string): void {
    const slider = document.getElementById(elementId) as HTMLInputElement;
    const valueDisplay = document.getElementById(`${elementId}-value`);

    if (slider && valueDisplay) {
      slider.addEventListener('input', () => {
        const value = parseFloat(slider.value);
        valueDisplay.textContent = formatter(value);
        this.pendingConfig[configKey] = value;
      });
    }
  }

  /**
   * Update UI with config from server
   */
  private updateUIFromConfig(config: ConfigState, specs: { [key: string]: ConfigSpec }): void {
    this.serverConfig = config;
    this.configSpecs = specs;
    this.pendingConfig = { ...config };

    // Update VAD sliders
    this.updateSliderFromConfig('vad-confidence', config['vad.confidence'], (v) => v.toFixed(2));
    this.updateSliderFromConfig('vad-stop-secs', config['vad.stop_secs'], (v) => `${v.toFixed(1)}s`);
    this.updateSliderFromConfig('vad-start-secs', config['vad.start_secs'], (v) => `${v.toFixed(2)}s`);
    this.updateSliderFromConfig('vad-min-volume', config['vad.min_volume'], (v) => v.toFixed(2));

    // Update Aggregator sliders
    this.updateSliderFromConfig('aggregator-max-words', config['aggregator.max_words'], (v) => v.toString());
    this.updateSliderFromConfig('aggregator-min-words', config['aggregator.min_words'], (v) => v.toString());
    this.updateSliderFromConfig('aggregator-silence-timeout', config['aggregator.silence_timeout'], (v) => `${v.toFixed(1)}s`);

    // Update console display toggle
    const consoleToggle = document.getElementById('aggregator-console-display') as HTMLInputElement;
    if (consoleToggle && config['aggregator.console_display'] !== undefined) {
      consoleToggle.checked = config['aggregator.console_display'];
    }

    // Update STT attention context
    const attContextSelect = document.getElementById('stt-att-context') as HTMLSelectElement;
    if (attContextSelect && config['stt.att_context']) {
      const value = Array.isArray(config['stt.att_context'])
        ? config['stt.att_context'].join(',')
        : config['stt.att_context'];
      attContextSelect.value = value;
    }

    // Update Diarization sliders
    this.updateSliderFromConfig('diar-max-speakers', config['diar.max_speakers'], (v) => v.toString());
    this.updateSliderFromConfig('diar-threshold', config['diar.threshold'], (v) => v.toFixed(2));
  }

  /**
   * Update a slider from config value
   */
  private updateSliderFromConfig(elementId: string, value: any, formatter: (v: number) => string): void {
    const slider = document.getElementById(elementId) as HTMLInputElement;
    const valueDisplay = document.getElementById(`${elementId}-value`);

    if (slider && value !== undefined) {
      slider.value = value.toString();
      if (valueDisplay) {
        valueDisplay.textContent = formatter(parseFloat(value));
      }
    }
  }

  /**
   * Apply pending configuration changes
   */
  private async applyConfiguration(): Promise<void> {
    if (!this.rtviClient) {
      this.addStreamEntry('error', 'Not connected - cannot apply configuration');
      return;
    }

    // Find changed values
    const changes: ConfigState = {};
    for (const [key, value] of Object.entries(this.pendingConfig)) {
      if (JSON.stringify(value) !== JSON.stringify(this.serverConfig[key])) {
        changes[key] = value;
      }
    }

    if (Object.keys(changes).length === 0) {
      this.addStreamEntry('system', 'No configuration changes to apply');
      return;
    }

    try {
      this.addStreamEntry('system', `Applying ${Object.keys(changes).length} config change(s)...`);

      await this.rtviClient.action({
        service: 'config',
        action: 'update',
        arguments: [{ name: 'params', value: changes }]
      });

    } catch (error) {
      this.addStreamEntry('error', `Config update failed: ${error}`);
    }
  }

  /**
   * Reset configuration to defaults
   */
  private async resetConfiguration(): Promise<void> {
    if (!this.rtviClient) {
      this.addStreamEntry('error', 'Not connected - cannot reset configuration');
      return;
    }

    try {
      this.addStreamEntry('system', 'Resetting configuration to defaults...');

      await this.rtviClient.action({
        service: 'config',
        action: 'reset',
        arguments: []
      });

    } catch (error) {
      this.addStreamEntry('error', `Config reset failed: ${error}`);
    }
  }

  /**
   * Toggle settings sidebar
   */
  private toggleSidebar(): void {
    this.settingsSidebar?.classList.toggle('collapsed');
  }

  /**
   * Add entry to stream log with color-coded styling
   * Extended with 'user' and 'assistant' types for text input mode
   */
  private addStreamEntry(type: 'partial' | 'final' | 'vad' | 'system' | 'error' | 'speaker' | 'user' | 'assistant', content: string): void {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });

    // Add to legacy stream log if available
    if (this.streamLog) {
      const welcome = this.streamLog.querySelector('.stream-welcome, .debug-welcome');
      if (welcome) welcome.remove();

      const entry = document.createElement('div');
      entry.className = `debug-item type-${type}`;
      entry.innerHTML = `
        <span class="debug-time">${time}</span>
        <span class="debug-type ${type}">${type.toUpperCase()}</span>
        <span class="debug-text">${this.escapeHtml(content)}</span>
      `;

      this.streamLog.appendChild(entry);
      this.streamLog.scrollTop = this.streamLog.scrollHeight;

      // Keep only last 100 entries
      while (this.streamLog.children.length > 100) {
        this.streamLog.removeChild(this.streamLog.firstChild!);
      }
    }

    // Add to new debug panel if available with color coding
    if (this.debugLog) {
      // Determine debug type based on content and type
      let debugType: 'info' | 'warn' | 'error' | 'system' | 'stt' | 'tts' | 'llm' = 'info';
      const contentLower = content.toLowerCase();

      if (type === 'error') {
        debugType = 'error';
      } else if (type === 'system') {
        debugType = 'system';
      } else if (type === 'user') {
        // User text input - show as STT (input from user)
        debugType = 'stt';
      } else if (type === 'assistant') {
        // Assistant response - show as LLM
        debugType = 'llm';
      } else if (contentLower.includes('stt') || contentLower.includes('asr') ||
                 contentLower.includes('transcript') || type === 'partial' || type === 'final') {
        debugType = 'stt';
      } else if (contentLower.includes('tts') || contentLower.includes('speak')) {
        debugType = 'tts';
      } else if (contentLower.includes('llm') || contentLower.includes('bot')) {
        debugType = 'llm';
      }

      this.addDebugEntry(debugType, `[${type.toUpperCase()}] ${content}`);
    }
  }

  /**
   * Clear stream log
   */
  private clearStreamLog(): void {
    if (!this.streamLog) return;
    this.streamLog.innerHTML = `
      <div class="stream-welcome">
        <span>Stream log cleared</span>
      </div>
    `;
  }

  /**
   * Clear transcripts (both history and flow)
   */
  private clearTranscripts(): void {
    this.transcriptCounter = 0;
    this.accumulatedText = '';
    this.flowLines = [];
    this.currentLineWordCount = 0;

    if (this.transcriptCount) {
      this.transcriptCount.textContent = '0';
    }

    // Clear legacy final transcripts
    if (this.finalTranscripts) {
      this.finalTranscripts.innerHTML = `
        <div class="transcript-empty">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
            <line x1="12" y1="19" x2="12" y2="23"/>
            <line x1="8" y1="23" x2="16" y2="23"/>
          </svg>
          <p>No transcriptions yet</p>
          <span>Connect and start speaking to see transcripts</span>
        </div>
      `;
    }

    // Clear new history panel
    if (this.historyContent) {
      // Remove all history items, keep the empty placeholder
      const items = this.historyContent.querySelectorAll('.history-item');
      items.forEach(item => item.remove());

      // Show empty placeholder
      if (this.historyEmpty) {
        this.historyEmpty.style.display = '';
      }
    }

    // Clear STT flow area
    if (this.sttFlow) {
      this.sttFlow.innerHTML = '';
    }

    this.updatePartialTranscript('');
    this.addStreamEntry('system', 'Transcripts cleared');
  }

  /**
   * Clear Voice Agent conversation view
   */
  private clearConversation(): void {
    this.conversationMessages = [];
    this.currentBotMessage = null;
    this.currentUserMessage = null;
    this.llmAccumulatedText = '';
    this.ttsChunks = [];
    this.currentTTSChunkIndex = 0;
    this.messageIdCounter = 0;
    this.botStatus = { status: 'idle', ttsPlaying: false, currentText: '' };

    // Reset conversation view to empty state
    if (this.conversationView) {
      this.conversationView.innerHTML = `
        <div class="conversation-empty">
          <div class="empty-icon"><span class="icon-chat"></span></div>
          <p>대화를 시작하세요</p>
          <span>마이크에 말하면 AI가 응답합니다</span>
        </div>
      `;
    }

    // Reset bot status indicator
    if (this.botStatusText) {
      this.botStatusText.textContent = '대기 중';
    }
    if (this.botStatusIndicator) {
      this.botStatusIndicator.className = 'bot-status-indicator idle';
    }

    this.addStreamEntry('system', 'Conversation cleared');
  }

  /**
   * Export transcript as text file
   */
  private exportTranscript(): void {
    if (!this.finalTranscripts) return;

    const items = this.finalTranscripts.querySelectorAll('.transcript-item');
    if (items.length === 0) {
      this.addStreamEntry('error', 'No transcripts to export');
      return;
    }

    let content = `NeMo Voice Agent - Meeting Transcript\n`;
    content += `Exported: ${new Date().toLocaleString()}\n`;
    content += `${'='.repeat(50)}\n\n`;

    items.forEach((item, index) => {
      const time = item.querySelector('.transcript-time')?.textContent || '';
      const speaker = item.querySelector('.speaker-badge')?.textContent || '';
      const text = item.querySelector('.transcript-text')?.textContent || '';

      if (speaker) {
        content += `[${time}] ${speaker}: ${text}\n`;
      } else {
        content += `[${time}] ${text}\n`;
      }
    });

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcript_${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);

    this.addStreamEntry('system', `Exported ${items.length} transcripts`);
  }

  /**
   * Start the audio visualizer (input)
   */
  private startVisualization(): void {
    console.log('[Audio] Starting input visualization...', {
      hasLegacyVisualizer: !!this.audioVisualizer,
      hasInputVisualizer: !!this.inputAudioVisualizer,
      hasAnalyser: !!this.analyser
    });
    // Start legacy visualizer
    if (this.audioVisualizer && this.analyser) {
      this.audioVisualizer.setAnalyser(this.analyser);
      this.audioVisualizer.setMode(this.vizMode);
      this.audioVisualizer.start();
      console.log('[Audio] Legacy visualizer started');
    }
    // Start new input visualizer
    if (this.inputAudioVisualizer && this.analyser) {
      this.inputAudioVisualizer.setAnalyser(this.analyser);
      this.inputAudioVisualizer.setMode(this.vizMode);
      this.inputAudioVisualizer.start();
      console.log('[Audio] Input visualizer started');
    }
  }

  /**
   * Stop the audio visualizer (input)
   */
  private stopVisualization(): void {
    if (this.audioVisualizer) {
      this.audioVisualizer.stop();
    }
    if (this.inputAudioVisualizer) {
      this.inputAudioVisualizer.stop();
    }
  }

  /**
   * Start the TTS output audio visualizer
   */
  private startOutputVisualization(): void {
    console.log('[Audio] Starting output visualization...', {
      hasOutputVisualizer: !!this.outputAudioVisualizer,
      hasOutputAnalyser: !!this.outputAnalyser
    });
    if (this.outputAudioVisualizer && this.outputAnalyser) {
      this.outputAudioVisualizer.setAnalyser(this.outputAnalyser);
      this.outputAudioVisualizer.setMode(this.vizMode);
      this.outputAudioVisualizer.setSpeaking(true);
      this.outputAudioVisualizer.start();
      console.log('[Audio] Output visualization started');
    } else {
      console.warn('[Audio] Cannot start output visualization - missing components');
    }
  }

  /**
   * Stop the TTS output audio visualizer
   */
  private stopOutputVisualization(): void {
    if (this.outputAudioVisualizer) {
      this.outputAudioVisualizer.setSpeaking(false);
      this.outputAudioVisualizer.stop();
    }
  }

  /**
   * Setup output audio analysis for TTS waveform visualization
   */
  private async setupOutputAudioAnalysis(): Promise<void> {
    console.log('[Audio] Setting up output audio analysis...');
    try {
      if (!this.outputAudioContext) {
        this.outputAudioContext = new AudioContext();
        console.log('[Audio] Created output AudioContext, state:', this.outputAudioContext.state);
      }

      if (this.outputAudioContext.state === 'suspended') {
        console.log('[Audio] Resuming output AudioContext...');
        await this.outputAudioContext.resume();
        console.log('[Audio] Output AudioContext resumed, state:', this.outputAudioContext.state);
      }

      this.outputAnalyser = this.outputAudioContext.createAnalyser();
      this.outputAnalyser.fftSize = 256;
      this.outputAnalyser.smoothingTimeConstant = 0.8;

      // Connect bot audio to analyser
      if (this.botAudio && this.botAudio.srcObject) {
        const source = this.outputAudioContext.createMediaStreamSource(this.botAudio.srcObject as MediaStream);
        source.connect(this.outputAnalyser);
        console.log('[Audio] Output analyser connected to bot audio stream');
      } else {
        console.warn('[Audio] Bot audio source not available for output analyser');
      }
    } catch (error) {
      console.error('[Audio] Output audio analysis setup failed:', error);
    }
  }

  /**
   * Update output volume display
   */
  private updateOutputVolumeDisplay(volume?: number): void {
    const bar = this.outputVolumeBar;
    const text = this.outputVolumeText;
    if (!bar) return;

    if (volume === undefined && this.outputAnalyser) {
      const dataArray = new Uint8Array(this.outputAnalyser.frequencyBinCount);
      this.outputAnalyser.getByteFrequencyData(dataArray);
      const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
      volume = (average / 255) * 100;
    }

    const displayVolume = Math.min(100, Math.max(0, volume || 0));
    bar.style.width = `${displayVolume}%`;
    if (text) {
      text.textContent = `${Math.round(displayVolume)}%`;
    }
  }

  /**
   * Handle bot audio level messages for visualization
   */
  private handleBotAudioLevel(data: { level: number; peak: number; is_speaking: boolean }): void {
    this.targetAudioLevel = data.level;
    this.currentPeakLevel = data.peak;
    this.isBotSpeaking = data.is_speaking;

    // Also update the output volume display
    this.updateOutputVolumeDisplay(data.level * 100);
  }

  /**
   * Start the circular visualization animation loop
   */
  private startCircularVisualization(): void {
    const animate = () => {
      this.drawCircularVisualizer();
      this.circularAnimationId = requestAnimationFrame(animate);
    };
    animate();
  }

  /**
   * Stop the circular visualization animation
   */
  private stopCircularVisualization(): void {
    if (this.circularAnimationId !== null) {
      cancelAnimationFrame(this.circularAnimationId);
      this.circularAnimationId = null;
    }
  }

  /**
   * Draw the ChatGPT-style circular visualizer
   * Creates a beautiful pulsating circle with dynamic waves based on audio level
   */
  private drawCircularVisualizer(): void {
    if (!this.circularVisualizer || !this.circularVisualizerCtx) return;

    const ctx = this.circularVisualizerCtx;
    const canvas = this.circularVisualizer;
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;

    // Smooth transition to target level
    const smoothingFactor = 0.15;
    this.currentAudioLevel += (this.targetAudioLevel - this.currentAudioLevel) * smoothingFactor;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Base radius
    const baseRadius = Math.min(width, height) * 0.25;
    const maxExpand = baseRadius * 0.5;

    // Calculate dynamic radius based on audio level
    const audioInfluence = this.currentAudioLevel * maxExpand;
    const peakInfluence = this.currentPeakLevel * maxExpand * 0.3;

    // Time-based animation for idle state
    const time = Date.now() * 0.001;
    const idlePulse = Math.sin(time * 2) * 3 + Math.sin(time * 3) * 2;

    // Combine all influences
    const dynamicRadius = this.isBotSpeaking
      ? baseRadius + audioInfluence + peakInfluence
      : baseRadius + idlePulse;

    // Create gradient for the main circle
    const gradient = ctx.createRadialGradient(
      centerX, centerY, 0,
      centerX, centerY, dynamicRadius * 1.5
    );

    if (this.isBotSpeaking) {
      // Speaking: vibrant blue-purple gradient
      gradient.addColorStop(0, 'rgba(99, 102, 241, 0.9)');
      gradient.addColorStop(0.5, 'rgba(139, 92, 246, 0.7)');
      gradient.addColorStop(0.8, 'rgba(168, 85, 247, 0.4)');
      gradient.addColorStop(1, 'rgba(192, 132, 252, 0)');
    } else {
      // Idle: subtle gray gradient
      gradient.addColorStop(0, 'rgba(156, 163, 175, 0.6)');
      gradient.addColorStop(0.5, 'rgba(156, 163, 175, 0.3)');
      gradient.addColorStop(1, 'rgba(156, 163, 175, 0)');
    }

    // Draw outer glow layers when speaking
    if (this.isBotSpeaking && this.currentAudioLevel > 0.1) {
      const numLayers = 3;
      for (let i = numLayers; i > 0; i--) {
        const layerRadius = dynamicRadius + (i * 15);
        const alpha = (0.15 / i) * this.currentAudioLevel;

        ctx.beginPath();
        ctx.arc(centerX, centerY, layerRadius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(139, 92, 246, ${alpha})`;
        ctx.fill();
      }
    }

    // Draw the main circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, dynamicRadius, 0, Math.PI * 2);
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw wave effect around the circle when speaking
    if (this.isBotSpeaking && this.currentAudioLevel > 0.05) {
      const numWaves = 24;
      const waveAmplitude = this.currentAudioLevel * 20;

      ctx.beginPath();
      for (let i = 0; i <= numWaves; i++) {
        const angle = (i / numWaves) * Math.PI * 2;
        const waveOffset = Math.sin(angle * 4 + time * 6) * waveAmplitude * (0.5 + this.currentAudioLevel * 0.5);
        const r = dynamicRadius + waveOffset;
        const x = centerX + Math.cos(angle) * r;
        const y = centerY + Math.sin(angle) * r;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.closePath();
      ctx.strokeStyle = `rgba(167, 139, 250, ${0.3 + this.currentAudioLevel * 0.5})`;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Draw inner highlight
    const innerGradient = ctx.createRadialGradient(
      centerX - dynamicRadius * 0.3,
      centerY - dynamicRadius * 0.3,
      0,
      centerX, centerY, dynamicRadius
    );
    innerGradient.addColorStop(0, 'rgba(255, 255, 255, 0.3)');
    innerGradient.addColorStop(0.5, 'rgba(255, 255, 255, 0.1)');
    innerGradient.addColorStop(1, 'rgba(255, 255, 255, 0)');

    ctx.beginPath();
    ctx.arc(centerX, centerY, dynamicRadius * 0.9, 0, Math.PI * 2);
    ctx.fillStyle = innerGradient;
    ctx.fill();

    // Draw status indicator text
    ctx.font = '12px "Pretendard Variable", system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    if (this.isBotSpeaking) {
      ctx.fillStyle = 'rgba(167, 139, 250, 0.9)';
      ctx.fillText('말하는 중...', centerX, centerY + dynamicRadius + 25);
    } else {
      ctx.fillStyle = 'rgba(156, 163, 175, 0.7)';
      ctx.fillText('대기 중', centerX, centerY + dynamicRadius + 25);
    }
  }

  /**
   * Escape HTML to prevent XSS
   */
  private escapeHtml(text: string): string {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * Clean STT text by removing/replacing special tokens
   * - SentencePiece token boundary: ▁ -> space
   * - Remove replacement characters: �
   */
  private cleanSTTText(text: string): string {
    if (!text) return '';
    return text
      .replace(/▁/g, ' ')           // SentencePiece token boundary -> space
      .replace(/\uFFFD/g, '')        // Unicode replacement character
      .replace(/\s+/g, ' ')          // Collapse multiple spaces
      .trim();
  }

  /**
   * Update partial transcript with continuous flowing paradigm
   * Text accumulates continuously like flowing water - never clears until explicit action
   */
  private updatePartialTranscript(text: string, speakerId?: number | null): void {
    const flowElement = this.sttFlow || this.partialTranscript;
    if (!flowElement) return;

    // Update accumulated text for flowing paradigm
    if (text) {
      this.accumulatedText = text;
      const newSpeakerId = speakerId !== undefined ? speakerId : null;
      this.currentSpeakerId = newSpeakerId;

      // Build the display content - accumulated lines + current partial
      const speakerClass = newSpeakerId !== null ? `speaker-${(newSpeakerId % 4) + 1}` : '';

      // Render accumulated flow lines (finalized text) + current partial
      let html = '';

      // Render finalized flow lines (solid styling)
      this.flowLines.forEach((line, index) => {
        html += `<span class="flow-line final-text">${this.escapeHtml(line)}</span>`;
      });

      // Add current partial with faded styling
      if (text.trim()) {
        // Create speaker tag if diarization is enabled
        const speakerTag = newSpeakerId !== null && this.diarizationEnabled
          ? `<span class="speaker-tag ${speakerClass}">${this.speakerNames[newSpeakerId % this.speakerNames.length]}</span> `
          : '';

        html += `<span class="partial-text speaker-text ${speakerClass}">${speakerTag}${this.escapeHtml(text)}</span>`;
      }

      flowElement.innerHTML = html;

      // Auto-scroll to bottom
      flowElement.scrollTop = flowElement.scrollHeight;

      // Show cursor while typing
      if (this.sttCursor) {
        this.sttCursor.style.display = 'inline-block';
      }

      // Activate live badge
      if (this.liveBadge) {
        this.liveBadge.classList.add('active');
      }
    } else {
      // Just hide cursor but keep accumulated text visible
      this.accumulatedText = '';

      // Hide cursor
      if (this.sttCursor) {
        this.sttCursor.style.display = 'none';
      }

      // Deactivate live badge when not actively typing
      if (this.liveBadge) {
        this.liveBadge.classList.remove('active');
      }
    }
  }

  /**
   * Commit current accumulated segment to transcript history
   */
  private commitCurrentSegment(reason: string = ''): void {
    if (!this.accumulatedText.trim()) return;

    const speakerId = this.currentSpeakerId;
    const color = speakerId !== null
      ? this.speakerColors[speakerId % this.speakerColors.length]
      : '';
    const name = speakerId !== null
      ? this.speakerNames[speakerId % this.speakerNames.length]
      : '';

    // Add to transcript history
    if (this.diarizationEnabled && speakerId !== null) {
      this.addSpeakerTranscript(this.accumulatedText, speakerId, name, color, true, reason);
    } else {
      this.addFinalTranscript(this.accumulatedText, reason);
    }

    // Clear accumulated text
    this.accumulatedText = '';
  }

  /**
   * Add a line to flowLines with windowed buffer to prevent OOM
   * Removes oldest lines when MAX_FLOW_LINES is exceeded
   *
   * @param line - The line to add to flowLines
   */
  private addToFlowLines(line: string): void {
    this.flowLines.push(line);

    // Windowed buffer: Remove oldest lines when limit exceeded
    if (this.flowLines.length > this.MAX_FLOW_LINES) {
      const overflow = this.flowLines.length - this.MAX_FLOW_LINES;
      this.flowLines.splice(0, overflow);
      console.log(`[FlowLines] Trimmed ${overflow} old lines, current count: ${this.flowLines.length}`);
    }
  }

  /**
   * Process final transcript - continuous flowing paradigm
   * Finalized text accumulates in the flow area with word-count line breaks
   * History records independently - flow never clears automatically
   */
  private processFinalTranscript(text: string, reason: string = '', speakerId?: number | null): void {
    if (!text.trim()) return;

    const finalSpeakerId = speakerId !== undefined ? speakerId : this.currentSpeakerId;
    const speakerClass = finalSpeakerId !== null ? `speaker-${(finalSpeakerId % 4) + 1}` : '';

    // Build the finalized text with speaker prefix if needed
    let finalText = text.trim();
    if (this.diarizationEnabled && finalSpeakerId !== null) {
      const speakerName = this.speakerNames[finalSpeakerId % this.speakerNames.length];
      finalText = `[${speakerName}] ${finalText}`;
    }

    // Add to flow lines with word-count based line breaks
    const words = finalText.split(/\s+/);
    let currentLine = '';

    // Check if we should continue the last line or start new
    if (this.flowLines.length > 0) {
      const lastLine = this.flowLines[this.flowLines.length - 1];
      const lastLineWordCount = lastLine.split(/\s+/).length;

      // If last line has room, append to it
      if (lastLineWordCount < this.maxWordsPerLine) {
        currentLine = lastLine;
        this.currentLineWordCount = lastLineWordCount;
        this.flowLines.pop(); // Remove last line, we'll add updated version
      } else {
        this.currentLineWordCount = 0;
      }
    }

    // Add words with line breaks based on maxWordsPerLine
    words.forEach(word => {
      if (this.currentLineWordCount >= this.maxWordsPerLine) {
        // Push current line and start new one (with windowed buffer)
        if (currentLine.trim()) {
          this.addToFlowLines(currentLine.trim());
        }
        currentLine = word;
        this.currentLineWordCount = 1;
      } else {
        currentLine = currentLine ? `${currentLine} ${word}` : word;
        this.currentLineWordCount++;
      }
    });

    // Add remaining content (with windowed buffer)
    if (currentLine.trim()) {
      this.addToFlowLines(currentLine.trim());
    }

    // Render the updated flow area
    const flowElement = this.sttFlow || this.partialTranscript;
    if (flowElement) {
      let html = '';
      this.flowLines.forEach(line => {
        html += `<span class="flow-line final-text ${speakerClass}">${this.escapeHtml(line)}</span>`;
      });
      flowElement.innerHTML = html;
      flowElement.scrollTop = flowElement.scrollHeight;
    }

    // Hide cursor after finalization
    if (this.sttCursor) {
      this.sttCursor.style.display = 'none';
    }

    // Clear the current partial (but keep flow lines)
    this.accumulatedText = '';

    // Add to history panel independently
    const historyText = text.trim();
    if (this.diarizationEnabled && finalSpeakerId !== null) {
      const name = this.speakerNames[finalSpeakerId % this.speakerNames.length];
      this.addHistoryItem(historyText, finalSpeakerId, name, reason);
    } else {
      this.addHistoryItem(historyText, null, '', reason);
    }
  }

  /**
   * Add item to history panel (new layout)
   */
  private addHistoryItem(text: string, speakerId: number | null, speakerName: string, reason: string = ''): void {
    // Use new historyContent if available, fallback to legacy finalTranscripts
    const historyElement = this.historyContent || this.finalTranscripts;
    if (!historyElement || !text) return;

    // Remove empty placeholder
    if (this.historyEmpty) {
      this.historyEmpty.style.display = 'none';
    }
    const emptyPlaceholder = historyElement.querySelector('.history-empty, .transcript-empty');
    if (emptyPlaceholder) {
      emptyPlaceholder.remove();
    }

    this.transcriptCounter++;
    const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const speakerClass = speakerId !== null ? `speaker-${(speakerId % 4) + 1}` : '';

    const item = document.createElement('div');
    item.className = `history-item ${speakerClass}`;

    if (speakerId !== null && this.diarizationEnabled) {
      item.innerHTML = `
        <div class="history-meta">
          <span class="history-time">${time}</span>
          <span class="history-speaker ${speakerClass}">${speakerName}</span>
        </div>
        <div class="history-text">${this.escapeHtml(text)}</div>
      `;
    } else {
      item.innerHTML = `
        <div class="history-meta">
          <span class="history-time">${time}</span>
        </div>
        <div class="history-text">${this.escapeHtml(text)}</div>
      `;
    }

    historyElement.appendChild(item);
    historyElement.scrollTop = historyElement.scrollHeight;

    // Update counter
    if (this.transcriptCount) {
      this.transcriptCount.textContent = this.transcriptCounter.toString();
    }
  }

  /**
   * Add final transcript to list with drop-down animation
   */
  private addFinalTranscript(text: string, reason?: string): void {
    if (!this.finalTranscripts || !text) return;

    const empty = this.finalTranscripts.querySelector('.transcript-empty');
    if (empty) empty.remove();

    this.transcriptCounter++;
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    const isTimeout = reason === 'timeout';

    const item = document.createElement('div');
    item.className = `transcript-item new-entry${isTimeout ? ' timeout' : ''}`;
    item.innerHTML = `
      <div class="transcript-meta">
        <span class="transcript-number">#${this.transcriptCounter}</span>
        <span class="transcript-time">${time}</span>
      </div>
      <div class="transcript-content">
        <span class="transcript-text">${this.escapeHtml(text)}</span>
        ${reason ? `<span class="transcript-reason">${reason}</span>` : ''}
      </div>
    `;

    this.finalTranscripts.appendChild(item);
    this.finalTranscripts.scrollTop = this.finalTranscripts.scrollHeight;

    // Remove animation class after animation completes
    setTimeout(() => {
      item.classList.remove('new-entry');
    }, 800);

    if (this.transcriptCount) {
      this.transcriptCount.textContent = this.transcriptCounter.toString();
    }

    this.updatePartialTranscript('');
  }

  /**
   * Add speaker-attributed transcript to list with drop-down animation
   */
  private addSpeakerTranscript(
    text: string,
    speakerId: number | null,
    speakerName: string,
    speakerColor: string,
    isFinal: boolean,
    reason?: string
  ): void {
    if (!this.finalTranscripts || !text) return;

    const empty = this.finalTranscripts.querySelector('.transcript-empty');
    if (empty) empty.remove();

    this.transcriptCounter++;
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });

    const item = document.createElement('div');
    item.className = `transcript-item speaker-transcript new-entry${isFinal ? '' : ' partial'}`;
    item.style.setProperty('--speaker-color', speakerColor);
    item.innerHTML = `
      <div class="transcript-meta">
        <span class="transcript-number">#${this.transcriptCounter}</span>
        <span class="transcript-time">${time}</span>
      </div>
      <div class="transcript-content">
        <span class="speaker-badge" style="background-color: ${speakerColor}20; color: ${speakerColor}; border-color: ${speakerColor}">${speakerName}</span>
        <span class="transcript-text" style="color: ${speakerColor}">${this.escapeHtml(text)}</span>
        ${reason ? `<span class="transcript-reason">${reason}</span>` : ''}
      </div>
    `;

    this.finalTranscripts.appendChild(item);
    this.finalTranscripts.scrollTop = this.finalTranscripts.scrollHeight;

    // Remove animation class after animation completes
    setTimeout(() => {
      item.classList.remove('new-entry');
    }, 800);

    if (this.transcriptCount) {
      this.transcriptCount.textContent = this.transcriptCounter.toString();
    }

    this.updatePartialTranscript('');
  }

  /**
   * Update speaker status indicators
   */
  private updateSpeakerStatus(
    speakers: Array<{id: number; active: boolean; name: string; color: string; activity_level: number}>,
    activeSpeakerId: number | null,
    totalSpeakers: number
  ): void {
    this.activeSpeakerId = activeSpeakerId;
    this.diarizationEnabled = true;

    if (this.speakerPanel) {
      this.speakerPanel.classList.add('active');
    }

    if (this.diarBadge) {
      this.diarBadge.style.display = 'inline-flex';
    }

    speakers.forEach((speaker) => {
      const indicator = this.speakerIndicators[speaker.id];
      if (indicator) {
        if (speaker.active) {
          indicator.classList.add('active');
          indicator.style.setProperty('--speaker-color', speaker.color);
        } else {
          indicator.classList.remove('active');
        }

        const nameEl = indicator.querySelector('.speaker-name');
        if (nameEl) {
          nameEl.textContent = speaker.name;
        }
      }
    });

    for (let i = totalSpeakers; i < 4; i++) {
      const indicator = this.speakerIndicators[i];
      if (indicator) {
        indicator.classList.remove('active');
      }
    }
  }

  /**
   * Update VAD status display with visual feedback
   */
  private updateVADStatus(speaking: boolean): void {
    if (!this.vadIndicator || !this.vadLabel) return;

    const vadMonitor = this.vadIndicator.closest('.vad-monitor');

    // Update visualizer speaking state
    if (this.audioVisualizer) {
      this.audioVisualizer.setSpeaking(speaking);
    }
    if (this.inputAudioVisualizer) {
      this.inputAudioVisualizer.setSpeaking(speaking);
    }

    if (speaking) {
      this.vadIndicator.classList.add('speaking');
      this.vadLabel.textContent = 'Speaking';
      this.vadLabel.style.color = 'var(--accent-green)';
      this.vadStartTime = Date.now();
      this.startVADTimer();

      // Add visual feedback to the monitor card
      if (vadMonitor) {
        vadMonitor.classList.add('active');
      }

      // Add body-level class for global effects
      document.body.classList.add('vad-active');

      // Update input zone status and pipeline
      this.updateInputStatus('listening');
      this.updateSTTLiveIndicator(true);
      this.updatePipelineStatus('listening');
      this.updatePipelineStage('stt', true);
    } else {
      this.vadIndicator.classList.remove('speaking');
      this.vadLabel.textContent = 'Silence';
      this.vadLabel.style.color = '';
      this.stopVADTimer();

      // Remove visual feedback
      if (vadMonitor) {
        vadMonitor.classList.remove('active');
      }

      document.body.classList.remove('vad-active');

      // Update input zone status and pipeline
      this.updateInputStatus('ready');
      this.updateSTTLiveIndicator(false);
      this.updatePipelineStage('stt', false);
    }
  }

  /**
   * Start VAD timer
   */
  private startVADTimer(): void {
    this.stopVADTimer();
    this.vadTimerInterval = window.setInterval(() => {
      if (this.vadTimer && this.vadStartTime) {
        const elapsed = (Date.now() - this.vadStartTime) / 1000;
        this.vadTimer.textContent = `${elapsed.toFixed(1)}s`;
      }
    }, 100);
  }

  /**
   * Stop VAD timer
   */
  private stopVADTimer(): void {
    if (this.vadTimerInterval) {
      clearInterval(this.vadTimerInterval);
      this.vadTimerInterval = null;
    }
    if (this.vadTimer) {
      this.vadTimer.textContent = '--';
    }
    this.vadStartTime = null;
  }

  /**
   * Update connection status display
   */
  private updateStatus(status: string): void {
    if (this.statusText) {
      this.statusText.textContent = status;
    }

    if (this.connectionIndicator) {
      this.connectionIndicator.classList.remove('connected', 'connecting');
      if (status === 'Connected') {
        this.connectionIndicator.classList.add('connected');
      } else if (status === 'Connecting...') {
        this.connectionIndicator.classList.add('connecting');
      }
    }

    this.addStreamEntry('system', `Status: ${status}`);
  }

  /**
   * Update footer stats
   */
  private updateFooterStats(text: string): void {
    if (this.footerStats) {
      this.footerStats.textContent = text;
    }
  }

  /**
   * Set up media tracks
   */
  setupMediaTracks() {
    if (!this.rtviClient) return;
    const tracks = this.rtviClient.tracks();
    if (tracks.bot?.audio) {
      this.setupAudioTrack(tracks.bot.audio);
    }
    // Set up volume monitoring with local audio track if available
    if (tracks.local?.audio && !this.analyser) {
      this.startVolumeMonitoring();
    }
  }

  /**
   * Set up track listeners
   */
  setupTrackListeners() {
    if (!this.rtviClient) return;

    try {
      this.rtviClient.on(RTVIEvent.TrackStarted, (track, participant) => {
        if (track.kind === 'audio') {
          if (participant?.local) {
            // Local audio track started - set up volume monitoring
            if (!this.analyser) {
              this.startVolumeMonitoring();
            }
          } else {
            // Bot audio track - set up for playback
            this.setupAudioTrack(track);
          }
        }
      });

      this.rtviClient.on(RTVIEvent.TrackStopped, (track, participant) => {
        this.addStreamEntry('system', `Track stopped: ${track.kind}`);
      });
    } catch (error) {
      this.addStreamEntry('error', `Track listener error: ${error}`);
    }
  }

  /**
   * Set up custom message handlers for speaker diarization and config
   * Uses the global WebSocket interceptor for reliable message capture
   */
  setupCustomMessageHandlers() {
    // CRITICAL FIX: Prevent duplicate handler registration
    // This function is called from both onConnected and onBotReady callbacks,
    // and registerGlobalMessageHandler() is also called in constructor.
    // Without this check, handlers would be registered 3+ times, causing
    // messages to be processed multiple times and text to accumulate incorrectly.
    if (this.messageHandlersSetup) {
      console.log('[CustomHandler] Message handlers already set up, skipping');
      return;
    }
    this.messageHandlersSetup = true;
    console.log('[CustomHandler] Setting up message handlers (first time)');

    // Also try RTVI client's message error handler as fallback
    if (this.rtviClient) {
      try {
        this.rtviClient.on(RTVIEvent.MessageError, (error: unknown) => {
          console.debug('Message handling error (expected for custom types):', error);
        });

        // Try transport's message event as additional fallback
        const transport = (this.rtviClient as any)['_transport'];
        if (transport && typeof transport.on === 'function') {
          transport.on('message', (message: unknown) => {
            this.handleCustomMessage(message);
          });
        }
      } catch (error) {
        console.debug('RTVI message handler setup:', error);
      }
    }
  }

  // Track if WebSocket interceptor is already attached (legacy - now using global interceptor)
  private wsInterceptorAttached: boolean = false;

  /**
   * Legacy function - now using global WebSocket interceptor
   * Kept for compatibility but no longer needed for message interception
   */
  private interceptWebSocketMessages(retryCount: number = 0): void {
    // Using global WebSocket interceptor instead - this function is now a no-op
    // The global interceptor (registered in constructor) handles all message interception
    console.log('[WS-Intercept] Using global WebSocket interceptor (legacy function skipped)');
    this.wsInterceptorAttached = true;

    // Still request server config after a delay if not received
    setTimeout(() => {
      if (!this.serverModelConfig) {
        console.log('[WS-Intercept] No config yet, requesting server config...');
        this.requestServerConfig();
      }
    }, 100);
  }

  /**
   * Wrap WebSocket's onmessage to intercept custom messages
   * Uses addEventListener for more reliable message interception
   */
  private wrapWebSocketOnMessage(ws: WebSocket): void {
    // Use addEventListener instead of wrapping onmessage - more reliable
    ws.addEventListener('message', async (event: MessageEvent) => {
      try {
        let data: any = null;

        // Handle different message data types
        if (typeof event.data === 'string') {
          // Direct JSON text message
          data = JSON.parse(event.data);
        } else if (event.data instanceof ArrayBuffer) {
          // Binary ArrayBuffer - try to decode as UTF-8 and parse as JSON
          const text = new TextDecoder('utf-8').decode(event.data);
          data = this.extractJsonFromBinary(text);
        } else if (event.data instanceof Blob) {
          // Blob - convert to text and parse
          const text = await event.data.text();
          data = this.extractJsonFromBinary(text);
        }

        // Check for custom message types
        if (data && data.type) {
          const customTypes = [
              'speaker-status', 'speaker-transcription',
              'config-available', 'config-update-result', 'config-reset-result',
              'bot-llm-text', 'bot-llm-stream', 'bot-tts-text', 'bot-status',
              'bot-audio-level',  // TTS audio level for visualization
              'server-config'  // Server model configuration
            ];
          if (customTypes.includes(data.type)) {
            console.log(`[WS-Intercept] Custom message: ${data.type}`, data);
            this.handleCustomMessage(data);
          }
        }
      } catch (e) {
        // Ignore parse errors for binary or non-JSON messages
      }
    });
    console.log('[WS-Intercept] Event listener attached to WebSocket');
  }

  /**
   * Extract JSON from binary/text data that may have Protobuf framing
   */
  private extractJsonFromBinary(text: string): any | null {
    // First, try direct JSON parse
    try {
      return JSON.parse(text);
    } catch (e) {
      // Not valid JSON directly
    }

    // Look for JSON object patterns embedded in binary data
    // RTVI messages have format: {"label": "rtvi-ai", "type": "...", ...}
    // or {"id": "...", "type": "server-config", ...}
    let i = 0;
    while (i < text.length) {
      if (text[i] === '{') {
        // Found potential JSON start - find matching brace
        let braceCount = 1;
        let j = i + 1;
        while (j < text.length && braceCount > 0) {
          if (text[j] === '{') braceCount++;
          else if (text[j] === '}') braceCount--;
          j++;
        }

        if (braceCount === 0) {
          try {
            const jsonStr = text.slice(i, j);
            const obj = JSON.parse(jsonStr);
            // Return first valid JSON object with 'type' field
            if (obj && obj.type) {
              return obj;
            }
          } catch (e) {
            // Not valid JSON, continue searching
          }
        }
      }
      i++;
    }

    return null;
  }

  /**
   * Handle custom RTVI messages
   */
  private handleCustomMessage(message: any): void {
    if (!message || !message.type) return;

    switch (message.type) {
      case 'speaker-status':
        if (message.data) {
          this.updateSpeakerStatus(
            message.data.speakers || [],
            message.data.active_speaker_id,
            message.data.total_speakers || 0
          );
        }
        break;

      case 'speaker-transcription':
        if (message.data) {
          const { text, speaker_id, speaker_name, speaker_color, is_final, reason } = message.data;
          if (is_final) {
            // Use flowing paradigm for speaker transcription
            this.processFinalTranscript(text, reason, speaker_id);
            this.addStreamEntry('final', `[${speaker_name}] ${text}`);
          } else {
            // Show speaker-attributed partial with flowing paradigm
            this.updatePartialTranscript(text, speaker_id);
            this.addStreamEntry('partial', `[${speaker_name}] ${text}`);
          }
        }
        break;

      case 'config-available':
        if (message.data) {
          const { config, specs, diarization_enabled } = message.data;
          this.updateUIFromConfig(config, specs);
          this.diarizationEnabled = diarization_enabled;

          if (this.diarSettings) {
            this.diarSettings.style.display = diarization_enabled ? 'block' : 'none';
          }
          if (this.diarBadge) {
            this.diarBadge.style.display = diarization_enabled ? 'inline-flex' : 'none';
          }

          this.addStreamEntry('system', 'Configuration received from server');
        }
        break;

      case 'config-update-result':
        if (message.data) {
          const { success, succeeded, total, results } = message.data;
          if (success) {
            this.addStreamEntry('system', `Configuration updated: ${succeeded}/${total} changes applied`);
            // Refresh config from server
            this.rtviClient?.action({
              service: 'config',
              action: 'get',
              arguments: []
            });
          } else {
            const errors = Object.entries(results || {})
              .filter(([_, r]: [string, any]) => !r.success)
              .map(([k, r]: [string, any]) => `${k}: ${r.error}`)
              .join(', ');
            this.addStreamEntry('error', `Config update failed: ${errors}`);
          }
        }
        break;

      case 'config-reset-result':
        if (message.data?.success) {
          this.addStreamEntry('system', 'Configuration reset to defaults');
          this.rtviClient?.action({
            service: 'config',
            action: 'get',
            arguments: []
          });
        }
        break;

      // Voice Agent: LLM streaming text (raw tokens)
      case 'bot-llm-text':
        if (message.data?.text) {
          this.handleLLMText(message.data.text);
        }
        break;

      // Voice Agent: Enhanced LLM streaming with accumulated text
      case 'bot-llm-stream':
        if (message.data) {
          this.handleLLMStream(message.data);
        }
        break;

      // Voice Agent: TTS text chunk (for sync display)
      case 'bot-tts-text':
        if (message.data) {
          this.handleTTSText(message.data);
        }
        break;

      // Voice Agent: Bot status updates
      case 'bot-status':
        if (message.data) {
          this.handleBotStatus(message.data);
        }
        break;

      // Voice Agent: Bot audio level for visualization
      case 'bot-audio-level':
        if (message.data) {
          this.handleBotAudioLevel(message.data);
        }
        break;

      // Server model configuration
      case 'server-config':
        if (message.data) {
          this.handleServerConfigMessage(message.data);
        }
        break;
    }
  }

  /**
   * Handle raw LLM text tokens (bot-llm-text)
   * This is a FALLBACK handler - bot-llm-stream is preferred as it has server-accumulated text
   */
  private handleLLMText(text: string): void {
    // Accumulate LLM text locally
    this.llmAccumulatedText += text;

    // Update current bot message
    if (!this.currentBotMessage) {
      this.createBotMessage();
    }

    if (this.currentBotMessage) {
      this.currentBotMessage.text = this.llmAccumulatedText;
      this.currentBotMessage.status = 'thinking';
      this.updateConversationView();
    }

    // Also update OUTPUT panel with accumulated text
    this.updateLLMStreamPanel(this.llmAccumulatedText, false);
  }

  /**
   * Handle enhanced LLM streaming (bot-llm-stream)
   * This is the PRIMARY handler for LLM text display - uses server-provided accumulated text
   */
  private handleLLMStream(data: {
    text: string;
    accumulated: string;
    is_sentence_end: boolean;
    timestamp: number;
  }): void {
    // Use server-provided accumulated text (most reliable source)
    this.llmAccumulatedText = data.accumulated;

    if (!this.currentBotMessage) {
      this.createBotMessage();
    }

    if (this.currentBotMessage) {
      this.currentBotMessage.text = data.accumulated;
      this.currentBotMessage.status = data.is_sentence_end ? 'speaking' : 'thinking';
      this.updateConversationView();
    }

    // Update LLM stream panel with FULL accumulated text (not chunks)
    // Using isChunk=false to REPLACE text instead of appending
    // This ensures the OUTPUT panel always shows the complete accumulated text
    this.updateLLMStreamPanel(data.accumulated, false);

    // Log streaming progress
    if (data.is_sentence_end) {
      this.addStreamEntry('system', `LLM: ${data.accumulated.slice(-80)}...`);
    }
  }

  /**
   * Handle TTS text chunk for synchronization (bot-tts-text)
   */
  private handleTTSText(data: { text: string; chunk_id: number; is_final: boolean }): void {
    // Store TTS chunk for sync display
    this.ttsChunks.push({
      id: data.chunk_id,
      text: data.text,
      isFinal: data.is_final
    });

    // Update bot message status
    if (this.currentBotMessage) {
      this.currentBotMessage.status = 'speaking';
      this.currentBotMessage.ttsChunkId = data.chunk_id;
      this.updateConversationView();
    }

    // Highlight the current TTS text
    this.highlightTTSText(data.text, data.chunk_id);

    this.addStreamEntry('system', `TTS #${data.chunk_id}: ${data.text.substring(0, 50)}...`);
  }

  /**
   * Handle bot status updates (bot-status)
   */
  private handleBotStatus(data: { status: string; tts_playing: boolean; current_text: string }): void {
    const previousStatus = this.botStatus.status;

    this.botStatus = {
      status: data.status as BotStatus['status'],
      ttsPlaying: data.tts_playing,
      currentText: data.current_text
    };

    this.updateBotStatusUI();

    // Update new IO zone output status
    this.updateOutputStatus(data.status as 'idle' | 'thinking' | 'speaking');

    // Update pipeline status
    if (data.status === 'thinking') {
      // CRITICAL FIX: Reset accumulated text when a new LLM response starts
      // This prevents text accumulation across conversation turns
      if (previousStatus !== 'thinking') {
        console.log('[BotStatus] New LLM response started, resetting accumulated text');
        this.llmAccumulatedText = '';
        this.ttsChunks = [];
        this.currentTTSChunkIndex = 0;
      }
      this.updatePipelineStatus('processing');
      this.updatePipelineStage('llm', true);
      this.updatePipelineStage('tts', false);
    } else if (data.status === 'speaking') {
      this.updatePipelineStatus('speaking');
      this.updatePipelineStage('llm', false);
      this.updatePipelineStage('tts', true);
      this.updateTTSSyncIndicator(true);
      // Start output visualization
      this.startOutputVisualization();
    } else {
      this.updatePipelineStatus('standby');
      this.updatePipelineStage('llm', false);
      this.updatePipelineStage('tts', false);
      this.updateTTSSyncIndicator(false);
      // Stop output visualization
      this.stopOutputVisualization();
      this.updateOutputVolumeDisplay(0);
    }

    // When bot becomes idle, finalize the current message
    if (data.status === 'idle' && this.currentBotMessage) {
      this.currentBotMessage.status = 'complete';
      this.finalizeBotMessage();
    }
  }

  /**
   * Create a new bot message
   */
  private createBotMessage(): void {
    this.messageIdCounter++;
    this.currentBotMessage = {
      id: `bot-${this.messageIdCounter}`,
      role: 'bot',
      text: '',
      timestamp: new Date(),
      status: 'thinking'
    };
    this.conversationMessages.push(this.currentBotMessage);
  }

  /**
   * Create a new user message
   */
  private createUserMessage(text: string, speakerId?: number | null): void {
    this.messageIdCounter++;
    this.currentUserMessage = {
      id: `user-${this.messageIdCounter}`,
      role: 'user',
      text: text,
      timestamp: new Date(),
      status: 'speaking',
      speakerId: speakerId
    };
    this.conversationMessages.push(this.currentUserMessage);
  }

  /**
   * Finalize current bot message
   */
  private finalizeBotMessage(): void {
    if (this.currentBotMessage) {
      this.currentBotMessage.status = 'complete';
      // Add to history panel
      if (this.currentBotMessage.text.trim()) {
        this.addHistoryEntry('bot', this.currentBotMessage.text);
      }
    }
    this.currentBotMessage = null;
    this.llmAccumulatedText = '';
    this.ttsChunks = [];
    this.currentTTSChunkIndex = 0;
    this.updateConversationView();
  }

  /**
   * Add a user message from text input (not STT) to conversation view.
   * This creates a user message bubble with 'text-input' source.
   * Used in text input modes (LLM-Only, TTS-Only, LLM+TTS).
   */
  private addTextInputUserMessage(text: string): void {
    if (!text.trim()) return;

    this.messageIdCounter++;
    const userMessage: ConversationMessage = {
      id: `user-text-${this.messageIdCounter}`,
      role: 'user',
      text: text,
      timestamp: new Date(),
      status: 'complete',  // Text input is immediately complete (unlike STT streaming)
      speakerId: null,
      source: 'text-input'  // Mark as text input source
    };

    this.conversationMessages.push(userMessage);

    // Also add to history panel
    this.addHistoryEntry('user', text);

    // Update the conversation view to show the new message
    this.updateConversationView();

    console.log('[TextInput] Added user message to conversation:', text.substring(0, 30) + (text.length > 30 ? '...' : ''));
  }

  /**
   * Update bot status UI
   */
  private updateBotStatusUI(): void {
    const statusLabels: { [key: string]: string } = {
      'idle': '대기 중',
      'thinking': '생각 중...',
      'speaking': '말하는 중...'
    };
    const statusLabel = statusLabels[this.botStatus.status] || this.botStatus.status;

    if (this.botStatusIndicator) {
      this.botStatusIndicator.className = `bot-status-indicator ${this.botStatus.status}`;
    }

    // Update bot status text (either direct reference or via selector)
    if (this.botStatusText) {
      this.botStatusText.textContent = statusLabel;
    } else if (this.botStatusIndicator) {
      const statusText = this.botStatusIndicator.querySelector('.status-text');
      if (statusText) {
        statusText.textContent = statusLabel;
      }
    }

    if (this.botThinkingIndicator) {
      this.botThinkingIndicator.style.display = this.botStatus.status === 'thinking' ? 'flex' : 'none';
    }
  }

  /**
   * Highlight the currently playing TTS text
   */
  private highlightTTSText(text: string, chunkId: number): void {
    const conversationView = this.conversationView;
    if (!conversationView) return;

    // Find the current bot message element
    const botMessage = conversationView.querySelector('.message.bot.current');
    if (!botMessage) return;

    const textElement = botMessage.querySelector('.message-text');
    if (!textElement) return;

    // Highlight the TTS text portion
    const fullText = this.llmAccumulatedText;
    const startIndex = fullText.indexOf(text);
    if (startIndex !== -1) {
      const before = fullText.substring(0, startIndex);
      const highlighted = text;
      const after = fullText.substring(startIndex + text.length);

      textElement.innerHTML = `
        <span class="text-complete">${this.escapeHtml(before)}</span>
        <span class="text-speaking">${this.escapeHtml(highlighted)}</span>
        <span class="text-pending">${this.escapeHtml(after)}</span>
      `;
    }
  }

  /**
   * Update the conversation view UI
   */
  private updateConversationView(): void {
    if (!this.conversationView) return;

    // OPTIMIZATION: Update only changed messages instead of re-rendering all
    // This prevents flickering caused by full innerHTML replacement

    const existingMessages = this.conversationView.querySelectorAll('.message');
    const existingIds = new Set<string>();

    // Update existing messages in place
    existingMessages.forEach((elem) => {
      const msgId = elem.getAttribute('data-id');
      if (msgId) existingIds.add(msgId);
    });

    // Check if we need a full re-render (new message added)
    const needsFullRender = this.conversationMessages.some(msg => !existingIds.has(msg.id));

    if (needsFullRender || existingMessages.length !== this.conversationMessages.length) {
      // Full re-render only when messages are added/removed
      let html = '';

      this.conversationMessages.forEach((msg, index) => {
        const isCurrent = msg.role === 'bot' && msg === this.currentBotMessage;
        const statusClass = msg.status;
        const roleClass = msg.role;
        const time = msg.timestamp.toLocaleTimeString('ko-KR', {
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit'
        });

        // Differentiate between STT (voice) and text input sources
        const isTextInput = msg.source === 'text-input';
        const iconClass = msg.role === 'user'
          ? 'user-icon'  // Same icon for both speech and text input
          : 'bot-icon';
        const name = msg.role === 'user'
          ? 'User'  // Unified label for both speech and text input
          : 'Assistant';
        const sourceClass = isTextInput ? 'text-input-source' : '';

        html += `
          <div class="message ${roleClass} ${statusClass} ${sourceClass} ${isCurrent ? 'current' : ''}" data-id="${msg.id}">
            <div class="message-header">
              <span class="message-icon ${iconClass}"></span>
              <span class="message-name">${name}</span>
              <span class="message-time">${time}</span>
              ${msg.role === 'bot' && msg.status !== 'complete' ? `
                <span class="message-status ${msg.status}">
                  ${msg.status === 'thinking' ? '생각 중...' : '응답 중...'}
                </span>
              ` : ''}
            </div>
            <div class="message-text">${this.escapeHtml(msg.text) || (msg.status === 'thinking' ? '<span class="thinking-dots"><span>.</span><span>.</span><span>.</span></span>' : '')}</div>
            ${msg.role === 'bot' && msg.status === 'speaking' && this.botStatus.ttsPlaying ? `
              <div class="tts-indicator">
                <div class="tts-bar"></div>
                <div class="tts-bar"></div>
                <div class="tts-bar"></div>
                <div class="tts-bar"></div>
              </div>
            ` : ''}
          </div>
        `;
      });

      this.conversationView.innerHTML = html;
    } else {
      // Incremental update: Only update text content of existing messages
      this.conversationMessages.forEach((msg) => {
        const elem = this.conversationView!.querySelector(`[data-id="${msg.id}"]`);
        if (elem) {
          // Update text content without replacing entire element
          const textElem = elem.querySelector('.message-text');
          if (textElem) {
            const newText = this.escapeHtml(msg.text) || (msg.status === 'thinking' ? '<span class="thinking-dots"><span>.</span><span>.</span><span>.</span></span>' : '');
            if (textElem.innerHTML !== newText) {
              textElem.innerHTML = newText;
            }
          }

          // Update status class (preserve text-input-source class for text input messages)
          const isCurrent = msg.role === 'bot' && msg === this.currentBotMessage;
          const isTextInput = msg.source === 'text-input';
          const sourceClass = isTextInput ? 'text-input-source' : '';
          elem.className = `message ${msg.role} ${msg.status} ${sourceClass} ${isCurrent ? 'current' : ''}`.trim();

          // Update status label for bot messages
          const statusElem = elem.querySelector('.message-status');
          if (msg.role === 'bot' && msg.status !== 'complete') {
            const statusText = msg.status === 'thinking' ? '생각 중...' : '응답 중...';
            if (statusElem) {
              statusElem.className = `message-status ${msg.status}`;
              statusElem.textContent = statusText;
            } else {
              // Add status element if missing
              const header = elem.querySelector('.message-header');
              if (header) {
                const newStatus = document.createElement('span');
                newStatus.className = `message-status ${msg.status}`;
                newStatus.textContent = statusText;
                header.appendChild(newStatus);
              }
            }
          } else if (statusElem) {
            statusElem.remove();
          }
        }
      });
    }

    // Auto-scroll to bottom
    this.conversationView.scrollTop = this.conversationView.scrollHeight;
  }

  /**
   * Set up audio track for playback
   */
  private async setupAudioTrack(track: MediaStreamTrack): Promise<void> {
    if (this.botAudio.srcObject && "getAudioTracks" in this.botAudio.srcObject) {
      const oldTrack = this.botAudio.srcObject.getAudioTracks()[0];
      if (oldTrack?.id === track.id) return;
    }
    this.botAudio.srcObject = new MediaStream([track]);

    // Setup output audio analysis for waveform visualization
    // This must be called after botAudio.srcObject is set
    // Await to ensure outputAnalyser is ready before startOutputVisualization is called
    await this.setupOutputAudioAnalysis();
    console.log('[Audio] Output audio analysis setup complete, outputAnalyser:', !!this.outputAnalyser);
  }

  /**
   * Connect to the server
   */
  public async connect(): Promise<void> {
    if (this.isConnecting) return;

    try {
      this.isConnecting = true;
      this.updateStatus('Connecting...');
      const startTime = Date.now();

      const transport = new WebSocketTransport();
      const RTVIConfig: RTVIClientOptions = {
        transport,
        params: {
          baseUrl: this.getServerConfig().baseUrl,
          endpoints: { connect: '/connect' },
        },
        enableMic: true,
        enableCam: false,
        callbacks: {
          onConnected: () => {
            this.updateStatus('Connected');
            this.updateFooterStats(`Connected in ${Date.now() - startTime}ms`);
            if (this.connectBtn) this.connectBtn.disabled = true;
            if (this.disconnectBtn) this.disconnectBtn.disabled = false;
            if (this.muteBtn) {
              this.muteBtn.disabled = false;
              this.muteBtn.querySelector('.btn-text')!.textContent = 'Mute';
            }
            if (!this.isMuted) {
              this.startVolumeMonitoring();
            }
            // Toggle input zone active state (glowing border when mic is active)
            if (this.inputZone && !this.isMuted) {
              this.inputZone.classList.add('active');
            }
            // Activate module badges
            this.updateBadgesState(true);
            // Setup stream panel resize
            this.setupStreamPanelResize();
            // Setup custom message handlers immediately after connection is established
            // Must be done synchronously to catch server-config message
            this.setupCustomMessageHandlers();
          },
          onDisconnected: () => {
            // Reset message handler flag to allow re-registration on reconnect
            this.messageHandlersSetup = false;

            if (!this.isConnecting) {
              this.updateStatus('Disconnected');
              this.updateFooterStats('Ready');
              if (this.connectBtn) this.connectBtn.disabled = false;
              if (this.disconnectBtn) this.disconnectBtn.disabled = true;
              if (this.muteBtn) {
                this.muteBtn.disabled = true;
                this.muteBtn.querySelector('.btn-text')!.textContent = 'Mute';
                this.muteBtn.classList.remove('muted');
              }
              // Remove active states from both zones
              this.inputZone?.classList.remove('active');
              this.outputZone?.classList.remove('active');
              // Reset speaking indicator
              this.speakingIndicator?.classList.remove('active');
              if (this.speakingStatusLabel) {
                this.speakingStatusLabel.textContent = '대기 중';
              }
              this.stopVolumeMonitoring();
              this.stopVADTimer();
              this.updateVADStatus(false);
            }
          },
          onBotReady: (data) => {
            this.addStreamEntry('system', 'Bot ready');
            this.setupMediaTracks();
            console.log('[onBotReady] Bot ready, setting up handlers');

            // Retry message handler setup and request config if not received
            // Server sends config right after bot-ready
            setTimeout(() => {
              this.setupCustomMessageHandlers();
              // Check if we received server config, if not, request it
              if (!this.serverModelConfig) {
                console.log('[onBotReady] No config yet, requesting...');
                this.requestServerConfig();
              }
            }, 100);

            // Multiple retry attempts for config
            setTimeout(() => {
              if (!this.serverModelConfig) {
                console.log('[onBotReady] 2nd attempt - requesting config...');
                this.requestServerConfig();
              }
            }, 500);

            setTimeout(() => {
              if (!this.serverModelConfig) {
                console.log('[onBotReady] 3rd attempt - requesting config...');
                this.requestServerConfig();
              }
            }, 1000);
          },
          onUserTranscript: (data: any) => {
            // Clean STT text (remove SentencePiece tokens like ▁, replacement chars, etc.)
            const cleanedText = this.cleanSTTText(data.text);

            if (data.final) {
              const reason = data.result?.reason || '';
              // Use flowing paradigm - upgrade partial to final
              this.processFinalTranscript(cleanedText, reason);
              this.addStreamEntry('final', cleanedText);

              // Update STT stream panel with final text
              this.updateSTTStreamPanel(cleanedText, true);

              // Voice Agent: Finalize user message in conversation view
              if (this.currentUserMessage) {
                this.currentUserMessage.text = cleanedText;
                this.currentUserMessage.status = 'complete';
                this.currentUserMessage = null;
                this.updateConversationView();
              }

              // Finalize previous LLM response and prepare for new one
              this.finalizeLLMResponse();

              // Also finalize bot message in VOICE AGENT view for proper turn separation
              if (this.currentBotMessage && this.currentBotMessage.text.trim()) {
                this.currentBotMessage.status = 'complete';
                this.finalizeBotMessage();
              }
            } else {
              this.updatePartialTranscript(cleanedText);
              this.addStreamEntry('partial', cleanedText);

              // Update STT stream panel with partial text
              this.updateSTTStreamPanel(cleanedText, false);

              // Voice Agent: Create or update user message in conversation view
              if (!this.currentUserMessage) {
                // NEW USER TURN: Finalize any existing bot message first
                // This ensures proper turn separation when user interrupts the bot
                if (this.currentBotMessage && this.currentBotMessage.text.trim()) {
                  this.currentBotMessage.status = 'complete';
                  this.finalizeBotMessage();
                }

                this.currentUserMessage = {
                  id: `user-${++this.messageIdCounter}`,
                  role: 'user',
                  text: cleanedText,
                  timestamp: new Date(),
                  status: 'speaking',
                };
                this.conversationMessages.push(this.currentUserMessage);
              } else {
                this.currentUserMessage.text = cleanedText;
              }
              this.updateConversationView();
            }
          },
          onBotTranscript: (data) => {
            // Bot transcription = finalized sentence sent to TTS
            // NOTE: Do NOT overwrite currentBotMessage.text here!
            // The accumulated text is already managed by handleLLMText/handleLLMStream.
            // onBotTranscript receives only ONE sentence, not the full accumulated text.
            this.addStreamEntry('system', `Bot: ${data.text}`);

            // Ensure bot message exists
            if (!this.currentBotMessage) {
              this.createBotMessage();
            }

            // Only update status to 'speaking', preserve accumulated text
            if (this.currentBotMessage) {
              // If accumulated text is empty (edge case), use transcript text
              if (!this.llmAccumulatedText || this.llmAccumulatedText.trim() === '') {
                this.currentBotMessage.text = data.text;
                this.llmAccumulatedText = data.text;
              }
              // Otherwise, keep the accumulated text as-is
              this.currentBotMessage.status = 'speaking';
              this.updateConversationView();
            }

            // Update OUTPUT panel with FULL accumulated text (not just this sentence)
            this.updateLLMStreamPanel(this.llmAccumulatedText, false);
          },
          // Handle streaming LLM tokens (bot-llm-text)
          onBotLlmText: (data: any) => {
            // This callback receives streamed LLM tokens
            if (data && data.text) {
              this.handleLLMText(data.text);
            }
          },
          // Handle TTS text (for sync)
          onBotTtsText: (data: any) => {
            // This callback receives text being sent to TTS
            if (data && data.text) {
              this.addStreamEntry('system', `TTS: ${data.text.substring(0, 50)}...`);
              // Update bot status to speaking
              if (this.currentBotMessage) {
                this.currentBotMessage.status = 'speaking';
                this.updateConversationView();
              }
            }
          },
          onUserStartedSpeaking: () => {
            this.updateVADStatus(true);
            this.addStreamEntry('vad', 'User started speaking');
          },
          onUserStoppedSpeaking: () => {
            this.updateVADStatus(false);
            this.addStreamEntry('vad', 'User stopped speaking');
          },
          onMessageError: (error) => {
            this.addStreamEntry('error', `Message error: ${error}`);
          },
          onError: (error) => {
            this.addStreamEntry('error', `Error: ${error}`);
          },
        },
      };

      this.rtviClient = new RTVIClient(RTVIConfig);
      this.setupTrackListeners();

      this.addStreamEntry('system', 'Initializing devices...');
      await this.rtviClient.initDevices();
      this.addStreamEntry('system', 'Devices initialized');

      this.addStreamEntry('system', 'Connecting to server...');
      await this.rtviClient.connect();

    } catch (error) {
      this.addStreamEntry('error', `Connection failed: ${(error as Error).message}`);
      this.updateStatus('Error');
      await this.cleanupOnError();
    } finally {
      this.isConnecting = false;
    }
  }

  /**
   * Clean up on error
   */
  private async cleanupOnError(): Promise<void> {
    this.isDisconnecting = true;
    const client = this.rtviClient;

    if (client) {
      try {
        if (typeof client.disconnect === 'function') {
          await client.disconnect();
        }
      } catch (e) {
        // Ignore disconnect errors
      } finally {
        this.rtviClient = null;
      }
    }

    if (this.connectBtn) this.connectBtn.disabled = false;
    if (this.disconnectBtn) this.disconnectBtn.disabled = true;
    if (this.muteBtn) {
      this.muteBtn.disabled = true;
      this.muteBtn.querySelector('.btn-text')!.textContent = 'Mute';
      this.muteBtn.classList.remove('muted');
    }

    this.stopVolumeMonitoring();
    this.stopVADTimer();
    this.isMuted = false;
    this.isDisconnecting = false;
  }

  /**
   * Disconnect from server
   */
  public async disconnect(): Promise<void> {
    if (this.isDisconnecting) return;

    this.isDisconnecting = true;
    this.addStreamEntry('system', 'Disconnecting...');

    try {
      const client = this.rtviClient;

      if (client) {
        try {
          if (typeof client.disconnect === 'function') {
            await client.disconnect();
          }
        } catch (error) {
          console.warn('Disconnect error (non-critical):', error);
        } finally {
          this.rtviClient = null;
          if (this.botAudio.srcObject && "getAudioTracks" in this.botAudio.srcObject) {
            this.botAudio.srcObject.getAudioTracks().forEach((track) => track.stop());
            this.botAudio.srcObject = null;
          }
        }
      }

      // Clean up resources
      this.stopVolumeMonitoring();
      this.stopVADTimer();
      this.isMuted = false;
      this.wsInterceptorAttached = false;
      this.serverModelConfig = null;

      // Update UI state - Critical fix: ensure buttons are properly reset
      this.updateStatus('Disconnected');
      this.updateFooterStats('Ready');

      if (this.connectBtn) this.connectBtn.disabled = false;
      if (this.disconnectBtn) this.disconnectBtn.disabled = true;
      if (this.muteBtn) {
        this.muteBtn.disabled = true;
        const btnText = this.muteBtn.querySelector('.btn-text');
        if (btnText) btnText.textContent = 'Mute';
        this.muteBtn.classList.remove('muted');
      }

      this.updateVADStatus(false);
      this.updateBadgesState(false);
      this.resetModelInfoDisplay();
      this.addStreamEntry('system', 'Disconnected from server');

    } finally {
      this.isDisconnecting = false;
    }
  }

  /**
   * Toggle mute
   */
  private toggleMute(): void {
    if (!this.rtviClient) return;

    this.isMuted = !this.isMuted;
    this.rtviClient.enableMic(!this.isMuted);

    if (this.muteBtn) {
      const btnText = this.muteBtn.querySelector('.btn-text');
      if (btnText) {
        btnText.textContent = this.isMuted ? 'Unmute' : 'Mute';
      }
      if (this.isMuted) {
        this.muteBtn.classList.add('muted');
      } else {
        this.muteBtn.classList.remove('muted');
      }
    }

    if (this.isMuted) {
      this.stopVolumeMonitoring();
    } else {
      this.startVolumeMonitoring();
    }

    // Toggle input zone active state (glowing border)
    if (this.inputZone) {
      this.inputZone.classList.toggle('active', !this.isMuted);
    }

    this.addStreamEntry('system', this.isMuted ? 'Microphone muted' : 'Microphone unmuted');
  }

  /**
   * Send text input to the server via RTVI action (text_input:send)
   * Used in text input modes (LLM-Only, TTS-Only, LLM+TTS)
   */
  private async sendTextInput(): Promise<void> {
    if (!this.rtviClient || !this.textInputTextarea) {
      console.warn('[TextInput] Cannot send: RTVI client or textarea not available');
      return;
    }

    const text = this.textInputTextarea.value.trim();
    if (!text) {
      console.debug('[TextInput] Empty text, skipping send');
      return;
    }

    console.log('[TextInput] Sending text:', text.substring(0, 50) + (text.length > 50 ? '...' : ''));

    // Finalize any existing bot message before sending new text input
    // This ensures each text input gets its own separate response bubble
    if (this.currentBotMessage) {
      console.log('[TextInput] Finalizing previous bot message before new input');
      this.finalizeBotMessage();
    }

    // Disable input while sending
    this.textInputTextarea.disabled = true;
    if (this.textSendBtn) {
      this.textSendBtn.disabled = true;
      this.textSendBtn.textContent = 'Sending...';
    }

    try {
      // Send text directly via WebSocket JSON message (bypass RTVI Protobuf serialization)
      // The RTVI client's action() method may not preserve argument values correctly
      const ws = getInterceptedWebSocket();
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        throw new Error('WebSocket not connected');
      }

      const actionMessage = {
        id: `text-input-${Date.now()}`,
        type: 'action',
        data: {
          service: 'text_input',
          action: 'send',
          arguments: [
            { name: 'text', value: text }
          ]
        }
      };

      console.log('[TextInput] Sending via WebSocket:', JSON.stringify(actionMessage));
      ws.send(JSON.stringify(actionMessage));

      // Note: We don't wait for response here since we're sending directly via WebSocket
      // The server will process the text and respond via LLM output frames
      console.log('[TextInput] Message sent successfully');

      // Clear textarea on success
      this.textInputTextarea.value = '';

      // Add to stream panel (show user input)
      this.addStreamEntry('user', text);

      // Add user message to conversation view (말풍선)
      // Use 'text-input' as source to differentiate from STT input
      this.addTextInputUserMessage(text);

      // Update text input status badge (update .status-label inside badge)
      if (this.textInputStatusBadge) {
        const statusLabel = this.textInputStatusBadge.querySelector('.status-label');
        if (statusLabel) {
          statusLabel.textContent = 'Sent';
        }
        this.textInputStatusBadge.classList.add('active');
        setTimeout(() => {
          if (this.textInputStatusBadge) {
            const label = this.textInputStatusBadge.querySelector('.status-label');
            if (label) {
              label.textContent = 'Ready';
            }
          }
        }, 1500);
      }

    } catch (error) {
      console.error('[TextInput] Failed to send:', error);
      this.addStreamEntry('error', `Failed to send text: ${error}`);

      // Update status badge to show error (update .status-label inside badge)
      if (this.textInputStatusBadge) {
        const statusLabel = this.textInputStatusBadge.querySelector('.status-label');
        if (statusLabel) {
          statusLabel.textContent = 'Error';
        }
        this.textInputStatusBadge.classList.remove('active');
        setTimeout(() => {
          if (this.textInputStatusBadge) {
            const label = this.textInputStatusBadge.querySelector('.status-label');
            if (label) {
              label.textContent = 'Ready';
            }
          }
        }, 2000);
      }
    } finally {
      // Re-enable input
      this.textInputTextarea.disabled = false;
      if (this.textSendBtn) {
        this.textSendBtn.disabled = false;
        this.textSendBtn.textContent = 'Send';
      }

      // Focus back to textarea for convenience
      this.textInputTextarea.focus();
    }
  }

  /**
   * Start volume monitoring using RTVI's local audio track
   * Falls back to getUserMedia if RTVI track is not available
   */
  private async startVolumeMonitoring(): Promise<void> {
    console.log('[Audio] Starting volume monitoring...');
    try {
      if (!this.audioContext) {
        this.audioContext = new AudioContext();
        console.log('[Audio] Created new AudioContext, state:', this.audioContext.state);
      }

      // Resume audio context if suspended (browser autoplay policy)
      if (this.audioContext.state === 'suspended') {
        console.log('[Audio] Resuming suspended AudioContext...');
        await this.audioContext.resume();
        console.log('[Audio] AudioContext resumed, state:', this.audioContext.state);
      }

      let stream: MediaStream | null = null;

      // Try to use RTVI's local audio track first (avoids device conflicts)
      if (this.rtviClient) {
        try {
          const tracks = this.rtviClient.tracks();
          console.log('[Audio] RTVI tracks:', {
            hasLocal: !!tracks.local,
            hasLocalAudio: !!tracks.local?.audio,
            hasBot: !!tracks.bot,
            hasBotAudio: !!tracks.bot?.audio
          });
          if (tracks.local?.audio) {
            stream = new MediaStream([tracks.local.audio]);
            console.log('[Audio] Using RTVI local audio track for volume monitoring');
          }
        } catch (e) {
          console.debug('[Audio] Could not get RTVI local track:', e);
        }
      }

      // Fallback: try getUserMedia if RTVI track not available
      if (!stream) {
        try {
          console.log('[Audio] Falling back to getUserMedia...');
          stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          console.log('[Audio] getUserMedia succeeded');
        } catch (e) {
          // Volume monitoring is non-critical - continue without it
          console.warn('[Audio] Volume monitoring unavailable:', e);
          this.addStreamEntry('system', 'Volume meter unavailable (microphone handled by RTVI)');
          return;
        }
      }

      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.smoothingTimeConstant = 0.8;

      this.microphone = this.audioContext.createMediaStreamSource(stream);
      this.microphone.connect(this.analyser);
      console.log('[Audio] Input analyser connected successfully');

      this.volumeUpdateInterval = window.setInterval(() => {
        this.updateVolumeDisplay();
      }, 50); // Faster updates for smoother visualization

      // Start the enhanced audio visualizer
      this.startVisualization();
      console.log('[Audio] Input visualization started');

    } catch (error) {
      // Volume monitoring is non-critical - log but don't show error to user
      console.error('[Audio] Volume monitoring setup failed:', error);
    }
  }

  /**
   * Stop volume monitoring
   */
  private stopVolumeMonitoring(): void {
    if (this.volumeUpdateInterval) {
      clearInterval(this.volumeUpdateInterval);
      this.volumeUpdateInterval = null;
    }

    if (this.microphone) {
      this.microphone.disconnect();
      this.microphone = null;
    }

    // Stop the enhanced audio visualizer
    this.stopVisualization();
    this.updateVolumeDisplay(0);
  }

  /**
   * Update volume display (input microphone)
   */
  private updateVolumeDisplay(volume?: number): void {
    if (volume === undefined && this.analyser) {
      const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
      this.analyser.getByteFrequencyData(dataArray);
      const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
      volume = (average / 255) * 100;
    }

    const displayVolume = Math.min(100, Math.max(0, volume || 0));

    // Update legacy volume display
    if (this.volumeBar) {
      this.volumeBar.style.width = `${displayVolume}%`;
    }
    if (this.volumeText) {
      this.volumeText.textContent = `${Math.round(displayVolume)}%`;
    }

    // Update new input volume display
    if (this.inputVolumeBar) {
      this.inputVolumeBar.style.width = `${displayVolume}%`;
    }
    if (this.inputVolumeText) {
      this.inputVolumeText.textContent = `${Math.round(displayVolume)}%`;
    }
  }

  private getServerConfig(): { name: string; baseUrl: string; wsPort: number } {
    return this.voiceAgentServer;
  }

  // ============================================
  // Badge and Model Info Management
  // ============================================

  /**
   * Update all module badges state (active/inactive)
   */
  private updateBadgesState(active: boolean): void {
    const badges = [this.sttBadge, this.llmBadge, this.ttsBadge];
    badges.forEach(badge => {
      if (badge) {
        badge.classList.toggle('active', active);
      }
    });
    // DIAR badge depends on diarization being enabled
    if (this.diarBadge) {
      this.diarBadge.classList.toggle('active', active && this.diarizationEnabled);
    }
  }

  /**
   * Update individual badge state
   */
  private updateBadgeState(badge: HTMLElement | null, active: boolean): void {
    if (badge) {
      badge.classList.toggle('active', active);
    }
  }

  /**
   * Handle server config message and update model info panel
   */
  private handleServerConfigMessage(config: any): void {
    console.log('[server-config] Received config:', config);
    this.serverModelConfig = config;

    // Debug: Log DOM element availability
    console.log('[server-config] DOM Elements check:', {
      sttModelName: !!this.sttModelName,
      llmModelName: !!this.llmModelName,
      ttsModelName: !!this.ttsModelName,
      diarModelName: !!this.diarModelName,
    });

    // Detect input mode: text input mode if STT is disabled
    // Server sends input_mode: "text" or "voice", or we check stt.enabled
    const inputMode = config.input_mode || (config.stt?.enabled === false ? 'text' : 'voice');
    this.isTextInputMode = inputMode === 'text';

    console.log('[server-config] Input mode:', inputMode, 'isTextInputMode:', this.isTextInputMode);

    // Update input zones visibility based on input mode
    this.updateInputZoneVisibility();

    // Update STT info (check if STT is enabled)
    const sttEnabled = config.stt?.enabled !== false && config.stt;
    if (sttEnabled) {
      const modelName = this.extractModelName(config.stt.model || '-');
      console.log('[server-config] STT model:', modelName);
      if (this.sttModelName) {
        this.sttModelName.textContent = modelName;
      }
      if (this.sttModelParams) {
        this.sttModelParams.textContent = `device: ${config.stt.device || 'cuda'}`;
      }
      // Activate STT badge
      this.updateBadgeState(this.sttBadge, true);
      this.addDebugEntry('info', `STT: ${modelName}`);
    } else {
      // STT is disabled (text input mode)
      console.log('[server-config] STT: Disabled (text input mode)');
      if (this.sttModelName) {
        this.sttModelName.textContent = 'Disabled';
      }
      if (this.sttModelParams) {
        this.sttModelParams.textContent = 'Text input mode';
      }
      this.updateBadgeState(this.sttBadge, false);
      this.addDebugEntry('info', 'STT: Disabled (text input mode)');
    }

    // Update LLM info (check if LLM is enabled)
    const llmEnabled = config.llm?.enabled !== false && config.llm;
    if (llmEnabled) {
      const modelName = this.extractModelName(config.llm.model || '-');
      console.log('[server-config] LLM model:', modelName);
      if (this.llmModelName) {
        this.llmModelName.textContent = modelName;
      }
      if (this.llmModelParams) {
        this.llmModelParams.textContent = `type: ${config.llm.type || 'vllm'}`;
      }
      // Activate LLM badge
      this.updateBadgeState(this.llmBadge, true);
      this.addDebugEntry('info', `LLM: ${modelName}`);
    } else {
      // LLM is disabled (TTS-Only mode or STT-Only mode)
      console.log('[server-config] LLM: Disabled');
      if (this.llmModelName) {
        this.llmModelName.textContent = 'Disabled';
      }
      if (this.llmModelParams) {
        this.llmModelParams.textContent = '-';
      }
      this.updateBadgeState(this.llmBadge, false);
      this.addDebugEntry('info', 'LLM: Disabled');
    }

    // Update TTS info (check if TTS is enabled)
    const ttsEnabled = config.tts?.enabled !== false && config.tts;
    if (ttsEnabled) {
      const modelName = config.tts.model ? this.extractModelName(config.tts.model) : config.tts.type || '-';
      console.log('[server-config] TTS model:', modelName);
      if (this.ttsModelName) {
        this.ttsModelName.textContent = modelName;
      }
      if (this.ttsModelParams) {
        this.ttsModelParams.textContent = `type: ${config.tts.type || '-'}`;
      }
      // Activate TTS badge
      this.updateBadgeState(this.ttsBadge, true);
      this.addDebugEntry('info', `TTS: ${modelName}`);
    } else {
      // TTS is disabled (STT-Only or LLM-Only mode)
      console.log('[server-config] TTS: Disabled');
      if (this.ttsModelName) {
        this.ttsModelName.textContent = 'Disabled';
      }
      if (this.ttsModelParams) {
        this.ttsModelParams.textContent = '-';
      }
      this.updateBadgeState(this.ttsBadge, false);
      this.addDebugEntry('info', 'TTS: Disabled');
    }

    // Update DIAR info
    if (config.diar) {
      if (config.diar.enabled) {
        const modelName = this.extractModelName(config.diar.model || '-');
        console.log('[server-config] DIAR model:', modelName);
        if (this.diarModelName) {
          this.diarModelName.textContent = modelName;
        }
        if (this.diarModelParams) {
          this.diarModelParams.textContent = `threshold: ${config.diar.threshold || 0.4}`;
        }
        this.diarizationEnabled = true;
        this.updateBadgeState(this.diarBadge, true);
        this.addDebugEntry('info', `DIAR: ${modelName}`);
      } else {
        console.log('[server-config] DIAR: Disabled');
        if (this.diarModelName) {
          this.diarModelName.textContent = 'Disabled';
        }
        if (this.diarModelParams) {
          this.diarModelParams.textContent = '-';
        }
        this.diarizationEnabled = false;
        this.updateBadgeState(this.diarBadge, false);
      }
    }

    // Detect STT-only mode (STT enabled, LLM disabled, TTS disabled)
    // In STT-only mode, hide the output zone to maximize input/transcript space
    this.isSTTOnlyMode = sttEnabled && !llmEnabled && !ttsEnabled;
    if (this.transcriptArea) {
      this.transcriptArea.classList.toggle('stt-only-mode', this.isSTTOnlyMode);
      console.log('[server-config] STT-only mode:', this.isSTTOnlyMode);
    }

    this.addStreamEntry('system', 'Server config received');
    this.addDebugEntry('system', 'Model configuration loaded');

    // Log the detected mode
    const modeDescription = this.isTextInputMode
      ? 'Text Input Mode (STT disabled)'
      : 'Voice Input Mode (STT enabled)';
    this.addStreamEntry('system', `Mode: ${modeDescription}`);
  }

  /**
   * Update input zone visibility based on input mode
   * Voice mode: Show microphone input zone, hide text input zone
   * Text mode: Show text input zone, hide microphone input zone
   */
  private updateInputZoneVisibility(): void {
    console.log('[InputZone] Updating visibility, isTextInputMode:', this.isTextInputMode);

    if (this.isTextInputMode) {
      // Text input mode: Show text input zone, hide voice input zone
      if (this.inputZone) {
        this.inputZone.style.display = 'none';
        console.log('[InputZone] Voice input zone hidden');
      }
      if (this.textInputZone) {
        this.textInputZone.style.display = 'flex';
        console.log('[InputZone] Text input zone shown');
      }
      // Disable mute button (not relevant for text input)
      if (this.muteBtn) {
        this.muteBtn.style.display = 'none';
      }
      // Update text input status badge
      if (this.textInputStatusBadge) {
        this.textInputStatusBadge.classList.add('active');
        const statusLabel = this.textInputStatusBadge.querySelector('.status-label');
        if (statusLabel) {
          statusLabel.textContent = 'Ready';
        }
      }
      this.addStreamEntry('system', 'Text input mode active - type your message and press Enter or click Send');
    } else {
      // Voice input mode: Show voice input zone, hide text input zone
      if (this.inputZone) {
        this.inputZone.style.display = 'flex';
        console.log('[InputZone] Voice input zone shown');
      }
      if (this.textInputZone) {
        this.textInputZone.style.display = 'none';
        console.log('[InputZone] Text input zone hidden');
      }
      // Enable mute button
      if (this.muteBtn) {
        this.muteBtn.style.display = '';
      }
    }
  }

  /**
   * Extract short model name from full path
   */
  private extractModelName(fullPath: string): string {
    if (!fullPath || fullPath === '-') return '-';
    // Handle paths like "nvidia/parakeet_realtime_eou_120m-v1" or "/path/to/model"
    const parts = fullPath.split('/');
    return parts[parts.length - 1] || fullPath;
  }

  /**
   * Reset model info display to default state
   */
  private resetModelInfoDisplay(): void {
    if (this.sttModelName) this.sttModelName.textContent = '-';
    if (this.sttModelParams) this.sttModelParams.textContent = 'Waiting for config...';
    if (this.llmModelName) this.llmModelName.textContent = '-';
    if (this.llmModelParams) this.llmModelParams.textContent = 'Waiting for config...';
    if (this.ttsModelName) this.ttsModelName.textContent = '-';
    if (this.ttsModelParams) this.ttsModelParams.textContent = 'Waiting for config...';
    if (this.diarModelName) this.diarModelName.textContent = '-';
    if (this.diarModelParams) this.diarModelParams.textContent = 'Waiting for config...';
  }

  /**
   * Request server config if not received automatically
   * This is a fallback mechanism for race condition handling
   */
  private requestServerConfig(): void {
    if (!this.rtviClient) return;

    console.log('[server-config] Requesting config from server (fallback)');

    try {
      // Send action to request server config
      this.rtviClient.action({
        service: 'config',
        action: 'get_server_config',
        arguments: []
      }).catch((error: Error) => {
        // Action might not be supported, log for debugging
        console.debug('[server-config] Request action not supported:', error.message);
      });
    } catch (error) {
      console.debug('[server-config] Failed to request config:', error);
    }
  }

  // ============================================
  // Real-time Stream Panel Methods
  // ============================================

  /**
   * Update STT stream panel with partial or final transcript
   */
  private updateSTTStreamPanel(text: string, isFinal: boolean): void {
    if (!this.sttStreamContent) return;

    let streamFlow = this.sttStreamContent.querySelector('.stream-flow');
    if (!streamFlow) {
      streamFlow = document.createElement('div');
      streamFlow.className = 'stream-flow';
      this.sttStreamContent.appendChild(streamFlow);
    }

    // Add streaming glow effect to input zone
    if (!isFinal && this.inputZone) {
      this.inputZone.classList.add('streaming');
    }

    if (isFinal) {
      // Remove streaming glow effect when final
      this.inputZone?.classList.remove('streaming');

      // Remove any partial line
      const partialLine = streamFlow.querySelector('.partial-text');
      if (partialLine) partialLine.remove();

      // Add final line
      const line = document.createElement('div');
      line.className = 'stream-line final-text';
      line.innerHTML = `<span class="stream-marker final-marker"></span> ${text}`;
      streamFlow.appendChild(line);
    } else {
      // Update or create partial line
      let currentLine = streamFlow.querySelector('.partial-text') as HTMLElement;
      if (!currentLine) {
        currentLine = document.createElement('div');
        currentLine.className = 'stream-line partial-text';
        streamFlow.appendChild(currentLine);
      }
      currentLine.innerHTML = `<span class="stream-marker typing-marker"></span> ${text}`;
    }

    this.sttStreamContent.scrollTop = this.sttStreamContent.scrollHeight;
  }

  /**
   * Update LLM stream panel with streaming response
   */
  private updateLLMStreamPanel(text: string, isChunk: boolean = true): void {
    if (!this.llmStreamContent) return;

    // Add streaming glow effect to output zone
    if (this.outputZone) {
      this.outputZone.classList.add('streaming');
    }

    let streamFlow = this.llmStreamContent.querySelector('.stream-flow');
    if (!streamFlow) {
      streamFlow = document.createElement('div');
      streamFlow.className = 'stream-flow';
      this.llmStreamContent.appendChild(streamFlow);
    }

    // Get or create current response container
    let currentResponse = streamFlow.querySelector('.current-response') as HTMLElement;
    if (!currentResponse) {
      currentResponse = document.createElement('div');
      currentResponse.className = 'stream-line current-response';
      currentResponse.innerHTML = '<span class="stream-marker llm-marker"></span> <span class="response-text"></span>';
      streamFlow.appendChild(currentResponse);
    }

    const responseText = currentResponse.querySelector('.response-text');
    if (responseText) {
      if (isChunk) {
        responseText.textContent += text;
      } else {
        responseText.textContent = text;
      }
    }

    this.llmStreamContent.scrollTop = this.llmStreamContent.scrollHeight;
  }

  /**
   * Finalize current LLM response and prepare for next
   */
  private finalizeLLMResponse(): void {
    if (!this.llmStreamContent) return;

    // Remove streaming glow effect
    this.outputZone?.classList.remove('streaming');

    const streamFlow = this.llmStreamContent.querySelector('.stream-flow');
    if (!streamFlow) return;

    const currentResponse = streamFlow.querySelector('.current-response');
    if (currentResponse) {
      currentResponse.classList.remove('current-response');
      currentResponse.classList.add('completed-response');
      const marker = currentResponse.querySelector('.stream-marker');
      if (marker) {
        marker.classList.remove('llm-marker');
        marker.classList.add('complete-marker');
      }
    }
  }

  /**
   * Clear STT stream panel
   */
  private clearSTTStream(): void {
    if (this.sttStreamContent) {
      const streamFlow = this.sttStreamContent.querySelector('.stream-flow');
      if (streamFlow) streamFlow.innerHTML = '';
    }
  }

  /**
   * Clear LLM stream panel
   */
  private clearLLMStream(): void {
    if (this.llmStreamContent) {
      const streamFlow = this.llmStreamContent.querySelector('.stream-flow');
      if (streamFlow) streamFlow.innerHTML = '';
    }
  }

  /**
   * Setup resize handle for stream panels
   */
  private setupStreamPanelResize(): void {
    if (!this.streamResizeHandle || !this.sttStreamPanel || !this.llmStreamPanel) return;

    let isResizing = false;
    let startX = 0;
    let startWidthLeft = 0;
    let startWidthRight = 0;

    this.streamResizeHandle.addEventListener('mousedown', (e: MouseEvent) => {
      isResizing = true;
      startX = e.clientX;
      startWidthLeft = this.sttStreamPanel!.offsetWidth;
      startWidthRight = this.llmStreamPanel!.offsetWidth;
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    });

    document.addEventListener('mousemove', (e: MouseEvent) => {
      if (!isResizing) return;

      const diff = e.clientX - startX;
      const totalWidth = startWidthLeft + startWidthRight;
      const newLeftWidth = Math.max(150, Math.min(totalWidth - 150, startWidthLeft + diff));
      const newRightWidth = totalWidth - newLeftWidth;

      const leftPercent = (newLeftWidth / totalWidth) * 100;
      const rightPercent = (newRightWidth / totalWidth) * 100;

      this.sttStreamPanel!.style.flex = `0 0 ${leftPercent}%`;
      this.llmStreamPanel!.style.flex = `0 0 ${rightPercent}%`;
    });

    document.addEventListener('mouseup', () => {
      if (isResizing) {
        isResizing = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      }
    });
  }
}

declare global {
  interface Window {
    WebsocketClientApp: typeof WebsocketClientApp;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  // Suppress non-critical microphone errors from Daily.js in headless/file-upload mode
  // These errors occur when no microphone is available but don't affect file upload functionality
  window.addEventListener('unhandledrejection', (event) => {
    const reason = event.reason;
    if (reason instanceof Error &&
        (reason.name === 'NotFoundError' ||
         reason.message?.includes('Requested device not found') ||
         reason.message?.includes('No Mic'))) {
      console.info('[Audio] Microphone not available - this is normal for file upload mode');
      event.preventDefault();
    }
  });

  window.WebsocketClientApp = WebsocketClientApp;
  new WebsocketClientApp();
});
