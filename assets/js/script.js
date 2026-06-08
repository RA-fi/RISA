// ========================= GLOBAL STATE =========================
var isVoice = 0;
var isVoiceConversation = false;
var _streamingActive = false;  // true while SSE stream is delivering TTS text
const VOICE_STORAGE_KEY = 'preferredVoiceName';
const LANG_STORAGE_KEY  = 'preferredLangCode';

// ========================= SILENCE KEEPALIVE (mobile browser fix) =========================
let silenceInterval = null;

function startSilence() {
    stopSilence();
    silenceInterval = setInterval(() => {
        if (!window.speechSynthesis.speaking) {
            const u = new SpeechSynthesisUtterance(' ');
            u.volume = 0;
            window.speechSynthesis.speak(u);
        }
    }, 10000);
}

function stopSilence() {
    if (silenceInterval) { clearInterval(silenceInterval); silenceInterval = null; }
}

// ========================= AUDIO STATE (Web Audio API — immune to autoplay blocking) =========================
let _audioCtx    = null;   // Shared AudioContext (created once, reused)
let _audioSource = null;   // Active BufferSourceNode
let _risaAudioEl = null;   // Fallback legacy <audio> element

function _getAudioCtx() {
    if (!_audioCtx || _audioCtx.state === 'closed') {
        _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    return _audioCtx;
}

// Unlock AudioContext on first user gesture (required by browser autoplay policy)
function _unlockAudioCtx() {
    try {
        const ctx = _getAudioCtx();
        if (ctx.state === 'suspended') ctx.resume().catch(() => {});
    } catch (_) {}
}
document.addEventListener('click',      _unlockAudioCtx, { once: true });
document.addEventListener('keydown',    _unlockAudioCtx, { once: true });
document.addEventListener('touchstart', _unlockAudioCtx, { once: true, passive: true });

function _stopRisaAudio() {
    if (_audioSource) {
        _audioSource.onended = null;
        try { _audioSource.stop(); } catch (_) {}
        _audioSource = null;
    }
    if (_risaAudioEl) {
        _risaAudioEl.pause();
        try { URL.revokeObjectURL(_risaAudioEl.src); } catch (_) {}
        _risaAudioEl = null;
    }
}

function stopSpeech() {
    _stopRisaAudio();
    if ('speechSynthesis' in window && window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
    }
}

function cleanupVoiceUI() {
    _stopRisaAudio();
    $('#voiceIndicator').removeClass('active speaking').hide();
    $('#voice_search').removeClass('voice-active speaking');
    stopSilence();
}

// Deferred cleanup — only fires when both streaming and TTS are fully done
function _maybeFinalCleanup() {
    if (_streamingActive) return;
    if (_ttsIsPlaying) return;
    if ('speechSynthesis' in window && (window.speechSynthesis.speaking || window.speechSynthesis.pending)) return;
    cleanupVoiceUI();
    isVoiceConversation = false;
}

// ========================= LANGUAGE UTILITIES =========================
const _LANG_FULL_MAP = {
    'bn':'bn-BD','BN':'bn-BD','bn-BD':'bn-BD','bn-IN':'bn-IN',
    'hi':'hi-IN','HI':'hi-IN','hi-IN':'hi-IN',
    'en':'en-US','EN':'en-US','en-US':'en-US','en-GB':'en-GB',
    'ar':'ar-SA','AR':'ar-SA','ar-SA':'ar-SA','ar-EG':'ar-EG',
    'ur':'ur-PK','ur-PK':'ur-PK',
    'fr':'fr-FR','fr-FR':'fr-FR','fr-CA':'fr-CA',
    'de':'de-DE','de-DE':'de-DE',
    'es':'es-ES','es-ES':'es-ES','es-MX':'es-MX',
    'it':'it-IT','it-IT':'it-IT',
    'pt':'pt-BR','pt-BR':'pt-BR','pt-PT':'pt-PT',
    'nl':'nl-NL','nl-NL':'nl-NL',
    'ru':'ru-RU','ru-RU':'ru-RU',
    'pl':'pl-PL','pl-PL':'pl-PL',
    'tr':'tr-TR','tr-TR':'tr-TR',
    'zh':'zh-CN','zh-CN':'zh-CN','zh-TW':'zh-TW',
    'ja':'ja-JP','ja-JP':'ja-JP',
    'ko':'ko-KR','ko-KR':'ko-KR',
    'id':'id-ID','id-ID':'id-ID',
    'ms':'ms-MY','ms-MY':'ms-MY',
    'th':'th-TH','th-TH':'th-TH',
    'vi':'vi-VN','vi-VN':'vi-VN',
    'tl':'fil-PH','fil-PH':'fil-PH',
    'ta':'ta-IN','ta-IN':'ta-IN',
    'te':'te-IN','te-IN':'te-IN',
    'ml':'ml-IN','ml-IN':'ml-IN',
    'gu':'gu-IN','gu-IN':'gu-IN',
    'kn':'kn-IN','kn-IN':'kn-IN',
    'mr':'mr-IN','mr-IN':'mr-IN',
    'pa':'pa-IN','pa-IN':'pa-IN',
    'ne':'ne-NP','ne-NP':'ne-NP',
    'si':'si-LK','si-LK':'si-LK',
};

function _normalizeLang(lang) {
    if (!lang || lang === 'unknown' || lang === 'auto') return 'en-US';
    if (_LANG_FULL_MAP[lang]) return _LANG_FULL_MAP[lang];
    if (lang.includes('-')) return lang;
    return 'en-US';
}

// ========================= BACKEND TTS — edge-tts neural voices via Web Audio API =========================
function _plainText(html) {
    return (html || '')
        .replace(/<[^>]+>/g, ' ')
        .replace(/\*\*([^*]+)\*\*/g, '$1')
        .replace(/\*([^*]+)\*/g, '$1')
        .replace(/#+\s*/g, '')
        .replace(/\s+/g, ' ')
        .trim()
        .substring(0, 2500);
}

async function speakAIResponse(text, detectedLang, alwaysSpeak = false) {
    const shouldSpeak = isVoiceConversation || alwaysSpeak ||
                        localStorage.getItem('alwaysSpeak') === 'true';
    if (!shouldSpeak) return;

    const clean = _plainText(text);
    if (!clean) return;

    const lang = _normalizeLang(detectedLang || 'en');
    stopSpeech();
    _ttsQueueClear();
    _streamingActive = false;
    $('#voiceIndicator').addClass('active').show();

    // For languages where browser TTS is good, use it directly (instant, no network)
    if (!_preferBackendSTT(lang) && _speakSentenceBrowser(clean, lang)) return;

    // Edge-tts via backend: best neural quality for all languages
    const API_BASE = (window.RISA_API_BASE || '').replace(/\/$/, '');
    try {
        const resp = await fetch(`${API_BASE}/tts`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: clean, lang }),
        });
        if (!resp.ok) throw new Error(`TTS HTTP ${resp.status}`);
        const arrayBuffer = await resp.arrayBuffer();
        if (!arrayBuffer || arrayBuffer.byteLength < 100) throw new Error('Empty audio');
        const ctx = _getAudioCtx();
        if (ctx.state === 'suspended') await ctx.resume();
        const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
        const source = ctx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(ctx.destination);
        _audioSource = source;
        source.onended = () => { _audioSource = null; _maybeFinalCleanup(); };
        $('#voiceIndicator').addClass('speaking');
        source.start(0);
    } catch (err) {
        console.warn('[TTS]', err.message);
        // Try browser TTS as last resort
        if (!_speakSentenceBrowser(clean, lang)) {
            cleanupVoiceUI();
            isVoiceConversation = false;
        }
    }
}

// ========================= DEVICE-NATIVE BROWSER TTS (primary) =========================
let _speechVoices = [];

function _cacheSpeechVoices() {
    if ('speechSynthesis' in window) {
        const v = window.speechSynthesis.getVoices();
        if (v.length) _speechVoices = v;
    }
}
if ('speechSynthesis' in window) {
    _cacheSpeechVoices();
    window.speechSynthesis.addEventListener('voiceschanged', _cacheSpeechVoices);
}

function _getBrowserVoice(lang) {
    if (!_speechVoices.length) _cacheSpeechVoices();
    const norm = _normalizeLang(lang || 'en');
    const base = norm.split('-')[0];
    return _speechVoices.find(v => v.lang === norm)
        || _speechVoices.find(v => v.lang.startsWith(base + '-'))
        || _speechVoices.find(v => v.lang.startsWith(base))
        || null;
}

function _speakSentenceBrowser(text, lang) {
    if (!('speechSynthesis' in window) || !text) return false;
    const voice = _getBrowserVoice(lang);
    if (!voice) return false;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = voice.lang;
    u.voice = voice;
    u.rate = 1.0;
    u.pitch = 1.0;
    u.volume = 1.0;
    u.onstart = () => $('#voiceIndicator').addClass('active speaking').show();
    u.onend = () => { _maybeFinalCleanup(); };
    window.speechSynthesis.speak(u);
    return true;
}

// ========================= TTS AUDIO QUEUE — edge-tts fallback when device has no voice =========================
const _ttsQueue = [];
let _ttsIsPlaying = false;

function _ttsQueueClear() {
    _ttsQueue.length = 0;
    _ttsIsPlaying = false;
    _streamingActive = false;
    _stopRisaAudio();
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    $('#voiceIndicator').removeClass('active speaking').hide();
}

function _ttsEnqueue(text, lang) {
    const clean = (text || '').replace(/\s+/g, ' ').trim();
    if (!clean || clean.length < 3) return;
    _ttsQueue.push({ text: clean, lang: lang || 'en' });
    if (!_ttsIsPlaying) _ttsPlayNext();
}

async function _ttsPlayNext() {
    if (!_ttsQueue.length) {
        _ttsIsPlaying = false;
        _maybeFinalCleanup();
        return;
    }
    _ttsIsPlaying = true;
    const { text, lang } = _ttsQueue.shift();
    await _playTTSSentence(text, lang);
    _ttsPlayNext();
}

async function _playTTSSentence(text, lang) {
    return new Promise(async (resolve) => {
        const API_BASE = (window.RISA_API_BASE || '').replace(/\/$/, '');
        const normalizedLang = _normalizeLang(lang || 'en');
        const controller = new AbortController();
        const tid = setTimeout(() => controller.abort(), 12000);
        try {
            const resp = await fetch(`${API_BASE}/tts`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, lang: normalizedLang }),
                signal: controller.signal,
            });
            clearTimeout(tid);
            if (!resp.ok) throw new Error(`TTS ${resp.status}`);

            const buf = await resp.arrayBuffer();
            if (!buf || buf.byteLength < 100) throw new Error('Empty TTS audio');

            const ctx = _getAudioCtx();
            if (ctx.state === 'suspended') await ctx.resume();

            const audioBuffer = await ctx.decodeAudioData(buf.slice(0));
            const source = ctx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(ctx.destination);
            _audioSource = source;

            $('#voiceIndicator').addClass('active speaking').show();
            source.onended = () => { _audioSource = null; resolve(); };
            source.start(0);
        } catch (err) {
            clearTimeout(tid);
            if (err.name !== 'AbortError') console.warn('[TTS Queue]', err.message);
            resolve();
        }
    });
}

// ── Fallback: browser speechSynthesis ───────────────────────────
function _browserTTSFallback(text, lang) {
    if (!('speechSynthesis' in window)) {
        cleanupVoiceUI(); isVoiceConversation = false; return;
    }
    const synth = window.speechSynthesis;
    synth.cancel();
    const voices = synth.getVoices();
    if (!voices.length) { cleanupVoiceUI(); isVoiceConversation = false; return; }

    const base  = lang.split('-')[0];
    const voice = voices.find(v => v.lang === lang)
               || voices.find(v => v.lang.startsWith(base))
               || voices.find(v => v.lang.startsWith('en'))
               || voices[0];

    const u = new SpeechSynthesisUtterance(text);
    u.lang   = lang;
    u.rate   = 1.0;
    u.pitch  = 1.0;
    u.volume = 1.0;
    if (voice) u.voice = voice;

    u.onstart = () => $('#voiceIndicator').addClass('speaking');
    u.onend   = () => { _maybeFinalCleanup(); };
    u.onerror = () => { cleanupVoiceUI(); isVoiceConversation = false; };
    synth.speak(u);
}

// ========================= BACKEND STT — Groq Whisper v3 via MediaRecorder =========================
let _mediaRecorder  = null;
let _audioChunks    = [];
let _mediaStream    = null;
let _micLang        = 'bn-BD';
let _webSpeechRec   = null;   // Active Web Speech recognition (for supported langs)
let _mediaAudioCtx  = null;
let _mediaAnalyser  = null;
let _mediaSource    = null;
let _mediaFrame     = null;
let _silenceTimer   = null;
let _maxTimer       = null;
let _speechDetected = false;

const _AUTO_STOP_MAX_MS = 15000;
const _AUTO_STOP_SILENCE_MS = 1200;
const _SPEECH_RMS_THRESHOLD = 0.010;

function _cleanupMediaMonitor() {
    if (_mediaFrame) {
        cancelAnimationFrame(_mediaFrame);
        _mediaFrame = null;
    }
    if (_silenceTimer) {
        clearTimeout(_silenceTimer);
        _silenceTimer = null;
    }
    if (_maxTimer) {
        clearTimeout(_maxTimer);
        _maxTimer = null;
    }
    if (_mediaSource) {
        try { _mediaSource.disconnect(); } catch (_) {}
        _mediaSource = null;
    }
    if (_mediaAudioCtx) {
        try { _mediaAudioCtx.close(); } catch (_) {}
        _mediaAudioCtx = null;
    }
    _mediaAnalyser = null;
    _speechDetected = false;
}

function _startAutoStopMonitor(stream) {
    _cleanupMediaMonitor();
    try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) return;

        _mediaAudioCtx = new AudioCtx();
        _mediaSource = _mediaAudioCtx.createMediaStreamSource(stream);
        _mediaAnalyser = _mediaAudioCtx.createAnalyser();
        _mediaAnalyser.fftSize = 2048;
        _mediaSource.connect(_mediaAnalyser);

        const samples = new Float32Array(_mediaAnalyser.fftSize);

        const monitor = () => {
            if (!_mediaRecorder || _mediaRecorder.state === 'inactive' || !_mediaAnalyser) return;

            _mediaAnalyser.getFloatTimeDomainData(samples);
            let sum = 0;
            for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
            const rms = Math.sqrt(sum / samples.length);

            if (rms > _SPEECH_RMS_THRESHOLD) {
                _speechDetected = true;
                if (_silenceTimer) {
                    clearTimeout(_silenceTimer);
                    _silenceTimer = null;
                }
                $('#mic-status').addClass('active').text('Listening… auto-stop when you pause');
            } else if (_speechDetected && !_silenceTimer) {
                _silenceTimer = setTimeout(() => {
                    if (_mediaRecorder && _mediaRecorder.state !== 'inactive') {
                        $('#mic-status').addClass('active').text('Processing…');
                        try { new Audio('assets/audio/end.mp3').play().catch(() => {}); } catch (_) {}
                        stopMediaRecording();
                    }
                }, _AUTO_STOP_SILENCE_MS);
            }

            _mediaFrame = requestAnimationFrame(monitor);
        };

        _maxTimer = setTimeout(() => {
            if (_mediaRecorder && _mediaRecorder.state !== 'inactive') {
                $('#mic-status').addClass('active').text('Processing…');
                try { new Audio('assets/audio/end.mp3').play().catch(() => {}); } catch (_) {}
                stopMediaRecording();
            }
        }, _AUTO_STOP_MAX_MS);

        _mediaFrame = requestAnimationFrame(monitor);
    } catch (err) {
        console.warn('[STT] Auto-stop monitor unavailable:', err);
    }
}

function _bestMimeType() {
    const types = [
        'audio/webm;codecs=opus', 'audio/webm',
        'audio/ogg;codecs=opus',  'audio/ogg', 'audio/mp4'
    ];
    return types.find(t => { try { return MediaRecorder.isTypeSupported(t); } catch { return false; } })
        || 'audio/webm';
}

// Languages where Whisper backend gives much better results than Web Speech API
function _preferBackendSTT(lang) {
    const base = (lang || '').split('-')[0].toLowerCase();
    const poor = ['bn','ur','ne','si','my','km','lo','as','or','gu','kn','ml','te','ta','pa'];
    return poor.includes(base);
}

function _hasWebSpeechAPI() {
    return !!(window.SpeechRecognition || window.webkitSpeechRecognition);
}

async function startMediaRecording(lang) {
    _micLang = lang || 'bn-BD';
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 16000
            }
        });
        _mediaStream  = stream;
        _audioChunks  = [];

        const mimeType = _bestMimeType();
        _mediaRecorder = new MediaRecorder(stream, { mimeType });

        _mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) _audioChunks.push(e.data);
        };

        _mediaRecorder.onstop = async () => {
            _cleanupMediaMonitor();
            if (_mediaStream) { _mediaStream.getTracks().forEach(t => t.stop()); _mediaStream = null; }
            if (!_audioChunks.length) {
                showCustomAlert('No audio captured. Please try again.');
                _resetMicUI(); return;
            }

            const blob = new Blob(_audioChunks, { type: mimeType });
            _audioChunks = [];
            $('#mic-status').text('Transcribing…');
            $('#recoredText').text('…');

            try {
                const result = await _transcribeWithGroq(blob, _micLang);
                const transcribed = result.text || '';

                if (transcribed && transcribed.trim().length > 1) {
                    $('#recoredText').text(transcribed);
                    // _micLang stays as the user-selected language — response will match UI selection
                    setTimeout(() => {
                        $('#microphone').removeClass('visible');
                        $('#prompt').val(transcribed);
                        isVoiceConversation = true;
                        _resetMicUI();
                        $('#sendBtn').click();
                    }, 500);
                } else {
                    showCustomAlert('No speech detected. Please speak clearly and try again.');
                    _resetMicUI();
                }
            } catch (err) {
                console.error('[STT] Backend transcription failed:', err);
                showCustomAlert('Transcription failed. Check your internet connection and try again.');
                _resetMicUI();
            }
        };

        _mediaRecorder.start(250);
    _startAutoStopMonitor(stream);
        return true;
    } catch (err) {
        console.error('[STT] MediaRecorder error:', err);
        if (err.name === 'NotAllowedError') {
            showCustomAlert('Microphone access denied. Please allow microphone access in your browser settings.');
        } else if (err.name === 'NotFoundError') {
            showCustomAlert('No microphone found. Please connect a microphone and try again.');
        } else {
            showCustomAlert('Could not access microphone. Please check your device settings.');
        }
        return false;
    }
}

function stopMediaRecording() {
    _cleanupMediaMonitor();
    if (_mediaRecorder && _mediaRecorder.state !== 'inactive') _mediaRecorder.stop();
    if (_mediaStream) { _mediaStream.getTracks().forEach(t => t.stop()); _mediaStream = null; }
}

async function _transcribeWithGroq(audioBlob, lang) {
    const API_BASE = (window.RISA_API_BASE || '').replace(/\/$/, '');
    const type = audioBlob.type || '';
    const ext  = type.includes('ogg') ? 'ogg' : type.includes('mp4') ? 'm4a' : 'webm';
    const fd   = new FormData();
    fd.append('file', audioBlob, `audio.${ext}`);
    fd.append('lang', lang || '');

    const resp = await fetch(`${API_BASE}/transcribe`, { method: 'POST', body: fd });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    // Return text; lang from backend mirrors user's UI selection
    return { text: data.text || '', lang: data.lang || lang || 'en' };
}

function _resetMicUI() {
    $('#speakBtn').removeClass('active recording');
    $('#mic-status').removeClass('active').text('Click the microphone to speak');
    $('#recoredText').text('');
}

// ========================= WEB SPEECH API WRAPPER (for well-supported languages) =========================
function _startWebSpeechRecognition(lang, { onResult, onError, onEnd } = {}) {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) throw new Error('Web Speech API not available');

    const rec = new SR();
    rec.continuous      = false;
    rec.interimResults  = true;
    rec.lang            = lang;
    rec.maxAlternatives = 1;

    // Safari: limit to supported languages
    const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent) &&
                     !/CriOS|FxiOS|EdgiOS/i.test(navigator.userAgent);
    if (isSafari && _preferBackendSTT(lang)) {
        rec.lang = 'en-US'; // Safari doesn't support Bangla recognition
    }

    rec.addEventListener('result', (e) => {
        let final = '', interim = '';
        for (let i = e.resultIndex; i < e.results.length; i++) {
            const t = e.results[i][0].transcript;
            if (e.results[i].isFinal) final += t; else interim += t;
        }
        if (onResult) onResult(final || interim, !!final);
    });
    rec.addEventListener('error', (e) => { if (onError) onError(e.error); });
    rec.addEventListener('end',   () => { if (onEnd) onEnd(); });
    rec.addEventListener('start', () => $('#voice_search').addClass('voice-active'));

    rec.start();
    return rec;
}

// ========================= LANGUAGE SELECTOR =========================
$(document).ready(function () {
    const LANGS = [
        { code: 'bn-BD',  label: 'বাংলা (Bangladesh)' },
        { code: 'bn-IN',  label: 'বাংলা (India)' },
        { code: 'en-US',  label: 'English (US)' },
        { code: 'en-GB',  label: 'English (UK)' },
        { code: 'hi-IN',  label: 'हिंदी' },
        { code: 'ur-PK',  label: 'اردو' },
        { code: 'ar-SA',  label: 'العربية' },
        { code: 'ar-EG',  label: 'العربية (مصر)' },
        { code: 'fr-FR',  label: 'Français' },
        { code: 'fr-CA',  label: 'Français (CA)' },
        { code: 'de-DE',  label: 'Deutsch' },
        { code: 'es-ES',  label: 'Español' },
        { code: 'es-MX',  label: 'Español (MX)' },
        { code: 'it-IT',  label: 'Italiano' },
        { code: 'pt-BR',  label: 'Português (BR)' },
        { code: 'pt-PT',  label: 'Português (PT)' },
        { code: 'nl-NL',  label: 'Nederlands' },
        { code: 'ru-RU',  label: 'Русский' },
        { code: 'pl-PL',  label: 'Polski' },
        { code: 'tr-TR',  label: 'Türkçe' },
        { code: 'zh-CN',  label: '中文 (简体)' },
        { code: 'zh-TW',  label: '中文 (繁體)' },
        { code: 'ja-JP',  label: '日本語' },
        { code: 'ko-KR',  label: '한국어' },
        { code: 'id-ID',  label: 'Bahasa Indonesia' },
        { code: 'ms-MY',  label: 'Bahasa Melayu' },
        { code: 'th-TH',  label: 'ภาษาไทย' },
        { code: 'vi-VN',  label: 'Tiếng Việt' },
        { code: 'fil-PH', label: 'Filipino' },
        { code: 'ta-IN',  label: 'தமிழ்' },
        { code: 'te-IN',  label: 'తెలుగు' },
        { code: 'ml-IN',  label: 'മലയാളം' },
        { code: 'gu-IN',  label: 'ગુજરાતી' },
        { code: 'kn-IN',  label: 'ಕನ್ನಡ' },
        { code: 'mr-IN',  label: 'मराठी' },
        { code: 'pa-IN',  label: 'ਪੰਜਾਬੀ' },
        { code: 'ne-NP',  label: 'नेपाली' },
        { code: 'si-LK',  label: 'සිංහල' },
    ];

    const $sel    = $('#lang');
    const stored  = localStorage.getItem(LANG_STORAGE_KEY) || 'bn-BD';

    $sel.empty();
    LANGS.forEach(({ code, label }) => {
        const useBackend = _preferBackendSTT(code);
        const badge      = useBackend ? ' ✦' : '';   // ✦ = Whisper-powered
        const $opt = $('<option>', { value: code, text: label + badge });
        $opt.attr('data-lang', code);
        if (code === stored) $opt.prop('selected', true);
        $sel.append($opt);
    });

    $sel.on('change', function () {
        const lang = $(this).find('option:selected').attr('data-lang') || 'bn-BD';
        localStorage.setItem(LANG_STORAGE_KEY, lang);
    });
});

// ========================= UI EVENT HANDLERS =========================
$(document).ready(function () {
    // Stop voice button
    $('#stopVoiceBtn').click(function () {
        _ttsQueueClear();
        stopSpeech();
        cleanupVoiceUI();
        isVoiceConversation = false;
        if (_webSpeechRec) { try { _webSpeechRec.stop(); } catch (_) {} _webSpeechRec = null; }
        stopMediaRecording();
    });

    // Click outside microphone panel to close
    $(document).click(function () { $('#microphone').removeClass('visible'); });

    // Main mic button — opens modal
    $('#voice_search').click(function (event) {
        stopSpeech();
        event.stopPropagation();
        $('#microphone').addClass('visible');
        $('#mic-status').text('Select language, then tap the microphone').removeClass('active');
        $('#recoredText').text('');
    });

    $('.recoder').click(function (e) { e.stopPropagation(); });
    $('#microphone .close').click(function () { $('#microphone').removeClass('visible'); });

    $('#customAlertClose').click(hideCustomAlert);
    if ($('#aboutLink').length)      $('#aboutLink').click(showAboutModal);
    if ($('#aboutModalClose').length) $('#aboutModalClose').click(hideAboutModal);
});

// ========================= SPEAK BUTTON — recording logic =========================
$(document).ready(function () {
    // Debounce flag & audio feedback
    function _playAudio(src) { try { new Audio(src).play().catch(() => {}); } catch (_) {} }

    $('#speakBtn').on('click', async function () {
        if ($(this).hasClass('processing')) return;
        $(this).addClass('processing');
        setTimeout(() => $(this).removeClass('processing'), 800);

        const lang = ($('#lang option:selected').attr('data-lang')
                   || localStorage.getItem(LANG_STORAGE_KEY) || 'bn-BD');

        if ($('#speakBtn').hasClass('active')) {
            // ── STOP recording ──────────────────────────────────────
            $('#speakBtn').removeClass('active recording');
            $('#mic-status').text('Processing…').addClass('active');
            _playAudio('assets/audio/end.mp3');

            if (_mediaRecorder && _mediaRecorder.state !== 'inactive') {
                stopMediaRecording();
            } else if (_webSpeechRec) {
                try { _webSpeechRec.stop(); } catch (_) {}
                _webSpeechRec = null;
            }

        } else {
            // ── START recording ─────────────────────────────────────
            $('#speakBtn').addClass('active');
            $('#recoredText').text('');
            _playAudio('assets/audio/start.mp3');

            // Bangla and other low-coverage languages always use Whisper backend
            const needsWhisper = _preferBackendSTT(lang) || !_hasWebSpeechAPI();

            if (needsWhisper) {
                // ── Path A: MediaRecorder → Groq Whisper ──────────────
                const started = await startMediaRecording(lang);
                if (started) {
                    $('#speakBtn').addClass('recording');
                    const hint = _preferBackendSTT(lang)
                        ? 'Recording (Whisper AI) — auto-stop when you pause'
                        : 'Recording — auto-stop when you pause';
                    $('#mic-status').addClass('active').text(hint);
                } else {
                    $('#speakBtn').removeClass('active');
                }

            } else {
                // ── Path B: Web Speech API (real-time, supported languages) ──
                $('#mic-status').addClass('active').text('Listening…');

                const isSecure = window.isSecureContext ||
                    window.location.protocol === 'https:' ||
                    ['localhost', '127.0.0.1'].includes(window.location.hostname);

                if (!isSecure) {
                    showCustomAlert('Voice recognition requires HTTPS or localhost.');
                    $('#speakBtn').removeClass('active');
                    return;
                }

                try {
                    _webSpeechRec = _startWebSpeechRecognition(lang, {
                        onResult: (text, isFinal) => {
                            $('#recoredText').text(text);
                            if (isFinal && text.length > 1) {
                                _webSpeechRec = null;
                                // Set _micLang to the selected language so source_lang is correct
                                _micLang = lang;
                                setTimeout(() => {
                                    $('#microphone').removeClass('visible');
                                    $('#prompt').val(text);
                                    isVoiceConversation = true;
                                    _resetMicUI();
                                    $('#sendBtn').click();
                                }, 400);
                            }
                        },
                        onError: async (err) => {
                            console.error('[WebSpeech]', err);
                            _webSpeechRec = null;
                            // Transparent fallback to Whisper
                            if (err === 'no-speech') {
                                showCustomAlert('No speech detected. Please try again.');
                                _resetMicUI();
                            } else {
                                $('#mic-status').text('Switching to Whisper…');
                                const ok = await startMediaRecording(lang);
                                if (ok) {
                                    $('#speakBtn').addClass('active recording');
                                    $('#mic-status').addClass('active').text('Recording (Whisper) — auto-stop when you pause');
                                } else {
                                    _resetMicUI();
                                }
                            }
                        },
                        onEnd: () => {
                            if ($('#speakBtn').hasClass('active') && !$('#recoredText').text()) {
                                _resetMicUI();
                            }
                        }
                    });
                } catch (err) {
                    console.error('[WebSpeech] Start failed:', err);
                    // Silent fallback to Whisper
                    const ok = await startMediaRecording(lang);
                    if (ok) {
                        $('#speakBtn').addClass('recording');
                        $('#mic-status').addClass('active').text('Recording (Whisper) — auto-stop when you pause');
                    } else {
                        _resetMicUI();
                    }
                }
            }
        }
    });

    // Legacy stop-voice click on stop button inside voice modal
    $('#stopVoiceBtn').off('click').on('click', function () {
        _ttsQueueClear();
        stopSpeech();
        cleanupVoiceUI();
        isVoiceConversation = false;
        if (_webSpeechRec) { try { _webSpeechRec.stop(); } catch (_) {} _webSpeechRec = null; }
        stopMediaRecording();
    });
});

// ========================= CUSTOM ALERT =========================
function showCustomAlert(message) {
    $('#customAlertMessage').html(message);
    $('#customAlert').addClass('visible');
}

function hideCustomAlert() {
    $('#customAlert').removeClass('visible');
}

// ========================= ABOUT MODAL =========================
function showAboutModal() {
    $('#aboutModal').addClass('visible');
    $('body').addClass('modal-open');
    $(document).on('keydown.aboutModal', function (e) {
        if (e.key === 'Escape') hideAboutModal();
    });
    $('#aboutModal').on('click.aboutModal', function (e) {
        if (e.target === this) hideAboutModal();
    });
    setTimeout(() => $('#aboutModalClose').focus(), 100);
}

function hideAboutModal() {
    $('#aboutModal').removeClass('visible');
    $('body').removeClass('modal-open');
    $(document).off('keydown.aboutModal');
    $('#aboutModal').off('click.aboutModal');
    $('#aboutLink').focus();
}
