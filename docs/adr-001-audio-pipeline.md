# ADR-001: Audio pipeline tuning (live + after-recording)

**Status:** Accepted
**Date:** 2026-10-16
**Deciders:** Murad (owner)

## Context

Murmur Notetaker must transcribe meetings with low latency, keep speakers apart (also several people
on one microphone, e.g. a phone call on speaker), handle mixed AZ/DE/EN, and work in noisy rooms
without a meeting bot. We studied Meetily, WhisperLiveKit, anarlog (Hyprnote), OpenWhispr,
screenpipe and Buzz. None fine-tunes models; their quality comes from the pipeline around stock
Whisper/Parakeet: VAD, echo cancellation, denoise, decoding guards, and post-processing.

## Decision

| Stage | Choice | Source / evidence |
|---|---|---|
| Capture | Mic + WASAPI loopback at 48 kHz; soxr to 16 kHz for recognition | Meetily (48 kHz chain) |
| Live echo cancellation | DTLN-aec 128 (streaming, 0.08x realtime, no added delay), loopback as reference | anarlog ships the same models |
| Live echo guards | Drop mic lines the canceller cut by >10 dB while speakers played; text match vs. the other channel within ±3 s (±2 s for one word) | own tests |
| Segmentation | Silero VAD (onset 0.5, min speech 250 ms, end after 500 ms silence, 300 ms pre-pad) | Meetily live path |
| Interim captions | Fast pass (beam 1, no word timestamps) every 0.8 s on the open sentence, only when the engine is idle | WhisperLiveKit idea; first text ~2.6 s after audio start |
| Decoding | large-v3-turbo; beam 3 live, beam 5 refine; faster-whisper thresholds (≈ Meetily: logprob -1.0, no-speech 0.6, compression 2.4); `hallucination_silence_threshold` 2 s on refine | Meetily whisper_engine.rs |
| Hallucination filter | Meetily `clean_repetitive_text` port + drop ≤3-word low-confidence lines | Meetily |
| Vocabulary | `hotwords` from Settings + known speaker names | OpenWhispr custom dictionary |
| Language | Allowed set AZ/DE/EN; on clips < 1.5 s keep detection only if confidence ≥ 0.8, else channel's last language | WhisperLiveKit notes on auto-detect bias |
| Live speakers | WeSpeaker voiceprint per sentence, online clustering at cosine 0.74 (measured: same ≥ 0.93, different ≤ 0.72 on our 4-voice test) | own measurement |
| After recording | DTLN-aec (or DTLN denoise when no loopback and room noise floor > -55 dBFS), guard if near speech loses > 10 dB, dynaudnorm; then refine pass, diarization | anarlog + own tests |
| Diarization | pyannote seg 3.0 + WeSpeaker ResNet34, threshold 0.35, per-line voiceprint reassignment (margin 0.08), names from self-introductions, live names carried over | sweeps on 4 test meetings: 4/3/3/2 speakers correct |
| Listening mix | 48 kHz, mic HPF 80 Hz + afftdn + dynaudnorm, per-channel speech level -20 dBFS | Meetily chain |

### Update (same day): live robustness, from a real recording (YouTube interview + own voice)

| Problem seen | Cause (measured) | Change |
|---|---|---|
| Audio dropped ("data discontinuity", 40-70 per recording) | sherpa-onnx holds the GIL for seconds; capture threads starved | Capture threads only enqueue; one processing thread does the rest; live re-diarization runs in a child process. Down to 4-6 warnings, all at start-up |
| Speaker echo not cancelled | Loopback leads the mic by 180-550 ms (Bluetooth output) | Echo delay measured every 5 s (10 ms log-energy cross-correlation), reference shifted; also offline (250 ms, corr 0.79) |
| Live merged two remote voices | Compressed remote voices are very similar (mean 0.82, max 0.94); same person as low as 0.52 | Background full diarization every ≥20 s on the last 3 minutes, mapped onto live keys by overlap |
| "Thank you." fragments on mic | Whisper on residual echo | While speakers play, mic lines need no-speech ≤ 0.3 and logprob ≥ -0.6 (live and refine). 10 → 1 line |
| Same person resumes after a pause | merged into the old block | Live view starts a new line after a 4 s pause |
| Muting a source | needed during calls | Mute mic / computer audio live; muted audio is written as silence (timeline kept) |

Tried and rejected: choosing the speaker count by silhouette score (picked 2 instead of 3 on both real
recordings); Sortformer (needs NeMo/PyTorch, ggml copy not loadable from Python).

### Update: Murmur Copilot answer latency and style, benchmarked against ParakeetAI

ParakeetAI was measured in a live mock interview (questions asked by its AI interviewer, answered with TTS) and
its stored answers were read from 2 past sessions (62 answers). Ours was measured the same way: TTS question
through the speakers, live capture, answer in the UI.

| Step | ParakeetAI | Murmur Copilot before | Murmur Copilot now |
|---|---|---|---|
| Question end -> answer starts | 1.25 s | 2.1-2.4 s | 0.53-0.65 s |
| Question end -> first answer words | ~1.4 s | 2.9-3.2 s | 1.8-2.3 s |
| Full answer (~100-120 words) | ~3 s | 2.3-3.6 s | 2.3-3.6 s |

What changed:
- Model call without reasoning (`reasoning_effort: none`): first token 3.3 s -> 0.9 s on gpt-6-sol; one stream
  instead of opening + continuation. Screenshots keep a little reasoning (code).
- Warm-up request at session start: the long prompt is cached, first answer 3.1 s -> 0.8 s.
- Speculative start: when the voice detector closes an interviewer sentence, the answer starts from the live
  caption; the final text (0.7 s later) confirms it (>= 80 % same words) or the answer is cancelled and redone.
- Copilot sessions transcribe live in fast mode (beam 1, no word timings); the after-recording pass redoes it.
- The trigger waits while the interviewer is talking (VAD) or a sentence is still being transcribed: no more
  answers to half questions ("Let's start by hearing a bit about you" ...).

Style taken from ParakeetAI's answers: a short question title on top, the first sentence answers directly,
70-160 words, short sentences, polite "could you repeat that?" for unclear questions. Kept from the user's own
rules: spoken and non-academic, stories from real projects (no invented incidents or numbers, no meta talk such
as "I can't point to..."), STAR order only for experience questions and without labels.

## Options considered

- **NLMS echo canceller (own)**: 25 dB on linear echo, no noise reduction; kept as fallback.
- **DTLN-aec**: 32 dB on non-linear echo plus ~19 dB noise reduction, 0.08x realtime. Chosen.
- **Cloud streaming ASR (Speechmatics, as ParakeetAI does)**: lowest latency, but audio leaves the
  machine and costs per minute. Not chosen; can be added later as an option.

## Consequences

- Easier: noisy rooms, phone calls on speaker, speakers without headphones, mixed languages.
- Harder: after-recording processing takes longer (~5 min CPU per hour for DTLN, plus the refine pass).
- Revisit: live speaker threshold on real voices (TTS voices are more consistent than people);
  LocalAgreement-style committed interim words; Ollama as a local summary provider.
