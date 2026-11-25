# Changelog

## 1.1.0 — 2026-11-26

- **`--json`**: structured output on stdout (`basic`, `events`, `timeline`, `masking`,
  `sections`, `roughness_hotspots`, `brightness`, `paths`, …), Markdown report still written.
- **Simultaneous voices** (model, up to 3): integer-supported candidates that form a common
  interval; octave/harmonic and inharmonic-partial candidates are filtered out.
- `--no-spectrogram`, `--version`.
- Fix: lossy codes (mp3/aac) smeared the attack over a few frames and the same-frame level
  gate silently dropped **every** onset in such files — level gate now looks 3 frames ahead.
- Fix: a clean single-partial tone was rejected as "not tonal" (added a spectral-concentration gate).
- Fix: onset threshold used p99, so one loud attack raised the bar and hid quieter strikes (now p90).
- Fix: all-silent files reported key confidence `-9.00` and guessed a tempo; both now report "unknown".
- Self-test grew from 12 to 21 cases (chords, JSON, 24-bit WAV, clipping, silence, echo negatives).

## 1.0.0 — 2026-11-25

- First release: pitch/cents, Sethares roughness, critical-band beating, masking hints,
  silence + spectral-novelty segmentation, key and tempo estimates, spectrogram,
  echo de-duplication, quiet-gate, 12 self-tests.
