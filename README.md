# audio-ears

**A tiny audio-analysis skill for AI agents — it measures the sound and tells you how far to trust each number.**

Agents can't hear. This skill gives an agent a ruler: pitch and cents, loudness, roughness,
masking, segmentation, spectrograms, and a cautious guess at simultaneous voices — plus a
self-test that proves the ruler is not bent. No librosa, no PyTorch, no model downloads:
**numpy + ffmpeg only**, so it runs on a phone.

```bash
python3 scripts/selftest.py                    # 24 checks against synthetic ground truth
python3 scripts/analyze.py song.mp3 out/       # -> out/song-声学分析报告.md (+ spectrogram)
python3 scripts/analyze.py song.mp3 out/ --json # -> JSON on stdout (+ the same .md)
```

A full sample report (from a synthetic demo with three bells, one chord and one masked partial):
[examples/sample-report.md](examples/sample-report.md) · [examples/sample-analysis.json](examples/sample-analysis.json).

## Why another audio analyzer

Most audio tooling answers "what is this signal". An agent writing or judging music needs
something slightly different: *"should I believe this reading?"* So every report splits into

| Tier | Examples | How to use it |
|---|---|---|
| **Measured** | duration, peak/RMS, clipping, onset times, peak frequency | state as fact |
| **Model** | key (Krumhansl), tempo, roughness (Sethares), beating, masking, cents | always hedge: "likely / model says" |
| **Unknown** | dense textures, sustained polyphony, non-Western scales | say so, don't invent |

The same idea shapes the numbers: quiet frames (below peak −45 dB) are excluded instead of
turned into fake roughness; onsets are gated 3 frames ahead because lossy codecs smear attacks;
echo/tail repeats are de-duplicated with a deliberately narrow window.

## What it measures

- **Basic**: duration, sample rate, channels, peak / RMS, clipping samples, L/R correlation.
- **Strikes / onsets**: time, fundamental (peak of 60–1500 Hz in a +20…+820 ms window,
  parabolic interpolation), nearest note, **cents deviation**, roughness, critical-band beat pairs.
- **Structure**: splits on silence (≥1.2 s) plus spectral novelty (2 s cosine distance);
  per-segment mean level and active ratio.
- **Key** (Krumhansl templates, with a confidence gate) and **tempo** (comb-scored onset-flux autocorrelation).
- **Time line**: spectral centroid (brightness) and roughness per 0.5 s, with hotspots.
- **Simultaneous voices** (model, up to 3): other fundamentals still ringing inside the same
  analysis window — including the tail of the previous strike, so it is *not* a chord transcription.
  Candidates must be integer-supported by at least two spectral peaks and form a common interval
  (±25 cents) with the main note; octave/harmonic and inharmonic-partial candidates are filtered out.
- **Masking hints**: weak partials sitting inside the critical band of a stronger one.
- **Spectrogram**: matplotlib if available, else `ffmpeg showspectrumpic`, else skipped.
- **JSON** (`--json`): every number above in a stable structure (`basic`, `events`, `timeline`,
  `masking`, `sections`, `paths`, …), for programs rather than eyes.

## Requirements

- Python 3.8+ and `numpy` (only hard dependency).
- `ffmpeg` — optional, for non-WAV input and the spectrogram fallback.
- `matplotlib` — optional, for a nicer spectrogram.

Tested on Python 3.12 / numpy 2.5 in a proot Ubuntu on Android 15 (no root), after being
used to mix and critique dozens of synthesized tracks.

## Validation, not vibes

`scripts/selftest.py` synthesizes signals with known ground truth and asserts the report matches:

| Case | Ground truth | Result |
|---|---|---|
| clean strike | 330.00 Hz | 330.00 Hz, +2 cents, E4 |
| inharmonic bell (partials 1.00/2.00/2.76/4.07/5.41/8.13) | 261.63 Hz | 261.6 Hz, 0 cents, C4 |
| just major third vs 12-TET | +15.6 cents | +16 cents |
| three strikes @ 1.5 s spacing | 3 onsets | 3 onsets @ 0.6 / 2.1 / 3.6 s |
| echo 0.4 s later, 10 dB down | 1 strike | echoed strike removed |
| one 1.5 s gap | 1 silence run | 1 run, 2 onsets |
| long fade tail | no roughness in silence | max roughness 0.000 (old build: 0.836) |
| 1000 Hz + 1050 Hz weak partial | masking hint | "1050 Hz may be masked" |
| C major arpeggio | C major | C major (0.86) |
| stereo file | 2 channels | 2 channels + correlation |
| MP3 round-trip | 440 Hz | 440.00 Hz |
| missing file | clean error | exit 1 + message |
| JSON output | valid JSON, 1 event | 23 top-level fields, paths exist |
| chord C4+E4+G4 struck together | 3 fundamentals | C4 + E4 + G4 (±15 cents) |
| single inharmonic bell | 1 voice | 1 voice (2.76/4.07 partials rejected) |
| unit rules | note names, beat pairs, roughness order | minor 2nd ≫ octave; 5 Hz beat pair found |
| overdriven signal | clipping | 5,145 clipped samples, peak 0 dBFS |
| 24-bit WAV | 261.63 Hz | C4, ±5 cents |
| 120 BPM click train | 120 BPM | 120 BPM |
| legato 120 BPM arpeggio (overlapping bells) | 120 BPM | 119 BPM (old build: 200) |
| all-silent file | nothing to report | 0 events, no key/tempo/brightness |
| two different notes 0.4 s apart | no echo removal | both kept |
| `--no-spectrogram` | no PNG | PNG path null, Markdown still written |

These tests were written *after* the failures they now cover: an earlier build read a clean
330.1 Hz E4 as 325.9 Hz (**−20 cents**, i.e. exactly a wolf fifth — the worst possible wrong
answer), reported roughness 0.836 inside a fade-out, silently lost every onset in MP3 input,
and missed a quiet strike that followed a loud one because a global p99 threshold let the
loudest attack set the bar. Each is now a regression test.

## Relation to existing work

This is a deliberate narrow tool, not a replacement for the ecosystem:

- **librosa / aubio / Essentia / basic-pitch / music21** — mature libraries with far better
  time-frequency, transcription and MIR algorithms. Use them when you can afford the
  dependencies and want accuracy over portability.
- **[oaustegard/claude-skills](https://github.com/oaustegard/claude-skills) `listening-to-music`**
  (~417★) — the closest "agent skill" in spirit: render → spectrogram → measure → compare,
  with pYIN, chroma, rub meters and sheet music. It needs Node + Playwright + Chromium and
  Python with librosa/matplotlib; excellent on a workstation, not on a phone.
- **audio-ears** sits at the other end: one file, numpy only, phone-sized, opinionated about
  *reporting uncertainty* (tiers, confidence gates, known bias) more than about state-of-the-art DSP.

## Use as a skill

The `SKILL.md` front matter follows the common agent-skill convention (name + description).
For DSH-like runtimes, drop the directory into the skills folder, e.g.:

```bash
cp -r audio-ears ~/.dsh/skills/audio-ears
```

Then ask the agent something like "why does this mix sound muddy" and let it run `analyze.py`.
For Claude-style skill bundles, the same directory works as a skill folder.

## Known limits

- **Voices are a guess.** The main series reports the loudest fundamental; up to three
  simultaneously ringing fundamentals are added when they clear the harmonic/interval filters.
  Because the window is 0.8 s long, a previous strike's tail can be reported as a "voice" — this
  is a decay-tail effect, not always a chord. Dense textures still under-report, and the reported
  "main" note of a chord is arbitrary (it's the loudest).
- **+1.5 cents systematic bias** on real reverberant material (synthetic material: within ±3).
  Compare relative deviations within one file, not absolute pitch, unless calibrated.
- **Key/tempo are models**, can lock to the wrong metrical level (half/double reported) or a
  relative key; the confidence gate is a filter, not a proof.
- **Roughness is Sethares' pairwise model**, useful for ranking, meaningless as an absolute
  perceptual unit.
- **Chinese output.** Human-readable report labels are Chinese; use `--json` for a machine-readable
  structure instead of parsing the Markdown.

## 中文说明

一个给 agent 用的音频分析技能。agent 听不见，但可以用这把尺子量：音高与音分偏差、
响度动态、Sethares 粗糙度、临界带拍频、掩蔽提示、结构分段、声谱图；
更重要的是报告会明确区分「测到的」「模型推的」「测不了」。

没有 librosa、不下载模型，只依赖 numpy + ffmpeg——所以在手机上（proot Android）也能跑。
这不是一个 MIR 库，是一个带误差棒的听诊器。

- 跑分析：`python3 scripts/analyze.py <音频> [输出目录]`（加 `--json` 出结构化结果）
- 自测：`python3 scripts/selftest.py`（24 项，全部用合成信号的真值验证）
- 边界：主音高取「最响的峰」；并行音最多猜 3 个、可能漏；真实混响素材系统性偏高约 +1.5 音分；
  结构分段是声学变化点，不是曲式。

## License

MIT. See `LICENSE`.
