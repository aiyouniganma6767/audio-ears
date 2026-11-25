# -*- coding: utf-8 -*-
"""声学分析：给 agent 用的一只「金耳朵」。

对一段音频做可复现的测量，输出一份 Markdown 报告（`--json` 时输出结构化 JSON）：

- 基本：时长、峰值、RMS、削顶、左右声道相关
- 敲击/起音：时间、基频、最近音名、音分偏差、Sethares 粗糙度、临界带拍频
- 并行音：一次起音里如果有多个基频（和弦/多声部），模型会猜最多 3 个
- 结构：静默点 + 频谱新颖度（2 秒窗余弦距离）切段
- 调性（Krumhansl 模板，带置信度门限）、速度（起音包络自相关）
- 时间轴：谱重心（亮度）、粗糙度热点
- 掩蔽提示：弱成分贴着更强的成分、同临界带时标出来
- 声谱图：matplotlib 优先，没有就 ffmpeg showspectrumpic，都没有就跳过

设计取舍（都是踩过坑之后的）：

1. **基频用「[+20ms, +820ms] 窗口里 60–1500 Hz 最响的峰」+ 抛物线插值**。
   倒谱会被钟声的非整数倍分音（1.00/2.00/2.76/4.07/5.41/8.13）带到泛音上，
   自相关又容易低八度；在这类素材上「最响的峰」反而最稳。
2. **并行音只报有把握的**：候选基频要至少被 2 个谱峰当整数倍支持（1.5% 容差），
   而且是已选基频的整数倍时直接当泛音扔掉。所以它不会把钟的八度分音报成第二个音，
   但真正的和弦（纯四/五度、三度）能报出来；密集织体仍然可能漏。
3. **回声去重只有 0.2–0.8 秒内、电平更低、音高相近才算回声**。
   窗口再放大会误伤加速段里真正的连续敲击。
4. **安静段（低于峰值 45 dB）不算粗糙度/亮度**。否则淡出尾巴里的噪声会被报成
   「粗糙度 0.8」，看着吓人，其实是零信号。
5. **起音电平门往后看 3 帧**：mp3/aac 的有损编码会把起音糊开几帧，
   用同一帧判断会把整首 MP3 的起音全漏掉。
6. **输入格式不挑**：WAV 直接用标准库读（8/16/24/32 bit），其它格式交给 ffmpeg。
7. **依赖只有 numpy**；ffmpeg / matplotlib 是可选的。

已知局限：音高是「最响的峰」而不是真正的多基频估计，密集织体只报最响的那几个；
真实混响素材系统性偏高约 +1.5 音分（纯合成素材在 ±3 音分内）；结构分段是声学变化点，
不是曲式分析；粗糙度是 Sethares 模型的相对值，不是听感。

用法:
    python3 analyze.py <音频文件> [输出目录] [--json] [--no-spectrogram]
输出:
    默认：Markdown 报告（stdout + `<名字>-声学分析报告.md`）+ 声谱图 PNG（有工具时）
    --json：stdout 只打一份 JSON（同时仍写 Markdown 报告），路径在 JSON 的 paths 里
"""
import argparse
import json
import os
import subprocess
import sys
import wave

import numpy as np

VERSION = "1.1.0"
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SR_TARGET = 32000


# ---------------------------------------------------------------- 读文件

def _read_wav(path):
    with wave.open(path, "rb") as w:
        ch, sw, sr, nf = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(nf)
    if sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128) / 128
    elif sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v & 0x800000, v - (1 << 24), v)
        x = v.astype(np.float64) / (1 << 23)
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float64) / (1 << 31)
    else:
        raise ValueError(f"不支持的采样位宽: {sw*8}bit")
    return x.reshape(-1, ch), sr


def _read_ffmpeg(path, sr=SR_TARGET):
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True,
    )
    if p.returncode != 0:
        raise RuntimeError("ffmpeg 解码失败：" + (p.stderr.decode("utf-8", "ignore")[:200]))
    x = np.frombuffer(p.stdout, dtype=np.float32).astype(np.float64)
    return x.reshape(-1, 1), sr


def load_audio(path):
    """返回 (多声道数组, 采样率)。WAV 用 wave，其它交给 ffmpeg。"""
    is_wav = False
    try:
        with open(path, "rb") as f:
            head = f.read(12)
        is_wav = head[:4] == b"RIFF" and head[8:12] == b"WAVE"
    except OSError:
        is_wav = False
    if is_wav:
        return _read_wav(path)
    try:
        return _read_ffmpeg(path)
    except (RuntimeError, FileNotFoundError):
        raise RuntimeError(
            f"不是 WAV，且这台机器没有可用的 ffmpeg 转码（{os.path.basename(path)}）。"
            "要么给 WAV，要么先装 ffmpeg。"
        )


# ---------------------------------------------------------------- 基础测量

def frame_rms(x, sr, hop_s=0.01, win_s=0.025):
    hop, win = max(1, int(hop_s * sr)), max(2, int(win_s * sr))
    if len(x) < win:
        return np.array([np.sqrt(np.mean(x**2) + 1e-20)]), hop
    cs = np.concatenate([[0.0], np.cumsum(x * x)])
    starts = np.arange(0, len(x) - win, hop)
    rms = np.sqrt(np.maximum(0, cs[starts + win] - cs[starts]) / win)
    return rms, hop


def spectrum_peaks(seg, sr, max_peaks=10, floor=0.04):
    n = len(seg)
    if n < 128:
        return []
    mag = np.abs(np.fft.rfft(seg * np.hanning(n), 1 << int(np.ceil(np.log2(n * 4)))))
    nfft = (len(mag) - 1) * 2
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    thr = mag.max() * floor
    idx = np.where((mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:]) & (mag[1:-1] > thr))[0] + 1
    peaks = []
    for i in idx:
        if not (40 <= freqs[i] <= 8000):
            continue
        den = mag[i - 1] - 2 * mag[i] + mag[i + 1]
        delta = 0.5 * (mag[i - 1] - mag[i + 1]) / den if abs(den) > 1e-12 else 0.0
        peaks.append(((i + delta) * sr / nfft, float(mag[i])))
    peaks.sort(key=lambda p: -p[1])
    return peaks[:max_peaks]


def estimate_f0(seg, sr, lo=55.0, hi=1500.0):
    """基频估计：0.8 秒窗口里最响的那个低频峰。

    为什么不用倒谱/自相关：钟声是非整数倍分音（1.00/2.00/2.76/4.07/5.41/8.13），
    倒谱会被分音间距带到泛音上，自相关又容易低八度。实测这套素材里，
    「60–1500Hz 里最响的峰」最稳。
    """
    seg = seg - np.mean(seg)
    n = len(seg)
    if n < 512:
        return None
    nfft = 1 << int(np.ceil(np.log2(n * 8)))
    mag = np.abs(np.fft.rfft(seg * np.hanning(n), nfft))
    fr = np.fft.rfftfreq(nfft, 1 / sr)
    band = (fr >= lo) & (fr <= hi)
    if not band.any():
        return None
    idx_band = np.where(band)[0]
    i = idx_band[int(np.argmax(mag[band]))]
    if 0 < i < len(mag) - 1:
        den = mag[i - 1] - 2 * mag[i] + mag[i + 1]
        d = 0.5 * (mag[i - 1] - mag[i + 1]) / den if abs(den) > 1e-12 else 0.0
        f0 = (i + d) * sr / nfft
    else:
        f0 = fr[i]
    return float(f0) if lo <= f0 <= hi else None


def _interval_ok(f, g, tol_cents=25.0):
    """f 和 g 的间距是不是一个「像和弦成员」的音程（±25 音分内落在半音格上，
    并且是半音阶上的 2/3/4/5/7/8/9 度——八度会被前面的整数倍判据挡掉）。"""
    c = 1200.0 * np.log2(f / g) % 1200.0
    st = int(round(c / 100.0))
    return abs(c - st * 100.0) <= tol_cents and st % 12 in (2, 3, 4, 5, 7, 8, 9)


def voices_at(seg, sr, max_voices=3):
    """估这一段里有哪几个并行的基频（单声部时只有一个）。

    做法：取谱峰，把 55–1200 Hz 的峰逐个当作候选基频；候选要至少被 2 个峰
    当作整数倍支持（1.5% 容差）；已经选中的基频的整数倍（八度/泛音）直接扔掉，
    免得把钟的第二个分音报成第二个声部。最多 max_voices 个，按响度排序。
    """
    peaks = spectrum_peaks(seg, sr, max_peaks=24, floor=0.02)
    if not peaks:
        return []
    amax = peaks[0][1]
    cands = []
    for f, a in peaks:
        if not (55.0 <= f <= 1200.0):
            continue
        n, support = 0, 0.0
        for g, aa in peaks:
            k = g / f
            if k < 1 or k > 8.2:
                continue
            if abs(k - round(k)) <= 0.015:
                n += 1
                support += aa
        if n >= 2:
            cands.append((f, a, support))
    cands.sort(key=lambda c: -c[1])
    strong = [(g, aa) for g, aa in peaks if aa > 0.10 * amax]
    chosen = []
    for f, a, _support in cands:
        if a < 0.10 * amax:
            break
        if any(abs(1200.0 * np.log2(f / g)) < 60.0 for g in chosen):
            continue  # 同音
        if any(abs(f / g - round(f / g)) <= 0.015 and round(f / g) >= 2 for g in chosen):
            continue  # 是已选基频的泛音
        # 上泛音判据：如果 f/k（k=2..4）附近有强峰，说明 f 更像某个低音的泛音。
        # 非整数倍分音（钟的 2.76/4.07/…）的子谐波和真分音对不齐，所以这条能把它挡掉。
        if any(abs(f / k - g) / g < 0.02 for k in (2, 3, 4) for g, _aa in strong if g < f):
            continue
        # 音程判据：只报和已选基频构成常见音程（±25 音分内的大二度到六度）的声部。
        # 钟的 2.76/4.07 分音和基频差 557/2440 音分，落不到半音格上，正好被挡掉。
        if chosen and not any(_interval_ok(f, g) for g in chosen):
            continue
        chosen.append(f)
        if len(chosen) >= max_voices:
            break
    return chosen


def note_of(f):
    midi = 69 + 12 * np.log2(f / 440.0)
    m = int(round(midi))
    cents = (midi - m) * 100
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}", cents


def roughness(peaks):
    """Sethares 成对不协和叠加（相对值）"""
    ps = peaks[:6]
    if len(ps) < 2:
        return 0.0
    amax = max(a for _, a in ps)
    r = 0.0
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            f1, a1 = ps[i]
            f2, a2 = ps[j]
            fmin = min(f1, f2)
            df = abs(f1 - f2)
            s = 0.24 / (0.0207 * fmin + 18.96)
            d = np.exp(-3.5 * s * df) - np.exp(-5.75 * s * df)
            r += (a1 / amax) * (a2 / amax) * max(0.0, d)
    return float(r)


def beat_pairs(peaks):
    out = []
    ps = peaks[:6]
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            f1, a1 = ps[i]
            f2, a2 = ps[j]
            fmin = min(f1, f2)
            df = abs(f1 - f2)
            cb = 100.0 if fmin < 500 else 0.2 * fmin
            if 0.5 < df < 30 and df < cb:
                out.append((f1, f2, df, min(a1, a2) / max(a1, a2)))
    return out


def harmonic_score(peaks):
    if len(peaks) < 3:
        return False
    amax = peaks[0][1]
    tot = sum(a for _, a in peaks)
    for f0, a in peaks:
        if f0 > 1500 or a < 0.2 * amax:
            continue
        n = 0
        ea = 0.0
        for f, aa in peaks:
            k = f / f0
            if abs(k - round(k)) < 0.02 and round(k) >= 1:
                n += 1
                ea += aa
        if n >= 3 and ea >= 0.35 * tot:
            return True
    return False


def is_tonal(seg, sr, peaks):
    """这个片段像不像一个有音高的音。

    两个门：谐波列（至少 3 个峰落在同一个基频的整数倍上、且占多数能量），
    或频谱集中度（最响 6 个 bin 占 40–6000Hz 总能量 35% 以上——纯音/钟声都满足）。
    第二个门是给「只有一个明显峰」的纯净单音留的：它排不出谐波列，但显然有音高。
    """
    if not peaks:
        return False
    if len(peaks) >= 2 and harmonic_score(peaks):
        return True
    mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    fr = np.fft.rfftfreq(len(seg), 1 / sr)
    m = np.sort(mag[(fr > 40) & (fr < 6000)])[::-1]
    tot = m.sum()
    return bool(tot > 0 and m[:6].sum() / tot > 0.35)


def dedupe_echoes(events, min_dt=0.2, max_dt=0.8, max_cents=60.0):
    """把「同一击的回声/拖尾」从起音列表里去掉。

    判据：距上一个保留事件 0.2–0.8 秒、电平明显更低（+1 dB 容差）、
    音高相差小于 60 音分。返回 (保留的事件, 剔除个数)。
    窗口不能放宽：加速段里真正连续的敲击间隔会掉进 1.5 秒窗口。
    """
    kept, n_echo = [], 0
    for e in events:
        dup = False
        for p in kept[-8:]:
            dt = e["t"] - p["t"]
            if min_dt < dt < max_dt and e["level"] < p["level"] + 1.0:
                if abs(1200 * np.log2(max(e["f0"], 1e-9) / max(p["f0"], 1e-9))) < max_cents:
                    dup = True
                    break
        if dup:
            n_echo += 1
        else:
            kept.append(e)
    return kept, n_echo


def find_onsets(env_db, hop, sr):
    flux = np.maximum(0.0, np.diff(env_db, prepend=env_db[0]))
    flux = np.convolve(flux, np.ones(5) / 5, mode="same")
    pos = flux[flux > 0]
    if len(pos) == 0:
        return []
    # 阈值用 p90 而不是 p99：一记很响的起音会把 p99 拉高，
    # 后面的轻起音（比如前一击的拖尾里又敲一下）就全被压掉了。
    thr = max(0.7, float(np.percentile(pos, 90)) * 0.6)
    ref = env_db.max()
    gate = ref - 45
    cand = []
    n = len(flux)
    for i in range(1, n - 1):
        # 电平门往后看 3 帧：mp3/aac 的有损编码会把起音糊开几帧，
        # 通量峰值出现时电平还没升上来，用同一帧判断会把整首的起音全漏掉。
        j = min(i + 3, n - 1)
        if flux[i] > thr and flux[i] >= flux[i - 1] and flux[i] > flux[i + 1] and env_db[j] > gate:
            cand.append(i)
    min_gap = int(0.18 * sr / hop)
    picked = []
    for i in cand:
        if picked and i - picked[-1] < min_gap:
            if flux[i] > flux[picked[-1]]:
                picked[-1] = i
        else:
            picked.append(i)
    return [i * hop / sr for i in picked]


# ---------------------------------------------------------------- 结构 / 调性 / 速度

def silence_bounds(env_db, hop, sr, total, drop=45, min_len=1.2):
    ref = env_db.max()
    quiet = env_db < ref - drop
    runs, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if (j - i) * hop / sr >= min_len:
                runs.append((i * hop / sr, j * hop / sr))
            i = j
        else:
            i += 1
    return [0.0] + [(a + b) / 2 for a, b in runs] + [total], runs


def novelty_bounds(mono, sr, total, win_s=2.0, min_gap=8.0):
    win = int(win_s * sr)
    blocks = []
    for i0 in range(0, len(mono) - win, win):
        seg = mono[i0:i0 + win]
        blocks.append(np.abs(np.fft.rfft(seg * np.hanning(win))))
    if len(blocks) < 4:
        return []
    nov = []
    for k in range(1, len(blocks)):
        a, b = blocks[k - 1], blocks[k]
        nov.append(1 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    nov = np.array(nov)
    cut = nov.mean() + 1.2 * nov.std()
    cand = [(k + 1) * win_s for k in np.where(nov > cut)[0]]
    picked = []
    for t in cand:
        if not picked or t - picked[-1] >= min_gap:
            picked.append(t)
    return picked


def estimate_key(mono, sr):
    nfft = 4096
    hop = 2048
    pc = np.zeros(12)
    win = np.hanning(nfft)
    for i0 in range(0, max(1, len(mono) - nfft), hop):
        mag = np.abs(np.fft.rfft(mono[i0:i0 + nfft] * win))
        fr = np.fft.rfftfreq(nfft, 1 / sr)
        for i in range(1, nfft // 2):
            if 60 <= fr[i] <= 2000:
                midi = int(round(69 + 12 * np.log2(fr[i] / 440.0)))
                pc[midi % 12] += mag[i]
    if pc.sum() <= 1e-9:  # 静音/近乎静音：没有可用的调性证据
        return 0.0, "", ""
    pc = pc / pc.sum()
    MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    best = (0.0, "", "")
    for r in range(12):
        maj = np.corrcoef(pc, np.roll(MAJ, r))[0, 1]
        mi = np.corrcoef(pc, np.roll(MIN, r))[0, 1]
        if not np.isfinite(maj):
            maj = 0.0
        if not np.isfinite(mi):
            mi = 0.0
        if maj > best[0]:
            best = (maj, NOTE_NAMES[r], "大调")
        if mi > best[0]:
            best = (mi, NOTE_NAMES[r], "小调")
    return best


def estimate_tempo(env_db, hop, sr):
    if float(np.ptp(env_db)) < 3.0:  # 没有动态可言（静音恒噪底）就别硬猜速度
        return None
    env = env_db - env_db.mean()
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    fps = sr / hop
    lo, hi = int(fps * 60 / 200), int(fps * 60 / 55)
    if hi >= len(ac) or lo >= hi:
        return None
    lag = lo + int(np.argmax(ac[lo:hi]))
    return 60 * fps / lag


# ---------------------------------------------------------------- 主分析

def analyze(path, outdir=None, want_spectrogram=True):
    """跑完整分析，返回结果 dict（JSON 可序列化）。"""
    outdir = outdir or os.path.dirname(os.path.abspath(path))
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]

    x, sr = load_audio(path)

    total = len(x) / sr
    truncated = False
    if total > 900:
        x = x[:int(900 * sr)]
        total = 900.0
        truncated = True
    ch = x.shape[1]
    mono = x.mean(axis=1) if ch > 1 else x[:, 0]

    rms, hop = frame_rms(mono, sr)
    env_db = 20 * np.log10(rms + 1e-9)
    ref = env_db.max()
    peak = float(np.max(np.abs(x)))
    full_rms = float(np.sqrt(np.mean(mono**2)) + 1e-12)
    clip_n = int(np.sum(np.abs(x) >= 0.999))

    onsets = find_onsets(env_db, hop, sr)

    events = []
    for t in onsets:
        i0 = int((t + 0.02) * sr)
        i1 = min(i0 + int(0.8 * sr), len(mono))
        if i1 - i0 < 2048:
            continue
        seg = mono[i0:i1]
        if np.sqrt(np.mean(seg**2)) < 10 ** ((ref - 45) / 20):
            continue
        f0 = estimate_f0(seg, sr)
        if f0 is None:
            continue
        pk = spectrum_peaks(seg, sr)
        if not pk or not is_tonal(seg, sr, pk):
            continue
        name, cents = note_of(f0)
        lv = 20 * np.log10(np.sqrt(np.mean(seg**2)) + 1e-12)
        voices = voices_at(seg, sr)
        if not any(abs(1200 * np.log2(f0 / f)) < 50 for f in voices):
            voices = [f0] + [f for f in voices if f < 0.9 * f0 or f > 1.1 * f0]
        voice_info = []
        for f in voices[:3]:
            n, c = note_of(f)
            voice_info.append({"f0_hz": round(float(f), 2), "note": n, "cents": round(float(c), 1)})
        events.append(dict(
            t=round(float(t), 3), f0=float(f0), name=name, cents=float(cents), level=float(lv),
            rough=roughness(pk),
            beats=[{"f1_hz": round(f1, 1), "f2_hz": round(f2, 1),
                    "delta_hz": round(df, 2), "amp_ratio": round(ratio, 2)}
                   for f1, f2, df, ratio in beat_pairs(pk)[:5]],
            peaks=pk, voices=voice_info,
        ))

    # 回声去重：0.2–0.8s 内、电平更低、音高相近 → 判为同一击的回声
    events, n_echo = dedupe_echoes(events)

    # 时间轴：亮度 + 粗糙度（安静段按 0 处理，不再把噪声当信号）
    step = win_n = int(0.5 * sr)
    tl_t, tl_c, tl_r = [], [], []
    gate = ref - 45
    for i0 in range(0, max(1, len(mono) - win_n), step):
        seg = mono[i0:i0 + win_n]
        if len(seg) < 512:
            break
        lv = 20 * np.log10(np.sqrt(np.mean(seg**2)) + 1e-12)
        mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        fr = np.fft.rfftfreq(len(seg), 1 / sr)
        band = (fr > 40) & (fr < 8000)
        m = mag[band]
        cen = float(np.sum(fr[band] * m) / (np.sum(m) + 1e-12))
        tl_t.append(i0 / sr)
        tl_c.append(cen if lv > gate else np.nan)
        if lv > gate:
            pk6 = spectrum_peaks(seg, sr, 6)
            tl_r.append(roughness(pk6) if is_tonal(seg, sr, pk6) else 0.0)
        else:
            tl_r.append(np.nan)
    tl_t = np.array(tl_t)
    tl_c = np.array(tl_c)
    tl_r = np.array(tl_r)

    sil_bounds, sil_runs = silence_bounds(env_db, hop, sr, total)
    nov_bounds = novelty_bounds(mono, sr, total)
    bounds = sorted(set([0.0, total] + [b for b in sil_bounds if 0 < b < total]
                        + [b for b in nov_bounds if 0 < b < total]))
    secs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        ia, ib = int(a * sr / hop), max(int(b * sr / hop), int(a * sr / hop) + 1)
        seg = env_db[ia:min(ib, len(env_db))]
        secs.append({"start": round(a, 2), "end": round(b, 2),
                     "mean_rel_db": round(float(np.mean(seg)) - ref, 1) if len(seg) else -99.0,
                     "active_ratio": round(float(np.mean(seg > ref - 35)), 3) if len(seg) else 0.0})

    key_conf, key_root, key_mode = estimate_key(mono, sr)
    bpm = estimate_tempo(env_db, hop, sr)

    # 粗糙度热点（时间轴前 5 名，按时间顺序铺开）
    valid = ~np.isnan(tl_r)
    hotspots = []
    if valid.any():
        order = np.argsort(-np.where(valid, tl_r, -1))[:5]
        for i in sorted(order):
            hotspots.append({"t": round(float(tl_t[i]), 1), "roughness": round(float(tl_r[i]), 3),
                             "brightness_hz": None if np.isnan(tl_c[i]) else round(float(tl_c[i]))})

    # 掩蔽：弱成分贴着更强的成分、同临界带
    masking = []
    for e in events:
        pk = e["peaks"]
        if len(pk) < 2:
            continue
        amax = pk[0][1]
        for f, a in pk[1:]:
            if a < 0.15 * amax:
                for g, b in pk:
                    if g < f and f - g > 15 and b > 6 * a:
                        cb = 100.0 if g < 500 else 0.2 * g
                        if f - g < cb:
                            masking.append({"t": round(e["t"], 2), "weak_hz": round(f),
                                            "strong_hz": round(g)})
                            break
                if len(masking) >= 6:
                    break
        if len(masking) >= 6:
            break

    # 亮度
    brightness = None
    if (~np.isnan(tl_c)).any():
        c = tl_c[~np.isnan(tl_c)]
        brightness = {
            "median_hz": round(float(np.median(c))),
            "p10_hz": round(float(np.percentile(c, 10))),
            "p90_hz": round(float(np.percentile(c, 90))),
            "brightest_s": round(float(tl_t[np.nanargmax(tl_c)]), 1),
            "darkest_s": round(float(tl_t[np.nanargmin(tl_c)]), 1),
            "brightest_hz": round(float(np.nanmax(tl_c))),
            "darkest_hz": round(float(np.nanmin(tl_c))),
        }

    # 起音间隔趋势（用原始起音，不用筛过音高的）
    gaps_info = None
    raw = [onsets[0]] if onsets else []
    for t in onsets[1:]:
        if t - raw[-1] >= 0.35:
            raw.append(t)
    gaps = np.array([b - a for a, b in zip(raw[:-1], raw[1:]) if b - a < 10.0])
    if len(gaps) >= 3:
        h1, h2 = gaps[: len(gaps) // 2], gaps[len(gaps) // 2:]
        trend = ("趋于变密（加速感）" if np.median(h2) < np.median(h1) * 0.8
                 else "趋于变疏（减速感）" if np.median(h2) > np.median(h1) * 1.25
                 else "大体均匀")
        gaps_info = {"median_s": round(float(np.median(gaps)), 2),
                     "first_half_median_s": round(float(np.median(h1)), 2),
                     "second_half_median_s": round(float(np.median(h2)), 2),
                     "trend": trend}

    res = {
        "tool": "audio-ears",
        "version": VERSION,
        "file": os.path.abspath(path),
        "basename": base,
        "duration_s": round(total, 2),
        "sample_rate": sr,
        "channels": ch,
        "truncated_at_15min": truncated,
        "basic": {
            "peak_dbfs": round(20 * np.log10(peak + 1e-12), 1),
            "rms_dbfs": round(20 * np.log10(full_rms), 1),
            "clip_samples": clip_n,
        },
        "key": None if key_conf < 0.45 else {
            "root": key_root, "mode": key_mode, "confidence": round(float(key_conf), 2)},
        "key_low_confidence": None if key_conf >= 0.45 else round(float(key_conf), 2),
        "tempo_bpm": round(float(bpm), 1) if bpm else None,
        "sections": secs,
        "silence_runs": [[round(a, 2), round(b, 2)] for a, b in sil_runs],
        "novelty_points": [round(float(t), 1) for t in nov_bounds],
        "events": events,
        "echoes_removed": n_echo,
        "onset_gaps": gaps_info,
        "timeline": {
            "step_s": 0.5,
            "t": [round(float(v), 2) for v in tl_t],
            "brightness_hz": [None if np.isnan(v) else round(float(v)) for v in tl_c],
            "roughness": [None if np.isnan(v) else round(float(v), 3) for v in tl_r],
        },
        "roughness_hotspots": hotspots,
        "masking": masking,
        "brightness": brightness,
        "paths": {"md": os.path.join(outdir, f"{base}-声学分析报告.md"),
                  "json": os.path.join(outdir, f"{base}-分析.json"),
                  "spectrogram": os.path.join(outdir, f"{base}-声谱图.png")},
    }
    if ch == 2:
        lr = np.sqrt(np.mean(x**2, axis=0))
        res["basic"]["lr_rms_dbfs"] = [round(20 * np.log10(lr[0] + 1e-12), 1),
                                       round(20 * np.log10(lr[1] + 1e-12), 1)]
        res["basic"]["lr_correlation"] = round(float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]), 2)

    if want_spectrogram:
        res["paths"]["spectrogram"] = _make_spectrogram(path, mono, sr, total, env_db, hop, tl_t, tl_r, res["paths"]["spectrogram"])
    else:
        res["paths"]["spectrogram"] = None
    return res


def _make_spectrogram(path, mono, sr, total, env_db, hop, tl_t, tl_r, png_path):
    """声谱图：matplotlib 优先，没有就 ffmpeg showspectrumpic，都没有就不出图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        win_n, hop_n = 1024, 512
        frames = []
        w = np.hanning(win_n)
        for i0 in range(0, len(mono) - win_n, hop_n):
            frames.append(np.abs(np.fft.rfft(mono[i0:i0 + win_n] * w)))
        S = 20 * np.log10(np.array(frames).T + 1e-9)
        fr = np.fft.rfftfreq(win_n, 1 / sr)
        keep = fr <= 5000
        ref = env_db.max()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                       gridspec_kw={"height_ratios": [2.2, 1]})
        ax1.imshow(S[keep], origin="lower", aspect="auto", cmap="magma",
                   extent=[0, total, 0, 5000], vmin=S.max() - 70, vmax=S.max())
        ax1.set_ylabel("Freq (Hz)")
        ax1.set_title("Spectrogram")
        tt = np.arange(len(env_db)) * hop / sr
        ax2.plot(tt, env_db - ref, lw=0.8, color="tab:blue")
        ax2.set_ylabel("Level (dB rel.)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylim(-70, 3)
        ax3 = ax2.twinx()
        ax3.plot(tl_t, np.nan_to_num(tl_r, nan=0.0), lw=1.2, color="tab:red")
        ax3.set_ylabel("Roughness", color="tab:red")
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        return png_path
    except Exception:
        try:
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-lavfi",
                            "showspectrumpic=s=1200x600:legend=1:color=intensity", png_path],
                           check=True)
            return png_path
        except Exception:
            return None


# ---------------------------------------------------------------- 报告渲染

def render_report(res):
    L = []
    L.append(f"# 声学分析报告 · {res['basename']}\n")
    L.append("> 物理测量与心理声学模型推断，不是真实听感。听感结论以耳朵为准。\n")
    L.append("## 基本信息\n")
    L.append(f"- 时长 {res['duration_s']:.1f}s | 采样率 {res['sample_rate']}Hz | 声道 "
             f"{'立体声' if res['channels'] == 2 else '单声道' if res['channels'] == 1 else str(res['channels'])}")
    b = res["basic"]
    L.append(f"- 峰值 {b['peak_dbfs']:.1f} dBFS | 整体 RMS {b['rms_dbfs']:.1f} dBFS | 削顶样本 {b['clip_samples']}")
    if res["channels"] == 2 and "lr_correlation" in b:
        L.append(f"- 左/右 RMS: {b['lr_rms_dbfs'][0]:.1f} / {b['lr_rms_dbfs'][1]:.1f} dBFS | "
                 f"声道相关 {b['lr_correlation']:.2f}（越低越宽）")
    if res["key"]:
        L.append(f"- 调性(模板推测) {res['key']['root']} {res['key']['mode']}，吻合度 {res['key']['confidence']:.2f}")
    else:
        L.append(f"- 调性(模板推测) 置信度低 ({res['key_low_confidence']:.2f})，跳过")
    if res["tempo_bpm"]:
        L.append(f"- 速度(起音包络推测) {res['tempo_bpm']:.0f} BPM（半速 {res['tempo_bpm']/2:.0f} / 双速 {res['tempo_bpm']*2:.0f}）")
    if res["truncated_at_15min"]:
        L.append("- ⚠️ 超过 15 分钟，仅分析前 15 分钟")

    L.append("\n## 结构分段（静默点 + 频谱新颖度）\n")
    L.append("| 起 | 止 | 平均相对响度 | 活跃占比 |")
    L.append("|---|---|---|---|")
    for s in res["sections"]:
        L.append(f"| {s['start']:.1f}s | {s['end']:.1f}s | {s['mean_rel_db']:.1f} dB | {s['active_ratio']*100:.0f}% |")
    if res["silence_runs"]:
        L.append(f"\n- 检测到 {len(res['silence_runs'])} 段 ≥1.2 秒的静默")
    L.append(f"- 频谱变化点（推测）: " + (
        ", ".join(f"{t:.1f}s" for t in res["novelty_points"]) if res["novelty_points"] else "没测到明显变化"))

    L.append(f"\n## 敲击/起音点（共 {len(res['events'])} 个，已剔除 {res['echoes_removed']} 个疑似回声）\n")
    if res["events"]:
        L.append("| 时间 | 基频(峰值) | 最近音名 | 偏差(音分) | 粗糙度 |")
        L.append("|---|---|---|---|---|")
        for e in res["events"]:
            L.append(f"| {e['t']:.2f}s | {e['f0']:.1f}Hz | {e['name']} | {e['cents']:+.0f} | {e['rough']:.3f} |")
        g = res["onset_gaps"]
        if g:
            L.append(f"\n- 起音间隔中位数 {g['median_s']:.2f}s，前半 {g['first_half_median_s']:.2f}s → "
                     f"后半 {g['second_half_median_s']:.2f}s：{g['trend']}")
        poly = [e for e in res["events"] if len(e["voices"]) > 1]
        if poly:
            L.append("\n- 并行音（模型推测，最多 3 个）：")
            for e in poly[:6]:
                parts = " + ".join(f"{v['note']} {v['f0_hz']:.1f}Hz" for v in e["voices"])
                L.append(f"  - {e['t']:.2f}s → {parts}")
    else:
        L.append("未检测到明显起音（可能是持续音或电平过低）。")
    L.append("\n注：基频取 [+20ms, +820ms] 窗口内 60–1500Hz 最响的峰（抛物线插值）；纯合成素材误差在 ±3 音分内，"
             "真实混响素材实测系统性偏高约 +1.5 音分——同一首里比较相对偏差比看绝对值可靠。"
             "并行音是模型猜的：同时敲的音只保证报出最响的那个，多声部可能漏。")

    L.append("\n## 拍频与粗糙度热点\n")
    if res["roughness_hotspots"]:
        L.append("| 时间 | 粗糙度 | 谱重心(亮度) |")
        L.append("|---|---|---|")
        for h in res["roughness_hotspots"]:
            cen = "—" if h["brightness_hz"] is None else f"{h['brightness_hz']:.0f}Hz"
            L.append(f"| {h['t']:.1f}s | {h['roughness']:.3f} | {cen} |")
        hot = max(res["events"], key=lambda e: e["rough"]) if res["events"] else None
        if hot and hot["beats"]:
            L.append(f"\n最粗糙起音 {hot['t']:.2f}s 处的临界带近频对（拍频来源）：")
            for bp in hot["beats"]:
                L.append(f"- {bp['f1_hz']:.1f}Hz × {bp['f2_hz']:.1f}Hz → 拍频约 {bp['delta_hz']:.1f}Hz"
                         f"（幅度比 {bp['amp_ratio']:.2f}）")
        L.append("\n注：粗糙度是 Sethares 模型的相对值；1–6Hz 拍频像晃动，8–30Hz 进入粗糙区。")
    else:
        L.append("没有可用的响段（整首太安静？），跳过。")

    L.append("\n## 掩蔽提示\n")
    if res["masking"]:
        for m in res["masking"]:
            L.append(f"- {m['t']:.2f}s：{m['weak_hz']}Hz 成分较弱且贴近更强的 {m['strong_hz']}Hz（同临界带），可能被掩蔽")
    else:
        L.append("未发现明显的弱成分掩蔽风险。")

    L.append("\n## 亮度（谱重心）\n")
    br = res["brightness"]
    if br:
        L.append(f"- 响段中位 {br['median_hz']}Hz，范围 {br['p10_hz']}–{br['p90_hz']}Hz")
        L.append(f"- 最亮 {br['brightest_s']:.1f}s（{br['brightest_hz']}Hz），最暗 {br['darkest_s']:.1f}s（{br['darkest_hz']}Hz）")
    else:
        L.append("没有可用的响段。")

    L.append("\n---\n")
    L.append(f"版本 {res['version']} · `--json` 可拿结构化结果")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="analyze.py",
        description="音频分析：测量、音高/音分、粗糙度、掩蔽、分段、声谱图（不假装听到了）。",
    )
    ap.add_argument("audio", help="音频文件（wav/mp3/m4a/flac…）")
    ap.add_argument("outdir", nargs="?", help="输出目录，默认和音频同目录")
    ap.add_argument("--json", action="store_true", help="stdout 输出 JSON（仍写 Markdown 报告）")
    ap.add_argument("--no-spectrogram", action="store_true", help="不生成声谱图")
    ap.add_argument("--version", action="version", version=f"audio-ears {VERSION}")
    args = ap.parse_args(argv)

    if not os.path.exists(args.audio):
        print(f"找不到文件: {args.audio}")
        return 1
    try:
        res = analyze(args.audio, args.outdir, want_spectrogram=not args.no_spectrogram)
    except Exception as ex:
        print(f"读取失败：{ex}")
        return 1

    md = render_report(res)
    with open(res["paths"]["md"], "w", encoding="utf-8") as f:
        f.write(md)
    with open(res["paths"]["json"], "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    if args.json:
        res["paths"]["md"] = os.path.abspath(res["paths"]["md"])
        res["paths"]["json"] = os.path.abspath(res["paths"]["json"])
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    print(md)
    print(f"MEDIA:{res['paths']['md']}")
    if res["paths"]["spectrogram"]:
        print(f"MEDIA:{res['paths']['spectrogram']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
