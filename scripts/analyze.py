# -*- coding: utf-8 -*-
"""声学分析：给 agent 用的一只「金耳朵」。

对一段音频做可复现的测量，输出一份 Markdown 报告：

- 基本：时长、峰值、RMS、削顶、左右声道相关
- 敲击/起音：时间、基频、最近音名、音分偏差、Sethares 粗糙度、临界带拍频
- 结构：静默点 + 频谱新颖度（2 秒窗余弦距离）切段
- 调性（Krumhansl 模板，带置信度门限）、速度（起音包络自相关）
- 时间轴：谱重心（亮度）、粗糙度热点
- 掩蔽提示：弱成分贴着更强的成分、同临界带时标出来
- 声谱图：matplotlib 优先，没有就 ffmpeg showspectrumpic，都没有就跳过

设计取舍（都是踩过坑之后的）：

1. **基频用「[+20ms, +820ms] 窗口里 60–1500 Hz 最响的峰」+ 抛物线插值**。
   倒谱会被钟声的非整数倍分音（1.00/2.00/2.76/4.07/5.41/8.13）带到泛音上，
   自相关又容易低八度；在这类素材上「最响的峰」反而最稳。
   代价：同时敲的音只报最响的那个（单声部假设），密集织体会漏音。
2. **回声去重只有 0.2–0.8 秒内、电平更低、音高相近才算回声**。
   窗口再放大会误伤加速段里真正的连续敲击。
3. **安静段（低于峰值 45 dB）不算粗糙度/亮度**。否则淡出尾巴里的噪声会被报成
   「粗糙度 0.8」，看着吓人，其实是零信号。
4. **输入格式不挑**：WAV 直接用标准库读，其它格式交给 ffmpeg；没有 ffmpeg 就直说。
5. **依赖只有 numpy**；ffmpeg / matplotlib 是可选的。

已知局限：单声部音高假设；真实混响素材系统性偏高约 +1.5 音分（纯合成素材在 ±3 音分内）；
结构分段是声学变化点，不是曲式分析；粗糙度是 Sethares 模型的相对值，不是听感。

用法: python3 analyze.py <音频文件> [输出目录]
输出: Markdown 报告（stdout + 文件）+ 声谱图 PNG（有工具时）；最后打 MEDIA: 行。
"""
import os
import subprocess
import sys
import wave

import numpy as np

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


def spectrum_peaks(seg, sr, max_peaks=10):
    n = len(seg)
    if n < 128:
        return []
    mag = np.abs(np.fft.rfft(seg * np.hanning(n), 1 << int(np.ceil(np.log2(n * 4)))))
    nfft = (len(mag) - 1) * 2
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    thr = mag.max() * 0.04
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
    「60–1500Hz 里最响的峰」最稳：干净 E4 报 330.3Hz（+2音分），窄狼报 325.7Hz（+1音分）。
    起音瞬间的噪声会把峰拉偏（0.32 秒窗时能偏到 −6 音分），所以窗口拉到 0.8 秒。
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
    pc = pc / (pc.sum() + 1e-12)
    MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    best = (-9, "", "")
    for r in range(12):
        maj = np.corrcoef(pc, np.roll(MAJ, r))[0, 1]
        mi = np.corrcoef(pc, np.roll(MIN, r))[0, 1]
        if maj > best[0]:
            best = (maj, NOTE_NAMES[r], "大调")
        if mi > best[0]:
            best = (mi, NOTE_NAMES[r], "小调")
    return best


def estimate_tempo(env_db, hop, sr):
    env = env_db - env_db.mean()
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    fps = sr / hop
    lo, hi = int(fps * 60 / 200), int(fps * 60 / 55)
    if hi >= len(ac) or lo >= hi:
        return None
    lag = lo + int(np.argmax(ac[lo:hi]))
    return 60 * fps / lag


# ---------------------------------------------------------------- 主流程

def main():
    if len(sys.argv) < 2:
        print("用法: python3 analyze.py <音频文件> [输出目录]")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"找不到文件: {path}")
        sys.exit(1)
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(path))
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]

    try:
        x, sr = load_audio(path)
    except Exception as ex:
        print(f"读取失败：{ex}")
        sys.exit(1)

    total = len(x) / sr
    truncated = False
    if total > 900:
        x = x[:int(900 * sr)]
        total = 900.0
        truncated = True
    ch = x.shape[1]
    mono = x.mean(axis=1)

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
        events.append(dict(t=t, f0=f0, name=name, cents=cents, level=lv,
                           rough=roughness(pk), beats=beat_pairs(pk), peaks=pk))

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
        secs.append((a, b, float(np.mean(seg)) - ref,
                     float(np.mean(seg > ref - 35))))

    key_conf, key_root, key_mode = estimate_key(mono, sr)
    bpm = estimate_tempo(env_db, hop, sr)

    L = []
    L.append(f"# 声学分析报告 · {base}\n")
    L.append("> 物理测量与心理声学模型推断，不是真实听感。听感结论以耳朵为准。\n")
    L.append("## 基本信息\n")
    L.append(f"- 时长 {total:.1f}s | 采样率 {sr}Hz | 声道 {'立体声' if ch == 2 else '单声道' if ch == 1 else str(ch)}")
    L.append(f"- 峰值 {20*np.log10(peak+1e-12):.1f} dBFS | 整体 RMS {20*np.log10(full_rms):.1f} dBFS | 削顶样本 {clip_n}")
    if ch == 2:
        lr = np.sqrt(np.mean(x**2, axis=0))
        corr = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])
        L.append(f"- 左/右 RMS: {20*np.log10(lr[0]+1e-12):.1f} / {20*np.log10(lr[1]+1e-12):.1f} dBFS | 声道相关 {corr:.2f}（越低越宽）")
    if key_conf >= 0.45:
        L.append(f"- 调性(模板推测) {key_root} {key_mode}，吻合度 {key_conf:.2f}")
    else:
        L.append(f"- 调性(模板推测) 置信度低 ({key_conf:.2f})，跳过")
    if bpm:
        L.append(f"- 速度(起音包络推测) {bpm:.0f} BPM（半速 {bpm/2:.0f} / 双速 {bpm*2:.0f}）")
    if truncated:
        L.append("- ⚠️ 超过 15 分钟，仅分析前 15 分钟")

    L.append("\n## 结构分段（静默点 + 频谱新颖度）\n")
    L.append("| 起 | 止 | 平均相对响度 | 活跃占比 |")
    L.append("|---|---|---|---|")
    for a, b, lv, act in secs:
        L.append(f"| {a:.1f}s | {b:.1f}s | {lv:.1f} dB | {act*100:.0f}% |")
    if sil_runs:
        L.append(f"\n- 检测到 {len(sil_runs)} 段 ≥1.2 秒的静默")
    L.append(f"- 频谱变化点（推测）: " + (", ".join(f"{t:.1f}s" for t in nov_bounds) if nov_bounds else "没测到明显变化"))

    L.append(f"\n## 敲击/起音点（共 {len(events)} 个，已剔除 {n_echo} 个疑似回声）\n")
    if events:
        L.append("| 时间 | 基频(峰值) | 最近音名 | 偏差(音分) | 粗糙度 |")
        L.append("|---|---|---|---|---|")
        for e in events:
            L.append(f"| {e['t']:.2f}s | {e['f0']:.1f}Hz | {e['name']} | {e['cents']:+.0f} | {e['rough']:.3f} |")
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
            L.append(f"\n- 起音间隔中位数 {np.median(gaps):.2f}s，前半 {np.median(h1):.2f}s → 后半 {np.median(h2):.2f}s：{trend}")
    else:
        L.append("未检测到明显起音（可能是持续音或电平过低）。")
    L.append("\n注：基频取 [+20ms, +820ms] 窗口内 60–1500Hz 最响的峰（抛物线插值）；纯合成素材误差在 ±3 音分内，真实混响素材实测系统性偏高约 +1.5 音分——同一首里比较相对偏差比看绝对值可靠。同时敲的音只报最响的那个。")

    L.append("\n## 拍频与粗糙度热点\n")
    valid = ~np.isnan(tl_r)
    if valid.any():
        order = np.argsort(-np.where(valid, tl_r, -1))[:5]
        L.append("| 时间 | 粗糙度 | 谱重心(亮度) |")
        L.append("|---|---|---|")
        for i in sorted(order):
            cen = tl_c[i]
            L.append(f"| {tl_t[i]:.1f}s | {tl_r[i]:.3f} | {'—' if np.isnan(cen) else f'{cen:.0f}Hz'} |")
        if events:
            hot = max(events, key=lambda e: e["rough"])
            if hot["beats"]:
                L.append(f"\n最粗糙起音 {hot['t']:.2f}s 处的临界带近频对（拍频来源）：")
                for f1, f2, df, ratio in hot["beats"][:5]:
                    L.append(f"- {f1:.1f}Hz × {f2:.1f}Hz → 拍频约 {df:.1f}Hz（幅度比 {ratio:.2f}）")
        L.append("\n注：粗糙度是 Sethares 模型的相对值；1–6Hz 拍频像晃动，8–30Hz 进入粗糙区。")
    else:
        L.append("没有可用的响段（整首太安静？），跳过。")

    L.append("\n## 掩蔽提示\n")
    masked = []
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
                            masked.append((e["t"], f, g))
                            break
    if masked:
        for t, f, g in masked[:6]:
            L.append(f"- {t:.2f}s：{f:.0f}Hz 成分较弱且贴近更强的 {g:.0f}Hz（同临界带），可能被掩蔽")
    else:
        L.append("未发现明显的弱成分掩蔽风险。")

    L.append("\n## 亮度（谱重心）\n")
    if (~np.isnan(tl_c)).any():
        c = tl_c[~np.isnan(tl_c)]
        L.append(f"- 响段中位 {np.median(c):.0f}Hz，范围 {np.percentile(c,10):.0f}–{np.percentile(c,90):.0f}Hz")
        L.append(f"- 最亮 {tl_t[np.nanargmax(tl_c)]:.1f}s（{np.nanmax(tl_c):.0f}Hz），最暗 {tl_t[np.nanargmin(tl_c)]:.1f}s（{np.nanmin(tl_c):.0f}Hz）")
    else:
        L.append("没有可用的响段。")

    report = "\n".join(L) + "\n"
    md_path = os.path.join(outdir, f"{base}-声学分析报告.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 声谱图：matplotlib 优先，没有就 ffmpeg
    png_path = os.path.join(outdir, f"{base}-声谱图.png")
    made = False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager  # noqa: F401
        win_n, hop_n = 1024, 512
        frames = []
        w = np.hanning(win_n)
        for i0 in range(0, len(mono) - win_n, hop_n):
            frames.append(np.abs(np.fft.rfft(mono[i0:i0 + win_n] * w)))
        S = 20 * np.log10(np.array(frames).T + 1e-9)
        fr = np.fft.rfftfreq(win_n, 1 / sr)
        keep = fr <= 5000
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
        for t in onsets:
            ax2.axvline(t, color="gray", lw=0.3, alpha=0.5)
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        made = True
    except Exception:
        try:
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-lavfi",
                            "showspectrumpic=s=1200x600:legend=1:color=intensity", png_path],
                           check=True)
            made = True
        except Exception:
            png_path = None

    print(report)
    print(f"MEDIA:{md_path}")
    if made:
        print(f"MEDIA:{png_path}")


if __name__ == "__main__":
    main()
