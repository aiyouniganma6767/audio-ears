#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自测：用合成信号验证 analyze.py 报出来的数字对不对。

每个用例都自带真值——我们知道信号里是什么频率、几段静默、哪里埋了个弱成分——
报告里的数字必须能对上。不依赖任何外部音频文件；ffmpeg 缺失时自动跳过 MP3 用例。

    python3 scripts/selftest.py

退出码 0 = 全过；1 = 有用例失败（会打出来差多少）。
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYZE = os.path.join(HERE, "analyze.py")
SR = 32000

# 一组故意非整数倍的分音（钟/金属的典型谱），用来验证基频不会被带到泛音上
BELL_PARTS = (1.0, 2.0, 2.76, 4.07, 5.41, 8.13)


# ---------------------------------------------------------------- 合成信号

def struck(f0, dur=2.5, amp=0.4, decay=2.0, partials=(1.0,), phase=0.7):
    """一次敲击：若干分音 + 指数衰减 + 2ms 起音。"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for k, ratio in enumerate(partials):
        a = amp / (k + 1) ** 0.7
        x += a * np.sin(2 * np.pi * f0 * ratio * t + phase * k) * np.exp(-t * (decay + k * 0.9))
    a = int(0.002 * SR)
    x[:a] *= np.linspace(0, 1, a)
    return x


def silence(sec):
    return np.zeros(int(sec * SR))


def mix(total, events):
    """events: [(起始秒, 波形), ...]"""
    y = np.zeros(int(total * SR))
    for t0, sig in events:
        i = int(t0 * SR)
        y[i:i + len(sig)] += sig[:len(y) - i]
    return y


def write_wav(path, x, channels=1):
    x = np.clip(x, -1, 1)
    if channels == 1:
        data = (x * 32767).astype("<i2").tobytes()
    else:  # 给立体声用例：右声道轻微不同
        l = (x * 32767).astype("<i2")
        r = (x * 0.6 * 32767).astype("<i2")
        data = np.stack([l, r], axis=1).tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data)


# ---------------------------------------------------------------- 跑分析、解析报告

def run(path, outdir):
    p = subprocess.run([sys.executable, ANALYZE, path, outdir],
                       capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        raise RuntimeError((p.stdout + p.stderr)[-400:])
    return p.stdout


def run_args(path, outdir, extra):
    p = subprocess.run([sys.executable, ANALYZE, path, outdir] + list(extra),
                       capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        raise RuntimeError((p.stdout + p.stderr)[-400:])
    return p.stdout


def run_json(path, outdir, extra=()):
    """跑 --json 模式，返回解析后的 dict（stdout 必须是一份合法 JSON）。"""
    import json
    return json.loads(run_args(path, outdir, ["--json"] + list(extra)))


def events_of(report):
    """解析敲击表：[(时间, 基频, 音名, 音分, 粗糙度), ...]"""
    sec = report.split("## 敲击/起音点")[1].split("## 拍频")[0]
    rows = []
    for line in sec.splitlines():
        m = re.match(r"\|\s*([\d.]+)s\s*\|\s*([\d.]+)Hz\s*\|\s*([A-G]#?\d)\s*\|\s*([+-]\d+)\s*\|\s*([\d.]+)\s*\|", line)
        if m:
            rows.append(tuple([float(m.group(1)), float(m.group(2)), m.group(3),
                               int(m.group(4)), float(m.group(5))]))
    return rows


def hot_roughness(report):
    """粗糙度热点表里的粗糙度值（跳过表头和 nan 行）"""
    sec = report.split("## 拍频与粗糙度热点")[1].split("## 掩蔽提示")[0]
    vals = []
    for line in sec.splitlines():
        m = re.match(r"\|\s*([\d.]+)s\s*\|\s*([\d.]+)\s*\|", line)
        if m:
            vals.append(float(m.group(2)))
    return vals


# ---------------------------------------------------------------- 用例

def test_pitch_clean(tmp):
    """干净的单分音敲击：330.0 Hz 应报在 ±3 音分内。"""
    w = os.path.join(tmp, "clean330.wav")
    write_wav(w, np.concatenate([silence(0.6), struck(330.0)]))
    ev = events_of(run(w, tmp))
    assert len(ev) == 1, f"起音数 {len(ev)} != 1"
    t, f, name, cents, _ = ev[0]
    assert abs(f - 330.0) < 0.7, f"基频 {f:.2f} Hz 偏了"
    assert abs(cents) <= 3, f"音分偏差 {cents:+d} 超过 ±3"
    assert name == "E4", f"音名 {name} != E4"
    return f"{f:.2f}Hz {cents:+d}音分 {name}"


def test_pitch_nonharmonic_bell(tmp):
    """非整数倍分音的钟声：基频不能被带到 2/2.76/4.07 倍分音上。"""
    w = os.path.join(tmp, "bell261.wav")
    write_wav(w, np.concatenate([silence(0.6), struck(261.63, partials=BELL_PARTS)]))
    ev = events_of(run(w, tmp))
    assert len(ev) == 1, f"起音数 {len(ev)} != 1"
    f, name, cents = ev[0][1], ev[0][2], ev[0][3]
    assert abs(1200 * np.log2(f / 261.63)) <= 5, f"基频 {f:.2f} Hz 错了半音以上"
    assert name == "C4", f"音名 {name} != C4"
    return f"{f:.2f}Hz {cents:+d}音分 {name}"


def test_comma(tmp):
    """纯律大三度 264.0 Hz 与十二平均律 261.63 Hz 相差约 15.6 音分，必须测出来。"""
    cents = {}
    for f0 in (261.63, 264.0):
        w = os.path.join(tmp, f"third{f0}.wav")
        write_wav(w, np.concatenate([silence(0.6), struck(f0)]))
        ev = events_of(run(w, tmp))
        assert len(ev) == 1, f"{f0}Hz 起音数 {len(ev)} != 1"
        cents[f0] = ev[0][3]
    diff = cents[264.0] - cents[261.63]
    assert abs(diff - 15.6) <= 4, f"测出 {diff:+d} 音分，应在 15.6±4"
    return f"261.63Hz→{cents[261.63]:+d}，264Hz→{cents[264.0]:+d}，差 {diff:+d} 音分（真值 +15.6）"


def test_onset_count(tmp):
    """三个相隔 1.5 秒的敲击 → 三个起音。"""
    w = os.path.join(tmp, "three.wav")
    write_wav(w, mix(6.0, [(0.6, struck(330.0, dur=1.2)),
                           (2.1, struck(392.0, dur=1.2)),
                           (3.6, struck(523.25, dur=1.2))]))
    ev = events_of(run(w, tmp))
    assert len(ev) == 3, f"起音数 {len(ev)} != 3"
    ts = [round(e[0], 2) for e in ev]
    assert all(abs(a - b) < 0.06 for a, b in zip(ts, (0.6, 2.1, 3.6))), f"起音时间 {ts} 偏了"
    return f"{len(ev)} 个起音 @ {ts}"


def test_echo_dedupe_unit(tmp):
    _ = tmp
    """回声去重：同一击 0.4 秒后、低 10 dB 的重复应为回声，不算新起音。"""
    sys.path.insert(0, HERE)
    import analyze as A
    base = dict(t=0.6, f0=330.0, name="E4", cents=0, level=-20.0, rough=0.0, beats=[], peaks=[])
    echo = dict(base, t=1.0, level=-30.0)
    other = dict(base, t=1.0, f0=440.0, name="A4", level=-5.0)
    kept, n_echo = A.dedupe_echoes([base, echo, other])
    assert n_echo == 1, f"剔除 {n_echo} != 1"
    assert len(kept) == 2, f"保留 {len(kept)} != 2"
    return "0.4s/低10dB/同音高 → 剔除；同刻不同音高、更响的 → 保留"


def test_silence_split(tmp):
    """1.5 秒静默必须被切成两段，且不影响起音计数。"""
    w = os.path.join(tmp, "gap.wav")
    write_wav(w, mix(6.8, [(0.6, struck(330.0, dur=2.0, decay=2.5)),
                           (4.1, struck(392.0, dur=2.0, decay=2.5))]))
    rep = run(w, tmp)
    assert "检测到 1 段 ≥1.2 秒的静默" in rep, "没有检到静默段"
    ev = events_of(rep)
    assert len(ev) == 2, f"起音数 {len(ev)} != 2"
    return "1 段静默、2 个起音"


def test_quiet_gate(tmp):
    """长衰减尾巴：安静段不能报出假的粗糙度（旧版在淡出里报 0.836）。"""
    w = os.path.join(tmp, "fade.wav")
    n = int(4.0 * SR)
    t = np.arange(n) / SR
    tail = 0.4 * np.sin(2 * np.pi * 330 * t) * np.exp(-t * 3.0)
    write_wav(w, np.concatenate([silence(0.6), tail]))
    rep = run(w, tmp)
    rough = hot_roughness(rep)
    assert rough, "热点表是空的"
    assert max(rough) < 0.05, f"安静段粗糙度 {max(rough):.3f} 应接近 0"
    return f"最大粗糙度 {max(rough):.3f}（旧版 0.836）"


def test_masking(tmp):
    """1050 Hz 弱成分贴着 1000 Hz 强成分（同临界带）→ 应提示可能被掩蔽。"""
    w = os.path.join(tmp, "mask.wav")
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    body = (0.3 * np.sin(2 * np.pi * 1000 * t) + 0.03 * np.sin(2 * np.pi * 1050 * t)) * np.exp(-t * 1.5)
    write_wav(w, np.concatenate([silence(0.6), body]))
    rep = run(w, tmp)
    assert "可能被掩蔽" in rep, "没有掩蔽提示"
    assert "1050Hz" in rep, "提示里没有点名 1050Hz"
    return "1000+1050 Hz 同临界带 → 已提示"


def test_key(tmp):
    """C 大调琶音 → 模板推测应为 C 大调。"""
    notes = [261.63, 329.63, 392.0, 523.25, 392.0, 329.63, 261.63]
    evs = [(0.6 + 0.8 * i, struck(f, dur=1.2, decay=2.5)) for i, f in enumerate(notes)]
    w = os.path.join(tmp, "cmaior.wav")
    write_wav(w, mix(0.8 * len(notes) + 2.0, evs))
    rep = run(w, tmp)
    line = next((l for l in rep.splitlines() if "调性" in l), "")
    assert "C 大调" in line, f"调性报的是：{line}"
    return line.strip("- ")


def test_stereo(tmp):
    """立体声文件：应识别成 2 声道并给出左右 RMS 与相关。"""
    w = os.path.join(tmp, "stereo.wav")
    write_wav(w, np.concatenate([silence(0.6), struck(440.0, dur=1.2)]), channels=2)
    rep = run(w, tmp)
    assert "立体声" in rep, "没识别成立体声"
    assert "声道相关" in rep, "缺左右声道相关"
    return "2 声道 + 相关读数"


def test_mp3(tmp):
    """MP3（ffmpeg 解码路径）；没有 ffmpeg 就跳过。"""
    if not shutil.which("ffmpeg"):
        return "SKIP: 没有 ffmpeg"
    wav = os.path.join(tmp, "tone440.wav")
    mp3 = os.path.join(tmp, "tone440.mp3")
    write_wav(wav, np.concatenate([silence(0.6), struck(440.0)]))
    p = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav, "-b:a", "128k", mp3],
                       capture_output=True)
    if p.returncode != 0:
        return "SKIP: ffmpeg 编码失败"
    ev = events_of(run(mp3, tmp))
    assert len(ev) == 1, f"MP3 起音数 {len(ev)} != 1"
    assert abs(ev[0][1] - 440.0) < 1.5, f"MP3 基频 {ev[0][1]:.1f} Hz 偏了"
    return f"MP3 解码 + {ev[0][1]:.2f}Hz"


def test_missing_file(tmp):
    _ = tmp
    """给不存在的文件要报错退出，不是崩溃。"""
    p = subprocess.run([sys.executable, ANALYZE, "/nonexistent/nope.wav"],
                       capture_output=True, text=True)
    assert p.returncode == 1, f"退出码 {p.returncode} != 1"
    assert "找不到文件" in p.stdout, "没有提示找不到文件"
    return "退出码 1 + 明确提示"


# ---------------------------------------------------------------- v1.1 新增

def test_json_output(tmp):
    """--json：合法 JSON，字段齐、和 Markdown 对得上。"""
    w = os.path.join(tmp, "json330.wav")
    write_wav(w, np.concatenate([silence(0.6), struck(330.0)]))
    d = run_json(w, tmp)
    for key in ("version", "duration_s", "sample_rate", "channels", "basic", "sections",
                "events", "timeline", "masking", "roughness_hotspots", "paths"):
        assert key in d, f"缺字段 {key}"
    assert len(d["events"]) == 1, f"事件数 {len(d['events'])} != 1"
    assert abs(d["events"][0]["f0"] - 330.0) < 0.7, "JSON 里的基频不对"
    assert os.path.exists(d["paths"]["md"]), "Markdown 报告没写出来"
    assert d["paths"]["json"].endswith(".json")
    assert abs(len(d["timeline"]["t"]) - len(d["timeline"]["roughness"])) <= 0, "时间轴长度不一致"
    return f"{len(d)} 个顶层字段，{len(d['events'])} 个事件"


def test_chord_polyphony(tmp):
    """同时敲 C4+E4+G4：并行音应报出三个音，误差 ±15 音分。"""
    chord = mix(3.5, [(0.6, sum(struck(f, dur=2.0, partials=BELL_PARTS, amp=0.22)
                                for f in (261.63, 329.63, 392.0)))])
    w = os.path.join(tmp, "chord.wav")
    write_wav(w, chord)
    d = run_json(w, tmp)
    ev = [e for e in d["events"] if e["voices"]]
    assert ev, "没有事件"
    voices = ev[0]["voices"]
    assert len(voices) == 3, f"并行音 {len(voices)} 个 != 3：{voices}"
    got = sorted(v["f0_hz"] for v in voices)
    for g, want in zip(got, (261.63, 329.63, 392.0)):
        assert abs(1200 * np.log2(g / want)) <= 15, f"{g:.1f}Hz 该是 {want}Hz"
    return "C4+E4+G4 → " + "+".join(v["note"] for v in voices)


def test_single_voice_no_false_polyphony(tmp):
    """单个钟（带 2.76/4.07 这些非整数倍分音）不能被误报成两个声部。"""
    w = os.path.join(tmp, "bell261b.wav")
    write_wav(w, np.concatenate([silence(0.6), struck(261.63, partials=BELL_PARTS)]))
    d = run_json(w, tmp)
    assert len(d["events"]) == 1, f"事件数 {len(d['events'])} != 1"
    voices = d["events"][0]["voices"]
    assert len(voices) == 1, f"误报并行音：{voices}"
    return "单钟 → 1 个声部（2.76/4.07 分音已排除）"


def test_unit_rules(tmp):
    """几个纯函数：音名/拍频/粗糙度顺序/音程判据。"""
    sys.path.insert(0, HERE)
    import analyze as A
    assert A.note_of(440.0)[0] == "A4", "440Hz 不是 A4"
    assert abs(A.note_of(466.16)[1]) <= 1, "466.16Hz 应接近 A#4"
    beats = A.beat_pairs([(1000.0, 1.0), (1005.0, 0.9)])
    assert len(beats) == 1 and abs(beats[0][2] - 5) < 0.5, f"拍频对不上：{beats}"
    assert A.beat_pairs([(1000.0, 1.0), (3000.0, 0.9)]) == [], "远距离频对不该算拍频"
    r2 = A.roughness([(261.63, 1.0), (277.18, 1.0)])   # 小二度
    r8 = A.roughness([(261.63, 1.0), (523.25, 1.0)])   # 八度
    assert r2 > r8 * 3, f"小二度粗糙度 {r2:.3f} 应远大于八度 {r8:.3f}"
    assert A._interval_ok(392.0, 261.63), "纯五度判据错了"
    assert not A._interval_ok(910.6, 330.0), "2.76 分音不该被当和弦成员"
    return f"小二度 {r2:.2f} vs 八度 {r8:.2f}；拍频 5Hz；音程判据 OK"


def test_clipping(tmp):
    """过载信号：削顶样本数必须报出来。"""
    w = os.path.join(tmp, "clip.wav")
    write_wav(w, np.concatenate([silence(0.6), 5.0 * struck(440.0)]))
    d = run_json(w, tmp)
    assert d["basic"]["clip_samples"] > 1000, f"削顶样本 {d['basic']['clip_samples']} 太少"
    assert d["basic"]["peak_dbfs"] >= -0.5, "峰值应接近 0 dBFS"
    return f"削顶 {d['basic']['clip_samples']} 个样本，峰值 {d['basic']['peak_dbfs']} dBFS"


def test_wav24(tmp):
    """24-bit WAV：标准库路径要能读，音高误差 ±5 音分。"""
    sr = SR
    y = np.concatenate([silence(0.6), struck(261.63, partials=BELL_PARTS)])
    yi = (np.clip(y, -1, 1) * (2 ** 23 - 1)).astype(np.int64)
    b = np.stack([(yi >> (8 * k)) & 0xFF for k in range(3)], axis=1).astype(np.uint8)
    w = os.path.join(tmp, "b24.wav")
    with wave.open(w, "wb") as ww:
        ww.setnchannels(1); ww.setsampwidth(3); ww.setframerate(sr)
        ww.writeframes(b.tobytes())
    d = run_json(w, tmp)
    assert len(d["events"]) == 1, f"事件数 {len(d['events'])} != 1"
    assert abs(1200 * np.log2(d["events"][0]["f0"] / 261.63)) <= 5, "24-bit 读出来音高偏了"
    return f"24-bit → {d['events'][0]['name']} {d['events'][0]['f0']:.1f}Hz"


def test_silence_only(tmp):
    """整首静音：不能崩、不能硬猜调性/速度。"""
    w = os.path.join(tmp, "silence.wav")
    write_wav(w, np.zeros(int(2.0 * SR)))
    d = run_json(w, tmp)
    assert d["events"] == [], "静音里报出了起音"
    assert d["key"] is None, "静音里猜出了调性"
    assert d["tempo_bpm"] is None, "静音里猜出了速度"
    assert d["brightness"] is None, "静音里算出了亮度"
    return "0 事件、无调性/速度/亮度"


def test_dedupe_negative(tmp):
    """不同音高、不同响度的连续两击：不能当回声删掉。"""
    sys.path.insert(0, HERE)
    import analyze as A
    base = dict(t=0.6, f0=261.63, name="C4", cents=0, level=-20.0, rough=0.0, beats=[], peaks=[])
    other = dict(base, t=1.0, f0=392.0, name="G4", level=-8.0)
    kept, n_echo = A.dedupe_echoes([base, other])
    assert n_echo == 0, f"误删 {n_echo} 个"
    assert len(kept) == 2, "两击应该都留着"
    return "不同音高/更响 → 不删"


def test_no_spectrogram(tmp):
    """--no-spectrogram：不生成 PNG，JSON 里为 null。"""
    w = os.path.join(tmp, "nosp.wav")
    write_wav(w, np.concatenate([silence(0.6), struck(440.0)]))
    d = run_json(w, tmp, extra=["--no-spectrogram"])
    assert d["paths"]["spectrogram"] is None, f"还是出了图：{d['paths']['spectrogram']}"
    assert os.path.exists(d["paths"]["md"])
    return "跳过声谱图，Markdown 照写"


# ---------------------------------------------------------------- 汇总

CASES = [
    ("音高·干净敲击", test_pitch_clean),
    ("音高·非整数倍分音", test_pitch_nonharmonic_bell),
    ("音高·纯律/平均律差 15.6 音分", test_comma),
    ("起音计数", test_onset_count),
    ("回声去重", test_echo_dedupe_unit),
    ("静默分段", test_silence_split),
    ("安静段不报假粗糙度", test_quiet_gate),
    ("掩蔽提示", test_masking),
    ("调性", test_key),
    ("立体声", test_stereo),
    ("MP3 解码", test_mp3),
    ("错误处理", test_missing_file),
    ("JSON 输出", test_json_output),
    ("并行音·三音和弦", test_chord_polyphony),
    ("并行音·单音不误报", test_single_voice_no_false_polyphony),
    ("纯函数规则", test_unit_rules),
    ("削顶检测", test_clipping),
    ("24-bit WAV", test_wav24),
    ("全静音", test_silence_only),
    ("回声去重·不误删", test_dedupe_negative),
    ("--no-spectrogram", test_no_spectrogram),
]


def main():
    print(f"analyze.py 自测 · {len(CASES)} 项\n" + "=" * 52)
    ok = skipped = 0
    failed = []
    tmp = tempfile.mkdtemp(prefix="audio-ears-selftest-")
    try:
        for i, (name, fn) in enumerate(CASES, 1):
            try:
                note = fn(tmp)
                if note.startswith("SKIP"):
                    skipped += 1
                    print(f"[{i:2d}] SKIP {name} — {note[6:]}")
                else:
                    ok += 1
                    print(f"[{i:2d}] PASS {name} — {note}")
            except Exception as e:
                failed.append((name, e))
                print(f"[{i:2d}] FAIL {name}")
                for line in str(e).splitlines()[:4]:
                    print(f"        └ {line}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("=" * 52)
    print(f"通过 {ok} / {len(CASES) - skipped}（跳过 {skipped}）")
    if failed:
        print("失败：" + "、".join(n for n, _ in failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
