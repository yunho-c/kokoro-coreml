#!/usr/bin/env python3
import argparse
import time
import numpy as np
import wave
from pathlib import Path
from test_ane_pipeline import HybridTTSPipeline


def save_wav(path: str, audio: np.ndarray, sample_rate: int = 24000):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Normalize to int16 safely
    if audio.size == 0:
        data = np.zeros((0,), dtype=np.int16)
    else:
        peak = max(1e-7, float(np.max(np.abs(audio))))
        scaled = np.clip(audio / peak, -1.0, 1.0)
        data = (scaled * 32767.0).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', choices=['coreml', 'pytorch'], default='coreml')
    ap.add_argument('--text', required=True)
    ap.add_argument('--voice', default='af_heart')
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--out', default='outputs/out.wav')
    args = ap.parse_args()

    p = HybridTTSPipeline(force_engine=args.engine)

    t0 = time.time()
    audio, sr = p.synthesize(args.text, voice=args.voice, speed=args.speed)
    t1 = time.time()

    if audio is None:
        print('FAIL synthesis')
        return

    save_wav(args.out, audio, sr)

    audio_len = len(audio) / sr
    synth_time = t1 - t0
    rtf = synth_time / audio_len if audio_len > 0 else float('inf')
    print(f"engine={args.engine} time_sec={synth_time:.3f} audio_sec={audio_len:.3f} rtf={rtf:.3f} out={args.out}")


if __name__ == '__main__':
    main()
