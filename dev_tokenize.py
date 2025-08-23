#!/usr/bin/env python3
import argparse, json
from kokoro import KPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--text", required=True)
parser.add_argument("--voice", default="af_heart")
parser.add_argument("--speed", type=float, default=1.0)
args = parser.parse_args()

# Minimal tokenizer: use Kokoro pipeline to get phoneme ids
pipe = KPipeline(lang_code='a')
text = args.text
voice = args.voice
speed = args.speed

# Grab first step ids
ids = []
for _, ps, _ in pipe(text, voice=voice, speed=speed):
    # ps is torch tensor-like; convert to list of ints
    try:
        arr = ps.cpu().numpy().tolist()
    except Exception:
        arr = list(map(int, ps))
    # Flatten
    if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], list):
        arr = arr[0]
    ids = [int(x) for x in arr]
    break
print(json.dumps({"ids": ids}))
