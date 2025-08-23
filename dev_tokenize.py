#!/usr/bin/env python3
import argparse, json, sys
from contextlib import redirect_stdout
from kokoro import KPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--text", required=True)
parser.add_argument("--voice", default="af_heart")
parser.add_argument("--speed", type=float, default=1.0)
args = parser.parse_args()

# Load vocab mapping from config.json
try:
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    vocab = config.get('vocab', {})
    print(f"Loaded vocab with {len(vocab)} symbols", file=__import__('sys').stderr)
except Exception as e:
    print(f"Error loading config {args.config}: {e}", file=__import__('sys').stderr)
    vocab = {}

# Minimal tokenizer: use Kokoro pipeline to get phoneme symbols/ids
# Redirect any library stdout (warnings, progress) to stderr so stdout is pure JSON
text = args.text
voice = args.voice
speed = args.speed
ids = []
with redirect_stdout(sys.stderr):
    pipe = KPipeline(lang_code='a')
    for _, ps, _ in pipe(text, voice=voice, speed=speed):
        # Handle both tensor and list cases
        try:
            if hasattr(ps, 'cpu'):
                # Torch tensor - convert to list
                arr = ps.cpu().numpy().tolist()
            else:
                # Already a list or similar
                arr = list(ps)
        except Exception as e:
            print(f"Error converting ps to list: {e}", file=__import__('sys').stderr)
            arr = list(ps)
        
        # Flatten if nested
        if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], list):
            arr = arr[0]
        
        # Convert symbols to numeric IDs
        numeric_ids = []
        for item in arr:
            if isinstance(item, str):
                # String phoneme symbol - look up in vocab
                if item in vocab:
                    numeric_ids.append(vocab[item])
                else:
                    print(f"Warning: Unknown phoneme symbol '{item}', skipping", file=__import__('sys').stderr)
            elif isinstance(item, (int, float)):
                # Already numeric - use as-is
                numeric_ids.append(int(item))
            else:
                print(f"Warning: Unexpected item type {type(item)}: {item}", file=__import__('sys').stderr)
        
        ids = numeric_ids
        break

# Emit only JSON to stdout
print(json.dumps({"ids": ids}))
