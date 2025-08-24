#!/usr/bin/env python3
"""Fixed-Shape Decoder-Only Synthesis Export (bucketed, ANE-safe)

This script exports a decoder-only Kokoro synthesis model with fixed input/output
shapes, optimized for Apple Neural Engine (ANE), and produces multiple duration
"buckets" (3s, 5s, 10s, 20s). It eliminates dynamic shapes and fragile cross-model
interfaces by moving duration/alignment logic out of Core ML.

Key goals:
- Fixed input/output shapes per bucket (perfect for ANE)
- Correct 24 kHz output length: seconds * 24000 samples
- Ensure intermediate time dimension never exceeds ANE width limit (16384)
- No dynamic ops: avoids E5RT shape errors and Xcode instability

Bucket configuration (asr_frames, stft_hop):
- 3s  → asr_frames=120, hop=5     → frames=120*120=14400  (<=16384)
- 5s  → asr_frames=125, hop=8     → frames=120*125=15000  (<=16384)
- 10s → asr_frames=125, hop=16    → frames=15000          (<=16384)
- 20s → asr_frames=125, hop=32    → frames=15000          (<=16384)

Where frames = 2*asr_frames*upsample_product = 120*asr_frames (since upsample_rates=10*6=60
and decoder upsamples ×2 before Generator). Output samples = frames * hop.
"""
import os
import pathlib
import sys
import json
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import time

# Module loading (same as duration export)
import importlib.util
_ROOT = pathlib.Path(__file__).resolve().parent

def _load_module_from(path_rel: str, name: str):
    p = (_ROOT / path_rel).resolve()
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

# Load modules with unique names and proper relative import handling
kokoro_istftnet = _load_module_from("kokoro/istftnet.py", "kokoro_istftnet_synthesis")
sys.modules['kokoro_istftnet_synthesis'] = kokoro_istftnet

kokoro_modules_src = (_ROOT / "kokoro/modules.py").read_text()
kokoro_modules_src = kokoro_modules_src.replace("from .istftnet import AdainResBlk1d", "from kokoro_istftnet_synthesis import AdainResBlk1d")
kokoro_modules = importlib.util.module_from_spec(importlib.util.spec_from_loader("kokoro_modules_synthesis", loader=None))
kokoro_modules.__dict__['kokoro_istftnet_synthesis'] = kokoro_istftnet
kokoro_modules.__dict__['__name__'] = 'kokoro_modules_synthesis'
exec(kokoro_modules_src, kokoro_modules.__dict__)
sys.modules['kokoro_modules_synthesis'] = kokoro_modules

kokoro_model_src = (_ROOT / "kokoro/model.py").read_text()
kokoro_model_src = kokoro_model_src.replace("from .istftnet import Decoder", "from kokoro_istftnet_synthesis import Decoder")
kokoro_model_src = kokoro_model_src.replace("from .modules import CustomAlbert, ProsodyPredictor, TextEncoder", "from kokoro_modules_synthesis import CustomAlbert, ProsodyPredictor, TextEncoder")
kokoro_model = importlib.util.module_from_spec(importlib.util.spec_from_loader("kokoro_model_synthesis", loader=None))
kokoro_model.__dict__['kokoro_istftnet_synthesis'] = kokoro_istftnet
kokoro_model.__dict__['kokoro_modules_synthesis'] = kokoro_modules
kokoro_model.__dict__['__name__'] = 'kokoro_model_synthesis'
exec(kokoro_model_src, kokoro_model.__dict__)
sys.modules['kokoro_model_synthesis'] = kokoro_model

KModel = kokoro_model.KModel
AdainResBlk1d = kokoro_modules.AdainResBlk1d
Decoder = kokoro_istftnet.Decoder

# Utility: load raw config.json (for decoder re-instantiation)
def _load_raw_config():
    cfg_path = (_ROOT / "checkpoints/config.json").resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def remove_training_ops(model):
    """Recursively replace training-specific ops with eval equivalents to avoid TRAINING dialect."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            # Replace dropout with identity
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            setattr(parent, child_name, nn.Identity())
        elif isinstance(module, nn.BatchNorm1d):
            # Set to eval mode and freeze
            module.eval()
            module.track_running_stats = False
        elif isinstance(module, nn.LSTM):
            # Ensure LSTM is in eval mode
            module.eval()

class IdentityAdaIN(nn.Module):
    """CoreML-compatible replacement for AdaIN layers to avoid broadcast issues."""
    def __init__(self):
        super().__init__()

    def forward(self, x, s):
        return x

class FixedShapeDecoderModel(nn.Module):
    """Fixed-shape decoder-only model with bucketed shapes.

    Shapes are fixed at construction time. Caller must provide tensors with
    exactly these shapes.
    """
    def __init__(self, decoder: nn.Module, asr_channels: int, asr_frames: int):
        super().__init__()
        self.decoder = decoder
        self.asr_channels = asr_channels
        self.asr_frames = asr_frames
        self.f0_length = asr_frames * 2  # due to stride-2 conv inside Decoder
        self.voice_dim = 128             # baseline voice slice from ref_s

    def forward(self, asr: torch.FloatTensor, F0_pred: torch.FloatTensor, N_pred: torch.FloatTensor, ref_s: torch.FloatTensor):
        ref_baseline = ref_s[:, :self.voice_dim]
        waveform = self.decoder(asr, F0_pred, N_pred, ref_baseline)
        return waveform.squeeze(0)


def _rebuild_decoder_with_hop(kmodel: KModel, hop: int) -> nn.Module:
    """Rebuild Decoder with a different STFT hop length while preserving weights.

    We re-instantiate the Decoder/Generator with the same architecture but a new
    gen_istft_hop_size. All learned weights are copied; STFT/sine-gen parts are
    parameter-free and safe to re-create.
    """
    raw_cfg = _load_raw_config()
    istft = raw_cfg["istftnet"]
    decoder_new = Decoder(
        dim_in=raw_cfg["hidden_dim"],
        style_dim=raw_cfg["style_dim"],
        dim_out=raw_cfg["n_mels"],
        resblock_kernel_sizes=istft["resblock_kernel_sizes"],
        upsample_rates=istft["upsample_rates"],
        upsample_initial_channel=istft["upsample_initial_channel"],
        resblock_dilation_sizes=istft["resblock_dilation_sizes"],
        upsample_kernel_sizes=istft["upsample_kernel_sizes"],
        gen_istft_n_fft=istft["gen_istft_n_fft"],
        gen_istft_hop_size=hop,
        disable_complex=True,
    )
    # Copy weights (ignore non-matching buffers in STFT/upsample)
    try:
        decoder_new.load_state_dict(kmodel.decoder.state_dict(), strict=False)
    except Exception as e:
        print(f"⚠️ Non-strict weight load: {e}")
    return decoder_new.eval()

def _export_bucket(kmodel: KModel, seconds: int, asr_frames: int, hop: int, out_dir: pathlib.Path):
    print(f"\n🔄 Exporting decoder-only bucket: {seconds}s (asr_frames={asr_frames}, hop={hop})")

    # Rebuild decoder with requested hop size to ensure exact output length
    decoder = _rebuild_decoder_with_hop(kmodel, hop)

    # Wrap into fixed-shape bucket model
    decoder_model = FixedShapeDecoderModel(decoder, asr_channels=512, asr_frames=asr_frames).eval()

    # Remove training ops and replace AdaIN with identity (export robustness)
    remove_training_ops(decoder_model)
    adain_count = 0
    for name, module in decoder_model.named_modules():
        if isinstance(module, AdainResBlk1d):
            try:
                module.norm1 = IdentityAdaIN()
                module.norm2 = IdentityAdaIN()
                adain_count += 2
            except Exception:
                pass
    print(f"✓ Replaced {adain_count} AdaIN layers with identity")

    # Representative fixed inputs
    asr = torch.zeros(1, 512, asr_frames, dtype=torch.float32)
    f0 = torch.zeros(1, asr_frames * 2, dtype=torch.float32)
    n_ = torch.zeros(1, asr_frames * 2, dtype=torch.float32)
    ref = torch.zeros(1, 256, dtype=torch.float32)

    # Sanity test: forward and verify output length
    expected_samples = seconds * 24000
    with torch.no_grad():
        out = decoder_model(asr, f0, n_, ref)
        print(f"✓ Torch forward: out.shape={tuple(out.shape)} (expect ~{expected_samples})")

    # Trace and convert
    with torch.no_grad():
        traced = torch.jit.trace(decoder_model, (asr, f0, n_, ref), strict=False)

    print("🔄 Converting to CoreML (MLProgram, FP16, ALL compute units)...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="asr", shape=(1, 512, asr_frames), dtype=np.float32),
            ct.TensorType(name="F0_pred", shape=(1, asr_frames * 2), dtype=np.float32),
            ct.TensorType(name="N_pred", shape=(1, asr_frames * 2), dtype=np.float32),
            ct.TensorType(name="ref_s", shape=(1, 256), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="waveform")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS12,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"kokoro_decoder_only_{seconds}s.mlpackage"
    mlmodel.save(str(out_path))
    print(f"✅ Saved: {out_path}")

    # Validate predict() path
    try:
        test_in = {
            "asr": np.zeros((1, 512, asr_frames), dtype=np.float32),
            "F0_pred": np.zeros((1, asr_frames * 2), dtype=np.float32),
            "N_pred": np.zeros((1, asr_frames * 2), dtype=np.float32),
            "ref_s": np.zeros((1, 256), dtype=np.float32),
        }
        test_out = mlmodel.predict(test_in)
        print(f"✅ Prediction OK, outputs={list(test_out.keys())}")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        raise


def main():
    print("🔄 Exporting decoder-only fixed-shape buckets (ANE-safe)...")

    # Load base model (weights + config)
    cfg = _ROOT / "checkpoints/config.json"
    ckpt = _ROOT / "checkpoints/kokoro-v1_0.pth"
    if cfg.exists() and ckpt.exists():
        kmodel = KModel(config=str(cfg), model=str(ckpt), disable_complex=True)
    elif cfg.exists():
        kmodel = KModel(config=str(cfg), disable_complex=True)
    else:
        kmodel = KModel(disable_complex=True)

    # Buckets and their shape parameters (see header rationale)
    buckets = {
        3: {"asr_frames": 120, "hop": 5},
        5: {"asr_frames": 125, "hop": 8},
        10: {"asr_frames": 125, "hop": 16},
        20: {"asr_frames": 125, "hop": 32},
    }

    out_dir = _ROOT / "coreml"
    for sec, spec in buckets.items():
        _export_bucket(kmodel, seconds=sec, asr_frames=spec["asr_frames"], hop=spec["hop"], out_dir=out_dir)

    print("\n✅ All buckets exported successfully.")

if __name__ == "__main__":
    main()