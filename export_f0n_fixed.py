#!/usr/bin/env python3
"""Export fixed-shape F0/N predictor for decoder-only pipeline (3s bucket).

Inputs (fixed):
- en: [1, 512, 120]   (aligned features; use t_en @ pred_aln_trg in Swift)
- s:  [1, 128]        (style embedding; same as duration model's "s")

Outputs (fixed):
- F0_pred: [1, 240]
- N_pred:  [1, 240]

This wraps ProsodyPredictor.F0Ntrain with static shapes for Core ML.
"""
import pathlib
import importlib.util
import sys
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent

def _load_module_from(path_rel: str, name: str):
    p = (_ROOT / path_rel).resolve()
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

# Load modules w/ remapped imports like other exporters
kokoro_istftnet = _load_module_from("kokoro/istftnet.py", "kokoro_istftnet_f0n")
kokoro_modules_src = (_ROOT / "kokoro/modules.py").read_text()
kokoro_modules_src = kokoro_modules_src.replace("from .istftnet import AdainResBlk1d", "from kokoro_istftnet_f0n import AdainResBlk1d")
sys.modules["kokoro_istftnet_f0n"] = kokoro_istftnet
kokoro_modules = importlib.util.module_from_spec(importlib.util.spec_from_loader("kokoro_modules_f0n", loader=None))
kokoro_modules.__dict__["kokoro_istftnet_f0n"] = kokoro_istftnet
kokoro_modules.__dict__["__name__"] = "kokoro_modules_f0n"
exec(kokoro_modules_src, kokoro_modules.__dict__)
sys.modules["kokoro_modules_f0n"] = kokoro_modules

kokoro_model_src = (_ROOT / "kokoro/model.py").read_text()
kokoro_model_src = kokoro_model_src.replace("from .istftnet import Decoder", "from kokoro_istftnet_f0n import Decoder")
kokoro_model_src = kokoro_model_src.replace("from .modules import CustomAlbert, ProsodyPredictor, TextEncoder", "from kokoro_modules_f0n import CustomAlbert, ProsodyPredictor, TextEncoder")
kokoro_model = importlib.util.module_from_spec(importlib.util.spec_from_loader("kokoro_model_f0n", loader=None))
kokoro_model.__dict__["kokoro_istftnet_f0n"] = kokoro_istftnet
kokoro_model.__dict__["kokoro_modules_f0n"] = kokoro_modules
kokoro_model.__dict__["__name__"] = "kokoro_model_f0n"
exec(kokoro_model_src, kokoro_model.__dict__)
sys.modules["kokoro_model_f0n"] = kokoro_model

KModel = kokoro_model.KModel
ProsodyPredictor = kokoro_modules.ProsodyPredictor

class F0NFixed(nn.Module):
    def __init__(self, predictor: ProsodyPredictor, frames: int = 120):
        super().__init__()
        self.predictor = predictor
        self.frames = frames

    def forward(self, en: torch.FloatTensor, s: torch.FloatTensor):
        # en: [B, 512, T], s: [B, 128]
        # F0Ntrain expects x with channels = d_hid + style_dim along channel axis before its transpose
        # Concatenate broadcasted style along channel dim: [B, 512+128, T]
        s_bc = s.unsqueeze(-1).expand(-1, s.shape[1], en.shape[2])  # [B,128,T]
        x = torch.cat([en, s_bc], dim=1)  # [B,640,T]
        F0, N = self.predictor.F0Ntrain(x, s)
        return F0, N


def remove_training_ops(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, nn.Identity())
        elif isinstance(module, nn.BatchNorm1d):
            module.eval(); module.track_running_stats = False
        elif isinstance(module, nn.LSTM):
            module.eval()


def main():
    # Load base model; we only need the predictor weights
    cfg = _ROOT / "checkpoints/config.json"
    ckpt = _ROOT / "checkpoints/kokoro-v1_0.pth"
    if cfg.exists() and ckpt.exists():
        kmodel = KModel(config=str(cfg), model=str(ckpt), disable_complex=True)
    elif cfg.exists():
        kmodel = KModel(config=str(cfg), disable_complex=True)
    else:
        kmodel = KModel(disable_complex=True)

    frames = 120
    f0n = F0NFixed(kmodel.predictor, frames=frames).eval()
    remove_training_ops(f0n)

    # Representative inputs
    en = torch.zeros(1, 512, frames, dtype=torch.float32)
    s = torch.zeros(1, 128, dtype=torch.float32)

    # Test
    with torch.no_grad():
        F0, N = f0n(en, s)
        print(f"Test F0 shape={tuple(F0.shape)} N shape={tuple(N.shape)}")

    # Trace
    with torch.no_grad():
        traced = torch.jit.trace(f0n, (en, s), strict=False)

    # Convert
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="en", shape=(1, 512, frames), dtype=np.float32),
            ct.TensorType(name="s", shape=(1, 128), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="F0_pred"),
            ct.TensorType(name="N_pred"),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS12,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )

    out_dir = _ROOT / "coreml"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kokoro_f0n_3s.mlpackage"
    mlmodel.save(str(out_path))
    print(f"Saved F0N model: {out_path}")

if __name__ == "__main__":
    main()
