#!/usr/bin/env python3
"""
Export Kokoro Duration model to CoreML ML Program (.mlpackage) with strict shape bounds
and aliasing fixes to avoid CoreML "tile reps >= 1" and BNNS input/output alias errors.

Inputs:
- checkpoints/config.json
- checkpoints/kokoro-v1_0.pth (optional; will fallback to default constructor)

Outputs:
- coreml/kokoro_duration.mlpackage
"""
import os
import pathlib
import numpy as np
import coremltools as ct
import torch
import torch.nn as nn

# Reuse local module loading approach from export_synthesizers
import importlib.util, sys
_ROOT = pathlib.Path(__file__).resolve().parent

def _load_module_from(path_rel: str, name: str):
    p = (_ROOT / path_rel).resolve()
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

kokoro_istftnet = _load_module_from("kokoro/istftnet.py", "kokoro_istftnet")
kokoro_modules_src = (_ROOT / "kokoro/modules.py").read_text()
kokoro_modules = importlib.util.module_from_spec(importlib.util.spec_from_loader("kokoro_modules", loader=None))
kokoro_modules.__dict__['kokoro_istftnet'] = kokoro_istftnet
kokoro_modules.__dict__['__name__'] = 'kokoro_modules'
exec(kokoro_modules_src, kokoro_modules.__dict__)
sys.modules['kokoro_modules'] = kokoro_modules
kokoro_model_src = (_ROOT / "kokoro/model.py").read_text()
kokoro_model = importlib.util.module_from_spec(importlib.util.spec_from_loader("kokoro_model", loader=None))
kokoro_model.__dict__['kokoro_modules'] = kokoro_modules
kokoro_model.__dict__['__name__'] = 'kokoro_model'
exec(kokoro_model_src, kokoro_model.__dict__)
sys.modules['kokoro_model'] = kokoro_model
KModel = kokoro_model.KModel
AdaLayerNorm = kokoro_modules.AdaLayerNorm

class CoreMLFriendlyTextEncoder(nn.Module):
    def __init__(self, original_encoder):
        super().__init__()
        self.embedding = original_encoder.embedding
        self.cnn = original_encoder.cnn
        self.lstm = original_encoder.lstm
    def forward(self, x, input_lengths, m):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        m = m.unsqueeze(1)
        x.masked_fill_(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = x.transpose(-1, -2)
        x.masked_fill_(m, 0.0)
        return x

class CoreMLFriendlyDurationEncoder(nn.Module):
    def __init__(self, original_encoder):
        super().__init__()
        self.lstms = original_encoder.lstms
        self.dropout = original_encoder.dropout
    def forward(self, x, style, text_lengths, m):
        masks = m
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        x = x.transpose(-1, -2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                block.flatten_parameters()
                x, _ = block(x)
                x = nn.functional.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)
        return x.transpose(-1, -2)

class DurationModel(nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel
        self.kmodel.text_encoder = CoreMLFriendlyTextEncoder(kmodel.text_encoder)
        self.kmodel.predictor.text_encoder = CoreMLFriendlyDurationEncoder(kmodel.predictor.text_encoder)
        if hasattr(self.kmodel.bert.embeddings, 'token_type_ids'):
            delattr(self.kmodel.bert.embeddings, 'token_type_ids')
    def forward(self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor, speed: torch.FloatTensor, attention_mask: torch.LongTensor):
        k = self.kmodel
        input_lengths = attention_mask.sum(dim=-1).to(torch.long)
        text_mask = attention_mask == 0
        token_type_ids = torch.zeros_like(input_ids)
        bert_dur = k.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        d_en = k.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]  # style half
        d = k.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = k.predictor.lstm(d)
        duration = k.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long()
        t_en = k.text_encoder(input_ids, input_lengths, text_mask)
        # Avoid CoreML aliasing: ensure ref_s output is distinct
        ref_s_out = ref_s + torch.zeros_like(ref_s)
        return pred_dur, d, t_en, s, ref_s_out

def main():
    cfg = _ROOT / "checkpoints/config.json"
    ckpt = _ROOT / "checkpoints/kokoro-v1_0.pth"
    if cfg.exists() and ckpt.exists():
        kmodel = KModel(config=str(cfg), model=str(ckpt), disable_complex=True)
    elif cfg.exists():
        kmodel = KModel(config=str(cfg), disable_complex=True)
    else:
        kmodel = KModel(disable_complex=True)
    duration_model = DurationModel(kmodel).eval()

    # Trace minimal representative inputs
    T = 32
    input_ids = torch.randint(0, 100, (1, T), dtype=torch.int32)
    ref_s = torch.zeros(1, 256, dtype=torch.float32)
    speed = torch.tensor([1.0], dtype=torch.float32)
    attention_mask = torch.ones(1, T, dtype=torch.int32)

    with torch.no_grad():
        _ = duration_model(input_ids, ref_s, speed, attention_mask)

    traced = torch.jit.trace(duration_model, (input_ids, ref_s, speed, attention_mask), strict=False)

    # Convert with strict shapes and RangeDim min=1 for token dims
    duration_ml = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids",      shape=(ct.RangeDim(1, 512),),  dtype=np.int32),
            ct.TensorType(name="ref_s",          shape=(256,),                  dtype=np.float32),
            ct.TensorType(name="speed",          shape=(1,),                    dtype=np.float32),
            ct.TensorType(name="attention_mask", shape=(ct.RangeDim(1, 512),),  dtype=np.int32),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS12,
        compute_precision=ct.precision.FLOAT16,
    )

    out_dir = _ROOT / "coreml"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kokoro_duration.mlpackage"
    duration_ml.save(str(out_path))
    print(f"✅ Saved duration model to: {out_path}")

if __name__ == "__main__":
    main()
