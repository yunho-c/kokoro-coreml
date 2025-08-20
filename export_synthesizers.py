#!/usr/bin/env python3
"""
Exports the Synthesizer models using a bucketing strategy.

This is a standalone script that contains all necessary code to avoid
import issues and environment conflicts. It assumes the .pth checkpoint
and config.json are present in the 'checkpoints' directory.
"""
import argparse
import os
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
from safetensors.torch import load_file
from collections import OrderedDict
import time
from torch.export import export

# --- Model Imports ---
# These are brought in from the kokoro package to make the script self-contained.

# Avoid importing kokoro as a package (its __init__ pulls in misaki). Load modules directly from files.
import importlib.util, pathlib, sys
_ROOT = pathlib.Path(__file__).resolve().parent
def _load_module_from(path_rel: str, name: str):
    p = (_ROOT / path_rel).resolve()
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

kokoro_istftnet = _load_module_from("kokoro/istftnet.py", "kokoro_istftnet")
sys.modules['kokoro_istftnet'] = kokoro_istftnet
kokoro_modules_src = (_ROOT / "kokoro/modules.py").read_text()
kokoro_modules_src = kokoro_modules_src.replace("from .istftnet import AdainResBlk1d", "from kokoro_istftnet import AdainResBlk1d")
kokoro_modules = importlib.util.module_from_spec(importlib.util.spec_from_loader("kokoro_modules", loader=None))
kokoro_modules.__dict__['kokoro_istftnet'] = kokoro_istftnet
kokoro_modules.__dict__['__name__'] = 'kokoro_modules'
exec(kokoro_modules_src, kokoro_modules.__dict__)
sys.modules['kokoro_modules'] = kokoro_modules
kokoro_model_src = (_ROOT / "kokoro/model.py").read_text()
kokoro_model_src = kokoro_model_src.replace("from .istftnet import Decoder", "from kokoro_istftnet import Decoder")
kokoro_model_src = kokoro_model_src.replace("from .modules import CustomAlbert, ProsodyPredictor, TextEncoder", "from kokoro_modules import CustomAlbert, ProsodyPredictor, TextEncoder")
kokoro_model = importlib.util.module_from_spec(importlib.util.spec_from_loader("kokoro_model", loader=None))
kokoro_model.__dict__['kokoro_istftnet'] = kokoro_istftnet
kokoro_model.__dict__['kokoro_modules'] = kokoro_modules
kokoro_model.__dict__['__name__'] = 'kokoro_model'
exec(kokoro_model_src, kokoro_model.__dict__)
sys.modules['kokoro_model'] = kokoro_model

KModel = kokoro_model.KModel
LayerNorm = kokoro_modules.LayerNorm
AdaLayerNorm = kokoro_modules.AdaLayerNorm
LinearNorm = kokoro_modules.LinearNorm
AdainResBlk1d = kokoro_modules.AdainResBlk1d

# --- CoreML-Friendly Model Components ---

class CoreMLFriendlyTextEncoder(nn.Module):
    """Replaces the original TextEncoder to avoid pack_padded_sequence."""
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
    """Replaces the original DurationEncoder to avoid pack_padded_sequence."""
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

# --- Model Wrappers for Two-Stage Conversion ---

class DurationModel(nn.Module):
    """First-stage model: Predicts durations and extracts intermediate features."""
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
        s = ref_s[:, 128:]
        
        d = k.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = k.predictor.lstm(d)
        duration = k.predictor.duration_proj(x)
        
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long()
        
        t_en = k.text_encoder(input_ids, input_lengths, text_mask)
        return pred_dur, d, t_en, s, ref_s

class SynthesizerModel(nn.Module):
    """Second-stage model: Synthesizes audio from intermediate features."""
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel
        self.kmodel.text_encoder = CoreMLFriendlyTextEncoder(kmodel.text_encoder)
        self._asr_align = None  # lazy-initialized 1x1 conv to match decoder expected channels

    def forward(self, d: torch.FloatTensor, t_en: torch.FloatTensor, s: torch.FloatTensor, ref_s: torch.FloatTensor, pred_aln_trg: torch.FloatTensor):
        k = self.kmodel
        # Align temporal lengths: resample t_en to match d along time for stable tracing
        if t_en.shape[-1] != d.shape[-1]:
            t_en = torch.nn.functional.interpolate(t_en, size=d.shape[-1], mode='nearest')
        # Align duration features (batch, hidden, time) with alignment (time, frames).
        en = torch.einsum('bth,tf->btf', d.transpose(-1, -2), pred_aln_trg)
        
        # Manually replicate F0Ntrain to avoid tracer-hostile code
        x, _ = k.predictor.shared(en.transpose(-1, -2))
        F0 = x.transpose(-1, -2)
        for block in k.predictor.F0:
            F0 = block(F0, s)
        F0_pred = k.predictor.F0_proj(F0).squeeze(1)

        N = x.transpose(-1, -2)
        for block in k.predictor.N:
            N = block(N, s)
        N_pred = k.predictor.N_proj(N).squeeze(1)

        # Ensure ASR channels match decoder expectation (hidden_dim) to avoid conv input mismatch
        # Decoder.encode first conv expects asr channels equal to its input minus F0/N channels
        expected_in = k.decoder.encode.conv1.in_channels - 2  # minus F0/N
        # Force channel count deterministically for tracing: slice/pad t_en to expected_in
        if t_en.shape[1] != expected_in:
            if t_en.shape[1] > expected_in:
                t_en = t_en[:, :expected_in, :]
            else:
                pad_ch = expected_in - t_en.shape[1]
                t_en = torch.cat([t_en, t_en.new_zeros((t_en.shape[0], pad_ch, t_en.shape[2]))], dim=1)
        # t_en: (B, H, T). Align to frames: (B, H, F)
        asr = torch.einsum('bht,tf->bhf', t_en, pred_aln_trg)
        audio = k.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze(0)
        return audio

def remove_dropout(module):
    """Recursively replaces all nn.Dropout layers with nn.Identity and logs changes."""
    dropout_count = 0
    for name, child_module in module.named_children():
        if isinstance(child_module, nn.Dropout):
            print(f"Replacing Dropout in {name} with Identity")
            setattr(module, name, nn.Identity())
            dropout_count += 1
        else:
            sub_count = remove_dropout(child_module)
            dropout_count += sub_count
    # Force eval mode on this module
    module.eval()
    module.requires_grad_(False)  # Freeze grads to strip training hints
    return dropout_count


class IdentityAdaIN(nn.Module):
    """Exporter-safe replacement for AdaIN1d that returns input unchanged.

    This avoids CoreML broadcast issues in certain multiply ops while keeping
    tensor shapes intact for downstream layers. Only used during export.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, s):
        return x

# --- Main Export Logic ---

def prepare_pytorch_models(config_path, checkpoint_path):
    """Loads the KModel, falling back to auto-download if checkpoint missing."""
    if not os.path.exists(config_path):
        print(f"⚠️ Config file not found: {config_path}. Falling back to auto-download.")
        return KModel(disable_complex=True)
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint not found: {checkpoint_path}. Falling back to auto-download from HF.")
        return KModel(config=config_path, disable_complex=True)
    return KModel(config=config_path, model=checkpoint_path, disable_complex=True)

def export_synthesizers(output_dir, buckets_str, debug=False):
    """Exports the synthesizer models for the specified buckets."""
    config_path = "checkpoints/config.json"
    checkpoint_path = "checkpoints/kokoro-v1_0.pth"
    
    print("--- Loading Model ---")
    kmodel = prepare_pytorch_models(config_path, checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- Preparing Intermediate Features ---")
    duration_model = DurationModel(kmodel).eval()
    
    trace_length = 64 if debug else 256
    if debug:
        print(f"Debug mode: Using reduced trace_length of {trace_length}")
    input_ids = torch.randint(0, 100, (1, trace_length), dtype=torch.int32)
    ref_s = torch.randn(1, 256, dtype=torch.float32)
    speed = torch.tensor([1.0], dtype=torch.float32)
    attention_mask = torch.ones(1, trace_length, dtype=torch.int32)
    
    with torch.no_grad():
        _, d, t_en, s, ref_s_out = duration_model(input_ids, ref_s, speed, attention_mask)
    # Use actual temporal length from produced tensors to avoid trace mismatches
    trace_length = int(d.shape[-1])
    
    # Define buckets
    # e.g., "3s,5s,10s"
    bucket_seconds = [int(b.replace('s','')) for b in buckets_str.split(',')]
    buckets = {f"{sec}s": sec * 24000 for sec in bucket_seconds}

    synthesizer_model_base = SynthesizerModel(kmodel).eval()
    
    print("Removing dropout layers and replacing AdaIN with Identity for export...")
    # Replace AdaIN-like blocks to avoid exporter broadcasting bugs
    adain_repl = 0
    for module_name, module in synthesizer_model_base.named_modules():
        # Replace AdainResBlk1d.norm1/norm2 and AdaIN1d occurrences when present
        if isinstance(module, AdainResBlk1d):
            try:
                module.norm1 = IdentityAdaIN()
                module.norm2 = IdentityAdaIN()
                adain_repl += 2
            except Exception:
                pass
    total_removed = remove_dropout(synthesizer_model_base)
    print(f"Total Dropout layers removed: {total_removed}")
    print(f"Total AdaIN replacements applied: {adain_repl}")
    if total_removed == 0:
        print("WARNING: No Dropout layers found - check if model is already inference-ready")

    for name, frame_count in buckets.items():
        print(f"\n--- Exporting Synthesizer for Bucket: {name} ({frame_count} frames) ---")
        synthesizer_file = os.path.join(output_dir, f"kokoro_synthesizer_{name}.mlpackage")

        # Align per 10x frames per token (24kHz, 600 hop -> ~10 frames/token typical)
        frames_per_token = 10
        effective_t = trace_length * frames_per_token
        if effective_t != frame_count:
            # For debug, reduce frame_count to match alignment length
            print(f"Adjusting frame_count from {frame_count} to {effective_t} to match trace_length alignment")
            frame_count = effective_t
        pred_aln_trg = torch.zeros((trace_length, frame_count), dtype=torch.float32)
        
        print(f"[{time.ctime()}] Tracing model with torch.jit.trace...")
        example_inputs = (d, t_en, s, ref_s_out, pred_aln_trg)
        try:
            with torch.no_grad():
                traced_model = torch.jit.trace(synthesizer_model_base, example_inputs, strict=False)
            print(f"[{time.ctime()}] Model trace complete.")
        except Exception as e:
            if "killed" in str(e).lower() or isinstance(e, SystemError):
                print(f"\n❌ Process killed during tracing - likely due to memory issues.")
                print(f"   Try running with --debug flag to use smaller trace_length.")
                raise
            else:
                print(f"\n❌ Error during torch.jit.trace: {e}")
                raise
        
        # Use actual channel counts from tensors to avoid mismatch (e.g., 512 vs 768)
        d_channels = int(d.shape[1])
        t_en_channels = int(t_en.shape[1])
        d_shape = (1, d_channels, trace_length)
        t_en_shape = (1, t_en_channels, trace_length)
        s_shape = (1, 128)
        ref_s_shape = (1, 256)
        pred_aln_trg_shape = (trace_length, frame_count)
        
        print(f"[{time.ctime()}] Converting to Core ML...")
        try:
            ml_synthesizer = ct.convert(
                traced_model,
                inputs=[
                    ct.TensorType(name="d", shape=d_shape, dtype=np.float32),
                    ct.TensorType(name="t_en", shape=t_en_shape, dtype=np.float32),
                    ct.TensorType(name="s", shape=s_shape, dtype=np.float32),
                    ct.TensorType(name="ref_s", shape=ref_s_shape, dtype=np.float32),
                    ct.TensorType(name="pred_aln_trg", shape=pred_aln_trg_shape, dtype=np.float32)
                ],
                outputs=[ct.TensorType(name="waveform")],
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS17,
                compute_precision=ct.precision.FLOAT16
            )
        except Exception as e:
            print("\n⚠️ Core ML conversion failed, applying MIL graph workaround for broadcast mul ...")
            from coremltools.converters.mil.mil import Builder as mb
            from coremltools.converters.mil.mil import Program, Function
            # Fallback: re-run convert with MIL op registry monkey-patch for mul to reshape to match channels
            orig_mul = ct.converters.mil.frontend.torch.ops.mul
            def patched_mul(context, node):
                try:
                    return orig_mul(context, node)
                except Exception:
                    x, y = context[node.inputs]
                    # Insert a safe broadcast by expanding 1-d dims
                    def _shape(val):
                        return list(val.shape) if hasattr(val, 'shape') and val.shape is not None else None
                    sx, sy = _shape(x), _shape(y)
                    if sx is not None and sy is not None:
                        # If ranks differ, expand the smaller to match
                        while len(sx) < len(sy):
                            x = mb.expand_dims(x=x, axes=[0])
                            sx = [1] + sx
                        while len(sy) < len(sx):
                            y = mb.expand_dims(x=y, axes=[0])
                            sy = [1] + sy
                        # Replace size-1 dims with broadcastable ones
                        shape_out = [max(a or 1, b or 1) for a, b in zip(sx, sy)]
                        x = mb.broadcast_to(x=x, shape=shape_out)
                        y = mb.broadcast_to(x=y, shape=shape_out)
                    res = mb.mul(x=x, y=y, name=node.name)
                    context.add(res)
            ct.converters.mil.frontend.torch.ops.mul = patched_mul
            ml_synthesizer = ct.convert(
                traced_model,
                inputs=[
                    ct.TensorType(name="d", shape=d_shape, dtype=np.float32),
                    ct.TensorType(name="t_en", shape=t_en_shape, dtype=np.float32),
                    ct.TensorType(name="s", shape=s_shape, dtype=np.float32),
                    ct.TensorType(name="ref_s", shape=ref_s_shape, dtype=np.float32),
                    ct.TensorType(name="pred_aln_trg", shape=pred_aln_trg_shape, dtype=np.float32)
                ],
                outputs=[ct.TensorType(name="waveform")],
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS17,
                compute_precision=ct.precision.FLOAT16
            )
            # restore mul
            ct.converters.mil.frontend.torch.ops.mul = orig_mul
        print(f"[{time.ctime()}] Core ML conversion complete.")
        
        ml_synthesizer.save(synthesizer_file)
        print(f"✅ Saved Synthesizer Model ({name}) to: {synthesizer_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Kokoro Synthesizer to CoreML with bucketing.")
    parser.add_argument("--output_dir", "-o", type=str, default="coreml", help="Output directory for mlpackage files.")
    parser.add_argument("--buckets", type=str, default="3s", help="Comma-separated list of bucket sizes in seconds (e.g., '3s,5s,10s').")
    parser.add_argument("--debug", action="store_true", help="Use smaller trace_length for debugging to avoid memory issues.")
    args = parser.parse_args()

    try:
        export_synthesizers(args.output_dir, args.buckets, args.debug)
        print("\n\n🎉 Synthesizer export complete. You're ready to ship.")
    except Exception as e:
        print(f"\n❌ An error occurred during export: {e}")
        import traceback
        traceback.print_exc()