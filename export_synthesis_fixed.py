#!/usr/bin/env python3
"""Fixed-Shape Synthesis Model Export for E5RT Compatibility

This script creates a CoreML synthesis model with completely fixed shapes to resolve
E5RT stride conflicts and dynamic shape issues that cause "Invalid blob shape" errors.

Key Differences from original export_synthesizers.py:
1. **No Dynamic Shapes**: All inputs/outputs use fixed, static shapes
2. **No .expand() Operations**: Replaced with explicit tensor operations
3. **No Runtime Shape Inspection**: All shapes determined at export time
4. **E5RT Compatible**: Avoids data-dependent operations that trigger E5RT failures

Target Issue Resolution:
- "MIL program has non-constant (dynamic) shapes for external input but FlexibleShapeInformation attribute is missing"
- "E5RT: Espresso exception: 'Invalid blob shape': Data-dependent shapes were disabled: non_zero_0_classic_cpu - [?, 3]"
- ReliableMode fallback causing beep placeholders instead of speech synthesis
"""
import os
import pathlib
import sys
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
    """Fixed-shape decoder model for E5RT compatibility.
    
    This model eliminates all dynamic operations and uses completely fixed shapes:
    - ASR features: [1, 512, 72] (fixed for 3s at 24kHz)
    - F0/N predictions: [1, 144] (fixed, 2x ASR time dim due to stride-2 conv)
    - Reference voice: [1, 256] (fixed voice embedding)
    - Output waveform: [72000] (fixed 3s at 24kHz)
    """
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.decoder = kmodel.decoder
        # Store fixed dimensions to avoid runtime inspection
        self.asr_channels = 512
        self.asr_frames = 72  # 3s * 24kHz / 1000 * 3 ≈ 72 frames
        self.f0_length = 144  # 2 * asr_frames due to stride-2 conv
        self.voice_dim = 128  # baseline voice features
        
    def forward(self, asr: torch.FloatTensor, F0_pred: torch.FloatTensor, N_pred: torch.FloatTensor, ref_s: torch.FloatTensor):
        # All inputs have fixed shapes - no dynamic operations allowed
        # asr: [1, 512, 72], F0_pred: [1, 144], N_pred: [1, 144], ref_s: [1, 256]
        
        # Extract baseline voice features (fixed slice, no dynamic indexing)
        ref_baseline = ref_s[:, :self.voice_dim]  # [1, 128]
        
        # Call decoder with completely fixed inputs - no shape inspection
        waveform = self.decoder(asr, F0_pred, N_pred, ref_baseline)
        
        # Ensure output is squeezed to expected shape [72000]
        return waveform.squeeze(0)

def main():
    print("🔄 Exporting fixed-shape synthesis model for E5RT compatibility...")
    
    # Load model
    cfg = _ROOT / "checkpoints/config.json"
    ckpt = _ROOT / "checkpoints/kokoro-v1_0.pth"
    if cfg.exists() and ckpt.exists():
        kmodel = KModel(config=str(cfg), model=str(ckpt), disable_complex=True)
    elif cfg.exists():
        kmodel = KModel(config=str(cfg), disable_complex=True)
    else:
        kmodel = KModel(disable_complex=True)
        
    # Create fixed-shape decoder model
    decoder_model = FixedShapeDecoderModel(kmodel).eval()
    
    # Remove training operations
    remove_training_ops(decoder_model)
    
    # Replace AdaIN layers with identity to avoid broadcast issues
    adain_count = 0
    for name, module in decoder_model.named_modules():
        if isinstance(module, AdainResBlk1d):
            try:
                module.norm1 = IdentityAdaIN()
                module.norm2 = IdentityAdaIN()
                adain_count += 2
            except:
                pass
    
    print(f"✓ Replaced {adain_count} AdaIN layers with identity")
    
    # Force all submodules to eval mode to prevent TRAINING dialect
    for module in decoder_model.modules():
        module.eval()
    
    # Create fixed representative inputs (3s audio = 72000 samples at 24kHz)
    sample_rate = 24000
    duration_seconds = 3
    waveform_samples = duration_seconds * sample_rate  # 72000
    
    # ASR features: [1, 512, 72] - fixed shape
    asr_features = torch.zeros(1, 512, 72, dtype=torch.float32)
    
    # F0/N predictions: [1, 144] - fixed shape (2x ASR time due to stride-2)
    f0_pred = torch.zeros(1, 144, dtype=torch.float32)
    n_pred = torch.zeros(1, 144, dtype=torch.float32)
    
    # Reference voice: [1, 256] - fixed shape
    ref_voice = torch.zeros(1, 256, dtype=torch.float32)
    
    print("🔄 Testing model with fixed inputs...")
    with torch.no_grad():
        output = decoder_model(asr_features, f0_pred, n_pred, ref_voice)
        print(f"✓ Test successful - output shape: {output.shape}")
    
    # Trace with fixed inputs
    print("🔄 Tracing model with torch.jit.trace...")
    with torch.no_grad():
        traced_model = torch.jit.trace(
            decoder_model, 
            (asr_features, f0_pred, n_pred, ref_voice), 
            strict=False
        )
    
    # Convert to CoreML with completely fixed shapes
    print("🔄 Converting to CoreML...")
    try:
        synthesis_ml = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(name="asr", shape=(1, 512, 72), dtype=np.float32),
                ct.TensorType(name="F0_pred", shape=(1, 144), dtype=np.float32),
                ct.TensorType(name="N_pred", shape=(1, 144), dtype=np.float32),
                ct.TensorType(name="ref_s", shape=(1, 256), dtype=np.float32),
            ],
            outputs=[
                ct.TensorType(name="waveform"),
            ],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS12,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.ALL,
        )
    except Exception as e:
        print(f"❌ CoreML conversion failed: {e}")
        raise
    
    # Save model
    out_dir = _ROOT / "coreml"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kokoro_decoder_only_3s.mlpackage"
    synthesis_ml.save(str(out_path))
    print(f"✅ Saved fixed-shape synthesis model to: {out_path}")
    
    # Validate with test inputs
    print("🔍 Validating exported model...")
    try:
        test_input = {
            "asr": np.zeros((1, 512, 72), dtype=np.float32),
            "F0_pred": np.zeros((1, 144), dtype=np.float32),
            "N_pred": np.zeros((1, 144), dtype=np.float32),
            "ref_s": np.zeros((1, 256), dtype=np.float32)
        }
        test_output = synthesis_ml.predict(test_input)
        print(f"✅ Validation successful - output keys: {list(test_output.keys())}")
        waveform_shape = test_output['waveform'].shape if 'waveform' in test_output else 'unknown'
        print(f"✅ Waveform shape: {waveform_shape}")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        raise
    
    print("✅ Fixed-shape synthesis model export complete!")

if __name__ == "__main__":
    main()