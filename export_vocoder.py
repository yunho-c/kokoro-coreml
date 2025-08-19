#!/usr/bin/env python3
"""
Kokoro Vocoder Extraction and CoreML Conversion Script

This script extracts the iSTFTNet vocoder (Decoder) from the full Kokoro model
and converts it to a CoreML package optimized for Apple Neural Engine (ANE).

The vocoder is the compute-heavy component that can run efficiently on ANE,
while the text processing components (BERT, LSTM) must remain on CPU.

Architecture Split:
- CPU: Text encoding, prosody prediction, duration alignment
- ANE: Audio synthesis via iSTFTNet vocoder (this script)

Key Technical Details:
- Extracts model.decoder (contains Generator from istftnet.py)
- Uses FP16 precision for ANE optimization
- Handles proper tensor shape layout for ANE memory efficiency
- Creates wrapper for CoreML-compatible I/O format
"""

import torch
import coremltools as ct
import numpy as np
from kokoro import KModel

# ANE-optimized conversion settings
COMPUTE_PRECISION = ct.precision.FLOAT16  # ANE native precision
MINIMUM_DEPLOYMENT_TARGET = ct.target.macOS13  # Match repo macOS 15+, allows FP16 inputs
COMPUTE_UNITS = ct.ComputeUnit.ALL  # Allow ANE + GPU + CPU as needed

# Model architecture constants
class ExportConstants:
    """Constants for vocoder export and CoreML conversion."""
    
    # Sample input dimensions (typical 2-3 second phrase)
    SEQUENCE_LENGTH_INPUT = 400      # Original F0 curve length
    SEQUENCE_LENGTH_ASR = 200        # ASR features after F0/N convolution (half of input)
    
    # Audio parameters
    SAMPLE_RATE = 24000              # Kokoro model sample rate in Hz
    HOP_LENGTH = 600                 # Samples per frame (24kHz / 40fps)
    FRAMES_PER_SECOND = 40           # Frame rate for duration predictions
    
    # Model dimensions
    ASR_FEATURE_DIM = 512            # Acoustic feature dimension
    STYLE_EMBEDDING_DIM = 128        # Voice style embedding size
    MEL_CHANNELS = 80                # Mel-spectrogram channels (unused but standard)
    
    # Conversion targets
    MIN_LENGTH = 64                  # Minimum sequence length for variable input
    MAX_LENGTH = 1024                # Maximum sequence length for variable input
    
    # CoreML optimization
    FALLBACK_TARGET = ct.target.macOS12  # Fallback deployment target for compatibility
    FALLBACK_PRECISION = ct.precision.FLOAT32  # Fallback precision for CPU-only

class VocoderWrapper(torch.nn.Module):
    """
    CoreML-compatible wrapper around the Kokoro decoder (vocoder).
    
    The original decoder expects multiple input tensors with specific shapes.
    This wrapper provides a clean interface that matches CoreML conversion
    requirements and handles proper tensor formatting.
    
    Key Considerations:
    - ANE memory layout: Last dimension should be largest for efficiency
    - Fixed input shapes for better ANE optimization
    - Proper handling of F0 curve and noise parameters
    """
    
    def __init__(self, decoder):
        """
        Initialize vocoder wrapper.
        
        Args:
            decoder: The extracted decoder module from KModel
        """
        super().__init__()
        self.decoder = decoder
        
    def forward(self, asr_4d, f0_curve_4d, n_4d, s):
        """
        Forward pass through the vocoder.
        
        Args:
            asr_4d: Aligned acoustic features, shape (1, 512, 1, T)
            f0_curve_4d: F0/pitch curve, shape (1, 1, 1, T) 
            n_4d: Noise parameters, shape (1, 1, 1, T)
            s: Voice style embedding, shape (1, 128)
            
        Returns:
            audio: Generated waveform, shape (1, 1, audio_length)
        """
        # Squeeze 4D (B,C,1,S)/(B,1,1,S) to decoder's expected shapes
        asr = asr_4d.squeeze(2)               # (1, 512, T)
        f0_curve = f0_curve_4d.squeeze(2).squeeze(1)  # (1, T)
        n = n_4d.squeeze(2).squeeze(1)       # (1, T)
        audio = self.decoder(asr, f0_curve, n, s)
        
        # Ensure output shape is consistent for CoreML
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # Add channel dimension
        
        return audio

class SimpleGeneratorWrapper(torch.nn.Module):
    """
    Simplified wrapper that extracts just the Generator component.
    
    This is a fallback approach that focuses on the core synthesis
    part which should be more ANE-compatible.
    """
    
    def __init__(self, decoder):
        """
        Initialize with just the generator from the decoder.
        
        Args:
            decoder: The decoder module containing the generator
        """
        super().__init__()
        self.generator = decoder.generator
        
    def forward(self, x, s, f0_curve):
        """
        Direct generator forward pass.
        
        Args:
            x: Processed features, shape (1, 512, T) 
            s: Style embedding, shape (1, 128)
            f0_curve: F0 curve, shape (1, T*2) (upsampled)
            
        Returns:
            audio: Generated waveform
        """
        return self.generator(x, s, f0_curve)

class GeneratorWrapper(torch.nn.Module):
    """
    CoreML-friendly wrapper for generator-only path, accepting ANE-friendly 4D inputs.
    """
    def __init__(self, decoder):
        super().__init__()
        self.generator = decoder.generator

    def forward(self, x_4d, s, f0_curve_4d):
        # x_4d: (B, 512, 1, T_asr) → (B, 512, T_asr)
        x = x_4d.squeeze(2)
        # f0_curve_4d: (B, 1, 1, T) → (B, T)
        f0_curve = f0_curve_4d.squeeze(2).squeeze(1)
        return self.generator(x, s, f0_curve)

class GeneratorNoSource(torch.nn.Module):
    """
    Generator variant that accepts precomputed harmonic source features.
    Expects `har` = concat([har_spec, har_phase], dim=1) with exact hn-nsf parity
    computed in PyTorch (same as model.decoder.generator.stft.transform on m_source output).
    """
    def __init__(self, generator: 'Generator'):
        super().__init__()
        # Copy submodules used after source creation
        self.num_kernels = generator.num_kernels
        self.num_upsamples = generator.num_upsamples
        self.noise_convs = generator.noise_convs
        self.noise_res = generator.noise_res
        self.ups = generator.ups
        self.resblocks = generator.resblocks
        self.post_n_fft = generator.post_n_fft
        self.conv_post = generator.conv_post
        self.reflection_pad = generator.reflection_pad

    def forward(self, x, s, har):
        # har is (B, n_fft+2, T)
        for i in range(self.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, s)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x, s)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x, s)
            x = xs / self.num_kernels
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv_post(x)
        # Return spec+phase like original prior to inverse; inverse handled outside of CoreML in this mode
        return x

class DecoderNoSourceWrapper(torch.nn.Module):
    """
    Wraps Decoder to accept precomputed hn-nsf harmonic source features via `har_spec` and `har_phase`.
    CoreML side will not generate source, only consume it, matching PyTorch exactly.
    """
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.gen_no_source = GeneratorNoSource(decoder.generator)

    def forward(self, asr_4d, f0_curve_4d, n_4d, s, har_spec_4d, har_phase_4d):
        # Squeeze 4D back to expected shapes
        asr = asr_4d.squeeze(2)  # (B, 512, T_asr)
        f0_curve = f0_curve_4d.squeeze(2).squeeze(1)  # (B, T)
        n = n_4d.squeeze(2).squeeze(1)  # (B, T)
        # Preprocess F0 and N as in Decoder.forward
        F0 = self.decoder.F0_conv(f0_curve.unsqueeze(1))
        N = self.decoder.N_conv(n.unsqueeze(1))
        x = torch.cat([asr, F0, N], axis=1)
        x = self.decoder.encode(x, s)
        asr_res = self.decoder.asr_res(asr)
        res = True
        for block in self.decoder.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], axis=1)
            x = block(x, s)
            if getattr(block, 'upsample_type', 'none') != 'none':
                res = False
        # Construct har from provided spec+phase
        har_spec = har_spec_4d.squeeze(2)
        har_phase = har_phase_4d.squeeze(2)
        har = torch.cat([har_spec, har_phase], dim=1)
        # Run generator up to spec/phase output
        x = self.gen_no_source(x, s, har)
        # Now apply the same final mapping as original: exp on spec channels, sin on phase channels is done outside CoreML
        return x

class CoreMLFriendlySource(torch.nn.Module):
    """
    CoreML-friendly multi-harmonic source (hn-nsf approx) that avoids unsupported ops.
    - Builds fundamental + overtones from f0 using cumsum/sin
    - Linear + tanh mixdown to single channel (matches original interface)
    - Deterministic noise shaped by uv for stability on Core ML
    """
    def __init__(
        self,
        sampling_rate: float = 24000.0,
        harmonic_num: int = 8,
        voiced_threshold: float = 1.0,
        sine_amp: float = 0.2,
        noise_std: float = 0.001,
    ):
        super().__init__()
        self.sampling_rate = float(sampling_rate)
        self.voiced_threshold = float(voiced_threshold)
        self.sine_amp = float(sine_amp)
        self.noise_std = float(noise_std)
        dim = harmonic_num + 1
        self.merge_tanh = torch.nn.Tanh()
        self.merge_linear = torch.nn.Linear(dim, 1, bias=False)
        # Initialize deterministic averaging weights (no randomness at inference)
        with torch.no_grad():
            inv = torch.reciprocal(torch.arange(1, dim + 1, dtype=torch.float32))
            w = (inv / inv.sum()).unsqueeze(0)  # emphasize low harmonics
            self.merge_linear.weight.copy_(w)
        for p in self.merge_linear.parameters():
            p.requires_grad_(False)
        # Register harmonic multipliers 1..(harmonic_num+1) as buffer
        harmonics = torch.arange(1, dim + 1, dtype=torch.float32).view(1, 1, dim)
        self.register_buffer("harmonics", harmonics)

    def forward(self, f0_upsampled):
        # f0_upsampled: (batch, length, 1)
        dtype = f0_upsampled.dtype
        device = f0_upsampled.device
        f0 = torch.clamp(f0_upsampled, min=0.0)
        # Broadcast f0 across harmonic dimension WITHOUT multiply: cumulative sum builds (i+1)*f0
        H = self.harmonics.numel()
        f0_rep = f0.expand(-1, -1, H).contiguous()  # (B, L, H)
        f0_h = torch.cumsum(f0_rep, dim=2)
        # Phase integration in radians per sample for each harmonic
        delta_phase = (f0_h / self.sampling_rate) * (2.0 * torch.pi)
        phase = torch.cumsum(delta_phase, dim=1)  # (B, L, H)
        sines = torch.sin(phase) * self.sine_amp  # (B, L, H)
        # Mixdown harmonics → 1 channel
        sine_merge = self.merge_tanh(self.merge_linear(sines))  # (B, L, 1)
        # uv and simple deterministic noise shaped by uv
        uv = (f0_upsampled > self.voiced_threshold).to(dtype)
        # Deterministic pseudo-noise from a higher frequency sinusoid
        noise_raw = torch.sin(phase * 13.0)[..., :1]  # higher frequency pseudo-noise
        noise_amp = uv * self.noise_std + (1.0 - uv) * (self.noise_std * 2.0)
        noise = noise_amp * noise_raw
        return sine_merge, noise, uv

def inspect_model_structure(model):
    """Inspect the model structure to understand the decoder architecture.
    
    This helps identify the exact input shapes and requirements for
    the decoder component that we'll be extracting.
    
    Args:
        model: The loaded KModel instance from kokoro.model
        
    Returns:
        decoder: The extracted decoder module for further processing
        
    Called by:
        main(): During model analysis before conversion
        
    Process:
        1. Analyzes top-level model components (bert, predictor, decoder)
        2. Inspects decoder submodule architecture  
        3. Returns decoder reference for conversion pipeline
    """
    print("\n🔍 Model Structure Analysis:")
    print(f"Model type: {type(model).__name__}")
    print("\nMain components:")
    for name, module in model.named_children():
        print(f"  - {name}: {type(module).__name__}")
        
    print(f"\n📊 Decoder details:")
    decoder = model.decoder
    print(f"Decoder type: {type(decoder).__name__}")
    print("Decoder submodules:")
    for name, module in decoder.named_children():
        print(f"  - {name}: {type(module).__name__}")
        
    return decoder

def create_sample_inputs():
    """Create realistic sample inputs that match the decoder's expected format.
    
    These inputs are based on the actual data flow from the full model:
    - asr: Aligned acoustic features from text encoder
    - f0_curve: F0/pitch predictions from prosody predictor  
    - noise: Noise parameters for vocoder
    - style: Voice style embedding (first 128 dims of ref_s)
    
    Returns:
        dict: Sample inputs for tracing with proper tensor shapes
        
    Called by:
        extract_and_convert_vocoder(): To create dummy inputs for torch.jit.trace
        export_decoder_with_har_input(): For HAR variant tracing
        
    Shape Rationale:
        - F0 and N use full temporal resolution
        - ASR features are downsampled 2x due to decoder stride-2 convolutions
        - 4D tensor layout (B,C,1,S) optimizes ANE memory access patterns
    """
    # Sample inputs matching decoder expectations
    sample_inputs = {
        "asr": torch.randn(1, ExportConstants.ASR_FEATURE_DIM, 1, ExportConstants.SEQUENCE_LENGTH_ASR),
        "f0_curve": torch.randn(1, 1, 1, ExportConstants.SEQUENCE_LENGTH_INPUT),
        "n": torch.randn(1, 1, 1, ExportConstants.SEQUENCE_LENGTH_INPUT),
        "s": torch.randn(1, ExportConstants.STYLE_EMBEDDING_DIM)
    }
    
    print("\n📝 Sample Input Shapes:")
    for name, tensor in sample_inputs.items():
        print(f"  - {name}: {tensor.shape}")
        
    return sample_inputs

def extract_and_convert_vocoder(model):
    """Extract the decoder and convert it to CoreML format.
    
    This is the main conversion process that:
    1. Extracts the decoder module 
    2. Wraps it for CoreML compatibility
    3. Traces with sample inputs
    4. Converts to CoreML with ANE optimization
    
    Args:
        model: The loaded KModel instance from kokoro.model
        
    Returns:
        str: Path to the saved CoreML package (.mlpackage)
        
    Called by:
        main(): Primary export pathway for full decoder conversion
        
    Process Flow:
        1. Extract decoder from KModel.decoder
        2. Wrap in VocoderWrapper for CoreML-compatible I/O
        3. Generate representative sample inputs via create_sample_inputs()
        4. Trace with torch.jit.trace for graph capture
        5. Convert to CoreML mlprogram with FP16 precision
        6. Apply ANE optimizations and fallback handling
        7. Save to coreml/KokoroVocoder.mlpackage
        
    Optimization Strategy:
        - Primary: FP16 precision + ALL compute units for ANE acceleration
        - Fallback: FP32 precision + CPU_ONLY for compatibility
        - Input shapes optimized for ANE memory layout (B,C,1,S)
    """
    print("\n🔧 Extracting decoder module...")
    decoder = model.decoder
    
    # Force full decoder conversion for correct tensor alignment
    print("🔄 Forcing full decoder conversion (generator-only path mismatched shapes)...")
    wrapper = VocoderWrapper(decoder)
    wrapper.eval()
    conversion_mode = "full_decoder"
    print("✅ Full decoder extracted and wrapped")
    # Replace source module with CoreML-friendly implementation (avoid unsupported ops)
    # Use the exact hn-nsf source implementation for parity with PyTorch.
    # Do NOT replace generator.m_source; preserving original SourceModuleHnNSF.
    print("🎯 Using exact hn-nsf source from PyTorch model (no replacement)")
    
    # Create sample inputs for tracing
    sample_inputs = create_sample_inputs()
    
    # Convert to tuple for tracing (matches forward signature)
    if conversion_mode == "full_decoder":
        trace_inputs = (
            sample_inputs["asr"],
            sample_inputs["f0_curve"], 
            sample_inputs["n"],
            sample_inputs["s"]
        )
    else:  # generator_only (unused in forced mode)
        raise RuntimeError("generator_only mode disabled due to shape mismatches")
    
    print("\n⚡ Tracing model with torch.jit.trace...")
    try:
        # Use torch.jit.trace with warnings suppressed
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress tracing warnings
            # Disable problematic source noise path by freezing uv/noise to zeros during trace
            # to avoid unsupported multiply op in converter.
            traced_vocoder = torch.jit.trace(wrapper, trace_inputs, strict=False)
        print("✅ Model traced successfully")
    except Exception as e:
        print(f"❌ Tracing failed: {e}")
        print("This may indicate incompatible operations for CoreML conversion")
        raise
    
    print("\n🍎 Converting to CoreML...")
    
    # Define CoreML input specifications with proper types and shapes
    # Ensure generator input alignment: f0 length must be 2x asr temporal dim due to upsampling in Generator
    sequence_length_asr = ExportConstants.SEQUENCE_LENGTH_ASR
    sequence_length_input = ExportConstants.SEQUENCE_LENGTH_INPUT
    
    if conversion_mode == "full_decoder":
        inputs = [
            ct.TensorType(name="asr", shape=(1, ExportConstants.ASR_FEATURE_DIM, 1, sequence_length_asr), dtype=np.float32),
            ct.TensorType(name="f0_curve", shape=(1, 1, 1, sequence_length_input), dtype=np.float32),
            ct.TensorType(name="n", shape=(1, 1, 1, sequence_length_input), dtype=np.float32), 
            ct.TensorType(name="s", shape=(1, ExportConstants.STYLE_EMBEDDING_DIM), dtype=np.float32)
        ]
    else:  # generator_only
        inputs = [
            ct.TensorType(name="x", shape=(1, ExportConstants.ASR_FEATURE_DIM, 1, sequence_length_asr), dtype=np.float16),
            ct.TensorType(name="s", shape=(1, ExportConstants.STYLE_EMBEDDING_DIM), dtype=np.float16),
            ct.TensorType(name="f0_curve", shape=(1, 1, 1, sequence_length_input), dtype=np.float16)
        ]
    
    # Convert with ANE optimization settings
    try:
        coreml_model = ct.convert(
            traced_vocoder,
            inputs=inputs,
            convert_to="mlprogram",
            compute_precision=COMPUTE_PRECISION,
            minimum_deployment_target=MINIMUM_DEPLOYMENT_TARGET,
            compute_units=COMPUTE_UNITS
        )
        print("✅ CoreML conversion successful with ANE optimization")
    except Exception as e:
        print(f"⚠️ ANE conversion failed: {e}")
        print("🔄 Trying fallback conversion with CPU-only...")
        coreml_model = ct.convert(
            traced_vocoder,
            inputs=inputs,
            convert_to="mlprogram",
            compute_precision=ExportConstants.FALLBACK_PRECISION,
            minimum_deployment_target=ExportConstants.FALLBACK_TARGET,
            compute_units=ct.ComputeUnit.CPU_ONLY
        )
        print("✅ CoreML conversion successful with CPU fallback")
    
    # Add model metadata
    coreml_model.author = "Kokoro TTS - Vocoder Module"
    if conversion_mode == "full_decoder":
        coreml_model.short_description = "Complete iSTFTNet decoder for high-quality audio synthesis on Apple Neural Engine"
    else:
        coreml_model.short_description = "iSTFTNet generator core for high-quality audio synthesis on Apple Neural Engine"
    coreml_model.version = "1.0.0"
    
    # Normalize I/O naming for app integration
    try:
        spec = coreml_model.get_spec()
        if spec.description.output and spec.description.output[0].name != "waveform":
            spec.description.output[0].name = "waveform"
        coreml_model = ct.models.MLModel(spec)
    except Exception as e:
        print(f"⚠️ Could not rename output to 'waveform': {e}")
    
    # Save the model under coreml/ directory
    output_path = "coreml/KokoroVocoder.mlpackage"
    import os
    os.makedirs("coreml", exist_ok=True)
    coreml_model.save(output_path)
    
    print(f"✅ CoreML model saved to: {output_path}")
    
    # Verify the conversion
    print("\n🧪 Verifying CoreML model...")
    # Simple load check
    try:
        _ = ct.models.MLModel(output_path)
        print("✅ CoreML model load verification successful")
    except Exception as e:
        print(f"⚠️  Load verification failed: {e}")
        print("Model was saved but may have issues")
    
    return output_path

def export_decoder_with_har_input(model):
    print("\n🚀 Exporting Decoder variant that accepts hn-nsf source as input (exact parity)")
    decoder = model.decoder
    wrapper = DecoderNoSourceWrapper(decoder).eval()
    sample_inputs = create_sample_inputs()
    # Create dummy har from PyTorch path to trace shapes
    with torch.no_grad():
        gen = decoder.generator
        # Build realistic har via exact PyTorch path
        f0 = sample_inputs["f0_curve"].squeeze(2).squeeze(1)
        f0_up = gen.f0_upsamp(f0[:, None]).transpose(1, 2)
        har_source, _, _ = gen.m_source(f0_up)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = gen.stft.transform(har_source)
    trace_inputs = (
        sample_inputs["asr"],
        sample_inputs["f0_curve"],
        sample_inputs["n"],
        sample_inputs["s"],
        har_spec.unsqueeze(2),  # add 4th dim back: (B, C, 1, T)
        har_phase.unsqueeze(2),
    )
    print("⚡ Tracing DecoderNoSourceWrapper...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, trace_inputs, strict=False)
    import coremltools as ct
    import numpy as np
    n_fft = decoder.generator.post_n_fft
    asr_shape = (1, 512, 1, 200)
    f0_shape = (1, 1, 1, 400)
    n_shape = (1, 1, 1, 400)
    s_shape = (1, 128)
    har_c = (n_fft // 2 + 1)
    # Match exact PyTorch hn-nsf STFT time length for f0_win=400
    har_t = 24001
    inputs = [
        ct.TensorType(name="asr", shape=asr_shape, dtype=np.float32),
        ct.TensorType(name="f0_curve", shape=f0_shape, dtype=np.float32),
        ct.TensorType(name="n", shape=n_shape, dtype=np.float32),
        ct.TensorType(name="s", shape=s_shape, dtype=np.float32),
        ct.TensorType(name="har_spec", shape=(1, har_c, 1, har_t), dtype=np.float32),
        ct.TensorType(name="har_phase", shape=(1, har_c, 1, har_t), dtype=np.float32),
    ]
    print("🍎 Converting DecoderNoSourceWrapper to CoreML (mlprogram, FP16)...")
    ml = ct.convert(
        traced,
        inputs=inputs,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )
    # Output is raw x (spec+phase pre-nonlinearity); we keep it as generic output name
    out_path = "coreml/KokoroDecoder_HAR.mlpackage"
    import os
    os.makedirs("coreml", exist_ok=True)
    ml.save(out_path)
    print(f"✅ Saved Decoder_HAR CoreML model to: {out_path}")
    return out_path

def _compute_har_shapes_for_f0_len(decoder, f0_len: int):
    """Compute (har_c, har_t) given a desired f0 length using exact PyTorch path."""
    with torch.no_grad():
        gen = decoder.generator
        device = next(gen.parameters()).device
        f0 = torch.zeros((1, f0_len), dtype=torch.float32, device=device)
        f0_up = gen.f0_upsamp(f0[:, None]).transpose(1, 2)
        har_source, _, _ = gen.m_source(f0_up)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = gen.stft.transform(har_source)
    har_c = har_spec.shape[1]
    har_t = har_spec.shape[2]
    return har_c, har_t


def export_decoder_har_bucket(decoder, seconds: int, output_dir: str = "coreml"):
    """
    Export a Decoder variant that consumes precomputed hn-nsf features for a single-shot bucket.

    - seconds: bucket duration in seconds (e.g., 5, 15, 30)
    """
    print(f"\n🚀 Exporting Decoder_HAR bucket: {seconds}s")
    wrapper = DecoderNoSourceWrapper(decoder).eval()

    # Determine target temporal sizes
    # Empirical mapping from earlier 5s window: f0_len=400 → asr_len=200
    f0_per_sec = 80  # 24kHz / 300 samples per f0 frame ≈ 80 Hz
    f0_len = int(seconds * f0_per_sec)
    asr_len = f0_len // 2

    # Build realistic dummy inputs and compute exact har shapes
    sample_inputs = {
        "asr": torch.zeros(1, 512, 1, asr_len, dtype=torch.float32),
        "f0_curve": torch.zeros(1, 1, 1, f0_len, dtype=torch.float32),
        "n": torch.zeros(1, 1, 1, f0_len, dtype=torch.float32),
        "s": torch.zeros(1, 128, dtype=torch.float32),
    }
    har_c, har_t = _compute_har_shapes_for_f0_len(decoder, f0_len)

    trace_inputs = (
        sample_inputs["asr"],
        sample_inputs["f0_curve"],
        sample_inputs["n"],
        sample_inputs["s"],
        torch.zeros(1, har_c, 1, har_t, dtype=torch.float32),
        torch.zeros(1, har_c, 1, har_t, dtype=torch.float32),
    )

    print("⚡ Tracing DecoderNoSourceWrapper for bucket...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, trace_inputs, strict=False)

    print("🍎 Converting to CoreML (mlprogram, FP16)...")
    ml = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="asr", shape=(1, 512, 1, asr_len), dtype=np.float32),
            ct.TensorType(name="f0_curve", shape=(1, 1, 1, f0_len), dtype=np.float32),
            ct.TensorType(name="n", shape=(1, 1, 1, f0_len), dtype=np.float32),
            ct.TensorType(name="s", shape=(1, 128), dtype=np.float32),
            ct.TensorType(name="har_spec", shape=(1, har_c, 1, har_t), dtype=np.float32),
            ct.TensorType(name="har_phase", shape=(1, har_c, 1, har_t), dtype=np.float32),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )

    import os
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"KokoroDecoder_HAR_{seconds}s.mlpackage")
    ml.save(out_path)
    print(f"✅ Saved Decoder_HAR bucket to: {out_path}")
    return out_path


def export_decoder_har_buckets(model, seconds_list):
    """Export multiple Decoder_HAR bucket variants in one go."""
    decoder = model.decoder
    exported = []
    for sec in seconds_list:
        try:
            exported.append(export_decoder_har_bucket(decoder, sec))
        except Exception as e:
            print(f"⚠️ Failed to export {sec}s bucket: {e}")
            import traceback
            traceback.print_exc()
    return exported


def main():
    """
    Main execution function for vocoder extraction and conversion.
    """
    print("🚀 Kokoro Vocoder Extraction & CoreML Conversion")
    print("=" * 50)

    # Lightweight flag parsing without adding dependencies
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-vocoder", action="store_true", help="Export KokoroVocoder.mlpackage (full decoder wrapper)")
    parser.add_argument("--export-decoder-har", action="store_true", help="Export Decoder_HAR window model (5s window)")
    parser.add_argument("--har-buckets", type=str, default="", help="Comma-separated seconds for Decoder_HAR buckets, e.g. '5,15,30'")
    args = parser.parse_args()

    print("\n📦 Loading full Kokoro model...")
    try:
        # Load the model with CoreML-friendly settings
        # disable_complex=True avoids complex ops (e.g., angle) that break Torch->CoreML
        model = KModel(disable_complex=True).to('cpu').eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Inspect the model structure to understand the decoder
    decoder = inspect_model_structure(model)

    # Export paths depending on flags
    try:
        if args.export_vocoder:
            output_path = extract_and_convert_vocoder(model)
            print(f"\n🎉 Conversion Complete!")
            print(f"📁 CoreML vocoder saved to: {output_path}")
            print("\nNext steps:")
            print("1. Test the vocoder with test_ane_pipeline.py")
            print("2. Verify ANE usage with Instruments or powermetrics")
            print("3. Compare performance vs CPU-only pipeline")

        if args.export_decoder_har:
            export_decoder_with_har_input(model)

        if args.har_buckets:
            seconds = [int(s.strip().replace('s','')) for s in args.har_buckets.split(',') if s.strip()]
            export_decoder_har_buckets(model, seconds)

        if not (args.export_vocoder or args.export_decoder_har or args.har_buckets):
            # Default behavior remains the same as before
            output_path = extract_and_convert_vocoder(model)
            print(f"\n🎉 Conversion Complete!")
            print(f"📁 CoreML vocoder saved to: {output_path}")

    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()