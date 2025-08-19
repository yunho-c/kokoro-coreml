# Engineering Spec: Rebuilding the Kokoro Vocoder Source Module for ANE

**Date:** 2025-08-18  
**Author:** Andy Hertzfeld  
**Status:** **Urgent** - Blocker for V1 Audio Quality

## 1. Objective

The current CoreML-converted vocoder (`KokoroVocoder.mlpackage`) produces audio with a thin, robotic, and "buzzy" quality. This is because we bypassed the original, complex harmonic and noise generation module (`generator.m_source`) with a placeholder `DummySource` to achieve an initial conversion.

This task is to replace that placeholder with a high-fidelity, 1:1 functional equivalent of the original PyTorch module, rebuilt as a custom composite operator in CoreML that is fully compatible with the Apple Neural Engine (ANE).

The goal is to achieve audio quality from our CoreML pipeline that is perceptually identical to the original PyTorch model, while retaining the performance benefits of ANE acceleration.

## 2. Technical Strategy: Deconstruct and Rebuild

We will use CoreML's [Composite Operator](https://coremltools.readme.io/docs/composite-operators) feature. This allows us to define a new, complex operation by composing a sequence of simpler, ANE-safe primitives. This is the standard, sanctioned way to handle unsupported PyTorch operations without sacrificing ANE compatibility.

### Phase 1: Deconstruction and Analysis
**Goal:** Create a complete mathematical and operational blueprint of the original `generator.m_source` module.

1.  **Locate the Source:** The target code is in the `kokoro.istftnet.SourceModuleHnSinc` class within the `kokoro-coreml` submodule.
2.  **Trace the Data Flow:** The engineer must meticulously trace the `forward` pass of this module. Identify every input, every output, and every intermediate tensor.
3.  **Map the Operations:** For every step in the `forward` pass, identify the exact PyTorch operation being used (e.g., `torch.cumsum`, `torch.sin`, `F.leaky_relu`, `F.pad`, various convolutions).
4.  **Document Tensor Shapes:** Document the exact shape of every tensor at every step of the process. This is critical for debugging.
5.  **Identify ANE-Unsafe Operations:** The key task is to identify which of these operations are not directly supported by the ANE and caused the original conversion to fail. The likely culprits are custom activation functions or complex slicing/padding logic.

### Phase 2: Rebuilding with CoreML Primitives
**Goal:** Implement a new `TalkToMeSource` module in Python as a custom composite operator.

1.  **Use `coremltools.converters.mil`:** The entire module will be rebuilt using the MIL (Model Intermediate Language) builder (`mb`).
2.  **Translate Operations 1:1:** For each operation identified in Phase 1, find the corresponding ANE-safe primitive in the MIL builder.
    -   `torch.sin` -> `mb.sin`
    -   `torch.cumsum` -> `mb.cumsum`
    -   `F.pad` -> `mb.pad`
    -   `nn.Conv1d` -> `mb.conv`
    -   ... and so on.
3.  **Register as a Custom Operator:** The new `TalkToMeSource` class will be registered with the CoreML converter using the `@register_torch_op` decorator. This tells the converter to use our custom implementation whenever it encounters the `SourceModuleHnSinc` class during tracing.
4.  **Handle Shape and Data Types:** Pay meticulous attention to tensor shapes and data types (`fp16` vs `fp32`) throughout the MIL graph to ensure compatibility.

### Phase 3: Integration and Verification
**Goal:** Export a new vocoder model and verify its quality and performance.

1.  **Modify `export_vocoder.py`:**
    -   Import the new `TalkToMeSource` module.
    -   Ensure the `@register_torch_op` decorator is correctly pointing to the original `SourceModuleHnSinc` class.
    -   Remove the old `DummySource` monkey-patching logic.
2.  **Export the New Model:** Run the modified export script to generate a new `KokoroVocoder_v2.mlpackage`.
3.  **Verify Audio Quality (Subjective):** Run the `test_ane_pipeline.py` script with the new model. The output audio in `kokoro-coreml/outputs/` should sound rich, pitched, and natural, without the "buzzy" artifact. It should be indistinguishable from the pure PyTorch baseline.
4.  **Verify ANE Execution (Objective):** Use Instruments and `powermetrics` to confirm that the new, complex `TalkToMeSource` operator is still executing entirely on the ANE. There should be no regressions in performance or fallbacks to the CPU/GPU.

## 3. Deliverables

1.  **A modified `export_vocoder.py`** script containing the implementation of the `TalkToMeSource` composite operator.
2.  **A new `KokoroVocoder_v2.mlpackage`** file that produces high-fidelity audio.
3.  **An update in `kokoro-coreml/docs/learnings.md`** documenting the final, successful conversion strategy.

This is a challenging but critical task. The success of our V1 product depends on getting the audio quality right.

## 4. Progress Log — 2025‑08‑19

- Implemented alternative path avoiding a full CoreML source rebuild by exporting `Decoder_HAR` variants that consume PyTorch‑computed hn‑nsf features (`har_spec`, `har_phase`).
- Added single‑shot bucket export (`--har-buckets 5,15,30`) and integrated single‑call inference in `test_ane_pipeline.py`.
- Latency improved materially (CoreML bucket RTF ~0.05 vs PyTorch ~0.15 for 23.3s clip), but audio quality is still subpar on some utterances.
- Observed artifacts point to temporal alignment mismatch in bucket inputs rather than source parity; exact hn‑nsf and inverse STFT are run in PyTorch.

### Next
- Add a per‑frame spectral parity test harness comparing PyTorch decoder vs CoreML Decoder_HAR output `x` given identical `(asr, F0, N, har_spec, har_phase, s)`.
- If mismatch localized to `x` split or residual addition, adjust bucket export to lock post_n_fft and residual timing; otherwise, proceed with composite operator rebuild of the source to regain internal alignment guarantees.
