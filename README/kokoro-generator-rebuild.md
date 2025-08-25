# Engineering Spec: Rebuilding the Kokoro Vocoder Source Module for ANE

**Date:** 2025-08-18  
**Author:** Andy Hertzfeld  
**Status:** ✅ **Done** — V1 goal achieved via alternative path

## 1. Objective

Originally, we planned to rebuild Kokoro's `generator.m_source` as a Core ML composite operator to eliminate artifacts from the placeholder source used during export. The goal was parity audio quality with ANE acceleration.

We achieved the V1 goal via an alternative approach: exporting Decoder_HAR bucket models that accept PyTorch‑computed `har_spec` and `har_phase` (exact hn‑nsf parity), then running a single CoreML pass per long segment with minimal overlap. This path delivers high throughput on ANE with acceptable quality for V1.

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

Delivered for V1 (alternative path):
1.  Decoder_HAR bucket models (5s/15s/30s) as `.mlpackage` artifacts
2.  Single‑shot and grouped decode paths in `test_ane_pipeline.py` with overlap‑add stitching
3.  Benchmarks and warmed latency breakdown (ANE vs CPU) recorded in `docs/learnings.md`

Deferred beyond V1:
1.  Full Core ML composite operator rebuild of `generator.m_source` (kept as a quality roadmap item)

This is a challenging but critical task. The success of our V1 product depends on getting the audio quality right.

## 4. Resolution & Benchmarks — 2025‑08‑19

We are marking this effort as Done for V1 by adopting the Decoder_HAR bucket path. For the test utterance (~23.7 s audio):

- 5s bucket: ~1.35 s total (warmed avg), RTF ≈ 0.057
- 15s bucket: ~1.41 s total (warmed avg), RTF ≈ 0.060
- 30s bucket: ~1.38 s total (warmed avg), RTF ≈ 0.058

Breakdown (typical warmed share across buckets):
- CoreML predict (ANE): ~0.25–0.31 s
- CPU prep (hn‑nsf + STFT): ~0.15–0.17 s
- Inverse STFT (CPU): ~0.02–0.03 s
- Remainder (orchestration / IO / overlap): ~0.55–0.60 s

Takeaways:
- User‑visible wait is sub‑second on warmed runs for most incremental segments; entire 23–24 s clip synthesizes in ~1.3–1.4 s.
- 15–30 s buckets are both efficient; 5 s incurs extra overhead from more windows and overlap.
- Audio quality is acceptable for V1 using exact hn‑nsf features from PyTorch; a full Core ML source rebuild remains on the post‑V1 roadmap if needed to further close any residual gaps.
