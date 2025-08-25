# TalkToMe CoreML TTS — Problem Summary and Investigation Brief

## Executive Summary

- We ship a two-stage, CoreML-based Kokoro TTS pipeline on macOS:
  - Duration model (variable length text → durations + intermediate features)
  - Synthesizer model (fixed-length bucket → waveform)
- Models load and preflight passes, but first real synthesis fails with:
  - Error: `Cannot retrieve vector from IRValue format int32`
  - App “beachballs” (UI stalls following the error)
- Our documentation and experiments strongly indicate this is a tensor shape mismatch between duration outputs and synthesizer inputs (not a dtype issue), exacerbated by bundling/compilation pitfalls and past exporter artifacts.
- We added runtime tensor adaptation and comprehensive logging to align shapes exactly to what the synthesizer declares, but we still need an external deep dive to confirm model I/O contracts, toolchain behavior, and eradicate remaining mismatches.

## What Works vs. What Fails

- Works
  - Model discovery and runtime compilation of `.mlpackage` via `MLModel.compileModel(at:)`
  - Duration preflight with short token sequence
  - Duration prediction timings ~20 ms (warmed)
  - Loader logs show correct `.mlpackage` model paths being compiled to temp `.mlmodelc`
- Fails
  - First real synthesizer prediction with duration outputs and alignment matrix → `IRValue int32` error
  - Historically: stale `.mlmodelc`, ANE/GPU width limit, BNNS tile(reps) validation, E5ML flexible-shape strides

## Current Architecture (High-Level)

- Two-stage CoreML pipeline (Kokoro-inspired):
  1) Duration model outputs: `pred_dur`, `d`, `t_en`, `s` (and optionally `ref_s` passthrough)
  2) Client builds alignment matrix `pred_aln_trg` from `pred_dur` to target bucket frames
  3) Synthesizer model consumes: `d`, `t_en`, `s`, `ref_s`, `pred_aln_trg` and produces waveform
- Buckets used: 5 s (currently preferred), 10 s, 20 s (larger ones can hit GPU width limits if GPU fallback happens)

## Environment & Toolchain

- Hardware: Apple Silicon (M2 Ultra primary), macOS 15.6 (24G84)
- Python venv: `torch==2.5.0`, `coremltools==8.3.0`, `numpy==1.26.4`, `safetensors`, `soundfile`
- Export scripts (kokoro-coreml):
  - Duration: `python examples/export_coreml.py --output_dir ../coreml --duration_only`
  - Synthesizer: `python export_synthesizers.py --buckets="5s" --debug --output_dir ../coreml`
- App bundling: `.mlpackage` copied into `Contents/Resources/coreml/`; runtime compiles to temp `.mlmodelc` before loading.

## Repro Steps (Current)

1) Launch TalkToMe (Debug build)
2) Startup logs show:
   - Bundle scan finds `.mlpackage` models (duration + 5s/10s synth)
   - `🛠️ compileModel: ... .mlpackage → /var/folders/... .mlmodelc`
   - `🔥 Model loading: … (Duration: ✅, Synth: ✅)`
   - `🔥 Duration prediction: ~20 ms`
   - `CoreML=ON (reason: preflight PASS)`
3) First synthesis attempt → `Error: Cannot retrieve vector from IRValue format int32`

## Key Logs (Representative)

- Model paths resolved to `.mlpackage`, compiled to temp `.mlmodelc` successfully
- Duration preflight PASS, e.g. `CoreML=ON (reason: preflight PASS)`
- Failure on first synth call: `Error: Cannot retrieve vector from IRValue format int32`

## Hypotheses (Ranked)

1) Shape mismatch (most likely)
   - Duration outputs `d`, `t_en`, or `pred_aln_trg` do not match the synthesizer’s expected input shapes for the chosen bucket. Historically this error message is a red herring for shape/stride issues.
2) Exporter/model contract drift
   - Synth model’s declared input shapes or ordering differ from what the Swift pipeline assumes.
   - Previously, stale artifacts and exporter changes led to mismatches (e.g., `ref_s` as both input and output, unguarded tile reps).
3) Hardware backend interaction (lower likelihood but worth isolating)
   - ANE or GPU fallback exposing shape/stride constraints; CPU-only run would isolate.

## Timeline of Attempts and Tweaks

- Model selection & bundling hygiene
  - Removed 20s-only bias; prefer 5s bucket (`CoreMLModelManager.swift` ordering updated)
  - Deleted stray `kokoro_synthesizer_20s.mlpackage` to avoid accidental selection
  - Ensured only `.mlpackage` is shipped; compile at runtime with `MLModel.compileModel(at:)`
  - Loader now logs exact `.mlpackage` and temp `.mlmodelc` paths
- Duration preflight and inputs normalization
  - Preflight updated to small, realistic sequence (16 tokens)
  - Fixed rank of duration model inputs in Swift (now rank-1 vectors: `input_ids [T]`, `attention_mask [T]`, `ref_s [256]`, `speed [1]`)
- Synthesis pipeline shape/dtype adaptation (Swift)
  - Added robust adaptation in `CoreMLTTSService.synthesizeChunkSeconds(...)`:
    - Read synthesizer input constraints via `model.modelDescription.inputDescriptionsByName`
    - Compute `expectedTokens` (from `pred_aln_trg` first dim) and `expectedFrames` (from `pred_aln_trg` last dim or seconds×fps)
    - Pad/crop `d`, `t_en` along last dim to `expectedTokens`
    - Ensure `s` is `[1,128]`, `ref_s` is `[1,256]` (batch if needed)
    - Pad/crop alignment `pred_aln_trg` to `[expectedTokens, expectedFrames]`
    - Cast everything to Float32 for safety
  - Added detailed DEBUG logs for target vs adapted shapes
- Synthesis call site diagnostics (Swift)
  - In `SynthesisPipeline.synthesize(...)`, added DEBUG prints:
    - Expected input constraints vs actual shapes passed in
    - Outputs enumeration and chosen key
- Exporter/tooling iterations (historical)
  - Duration model: added `RangeDim(min=1)` on sequence inputs; removed `ref_s` from outputs to avoid BNNS aliasing; pinned toolchain versions
  - Synthesizer model: used `--debug` mode to reduce `trace_length` (memory), start with smallest viable bucket (5s)

## Relevant Learnings (from docs/learnings.md)

- Dynamic shapes vs. static graphs
  - `repeat_interleave` and value-derived shapes break CoreML static graphs
  - Split pipeline: keep dynamic logic (alignment creation) on CPU; compile fixed buckets for synth
- Tile(reps) ≥ 1 and BNNS aliasing
  - Ensure `RangeDim(min=1)` on sequence dims; do not expose `ref_s` as output
- E5ML flexible-shape strides error
  - Duration exported as fixed-shape to avoid flexible-shape + known-strides mismatch in MLProgram
- Bundling gotcha: `.mlmodel` vs `.mlpackage`
  - Standalone `.mlmodel` can be auto-compiled to `.mlmodelc` and overshadow `.mlpackage` at runtime; remove `.mlmodel` and ship `.mlpackage` only
- Memory constraints
  - `pred_aln_trg` size explodes with `trace_length` and frames; use reduced `trace_length` and smaller buckets (5s) for export stability

## Current Code-State (Important Files)

- `Sources/TalkToMe/CoreMLModelManager.swift`
  - Prefer 5s bucket first; discover `.mlpackage` in bundle; runtime compile
- `Sources/TalkToMe/CoreMLTTSService.swift`
  - Preflight with rank-1 vectors; duration predict wrapper with timing
  - `synthesizeChunkSeconds(...)` adapts tensor shapes/dtypes to synthesizer expectations; DEBUG logs of target/adapted shapes
  - `buildInputsNative(...)` emits rank-1 vectors for duration inputs
- `Sources/TalkToMe/SynthesisPipeline.swift`
  - Logs synthesizer expected inputs (constraints) and actual passed shapes; prints output feature candidates

## Outstanding Issues / Open Questions

- Despite runtime adaptation, first synth prediction fails with `IRValue int32` message
  - We need to capture the new DEBUG logs (`--- Shape Adaptation ---` and `Synth expected inputs/actual`) from a failing run to pinpoint misalignment
- Confirm exact synthesizer input contracts from the exported `.mlpackage`
  - MIL inspection (Netron or `coremltools` MIL dump) to verify input names, ranks, and shapes for `d`, `t_en`, `s`, `ref_s`, `pred_aln_trg`
- Verify compute unit behavior
  - Force `.cpuOnly` via `MLModelConfiguration.computeUnits` for synth to isolate hardware backend issues (should still fail if shape-related)
- Cross-check exporter provenance
  - Ensure the `.mlpackage` in bundle matches the latest exporter commit (no stale graphs)

## Suggested Investigation Plan (for Research)

1) Shape Contract Validation
   - Programmatically print synthesizer input constraints from the bundled `.mlpackage`
   - Cross-check with Swift adaptation (log both expected and actual shapes at call-site)
   - Confirm final adapted shapes match exactly: names, ranks, dims
2) MIL Graph Inspection
   - Dump MIL with `ct.convert(..., debug=True)` or load and print via `coremltools` to verify ops around inputs and early layers
   - Confirm `pred_aln_trg` rank and ordering the synth expects
3) Backend Isolation
   - Run synth with `.cpuOnly` compute units to rule out ANE/GPU backend idiosyncrasies
   - Check for GPU width limits if GPU fallback occurs (Metal 16384 texture width)
4) Re-export Contract Check
   - Re-export synthesizer and duration with clear, fixed input contracts (consider EnumeratedShapes on sequence dims for synth inputs)
   - Confirm duration outputs carry a batch dimension where expected and that client code preserves/batches consistently
5) Minimal Repro Harness
   - Build a small Swift/Python harness that:
     - Loads the synth `.mlpackage`, prints constraints
     - Builds dummy tensors in-memory with the exact expected shapes
     - Runs a single prediction to validate the model works independently
6) Bundle Integrity
   - Verify app bundle only contains intended `.mlpackage` models
   - Confirm runtime is compiling from the expected paths (logs already added)

## Artifacts & References

- Models in bundle (iteration mode):
  - `coreml/kokoro_duration.mlpackage`
  - `coreml/kokoro_synthesizer_5s.mlpackage` (and 10s)
- Export commands used:
  - `python kokoro-coreml/examples/export_coreml.py --output_dir ../coreml --duration_only`
  - `python kokoro-coreml/export_synthesizers.py --buckets="5s" --debug --output_dir ../coreml`
- Toolchain:
  - `torch==2.5.0`, `coremltools==8.3.0`, `numpy==1.26.4`, `safetensors`, `soundfile`
- Relevant docs:
  - `kokoro-coreml/docs/learnings.md` (dynamic shapes, tile reps, E5ML, bundling, memory)
  - `README/Notes/xcode-debug-talk2me.md` (bundling timeline, exporter fixes, hypotheses)
  - `README/Guides/CoreML-deployment.md` (modern `.mlpackage`, build rules, compute units)

## Success Criteria

- No `IRValue int32` errors
- Synth prediction returns a float waveform tensor
- DEBUG logs show expected vs actual shapes identical
- End-to-end speech generation from text succeeds using the 5s bucket
- Confirmed via CPU-only and default compute units

---

## External Research Task Brief (Non-coding)

- Objective
  - Rapidly gather, synthesize, and cite external knowledge on debugging CoreML MLProgram runtime errors matching our symptoms (especially “Cannot retrieve vector from IRValue format int32”), shape/stride issues, ANE/GPU fallbacks, and `.mlpackage` deployment pitfalls.
  - Outcome is a concise report that short-circuits multiple days of ad‑hoc Googling and points us to actionable next steps.

- Scope (In-scope research sources)
  - Apple Developer Forums (Core ML / Metal / ANE), Stack Overflow, GitHub Issues/Discussions (coremltools, Metal, Core ML sample repos), blog posts, Medium/Dev.to, WWDC sessions/transcripts, academic/industry writeups.
  - Keywords/strings to use (examples):
    - "Cannot retrieve vector from IRValue format int32", "E5RT flexible-shape strides", "tile reps >= 1 coreml", "MTLTextureDescriptor width 16384 CoreML", "ANECCompile failed", "mlprogram shape mismatch", "CoreML MLProgram MLMultiArray strides", "coremltools enumerated shapes", "RangeDim min=1", "mlpackage stale .mlmodelc", "CoreML inputDescriptionsByName multiArrayConstraint".

- Non-goals
  - No code changes, no re-exports, no running our project. Provide a literature review and an actionable, cited report only.

- Context to assume (constraints)
  - macOS 15.6 (Apple Silicon), Core ML MLProgram in `.mlpackage`, coremltools 7.x/8.x with Torch 2.x, Kokoro-like 2-stage pipeline, alignment matrix built on CPU, runtime compilation with `MLModel.compileModel(at:)`, and prior issues including tile(reps), flexible-shape strides, ANE/GPU width limits, and stale `.mlmodelc`.

- Key questions to answer
  1) In real-world reports, what root causes map to our specific error signature (int32 IRValue) and to similar early-runtime failures on synth calls?
  2) What shape/stride contract patterns and pitfalls are most cited for MLProgram models taking multi-array inputs (esp. 1D/2D/3D audio/NLP tensors)?
  3) Which coremltools versions/regressions are implicated, and what mitigations (e.g., EnumeratedShapes vs RangeDim, fixed-shape exports, dtype casting) are recommended?
  4) Best-practice playbooks for isolating ANE/GPU/CPU, dumping MIL, validating model IO contracts, and proving fallback reasons.
  5) Deployment pitfalls with `.mlpackage` (cache invalidation, folder references, stale `.mlmodelc`) and the most reliable workflows cited by others.

- Deliverables (what we expect)
  - Executive summary (1–2 pages): ranked likely causes for our symptom set, with rationale.
  - Annotated bibliography: links with 1–3 bullet takeaways each (focus on directly applicable guidance).
  - Decision tree / triage checklist we can follow in <30 minutes to isolate shape/stride vs backend vs bundling issues.
  - Concrete, copy‑pasteable snippets/commands:
    - Python: printing model IO constraints, dumping MIL, small synthetic input builders that match constraints.
    - Swift: reading `modelDescription.inputDescriptionsByName`, forcing `.cpuOnly`, logging constraints.
  - Table mapping common CoreML runtime errors to likely root causes + recommended actions.
  - Version notes: known coremltools regressions/fixes by version relevant to macOS 14/15 and Torch 2.x.

- Format
  - Markdown with clickable links; include inline code blocks for commands/snippets; footnote-style citations acceptable.

- Timeline / process
  - 1–2 months of focused research
  - If evidence is sparse/contradictory, flag clearly and propose follow‑up searches.

- Success criteria
  - We gain a prioritized, evidence‑based path to test next (e.g., specific shape checks, MIL inspection steps, compute-unit isolation) without broad, time‑consuming searches.
  - The report materially reduces further debugging time and points to concrete, cited solutions/examples others used for the same errors.

## Appendix: Known Pitfalls We Already Guarded Against

- Stale `.mlmodelc` overshadowing `.mlpackage` — we compile from `.mlpackage` at runtime and log paths
- BNNS “tile reps ≥ 1” — exporter adds `RangeDim(min=1)`; client clamps durations to ≥1 when building alignment
- `ref_s` aliasing (input == output) — exporter no longer exposes `ref_s` as output
- E5ML flexible-shape strides — avoided by fixed-shape duration or ensuring compatible shapes
- Memory OOM during synth export — mitigated by reduced `trace_length` (`--debug`) and starting with 5s bucket