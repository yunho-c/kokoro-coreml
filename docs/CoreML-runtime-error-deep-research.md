
CoreML TTS Pipeline Investigation: A Report on Mitigating IRValue Errors and Validating Tensor Contracts


I. Executive Summary & Prioritized Action Plan

This report presents a comprehensive analysis of the Cannot retrieve vector from IRValue format int32 runtime error encountered in the TalkToMe CoreML-based Text-to-Speech (TTS) pipeline. The investigation synthesizes findings from official Apple documentation, developer forums, and public issue trackers to provide a prioritized, evidence-based action plan for diagnosis and resolution.
The primary diagnosis concurs with the internal hypothesis that the error is symptomatic of a tensor shape, rank, or stride mismatch. However, this investigation concludes with high confidence that the issue is not a simple discrepancy at the model-to-model interface. Instead, it is likely an internal graph inconsistency within the synthesizer model itself, introduced as an artifact during the PyTorch-to-CoreML export process. The IRValue int32 error is a misleading symptom of an internal operation (such as reshape or tile) failing because it expects a static, constant integer for a shape-defining parameter but receives an incorrectly typed or shaped tensor value at runtime. This behavior is a well-documented pitfall of torch.jit.trace when dynamic control flow or non-tensor data types are involved in shape calculations.1
A critical secondary diagnosis identifies a fundamental flaw in the current deployment workflow: compiling the .mlpackage to a new temporary .mlmodelc directory on every application launch. This practice guarantees a "cold" (uncached) model load for every session, incurring a severe and unnecessary performance penalty. More importantly for debugging, it prevents the observation of the model's behavior under the more common and highly optimized "cached" load path, potentially masking or even causing the observed error.4 Rectifying this workflow is essential for both accurate debugging and achieving production-level performance.
Based on these findings, the following prioritized action plan is recommended to systematically isolate and resolve the issue:
Isolate the Compute Backend: Immediately re-run the failing synthesis call with the synthesizer model's MLModelConfiguration set to force .cpuOnly execution. This is a non-invasive, five-minute test that will definitively confirm or rule out the Apple Neural Engine (ANE) or GPU hardware backends as the source of the error. An error that persists on the CPU points to a fundamental graph issue, whereas an error that disappears implicates a hardware-specific constraint or bug.5
Verify the External I/O Contract: Implement programmatic logging on both the Python export script and the Swift application runtime to print the exact input specifications of the synthesizer model. This includes tensor names, data types, ranks, and shape constraints. These logs must be compared side-by-side with the logs of the adapted tensors being passed to the prediction() call at the moment of failure. This will provide incontrovertible evidence of whether the external contract is being met.9
Inspect the Internal Graph Contract: Utilize coremltools to load the synthesizer .mlpackage and dump its Model Intermediate Language (MIL) representation to a text file. Manually inspect the resulting MIL graph, focusing on operations that consume the pred_aln_trg tensor or other inputs with variable shapes. Pay close attention to shape-deriving operations like reshape, tile, expand, and gather. This inspection will reveal if the graph contains hardcoded shape assumptions baked in from the original PyTorch trace, which is a likely source of the internal conflict.
Build a Minimal Reproducer: Create a standalone Swift command-line application or XCTest case that performs only three actions: loads the synthesizer .mlpackage, synthetically generates MLMultiArray inputs that perfectly match the logged specifications from step 2, and attempts a single prediction. A failure in this isolated environment would prove the issue is inherent to the model file itself, independent of the main application's complex data pipeline.
Correct the Caching and Compilation Workflow: Modify the CoreMLModelManager.swift service to compile the bundled .mlpackage files to a persistent, stable path within the application's sandboxed support directory (e.g., Library/Application Support/). The application should check for the existence of the compiled .mlmodelc at this path on launch and only re-compile if it is missing or if the source .mlpackage has been updated. This change is critical for enabling the device-specialized cache, which will drastically improve load times and create a more stable, production-representative debugging environment.4

II. Deconstructing the IRValue int32 Error Signature

The runtime error Cannot retrieve vector from IRValue format int32 is a low-level message from the CoreML execution engine, specifically the modern runtime responsible for executing ML Program models (often referred to as E5ML). A careful deconstruction of this message, informed by community reports and an understanding of the model conversion toolchain, reveals that it is most often a "red herring" pointing to a deeper, structural issue within the model graph rather than a simple data type mismatch on the model's inputs.

Analysis of the Error Message

The message indicates that the runtime encountered an intermediate value (IRValue) during graph execution. It expected this value to be in a format that could be interpreted as a vector (a one-dimensional array of numbers), but instead found it to be a singular int32 value. This is a fundamental type incompatibility at the operation level.
This does not imply that the application is passing an Int32 tensor where a Float32 tensor is expected. The robust runtime adaptation and type casting logic already implemented in Swift makes such a high-level error highly improbable. The term IRValue is key, as it refers to a value that exists within the model's compiled intermediate representation, not necessarily a named input feature. This value is often a parameter for an operation, such as the reps argument for a tile operation or the shape argument for a reshape operation, which must be provided as a vector of integers. The error suggests that the runtime received a single integer where it expected such a vector.

The PyTorch Tracing Artifact Hypothesis

The most compelling evidence points to this error being an artifact of the PyTorch-to-CoreML conversion process, particularly when using torch.jit.trace. The tracer operates by executing the model with example inputs and recording the sequence of operations performed. This method has a significant limitation: it cannot capture data-dependent control flow or operations on non-tensor Python types in a dynamic way.
Tracer Warnings and Constant Baking: During tracing, if the model's code contains Python-level logic like if tensor.shape > 0: or constructs a list of integers for a reshape operation, the tracer may issue a TracerWarning.3 This warning signifies that a Python value is being converted to a tensor or that a data-dependent condition is being evaluated. The tracer's behavior in these cases is to "bake" the values observed
during the trace into the graph as constants. For example, if the trace was performed with a tensor of shape (1, 50, 128), an internal reshape operation might be recorded as reshape(input, shape=), where `` is now a hardcoded constant in the graph.1
Runtime Mismatch: When the application later provides an input with a different shape—for instance, a tensor of shape (1, 75, 128)—the execution reaches the same reshape operation. If the operation was designed to dynamically derive its shape from this new input, but the graph has a hardcoded constant, a conflict arises. In more complex scenarios, an operation might expect a shape parameter to be a dynamically computed 1D tensor (a vector), but due to a tracing artifact, it receives a single scalar integer (int32). The runtime then fails when it tries to interpret this scalar integer as the vector it needs, resulting in the Cannot retrieve vector from IRValue format int32 error. A related failure mode occurs when the JIT scripter cannot infer an object's type and incorrectly assumes it is a tensor, when it is in fact a list of integers, leading to a similar type mismatch at runtime.2

Connection to Historical Issues

The team's internal documentation of a past issue with "E5ML flexible-shape strides" provides a strong historical precedent. This, combined with public developer forum posts regarding warnings like Type of hiddenStates in function main's I/O contains unknown strides 17, indicates a high degree of sensitivity within the ML Program runtime to the definition and propagation of tensor shapes and their corresponding memory layouts (strides). Flexible input shapes exacerbate this sensitivity. The
IRValue int32 error is likely a new manifestation of this same underlying problem: the compiled graph contains an assumption about shape or stride that is violated by the real-world data provided at runtime, leading to a low-level execution failure.
The central conclusion from this analysis is that the debugging focus must pivot. Validating the external data types and shapes being passed to the model, while necessary, is insufficient. The root cause is almost certainly located within the internal logic of the compiled synthesizer model graph. Therefore, direct inspection of this graph and a critical review of the export process that created it are the most crucial next steps.

III. Validating the Model I/O Contract: A Deep Dive into Shape, Rank, and Strides

To definitively resolve the runtime error, it is imperative to establish and rigorously validate the Input/Output (I/O) contract of the synthesizer model. This contract encompasses not only the shape and data type of each tensor but also how flexibility is defined and handled by the coremltools converter and the CoreML runtime. Any ambiguity or mismatch in this contract is a primary suspect for the observed failure.

3.1. Defining Flexible Shapes in coremltools: EnumeratedShapes vs. RangeDim

The synthesizer model's inputs, particularly d, t_en, and pred_aln_trg, have dimensions that vary based on the output of the preceding duration model. The method used to declare this flexibility during the coremltools conversion process has profound implications for both correctness and performance.
EnumeratedShapes: This approach defines a discrete, finite list of specific, complete shapes that the model can accept (e.g., [(1, 100, 256), (1, 200, 256)]).
Advantages: It offers the highest performance, particularly on the ANE. The CoreML runtime can pre-compile and heavily optimize a specialized execution path for each shape in the enumeration, as the set of possibilities is known ahead of time.18
Disadvantages: This method is inherently inflexible. If the model is fed a tensor with a shape not present in the enumerated list, the prediction will fail. Prior to iOS 18, a significant limitation was that only one input to a model could be defined with EnumeratedShapes; all other inputs had to be fixed-shape.18 Furthermore, mixing
EnumeratedShapes on one input with RangeDim on another has been reported to cause conversion failures in some versions of coremltools.20
RangeDim: This approach provides greater flexibility by defining a continuous range (a lower and upper bound) for one or more dimensions of a shape (e.g., (1, ct.RangeDim(1, 512), 256)).
Advantages: It can handle a much wider variety of input sizes without requiring each one to be explicitly listed, which is well-suited for the dynamic nature of the TTS pipeline.
Disadvantages: The performance can be lower than with EnumeratedShapes because the runtime cannot pre-specialize the graph as effectively for an entire range of possibilities. ANE support for RangeDim is known to be less robust; frequently, only the specified default shape will execute on the ANE, while any other shape within the range will cause a silent fallback to the GPU or CPU.21 A critical limitation for the current project is that unbounded ranges (e.g.,
upper_bound=-1) are not permitted when converting to the mlprogram format, which is the default for modern deployment targets.18
Given the variable number of tokens produced by the duration model, RangeDim is the more logical choice for defining the synthesizer's input shapes. However, the team must be aware of its potential to trigger backend fallbacks and ensure a finite upper_bound is always specified during export.

3.2. The MLProgram Backend and Typed Execution

The project correctly utilizes the modern .mlpackage format, which, for the target macOS version, defaults to producing a model with the MLProgram backend.22 Understanding the distinct characteristics of this backend is crucial for debugging.
The key differences from the legacy NeuralNetwork format are 24:
Typed Execution: MLProgram is a strongly-typed representation. Every intermediate tensor within the graph has an explicit data type (e.g., fp16, fp32, int32). The runtime strictly adheres to these types. This provides granular control over precision but also means that any type mismatch, however minor, will lead to a hard failure.
Decoupled Weights: In an .mlpackage, the model's architecture is stored in a human-readable MIL file (model.mil inside the compiled .mlmodelc), while the numerical weights are serialized into a separate binary file (e.g., weights.bin). This separation improves compilation efficiency.
Operations vs. Layers: The graph is defined by a set of primitive "ops" (e.g., add, conv, reshape) rather than high-level "layers". This offers more flexibility but also exposes the developer to lower-level implementation details and potential inconsistencies.
The strictness of the MLProgram runtime means that even subtle inconsistencies introduced during the PyTorch export—such as a shape parameter being treated as a scalar int32 instead of a 1D tensor—are more likely to cause a fatal runtime error.

3.3. Programmatic Inspection of the I/O Contract

To eliminate any ambiguity about the model's expected inputs, the contract should be programmatically inspected at both ends of the pipeline: after export in Python and before prediction in Swift.

Python (coremltools) Inspection

After the synthesizer model is exported, a Python script should be used to load the resulting .mlpackage and print its spec. This provides the canonical "source of truth" for the model's I/O contract as understood by the coremltools framework.
Code Snippet (Python):

Python


import coremltools as ct
import coremltools.proto.FeatureTypes_pb2 as ft

def inspect_model_spec(model_path: str):
    """Loads a.mlpackage and prints its input specifications."""
    try:
        model = ct.models.MLModel(model_path)
        spec = model.get_spec()
        
        print(f"--- Input Specification for {model_path} ---")
        if not spec.description.input:
            print("  No inputs found in model specification.")
            return

        for input_desc in spec.description.input:
            name = input_desc.name
            feature_type = input_desc.type.WhichOneof('Type')
            
            print(f"  Input: '{name}'")
            print(f"    - Feature Type: {feature_type}")

            if feature_type == 'multiArrayType':
                multi_array_type = input_desc.type.multiArrayType
                shape = [dim for dim in multi_array_type.shape]
                dtype_enum = multi_array_type.dataType
                dtype_name = ft.ArrayFeatureType.DataType.Name(dtype_enum)
                
                print(f"    - DType: {dtype_name}")
                print(f"    - Shape: {shape}")
            elif feature_type == 'imageType':
                image_type = input_desc.type.imageType
                print(f"    - Dimensions: {image_type.width}x{image_type.height}")
                print(f"    - Color Space: {image_type.colorSpace}")

    except Exception as e:
        print(f"Error inspecting model at {model_path}: {e}")

# Usage
synthesizer_model_path = "coreml/kokoro_synthesizer_5s.mlpackage"
inspect_model_spec(synthesizer_model_path)


Relevant Sources: 10

Swift Runtime Inspection

Correspondingly, the Swift application should inspect the loaded MLModel object at runtime, just before making a prediction. The modelDescription property provides access to the same contract information. The output of this Swift log should be compared character-for-character with the output of the Python inspection script and the debug logs of the tensors being passed to the model.
Code Snippet (Swift):

Swift


import CoreML

// Helper extension to make MLMultiArrayDataType printable
extension MLMultiArrayDataType {
    func toString() -> String {
        switch self {
        case.double: return "Double"
        case.float32: return "Float32"
        case.int32: return "Int32"
        case.float16: return "Float16"
        case.float64: return "Float64" // Alias for.double
        @unknown default: return "Unknown"
        }
    }
}

func logModelInputConstraints(model: MLModel) {
    let modelId = model.modelDescription.metadata[MLModelMetadataKey.description] as? String?? "Unnamed Model"
    print("--- Expected Input Constraints for \(modelId) ---")
    
    let inputs = model.modelDescription.inputDescriptionsByName
    guard!inputs.isEmpty else {
        print("  No inputs found in model description.")
        return
    }

    for (name, description) in inputs {
        print("  Input: '\(name)'")
        print("    - Optional: \(description.isOptional)")
        
        if let constraint = description.multiArrayConstraint {
            let shape = constraint.shape.map { $0.intValue }
            let dtype = constraint.dataType
            
            print("    - Type: MLMultiArray")
            print("    - DType: \(dtype.toString())")
            print("    - Shape (from constraint): \(shape)")
            
            // Log detailed flexible shape constraints
            let shapeConstraint = constraint.shapeConstraint
            switch shapeConstraint.type {
            case.unspecified:
                print("    - Shape Constraint: Unspecified")
            case.enumerated:
                let enumeratedShapes = shapeConstraint.enumeratedShapes
                print("    - Shape Constraint: Enumerated")
                for (i, enumeratedShape) in enumeratedShapes.enumerated() {
                    print("      - Shape \(i): \(enumeratedShape.map { $0.intValue })")
                }
            case.range:
                let shapeRange = shapeConstraint.shapeRange
                print("    - Shape Constraint: Range")
                for (i, dimRange) in shapeRange.enumerated() {
                    let lower = dimRange.lowerBound
                    let upper = dimRange.upperBound == -1? "unbounded" : "\(dimRange.upperBound)"
                    print("      - Dimension \(i): [\(lower), \(upper)]")
                }
            @unknown default:
                print("    - Shape Constraint: Unknown Type")
            }
        } else if let imageConstraint = description.imageConstraint {
            print("    - Type: Image")
            print("    - Dimensions: \(imageConstraint.pixelsWide)x\(imageConstraint.pixelsHigh)")
        } else {
            print("    - Type: Other (\(description.type))")
        }
    }
}


Relevant Sources: 9
This two-pronged validation approach will definitively establish whether the error originates from a mismatch between the application's data preparation and the model's declared external contract. If these logs show a perfect match, the investigation must proceed to the model's internal graph and backend execution environment.

IV. Isolating Hardware Backend Dependencies

The CoreML framework abstracts away the underlying hardware, dynamically scheduling operations across the CPU, GPU, and ANE to optimize performance.31 While powerful, this abstraction can mask hardware-specific bugs or constraints. The
IRValue int32 error could be universal, or it could be triggered only when the model executes on a specific compute unit. A systematic process of elimination is required to isolate this variable.

4.1. Forcing Compute Units for Isolation

The most direct way to test for backend-specific issues is to force the model to load and execute on a single compute unit. The MLModelConfiguration object provides the necessary control.
Rationale: If the error disappears when the model is constrained to .cpuOnly, the bug is definitively linked to the ANE or GPU backends. This could stem from an unsupported operation, a violation of a hardware-specific shape or stride constraint, or a lower-level driver or OS bug. Conversely, if the error persists even on the CPU, it is almost certainly a fundamental issue within the model's graph structure or a violation of its I/O contract, independent of the underlying hardware.
Implementation: The MLModelConfiguration should be instantiated and its computeUnits property set before the synthesizer model is loaded. This ensures that the entire model lifecycle, from device-specific compilation to prediction, is restricted to the specified backend.
Code Snippet (Swift):

Swift


import CoreML

/// Loads a CoreML model with a specific compute unit configuration for debugging.
///
/// - Parameters:
///   - url: The file URL of the.mlpackage or.mlmodelc to load.
///   - computeUnits: The MLComputeUnits to force for execution.
/// - Returns: An initialized MLModel instance.
/// - Throws: An error if the model cannot be loaded.
func loadModel(url: URL, with computeUnits: MLComputeUnits) throws -> MLModel {
    let config = MLModelConfiguration()
    config.computeUnits = computeUnits
    
    let unitString: String
    switch computeUnits {
    case.cpuOnly:
        unitString = ".cpuOnly"
    case.cpuAndGPU:
        unitString = ".cpuAndGPU"
    case.all:
        unitString = ".all"
    case.cpuAndNeuralEngine:
        unitString = ".cpuAndNeuralEngine"
    @unknown default:
        unitString = "unknown"
    }
    
    print("Attempting to load model at \(url.lastPathComponent) with compute units: \(unitString)...")
    
    let compiledUrl = try MLModel.compileModel(at: url)
    return try MLModel(contentsOf: compiledUrl, configuration: config)
}

// Example Usage in the model loading service:
// let synthesizerModel = try loadModel(url: modelUrl, with:.cpuOnly)


Relevant Sources: 7

4.2. Diagnosing Silent Fallbacks with Xcode Instruments

A common pitfall is assuming that computeUnits =.all means the model will run entirely on the ANE. In reality, CoreML may determine that certain operations are unsupported by the ANE or would run faster on the GPU or CPU. It will then partition the model graph, executing some parts on the ANE and "falling back" to other units for the rest. This silent fallback can lead to unexpected performance and, critically, can expose bugs specific to the fallback backend. The Core ML Instrument in Xcode is the definitive tool for visualizing this behavior.
Methodology for Detection 33:
Profile the App: Launch the application from Xcode using Product > Profile (or Cmd+I).
Select Instrument: Choose the "Core ML" instrument template from the profiling options.
Record Execution: Start the recording in Instruments, then switch back to the application and trigger the TTS synthesis call that causes the error.
Analyze the Track: Stop the recording and examine the "Core ML" track in the Instruments timeline. This track will show discrete events for model loading (load) and prediction (predict).
Inspect Prediction Details: Select the specific predict event corresponding to the synthesizer model. The detail pane below the timeline will populate with a table showing every layer or operation within the model.
Identify Fallbacks: This table includes a "Compute Unit" column. For a model loaded with .all, this column will show precisely where each operation ran: ANE, GPU, or CPU. Any operation not running on the ANE represents a silent fallback. This allows for pinpointing the exact part of the model that is incompatible with the Neural Engine.

4.3. Known Hardware Constraints

Certain hardware backends have well-documented limitations that can cause runtime failures if violated.
Metal Texture Width Limit: The GPU backend frequently represents tensors as Metal textures for processing. Apple Silicon GPUs have a hard limit on the maximum width of a 1D or 2D texture, which is consistently documented as 16,384 pixels.38 For a large 2D tensor, such as the
pred_aln_trg alignment matrix (``), if either dimension exceeds this limit, prediction will fail with an MTLTextureDescriptor error. While the 5s bucket is unlikely to produce a matrix this large, longer synthesis chunks or future model changes could approach this limit. This is a known, non-negotiable hardware constraint.39
ANE Shape and Layer Constraints: The ANE is the most specialized and also the most restrictive of the compute units. Its performance benefits are derived from hardware optimized for specific operations and data layouts. Consequently, it has stricter limitations on supported layer types and dynamic shapes. As noted previously, RangeDim inputs often cause a fallback to the GPU/CPU for any shape other than the specified default.21 If Instruments reveals a fallback, it is a strong indication that an ANE constraint has been violated.
The default computeUnits =.all setting should be treated as an instruction to an optimizer, not as a guarantee of execution on a specific backend. The CoreML planner's goal is to achieve the lowest possible latency, and it will partition the graph across all available units to achieve this.40 This can lead to complex and sometimes counter-intuitive execution patterns. For the purpose of systematic debugging, this uncertainty is a liability. The model's behavior must be tested explicitly on each compute configuration (
.cpuOnly, .cpuAndGPU, .cpuAndNeuralEngine) to build a complete picture and reliably isolate the failure.

V. Toolchain and Deployment Pipeline Integrity

The journey from a PyTorch model to a running CoreML feature in a Swift application involves a complex toolchain and deployment pipeline. Errors can be introduced at any stage, from the initial export in Python to the final bundling and loading process in Xcode. A thorough audit of this pipeline is necessary to ensure its integrity and rule out process-related causes for the runtime error.

5.1. coremltools Version Analysis (v8.0-8.3)

The project is utilizing a modern toolchain with coremltools==8.3.0 and torch==2.5.0. While staying current is generally a best practice, it is important to be aware of recent changes and potential regressions in the converter.
Enhanced torch.export Support: A major focus of recent coremltools releases has been improving support for the torch.export pathway, which is designed to be more robust for capturing dynamic shapes compared to the older torch.jit.trace method.41 The export scripts should be reviewed to determine which method is being used. If
torch.jit.trace is still in use, it remains a prime suspect for the tracing artifacts discussed in Section II. Migrating to torch.export, if feasible for the model architecture, could resolve these issues.
New Debugging Utilities: Version 8.3.0 introduced a suite of powerful debugging tools, including MLModelComparator and MLModelBenchmarker.41 These utilities can be integrated into the Python-side workflow. For instance,
MLModelComparator can be used to programmatically compare the numerical output of the converted CoreML model against the original PyTorch model using a set of synthetic inputs. This could reveal subtle numerical discrepancies that precede the catastrophic runtime failure, providing an earlier and more informative error signal.
Known Bugs and Regressions: The coremltools release history shows a continuous stream of fixes and improvements. Notably, past versions have had bugs related to the interaction between different flexible shape types, such as an exception when mixing EnumeratedShapes and RangeDim inputs in a single model.20 While the current version may have fixed this specific issue, it highlights the complexity of handling flexible shapes in the converter. The release notes also mention fixes for specific layers like
batch_norm and ConvTranspose1d 41, demonstrating that the conversion logic for individual ops is an ongoing area of development.

5.2. Runtime Compilation and Caching (.mlpackage -> .mlmodelc)

The investigation has identified a critical flaw in the application's model loading workflow. The brief states that the .mlpackage is compiled to a temporary .mlmodelc directory at runtime. This approach fundamentally misunderstands and defeats CoreML's caching mechanism.
The Correct Caching Mechanism 4:
Initial Load (Uncached): When an MLModel is first instantiated from a .mlpackage or .mlmodelc at a specific file path, CoreML performs an expensive, multi-stage compilation. This includes a quick compilation to the generic .mlmodelc format, followed by a much slower "device specialization" step where the model graph is optimized and compiled into native code for the specific hardware (ANE, GPU) of the user's device.
Cache Storage: The result of this device specialization is stored in a secure, system-managed cache. The key for this cache entry is derived from several factors, most importantly the full, absolute file path of the source .mlmodelc directory.
Subsequent Loads (Cached): On all future application launches, when the code attempts to load a model from the exact same file path, CoreML finds the corresponding entry in its cache and loads the pre-compiled, device-specialized assets directly. This "cached load" is orders of magnitude faster than the initial uncached load.
The Flaw in the Current Workflow:
By compiling the model to a new temporary directory in /var/folders/ on each application launch, the file path is guaranteed to be different every single time. This forces CoreML to perform the slow, uncached, device-specialization compilation on every launch, completely negating the benefit of the cache.
Recommended Correction:
The model loading logic in CoreMLModelManager.swift must be redesigned. The recommended approach is:
On first launch (or when a new model version is detected), determine a persistent, stable path within the app's sandboxed Application Support directory.
Programmatically call MLModel.compileModel(at:) to compile the bundled .mlpackage to this persistent path.
On all subsequent launches, the app should first check if the .mlmodelc directory already exists at the persistent path.
If it exists, the app should instantiate the MLModel directly from the URL of the persistent .mlmodelc directory, thus ensuring a fast, cached load.
If it does not exist (or if the app version has changed, indicating a potential model update), the compilation step should be re-run.
This change is not merely a performance optimization; it is a prerequisite for correct debugging. The code path for an uncached load is significantly different and more complex than for a cached load. It is plausible that the IRValue int32 error is a bug that only manifests on the uncached path. By fixing the caching workflow, the team will create a debugging environment that more accurately reflects real-world usage and may, in itself, resolve the runtime error.

5.3. Bundling and Deployment Best Practices

The team's current approach to bundling aligns with modern best practices, but there are opportunities for increased robustness.
Continue Shipping .mlpackage: The decision to ship the architecture-neutral .mlpackage and perform compilation on-device is correct. This ensures the smallest app bundle size and allows each user's device to apply the latest and most relevant hardware-specific optimizations at runtime.42
Implement Model Versioning: The model assets should include explicit versioning information in their metadata. The application's loading logic should compare the version of the bundled .mlpackage with the version of any existing compiled model in the persistent cache directory. This ensures that when the app is updated with a new model, the old cached version is properly invalidated and the new model is compiled.45
Consider Multi-Function Models for Future Expansion: As the TTS system evolves to support multiple speakers or styles, the team should investigate using multi-function models. This allows multiple model "heads" (e.g., different synthesizer adapters) to share a common base model, with all variants contained within a single .mlpackage. This approach reduces the overall app size by de-duplicating shared weights and simplifies asset management.47
By addressing the critical caching flaw and implementing robust versioning, the deployment pipeline can be made more performant, reliable, and debuggable.

VI. Appendices


A. Triage Checklist & Decision Tree

This checklist provides a systematic, step-by-step process for isolating the root cause of the IRValue int32 error.
Baseline & Logging:
[ ] Ensure comprehensive debug logging is enabled in Swift for both shape adaptation and model I/O constraint inspection.
[ ] Reproduce the crash to capture a baseline set of logs.
Step 1: Isolate the Compute Backend.
[ ] Modify the synthesizer's MLModelConfiguration to set computeUnits =.cpuOnly.
[ ] Re-run the synthesis task.
Decision: Does the error still occur?
YES (Error Persists): The issue is backend-independent and lies within the model graph or I/O contract. Proceed to Step 3: Shape Validation.
NO (Error Disappears): The issue is specific to the ANE or GPU backend. Proceed to Step 4: Backend Investigation.
Step 2: Validate the External I/O Contract.
[ ] Run the Python inspection script (Section III.3) on the bundled synthesizer .mlpackage to generate the "source of truth" for input specifications.
[ ] In the failing run logs from Swift, compare the "Expected Input Constraints" log with the "Shape Adaptation" log showing the actual tensor shapes being passed to the prediction() call.
Decision: Do the names, ranks, dimensions, and data types of the actual tensors passed exactly match the model's specification from both Python and Swift inspection?
YES (Perfect Match): The external contract is being met. The problem is internal to the model's graph. Proceed to Step 5: MIL Inspection.
NO (Mismatch Found): The Swift shape adaptation logic is faulty. Correct the padding, cropping, or batching logic to ensure a perfect match. Return to Step 1.
Step 3: Investigate the Hardware Backend. (Execute if error disappeared in Step 2)
[ ] Profile the application with the Core ML Instrument in Xcode while running with computeUnits =.all.
[ ] Identify which specific layers are falling back from ANE to GPU/CPU. This is the likely location of the incompatibility.
[ ] Check the dimensions of all input tensors against known hardware limits, especially the Metal texture width limit of 16,384 for the pred_aln_trg matrix.
[ ] Hypothesize and Test:
If a fallback is caused by RangeDim on the ANE, consider re-exporting with a limited set of EnumeratedShapes as a test.
If a dimension is approaching the GPU width limit, test with shorter synthesis chunks to see if the error is size-dependent.
Step 4: Inspect the Internal Model Graph. (Execute if Step 3 showed a perfect match)
[ ] Use coremltools to load the model and dump its MIL representation to a text file.
[ ] Perform a text search within the MIL file for operations that consume the variable-shape inputs (e.g., pred_aln_trg).
[ ] Scrutinize shape-deriving ops like reshape, tile, expand, gather. Check if their shape-defining parameters are derived from other tensors or are hardcoded constants.
[ ] Cross-reference these findings with the original PyTorch exporter code. Look for potential sources of TracerWarnings where Python integers or shape calculations might have been improperly baked into the graph.
[ ] Resolution: If a hardcoded shape is found, the PyTorch exporter must be modified to express the shape calculation using pure tensor operations that can be correctly translated by coremltools.

B. Actionable Code Snippets

This section consolidates the key Python and Swift code snippets provided throughout this report for easy reference and implementation.

Python: Inspecting Model I/O Specification


Python


import coremltools as ct
import coremltools.proto.FeatureTypes_pb2 as ft

def inspect_model_spec(model_path: str):
    """Loads a.mlpackage and prints its input specifications."""
    try:
        model = ct.models.MLModel(model_path)
        spec = model.get_spec()
        
        print(f"--- Input Specification for {model_path} ---")
        for input_desc in spec.description.input:
            name = input_desc.name
            feature_type = input_desc.type.WhichOneof('Type')
            print(f"  Input: '{name}' (Type: {feature_type})")

            if feature_type == 'multiArrayType':
                multi_array_type = input_desc.type.multiArrayType
                shape = [dim for dim in multi_array_type.shape]
                dtype_name = ft.ArrayFeatureType.DataType.Name(multi_array_type.dataType)
                print(f"    - DType: {dtype_name}, Shape: {shape}")

    except Exception as e:
        print(f"Error inspecting model at {model_path}: {e}")

# Usage:
# inspect_model_spec("path/to/your/model.mlpackage")



Swift: Inspecting Model I/O Constraints at Runtime


Swift


import CoreML

extension MLMultiArrayDataType {
    func toString() -> String {
        switch self {
        case.double: return "Double"
        case.float32: return "Float32"
        case.int32: return "Int32"
        case.float16: return "Float16"
        case.float64: return "Float64"
        @unknown default: return "Unknown"
        }
    }
}

func logModelInputConstraints(model: MLModel) {
    print("--- Expected Input Constraints for Model ---")
    let inputs = model.modelDescription.inputDescriptionsByName
    for (name, description) in inputs {
        print("  Input: '\(name)'")
        if let constraint = description.multiArrayConstraint {
            let shape = constraint.shape.map { $0.intValue }
            print("    - Type: MLMultiArray")
            print("    - DType: \(constraint.dataType.toString())")
            print("    - Shape: \(shape)")
            print("    - Shape Constraint Type: \(constraint.shapeConstraint.type)")
        }
    }
}



Swift: Forcing a Specific Compute Unit


Swift


import CoreML

func loadModel(url: URL, with computeUnits: MLComputeUnits) throws -> MLModel {
    let config = MLModelConfiguration()
    config.computeUnits = computeUnits
    
    // Assumes runtime compilation. For pre-compiled.mlmodelc, use this directly.
    let compiledUrl = try MLModel.compileModel(at: url)
    return try MLModel(contentsOf: compiledUrl, configuration: config)
}

// Usage:
// let cpuModel = try loadModel(url: modelUrl, with:.cpuOnly)
// let gpuModel = try loadModel(url: modelUrl, with:.cpuAndGPU)
// let aneModel = try loadModel(url: modelUrl, with:.all)



Python: Dumping MIL Graph for Inspection


Python


import coremltools as ct

def dump_mil_representation(model_path: str, output_path: str):
    """
    Converts a model to MIL and prints the program representation for debugging.
    This requires re-running the conversion with debug flags.
    """
    # This is an example assuming a PyTorch source model `torch_model`
    # The key is the `debug=True` flag in ct.convert()
    
    # traced_model = torch.jit.trace(torch_model, example_input)
    # mlmodel = ct.convert(
    #     traced_model,
    #     convert_to="mlprogram",
    #     inputs=,
    #     debug=True  # This will print MIL to stdout during conversion
    # )
    
    # If you already have the.mlpackage, getting the MIL is harder post-facto.
    # The best way is to find the model.mil file inside the compiled.mlmodelc
    # xcrun coremlcompiler compile model.mlpackage.
    # Then inspect the contents of model.mlmodelc/
    print("To inspect MIL, re-run the original conversion with `debug=True`")
    print("Alternatively, compile the.mlpackage with `xcrun coremlcompiler` and find 'model.mil' inside the resulting.mlmodelc directory.")




C. Common CoreML Runtime Error Matrix

This table maps common, often cryptic, CoreML runtime errors to their most likely root causes and recommended first-line diagnostic actions.

Error Signature
Likely Root Cause(s)
Recommended First Actions
Relevant Sources
Cannot retrieve vector from IRValue format int32
1. Internal Graph Inconsistency: An op (e.g., reshape) expects a constant shape parameter but receives a dynamic tensor due to a PyTorch tracing artifact. 2. Severe Shape/Rank Mismatch: A gross mismatch between a passed tensor and the model's internal expectations.
1. Force .cpuOnly execution to isolate the backend. 2. Inspect the MIL graph for shape-deriving ops (reshape, tile). 3. Rigorously log and compare runtime tensor shapes vs. model spec shapes.
1
E5RT flexible-shape strides / unknown strides
1. Unsupported Shape for Backend: The ANE or GPU cannot compute the memory layout (strides) for the given flexible shape. 2. Incorrect RangeDim Usage: Using unbounded ranges (upper_bound=-1) with the mlprogram backend is forbidden.
1. Force .cpuOnly execution. 2. Use the Core ML Instrument to check for backend fallbacks. 3. Switch to EnumeratedShapes or a more constrained RangeDim during export.
17
MTLTextureDescriptor has width (...) greater than 16384
GPU Backend Limit: A tensor dimension exceeds the maximum texture width supported by the Metal framework on the GPU.
1. Verify the dimensions of all input tensors, especially 2D matrices like alignment maps. 2. Reduce batch size or sequence length if possible. 3. Force .cpuOnly or .cpuAndNeuralEngine to bypass the GPU.
38
ANECCompile failed
ANE Incompatibility: An operation, data type, or shape configuration in the model graph is not supported by the Apple Neural Engine hardware.
1. Use the Core ML Instrument to identify the specific layer that is failing on the ANE. 2. Force .cpuAndGPU execution to confirm the model works without the ANE. 3. Modify the source model architecture to replace the unsupported layer or operation.
33
validator error:... input rank X but expects rank Y
Rank Mismatch: The number of dimensions of a provided tensor is incorrect (e.g., passing a 2D tensor where a 3D tensor with a batch dimension is expected).
1. Programmatically log the rank (tensor.shape.count in Swift) of all input tensors. 2. Compare against the rank defined in the model specification. 3. Ensure batch dimensions are consistently added/removed as required by the model.
48
BlobWriter not loaded
Toolchain/Dependency Issue: Often seen with coremltools when there is an incompatibility with the protobuf library version, or an incomplete installation.
1. Create a clean Python virtual environment. 2. Reinstall coremltools and its dependencies (pip install --force-reinstall coremltools). 3. Ensure protobuf version is compatible with the coremltools version.
50


D. Annotated Bibliography

This section provides a summary of the key takeaways from the source materials consulted for this report.
1 Hugging Face Forums - CLIP to CoreML Conversion:
Highlights that PyTorch tracing can produce incorrect graphs if the model's behavior changes with input shape.
Discusses the difference between Neural Network and ML Program formats, noting ML Program allows for FP32 precision on the GPU.
Recommends verifying conversion correctness by first running on CPU-only to establish a numerical baseline.
52
coremltools Source - Input Types:
Shows internal coremltools type checking logic.
Confirms that EnumeratedShapes cannot be used with optional inputs that have a default_value.
Lists the supported NumPy data types for model inputs/outputs.
2 Apple Tech Talk - Convert PyTorch models to Core ML:
Explicitly states that a runtime error hinting at a type mismatch can occur if the JIT scripter cannot infer an object's type and incorrectly assumes it is a tensor (e.g., a list of integers becomes a list of tensors).
This is a direct parallel to the primary hypothesis for the IRValue int32 error.
53 ONNX Runtime - CoreML Execution Provider:
Details how to use CoreML as a backend for ONNX Runtime.
Shows that the MLProgram format can be explicitly requested as a configuration option.
18
coremltools Docs - Flexible Inputs:
Details EnumeratedShapes and RangeDim for defining flexible input shapes.
States that unbounded ranges are not permitted when converting to an ML Program.
Notes that for multi-input models with EnumeratedShapes (pre-iOS 18), all inputs must have the same number of shapes and are matched by index.
6
coremltools FAQs:
Recommends using ComputeUnit.CPU_ONLY as a workaround and debugging step for conversion or runtime errors.
Explains the evolution from .mlmodel to the .mlpackage directory format.
Notes that converting a fixed-shape model to use flexible inputs with EnumeratedShapes is the best way to maintain ANE compatibility.
28 Apple Docs -
MLShapedArray:
Describes MLShapedArray as the modern Swift counterpart to MLMultiArray.
Explains how to programmatically inspect a model's input/output constraints at runtime by accessing modelDescription.inputDescriptionsByName and the multiArrayConstraint property.
29 Apple Docs -
MLMultiArray:
Provides the Objective-C/Swift API for MLMultiArray.
Details properties like .shape and .strides, which are crucial for creating and debugging multi-dimensional arrays.
54 GitHub Issue - CoreMLHelpers:
Discusses the necessity of converting UIImage to MLMultiArray when a model does not accept CVPixelBuffer.
Highlights potential complexities with memory layout and strides when manually populating an MLMultiArray.
55 WWDC 2021 - Tune your Core ML models:
Introduces MLShapedArray as an easier way to work with multi-dimensional data in Swift.
Promotes the .mlpackage format for its flexibility and more efficient compilation.
56 Microsoft Docs -
MLMultiArray.Strides:
Explains that the strides property defines the memory layout of the multi-array, indicating how many elements to skip to advance one index in a given dimension. A mismatch in stride calculation can lead to memory corruption or errors.
57 Medium - Core ML and
MLMultiArray:
Provides a basic tutorial on creating and accessing MLMultiArray elements in Swift.
18
coremltools Docs - Flexible Input Shapes (Multiple Versions):
Consistently emphasizes that EnumeratedShapes provides the best performance due to on-device optimization for a finite set of shapes.
Warns that RangeDim offers more flexibility but is harder for the runtime to optimize and has limitations (e.g., no unbounded ranges for MLProgram).
Documents the pre-iOS 18 limitation that only one input could use EnumeratedShapes.
58 Medium - Set Flexible Input Shape for CoreML:
Provides a simple example of using ct.RangeDim() to allow variable-sized image inputs.
22
coremltools Docs - New Conversion Options:
Clarifies that the converter produces an MLProgram by default for modern deployment targets (iOS 15+) and that the default precision for MLProgram is FP16.
Re-iterates that compute_units can be specified during conversion for debugging purposes.
59
coremltools Docs - Model Input and Output Types:
States that for PyTorch models, the input shape must be provided during conversion.
Recommends providing a static shape even for TensorFlow models to enable graph optimizations.
20 GitHub Issue -
coremltools #2548:
Reports a specific bug: A model doesn't allow a mixture of enumerated and range shape flexibility. This demonstrates that interactions between different flexibility types can be a source of errors.
21 GitHub Issue -
coremltools #2370:
Provides critical community-driven performance analysis.
Reports that a model with RangeDim input runs on the ANE only with the default shape, falling back for other shapes in the range.
Reports that for mlprogram models, even EnumeratedShapes may only run on the ANE for the default shape, a regression from the neuralnetwork format.
60
coremltools Docs - RangeDim API:
Defines the RangeDim class constructor, showing its parameters lower_bound, upper_bound, and default.
61
coremltools Docs - Image Inputs:
Explains how to specify ImageType vs. the default MLMultiArray for inputs.
Notes that using ImageType is more efficient for CVPixelBuffer data paths.
62 GitHub Issue -
coremltools #1827:
A user suggests watching WWDC talks on optimization and using Instruments to determine if a flexible shape is causing a fallback from the ANE to the CPU.
42
coremltools Docs - New in coremltools:
Summarizes features in recent versions, including the introduction of the MLProgram model type and .mlpackage container format in version 5.
63 NVIDIA Docs - TensorRT-LLM Troubleshooting:
While for a different framework, it provides a valuable parallel for debugging shape errors.
Recommends double-checking the rank and dimensions of input tensors and using verbose logging to print expected vs. actual shapes at runtime.
64 Medium - How to Visualize Tensors:
Provides a basic primer on tensor terminology (rank, shape), which is fundamental to debugging these types of issues.
48 Stack Overflow - CoreML Tensor Rank Mismatch:
An answer suggests that rank mismatch errors are common and often occur when a layer expects a tensor of a certain rank but receives another (e.g., expecting 2D but getting 3D).
50 PyTorch Docs - ExecuTorch Core ML Backend:
Documents the coremltools.ComputeUnit enum values (ALL, CPU_ONLY, CPU_AND_GPU, CPU_AND_NE).
Notes that the ANE only supports FP16 precision, and the converter defaults to FP16 for MLProgram.
Lists a common ValueError related to dtype mismatches between different inputs to an op.
17 Apple Developer Forums - Core ML Topics:
A user reports a memory stride warning: Using unknown strides for MIL tensor buffers with unknown shapes is not recommended in E5ML. This directly links flexible shapes to potential memory layout issues in the MLProgram runtime.
65 GitHub Issue -
coremltools #1953:
Highlights that toolchain version incompatibility (e.g., TensorFlow version used to save a model vs. version used for conversion) can cause internal conversion errors.
31 Apple Docs - Core ML Overview:
States that Core ML is designed to leverage CPU, GPU, and Neural Engine to maximize performance.
Introduces Xcode performance reports for analyzing compute unit usage.
36 Apple Docs -
MLComputeUnits:
Defines the cases of the MLComputeUnits enum, including .cpuOnly.
67 WWDC Videos - Optimize your Core ML usage:
Reinforces the use of the Core ML Instrument for understanding model performance and backend execution.
69
coremltools Docs - Optimization Overview:
Discusses how runtime performance gains from compression depend heavily on the specific hardware and compute unit, as implementations of compressed kernels vary.
Recommends testing on the specific target Apple Silicon to verify performance.
70 Reddit - How Neural Engine benefits:
Provides a high-level explanation that developers can choose which compute unit is best suited for their algorithm (CPU, GPU, or ANE).
40 GitHub Issue - ml-stable-diffusion #122:
A collaborator confirms a known issue where specifying a more restricted compute unit set (e.g., .cpuAndNeuralEngine) can yield better performance than .all, demonstrating the complexity of the Core ML planner.
9 Apple Docs -
MLFeatureDescription:
The foundational documentation for programmatically inspecting a model's features and their constraints in Swift.
71 Zignuts Blog - How to use Core ML:
Notes that some operations work best on the ANE while others are better suited for the GPU, and profiling is necessary to understand a model's hardware affinity.
72
coremltools Docs - NeuralNetwork.proto:
Provides low-level details of the legacy Neural Network format specification.
Explains how axes are interpreted in an N-dimensional setting (e.g., "width" is axis -1).
55 WWDC 2021 - Tune your Core ML models:
Introduces MLShapedArray and the .mlpackage format.
Shows that a model's prediction result can have a shapedArray property for easier manipulation in Swift.
38 Apple Docs - Metal Feature Set Tables:
Provides the definitive hardware limits for Apple GPUs.
Explicitly states the maximum 2D texture width and height is 16,384 pixels for Apple Silicon GPUs (Apple Family 2 through 9).
39 GitHub Issue -
coremltools #283:
A real-world example of a user hitting the MTLTextureDescriptor has width (...) greater than the maximum allowed size of 16384 error.
The user's workaround was to resize their input and retrain the model, confirming this is a hard limit.
73 Stack Overflow - Metal & CoreML:
Discusses the complexities of synchronizing CPU and GPU timelines when working with Metal resources.
Shows an example of incorrect stride calculation (destinationBytesPerRow) when copying data from an MTLTexture.
75 WWDC 2025 - Metal 4 (Hypothetical/Future-dated):
Introduces MTLTensors as a more flexible data container for ML workloads compared to MTLTextures, which have strict channel and extent limits.
76 Apple Docs - Metal Performance Shaders:
Describes MPS as a framework of highly optimized compute and graphics shaders tuned for each Metal GPU family. Core ML's GPU backend relies heavily on this framework.
5
coremltools Docs - Model Prediction:
Shows how to set compute_units=ct.ComputeUnit.CPU_ONLY when loading a model in Python for debugging.
Explains the two-stage compilation process (.mlpackage -> .mlmodelc -> device-specialized cache) and how the cache is keyed by the .mlmodelc path.
77 Krisp Blog - Integrate CoreML into C++:
Explains that CoreML models must be compiled to .mlmodelc before use.
Notes that this compilation can be done ahead-of-time with xcrun coremlc or at runtime by Xcode.
78 Reddit - iOS Programming:
A comment mentions that a CoreML model may fall back from ANE to CPU for some layers.
32 Apple ML Research - Neural Engine Transformers:
States that Core ML "seamlessly blends CPU, GPU, and ANE... to create the most effective hybrid execution plan."
33 Fritz AI Blog - Does my model run on ANE?:
Provides a detailed, step-by-step guide on using Xcode Instruments to determine which hardware a model is running on.
Explains how to filter instrument logs for "ANE" calls (indicating ANE usage) or "Metal"/"MTL" calls (indicating GPU usage).
41
coremltools GitHub Releases:
The primary source for changes, bug fixes, and new features in coremltools.
Version 8.3.0 introduced a suite of debugging utilities (MLModelValidator, MLModelComparator, MLModelBenchmarker) highly relevant to the current problem.
79 fast.ai Forums - coremltools & PyTorch incompatibility:
A historical example of how coremltools versions can be tightly coupled with specific PyTorch versions, causing conversion failures.
80 Stack Overflow - PyTorch 'len' op not implemented:
An example of a standard Python function (len()) causing a conversion error because the tracer attempts to convert it into a graph operation, which is not supported.
51 Stack Overflow -
coremltools tag:
A collection of user-reported issues, showing common problems like BlobWriter not loaded errors, which often point to environment or dependency issues.
81 GitHub Issue -
coremltools #1535:
A coremltools maintainer notes that support for PyTorch Scripting is experimental and recommends using traced models, but acknowledges that complex models can reveal bugs in the converter.
82 Apple Developer Forums - sklearn conversion error:
A user discovers that their model conversion failed because CoreML did not support float as a target class label type; it had to be int or string. This shows how CoreML can have strict type requirements.
6
coremltools FAQs (v6.3):
A comprehensive guide that recommends filing GitHub issues for persistent errors and trying CPUOnly as a debugging step.
3 Hugging Face Forums - BigBird conversion:
A log file filled with TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. This is a canonical example of the tracing pitfalls that are the likely root cause of the IRValue error.
83 Stack Overflow - Error converting TensorFlow model:
An answer clarifies the different values for the convert_to parameter, including "mlprogram" and "neuralnetwork".
24
coremltools Docs - Comparing ML Programs and Neural Networks:
The definitive guide to the differences between the two formats, covering typed execution, GPU runtime precision, and weight serialization.
Explains that MLProgram can only be saved as an .mlpackage.
43 Stack Overflow -
.mlmodel vs .mlpackage:
Provides a community-sourced summary of the differences, reinforcing that .mlpackage is the more modern, flexible format that separates architecture from weights.
17 Apple Developer Forums - Various Topics:
Contains user reports on various CoreML issues, including model decryption, caching behavior after app updates, and regressions in flexible shape support. The caching question in 84 is particularly relevant.
4 WWDC 2023 - Improve Core ML integration with async prediction:
Provides a clear, authoritative explanation of the model lifecycle and caching.
States that the cache is tied to the model's path and configuration, and that the OS deletes the cache when disk space is low, on system updates, or if the compiled model is modified.
5
coremltools Docs - Using Compiled Python Models:
Explicitly warns that because coremltools in Python uses a temporary directory for the .mlmodelc, the device specialization cache is not used across sessions.
Recommends a workaround: manually compiling to a persistent path and loading from there to leverage caching, which is the exact solution proposed in this report.
45 Articles on ML Deployment Best Practices:
Emphasize the importance of model versioning, containerization, CI/CD automation, and monitoring for robust production deployments.
47
coremltools Docs - Multifunction Models:
Introduces the concept of merging multiple models (e.g., LoRA adapters) into a single .mlpackage to share weights and reduce asset size.
44 Ultralytics Docs - CoreML Deployment:
Describes on-device vs. cloud-based deployment options for CoreML models.
Shows a simple workflow for exporting a YOLO model to .mlpackage.
87
coremltools Docs - Introductory Quickstart:
A tutorial that demonstrates setting model metadata (author, license, input/output descriptions) to improve integration with Xcode.
29 Docs/Code on
MLMultiArray:
A collection of sources providing API details, constructors, and usage examples for MLMultiArray in Swift, Objective-C, and C#.
19
coremltools Docs/Issues on Flexible Inputs:
Provide examples of how to define flexible shapes and debug issues related to them, such as needing to manually modify the protobuf spec to adjust tensor ranks.
7 Community Discussions on Forcing CPU:
A collection of Reddit, Stack Overflow, and GitHub threads where users discuss forcing CPU execution for performance testing, debugging, or working around GPU/ANE issues.
95 Docs/Blogs on CoreML Debugging:
General guides on debugging CoreML, recommending the use of Xcode's built-in debugger, Instruments, and model optimization techniques like quantization and pruning.
10
coremltools Docs/Examples on get_spec():
Numerous examples demonstrating how to load a model and access its underlying protobuf spec to inspect and modify its properties, such as input/output descriptions.
95 Apple Docs - Core ML (Top Level):
High-level documentation introducing the MLComputePlan and MLTensor APIs for more advanced control over execution.
1 Hugging Face Forums - CLIP Conversion Discrepancy:
A user reports numerical differences between their original PyTorch model and the converted CoreML model, and an expert suggests forcing CPU execution to debug and measure the signal-to-noise ratio.
11 Swift Code Examples:
Real-world Swift code from Hugging Face, Apple, and Stack Overflow that demonstrates accessing model.modelDescription.inputDescriptionsByName to get shape constraints and other model properties at runtime.
30 Docs on
MLMultiArrayConstraint:
API documentation detailing the MLMultiArrayConstraint class, which holds the shape and data type constraints for a multi-array feature.
110 PyPI -
coremltools:
Project description page, confirming the purpose of the package.
Works cited
Converting CLIP to CoreML - Transformers - Hugging Face Forums, accessed August 21, 2025, https://discuss.huggingface.co/t/converting-clip-to-coreml/31345
Convert PyTorch models to Core ML - Tech Talks - Videos - Apple ..., accessed August 21, 2025, https://developer.apple.com/videos/play/tech-talks/10154/
Conversion to CoreML for On-Device Use - Models - Hugging Face Forums, accessed August 21, 2025, https://discuss.huggingface.co/t/conversion-to-coreml-for-on-device-use/13284
Improve Core ML integration with async prediction - WWDC23 ..., accessed August 21, 2025, https://developer.apple.com/videos/play/wwdc2023/10049/
Model Prediction — Guide to Core ML Tools - Apple, accessed August 21, 2025, https://apple.github.io/coremltools/docs-guides/source/model-prediction.html
Core ML Tools FAQs, accessed August 21, 2025, https://coremltools.readme.io/v6.3/docs/faqs
Force your model to Run CPU in iOS | by Marc StevenCoder - Medium, accessed August 21, 2025, https://medium.com/@MarcStevenCoder/force-your-model-to-run-cpu-in-ios-58c53398c901
Unable to run model on iPhone 14 Pro with error: "failed to load ANE model" — works fine from CLI · Issue #51 · apple/ml-stable-diffusion - GitHub, accessed August 21, 2025, https://github.com/apple/ml-stable-diffusion/issues/51
MLFeatureDescription | Apple Developer Documentation, accessed August 21, 2025, https://developer.apple.com/documentation/coreml/mlfeaturedescription
MLModel Utilities — Guide to Core ML Tools - Apple, accessed August 21, 2025, https://apple.github.io/coremltools/docs-guides/source/mlmodel-utilities.html
swift-transformers/Sources/Models/LanguageModel.swift at main - GitHub, accessed August 21, 2025, https://github.com/huggingface/swift-transformers/blob/main/Sources/Models/LanguageModel.swift
inputDescriptionsByName | Apple Developer Documentation, accessed August 21, 2025, https://developer.apple.com/documentation/coreml/mlmodeldescription/inputdescriptionsbyname
MLModel Utilities - Core ML Tools, accessed August 21, 2025, https://coremltools.readme.io/v6.3/docs/mlmodel-utilities
MLModel Overview — Guide to Core ML Tools - Apple, accessed August 21, 2025, https://apple.github.io/coremltools/docs-guides/source/mlmodel.html
Model Prediction - Core ML Tools, accessed August 21, 2025, https://coremltools.readme.io/v6.3/docs/model-prediction
Split a pre-trained CoreML model into two CoreML Models ? · Issue #427 · apple/coremltools - GitHub, accessed August 21, 2025, https://github.com/apple/coremltools/issues/427
Core ML - Apple Developer Forums, accessed August 21, 2025, https://developer.apple.com/forums/forums/topics/machine-learning-and-ai/machine-learning-topic-core-ml
Flexible Input Shapes — Guide to Core ML Tools - Apple, accessed August 21, 2025, https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html
Flexible Input Shapes - Core ML Tools, accessed August 21, 2025, https://coremltools.readme.io/v4.0/docs/flexible-inputs
Stateful MIL to CoreML breaks when fixed and flexible inputs are ..., accessed August 21, 2025, https://github.com/apple/coremltools/issues/2548
Flexible Input Shapes on Neural Engine · Issue #2370 · apple/coremltools - GitHub, accessed August 21, 2025, https://github.com/apple/coremltools/issues/2370
Conversion Options - Core ML Tools, accessed August 21, 2025, https://coremltools.readme.io/v6.3/docs/neural-network-conversion
New Conversion Options — Guide to Core ML Tools - Apple, accessed August 21, 2025, https://apple.github.io/coremltools/docs-guides/source/new-conversion-options.html
Comparing ML Programs and Neural Networks — Guide to Core ML ..., accessed August 21, 2025, https://apple.github.io/coremltools/docs-guides/source/comparing-ml-programs-and-neural-networks.html
coremltools/coremltools/models/utils.py at main · apple/coremltools - GitHub, accessed August 21, 2025, https://github.com/apple/coremltools/blob/master/coremltools/models/utils.py
ML Programs - Core ML Tools, accessed August 21, 2025, https://coremltools.readme.io/v6.3/docs/ml-programs
Model Prediction - Core ML Tools, accessed August 21, 2025, https://coremltools.readme.io/v4.0/docs/model-prediction
MLShapedArray | Apple Developer Documentation, accessed August 21, 2025, https://developer.apple.com/documentation/coreml/mlshapedarray
MLMultiArray | Apple Developer Documentation, accessed August 21, 2025, https://developer.apple.com/documentation/coreml/mlmultiarray
MLMultiArrayConstraint | Apple Developer Documentation, accessed August 21, 2025, https://developer.apple.com/documentation/coreml/mlmultiarrayconstraint
Core ML - Machine Learning - Apple Developer, accessed August 21, 2025, https://developer.apple.com/machine-learning/core-ml/
Deploying Transformers on the Apple Neural Engine - Apple Machine Learning Research, accessed August 21, 2025, https://machinelearning.apple.com/research/neural-engine-transformers
Does my Core ML model run on Apple's Neural Engine? - Fritz ai, accessed August 21, 2025, https://fritz.ai/does-my-core-ml-model-run-on-apples-neural-engine/
Vision Pro CoreML seem to only run on CPU (10x slower) - Reddit, accessed August 21, 2025, https://www.reddit.com/r/visionosdev/comments/1aue2jw/vision_pro_coreml_seem_to_only_run_on_cpu_10x/
CoreML / MLModelConfig preferredMetalDevice - understanding device placement heuristics - Stack Overflow, accessed August 21, 2025, https://stackoverflow.com/questions/58437789/coreml-mlmodelconfig-preferredmetaldevice-understanding-device-placement-heu
MLComputeUnits.cpuOnly | Apple Developer Documentation, accessed August 21, 2025, https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuonly
diffusers/docs/source/en/optimization/coreml.md at main - GitHub, accessed August 21, 2025, https://github.com/huggingface/diffusers/blob/main/docs/source/en/optimization/coreml.md
Metal Feature Set Tables | Apple Developer, accessed August 21, 2025, https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
MTLTextureDescriptor has width (16900) greater than the maximum ..., accessed August 21, 2025, https://github.com/apple/coremltools/issues/283
.all vs .cpuAndNeuralEngine? · Issue #122 · apple/ml-stable-diffusion - GitHub, accessed August 21, 2025, https://github.com/apple/ml-stable-diffusion/issues/122
Releases · apple/coremltools - GitHub, accessed August 21, 2025, https://github.com/apple/coremltools/releases
New Features - Core ML Tools, accessed August 21, 2025, https://coremltools.readme.io/v6.3/docs/new-in-coremltools
Difference between mlmodel and mlpackage - coreml - Stack Overflow, accessed August 21, 2025, https://stackoverflow.com/questions/78229310/difference-between-mlmodel-and-mlpackage
CoreML Export for YOLO11 Models - Ultralytics YOLO Docs, accessed August 21, 2025, https://docs.ultralytics.com/integrations/coreml/
Best Practices for Deploying Machine Learning Models in Production - Medium, accessed August 21, 2025, https://medium.com/@nemagan/best-practices-for-deploying-machine-learning-models-in-production-10b690503e6d
ML Model Packaging [The Ultimate Guide] - Neptune.ai, accessed August 21, 2025, https://neptune.ai/blog/ml-model-packaging
Multifunction Models — Guide to Core ML Tools - Apple, accessed August 21, 2025, https://apple.github.io/coremltools/docs-guides/source/multifunction-models.html
coremltools Error: ValueError: perm should have the same length as rank(x): 3 != 2, accessed August 21, 2025, https://stackoverflow.com/questions/79153512/coremltools-error-valueerror-perm-should-have-the-same-length-as-rankx-3
coremltools: how to properly use NeuralNetworkMultiArrayShapeRange? - Stack Overflow, accessed August 21, 2025, https://stackoverflow.com/questions/59662399/coremltools-how-to-properly-use-neuralnetworkmultiarrayshaperange
Core ML Backend — ExecuTorch 0.7 documentation, accessed August 21, 2025, https://docs.pytorch.org/executorch/stable/backends-coreml.html
Newest 'coremltools' Questions - Stack Overflow, accessed August 21, 2025, https://stackoverflow.com/questions/tagged/coremltools
Source code for coremltools.converters.mil.input_types - Apple, accessed August 21, 2025, https://apple.github.io/coremltools/_modules/coremltools/converters/mil/input_types.html
CoreML - Apple - ONNX Runtime, accessed August 21, 2025, https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
UIImage to MLMultiArray ? · Issue #5 · hollance/CoreMLHelpers - GitHub, accessed August 21, 2025, https://github.com/hollance/CoreMLHelpers/issues/5
Tune your Core ML models - WWDC21 - Videos - Apple Developer, accessed August 21, 2025, https://developer.apple.com/videos/play/wwdc2021/10038/
MLMultiArray.Strides Property (CoreML) | Microsoft Learn, accessed August 21, 2025, https://learn.microsoft.com/en-us/dotnet/api/coreml.mlmultiarray.strides?view=xamarin-ios-sdk-12
Working with Multi-Dimensional Arrays in iOS Development with Swift and Core ML, accessed August 21, 2025, https://medium.com/@ios_guru/core-ml-and-mlmultiarray-for-working-with-multi-dimensional-arrays-7c51ceefc54e
Set flexible input shape for CoreML Model | by MLBoy - Medium, accessed August 21, 2025, https://rockyshikoku.medium.com/set-flexible-input-shape-for-coreml-model-9b873fda5310
Model Input and Output Types — Guide to Core ML Tools - Apple, accessed August 21, 2025, https://apple.github.io/coremltools/docs-guides/source/model-input-and-output-types.html
MIL Input Types — coremltools API Reference 8.1 documentation - Apple, accessed August 21, 2025, https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html
Image Input and Output - Core ML Tools, accessed August 21, 2025, https://coremltools.readme.io/v6.3/docs/image-inputs
Can multi-input networks support enumerate shape? · Issue #1827 · apple/coremltools, accessed August 21, 2025, https://github.com/apple/coremltools/issues/1827
Troubleshooting — TensorRT-LLM - GitHub Pages, accessed August 21, 2025, https://nvidia.github.io/TensorRT-LLM/reference/troubleshooting.html
How to visualize tensors while debugging? | by Adrian Boguszewski - Medium, accessed August 21, 2025, https://medium.com/@adrianboguszewski/how-to-visualize-tensors-while-debugging-1d3d50d9b3f1
Problem converting Tenserflow model to CoreML · Issue #1953 · apple/coremltools - GitHub, accessed August 21, 2025, https://github.com/apple/coremltools/issues/1953
MLComputeUnits.cpuAndGPU | Apple Developer Documentation, accessed August 21, 2025, https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandgpu
Optimize your Core ML usage - WWDC22 - Videos - Apple Developer, accessed August 21, 2025, https://developer.apple.com/videos/play/wwdc2022/10027/
WWDC22: Optimize your Core ML usage | Apple - YouTube, accessed August 21, 2025, https://www.youtube.com/watch?v=THXq071qZ6E&vl=en
Overview — Guide to Core ML Tools - Apple, accessed August 21, 2025, https://apple.github.io/coremltools/docs-guides/source/opt-overview.html
How exactly does the Neural Engine benefit the consumer? : r/apple - Reddit, accessed August 21, 2025, https://www.reddit.com/r/apple/comments/qbawpk/how_exactly_does_the_neural_engine_benefit_the/
How to Use Core ML in iOS: A Complete Guide with Examples - Zignuts Technolab, accessed August 21, 2025, https://www.zignuts.com/blog/how-to-use-core-ml-in-ios-guide
NeuralNetwork — Core ML Format Reference documentation - Apple, accessed August 21, 2025, https://apple.github.io/coremltools/mlmodel/Format/NeuralNetwork.html
In Metal, how to wait for all GPU operations in my process to complete? - Stack Overflow, accessed August 21, 2025, https://stackoverflow.com/questions/78171909/in-metal-how-to-wait-for-all-gpu-operations-in-my-process-to-complete
Issue creating MTLBuffer from MTLTexture used as inputs in CoreML Custom Layer for GPU execution - Stack Overflow, accessed August 21, 2025, https://stackoverflow.com/questions/72031359/issue-creating-mtlbuffer-from-mtltexture-used-as-inputs-in-coreml-custom-layer-f
Combine Metal 4 machine learning and graphics - WWDC25 - Videos - Apple Developer, accessed August 21, 2025, https://developer.apple.com/videos/play/wwdc2025/262/
Metal Performance Shaders | Apple Developer Documentation, accessed August 21, 2025, https://developer.apple.com/documentation/metalperformanceshaders
How to Integrate CoreML Models Into C/C++ Codebase - Krisp, accessed August 21, 2025, https://krisp.ai/blog/how-to-integrate-coreml-models-into-c-c-codebase/
Does running models on mobile consume so many resources? : r, accessed August 21, 2025, https://www.reddit.com/r/iOSProgramming/comments/1gcc2xh/does_running_models_on_mobile_consume_so_many/
Coremltools & Pytorch incompatibility - Part 1 (2020) - fast.ai Course Forums, accessed August 21, 2025, https://forums.fast.ai/t/coremltools-pytorch-incompatibility/83899
PyTorch convert function for op 'len' not implemented for coremltools - Stack Overflow, accessed August 21, 2025, https://stackoverflow.com/questions/69948623/pytorch-convert-function-for-op-len-not-implemented-for-coremltools
Facing issues while converting a pre-trained PyTorch model to CoreML #1535 - GitHub, accessed August 21, 2025, https://github.com/apple/coremltools/issues/1535
CoreML Error when converting from sklearn RandomForestClassifier - Apple Developer, accessed August 21, 2025, https://developer.apple.com/forums/thread/99785
Error converting Tensorflow model to CoreML model - Stack Overflow, accessed August 21, 2025, https://stackoverflow.com/questions/78761234/error-converting-tensorflow-model-to-coreml-model
Core ML | Apple Developer Forums, accessed August 21, 2025, https://developer.apple.com/forums/tags/core-ml/?page=4&sortBy=oldest
Core ML | Apple Developer Forums, accessed August 21, 2025, https://developer.apple.com/forums/tags/core-ml
Model packages for deployment (preview) - Azure Machine Learning, accessed August 21, 2025, https://learn.microsoft.com/en-us/azure/machine-learning/concept-package-models?view=azureml-api-2
Getting Started — Guide to Core ML Tools - Apple, accessed August 21, 2025, https://apple.github.io/coremltools/docs-guides/source/introductory-quickstart.html
MLMultiArray Constructor (CoreML) - Learn Microsoft, accessed August 21, 2025, https://learn.microsoft.com/en-us/dotnet/api/coreml.mlmultiarray.-ctor?view=xamarin-mac-sdk-14
MLMultiArray+Image.swift - hollance/CoreMLHelpers - GitHub, accessed August 21, 2025, https://github.com/hollance/CoreMLHelpers/blob/master/CoreMLHelpers/MLMultiArray%2BImage.swift
MLMultiArray in objc2_core_ml - Rust - Docs.rs, accessed August 21, 2025, https://docs.rs/objc2-core-ml/latest/objc2_core_ml/struct.MLMultiArray.html
Objective-C Runtime | Apple Developer Forums, accessed August 21, 2025, https://developer.apple.com/forums/tags/objective-c-runtime
Xcode and CoreML : r/swift - Reddit, accessed August 21, 2025, https://www.reddit.com/r/swift/comments/1i3pkv9/xcode_and_coreml/
Can I force Swift code to utilize all cores? - Stack Overflow, accessed August 21, 2025, https://stackoverflow.com/questions/41839948/can-i-force-swift-code-to-utilize-all-cores
How to enable CoreML Execution Provider on Mac · Issue #166 · s0md3v/sd-webui-roop, accessed August 21, 2025, https://github.com/s0md3v/sd-webui-roop/issues/166
Core ML | Apple Developer Documentation, accessed August 21, 2025, https://developer.apple.com/documentation/coreml
Core ML Tools, accessed August 21, 2025, https://coremltools.readme.io/
Mastering Core ML for Advanced Apps - Number Analytics, accessed August 21, 2025, https://www.numberanalytics.com/blog/mastering-core-ml
Mastering Core ML - A Comprehensive Guide for Advanced iOS Developers - MoldStud, accessed August 21, 2025, https://moldstud.com/articles/p-mastering-core-ml-a-comprehensive-guide-for-advanced-ios-developers
Model APIs — coremltools API Reference 8.1 documentation - Apple, accessed August 21, 2025, https://apple.github.io/coremltools/source/coremltools.models.html
Quickstart Example - Core ML Tools, accessed August 21, 2025, https://coremltools.readme.io/v4.0/docs/introductory-quickstart
Typed Execution Workflow Example - Core ML Tools, accessed August 21, 2025, https://coremltools.readme.io/v6.3/docs/typed-execution-example
coremltools常用操作原创 - CSDN博客, accessed August 21, 2025, https://blog.csdn.net/ssunshining/article/details/116352515
Using Core ML and MLModel.prediction(from:options:) in iOS Development with Swift, accessed August 21, 2025, https://medium.com/@ios_guru/core-ml-and-mlmodel-prediction-from-options-for-making-predictions-with-a-model-12c21c9d9a53
Unet.swift - apple/ml-stable-diffusion - GitHub, accessed August 21, 2025, https://github.com/apple/ml-stable-diffusion/blob/main/swift/StableDiffusion/pipeline/Unet.swift
Does VNCoreMLFeatureValueObservation output softmax probabilities? If so, how to extract top values? - Stack Overflow, accessed August 21, 2025, https://stackoverflow.com/questions/51886316/does-vncoremlfeaturevalueobservation-output-softmax-probabilities-if-so-how-to
How to programmatically increase the height of UIView with Swift - Stack Overflow, accessed August 21, 2025, https://stackoverflow.com/questions/43010173/how-to-programmatically-increase-the-height-of-uiview-with-swift
MLMultiArrayConstraint | Apple Developer Documentation, accessed August 21, 2025, https://developer.apple.com/documentation/coreml/mlmultiarrayconstraint?language=objc
MLMultiArrayConstraint.Shape Property (CoreML) | Microsoft Learn, accessed August 21, 2025, https://learn.microsoft.com/en-us/dotnet/api/coreml.mlmultiarrayconstraint.shape?view=xamarin-ios-sdk-12
How to reshape MLMultipArray in Swift - Stack Overflow, accessed August 21, 2025, https://stackoverflow.com/questions/76138724/how-to-reshape-mlmultiparray-in-swift
coremltools - PyPI, accessed August 21, 2025, https://pypi.org/project/coremltools/
