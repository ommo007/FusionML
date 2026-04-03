// benchmark_pipeline.cpp
// This runs entirely in userspace, no root needed

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/delegates/gpu/delegate.h>
#include <tensorflow/lite/delegates/nnapi/nnapi_delegate.h>
#include <chrono>
#include <thread>
#include <fstream>
#include <iostream>
#include <vector>
#include <mutex>

struct BenchmarkResult {
    std::string config;
    double throughput;
    double latency_ms;
    int num_iterations;
};

// ============================================
// ISOLATED DEVICE BENCHMARKS (no contention)
// ============================================

double benchmark_single_device(const std::string& model_path, 
                                const std::string& device,
                                int warmup, int iterations) {
    auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    
    // Configure delegate based on device
    if (device == "GPU") {
        TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
        gpu_opts.inference_preference = 
            TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
        auto* gpu_delegate = TfLiteGpuDelegateV2Create(&gpu_opts);
        interpreter->ModifyGraphWithDelegate(gpu_delegate);
    } 
    else if (device == "NPU") {
        tflite::StatefulNnApiDelegate::Options nnapi_opts;
        // Modified: Let NNAPI select best NPU fallback instead of forcing hta only
        nnapi_opts.execution_preference = 
            tflite::StatefulNnApiDelegate::Options::kSustainedSpeed;
        auto* nnapi_delegate = new tflite::StatefulNnApiDelegate(nnapi_opts);
        interpreter->ModifyGraphWithDelegate(nnapi_delegate);
    }
    // else CPU - no delegate needed
    
    interpreter->AllocateTensors();
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        interpreter->Invoke();
    }
    
    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        interpreter->Invoke();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return iterations / (elapsed_ms / 1000.0);  // throughput
}

// ============================================
// CONCURRENT EXECUTION (measures contention)
// ============================================

struct ConcurrentResult {
    double gpu_throughput;
    double npu_throughput;
    double gpu_isolated;
    double npu_isolated;
    double alpha_contention;
};

ConcurrentResult benchmark_concurrent(const std::string& model_path,
                                       int warmup, int iterations) {
    ConcurrentResult result;
    
    // First: isolated measurements
    result.gpu_isolated = benchmark_single_device(model_path, "GPU", warmup, iterations);
    result.npu_isolated = benchmark_single_device(model_path, "NPU", warmup, iterations);
    
    // Second: concurrent measurements
    double gpu_concurrent = 0, npu_concurrent = 0;
    
    std::thread gpu_thread([&]() {
        gpu_concurrent = benchmark_single_device(model_path, "GPU", warmup, iterations);
    });
    
    std::thread npu_thread([&]() {
        npu_concurrent = benchmark_single_device(model_path, "NPU", warmup, iterations);
    });
    
    gpu_thread.join();
    npu_thread.join();
    
    result.gpu_throughput = gpu_concurrent;
    result.npu_throughput = npu_concurrent;
    
    // Alpha contention = throughput retained under contention
    result.alpha_contention = gpu_concurrent / result.gpu_isolated;
    
    return result;
}

// ============================================
// PIPELINED EXECUTION (your FusionML approach)
// ============================================

double benchmark_pipeline(const std::string& model_path,
                          int warmup, int iterations) {
    // Build two interpreters: one for GPU, one for NPU
    auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    tflite::ops::builtin::BuiltinOpResolver resolver;
    
    // GPU interpreter
    std::unique_ptr<tflite::Interpreter> gpu_interp;
    tflite::InterpreterBuilder(*model, resolver)(&gpu_interp);
    TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
    auto* gpu_delegate = TfLiteGpuDelegateV2Create(&gpu_opts);
    gpu_interp->ModifyGraphWithDelegate(gpu_delegate);
    gpu_interp->AllocateTensors();
    
    // NPU interpreter  
    std::unique_ptr<tflite::Interpreter> npu_interp;
    tflite::InterpreterBuilder(*model, resolver)(&npu_interp);
    tflite::StatefulNnApiDelegate::Options nnapi_opts;
    auto* nnapi_delegate = new tflite::StatefulNnApiDelegate(nnapi_opts);
    npu_interp->ModifyGraphWithDelegate(nnapi_delegate);
    npu_interp->AllocateTensors();
    
    // Double-buffered pipeline execution
    // Sample k on GPU while sample k-1 on NPU
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        gpu_interp->Invoke();
        npu_interp->Invoke();
    }
    
    int completed = 0;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Pipeline: overlap GPU and NPU execution
    for (int i = 0; i < iterations; i++) {
        std::thread gpu_work([&]() { gpu_interp->Invoke(); });
        std::thread npu_work([&]() { npu_interp->Invoke(); });
        gpu_work.join();
        npu_work.join();
        completed += 2;  // Both devices processed a sample
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    return completed / (elapsed_ms / 1000.0);
}

// ============================================
// MAIN - Run all benchmarks, save JSON results
// ============================================

int main(int argc, char* argv[]) {
    std::string output_dir = "/data/local/tmp/fusionml_results/";
    int warmup = 20;
    int iterations = 100;
    
    std::vector<std::string> models = {
        "/data/local/tmp/fusionml/resnet50.tflite",
        "/data/local/tmp/fusionml/bert_base.tflite",
        "/data/local/tmp/fusionml/vit_b16.tflite"
    };
    
    std::ofstream results(output_dir + "results.json");
    results << "{
";
    
    for (auto& model_path : models) {
        std::cout << "Benchmarking: " << model_path << std::endl;
        
        // Isolated benchmarks
        double cpu_tp = benchmark_single_device(model_path, "CPU", warmup, iterations);
        double gpu_tp = benchmark_single_device(model_path, "GPU", warmup, iterations);
        double npu_tp = benchmark_single_device(model_path, "NPU", warmup, iterations);
        
        // Contention measurement
        auto concurrent = benchmark_concurrent(model_path, warmup, iterations);
        
        // Pipeline
        double pipeline_tp = benchmark_pipeline(model_path, warmup, iterations);
        
        // Write results
        results << "  "" << model_path << "": {
";
        results << "    "cpu_throughput": " << cpu_tp << ",
";
        results << "    "gpu_throughput": " << gpu_tp << ",
";
        results << "    "npu_throughput": " << npu_tp << ",
";
        results << "    "gpu_under_contention": " << concurrent.gpu_throughput << ",
";
        results << "    "npu_under_contention": " << concurrent.npu_throughput << ",
";
        results << "    "alpha_contention": " << concurrent.alpha_contention << ",
";
        results << "    "pipeline_throughput": " << pipeline_tp << "
";
        results << "  },
";
        
        std::cout << "  CPU: " << cpu_tp << " items/s" << std::endl;
        std::cout << "  GPU: " << gpu_tp << " items/s" << std::endl;
        std::cout << "  NPU: " << npu_tp << " items/s" << std::endl;
        std::cout << "  Pipeline: " << pipeline_tp << " items/s" << std::endl;
        std::cout << "  Alpha: " << concurrent.alpha_contention << std::endl;
    }
    
    results << "}
";
    results.close();
    
    return 0;
}
