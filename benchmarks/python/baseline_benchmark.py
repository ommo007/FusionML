"""
Baseline Comparisons
====================
Compare FusionML end-to-end pipeline against:
- CoreML Default (ANE + GPU mapping by Apple)
- PyTorch MPS (Native GPU)
- Pure MLX (Native GPU)
"""
import sys
import os
import time
import json
import numpy as np
import torch
import mlx.core as mx
import coremltools as ct

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from fusionml.models.resnet50 import get_pytorch_resnet50, build_fusionml_resnet50_pipeline
from fusionml.models.bert_base import get_pytorch_bert, build_fusionml_bert_pipeline

def get_system_info():
    import platform, subprocess
    info = {"machine": platform.machine()}
    try:
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        info["cpu_brand"] = r.stdout.strip()
    except:
        info["cpu_brand"] = "Unknown"
    return info

def bench_pytorch_mps(model, dummy_input, iters=20):
    try:
        device = torch.device("mps")
        model = model.to(device)
        dummy_input = dummy_input.to(device)
        
        # Warmup
        for _ in range(5):
            model(dummy_input)
            
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
            
        t0 = time.perf_counter()
        for _ in range(iters):
            model(dummy_input)
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        t1 = time.perf_counter()
        
        return (t1 - t0) * 1000.0 / iters
    except Exception as e:
        print(f"PyTorch MPS failed: {e}")
        return float('inf')

def bench_coreml(model, dummy_input, model_name, seq_len=None, iters=20):
    try:
        # Important: move model back to CPU for CoreML tracing
        model = model.to('cpu')
        dummy_input = dummy_input.to('cpu')
        
        if "bert" in model_name.lower():
            class BertWrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x):
                    return self.m(x).last_hidden_state
            wrapped = BertWrapper(model).eval()
            traced = torch.jit.trace(wrapped, dummy_input)
        else:
            traced = torch.jit.trace(model.eval(), dummy_input)
            
        shape = dummy_input.shape
        if "bert" in model_name.lower():
            cml_model = ct.convert(
                traced, 
                inputs=[ct.TensorType(name="x", shape=shape, dtype=np.int32)],
                minimum_deployment_target=ct.target.macOS13
            )
            inp_dict = {"x": dummy_input.numpy().astype(np.int32)}
        else:
            cml_model = ct.convert(
                traced, 
                inputs=[ct.TensorType(name="x", shape=shape)],
                minimum_deployment_target=ct.target.macOS13
            )
            inp_dict = {"x": dummy_input.numpy()}
            
        # Warmup
        for _ in range(5):
            cml_model.predict(inp_dict)
            
        t0 = time.perf_counter()
        for _ in range(iters):
            cml_model.predict(inp_dict)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0 / iters
    except Exception as e:
        print(f"CoreML failed: {e}")
        return float('inf')

def bench_onnx_runtime(model, dummy_input, model_name, iters=20):
    try:
        import onnxruntime as ort
        import tempfile
        import warnings
        
        # Export model to onnx
        model = model.to('cpu')
        dummy_input = dummy_input.to('cpu')
        
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            onnx_path = tmp.name
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if "bert" in model_name.lower():
                torch.onnx.export(model, dummy_input, onnx_path, 
                                 input_names=["input"], output_names=["output"],
                                 opset_version=14)
            else:
                torch.onnx.export(model, dummy_input, onnx_path, 
                                 input_names=["input"], output_names=["output"],
                                 opset_version=14)
                
        # Load ONNX Runtime session with CoreML
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        sess = ort.InferenceSession(onnx_path, providers=providers)
        
        inp_name = sess.get_inputs()[0].name
        if "bert" in model_name.lower():
            inp_data = dummy_input.numpy().astype(np.int64)
        else:
            inp_data = dummy_input.numpy()
            
        # Warmup
        for _ in range(5):
            sess.run(None, {inp_name: inp_data})
            
        t0 = time.perf_counter()
        for _ in range(iters):
            sess.run(None, {inp_name: inp_data})
        t1 = time.perf_counter()
        
        os.unlink(onnx_path)
        return (t1 - t0) * 1000.0 / iters
    except Exception as e:
        print(f"ONNX Runtime failed: {e}")
        return float('inf')

def bench_fusionml_auto(sched, inputs, iters=20):
    try:
        # Warmup
        sched.run_pipelined_batch(inputs[:2])
        
        t0 = time.perf_counter()
        sched.run_pipelined_batch(inputs)
        t1 = time.perf_counter()
        
        return (t1 - t0) * 1000.0 / len(inputs)
    except Exception as e:
        print(f"FusionML Auto failed: {e}")
        return float('inf')

def bench_fusionml_mlx_only(sched, inputs, iters=20):
    try:
        # Save original and temp map all to GPU
        orig = [e.backend for e in sched.schedule]
        for e in sched.schedule:
            e.backend = "gpu"
            
        sched.run_pipelined_batch(inputs[:2])
        t0 = time.perf_counter()
        sched.run_pipelined_batch(inputs)
        t1 = time.perf_counter()
        
        # Restore
        for i, e in enumerate(sched.schedule):
            e.backend = orig[i]
            
        return (t1 - t0) * 1000.0 / len(inputs)
    except Exception as e:
        print(f"FusionML MLX Only failed: {e}")
        return float('inf')

def bench_fusionml_concurrent(sched, inputs, iters=20):
    try:
        # Warmup
        sched.run_true_pipeline(inputs[:2])
        
        t0 = time.perf_counter()
        sched.run_true_pipeline(inputs)
        t1 = time.perf_counter()
        
        return (t1 - t0) * 1000.0 / len(inputs)
    except Exception as e:
        print(f"FusionML Concurrent failed: {e}")
        return float('inf')

def main():
    info = get_system_info()
    print("System:", info["cpu_brand"])
    
    iters = 20
    results = {"system": info, "benchmarks": {}}
    
    print("\n--- Benchmarking ResNet-50 ---")
    pt_resnet = get_pytorch_resnet50().eval()
    dummy_img = torch.randn(1, 3, 224, 224)
    
    print("Compiling FusionML ResNet pipeline...")
    fm_resnet_sched = build_fusionml_resnet50_pipeline()
    fm_resnet_sched.compile(profile_iters=2)
    fm_resnet_inputs = [np.random.randn(1, 3, 224, 224).astype(np.float32) for _ in range(iters)]
    
    print("Running PyTorch MPS...")
    mps_ms = bench_pytorch_mps(pt_resnet, dummy_img, iters)
    print("Running CoreML...")
    cml_ms = bench_coreml(pt_resnet, dummy_img, "resnet50", iters=iters)
    print("Running Pure MLX...")
    mlx_ms = bench_fusionml_mlx_only(fm_resnet_sched, fm_resnet_inputs, iters)
    print("Running ONNX Runtime...")
    onnx_ms = bench_onnx_runtime(pt_resnet, dummy_img, "resnet50", iters)
    print("Running FusionML Auto (Sequential)...")
    fm_ms = bench_fusionml_auto(fm_resnet_sched, fm_resnet_inputs, iters)
    print("Running FusionML Auto (Concurrent)...")
    fm_conc_ms = bench_fusionml_concurrent(fm_resnet_sched, fm_resnet_inputs, iters)
    
    results["benchmarks"]["resnet50"] = {
        "pytorch_mps_ms": mps_ms,
        "coreml_ms": cml_ms,
        "onnx_ms": onnx_ms,
        "pure_mlx_ms": mlx_ms,
        "fusionml_auto_ms": fm_ms,
        "fusionml_concurrent_ms": fm_conc_ms
    }
    
    print(f"ResNet-50: PyTorch MPS = {mps_ms:.1f}ms, CoreML = {cml_ms:.1f}ms, ONNX = {onnx_ms:.1f}ms, Pure MLX = {mlx_ms:.1f}ms, Fusion Auto(Seq) = {fm_ms:.1f}ms, Fusion Auto(Conc) = {fm_conc_ms:.1f}ms")

    print("\n--- Benchmarking BERT-base ---")
    seq_len = 128
    pt_bert = get_pytorch_bert().eval()
    dummy_ids = torch.randint(0, 30000, (1, seq_len), dtype=torch.long)
    
    print("Compiling FusionML BERT pipeline...")
    fm_bert_sched = build_fusionml_bert_pipeline(seq_len=seq_len)
    fm_bert_sched.compile(profile_iters=2)
    fm_bert_inputs = [np.random.randint(0, 30000, (1, seq_len)).astype(np.int32) for _ in range(iters)]
    
    print("Running PyTorch MPS...")
    mps_bert_ms = bench_pytorch_mps(pt_bert, dummy_ids, iters)
    print("Running CoreML...")
    cml_bert_ms = bench_coreml(pt_bert, dummy_ids, "bert", seq_len=seq_len, iters=iters)
    print("Running Pure MLX...")
    mlx_bert_ms = bench_fusionml_mlx_only(fm_bert_sched, fm_bert_inputs, iters)
    print("Running ONNX Runtime...")
    onnx_bert_ms = bench_onnx_runtime(pt_bert, dummy_ids, "bert", iters)
    print("Running FusionML Auto (Sequential)...")
    fm_bert_ms = bench_fusionml_auto(fm_bert_sched, fm_bert_inputs, iters)
    print("Running FusionML Auto (Concurrent)...")
    fm_bert_conc_ms = bench_fusionml_concurrent(fm_bert_sched, fm_bert_inputs, iters)
    
    results["benchmarks"]["bert_base"] = {
        "pytorch_mps_ms": mps_bert_ms,
        "coreml_ms": cml_bert_ms,
        "onnx_ms": onnx_bert_ms,
        "pure_mlx_ms": mlx_bert_ms,
        "fusionml_auto_ms": fm_bert_ms,
        "fusionml_concurrent_ms": fm_bert_conc_ms
    }
    
    print(f"BERT-base: PyTorch MPS = {mps_bert_ms:.1f}ms, CoreML = {cml_bert_ms:.1f}ms, ONNX = {onnx_bert_ms:.1f}ms, Pure MLX = {mlx_bert_ms:.1f}ms, Fusion Auto(Seq) = {fm_bert_ms:.1f}ms, Fusion Auto(Conc) = {fm_bert_conc_ms:.1f}ms")
    
    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"baseline_results_{info['cpu_brand'].replace(' ', '_')}.json")
    
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved baseline results to {out_file}")

if __name__ == "__main__":
    main()
