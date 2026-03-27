#!/usr/bin/env python3
"""
Qualcomm AI Hub Benchmark Script
Measures the physical inference latency of ResNet-50 and BERT-base on real
Snapdragon processors. This provides the crucial "Non-Apple" baseline for our NeurIPS 2026 submission.

Prerequisites:
- pip install qai-hub qai-hub-models
- Set QAI_HUB_API_TOKEN in your environment or run `qai-hub configure`
"""

import os
import sys
import json
import time
import torch
import torchvision.models as vision_models
from transformers import BertModel, BertConfig

try:
    import qai_hub as hub
except ImportError:
    print("❌ qai_hub module not found.")
    print("Please install via: pip install qai-hub qai-hub-models")
    sys.exit(1)


def profile_model_on_snapdragon(model, input_specs, dummy_input, model_name, device_name="Samsung Galaxy S24 (Family)", options=""):
    """
    Submits a PyTorch model to Qualcomm AI Hub for compilation and physical device profiling.
    """
    print(f"\n[QAI Hub] Processing {model_name}...")
    
    model.eval()
    print(f"  -> Tracing PyTorch model locally...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
    
    print(f"  -> Submitting Cloud Compile Job for Hexagon NPU on {device_name}...")
    compile_job = hub.submit_compile_job(
        model=traced_model,
        device=hub.Device(device_name),
        input_specs=input_specs,
        options=options,
    )
    
    target_model = compile_job.get_target_model()
    if not target_model:
        raise RuntimeError(f"Compilation failed for {model_name}")
    print("  -> Compilation successful!")

    print(f"  -> Submitting Physical Profile Job (this runs on read hardware!)...")
    profile_job = hub.submit_profile_job(
        model=target_model,
        device=hub.Device(device_name)
    )
    
    profile_data = profile_job.download_profile()
    
    # Typically, the execution summary contains median inference time in microseconds (us)
    # The profile_data returned by download_profile() is a dict in recent qai_hub versions
    execution_summary = profile_data.get("execution_summary", {})
    estimated_inference_time = execution_summary.get("estimated_inference_time", 0)
    estimated_inference_time_ms = float(estimated_inference_time) / 1000.0  # Convert us to ms
    
    print(f"  ✅ {model_name} Profiling Complete: {estimated_inference_time_ms:.2f} ms")
    
    # Compute Throughput (items per second)
    throughput = 1000.0 / estimated_inference_time_ms
    print(f"  🚀 {model_name} Throughput: {throughput:.1f} items/s")
    
    return {
        "latency_ms": estimated_inference_time_ms,
        "throughput_sps": throughput,
        "device": device_name,
        "raw_summary": str(execution_summary)
    }

def main():
    print("================================================================")
    print("   Qualcomm AI Hub Snapdragon Benchmark (Physical Hardware)     ")
    print("================================================================")
    
    # Fast-fail if not authenticated
    try:
        devices = hub.get_devices()
        print(f"✅ Successfully authenticated with Qualcomm AI Hub.")
    except Exception as e:
        print("❌ Authentication failed. You must provide a valid API token.")
        print("Run `qai-hub configure` and enter your token, or set QAI_HUB_API_TOKEN environment variable.")
        sys.exit(1)

    # We will target the Samsung Galaxy S24, which features the Snapdragon 8 Gen 3
    target_device = "Samsung Galaxy S24 (Family)"
    results = {
        "qualcomm_device": target_device,
        "resnet50": {},
        "bert_base": {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    }

    # 1. ResNet-50
    resnet = vision_models.resnet50(pretrained=False) # Random weights are fine for latency profiling
    dummy_img = torch.randn(1, 3, 224, 224)
    try:
        res = profile_model_on_snapdragon(
            model=resnet,
            input_specs=dict(image=(1, 3, 224, 224)),
            dummy_input=dummy_img,
            model_name="ResNet-50",
            device_name=target_device
        )
        results["resnet50"] = res
    except Exception as e:
        print(f"Failed ResNet-50: {e}")

    # 2. BERT-base
    # We use dynamic sequence length handling
    config = BertConfig()
    bert = BertModel(config)
    dummy_input_ids = torch.randint(0, 1000, (1, 128))
    
    try:
        # Note: BERT typically takes multiple inputs (attention mask, token types).
        # We wrap it to accept just input_ids for simplicity in tracing/compilation.
        class BertWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, input_ids):
                # Ensure input_ids are cast strictly to long (int64) for NPU indexing compatibility
                return self.m(input_ids.long())[0]
                
        wrapped_bert = BertWrapper(bert)
        
        res = profile_model_on_snapdragon(
            model=wrapped_bert,
            input_specs=dict(input_ids=(1, 128)),
            dummy_input=dummy_input_ids,
            model_name="BERT-base (128 seq)",
            device_name=target_device,
            options="--truncate_64bit_tensors"
        )
        results["bert_base"] = res
    except Exception as e:
        print(f"Failed BERT-base: {e}")

    # Save output
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "results", "qualcomm_snapdragon_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSnapdragon hardware results saved to {out_path}")

if __name__ == "__main__":
    main()
