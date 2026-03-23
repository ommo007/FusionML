import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Tuple
from .pipeline_scheduler import ScheduleEntry, LayerProfile

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3) # CPU, GPU, ANE
        )

    def forward(self, x):
        return self.net(x)

class NeuralDeviceScheduler:
    """
    Neural Device Scheduler (NDS) - 
    A learned policy regressor that outputs contention-aware device assignments.
    Replaces brute-force ILP and greedy heuristics with a trained Neural Network.
    """
    def __init__(self, model_dir: str = None):
        import pathlib
        if model_dir is None:
            model_dir = os.path.join(pathlib.Path(__file__).parent.parent.parent.parent, "benchmarks", "models")
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, "nds_policy.pt")
        
        self.policy = PolicyNetwork()
        self.is_trained = False
        
        # Features mapping
        # 1. flops
        # 2. memory_bytes
        # 3. is_conv (0/1)
        # 4. is_linear (0/1)
        # 5. current_gpu_load (0-1)
        # 6. current_ane_load (0-1)
        # 7. alpha_contention (environment factor)
        # 8. is_attention (0/1)
        
        if os.path.exists(self.model_path):
            self.load_model(self.model_path)
    
    def extract_features(self, layer_config: Any, current_gpu_load: float, current_ane_load: float, alpha: float) -> np.ndarray:
        flops = 0.0
        memory = 0.0
        is_conv = 0.0
        is_linear = 0.0
        is_attn = 0.0
        
        op = layer_config.op_type.lower()
        if "conv" in op:
            is_conv = 1.0
        elif "linear" in op or "dense" in op:
            is_linear = 1.0
        elif "attn" in op or "mha" in op:
            is_attn = 1.0
            
        shape = layer_config.input_shape
        memory = float(np.prod(shape) * 4) # 4 bytes per fp32
        
        return np.array([
            flops, memory, is_conv, is_linear, current_gpu_load, current_ane_load, alpha, is_attn
        ], dtype=np.float32)

    def train_from_simulated_traces(self, num_samples=1000):
        """
        Trains the policy network using offline profiling traces or simulated optimal ILP resolutions.
        In a real deployment, this would train on thousands of edge device profiling logs.
        For NeurIPS UMHA demonstration, we simulate the ILP/Greedy oracle.
        """
        print("Training Neural Device Scheduler (NDS) on UMHA simulated traces...")
        optimizer = optim.Adam(self.policy.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()
        
        # Generate synthetic UMHA contention data
        # If ANE load is high + contention alpha is high -> GPU is better
        # If OP is Attention -> NPU struggles -> GPU is better
        # If OP is Conv -> NPU excels -> ANE is better
        
        X = []
        Y = []
        for _ in range(num_samples):
            flops = np.random.rand()
            memory = np.random.rand() * 1e6
            is_conv = np.random.choice([0, 1])
            is_linear = np.random.choice([0, 1]) if not is_conv else 0
            is_attn = 1 if (not is_conv and not is_linear) else 0
            gpu_load = np.random.rand()
            ane_load = np.random.rand()
            alpha = np.random.uniform(0.1, 0.5)
            
            x = [flops, memory, is_conv, is_linear, gpu_load, ane_load, alpha, is_attn]
            X.append(x)
            
            # Oracle logic
            if is_attn == 1:
                y = 1 # GPU
            elif is_conv == 1 and ane_load < 0.8:
                y = 2 # ANE
            elif is_linear == 1 and gpu_load < 0.5:
                y = 1 # GPU
            elif ane_load > 0.8 and alpha > 0.3:
                y = 1 # GPU to avoid contention
            else:
                y = 0 # CPU fallback
                
            Y.append(y)
            
        X_t = torch.tensor(X, dtype=torch.float32)
        Y_t = torch.tensor(Y, dtype=torch.long)
        
        for epoch in range(50):
            optimizer.zero_grad()
            out = self.policy(X_t)
            loss = criterion(out, Y_t)
            loss.backward()
            optimizer.step()
            
        print(f"NDS Training Complete. Final Loss: {loss.item():.4f}")
        self.is_trained = True
        torch.save(self.policy.state_dict(), self.model_path)
        
    def load_model(self, path: str):
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()
        self.is_trained = True
        
    def predict_schedule(self, layers: List[Any], profiles: List[LayerProfile], alpha: float = 0.2) -> List[ScheduleEntry]:
        """
        Uses the trained Neural Network to emit a full UMHA Contention-Aware Schedule.
        """
        if not self.is_trained:
            self.train_from_simulated_traces()
            
        self.policy.eval()
        schedule = []
        
        gpu_load = 0.0
        ane_load = 0.0
        
        devices = ["cpu", "gpu", "ane"]
        
        with torch.no_grad():
            for i, layer in enumerate(layers):
                feats = self.extract_features(layer, gpu_load, ane_load, alpha)
                logits = self.policy(torch.tensor(feats).unsqueeze(0))
                device_idx = torch.argmax(logits, dim=1).item()
                best_dev = devices[device_idx]
                
                # Hard fallback if hardware doesn't support the profile entirely
                if best_dev == "ane" and profiles[i].ane_ms == float('inf'):
                    best_dev = "gpu"
                    
                schedule.append(ScheduleEntry(layer_idx=i, backend=best_dev))
                
                # Update loads synthetically
                if best_dev == "gpu":
                    gpu_load = min(1.0, gpu_load + 0.2)
                    ane_load = max(0.0, ane_load - 0.1)
                elif best_dev == "ane":
                    ane_load = min(1.0, ane_load + 0.2)
                    gpu_load = max(0.0, gpu_load - 0.1)
                    
        return schedule
