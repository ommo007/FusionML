#!/usr/bin/env python3
"""
bench_hw.py — shared hardware identification for benchmark scripts.
Slug format: Apple_M1_8GB_8CPU_7GPU_16ANE (matches model_comparison.py).
Kept dependency-free (no fusionml import) so orchestrators can use it cheaply.
"""

import re
import subprocess


def get_system_info():
    cpu = "Unknown"
    try:
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                           capture_output=True, text=True)
        cpu = r.stdout.strip()
    except Exception:
        pass

    mem_gb = "?GB"
    try:
        r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                           capture_output=True, text=True)
        mem_gb = f"{int(r.stdout.strip()) // (1024 ** 3)}GB"
    except Exception:
        pass

    cpu_cores = "?"
    try:
        r = subprocess.run(["sysctl", "-n", "hw.physicalcpu"],
                           capture_output=True, text=True)
        v = r.stdout.strip()
        if v.isdigit():
            cpu_cores = v
    except Exception:
        pass

    gpu_cores = "?"
    try:
        r = subprocess.run(["ioreg", "-r", "-c", "AGXAccelerator"],
                           capture_output=True, text=True)
        m = re.search(r'"gpu-core-count"\s*=\s*(\d+)', r.stdout)
        if m:
            gpu_cores = m.group(1)
    except Exception:
        pass

    _ANE_LOOKUP = [
        ("M1 Ultra", "32"), ("M2 Ultra", "32"), ("M3 Ultra", "36"), ("M4 Ultra", "64"),
        ("M3 Pro",   "18"), ("M3 Max",   "18"),
        ("M4 Pro",   "20"), ("M4 Max",   "32"),
    ]
    ane_cores = "16"  # M1/M2/M3-base/M4-base all have 16-core ANE
    for chip_key, cores in _ANE_LOOKUP:
        if chip_key in cpu:
            ane_cores = cores
            break

    model_id = ""
    try:
        r = subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True)
        model_id = r.stdout.strip()  # e.g. Macmini10,1 / MacBookAir10,1
    except Exception:
        pass
    # MacBook Air is the only fanless Apple Silicon chassis
    passively_cooled = model_id.startswith("MacBookAir") or "Air" in model_id

    slug = (
        f"{cpu}_{mem_gb}_{cpu_cores}CPU_{gpu_cores}GPU_{ane_cores}ANE"
        .replace(" ", "_")
    )

    return {
        "cpu":       cpu,
        "memory":    mem_gb,
        "cpu_cores": cpu_cores,
        "gpu_cores": gpu_cores,
        "ane_cores": ane_cores,
        "model_id":  model_id,
        "passively_cooled": passively_cooled,
        "cpu_slug":  slug,
    }


def get_power_state():
    """Power/thermal context — battery-vs-AC and accumulated heat dominate
    run-to-run variance on passively cooled machines."""
    env = dict(get_system_info())
    try:
        batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
        env["power_source"] = "AC" if "AC Power" in batt else "battery"
        for tok in batt.split():
            if tok.rstrip(";").endswith("%"):
                env["battery_pct"] = tok.rstrip(";")
                break
    except Exception:
        pass
    try:
        therm = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout
        env["thermal_notes"] = therm.strip()
    except Exception:
        pass
    # Free RAM at run start — a benchmark whose working set exceeds this is
    # measuring swap, not compute (see the invalidated M4 Pro full-depth run:
    # 15.6GB model, 1.8GB resident, 5.5x-slow nonsense timings).
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        import re as _re
        page_size = 16384
        m = _re.search(r"page size of (\d+)", vm)
        if m:
            page_size = int(m.group(1))
        pages = {}
        for key in ["Pages free", "Pages inactive", "Pages speculative"]:
            m = _re.search(rf"{key}:\s+(\d+)", vm)
            if m:
                pages[key] = int(m.group(1))
        if pages:
            env["free_ram_gb"] = round(sum(pages.values()) * page_size / (1024 ** 3), 2)
    except Exception:
        pass
    return env
