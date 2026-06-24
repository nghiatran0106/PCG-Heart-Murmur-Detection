from pathlib import Path
import copy
import yaml

BASE_CONFIG = Path("configs/baseline_fusion.yaml")
OUT_DIR = Path("configs/ablation_preprocessing")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE_CONFIG, "r", encoding="utf-8") as f:
    base = yaml.safe_load(f)

experiments = [
    {
        "name": "bp_none_sr4000_seg5",
        "bandpass_low": None,
        "bandpass_high": None,
    },
    {
        "name": "bp_20_400_sr4000_seg5",
        "bandpass_low": 20,
        "bandpass_high": 400,
    },
    {
        "name": "bp_25_400_sr4000_seg5",
        "bandpass_low": 25,
        "bandpass_high": 400,
    },
    {
        "name": "bp_25_800_sr4000_seg5",
        "bandpass_low": 25,
        "bandpass_high": 800,
    },
    {
        "name": "bp_50_800_sr4000_seg5",
        "bandpass_low": 50,
        "bandpass_high": 800,
    },
]

for exp in experiments:
    cfg = copy.deepcopy(base)

    cfg["experiment_name"] = exp["name"]

    if "preprocessing" not in cfg:
        cfg["preprocessing"] = {}

    cfg["preprocessing"]["target_sr"] = 4000
    cfg["preprocessing"]["bandpass_low"] = exp["bandpass_low"]
    cfg["preprocessing"]["bandpass_high"] = exp["bandpass_high"]

    if "segmentation" not in cfg:
        cfg["segmentation"] = {}

    cfg["segmentation"]["window_sec"] = 5.0
    cfg["segmentation"]["mode"] = "fixed"

    if "output_dir" in cfg:
        cfg["output_dir"] = f"outputs/ablation_preprocessing/{exp['name']}"
    elif "project" in cfg:
        cfg.setdefault("project", {})
        cfg["project"]["output_dir"] = f"outputs/ablation_preprocessing/{exp['name']}"

    out_path = OUT_DIR / f"{exp['name']}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print("Saved:", out_path)
