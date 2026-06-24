from pathlib import Path
import copy
import csv
import yaml


BASE_CONFIG = Path("configs/baseline_fusion.yaml")
CONFIG_DIR = Path("configs/experiments")
QUEUE_DIR = Path("outputs/experiment_queue")

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
QUEUE_DIR.mkdir(parents=True, exist_ok=True)


def load_base_config():
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(f"Missing baseline config: {BASE_CONFIG}")

    with open(BASE_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_set(cfg, keys, value):
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or cur[k] is None:
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def set_common(cfg, exp_name, phase, dataset="CirCor2022", task="murmur"):
    cfg["experiment_name"] = exp_name
    cfg["phase"] = phase

    deep_set(cfg, ["data", "dataset_name"], dataset)
    deep_set(cfg, ["task", "name"], task)

    # Default output dir
    cfg["output_dir"] = f"outputs/experiment_results/{phase}/{exp_name}"

    return cfg


def save_config(cfg, phase, exp_name):
    out_dir = CONFIG_DIR / phase
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{exp_name}.yaml"

    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    return out_path


def add_command(rows, phase, exp_name, cfg_path, supported=True, note=""):
    # supported=True means current train_baseline.py is expected to run it
    # supported=False means config is planned and may need train_experiment.py support
    command = (
        f"for FOLD in 0 1 2 3 4; do "
        f"python src/training/train_baseline.py --config {cfg_path} --fold $FOLD; "
        f"done"
    )

    rows.append(
        {
            "phase": phase,
            "experiment_name": exp_name,
            "config": str(cfg_path),
            "supported_by_current_baseline_runner": supported,
            "command": command,
            "note": note,
        }
    )


def make_phase1_preprocessing(base, rows):
    phase = "phase1_preprocessing"

    # 1A. Bandpass ablation
    bandpass_exps = [
        ("bp_none_sr4000_seg5_logmel", None, None),
        ("bp_20_400_sr4000_seg5_logmel", 20, 400),
        ("bp_25_400_sr4000_seg5_logmel", 25, 400),
        ("bp_25_800_sr4000_seg5_logmel", 25, 800),
        ("bp_50_800_sr4000_seg5_logmel", 50, 800),
    ]

    for name, low, high in bandpass_exps:
        cfg = copy.deepcopy(base)
        set_common(cfg, name, phase)
        deep_set(cfg, ["preprocessing", "target_sr"], 4000)
        deep_set(cfg, ["preprocessing", "bandpass_low"], low)
        deep_set(cfg, ["preprocessing", "bandpass_high"], high)
        deep_set(cfg, ["segmentation", "window_sec"], 5.0)
        deep_set(cfg, ["features", "time_frequency"], "logmel")
        deep_set(cfg, ["features", "spectrogram_enhancement"], "none")

        p = save_config(cfg, phase, name)
        add_command(rows, phase, name, p, True, "Bandpass ablation; keep sr=4000, segment=5s, log-Mel.")

    # 1B. Resampling ablation
    for sr in [1000, 2000, 4000]:
        name = f"sr{sr}_bp25_800_seg5_logmel"
        cfg = copy.deepcopy(base)
        set_common(cfg, name, phase)
        deep_set(cfg, ["preprocessing", "target_sr"], sr)
        deep_set(cfg, ["preprocessing", "bandpass_low"], 25)
        deep_set(cfg, ["preprocessing", "bandpass_high"], 800)
        deep_set(cfg, ["segmentation", "window_sec"], 5.0)
        deep_set(cfg, ["features", "time_frequency"], "logmel")
        deep_set(cfg, ["features", "spectrogram_enhancement"], "none")

        p = save_config(cfg, phase, name)
        add_command(rows, phase, name, p, True, "Resampling ablation; keep bandpass=25-800, segment=5s.")

    # 1C. Segmentation ablation
    segment_exps = [
        ("seg5_bp25_800_sr4000_logmel", 5.0, "fixed"),
        ("seg10_bp25_800_sr4000_logmel", 10.0, "fixed"),
        ("seg15_bp25_800_sr4000_logmel", 15.0, "fixed"),
        ("segfull_bp25_800_sr4000_logmel", None, "full"),
    ]

    for name, win, mode in segment_exps:
        cfg = copy.deepcopy(base)
        set_common(cfg, name, phase)
        deep_set(cfg, ["preprocessing", "target_sr"], 4000)
        deep_set(cfg, ["preprocessing", "bandpass_low"], 25)
        deep_set(cfg, ["preprocessing", "bandpass_high"], 800)
        deep_set(cfg, ["segmentation", "mode"], mode)
        deep_set(cfg, ["segmentation", "window_sec"], win)
        deep_set(cfg, ["features", "time_frequency"], "logmel")
        deep_set(cfg, ["features", "spectrogram_enhancement"], "none")

        p = save_config(cfg, phase, name)
        add_command(rows, phase, name, p, True, "Segmentation ablation.")

    # 1D. Spectrogram enhancement ablation
    for enh in ["none", "spectral_gating", "peak_reduction", "pcen", "specaugment"]:
        name = f"enh_{enh}_bp25_800_sr4000_seg5"
        cfg = copy.deepcopy(base)
        set_common(cfg, name, phase)
        deep_set(cfg, ["preprocessing", "target_sr"], 4000)
        deep_set(cfg, ["preprocessing", "bandpass_low"], 25)
        deep_set(cfg, ["preprocessing", "bandpass_high"], 800)
        deep_set(cfg, ["segmentation", "window_sec"], 5.0)
        deep_set(cfg, ["features", "time_frequency"], "logmel")
        deep_set(cfg, ["features", "spectrogram_enhancement"], enh)

        p = save_config(cfg, phase, name)
        add_command(
            rows,
            phase,
            name,
            p,
            False,
            "May require preprocessing code support for spectral gating/PCEN/SpecAugment.",
        )


def make_phase2_features(base, rows):
    phase = "phase2_feature_ablation"

    # Time-frequency representations
    for rep in ["mel", "logmel", "stft", "cwt", "wavelet_scalogram", "pcen_mel"]:
        name = f"tf_{rep}_fusion_baseline"
        cfg = copy.deepcopy(base)
        set_common(cfg, name, phase)
        deep_set(cfg, ["preprocessing", "target_sr"], 4000)
        deep_set(cfg, ["preprocessing", "bandpass_low"], 25)
        deep_set(cfg, ["preprocessing", "bandpass_high"], 800)
        deep_set(cfg, ["segmentation", "window_sec"], 5.0)
        deep_set(cfg, ["features", "time_frequency"], rep)
        deep_set(cfg, ["features", "handcrafted_group"], "all")

        p = save_config(cfg, phase, name)
        add_command(rows, phase, name, p, rep == "logmel", "Time-frequency representation ablation.")

    # Handcrafted feature groups
    for group in ["mfcc", "spectral", "zcr", "shannon_energy", "envelope", "wavelet_stats", "all"]:
        name = f"hand_{group}_fusion_baseline"
        cfg = copy.deepcopy(base)
        set_common(cfg, name, phase)
        deep_set(cfg, ["features", "time_frequency"], "logmel")
        deep_set(cfg, ["features", "handcrafted_group"], group)

        p = save_config(cfg, phase, name)
        add_command(
            rows,
            phase,
            name,
            p,
            group == "all",
            "Handcrafted feature group ablation; may require feature extractor group selection support.",
        )

    # Metadata ablation
    for setting in ["audio_only", "metadata_only", "audio_metadata"]:
        name = f"metadata_{setting}"
        cfg = copy.deepcopy(base)
        set_common(cfg, name, phase)
        deep_set(cfg, ["features", "metadata_mode"], setting)

        p = save_config(cfg, phase, name)
        add_command(rows, phase, name, p, False, "Metadata branch requires implementation.")


def make_phase3_fusion(base, rows):
    phase = "phase3_fusion_ablation"

    fusion_settings = [
        ("spectrogram_only", "spectrogram_only"),
        ("handcrafted_only", "handcrafted_only"),
        ("late_concat_fusion", "late_concat"),
        ("early_fusion", "early_fusion"),
        ("gated_late_fusion", "gated_late"),
        ("attention_late_fusion", "attention_late"),
    ]

    for name, fusion in fusion_settings:
        cfg = copy.deepcopy(base)
        set_common(cfg, name, phase)
        deep_set(cfg, ["model", "fusion_strategy"], fusion)
        deep_set(cfg, ["features", "time_frequency"], "logmel")
        deep_set(cfg, ["features", "handcrafted_group"], "all")

        p = save_config(cfg, phase, name)
        add_command(
            rows,
            phase,
            name,
            p,
            fusion == "late_concat",
            "Fusion ablation. Current baseline supports late concat; others require model implementation.",
        )


def make_phase4_model_blocks(base, rows):
    phase = "phase4_model_block_ablation"

    # Spectrogram backbone ablation
    for backbone in ["cnn_small", "resnet18", "resnet34", "efficientnet_b0", "mobilenetv3_small"]:
        name = f"backbone_{backbone}_fusion"
        cfg = copy.deepcopy(base)
        set_common(cfg, name, phase)
        deep_set(cfg, ["model", "spectrogram_backbone"], backbone)
        deep_set(cfg, ["model", "fusion_strategy"], "late_concat")

        p = save_config(cfg, phase, name)
        add_command(
            rows,
            phase,
            name,
            p,
            backbone == "resnet18",
            "Spectrogram backbone ablation. Only ResNet18 is current baseline unless other backbones are implemented.",
        )

    # Handcrafted branch classifier/model
    for hand_model in ["mlp", "logistic_regression", "random_forest", "xgboost", "svm_rbf", "tabnet"]:
        name = f"hand_model_{hand_model}"
        cfg = copy.deepcopy(base)
        set_common(cfg, name, phase)
        deep_set(cfg, ["model", "handcrafted_model"], hand_model)

        p = save_config(cfg, phase, name)
        add_command(
            rows,
            phase,
            name,
            p,
            hand_model == "mlp",
            "Handcrafted branch model ablation.",
        )

    # Classifier head
    for head in ["linear", "mlp", "gated_mlp", "attention_head", "lstm_segment_aggregator", "transformer_segment_aggregator"]:
        name = f"classifier_head_{head}"
        cfg = copy.deepcopy(base)
        set_common(cfg, name, phase)
        deep_set(cfg, ["model", "classifier_head"], head)

        p = save_config(cfg, phase, name)
        add_command(
            rows,
            phase,
            name,
            p,
            head == "mlp",
            "Classifier head ablation.",
        )


def make_phase5_xai(base, rows):
    phase = "phase5_xai"

    xai_jobs = [
        ("gradcam_spectrogram", "python experiments/xai_baseline/02_gradcam_spectrogram_branch.py --fold $FOLD --max-per-class 4"),
        ("occlusion_sensitivity", "python experiments/xai_baseline/02_occlusion_sensitivity.py --fold $FOLD --max-per-class 3"),
        ("tsne_pca_resnet_embedding", "python experiments/xai_baseline/03_tsne_pca_resnet_embedding.py --fold $FOLD --max-per-class 150"),
        ("handcrafted_weight_importance", "python experiments/xai_baseline/05_handcrafted_weight_importance.py"),
    ]

    for name, cmd in xai_jobs:
        cfg = copy.deepcopy(base)
        set_common(cfg, name, phase)
        deep_set(cfg, ["xai", "method"], name)

        p = save_config(cfg, phase, name)

        if name == "handcrafted_weight_importance":
            command = cmd
        else:
            command = f"for FOLD in 0 1 2 3 4; do {cmd}; done"

        rows.append(
            {
                "phase": phase,
                "experiment_name": name,
                "config": str(p),
                "supported_by_current_baseline_runner": True,
                "command": command,
                "note": "XAI job; uses trained checkpoints and saved predictions/embeddings.",
            }
        )


def make_phase6_datasets_multitask(base, rows):
    phase = "phase6_multidataset_multitask"

    dataset_settings = [
        ("circor2022_murmur", "CirCor2022", "murmur"),
        ("circor2022_outcome", "CirCor2022", "outcome"),
        ("circor2022_multitask", "CirCor2022", "multitask"),
        ("physionet2016_binary", "PhysioNet2016", "murmur"),
        ("cvd_binary", "CVD", "murmur"),
        ("all_datasets_murmur", "all", "murmur"),
        ("all_datasets_multitask", "all", "multitask"),
    ]

    for name, dataset, task in dataset_settings:
        cfg = copy.deepcopy(base)
        set_common(cfg, name, phase, dataset=dataset, task=task)
        deep_set(cfg, ["data", "metadata_csv"], "data/metadata/unified_metadata_all_datasets.csv")
        deep_set(cfg, ["model", "fusion_strategy"], "late_concat")
        deep_set(cfg, ["training", "task_mode"], task)

        p = save_config(cfg, phase, name)
        add_command(
            rows,
            phase,
            name,
            p,
            dataset == "CirCor2022" and task == "murmur",
            "Dataset/task ablation. Non-CirCor or multitask may require dataset adapter/training support.",
        )


def make_phase7_optimization(base, rows):
    phase = "phase7_optimization"

    schedulers = ["cosine", "onecycle", "plateau"]
    losses = ["cross_entropy", "weighted_cross_entropy", "focal_loss", "label_smoothing"]
    optimizers = ["adam", "adamw"]

    for opt in optimizers:
        for sch in schedulers:
            for loss in losses:
                name = f"opt_{opt}_sch_{sch}_loss_{loss}"
                cfg = copy.deepcopy(base)
                set_common(cfg, name, phase)
                deep_set(cfg, ["training", "optimizer"], opt)
                deep_set(cfg, ["training", "scheduler"], sch)
                deep_set(cfg, ["training", "loss"], loss)
                deep_set(cfg, ["training", "lr"], 1e-4)
                deep_set(cfg, ["training", "weight_decay"], 1e-4)

                p = save_config(cfg, phase, name)
                add_command(
                    rows,
                    phase,
                    name,
                    p,
                    False,
                    "Optimization ablation; requires training script support for scheduler/loss choices.",
                )

    # Optuna config placeholder
    name = "optuna_search_fusion_baseline"
    cfg = copy.deepcopy(base)
    set_common(cfg, name, phase)
    deep_set(cfg, ["optuna", "enabled"], True)
    deep_set(cfg, ["optuna", "n_trials"], 30)
    deep_set(cfg, ["optuna", "search_space", "lr"], [1e-5, 3e-3])
    deep_set(cfg, ["optuna", "search_space", "weight_decay"], [1e-6, 1e-2])
    deep_set(cfg, ["optuna", "search_space", "dropout"], [0.1, 0.5])

    p = save_config(cfg, phase, name)
    rows.append(
        {
            "phase": phase,
            "experiment_name": name,
            "config": str(p),
            "supported_by_current_baseline_runner": False,
            "command": f"python experiments/optuna/run_optuna_search.py --config {p}",
            "note": "Optuna hyperparameter search placeholder.",
        }
    )


def write_registry(rows):
    registry_path = QUEUE_DIR / "experiment_registry.csv"

    with open(registry_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "experiment_name",
                "config",
                "supported_by_current_baseline_runner",
                "command",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved registry: {registry_path}")

    supported_sh = QUEUE_DIR / "run_supported_baseline_jobs_later.sh"
    with open(supported_sh, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -e\n\n")
        f.write("cd ~/Nghia/PCG-Heart-Murmur-Detection\n")
        f.write("source .venv_pcg/bin/activate\n")
        f.write("export PYTHONPATH=$(pwd)\n\n")
        f.write("mkdir -p outputs/experiment_logs\n\n")

        for r in rows:
            if str(r["supported_by_current_baseline_runner"]) == "True":
                exp = r["experiment_name"]
                phase = r["phase"]
                cmd = r["command"]
                log = f"outputs/experiment_logs/{phase}_{exp}.log"
                f.write(f'echo "===== {phase} | {exp} ====="\n')
                f.write(f"({cmd}) 2>&1 | tee {log}\n\n")

    supported_sh.chmod(0o755)
    print(f"Saved runnable supported queue: {supported_sh}")

    planned_sh = QUEUE_DIR / "planned_jobs_require_implementation.sh"
    with open(planned_sh, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("# These jobs are planned but may require code support before running.\n")
        f.write("# Do not execute blindly.\n\n")

        for r in rows:
            if str(r["supported_by_current_baseline_runner"]) != "True":
                f.write(f"# ===== {r['phase']} | {r['experiment_name']} =====\n")
                f.write(f"# note: {r['note']}\n")
                f.write(f"# {r['command']}\n\n")

    planned_sh.chmod(0o755)
    print(f"Saved planned queue: {planned_sh}")


def main():
    base = load_base_config()
    rows = []

    make_phase1_preprocessing(base, rows)
    make_phase2_features(base, rows)
    make_phase3_fusion(base, rows)
    make_phase4_model_blocks(base, rows)
    make_phase5_xai(base, rows)
    make_phase6_datasets_multitask(base, rows)
    make_phase7_optimization(base, rows)

    write_registry(rows)

    print("\n===== Summary by phase =====")
    counts = {}
    for r in rows:
        counts[r["phase"]] = counts.get(r["phase"], 0) + 1

    for k, v in counts.items():
        print(f"{k}: {v} experiments")

    print("\nNo training was executed.")


if __name__ == "__main__":
    main()
