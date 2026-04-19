from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "ablation_outputs"
FULL_VARIANT_NAME = "full_model"
DEFAULT_SEED = 20260415


@dataclass(frozen=True)
class AblationConfig:
    name: str
    aspect: str
    description: str
    top_raw: int
    top_semantic: int
    selection_mode: str = "correlation"
    fixed_raw: Tuple[str, ...] = ()
    fixed_semantic: Tuple[str, ...] = ()
    use_quantum: bool = True
    enabled_bases: Tuple[str, ...] = ("Z", "X", "Y")
    include_cross_basis: bool = True
    disable_entanglement: bool = False
    combined_from: Tuple[str, ...] = ()


SINGLE_ABLATION_CONFIGS: Tuple[AblationConfig, ...] = (
    AblationConfig(
        name=FULL_VARIANT_NAME,
        aspect="reference",
        description="Full model with correlation-guided hybrid slots and multi-basis quantum readout.",
        top_raw=5,
        top_semantic=3,
    ),
    AblationConfig(
        name="no_corr_selection",
        aspect="selection_strategy",
        description="Remove correlation guidance and keep fixed-order raw/semantic slots.",
        top_raw=5,
        top_semantic=3,
        selection_mode="fixed",
        fixed_raw=("Gender", "Age", "Palpitations", "Sleep_Issues", "Headaches"),
        fixed_semantic=("somatic_load", "affective_load", "confidence_load"),
    ),
    AblationConfig(
        name="semantic_only_slots",
        aspect="slot_source",
        description="Keep only semantic slots to test the contribution of raw feature injection.",
        top_raw=0,
        top_semantic=8,
    ),
    AblationConfig(
        name="single_basis_z",
        aspect="readout_basis",
        description="Keep the reservoir but only measure in the Z basis.",
        top_raw=5,
        top_semantic=3,
        enabled_bases=("Z",),
        include_cross_basis=False,
    ),
    AblationConfig(
        name="no_quantum_readout",
        aspect="quantum_readout",
        description="Use only the 8 hybrid slots and remove the quantum readout entirely.",
        top_raw=5,
        top_semantic=3,
        use_quantum=False,
        enabled_bases=(),
        include_cross_basis=False,
    ),
)

COMPOUND_ABLATION_CONFIGS: Tuple[AblationConfig, ...] = (
    AblationConfig(
        name="raw_only_no_entanglement",
        aspect="compound_combo",
        description="Combine two strong single ablations: raw-only slots with the entanglement layers removed.",
        top_raw=8,
        top_semantic=0,
        disable_entanglement=True,
        combined_from=("raw_only_slots", "no_entanglement"),
    ),
    AblationConfig(
        name="raw_only_no_quantum_readout",
        aspect="compound_combo",
        description="Combine two strong single ablations: raw-only slots and no quantum readout.",
        top_raw=8,
        top_semantic=0,
        use_quantum=False,
        enabled_bases=(),
        include_cross_basis=False,
        combined_from=("raw_only_slots", "no_quantum_readout"),
    ),
    AblationConfig(
        name="raw_only_single_basis_z",
        aspect="compound_combo",
        description="Pair the strong raw-only slot setting with a lighter Z-only quantum readout.",
        top_raw=8,
        top_semantic=0,
        enabled_bases=("Z",),
        include_cross_basis=False,
        combined_from=("raw_only_slots", "single_basis_z"),
    ),
    AblationConfig(
        name="raw_only_single_basis_z_no_entanglement",
        aspect="compound_combo",
        description="Stack three lighter settings together: raw-only slots, Z-only readout and no entanglement.",
        top_raw=8,
        top_semantic=0,
        enabled_bases=("Z",),
        include_cross_basis=False,
        disable_entanglement=True,
        combined_from=("raw_only_slots", "single_basis_z", "no_entanglement"),
    ),
)

ABLATION_CONFIGS: Tuple[AblationConfig, ...] = SINGLE_ABLATION_CONFIGS + COMPOUND_ABLATION_CONFIGS


def _seed_c_runtime(seed: int) -> None:
    seed_u32 = int(seed) & 0xFFFFFFFF
    try:
        if os.name == "nt":
            ctypes.CDLL("msvcrt.dll").srand(seed_u32)
            return
        libc = ctypes.CDLL(None)
        if hasattr(libc, "srand"):
            libc.srand(seed_u32)
    except Exception:
        pass


def set_global_seed(seed: int) -> int:
    seed_u32 = int(seed) & 0xFFFFFFFF
    os.environ["PYTHONHASHSEED"] = str(seed_u32)
    random.seed(seed_u32)
    np.random.seed(seed_u32)
    _seed_c_runtime(seed_u32)
    return seed_u32


def derive_variant_seed(base_seed: int, variant_name: str) -> int:
    digest = hashlib.sha256(variant_name.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "little")
    return int((int(base_seed) + offset) & 0xFFFFFFFF)


def resolve_existing_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate.resolve()

    search_roots = (
        SCRIPT_DIR,
        SCRIPT_DIR.parent,
    )
    for root in search_roots:
        merged = (root / candidate).resolve()
        if merged.exists():
            return merged
    raise FileNotFoundError(f"Could not find file: {path}")


def load_local_train_module() -> ModuleType:
    train_path = SCRIPT_DIR / "train.py"
    spec = importlib.util.spec_from_file_location("corr_semantic_hybrid_qrr_train", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for: {train_path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown_dependency"
        raise SystemExit(
            f"Missing dependency while loading {train_path.name}: {missing}. "
            "Please install the required runtime first, then rerun the ablation script."
        ) from exc
    return module


def apply_semantic_preprocess(
    slots: pd.DataFrame,
    stats: Dict[str, Dict[str, float]],
    semantic_names: Sequence[str],
) -> pd.DataFrame:
    transformed = slots.copy()
    for col in semantic_names:
        col_stats = stats[col]
        transformed[col] = (transformed[col] - float(col_stats["mean"])) / float(col_stats["std"])
    return transformed


def make_reservoir(
    train_mod: ModuleType,
    config: AblationConfig,
    shots_per_basis: int,
    rng_seed: int,
):
    enabled_bases = tuple(dict.fromkeys(config.enabled_bases))
    invalid = set(enabled_bases) - set(train_mod.BASIS_NAMES)
    if invalid:
        raise ValueError(f"Unsupported measurement bases: {sorted(invalid)}")

    include_cross_basis = bool(
        config.include_cross_basis and {"Z", "X", "Y"}.issubset(set(enabled_bases))
    )

    class AblationReservoir(train_mod.CorrelationSemanticReservoir):
        def __init__(self) -> None:
            super().__init__(shots_per_basis=shots_per_basis)
            self.enabled_bases = enabled_bases
            self.include_cross_basis = include_cross_basis
            self._rng = np.random.default_rng(int(rng_seed) & 0xFFFFFFFF)
            if config.disable_entanglement:
                self.GRAPH_LAYER1 = []
                self.GRAPH_LAYER2 = []

        def _run_counts_seeded(self, qvm, base_circuit, basis: str) -> Dict[str, int]:
            # Build a no-measure program to obtain exact basis probabilities,
            # then draw deterministic pseudo-shots from a seeded RNG.
            prog = train_mod.QProg()
            prog << base_circuit
            prog << self._basis_change(basis)
            qvm.run(prog, 1)
            prob_dict = qvm.result().get_prob_dict(list(range(self.n_qubits)))

            keys = [format(i, f"0{self.n_qubits}b") for i in range(1 << self.n_qubits)]
            probs = np.asarray([max(float(prob_dict.get(key, 0.0)), 0.0) for key in keys], dtype=float)
            prob_sum = float(np.sum(probs))
            if prob_sum <= 0.0:
                probs = np.full_like(probs, 1.0 / len(probs), dtype=float)
            else:
                probs /= prob_sum

            sampled = self._rng.multinomial(self.shots_per_basis, probs)
            return {key: int(cnt) for key, cnt in zip(keys, sampled) if int(cnt) > 0}

        def transform_row(self, sample8: np.ndarray, qvm=None) -> np.ndarray:
            owns_qvm = qvm is None
            if qvm is None:
                qvm = self._make_qvm()
            try:
                base_circuit = self.build_base_circuit(sample8)
                basis_stats: Dict[str, np.ndarray] = {}
                all_features: List[float] = []

                for basis in self.enabled_bases:
                    counts = self._run_counts_seeded(qvm, base_circuit, basis)
                    stats = self._counts_to_stats(counts)
                    basis_stats[basis] = stats
                    all_features.extend(stats.tolist())

                features = np.asarray(all_features, dtype=float)
                if self.include_cross_basis:
                    z_stats = basis_stats["Z"]
                    x_stats = basis_stats["X"]
                    y_stats = basis_stats["Y"]
                    cross_basis = np.array(
                        [
                            float(x_stats[1] - z_stats[1]),
                            float(y_stats[1] - z_stats[1]),
                            float(z_stats[4] - 0.5 * (x_stats[4] + y_stats[4])),
                        ],
                        dtype=float,
                    )
                    features = np.concatenate([features, cross_basis], axis=0)
                return features
            finally:
                if owns_qvm:
                    self._finalize_qvm(qvm)

    return AblationReservoir()


def select_slots(
    train_mod: ModuleType,
    x_raw: pd.DataFrame,
    semantic_slots: pd.DataFrame,
    y: np.ndarray,
    config: AblationConfig,
) -> Tuple[List[str], List[str], Dict[str, float], Dict[str, float]]:
    if config.top_raw + config.top_semantic != train_mod.SLOT_DIM:
        raise ValueError(
            f"Variant {config.name} must satisfy top_raw + top_semantic == {train_mod.SLOT_DIM}."
        )

    if config.selection_mode == "correlation":
        return train_mod.select_hybrid_slots(
            x_raw,
            semantic_slots,
            y,
            top_raw=config.top_raw,
            top_semantic=config.top_semantic,
        )

    if config.selection_mode == "fixed":
        raw_scores, semantic_scores = train_mod.rank_feature_correlations(x_raw, semantic_slots, y)
        raw_selected = list(config.fixed_raw or tuple(train_mod.RAW_FEATURES[: config.top_raw]))
        semantic_selected = list(
            config.fixed_semantic
            or tuple(train_mod.SEMANTIC_CANDIDATE_NAMES[: config.top_semantic])
        )
        if len(raw_selected) != config.top_raw or len(semantic_selected) != config.top_semantic:
            raise ValueError(f"Variant {config.name} has inconsistent fixed slot lengths.")
        return raw_selected, semantic_selected, raw_scores, semantic_scores

    raise ValueError(f"Unsupported selection mode: {config.selection_mode}")


def build_fused_features(
    train_mod: ModuleType,
    slot_matrix: np.ndarray,
    config: AblationConfig,
    shots_per_basis: int,
    rng_seed: int,
) -> Tuple[np.ndarray, int]:
    if not config.use_quantum:
        return slot_matrix, 0

    reservoir = make_reservoir(train_mod, config, shots_per_basis, rng_seed=rng_seed)
    quantum_features = reservoir.transform(slot_matrix)
    fused_features = np.hstack([slot_matrix, quantum_features])
    return fused_features, int(quantum_features.shape[1])


def init_head_bias(train_mod: ModuleType, y: np.ndarray) -> float:
    mean_target = float(np.mean(y))
    p0 = min(
        max(
            (mean_target - float(train_mod.OUTPUT_MIN))
            / (float(train_mod.OUTPUT_MAX) - float(train_mod.OUTPUT_MIN)),
            1e-4,
        ),
        1.0 - 1e-4,
    )
    return float(math.log(p0 / (1.0 - p0)))


def eval_metrics(train_mod: ModuleType, y_true: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    clipped = np.clip(preds, float(train_mod.OUTPUT_MIN), float(train_mod.OUTPUT_MAX))
    rounded = np.round(clipped, 1)
    mae_raw = float(np.mean(np.abs(np.asarray(y_true, dtype=float) - clipped)))
    mae_eval = float(np.mean(np.abs(np.asarray(y_true, dtype=float) - rounded)))
    return {
        "eval_mae_raw": mae_raw,
        "eval_mae": mae_eval,
        "eval_score": float(train_mod.score_from_mae(mae_eval)),
    }


def save_variant_details(
    output_dir: Path,
    config: AblationConfig,
    raw_selected: Sequence[str],
    semantic_selected: Sequence[str],
    raw_scores: Dict[str, float],
    semantic_scores: Dict[str, float],
    variant_seed: int,
) -> Path:
    detail_path = output_dir / f"details__{config.name}.json"
    payload = {
        "variant_name": config.name,
        "aspect": config.aspect,
        "description": config.description,
        "combined_from": list(config.combined_from),
        "selection_mode": config.selection_mode,
        "seed": int(variant_seed),
        "selected_raw_features": list(raw_selected),
        "selected_semantic_slots": list(semantic_selected),
        "raw_correlation_scores": {key: float(val) for key, val in raw_scores.items()},
        "semantic_correlation_scores": {key: float(val) for key, val in semantic_scores.items()},
        "variant_config": {
            "top_raw": config.top_raw,
            "top_semantic": config.top_semantic,
            "use_quantum": config.use_quantum,
            "enabled_bases": list(config.enabled_bases),
            "include_cross_basis": config.include_cross_basis,
            "disable_entanglement": config.disable_entanglement,
            "combined_from": list(config.combined_from),
        },
    }
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return detail_path


def run_variant(
    train_mod: ModuleType,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    config: AblationConfig,
    output_dir: Path,
    shots_per_basis: int,
    head_steps: int,
    head_lr: float,
    lambda_w: float,
    delta: float,
    variant_order: int,
    variant_seed: int,
) -> Dict[str, Any]:
    variant_t0 = time.time()
    set_global_seed(variant_seed)
    y_train = train_df[train_mod.TARGET_COL].to_numpy(dtype=float)

    x_raw_train, raw_stats = train_mod.fit_raw_preprocess(train_df)
    semantic_raw_train = train_mod.build_semantic_candidates(x_raw_train)
    semantic_slots_train, semantic_stats = train_mod.fit_semantic_preprocess(semantic_raw_train)

    raw_selected, semantic_selected, raw_scores, semantic_scores = select_slots(
        train_mod,
        x_raw_train,
        semantic_slots_train,
        y_train,
        config,
    )
    slot_df_train, slot_names, slot_roles = train_mod.pack_hybrid_slots(
        x_raw_train,
        semantic_slots_train,
        raw_selected,
        semantic_selected,
    )

    slot_matrix_train = slot_df_train.to_numpy(dtype=float)
    fused_train, quantum_feature_dim = build_fused_features(
        train_mod,
        slot_matrix_train,
        config,
        shots_per_basis,
        rng_seed=variant_seed,
    )

    w = np.zeros(fused_train.shape[1], dtype=float)
    b = init_head_bias(train_mod, y_train)
    w, b = train_mod.train_head(
        fused_train,
        y_train,
        w,
        b,
        head_steps,
        head_lr,
        lambda_w,
        delta,
    )

    train_preds = train_mod.bounded_forward(fused_train, w, b)
    train_mae = float(train_mod.compute_mae(y_train, train_preds))
    train_score = float(train_mod.score_from_mae(train_mae))

    x_raw_eval = train_mod.apply_raw_preprocess(eval_df, raw_stats)
    semantic_raw_eval = train_mod.build_semantic_candidates(x_raw_eval)
    semantic_slots_eval = apply_semantic_preprocess(
        semantic_raw_eval,
        semantic_stats,
        train_mod.SEMANTIC_CANDIDATE_NAMES,
    )
    slot_df_eval, _, _ = train_mod.pack_hybrid_slots(
        x_raw_eval,
        semantic_slots_eval,
        raw_selected,
        semantic_selected,
    )
    slot_matrix_eval = slot_df_eval.to_numpy(dtype=float)
    fused_eval, _ = build_fused_features(
        train_mod,
        slot_matrix_eval,
        config,
        shots_per_basis,
        rng_seed=(variant_seed ^ 0x9E3779B9),
    )

    preds = np.clip(
        train_mod.bounded_forward(fused_eval, w, b),
        float(train_mod.OUTPUT_MIN),
        float(train_mod.OUTPUT_MAX),
    )
    rounded_preds = np.round(preds, 1)

    y_eval = eval_df[train_mod.TARGET_COL].to_numpy(dtype=float)
    metrics = eval_metrics(train_mod, y_eval, preds)

    pred_path = output_dir / f"predictions__{config.name}.csv"
    pred_df = pd.DataFrame(
        {
            "actual": y_eval,
            "predicted": preds,
            "predicted_rounded_1dp": rounded_preds,
            "abs_error": np.abs(y_eval - preds),
            "abs_error_rounded_1dp": np.abs(y_eval - rounded_preds),
        }
    )
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    detail_path = save_variant_details(
        output_dir,
        config,
        raw_selected,
        semantic_selected,
        raw_scores,
        semantic_scores,
        variant_seed,
    )

    return {
        "variant_order": variant_order,
        "variant_name": config.name,
        "aspect": config.aspect,
        "description": config.description,
        "combined_from": "|".join(config.combined_from),
        "selection_mode": config.selection_mode,
        "top_raw": config.top_raw,
        "top_semantic": config.top_semantic,
        "use_quantum": bool(config.use_quantum),
        "enabled_bases": ",".join(config.enabled_bases),
        "include_cross_basis": bool(
            config.use_quantum
            and config.include_cross_basis
            and {"Z", "X", "Y"}.issubset(set(config.enabled_bases))
        ),
        "disable_entanglement": bool(config.disable_entanglement),
        "slot_dim": int(slot_matrix_train.shape[1]),
        "quantum_feature_dim": int(quantum_feature_dim),
        "fused_feature_dim": int(fused_train.shape[1]),
        "selected_raw_features": "|".join(raw_selected),
        "selected_semantic_slots": "|".join(semantic_selected),
        "hybrid_slot_names": "|".join(slot_names),
        "hybrid_slot_roles": "|".join(slot_roles),
        "train_mae": train_mae,
        "train_score": train_score,
        "eval_mae_raw": metrics["eval_mae_raw"],
        "eval_mae": metrics["eval_mae"],
        "eval_score": metrics["eval_score"],
        "prediction_file": pred_path.name,
        "detail_file": detail_path.name,
        "variant_seconds": float(time.time() - variant_t0),
        "shots_per_basis": int(shots_per_basis),
        "head_steps": int(head_steps),
        "head_lr": float(head_lr),
        "lambda_w": float(lambda_w),
        "delta": float(delta),
        "seed": int(variant_seed),
    }


def add_relative_deltas(df: pd.DataFrame) -> pd.DataFrame:
    if FULL_VARIANT_NAME in set(df["variant_name"]):
        reference_row = df.loc[df["variant_name"] == FULL_VARIANT_NAME].iloc[0]
    else:
        reference_row = df.sort_values("eval_score", ascending=False).iloc[0]

    reference_eval_score = float(reference_row["eval_score"])
    reference_eval_mae = float(reference_row["eval_mae"])
    reference_train_score = float(reference_row["train_score"])

    enriched = df.copy()
    enriched["eval_score_delta_vs_full"] = enriched["eval_score"] - reference_eval_score
    enriched["eval_mae_delta_vs_full"] = enriched["eval_mae"] - reference_eval_mae
    enriched["train_score_delta_vs_full"] = enriched["train_score"] - reference_train_score
    return enriched


def parse_requested_variants(raw_value: str) -> Tuple[AblationConfig, ...]:
    if raw_value.strip().lower() == "all":
        return ABLATION_CONFIGS

    wanted = {item.strip() for item in raw_value.split(",") if item.strip()}
    configs = tuple(config for config in ABLATION_CONFIGS if config.name in wanted)
    if not configs:
        available = ", ".join(config.name for config in ABLATION_CONFIGS)
        raise ValueError(f"No variants matched. Available variants: {available}")

    missing = sorted(wanted - {config.name for config in configs})
    if missing:
        raise ValueError(f"Unknown variant names: {', '.join(missing)}")
    return configs


def save_summary_files(df: pd.DataFrame, output_dir: Path) -> Tuple[Path, Path]:
    csv_path = output_dir / "ablation_results.csv"
    json_path = output_dir / "ablation_results.json"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    return csv_path, json_path


def print_summary(df: pd.DataFrame) -> None:
    cols = ["variant_name", "aspect", "eval_mae", "eval_score"]
    summary = df[cols].sort_values("eval_score", ascending=False)
    print("[Info] Ablation ranking by eval_score:")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation study for CSMB_QRR")
    parser.add_argument("--train", type=str, default="train.csv", help="Training csv path")
    parser.add_argument("--eval", type=str, default="eval.csv", help="Evaluation csv path")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for ablation tables, details, predictions and plots",
    )
    parser.add_argument("--shots_per_basis", type=int, default=384, help="Quantum shots for each basis")
    parser.add_argument("--head_steps", type=int, default=3200, help="Optimization steps for the FC head")
    parser.add_argument("--head_lr", type=float, default=0.015, help="Learning rate for the FC head")
    parser.add_argument("--lambda_w", type=float, default=8e-4, help="L2 coefficient for the FC head")
    parser.add_argument("--delta", type=float, default=0.35, help="Pseudo-Huber delta")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Global seed for reproducible ablation runs",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help="Comma-separated variant names or 'all'",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Only save numeric results and skip figure generation",
    )
    args = parser.parse_args()

    selected_configs = parse_requested_variants(args.variants)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_seed = set_global_seed(args.seed)

    train_mod = load_local_train_module()
    train_df = pd.read_csv(resolve_existing_path(args.train))
    eval_df = pd.read_csv(resolve_existing_path(args.eval))
    train_df, removed_age_rows = train_mod.clean_training_table(train_df)

    results: List[Dict[str, Any]] = []
    total_t0 = time.time()
    for index, config in enumerate(selected_configs):
        variant_seed = derive_variant_seed(base_seed, config.name)
        print(f"[Info] Running variant {index + 1}/{len(selected_configs)}: {config.name}")
        print(f"[Info] variant_seed = {variant_seed}")
        result = run_variant(
            train_mod=train_mod,
            train_df=train_df,
            eval_df=eval_df,
            config=config,
            output_dir=output_dir,
            shots_per_basis=args.shots_per_basis,
            head_steps=args.head_steps,
            head_lr=args.head_lr,
            lambda_w=args.lambda_w,
            delta=args.delta,
            variant_order=index,
            variant_seed=variant_seed,
        )
        result["removed_bad_age_rows"] = int(removed_age_rows)
        result["base_seed"] = int(base_seed)
        results.append(result)

    result_df = add_relative_deltas(pd.DataFrame(results))
    csv_path, json_path = save_summary_files(result_df, output_dir)
    print_summary(result_df)

    if not args.skip_plots:
        from visualize_ablation import generate_plots

        generated = generate_plots(csv_path, output_dir=output_dir)
        for fig_path in generated:
            print(f"[Done] plot saved to: {fig_path}")

    print(f"[Done] results csv saved to: {csv_path}")
    print(f"[Done] results json saved to: {json_path}")
    print(f"[Done] total_seconds = {time.time() - total_t0:.2f}")


if __name__ == "__main__":
    main()
