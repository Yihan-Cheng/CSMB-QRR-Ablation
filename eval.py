from __future__ import annotations

"""
推理阶段会严格复用训练时保存的：
1. 原始特征预处理；
2. 语义候选槽位构造与标准化；
3. 相关性筛出的原始特征与语义槽位顺序；
4. 固定量子蓄水池与多测量基读出；
5. 最后的有界 FC 头。
"""

import argparse
import ctypes
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pyqpanda3.core import CPUQVM, CNOT, H, QCircuit, QProg, RY, RZ, measure

TARGET_COL = "Recent_Stress"
METHOD_NAME = "CSMB_QRR"
OUTPUT_MIN = 0.0
OUTPUT_MAX = 5.0
N_QUBITS = 4
SLOT_DIM = 8
BASIS_NAMES = ["Z", "X", "Y"]
FEATURES_PER_BASIS = 5
CROSS_BASIS_DIM = 3
READOUT_DIM = len(BASIS_NAMES) * FEATURES_PER_BASIS + CROSS_BASIS_DIM
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SEED = 20260415

RAW_FEATURES = [
    "Gender",
    "Age",
    "Palpitations",
    "Sleep_Issues",
    "Headaches",
    "Irritability",
    "Concentration",
    "Low_Mood",
    "Health_Issues",
    "Loneliness",
    "Peer_Comp",
    "Prof_Issues",
    "Work_Env",
    "Relax_Struggle",
    "Home_Env",
    "Acad_Conf",
    "Subj_Conf",
    "Act_Conflict",
]

SCALE_COLS = [
    "Palpitations",
    "Sleep_Issues",
    "Headaches",
    "Irritability",
    "Concentration",
    "Low_Mood",
    "Health_Issues",
    "Loneliness",
    "Peer_Comp",
    "Prof_Issues",
    "Work_Env",
    "Relax_Struggle",
    "Home_Env",
    "Acad_Conf",
    "Subj_Conf",
    "Act_Conflict",
]

AUX_FEATURES = [
    "somatic_load",
    "affective_load",
    "confidence_load",
    "social_env_load",
    "time_conflict_load",
    "overall_load",
]

SEMANTIC_CANDIDATE_NAMES = AUX_FEATURES + [
    "demographic_context",
    "internal_external_gap",
]


def resolve_existing_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.exists():
        candidate = (SCRIPT_DIR / path).resolve()
    return candidate


def load_table(path: str) -> pd.DataFrame:
    return pd.read_csv(resolve_existing_path(path))


def _seed_c_runtime(seed: int) -> None:
    # Best-effort C runtime seeding for third-party native extensions.
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


def apply_raw_preprocess(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """用训练期统计量对原始特征做同构变换。"""
    x = df[RAW_FEATURES].copy()
    for col in RAW_FEATURES:
        s = stats[col]
        x[col] = (x[col] - float(s["mean"])) / float(s["std"])
    return x


def build_semantic_candidates(x_raw: pd.DataFrame) -> pd.DataFrame:
    """直接从标准化后的原始特征构造语义槽位，和 train.py 保持一致。"""
    slots = pd.DataFrame(index=x_raw.index)
    slots["somatic_load"] = (
        x_raw["Palpitations"]
        + x_raw["Sleep_Issues"]
        + x_raw["Headaches"]
        + x_raw["Health_Issues"]
    ) / 4.0
    slots["affective_load"] = (
        x_raw["Irritability"]
        + x_raw["Low_Mood"]
        + x_raw["Loneliness"]
    ) / 3.0
    slots["confidence_load"] = (
        x_raw["Concentration"]
        + x_raw["Acad_Conf"]
        + x_raw["Subj_Conf"]
    ) / 3.0
    slots["social_env_load"] = (
        x_raw["Peer_Comp"]
        + x_raw["Prof_Issues"]
        + x_raw["Work_Env"]
        + x_raw["Home_Env"]
    ) / 4.0
    slots["time_conflict_load"] = (
        x_raw["Relax_Struggle"] + x_raw["Act_Conflict"]
    ) / 2.0
    slots["overall_load"] = x_raw[SCALE_COLS].mean(axis=1)
    slots["demographic_context"] = 0.75 * x_raw["Age"] + 0.25 * x_raw["Gender"]
    slots["internal_external_gap"] = 0.5 * (
        slots["somatic_load"] + slots["affective_load"]
    ) - 0.5 * (
        slots["social_env_load"] + slots["time_conflict_load"]
    )
    return slots[SEMANTIC_CANDIDATE_NAMES]


def apply_semantic_preprocess(
    slots: pd.DataFrame,
    stats: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """按训练阶段统计量处理语义候选槽位。"""
    x = slots.copy()
    for col in SEMANTIC_CANDIDATE_NAMES:
        s = stats[col]
        x[col] = (x[col] - float(s["mean"])) / float(s["std"])
    return x


def pack_hybrid_slots(
    x_raw: pd.DataFrame,
    x_semantic: pd.DataFrame,
    raw_selected: List[str],
    semantic_selected: List[str],
) -> pd.DataFrame:
    ordered_frames: List[pd.DataFrame] = []
    slot_names: List[str] = []
    slot_roles: List[str] = []
    max_len = max(len(raw_selected), len(semantic_selected))

    for idx in range(max_len):
        if idx < len(raw_selected):
            name = raw_selected[idx]
            ordered_frames.append(x_raw[[name]])
            slot_names.append(name)
            slot_roles.append("raw")
        if idx < len(semantic_selected):
            name = semantic_selected[idx]
            ordered_frames.append(x_semantic[[name]])
            slot_names.append(name)
            slot_roles.append("semantic")

    slot_df = pd.concat(ordered_frames, axis=1)
    slot_df.columns = [f"{role}::{name}" for role, name in zip(slot_roles, slot_names)]
    return slot_df


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float).round(1)))
    )


def score_from_mae(mae: float) -> float:
    return float(30.0 * math.exp(-0.1 * float(mae)))


class CorrelationSemanticReservoir:
    ROUND1_RY = np.array([0.92, 0.88, 0.95, 0.84], dtype=float)
    ROUND1_RZ = np.array([0.68, 0.62, 0.72, 0.58], dtype=float)
    ROUND2_RY = np.array([0.54, 0.50, 0.57, 0.52], dtype=float)
    ROUND2_RZ = np.array([0.40, 0.36, 0.43, 0.38], dtype=float)
    MIX = np.array([0.18, -0.12, 0.16, -0.10], dtype=float)
    FINAL = np.array([0.10, -0.08, 0.09, -0.07], dtype=float)
    GRAPH_LAYER1 = [(0, 1), (1, 2), (0, 3), (3, 2)]
    GRAPH_LAYER2 = [(1, 0), (2, 1), (3, 0), (2, 3)]

    def __init__(
        self,
        shots_per_basis: int = 256,
        n_qubits: int = N_QUBITS,
        rng_seed: int = DEFAULT_SEED,
    ) -> None:
        self.shots_per_basis = int(shots_per_basis)
        self.n_qubits = int(n_qubits)
        self._rng = np.random.default_rng(int(rng_seed) & 0xFFFFFFFF)

    @staticmethod
    def _bit_to_z(bit_char: str) -> float:
        return 1.0 if bit_char == "0" else -1.0

    def _make_qvm(self) -> CPUQVM:
        qvm = CPUQVM()
        if hasattr(qvm, "init_qvm"):
            qvm.init_qvm()
        return qvm

    @staticmethod
    def _finalize_qvm(qvm: CPUQVM) -> None:
        if hasattr(qvm, "finalize"):
            qvm.finalize()

    @staticmethod
    def _hybrid_profile(
        xs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw_view = xs[0::2]
        semantic_view = xs[1::2]
        pair_view = 0.62 * raw_view + 0.38 * semantic_view
        bridge_view = np.array(
            [
                0.50 * semantic_view[0] + 0.50 * raw_view[1],
                0.50 * semantic_view[1] + 0.50 * raw_view[2],
                0.50 * semantic_view[2] + 0.50 * raw_view[3],
                0.50 * semantic_view[3] + 0.50 * raw_view[0],
            ],
            dtype=float,
        )
        cross_view = np.array(
            [
                0.50 * raw_view[0] + 0.50 * semantic_view[1],
                0.50 * raw_view[1] + 0.50 * semantic_view[2],
                0.50 * raw_view[2] + 0.50 * semantic_view[3],
                0.50 * raw_view[3] + 0.50 * semantic_view[0],
            ],
            dtype=float,
        )
        return raw_view, semantic_view, pair_view, bridge_view, cross_view

    def build_base_circuit(self, sample8: np.ndarray) -> QCircuit:
        xs = np.tanh(np.asarray(sample8, dtype=float))
        raw_view, semantic_view, pair_view, bridge_view, cross_view = self._hybrid_profile(xs)
        circuit = QCircuit()

        for q in range(self.n_qubits):
            circuit << H(q)

        for q in range(self.n_qubits):
            raw_signal = float(raw_view[q])
            semantic_signal = float(semantic_view[q])
            pair_signal = float(pair_view[q])
            circuit << RY(
                q,
                float(
                    np.pi
                    * (
                        0.50 * self.ROUND1_RY[q] * raw_signal
                        + 0.24 * semantic_signal
                        + 0.12 * pair_signal
                    )
                ),
            )
            circuit << RZ(
                q,
                float(
                    np.pi
                    * (
                        0.34 * self.ROUND1_RZ[q] * semantic_signal
                        - 0.18 * raw_signal
                        + 0.10 * pair_signal
                    )
                ),
            )

        for control, target in self.GRAPH_LAYER1:
            circuit << CNOT(control, target)

        for q in range(self.n_qubits):
            circuit << RY(q, float(np.pi * self.MIX[q] * bridge_view[q]))

        for q in range(self.n_qubits):
            cross_signal = float(cross_view[q])
            pair_signal = float(pair_view[q])
            circuit << RY(q, float(np.pi * (0.46 * self.ROUND2_RY[q] * cross_signal + 0.14 * pair_signal)))
            circuit << RZ(q, float(np.pi * (0.28 * self.ROUND2_RZ[q] * pair_signal - 0.10 * cross_signal)))

        for control, target in self.GRAPH_LAYER2:
            circuit << CNOT(control, target)

        final_mix = np.array(
            [
                0.55 * semantic_view[0] + 0.45 * bridge_view[3],
                0.55 * raw_view[1] + 0.45 * cross_view[0],
                0.55 * semantic_view[2] + 0.45 * bridge_view[1],
                0.55 * raw_view[3] + 0.45 * cross_view[2],
            ],
            dtype=float,
        )
        for q in range(self.n_qubits):
            circuit << RY(q, float(np.pi * self.FINAL[q] * final_mix[q]))

        return circuit

    def _basis_change(self, basis: str) -> QCircuit:
        circuit = QCircuit()
        if basis == "X":
            for q in range(self.n_qubits):
                circuit << H(q)
        elif basis == "Y":
            for q in range(self.n_qubits):
                circuit << RZ(q, -np.pi / 2.0)
                circuit << H(q)
        return circuit

    def _build_measurement_prog(self, base_circuit: QCircuit, basis: str) -> QProg:
        prog = QProg()
        prog << base_circuit
        prog << self._basis_change(basis)
        for q in range(self.n_qubits):
            prog << measure(q, q)
        return prog

    def _run_counts_seeded(self, qvm: CPUQVM, base_circuit: QCircuit, basis: str) -> Dict[str, int]:
        prog = QProg()
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

    def _counts_to_stats(self, counts: Dict[str, int]) -> np.ndarray:
        total = int(sum(counts.values()))
        single_vals = np.zeros(4, dtype=float)
        pair_vals = np.zeros(6, dtype=float)
        parity = 0.0
        entropy = 0.0

        for bitstring, cnt in counts.items():
            bits = bitstring.replace(" ", "")[-4:]
            z = np.array([self._bit_to_z(ch) for ch in bits], dtype=float)
            single_vals += cnt * z
            pair_vals += cnt * np.array(
                [
                    z[0] * z[1],
                    z[0] * z[2],
                    z[0] * z[3],
                    z[1] * z[2],
                    z[1] * z[3],
                    z[2] * z[3],
                ],
                dtype=float,
            )
            parity += cnt * float(z[0] * z[1] * z[2] * z[3])
            prob = cnt / total
            entropy -= prob * math.log(prob + 1e-12, 2)

        single_vals /= total
        pair_vals /= total
        entropy /= 4.0

        return np.array(
            [
                float(np.mean(single_vals)),
                float(np.mean(np.abs(single_vals))),
                float(np.mean(pair_vals)),
                float(parity / total),
                float(entropy),
            ],
            dtype=float,
        )

    def transform_row(self, sample8: np.ndarray, qvm: CPUQVM | None = None) -> np.ndarray:
        owns_qvm = qvm is None
        if qvm is None:
            qvm = self._make_qvm()
        try:
            base_circuit = self.build_base_circuit(sample8)
            basis_stats: Dict[str, np.ndarray] = {}
            all_features: List[float] = []
            for basis in BASIS_NAMES:
                counts = self._run_counts_seeded(qvm, base_circuit, basis)
                stats = self._counts_to_stats(counts)
                basis_stats[basis] = stats
                all_features.extend(stats.tolist())

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
            return np.concatenate([np.asarray(all_features, dtype=float), cross_basis], axis=0)
        finally:
            if owns_qvm:
                self._finalize_qvm(qvm)

    def transform(self, x_slots: np.ndarray) -> np.ndarray:
        qvm = self._make_qvm()
        try:
            return np.vstack([self.transform_row(row, qvm=qvm) for row in x_slots])
        finally:
            self._finalize_qvm(qvm)


def bounded_forward(
    features: np.ndarray,
    w: np.ndarray,
    b: float,
    out_min: float = OUTPUT_MIN,
    out_max: float = OUTPUT_MAX,
) -> np.ndarray:
    return out_min + (out_max - out_min) * sigmoid(features @ w + float(b))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CSMB_QRR inference")
    parser.add_argument("--model", type=str, default="model.json", help="Model file path")
    parser.add_argument("--eval", type=str, default="eval.csv", help="Evaluation csv path")
    parser.add_argument("--output", type=str, default="eval_predictions.csv", help="Prediction output path")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override; defaults to model seed if available",
    )
    args = parser.parse_args()

    with open(resolve_existing_path(args.model), "r", encoding="utf-8") as f:
        model = json.load(f)

    infer_seed = int(model.get("seed", DEFAULT_SEED) if args.seed is None else args.seed)
    infer_seed = set_global_seed(infer_seed)

    df = load_table(args.eval)

    x_raw = apply_raw_preprocess(df, model["raw_preprocess_stats"])
    semantic_raw = build_semantic_candidates(x_raw)
    semantic_slots = apply_semantic_preprocess(
        semantic_raw,
        model["semantic_preprocess_stats"],
    )
    slot_df = pack_hybrid_slots(
        x_raw,
        semantic_slots,
        model["selected_raw_features"],
        model["selected_semantic_slots"],
    )

    slot_matrix = slot_df.to_numpy(dtype=float)
    reservoir = CorrelationSemanticReservoir(
        shots_per_basis=int(model["shots_per_basis"]),
        rng_seed=infer_seed,
    )
    quantum_features = reservoir.transform(slot_matrix)
    fused_features = np.hstack([slot_matrix, quantum_features])

    w = np.array(model["head_weights"], dtype=float)
    b = float(model["head_bias"])
    preds = bounded_forward(
        fused_features,
        w,
        b,
        out_min=float(model.get("output_min", OUTPUT_MIN)),
        out_max=float(model.get("output_max", OUTPUT_MAX)),
    )
    
    preds = np.clip(preds, OUTPUT_MIN, OUTPUT_MAX)

    out_df = df.copy()
    out_df[TARGET_COL] = preds
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"[Done] method = {model.get('method_name', METHOD_NAME)}")
    print(f"[Done] seed = {infer_seed}")
    print(f"[Done] selected_raw = {model.get('selected_raw_features', [])}")
    print(f"[Done] selected_semantic = {model.get('selected_semantic_slots', [])}")
    print(f"[Done] predictions saved to: {args.output}")

    if TARGET_COL in df.columns and df[TARGET_COL].notna().all():
        mae = compute_mae(df[TARGET_COL].to_numpy(dtype=float), preds)
        score = score_from_mae(mae)
        print(f"[Done] MAE = {mae:.6f}")
        print(f"[Done] Score = {score:.6f}")


if __name__ == "__main__":
    main()
