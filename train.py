from __future__ import annotations

"""
1. 用相关性挑出更强的原始特征，尽量保住分数；
2. 继续保留语义辅助槽位，避免方案退化成“只做筛特征”；
3. 保留多测量基读出与固定量子蓄水池，让整体仍然有创新点；
4. 只训练最后一个有界 FC 头，维持低复杂度和稳定性。
"""

import argparse
import ctypes
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pyqpanda3.core import CPUQVM, CNOT, H, QCircuit, QProg, RY, RZ, measure

# 监督学习目标列：表示最近压力评分。
TARGET_COL = "Recent_Stress"
# 方法标识：会写入模型文件，便于区分实验版本。
METHOD_NAME = "CSMB_QRR"
# 有界回归输出区间，最终预测会被限制在[0,5]。
OUTPUT_MIN = 0.0
OUTPUT_MAX = 5.0
# 数据集中明显异常的年龄值，训练前直接移除。
BAD_AGE_VALUE = 28
# 量子蓄水池基础规模设定。
N_QUBITS = 4
SLOT_DIM = 8
# 混合槽位默认拆分：5个原始特征+3个语义特征。
DEFAULT_TOP_RAW = 5
DEFAULT_TOP_SEMANTIC = SLOT_DIM - DEFAULT_TOP_RAW
# 三种测量基：用于构造多视角读出。
BASIS_NAMES = ["Z", "X", "Y"]
# 每个测量基压缩为5个统计量，再拼接跨基特征。
FEATURES_PER_BASIS = 5
CROSS_BASIS_DIM = 3
READOUT_DIM = len(BASIS_NAMES) * FEATURES_PER_BASIS + CROSS_BASIS_DIM
# 脚本目录：用于在不同工作目录下定位数据文件。
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SEED = 20260415

# 原始输入特征
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

# 用于构建overall_load的量表项。
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

# 辅助聚合特征：由原始量表线性组合而来。
AUX_FEATURES = [
    "somatic_load",
    "affective_load",
    "confidence_load",
    "social_env_load",
    "time_conflict_load",
    "overall_load",
]

# 候选语义槽位：前 6 个沿用 AUX 语义，后 2 个是额外上下文组合项。
SEMANTIC_CANDIDATE_NAMES = AUX_FEATURES + [
    "demographic_context",
    "internal_external_gap",
]


def load_table(path: str) -> pd.DataFrame:
    """读取 CSV，优先使用传入路径，否则回退到脚本目录下同名文件。"""
    candidate = Path(path)
    if not candidate.exists():
        candidate = (SCRIPT_DIR / path).resolve()
    return pd.read_csv(candidate)


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


def clean_training_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """训练期移除不符合的年龄"""
    cleaned = df.loc[df["Age"] < BAD_AGE_VALUE].copy()
    return cleaned, int(len(df) - len(cleaned))


def fit_raw_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    对原始特征执行简单标准化：
    1) 训练前删除 Age>28 的明显异常记录；
    2) 对每列做 z-score 标准化；
    3) 返回处理后数据及可复用的统计参数。
    """
    x = df[RAW_FEATURES].copy()
    stats: Dict[str, Dict[str, float]] = {}
    for col in RAW_FEATURES:
        # 保存标准化参数，供评估阶段复用同一尺度。
        mean = float(x[col].mean())
        std = float(x[col].std(ddof=0) + 1e-8)
        x[col] = (x[col] - mean) / std
        stats[col] = {
            "mean": mean,
            "std": std,
        }
    return x, stats


def apply_raw_preprocess(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """用训练期统计量对原始特征做同构变换，保证train/eval对齐。"""
    x = df[RAW_FEATURES].copy()
    for col in RAW_FEATURES:
        s = stats[col]
        x[col] = (x[col] - float(s["mean"])) / float(s["std"])
    return x


def build_semantic_candidates(x_raw: pd.DataFrame) -> pd.DataFrame:
    """
    构造8维语义候选槽位：
    前6维沿用原先辅助语义的心理学拆分方式；
    demographic_context强调年龄与性别背景；
    internal_external_gap刻画“内部负担vs外部环境”差值。
    """
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


def fit_semantic_preprocess(
    slots: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """拟合并应用语义候选槽位的标准化参数。"""
    x = slots.copy()
    stats: Dict[str, Dict[str, float]] = {}
    for col in SEMANTIC_CANDIDATE_NAMES:
        mean = float(x[col].mean())
        std = float(x[col].std(ddof=0) + 1e-8)
        x[col] = (x[col] - mean) / std
        stats[col] = {"mean": mean, "std": std}
    return x, stats


def safe_abs_corr(x: np.ndarray | pd.Series, y: np.ndarray) -> float:
    """
    计算绝对皮尔逊相关系数。
    """
    corr = float(np.corrcoef(np.asarray(x, dtype=float), np.asarray(y, dtype=float))[0, 1])
    if np.isnan(corr):
        return 0.0
    return abs(corr)


def rank_feature_correlations(
    x_raw: pd.DataFrame,
    x_semantic: pd.DataFrame,
    y: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """分别计算原始特征与语义候选槽位对目标的绝对相关性评分。"""
    raw_scores = {col: safe_abs_corr(x_raw[col], y) for col in RAW_FEATURES}
    semantic_scores = {
        col: safe_abs_corr(x_semantic[col], y) for col in SEMANTIC_CANDIDATE_NAMES
    }
    return raw_scores, semantic_scores


def select_hybrid_slots(
    x_raw: pd.DataFrame,
    x_semantic: pd.DataFrame,
    y: np.ndarray,
    top_raw: int,
    top_semantic: int,
) -> Tuple[List[str], List[str], Dict[str, float], Dict[str, float]]:
    """
    按相关性选择混合槽位：
    从原始特征里选top_raw；
    从语义候选里选top_semantic；
    两者之和必须等于SLOT_DIM。
    """
    if top_raw + top_semantic != SLOT_DIM:
        raise ValueError(f"top_raw + top_semantic must equal {SLOT_DIM}.")

    raw_scores, semantic_scores = rank_feature_correlations(x_raw, x_semantic, y)
    # 相关性越大，越优先保留。
    raw_selected = [
        key
        for key, _ in sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)[:top_raw]
    ]
    semantic_selected = [
        key
        for key, _ in sorted(semantic_scores.items(), key=lambda item: item[1], reverse=True)[:top_semantic]
    ]
    return raw_selected, semantic_selected, raw_scores, semantic_scores


def pack_hybrid_slots(
    x_raw: pd.DataFrame,
    x_semantic: pd.DataFrame,
    raw_selected: List[str],
    semantic_selected: List[str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    将选出的 raw/semantic 槽位按“交替穿插”方式打包成 8 维输入：
    raw0, semantic0, raw1, semantic1, ...
    这样可在后续电路中更自然地构建跨类型耦合。
    """
    ordered_frames: List[pd.DataFrame] = []
    slot_names: List[str] = []
    slot_roles: List[str] = []
    max_len = max(len(raw_selected), len(semantic_selected))

    for idx in range(max_len):
        # 先放 raw 再放 semantic，保证槽位顺序可解释、可复现。
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
    # 列名编码角色信息，便于调试和结果解释。
    slot_df.columns = [f"{role}::{name}" for role, name in zip(slot_roles, slot_names)]
    return slot_df, slot_names, slot_roles


def sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定版本 sigmoid，先裁剪输入避免 exp 上溢。"""
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算平均绝对误差（MAE）。"""
    return float(
        np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)))
    )


def score_from_mae(mae: float) -> float:
    """将 MAE 映射为分数，误差越小分数越高。"""
    return float(30.0 * math.exp(-0.1 * float(mae)))


class CorrelationSemanticReservoir:
    """
    相关性引导的语义混合量子蓄水池。
    设计原则：
    - 参数全部固定（不训练量子参数），提升稳定性与复现性；
    - 输入是 8 维 hybrid slots（raw/semantic 交错）；
    - 通过多层单比特旋转 + 两层纠缠图 + 多测量基读出，输出可供线性头学习的统计特征。
    """
    # 第 1 轮编码系数：强调 raw 与 semantic 的基础注入。
    ROUND1_RY = np.array([0.92, 0.88, 0.95, 0.84], dtype=float)
    ROUND1_RZ = np.array([0.68, 0.62, 0.72, 0.58], dtype=float)

    # 第 2 轮编码系数：使用跨槽位混合后的信号做补充编码。
    ROUND2_RY = np.array([0.54, 0.50, 0.57, 0.52], dtype=float)
    ROUND2_RZ = np.array([0.40, 0.36, 0.43, 0.38], dtype=float)

    # 中间融合层与末端轻量修正层。
    MIX = np.array([0.18, -0.12, 0.16, -0.10], dtype=float)
    FINAL = np.array([0.10, -0.08, 0.09, -0.07], dtype=float)

    # 两层有向纠缠拓扑，形成不同的相关传播路径。
    GRAPH_LAYER1 = [(0, 1), (1, 2), (0, 3), (3, 2)]
    GRAPH_LAYER2 = [(1, 0), (2, 1), (3, 0), (2, 3)]

    def __init__(
        self,
        shots_per_basis: int = 256,
        n_qubits: int = N_QUBITS,
        rng_seed: int = DEFAULT_SEED,
    ) -> None:
        """初始化量子蓄水池运行参数。"""
        self.shots_per_basis = int(shots_per_basis)
        self.n_qubits = int(n_qubits)
        self._rng = np.random.default_rng(int(rng_seed) & 0xFFFFFFFF)

    @staticmethod
    def _bit_to_z(bit_char: str) -> float:
        """将测量比特映射到泡利 Z 本征值：0 -> +1, 1 -> -1。"""
        return 1.0 if bit_char == "0" else -1.0

    def _make_qvm(self) -> CPUQVM:
        """创建并初始化 QVM 实例。"""
        qvm = CPUQVM()
        if hasattr(qvm, "init_qvm"):
            qvm.init_qvm()
        return qvm

    @staticmethod
    def _finalize_qvm(qvm: CPUQVM) -> None:
        """安全释放 QVM 资源。"""
        if hasattr(qvm, "finalize"):
            qvm.finalize()

    @staticmethod
    def _hybrid_profile(
        xs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        将8维交错槽位拆解为多种视角：
        raw_view:4维原始子视角；
        semantic_view:4维语义子视角；
        pair_view:同位融合（raw_i 与 semantic_i）；
        bridge_view:邻接桥接融合；
        cross_view:交叉错位融合。
        """
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
        """
        构造单样本基础电路（不含测量基变换与测量）。
        电路结构：
        1) 全局 Hadamard 初始化；
        2) Round-1 按 raw/semantic/pair 编码；
        3) 第一层纠缠；
        4) 桥接混合旋转；
        5) Round-2 按 cross/pair 继续编码；
        6) 第二层纠缠；
        7) FINAL轻量收尾旋转。
        """
        # 先压缩输入幅度，避免旋转角过大导致训练不稳定。
        xs = np.tanh(np.asarray(sample8, dtype=float))

        raw_view, semantic_view, pair_view, bridge_view, cross_view = self._hybrid_profile(xs)
        circuit = QCircuit()

        # 初态扩展到均匀叠加，增加后续可表达的干涉模式。
        for q in range(self.n_qubits):
            circuit << H(q)

        # Round-1：注入“原始 + 语义 + 同位融合”信息。
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

        # 第一层图结构纠缠：传播局部依赖关系。
        for control, target in self.GRAPH_LAYER1:
            circuit << CNOT(control, target)

        # 中间桥接混合：连接跨位邻接信息。
        for q in range(self.n_qubits):
            circuit << RY(q, float(np.pi * self.MIX[q] * bridge_view[q]))

        # Round-2：注入交叉视角信息，提高多视角区分能力。
        for q in range(self.n_qubits):
            cross_signal = float(cross_view[q])
            pair_signal = float(pair_view[q])
            circuit << RY(q, float(np.pi * (0.46 * self.ROUND2_RY[q] * cross_signal + 0.14 * pair_signal)))
            circuit << RZ(q, float(np.pi * (0.28 * self.ROUND2_RZ[q] * pair_signal - 0.10 * cross_signal)))

        # 第二层纠缠：补充另一方向的信息交换路径。
        for control, target in self.GRAPH_LAYER2:
            circuit << CNOT(control, target)

        # 收尾混合：以较小幅度做最终特征重整。
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
        """根据目标测量基添加末端基变换门。"""
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
        """将基础电路、基变换和测量操作拼成完整程序。"""
        prog = QProg()
        prog << base_circuit
        prog << self._basis_change(basis)
        for q in range(self.n_qubits):
            prog << measure(q, q)
        return prog

    def _run_counts_seeded(self, qvm: CPUQVM, base_circuit: QCircuit, basis: str) -> Dict[str, int]:
        # Execute a no-measure circuit to get exact basis probabilities,
        # then generate deterministic pseudo-shots from a seeded RNG.
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
        """
        将测量计数压缩为 5 维统计特征：
        1) 单体 Z 均值；
        2) 单体 |Z| 均值；
        3) 两体相关均值；
        4) 四体奇偶项；
        5) 归一化熵。
        """
        total = int(sum(counts.values()))

        # 4 个单体、6 个两体组合，外加 parity/entropy。
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

            # 四体项衡量全局协同符号。
            parity += cnt * float(z[0] * z[1] * z[2] * z[3])

            # 熵基于测量分布，反映状态离散度。
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
        """
        单样本量子特征提取。

        输出维度：
        3个测量基×每基5维统计=15；
        跨基补充统计3维；
        总计READOUT_DIM=18。
        """
        owns_qvm = qvm is None
        if qvm is None:
            qvm = self._make_qvm()
        try:
            base_circuit = self.build_base_circuit(sample8)
            basis_stats: Dict[str, np.ndarray] = {}
            all_features: List[float] = []

            # 分别在 Z/X/Y 测量基提取统计量。
            for basis in BASIS_NAMES:
                counts = self._run_counts_seeded(qvm, base_circuit, basis)
                stats = self._counts_to_stats(counts)
                basis_stats[basis] = stats
                all_features.extend(stats.tolist())

            z_stats = basis_stats["Z"]
            x_stats = basis_stats["X"]
            y_stats = basis_stats["Y"]
            # 跨基特征：反映不同观测视角下统计差异。
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
        """批量提取量子读出特征，共享单个 QVM 以减少开销。"""
        qvm = self._make_qvm()
        try:
            return np.vstack([self.transform_row(row, qvm=qvm) for row in x_slots])
        finally:
            self._finalize_qvm(qvm)


def bounded_forward(features: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """线性层+sigmoid 映射到 [OUTPUT_MIN, OUTPUT_MAX]。"""
    logits = features @ w + float(b)
    return OUTPUT_MIN + (OUTPUT_MAX - OUTPUT_MIN) * sigmoid(logits)


def pseudo_huber_grad(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 0.35) -> np.ndarray:
    """
    Pseudo-Huber损失对预测值的梯度。
    相比L2，对离群点更鲁棒；相比L1，在0附近可导更平滑。
    """
    residual = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return residual / np.sqrt(1.0 + (residual / delta) ** 2)


def train_head(
    features: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    steps: int,
    lr: float,
    lambda_w: float,
    delta: float,
) -> Tuple[np.ndarray, float]:
    """
    仅训练最后一层FC头（权重w和偏置b）。
    """
    w = np.asarray(w, dtype=float).copy()
    b = float(b)

    # Adam 一阶/二阶动量缓冲。
    m_w = np.zeros_like(w)
    v_w = np.zeros_like(w)
    m_b = 0.0
    v_b = 0.0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    scale = OUTPUT_MAX - OUTPUT_MIN

    for t in range(1, steps + 1):
        # 前向：bounded regression。
        logits = features @ w + b
        sig = sigmoid(logits)
        y_pred = OUTPUT_MIN + scale * sig

        # 链式法则：dL/dy -> dL/dz。
        dloss_dy = pseudo_huber_grad(y, y_pred, delta=delta) / len(y)
        dloss_dz = dloss_dy * scale * sig * (1.0 - sig)
        grad_w = features.T @ dloss_dz + 2.0 * lambda_w * w
        grad_b = float(np.sum(dloss_dz))

        # Adam 动量更新。
        m_w = beta1 * m_w + (1.0 - beta1) * grad_w
        v_w = beta2 * v_w + (1.0 - beta2) * (grad_w * grad_w)
        m_b = beta1 * m_b + (1.0 - beta1) * grad_b
        v_b = beta2 * v_b + (1.0 - beta2) * (grad_b * grad_b)

        # 偏差修正。
        m_w_hat = m_w / (1.0 - beta1**t)
        v_w_hat = v_w / (1.0 - beta2**t)
        m_b_hat = m_b / (1.0 - beta1**t)
        v_b_hat = v_b / (1.0 - beta2**t)

        # 参数更新。
        w -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
        b -= lr * m_b_hat / (math.sqrt(v_b_hat) + eps)

    return w, b


def summarize_complexity(
    fused_dim: int,
    top_raw: int,
    top_semantic: int,
) -> Dict[str, float | int | str]:
    """汇总模型复杂度与结构配置，便于实验记录。"""
    return {
        "method_name": METHOD_NAME,
        "n_qubits": N_QUBITS,
        "quantum_params": 0,
        "slot_dim": SLOT_DIM,
        "selected_raw_dim": top_raw,
        "selected_semantic_dim": top_semantic,
        "readout_dim": READOUT_DIM,
        "fused_feature_dim": fused_dim,
        "fc_params": fused_dim + 1,
        "measurement_passes_per_sample": len(BASIS_NAMES),
        "reservoir_rounds": 2,
        "one_qubit_gates_per_sample": 40,
        "two_qubit_gates_per_sample": 8,
        "approx_depth": 10,
        "optimizer": "adam_for_fc_only",
        "slot_strategy": "correlation_guided_raw_plus_direct_semantic_slots",
        "readout_strategy": "compressed_multi_basis",
    }


def main() -> None:
    """训练入口：数据预处理 -> 槽位选择 -> 量子读出 -> FC 头训练 -> 模型落盘。"""
    # 训练超参数与输入输出路径。
    parser = argparse.ArgumentParser(description="Train CSMB_QRR")
    parser.add_argument("--train", type=str, default="train.csv", help="Training csv path")
    parser.add_argument("--model_out", type=str, default="model.json", help="Model output path")
    parser.add_argument("--shots_per_basis", type=int, default=384, help="Quantum shots for each basis")
    parser.add_argument("--head_steps", type=int, default=3200, help="Optimization steps for the FC head")
    parser.add_argument("--head_lr", type=float, default=0.015, help="Learning rate for the FC head")
    parser.add_argument("--lambda_w", type=float, default=8e-4, help="L2 coefficient for the FC head")
    parser.add_argument("--delta", type=float, default=0.35, help="Pseudo-Huber delta")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Global seed for reproducible training",
    )
    parser.add_argument("--top_raw", type=int, default=DEFAULT_TOP_RAW, help="How many top raw features to keep")
    parser.add_argument(
        "--top_semantic",
        type=int,
        default=DEFAULT_TOP_SEMANTIC,
        help="How many top semantic slots to keep",
    )
    args = parser.parse_args()
    base_seed = set_global_seed(args.seed)

    t0 = time.time()

    # 1) 读取数据，并只删除明显异常的年龄脏数据。
    df = load_table(args.train)
    df, removed_age_rows = clean_training_table(df)
    y = df[TARGET_COL].to_numpy(dtype=float)

    # 2) 原始特征标准化 -> 直接构造语义候选，并分别标准化。
    x_raw, raw_stats = fit_raw_preprocess(df)
    semantic_raw = build_semantic_candidates(x_raw)
    semantic_slots, semantic_stats = fit_semantic_preprocess(semantic_raw)

    # 3) 相关性筛选并打包成 8 维 hybrid slots。
    raw_selected, semantic_selected, raw_scores, semantic_scores = select_hybrid_slots(
        x_raw,
        semantic_slots,
        y,
        top_raw=int(args.top_raw),
        top_semantic=int(args.top_semantic),
    )
    slot_df, slot_names, slot_roles = pack_hybrid_slots(
        x_raw,
        semantic_slots,
        raw_selected,
        semantic_selected,
    )

    # 4) 量子蓄水池提取读出特征，并与原 slots 融合。
    slot_matrix = slot_df.to_numpy(dtype=float)
    reservoir = CorrelationSemanticReservoir(
        shots_per_basis=args.shots_per_basis,
        rng_seed=base_seed,
    )
    quantum_features = reservoir.transform(slot_matrix)
    fused_features = np.hstack([slot_matrix, quantum_features])

    # 5) 训练有界 FC 头，初始化偏置使初始输出接近目标均值。
    mean_target = float(np.mean(y))
    p0 = min(max((mean_target - OUTPUT_MIN) / (OUTPUT_MAX - OUTPUT_MIN), 1e-4), 1.0 - 1e-4)
    w = np.zeros(fused_features.shape[1], dtype=float)
    b = float(math.log(p0 / (1.0 - p0)))
    w, b = train_head(fused_features, y, w, b, args.head_steps, args.head_lr, args.lambda_w, args.delta)

    # 6) 训练集指标。
    train_preds = bounded_forward(fused_features, w, b)
    train_mae = compute_mae(y, train_preds)
    train_score = score_from_mae(train_mae)
    elapsed = time.time() - t0

    # 7) 保存完整模型元信息，保证评估阶段可无损复现预处理与推理路径。
    model = {
        "method_name": METHOD_NAME,
        "target_col": TARGET_COL,
        "output_min": OUTPUT_MIN,
        "output_max": OUTPUT_MAX,
        "raw_features": RAW_FEATURES,
        "aux_features": AUX_FEATURES,
        "semantic_candidate_names": SEMANTIC_CANDIDATE_NAMES,
        "selected_raw_features": raw_selected,
        "selected_semantic_slots": semantic_selected,
        "hybrid_slot_names": slot_names,
        "hybrid_slot_roles": slot_roles,
        "slot_dim": SLOT_DIM,
        "removed_bad_age_value": BAD_AGE_VALUE,
        "removed_bad_age_rows": int(removed_age_rows),
        "raw_preprocess_stats": raw_stats,
        "semantic_preprocess_stats": semantic_stats,
        "raw_correlation_scores": {key: float(val) for key, val in raw_scores.items()},
        "semantic_correlation_scores": {key: float(val) for key, val in semantic_scores.items()},
        "shots_per_basis": int(args.shots_per_basis),
        "seed": int(base_seed),
        "head_weights": w.tolist(),
        "head_bias": float(b),
        "readout_dim": READOUT_DIM,
        "train_mae": float(train_mae),
        "train_score": float(train_score),
        "training_seconds": float(elapsed),
        "innovation_tags": [
            "correlation_guided_hybrid_slots",
            "direct_semantic_slot_construction",
            "semantic_graph_entanglement",
            "compressed_multi_basis_readout",
            "fc_only_low_parameter_training",
        ],
        "hyperparameters": {
            "head_steps": int(args.head_steps),
            "head_lr": float(args.head_lr),
            "lambda_w": float(args.lambda_w),
            "delta": float(args.delta),
            "top_raw": int(args.top_raw),
            "top_semantic": int(args.top_semantic),
        },
        "complexity": summarize_complexity(
            fused_features.shape[1],
            top_raw=int(args.top_raw),
            top_semantic=int(args.top_semantic),
        ),
    }

    # 持久化模型。
    with open(args.model_out, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

    # 控制台输出训练摘要。
    print(f"[Info] method = {METHOD_NAME}")
    print(f"[Info] removed_age_rows = {removed_age_rows}")
    print(f"[Info] selected_raw = {raw_selected}")
    print(f"[Info] selected_semantic = {semantic_selected}")
    print(f"[Info] seed = {base_seed}")
    print(f"[Info] hybrid_slot_order = {list(zip(slot_roles, slot_names))}")
    print(f"[Done] train_MAE = {train_mae:.6f}")
    print(f"[Done] train_score = {train_score:.6f}")
    print(f"[Done] training_seconds = {elapsed:.2f}")
    print(f"[Done] model saved to: {args.model_out}")


if __name__ == "__main__":
    main()
