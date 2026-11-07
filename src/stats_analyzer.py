"""
src/stats_analyzer.py
Real-time Streaming Cronbach's Alpha & Psychometric Analytics Engine.
Implements vector Welford online covariance updates with O(1) memory and O(K^2) time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.tally_schema import TallySchema


@dataclass
class ItemStats:
    """SPSS-grade Item-level statistics."""
    code: str
    item_number: int
    mean: float
    std: float
    citc: float
    alpha_if_deleted: float


@dataclass
class ConstructStats:
    """Construct-level reliability & descriptive statistics."""
    code: str
    name: str
    item_count: int
    is_reverse: bool
    mean: float
    std: float
    alpha: float
    mean_raw: Optional[float] = None          # Populated for reverse constructs (PC)
    mean_reversed: Optional[float] = None     # Populated for reverse constructs (PC)
    item_stats: Dict[str, ItemStats] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "code": self.code,
            "name": self.name,
            "item_count": self.item_count,
            "is_reverse": self.is_reverse,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "alpha": round(self.alpha, 4),
            "items": {
                k: {
                    "mean": round(v.mean, 4),
                    "std": round(v.std, 4),
                    "citc": round(v.citc, 4),
                    "alpha_if_deleted": round(v.alpha_if_deleted, 4),
                }
                for k, v in self.item_stats.items()
            },
        }
        if self.is_reverse:
            res["mean_raw"] = round(self.mean_raw or 0.0, 4)
            res["mean_reversed"] = round(self.mean_reversed or 0.0, 4)
        return res


@dataclass
class StatsSummary:
    """Overall executive summary for Streamlit UI and SPSS Exporter."""
    sample_count: int
    overall_alpha: float
    status: str                              # "INSUFFICIENT_DATA", "PROVISIONAL", "RELIABLE", "ZERO_VARIANCE"
    is_reliable: bool                        # True when sample_count >= 10 and total_var > 1e-9
    construct_stats: Dict[str, ConstructStats] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "overall_alpha": round(self.overall_alpha, 4),
            "status": self.status,
            "is_reliable": self.is_reliable,
            "constructs": {k: v.to_dict() for k, v in self.construct_stats.items()},
        }


class StreamingCovarianceTracker:
    """
    O(1) memory, numerically stable multivariate online covariance & Cronbach's Alpha calculator.
    Uses extended Welford algorithm with bit-level symmetry enforcement.
    """

    def __init__(self, k_items: int):
        if k_items < 2:
            raise ValueError(f"Cronbach's Alpha requires at least 2 items, got {k_items}")
        self.k: int = k_items
        self.n: int = 0
        self.mean: np.ndarray = np.zeros(k_items, dtype=np.float64)
        self.M2: np.ndarray = np.zeros((k_items, k_items), dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """Processes one observation vector of shape (k,)."""
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.shape != (self.k,):
            raise ValueError(f"Expected shape ({self.k},), got {x_arr.shape}")

        self.n += 1
        delta = x_arr - self.mean
        self.mean += delta / self.n
        delta2 = x_arr - self.mean

        # Enforce exact bit-level symmetry
        term = np.outer(delta, delta2)
        self.M2 += 0.5 * (term + term.T)

    def covariance_matrix(self) -> np.ndarray:
        """Returns unbiased sample covariance matrix. Shape (k, k)."""
        if self.n < 2:
            return np.zeros((self.k, self.k), dtype=np.float64)
        return self.M2 / (self.n - 1)

    def item_variances(self) -> np.ndarray:
        """Returns unbiased sample variances for all items."""
        if self.n < 2:
            return np.zeros(self.k, dtype=np.float64)
        return np.diag(self.M2) / (self.n - 1)

    def total_variance(self) -> float:
        """Returns unbiased sample variance of the sum of items."""
        if self.n < 2:
            return 0.0
        return float(np.sum(self.M2) / (self.n - 1))

    def compute_alpha(self) -> Tuple[float, str]:
        """
        Computes Cronbach's Alpha directly from M2 with ratio cancellation.
        Returns: (alpha_value, status_code)
        """
        if self.n < 3:
            return 0.0, "INSUFFICIENT_DATA"

        item_vars_sum = float(np.trace(self.M2))
        total_var_scaled = float(np.sum(self.M2))

        if total_var_scaled <= 1e-9:
            return 0.0, "ZERO_VARIANCE"

        ratio = item_vars_sum / total_var_scaled
        alpha = (self.k / (self.k - 1)) * (1.0 - ratio)

        clamped_alpha = max(0.0, min(1.0, float(alpha)))
        status = "RELIABLE" if self.n >= 10 else "PROVISIONAL"
        return clamped_alpha, status

    def cronbach_alpha(self) -> float:
        """Convenience method returning float alpha."""
        alpha, _ = self.compute_alpha()
        return alpha

    def reset(self) -> None:
        self.n = 0
        self.mean.fill(0.0)
        self.M2.fill(0.0)


class StatsAnalyzer:
    """
    Central Psychometric & Statistical Analyzer for Milestone M2.
    Integrates 11 construct trackers, 1 global 39-item tracker, automatic PC inversion,
    and SPSS-grade CITC diagnostic metrics.
    """

    CONSTRUCT_DEFS = [
        ("PU", "Perceived Usefulness (Nhận thức về sự hữu ích)", 4, False),
        ("PEOU", "Perceived Ease of Use (Nhận thức về tính dễ sử dụng)", 4, False),
        ("PES", "Perceived Environmental Sustainability (Bền vững môi trường)", 4, False),
        ("PC", "Perceived Costs (Nhận thức về chi phí)", 3, True),
        ("PI", "Peer Influence (Ảnh hưởng từ người xung quanh)", 3, False),
        ("SE", "Self-Efficacy (Năng lực tự thân)", 4, False),
        ("LC", "Location Convenience (Thuận tiện vị trí)", 3, False),
        ("ATT", "Attitude (Thái độ đối với dịch vụ)", 3, False),
        ("SN", "Subjective Norm (Chuẩn chủ quan)", 3, False),
        ("PBC", "Perceived Behavioral Control (Kiểm soát hành vi)", 4, False),
        ("ITU", "Intention to Use (Ý định sử dụng)", 3, False),
    ]

    def __init__(self):
        TallySchema.initialize()

        self._global_tracker = StreamingCovarianceTracker(38)
        self._construct_trackers: Dict[str, StreamingCovarianceTracker] = {}
        self._construct_indices: Dict[str, List[int]] = {}
        self._code_to_idx: Dict[str, int] = {}
        self._idx_to_code: Dict[int, str] = {}
        self._uuid_to_idx: Dict[str, int] = {}
        self._pc_indices: List[int] = []
        self._raw_pc_tracker = StreamingCovarianceTracker(3)

        curr_idx = 0
        for code, _, count, is_rev in self.CONSTRUCT_DEFS:
            self._construct_trackers[code] = StreamingCovarianceTracker(count)
            indices = list(range(curr_idx, curr_idx + count))
            self._construct_indices[code] = indices

            for c_i, g_idx in enumerate(indices):
                item_code = f"{code}{c_i+1}"
                self._code_to_idx[item_code] = g_idx
                self._idx_to_code[g_idx] = item_code

                # Map group UUID if available in TallySchema
                q_def = TallySchema.CODE_MAP.get(item_code)
                if q_def:
                    self._uuid_to_idx[q_def.group_uuid] = g_idx

                if is_rev:
                    self._pc_indices.append(g_idx)

            curr_idx += count

        self._sample_count: int = 0

    def _extract_score(self, likert_scores: Dict[str, Any], item_code: str) -> float:
        """Extracts numeric score for item_code from dict keyed by code, UUID, or response list."""
        if item_code in likert_scores:
            val = likert_scores[item_code]
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, list) and len(val) > 0:
                opt_uuid = val[0]
                q_def = TallySchema.CODE_MAP.get(item_code)
                if q_def and opt_uuid in q_def.uuid_to_score:
                    return float(q_def.uuid_to_score[opt_uuid])
            return float(val)

        # Try groupUuid
        q_def = TallySchema.CODE_MAP.get(item_code)
        if q_def and q_def.group_uuid in likert_scores:
            val = likert_scores[q_def.group_uuid]
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, list) and len(val) > 0:
                opt_uuid = val[0]
                if opt_uuid in q_def.uuid_to_score:
                    return float(q_def.uuid_to_score[opt_uuid])
            return float(val)

        raise KeyError(f"Missing score for Likert item '{item_code}'")

    def add_sample(self, likert_scores: Dict[str, Any]) -> None:
        """
        Accepts dict of scores keyed by item code ('PU1'..'ITU3') or group UUID.
        Automatically tracks raw PC and inverts PC items (7 - raw) for full-model psychometrics.
        """
        vector_38 = np.zeros(38, dtype=np.float64)
        pc_raw_vector = np.zeros(3, dtype=np.float64)

        # 1. Normalize scores to 38-item canonical vector
        for code, g_idx in self._code_to_idx.items():
            val = self._extract_score(likert_scores, code)
            val = float(np.clip(val, 1.0, 6.0))
            vector_38[g_idx] = val

        # 2. Extract raw PC for raw tracking
        for p_i, g_idx in enumerate(self._pc_indices):
            pc_raw_vector[p_i] = vector_38[g_idx]

        # 3. Update raw PC tracker
        self._raw_pc_tracker.update(pc_raw_vector)

        # 4. Invert PC items for the global scale & construct scale
        for g_idx in self._pc_indices:
            vector_38[g_idx] = 7.0 - vector_38[g_idx]

        # 5. Update global tracker
        self._global_tracker.update(vector_38)

        # 6. Update each construct tracker
        for code, indices in self._construct_indices.items():
            sub_vec = vector_38[indices]
            self._construct_trackers[code].update(sub_vec)

        self._sample_count += 1

    def get_cronbach_alpha(self) -> Dict[str, float]:
        """Returns dict of Cronbach's Alpha for all 11 constructs and 'overall'."""
        res: Dict[str, float] = {}
        for code in self._construct_trackers:
            alpha, _ = self._construct_trackers[code].compute_alpha()
            res[code] = alpha
        overall_alpha, _ = self._global_tracker.compute_alpha()
        res["overall"] = overall_alpha
        return res

    def get_construct_stats(self, construct_code: str) -> ConstructStats:
        """Computes comprehensive SPSS-grade construct and item statistics."""
        target_def = None
        for item in self.CONSTRUCT_DEFS:
            if item[0] == construct_code:
                target_def = item
                break

        if target_def is None:
            raise KeyError(f"Unknown construct code: '{construct_code}'")

        code, name, count, is_rev = target_def
        tracker = self._construct_trackers[code]
        alpha, _ = tracker.compute_alpha()

        cov_matrix = tracker.covariance_matrix()
        item_means = tracker.mean
        item_variances = tracker.item_variances()
        total_var = tracker.total_variance()

        # Mean and Std of scale sum
        scale_mean = float(np.sum(item_means))
        scale_std = math.sqrt(max(0.0, total_var))

        # Average item mean and std
        construct_mean = float(np.mean(item_means)) if count > 0 else 0.0
        construct_std = math.sqrt(max(0.0, total_var / (count**2))) if count > 0 else 0.0

        item_stats_dict: Dict[str, ItemStats] = {}
        indices = self._construct_indices[code]

        for i in range(count):
            item_code = f"{code}{i+1}"
            it_mean = float(item_means[i])
            it_std = math.sqrt(max(0.0, float(item_variances[i])))

            # Variance if item i is deleted: s_{T-i}^2 = s_T^2 + s_{ii} - 2 * sum_l s_{il}
            cov_sum_i = float(np.sum(cov_matrix[i, :]))
            s_T_minus_i_sq = max(0.0, total_var + float(item_variances[i]) - 2.0 * cov_sum_i)

            # Corrected Item-Total Correlation: (sum_{l != i} s_{il}) / sqrt(s_{ii} * s_{T-i}^2)
            cov_other_sum = cov_sum_i - float(item_variances[i])
            denom = math.sqrt(max(0.0, float(item_variances[i]) * s_T_minus_i_sq))
            citc = float(cov_other_sum / denom) if denom > 1e-9 else 0.0

            # Alpha if item deleted (requires at least 3 items originally, i.e., count >= 3)
            if count >= 3 and s_T_minus_i_sq > 1e-9:
                tr_sub = float(np.trace(cov_matrix)) - float(item_variances[i])
                ratio_sub = tr_sub / s_T_minus_i_sq
                alpha_if_del = ((count - 1) / (count - 2)) * (1.0 - ratio_sub)
                alpha_if_del = max(0.0, min(1.0, float(alpha_if_del)))
            else:
                alpha_if_del = 0.0

            item_stats_dict[item_code] = ItemStats(
                code=item_code,
                item_number=i + 1,
                mean=it_mean,
                std=it_std,
                citc=citc,
                alpha_if_deleted=alpha_if_del,
            )

        mean_raw = None
        mean_reversed = None
        if is_rev:
            mean_raw = float(np.mean(self._raw_pc_tracker.mean))
            mean_reversed = 7.0 - mean_raw

        return ConstructStats(
            code=code,
            name=name,
            item_count=count,
            is_reverse=is_rev,
            mean=construct_mean,
            std=construct_std,
            alpha=alpha,
            mean_raw=mean_raw,
            mean_reversed=mean_reversed,
            item_stats=item_stats_dict,
        )

    def get_summary(self) -> StatsSummary:
        """Returns comprehensive summary containing all constructs and overall status."""
        overall_alpha, status = self._global_tracker.compute_alpha()
        is_reliable = (self._sample_count >= 10) and (self._global_tracker.total_variance() > 1e-9)

        construct_stats: Dict[str, ConstructStats] = {}
        for code, _, _, _ in self.CONSTRUCT_DEFS:
            construct_stats[code] = self.get_construct_stats(code)

        return StatsSummary(
            sample_count=self._sample_count,
            overall_alpha=overall_alpha,
            status=status,
            is_reliable=is_reliable,
            construct_stats=construct_stats,
        )

    def reset(self) -> None:
        """Clears all trackers and resets state."""
        self._sample_count = 0
        self._global_tracker.reset()
        self._raw_pc_tracker.reset()
        for t in self._construct_trackers.values():
            t.reset()
