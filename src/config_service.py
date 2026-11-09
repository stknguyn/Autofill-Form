"""
src/config_service.py - Session Configuration Persistence Service.
Manages saving and loading operational configurations (Campaign, Pacing, Scheduler, Demographics, TAM/TPB Constructs)
between user sessions via JSON persistence.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Default persistence file in repository root
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "session_config.json"

DEFAULT_CONSTRUCT_BENCHMARKS: Dict[str, float] = {
    "PU": 4.65,    # Cảm nhận hữu ích (Perceived Usefulness)
    "PEOU": 4.70,  # Cảm nhận dễ sử dụng (Perceived Ease of Use)
    "PES": 4.75,   # Tiết kiệm kinh tế (Perceived Economic Savings)
    "PC": 2.35,    # Độ phức tạp công nghệ (Perceived Complexity - nghịch đảo)
    "PI": 4.25,    # Đổi mới cá nhân (Personal Innovativeness)
    "SE": 4.75,    # Tự tin công nghệ (Self-Efficacy)
    "LC": 4.70,    # Vị trí thuận tiện (Location Convenience)
    "ATT": 4.65,   # Thái độ đối với Smart Locker (Attitude Toward Using)
    "SN": 4.30,    # Chuẩn chủ quan xã hội (Subjective Norms)
    "PBC": 4.70,   # Kiểm soát hành vi nhận thức (Behavioral Control)
    "ITU": 4.65,   # Ý định sử dụng trong tương lai (Intention to Use)
}

DEFAULT_SESSION_CONFIG: Dict[str, Any] = {
    "campaign": {
        "form_url": "https://tally.so/r/laqVzV",
        "target_count": 150,
        "num_workers": 6,
        "master_seed": 42,
    },
    "pacing": {
        "dist_choice": "Truncated Gaussian (Người thật)",
        "delay_min": 1800.0,
        "delay_max": 3600.0,
        "enable_micro": True,
        "break_freq_min": 15,
        "break_freq_max": 25,
        "break_dur_min": 10.0,
        "break_dur_max": 30.0,
    },
    "scheduler": {
        "enable_sched": False,
        "start_delay_sec": 0,
        "active_windows_str": "08:00-12:00, 13:30-17:30",
        "batch_limit_val": 50,
        "cooldown_sec": 300.0,
    },
    "demographics": {
        "min_age": 19,
        "age_mean": 21.8,
        "age_std": 1.9,
        "female_pct": 53,
        "hcm_pct": 50,
        "hanoi_pct": 40,
        "danang_pct": 10,
    },
    "psychometrics": {
        "alpha_threshold": 0.780,
        "alpha_ceiling": 0.790,
        "construct_means": DEFAULT_CONSTRUCT_BENCHMARKS.copy(),
    },
}


class ConfigService:
    """Service handling configuration persistence across user sessions."""

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Returns a deep-copy of the standard default configuration."""
        import copy
        return copy.deepcopy(DEFAULT_SESSION_CONFIG)

    @classmethod
    def save_session_config(
        cls,
        config_data: Dict[str, Any],
        filepath: Optional[Union[Path, str]] = None,
    ) -> Path:
        """
        Saves configuration dictionary to a local JSON file with current timestamp.
        """
        target_path = Path(filepath) if filepath else DEFAULT_CONFIG_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "saved_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0",
            **config_data,
        }

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return target_path

    @classmethod
    def load_session_config(
        cls,
        filepath: Optional[Union[Path, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Loads configuration from the local JSON file.
        Returns None if file does not exist or contains invalid JSON.
        """
        target_path = Path(filepath) if filepath else DEFAULT_CONFIG_PATH
        if not target_path.exists():
            return None

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            return None
        except Exception:
            return None

    @classmethod
    def has_saved_config(cls, filepath: Optional[Union[Path, str]] = None) -> bool:
        """Returns True if a valid saved configuration file exists."""
        target_path = Path(filepath) if filepath else DEFAULT_CONFIG_PATH
        return target_path.exists() and target_path.is_file()

    @classmethod
    def get_saved_config_summary(
        cls,
        filepath: Optional[Union[Path, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Returns brief metadata summary of the saved configuration."""
        data = cls.load_session_config(filepath)
        if not data:
            return None

        saved_at = data.get("saved_at", "Không rõ")
        campaign = data.get("campaign", {})
        pacing = data.get("pacing", {})

        return {
            "saved_at": saved_at,
            "target_count": campaign.get("target_count", 150),
            "num_workers": campaign.get("num_workers", 6),
            "delay_min": pacing.get("delay_min", 1800.0),
            "delay_max": pacing.get("delay_max", 3600.0),
        }

    @classmethod
    def apply_to_session_state(
        cls,
        config: Dict[str, Any],
        session_state_obj: Any,
    ) -> None:
        """
        Applies a loaded configuration dictionary into Streamlit's session_state keys.
        """
        campaign = config.get("campaign", {})
        pacing = config.get("pacing", {})
        scheduler = config.get("scheduler", {})
        demographics = config.get("demographics", {})
        psychometrics = config.get("psychometrics", {})

        # Campaign
        if "form_url" in campaign:
            session_state_obj["cfg_form_url"] = str(campaign["form_url"])
        if "target_count" in campaign:
            session_state_obj["cfg_target_count"] = int(campaign["target_count"])
        if "num_workers" in campaign:
            session_state_obj["cfg_num_workers"] = int(campaign["num_workers"])
        if "master_seed" in campaign:
            session_state_obj["cfg_master_seed"] = int(campaign["master_seed"])

        # Pacing
        if "dist_choice" in pacing:
            session_state_obj["cfg_dist_choice"] = str(pacing["dist_choice"])
        d_min = float(pacing.get("delay_min", 1800.0))
        d_max = float(pacing.get("delay_max", 3600.0))
        session_state_obj["cfg_delay_range"] = (d_min, d_max)

        if "enable_micro" in pacing:
            session_state_obj["cfg_enable_micro"] = bool(pacing["enable_micro"])
        bf_min = int(pacing.get("break_freq_min", 15))
        bf_max = int(pacing.get("break_freq_max", 25))
        session_state_obj["cfg_break_freq"] = (bf_min, bf_max)
        bd_min = float(pacing.get("break_dur_min", 10.0))
        bd_max = float(pacing.get("break_dur_max", 30.0))
        session_state_obj["cfg_break_dur"] = (bd_min, bd_max)

        # Scheduler
        if "enable_sched" in scheduler:
            session_state_obj["cfg_enable_sched"] = bool(scheduler["enable_sched"])
        if "start_delay_sec" in scheduler:
            session_state_obj["cfg_start_delay_sec"] = int(scheduler["start_delay_sec"])
        if "active_windows_str" in scheduler:
            session_state_obj["cfg_active_windows_str"] = str(scheduler["active_windows_str"])
        if "batch_limit_val" in scheduler:
            session_state_obj["cfg_batch_limit_val"] = int(scheduler["batch_limit_val"])
        if "cooldown_sec" in scheduler:
            session_state_obj["cfg_cooldown_sec"] = float(scheduler["cooldown_sec"])

        # Demographics
        if "min_age" in demographics:
            session_state_obj["cfg_min_age"] = int(demographics["min_age"])
        if "age_mean" in demographics:
            session_state_obj["cfg_age_mean"] = float(demographics["age_mean"])
        if "age_std" in demographics:
            session_state_obj["cfg_age_std"] = float(demographics["age_std"])
        if "female_pct" in demographics:
            session_state_obj["cfg_female_pct"] = int(demographics["female_pct"])
        if "hcm_pct" in demographics:
            session_state_obj["cfg_hcm_pct"] = int(demographics["hcm_pct"])
        if "hanoi_pct" in demographics:
            session_state_obj["cfg_hanoi_pct"] = int(demographics["hanoi_pct"])
        if "danang_pct" in demographics:
            session_state_obj["cfg_danang_pct"] = int(demographics["danang_pct"])

        # Psychometrics
        if "alpha_threshold" in psychometrics:
            val = float(psychometrics["alpha_threshold"])
            session_state_obj["cfg_alpha_threshold"] = val
            session_state_obj["alpha_threshold"] = val
        if "alpha_ceiling" in psychometrics:
            val = float(psychometrics["alpha_ceiling"])
            session_state_obj["cfg_alpha_ceiling"] = val
            session_state_obj["alpha_ceiling"] = val

        construct_means = psychometrics.get("construct_means", {})
        if isinstance(construct_means, dict):
            session_state_obj["custom_constructs"] = construct_means.copy()
            mapping = {
                "PU": "cfg_pu_mean",
                "PEOU": "cfg_peou_mean",
                "PES": "cfg_pes_mean",
                "PI": "cfg_pi_mean",
                "SE": "cfg_se_mean",
                "LC": "cfg_lc_mean",
                "PC": "cfg_pc_mean",
                "ATT": "cfg_att_mean",
                "SN": "cfg_sn_mean",
                "PBC": "cfg_pbc_mean",
                "ITU": "cfg_itu_mean",
            }
            for code, key in mapping.items():
                if code in construct_means:
                    session_state_obj[key] = float(construct_means[code])

    @classmethod
    def init_session_state(cls, session_state_obj: Any) -> None:
        """Initializes default keys in session_state if they do not exist."""
        defaults = cls.get_default_config()

        # Check if already initialized
        if "cfg_target_count" in session_state_obj:
            return

        cls.apply_to_session_state(defaults, session_state_obj)

    @classmethod
    def extract_from_session_state(cls, session_state_obj: Any) -> Dict[str, Any]:
        """
        Extracts current configuration settings from Streamlit's session_state into a structured dictionary.
        """
        cfg_delay_range = session_state_obj.get("cfg_delay_range", (1800.0, 3600.0))
        if isinstance(cfg_delay_range, (list, tuple)) and len(cfg_delay_range) >= 2:
            delay_min = float(cfg_delay_range[0])
            delay_max = float(cfg_delay_range[1])
        else:
            delay_min, delay_max = 1800.0, 3600.0

        cfg_break_freq = session_state_obj.get("cfg_break_freq", (15, 25))
        if isinstance(cfg_break_freq, (list, tuple)) and len(cfg_break_freq) >= 2:
            break_freq_min = int(cfg_break_freq[0])
            break_freq_max = int(cfg_break_freq[1])
        else:
            break_freq_min, break_freq_max = 15, 25

        cfg_break_dur = session_state_obj.get("cfg_break_dur", (10.0, 30.0))
        if isinstance(cfg_break_dur, (list, tuple)) and len(cfg_break_dur) >= 2:
            break_dur_min = float(cfg_break_dur[0])
            break_dur_max = float(cfg_break_dur[1])
        else:
            break_dur_min, break_dur_max = 10.0, 30.0

        constructs = {
            "PU": float(session_state_obj.get("cfg_pu_mean", DEFAULT_CONSTRUCT_BENCHMARKS["PU"])),
            "PEOU": float(session_state_obj.get("cfg_peou_mean", DEFAULT_CONSTRUCT_BENCHMARKS["PEOU"])),
            "PES": float(session_state_obj.get("cfg_pes_mean", DEFAULT_CONSTRUCT_BENCHMARKS["PES"])),
            "PC": float(session_state_obj.get("cfg_pc_mean", DEFAULT_CONSTRUCT_BENCHMARKS["PC"])),
            "PI": float(session_state_obj.get("cfg_pi_mean", DEFAULT_CONSTRUCT_BENCHMARKS["PI"])),
            "SE": float(session_state_obj.get("cfg_se_mean", DEFAULT_CONSTRUCT_BENCHMARKS["SE"])),
            "LC": float(session_state_obj.get("cfg_lc_mean", DEFAULT_CONSTRUCT_BENCHMARKS["LC"])),
            "ATT": float(session_state_obj.get("cfg_att_mean", DEFAULT_CONSTRUCT_BENCHMARKS["ATT"])),
            "SN": float(session_state_obj.get("cfg_sn_mean", DEFAULT_CONSTRUCT_BENCHMARKS["SN"])),
            "PBC": float(session_state_obj.get("cfg_pbc_mean", DEFAULT_CONSTRUCT_BENCHMARKS["PBC"])),
            "ITU": float(session_state_obj.get("cfg_itu_mean", DEFAULT_CONSTRUCT_BENCHMARKS["ITU"])),
        }

        return {
            "campaign": {
                "form_url": str(session_state_obj.get("cfg_form_url", "https://tally.so/r/laqVzV")),
                "target_count": int(session_state_obj.get("cfg_target_count", 150)),
                "num_workers": int(session_state_obj.get("cfg_num_workers", 6)),
                "master_seed": int(session_state_obj.get("cfg_master_seed", 42)),
            },
            "pacing": {
                "dist_choice": str(session_state_obj.get("cfg_dist_choice", "Truncated Gaussian (Người thật)")),
                "delay_min": delay_min,
                "delay_max": delay_max,
                "enable_micro": bool(session_state_obj.get("cfg_enable_micro", True)),
                "break_freq_min": break_freq_min,
                "break_freq_max": break_freq_max,
                "break_dur_min": break_dur_min,
                "break_dur_max": break_dur_max,
            },
            "scheduler": {
                "enable_sched": bool(session_state_obj.get("cfg_enable_sched", False)),
                "start_delay_sec": int(session_state_obj.get("cfg_start_delay_sec", 0)),
                "active_windows_str": str(session_state_obj.get("cfg_active_windows_str", "08:00-12:00, 13:30-17:30")),
                "batch_limit_val": int(session_state_obj.get("cfg_batch_limit_val", 50)) if session_state_obj.get("cfg_batch_limit_val") is not None else 50,
                "cooldown_sec": float(session_state_obj.get("cfg_cooldown_sec", 300.0)),
            },
            "demographics": {
                "min_age": int(session_state_obj.get("cfg_min_age", 19)),
                "age_mean": float(session_state_obj.get("cfg_age_mean", 21.8)),
                "age_std": float(session_state_obj.get("cfg_age_std", 1.9)),
                "female_pct": int(session_state_obj.get("cfg_female_pct", 53)),
                "hcm_pct": int(session_state_obj.get("cfg_hcm_pct", 50)),
                "hanoi_pct": int(session_state_obj.get("cfg_hanoi_pct", 40)),
                "danang_pct": int(session_state_obj.get("cfg_danang_pct", 10)),
            },
            "psychometrics": {
                "alpha_threshold": float(session_state_obj.get("cfg_alpha_threshold", 0.780)),
                "alpha_ceiling": float(session_state_obj.get("cfg_alpha_ceiling", 0.790)),
                "construct_means": constructs,
            },
        }
