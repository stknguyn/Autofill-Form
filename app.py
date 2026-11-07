"""
app.py - Scientific Survey Automation Web Dashboard & Telemetry Monitor.
Milestone M4: Streamlit Dashboard & SPSS Export.

Adheres strictly to ui-ux-pro-max:
- Swiss Minimalism & Data-Dense Dashboard aesthetic (Slate 900 #0F172A theme)
- Non-blocking scoped telemetry: Isolated @st.fragment(run_every="1s") for KPI & Fleet Grid
- Independent tab layout outside high-frequency fragments: Zero tab jumping, zero screen flicker
- 5 Top-Level KPI Metric Cards with dynamic status pills
- Dynamic Worker Fleet Grid with 6-state FSM transitions and latency stats
- Psychometrics & Demographics with 8 Plotly figures (TAM & TPB Radar, Cronbach Alpha Gauge, 95% CI Bars)
- Direct downloads for SPSS Pure Numeric CSV and 3-Sheet Excel with Variable Codebook
- Fully compliant with Streamlit 1.55+ (width="stretch", zero deprecation warnings)
- Bulletproof unindented HTML rendering: zero raw div tag leaks or markdown codeblock glitches
- PyArrow-safe dataframe typing (float nulls instead of string hyphens in numeric columns)
- Comprehensive customization of default variables: Campaign, Pacing, Scheduler, Demographics & 11 TAM/TPB Latent Constructs
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure root directory is in sys.path for direct module resolution
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import streamlit as st

from src.config_service import ConfigService
from src.export_service import ExportService
from src.scheduler import JitterConfig, JitterDistribution, SchedulerConfig, SchedulerState
from src.stats_analyzer import StatsSummary
from src.ui_components import (
    PLOTLY_CONFIG,
    SLATE_THEME,
    create_accessible_construct_bar_chart,
    create_age_distribution_chart,
    create_construct_alpha_bars,
    create_cronbach_alpha_gauge,
    create_demographic_charts,
    create_employment_donut_chart,
    create_frequency_chart,
    create_gender_donut_chart,
    create_location_chart,
    create_psychometric_radar_chart,
)
from src.worker_manager import FleetConfig, FleetState, WorkerCardSnapshot, WorkerFleetSnapshot, WorkerManager, WorkerState

# ---------------------------------------------------------------------------
# Default Research Model & Theoretical Benchmarks
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Shopee Gen Z Survey Fleet | TAM & TPB",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Swiss Slate 900 Theme & Data-Dense CSS Injection (ui-ux-pro-max)
# ---------------------------------------------------------------------------
SWISS_SLATE_CSS = """<style>
/* ---------------- Google Fonts (Inter & JetBrains Mono) ---------------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ---------------- Root Palette & Reset ---------------- */
:root {
  --bg-main: #0b1120;
  --bg-sidebar: #070d19;
  --bg-card: #131d31;
  --bg-card-hover: #1e293b;
  --border-card: #1e293b;
  --border-focus: #38bdf8;
  --text-primary: #f8fafc;
  --text-secondary: #cbd5e1;
  --text-muted: #94a3b8;
  --accent-blue: #3b82f6;
  --accent-emerald: #10b981;
  --accent-amber: #f59e0b;
  --accent-red: #ef4444;
  --accent-sky: #38bdf8;
  --accent-indigo: #818cf8;
}

/* Page container reset */
.stApp {
  background-color: var(--bg-main) !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

div[data-testid="stAppViewContainer"] {
  background-color: var(--bg-main) !important;
}

header[data-testid="stHeader"] {
  background-color: transparent !important;
}

.main .block-container {
  padding-top: 1.0rem !important;
  padding-bottom: 2.5rem !important;
  max-width: 98% !important;
  background-color: var(--bg-main) !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

/* Custom Sleek Scrollbar */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}
::-webkit-scrollbar-track {
  background: var(--bg-main);
}
::-webkit-scrollbar-thumb {
  background: #334155;
  border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
  background: #475569;
}

/* ---------------- Sidebar Styling ---------------- */
section[data-testid="stSidebar"] {
  background-color: var(--bg-sidebar) !important;
  border-right: 1px solid #1e293b !important;
}

div[data-testid="stSidebarContent"] {
  background-color: var(--bg-sidebar) !important;
}

div[data-testid="stSidebar"] [data-testid="stExpander"] {
  background-color: var(--bg-card) !important;
  border: 1px solid var(--border-card) !important;
  border-radius: 8px !important;
  margin-bottom: 10px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
  transition: all 0.2s ease-in-out;
}

div[data-testid="stSidebar"] [data-testid="stExpander"]:hover {
  border-color: #334155 !important;
}

div[data-testid="stSidebar"] [data-testid="stExpander"] details {
  border: none !important;
}

div[data-testid="stSidebar"] [data-testid="stExpander"] summary {
  color: #f1f5f9 !important;
  font-size: 0.84rem !important;
  font-weight: 600 !important;
}

div[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
  color: #38bdf8 !important;
}

/* Sidebar master action buttons semantic styling */
div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4)) > div:nth-child(1) button {
  background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
  color: #ffffff !important;
  border: 1px solid #059669 !important;
  font-weight: 600 !important;
  border-radius: 6px !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4)) > div:nth-child(1) button:hover:not(:disabled) {
  background: linear-gradient(135deg, #047857 0%, #059669 100%) !important;
  box-shadow: 0 4px 14px rgba(16, 185, 129, 0.45) !important;
  transform: translateY(-1px);
}

div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4)) > div:nth-child(2) button {
  background: rgba(245, 158, 11, 0.15) !important;
  color: #fbbf24 !important;
  border: 1px solid rgba(245, 158, 11, 0.4) !important;
  font-weight: 600 !important;
  border-radius: 6px !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4)) > div:nth-child(2) button:hover:not(:disabled) {
  background: rgba(245, 158, 11, 0.25) !important;
  border-color: #f59e0b !important;
  box-shadow: 0 4px 12px rgba(245, 158, 11, 0.25) !important;
  transform: translateY(-1px);
}

div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4)) > div:nth-child(3) button {
  background: rgba(239, 68, 68, 0.15) !important;
  color: #f87171 !important;
  border: 1px solid rgba(239, 68, 68, 0.4) !important;
  font-weight: 600 !important;
  border-radius: 6px !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4)) > div:nth-child(3) button:hover:not(:disabled) {
  background: rgba(239, 68, 68, 0.25) !important;
  border-color: #ef4444 !important;
  box-shadow: 0 4px 12px rgba(239, 68, 68, 0.25) !important;
  transform: translateY(-1px);
}

div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4)) > div:nth-child(4) button {
  background: rgba(148, 163, 184, 0.12) !important;
  color: #cbd5e1 !important;
  border: 1px solid rgba(148, 163, 184, 0.3) !important;
  font-weight: 600 !important;
  border-radius: 6px !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4)) > div:nth-child(4) button:hover:not(:disabled) {
  background: rgba(148, 163, 184, 0.22) !important;
  border-color: #94a3b8 !important;
  transform: translateY(-1px);
}

/* Sidebar session config buttons styling (Load & Save) */
div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:not(:has(> div:nth-child(3))) > div:nth-child(1) button {
  background: rgba(56, 189, 248, 0.12) !important;
  color: #38bdf8 !important;
  border: 1px solid rgba(56, 189, 248, 0.35) !important;
  font-weight: 600 !important;
  font-size: 0.80rem !important;
  border-radius: 6px !important;
  transition: all 0.2s ease-in-out;
}
div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:not(:has(> div:nth-child(3))) > div:nth-child(1) button:hover:not(:disabled) {
  background: rgba(56, 189, 248, 0.22) !important;
  border-color: #38bdf8 !important;
  box-shadow: 0 2px 10px rgba(56, 189, 248, 0.3) !important;
}

div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:not(:has(> div:nth-child(3))) > div:nth-child(2) button {
  background: rgba(16, 185, 129, 0.12) !important;
  color: #34d399 !important;
  border: 1px solid rgba(16, 185, 129, 0.35) !important;
  font-weight: 600 !important;
  font-size: 0.80rem !important;
  border-radius: 6px !important;
  transition: all 0.2s ease-in-out;
}
div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:not(:has(> div:nth-child(3))) > div:nth-child(2) button:hover:not(:disabled) {
  background: rgba(16, 185, 129, 0.22) !important;
  border-color: #10b981 !important;
  box-shadow: 0 2px 10px rgba(16, 185, 129, 0.3) !important;
}

div[data-testid="stSidebar"] button:disabled {
  opacity: 0.40 !important;
  cursor: not-allowed !important;
  background: #131d31 !important;
  color: #64748b !important;
  border-color: #1e293b !important;
}

/* ---------------- Tabs Styling ---------------- */
.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
  background-color: transparent !important;
  border-bottom: 1px solid #1e293b !important;
  padding-bottom: 4px;
  margin-bottom: 12px;
}
.stTabs [data-baseweb="tab"] {
  background-color: transparent !important;
  border: none !important;
  border-radius: 6px 6px 0 0 !important;
  color: #94a3b8 !important;
  font-size: 0.88rem !important;
  font-weight: 600 !important;
  padding: 8px 16px !important;
  transition: all 0.2s ease-in-out;
}
.stTabs [data-baseweb="tab"]:hover {
  color: #f8fafc !important;
  background-color: rgba(30, 41, 59, 0.4) !important;
}
.stTabs [aria-selected="true"] {
  color: #38bdf8 !important;
  border-bottom: 2px solid #38bdf8 !important;
  background-color: rgba(56, 189, 248, 0.08) !important;
}

/* ---------------- Download Buttons in Tab 3 ---------------- */
.stDownloadButton button {
  background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
  color: #ffffff !important;
  border: 1px solid #3b82f6 !important;
  font-weight: 600 !important;
  padding: 10px 18px !important;
  border-radius: 8px !important;
  box-shadow: 0 2px 6px rgba(37, 99, 235, 0.3) !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
.stDownloadButton button:hover {
  background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
  box-shadow: 0 4px 16px rgba(59, 130, 246, 0.45) !important;
  transform: translateY(-1px);
}

/* ---------------- Top-level KPI Metric Cards ---------------- */
.kpi-row {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px;
  margin-bottom: 18px;
}
@media (max-width: 1200px) {
  .kpi-row { grid-template-columns: repeat(3, 1fr); }
}
@media (max-width: 768px) {
  .kpi-row { grid-template-columns: 1fr; }
}

.kpi-card {
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  border-top: 2px solid #3b82f6;
  border-radius: 10px;
  padding: 14px 16px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  min-height: 108px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.28);
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
.kpi-card:hover {
  border-color: #334155;
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.38);
}
.kpi-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}
.kpi-title {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
}
.kpi-value {
  font-family: 'JetBrains Mono', 'Segoe UI', monospace;
  font-size: 1.65rem;
  font-weight: 700;
  color: #ffffff;
  line-height: 1.15;
  font-variant-numeric: tabular-nums;
  margin: 4px 0;
}
.kpi-subtext {
  font-size: 0.74rem;
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ---------------- Status Pills ---------------- */
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 2.5px 8px;
  border-radius: 9999px;
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  white-space: nowrap;
}
.pill-emerald { background: rgba(34, 197, 94, 0.16); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }
.pill-blue    { background: rgba(59, 130, 246, 0.16); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
.pill-amber   { background: rgba(245, 158, 11, 0.16); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
.pill-red     { background: rgba(239, 68, 68, 0.16); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
.pill-sky     { background: rgba(56, 189, 248, 0.16); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }
.pill-indigo  { background: rgba(129, 140, 248, 0.16); color: #a5b4fc; border: 1px solid rgba(129, 140, 248, 0.4); }
.pill-slate   { background: rgba(148, 163, 184, 0.16); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.4); }

/* Latency Micro Badges */
.lat-pill {
  display: inline-flex;
  align-items: center;
  padding: 1.5px 6px;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}
.lat-fast { background: rgba(16, 185, 129, 0.16); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }
.lat-med  { background: rgba(245, 158, 11, 0.16); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }
.lat-slow { background: rgba(239, 68, 68, 0.16); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.35); }
.lat-na   { background: rgba(148, 163, 184, 0.12); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.25); }

/* ---------------- Worker Fleet Grid ---------------- */
.worker-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(270px, 1fr));
  gap: 12px;
  margin-top: 12px;
}
.worker-card {
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  border-radius: 8px;
  padding: 12px 14px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.24);
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
.worker-card:hover {
  border-color: #334155;
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.35);
}
.worker-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.worker-name {
  font-family: 'JetBrains Mono', 'Segoe UI', monospace;
  font-weight: 700;
  font-size: 0.88rem;
  color: #ffffff;
}
.worker-task {
  font-size: 0.76rem;
  color: var(--text-secondary);
  margin-bottom: 8px;
  min-height: 20px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.worker-stats {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.72rem;
  color: var(--text-muted);
}
.worker-error-box {
  margin-top: 8px;
  padding: 5px 8px;
  background: rgba(239, 68, 68, 0.14);
  border-left: 3px solid var(--accent-red);
  color: #fca5a5;
  font-size: 0.70rem;
  border-radius: 3px;
}

/* ---------------- Banner Boxes ---------------- */
.dryrun-amber-box {
  background: rgba(245, 158, 11, 0.10);
  border: 1px solid rgba(245, 158, 11, 0.35);
  border-left: 4px solid var(--accent-amber);
  border-radius: 8px;
  padding: 10px 14px;
  margin-bottom: 12px;
  font-size: 0.80rem;
  color: #fef08a;
  line-height: 1.45;
}
.live-red-box {
  background: rgba(239, 68, 68, 0.10);
  border: 1px solid rgba(239, 68, 68, 0.35);
  border-left: 4px solid var(--accent-red);
  border-radius: 8px;
  padding: 10px 14px;
  margin-bottom: 12px;
  font-size: 0.80rem;
  color: #fca5a5;
  line-height: 1.45;
}
</style>"""
st.markdown(SWISS_SLATE_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# State Management Lifecycle (WorkerManager Singleton in session_state)
# ---------------------------------------------------------------------------
def get_or_init_worker_manager() -> WorkerManager:
    """Initializes and returns the singleton WorkerManager stored in session_state."""
    ConfigService.init_session_state(st.session_state)
    if "worker_manager" not in st.session_state:
        default_config = FleetConfig(
            target_count=150,
            num_workers=6,
            dry_run=True,
            delay_range=(1800.0, 3600.0),
            master_seed=42,
        )
        st.session_state.worker_manager = WorkerManager(config=default_config)
        st.session_state.action_feedback = None
        st.session_state.custom_constructs = DEFAULT_CONSTRUCT_BENCHMARKS.copy()
        st.session_state.alpha_threshold = 0.820

    return st.session_state.worker_manager


wm = get_or_init_worker_manager()


# ---------------------------------------------------------------------------
# Sidebar: Controls & Parameter Deck
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Điều Khiển Hệ Thống")

    # Fleet State Pill
    current_state = wm.fleet_state
    if current_state == FleetState.RUNNING:
        state_pill_html = '<span class="status-pill pill-emerald">● ĐANG CHẠY</span>'
    elif current_state == FleetState.PAUSED:
        state_pill_html = '<span class="status-pill pill-amber">❚❚ TẠM DỪNG</span>'
    elif current_state == FleetState.COMPLETED:
        state_pill_html = '<span class="status-pill pill-blue">✓ HOÀN THÀNH</span>'
    else:
        state_pill_html = '<span class="status-pill pill-slate">○ SẴN SÀNG</span>'

    st.markdown(f"**Trạng thái Fleet**: {state_pill_html}", unsafe_allow_html=True)

    # Dry-Run Safe Mode Toggle
    dry_run = st.toggle("Chế độ Dry-Run An Toàn", value=True, help="Giả lập toàn bộ quy trình mà không gửi HTTP POST thật tới máy chủ Tally.")
    if dry_run:
        dryrun_box_html = (
            '<div class="dryrun-amber-box">'
            '<b>⚠️ DRY-RUN MODE ĐANG BẬT</b><br>'
            'Mọi hồ sơ Gen Z và điểm Likert được sinh đầy đủ và kiểm tra ràng buộc nhưng KHÔNG nộp thật lên Tally.'
            '</div>'
        )
        st.markdown(dryrun_box_html, unsafe_allow_html=True)
    else:
        live_box_html = (
            '<div class="live-red-box">'
            '<b>🚨 CHẾ ĐỘ NỘP THẬT (LIVE MODE)</b><br>'
            'Dữ liệu sẽ được gửi HTTP POST thật tới <code>api.tally.so/forms/laqVzV/respond</code>. Khảo sát sẽ được ghi nhận vĩnh viễn!'
            '</div>'
        )
        st.markdown(live_box_html, unsafe_allow_html=True)

    # Master Action Buttons
    c_btn1, c_btn2, c_btn3, c_btn4 = st.columns(4)

    is_running = wm.is_running
    is_paused = wm.is_paused

    btn_start = c_btn1.button("▶ Chạy", disabled=is_running, width="stretch")
    if is_paused:
        btn_pause_resume = c_btn2.button("⏯ Tiếp", disabled=not is_running, width="stretch")
    else:
        btn_pause_resume = c_btn2.button("⏸ Dừng", disabled=not is_running, width="stretch")
    btn_stop = c_btn3.button("⏹ Ngắt", disabled=not is_running, width="stretch")
    btn_reset = c_btn4.button("🔄 Đặt lại", disabled=is_running, width="stretch")

    st.markdown("---")

    # Session Configuration Storage & Recovery (Persistent session_config.json)
    st.markdown("#### 💾 Cấu Hình Phiên (Session Config)")

    saved_summary = ConfigService.get_saved_config_summary()
    if saved_summary:
        saved_time = saved_summary.get("saved_at", "Không rõ")
        tgt = saved_summary.get("target_count", 150)
        wkr = saved_summary.get("num_workers", 6)
        d_min = saved_summary.get("delay_min", 1800.0)
        d_max = saved_summary.get("delay_max", 3600.0)
        st.markdown(
            f'<div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 6px; padding: 8px 12px; margin-bottom: 8px; font-size: 0.76rem; color: #cbd5e1;">'
            f'<div style="display: flex; justify-content: space-between; margin-bottom: 4px;">'
            f'<span style="font-weight: 600; color: #38bdf8;">🕒 Phiên đã lưu</span>'
            f'<span style="color: #94a3b8;">{saved_time}</span>'
            f'</div>'
            f'<div style="display: flex; gap: 8px; font-family: monospace; color: #94a3b8; font-size: 0.72rem;">'
            f'<span>🎯 N={tgt}</span> • '
            f'<span>⚙️ {wkr}w</span> • '
            f'<span>⏱️ {d_min/60:.0f}–{d_max/60:.0f}p</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background: rgba(30, 41, 59, 0.4); border: 1px dashed #334155; border-radius: 6px; padding: 8px 12px; margin-bottom: 8px; font-size: 0.76rem; color: #64748b;">'
            '📁 <i>Chưa có cấu hình phiên trước (session_config.json).</i>'
            '</div>',
            unsafe_allow_html=True,
        )

    c_cfg1, c_cfg2 = st.columns(2)
    has_saved = ConfigService.has_saved_config()
    btn_load_config = c_cfg1.button(
        "📂 Tải Cấu Hình Cũ",
        disabled=not has_saved or is_running,
        width="stretch",
        help="Nạp lại toàn bộ cấu hình phiên trước từ session_config.json",
    )
    btn_save_config = c_cfg2.button(
        "💾 Lưu Cấu Hình",
        disabled=is_running,
        width="stretch",
        help="Lưu lại cấu hình hiện tại vào session_config.json để dùng cho các phiên sau",
    )

    if btn_load_config:
        loaded = ConfigService.load_session_config()
        if loaded:
            ConfigService.apply_to_session_state(loaded, st.session_state)
            saved_time = loaded.get("saved_at", "trước")
            st.session_state.action_feedback = f"Đã nạp thành công cấu hình phiên trước (lưu lúc {saved_time})!"
            st.rerun()
        else:
            st.error("Không thể đọc file cấu hình phiên trước.")

    if btn_save_config:
        current_cfg = ConfigService.extract_from_session_state(st.session_state)
        saved_path = ConfigService.save_session_config(current_cfg)
        st.session_state.action_feedback = f"Đã lưu cấu hình phiên làm việc hiện tại vào {saved_path.name}!"
        st.rerun()

    st.markdown("---")

    # 1. Quota & Worker Pool
    with st.expander("🎯 Số Lượng Mẫu & Luồng Thực Thi", expanded=True):
        form_url = st.text_input("Form URL", key="cfg_form_url", disabled=is_running)
        target_count = st.number_input("Số mẫu mục tiêu (N)", min_value=1, max_value=2000, step=10, key="cfg_target_count", disabled=is_running)
        num_workers = st.slider("Số lượng Worker Threads", min_value=1, max_value=20, step=1, key="cfg_num_workers", disabled=is_running)
        master_seed = st.number_input("PRNG Master Seed", min_value=0, max_value=999999, step=1, key="cfg_master_seed", disabled=is_running, help="Đảm bảo tính tái lập (reproducibility) của các mẫu sinh ngẫu nhiên.")

    # 2. Pacing & Anti-Detection Jitter
    with st.expander("⏳ Pacing Delay & Jitter Chống Bot", expanded=False):
        dist_choice = st.radio("Kiểu phân phối Jitter", ["Truncated Gaussian (Người thật)", "Uniform (Đều)"], key="cfg_dist_choice", disabled=is_running)
        pacing_dist = JitterDistribution.TRUNCATED_GAUSSIAN if "Gaussian" in dist_choice else JitterDistribution.UNIFORM
        delay_min, delay_max = st.slider(
            "Khoảng trễ nộp form (giây)",
            min_value=1.0,
            max_value=7200.0,
            step=30.0,
            key="cfg_delay_range",
            disabled=is_running,
            help="Khoảng trễ giãn cách giữa các lần gửi form (mặc định 1800s - 3600s, tức 30 phút - 1 tiếng) để mô phỏng nhịp truy cập tự nhiên của người thật, xóa bỏ hoàn toàn dấu vết bot."
        )
        st.caption(
            f"⏱️ **Cấu hình trễ hiện tại**: {int(delay_min)}s ({delay_min/60:.1f} phút) đến {int(delay_max)}s ({delay_max/60:.1f} phút) "
            f"| Trung bình: ~{(delay_min+delay_max)/120:.1f} phút/lượt gửi."
        )
        enable_micro = st.checkbox("Mô phỏng giải lao ngắt quãng (Micro-breaks)", key="cfg_enable_micro", disabled=is_running)
        break_freq = (15, 25)
        break_dur = (10.0, 30.0)
        if enable_micro:
            break_freq = st.slider("Tần suất giải lao (mẫu)", 5, 50, key="cfg_break_freq", disabled=is_running)
            break_dur = st.slider("Thời lượng nghỉ (giây)", 5.0, 60.0, key="cfg_break_dur", disabled=is_running)
        else:
            break_freq = st.session_state.get("cfg_break_freq", (15, 25))
            break_dur = st.session_state.get("cfg_break_dur", (10.0, 30.0))

    # 3. Background Automation & Scheduling
    with st.expander("⏰ Lên Lịch Thực Thi (Scheduler)", expanded=False):
        enable_sched = st.toggle("Bật Tự Động Hóa & Khung Giờ", key="cfg_enable_sched", disabled=is_running)
        start_delay_sec = 0
        active_windows_str = None
        batch_limit_val = None
        cooldown_sec = 300.0

        if enable_sched:
            start_delay_sec = st.number_input("Hẹn giờ chạy sau (giây)", min_value=0, max_value=7200, step=30, key="cfg_start_delay_sec", disabled=is_running)
            active_windows_str = st.text_input("Khung giờ chạy (HH:MM-HH:MM)", key="cfg_active_windows_str", disabled=is_running)
            batch_limit_val = st.number_input("Giới hạn theo đợt (Batch size, 0 = không giới hạn)", min_value=0, max_value=500, step=10, key="cfg_batch_limit_val", disabled=is_running)
            if batch_limit_val == 0:
                batch_limit_val = None
            cooldown_sec = st.number_input("Thời gian hạ nhiệt (Cooldown giây)", min_value=10.0, max_value=3600.0, step=30.0, key="cfg_cooldown_sec", disabled=is_running)

    # 4. Demographic Biases Tuning (Default Demographic Variables)
    with st.expander("👥 Tùy Biến Trọng Số Nhân Khẩu Học Mặc Định", expanded=False):
        age_mean = st.slider("Độ tuổi trung bình (Gen Z)", 18.0, 27.0, step=0.1, key="cfg_age_mean", disabled=is_running)
        age_std = st.slider("Độ lệch chuẩn tuổi", 0.5, 4.0, step=0.1, key="cfg_age_std", disabled=is_running)
        female_pct = st.slider("Tỷ lệ Nữ (%)", 0, 100, step=1, key="cfg_female_pct", disabled=is_running)
        hcm_pct = st.slider("Tỷ lệ TP. Hồ Chí Minh (%)", 0, 100, step=1, key="cfg_hcm_pct", disabled=is_running)
        hanoi_pct = st.slider("Tỷ lệ Hà Nội (%)", 0, 100, step=1, key="cfg_hanoi_pct", disabled=is_running)
        danang_pct = st.slider("Tỷ lệ Đà Nẵng (%)", 0, 100, step=1, key="cfg_danang_pct", disabled=is_running)

        total_loc = hcm_pct + hanoi_pct + danang_pct
        if total_loc == 0:
            sum_loc = 1.0
            loc_w = [0.5, 0.4, 0.1, 0.0]
        else:
            sum_loc = float(total_loc)
            loc_w = [hcm_pct / sum_loc, hanoi_pct / sum_loc, danang_pct / sum_loc, 0.0]

        st.caption(
            f"📍 **Phân bổ 3 đô thị**: TP.HCM ({loc_w[0]*100:.1f}%), Hà Nội ({loc_w[1]*100:.1f}%), Đà Nẵng ({loc_w[2]*100:.1f}%) • Không lấy mẫu tỉnh khác (0%)."
        )

    # 5. TAM & TPB Latent Construct Defaults Customization
    with st.expander("📐 Tùy Biến Điểm Kỳ Vọng 11 Thang Đo TAM & TPB", expanded=False):
        st.caption(r"Tùy biến kỳ vọng điểm trung bình $\mu$ cho 11 thang đo Likert (1–6) và ngưỡng độ tin cậy NCKH.")
        alpha_threshold = st.slider(
            r"Ngưỡng Cronbach's Alpha NCKH ($\ge$ chuẩn)",
            min_value=0.600,
            max_value=0.950,
            step=0.005,
            key="cfg_alpha_threshold",
            disabled=is_running,
        )
        st.session_state.alpha_threshold = alpha_threshold

        c_t1, c_t2 = st.columns(2)
        with c_t1:
            pu_mean = st.slider("PU - Hữu ích", 1.0, 6.0, step=0.05, key="cfg_pu_mean", disabled=is_running)
            peou_mean = st.slider("PEOU - Dễ dùng", 1.0, 6.0, step=0.05, key="cfg_peou_mean", disabled=is_running)
            pes_mean = st.slider("PES - Tiết kiệm", 1.0, 6.0, step=0.05, key="cfg_pes_mean", disabled=is_running)
            pi_mean = st.slider("PI - Đổi mới cá nhân", 1.0, 6.0, step=0.05, key="cfg_pi_mean", disabled=is_running)
            se_mean = st.slider("SE - Tự tin công nghệ", 1.0, 6.0, step=0.05, key="cfg_se_mean", disabled=is_running)
            lc_mean = st.slider("LC - Vị trí tiện lợi", 1.0, 6.0, step=0.05, key="cfg_lc_mean", disabled=is_running)

        with c_t2:
            pc_mean = st.slider("PC - Phức tạp (nghịch)", 1.0, 6.0, step=0.05, key="cfg_pc_mean", disabled=is_running, help="Điểm thấp thể hiện Gen Z đánh giá tủ thông minh không hề phức tạp.")
            att_mean = st.slider("ATT - Thái độ tích cực", 1.0, 6.0, step=0.05, key="cfg_att_mean", disabled=is_running)
            sn_mean = st.slider("SN - Chuẩn chủ quan", 1.0, 6.0, step=0.05, key="cfg_sn_mean", disabled=is_running)
            pbc_mean = st.slider("PBC - Kiểm soát hành vi", 1.0, 6.0, step=0.05, key="cfg_pbc_mean", disabled=is_running)
            itu_mean = st.slider("ITU - Ý định sử dụng", 1.0, 6.0, step=0.05, key="cfg_itu_mean", disabled=is_running)

        custom_construct_means = {
            "PU": pu_mean,
            "PEOU": peou_mean,
            "PES": pes_mean,
            "PC": pc_mean,
            "PI": pi_mean,
            "SE": se_mean,
            "LC": lc_mean,
            "ATT": att_mean,
            "SN": sn_mean,
            "PBC": pbc_mean,
            "ITU": itu_mean,
        }
        st.session_state.custom_constructs = custom_construct_means

        if st.button("↺ Khôi Phục Điểm Chuẩn Lý Thuyết", disabled=is_running, width="stretch"):
            for code, key in [
                ("PU", "cfg_pu_mean"), ("PEOU", "cfg_peou_mean"), ("PES", "cfg_pes_mean"),
                ("PC", "cfg_pc_mean"), ("PI", "cfg_pi_mean"), ("SE", "cfg_se_mean"),
                ("LC", "cfg_lc_mean"), ("ATT", "cfg_att_mean"), ("SN", "cfg_sn_mean"),
                ("PBC", "cfg_pbc_mean"), ("ITU", "cfg_itu_mean")
            ]:
                st.session_state[key] = DEFAULT_CONSTRUCT_BENCHMARKS[code]
            st.session_state.cfg_alpha_threshold = 0.820
            st.session_state.alpha_threshold = 0.820
            st.session_state.custom_constructs = DEFAULT_CONSTRUCT_BENCHMARKS.copy()
            st.session_state.action_feedback = "Đã khôi phục 11 thang đo TAM&TPB về giá trị chuẩn mô hình ban đầu!"
            st.rerun()

    # Build configs and handle button clicks
    if btn_start:
        try:
            # Auto-save current session config so the active run is always persisted
            current_cfg = ConfigService.extract_from_session_state(st.session_state)
            ConfigService.save_session_config(current_cfg)
            # Build JitterConfig
            jitter_cfg = JitterConfig(
                distribution=pacing_dist,
                min_delay=delay_min,
                max_delay=delay_max,
                micro_break_enabled=enable_micro,
                micro_break_interval=break_freq,
                micro_break_duration=break_dur,
            )

            # Build SchedulerConfig
            sched_cfg = None
            if enable_sched:
                sched_cfg = SchedulerConfig(
                    start_delay_seconds=float(start_delay_sec) if start_delay_sec > 0 else None,
                    active_windows=active_windows_str if active_windows_str else None,
                    batch_limit=batch_limit_val,
                    cooldown_seconds=float(cooldown_sec),
                )

            # Build demographic weights
            f_ratio = female_pct / 100.0
            m_ratio = max(0.0, (1.0 - f_ratio) * 0.94)
            o_ratio = max(0.0, 1.0 - f_ratio - m_ratio)
            gender_w = [m_ratio, f_ratio, o_ratio]

            demo_w = {
                "age_mean": float(age_mean),
                "age_std": float(age_std),
                "gender_weights": gender_w,
                "location_weights": loc_w,
                "construct_means": custom_construct_means,
                "alpha_threshold": float(alpha_threshold),
            }

            wm.configure(
                target_count=int(target_count),
                num_workers=int(num_workers),
                dry_run=bool(dry_run),
                delay_range=(float(delay_min), float(delay_max)),
                jitter_config=jitter_cfg,
                scheduler_config=sched_cfg,
                demographic_weights=demo_w,
                master_seed=int(master_seed),
            )
            wm.start()
            st.session_state.action_feedback = f"Khởi động thành công {num_workers} workers mục tiêu {target_count} mẫu (Seed={master_seed})!"
            st.rerun()
        except Exception as exc:
            st.error(f"Lỗi khởi động Fleet: {exc}")

    if btn_pause_resume:
        try:
            if is_paused:
                wm.resume()
                st.session_state.action_feedback = "Đã tiếp tục chạy Worker Fleet."
            else:
                wm.pause()
                st.session_state.action_feedback = "Đã tạm dừng Worker Fleet an toàn."
            st.rerun()
        except Exception as exc:
            st.error(f"Lỗi thao tác: {exc}")

    if btn_stop:
        try:
            wm.stop(timeout=5.0)
            st.session_state.action_feedback = "Đã dừng hẳn Worker Fleet."
            st.rerun()
        except Exception as exc:
            st.error(f"Lỗi dừng Fleet: {exc}")

    if btn_reset:
        try:
            wm.reset()
            st.session_state.action_feedback = "Đã đặt lại dữ liệu, bộ đếm và tiến trình."
            st.rerun()
        except Exception as exc:
            st.error(f"Lỗi đặt lại: {exc}")

# Display feedback toast if any
if st.session_state.get("action_feedback"):
    st.toast(st.session_state.action_feedback)
    st.session_state.action_feedback = None


# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
header_html = (
    '<div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 8px;">'
    '<div>'
    '<h2 style="margin: 0; color: #f8fafc; font-weight: 700; letter-spacing: -0.02em;">'
    '⚡ Khảo sát Ý định sử dụng Shopee Smart Locker (Gen Z)'
    '</h2>'
    '<div style="color: #94a3b8; font-size: 0.82rem; margin-top: 2px;">'
    'Mô hình Nghiên cứu TAM & TPB • 38 Biến Quan Sát • 11 Thang Đo • Cronbach\'s Alpha Streaming • Xuất Chuẩn SPSS & Excel'
    '</div>'
    '</div>'
    '<div>'
    '<a href="https://tally.so/r/laqVzV" target="_blank" style="text-decoration: none;">'
    '<span class="status-pill pill-blue">🔗 MỞ BIỂU MẪU TALLY LAQVZV</span>'
    '</a>'
    '</div>'
    '</div>'
)
st.markdown(header_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper Formatters
# ---------------------------------------------------------------------------
def format_seconds(seconds: float) -> str:
    """Formats seconds into mm:ss or hh:mm:ss."""
    s = int(seconds)
    if s < 3600:
        return f"{s // 60:02d}:{s % 60:02d}"
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def get_worker_state_badge(state: WorkerState) -> Tuple[str, str]:
    """Returns (badge_class, label) for a worker state."""
    mapping = {
        WorkerState.IDLE: ("pill-slate", "IDLE"),
        WorkerState.SYNTHESIZING: ("pill-indigo", "SYNTH"),
        WorkerState.PACING: ("pill-amber", "PACING"),
        WorkerState.SUBMITTING: ("pill-sky", "SUBMIT"),
        WorkerState.SUCCESS: ("pill-emerald", "SUCCESS"),
        WorkerState.ERROR: ("pill-red", "ERROR"),
    }
    return mapping.get(state, ("pill-slate", str(state.value)))


# Support older Streamlit versions gracefully if fragment is not present
fragment_decorator = st.fragment(run_every="1s") if hasattr(st, "fragment") else (lambda fn: fn)


# ---------------------------------------------------------------------------
# 1. Top-Level Live KPI Metric Cards Fragment (1-Second Non-Blocking Update)
# ---------------------------------------------------------------------------
@fragment_decorator
def render_live_kpi_cards(manager: WorkerManager):
    """
    Sub-second reactive monitoring container for the 5 Top-Level KPI Cards.
    Updates in isolation without re-rendering tabs or causing chart flickers.
    """
    fleet_snap: WorkerFleetSnapshot = manager.get_snapshot()
    sched_snap = manager.scheduler.get_snapshot()
    stats_sum: StatsSummary = manager.get_stats_summary()
    target_alpha = float(st.session_state.get("alpha_threshold", 0.820))

    succ = fleet_snap.success_count
    target = fleet_snap.total_target
    err = fleet_snap.error_count
    submitted = fleet_snap.submitted_count
    spm = fleet_snap.submissions_per_minute
    elapsed = fleet_snap.elapsed_seconds
    active = fleet_snap.active_workers
    total_w = fleet_snap.total_workers

    pct_done = (succ / target * 100.0) if target > 0 else 0.0

    # Pill 1: Target Quota
    if fleet_snap.is_completed:
        pill_q = '<span class="status-pill pill-emerald">✓ 100% XONG</span>'
    elif fleet_snap.is_running:
        pill_q = f'<span class="status-pill pill-blue">TIẾN ĐỘ {pct_done:.0f}%</span>'
    elif fleet_snap.is_paused:
        pill_q = '<span class="status-pill pill-amber">❚❚ TẠM DỪNG</span>'
    else:
        pill_q = '<span class="status-pill pill-slate">○ SẴN SÀNG</span>'

    # Pill 2: Submissions
    err_rate = (err / submitted * 100.0) if submitted > 0 else 0.0
    if err == 0:
        pill_s = '<span class="status-pill pill-emerald">100% ỔN ĐỊNH</span>'
    else:
        pill_s = f'<span class="status-pill pill-red">{err_rate:.1f}% LỖI</span>'

    # Pill 3: Throughput
    if fleet_snap.is_running:
        pill_t = f'<span class="status-pill pill-sky">{active}/{total_w} HOẠT ĐỘNG</span>'
    else:
        pill_t = '<span class="status-pill pill-slate">STANDBY</span>'

    # Pill 4: Cronbach Alpha
    alpha_val = stats_sum.overall_alpha
    if alpha_val >= target_alpha and stats_sum.sample_count >= 10:
        pill_a = f'<span class="status-pill pill-emerald">ĐẠT CHUẨN NCKH (>= {target_alpha:.2f})</span>'
    elif alpha_val >= (target_alpha - 0.12) and stats_sum.sample_count >= 10:
        pill_a = '<span class="status-pill pill-amber">CHẤP NHẬN ĐƯỢC</span>'
    elif stats_sum.sample_count >= 10:
        pill_a = '<span class="status-pill pill-red">CHƯA ĐẠT CHUẨN</span>'
    else:
        pill_a = f'<span class="status-pill pill-slate">THU MẪU (N={stats_sum.sample_count}/10)</span>'

    # Pill 5: Scheduler Status
    s_state = sched_snap.state
    if s_state == SchedulerState.ACTIVE:
        pill_sc = '<span class="status-pill pill-emerald">ĐANG CHẠY</span>'
        sc_detail = f"Khung giờ: {sched_snap.active_windows_str}" if sched_snap.active_windows_str else "Toàn thời gian"
    elif s_state == SchedulerState.COUNTDOWN:
        pill_sc = f'<span class="status-pill pill-sky">T-{sched_snap.start_countdown_remaining:.0f}s</span>'
        sc_detail = "Đang đếm ngược giờ chạy"
    elif s_state == SchedulerState.COOLDOWN:
        pill_sc = f'<span class="status-pill pill-amber">HẠ NHIỆT {sched_snap.cooldown_remaining:.0f}s</span>'
        sc_detail = f"Đợt #{sched_snap.total_batches_completed + 1} hoàn tất"
    elif s_state == SchedulerState.WINDOW_WAIT:
        pill_sc = '<span class="status-pill pill-indigo">NGOÀI KHUNG GIỜ</span>'
        sc_detail = f"Chờ khung giờ: {sched_snap.active_windows_str}"
    else:
        pill_sc = '<span class="status-pill pill-slate">CHẾ ĐỘ TỰ DO</span>'
        sc_detail = "Không kích hoạt giới hạn giờ"

    # Render HTML KPI Cards with clean unindented tags
    kpi_html = (
        f'<div class="kpi-row">'
        f'<div class="kpi-card">'
        f'<div class="kpi-top"><span class="kpi-title">Mục tiêu Khảo sát</span>{pill_q}</div>'
        f'<div class="kpi-value">{succ} <span style="font-size:0.95rem;color:var(--text-muted);font-weight:400;">/ {target}</span></div>'
        f'<div class="kpi-subtext">Đã hoàn thành {pct_done:.1f}% chỉ tiêu</div>'
        f'</div>'
        f'<div class="kpi-card">'
        f'<div class="kpi-top"><span class="kpi-title">Kết quả Gửi form</span>{pill_s}</div>'
        f'<div class="kpi-value">{succ} <span style="font-size:0.85rem;color:var(--accent-emerald);">✓</span> {err} <span style="font-size:0.85rem;color:var(--accent-red);">✗</span></div>'
        f'<div class="kpi-subtext">Tổng số lượt gửi: {submitted}</div>'
        f'</div>'
        f'<div class="kpi-card">'
        f'<div class="kpi-top"><span class="kpi-title">Tốc độ & Thông lượng</span>{pill_t}</div>'
        f'<div class="kpi-value">{spm:.1f} <span style="font-size:0.85rem;font-weight:500;color:var(--text-muted);">mẫu/phút</span></div>'
        f'<div class="kpi-subtext">Thời gian chạy: {format_seconds(elapsed)}</div>'
        f'</div>'
        f'<div class="kpi-card">'
        f'<div class="kpi-top"><span class="kpi-title">Độ tin cậy Cronbach</span>{pill_a}</div>'
        f'<div class="kpi-value">α = {alpha_val:.3f}</div>'
        f'<div class="kpi-subtext">38 câu Likert • Chuẩn NCKH {target_alpha:.2f}</div>'
        f'</div>'
        f'<div class="kpi-card">'
        f'<div class="kpi-top"><span class="kpi-title">Tự động hóa Scheduler</span>{pill_sc}</div>'
        f'<div class="kpi-value" style="font-size:1.35rem;">{s_state.value}</div>'
        f'<div class="kpi-subtext">{sc_detail}</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(kpi_html, unsafe_allow_html=True)


# Render Top-Level KPI Metric Cards
render_live_kpi_cards(wm)


# ---------------------------------------------------------------------------
# 2. Main Navigation Tabs (At Page Root - Zero Tab Jumping or Re-renders)
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Giám Sát Đa Luồng (Worker Fleet)",
    "📊 Phân Tích Thống Kê & Likert TAM/TPB",
    "💾 Xem Dữ Liệu & Xuất File (SPSS/Excel)",
    "⚙️ Tự Động Hóa & Nhật Ký Hệ Thống",
])


# ---------------------------------------------------------------------------
# TAB 1: Worker Fleet Live Monitor Grid (Isolated 1s Telemetry Fragment)
# ---------------------------------------------------------------------------
with tab1:
    @fragment_decorator
    def render_fleet_grid_live(manager: WorkerManager):
        fleet_snap: WorkerFleetSnapshot = manager.get_snapshot()

        st.markdown(f"#### 🤖 Trạng Thái Hoạt Động {fleet_snap.total_workers} Worker Threads")

        # Render Worker Cards Grid without any leading indentation
        cards = ['<div class="worker-grid">']
        for w in fleet_snap.workers:
            badge_class, badge_lbl = get_worker_state_badge(w.state)
            if w.latency_ms is not None:
                if w.latency_ms < 250:
                    lat_pill = f'<span class="lat-pill lat-fast">{w.latency_ms:.0f}ms</span>'
                elif w.latency_ms < 800:
                    lat_pill = f'<span class="lat-pill lat-med">{w.latency_ms:.0f}ms</span>'
                else:
                    lat_pill = f'<span class="lat-pill lat-slow">{w.latency_ms:.0f}ms</span>'
            else:
                lat_pill = '<span class="lat-pill lat-na">—</span>'
            err_box = f'<div class="worker-error-box">Lỗi: {w.last_error[:42]}</div>' if w.last_error else ""

            card = (
                f'<div class="worker-card">'
                f'<div class="worker-header">'
                f'<span class="worker-name">Worker #{w.worker_id:02d}</span>'
                f'<span class="status-pill {badge_class}">{badge_lbl}</span>'
                f'</div>'
                f'<div class="worker-task" title="{w.current_task}">{w.current_task}</div>'
                f'<div class="worker-stats">'
                f'<span>Độ trễ: {lat_pill}</span>'
                f'<span>Thành công: <b>{w.success_count}</b> / {w.submissions_count}</span>'
                f'</div>'
                f'{err_box}'
                f'</div>'
            )
            cards.append(card)
        cards.append('</div>')
        st.markdown("".join(cards), unsafe_allow_html=True)

        st.markdown("---")

        # Quick summary table for fleet (PyArrow-safe with float nulls and NumberColumn config)
        if fleet_snap.workers:
            worker_rows = [
                {
                    "Worker ID": f"Worker #{w.worker_id:02d}",
                    "Trạng thái": w.state.value,
                    "Tác vụ hiện tại": w.current_task,
                    "Độ trễ phản hồi (ms)": float(round(w.latency_ms, 1)) if w.latency_ms is not None else None,
                    "Số lượt thành công": w.success_count,
                    "Số lượt lỗi": w.error_count,
                    "Lần cập nhật cuối": dt.datetime.fromtimestamp(w.last_updated).strftime("%H:%M:%S") if w.last_updated else "—",
                }
                for w in fleet_snap.workers
            ]
            st.dataframe(
                pd.DataFrame(worker_rows),
                column_config={
                    "Độ trễ phản hồi (ms)": st.column_config.NumberColumn(
                        "Độ trễ phản hồi (ms)",
                        format="%.1f ms",
                    ),
                },
                width="stretch",
                hide_index=True,
            )

    render_fleet_grid_live(wm)


# ---------------------------------------------------------------------------
# TAB 2: Psychometrics & Demographics (Stable, Zero-Flicker Plotly Charts)
# ---------------------------------------------------------------------------
with tab2:
    active_threshold = float(st.session_state.get("alpha_threshold", 0.820))
    active_benchmarks = st.session_state.get("custom_constructs", DEFAULT_CONSTRUCT_BENCHMARKS)

    col_t2_head, col_t2_btn = st.columns([3.5, 1.5])
    with col_t2_head:
        st.markdown("#### 📈 Độ Tin Cậy Nội Tại & Phân Tích Thang Đo TAM & TPB")
        st.caption(rf"38 Biến Quan Sát • 11 Thang Đo Khái Niệm • Chuẩn Cronbach's Alpha $\ge {active_threshold:.3f}$")
    with col_t2_btn:
        if st.button("🔄 Cập Nhật Biểu Đồ", key="btn_refresh_charts", width="stretch"):
            st.rerun()

    stats_sum: StatsSummary = wm.get_stats_summary()
    history_df: pd.DataFrame = wm.get_history_df()

    if stats_sum.sample_count == 0:
        empty_t2_html = (
            '<div style="background: rgba(30, 41, 59, 0.7); border: 1px dashed #334155; border-radius: 8px; padding: 24px; text-align: center; margin: 20px 0;">'
            '<div style="font-size: 1.8rem; margin-bottom: 8px;">📊</div>'
            '<div style="color: #f8fafc; font-weight: 600; font-size: 1rem;">Chưa có dữ liệu khảo sát trong phiên làm việc</div>'
            '<div style="color: #94a3b8; font-size: 0.85rem; margin-top: 4px;">'
            'Nhấn <b>▶ Chạy</b> ở thanh bên trái để khởi động các worker threads. Hệ thống sẽ streaming tính toán chỉ số Cronbach\'s Alpha, vẽ biểu đồ Radar TAM/TPB và cơ cấu nhân khẩu học Gen Z ngay khi có mẫu nộp!'
            '</div>'
            '</div>'
        )
        st.markdown(empty_t2_html, unsafe_allow_html=True)
    else:
        # Row 1: Cronbach Alpha Radial Gauge & Construct Mini Bars
        col_g1, col_g2 = st.columns([1, 1.25])
        with col_g1:
            st.markdown(rf"##### Hệ Số Cronbach's Alpha Toàn Mô Hình (Chuẩn $\ge {active_threshold:.3f}$)")
            fig_gauge = create_cronbach_alpha_gauge(
                overall_alpha=stats_sum.overall_alpha,
                sample_count=stats_sum.sample_count,
                target_threshold=active_threshold,
                height=260,
            )
            st.plotly_chart(fig_gauge, width="stretch", config=PLOTLY_CONFIG, theme=None)

        with col_g2:
            st.markdown(rf"##### Hệ Số Cronbach's Alpha Từng Thang Đo (Ngưỡng {active_threshold:.3f})")
            fig_alpha_bars = create_construct_alpha_bars(
                stats_summary=stats_sum,
                target_threshold=active_threshold,
                height=260,
            )
            st.plotly_chart(fig_alpha_bars, width="stretch", config=PLOTLY_CONFIG, theme=None)

        st.markdown("---")

        # Row 2: Radar Chart & Accessible Horizontal Bar Chart with 95% CI
        col_c1, col_c2 = st.columns([1, 1.25])
        with col_c1:
            st.markdown("##### Biểu Đồ Radar 11 Thang Đo TAM & TPB (So Với Điểm Kỳ Vọng Tùy Chỉnh)")
            fig_radar = create_psychometric_radar_chart(
                stats_summary=stats_sum,
                benchmark_targets=active_benchmarks,
                height=420,
            )
            st.plotly_chart(fig_radar, width="stretch", config=PLOTLY_CONFIG, theme=None)

        with col_c2:
            st.markdown(r"##### Biểu Đồ Thanh Ngang Kèm Khoảng Tin Cậy 95% ($\pm 1.96 \times SEM$)")
            fig_bar = create_accessible_construct_bar_chart(stats_summary=stats_sum, height=420)
            st.plotly_chart(fig_bar, width="stretch", config=PLOTLY_CONFIG, theme=None)

        st.markdown("---")

        # Row 3: Demographics Visualizations (Gender, Location, Age, Shopping Frequency)
        st.markdown("#### 👥 Cơ Cấu Mẫu Nhân Khẩu Học Thế Hệ Z")
        d_col1, d_col2, d_col3, d_col4 = st.columns(4)

        with d_col1:
            fig_gender = create_gender_donut_chart(history_df, height=270)
            st.plotly_chart(fig_gender, width="stretch", config=PLOTLY_CONFIG, theme=None)

        with d_col2:
            fig_loc = create_location_chart(history_df, height=270)
            st.plotly_chart(fig_loc, width="stretch", config=PLOTLY_CONFIG, theme=None)

        with d_col3:
            fig_age = create_age_distribution_chart(history_df, height=270)
            st.plotly_chart(fig_age, width="stretch", config=PLOTLY_CONFIG, theme=None)

        with d_col4:
            fig_freq = create_frequency_chart(history_df, height=270)
            st.plotly_chart(fig_freq, width="stretch", config=PLOTLY_CONFIG, theme=None)

        with st.expander("💼 Xem Chi Tiết Cơ Cấu Việc Làm & Nghề Nghiệp", expanded=False):
            fig_emp = create_employment_donut_chart(history_df, height=300)
            st.plotly_chart(fig_emp, width="stretch", config=PLOTLY_CONFIG, theme=None)


# ---------------------------------------------------------------------------
# TAB 3: Data Inspector & SPSS Export (Rock-Solid Download Streams)
# ---------------------------------------------------------------------------
with tab3:
    col_t3_head, col_t3_btn = st.columns([3.5, 1.5])
    with col_t3_head:
        st.markdown("#### 💾 Dữ Liệu Khảo Sát & Xuất File Phân Tích (SPSS, SmartPLS, AMOS)")
        st.caption("Xuất tệp CSV số hóa thuần túy (Numeric Matrix) và sổ tay mã hóa 3-sheet Excel")
    with col_t3_btn:
        if st.button("🔄 Làm Mới Dữ Liệu", key="btn_refresh_data", width="stretch"):
            st.rerun()

    history_df: pd.DataFrame = wm.get_history_df()
    if history_df.empty:
        empty_t3_html = (
            '<div style="background: rgba(30, 41, 59, 0.7); border: 1px dashed #334155; border-radius: 8px; padding: 24px; text-align: center; margin: 20px 0;">'
            '<div style="font-size: 1.8rem; margin-bottom: 8px;">💾</div>'
            '<div style="color: #f8fafc; font-weight: 600; font-size: 1rem;">Chưa có bản ghi khảo sát nào để xuất dữ liệu</div>'
            '<div style="color: #94a3b8; font-size: 0.85rem; margin-top: 4px;">'
            'Nhấn <b>▶ Chạy</b> ở thanh bên trái để bắt đầu thu thập mẫu. File xuất chuẩn SPSS (Pure Numeric CSV) và 3-Sheet Excel Codebook sẽ sẵn sàng tải xuống ngay lập tức.'
            '</div>'
            '</div>'
        )
        st.markdown(empty_t3_html, unsafe_allow_html=True)
    else:
        # Generate SPSS pure numeric matrix
        numeric_matrix_df = ExportService.to_numeric_matrix(history_df)
        codebook_df = ExportService.get_variable_codebook()

        meta_t3_html = (
            f'<div style="display:flex;gap:20px;margin-bottom:14px;padding:10px 14px;background:#1e293b;border:1px solid #334155;border-radius:6px;font-size:0.85rem;color:var(--text-secondary);">'
            f'<span>Số lượng mẫu: <b style="color:var(--text-primary);">{len(numeric_matrix_df)}</b></span>'
            f'<span>Số lượng biến số: <b style="color:var(--text-primary);">{len(numeric_matrix_df.columns)}</b> cột</span>'
            f'<span>Thang đo Likert: <b style="color:var(--accent-emerald);">1 – 6 (Đã đảo chiều PC)</b></span>'
            f'<span>Mã hóa nhân khẩu: <b style="color:var(--accent-blue);">Số nguyên thuần túy (SPSS-ready)</b></span>'
            f'</div>'
        )
        st.markdown(meta_t3_html, unsafe_allow_html=True)

        # Export Action Buttons
        c_exp1, c_exp2 = st.columns(2)

        with c_exp1:
            csv_bytes = ExportService.export_spss_csv(history_df)
            st.download_button(
                label="📥 Tải Tệp CSV Số Hóa (SPSS, SmartPLS, AMOS)",
                data=csv_bytes,
                file_name=f"shopee_genz_survey_spss_{len(numeric_matrix_df)}obs.csv",
                mime="text/csv",
                width="stretch",
            )

        with c_exp2:
            excel_bytes = ExportService.export_excel(history_df, stats_summary=wm.get_stats_summary())
            st.download_button(
                label="📥 Tải Sổ Tay Mã Hóa 3-Sheet Excel (.xlsx)",
                data=excel_bytes,
                file_name=f"shopee_genz_survey_codebook_{len(numeric_matrix_df)}obs.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )

        st.markdown("---")

        # Data Preview Accordions
        st.markdown(f"##### 🔍 Xem Trước Bảng Dữ Liệu Số Hóa ({min(50, len(numeric_matrix_df))}/{len(numeric_matrix_df)} Bản ghi đầu tiên)")
        st.dataframe(numeric_matrix_df.head(50), width="stretch")

        with st.expander("📖 Xem Sổ Tay Mã Hóa Biến Số (Variable Codebook)", expanded=False):
            st.dataframe(codebook_df, width="stretch")


# ---------------------------------------------------------------------------
# TAB 4: Automation & Audit Logs (Real-Time Background Scheduler Telemetry)
# ---------------------------------------------------------------------------
with tab4:
    col_t4_head, col_t4_btn = st.columns([3.5, 1.5])
    with col_t4_head:
        st.markdown("#### ⚙️ Kiểm Tra Tự Động Hóa & Nhật Ký Hệ Thống")
        st.caption("Theo dõi chi tiết trạng thái bộ lập lịch (Scheduler), thời gian giải lao và nhật ký gửi form")
    with col_t4_btn:
        if st.button("🔄 Làm Mới Nhật Ký", key="btn_refresh_audit", width="stretch"):
            st.rerun()

    @fragment_decorator
    def render_audit_telemetry_live(manager: WorkerManager):
        fleet_snap = manager.get_snapshot()
        sched_snap = manager.scheduler.get_snapshot()
        history_df = manager.get_history_df()

        col_a1, col_a2 = st.columns(2)

        with col_a1:
            st.markdown("##### ⏱️ Thông Số Lên Lịch (Background Scheduler)")
            sched_details = [
                {"Thuộc tính": "Trạng thái", "Giá trị": str(sched_snap.state.value)},
                {"Thuộc tính": "Đang hoạt động", "Giá trị": "Có" if sched_snap.is_active else "Không"},
                {"Thuộc tính": "Đếm ngược bắt đầu", "Giá trị": f"{sched_snap.start_countdown_remaining:.1f} giây" if sched_snap.start_countdown_remaining > 0 else "0 giây"},
                {"Thuộc tính": "Khung giờ quy định", "Giá trị": sched_snap.active_windows_str if sched_snap.active_windows_str else "Tất cả các giờ"},
                {"Thuộc tính": "Chờ mở khung giờ", "Giá trị": f"{sched_snap.window_wait_remaining:.1f} giây" if sched_snap.window_wait_remaining > 0 else "Đang trong khung giờ"},
                {"Thuộc tính": "Hạ nhiệt sau đợt (Cooldown)", "Giá trị": f"{sched_snap.cooldown_remaining:.1f} giây" if sched_snap.cooldown_remaining > 0 else "Sẵn sàng"},
                {"Thuộc tính": "Số mẫu đợt hiện tại", "Giá trị": str(sched_snap.current_batch_count)},
                {"Thuộc tính": "Số đợt đã hoàn thành", "Giá trị": str(sched_snap.total_batches_completed)},
                {"Thuộc tính": "Thông điệp", "Giá trị": str(sched_snap.message)},
            ]
            st.dataframe(pd.DataFrame(sched_details), width="stretch", hide_index=True)

        with col_a2:
            st.markdown("##### 🚀 Hiệu Năng & Độ Trễ Fleet")
            latencies = [w.latency_ms for w in fleet_snap.workers if w.latency_ms is not None]
            avg_lat = float(np.mean(latencies)) if latencies else 0.0
            max_lat = float(np.max(latencies)) if latencies else 0.0
            min_lat = float(np.min(latencies)) if latencies else 0.0

            perf_details = [
                {"Chỉ số": "Tổng số worker threads", "Giá trị": f"{fleet_snap.total_workers} threads"},
                {"Chỉ số": "Số workers đang hoạt động", "Giá trị": f"{fleet_snap.active_workers} threads"},
                {"Chỉ số": "Tốc độ gửi trung bình", "Giá trị": f"{fleet_snap.submissions_per_minute:.1f} lượt/phút"},
                {"Chỉ số": "Độ trễ trung bình", "Giá trị": f"{avg_lat:.1f} ms"},
                {"Chỉ số": "Độ trễ nhỏ nhất", "Giá trị": f"{min_lat:.1f} ms"},
                {"Chỉ số": "Độ trễ lớn nhất", "Giá trị": f"{max_lat:.1f} ms"},
                {"Chỉ số": "Thời gian hoạt động", "Giá trị": format_seconds(fleet_snap.elapsed_seconds)},
            ]
            st.dataframe(pd.DataFrame(perf_details), width="stretch", hide_index=True)

        st.markdown("---")

        # History telemetry log
        st.markdown("##### 📜 Nhật Ký Gửi Mẫu Gần Nhất")
        if history_df.empty:
            st.info("Chưa có bản ghi nào được ghi nhận.")
        else:
            recent_df = history_df.tail(20).copy()
            recent_display = pd.DataFrame({
                "Mã lượt gửi": recent_df["submission_id"].astype(str).str[:16],
                "Thời gian": recent_df["timestamp"].apply(lambda t: dt.datetime.fromtimestamp(t).strftime("%H:%M:%S")),
                "Worker ID": recent_df["worker_id"].apply(lambda wid: f"Worker #{wid:02d}"),
                "Độ trễ (ms)": recent_df["latency_ms"].round(1),
                "Tuổi": recent_df["age"],
                "Chế độ": recent_df["dry_run"].apply(lambda d: "Dry-Run" if d else "Live"),
            })
            st.dataframe(recent_display, width="stretch", hide_index=True)

    render_audit_telemetry_live(wm)
