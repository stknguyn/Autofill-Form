"""
src/ui_components.py - Reusable Plotly Psychometrics Visualizations & UI Components.
Milestone M4: Streamlit Dashboard & SPSS Export.
Adheres to ui-ux-pro-max Swiss Minimalism & Data-Dense specifications.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.stats_analyzer import ConstructStats, StatsSummary
from src.tally_schema import TallySchema

# ---------------------------------------------------------------------------
# Design System Tokens (ui-ux-pro-max Slate & High Contrast)
# ---------------------------------------------------------------------------
SLATE_THEME: Dict[str, str] = {
    "bg_canvas": "#0f172a",          # Slate 900
    "bg_card": "#1e293b",            # Slate 800
    "border": "#334155",             # Slate 700
    "text_primary": "#f8fafc",       # Slate 50
    "text_secondary": "#cbd5e1",     # Slate 300
    "text_muted": "#94a3b8",         # Slate 400
    "primary_blue": "#3b82f6",       # Blue 500
    "primary_blue_light": "#60a5fa", # Blue 400
    "primary_fill": "rgba(59, 130, 246, 0.22)",
    "benchmark_green": "#10b981",    # Emerald 500
    "benchmark_fill": "rgba(16, 185, 129, 0.25)",
    "warning_amber": "#f59e0b",      # Amber 500
    "danger_rose": "#ef4444",        # Red 500
    "font_family": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
}

PLOTLY_CONFIG: Dict[str, Any] = {
    "responsive": True,
    "displayModeBar": False,
    "displaylogo": False,
    "scrollZoom": False,
}

# Theoretical Model Benchmarks from persona_engine CONSTRUCT_CONFIGS
DEFAULT_BENCHMARKS: Dict[str, float] = {
    "PU": 4.65,
    "PEOU": 4.70,
    "PES": 4.75,
    "PC": 4.65,  # Reversed target: 7.0 - 2.35
    "PI": 4.25,
    "SE": 4.75,
    "LC": 4.70,
    "ATT": 4.65,
    "SN": 4.30,
    "PBC": 4.70,
    "ITU": 4.65,
}

CONSTRUCT_METADATA: Dict[str, Dict[str, str]] = {
    "PU": {"short": "PU", "vi": "Hữu ích", "full": "PU: Nhận thức hữu ích"},
    "PEOU": {"short": "PEOU", "vi": "Dễ sử dụng", "full": "PEOU: Nhận thức dễ sử dụng"},
    "PES": {"short": "PES", "vi": "Bền vững môi trường", "full": "PES: Bền vững môi trường"},
    "PC": {"short": "PC (Đảo)", "vi": "Chi phí thấp (Đảo)", "full": "PC: Chi phí (Đảo chiều)"},
    "PI": {"short": "PI", "vi": "Ảnh hưởng xã hội", "full": "PI: Ảnh hưởng xã hội"},
    "SE": {"short": "SE", "vi": "Tự tin năng lực", "full": "SE: Năng lực tự thân"},
    "LC": {"short": "LC", "vi": "Thuận tiện vị trí", "full": "LC: Thuận tiện vị trí"},
    "ATT": {"short": "ATT", "vi": "Thái độ", "full": "ATT: Thái độ đối với dịch vụ"},
    "SN": {"short": "SN", "vi": "Chuẩn chủ quan", "full": "SN: Chuẩn chủ quan"},
    "PBC": {"short": "PBC", "vi": "Kiểm soát hành vi", "full": "PBC: Kiểm soát hành vi"},
    "ITU": {"short": "ITU", "vi": "Ý định sử dụng", "full": "ITU: Ý định sử dụng"},
}


# ---------------------------------------------------------------------------
# Styling Helpers & Empty State Handler
# ---------------------------------------------------------------------------
def apply_slate_theme(
    fig: go.Figure,
    title: Optional[str] = None,
    height: Optional[int] = None,
    show_legend: bool = True,
) -> go.Figure:
    """Applies unified ui-ux-pro-max slate styling tokens to any Plotly figure."""
    layout_update: Dict[str, Any] = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(15, 23, 42, 0.4)",
        "font": dict(
            family=SLATE_THEME["font_family"],
            color=SLATE_THEME["text_primary"],
            size=12,
        ),
        "showlegend": show_legend,
    }
    if title:
        layout_update["title"] = dict(
            text=f"<b>{title}</b>",
            font=dict(size=14, color=SLATE_THEME["text_primary"]),
            x=0.02,
            y=0.96,
        )
    if height:
        layout_update["height"] = height
    fig.update_layout(**layout_update)
    return fig


def create_empty_state_figure(
    message: str = "Chưa có dữ liệu khảo sát",
    subtext: str = "Vui lòng khởi động Worker Fleet để bắt đầu thu thập dữ liệu.",
    height: int = 320,
) -> go.Figure:
    """Renders a graceful placeholder figure when no samples are available."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"<b>{message}</b><br><span style='font-size:11px;color:{SLATE_THEME['text_muted']}'>{subtext}</span>",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            size=13,
            color=SLATE_THEME["text_secondary"],
            family=SLATE_THEME["font_family"],
        ),
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.3)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=height,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# 11-Construct TAM & TPB Radar Chart
# ---------------------------------------------------------------------------
def create_psychometric_radar_chart(
    stats_summary: Optional[StatsSummary],
    benchmark_targets: Optional[Dict[str, float]] = None,
    height: int = 420,
) -> go.Figure:
    """
    Renders the 11-construct TAM & TPB Radar Chart with Benchmark comparison curve.
    Closed polar polygon, fixed 1.0 to 6.0 scale, PC reverse-coded.
    """
    benchmarks = benchmark_targets or DEFAULT_BENCHMARKS
    construct_order = list(DEFAULT_BENCHMARKS.keys())
    closed_theta = [CONSTRUCT_METADATA[c]["short"] for c in construct_order]
    closed_theta.append(closed_theta[0])  # Close the polygon

    # Benchmark trace coordinates
    bench_r = [benchmarks.get(c, 4.5) for c in construct_order]
    bench_r.append(bench_r[0])

    fig = go.Figure()

    # 1. Benchmark Trace (Dashed Emerald)
    fig.add_trace(go.Scatterpolar(
        r=bench_r,
        theta=closed_theta,
        name="Mô hình chuẩn (Benchmark)",
        line=dict(color=SLATE_THEME["benchmark_green"], width=2, dash="dash"),
        mode="lines",
        hovertemplate="<b>%{theta}</b> (Benchmark): <b>%{r:.2f}</b> / 6.0<extra></extra>",
    ))

    # 2. Observed Sample Trace (Primary Blue with semi-transparent fill)
    if stats_summary and stats_summary.sample_count > 0:
        obs_r = []
        hover_texts = []
        for c in construct_order:
            cs = stats_summary.construct_stats.get(c)
            val = cs.mean if cs else 0.0
            obs_r.append(val)
            if c == "PC" and cs and cs.mean_raw is not None:
                hover_texts.append(
                    f"<b>{CONSTRUCT_METADATA[c]['full']}</b><br>"
                    f"Điểm đảo (PC_rev): <b>{val:.2f}</b> / 6.0<br>"
                    f"Điểm gốc (PC_raw): <b>{cs.mean_raw:.2f}</b> / 6.0"
                )
            else:
                hover_texts.append(
                    f"<b>{CONSTRUCT_METADATA[c]['full']}</b><br>"
                    f"Điểm trung bình: <b>{val:.2f}</b> / 6.0"
                )
        obs_r.append(obs_r[0])
        hover_texts.append(hover_texts[0])

        fig.add_trace(go.Scatterpolar(
            r=obs_r,
            theta=closed_theta,
            name=f"Mẫu thực nghiệm (N={stats_summary.sample_count})",
            line=dict(color=SLATE_THEME["primary_blue"], width=2.5),
            fill="toself",
            fillcolor=SLATE_THEME["primary_fill"],
            mode="lines+markers",
            marker=dict(size=6, color=SLATE_THEME["primary_blue_light"], symbol="circle"),
            customdata=hover_texts,
            hovertemplate="%{customdata}<extra></extra>",
        ))

    # Polar layout configuration
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[1.0, 6.0],
                tickvals=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                gridcolor=SLATE_THEME["border"],
                linecolor=SLATE_THEME["border"],
                tickfont=dict(color=SLATE_THEME["text_muted"], size=10),
            ),
            angularaxis=dict(
                gridcolor=SLATE_THEME["border"],
                linecolor=SLATE_THEME["border"],
                tickfont=dict(
                    color=SLATE_THEME["text_primary"],
                    size=11,
                    family=SLATE_THEME["font_family"],
                ),
            ),
            bgcolor="rgba(15, 23, 42, 0.4)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.06,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color=SLATE_THEME["text_secondary"]),
        ),
        margin=dict(l=45, r=45, t=55, b=45),
        height=height,
    )
    return apply_slate_theme(fig, show_legend=True)


# Alias for dispatch compatibility
create_radar_chart = create_psychometric_radar_chart


# ---------------------------------------------------------------------------
# Dual-Mode Accessible Horizontal Bar Chart with Error Bars
# ---------------------------------------------------------------------------
def create_accessible_construct_bar_chart(
    stats_summary: Optional[StatsSummary],
    benchmark_targets: Optional[Dict[str, float]] = None,
    height: int = 420,
) -> go.Figure:
    """
    Renders the Accessible Horizontal Bar Chart with 95% Confidence Intervals (+-1.96*SEM).
    Features direct value labels, 3.5 neutral midpoint line, and benchmark markers.
    """
    if stats_summary is None or stats_summary.sample_count == 0:
        return create_empty_state_figure(
            message="Chưa có dữ liệu phân tích thang đo",
            subtext="Biểu đồ thanh ngang với khoảng tin cậy 95% sẽ hiển thị khi có dữ liệu mẫu.",
            height=height,
        )

    benchmarks = benchmark_targets or DEFAULT_BENCHMARKS
    construct_order = list(DEFAULT_BENCHMARKS.keys())
    labels = [CONSTRUCT_METADATA[c]["vi"] for c in construct_order]

    n = stats_summary.sample_count
    means = []
    ci95_margins = []
    bench_values = []
    text_labels = []

    for c in construct_order:
        cs = stats_summary.construct_stats.get(c)
        m = cs.mean if cs else 0.0
        s = cs.std if cs else 0.0
        sem = (s / math.sqrt(n)) if n >= 2 else 0.0
        ci = 1.96 * sem

        means.append(m)
        ci95_margins.append(ci)
        bench_values.append(benchmarks.get(c, 4.5))
        text_labels.append(f"{m:.2f} ± {ci:.2f}" if n >= 2 else f"{m:.2f}")

    fig = go.Figure()

    # 1. Horizontal Bars with Error Bars (95% CI)
    fig.add_trace(go.Bar(
        y=labels,
        x=means,
        orientation="h",
        name=f"Điểm TB thực nghiệm (N={n})",
        marker=dict(
            color=SLATE_THEME["primary_blue"],
            line=dict(color=SLATE_THEME["primary_blue_light"], width=1),
        ),
        error_x=dict(
            type="data",
            array=ci95_margins,
            visible=(n >= 2),
            color=SLATE_THEME["text_secondary"],
            thickness=2,
            width=5,
        ),
        text=text_labels,
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="#ffffff", size=11, family=SLATE_THEME["font_family"]),
        hovertemplate="<b>%{y}</b><br>Trung bình: <b>%{x:.2f}</b><br>95%% CI: [-%{error_x.array:.2f}, +%{error_x.array:.2f}]<extra></extra>",
    ))

    # 2. Benchmark Target Diamond Markers
    fig.add_trace(go.Scatter(
        y=labels,
        x=bench_values,
        mode="markers",
        name="Mục tiêu chuẩn (Benchmark)",
        marker=dict(
            symbol="diamond",
            size=9,
            color=SLATE_THEME["benchmark_green"],
            line=dict(color="#047857", width=1.5),
        ),
        hovertemplate="<b>%{y}</b> (Benchmark): <b>%{x:.2f}</b><extra></extra>",
    ))

    # 3. Reference Line at Neutral Scale Midpoint (3.5)
    fig.add_vline(
        x=3.5,
        line_dash="dash",
        line_color=SLATE_THEME["warning_amber"],
        line_width=1.5,
        annotation_text="Trung hòa (3.5)",
        annotation_position="bottom right",
        annotation_font=dict(color=SLATE_THEME["warning_amber"], size=10),
    )

    fig.update_layout(
        xaxis=dict(
            title="Thang đo Likert (1.0 = Hoàn toàn không đồng ý, 6.0 = Hoàn toàn đồng ý)",
            range=[1.0, 6.2],
            tickvals=[1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 6.0],
            gridcolor=SLATE_THEME["border"],
            tickfont=dict(color=SLATE_THEME["text_muted"], size=10),
        ),
        yaxis=dict(
            autorange="reversed",  # Keeps PU at the top down to ITU
            gridcolor="rgba(30, 41, 59, 0.4)",
            tickfont=dict(color=SLATE_THEME["text_primary"], size=11),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color=SLATE_THEME["text_secondary"]),
        ),
        margin=dict(l=140, r=25, t=40, b=45),
        height=height,
    )
    return apply_slate_theme(fig, show_legend=True)


# Alias for dispatch compatibility
create_horizontal_bar_chart = create_accessible_construct_bar_chart


# ---------------------------------------------------------------------------
# Cronbach's Alpha Real-time Gauges & Mini Indicator Bars
# ---------------------------------------------------------------------------
def create_cronbach_alpha_gauge(
    overall_alpha: float,
    sample_count: int = 10,
    target_threshold: float = 0.82,
    height: int = 260,
) -> go.Figure:
    """
    Renders real-time radial gauge for Overall Model Cronbach's Alpha.
    Displays threshold delta, benchmark line at 0.820, and reliability color zones.
    """
    val = max(0.0, min(1.0, overall_alpha)) if sample_count >= 3 else 0.0

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=val,
        domain={"x": [0, 1], "y": [0, 1]},
        delta={
            "reference": target_threshold,
            "increasing": {"color": SLATE_THEME["benchmark_green"]},
            "decreasing": {"color": SLATE_THEME["danger_rose"]},
            "valueformat": "+.3f",
        },
        number={
            "valueformat": ".3f",
            "font": {"size": 34, "color": SLATE_THEME["text_primary"], "family": SLATE_THEME["font_family"]},
        },
        gauge={
            "axis": {
                "range": [0.0, 1.0],
                "tickvals": [0.0, 0.6, 0.7, 0.82, 1.0],
                "tickwidth": 1,
                "tickcolor": SLATE_THEME["text_muted"],
            },
            "bar": {"color": SLATE_THEME["primary_blue"], "thickness": 0.28},
            "bgcolor": "rgba(15, 23, 42, 0.6)",
            "borderwidth": 1,
            "bordercolor": SLATE_THEME["border"],
            "steps": [
                {"range": [0.00, 0.70], "color": "rgba(239, 68, 68, 0.22)"},
                {"range": [0.70, 0.82], "color": "rgba(245, 158, 11, 0.22)"},
                {"range": [0.82, 1.00], "color": "rgba(16, 185, 129, 0.28)"},
            ],
            "threshold": {
                "line": {"color": SLATE_THEME["benchmark_green"], "width": 4},
                "thickness": 0.75,
                "value": target_threshold,
            },
        },
    ))

    status_tag = (
        "ĐẠT CHUẨN NCKH (>= 0.820)"
        if (val >= target_threshold and sample_count >= 10)
        else ("ĐANG THU THẬP MẪU" if sample_count < 10 else "CHƯA ĐẠT NGƯỠNG")
    )
    status_color = SLATE_THEME["benchmark_green"] if val >= target_threshold else SLATE_THEME["warning_amber"]

    fig.add_annotation(
        text=f"<span style='color:{status_color};font-weight:600;font-size:11px'>● {status_tag}</span>",
        x=0.5,
        y=-0.12,
        showarrow=False,
        xref="paper",
        yref="paper",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=25, r=25, t=35, b=25),
        height=height,
    )
    return fig


# Alias for dispatch compatibility
create_alpha_gauge = create_cronbach_alpha_gauge


def create_construct_alpha_bars(
    stats_summary: Optional[StatsSummary],
    target_threshold: float = 0.82,
    height: int = 340,
) -> go.Figure:
    """
    Renders mini horizontal indicator bars showing construct-level Cronbach's Alpha compliance.
    Color codes: Green (>= 0.82), Amber (0.70 - 0.82), Red (< 0.70).
    """
    if stats_summary is None or stats_summary.sample_count == 0:
        return create_empty_state_figure(
            message="Chưa có dữ liệu hệ số Alpha",
            subtext="Độ tin cậy nội tại từng thang đo sẽ hiển thị tại đây.",
            height=height,
        )

    construct_order = list(DEFAULT_BENCHMARKS.keys())
    alphas = []
    bar_colors = []
    text_labels = []

    for c in construct_order:
        cs = stats_summary.construct_stats.get(c)
        a = cs.alpha if cs else 0.0
        alphas.append(a)
        if a >= target_threshold:
            bar_colors.append(SLATE_THEME["benchmark_green"])
        elif a >= 0.70:
            bar_colors.append(SLATE_THEME["warning_amber"])
        else:
            bar_colors.append(SLATE_THEME["danger_rose"])
        text_labels.append(f"{a:.3f}")

    labels = [CONSTRUCT_METADATA[c]["short"] for c in construct_order]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels,
        x=alphas,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color=SLATE_THEME["bg_canvas"], width=1)),
        text=text_labels,
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(color="#ffffff", size=11, family=SLATE_THEME["font_family"]),
        hovertemplate="<b>Thang đo %{y}</b><br>Cronbach's Alpha: <b>%{x:.3f}</b> (Chuẩn >= 0.820)<extra></extra>",
    ))

    # Reference Threshold Line at 0.820
    fig.add_vline(
        x=target_threshold,
        line_dash="dash",
        line_color=SLATE_THEME["benchmark_green"],
        line_width=2,
        annotation_text="Chuẩn NCKH (0.820)",
        annotation_position="top left",
        annotation_font=dict(color=SLATE_THEME["benchmark_green"], size=10),
    )

    fig.update_layout(
        xaxis=dict(
            range=[0.5, 1.0],
            tickvals=[0.5, 0.6, 0.7, 0.82, 0.9, 1.0],
            gridcolor=SLATE_THEME["border"],
            tickfont=dict(color=SLATE_THEME["text_muted"], size=10),
        ),
        yaxis=dict(
            autorange="reversed",
            gridcolor="rgba(30, 41, 59, 0.4)",
            tickfont=dict(color=SLATE_THEME["text_primary"], size=11),
        ),
        margin=dict(l=65, r=25, t=30, b=40),
        height=height,
    )
    return apply_slate_theme(fig, show_legend=False)


# ---------------------------------------------------------------------------
# Demographic Distribution Charts
# ---------------------------------------------------------------------------
def create_gender_donut_chart(
    history_df: Optional[pd.DataFrame],
    height: int = 280,
) -> go.Figure:
    """
    Renders Gender distribution Donut Chart with center total count.
    Maps option UUIDs or labels to Nam (#3b82f6), Nữ (#ec4899), Khác (#8b5cf6).
    """
    if history_df is None or history_df.empty or "gender" not in history_df.columns:
        return create_empty_state_figure("Chưa có dữ liệu giới tính", height=height)

    gender_series = history_df["gender"].dropna().astype(str)
    if gender_series.empty:
        return create_empty_state_figure("Chưa có dữ liệu giới tính", height=height)

    # UUID, integer, and label mapping dictionary
    gender_map = {
        "7a33c383-e672-4bdc-97e1-985ba902247d": "Nam",
        "428cad0c-6d8e-4f12-bbec-8c593b428fc4": "Nữ",
        "200936e3-785a-4d5b-8c00-226b5ca1735a": "Khác",
        "Nam": "Nam",
        "Nữ": "Nữ",
        "Khác": "Khác",
        "1": "Nam", "2": "Nữ", "3": "Khác",
        1: "Nam", 2: "Nữ", 3: "Khác",
    }
    mapped_series = gender_series.map(lambda x: gender_map.get(x, gender_map.get(int(x) if str(x).isdigit() else x, "Khác")))
    counts = mapped_series.value_counts()

    color_map = {
        "Nam": SLATE_THEME["primary_blue"],
        "Nữ": "#ec4899",
        "Khác": "#8b5cf6",
    }
    labels = list(counts.index)
    values = [int(v) for v in counts.values]
    colors = [color_map.get(lbl, "#64748b") for lbl in labels]
    total_n = sum(values)

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color=SLATE_THEME["bg_canvas"], width=2)),
        textinfo="percent+label",
        textfont=dict(color="#ffffff", size=11, family=SLATE_THEME["font_family"]),
        hovertemplate="<b>%{label}</b><br>Số lượng: <b>%{value}</b> (%{percent:.1%})<extra></extra>",
        showlegend=False,
    )])

    fig.add_annotation(
        text=f"<b>{total_n}</b><br><span style='font-size:10px;color:{SLATE_THEME['text_muted']}'>ĐÁP VIÊN</span>",
        x=0.5,
        y=0.5,
        font=dict(size=16, color=SLATE_THEME["text_primary"], family=SLATE_THEME["font_family"]),
        showarrow=False,
    )

    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=height)
    return apply_slate_theme(fig, title="Phân bổ Giới tính", show_legend=False)


def create_location_chart(
    history_df: Optional[pd.DataFrame],
    height: int = 280,
) -> go.Figure:
    """
    Renders Location distribution Donut Chart for 3 key urban markets:
    TP. Hồ Chí Minh (#3b82f6), Hà Nội (#f59e0b), Đà Nẵng (#10b981), and Tỉnh khác (#64748b).
    """
    if history_df is None or history_df.empty or "location" not in history_df.columns:
        return create_empty_state_figure("Chưa có dữ liệu địa bàn", height=height)

    loc_series = history_df["location"].dropna().astype(str)
    if loc_series.empty:
        return create_empty_state_figure("Chưa có dữ liệu địa bàn", height=height)

    loc_map = {
        "78ef19e6-8b27-4801-ae86-30118c8f333b": "TP. Hồ Chí Minh",
        "a1561c8f-20a1-4571-b447-a23943e98f0c": "Hà Nội",
        "4d4962c3-86f5-4d5b-bc59-536d23df2db5": "Đà Nẵng",
        "5e069c99-7562-4c9d-88d8-af4fc3a197d9": "Tỉnh khác",
        "TP. Hồ Chí Minh": "TP. Hồ Chí Minh",
        "Hà Nội": "Hà Nội",
        "Đà Nẵng": "Đà Nẵng",
        "Tỉnh khác": "Tỉnh khác",
        "1": "TP. Hồ Chí Minh", "2": "Hà Nội", "3": "Đà Nẵng", "4": "Tỉnh khác",
        1: "TP. Hồ Chí Minh", 2: "Hà Nội", 3: "Đà Nẵng", 4: "Tỉnh khác",
    }
    mapped = loc_series.map(lambda x: loc_map.get(x, loc_map.get(int(x) if str(x).isdigit() else x, "Tỉnh khác")))
    counts = mapped.value_counts()

    color_map = {
        "TP. Hồ Chí Minh": SLATE_THEME["primary_blue"],
        "Hà Nội": SLATE_THEME["warning_amber"],
        "Đà Nẵng": SLATE_THEME["benchmark_green"],
        "Tỉnh khác": "#64748b",
    }
    labels = list(counts.index)
    values = [int(v) for v in counts.values]
    colors = [color_map.get(lbl, "#64748b") for lbl in labels]
    total_n = sum(values)

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color=SLATE_THEME["bg_canvas"], width=2)),
        textinfo="percent+label",
        textfont=dict(color="#ffffff", size=10, family=SLATE_THEME["font_family"]),
        hovertemplate="<b>%{label}</b><br>Số lượng: <b>%{value}</b> (%{percent:.1%})<extra></extra>",
        showlegend=False,
    )])

    fig.add_annotation(
        text=f"<b>{total_n}</b><br><span style='font-size:10px;color:{SLATE_THEME['text_muted']}'>ĐÁP VIÊN</span>",
        x=0.5,
        y=0.5,
        font=dict(size=16, color=SLATE_THEME["text_primary"], family=SLATE_THEME["font_family"]),
        showarrow=False,
    )

    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=height)
    return apply_slate_theme(fig, title="Phân bổ Nơi sinh sống", show_legend=False)


def create_age_distribution_chart(
    history_df: Optional[pd.DataFrame],
    height: int = 280,
) -> go.Figure:
    """
    Renders discrete age distribution bar chart for Gen Z (18 - 27 years old).
    Features individual age bars, count labels, and mean age reference line.
    """
    if history_df is None or history_df.empty or "age" not in history_df.columns:
        return create_empty_state_figure("Chưa có dữ liệu độ tuổi", height=height)

    age_series = pd.to_numeric(history_df["age"], errors="coerce").dropna()
    if age_series.empty:
        return create_empty_state_figure("Chưa có dữ liệu độ tuổi", height=height)

    # 18-27 Gen Z categories
    possible_ages = list(range(18, 28))
    counts_map = age_series.value_counts().to_dict()
    counts = [int(counts_map.get(a, 0)) for a in possible_ages]
    mean_age = float(age_series.mean())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"{a}t" for a in possible_ages],
        y=counts,
        marker=dict(
            color=SLATE_THEME["primary_blue"],
            line=dict(color=SLATE_THEME["primary_blue_light"], width=1),
        ),
        text=[f"{c}" if c > 0 else "" for c in counts],
        textposition="outside",
        textfont=dict(color=SLATE_THEME["text_secondary"], size=10),
        hovertemplate="Độ tuổi <b>%{x}</b>: <b>%{y}</b> đáp viên<extra></extra>",
    ))

    # Mean Age Indicator
    mean_idx = max(0.0, min(9.0, mean_age - 18.0))
    fig.add_vline(
        x=mean_idx,
        line_dash="dot",
        line_color=SLATE_THEME["benchmark_green"],
        line_width=2,
        annotation_text=f"TB: {mean_age:.1f}t",
        annotation_position="top right",
        annotation_font=dict(color=SLATE_THEME["benchmark_green"], size=10),
    )

    fig.update_layout(
        xaxis=dict(title="Độ tuổi Gen Z", gridcolor=SLATE_THEME["border"]),
        yaxis=dict(title="Số lượng", gridcolor=SLATE_THEME["border"]),
        margin=dict(l=45, r=25, t=35, b=40),
        height=height,
    )
    return apply_slate_theme(fig, title="Phân bổ Độ tuổi (18 - 27 Gen Z)", show_legend=False)


def create_employment_donut_chart(
    history_df: Optional[pd.DataFrame],
    height: int = 280,
) -> go.Figure:
    """
    Renders Occupation / Employment distribution chart (Student vs Working).
    """
    if history_df is None or history_df.empty or "employment" not in history_df.columns:
        return create_empty_state_figure("Chưa có dữ liệu nghề nghiệp", height=height)

    emp_series = history_df["employment"].dropna().astype(str)
    if emp_series.empty:
        return create_empty_state_figure("Chưa có dữ liệu nghề nghiệp", height=height)

    emp_map = {
        "af1929cf-8d02-4c2a-b152-33facb89b2ae": "Bán thời gian",
        "226d694d-7ba6-4dfc-a600-ad508d32a2fd": "Toàn thời gian",
        "690cab48-5280-4e7c-91c8-bf2b8b751d35": "Freelancer",
        "5bceec63-270c-4305-b399-834a1ef7244c": "Đang tìm việc",
        "ff63da71-296c-4e14-9bb7-26a2afb35458": "Học sinh / Sinh viên",
        "87bdb602-4a9b-43cf-9f73-22af4be99496": "Khác",
        "1": "Bán thời gian", "2": "Toàn thời gian", "3": "Freelancer",
        "4": "Đang tìm việc", "5": "Học sinh / Sinh viên", "6": "Khác",
        1: "Bán thời gian", 2: "Toàn thời gian", 3: "Freelancer",
        4: "Đang tìm việc", 5: "Học sinh / Sinh viên", 6: "Khác",
    }
    mapped = emp_series.map(lambda x: emp_map.get(x, emp_map.get(int(x) if str(x).isdigit() else x, x)))
    counts = mapped.value_counts()

    palette = [
        SLATE_THEME["primary_blue"],
        SLATE_THEME["benchmark_green"],
        SLATE_THEME["warning_amber"],
        "#8b5cf6",
        "#06b6d4",
        "#ec4899",
    ]
    labels = list(counts.index)
    values = [int(v) for v in counts.values]
    colors = [palette[i % len(palette)] for i in range(len(labels))]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color=SLATE_THEME["bg_canvas"], width=2)),
        textinfo="percent",
        textfont=dict(color="#ffffff", size=10),
        hovertemplate="<b>%{label}</b>: %{value} đáp viên (%{percent:.1%})<extra></extra>",
    )])

    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=10, color=SLATE_THEME["text_secondary"]),
        ),
        margin=dict(l=20, r=90, t=30, b=20),
        height=height,
    )
    return apply_slate_theme(fig, title="Tình trạng Việc làm / Nghề nghiệp", show_legend=True)


def create_frequency_chart(
    history_df: Optional[pd.DataFrame],
    height: int = 280,
) -> go.Figure:
    """
    Renders Shopee Shopping Frequency horizontal bar chart.
    """
    if history_df is None or history_df.empty or "frequency" not in history_df.columns:
        return create_empty_state_figure("Chưa có dữ liệu tần suất Shopee", height=height)

    freq_series = history_df["frequency"].dropna().astype(str)
    if freq_series.empty:
        return create_empty_state_figure("Chưa có dữ liệu tần suất Shopee", height=height)

    freq_map = {
        "338ca9ac-a9be-4f70-8d8d-52bf6c9cd1c8": "< 2 đơn/tháng",
        "715ebb81-cfee-43f1-bf78-de86368dbac0": "2 - 4 đơn/tháng",
        "3ead4f73-ee69-496c-8872-d4df6d53d89b": "4 - 6 đơn/tháng",
        "cf2b1d3d-5307-4220-a6b9-9649ee940d75": "6 - 8 đơn/tháng",
        "d5586e4c-5b04-4174-8200-3162232169b7": "> 8 đơn/tháng",
        "1": "< 2 đơn/tháng", "2": "2 - 4 đơn/tháng", "3": "4 - 6 đơn/tháng",
        "4": "6 - 8 đơn/tháng", "5": "> 8 đơn/tháng",
        1: "< 2 đơn/tháng", 2: "2 - 4 đơn/tháng", 3: "4 - 6 đơn/tháng",
        4: "6 - 8 đơn/tháng", 5: "> 8 đơn/tháng",
    }
    mapped = freq_series.map(lambda x: freq_map.get(x, freq_map.get(int(x) if str(x).isdigit() else x, x)))
    ordered_tiers = ["< 2 đơn/tháng", "2 - 4 đơn/tháng", "4 - 6 đơn/tháng", "6 - 8 đơn/tháng", "> 8 đơn/tháng"]
    counts_map = mapped.value_counts().to_dict()
    counts = [int(counts_map.get(t, 0)) for t in ordered_tiers]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=ordered_tiers,
        x=counts,
        orientation="h",
        marker=dict(
            color=SLATE_THEME["primary_blue"],
            line=dict(color=SLATE_THEME["primary_blue_light"], width=1),
        ),
        text=[f"{c}" if c > 0 else "" for c in counts],
        textposition="outside",
        textfont=dict(color=SLATE_THEME["text_secondary"], size=10),
        hovertemplate="Tần suất <b>%{y}</b>: <b>%{x}</b> đáp viên<extra></extra>",
    ))

    fig.update_layout(
        xaxis=dict(title="Số lượng", gridcolor=SLATE_THEME["border"]),
        yaxis=dict(gridcolor="rgba(30, 41, 59, 0.4)"),
        margin=dict(l=100, r=25, t=35, b=40),
        height=height,
    )
    return apply_slate_theme(fig, title="Tần suất Mua sắm Shopee", show_legend=False)


def create_demographic_charts(history_df: Optional[pd.DataFrame]) -> Dict[str, go.Figure]:
    """Factory helper returning all 4 demographic distribution figures in a dictionary."""
    return {
        "gender": create_gender_donut_chart(history_df),
        "age": create_age_distribution_chart(history_df),
        "employment": create_employment_donut_chart(history_df),
        "frequency": create_frequency_chart(history_df),
    }
