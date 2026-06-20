"""
Plotly 시각화 모듈
- 공정능력분석: 히스토그램+박스플롯(규격선, 정규분포곡선), Q-Q plot
- 관리도: 관리도(점/CL/UCL/LCL), 이탈점 강조
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# 색상 팔레트
C_POINT = "#1f77b4"
C_CL = "#2e7d32"
C_LIM = "#d32f2f"
C_SPEC = "#7b1fa2"
C_OOC = "#e53935"


# ===========================================================================
# 공정능력분석 시각화
# ===========================================================================
def capability_figure(values, USL, LSL, x_bar, sigma_within, title="공정능력 분석"):
    """히스토그램 + 정규분포곡선 + 규격선(USL/LSL/Target)."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=x, nbinsx=min(30, max(10, int(np.sqrt(len(x)) * 2))),
        histnorm="probability density", name="데이터",
        marker_color="rgba(31,119,180,0.45)"))

    # 정규분포 곡선 (군내변동 기준)
    lo = min(x.min(), LSL) - sigma_within
    hi = max(x.max(), USL) + sigma_within
    xs = np.linspace(lo, hi, 400)
    fig.add_trace(go.Scatter(
        x=xs, y=norm.pdf(xs, loc=x_bar, scale=sigma_within),
        mode="lines", name="정규분포(σ_within)", line=dict(color=C_POINT, width=2)))

    for val, txt, col in [(LSL, "LSL", C_LIM), (USL, "USL", C_LIM),
                          ((USL + LSL) / 2, "Target", C_SPEC)]:
        fig.add_vline(x=val, line=dict(color=col, width=2, dash="dash"),
                      annotation_text=txt, annotation_position="top")
    fig.add_vline(x=x_bar, line=dict(color=C_CL, width=2),
                  annotation_text="x̄", annotation_position="bottom")

    fig.update_layout(title=title, height=430, bargap=0.05,
                      margin=dict(l=20, r=20, t=50, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      xaxis_title="측정값", yaxis_title="밀도")
    return fig


def box_by_group_figure(df, subgroup_col, value_col, USL=None, LSL=None):
    """부분군별 박스플롯."""
    fig = go.Figure()
    for name, g in df.groupby(subgroup_col, sort=True):
        fig.add_trace(go.Box(y=g[value_col], name=str(name), boxpoints="all",
                             jitter=0.4, marker_size=4, line_width=1))
    if USL is not None:
        fig.add_hline(y=USL, line=dict(color=C_LIM, dash="dash"), annotation_text="USL")
    if LSL is not None:
        fig.add_hline(y=LSL, line=dict(color=C_LIM, dash="dash"), annotation_text="LSL")
    fig.update_layout(height=380, showlegend=False,
                      margin=dict(l=20, r=20, t=30, b=20),
                      xaxis_title=str(subgroup_col), yaxis_title=str(value_col))
    return fig


def qq_figure(osm, osr, slope, intercept):
    """Q-Q plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                             marker=dict(color=C_POINT, size=6), name="표본"))
    line_x = np.array([osm.min(), osm.max()])
    fig.add_trace(go.Scatter(x=line_x, y=slope * line_x + intercept,
                             mode="lines", line=dict(color=C_LIM, width=2), name="기준선"))
    fig.update_layout(title="Q-Q Plot (정규성)", height=380,
                      margin=dict(l=20, r=20, t=50, b=20), showlegend=False,
                      xaxis_title="이론 분위수", yaxis_title="표본 분위수")
    return fig


# ===========================================================================
# 관리도 시각화
# ===========================================================================
def control_chart_figure(charts, names, var_name="값", ooc_map=None, title=None):
    """
    관리도 (다중 서브플롯).
    charts: [DataFrame, ...]  각 'point','CL','UCL','LCL'
    names : [차트 이름, ...]
    ooc_map: {chart_idx: [이탈 위치(0-base) ...]}  강조 표시
    """
    ooc_map = ooc_map or {}
    n = len(charts)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=False,
                        subplot_titles=[f"{nm} 관리도" for nm in names],
                        vertical_spacing=0.12)

    for i, (ch, nm) in enumerate(zip(charts, names)):
        row = i + 1
        xidx = list(ch.index)
        # 관리한계/중심선
        fig.add_trace(go.Scatter(x=xidx, y=ch["UCL"], mode="lines",
                                 line=dict(color=C_LIM, dash="dot", width=1.3),
                                 name="UCL", showlegend=(i == 0)), row=row, col=1)
        fig.add_trace(go.Scatter(x=xidx, y=ch["CL"], mode="lines",
                                 line=dict(color=C_CL, dash="dashdot", width=1.3),
                                 name="CL", showlegend=(i == 0)), row=row, col=1)
        fig.add_trace(go.Scatter(x=xidx, y=ch["LCL"], mode="lines",
                                 line=dict(color=C_LIM, dash="dot", width=1.3),
                                 name="LCL", showlegend=(i == 0)), row=row, col=1)
        # 데이터 점
        pts = ch["point"].to_numpy(dtype=float)
        colors = [C_POINT] * len(pts)
        for j in ooc_map.get(i, []):
            if 0 <= j < len(colors):
                colors[j] = C_OOC
        fig.add_trace(go.Scatter(x=xidx, y=pts, mode="lines+markers",
                                 line=dict(color=C_POINT, width=1.5),
                                 marker=dict(color=colors, size=8,
                                             line=dict(width=1, color="white")),
                                 name=nm, showlegend=False), row=row, col=1)
        # 한계값 주석
        last = xidx[-1]
        fig.add_annotation(x=last, y=ch["UCL"].iloc[-1], text=f"UCL={ch['UCL'].iloc[-1]:.4f}",
                           showarrow=False, xanchor="left", font=dict(color=C_LIM, size=10),
                           row=row, col=1)
        fig.add_annotation(x=last, y=ch["CL"].iloc[-1], text=f"CL={ch['CL'].iloc[-1]:.4f}",
                           showarrow=False, xanchor="left", font=dict(color=C_CL, size=10),
                           row=row, col=1)
        fig.add_annotation(x=last, y=ch["LCL"].iloc[-1], text=f"LCL={ch['LCL'].iloc[-1]:.4f}",
                           showarrow=False, xanchor="left", font=dict(color=C_LIM, size=10),
                           row=row, col=1)
        fig.update_yaxes(title_text=nm, row=row, col=1)
        fig.update_xaxes(title_text="부분군", row=row, col=1)

    fig.update_layout(height=260 * n + 60,
                      title=title or f"{var_name} 관리도",
                      margin=dict(l=40, r=90, t=70, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0))
    return fig
