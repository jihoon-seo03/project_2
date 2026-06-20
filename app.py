"""
스마트제조 프로젝트 2 — 공정능력분석 & 통계적공정관리(SPC) 웹앱
홍익대학교 스마트제조

데이터를 바꾸면 즉시 공정능력분석 + SPC 관리도가 재계산되고,
시각화 대시보드로 현재 공정 상태를 판단할 수 있습니다.
"""
import io
import numpy as np
import pandas as pd
import streamlit as st

import capability as cap
import spc
import viz
from sample_data import generate_value_data, generate_count_data, add_outliers_value

st.set_page_config(page_title="공정능력분석 · SPC", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")


def _render_ooc(chart, nelson, name):
    """관리이탈(Nelson's Rules) 판정 결과를 표로 렌더링."""
    viol = nelson["violations"]
    ooc = nelson["ooc_index"]
    idx_list = list(chart.index)
    if not viol:
        st.success(f"✅ '{name}' 차트: 관리상태 — Nelson's Rule 위반 없음 (모든 점이 관리상태)")
        return
    st.error(f"⚠️ '{name}' 차트: {len(ooc)}개 점에서 관리이탈 패턴 감지")
    rows = []
    for rule, idxs in sorted(viol.items()):
        labels = [str(idx_list[i]) for i in idxs if 0 <= i < len(idx_list)]
        shown = ", ".join(labels[:25]) + (" ..." if len(labels) > 25 else "")
        rows.append({"규칙": f"Rule {rule}", "내용": spc.RULE_DESC[rule], "해당 부분군": shown})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# 헤더
# ---------------------------------------------------------------------------
st.title("공정능력분석 & 통계적공정관리(SPC)")

# ===========================================================================
# 사이드바 — 데이터 설정
# ===========================================================================
with st.sidebar:
    st.header("⚙️ 데이터 설정")

    data_type = st.radio("데이터 유형", ["계량형 (연속형 측정값)", "계수형 (불량·결점 개수)"],
                         help="계량형: 길이·두께·점도 등 / 계수형: 불량개수·결점수 등")
    is_value = data_type.startswith("계량형")

    source = st.radio("데이터 소스", ["샘플 데이터 생성", "CSV 업로드", "직접 입력·편집"])

    st.divider()

    df = None
    colmap = {}

    # ----- 계량형 -----
    if is_value:
        SG, VAL = "부분군", "측정값"
        if source == "샘플 데이터 생성":
            st.subheader("샘플 생성 파라미터")
            var_name = st.text_input("측정 특성명", "두께", key="v_var")
            target = st.number_input("목표값(중심)", value=40.0, step=1.0, key="v_target")
            tol = st.number_input("허용오차(±공차)", value=2.0, min_value=0.0, step=0.5, key="v_tol")
            num_sg = st.slider("부분군 수", 5, 60, 30, key="v_numsg")
            sg_size = st.slider("부분군 크기(표본 수)", 1, 15, 4, key="v_sgsize")
            sg_std = st.number_input("부분군 표준편차", value=0.6, min_value=0.01, step=0.1, key="v_std")
            mean_shift = st.number_input("부분군 평균 이동폭", value=0.0, min_value=0.0, step=0.1, key="v_shift")
            add_out = st.checkbox("이상치 주입(이상점 제거 데모용)", value=False, key="v_out")
            df = generate_value_data(var_name=VAL, sg_name=SG, target=target,
                                     num_sg=num_sg, sg_size=sg_size, sg_std=sg_std,
                                     mean_shift=mean_shift)
            if add_out:
                df = add_outliers_value(df, VAL, SG, n_out=max(1, num_sg // 10), magnitude=4.0)
            st.session_state["spec_target"] = target
            st.session_state["spec_tol"] = tol

        elif source == "CSV 업로드":
            st.subheader("CSV 업로드")
            up = st.file_uploader("CSV 파일", type=["csv"], key="v_csv")
            if up is not None:
                try:
                    raw = pd.read_csv(up)
                    st.caption(f"행 {len(raw)} · 열 {list(raw.columns)}")
                    SG = st.selectbox("부분군 열", list(raw.columns), key="v_sgcol")
                    num_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
                    VAL = st.selectbox("측정값 열", num_cols or list(raw.columns), key="v_valcol")
                    df = raw[[SG, VAL]].dropna().copy()
                except Exception as e:
                    st.error(f"CSV 읽기 오류: {e}")

        else:  # 직접 입력·편집
            st.subheader("표 직접 편집")
            if "v_edit_df" not in st.session_state:
                st.session_state["v_edit_df"] = generate_value_data(
                    var_name=VAL, sg_name=SG, target=40, num_sg=10, sg_size=4, sg_std=0.6)
            edited = st.data_editor(st.session_state["v_edit_df"], num_rows="dynamic",
                                    width="stretch", key="v_editor", height=300)
            df = edited.dropna()

    # ----- 계수형 -----
    else:
        SG, SIZE, CNT = "부분군", "표본크기", "개수"
        if source == "샘플 데이터 생성":
            st.subheader("샘플 생성 파라미터")
            defect_kind = st.radio("관리 대상", ["불량 (이항분포 → P/NP)", "결점 (포아송 → C/U)"], key="c_kind")
            var_name = st.text_input("특성명", "불량수" if "불량" in defect_kind else "결점수", key="c_var")
            num_sg = st.slider("부분군 수", 5, 60, 25, key="c_numsg")
            sg_size = st.slider("부분군 표본크기", 50, 1000, 300, step=10, key="c_sgsize")
            p = st.number_input("발생 확률(p)", value=0.02, min_value=0.001, max_value=0.5,
                                step=0.005, format="%.3f", key="c_p")
            size_var = st.checkbox("표본 크기 변동 허용(가변)", value=False, key="c_sizevar")
            CNT = var_name
            df = generate_count_data(var_name=CNT, sg_name=SG, size_name=SIZE,
                                     num_sg=num_sg, sg_size=sg_size, p=p,
                                     sg_size_variation=(sg_size // 10 if size_var else 0))
            st.session_state["defect_kind"] = "불량" if "불량" in defect_kind else "결점"

        elif source == "CSV 업로드":
            st.subheader("CSV 업로드")
            up = st.file_uploader("CSV 파일", type=["csv"], key="c_csv")
            if up is not None:
                try:
                    raw = pd.read_csv(up)
                    st.caption(f"행 {len(raw)} · 열 {list(raw.columns)}")
                    SG = st.selectbox("부분군 열", list(raw.columns), key="c_sgcol")
                    num_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
                    SIZE = st.selectbox("표본크기 열", num_cols or list(raw.columns), key="c_sizecol")
                    CNT = st.selectbox("개수(불량·결점) 열",
                                       [c for c in (num_cols or list(raw.columns)) if c != SIZE]
                                       or list(raw.columns), key="c_cntcol")
                    df = raw[[SG, SIZE, CNT]].dropna().copy()
                    st.session_state["defect_kind"] = st.radio(
                        "관리 대상", ["불량", "결점"], key="c_kind_csv")
                except Exception as e:
                    st.error(f"CSV 읽기 오류: {e}")

        else:  # 직접 편집
            st.subheader("표 직접 편집")
            CNT = "불량수"
            if "c_edit_df" not in st.session_state:
                st.session_state["c_edit_df"] = generate_count_data(
                    var_name=CNT, sg_name=SG, size_name=SIZE, num_sg=10, sg_size=300, p=0.02)
            edited = st.data_editor(st.session_state["c_edit_df"], num_rows="dynamic",
                                    width="stretch", key="c_editor", height=300)
            df = edited.dropna()
            st.session_state["defect_kind"] = st.radio("관리 대상", ["불량", "결점"], key="c_kind_edit")

    colmap = {"SG": SG, "VAL": VAL} if is_value else {"SG": SG, "SIZE": SIZE, "CNT": CNT}


# ===========================================================================
# 데이터 유효성 체크
# ===========================================================================
def _data_ready(df, is_value, colmap):
    if df is None or len(df) == 0:
        return False, "데이터가 없습니다. 사이드바에서 데이터를 생성하거나 업로드하세요."
    if is_value:
        if df[colmap["SG"]].nunique() < 2:
            return False, "부분군이 2개 이상 필요합니다."
    else:
        if not set([colmap["SIZE"], colmap["CNT"]]).issubset(df.columns):
            return False, "표본크기·개수 열이 필요합니다."
    return True, ""


ready, msg = _data_ready(df, is_value, colmap)

# ===========================================================================
# 본문 탭
# ===========================================================================
if is_value:
    tab_data, tab_cap, tab_spc, tab_help = st.tabs(
        ["📁 데이터", "🎯 공정능력분석", "📈 관리도(SPC)", "❓ 도움말"])
else:
    tab_data, tab_spc, tab_help = st.tabs(["📁 데이터", "📈 관리도(SPC)", "❓ 도움말"])
    tab_cap = None

# ---------------------------------------------------------------------------
# 데이터 탭
# ---------------------------------------------------------------------------
with tab_data:
    if not ready:
        st.info(msg)
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("현재 데이터")
            st.dataframe(df, width="stretch", height=360)
            buff = io.StringIO(); df.to_csv(buff, index=False)
            st.download_button("⬇️ 현재 데이터 CSV 다운로드", buff.getvalue(),
                               "data.csv", "text/csv")
        with c2:
            st.subheader("요약")
            if is_value:
                v = df[colmap["VAL"]]
                st.metric("관측치 수", len(df))
                st.metric("부분군 수", df[colmap["SG"]].nunique())
                st.metric("평균", f"{v.mean():.4f}")
                st.metric("표준편차", f"{v.std(ddof=1):.4f}")
            else:
                st.metric("부분군 수", df[colmap["SG"]].nunique())
                st.metric("총 표본수", int(df[colmap["SIZE"]].sum()))
                st.metric("총 개수", int(df[colmap["CNT"]].sum()))

# ---------------------------------------------------------------------------
# 공정능력분석 탭
# ---------------------------------------------------------------------------
if tab_cap is not None:
    with tab_cap:
        if not ready:
            st.info(msg)
        else:
            SG, VAL = colmap["SG"], colmap["VAL"]
            vals = df[VAL].to_numpy(dtype=float)

            st.subheader("규격(스펙) 설정")
            sc1, sc2, sc3 = st.columns(3)
            spec_mode = sc1.radio("입력 방식", ["목표값 ± 공차", "USL / LSL 직접"], horizontal=True)
            if spec_mode == "목표값 ± 공차":
                tgt = sc2.number_input("목표값", value=float(st.session_state.get("spec_target", round(float(np.mean(vals)), 3))))
                tol = sc3.number_input("허용오차(±)", value=float(st.session_state.get("spec_tol", round(float(np.std(vals) * 3), 3))), min_value=0.0)
                USL, LSL = tgt + tol, tgt - tol
            else:
                USL = sc2.number_input("USL(상한)", value=float(np.max(vals)))
                LSL = sc3.number_input("LSL(하한)", value=float(np.min(vals)))
            st.caption(f"USL = {USL:.4f} · LSL = {LSL:.4f}")

            if USL <= LSL:
                st.error("USL은 LSL보다 커야 합니다.")
            else:
                # 정규성 검정
                nt = cap.normality_test(vals)
                use_transform = False
                trans = None
                st.subheader("정규성 검정 (Shapiro-Wilk)")
                ncol1, ncol2 = st.columns([1, 2])
                if nt["normal"] is None:
                    ncol1.warning(nt["msg"])
                elif nt["normal"]:
                    ncol1.success(f"p-value = {nt['p']:.4f}\n\n{nt['msg']}")
                else:
                    ncol1.error(f"p-value = {nt['p']:.4f}\n\n{nt['msg']}")
                    use_transform = ncol2.checkbox(
                        "비정규 → Box-Cox/Yeo-Johnson 변환 후 분석", value=True)

                # 변환 적용
                work_df = df.copy()
                wUSL, wLSL, wVAL = USL, LSL, VAL
                if use_transform:
                    try:
                        trans = cap.transform_if_needed(vals, USL, LSL)
                        work_df = df.copy()
                        # 변환은 그룹 구조 유지하며 값만 변환
                        if np.all(vals > 0) and trans["lmbda"] != 0:
                            work_df[VAL] = (work_df[VAL] ** trans["lmbda"] - 1) / trans["lmbda"]
                        elif np.all(vals > 0):
                            work_df[VAL] = np.log(work_df[VAL])
                        else:
                            from scipy.stats import yeojohnson
                            work_df[VAL] = yeojohnson(work_df[VAL].to_numpy(float), lmbda=trans["lmbda"])
                        wUSL, wLSL = trans["USL_t"], trans["LSL_t"]
                        ncol2.info(f"{trans['method']} 변환 적용 (λ = {trans['lmbda']:.4f})")
                    except Exception as e:
                        ncol2.warning(f"변환 실패, 원본으로 진행: {e}")
                        use_transform = False

                # 공정능력지수
                res = cap.capability_indices(work_df, SG, wVAL, wUSL, wLSL)
                st.subheader("공정능력지수")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Cp (단기)", f"{res['Cp']:.4f}")
                m2.metric("Cpk (단기)", f"{res['Cpk']:.4f}")
                m3.metric("Pp (장기)", f"{res['Pp']:.4f}")
                m4.metric("Ppk (장기)", f"{res['Ppk']:.4f}")

                grade = cap.capability_grade(res["Cpk"])
                st.markdown(
                    f"<div style='padding:12px;border-radius:8px;background:{grade['color']}22;"
                    f"border-left:6px solid {grade['color']}'>"
                    f"<b>판정 (Cpk 기준): 등급 {grade['grade']} — {grade['judgment']}</b><br>"
                    f"{grade['action']}</div>", unsafe_allow_html=True)

                with st.expander("계산 상세 (σ_within / σ_overall)"):
                    st.write({
                        "전체 평균 x̄": round(res["x_bar"], 4),
                        "σ_within (군내변동)": round(res["sigma_within"], 4),
                        "σ_overall (전체변동)": round(res["sigma_overall"], 4),
                        "합동표준편차 s_p": round(res["s_pooled"], 4),
                        "자유도 d": res["dof"] + 1,
                        "표본수 N": res["N"], "부분군수 k": res["k"],
                    })

                # 시각화
                st.subheader("시각화")
                g1, g2 = st.columns(2)
                with g1:
                    fig = viz.capability_figure(work_df[wVAL].to_numpy(float), wUSL, wLSL,
                                                res["x_bar"], res["sigma_within"],
                                                title="공정능력 (히스토그램 + 정규분포)")
                    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                with g2:
                    osm, osr, slope, intercept, r = cap.qq_data(work_df[wVAL].to_numpy(float))
                    st.plotly_chart(viz.qq_figure(osm, osr, slope, intercept),
                                    width="stretch", config={"displayModeBar": False})
                st.plotly_chart(
                    viz.box_by_group_figure(work_df, SG, wVAL, wUSL, wLSL),
                    width="stretch", config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# 관리도(SPC) 탭
# ---------------------------------------------------------------------------
with tab_spc:
    if not ready:
        st.info(msg)
    elif is_value:
        SG, VAL = colmap["SG"], colmap["VAL"]
        sizes = df.groupby(SG)[VAL].count()
        mode_n = int(sizes.mode().iloc[0])
        # 자동 추천
        if mode_n == 1:
            rec = "I-MR"
        elif mode_n <= 10:
            rec = "Xbar-R"
        else:
            rec = "Xbar-S"
        st.subheader("관리도 선택")
        chart_type = st.radio("관리도 종류", ["Xbar-R", "Xbar-S", "I-MR"],
                              index=["Xbar-R", "Xbar-S", "I-MR"].index(rec), horizontal=True)
        window = 3
        if chart_type == "I-MR":
            window = st.slider("이동범위 윈도우(w)", 2, 5, 3)

        try:
            if chart_type == "Xbar-R":
                out = spc.xbar_r(df, SG, VAL); names = ["Xbar", "R"]; charts = [out["Xbar"], out["R"]]
            elif chart_type == "Xbar-S":
                out = spc.xbar_s(df, SG, VAL); names = ["Xbar", "S"]; charts = [out["Xbar"], out["S"]]
            else:
                out = spc.i_mr(df, VAL, window=window); names = ["I", "MR"]; charts = [out["I"], out["MR"]]

            # 이탈 판정 (Nelson) - 주 차트(첫 번째) 기준
            main = charts[0]
            nelson = spc.nelson_rules(main)
            limit_ooc = {i: spc.ooc_by_limits(ch) for i, ch in enumerate(charts)}

            fig = viz.control_chart_figure(charts, names, var_name=VAL,
                                           ooc_map=limit_ooc, title=f"{chart_type} 관리도 — {VAL}")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

            _render_ooc(main, nelson, names[0])

            # 이상치 제거 후 관리한계 재계산 (강의록 예제)
            limit_pts = sorted(set(limit_ooc.get(0, []) + limit_ooc.get(1, [])))
            if limit_pts and chart_type in ("Xbar-R", "Xbar-S"):
                ooc_labels = [list(main.index)[i] for i in limit_pts]
                with st.expander(f"🔧 관리한계 이탈 부분군 제거 후 재작성 (이탈 {len(ooc_labels)}개)"):
                    st.write(f"이탈 부분군(Lot): {ooc_labels}")
                    df2 = df[~df[SG].isin(ooc_labels)]
                    if df2[SG].nunique() >= 2:
                        out2 = (spc.xbar_r(df2, SG, VAL) if chart_type == "Xbar-R"
                                else spc.xbar_s(df2, SG, VAL))
                        m2 = out2["Xbar"]
                        b1, b2 = st.columns(2)
                        b1.metric("초기 UCL", f"{main['UCL'].iloc[0]:.4f}")
                        b1.metric("초기 LCL", f"{main['LCL'].iloc[0]:.4f}")
                        b2.metric("재계산 UCL", f"{m2['UCL'].iloc[0]:.4f}",
                                  delta=f"{m2['UCL'].iloc[0]-main['UCL'].iloc[0]:.4f}")
                        b2.metric("재계산 LCL", f"{m2['LCL'].iloc[0]:.4f}",
                                  delta=f"{m2['LCL'].iloc[0]-main['LCL'].iloc[0]:.4f}")
                        sub = ["Xbar", "R"] if chart_type == "Xbar-R" else ["Xbar", "S"]
                        ch2 = [out2[sub[0]], out2[sub[1]]]
                        oo2 = {k: spc.ooc_by_limits(c) for k, c in enumerate(ch2)}
                        st.plotly_chart(
                            viz.control_chart_figure(ch2, sub, var_name=VAL, ooc_map=oo2,
                                                     title="이상치 제거 후 재작성된 관리도"),
                            width="stretch", config={"displayModeBar": False})
                        st.caption("이상치를 제거하면 관리한계 폭이 좁아져, 기존에 정상이던 점이 새로운 이상점이 될 수 있습니다. 모든 점이 관리상태가 될 때까지 반복합니다.")
                    else:
                        st.warning("제거 후 부분군이 부족하여 재계산할 수 없습니다.")
        except Exception as e:
            st.error(f"관리도 계산 오류: {e}")

    else:  # 계수형
        SG, SIZE, CNT = colmap["SG"], colmap["SIZE"], colmap["CNT"]
        kind = st.session_state.get("defect_kind", "불량")
        const_n = df[SIZE].nunique() == 1
        if kind == "불량":
            rec = "NP" if const_n else "P"
            options = ["NP", "P"]
        else:
            rec = "C" if const_n else "U"
            options = ["C", "U"]
        st.subheader("관리도 선택")
        chart_type = st.radio("관리도 종류", options, index=options.index(rec), horizontal=True)

        try:
            if chart_type == "NP":
                if not const_n:
                    st.warning("NP 관리도는 표본 크기가 동일해야 합니다. 표본이 가변이면 P 관리도를 권장합니다.")
                out = spc.np_chart(df, SG, SIZE, CNT); ch = out["NP"]; nm = "NP"
            elif chart_type == "P":
                out = spc.p_chart(df, SG, SIZE, CNT); ch = out["P"]; nm = "P"
            elif chart_type == "C":
                if not const_n:
                    st.warning("C 관리도는 표본 크기가 동일해야 합니다. 표본이 가변이면 U 관리도를 권장합니다.")
                out = spc.c_chart(df, SG, CNT); ch = out["C"]; nm = "C"
            else:
                out = spc.u_chart(df, SG, SIZE, CNT); ch = out["U"]; nm = "U"

            nelson = spc.nelson_rules(ch)
            limit_ooc = {0: spc.ooc_by_limits(ch)}
            fig = viz.control_chart_figure([ch], [nm], var_name=CNT,
                                           ooc_map=limit_ooc, title=f"{nm} 관리도 — {CNT}")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            _render_ooc(ch, nelson, nm)
        except Exception as e:
            st.error(f"관리도 계산 오류: {e}")

# ---------------------------------------------------------------------------
# 도움말 탭
# ---------------------------------------------------------------------------
with tab_help:
    st.markdown("""
### 사용 방법
1. **사이드바**에서 데이터 유형(계량형/계수형)과 소스(샘플 생성 · CSV 업로드 · 직접 편집)를 선택합니다.
2. **공정능력분석 탭**: 규격(USL/LSL 또는 목표±공차)을 입력하면 Cp·Cpk·Pp·Ppk와 판정 등급, 정규성 검정, 시각화가 자동 계산됩니다. 비정규 데이터는 Box-Cox 변환 후 분석할 수 있습니다.
3. **관리도(SPC) 탭**: 데이터에 맞는 관리도를 자동 추천하고, 관리한계와 이탈점(Nelson's Rules)을 표시합니다.

### 공정능력지수
- **Cp** = (USL−LSL) / (6·σ_within) — 산포만 고려(단기)
- **Cpk** = min((USL−x̄), (x̄−LSL)) / (3·σ_within) — 치우침 포함(단기)
- **Pp, Ppk**: σ_within 대신 σ_overall 사용(장기)
- **판정**: Cpk ≥ 1.33 충분 / 1.00~1.33 주의 / < 1.00 개선 필요

### 관리도 선택 기준
- **계량형**: 부분군 크기 1 → I-MR, 2~10 → Xbar-R, 10 초과 → Xbar-S
- **계수형**: 불량(이항) → NP(표본 일정)·P(가변) / 결점(포아송) → C(표본 일정)·U(가변)

*불편화 상수는 강의록 표를 내장하여 외부 파일 없이 동작합니다.*
""")

st.divider()
st.caption("스마트제조 프로젝트2 - C221024 서지훈")
