"""
통계적공정관리(SPC) 모듈
- 계량형 관리도: Xbar-R, Xbar-S, I-MR
- 계수형 관리도: P, NP, C, U
- 관리이탈 판정: Nelson's Rules (8 rules)
- 이상치 제거 후 관리한계 재계산
"""
import numpy as np
import pandas as pd

from constants import A2, A3, D3, D4, B3, B4, d2 as _d2


# ===========================================================================
# 계량형 관리도
# ===========================================================================
def xbar_r(df, subgroup_col, value_col):
    """Xbar-R 관리도. 반환: (xbar_chart, r_chart) DataFrame."""
    grp = df.groupby(subgroup_col, sort=True)[value_col]
    xbar = grp.mean()
    R = grp.max() - grp.min()
    n = int(round(grp.count().mode().iloc[0]))   # 최빈 부분군 크기

    xbar_bar, r_bar = xbar.mean(), R.mean()
    a2, d3c, d4c = A2(n), D3(n), D4(n)

    xc = pd.DataFrame({"point": xbar.values}, index=xbar.index)
    xc["CL"], xc["UCL"], xc["LCL"] = xbar_bar, xbar_bar + a2 * r_bar, xbar_bar - a2 * r_bar

    rc = pd.DataFrame({"point": R.values}, index=R.index)
    rc["CL"], rc["UCL"], rc["LCL"] = r_bar, d4c * r_bar, d3c * r_bar
    return {"Xbar": xc, "R": rc, "n": n}


def xbar_s(df, subgroup_col, value_col):
    """Xbar-S 관리도. 반환: (xbar_chart, s_chart)."""
    grp = df.groupby(subgroup_col, sort=True)[value_col]
    xbar = grp.mean()
    S = grp.std(ddof=1)
    n = int(round(grp.count().mode().iloc[0]))

    xbar_bar, s_bar = xbar.mean(), S.mean()
    a3, b3, b4 = A3(n), B3(n), B4(n)

    xc = pd.DataFrame({"point": xbar.values}, index=xbar.index)
    xc["CL"], xc["UCL"], xc["LCL"] = xbar_bar, xbar_bar + a3 * s_bar, xbar_bar - a3 * s_bar

    sc = pd.DataFrame({"point": S.values}, index=S.index)
    sc["CL"], sc["UCL"], sc["LCL"] = s_bar, b4 * s_bar, b3 * s_bar
    return {"Xbar": xc, "S": sc, "n": n}


def i_mr(df, value_col, window=3):
    """I-MR 관리도 (부분군 크기 1). 반환: (i_chart, mr_chart)."""
    x = df[value_col].reset_index(drop=True).astype(float)
    idx = np.arange(1, len(x) + 1)
    w = max(2, int(window))

    xbar = x.mean()
    mr = x.rolling(window=w).apply(lambda v: v.max() - v.min(), raw=True)
    mr_bar = mr[w - 1:].mean()
    d2c, d3c, d4c = _d2(w), D3(w), D4(w)

    ic = pd.DataFrame({"point": x.values}, index=idx)
    ic["CL"] = xbar
    ic["UCL"] = xbar + 3 * mr_bar / d2c
    ic["LCL"] = xbar - 3 * mr_bar / d2c

    mc = pd.DataFrame({"point": mr.values}, index=idx)
    mc["CL"] = mr_bar
    mc["UCL"] = d4c * mr_bar
    mc["LCL"] = d3c * mr_bar
    return {"I": ic, "MR": mc, "n": 1}


# ===========================================================================
# 계수형 관리도
# ===========================================================================
def p_chart(df, subgroup_col, size_col, count_col):
    """P 관리도 (불량률). 표본 크기가 달라도 사용 가능 (관리한계 가변)."""
    g = df.groupby(subgroup_col, sort=True)
    cnt = g[count_col].sum()
    n_i = g[size_col].sum()
    p_bar = cnt.sum() / n_i.sum()
    p = cnt / n_i

    c = pd.DataFrame({"point": p.values}, index=p.index)
    c["CL"] = p_bar
    se = np.sqrt(p_bar * (1 - p_bar) / n_i.values)
    c["UCL"] = p_bar + 3 * se
    c["LCL"] = p_bar - 3 * se
    return {"P": c, "p_bar": float(p_bar)}


def np_chart(df, subgroup_col, size_col, count_col):
    """NP 관리도 (불량개수). 표본 크기가 동일해야 함."""
    g = df.groupby(subgroup_col, sort=True)
    cnt = g[count_col].sum()
    n_i = g[size_col].sum()
    n = n_i.mode().iloc[0]
    p_bar = cnt.sum() / n_i.sum()
    np_bar = p_bar * n

    c = pd.DataFrame({"point": cnt.values}, index=cnt.index)
    c["CL"] = np_bar
    se = np.sqrt(np_bar * (1 - p_bar))
    c["UCL"] = np_bar + 3 * se
    c["LCL"] = np_bar - 3 * se
    return {"NP": c, "np_bar": float(np_bar), "n": int(n)}


def c_chart(df, subgroup_col, count_col):
    """C 관리도 (결점수). 표본 크기가 동일해야 함."""
    g = df.groupby(subgroup_col, sort=True)
    cnt = g[count_col].sum()
    c_bar = cnt.mean()

    c = pd.DataFrame({"point": cnt.values}, index=cnt.index)
    c["CL"] = c_bar
    se = np.sqrt(c_bar)
    c["UCL"] = c_bar + 3 * se
    c["LCL"] = c_bar - 3 * se
    return {"C": c, "c_bar": float(c_bar)}


def u_chart(df, subgroup_col, size_col, count_col):
    """U 관리도 (단위당 결점수). 표본 크기가 달라도 사용 가능."""
    g = df.groupby(subgroup_col, sort=True)
    cnt = g[count_col].sum()
    n_i = g[size_col].sum()
    u_bar = cnt.sum() / n_i.sum()
    u = cnt / n_i

    c = pd.DataFrame({"point": u.values}, index=u.index)
    c["CL"] = u_bar
    se = np.sqrt(u_bar / n_i.values)
    c["UCL"] = u_bar + 3 * se
    c["LCL"] = u_bar - 3 * se
    return {"U": c, "u_bar": float(u_bar)}


# ===========================================================================
# 관리이탈 판정 - Nelson's Rules
# ===========================================================================
def nelson_rules(chart):
    """
    Nelson's 8 rules 로 관리이탈 패턴 탐지.
    chart: 'point','CL','UCL','LCL' 컬럼을 가진 DataFrame.
    반환: {rule_no: [위치 인덱스...]} 및 각 점의 위반 여부.
    sigma 는 (UCL-CL)/3 로 추정.
    """
    x = chart["point"].to_numpy(dtype=float)
    cl = chart["CL"].to_numpy(dtype=float)
    ucl = chart["UCL"].to_numpy(dtype=float)
    sigma = (ucl - cl) / 3.0
    sigma = np.where(sigma == 0, np.nan, sigma)
    z = (x - cl) / sigma           # 표준화 (CL 기준 sigma 단위)
    n = len(x)
    viol = {i: [] for i in range(1, 9)}

    def add(rule, idxs):
        for i in idxs:
            if 0 <= i < n and i not in viol[rule]:
                viol[rule].append(i)

    # Rule 1: 1점이 ±3σ 밖
    add(1, list(np.where(np.abs(z) > 3)[0]))

    # Rule 2: 연속 9점이 중심선 한쪽
    for i in range(n - 8):
        seg = z[i:i + 9]
        if np.all(seg > 0) or np.all(seg < 0):
            add(2, range(i, i + 9))

    # Rule 3: 연속 6점이 지속 증가 또는 감소
    for i in range(n - 5):
        seg = x[i:i + 6]
        if np.all(np.diff(seg) > 0) or np.all(np.diff(seg) < 0):
            add(3, range(i, i + 6))

    # Rule 4: 연속 14점이 교대로 증감
    for i in range(n - 13):
        d = np.diff(x[i:i + 14])
        if np.all(d != 0) and np.all(np.sign(d[:-1]) != np.sign(d[1:])):
            add(4, range(i, i + 14))

    # Rule 5: 연속 3점 중 2점이 같은 쪽 2σ 밖
    for i in range(n - 2):
        seg = z[i:i + 3]
        for side in (1, -1):
            if np.sum(side * seg > 2) >= 2:
                add(5, range(i, i + 3))

    # Rule 6: 연속 5점 중 4점이 같은 쪽 1σ 밖
    for i in range(n - 4):
        seg = z[i:i + 5]
        for side in (1, -1):
            if np.sum(side * seg > 1) >= 4:
                add(6, range(i, i + 5))

    # Rule 7: 연속 15점이 ±1σ 이내
    for i in range(n - 14):
        if np.all(np.abs(z[i:i + 15]) < 1):
            add(7, range(i, i + 15))

    # Rule 8: 연속 8점이 ±1σ 밖 (양쪽)
    for i in range(n - 7):
        if np.all(np.abs(z[i:i + 8]) > 1):
            add(8, range(i, i + 8))

    viol = {k: v for k, v in viol.items() if v}
    any_idx = sorted(set(i for v in viol.values() for i in v))
    return {"violations": viol, "ooc_index": any_idx}


RULE_DESC = {
    1: "1점이 관리한계(±3σ)를 벗어남",
    2: "연속 9점이 중심선 한쪽에 위치",
    3: "연속 6점이 지속적으로 증가 또는 감소",
    4: "연속 14점이 교대로 증감(진동)",
    5: "연속 3점 중 2점이 같은 쪽 2σ 밖",
    6: "연속 5점 중 4점이 같은 쪽 1σ 밖",
    7: "연속 15점이 ±1σ 이내 (변동 과소)",
    8: "연속 8점이 ±1σ 밖 (양쪽)",
}


def ooc_by_limits(chart):
    """관리한계(UCL/LCL) 이탈 점의 인덱스만 반환 (Rule 1 단순판정)."""
    x = chart["point"].to_numpy(dtype=float)
    ucl = chart["UCL"].to_numpy(dtype=float)
    lcl = chart["LCL"].to_numpy(dtype=float)
    mask = (x > ucl) | (x < lcl)
    return list(np.where(mask)[0])
