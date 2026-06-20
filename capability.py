"""
공정능력분석(Process Capability Analysis) 모듈
- 단기 공정능력지수 Cp, Cpk  (군내변동 sigma_within)
- 장기 공정능력지수 Pp, Ppk  (전체변동 sigma_overall)
- 정규성 검정 / Box-Cox(또는 Yeo-Johnson) 변환
- 공정능력 판정 등급
"""
import numpy as np
import pandas as pd
from scipy import stats

from constants import c4


def estimate_sigma(df, subgroup_col, value_col):
    """sigma_within(합동표준편차 기반) 과 sigma_overall(전체표준편차 기반) 추정."""
    s = df[[subgroup_col, value_col]].dropna()
    x = s[value_col].to_numpy(dtype=float)
    N = len(x)

    # 전체변동
    s_total = x.std(ddof=1)
    sigma_overall = s_total / c4(N)

    # 군내변동 (합동표준편차)  s_p = sqrt( SS_within / dof ),  dof = sum(n_i - 1)
    grp = s.groupby(subgroup_col)[value_col]
    ss_within = grp.apply(lambda g: ((g - g.mean()) ** 2).sum()).sum()
    dof = int((grp.count() - 1).sum())
    if dof <= 0:
        sigma_within = np.nan
        d = np.nan
    else:
        s_p = np.sqrt(ss_within / dof)
        d = dof + 1
        sigma_within = s_p / c4(d)

    return {
        "N": N,
        "k": int(grp.ngroups),
        "x_bar": float(x.mean()),
        "s_total": float(s_total),
        "sigma_overall": float(sigma_overall),
        "s_pooled": float(np.sqrt(ss_within / dof)) if dof > 0 else np.nan,
        "dof": dof,
        "c4_overall": float(c4(N)),
        "c4_within": float(c4(d)) if dof > 0 else np.nan,
        "sigma_within": float(sigma_within),
    }


def capability_indices(df, subgroup_col, value_col, USL, LSL):
    """Cp, Cpk, Pp, Ppk 계산."""
    est = estimate_sigma(df, subgroup_col, value_col)
    x_bar = est["x_bar"]
    sw = est["sigma_within"]
    so = est["sigma_overall"]

    def _cp(sig):
        return (USL - LSL) / (6 * sig) if sig and sig > 0 else np.nan

    def _cpk(sig):
        if not sig or sig <= 0:
            return np.nan
        return min((USL - x_bar) / (3 * sig), (x_bar - LSL) / (3 * sig))

    res = dict(est)
    res.update({
        "USL": USL, "LSL": LSL,
        "Cp": _cp(sw), "Cpk": _cpk(sw),
        "Pp": _cp(so), "Ppk": _cpk(so),
    })
    return res


def normality_test(values):
    """Shapiro-Wilk 정규성 검정. p>=0.05 이면 정규성 만족."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return {"stat": np.nan, "p": np.nan, "normal": None,
                "msg": "표본이 너무 적어 검정 불가 (n>=3 필요)"}
    stat, p = stats.shapiro(x)
    return {"stat": float(stat), "p": float(p), "normal": bool(p >= 0.05),
            "msg": "정규성 만족" if p >= 0.05 else "정규성 불만족"}


def qq_data(values):
    """Q-Q plot 용 (theoretical, sample) 분위수 + 회귀선 정보."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    z = stats.zscore(x)
    (osm, osr), (slope, intercept, r) = stats.probplot(z, dist="norm")
    return osm, osr, slope, intercept, r


def transform_if_needed(values, USL, LSL):
    """
    정규성 불만족 시 Box-Cox(양수) 또는 Yeo-Johnson(음수 포함) 변환.
    규격(USL, LSL)도 동일 변환을 적용하여 반환.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]

    if np.all(x > 0):
        xt, lmbda = stats.boxcox(x)
        method = "Box-Cox"

        def _t(v):
            return stats.boxcox(np.array([v]), lmbda=lmbda)[0] if lmbda != 0 else np.log(v)

        usl_t = (USL ** lmbda - 1) / lmbda if lmbda != 0 else np.log(USL)
        lsl_t = (LSL ** lmbda - 1) / lmbda if lmbda != 0 else np.log(LSL)
    else:
        xt, lmbda = stats.yeojohnson(x)
        method = "Yeo-Johnson"
        usl_t = stats.yeojohnson(np.array([USL]), lmbda=lmbda)[0]
        lsl_t = stats.yeojohnson(np.array([LSL]), lmbda=lmbda)[0]

    return {"data": xt, "lmbda": float(lmbda), "method": method,
            "USL_t": float(usl_t), "LSL_t": float(lsl_t)}


# 공정능력 판정 기준 (강의록 판정표)
def capability_grade(cp):
    """Cp(또는 Cpk) 값에 따른 공정능력 등급/판정."""
    if cp is None or np.isnan(cp):
        return {"grade": None, "judgment": "-", "action": "-", "color": "#9e9e9e"}
    if cp >= 1.67:
        return {"grade": 0, "judgment": "매우 충분",
                "action": "관리 간소화를 고려해도 좋음 (±5σ 수준)", "color": "#2e7d32"}
    if cp >= 1.33:
        return {"grade": 1, "judgment": "충분",
                "action": "이상적인 상태이므로 현재 상태 유지 (±4σ 수준)", "color": "#66bb6a"}
    if cp >= 1.00:
        return {"grade": 2, "judgment": "충분하지는 않지만 괜찮음",
                "action": "관리상태를 확실히 유지, 불량 발생 가능성 주의 (±3σ 수준)", "color": "#fbc02d"}
    if cp >= 0.67:
        return {"grade": 3, "judgment": "부족",
                "action": "전체 선별, 공정 개선·관리 필요 (±2σ 수준)", "color": "#fb8c00"}
    return {"grade": 4, "judgment": "매우 부족",
            "action": "긴급 대책 필요, 규격 재검토 (±1σ 수준)", "color": "#e53935"}
