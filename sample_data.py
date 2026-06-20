"""
샘플 데이터 생성 모듈 (강의록 generate_data 방식)
- 계량형: 부분군별 정규분포 데이터
- 계수형: 이항분포 기반 불량/결점수 데이터
"""
import numpy as np
import pandas as pd


def generate_value_data(var_name="측정값", sg_name="부분군", target=40.0,
                        num_sg=20, sg_size=5, sg_std=2.0, mean_shift=0.0,
                        sg_size_variation=0, seed=42):
    """계량형(연속형) 데이터 생성. 반환: long format DataFrame[sg_name, var_name]."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(num_sg):
        shift = rng.uniform(-mean_shift, mean_shift) if mean_shift > 0 else 0.0
        if sg_size_variation > 0:
            n_i = sg_size + rng.integers(-sg_size_variation, sg_size_variation + 1)
            n_i = max(1, int(n_i))
        else:
            n_i = sg_size
        vals = rng.normal(loc=target + shift, scale=sg_std, size=n_i)
        frames.append(pd.DataFrame({sg_name: i + 1, var_name: vals}))
    return pd.concat(frames, ignore_index=True)


def generate_count_data(var_name="불량수", sg_name="부분군", size_name="표본크기",
                        num_sg=20, sg_size=300, p=0.02, sg_size_variation=0, seed=42):
    """계수형 데이터 생성. 반환: DataFrame[sg_name, size_name, var_name]."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(num_sg):
        if sg_size_variation > 0:
            n_i = sg_size + rng.integers(-sg_size_variation, sg_size_variation + 1)
            n_i = max(1, int(n_i))
        else:
            n_i = sg_size
        c = rng.binomial(n=n_i, p=p)
        rows.append({sg_name: i + 1, size_name: n_i, var_name: int(c)})
    return pd.DataFrame(rows)


def add_outliers_value(df, var_name, sg_name, n_out=1, magnitude=4.0, seed=7):
    """일부 부분군에 이상치를 주입(이상치 제거 데모용)."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    std = df[var_name].std()
    idxs = rng.choice(df.index, size=min(n_out, len(df)), replace=False)
    df.loc[idxs, var_name] = df.loc[idxs, var_name] + magnitude * std
    return df
