"""公開版の最小基盤。

本体（研究用リポジトリ）の pu_lib.py / label_lib.py のうち、7 節の
シミュレーションが実際に使う部分だけを切り出したもの。BigQuery にも
機関データにも一切触れない。

  - NUM_ASOF / CAT_ASOF : 特徴量名の一覧（列名だけ。値は含まない）
  - oof_predict         : out-of-fold 予測（pu_lib と同一実装）
  - surrogate_features  : 実データの代わりに使う代理特徴量行列の生成

代理特徴量について
------------------
論文の 7 節は「実コホートの特徴量行列 X を固定し、ラベルだけを合成する」
設計である。X はこのリポジトリには同梱していない。そこで公開版では、行数・列数・
数値列の相関構造（冗長な観測系列があるという性質）だけを模した代理行列を
生成する。

これで再現できるのは、図 11・図 12 が示す**定性的な形**（Δ に対する
改善幅の単調増加、rho による減衰、二つの診断の役割分担）である。
個々の数値は実データの X に依存するので一致しない。論文の図そのものを
再現したい場合は、同梱の exp22_*.csv（実 X で走らせた出力）から
exp23_fig_simulation.py を回すこと。
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SEED = 0
N_SPLITS = 5

# 予測時点（T0 = 2021-10-01）より前にのみ観測される特徴量の列名。
# 出席・レポート提出・担任とのやりとり・部活・アンケート・在籍情報から成る。
NUM_ASOF = [
    'att_cnt_all', 'att_st_all', 'att_rate_all', 'att_online_share', 'att_intensity',
    'att_rate_jul', 'att_rate_jun', 'att_rate_may', 'att_rate_apr',
    'att_zero_months', 'att_months',
    'rep_cnt', 'rep_subjects', 'rep_days_since_last', 'rep_30d', 'rep_90d',
    'rep_before90', 'rep_active_days',
    'tr_cnt', 'tr_out', 'tr_in', 'tr_out_ratio', 'tr_tools', 'tr_cats',
    'tr_days_since_last', 'tr_30d',
    'club_cnt', 'sv_sat', 'days_enrolled', 'age_months',
]
CAT_ASOF = [
    'school', 'admission_type', 'living_separation', 'gender', 'prefecture',
    'sv_bullying', 'sv_friends', 'sv_class_sat', 'sv_pride', 'sv_dif_friends',
    'sv_friend_intent',
]

# 代理カテゴリ列の水準数。実データのおおよその粒度に合わせてある。
CAT_LEVELS = {'school': 6, 'admission_type': 3, 'living_separation': 2,
              'gender': 3, 'prefecture': 47, 'sv_bullying': 4, 'sv_friends': 5,
              'sv_class_sat': 5, 'sv_pride': 5, 'sv_dif_friends': 4,
              'sv_friend_intent': 4}

N_COHORT = 2437      # 論文のリスク集合（T0 時点で通学系）と同じ人数


def oof_predict(make_model, X, s, n_splits=N_SPLITS, seed=SEED):
    """s を教師にして out-of-fold 予測を返す。返り値は P(s=1|x) の推定値。"""
    oof = np.zeros(len(X))
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, s):
        m = make_model()
        m.fit(X.iloc[tr], s[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof


def surrogate_features(n=N_COHORT, seed=SEED, n_factors=6):
    """代理の特徴量行列を作る。返り値は (X, Z)。

    数値列は共通因子 + 固有ノイズの因子モデルで作る（実データでも出席系・
    レポート系の列は互いに強く相関しているため）。そのうえで、率は [0,1]、
    回数は非負整数、というように列ごとの型を実データに合わせる。
    Z は exp22 側と同じ「数値列を標準化した行列」。
    """
    rng = np.random.default_rng(seed)
    d = len(NUM_ASOF)

    # --- 因子モデル -------------------------------------------------------
    F = rng.normal(size=(n, n_factors))
    Lo = rng.normal(size=(n_factors, d)) * (rng.random((n_factors, d)) < 0.45)
    U = F @ Lo + rng.normal(size=(n, d)) * 0.8
    U = (U - U.mean(0)) / U.std(0)

    # --- 列ごとに実データっぽい型へ写す -----------------------------------
    cols = {}
    for j, name in enumerate(NUM_ASOF):
        u = U[:, j]
        if 'rate' in name or 'share' in name or 'ratio' in name:
            cols[name] = 1 / (1 + np.exp(-u))                    # 率 → (0,1)
        elif name in ('sv_sat',):
            cols[name] = np.clip(np.round(3 + u), 1, 5)          # 5 件法
        elif name in ('days_enrolled',):
            cols[name] = np.clip(np.round(400 + 120 * u), 30, None)
        elif name in ('age_months',):
            cols[name] = np.clip(np.round(198 + 12 * u), 168, 260)
        elif 'days_since_last' in name:
            cols[name] = np.clip(np.round(30 * np.exp(u)), 0, 365)
        else:
            cols[name] = np.round(np.expm1(np.clip(u + 2.2, 0, 8)))   # 回数 → 非負整数
    X = pd.DataFrame(cols)

    # --- カテゴリ列。一部は因子に相関させる（実データでも学校差がある）----
    for i, name in enumerate(CAT_ASOF):
        k = CAT_LEVELS[name]
        lin = F[:, i % n_factors] + rng.normal(size=n) * 1.5
        lv = pd.qcut(lin.argsort().argsort(), k, labels=False)
        X[name] = pd.Series([f'{name[:3]}{v:02d}' for v in lv]).astype('category')

    # --- 欠測を実データ並みに入れる（数値列のみ、約 3%）--------------------
    for name in NUM_ASOF:
        X.loc[rng.random(n) < 0.03, name] = np.nan

    Zdf = X[NUM_ASOF].apply(pd.to_numeric, errors='coerce')
    Zdf = Zdf.fillna(Zdf.median())
    Z = ((Zdf - Zdf.mean()) / Zdf.std().replace(0, 1)).values
    return X, Z
