"""実験22: 事象混合のシミュレーション研究。

本研究の中心的な主張は「ラベル純化が +0.109 で最大の施策だった」だが、
その +0.109 という数値そのものは本コホート固有である。
他機関で同じ点検をした人が得る改善幅は分からない。

そこで、**特徴量分布は実データのまま固定し、ラベルの作り方だけを人工的に
制御して**、次の二点を測る。

  (1) 純化による ROC-AUC の改善幅は、束ねられた二事象の「予測しやすさの差」
      Δ の関数としてどう動くか。実測の +0.098（純化のうち事象分離の寄与分）は
      その関数の上に乗るか。
  (2) 本稿の三つの診断（c(x) のリスク分位変動 / 陽性の事象構成 / 窓長）は、
      混合を検出できるか。そして混合がないときに誤検出しないか（特異度）。

生成モデル
----------
実コホート（T0 時点で通学系の 2,437 名）の as-of 特徴量行列 X をそのまま使い、
ラベルだけを次のように合成する。

  z(x)              数値特徴量を標準化したもの
  eta_A = w_A^T z   事象 A の潜在リスク（標準化）
  eta_B = rho*eta_A + sqrt(1-rho^2)*eta_perp   事象 B の潜在リスク
  P(A=1|x) = sigmoid(a_A + b_A*eta_A),  P(B=1|x) = sigmoid(a_B + b_B*eta_B)
  y_mix = A or B                    （実データでも A と B の重複は 0 件）
  s_A = A * Bern(c_A),  s_B = B * Bern(c_B),  s_mix = s_A or s_B

(a_k, b_k) は「事象 k 単独での到達可能 ROC-AUC（ベイズ AUC）」と「発生率」の
二つを狙って数値的に解く。**成分内は SCAR（c_k は x に依存しない）**である点が
要で、混合ラベルで観測される SCAR の破れは、成分ごとに c が違うことだけから
生じる。本稿の主張の生成モデル版がこれにあたる。

実データから取った定数（下の EMP）:
  発生率     y_A(ネット転換) 196/2437 = .080、y_B(日数変更) 336/2437 = .138、重複 0
  ラベル頻度 c_A = 0.990、c_B = 0.458
  到達 AUC   A = 0.868、B = 0.664（差 Δ = 0.204）
  rho        実測の成分間リスク相関は spearman 0.009 ≒ 0（別々の要因で動く）

出力: exp22_sweep.csv / exp22_diag.csv / exp22_window.csv

公開版についての注記
--------------------
上の設計では特徴量行列 X に実コホートのものを使うが、X はこのリポジトリ
には同梱していない。このスクリプトは既定で simlib.surrogate_features() が作る
代理行列を使う（行数・列数・数値列の相関構造だけを模したもの）。
再現できるのは図 11・図 12 の定性的な形であって、論文中の個々の数値では
ない。論文の図そのものは、同梱の exp22_*.csv から exp23_fig_simulation.py
を回せば再現できる。
"""
import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

import simlib as P

SEED = 0
N_JOBS = 4

# ---- 実データから取った定数 -------------------------------------------
EMP = dict(pi_mix=(196 + 336) / 2437,   # 0.2183 混合ラベルの発生率
           w_B=336 / (196 + 336),       # 0.6316 混合陽性に占める B の割合
           c_A=0.990, c_B=0.458,
           auc_A=0.868, auc_B=0.664)
EMP['delta'] = EMP['auc_A'] - EMP['auc_B']   # 0.204

# 到達可能 AUC（ベイズ AUC）と、有限標本で実際に当てられる AUC の差。
# 事前掃引（tau 7 水準 × 発生率 4 水準 × 5 反復 = 140 本）での実測は 0.014〜0.100、
# 平均 0.037。単純に一定値として目標に上乗せする。これで合成事象 A の実現 AUC が
# 実データのネット転換（0.868）とほぼ揃う。
SHRINK = 0.04

# 成分間のリスク相関 rho は直接は測れない唯一の自由パラメータなので、
# 混合ラベルでの実現 AUC（実データ 0.750）を再現する値に合わせる。
# rho=0 で 0.697、0.5 で 0.733、0.9 で 0.761 なので 0.8 前後。
RHO_FIT = 0.8


def sim_lgbm():
    """label_lib.lgbm と同一設定。並列化のため n_jobs だけ 1 に落とす。"""
    return lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=3,
                              min_child_samples=40, subsample=.8, subsample_freq=1,
                              colsample_bytree=.6, reg_lambda=5.,
                              random_state=SEED, n_jobs=1, verbose=-1)


# ---- ベイズ AUC と、それを狙った係数の逆解き ----------------------------
def bayes_auc(eta, p):
    """スコア eta で順位付けしたときの、ラベル ~ Bern(p) に対する期待 ROC-AUC。

    Σ_i p_i * (Σ_{j: eta_j < eta_i} (1-p_j)) / [(Σp)(Σ(1-p)) - Σ p(1-p)]
    を O(n log n) で。y_i と y_j の独立性を使っているだけで近似はない。
    """
    o = np.argsort(eta, kind='mergesort')
    ps, qs = p[o], 1.0 - p[o]
    below = np.concatenate([[0.0], np.cumsum(qs)[:-1]])
    num = float(ps @ below)
    den = float(ps.sum() * qs.sum() - (ps * qs).sum())
    return num / den


def _solve_a(eta, b, prev):
    """平均発生率が prev になる切片 a を二分法で求める。"""
    lo, hi = -30.0, 30.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(mid + b * eta)))).mean() < prev:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def calibrate(eta, target_auc, prev):
    """(到達可能 AUC, 発生率) を狙って (a, b) を解く。AUC は b について単調増加。"""
    lo, hi = 0.0, 40.0
    for _ in range(40):
        mid = (lo + hi) / 2
        a = _solve_a(eta, mid, prev)
        if bayes_auc(eta, 1 / (1 + np.exp(-(a + mid * eta)))) < target_auc:
            lo = mid
        else:
            hi = mid
    b = (lo + hi) / 2
    return _solve_a(eta, b, prev), b


# ---- 潜在リスクとラベルの生成 ------------------------------------------
def latents(Z, rho, rng):
    """疎なランダム線形結合で二本の潜在リスクを作る。相関を rho に合わせる。"""
    d = Z.shape[1]
    def draw():
        w = rng.normal(size=d) * (rng.random(d) < 0.5)   # 半数を 0 にした疎な係数
        if not w.any():
            w[rng.integers(d)] = 1.0
        e = Z @ w
        return (e - e.mean()) / e.std()
    eA, ep = draw(), draw()
    ep = ep - (ep @ eA) / (eA @ eA) * eA                 # eA と直交化
    ep = (ep - ep.mean()) / ep.std()
    eB = rho * eA + np.sqrt(max(1 - rho ** 2, 0.0)) * ep
    return eA, (eB - eB.mean()) / eB.std()


def make_labels(Z, rng, auc_A, auc_B, pi_mix, w_B, c_A, c_B, rho):
    """A / B と、その混合ラベル・観測ラベルを引く。"""
    eA, eB = latents(Z, rho, rng)
    piA, piB = pi_mix * (1 - w_B), pi_mix * w_B
    aA, bA = calibrate(eA, auc_A, piA)
    aB, bB = calibrate(eB, auc_B, piB)
    pA = 1 / (1 + np.exp(-(aA + bA * eA)))
    pB = 1 / (1 + np.exp(-(aB + bB * eB)))
    A = (rng.random(len(pA)) < pA).astype(int)
    B = (rng.random(len(pB)) < pB).astype(int)
    B[A == 1] = 0                        # 実データでも重複は 0 件
    sA = A * (rng.random(len(A)) < c_A)
    sB = B * (rng.random(len(B)) < c_B)
    return dict(A=A, B=B, y_mix=np.maximum(A, B),
                s_A=sA.astype(int), s_B=sB.astype(int),
                s_mix=np.maximum(sA, sB).astype(int), pA=pA, pB=pB)


# ---- 評価と三つの診断 ---------------------------------------------------
def oof(X, s):
    return P.oof_predict(sim_lgbm, X, np.asarray(s))


def diagnostics(g, y_mix, s_mix, A):
    """混合ラベルの上で本稿の診断 (a)(b) を回す。実務者が実際に打てる検査。"""
    t = pd.DataFrame({'g': g, 'y': y_mix, 's': s_mix, 'A': A})
    t['q'] = pd.qcut(t.g.rank(method='first'), 5, labels=False)
    agg = t.groupby('q').apply(lambda d: pd.Series({
        'y': d.y.sum(), 's': d.s.sum(), 'A': d[d.y == 1].A.sum()}))
    cx = (agg.s / agg.y.replace(0, np.nan))
    share = (agg.A / agg.y.replace(0, np.nan))          # 陽性に占める A の割合
    return dict(cx_min=cx.min(), cx_max=cx.max(),
                cx_ratio=cx.max() / cx.min() if cx.min() and cx.min() > 0 else np.nan,
                comp_q1=share.iloc[0], comp_q5=share.iloc[-1],
                comp_gap=share.iloc[-1] - share.iloc[0])


def one_rep(X, Z, rep, auc_A, delta, w_B, c_A, c_B, rho, do_diag=True):
    """auc_A / delta は「実現させたい AUC」で指定する（SHRINK を内部で上乗せする）。"""
    rng = np.random.default_rng(1000 * rep + 7)
    L = make_labels(Z, rng, auc_A + SHRINK, auc_A - delta + SHRINK,
                    EMP['pi_mix'], w_B, c_A, c_B, rho)
    if L['A'].sum() < 25 or L['B'].sum() < 25 or L['s_mix'].sum() < 25:
        return None
    g_mix = oof(X, L['s_mix'])
    g_A = oof(X, L['s_A'])
    g_B = oof(X, L['s_B'])
    out = dict(rep=rep, delta_target=delta, w_B=w_B, rho=rho, c_A=c_A, c_B=c_B,
               n_A=int(L['A'].sum()), n_B=int(L['B'].sum()),
               n_pos_mix=int(L['y_mix'].sum()), n_s_mix=int(L['s_mix'].sum()),
               auc_mix=roc_auc_score(L['y_mix'], g_mix),
               auc_A=roc_auc_score(L['A'], g_A),
               auc_B=roc_auc_score(L['B'], g_B),
               bayes_A=bayes_auc(L['pA'], L['pA']), bayes_B=bayes_auc(L['pB'], L['pB']))
    out['delta_realized'] = out['auc_A'] - out['auc_B']
    out['gain'] = out['auc_A'] - out['auc_mix']
    if do_diag:
        out.update(diagnostics(g_mix, L['y_mix'], L['s_mix'], L['A']))
    return out


def run(grid, X, Z, tag):
    t0 = time.time()
    res = Parallel(n_jobs=N_JOBS, verbose=0)(delayed(one_rep)(X, Z, **cfg) for cfg in grid)
    res = [r for r in res if r]
    print(f'  [{tag}] {len(res)}/{len(grid)} 本  {time.time() - t0:.0f} 秒')
    return pd.DataFrame(res)


# ---- 窓長の診断 ---------------------------------------------------------
def window_rep(X, Z, rep, delta, months=(1, 2, 3, 5, 8, 12)):
    """窓を伸ばすと何が起きるか。A は初月に集中、B は年間に散る。

    窓長 L を伸ばすほど追加される陽性は B に偏るので、Δ>0 なら AUC は下がる。
    Δ=0 ならほぼ平坦のはず（特異度の確認）。
    """
    rng = np.random.default_rng(5000 * rep + 11)
    L = make_labels(Z, rng, EMP['auc_A'] + SHRINK, EMP['auc_A'] - delta + SHRINK,
                    EMP['pi_mix'], EMP['w_B'], EMP['c_A'], EMP['c_B'], RHO_FIT)
    n = len(X)
    mA = np.where(rng.random(n) < EMP['c_A'], 1, rng.integers(2, 13, n))
    mB = np.where(rng.random(n) < EMP['c_B'], 1, rng.integers(2, 13, n))
    rows = []
    for Lw in months:
        y = np.maximum(L['A'] * (mA <= Lw), L['B'] * (mB <= Lw)).astype(int)
        if y.sum() < 25:
            continue
        rows.append(dict(rep=rep, delta_target=delta, window=Lw, n_pos=int(y.sum()),
                         share_A=float(L['A'][y == 1].mean()),
                         auc=roc_auc_score(y, oof(X, y))))
    return rows


def main():
    X, Z = P.surrogate_features(seed=SEED)
    print('代理特徴量行列を使用（実データは同梱していない。README を参照）')
    print(f'コホート {len(X)} 名 / 特徴量 {X.shape[1]} 列（潜在リスクは数値 {Z.shape[1]} 列から構成）')
    print(f'実測の基準値: Δ={EMP["delta"]:.3f}, w_B={EMP["w_B"]:.3f}, '
          f'c_A={EMP["c_A"]}, c_B={EMP["c_B"]}\n')

    REPS = 20
    A0 = EMP['auc_A']
    DELTAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)

    # --- 掃引 A: 予測しやすさの差 Δ × 成分間リスク相関 rho（主表面）------
    print('=== 掃引A: Δ × rho（w_B と c は実測値に固定）===')
    gA = [dict(rep=r, auc_A=A0, delta=d, w_B=EMP['w_B'],
               c_A=EMP['c_A'], c_B=EMP['c_B'], rho=rho)
          for d in DELTAS for rho in (0.0, 0.4, RHO_FIT) for r in range(REPS)]
    SA = run(gA, X, Z, 'sweepA')
    SA['sweep'] = 'delta_x_rho'

    # --- 掃引 B: 混合比 w_B（rho は実データに合わせた値に固定）------------
    print('=== 掃引B: Δ × w_B（rho=RHO_FIT）===')
    gB = [dict(rep=r, auc_A=A0, delta=d, w_B=w,
               c_A=EMP['c_A'], c_B=EMP['c_B'], rho=RHO_FIT)
          for d in (0.0, 0.10, 0.20) for w in (0.20, 0.40, 0.80) for r in range(REPS)]
    SB = run(gB, X, Z, 'sweepB')
    SB['sweep'] = 'delta_x_wB'

    pd.concat([SA, SB], ignore_index=True).to_csv('exp22_sweep.csv', index=False)

    # --- 研究2: 診断の感度と特異度（Δ × タイミング差の 2x2）--------------
    print('=== 研究2: 診断の 2x2（Δ あり/なし × タイミング差 あり/なし）===')
    g3 = [dict(rep=r, auc_A=A0, delta=d, w_B=EMP['w_B'],
               c_A=EMP['c_A'], c_B=cb, rho=RHO_FIT)
          for d in (0.0, EMP['delta'])
          for cb in (EMP['c_A'], EMP['c_B'])
          for r in range(30)]
    S3 = run(g3, X, Z, 'diag2x2')
    S3.to_csv('exp22_diag.csv', index=False)

    # --- 研究3: 窓長 ------------------------------------------------------
    print('=== 研究3: 窓長を伸ばしたときの AUC ===')
    t0 = time.time()
    g4 = [(r, d) for d in (0.0, 0.10, EMP['delta']) for r in range(15)]
    W = Parallel(n_jobs=N_JOBS)(delayed(window_rep)(X, Z, r, d) for r, d in g4)
    W = pd.DataFrame([x for rows in W for x in rows])
    print(f'  [window] {len(W)} 行  {time.time() - t0:.0f} 秒')
    W.to_csv('exp22_window.csv', index=False)

    # --- 要約 -------------------------------------------------------------
    def ci(s):
        return f'{s.mean():.3f} [{np.percentile(s, 2.5):.3f}, {np.percentile(s, 97.5):.3f}]'

    print('\n=== 掃引A 要約: 純化による ROC-AUC 改善（行 Δ × 列 rho）===')
    print(SA.pivot_table(index='delta_target', columns='rho', values='gain').round(3).to_string())
    print('\n同 実現した混合ラベルの AUC:')
    print(SA.pivot_table(index='delta_target', columns='rho', values='auc_mix').round(3).to_string())

    print('\n実測配置 (Δ=0.20, w_B=0.632, rho=RHO_FIT) のセル:')
    cell = SA[(SA.delta_target == 0.20) & (SA.rho == RHO_FIT)]
    print(f'  gain      = {ci(cell.gain)}   （実データ PU 設定 0.750→0.867 は +0.117）')
    print(f'  Δ_realized= {ci(cell.delta_realized)}   （実データは 0.238）')
    print(f'  auc_mix   = {ci(cell.auc_mix)}   （実データは 0.750）')
    print(f'  auc_A     = {ci(cell.auc_A)}   （実データは 0.867）')

    print('\n=== 掃引A: Δ=0（束ねても両成分が同じ予測しやすさ）===')
    for rho, z in SA[SA.delta_target == 0.0].groupby('rho'):
        print(f'  rho={rho:.1f}  gain = {ci(z.gain)}')

    print('\n=== 掃引B 要約: 行 Δ × 列 w_B（rho={:.1f}）==='.format(RHO_FIT))
    print(SB.pivot_table(index='delta_target', columns='w_B', values='gain').round(3).to_string())

    print('\n=== 研究2: 三つの診断の 2x2 ===')
    S3['cell'] = np.where(S3.delta_target > 0, 'Δあり', 'Δなし') + '×' + \
                 np.where(S3.c_B < 0.9, 'タイミング差あり', 'タイミング差なし')
    print(S3.groupby('cell')[['cx_ratio', 'comp_gap', 'gain']]
            .agg(['mean', 'std']).round(3).to_string())

    print('\n=== 研究3: 窓長 ===')
    print(W.pivot_table(index='window', columns='delta_target',
                        values=['auc', 'share_A']).round(3).to_string())
    print('\nexp22_sweep.csv / exp22_diag.csv / exp22_window.csv を書き出した。')


if __name__ == '__main__':
    main()
