"""実験23: シミュレーション研究の図 (fig11 / fig12)。

fig11  純化の効き幅がどこから来るかの表面。
       左   実測できる成分間の予測しやすさの差 Δ̂ に対する改善幅（成分間相関 rho 別）。
            実データの配置（Δ̂=0.238, +0.117）を重ねる。
       中   混合陽性に占める「予測しにくい側」の割合 w_B に対する改善幅。
       右   窓を伸ばしたときの AUC。Δ>0 のときだけ単調に下がる（診断3の特異度）。

fig12  三つの診断が何を検出しているか。
       左   comp_gap（診断2）と c(x) 比（診断1）の散布図。2x2 の四条件で色分け。
       右   「純化が効くか」と「SCAR が破れているか」の二つの問いに対する検出力。
            各診断はどちらか一方にだけほぼ完全で、他方には無力。

軸ラベルは既存の thesis 図に合わせて英語。
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import pu_style

pu_style.use(); C = pu_style.COLORS
S = pd.read_csv('exp22_sweep.csv')
D = pd.read_csv('exp22_diag.csv')
W = pd.read_csv('exp22_window.csv')

# 実データの配置（PU 設定。6 節の表と同じ数値）
OBS = dict(delta=0.867 - 0.629, gain=0.867 - 0.750, comp_gap=144 / 279 - 5 / 44, cx=4.80)

RHO_COL = {0.0: C['proposed'], 0.4: C['leak'], 0.8: C['baseline']}
RHO_LAB = {0.0: r'$\rho = 0$  (disjoint drivers)',
           0.4: r'$\rho = 0.4$',
           0.8: r'$\rho = 0.8$  (fits our data)'}


def band(g, col):
    return (g.gain.mean(), np.percentile(g.gain, 2.5), np.percentile(g.gain, 97.5))


# ===== fig11 =============================================================
fig, ax = plt.subplots(1, 3, figsize=(5.645, 2.35))

# --- 左: gain vs Δ̂ -------------------------------------------------------
A = S[S.sweep == 'delta_x_rho']
for rho, g in A.groupby('rho'):
    m = g.groupby('delta_target').agg(x=('delta_realized', 'mean'), y=('gain', 'mean'),
                                      lo=('gain', lambda s: np.percentile(s, 2.5)),
                                      hi=('gain', lambda s: np.percentile(s, 97.5)))
    ax[0].fill_between(m.x, m.lo, m.hi, color=RHO_COL[rho], alpha=.13, lw=0)
    ax[0].plot(m.x, m.y, 'o-', ms=3, lw=1.2, color=RHO_COL[rho], label=RHO_LAB[rho])
ax[0].axhline(0, ls=':', lw=.8, color='k')
ax[0].plot(OBS['delta'], OBS['gain'], '*', ms=11, color=C['oracle'],
           mec='k', mew=.4, zorder=5, label='observed (this cohort)')
ax[0].set_xlabel(r'measured gap $\hat\Delta$ between components', fontsize=7)
ax[0].set_ylabel('ROC-AUC gain from purification', fontsize=7)
ax[0].legend(frameon=False, fontsize=5.2, loc='upper left')
ax[0].tick_params(labelsize=6.5)

# --- 中: gain vs w_B -----------------------------------------------------
B = S[S.sweep == 'delta_x_wB']
DCOL = {0.0: '#999999', 0.1: C['leak'], 0.2: C['proposed']}
for d, g in B.groupby('delta_target'):
    m = g.groupby('w_B').agg(y=('gain', 'mean'),
                             lo=('gain', lambda s: np.percentile(s, 2.5)),
                             hi=('gain', lambda s: np.percentile(s, 97.5)))
    ax[1].fill_between(m.index, m.lo, m.hi, color=DCOL[d], alpha=.13, lw=0)
    ax[1].plot(m.index, m.y, 'o-', ms=3, lw=1.2, color=DCOL[d],
               label=rf'$\Delta = {d:.2f}$')
ax[1].axhline(0, ls=':', lw=.8, color='k')
ax[1].axvline(336 / 532, ls='--', lw=.8, color=C['oracle'])
ax[1].set_ylim(-.02, .235)
ax[1].text(336 / 532 - .015, .225, 'ours', ha='right', va='top',
           fontsize=5.5, color=C['oracle'])
ax[1].set_xlabel('share of bundle contributed by\nthe less predictable event', fontsize=7)
ax[1].set_ylabel('ROC-AUC gain', fontsize=7)
ax[1].legend(frameon=False, fontsize=5.5, loc='upper left')
ax[1].tick_params(labelsize=6.5)

# --- 右: 窓長 ------------------------------------------------------------
for d, g in W.groupby('delta_target'):
    m = g.groupby('window').auc.agg(['mean', lambda s: np.percentile(s, 2.5),
                                     lambda s: np.percentile(s, 97.5)])
    m.columns = ['y', 'lo', 'hi']
    col = DCOL[0.0] if d == 0 else (DCOL[0.1] if d < .15 else DCOL[0.2])
    ax[2].fill_between(m.index, m.lo, m.hi, color=col, alpha=.13, lw=0)
    ax[2].plot(m.index, m.y, 'o-', ms=3, lw=1.2, color=col,
               label=rf'$\Delta = {d:.2f}$')
ax[2].set_xlabel('observation window (months)', fontsize=7)
ax[2].set_ylabel('ROC-AUC', fontsize=7)
ax[2].legend(frameon=False, fontsize=5.5, loc='lower left')
ax[2].tick_params(labelsize=6.5)

fig.suptitle('What the purification gain depends on', fontsize=8)
fig.tight_layout()
fig.savefig('fig11_sim_surface.pdf'); fig.savefig('fig11_sim_surface.png', dpi=150)

# ===== fig12 =============================================================
D['gap'] = D.delta_target > 0
D['tim'] = D.c_B < 0.9
CELL = [(True,  True,  C['proposed'], 'o', 'gap + timing difference'),
        (True,  False, C['leak'],     's', 'gap only'),
        (False, True,  C['baseline'], '^', 'timing difference only'),
        (False, False, C['random'],   'x', 'neither')]

fig, ax = plt.subplots(1, 2, figsize=(5.645, 2.35))

for gp, tm, col, mk, lab in CELL:
    d = D[(D.gap == gp) & (D.tim == tm)]
    ax[0].scatter(d.cx_ratio, d.comp_gap, s=11, c=col, marker=mk, lw=.7,
                  alpha=.8, label=lab)
ax[0].axhline(.40, ls='--', lw=.8, color='k')
ax[0].text(1.02, .77, 'purification pays above this line', fontsize=5.2, ha='left')
ax[0].plot(OBS['cx'], OBS['comp_gap'], '*', ms=11, color=C['oracle'],
           mec='k', mew=.4, zorder=5)
ax[0].annotate('observed\ncohort', xy=(OBS['cx'], OBS['comp_gap']),
               xytext=(0, -9), textcoords='offset points', ha='center',
               va='top', fontsize=5.5, color=C['oracle'])
ax[0].set_xscale('log')
ax[0].set_xticks([1, 1.5, 2, 3, 5], ['1', '1.5', '2', '3', '5'])
ax[0].set_xticks([], minor=True)
ax[0].set_ylim(-.72, .92)
ax[0].set_xlabel(r'Diagnostic 1: $c(x)$ spread (max/min)', fontsize=7)
ax[0].set_ylabel('Diagnostic 2: composition gap Q5 $-$ Q1', fontsize=7)
ax[0].tick_params(labelsize=6.5)

d = D.dropna(subset=['comp_gap', 'cx_ratio'])
Q = {'Does purification pay?': d.gap.astype(int), 'Is SCAR violated?': d.tim.astype(int)}
T = {'Diagnostic 1: $c(x)$ spread': d.cx_ratio, 'Diagnostic 2: composition': d.comp_gap}
x = np.arange(len(Q)); w = .34
for j, (tn, tv) in enumerate(T.items()):
    v = [roc_auc_score(y, tv) for y in Q.values()]
    ax[1].bar(x + (j - .5) * w, v, w, label=tn,
              color=[C['baseline'], C['proposed']][j])
    for xi, vi in zip(x + (j - .5) * w, v):
        ax[1].text(xi, vi + .015, f'{vi:.2f}', ha='center', fontsize=6)
ax[1].axhline(.5, ls=':', lw=.8, color='k')
ax[1].set_xlim(-.62, 1.5)
ax[1].text(-.60, .53, 'chance', fontsize=5.5, ha='left', va='bottom')
ax[1].set_xticks(x, list(Q), fontsize=6.5)
ax[1].set_ylim(0, 1.38); ax[1].set_ylabel('detection ROC-AUC', fontsize=7)
ax[1].legend(frameon=False, fontsize=5.5, loc='upper center')
ax[1].tick_params(labelsize=6.5)

fig.suptitle('The two diagnostics answer two different questions', fontsize=8)
# 凡例は軸の外（図の下端）へ。軸の中に置くと緑の星がデータ点に見える。
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=5.8,
           handletextpad=.2, columnspacing=1.1, bbox_to_anchor=(.5, -.005))
fig.tight_layout(rect=(0, .075, 1, 1))
fig.savefig('fig12_sim_diagnostics.pdf'); fig.savefig('fig12_sim_diagnostics.png', dpi=150)
print('fig11_sim_surface / fig12_sim_diagnostics を書き出した。')
