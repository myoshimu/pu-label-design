"""論文図の共通設定 (親ディレクトリの style_config.py と同じパレットを使う)。"""
import os
import matplotlib.pyplot as plt

MPLSTYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thesis_style.mplstyle')

# 親の style_config.COLORS と同じ色を、本稿の比較軸に割り当て直したもの
COLORS = {
    'baseline': '#0072B2',   # 従来手法 (naive / Elkan-Noto)
    'proposed': '#D55E00',   # 提案 (ハザードモデル)
    'oracle':   '#009E73',   # 完全ラベル学習 (到達上限)
    'random':   '#999999',   # ランダム基準
    'leak':     '#CC79A7',   # リークを含む参考値
}


def use():
    plt.style.use(MPLSTYLE)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'
