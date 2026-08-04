"""可跑的 demo：从 S = k log W 到 gpt.py 的 softmax —— 一条线。

运行:
    python docs/examples/boltzmann_machine_demo.py

一路推下来，每一步都是上一步的直接后果：

    1.  1877 玻尔兹曼   S = k log W，以及 p ∝ exp(-E/kT) 是怎么冒出来的
    2.  1982 Hopfield   把 exp(-E) 里的 E 换成 -½sᵀWs，得到一台确定性的能量下降机
    3.  1985 Hinton     把"下降"换成"按玻尔兹曼分布采样"，模拟退火跳出局部极小
    3b. 名字的由来      测出来：这台机器的平稳分布**精确等于** exp(-E/T)/Z
    4a. "受限"是什么     两个恒等式的数值验证：Σ_h 可解析求和、p(h|v) 可分解
    4.  2002 CD-1       受限玻尔兹曼机的学习律：正相 - 负相（数据项 - 模型项）
    5.  今天            codechat/gpt.py:141-146 的 logits/temperature + softmax
                        就是同一个 exp(-E/kT)/Z

产出（写到 docs/images/）:
    boltzmann_01_entropy_and_distribution.png
    boltzmann_02_hopfield_vs_boltzmann.png
    boltzmann_02b_why_the_name.png
    boltzmann_03_rbm_cd1.png
    boltzmann_04_softmax_is_boltzmann.png

配文：docs/boltzmann_to_hinton.md

说明：
  * 第 5 部分的采样代码和 codechat/gpt.py:141-146 逐行同构（同样的
    logits/temperature → top-k → softmax → multinomial）。
  * 第 1-4 部分是玩具尺寸（16 个可见单元、8 个隐单元），目的是把公式跑出数来，
    不是复现任何论文的实验规模。
无显示环境也能跑（Agg 后端），全程在终端打印 ASCII 版本。
"""
import math
import os

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  # 无头环境
import matplotlib.pyplot as plt

# 让中文能正常显示（按可用性回退；找不到就退回 DejaVu，中文会变方框但不报错）
matplotlib.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "Hiragino Sans GB", "STHeiti", "Heiti TC",
    "PingFang HK", "Songti SC", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.abspath(os.path.join(HERE, "..", "images"))
os.makedirs(IMG, exist_ok=True)

torch.manual_seed(0)


def title(n, s):
    print("\n" + "=" * 78)
    print(f"[{n}] {s}")
    print("=" * 78)


def bar(x, width=40, ch="#"):
    return ch * int(round(x * width))


# ---------------------------------------------------------------------------
# 1. 1877：S = k log W，以及玻尔兹曼分布
# ---------------------------------------------------------------------------
def part1_entropy_and_distribution():
    title(1, "1877 玻尔兹曼：S = k log W —— 熵是在数数")

    # 1a. N 个可分辨粒子分到左右两个盒子。宏观态 = "左边有 n 个"。
    #     微观态数 W(n) = C(N, n)。熵 S = ln W。
    N = 20
    ns = torch.arange(0, N + 1)
    logW = torch.tensor([math.lgamma(N + 1) - math.lgamma(int(n) + 1) - math.lgamma(N - int(n) + 1)
                         for n in ns])
    W = logW.exp()

    print(f"\nN = {N} 个粒子分进左右两盒，宏观态用「左边 n 个」描述：")
    print(f"{'n':>4} {'W(n)=C(N,n)':>14} {'S=ln W':>9}  {'':<42}")
    for n in range(0, N + 1, 2):
        print(f"{n:>4} {int(W[n].item()):>14,} {logW[n].item():>9.3f}  {bar(logW[n].item() / logW.max().item())}")

    p_ordered = (W[0] / W.sum()).item()     # 全部跑到左边
    p_equil = (W[N // 2] / W.sum()).item()  # 均分
    print(f"\n  全部在左边（'有序'）的概率：{p_ordered:.3e}")
    print(f"  正好均分（'平衡'）的概率  ：{p_equil:.3e}")
    print(f"  倍数：{p_equil / p_ordered:,.0f} 倍")
    print("  → 第二定律不是一条力学定律，是一句数数的结论：系统往微观态多的宏观态跑。")
    print("    这就是玻尔兹曼墓碑上的 S = k log W。")

    # 1b. 从「最大化 S，约束总能量」得到 p_i ∝ exp(-E_i / kT)
    #     这里直接验证：给定能级，玻尔兹曼分布在同样平均能量下熵最大。
    E = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    temps = [0.2, 0.5, 1.0, 2.0, 5.0, 100.0]
    print("\n玻尔兹曼因子 p_i = exp(-E_i/T) / Z，同一组能级在不同温度下的占据：")
    print(f"{'T':>7} | " + " ".join(f"E={float(e):.0f}" for e in E) + "   |   S=-Σp·lnp")
    rows = []
    for T in temps:
        p = F.softmax(-E / T, dim=0)          # ← 注意：softmax(-E/T) 就是玻尔兹曼分布
        S = max(-(p * p.clamp_min(1e-12).log()).sum().item(), 0.0)
        rows.append((T, p, S))
        print(f"{T:>7.1f} | " + " ".join(f"{v:.3f}" for v in p) + f"   |   {S:.4f}")
    print(f"\n  T → 0   ：全部塌到基态，S → 0        （= 贪心解码 / argmax）")
    print(f"  T → ∞   ：均匀分布，S → ln 5 = {math.log(5):.4f}  （= 完全随机）")
    print("  中间的每一个 T 都是「能量」和「熵」的一次配比。自由能 F = E - T·S 取极小。")

    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.bar(ns.numpy(), W.numpy(), color="#4c72b0")
    ax.set_title("① 微观态计数 W(n) = C(20, n)")
    ax.set_xlabel("左盒粒子数 n"); ax.set_ylabel("W(n)")
    ax.annotate("熵最大 = 微观态最多\n= 「平衡态」",
                xy=(10, W[10].item()), xytext=(13, W[10].item() * 0.75),
                arrowprops=dict(arrowstyle="->", color="crimson"), color="crimson", fontsize=9)

    ax = axes[1]
    for T, p, S in rows[:5]:
        ax.plot(E.numpy(), p.numpy(), "o-", label=f"T={T}")
    ax.set_title("② 玻尔兹曼分布 p ∝ exp(−E/T)")
    ax.set_xlabel("能量 E"); ax.set_ylabel("占据概率 p"); ax.legend(fontsize=8)

    ax = axes[2]
    Tsweep = torch.logspace(-1, 2, 120)
    Ss, Es = [], []
    for T in Tsweep:
        p = F.softmax(-E / T, dim=0)
        Ss.append(-(p * p.clamp_min(1e-12).log()).sum().item())
        Es.append((p * E).sum().item())
    ax.semilogx(Tsweep.numpy(), Ss, label="熵 S", color="#c44e52")
    ax.semilogx(Tsweep.numpy(), Es, label="平均能量 <E>", color="#4c72b0")
    ax.axhline(math.log(5), ls="--", c="gray", lw=1)
    ax.text(0.12, math.log(5) + 0.05, "ln 5（均匀上限）", fontsize=8, color="gray")
    ax.set_title("③ 温度调的是「熵 ↔ 能量」的配比")
    ax.set_xlabel("温度 T"); ax.legend(fontsize=9)

    fig.suptitle("1877 → 玻尔兹曼：熵是数数，分布是 exp(−E/T)/Z", fontsize=13)
    fig.tight_layout()
    out = os.path.join(IMG, "boltzmann_01_entropy_and_distribution.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\n  图已保存：{out}")


# ---------------------------------------------------------------------------
# 2 & 3. 1982 Hopfield（确定性下降） vs 1985 玻尔兹曼机（随机 + 退火）
# ---------------------------------------------------------------------------
PATTERNS = torch.tensor([
    # 4x4 的三个"记忆"，展平成 16 维 ±1 向量
    [1, 1, 1, 1,  -1, -1, -1, -1,  1, 1, 1, 1,  -1, -1, -1, -1],   # 横条
    [1, -1, 1, -1,  1, -1, 1, -1,  1, -1, 1, -1,  1, -1, 1, -1],   # 竖条
    [1, 1, -1, -1,  1, 1, -1, -1,  -1, -1, 1, 1,  -1, -1, 1, 1],   # 棋盘块
], dtype=torch.float32)


def hebb_weights(P):
    """Hebb 规则：W = (1/N) Σ_μ ξ^μ (ξ^μ)ᵀ，对角清零。1982 Hopfield 论文的存储律。"""
    N = P.shape[1]
    W = (P.t() @ P) / N
    W.fill_diagonal_(0.0)
    return W


def energy(s, W):
    """E(s) = -½ sᵀ W s。Hopfield/玻尔兹曼机共用的同一个能量函数（Ising 模型）。"""
    return -0.5 * (s @ W @ s)


def hopfield_settle(s, W, max_sweeps=50):
    """确定性异步更新：s_i ← sign(Σ_j W_ij s_j)。能量单调不增 → 必收敛到局部极小。"""
    s = s.clone()
    traj = [energy(s, W).item()]
    for _ in range(max_sweeps):
        changed = False
        for i in torch.randperm(s.numel()):
            h = W[i] @ s
            new = 1.0 if h >= 0 else -1.0
            if new != s[i]:
                s[i] = new
                changed = True
                traj.append(energy(s, W).item())
        if not changed:
            break
    return s, traj


def boltzmann_settle(s, W, T_hi=2.0, T_lo=0.05, sweeps=60):
    """玻尔兹曼机：同样的 W，但翻转是随机的。

    p(s_i = +1) = σ(2·h_i / T)，即按 exp(-E/T) 采样（Gibbs sampling）。
    T 从高到低退火 = Kirkpatrick 1983 的模拟退火，也是 Hinton & Sejnowski 1985
    用来让网络找到全局极小的机制。T → 0 时它退化成上面的 Hopfield。
    """
    s = s.clone()
    traj = [energy(s, W).item()]
    for k in range(sweeps):
        T = T_hi * (T_lo / T_hi) ** (k / max(sweeps - 1, 1))   # 几何退火
        for i in torch.randperm(s.numel()):
            h = W[i] @ s
            p_up = torch.sigmoid(2.0 * h / T)
            s[i] = 1.0 if torch.rand(()) < p_up else -1.0
        traj.append(energy(s, W).item())
    return s, traj


def part23_hopfield_vs_boltzmann():
    title(2, "1982 Hopfield：能量下降机 —— 确定性，但会卡在局部极小")
    W = hebb_weights(PATTERNS)
    N = PATTERNS.shape[1]

    print("\n存了 3 个 16 位模式（Hebb 外积），它们的能量：")
    for i, p in enumerate(PATTERNS):
        print(f"  模式 {i}: E = {energy(p, W).item():+.4f}")

    # 全局最低能量：16 位穷举（65536 个态，跑得动）
    alls = torch.tensor([[1.0 if (m >> b) & 1 else -1.0 for b in range(N)] for m in range(1 << N)])
    allE = -0.5 * ((alls @ W) * alls).sum(1)
    Emin = allE.min().item()
    n_global = int((allE <= Emin + 1e-6).sum().item())
    print(f"\n穷举 2^16 = 65536 个状态：全局最低能量 E* = {Emin:.4f}，共 {n_global} 个态达到它")
    print(f"  能量分布：min={allE.min():.3f}  median={allE.median():.3f}  max={allE.max():.3f}")

    title(3, "1985 Hinton & Sejnowski：把 sign() 换成 exp(−ΔE/T) 采样")
    print("\n同一个 W，同一个 E(s) = -½sᵀWs，只改一行更新规则：")
    print("    Hopfield  : s_i ← sign(h_i)                   （确定性，只下山）")
    print("    玻尔兹曼机 : s_i ← +1 with p = σ(2·h_i / T)     （随机，允许上山）")
    print("\n从 200 个随机初态出发，看谁能走到全局最低能量：")

    trials = 200
    torch.manual_seed(7)
    inits = torch.where(torch.rand(trials, N) < 0.5, -1.0, 1.0)

    hop_E, bm_E = [], []
    hop_traj = bm_traj = None
    stuck_at = float("nan")
    for t in range(trials):
        sh, th = hopfield_settle(inits[t], W)
        sb, tb = boltzmann_settle(inits[t], W)
        hop_E.append(energy(sh, W).item())
        bm_E.append(energy(sb, W).item())
        # 挑第一个「Hopfield 卡住、玻尔兹曼机走到底」的例子来画图
        if hop_traj is None and hop_E[-1] > Emin + 1e-6 and bm_E[-1] <= Emin + 1e-6:
            hop_traj, bm_traj, stuck_at = th, tb, hop_E[-1]

    hop_E_t, bm_E_t = torch.tensor(hop_E), torch.tensor(bm_E)
    hop_hit = (hop_E_t <= Emin + 1e-6).float().mean().item()
    bm_hit = (bm_E_t <= Emin + 1e-6).float().mean().item()

    print("\n  更新规则              命中全局极小   平均终态能量")
    print(f"  Hopfield（确定性）     {hop_hit * 100:>9.1f}%   {hop_E_t.mean().item():>11.4f}")
    print(f"  玻尔兹曼机（退火）     {bm_hit * 100:>9.1f}%   {bm_E_t.mean().item():>11.4f}")
    print(f"\n  → 随机性不是噪声，是搜索工具。这是 Hinton 1985 相对 Hopfield 1982 的第一处关键改动。")

    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.plot(hop_traj, lw=1.6, color="#4c72b0", label=f"Hopfield：单调下降，卡在 E={stuck_at:.2f}")
    ax.axhline(Emin, ls="--", c="crimson", lw=1, label=f"全局极小 E*={Emin:.2f}")
    ax.set_title("① Hopfield：只下山 → 停在伪极小")
    ax.set_xlabel("异步翻转次数"); ax.set_ylabel("E(s)"); ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(bm_traj, lw=1.2, color="#c44e52", label="玻尔兹曼机：高温时允许上山")
    ax.axhline(Emin, ls="--", c="crimson", lw=1)
    ax.set_title("② 退火轨迹：先乱走，再冷却")
    ax.set_xlabel("退火 sweep"); ax.legend(fontsize=8)

    ax = axes[2]
    bins = torch.linspace(min(hop_E_t.min(), bm_E_t.min()).item() - 0.1, 0.1, 30).numpy()
    ax.hist(hop_E_t.numpy(), bins=bins, alpha=0.6, label=f"Hopfield（命中 {hop_hit*100:.0f}%）", color="#4c72b0")
    ax.hist(bm_E_t.numpy(), bins=bins, alpha=0.6, label=f"玻尔兹曼机（命中 {bm_hit*100:.0f}%）", color="#c44e52")
    ax.axvline(Emin, ls="--", c="k", lw=1)
    ax.set_title(f"③ {trials} 次随机初态的终态能量分布")
    ax.set_xlabel("终态 E"); ax.set_ylabel("次数"); ax.legend(fontsize=8)

    fig.suptitle("1982 → 1985：同一个能量函数，把 sign() 换成采样", fontsize=13)
    fig.tight_layout()
    out = os.path.join(IMG, "boltzmann_02_hopfield_vs_boltzmann.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\n  图已保存：{out}")
    return W


# ---------------------------------------------------------------------------
# 3b. 名字的由来：这台机器的稳态分布「就是」玻尔兹曼分布
# ---------------------------------------------------------------------------
def part3b_why_the_name():
    """为什么叫「玻尔兹曼机」而不是「随机 Hopfield 网络」。

    因为它不是「用了玻尔兹曼的想法」，而是：让它自由跑下去，
    它停留在状态 s 上的频率，精确等于 exp(-E(s)/T) / Z。
    这里用 8 个单元（2^8 = 256 个态，可以穷举出真实分布）把这件事测出来。
    """
    title("3b", "为什么叫「玻尔兹曼机」：稳态分布 = exp(−E/T)/Z，可测")

    N, T = 8, 1.0
    g = torch.Generator().manual_seed(42)
    W = torch.randn(N, N, generator=g) * 0.6
    W = (W + W.t()) / 2          # 对称：细致平衡的前提
    W.fill_diagonal_(0.0)

    # 理论分布：穷举 2^8 = 256 个状态
    states = torch.tensor([[1.0 if (m >> b) & 1 else -1.0 for b in range(N)] for m in range(1 << N)])
    E_all = -0.5 * ((states @ W) * states).sum(1)
    p_theory = F.softmax(-E_all / T, dim=0)          # exp(-E/T) / Z

    # 经验分布：定温 Gibbs 采样（不退火 —— 退火是找极小，这里是要平衡分布）
    torch.manual_seed(5)
    s = torch.where(torch.rand(N) < 0.5, -1.0, 1.0)
    counts = torch.zeros(1 << N)
    burn_in, n_sweep = 2000, 200_000
    for k in range(burn_in + n_sweep):
        for i in torch.randperm(N):
            s[i] = 1.0 if torch.rand(()) < torch.sigmoid(2.0 * (W[i] @ s) / T) else -1.0
        if k >= burn_in:
            idx = int(((s > 0).long() * (2 ** torch.arange(N))).sum())
            counts[idx] += 1
    p_emp = counts / counts.sum()

    print(f"\n  {N} 个单元、对称 W、定温 T={T} 的 Gibbs 采样，跑 {n_sweep:,} 个 sweep。")
    print("  把访问频率和 exp(−E/T)/Z 逐个状态对照（按理论概率排序，取前 10）：")
    order = torch.argsort(p_theory, descending=True)
    print(f"\n  {'状态':>10} {'E(s)':>9} {'理论 exp(−E/T)/Z':>18} {'实测频率':>12} {'比值':>8}")
    for idx in order[:10]:
        i = int(idx)
        print(f"  {format(i, '08b'):>10} {E_all[i].item():>9.3f} {p_theory[i].item():>18.5f} "
              f"{p_emp[i].item():>12.5f} {p_emp[i].item() / p_theory[i].item():>8.3f}")

    # 量化吻合程度
    kl = (p_theory * (p_theory.clamp_min(1e-12) / p_emp.clamp_min(1e-12)).log()).sum().item()
    tv = 0.5 * (p_theory - p_emp).abs().sum().item()
    mask = p_theory > 1e-4
    corr = torch.corrcoef(torch.stack([p_theory[mask].log(), p_emp[mask].clamp_min(1e-9).log()]))[0, 1].item()
    print(f"\n    KL(理论‖实测)   = {kl:.5f}")
    print(f"    总变差距离 TV   = {tv:.5f}")
    print(f"    log-log 相关系数 = {corr:.5f}")
    print("\n  → 不是「近似」，不是「受启发于」。这台机器的平稳分布**就是**玻尔兹曼分布。")
    print("    名字是描述，不是致敬。")
    print("\n  三条直接推论：")
    print("    (a) log p(s_a) − log p(s_b) = −(E_a − E_b)/T   能量差 = 对数概率差")
    print("    (b) T → 0 时全部质量塌到最低能量态 → 退化成 Hopfield 网络")
    print("    (c) 因为 E 对 w_ij 是线性的（∂E/∂w_ij = −s_i s_j），")
    print("        而 log p = −E/T − log Z，所以梯度必然是「两个相关之差」——")
    print("        1985 年那条学习律是「玻尔兹曼」这三个字的直接后果。")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.loglog(p_theory[mask].numpy(), p_emp[mask].clamp_min(1e-9).numpy(), ".", ms=4, color="#c44e52")
    lo = float(p_theory[mask].min()) * 0.6
    ax.loglog([lo, 1], [lo, 1], "--", c="gray", lw=1, label="y = x")
    ax.set_xlabel("理论 exp(−E/T)/Z"); ax.set_ylabel("Gibbs 实测频率")
    ax.set_title(f"① 稳态分布 vs 玻尔兹曼分布（TV={tv:.4f}）")
    ax.legend(fontsize=9)

    ax = axes[1]
    seen = counts > 0     # 从没被访问到的高能态没有 log 频率，不画
    ax.plot(E_all[seen].numpy(), p_emp[seen].log().numpy(), ".", ms=4, color="#4c72b0",
            label=f"实测 log 频率（{int(seen.sum())}/{1 << N} 个被访问到）")
    ax.plot(E_all.numpy(), (-E_all / T - torch.logsumexp(-E_all / T, 0)).numpy(), "-",
            c="crimson", lw=1.2, label="−E/T − log Z（直线）")
    ax.set_xlabel("能量 E(s)"); ax.set_ylabel("log p(s)")
    ax.set_title("② log p 对 E 是一条直线，斜率 = −1/T")
    ax.legend(fontsize=9)

    fig.suptitle("名字的由来：平稳分布精确等于 exp(−E/T)/Z", fontsize=13)
    fig.tight_layout()
    out = os.path.join(IMG, "boltzmann_02b_why_the_name.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\n  图已保存：{out}")


# ---------------------------------------------------------------------------
# 4. 1986 Smolensky / 2002 Hinton：受限玻尔兹曼机 + Contrastive Divergence
# ---------------------------------------------------------------------------
def bars_and_stripes(n=4):
    """经典玩具数据集：n×n 的全横条或全竖条图案。共 2^n + 2^n - 2 = 30 个（n=4）。"""
    pats = []
    for m in range(1 << n):
        row = torch.tensor([(m >> i) & 1 for i in range(n)], dtype=torch.float32)
        pats.append(row.view(n, 1).expand(n, n).reshape(-1))   # 横条
        pats.append(row.view(1, n).expand(n, n).reshape(-1))   # 竖条
    X = torch.stack(pats)
    X = torch.unique(X, dim=0)                                  # 去掉全 0 / 全 1 的重复
    return X


class RBM:
    """受限玻尔兹曼机：可见层 v ↔ 隐层 h，层内无连接。

    E(v,h) = -vᵀW h - bᵀv - cᵀh
    p(v,h) = exp(-E(v,h)) / Z         ← 还是玻尔兹曼分布，只是 E 里多了隐变量

    "受限"（层内无边）带来的唯一好处：p(h|v) 和 p(v|h) 都可以整层并行采样，
    Gibbs sampling 从 O(单元数) 步变成 2 步。这就是 1986 Smolensky 的 Harmonium
    在 2002 年被 Hinton 重新捡起来的原因。
    """

    def __init__(self, n_vis, n_hid, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(n_vis, n_hid, generator=g) * 0.1
        self.b = torch.zeros(n_vis)   # 可见偏置
        self.c = torch.zeros(n_hid)   # 隐层偏置

    def h_given_v(self, v):
        return torch.sigmoid(v @ self.W + self.c)

    def v_given_h(self, h):
        return torch.sigmoid(h @ self.W.t() + self.b)

    def free_energy(self, v):
        """F(v) = -bᵀv - Σ_j softplus(c_j + (vW)_j)，满足 p(v) = exp(-F(v))/Z。

        隐层被解析地积掉了 —— 这一步是 RBM 能训起来的全部代数原因。
        """
        return -(v @ self.b) - F.softplus(v @ self.W + self.c).sum(1)

    def cd_k(self, v0, k=1, lr=0.1):
        """Contrastive Divergence（Hinton 2002）。

        真正的梯度是   ∂logp/∂W = ⟨v hᵀ⟩_data - ⟨v hᵀ⟩_model
                                    ─正相─      ─负相─
        负相需要跑到平衡分布（不可行）。CD-k 用「从数据出发跑 k 步 Gibbs」近似它。
        k=1 就够用 —— 这是 2002 年那篇论文最反直觉、也最有工程价值的发现。
        """
        # 正相：数据钳在可见层上
        ph0 = self.h_given_v(v0)
        # 负相：k 步 Gibbs
        h = torch.bernoulli(ph0)
        for _ in range(k):
            pv = self.v_given_h(h)
            vk = torch.bernoulli(pv)
            phk = self.h_given_v(vk)
            h = torch.bernoulli(phk)
        B = v0.shape[0]
        self.W += lr * (v0.t() @ ph0 - vk.t() @ phk) / B
        self.b += lr * (v0 - vk).mean(0)
        self.c += lr * (ph0 - phk).mean(0)
        return ((v0 - pv) ** 2).mean().item()   # 重构误差（诊断量，不是目标函数）


def part4a_what_restricted_means():
    """「受限」到底限掉了什么，又换来了什么 —— 两个恒等式，逐个数值验证。"""
    title("4a", "「受限」是什么意思：删掉层内的边，换来两个恒等式")

    n_vis, n_hid = 4, 3
    rbm = RBM(n_vis, n_hid, seed=7)
    rbm.W = torch.randn(n_vis, n_hid, generator=torch.Generator().manual_seed(9)) * 1.2
    rbm.b = torch.randn(n_vis, generator=torch.Generator().manual_seed(10)) * 0.5
    rbm.c = torch.randn(n_hid, generator=torch.Generator().manual_seed(11)) * 0.5

    V = torch.tensor([[float((m >> i) & 1) for i in range(n_vis)] for m in range(1 << n_vis)])
    H = torch.tensor([[float((m >> j) & 1) for j in range(n_hid)] for m in range(1 << n_hid)])

    def joint_E(v, h):
        return -(v @ rbm.W @ h) - (v @ rbm.b) - (h @ rbm.c)

    print("\n  一个 4 可见 / 3 隐 的小 RBM，E(v,h) = −vᵀWh − bᵀv − cᵀh。")
    print("  「受限」= 没有 v–v 的边，也没有 h–h 的边（能量里只有交叉项 vᵀWh）。")

    # 恒等式 1：隐层可以被解析地积掉 —— Σ_h exp(-E(v,h)) == exp(-F(v))
    print("\n  恒等式 ①  Σ_h exp(−E(v,h)) == exp(−F(v))，其中")
    print("            F(v) = −bᵀv − Σ_j softplus(c_j + (vW)_j)")
    print(f"\n  {'v':>6} {'穷举 Σ_h（8 项）':>18} {'解析 exp(−F(v))':>18} {'相对误差':>12}")
    max_err = 0.0
    for k in range(1 << n_vis):
        v = V[k]
        brute = torch.stack([torch.exp(-joint_E(v, H[j])) for j in range(1 << n_hid)]).sum()
        analytic = torch.exp(-rbm.free_energy(v.unsqueeze(0))).squeeze()
        err = abs((brute - analytic) / brute).item()
        max_err = max(max_err, err)
        if k < 5:
            print(f"  {''.join(str(int(x)) for x in v):>6} {brute.item():>18.6f} "
                  f"{analytic.item():>18.6f} {err:>12.2e}")
    print(f"  ... 全部 16 个 v 的最大相对误差：{max_err:.2e}")
    print("    → 隐层被 softplus 一行吃掉了。全连接玻尔兹曼机做不到这件事，")
    print("      因为 h–h 的边会让 Σ_h 不能拆成每个 h_j 独立的乘积。")

    # 恒等式 2：p(h|v) 完全分解成每个隐单元的独立 sigmoid
    print("\n  恒等式 ②  p(h | v) == Π_j p(h_j | v)，且 p(h_j=1|v) = σ(c_j + (vW)_j)")
    v = V[11]
    logits = torch.stack([-joint_E(v, H[j]) for j in range(1 << n_hid)])
    p_exact = F.softmax(logits, dim=0)                      # 穷举得到的 p(h|v)
    p_marg = rbm.h_given_v(v.unsqueeze(0)).squeeze()        # 每个 h_j 的 sigmoid
    p_fact = torch.stack([torch.prod(torch.where(H[j] > 0.5, p_marg, 1 - p_marg))
                          for j in range(1 << n_hid)])      # 独立假设下的乘积
    print(f"\n  取 v = {''.join(str(int(x)) for x in v)}，逐个 h 对照：")
    print(f"  {'h':>6} {'穷举 p(h|v)':>14} {'Π_j σ(...)':>14} {'差':>10}")
    for j in range(1 << n_hid):
        print(f"  {''.join(str(int(x)) for x in H[j]):>6} {p_exact[j].item():>14.6f} "
              f"{p_fact[j].item():>14.6f} {abs(p_exact[j] - p_fact[j]).item():>10.2e}")
    print(f"\n    最大偏差 {(p_exact - p_fact).abs().max().item():.2e} —— 严格相等（数值误差量级）。")
    print("    → 给定 v，3 个隐单元互相独立，可以**一次性并行**全部采完。")

    # 代价与收益
    print("""
  受限的代价与收益：

    没了什么                      换来什么
    ─────────────────────────────────────────────────────────────────
    v–v 的横向连接                p(h|v) 分解 → 整层并行采样
      （可见单元之间的直接相关     Gibbs 一个 sweep 从「逐个单元 N 次串行」
        只能通过 h 间接表达）       变成「两次矩阵乘法」
    h–h 的横向连接                Σ_h 可解析求和 → 自由能 F(v) 有闭式
      （隐单元之间不能互相解释      → 可以直接比较任意两个 v 的相对概率
        away，表达力下降）           （虽然 Z 仍然算不动）

  一句话：把无向图砍成二部图，损失一部分表达力，
          换来「一步就能从 v 采到 h、再从 h 采回 v」。
          2002 年的 CD-1 之所以可能，全部建立在这两个恒等式上。""")


def part4_rbm():
    title(4, "1986 Smolensky / 2002 Hinton：RBM 与 Contrastive Divergence")

    X = bars_and_stripes(4)
    n_vis, n_hid = X.shape[1], 8
    print(f"\n数据集：4×4 bars-and-stripes，{X.shape[0]} 个模式，可见单元 {n_vis}，隐单元 {n_hid}")
    print("  样例（■ = 1）：")
    for r in range(3):
        rows = X[r].view(4, 4)
        print("    " + "  ".join("".join("■" if c > 0.5 else "·" for c in row) for row in rows))

    rbm = RBM(n_vis, n_hid, seed=1)
    torch.manual_seed(3)
    rand_v = torch.bernoulli(torch.full((256, n_vis), 0.5))   # 对照组：纯随机图案

    steps, recon, gap = [], [], []
    print(f"\n{'step':>6} {'重构误差':>10} {'F(数据)':>10} {'F(随机)':>10} {'ΔF':>9}")
    for step in range(3001):
        idx = torch.randint(0, X.shape[0], (16,))
        err = rbm.cd_k(X[idx], k=1, lr=0.1)
        if step % 100 == 0:
            with torch.no_grad():
                fd = rbm.free_energy(X).mean().item()
                fr = rbm.free_energy(rand_v).mean().item()
            steps.append(step); recon.append(err); gap.append(fr - fd)
            if step % 500 == 0:
                print(f"{step:>6} {err:>10.4f} {fd:>10.3f} {fr:>10.3f} {fr - fd:>9.3f}")

    print(f"\n  ΔF = F(随机) − F(数据) 从 {gap[0]:+.3f} 涨到 {gap[-1]:+.3f}")
    print("  自由能低 = 概率高。模型学会了「把概率质量搬到数据所在的那 30 个态上」，")
    print("  搬运的动力就是 正相 − 负相。")

    # 从模型采样：Gibbs 链跑久一点，看它吐什么
    torch.manual_seed(11)
    v = torch.bernoulli(torch.full((1, n_vis), 0.5))
    for _ in range(500):
        v = torch.bernoulli(rbm.v_given_h(torch.bernoulli(rbm.h_given_v(v))))
    print("\n  Gibbs 链跑 500 步后模型自己「幻想」出的图案：")
    for row in v.view(4, 4):
        print("      " + "".join("■" if c > 0.5 else "·" for c in row))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.plot(steps, recon, color="#4c72b0")
    ax.set_title("① CD-1 重构误差"); ax.set_xlabel("step"); ax.set_ylabel("MSE")

    ax = axes[1]
    ax.plot(steps, gap, color="#c44e52")
    ax.axhline(0, ls="--", c="gray", lw=1)
    ax.set_title("② 自由能间隙 F(随机) − F(数据)")
    ax.set_xlabel("step"); ax.set_ylabel("ΔF（越大越好）")

    ax = axes[2]
    Wimg = rbm.W.t().reshape(n_hid, 4, 4)
    grid = torch.cat([torch.cat([Wimg[r * 4 + c] for c in range(4)], dim=1)
                      for r in range(2)], dim=0)
    im = ax.imshow(grid.detach().numpy(), cmap="RdBu_r")
    ax.set_title("③ 8 个隐单元学到的「特征」(W 的列)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.03)

    fig.suptitle("1986/2002 → RBM：把玻尔兹曼机砍成两层，负相用 1 步 Gibbs 近似", fontsize=13)
    fig.tight_layout()
    out = os.path.join(IMG, "boltzmann_03_rbm_cd1.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\n  图已保存：{out}")


# ---------------------------------------------------------------------------
# 5. 今天：codechat/gpt.py 的 softmax 就是玻尔兹曼分布
# ---------------------------------------------------------------------------
def part5_softmax_is_boltzmann():
    title(5, "今天：gpt.py:141-146 的采样，就是 exp(−E/kT)/Z")

    print("""
    codechat/gpt.py:141-146
    ────────────────────────────────────────────────────────────
        logits = logits[:, -1, :].float() / max(temperature, 1e-5)
        if top_k is not None: ...                       # 截断
        probs  = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
    ────────────────────────────────────────────────────────────

    改写一下就认得出来了：令 E_i := −logit_i，T := temperature

        p_i = exp(logit_i / T) / Σ_j exp(logit_j / T)
            = exp(−E_i / T) / Z          ← 一模一样的玻尔兹曼分布

    也就是说：Transformer 的最后一层，是在给 50257 个 token 各打一个「能量」，
    然后按 1877 年那条公式在这 50257 个「能级」上做一次热采样。
    温度这个词不是比喻，它就是 kT。
    """)

    # 造一组接近真实的 logits（一个"该输出 def 还是 class"的位置）
    vocab = ["def", "class", "import", "return", "for", "x", "#", "\\n", "(", "self"]
    logits0 = torch.tensor([8.4, 6.9, 6.1, 4.2, 3.8, 2.5, 2.1, 1.6, 1.0, 0.4])

    print(f"  某个位置上 10 个候选 token 的 logits（= −能量）：")
    print("    " + "  ".join(f"{t}:{l:.1f}" for t, l in zip(vocab, logits0)))

    print(f"\n{'温度 T':>8} {'熵 S(nats)':>12} {'有效候选 e^S':>13}  top-3")
    rows = []
    for T in [0.01, 0.3, 0.7, 1.0, 1.5, 3.0]:
        p = F.softmax(logits0 / T, dim=0)
        S = max(-(p * p.clamp_min(1e-12).log()).sum().item(), 0.0)
        top = torch.topk(p, 3)
        s3 = ", ".join(f"{vocab[i]}={v:.2f}" for v, i in zip(top.values, top.indices))
        rows.append((T, p, S))
        print(f"{T:>8.2f} {S:>12.4f} {math.exp(S):>13.2f}  {s3}")

    print(f"\n  T=0.01 → S≈0，塌到基态「def」            = 贪心解码 = 淬火(quench)")
    print(f"  T=0.7  → 本仓库 chat_cli.py:15 的默认值   = 保留 ~{math.exp(rows[2][2]):.1f} 个有效候选")
    print(f"  T=3.0  → 接近均匀，S→ln10={math.log(10):.3f}  = 胡说八道")
    print("\n  chat_cli 用 T=0.7、chat_rl_funcall 用 T=1.0（RL 采样必须无偏，见 chat_rl_funcall.py:242），")
    print("  这两个数字的物理含义，和 1877 年那个 kT 是同一个东西。")

    # 交叉熵 = 自由能
    print("""
  更深一层：训练用的 F.cross_entropy（gpt.py:129）本身就是自由能。

      loss = −log p(target) = E_target + log Z
                              ─────    ─────
                              能量项    配分函数（log Z = 自由能）

  「降 loss」= 压低正确 token 的能量，同时压低整个 log Z。
  这正是 RBM 的 正相 − 负相：拉低数据的能量、抬高其他一切的概率质量。
  区别只在于 Transformer 的 Z 是 50257 项的显式求和（算得起），
  而玻尔兹曼机的 Z 是 2^N 项（算不起，只能采样）——
  ★ 这就是 1985 年那条路走不通、而 2017 年这条路走通了的全部原因。
    """)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.bar(range(len(vocab)), (-logits0).numpy(), color="#55a868")
    ax.set_xticks(range(len(vocab))); ax.set_xticklabels(vocab, rotation=45, ha="right", fontsize=8)
    ax.set_title("① Transformer 给每个 token 的「能量」E = −logit")
    ax.set_ylabel("E")

    ax = axes[1]
    for T, p, S in rows[1:5]:
        ax.plot(range(len(vocab)), p.numpy(), "o-", label=f"T={T}")
    ax.set_xticks(range(len(vocab))); ax.set_xticklabels(vocab, rotation=45, ha="right", fontsize=8)
    ax.set_title("② p = exp(−E/T)/Z：温度改的是分布形状")
    ax.legend(fontsize=8)

    ax = axes[2]
    Tsweep = torch.logspace(-2, 0.7, 150)
    Ss = [-(F.softmax(logits0 / T, dim=0) * F.softmax(logits0 / T, dim=0).clamp_min(1e-12).log()).sum().item()
          for T in Tsweep]
    ax.semilogx(Tsweep.numpy(), Ss, color="#c44e52")
    ax.axhline(math.log(10), ls="--", c="gray", lw=1); ax.text(0.011, math.log(10) + 0.04, "ln 10", fontsize=8, color="gray")
    for T, lab, c in [(0.7, "chat_cli\nT=0.7", "#4c72b0"), (1.0, "chat_rl_funcall\nT=1.0", "#dd8452")]:
        ax.axvline(T, ls=":", c=c, lw=1.4)
        ax.text(T * 1.03, 0.15, lab, fontsize=8, color=c)
    ax.set_title("③ 采样熵 vs 温度（本仓库两个默认值）")
    ax.set_xlabel("temperature"); ax.set_ylabel("S (nats)")

    fig.suptitle("2017 → 今天：softmax(logits/T) 就是 1877 年的 exp(−E/kT)/Z", fontsize=13)
    fig.tight_layout()
    out = os.path.join(IMG, "boltzmann_04_softmax_is_boltzmann.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  图已保存：{out}")


# ---------------------------------------------------------------------------
def main():
    print(__doc__.split("配文")[0].rstrip())
    part1_entropy_and_distribution()
    part23_hopfield_vs_boltzmann()
    part3b_why_the_name()
    part4a_what_restricted_means()
    part4_rbm()
    part5_softmax_is_boltzmann()

    print("\n" + "=" * 78)
    print("一条线：")
    print("  1877 S = k log W          熵是数数")
    print("  1877 p ∝ exp(−E/kT)       分布由能量决定")
    print("  1982 E = −½sᵀWs           Hopfield：把 E 变成可学的权重")
    print("  1985 p(s) = exp(−E/T)/Z   Hinton：把下降换成采样 → 玻尔兹曼机")
    print("  2002 CD-1                 负相用 1 步 Gibbs 近似 → RBM 能训了")
    print("  2006 DBN 逐层预训练        深度网络第一次训得动 → 深度学习复活")
    print("  2012 AlexNet / 2017 Transformer  反向传播 + 算力接管")
    print("  今天 softmax(logits/T)     Z 变成 50257 项的显式求和，采样不再需要 Gibbs")
    print("=" * 78)


if __name__ == "__main__":
    main()
