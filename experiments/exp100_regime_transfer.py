# -*- coding: utf-8 -*-
"""실험 100: 체제 전이학습 — 데이터를 버리지 않고 현 체제에 적응시킨다.

우리가 확인한 유일한 구조적 문제는 라벨 체제 전환이다. 그런데 지금까지
`abs_era` 플래그 하나로만 처리했다.

  2025 test 체제      : 1군 ABS + 퓨처스 ABS
  같은 체제 학습 데이터 : R 2024 + F 2023~24  (약 278K행)
  나머지               : 약 120만행, 다른 체제

시도된 것과 실패 이유:
  exp60 최근 창만 학습  → 세 fold 모두 악화. 데이터 감소 손해가 큼
  exp11 최근 시즌 가중  → 중립~악화
  둘 다 **데이터를 버리는** 방식이다.

여기서는 버리지 않는다. 전체로 표현을 만들고(사전학습) 체제 일치분으로
결정 경계만 옮긴다(미세조정). GBDT로는 불가능하고 NN이라 가능하며,
NN 계열이 챔피언 유효 가중의 54.5%다.

변형:
  A pretrain_only   전체 학습만 (대조군)
  B finetune_head   사전학습 후 출력층만 미세조정 (표현 동결)
  C finetune_all    사전학습 후 전체를 낮은 LR로 미세조정
  D finetune_resid  사전학습 예측을 offset으로 고정하고 잔차만 학습

D가 개념적으로 가장 보수적이다. 사전학습 로짓을 상수 offset으로 두고
체제 일치분에서 보정항만 학습하므로, 미세조정이 실패해도 사전학습
수준 아래로 내려가지 않는다.

구조 임베딩은 검증된 축만: 투수 x (같은손 2 x 카운트우열 3) = 6벌.
채점은 2025 유형 행. 노이즈 바닥 SD 2.8e-5이므로 3시드 평균으로 비교하고
채택 문턱은 2시그마 = 3.2e-5로 사전 등록한다.
"""
from pathlib import Path
import sys, time, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import brier_score_loss, roc_auc_score
from features import add_features

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ID, TARGET = "row_id", "control_success"
EMB = [("pitcher_id", 16), ("batter_id", 16), ("pitcher_team_id", 4), ("batter_team_id", 4),
       ("base_state", 4), ("pitcher_hand", 2), ("batter_hand", 2), ("top_bottom", 2),
       ("game_type", 2)]
BATCH = 8192
SEEDS = [42, 7, 2024]
PRE_EP = 3          # 사전학습 (exp99에서 조기종료가 3에폭에 멈춤을 확인)
FT_EP = 4           # 미세조정


def regime_mask(d):
    """미세조정 대상 = 학습 구간의 최신 시즌.

    엄밀한 ABS 체제 정의(1군 2024~ / 퓨처스 2023~)를 쓰면 2024 fold에서
    체제 일치가 F2023 25,686행(2.1%)뿐이라 2025 상황을 재현할 수 없다.
    2025 제출 시에는 R2024+F2023~24로 약 19%가 되기 때문이다.
    따라서 fold마다 재현 가능하고 비율이 맞는 '최신 시즌' 정의를 쓴다.
      2024 fold -> 2023으로 미세조정 (약 20%)
      2025 제출 -> 2024로 미세조정 (약 17%)
    """
    return (d.season == d.season.max()).to_numpy()


def build(df, year):
    tr = df[df.season < year].reset_index(drop=True)
    va = df[df.season == year].reset_index(drop=True)
    ct = np.zeros((len(tr), len(EMB)), dtype=np.int64)
    cv = np.zeros((len(va), len(EMB)), dtype=np.int64)
    sizes = []
    for j, (c, _) in enumerate(EMB):
        vals = sorted(tr[c].dropna().astype(str).unique())
        mp = {v: i + 1 for i, v in enumerate(vals)}
        ct[:, j] = tr[c].astype(str).map(mp).fillna(0).to_numpy(dtype=np.int64)
        cv[:, j] = va[c].astype(str).map(mp).fillna(0).to_numpy(dtype=np.int64)
        sizes.append(len(vals) + 1)
    names = {c for c, _ in EMB}
    ncols = [c for c in df.columns if c not in names and c not in (ID, TARGET)]
    xt = pd.concat([tr[ncols], add_features(tr)], axis=1).astype(np.float32)
    xv = pd.concat([va[ncols], add_features(va)], axis=1).astype(np.float32)
    med = xt.median(); xt = xt.fillna(med); xv = xv.fillna(med)
    mu, sd = xt.mean(), xt.std().replace(0, 1)
    xt = ((xt - mu) / sd).to_numpy(np.float32); xv = ((xv - mu) / sd).to_numpy(np.float32)

    def cell(d):
        same = (d.pitcher_hand.to_numpy() == d.batter_hand.to_numpy()).astype(np.int64)
        adv = np.sign(d.strikes_before.to_numpy() - d.balls_before.to_numpy()) + 1
        return same * 3 + adv
    return tr, va, ct, cv, sizes, xt, xv, cell(tr), cell(va)


class Net(nn.Module):
    def __init__(self, sizes, ndim):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(s, d) for s, (_, d) in zip(sizes, EMB)])
        self.pc = nn.ModuleList([nn.Embedding(sizes[0], 16) for _ in range(6)])
        dim = ndim + sum(d for _, d in EMB) + 16
        self.trunk = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.15),
                                   nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.15))
        self.head = nn.Linear(128, 1)

    def feat(self, xc, xn, cell):
        e = [emb(xc[:, j]) for j, emb in enumerate(self.embs)]
        ph = torch.zeros(xc.shape[0], 16, device=xc.device)
        for g in range(6):
            m = cell == g
            if m.any():
                ph[m] = self.pc[g](xc[m, 0])
        return self.trunk(torch.cat(e + [ph, xn], dim=1))

    def forward(self, xc, xn, cell):
        return self.head(self.feat(xc, xn, cell)).squeeze(1)


def epochs(net, opt, Xc, Xn, CE, Y, idx, n_ep, offset=None):
    bce = nn.BCEWithLogitsLoss()
    for _ in range(n_ep):
        np.random.shuffle(idx); net.train()
        for s in range(0, len(idx), BATCH):
            b = idx[s:s + BATCH]; opt.zero_grad()
            z = net(Xc[b], Xn[b], CE[b])
            if offset is not None:
                z = z + offset[b]
            bce(z, Y[b]).backward(); opt.step()


def infer(net, Xc, Xn, CE, offset=None, logit=False):
    net.eval(); zs = []
    with torch.no_grad():
        for s in range(0, Xc.shape[0], BATCH * 4):
            e = min(s + BATCH * 4, Xc.shape[0])
            z = net(Xc[s:e], Xn[s:e], CE[s:e])
            if offset is not None:
                z = z + offset[s:e]
            zs.append(z)
    z = torch.cat(zs)
    return z if logit else torch.sigmoid(z).numpy()


def run(tr, va, ct, cv, sizes, xt, xv, ce_tr, ce_va, seed, mode):
    torch.manual_seed(seed); np.random.seed(seed)
    Xc, Xn, CE = torch.from_numpy(ct), torch.from_numpy(xt), torch.from_numpy(ce_tr)
    Y = torch.from_numpy(tr[TARGET].to_numpy(np.float32))
    Vc, Vn, VE = torch.from_numpy(cv), torch.from_numpy(xv), torch.from_numpy(ce_va)
    net = Net(sizes, xt.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    epochs(net, opt, Xc, Xn, CE, Y, np.arange(len(tr)), PRE_EP)
    if mode == "pretrain_only":
        return infer(net, Vc, Vn, VE)

    rm = np.flatnonzero(regime_mask(tr))          # 최신 시즌 행만
    if mode == "finetune_head":
        for p in net.trunk.parameters(): p.requires_grad = False
        for p in net.embs.parameters(): p.requires_grad = False
        for p in net.pc.parameters(): p.requires_grad = False
        o2 = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3)
        epochs(net, o2, Xc, Xn, CE, Y, rm, FT_EP)
        return infer(net, Vc, Vn, VE)
    if mode == "finetune_all":
        o2 = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-5)
        epochs(net, o2, Xc, Xn, CE, Y, rm, FT_EP)
        return infer(net, Vc, Vn, VE)
    if mode == "finetune_resid":
        # 사전학습 로짓을 상수 offset으로 고정하고 보정항만 새로 학습
        off_tr = infer(net, Xc, Xn, CE, logit=True).detach()
        off_va = infer(net, Vc, Vn, VE, logit=True).detach()
        torch.manual_seed(seed + 1)
        net2 = Net(sizes, xt.shape[1])
        nn.init.zeros_(net2.head.weight); nn.init.zeros_(net2.head.bias)
        o2 = torch.optim.AdamW(net2.parameters(), lr=5e-4, weight_decay=1e-4)
        epochs(net2, o2, Xc, Xn, CE, Y, rm, FT_EP, offset=off_tr)
        return infer(net2, Vc, Vn, VE, offset=off_va)
    raise ValueError(mode)


def main():
    df = pd.read_csv(HERE / "data" / "train.csv", encoding="utf-8-sig")
    YEAR = 2024
    tr, va, ct, cv, sizes, xt, xv, ce_tr, ce_va = build(df, YEAR)
    keep = ((va.game_type == "R") | (va.season >= 2023)).to_numpy()
    yv = va[TARGET].to_numpy()
    rm = regime_mask(tr)
    print("학습 %d행 (최신시즌 %d행, %.1f%%) / 검증 %d행"
          % (len(tr), rm.sum(), 100 * rm.mean(), len(va)), flush=True)
    res = {}
    for mode in ["pretrain_only", "finetune_head", "finetune_all", "finetune_resid"]:
        ps = []
        for sd in SEEDS:
            t0 = time.time()
            p = run(tr, va, ct, cv, sizes, xt, xv, ce_tr, ce_va, sd, mode)
            ps.append(p)
            print("  %-16s seed=%-5d brier=%.6f (%ds)"
                  % (mode, sd, brier_score_loss(yv[keep], p[keep]), round(time.time() - t0)),
                  flush=True)
            gc.collect()
        avg = np.mean(ps, axis=0); res[mode] = avg
        print("  %-16s 3시드평균 brier=%.6f auc=%.4f"
              % (mode, brier_score_loss(yv[keep], avg[keep]),
                 roc_auc_score(yv[keep], avg[keep])), flush=True)
    b0 = brier_score_loss(yv[keep], res["pretrain_only"][keep])
    print("\n=== pretrain_only 대비 (e-5) | 3시드 유효SD 1.62e-5, 채택문턱 2시그마=3.2e-5 ===")
    for k, v in res.items():
        g = (b0 - brier_score_loss(yv[keep], v[keep])) * 1e5
        mk = "채택" if g >= 3.2 else ("경계" if g >= 1.6 else "미달")
        print("  %-16s %+8.2f  %s" % (k, g, mk))
    pd.DataFrame({ID: va[ID], TARGET: yv, "keep": keep,
                  **{"p_" + k: res[k] for k in res}}).to_csv(
        HERE / "results" / "exp100_regime_transfer_oof.csv.gz", index=False, compression="gzip")


if __name__ == "__main__":
    main()
