# -*- coding: utf-8 -*-
"""실험 99: 멀티태스크 NN — 새 판으로 짠 모델.

동기 두 가지.

[1] 가장 비중 큰 계열이 가장 대충 학습되고 있다.
챔피언 ms65의 유효 가중은 NN 분할 계열 54.5% / GBDT 38.7%인데,
그 NN은 EPOCHS=4 고정, LR 2e-3 고정, 검증 기반 조기종료 없음이다.
CLAUDE.md에 이미 증거가 있다: 2에폭 0.248538 < 4에폭 0.248685.
즉 4에폭은 과학습 구간이며 에폭 수가 튜닝된 적이 없다.

[2] 역산 라벨이 안 쓰이고 있다.
exp24_reconstructed_labels.npy는 147만 행의 4분류 라벨이고 복원률 99.89%,
class0과 control_success 일치율 1.0000이다. 이진 라벨의 4배 정보량인데
exp24(다중분류)·exp52(앙상블 멤버) 둘 다 실패했다. 그러나 둘 다
**다중분류 헤드로 예측**했다. 그건 이진 헤드와 항등에 가까우니 당연히 실패한다.

여기서는 보조 손실로 **표현을 규제**하고 예측은 이진 헤드에서 뽑는다.
근거: 세 실패 유형은 의존 피처가 다르다(exp45 측정).
  의도반대  팔각도x매치업 -0.0166, 직구비중 +0.0209
  한가운데  카운트 지배(3-0 0.1815 vs 0-2 0.1180), 구종 무관
  크게벗어남 슬라이더비중 +0.0225, 2스트라이크
공유 몸통이 셋을 모두 설명하게 강제하면 표현이 풍부해진다.

세 변형을 한 fold(2024)에서 분리 측정한다.
  A base    현행 재현 (4에폭 고정, 스케줄 없음)
  B sched   검증 기반 조기종료 + 코사인 스케줄
  C multi   B + 4분류 보조 손실 (예측은 이진 헤드)

구조 임베딩은 검증된 축만 쓴다: 투수 x (같은손 2 x 카운트우열 3) = 6벌.
채점은 2025 유형 행(R 또는 season>=2023). 노이즈 바닥 2.8e-5를 감안해
시드 3개 평균으로 비교한다.
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
VAL_FRAC = 0.12          # 학습 구간 끝부분을 시간순으로 떼어 조기종료에 쓴다
SEEDS = [42, 7, 2024]


def build_arrays(df, year):
    tr = df[df.season < year].reset_index(drop=True)
    va = df[df.season == year].reset_index(drop=True)
    cat_tr = np.zeros((len(tr), len(EMB)), dtype=np.int64)
    cat_va = np.zeros((len(va), len(EMB)), dtype=np.int64)
    sizes = []
    for j, (c, _) in enumerate(EMB):
        vals = sorted(tr[c].dropna().astype(str).unique())
        mp = {v: i + 1 for i, v in enumerate(vals)}
        cat_tr[:, j] = tr[c].astype(str).map(mp).fillna(0).to_numpy(dtype=np.int64)
        cat_va[:, j] = va[c].astype(str).map(mp).fillna(0).to_numpy(dtype=np.int64)
        sizes.append(len(vals) + 1)
    names = {c for c, _ in EMB}
    num_cols = [c for c in df.columns if c not in names and c not in (ID, TARGET, "cls4")]
    xt = pd.concat([tr[num_cols], add_features(tr)], axis=1).astype(np.float32)
    xv = pd.concat([va[num_cols], add_features(va)], axis=1).astype(np.float32)
    med = xt.median(); xt = xt.fillna(med); xv = xv.fillna(med)
    mu, sd = xt.mean(), xt.std().replace(0, 1)
    xt = ((xt - mu) / sd).to_numpy(np.float32)
    xv = ((xv - mu) / sd).to_numpy(np.float32)

    def cell(d):
        same = (d.pitcher_hand.to_numpy() == d.batter_hand.to_numpy()).astype(np.int64)
        adv = np.sign(d.strikes_before.to_numpy() - d.balls_before.to_numpy()) + 1
        return same * 3 + adv
    return (tr, va, cat_tr, cat_va, sizes, xt, xv, cell(tr), cell(va))


class Net(nn.Module):
    def __init__(self, sizes, ndim, multi):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(s, d) for s, (_, d) in zip(sizes, EMB)])
        self.pc = nn.ModuleList([nn.Embedding(sizes[0], 16) for _ in range(6)])
        dim = ndim + sum(d for _, d in EMB) + 16
        self.trunk = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.15),
                                   nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.15))
        self.head = nn.Linear(128, 1)
        self.aux = nn.Linear(128, 4) if multi else None

    def forward(self, xc, xn, cell):
        e = [emb(xc[:, j]) for j, emb in enumerate(self.embs)]
        ph = torch.zeros(xc.shape[0], 16, device=xc.device)
        for g in range(6):
            m = cell == g
            if m.any():
                ph[m] = self.pc[g](xc[m, 0])
        h = self.trunk(torch.cat(e + [ph, xn], dim=1))
        return self.head(h).squeeze(1), (self.aux(h) if self.aux is not None else None)


def run(tr, va, cat_tr, cat_va, sizes, xt, xv, cell_tr, cell_va,
        seed, mode, max_ep, lam=0.3):
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(tr)
    ncut = int(n * (1 - VAL_FRAC))          # 시간순 뒷부분을 내부 검증으로
    y = tr[TARGET].to_numpy(np.float32)
    c4 = tr["cls4"].to_numpy(np.int64)
    Xc = torch.from_numpy(cat_tr); Xn = torch.from_numpy(xt)
    Y = torch.from_numpy(y); C = torch.from_numpy(c4); CE = torch.from_numpy(cell_tr)
    Vc = torch.from_numpy(cat_va); Vn = torch.from_numpy(xv); VE = torch.from_numpy(cell_va)
    net = Net(sizes, xt.shape[1], mode == "multi")
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
             if mode != "base" else None)
    idx = np.arange(ncut)
    best, best_state, bad = 1e9, None, 0
    for ep in range(max_ep):
        np.random.shuffle(idx); net.train()
        for s in range(0, ncut, BATCH):
            b = idx[s:s + BATCH]; opt.zero_grad()
            z, a = net(Xc[b], Xn[b], CE[b])
            loss = bce(z, Y[b])
            if a is not None:
                loss = loss + lam * ce(a, C[b])
            loss.backward(); opt.step()
        if sched: sched.step()
        if mode == "base":
            continue
        net.eval()                                    # 내부 검증으로 조기종료
        with torch.no_grad():
            zs = []
            for s in range(ncut, n, BATCH * 4):
                e = min(s + BATCH * 4, n)
                zs.append(net(Xc[s:e], Xn[s:e], CE[s:e])[0])
            p = torch.sigmoid(torch.cat(zs)).numpy()
        vb = float(np.mean((y[ncut:] - p) ** 2))
        if vb < best - 1e-7:
            best, bad = vb, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        zs = []
        for s in range(0, len(va), BATCH * 4):
            e = min(s + BATCH * 4, len(va))
            zs.append(net(Vc[s:e], Vn[s:e], VE[s:e])[0])
        out = torch.sigmoid(torch.cat(zs)).numpy()
    del net; gc.collect()
    return out, (ep + 1)


def main():
    df = pd.read_csv(HERE / "data" / "train.csv", encoding="utf-8-sig")
    df["cls4"] = np.load(HERE / "results" / "exp24_reconstructed_labels.npy")
    YEAR = 2024
    tr, va, cat_tr, cat_va, sizes, xt, xv, ce_tr, ce_va = build_arrays(df, YEAR)
    keep = ((va.game_type == "R") | (va.season >= 2023)).to_numpy()
    yv = va[TARGET].to_numpy()
    print("학습 %d행 / 검증 %d행(채점 %d) / 수치피처 %d개"
          % (len(tr), len(va), keep.sum(), xt.shape[1]), flush=True)

    res = {}
    for name, mode, mx in [("A_base(4에폭고정)", "base", 4),
                           ("B_sched(조기종료)", "sched", 20),
                           ("C_multi(보조손실)", "multi", 20)]:
        ps, eps = [], []
        for sd in SEEDS:
            t0 = time.time()
            p, e = run(tr, va, cat_tr, cat_va, sizes, xt, xv, ce_tr, ce_va, sd, mode, mx)
            ps.append(p); eps.append(e)
            print("  %-18s seed=%-5d ep=%2d brier=%.6f (%ds)"
                  % (name, sd, e, brier_score_loss(yv[keep], p[keep]), round(time.time() - t0)),
                  flush=True)
        avg = np.mean(ps, axis=0)
        res[name] = avg
        print("  %-18s 3시드평균 brier=%.6f auc=%.4f  (에폭 %s)"
              % (name, brier_score_loss(yv[keep], avg[keep]),
                 roc_auc_score(yv[keep], avg[keep]), eps), flush=True)
    print("\n=== A 대비 (e-5, 3시드 평균, 노이즈 바닥 SD 2.8e-5/√3=1.6e-5) ===")
    b0 = brier_score_loss(yv[keep], res["A_base(4에폭고정)"][keep])
    for k, v in res.items():
        print("  %-18s %+8.2f" % (k, (b0 - brier_score_loss(yv[keep], v[keep])) * 1e5))
    np.save(HERE / "results" / "exp99_preds.npy",
            np.vstack([res[k] for k in res]))
    pd.DataFrame({ID: va[ID], TARGET: yv, "keep": keep,
                  **{("p_" + k.split("_")[0]): res[k] for k in res}}).to_csv(
        HERE / "results" / "exp99_multitask_oof.csv.gz", index=False, compression="gzip")


if __name__ == "__main__":
    main()
