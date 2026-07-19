"""世界模型判别力诊断：把关人为什么从不干预？

三个问题：
1. fail 头能否区分成功/失败 episode 的状态-动作对？（分离度）
2. 候选块之间的评分展布多大？（随机噪声候选是否产生实质差异）
3. Δ进度头对"实际推进 vs 停滞"的区分？

运行:
    ../../.venv/Scripts/python.exe diagnose_wm.py
"""

import io
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from eval_act_wm import NOISE_SIGMA, WmGate
from train_act import CAMS, CHUNK, IMG_MEAN, IMG_STD
from train_wm import ROLL_DIR

HERE = Path(__file__).resolve().parent


def load_sample(f, t):
    obs = {"qpos": f["observations/qpos"][t]}
    for cam in CAMS:
        obs[cam] = np.asarray(Image.open(io.BytesIO(
            f[f"observations/images/{cam}"][t].tobytes())))
    chunk = f["action"][t:t + CHUNK]
    row = int(json.loads(f.attrs["cfg"])["target_seg"][1])
    return obs, chunk, row


def main():
    gate = WmGate(HERE / "ckpt" / "wm_latest.pt")
    rng = np.random.default_rng(0)

    ok_files = sorted(ROLL_DIR.glob("rollout_*_ok.hdf5"))[:15]
    fail_files = sorted(ROLL_DIR.glob("rollout_*_fail.hdf5"))[:15]

    # 1) fail 头分离度：各 episode 中段采样
    def fail_scores(files):
        out = []
        for p in files:
            with h5py.File(p, "r") as f:
                n = f["observations/qpos"].shape[0]
                for t in (int(n * 0.3), int(n * 0.5), int(n * 0.7)):
                    obs, chunk, row = load_sample(f, t)
                    s = gate.score(obs, chunk[None], row)   # 单候选
                    # score = prog - λ·fail；重新拆出 fail 需要直接调模型
                    out.append(s[0])
        return np.array(out)

    s_ok = fail_scores(ok_files)
    s_fail = fail_scores(fail_files)
    print(f"成功 episode 样本评分: {s_ok.mean():.3f} ± {s_ok.std():.3f}")
    print(f"失败 episode 样本评分: {s_fail.mean():.3f} ± {s_fail.std():.3f}")
    # 简易 AUC
    labels = np.r_[np.zeros(len(s_ok)), np.ones(len(s_fail))]
    scores = np.r_[-s_ok, -s_fail]     # 分数低 = 预判失败
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores))
    auc = (ranks[labels == 1].mean() - (labels.sum() - 1) / 2) / (labels == 0).sum()
    print(f"成败判别 AUC（训练集样本，供参考）: {auc:.3f}")

    # 2) 候选评分展布
    spreads = []
    with h5py.File(ok_files[0], "r") as f:
        n = f["observations/qpos"].shape[0]
        for t in range(0, n - CHUNK, 200):
            obs, chunk, row = load_sample(f, t)
            cands = gate.make_candidates(chunk, 8, rng)
            sc = gate.score(obs, cands, row)
            spreads.append(sc.max() - sc.min())
            print(f"  t={t:5d}: 原块 {sc[0]:+.3f}, 候选最优 {sc.max():+.3f}, "
                  f"展布 {sc.max()-sc.min():.4f}")
    print(f"候选评分展布（K=8, σ={NOISE_SIGMA}）: "
          f"中位 {np.median(spreads):.4f}, 最大 {np.max(spreads):.4f}")


if __name__ == "__main__":
    main()
