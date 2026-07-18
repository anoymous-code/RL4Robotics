"""对照组实验：错误标定下的脚本专家成功率。

模拟真机"手眼标定/位姿估计有固定偏差"的场景：专家的所有感知性位姿读取
（板位姿、目标格位置、盒 B 位置）统一加一个水平偏移（默认 3 cm 随机方向），
成功判定仍用真值。场景序列与 eval_act.py 相同（seed 10000+ep），可直接对比。

这就是预注册协议里的对照组：脚本在错标定下崩溃的程度 vs 视觉策略的表现，
量化"从图像闭环适应位置"的价值。

运行:
    ../../.venv/Scripts/python.exe eval_miscalib.py --n 20 --offset 0.03
"""

import argparse

import numpy as np

import tear_scene as ts
from run_full_demo import FullDemo


def main(n, offset_mag):
    n_seg, n_full = 0, 0
    for ep in range(n):
        rng = np.random.default_rng(10000 + ep)      # 与 eval_act 相同的场景序列
        cfg = ts.sample_cfg(rng, 1.0)
        ang = np.random.default_rng(500 + ep).uniform(0, 2 * np.pi)
        demo = FullDemo(cfg=cfg, skip_drive=True, targets=[tuple(cfg.target_seg)],
                        make_video=False, verbose=False)
        demo.sense_offset = np.array([np.cos(ang), np.sin(ang), 0.0]) * offset_mag
        try:
            n_ok, returned = demo.run()
        except Exception as exc:
            print(f"[ep {ep:02d}] 执行异常: {exc}")
            n_ok, returned = 0, False
        n_seg += int(n_ok >= 1)
        n_full += int(n_ok >= 1 and returned)
        print(f"[ep {ep:02d}] 撕剪入盒 B: {n_ok >= 1}, 全流程: {n_ok >= 1 and returned}"
              f"（感知偏移 {offset_mag*1000:.0f}mm @ {np.degrees(ang):.0f}°）")
    print(f"\n错标定专家成功率（偏移 {offset_mag*1000:.0f}mm）: "
          f"撕剪入盒 B {n_seg}/{n}, 全流程 {n_full}/{n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--offset", type=float, default=0.03, help="感知偏移幅度 (m)")
    args = parser.parse_args()
    main(args.n, args.offset)
