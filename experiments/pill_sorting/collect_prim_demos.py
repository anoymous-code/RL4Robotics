"""原语级补数据（Stage C）：从入口状态池直采单原语演示片段。

与全流程采集（collect_demos.py）互补：不用跑完整 episode，直接
restore 池条目 → 脚本原语执行（DART 噪声）→ 录制该原语窗口。
成本 = 原语时长（数秒）而非全流程（~45 s），弱原语可以廉价堆数据。

产物 demos/boost_{prim}_{i}_{ok|fail}.hdf5，格式与全流程演示一致
（/phase 相位标签 + prim_ok 原语级成功标注），train_prim.py 用
--prefix "prim,boost_p2_tear_seg" 合并训练。

运行:
    ../../.venv/Scripts/python.exe collect_prim_demos.py --prim p2_tear_seg ^
        --pools train,extra --passes 1
"""

import argparse
import time

import numpy as np

from collect_demos import DEMO_DIR, DemoRecorder
from primitives import PRIM_SPECS, load_pool, restore_session


def collect(prim, pool_tags, passes=1, action_noise=0.004, seed=0,
            img_hw=(480, 640), start=0):
    spec = PRIM_SPECS[prim]
    entries = []
    for tag in pool_tags:
        entries += load_pool(prim, tag=tag, only_ok=True)
    print(f"{prim}: {len(entries)} 条入口 × {passes} 遍", flush=True)
    np.random.seed(seed)
    idx, n_ok = start, 0
    for p in range(passes):
        for e in entries:
            t0 = time.time()
            sess = restore_session(e, img_hw=img_hw)
            demo = sess.demo
            demo.action_noise = action_noise
            recorder = DemoRecorder(demo.model, jpeg_quality=50, img_hw=img_hw,
                                    phase_fn=lambda d=demo: d.phase_name,
                                    cams=(spec.cam,))
            demo.recorder = recorder
            try:
                ok = sess.run_scripted()
            except Exception as exc:
                print(f"[{idx:03d}] 原语执行异常: {exc}")
                ok = False
            path = DEMO_DIR / f"boost_{prim}_{idx:03d}_{'ok' if ok else 'fail'}.hdf5"
            recorder.save(path, demo.cfg, ok, prim_ok={prim: bool(ok)})
            recorder.close()
            sess.close()
            n_ok += ok
            print(f"[{idx:03d}] {'成功' if ok else '失败'} "
                  f"{len(recorder.qpos)} ticks {time.time()-t0:.0f}s -> {path.name}",
                  flush=True)
            idx += 1
    print(f"\n完成: {n_ok}/{idx - start} 成功", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="入口池直采原语演示")
    parser.add_argument("--prim", type=str, required=True, choices=list(PRIM_SPECS))
    parser.add_argument("--pools", type=str, default="train",
                        help="入口池 tag 列表（逗号分隔）")
    parser.add_argument("--passes", type=int, default=1,
                        help="每条入口采几遍（DART 噪声保证遍间差异）")
    parser.add_argument("--noise", type=float, default=0.004)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    collect(args.prim, args.pools.split(","), passes=args.passes,
            action_noise=args.noise, seed=args.seed, start=args.start)
