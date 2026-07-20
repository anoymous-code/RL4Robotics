"""原语分解全流程编排器：脚本衔接段 + 学习原语 + 原语级重试。

架构（原语分解 + 经典控制衔接）：
    衔接段永远走 FullDemo 的高精度 IK 编排（approach_board / lift_and_hold /
    approach_seg / transport_to_boxb / retreat_right / approach_slot / ...）；
    4 个接触原语窗口由学习策略接管（--prims 选择替换哪些，未替换的
    保持脚本原语——支持逐原语上线）。

原语失败处理：
    p1/p2  张爪 → 撤回入口关节位形 → 重试（--retries 次）；
    p4     仅 latch 未松时可重试；
    p2 最终失败时张爪撤回，避免夹着未撕断的格拖走整板；
    流程不中断（与脚本专家一致：失败照常走完，按终态判定）。

运行:
    ../../.venv/Scripts/python.exe run_primitives.py --n 20                 # 4 原语全学习
    ../../.venv/Scripts/python.exe run_primitives.py --n 20 --prims p2_tear_seg
    ../../.venv/Scripts/python.exe run_primitives.py --n 20 --prims none    # 纯脚本对照
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

import tear_scene as ts
from eval_prim import EXEC_STEPS, PrimPolicy
from primitives import PRIM_NAMES, PRIM_SPECS, PrimitiveSession, prim_success
from run_full_demo import GRIP_OPEN, FullDemo

HERE = Path(__file__).resolve().parent
VIDEO_DIR = HERE.parents[1] / "docs" / "assets" / "videos"


class FlowVideo:
    """QuadCam 四视角录像（全景/主视角/双腕，1920x1440），挂 FullDemo.recorder。"""

    def __init__(self, model, path):
        from run_full_demo import QuadCam

        self.quad = QuadCam(model)
        self.writer = imageio.get_writer(path, fps=25, quality=7, macro_block_size=1)
        self.k = 0

    def tick(self, model, data):
        self.k += 1
        if self.k % 2:
            return
        self.writer.append_data(self.quad.composite(data))

    def close(self):
        self.writer.close()
        self.quad.close()


def make_impl(policy, retries=1, exec_steps=EXEC_STEPS, img_hw=(480, 640),
              shared=None):
    """把原语策略包装成 FullDemo.primitive_impls 的实现（含入口重试）。

    shared: episode 级共享渲染器缓存 dict——原语窗口各自建/关渲染器会
    杀掉共享 GL 上下文（录像黑屏），改为整个 episode 共用一个。"""
    shared = shared if shared is not None else {}

    def impl(demo, ctx):
        name = demo.phase_name
        spec = PRIM_SPECS[name]
        if shared.get("demo") is not demo:
            # 新 episode：旧渲染器不 close（close 会连坐杀掉本 episode
            # 录像上下文），交给进程退出回收；20 集级别显存可接受
            shared["renderer"] = mujoco.Renderer(demo.model, height=img_hw[0],
                                                 width=img_hw[1])
            shared["demo"] = demo
        sess = PrimitiveSession(demo, name, ctx, img_hw=img_hw,
                                renderer=shared["renderer"])
        arm = demo.left if spec.arm == "left" else demo.right
        grip_act = demo.left_grip if spec.arm == "left" else demo.right_grip
        q_entry = arm.q_now(demo.data)
        grip_entry = demo.data.ctrl[grip_act]
        target_row = int(demo.cfg.target_seg[1])
        latch_id = demo.model.equality("grasp_latch").id

        ok = False
        for attempt in range(retries + 1):
            sess.t0 = demo.t                       # 每次尝试独立计时
            while not sess.timeout() and not ok:
                chunk = policy.predict_chunk(sess.obs(), target_row)
                for k in range(min(exec_steps, len(chunk))):
                    sess.step(chunk[k])
                    if sess.success():
                        ok = True
                        break
                    if sess.timeout():
                        break
            if ok or attempt >= retries:
                break
            # 重试可行性：p3 无法收回；p4 只有板仍被锁定（没松手）时才能再插
            if name == "p3_place":
                break
            if name == "p4_insert" and not demo.data.eq_active[latch_id]:
                break
            demo.log(f"[t={demo.t:6.2f}s] {name} 超时，撤回入口重试 #{attempt + 1}")
            if name in ("p1_grasp_board", "p2_tear_seg"):
                demo.data.ctrl[grip_act] = GRIP_OPEN    # p4 撤回时板必须留在手里
                demo.dwell(0.4)
            arm_q = arm.q_now(demo.data)
            demo.move_joint(arm, q_entry, max(0.8, float(
                np.max(np.abs(arm_q - q_entry)) / 0.6)))
            demo.data.ctrl[grip_act] = grip_entry
            demo.dwell(0.3)

        if name == "p2_tear_seg" and not ok:
            # 防拖板：夹着未撕断的格进运送段会把整板拽走
            demo.data.ctrl[grip_act] = GRIP_OPEN
            demo.dwell(0.4)
            demo.move_joint(arm, q_entry, 1.2)
        if name == "p1_grasp_board" and not ok:
            # 后续 lift_and_hold 依赖 strip_in_site（板-手相对位姿）；
            # 抓取失败也记录当前相对位姿，让流程空手走完而非崩溃
            demo.record_grasp_frame()
        sess.close()
        demo.log(f"[t={demo.t:6.2f}s] 原语 {name}: {'成功' if ok else '失败'}")
        return ok

    return impl


def run_episode(cfg, policies, retries=1, exec_steps=EXEC_STEPS, strict=True,
                entry_jitter=0.0, video_path=None, verbose=False):
    demo = FullDemo(cfg=cfg, skip_drive=True, targets=[tuple(cfg.target_seg)],
                    make_video=False, verbose=verbose, strict_tear=strict,
                    entry_jitter=entry_jitter)
    shared = {}
    for prim, policy in policies.items():
        demo.primitive_impls[prim] = make_impl(policy, retries=retries,
                                               exec_steps=exec_steps,
                                               shared=shared)
    prim_ok = {}

    def on_exit(d, name, ctx):
        prim_ok[name] = bool(prim_success(d, name,
                                          seg=ts.seg_name(*d.cfg.target_seg)))

    for p in PRIM_NAMES:
        demo.primitive_hooks[f"{p}/exit"] = on_exit
    vid = None
    if video_path:
        vid = FlowVideo(demo.model, video_path)
        demo.recorder = vid
    try:
        n_ok, returned = demo.run()
    except Exception as exc:
        print(f"  episode 异常: {exc}")
        n_ok, returned = 0, False
    if vid:
        vid.close()
    return n_ok, returned, prim_ok


def main(n, prims, seed=2000, retries=1, exec_steps=EXEC_STEPS,
         entry_jitter=0.0, video=0, ckpts=None):
    policies = {}
    if prims:
        for prim in prims:
            path = HERE / "ckpt" / ((ckpts or {}).get(prim) or f"{prim}_latest.pt")
            policies[prim] = PrimPolicy(path)
    label = "+".join(prims) if prims else "纯脚本"
    print(f"编排器评测: 学习原语 [{label}], 重试 {retries}, n={n}")

    rng = np.random.default_rng(seed)
    stats = {p: [0, 0] for p in PRIM_NAMES}   # [成功, 出现]
    n_full = n_seg = 0
    quota = {"ok": video, "fail": video}      # 成败案例各录 video 条
    for ep in range(n):
        cfg = ts.sample_cfg(rng)
        vp = None
        if quota["ok"] > 0 or quota["fail"] > 0:
            vp = VIDEO_DIR / f"_tmp_prim_flow_{ep}.mp4"
        n_ok, returned, prim_ok = run_episode(
            cfg, policies, retries=retries, exec_steps=exec_steps,
            entry_jitter=entry_jitter, video_path=vp)
        full = (n_ok == 1) and returned
        if vp is not None:
            kind = "ok" if full else "fail"
            if quota[kind] > 0:
                idx = video - quota[kind]
                dst = VIDEO_DIR / f"prim_flow_{kind}_{idx}.mp4"
                vp.replace(dst)
                quota[kind] -= 1
                print(f"    -> 视频归档 {dst.name}")
            else:
                vp.unlink(missing_ok=True)
        n_full += full
        n_seg += (n_ok == 1)
        for p, ok in prim_ok.items():
            stats[p][0] += ok
            stats[p][1] += 1
        marks = " ".join(f"{p.split('_')[0]}:{'√' if ok else '×'}"
                         for p, ok in prim_ok.items())
        print(f"[ep {ep:02d}] 撕剪入盒 {n_ok == 1}, 全流程 {full} | {marks}",
              flush=True)

    print(f"\n=== 学习原语 [{label}] ===")
    for p in PRIM_NAMES:
        w, t = stats[p]
        src = "策略" if p in policies else "脚本"
        if t:
            print(f"  {p:16s}（{src}）: {w}/{t} = {w/t*100:.0f}%")
    print(f"  撕剪入盒 B: {n_seg}/{n}, 全流程: {n_full}/{n} = {n_full/n*100:.0f}%")
    return n_full / n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--prims", type=str, nargs="*", default=list(PRIM_NAMES),
                        help="用学习策略替换的原语（none = 纯脚本对照）")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--exec-steps", type=int, default=EXEC_STEPS)
    parser.add_argument("--entry-jitter", type=float, default=0.0,
                        help="衔接段到位偏差（评测鲁棒性用）")
    parser.add_argument("--video", type=int, default=0,
                        help="成功/失败案例各录 N 条四视角视频")
    parser.add_argument("--ckpt", type=str, nargs="*", default=[],
                        help="覆盖检查点：prim=ckpt文件名（默认 {prim}_latest.pt）")
    args = parser.parse_args()
    prims = [p for p in args.prims if p != "none"]
    for p in prims:
        assert p in PRIM_NAMES, f"未知原语 {p}"
    main(args.n, prims, seed=args.seed, retries=args.retries,
         exec_steps=args.exec_steps, entry_jitter=args.entry_jitter,
         video=args.video, ckpts=dict(kv.split("=", 1) for kv in args.ckpt))
