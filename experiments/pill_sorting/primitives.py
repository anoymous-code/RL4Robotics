"""操作原语层：规格、出口谓词、入口状态快照/恢复与执行会话。

原语分解 + 经典控制衔接架构：FullDemo 的流程 = 衔接控制段（IK 高精度
运送，永远走脚本）+ 4 个接触密集原语窗口（可被学习策略替换）：

    p1_grasp_board  左臂抓药板手柄 → latch 锁定
    p2_tear_seg     右臂夹目标格缘 + 扭断易撕线（毫米级定位瓶颈所在）
    p3_place        盒 B 上方松爪投放 + 抖净
    p4_insert       剩板对槽下插 + 松爪

本文件提供：
    PRIM_SPECS        原语规格（执行臂 / 观测相机 / 14 维动作切片 / 时长上限）
    prim_success      出口谓词（真值判定；训练打标与评测计分共用同一口径）
    snapshot_entry    原语入口状态快照（入口状态池条目，可 pickle）
    restore_session   从池条目重建 FullDemo → PrimitiveSession
    PrimitiveSession  单原语执行会话：渲染观测 / 单臂步进 / 物理规则回调
                      （易撕线断裂 watcher、左爪 sticky latch）/ 谓词判定
"""

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

import tear_scene as ts
from pill_env import actuator_ids14, qpos_ids14
from run_full_demo import FullDemo

HERE = Path(__file__).resolve().parent
PRIM_DIR = HERE / "demos" / "prim"


@dataclass(frozen=True)
class PrimSpec:
    arm: str            # 执行臂："left" / "right"
    cam: str            # 观测相机（该臂腕相机）
    act_lo: int         # 14 维动作布局中该臂的切片 [act_lo, act_hi)
    act_hi: int
    max_secs: float     # 学习策略执行时长上限（脚本版时长 × ~1.5）
    scripted: str       # FullDemo 上的脚本原语方法名


# 14 维布局：[左臂 6 关节, 左爪, 右臂 6 关节, 右爪]
PRIM_SPECS = {
    "p1_grasp_board": PrimSpec("left", "wrist_cam_left", 0, 7, 6.0,
                               "grasp_board_scripted"),
    "p2_tear_seg": PrimSpec("right", "wrist_cam_right", 7, 14, 18.0,
                            "tear_seg_scripted"),
    "p3_place": PrimSpec("right", "wrist_cam_right", 7, 14, 5.0,
                         "place_scripted"),
    "p4_insert": PrimSpec("left", "wrist_cam_left", 0, 7, 8.0,
                          "insert_scripted"),
}
PRIM_NAMES = tuple(PRIM_SPECS)


# ---------------- 出口谓词（真值判定） ----------------
def prim_success(demo, name, seg=None):
    """原语出口成功谓词。seg = 目标格 body 名（p2/p3 需要）。"""
    model, data = demo.model, demo.data
    mujoco.mj_forward(model, data)
    if name == "p1_grasp_board":
        # 手柄夹持建立（sticky latch 锁定）
        return bool(data.eq_active[model.equality("grasp_latch").id])
    if name == "p2_tear_seg":
        # 目标格全部易撕线断开（strict 物理下断裂本身要求双指夹持，
        # 即"规范撕剪"），且格仍夹在右爪中（可运送）
        welds = ts.weld_names_of(*demo.cfg.target_seg)
        torn = all(w in demo.broken for w in welds)
        return torn and demo.finger_touch_count(seg) >= 1
    if name == "p3_place":
        p = data.body(seg).xpos
        return bool(abs(p[0] - demo.cfg.box_b_xy[0]) < ts.BOX_B_HX
                    and abs(p[1] - demo.cfg.box_b_xy[1]) < ts.BOX_B_HY
                    and p[2] < 0.05)
    if name == "p4_insert":
        # 必须已松开（latch 解除）：板到位但仍被夹持时撤臂会把板拔出
        if data.eq_active[model.equality("grasp_latch").id]:
            return False
        bp = data.body("strip").xpos          # 真值判定（不加 sense_offset）
        bR = data.body("strip").xmat.reshape(3, 3)
        return bool(np.linalg.norm(bp[:2] - demo.cfg.board_home[:2]) < 0.025
                    and bp[2] < demo.cfg.board_home[2] + 0.015
                    and bR[2, 0] < -0.9)
    raise ValueError(name)


# ---------------- 入口状态快照 / 恢复 ----------------
def snapshot_entry(demo, name, ctx):
    """原语入口的完整状态快照（挂 FullDemo.primitive_hooks 时调用）。"""
    model, data = demo.model, demo.data
    latch_id = model.equality("grasp_latch").id
    return {
        "prim": name,
        "cfg": asdict(demo.cfg),
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "ctrl": data.ctrl.copy(),
        "eq_active": data.eq_active.copy(),
        "eq_latch": model.eq_data[latch_id].copy(),
        "broken": sorted(demo.broken),
        "strip_in_site": demo.strip_in_site,
        # ctx 中可 pickle 的字段（watch 闭包恢复时按 broken 重建）
        "ctx": {k: v for k, v in ctx.items() if k != "watch"},
    }


def restore_session(entry, strict_tear=True, img_hw=(480, 640)):
    """从池条目重建 FullDemo（状态恢复到原语入口）→ PrimitiveSession。"""
    cfg = ts.SceneCfg(**entry["cfg"])
    demo = FullDemo(cfg=cfg, skip_drive=True, targets=[tuple(cfg.target_seg)],
                    make_video=False, verbose=False, strict_tear=strict_tear)
    demo.reset()
    model, data = demo.model, demo.data
    data.qpos[:] = entry["qpos"]
    data.qvel[:] = entry["qvel"]
    data.ctrl[:] = entry["ctrl"]
    model.eq_data[model.equality("grasp_latch").id] = entry["eq_latch"]
    data.eq_active[:] = entry["eq_active"]
    demo.broken = set(entry["broken"])
    for w in demo.broken:
        model.eq_active0[model.equality(w).id] = 0
    demo.strip_in_site = entry["strip_in_site"]
    mujoco.mj_forward(model, data)
    return PrimitiveSession(demo, entry["prim"], dict(entry["ctx"]),
                            img_hw=img_hw)


class PrimitiveSession:
    """单原语执行会话（评测 / 基线复现用）。

    step() 只接受该原语执行臂的 7 维动作，另一臂 ctrl 保持入口值；
    每控制周期执行物理规则回调：
        p2 → 易撕线断裂 watcher（与脚本专家一致的 strict 物理）
        p1/p4 → 左爪 sticky latch（闭爪触手柄自动锁定 / 张爪解除，
                 与 PillTearEnv._check_grasp_latch 同规则——学习策略
                 没有脚本的显式 engage_latch 调用，物理必须自动生效）
    """

    def __init__(self, demo, prim, ctx, img_hw=(480, 640), renderer=None):
        self.demo = demo
        self.prim = prim
        self.spec = PRIM_SPECS[prim]
        self.ctx = ctx
        self.img_hw = tuple(img_hw)
        # 渲染器可外部注入（编排器多原语共享）：mujoco Renderer.close()
        # 会连带杀掉共享 GL 上下文，把同 episode 其他渲染器（如录像）
        # 变成黑屏——注入的渲染器由持有者负责生命周期
        self._ext_renderer = renderer is not None
        model = demo.model
        self.seg = ts.seg_name(*demo.cfg.target_seg)
        self.act14 = actuator_ids14(model)
        self.qpos14 = qpos_ids14(model)
        self._renderer = renderer
        self.t0 = demo.t

        # p2：重建断裂 watcher（ctx 里的 watch 闭包不进快照）
        if prim == "p2_tear_seg":
            welds = [w for w in ts.weld_names_of(*demo.cfg.target_seg)
                     if w not in demo.broken]
            self.ctx.setdefault("target_welds", welds)
            self.watch = demo.make_watcher(welds, seg=self.seg)
            self.ctx["watch"] = self.watch
        else:
            self.watch = None

        # p1/p4：sticky latch 自动规则的初始状态
        self._latch_id = model.equality("grasp_latch").id
        self._latched = bool(demo.data.eq_active[self._latch_id])
        self._tab_geom = model.geom("strip_tab").id
        self._lfingers = {model.body("left/left_finger_link").id,
                          model.body("left/right_finger_link").id}
        self._lgrip = model.actuator("left/gripper").id

    # ---------- 观测 ----------
    def obs(self):
        """原语观测：该原语相机图像 + 14 维 qpos。"""
        demo = self.demo
        mujoco.mj_forward(demo.model, demo.data)
        if self._renderer is None:
            self._renderer = mujoco.Renderer(demo.model, height=self.img_hw[0],
                                             width=self.img_hw[1])
        self._renderer.update_scene(demo.data, camera=self.spec.cam)
        return {self.spec.cam: self._renderer.render().copy(),
                "qpos": demo.data.qpos[self.qpos14].copy()}

    # ---------- 步进 ----------
    def _auto_latch(self):
        demo = self.demo
        model, data = demo.model, demo.data
        grip_cmd = data.ctrl[self._lgrip]
        if not self._latched:
            if grip_cmd < 0.008:
                ncon = sum(
                    1 for i in range(data.ncon)
                    if (data.contact[i].geom1 == self._tab_geom
                        and model.geom_bodyid[data.contact[i].geom2] in self._lfingers)
                    or (data.contact[i].geom2 == self._tab_geom
                        and model.geom_bodyid[data.contact[i].geom1] in self._lfingers))
                if ncon >= 2:
                    ts.engage_latch(model, data)
                    self._latched = True
                    demo.record_grasp_frame()
        elif grip_cmd > 0.015:
            ts.release_latch(model, data)
            self._latched = False

    def step(self, action7):
        """执行臂 7 维位置伺服目标（另一臂保持），推进一个控制周期。"""
        data = self.demo.data
        ids = self.act14[self.spec.act_lo:self.spec.act_hi]
        lo, hi = self.demo.model.actuator_ctrlrange[ids].T
        data.ctrl[ids] = np.clip(np.asarray(action7, dtype=np.float64), lo, hi)
        self.demo.step_ctrl(watch=self.watch)
        if self.prim in ("p1_grasp_board", "p4_insert"):
            self._auto_latch()

    def elapsed(self):
        return self.demo.t - self.t0

    def timeout(self):
        return self.elapsed() >= self.spec.max_secs

    # ---------- 判定与基线 ----------
    def success(self):
        return prim_success(self.demo, self.prim, seg=self.seg)

    def run_scripted(self):
        """脚本原语基线（快照恢复一致性检查）。"""
        scripted = getattr(self.demo, self.spec.scripted)
        self.demo.run_primitive(self.prim, scripted, ctx=self.ctx)
        return self.success()

    def close(self):
        if self._renderer is not None and not self._ext_renderer:
            self._renderer.close()
        self._renderer = None


# ---------------- 入口状态池 ----------------
def pool_path(prim, tag="train"):
    return PRIM_DIR / f"pool_{prim}_{tag}.pkl"


def load_pool(prim, tag="train", only_ok=True):
    """载入原语入口状态池。only_ok：只取脚本原语从该入口能成功的条目
    （可行性保证——评测口径 = "衔接段送到位后，原语能否完成"）。"""
    with open(pool_path(prim, tag), "rb") as f:
        pool = pickle.load(f)
    if only_ok:
        pool = [e for e in pool if e["scripted_ok"]]
    return pool


def gen_pools(n_episodes, seed=0, entry_jitter=0.004, strict=True, tag="train",
              prims=PRIM_NAMES):
    """跑 n 条全流程脚本专家，在每个原语入口打状态快照。

    出口谓词回填 scripted_ok（脚本原语从该入口是否成功）；衔接段带
    entry_jitter 到位偏差，池覆盖"经典控制交接时的定位误差"分布。
    快照在 episode 结束后统一落盘（谓词已回填）。
    """
    rng = np.random.default_rng(seed)
    pools = {p: [] for p in prims}
    for ep in range(n_episodes):
        cfg = ts.sample_cfg(rng)
        demo = FullDemo(cfg=cfg, skip_drive=True, targets=[tuple(cfg.target_seg)],
                        make_video=False, verbose=False, strict_tear=strict,
                        entry_jitter=entry_jitter)
        pending = {}

        def on_entry(d, name, ctx):
            pending[name] = snapshot_entry(d, name, ctx)

        def on_exit(d, name, ctx):
            snap = pending.pop(name, None)
            if snap is not None:
                seg = ts.seg_name(*d.cfg.target_seg)
                snap["scripted_ok"] = prim_success(d, name, seg=seg)
                pools[name].append(snap)

        for p in prims:
            demo.primitive_hooks[p] = on_entry
            demo.primitive_hooks[f"{p}/exit"] = on_exit
        try:
            n_ok, returned = demo.run()
        except Exception as exc:
            print(f"[pool ep {ep:03d}] 专家执行异常: {exc}")
            continue
        marks = " ".join(f"{p}:{'√' if pools[p] and pools[p][-1]['scripted_ok'] else '×'}"
                         for p in prims if pools[p])
        print(f"[pool ep {ep:03d}] 撕剪 {n_ok}/1 回槽 {returned} | {marks}", flush=True)

    PRIM_DIR.mkdir(parents=True, exist_ok=True)
    for p in prims:
        path = pool_path(p, tag)
        with open(path, "wb") as f:
            pickle.dump(pools[p], f)
        ok = sum(e["scripted_ok"] for e in pools[p])
        print(f"{p}: {len(pools[p])} 条（脚本可行 {ok}）-> {path.name}")


def smoke(n=4, seed=7):
    """快照恢复一致性检查：入口池条目 → restore → 脚本原语应大概率成功。"""
    rng = np.random.default_rng(seed)
    for prim in PRIM_NAMES:
        pool = load_pool(prim, only_ok=True)
        if not pool:
            print(f"{prim}: 池为空，跳过")
            continue
        wins = 0
        idxs = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        for i in idxs:
            sess = restore_session(pool[i])
            ok = sess.run_scripted()
            wins += ok
            sess.close()
        print(f"{prim}: 恢复后脚本原语 {wins}/{len(idxs)} 成功")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="原语入口状态池")
    parser.add_argument("--gen", type=int, default=0, help="生成池：全流程 episode 数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jitter", type=float, default=0.004,
                        help="衔接段到位偏差半径 (m)")
    parser.add_argument("--tag", type=str, default="train", help="池文件名后缀")
    parser.add_argument("--smoke", action="store_true", help="快照恢复一致性检查")
    args = parser.parse_args()
    if args.gen:
        gen_pools(args.gen, seed=args.seed, entry_jitter=args.jitter, tag=args.tag)
    if args.smoke:
        smoke()
