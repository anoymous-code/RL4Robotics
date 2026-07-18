"""撕剪-投放子任务的 RL 精修环境（相位级修正动作 + 物理随机化 + 特权观测）。

动机：ACT 策略的残余失败集中在接触环节（撕剪滑脱、投放弹出）。
本环境把"撕剪目标格 → 投入盒 B"独立成短子任务，在脚本专家的名义编排
之上学习**相位级修正**，用特权状态反馈适应脚本看不见的物理变化：

    - 右爪指垫摩擦系数  ×[0.25, 1.3]（低摩擦 → 撕剪反力下滑脱）
    - 板格质量          ×[0.6, 2.0]
    - 易撕线断裂阈值    ×[0.8, 2.2]（高阈值 → 扭腕不断需补拉）
    - 感知偏移          水平 ±25 mm / 竖直 ±5 mm（手眼标定误差 → 抓浅/投偏）

动作（5 维, [-1,1]，**每编排相位决策一次**，semi-MDP）：
    [0:3] 该相位 IK 目标位置修正 ±30 mm（任务空间残差）
    [3]   夹爪闭合目标修正 ±2 mm（过盈量 = 夹持力）
    [4]   扭腕幅度缩放 ×[0.5, 1.5]
零动作 = 完整复现脚本专家。episode 只有 ~14 个决策点，
credit assignment 远易于逐步残差（第一轮训练的教训：750 步 × 残差惩罚
累积远超成功奖励，策略学成"躺平"还是破坏脚本两难）。

观测（相位入口取样，float32）：相位 one-hot、右臂本体、真值格/板相对
位姿、易撕线载荷、接触数、物理参数 θ 与感知偏移真值（特权信息教师设定）。

Episode 从重置池的状态快照开始（脚本已完成取板→工作位）。重置池由
本文件 --gen-pool 预生成：每个条目 = 场景 XML + 取板后的完整状态。

用法:
    python tear_refine_env.py --gen-pool 64        # 生成重置池
    python tear_refine_env.py --smoke              # 零动作 + 标称物理（应全成）
    python tear_refine_env.py --baseline 40        # 零动作 + 物理随机化（基线）
"""

import argparse
import pickle
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

import tear_scene as ts
from ik_utils import ARM_JOINTS, ArmKinematics
from pill_env import CTRL_HZ, TEAR_HOLD, TEAR_LOAD, actuator_ids14, tune_model
from run_full_demo import (AXES_TEAR, DOWN, EDGE_OVERLAP, GRIP_CLOSE_R,
                           GRIP_OPEN, PADS_LOCAL, PREGRASP_BACKOFF, TWIST_RAD,
                           X, axes_rot)

HERE = Path(__file__).resolve().parent
POOL_PATH = HERE / "demos" / "refine_pool.pkl"

ACT_POS = 0.03      # IK 目标修正上限 (m)
ACT_GRIP = 0.002    # 夹爪目标修正上限 (m)
ACT_TWIST = 0.5     # 扭幅缩放幅度（×[1-0.5, 1+0.5]）

# 名义编排相位（与脚本专家 tear_segments 一致的途经点/时长）
PHASES = ("pre", "grasp", "close", "settle", "twist", "pull",
          "stab", "lift", "drop", "dwell2", "open", "shake", "rest")
DUR = {"pre": 1.8, "grasp": 1.2, "close": 0.8, "settle": 0.6,
       "twist": 3.0, "pull": 1.2, "stab": 0.5, "lift": 1.0, "drop": 2.4,
       "dwell2": 0.3, "open": 0.4, "shake": 1.2, "rest": 1.2}
IK_PHASES = ("pre", "grasp", "pull", "lift", "drop")


def minjerk(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def sample_phys(rng, level=1.0):
    """物理随机化参数 θ（level=0 → 标称）。

    幅度标定原则：让零动作脚本明显退化（不然精修没有可测的提升空间）。
    实测脚本编排对纯物理量（摩擦/质量/阈值）异常鲁棒（40 集仅 1 败），
    真正的失效来源与错标定实验一致——感知偏移：抓浅滑脱 / 投放弹出。
    """
    return {
        "fric": 1.0 + (rng.uniform(-0.75, 0.3)) * level,    # 指垫摩擦 ×[0.25, 1.3]
        "mass": 1.0 + (rng.uniform(-0.4, 1.0)) * level,     # 格质量 ×[0.6, 2.0]
        "thresh": 1.0 + (rng.uniform(-0.2, 1.2)) * level,   # 断裂阈值 ×[0.8, 2.2]
        # 感知偏移取错标定实验的量级（±2.5 cm 手眼标定误差）
        "sense": np.concatenate([rng.uniform(-0.025, 0.025, 2),
                                 rng.uniform(-0.005, 0.005, 1)]) * level,
    }


class TearRefineEnv(gym.Env):
    """相位级修正 RL 精修环境。零动作 = 完整复现脚本专家的撕剪编排。"""

    metadata = {"render_modes": []}

    def __init__(self, pool_path=POOL_PATH, seed=None, phys_level=1.0,
                 pool_indices=None):
        super().__init__()
        with open(pool_path, "rb") as f:
            self.pool = pickle.load(f)
        self.pool_indices = list(pool_indices) if pool_indices is not None \
            else list(range(len(self.pool)))
        self.rng = np.random.default_rng(seed)
        self.phys_level = phys_level
        self._cache = {}
        self.render_hook = None      # 每控制周期回调 hook(data)（评测录像用）

        self.action_space = spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (59,), np.float32)

    # ---------- 模型缓存与重置 ----------
    def _get_model(self, idx):
        """模型/数据/运动学按池条目缓存复用——每次 reset 重新分配 MjData
        会在长训练中耗尽内存（MuJoCo engine error: Could not allocate memory）。"""
        if idx not in self._cache:
            model = tune_model(mujoco.MjModel.from_xml_string(self.pool[idx]["xml"]))
            rf_bodies = [model.body("right/left_finger_link").id,
                         model.body("right/right_finger_link").id]
            pad_geoms = [g for g in range(model.ngeom)
                         if model.geom_bodyid[g] in rf_bodies]
            seg_bodies = [model.body(ts.seg_name(c, r)).id
                          for c, r in ts.all_segments()]
            base = {"fric": model.geom_friction[pad_geoms, 0].copy(),
                    "mass": model.body_mass[seg_bodies].copy(),
                    "inertia": model.body_inertia[seg_bodies].copy()}
            self._cache[idx] = (model, pad_geoms, seg_bodies, base,
                                mujoco.MjData(model),
                                ArmKinematics(model, "right", "right/gripper"))
        return self._cache[idx]

    def _peek_model(self, idx):
        """取某池条目的已缓存模型（渲染器需在 reset 前按模型创建）。"""
        return self._get_model(idx)[0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        idx = (options or {}).get("pool_idx")
        if idx is None:
            idx = int(self.rng.choice(self.pool_indices))
        entry = self.pool[idx]
        self.cfg = ts.SceneCfg(**entry["cfg"])
        self.phys = (options or {}).get("phys") or sample_phys(self.rng, self.phys_level)

        model, pad_geoms, seg_bodies, base, data, right = self._get_model(idx)
        model.geom_friction[pad_geoms, 0] = base["fric"] * self.phys["fric"]
        model.body_mass[seg_bodies] = base["mass"] * self.phys["mass"]
        model.body_inertia[seg_bodies] = base["inertia"] * self.phys["mass"]
        latch_id = model.equality("grasp_latch").id
        model.eq_data[latch_id] = entry["eq_latch"]

        self.model = model
        self.data = data
        mujoco.mj_resetData(model, data)
        data.qpos[:] = entry["qpos"]
        data.qvel[:] = entry["qvel"]
        data.ctrl[:] = entry["ctrl"]
        data.eq_active[latch_id] = 1
        self.n_sub = int(1.0 / CTRL_HZ / model.opt.timestep)
        self.right = right
        self.right_grip = model.actuator("right/gripper").id
        self.wrist_act = model.actuator("right/wrist_rotate").id
        self.act14 = actuator_ids14(model)

        seg = ts.seg_name(*self.cfg.target_seg)
        self.seg_body = model.body(seg).id
        self.seg_geom = model.geom(f"{seg}_plate").id
        self.strip_body = model.body("strip").id
        self.rf_bodies = {model.body("right/left_finger_link").id,
                          model.body("right/right_finger_link").id}
        self.welds = [model.equality(w).id
                      for w in ts.weld_names_of(*self.cfg.target_seg)]
        for e in self.welds:
            data.eq_active[e] = 1          # 模型缓存复用，显式复位易撕线
        self.aws = set(self.welds)
        self.over_cnt = {}
        self.thresh = TEAR_LOAD * self.phys["thresh"]
        mujoco.mj_forward(model, data)
        for _ in range(5 * self.n_sub):    # 快照落地后短沉降
            mujoco.mj_step(model, data)

        self.strip_home = data.body("strip").xpos.copy()
        self.ctrl_nom = data.ctrl.copy()
        self.R_tear = axes_rot(AXES_TEAR)
        self.phase_i = 0
        self._torn_bonus = False
        return self._obs(), {"cfg": self.cfg, "phys": self.phys, "pool_idx": idx}

    # ---------- 名义编排 ----------
    def _sense_seg(self):
        return self.data.body(self.seg_body).xpos.copy() + self.phys["sense"]

    def _site_target(self, true_pos=False):
        seg = (self.data.body(self.seg_body).xpos.copy() if true_pos
               else self._sense_seg())
        grasp = seg + np.array([ts.SEG_HX - EDGE_OVERLAP, 0, 0])
        return grasp - self.R_tear @ PADS_LOCAL

    @property
    def phase(self):
        return PHASES[self.phase_i]

    def _phase_plan(self, corr_pos, corr_grip, twist_scale):
        """相位入口：由名义途经点 + RL 修正生成该相位的执行计划。"""
        p = self.phase
        q_now = self.right.q_now(self.data)
        plan = {"q0": q_now, "qT": q_now}
        if p in IK_PHASES:
            if p == "pre":
                self.ctrl_nom[self.right_grip] = GRIP_OPEN
                tgt = self._site_target() + X * PREGRASP_BACKOFF
            elif p == "grasp":
                tgt = self._site_target()
            elif p == "pull":
                tgt = self.right.site_pos(self.data) + np.array([0.03, 0, -0.04])
            elif p == "lift":
                tgt = self.right.site_pos(self.data) + np.array([0.04, 0, 0.05])
            else:   # drop
                tgt = (self.cfg.box_b_center + self.phys["sense"]
                       + np.array([0, 0, 0.075]))
            axes = {"pre": AXES_TEAR, "grasp": AXES_TEAR, "pull": None,
                    "lift": AXES_TEAR, "drop": [(0, DOWN)]}[p]
            plan["qT"], _, _ = self.right.solve(self.data, tgt + corr_pos,
                                                axes=axes, q_init=q_now)
        elif p == "close":
            plan["grip0"] = self.ctrl_nom[self.right_grip]
            plan["gripT"] = max(0.0, GRIP_CLOSE_R + corr_grip)
        elif p == "twist":
            plan["twist0"] = self.ctrl_nom[self.wrist_act]
            plan["twist_amp"] = TWIST_RAD * twist_scale
        elif p == "open":
            self.ctrl_nom[self.right_grip] = GRIP_OPEN
        elif p == "shake":
            plan["shake0"] = self.ctrl_nom[self.wrist_act]
        return plan

    def _phase_ctrl(self, plan, k, dur):
        p = self.phase
        s = minjerk(min(1.0, k / dur))
        if p in IK_PHASES:
            self.right.command_ctrl(self.ctrl_nom,
                                    plan["q0"] + s * (plan["qT"] - plan["q0"]))
        elif p == "close":
            self.ctrl_nom[self.right_grip] = (
                plan["grip0"] + (k / dur) * (plan["gripT"] - plan["grip0"]))
        elif p == "twist":
            self.ctrl_nom[self.wrist_act] = plan["twist0"] + s * plan["twist_amp"]
        elif p == "shake":
            self.ctrl_nom[self.wrist_act] = plan["shake0"] + 0.28 * np.sin(
                2 * np.pi * 3.5 * k / CTRL_HZ)
            if k >= dur:
                self.ctrl_nom[self.wrist_act] = plan["shake0"]

    # ---------- 物理规则（与 PillTearEnv 一致） ----------
    def _torn(self):
        return not self.aws

    def _gripped(self):
        """撕裂物理前提"边界受夹" = 两侧手指**同时**接触板格。
        单指接触（如张爪怼压）不构成夹持——否则 RL 会学出"猛怼撕断、
        格子自由落体进正下方盒 B"的 reward hacking 捷径。"""
        data, model = self.data, self.model
        touched = set()
        for i in range(data.ncon):
            g1, g2 = data.contact[i].geom1, data.contact[i].geom2
            if g1 == self.seg_geom and model.geom_bodyid[g2] in self.rf_bodies:
                touched.add(model.geom_bodyid[g2])
            elif g2 == self.seg_geom and model.geom_bodyid[g1] in self.rf_bodies:
                touched.add(model.geom_bodyid[g1])
        return len(touched) >= 2

    def _n_contacts(self):
        data, model = self.data, self.model
        return sum(1 for i in range(data.ncon)
                   if (data.contact[i].geom1 == self.seg_geom
                       and model.geom_bodyid[data.contact[i].geom2] in self.rf_bodies)
                   or (data.contact[i].geom2 == self.seg_geom
                       and model.geom_bodyid[data.contact[i].geom1] in self.rf_bodies))

    def _weld_loads(self):
        data = self.data
        loads = {e: 0.0 for e in self.aws}
        for i in range(data.nefc):
            if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY:
                e = data.efc_id[i]
                if e in loads:
                    loads[e] += abs(data.efc_force[i])
        return loads

    def _check_tears(self):
        if not self.aws or not self._gripped():
            self.over_cnt = {}
            return
        for e, load in self._weld_loads().items():
            if load >= self.thresh:
                self.over_cnt[e] = self.over_cnt.get(e, 0) + 1
                if self.over_cnt[e] >= TEAR_HOLD:
                    self.data.eq_active[e] = 0
                    self.aws.discard(e)
            else:
                self.over_cnt[e] = 0

    # ---------- Gym API（semi-MDP：一步 = 一个编排相位） ----------
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        corr_pos = a[:3] * ACT_POS
        corr_grip = a[3] * ACT_GRIP
        twist_scale = 1.0 + a[4] * ACT_TWIST
        plan = self._phase_plan(corr_pos, corr_grip, twist_scale)
        p = self.phase
        dur = max(1, int(DUR[p] * CTRL_HZ))

        reward = -0.05 * float(np.mean(a**2))
        terminated, info = False, {}
        for k in range(1, dur + 1):
            self._phase_ctrl(plan, k, dur)
            lo, hi = self.model.actuator_ctrlrange.T
            self.data.ctrl[:] = np.clip(self.ctrl_nom, lo, hi)
            for _ in range(self.n_sub):
                mujoco.mj_step(self.model, self.data)
            self._check_tears()
            if self.render_hook is not None:
                self.render_hook(self.data)

            if self._torn() and not self._torn_bonus:
                self._torn_bonus = True
                reward += 3.0
            if p in ("twist", "pull") and self._torn():
                break                        # 断裂即停（与脚本一致）

            seg_p = self.data.body(self.seg_body).xpos
            in_box = (abs(seg_p[0] - self.cfg.box_b_xy[0]) < ts.BOX_B_HX
                      and abs(seg_p[1] - self.cfg.box_b_xy[1]) < ts.BOX_B_HY
                      and seg_p[2] < 0.05)
            if self._torn() and not self._gripped():
                seg_v = np.linalg.norm(self.data.body(self.seg_body).cvel[3:])
                if in_box and seg_v < 0.08:
                    reward += 10.0
                    terminated, info["success"] = True, True
                    break
                if seg_p[2] < 0.03 and seg_v < 0.05 and not in_box:
                    reward -= 3.0            # 落到盒外（桌面/弹飞）
                    terminated = True
                    break
            if np.linalg.norm(self.data.body(self.strip_body).xpos
                              - self.strip_home) > 0.05:
                reward -= 3.0                # 把整板拽脱位
                terminated = True
                break

        # 相位级塑形
        if not terminated:
            if p == "grasp":                 # 抓取深度（相对真值抓取点）
                d = np.linalg.norm(self.data.site_xpos[self.right.site_id]
                                   - self._site_target(true_pos=True))
                reward += 1.0 * max(0.0, 1.0 - d / 0.03)
            elif p == "close" and self._n_contacts() >= 2:
                reward += 0.5
            elif p == "pull" and not self._torn():
                reward -= 3.0                # 扭+拉都没撕断
                terminated = True
            elif p == "drop":                # 投放对准（相对真值盒 B 中心）
                d_xy = np.linalg.norm(
                    self.data.body(self.seg_body).xpos[:2] - self.cfg.box_b_xy)
                reward += 1.0 * max(0.0, 1.0 - d_xy / 0.06)

        if not terminated:
            if p == "twist" and self._torn():
                self.phase_i = PHASES.index("stab")
            else:
                self.phase_i += 1
        truncated = (not terminated) and self.phase_i >= len(PHASES)
        if truncated and self._torn() and not self._gripped():
            # 编排走完仍未触发逐步判定（如最后一刻才停稳）：按终态补判
            seg_p = self.data.body(self.seg_body).xpos
            if (abs(seg_p[0] - self.cfg.box_b_xy[0]) < ts.BOX_B_HX
                    and abs(seg_p[1] - self.cfg.box_b_xy[1]) < ts.BOX_B_HY
                    and seg_p[2] < 0.05):
                reward += 10.0
                info["success"] = True
        info.setdefault("success", False)
        info["torn"] = self._torn()
        return self._obs(), float(reward), terminated, truncated, info

    def _obs(self):
        model, data = self.model, self.data
        one_hot = np.zeros(len(PHASES))
        one_hot[min(self.phase_i, len(PHASES) - 1)] = 1.0
        qr = self.right.q_now(data)
        vr = data.qvel[self.right.dof_ids]
        grip_q = data.qpos[model.joint("right/left_finger").qposadr[0]]
        sp = data.site_xpos[self.right.site_id]
        sR = data.site_xmat[self.right.site_id].reshape(3, 3)
        seg_p = data.body(self.seg_body).xpos
        seg_v = data.body(self.seg_body).cvel[3:]
        strip_p = data.body(self.strip_body).xpos
        load = max(self._weld_loads().values(), default=0.0) if self.aws else 0.0
        boxb = self.cfg.box_b_center
        theta = np.array([self.phys["fric"], self.phys["mass"],
                          self.phys["thresh"]])
        return np.concatenate([
            one_hot,
            qr, vr, [grip_q],
            sp, sR[:, 0], sR[:, 1],
            (seg_p - sp), seg_v,
            (strip_p - self.strip_home),
            [min(load / self.thresh, 2.0), float(self._torn()),
             self._n_contacts() / 4.0],
            (boxb - sp), (boxb - seg_p),
            theta, self.phys["sense"] * 100,
        ]).astype(np.float32)


# ArmKinematics 扩展：把关节目标写进给定 ctrl 数组（而非 data.ctrl）
def _command_ctrl(self, ctrl, q_des):
    for act_id, q in zip(self.act_ids, q_des):
        ctrl[act_id] = q


ArmKinematics.command_ctrl = _command_ctrl


# ---------- 重置池生成 ----------
def gen_pool(n, seed=0, path=POOL_PATH):
    from dataclasses import asdict

    from run_full_demo import FullDemo

    rng = np.random.default_rng(seed)
    entries = []
    tries = 0
    while len(entries) < n and tries < n * 2:
        tries += 1
        cfg = ts.sample_cfg(rng)
        demo = FullDemo(cfg=cfg, skip_drive=True, targets=[tuple(cfg.target_seg)],
                        make_video=False, verbose=False)
        try:
            demo.reset()
            demo.dwell(0.5)
            demo.pick_board()
        except Exception as exc:
            print(f"[pool {tries}] 取板异常: {exc}")
            continue
        latch_id = demo.model.equality("grasp_latch").id
        _, bR = demo.strip_pose()
        if demo.data.eq_active[latch_id] != 1 or bR[2, 2] < 0.97:
            print(f"[pool {tries}] 取板质量不合格（latch={demo.data.eq_active[latch_id]}, "
                  f"tilt cos={bR[2, 2]:.3f}），跳过")
            continue
        entries.append({
            "xml": ts.build_xml(cfg),
            "cfg": asdict(cfg),
            "qpos": demo.data.qpos.copy(),
            "qvel": demo.data.qvel.copy(),
            "ctrl": demo.data.ctrl.copy(),
            "eq_latch": demo.model.eq_data[latch_id].copy(),
        })
        print(f"[pool {len(entries)}/{n}] 目标 {ts.seg_name(*cfg.target_seg)} "
              f"盒B ({cfg.box_b_xy[0]:+.3f},{cfg.box_b_xy[1]:+.3f})")
    path.parent.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(entries, f)
    print(f"重置池已保存: {path}（{len(entries)} 条）")


def run_zero(n, phys_level, seed=100):
    """零动作 rollout：phys_level=0 是一致性检查，=1 是脚本基线。"""
    env = TearRefineEnv(seed=seed, phys_level=phys_level)
    wins, torns = 0, 0
    for ep in range(n):
        env.reset()
        done = False
        while not done:
            _, r, term, trunc, info = env.step(np.zeros(5))
            done = term or trunc
        wins += info["success"]
        torns += info["torn"]
        print(f"[ep {ep:02d}] 撕断 {info['torn']}, 入盒 {info['success']} "
              f"(fric {env.phys['fric']:.2f} mass {env.phys['mass']:.2f} "
              f"thr {env.phys['thresh']:.2f} "
              f"sense {np.round(env.phys['sense']*1000, 1)}mm)")
    print(f"\n零动作 @ phys_level={phys_level}: 撕断 {torns}/{n}, 入盒 B {wins}/{n}")
    return wins / n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-pool", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--baseline", type=int, default=0)
    args = parser.parse_args()
    if args.gen_pool:
        gen_pool(args.gen_pool, seed=args.seed)
    if args.smoke:
        run_zero(8, phys_level=0.0)
    if args.baseline:
        run_zero(args.baseline, phys_level=1.0)
