"""分药任务 Gymnasium 环境（v5 场景，域随机化）。

Episode 从"机器人已停在桌前（含停车扰动）"开始，任务：
盒 A 取板 → 撕下目标格入盒 B → 剩板放回盒 A。

- 动作（14 维，ALOHA/ACT 标准布局）：
    [左臂 6 关节目标, 左爪开度, 右臂 6 关节目标, 右爪开度]（位置伺服目标，50 Hz）
- 观测：
    qpos（14 维，与动作同布局）+ 三路机载相机图像（head / wrist_left / wrist_right）
- 奖励：稀疏——目标格入盒 B (+1) 与剩板回槽 (+1)；IL 不用奖励，RL 后续再塑形。

用法:
    env = PillTearEnv(seed=0)
    obs, info = env.reset()
    obs, r, terminated, truncated, info = env.step(action)
"""

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

import tear_scene as ts
from ik_utils import ARM_JOINTS

NEUTRAL_ARM = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0])
CTRL_HZ = 50
IMG_H, IMG_W = 240, 320
CAMS = ("head_cam", "wrist_cam_left", "wrist_cam_right")
GRIP_OPEN = 0.025
TEAR_LOAD = 4.5    # 易撕线断裂载荷阈值（与脚本专家一致）
TEAR_HOLD = 3      # 断裂需持续超阈值的控制周期数（滤掉夹持瞬间冲击）


def actuator_ids14(model):
    """14 维动作布局对应的 actuator id。"""
    ids = []
    for side in ("left", "right"):
        ids += [model.actuator(f"{side}/{j}").id for j in ARM_JOINTS]
        ids.append(model.actuator(f"{side}/gripper").id)
    return np.array(ids)


def qpos_ids14(model):
    """14 维本体感知（关节角 + 爪开度）对应的 qpos 地址。"""
    adr = []
    for side in ("left", "right"):
        adr += [model.joint(f"{side}/{j}").qposadr[0] for j in ARM_JOINTS]
        adr.append(model.joint(f"{side}/left_finger").qposadr[0])
    return np.array(adr)


def tune_model(model):
    """与脚本专家一致的模型调参（伺服刚度、夹爪增益/量程）。"""
    from run_demo import stiffen_arm

    stiffen_arm(model, "left", 15.0)
    stiffen_arm(model, "right", 8.0)
    for side, scale in (("left", 8.0), ("right", 6.0)):
        ga = model.actuator(f"{side}/gripper")
        model.actuator_gainprm[ga.id, 0] *= scale
        model.actuator_biasprm[ga.id, 1] *= scale
        model.actuator_ctrlrange[ga.id, 0] = 0.0
    return model


class PillTearEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": CTRL_HZ}

    def __init__(self, seed=None, rand_level=1.0, image_obs=True,
                 max_secs=60.0, strict_grip=False, img_hw=(IMG_H, IMG_W)):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.rand_level = rand_level
        self.image_obs = image_obs
        # strict_grip=True：断裂前提"边界受夹"要求双指同时接触（规范撕剪物理，
        # 堵死"单指敲断"）；False 保持旧物理（兼容在旧物理下采集/训练的策略）
        self.strict_grip = strict_grip
        # 观测分辨率：默认 240x320；高分辨率实验（毫米级抓取定位的
        # 信息瓶颈检验）用 480x640，须与训练数据分辨率一致
        self.img_hw = tuple(img_hw)
        self.max_steps = int(max_secs * CTRL_HZ)
        self.cfg = None
        self.model = None
        self.data = None
        self.renderer = None
        self._step_count = 0

        obs_dict = {"qpos": spaces.Box(-np.inf, np.inf, (14,), np.float64)}
        if image_obs:
            for cam in CAMS:
                obs_dict[cam] = spaces.Box(0, 255, (*self.img_hw, 3), np.uint8)
        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.Box(-np.pi, np.pi, (14,), np.float64)

    # ---------- 生命周期 ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

        self.cfg = (options or {}).get("cfg") or ts.sample_cfg(self.rng, self.rand_level)
        self.model = tune_model(ts.load_model(self.cfg))
        # 物理随机化档位（评测蒸馏策略的抗物理扰动能力）：
        # 摩擦/质量缩放进模型，断裂阈值缩放存 self._tear_load
        phys = (options or {}).get("phys") or {}
        self._tear_load = TEAR_LOAD * phys.get("thresh", 1.0)
        if phys:
            rf = [self.model.body("right/left_finger_link").id,
                  self.model.body("right/right_finger_link").id]
            pads = [g for g in range(self.model.ngeom)
                    if self.model.geom_bodyid[g] in rf]
            self.model.geom_friction[pads, 0] *= phys.get("fric", 1.0)
            segs = [self.model.body(ts.seg_name(c, r)).id
                    for c, r in ts.all_segments()]
            self.model.body_mass[segs] *= phys.get("mass", 1.0)
            self.model.body_inertia[segs] *= phys.get("mass", 1.0)
        self.data = mujoco.MjData(self.model)
        self.n_sub = int(1.0 / CTRL_HZ / self.model.opt.timestep)
        self.act_ids = actuator_ids14(self.model)
        self.qpos_ids = qpos_ids14(self.model)
        self._step_count = 0

        model, data = self.model, self.data
        ts.set_base(model, data, self.cfg.base_work)   # 停车扰动已含在 cfg 内
        for side in ("left", "right"):
            for jname, q in zip(ARM_JOINTS, NEUTRAL_ARM):
                data.qpos[model.joint(f"{side}/{jname}").qposadr[0]] = q
                data.ctrl[model.actuator(f"{side}/{jname}").id] = q
            data.qpos[model.joint(f"{side}/left_finger").qposadr[0]] = 0.0084
            data.qpos[model.joint(f"{side}/right_finger").qposadr[0]] = 0.0084
            data.ctrl[model.actuator(f"{side}/gripper").id] = GRIP_OPEN
        mujoco.mj_forward(model, data)
        # 可断裂易撕线 = 仅目标格相邻的焊线（与采集演示时的物理一致：
        # 专家只在撕剪目标格时监控断裂；非目标焊线视为不可断）
        self._welds = [model.equality(w).id
                       for w in ts.weld_names_of(*self.cfg.target_seg)]
        self._aws = set(self._welds)
        self._over_cnt = {}
        self._tear_events = []   # 每次断裂事件：是否双指夹持（过程质量口径）
        self._latched = False
        self._tab_geom = model.geom("strip_tab").id
        self._seg_geom = model.geom(f"{ts.seg_name(*self.cfg.target_seg)}_plate").id
        self._lfinger_bodies = {model.body("left/left_finger_link").id,
                                model.body("left/right_finger_link").id}
        self._rfinger_bodies = {model.body("right/left_finger_link").id,
                                model.body("right/right_finger_link").id}
        self._lgrip_act = model.actuator("left/gripper").id
        for _ in range(int(0.5 * CTRL_HZ) * self.n_sub):   # 沉降
            mujoco.mj_step(model, data)
        return self._obs(), {"cfg": self.cfg}

    def _check_grasp_latch(self):
        """左爪锁定作为环境物理（sticky gripper）：闭爪且两指触到手柄 → 锁定；
        张爪 → 解除。与采集演示时脚本专家的 latch 行为一致。"""
        model, data = self.model, self.data
        grip_cmd = data.ctrl[self._lgrip_act]
        if not self._latched:
            if grip_cmd < 0.008:
                ncon = sum(1 for i in range(data.ncon)
                           if (data.contact[i].geom1 == self._tab_geom
                               and model.geom_bodyid[data.contact[i].geom2] in self._lfinger_bodies)
                           or (data.contact[i].geom2 == self._tab_geom
                               and model.geom_bodyid[data.contact[i].geom1] in self._lfinger_bodies))
                if ncon >= 2:
                    ts.engage_latch(model, data)
                    self._latched = True
        elif grip_cmd > 0.015:
            ts.release_latch(model, data)
            self._latched = False

    def _finger_touch_count(self):
        """右爪两根手指中与目标格板接触的手指数（0/1/2）。"""
        model, data = self.model, self.data
        touched = set()
        for i in range(data.ncon):
            g1, g2 = data.contact[i].geom1, data.contact[i].geom2
            if g1 == self._seg_geom and model.geom_bodyid[g2] in self._rfinger_bodies:
                touched.add(model.geom_bodyid[g2])
            elif g2 == self._seg_geom and model.geom_bodyid[g1] in self._rfinger_bodies:
                touched.add(model.geom_bodyid[g1])
        return len(touched)

    def _seg_gripped(self):
        """断裂前提"边界受夹"是否成立。

        strict_grip=True 要求双指同时接触（规范撕剪，与 RL 精修环境一致）；
        False 为旧物理（任一指接触即可，兼容旧数据训练的策略——评测
        环境不能对着旧策略事后改规则，实测改了 ACT v1 从 75% 跌 20%）。"""
        n = self._finger_touch_count()
        return n >= (2 if self.strict_grip else 1)

    def _check_tears(self):
        """易撕线断裂：仅当右爪夹住目标格且载荷连续 TEAR_HOLD 步超阈值。

        纯惯性冲击（提板/转体加速度尖峰）不会沿易撕线撕开——撕裂需要
        "边界受夹 + 持续弯折"。这也与采集演示时专家仅在撕剪阶段判定
        断裂的物理规则一致。"""
        data = self.data
        if not self._active_weld_set or not self._seg_gripped():
            self._over_cnt = {}
            return
        loads = {e: 0.0 for e in self._active_weld_set}
        for i in range(data.nefc):
            if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY:
                e = data.efc_id[i]
                if e in loads:
                    loads[e] += abs(data.efc_force[i])
        for e, load in loads.items():
            if load >= self._tear_load:
                self._over_cnt[e] = self._over_cnt.get(e, 0) + 1
                if self._over_cnt[e] >= TEAR_HOLD:
                    self.model.eq_active0[e] = 0
                    data.eq_active[e] = 0
                    self._active_weld_set.discard(e)
                    # 过程质量记录：断裂瞬间是否双指规范夹持（"撕"）
                    # 还是单指压击（"敲"）——统计口径用，不影响物理
                    self._tear_events.append(self._finger_touch_count() >= 2)
            else:
                self._over_cnt[e] = 0

    @property
    def _active_weld_set(self):
        return self._aws

    def step(self, action):
        self.data.ctrl[self.act_ids] = np.asarray(action, dtype=np.float64)
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        self._check_tears()
        self._check_grasp_latch()
        self._step_count += 1
        seg_ok, board_ok = self._success()
        # 板初始就在槽中，"回槽"只有在目标格已入盒 B 后才有意义
        reward = float(seg_ok) + float(seg_ok and board_ok)
        terminated = bool(seg_ok and board_ok)
        truncated = self._step_count >= self.max_steps
        info = {"seg_in_box_b": seg_ok, "board_returned": seg_ok and board_ok,
                # 规范撕剪 = 所有断裂事件都在双指夹持下发生（"撕"而非"敲"）
                "clean_tear": bool(self._tear_events) and all(self._tear_events)}
        return self._obs(), reward, terminated, truncated, info

    # ---------- 观测与判定 ----------
    def _obs(self):
        mujoco.mj_forward(self.model, self.data)
        obs = {"qpos": self.data.qpos[self.qpos_ids].copy()}
        if self.image_obs:
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=self.img_hw[0],
                                                width=self.img_hw[1])
            for cam in CAMS:
                self.renderer.update_scene(self.data, camera=cam)
                obs[cam] = self.renderer.render().copy()
        return obs

    def _success(self):
        seg = ts.seg_name(*self.cfg.target_seg)
        p = self.data.body(seg).xpos
        seg_ok = (abs(p[0] - self.cfg.box_b_xy[0]) < ts.BOX_B_HX
                  and abs(p[1] - self.cfg.box_b_xy[1]) < ts.BOX_B_HY and p[2] < 0.05)
        bp = self.data.body("strip").xpos
        bR = self.data.body("strip").xmat.reshape(3, 3)
        board_ok = (np.linalg.norm(bp[:2] - self.cfg.board_home[:2]) < 0.025
                    and bp[2] < self.cfg.board_home[2] + 0.015 and bR[2, 0] < -0.9)
        return seg_ok, board_ok

    def privileged(self):
        """特权信息（真值位姿，专家/RL 教师用；策略学习不可用）。"""
        mujoco.mj_forward(self.model, self.data)
        seg = ts.seg_name(*self.cfg.target_seg)
        return {
            "strip_pos": self.data.body("strip").xpos.copy(),
            "strip_mat": self.data.body("strip").xmat.reshape(3, 3).copy(),
            "seg_pos": self.data.body(seg).xpos.copy(),
        }

    def render(self):
        return self._obs().get("head_cam")

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


if __name__ == "__main__":
    env = PillTearEnv(seed=0)
    obs, info = env.reset()
    print("reset ok, cfg:", info["cfg"])
    print("qpos shape:", obs["qpos"].shape,
          "| images:", {c: obs[c].shape for c in CAMS})
    for _ in range(10):
        obs, r, term, trunc, info = env.step(obs["qpos"])   # 保持当前位形
    print("step ok, reward:", r, "info:", info)
    env.close()
    print("环境烟测通过")
