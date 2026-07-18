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
                 max_secs=60.0):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.rand_level = rand_level
        self.image_obs = image_obs
        self.max_steps = int(max_secs * CTRL_HZ)
        self.cfg = None
        self.model = None
        self.data = None
        self.renderer = None
        self._step_count = 0

        obs_dict = {"qpos": spaces.Box(-np.inf, np.inf, (14,), np.float64)}
        if image_obs:
            for cam in CAMS:
                obs_dict[cam] = spaces.Box(0, 255, (IMG_H, IMG_W, 3), np.uint8)
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
        for _ in range(int(0.5 * CTRL_HZ) * self.n_sub):   # 沉降
            mujoco.mj_step(model, data)
        return self._obs(), {"cfg": self.cfg}

    def step(self, action):
        self.data.ctrl[self.act_ids] = np.asarray(action, dtype=np.float64)
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        seg_ok, board_ok = self._success()
        # 板初始就在槽中，"回槽"只有在目标格已入盒 B 后才有意义
        reward = float(seg_ok) + float(seg_ok and board_ok)
        terminated = bool(seg_ok and board_ok)
        truncated = self._step_count >= self.max_steps
        info = {"seg_in_box_b": seg_ok, "board_returned": seg_ok and board_ok}
        return self._obs(), reward, terminated, truncated, info

    # ---------- 观测与判定 ----------
    def _obs(self):
        mujoco.mj_forward(self.model, self.data)
        obs = {"qpos": self.data.qpos[self.qpos_ids].copy()}
        if self.image_obs:
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)
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
