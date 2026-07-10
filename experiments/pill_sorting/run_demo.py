"""分药任务 v0：ALOHA 双臂脚本化演示。

编排方式：离线 IK 解出各关键位姿的关节构型，再用关节空间最小加加速度插值执行。

流程：
  1. 左臂夹持药板（3 格泡罩，底面朝下为铝膜），移动到药杯正上方；
  2. 右臂持按压杆逐格下压；
  3. 铝膜以"药片-铝膜法向接触力阈值"模拟破裂——超阈值后该格铝膜失效，药片坠入药杯；
  4. 全程录像，并记录按压力曲线。

运行:
    cd experiments/pill_sorting && ../../.venv/Scripts/python.exe run_demo.py
"""

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from ik_utils import ARM_JOINTS, ArmKinematics

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
VIDEO_DIR = PROJECT_ROOT / "docs" / "assets" / "videos"
IMAGE_DIR = PROJECT_ROOT / "docs" / "assets" / "images"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- 参数 ----------------
NEUTRAL_ARM = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0])
GRIPPER_Q = 0.0084
UP = np.array([0.0, 0.0, 1.0])

CTRL_HZ = 50
FPS = 25
RUPTURE_FORCE = 1.0      # 模拟铝膜破裂的法向力阈值 (N)
CUP_TOP = np.array([0.08, 0.05, 0.15])  # 杯口正上方（按压时目标泡罩对准这里）
HOVER = 0.03             # 按压前悬停高度 (m)
PRESS_DEPTH = 0.02       # 相对药片中心的下压目标深度 (m)
LEFT_KP_SCALE = 15.0     # 左臂（持板）伺服刚度放大
RIGHT_KP_SCALE = 8.0     # 右臂（按压）伺服刚度放大


def stiffen_arm(model, side, scale):
    for jname in ARM_JOINTS:
        a = model.actuator(f"{side}/{jname}")
        model.actuator_gainprm[a.id, 0] *= scale
        model.actuator_biasprm[a.id, 1] *= scale


def minjerk(alpha):
    """最小加加速度插值系数 s(α)，α∈[0,1]。"""
    return 10 * alpha**3 - 15 * alpha**4 + 6 * alpha**5


def contact_normal_force(model, data, g1_id, g2_id):
    total = 0.0
    force = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if {c.geom1, c.geom2} == {g1_id, g2_id}:
            mujoco.mj_contactForce(model, data, i, force)
            total += abs(force[0])
    return total


class Demo:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(HERE / "pill_scene.xml"))
        stiffen_arm(self.model, "left", LEFT_KP_SCALE)
        stiffen_arm(self.model, "right", RIGHT_KP_SCALE)
        self.data = mujoco.MjData(self.model)
        self.n_sub = int(1.0 / CTRL_HZ / self.model.opt.timestep)
        self.left = ArmKinematics(self.model, "left", "pack_center")
        self.right = ArmKinematics(self.model, "right", "stylus_tip_site")
        self.renderer = mujoco.Renderer(self.model, height=720, width=1280)
        self.frames = []
        self.t = 0.0
        self.force_log = {"t": [], "force": [], "pocket": []}
        self.ruptures = []

    # ---------- 基础设施 ----------
    def reset(self):
        model, data = self.model, self.data
        for side in ("left", "right"):
            for jname, q in zip(ARM_JOINTS, NEUTRAL_ARM):
                data.qpos[model.joint(f"{side}/{jname}").qposadr[0]] = q
            for fname in ("left_finger", "right_finger"):
                data.qpos[model.joint(f"{side}/{fname}").qposadr[0]] = GRIPPER_Q
            for aname, q in zip(ARM_JOINTS, NEUTRAL_ARM):
                data.ctrl[model.actuator(f"{side}/{aname}").id] = q
            data.ctrl[model.actuator(f"{side}/gripper").id] = GRIPPER_Q
        mujoco.mj_forward(model, data)

        # 药片放入泡罩格（贴住铝膜上表面）
        pack = data.body("pill_pack")
        R = pack.xmat.reshape(3, 3)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, pack.xmat)
        for i, px in enumerate(self.model.numeric("pocket_xs").data):
            world = pack.xpos + R @ np.array([px, 0.0, -0.001])
            adr = model.joint(f"pill_{i}_joint").qposadr[0]
            data.qpos[adr : adr + 3] = world
            data.qpos[adr + 3 : adr + 7] = quat
        mujoco.mj_forward(model, data)

    def step_ctrl(self, extra_cb=None):
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        self.t += 1.0 / CTRL_HZ
        if extra_cb:
            extra_cb()
        if len(self.frames) * (1.0 / FPS) <= self.t:
            self.renderer.update_scene(self.data, camera="follow_pack")
            self.frames.append(self.renderer.render().copy())

    def move_joint(self, arm, q_target, secs, extra_cb=None):
        """关节空间最小加加速度轨迹。"""
        q0 = arm.q_now(self.data)
        steps = max(1, int(secs * CTRL_HZ))
        for k in range(steps):
            s = minjerk((k + 1) / steps)
            arm.command(self.data, q0 + s * (q_target - q0))
            self.step_ctrl(extra_cb)

    def dwell(self, secs, extra_cb=None):
        for _ in range(int(secs * CTRL_HZ)):
            self.step_ctrl(extra_cb)

    # ---------- 任务编排 ----------
    def run(self):
        self.reset()
        model, data = self.model, self.data

        # 阶段 0：沉降
        self.dwell(0.5)

        # 阶段 1：左臂将药板移到药杯上方（经由中间过渡点，避免大摆动）
        q_mid, e, a = self.left.solve(
            data, np.array([-0.12, 0.02, 0.22]), target_zaxis=UP, q_init=NEUTRAL_ARM)
        print(f"IK 左臂过渡点: 误差 {e*1000:.1f}mm/{a:.1f}°")
        self.move_joint(self.left, q_mid, 1.6)
        q_hold, e, a = self.left.solve(
            data, CUP_TOP - np.array([0.02, 0, 0]), target_zaxis=UP, q_init=q_mid)
        print(f"IK 左臂持板位: 误差 {e*1000:.1f}mm/{a:.1f}°")
        self.move_joint(self.left, q_hold, 1.8)
        self.dwell(0.8)

        # 阶段 2：右臂逐格按压
        for pocket in range(3):
            pill_body = data.body(f"pill_{pocket}")
            pill_geom_id = model.geom(f"pill_{pocket}_geom").id
            foil_id = model.geom(f"foil_{pocket}").id

            # 2a. 左臂微调：把当前泡罩格对准杯口正中（迭代两次消除偏航影响）
            for _ in range(2):
                mujoco.mj_forward(model, data)
                offset = data.site(f"pocket_{pocket}").xpos - data.site("pack_center").xpos
                pack_target = CUP_TOP - offset
                pack_target[2] = CUP_TOP[2]
                q_aim, e, a = self.left.solve(
                    data, pack_target, target_zaxis=UP, q_init=self.left.q_now(data))
                self.move_joint(self.left, q_aim, 1.0)
            mujoco.mj_forward(model, data)
            aim_err = np.linalg.norm(
                data.site(f"pocket_{pocket}").xpos[:2] - CUP_TOP[:2])
            print(f"[t={self.t:6.2f}s] 泡罩 {pocket} 对准杯口，偏差 {aim_err*1000:.1f} mm")

            # 2b. 悬停目标 = 药片当前位置上方 HOVER
            hover_pos = pill_body.xpos + np.array([0, 0, HOVER])
            q_hover, e, a = self.right.solve(
                data, hover_pos, target_zaxis=UP, q_init=NEUTRAL_ARM)
            print(f"IK 右臂悬停[{pocket}]: 误差 {e*1000:.1f}mm/{a:.1f}°")
            self.move_joint(self.right, q_hover, 1.6)
            self.dwell(0.4)

            # 2b. 下压：慢速压至目标深度并保压观察；未破膜则加深重试
            peak = 0.0
            state = {"ruptured": False}

            def watch_force():
                nonlocal peak
                if state["ruptured"]:
                    return
                f = contact_normal_force(model, data, pill_geom_id, foil_id)
                self.force_log["t"].append(self.t)
                self.force_log["force"].append(f)
                self.force_log["pocket"].append(pocket)
                peak = max(peak, f)
                if f >= RUPTURE_FORCE:
                    model.geom_contype[foil_id] = 0
                    model.geom_conaffinity[foil_id] = 0
                    model.geom_rgba[foil_id] = [0.76, 0.76, 0.8, 0.15]
                    self.ruptures.append((self.t, pocket, f))
                    state["ruptured"] = True
                    print(f"[t={self.t:6.2f}s] 泡罩 {pocket} 铝膜破裂，触发力 {f:.2f} N")

            for attempt, depth in enumerate((PRESS_DEPTH, PRESS_DEPTH + 0.012)):
                mujoco.mj_forward(model, data)
                press_pos = pill_body.xpos + np.array([0, 0, -depth])
                q_press, e, a = self.right.solve(
                    data, press_pos, target_zaxis=UP, q_init=self.right.q_now(data))
                print(f"IK 右臂按压[{pocket}] 第{attempt+1}次(深度{depth*1000:.0f}mm): "
                      f"误差 {e*1000:.1f}mm/{a:.1f}°")
                q0 = self.right.q_now(data)
                steps = int(2.0 * CTRL_HZ)
                for k in range(steps):
                    if state["ruptured"]:
                        break
                    s = minjerk((k + 1) / steps)
                    self.right.command(data, q0 + s * (q_press - q0))
                    self.step_ctrl(watch_force)
                # 保压观察 0.8 s
                for _ in range(int(0.8 * CTRL_HZ)):
                    if state["ruptured"]:
                        break
                    self.step_ctrl(watch_force)
                if state["ruptured"]:
                    break
            if not state["ruptured"]:
                print(f"[t={self.t:6.2f}s] 泡罩 {pocket} 未破膜（峰值 {peak:.2f} N）")

            # 2c. 等药片落下，再抬杆
            self.dwell(0.7)
            self.move_joint(self.right, q_hover, 1.0)

        # 阶段 3：右臂回中立位，定格
        q_neutral = NEUTRAL_ARM.copy()
        self.move_joint(self.right, q_neutral, 1.5)
        self.dwell(1.0)

        # 统计
        cup_xy = np.array(model.numeric("cup_center").data)
        in_cup = 0
        for i in range(3):
            p = data.body(f"pill_{i}").xpos
            ok = np.linalg.norm(p[:2] - cup_xy) < 0.055 and p[2] < 0.08
            in_cup += ok
            print(f"pill_{i} 最终位置 {np.round(p, 3)} -> {'√ 入杯' if ok else '× 未入杯'}")
        print(f"成功入杯: {in_cup}/3")

        out = VIDEO_DIR / "pill_demo_v0.mp4"
        imageio.mimsave(out, self.frames, fps=FPS, macro_block_size=1)
        print(f"视频: {out}（{len(self.frames)} 帧, {len(self.frames)/FPS:.1f} s）")
        self.renderer.close()
        self.plot_force()
        return in_cup

    def plot_force(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(9, 4.2), dpi=130)
        tarr = np.array(self.force_log["t"])
        farr = np.array(self.force_log["force"])
        parr = np.array(self.force_log["pocket"])
        colors = ["#e4604e", "#eec84d", "#66bf8c"]
        for pk in range(3):
            m = parr == pk
            if m.any():
                ax.plot(tarr[m], farr[m], color=colors[pk], lw=1.8, label=f"泡罩 {pk}")
        for rt, pk, f in self.ruptures:
            ax.axvline(rt, color=colors[pk], ls="--", lw=1, alpha=0.6)
            ax.annotate(f"破膜 {f:.1f}N", (rt, f), textcoords="offset points",
                        xytext=(6, 6), fontsize=9, color=colors[pk])
        ax.axhline(RUPTURE_FORCE, color="gray", ls=":", lw=1.2,
                   label=f"破膜阈值 {RUPTURE_FORCE}N")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("药片-铝膜法向接触力 (N)")
        ax.set_title("分药演示 v0：按压力曲线与破膜事件")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(IMAGE_DIR / "pill_demo_v0_force.png")
        print(f"力曲线: {IMAGE_DIR / 'pill_demo_v0_force.png'}")


if __name__ == "__main__":
    Demo().run()
