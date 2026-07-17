"""分药任务 v2：双臂协同撕剪 8 格铝塑板，单格（连药）投入托盘。

流程（对每个目标格）：
  1. 左臂持板移到工作位；
  2. 右臂张开夹爪，从前方水平接近，夹住目标格外缘（真实摩擦夹持，无吸附作弊）；
  3. 扭转 + 下拉，易撕线（焊接约束）载荷超阈值即断裂；
  4. 提起撕下的单格，移到托盘上方松爪投放；
  5. 记录易撕线载荷曲线与断裂事件，三机位同步录像。

运行:
    cd experiments/pill_sorting && ../../.venv/Scripts/python.exe run_tear_demo.py
"""

import argparse

import imageio.v2 as imageio
import mujoco
import numpy as np

import tear_scene as ts
from ik_utils import ARM_JOINTS, ArmKinematics
from run_demo import (FPS, CTRL_HZ, IMAGE_DIR, VIDEO_DIR, LiveWindow, MultiCam,
                      minjerk, stiffen_arm)

NEUTRAL_ARM = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0])
UP = np.array([0.0, 0.0, 1.0])

STRIP_HOLD = np.array([0.00, 0.03, 0.17])   # 左手持板位（strip 中心）
APPROACH_DIR = np.array([1.0, 0.0, 0.0])    # 右手从 +x 自由端水平接近（指向 -x）
GRIP_OPEN = 0.025
GRIP_CLOSE = 0.0018
TIP_BEHIND_SITE = 0.0138  # 指尖在 gripper 站点后方的距离（实测标定）
EDGE_OVERLAP = 0.007      # 指尖与板缘的重叠量（穹顶无碰撞，可放心深捏）
PREGRASP_BACKOFF = 0.05   # 预抓取后撤距离 (m)
TEAR_LOAD = 6.0           # 易撕线断裂载荷阈值（efc 合力）
TWIST_RAD = 0.7           # 撕剪扭转角 (rad)，断裂即停
LEFT_KP_SCALE = 15.0
RIGHT_KP_SCALE = 8.0

# 从自由端顺序撕（人撕铝板的方式）：先第 4 列前排，再第 4 列后排
TEAR_TARGETS = [(3, 0), (3, 1)]


class TearDemo:
    def __init__(self, live=False):
        self.model = ts.load_model()
        stiffen_arm(self.model, "left", LEFT_KP_SCALE)
        stiffen_arm(self.model, "right", RIGHT_KP_SCALE)
        # 右夹爪握力增强（薄板夹持全靠指尖过盈量，原 kp 握力不足）
        ga = self.model.actuator("right/gripper")
        self.model.actuator_gainprm[ga.id, 0] *= 3.0
        self.model.actuator_biasprm[ga.id, 1] *= 3.0
        self.data = mujoco.MjData(self.model)
        self.n_sub = int(1.0 / CTRL_HZ / self.model.opt.timestep)
        self.left = ArmKinematics(self.model, "left", "strip_center")
        self.right = ArmKinematics(self.model, "right", "right/gripper")
        self.right_grip = self.model.actuator("right/gripper").id
        self.cams = MultiCam(self.model)
        self.live = None
        if live:
            try:
                self.live = LiveWindow()
            except Exception as exc:
                print(f"实时窗口不可用（{exc}），仅录制视频")
        self.frames = []
        self.t = 0.0
        self.load_log = {"t": [], "load": [], "weld": []}
        self.breaks = []
        self.broken = set()

    # ---------- 基础设施 ----------
    def reset(self):
        model, data = self.model, self.data
        for side in ("left", "right"):
            for jname, q in zip(ARM_JOINTS, NEUTRAL_ARM):
                data.qpos[model.joint(f"{side}/{jname}").qposadr[0]] = q
                data.ctrl[model.actuator(f"{side}/{jname}").id] = q
            data.qpos[model.joint(f"{side}/left_finger").qposadr[0]] = 0.0084
            data.qpos[model.joint(f"{side}/right_finger").qposadr[0]] = 0.0084
            data.ctrl[model.actuator(f"{side}/gripper").id] = 0.0084
        data.ctrl[model.actuator("left/gripper").id] = 0.005
        data.ctrl[self.right_grip] = GRIP_OPEN
        ts.place_segments(model, data)

    def step_ctrl(self, watch=None):
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        self.t += 1.0 / CTRL_HZ
        if watch:
            watch()
        if len(self.frames) * (1.0 / FPS) <= self.t:
            frame = self.cams.composite(self.data)
            self.frames.append(frame)
            if self.live is not None:
                self.live.show(frame)

    def move_joint(self, arm, q_target, secs, watch=None):
        q0 = arm.q_now(self.data)
        steps = max(1, int(secs * CTRL_HZ))
        for k in range(steps):
            s = minjerk((k + 1) / steps)
            arm.command(self.data, q0 + s * (q_target - q0))
            self.step_ctrl(watch)

    def dwell(self, secs, watch=None):
        for _ in range(int(secs * CTRL_HZ)):
            self.step_ctrl(watch)

    def right_axes(self):
        """右手姿态：指向 = 接近方向的反向（-x），手指开合轴（站点 y）竖直。"""
        return [(0, -APPROACH_DIR), (1, UP)]

    # ---------- 载荷监控 ----------
    def make_watcher(self, target_welds):
        model, data = self.model, self.data

        def watch():
            for w in target_welds:
                if w in self.broken:
                    continue
                load = ts.weld_load(model, data, w)
                self.load_log["t"].append(self.t)
                self.load_log["load"].append(load)
                self.load_log["weld"].append(w)
                if load >= TEAR_LOAD:
                    model.eq_active0[model.equality(w).id] = 0
                    data.eq_active[model.equality(w).id] = 0
                    self.broken.add(w)
                    self.breaks.append((self.t, w, load))
                    print(f"[t={self.t:6.2f}s] 易撕线 {w} 断裂（载荷 {load:.1f}）")

        return watch

    def grasp_contacts(self, seg):
        """右手指与目标板格的接触点数。"""
        model, data = self.model, self.data
        plate = model.geom(f"{seg}_plate").id
        finger_bodies = {model.body("right/left_finger_link").id,
                         model.body("right/right_finger_link").id}
        n = 0
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = model.geom_bodyid[c.geom1]
            b2 = model.geom_bodyid[c.geom2]
            if (c.geom1 == plate and b2 in finger_bodies) or \
               (c.geom2 == plate and b1 in finger_bodies):
                n += 1
        return n

    # ---------- 主流程 ----------
    def run(self):
        self.reset()
        model, data = self.model, self.data
        self.dwell(0.5)

        # 阶段 1：左臂持板到工作位
        q_mid, e, a = self.left.solve(
            data, np.array([-0.12, 0.03, 0.22]), target_zaxis=UP, q_init=NEUTRAL_ARM)
        self.move_joint(self.left, q_mid, 1.6)
        q_hold, e, a = self.left.solve(data, STRIP_HOLD, target_zaxis=UP, q_init=q_mid)
        print(f"IK 左臂持板位: 误差 {e*1000:.1f}mm/{a:.1f}°")
        self.move_joint(self.left, q_hold, 1.8)
        self.dwell(0.6)

        results = []
        for idx, (col, row) in enumerate(TEAR_TARGETS):
            seg = ts.seg_name(col, row)
            target_welds = [w for w in ts.weld_names_of(col, row) if w not in self.broken]
            watch = self.make_watcher(target_welds)
            print(f"—— 目标 {idx+1}: {seg}，需断开易撕线 {target_welds} ——")

            # 阶段 2：张开夹爪，从 +x 自由端水平接近，指尖捏住板缘
            def site_target():
                mujoco.mj_forward(model, data)
                seg_pos = data.body(seg).xpos.copy()
                # 指尖落在 板缘 - EDGE_OVERLAP 处；站点在指尖前方 TIP_BEHIND_SITE
                return seg_pos + np.array(
                    [ts.SEG_HX - EDGE_OVERLAP - TIP_BEHIND_SITE, 0, 0])

            data.ctrl[self.right_grip] = GRIP_OPEN
            pre = site_target() + APPROACH_DIR * PREGRASP_BACKOFF
            q_pre, e, a = self.right.solve(
                data, pre, axes=self.right_axes(), q_init=self.right.q_now(data))
            print(f"IK 右臂预抓取: 误差 {e*1000:.1f}mm/{a:.1f}°")
            self.move_joint(self.right, q_pre, 1.8)
            self.dwell(0.3)

            q_grasp, e, a = self.right.solve(
                data, site_target(), axes=self.right_axes(), q_init=q_pre)
            print(f"IK 右臂抓取位: 误差 {e*1000:.1f}mm/{a:.1f}°")
            self.move_joint(self.right, q_grasp, 1.2)

            # 阶段 3：闭合夹爪（真实摩擦夹持）
            for k in range(int(0.8 * CTRL_HZ)):
                s = (k + 1) / (0.8 * CTRL_HZ)
                data.ctrl[self.right_grip] = GRIP_OPEN + s * (GRIP_CLOSE - GRIP_OPEN)
                self.step_ctrl()
            self.dwell(0.3)
            ncon = self.grasp_contacts(seg)
            print(f"[t={self.t:6.2f}s] 夹持接触点: {ncon}")

            # 阶段 4：缓慢扭转撕剪（沿接近轴转腕），断裂即停；必要时补一段下拉
            wrist_act = model.actuator("right/wrist_rotate").id
            twist0 = data.ctrl[wrist_act]
            for k in range(int(2.2 * CTRL_HZ)):
                if all(w in self.broken for w in target_welds):
                    break
                s = minjerk((k + 1) / (2.2 * CTRL_HZ))
                data.ctrl[wrist_act] = twist0 + s * TWIST_RAD
                self.step_ctrl(watch)
            if not all(w in self.broken for w in target_welds):
                q_now = self.right.q_now(data)
                q_pull, _, _ = self.right.solve(
                    data, self.right.site_pos(data) + np.array([0.03, 0, -0.04]),
                    axes=None, q_init=q_now)
                self.move_joint(self.right, q_pull, 1.2, watch)
            torn = all(w in self.broken for w in target_welds)
            print(f"[t={self.t:6.2f}s] {seg} 撕剪{'成功' if torn else '失败'}")

            # 阶段 5：保持夹持提起后撤 → 到托盘上方 → 指尖朝下"倒手"松爪
            mujoco.mj_forward(model, data)
            lift = self.right.site_pos(data) + np.array([0.04, 0, 0.05])
            q_lift, _, _ = self.right.solve(data, lift, axes=None,
                                            q_init=self.right.q_now(data))
            self.move_joint(self.right, q_lift, 1.0)
            drop = ts.TRAY_CENTER + np.array([0, 0, 0.10])
            q_drop, e, a = self.right.solve(
                data, drop, axes=[(0, -UP)], q_init=q_lift)
            print(f"IK 右臂投放位(指尖朝下): 误差 {e*1000:.1f}mm/{a:.1f}°")
            self.move_joint(self.right, q_drop, 1.8)
            self.dwell(0.3)
            data.ctrl[self.right_grip] = GRIP_OPEN
            self.dwell(0.4)
            shake_base = data.ctrl[wrist_act]
            for k in range(int(0.8 * CTRL_HZ)):  # 抖腕帮助脱落
                data.ctrl[wrist_act] = shake_base + 0.22 * np.sin(2 * np.pi * 3 * k / CTRL_HZ)
                self.step_ctrl()
            data.ctrl[wrist_act] = shake_base
            self.dwell(0.5)

            # 判定
            p = data.body(seg).xpos
            ok = (torn and abs(p[0] - ts.TRAY_CENTER[0]) < ts.TRAY_HX
                  and abs(p[1] - ts.TRAY_CENTER[1]) < ts.TRAY_HY and p[2] < 0.05)
            results.append((seg, ok))
            print(f"{seg} 最终位置 {np.round(p, 3)} -> {'√ 入托盘' if ok else '× 未入托盘'}")

            # 右手退回待命位
            q_home, _, _ = self.right.solve(
                data, np.array([0.20, -0.05, 0.25]), axes=self.right_axes(),
                q_init=self.right.q_now(data))
            self.move_joint(self.right, q_home, 1.4)

        self.dwell(1.0)
        n_ok = sum(ok for _, ok in results)
        print(f"成功入托盘: {n_ok}/{len(results)}")

        out = VIDEO_DIR / "pill_tear_v2_multicam.mp4"
        imageio.mimsave(out, self.frames, fps=FPS, macro_block_size=1)
        print(f"视频: {out}（{len(self.frames)} 帧, {len(self.frames)/FPS:.1f} s）")
        self.cams.close()
        if self.live is not None:
            self.live.close()
        self.plot_load()
        return n_ok

    def plot_load(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=130)
        tarr = np.array(self.load_log["t"])
        larr = np.array(self.load_log["load"])
        warr = np.array(self.load_log["weld"])
        for i, w in enumerate(sorted(set(self.load_log["weld"]))):
            m = warr == w
            ax.plot(tarr[m], larr[m], lw=1.6, label=f"易撕线 {w}")
        for bt, w, load in self.breaks:
            ax.axvline(bt, ls="--", lw=1, alpha=0.5, color="gray")
            ax.annotate(f"{w} 断裂", (bt, load), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)
        ax.axhline(TEAR_LOAD, color="gray", ls=":", lw=1.2, label=f"断裂阈值 {TEAR_LOAD}")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("易撕线约束载荷（efc 合力）")
        ax.set_title("分药 v2（撕剪）：易撕线载荷曲线与断裂事件")
        ax.legend(loc="upper left", fontsize=8, ncols=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(IMAGE_DIR / "pill_tear_v2_load.png")
        print(f"载荷曲线: {IMAGE_DIR / 'pill_tear_v2_load.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALOHA 双臂撕剪分药演示（三机位）")
    parser.add_argument("--live", action="store_true", help="弹窗实时显示三机位画面")
    args = parser.parse_args()
    TearDemo(live=args.live).run()
