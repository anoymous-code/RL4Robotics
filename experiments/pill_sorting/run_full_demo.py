"""分药任务 v5：轮式移动操作机器人（双臂并排朝前）+ 固定桌子，完整闭环流程。

  驶离充电桩 → 停到桌前 → 桌上盒 A 取板 → 双臂撕剪单格 → 入盒 B → 剩板放回盒 A

流程：
  0. 机器人从充电桩出发：原地转向 → 直线行驶 → 回正停到桌前（盒 A/B 都在固定桌上）；
  1. 左臂移到盒 A 上方，张开夹爪竖直下降，夹住药板手柄提出槽位；
  2. 空中转体 90°，把板转到水平工作位（格朝上，自由端指向右臂侧）；
  3. 右臂从自由端夹住目标格外缘，扭腕撕断易撕线（可断裂焊接约束）；
  4. 撕下的单格（药片保持密封）运到盒 B 上方，指尖朝下松爪投放；
  5. 重复撕剪第二格；
  6. 左臂把剩板转回竖直，插回盒 A 中间槽位，松爪撤离。

运行:
    cd experiments/pill_sorting && ../../.venv/Scripts/python.exe run_full_demo.py
    可选 --live 实时弹窗；--no-latch 左手纯摩擦对照
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
DOWN = -UP
X = np.array([1.0, 0.0, 0.0])

# 指腹中心在 gripper 站点系中的位置（实测标定，两指平均）
PADS_LOCAL = np.array([-0.0138, 0.0022, 0.0])

STRIP_HOLD = np.array([0.00, 0.10, 0.17])   # 撕剪工作位（strip 中心目标，桌前两臂中间上方）
GRIP_OPEN = 0.025
GRIP_CLOSE_R = 0.0002     # 右爪捏 2.4mm 薄板（大过盈：断裂冲击下不脱手）
GRIP_CLOSE_L = 0.0028     # 左爪捏 9mm 手柄
TIP_BEHIND_SITE = 0.0138
EDGE_OVERLAP = 0.0085     # 右手指尖与目标格板缘的重叠量
TAB_BITE = 0.010          # 左手指腹咬入手柄顶部的深度
PREGRASP_BACKOFF = 0.05
TEAR_LOAD = 4.5           # 悬持板柔性大，阈值调低使断裂发生在小变形处、减小弹射冲击
TWIST_RAD = 0.7
LEFT_KP_SCALE = 15.0
RIGHT_KP_SCALE = 8.0

TEAR_TARGETS = [(3, 0), (3, 1)]  # 从自由端顺序撕：第 4 列前排、后排

# 站点姿态（axes = [(站点局部轴, 世界方向), ...]）
AXES_GRASP_DOWN = [(0, DOWN), (1, X)]    # 竖直向下抓手柄：指向下，开合轴沿板厚(x)
AXES_HOLD = [(0, X), (1, UP)]            # 水平持板：指向 +x，开合轴竖直
AXES_TEAR = [(0, -X), (1, UP)]           # 右手撕剪：指向 -x，开合轴竖直


def axes_rot(axes):
    """由两条轴约束构造站点旋转矩阵（列 = 站点局部轴的世界方向）。"""
    c0 = np.asarray(axes[0][1], dtype=float)
    c1 = np.asarray(axes[1][1], dtype=float)
    c2 = np.cross(c0, c1)
    return np.column_stack([c0, c1, c2])


def _quat_mat(quat):
    m = np.empty(9)
    mujoco.mju_quat2Mat(m, quat)
    return m.reshape(3, 3)


class FullDemo:
    def __init__(self, live=False, latch=False):
        self.latch = latch
        self.model = ts.load_model()
        stiffen_arm(self.model, "left", LEFT_KP_SCALE)
        stiffen_arm(self.model, "right", RIGHT_KP_SCALE)
        # 夹爪增强：薄物夹持全靠指尖过盈；ctrlrange 下限 0.002 是真机软件限位，放开到 0
        for side, scale in (("left", 8.0), ("right", 6.0)):
            ga = self.model.actuator(f"{side}/gripper")
            self.model.actuator_gainprm[ga.id, 0] *= scale
            self.model.actuator_biasprm[ga.id, 1] *= scale
            self.model.actuator_ctrlrange[ga.id, 0] = 0.0
        self.data = mujoco.MjData(self.model)
        self.n_sub = int(1.0 / CTRL_HZ / self.model.opt.timestep)
        self.left = ArmKinematics(self.model, "left", "left/gripper")
        self.right = ArmKinematics(self.model, "right", "right/gripper")
        self.left_grip = self.model.actuator("left/gripper").id
        self.right_grip = self.model.actuator("right/gripper").id
        self.cams = MultiCam(self.model)
        self.live = None
        if live:
            try:
                self.live = LiveWindow()
            except Exception as exc:
                print(f"实时窗口不可用（{exc}），仅录制视频")
        # 流式写盘：全流程 1500+ 帧若囤内存（>6GB）会让编码器分配失败
        self.video_path = VIDEO_DIR / "pill_full_v5_mobile_multicam.mp4"
        self.writer = imageio.get_writer(self.video_path, fps=FPS, macro_block_size=1)
        self.n_frames = 0
        self.t = 0.0
        self.load_log = {"t": [], "load": [], "weld": []}
        self.breaks = []
        self.broken = set()
        self.strip_in_site = None   # 抓稳后 strip 在左站点系中的位姿（滑移监控/放回规划）

    # ---------- 基础设施 ----------
    def reset(self):
        model, data = self.model, self.data
        mujoco.mj_resetData(model, data)
        ts.set_base(model, data, ts.BASE_START)
        for side in ("left", "right"):
            for jname, q in zip(ARM_JOINTS, NEUTRAL_ARM):
                data.qpos[model.joint(f"{side}/{jname}").qposadr[0]] = q
                data.ctrl[model.actuator(f"{side}/{jname}").id] = q
            data.qpos[model.joint(f"{side}/left_finger").qposadr[0]] = 0.0084
            data.qpos[model.joint(f"{side}/right_finger").qposadr[0]] = 0.0084
        data.ctrl[self.left_grip] = GRIP_OPEN
        data.ctrl[self.right_grip] = GRIP_OPEN
        mujoco.mj_forward(model, data)

    def step_ctrl(self, watch=None):
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        self.t += 1.0 / CTRL_HZ
        if watch:
            watch()
        if self.n_frames * (1.0 / FPS) <= self.t:
            frame = self.cams.composite(self.data)
            self.writer.append_data(frame)
            self.n_frames += 1
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

    def close_gripper(self, act_id, target, secs=0.8):
        data = self.data
        v0 = data.ctrl[act_id]
        steps = int(secs * CTRL_HZ)
        for k in range(steps):
            s = (k + 1) / steps
            data.ctrl[act_id] = v0 + s * (target - v0)
            self.step_ctrl()

    def site_pose(self, arm):
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.site_xpos[arm.site_id].copy()
        rot = self.data.site_xmat[arm.site_id].reshape(3, 3).copy()
        return pos, rot

    def strip_pose(self):
        mujoco.mj_forward(self.model, self.data)
        s = self.data.body("strip")
        return s.xpos.copy(), s.xmat.reshape(3, 3).copy()

    def record_grasp_frame(self):
        """记录抓稳时 strip 在左站点系中的相对位姿。"""
        sp, sR = self.site_pose(self.left)
        bp, bR = self.strip_pose()
        self.strip_in_site = (sR.T @ (bp - sp), sR.T @ bR)

    def strip_slip(self):
        """当前 strip 相对左站点位姿与抓稳时的偏差（mm, deg）。"""
        sp, sR = self.site_pose(self.left)
        bp, bR = self.strip_pose()
        rel_p = sR.T @ (bp - sp)
        rel_R = sR.T @ bR
        dp = np.linalg.norm(rel_p - self.strip_in_site[0]) * 1000
        dR = rel_R @ self.strip_in_site[1].T
        ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        return dp, ang

    def left_pose_for_strip(self, strip_target_pos, R_strip_target):
        """给定 strip 目标位姿，反推左站点位置与轴约束（用抓稳时的相对位姿补偿板倾斜）。"""
        rel_p, rel_R = self.strip_in_site
        R_site = R_strip_target @ rel_R.T
        site_pos = strip_target_pos - R_site @ rel_p
        axes = [(0, R_site[:, 0]), (1, R_site[:, 1])]
        return site_pos, axes

    def grasp_contacts(self, seg):
        model, data = self.model, self.data
        mujoco.mj_forward(model, data)
        plate = model.geom(f"{seg}_plate").id
        fb = {model.body("right/left_finger_link").id,
              model.body("right/right_finger_link").id}
        return sum(1 for i in range(data.ncon)
                   if (data.contact[i].geom1 == plate and model.geom_bodyid[data.contact[i].geom2] in fb)
                   or (data.contact[i].geom2 == plate and model.geom_bodyid[data.contact[i].geom1] in fb))

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

    # ---------- 阶段 0：驶离充电桩到桌前 ----------
    def drive_to_work(self):
        """差速底盘式编排：原地转向 → 直线行驶 → 回正停到桌前（车头 = 车体局部 +y）。"""
        model, data = self.model, self.data
        acts = {n: model.actuator(n).id for n in ("base_x", "base_y", "base_yaw")}
        start, work = ts.BASE_START, ts.BASE_WORK
        delta = work[:2] - start[:2]
        heading = np.arctan2(-delta[0], delta[1])   # 使车头(+y 局部)对准位移方向的 yaw

        def base_ctrl(x, y, yaw):
            data.ctrl[acts["base_x"]] = x
            data.ctrl[acts["base_y"]] = y
            data.ctrl[acts["base_yaw"]] = yaw

        # 原地转向车头对准目标
        for k in range(int(1.6 * CTRL_HZ)):
            s = minjerk((k + 1) / (1.6 * CTRL_HZ))
            base_ctrl(start[0], start[1], start[2] + s * (heading - start[2]))
            self.step_ctrl()
        # 沿车头方向直线行驶（min-jerk 位置轨迹，平滑加减速）
        for k in range(int(4.5 * CTRL_HZ)):
            s = minjerk((k + 1) / (4.5 * CTRL_HZ))
            base_ctrl(start[0] + s * delta[0], start[1] + s * delta[1], heading)
            self.step_ctrl()
        # 原地回正：车头朝向桌子
        for k in range(int(1.6 * CTRL_HZ)):
            s = minjerk((k + 1) / (1.6 * CTRL_HZ))
            base_ctrl(work[0], work[1], heading + s * (work[2] - heading))
            self.step_ctrl()
        self.dwell(0.8)

        bx = data.qpos[model.joint("base_x").qposadr[0]]
        by = data.qpos[model.joint("base_y").qposadr[0]]
        yaw = data.qpos[model.joint("base_yaw").qposadr[0]]
        err = np.hypot(bx - work[0], by - work[1]) * 1000
        print(f"[t={self.t:6.2f}s] 停到桌前 ({bx:.3f}, {by:.3f}, yaw {np.degrees(yaw):.1f}°)，"
              f"停车误差 {err:.1f} mm")

    # ---------- 阶段 1-2：盒 A 取板 → 工作位 ----------
    def pick_board(self):
        model, data = self.model, self.data
        R_down = axes_rot(AXES_GRASP_DOWN)

        def tab_grasp_site():
            """指腹咬住手柄顶部下方 TAB_BITE 处。"""
            bp, bR = self.strip_pose()
            tab_top = bp + bR @ np.array([ts.TAB_LOCAL[0], 0, ts.TAB_LOCAL[2]])
            grasp = tab_top + np.array([0, 0, -TAB_BITE])
            return grasp - R_down @ PADS_LOCAL

        pre = tab_grasp_site() + np.array([0, 0, 0.05])
        q_pre, e, a = self.left.solve(data, pre, axes=AXES_GRASP_DOWN, q_init=NEUTRAL_ARM)
        print(f"IK 左臂预抓取: 误差 {e*1000:.1f}mm/{a:.1f}°")
        self.move_joint(self.left, q_pre, 2.0)
        self.dwell(0.3)

        q_grasp, e, a = self.left.solve(data, tab_grasp_site(), axes=AXES_GRASP_DOWN, q_init=q_pre)
        print(f"IK 左臂抓取位: 误差 {e*1000:.1f}mm/{a:.1f}°")
        self.move_joint(self.left, q_grasp, 1.2)
        self.close_gripper(self.left_grip, GRIP_CLOSE_L)
        self.dwell(0.3)
        self.record_grasp_frame()
        if self.latch:
            ts.engage_latch(model, data)
            print("已启用左爪锁定（latch）")

        # 竖直提出槽位
        sp, _ = self.site_pose(self.left)
        q_lift, e, a = self.left.solve(data, sp + np.array([0, 0, 0.17]),
                                       axes=AXES_GRASP_DOWN, q_init=q_grasp)
        self.move_joint(self.left, q_lift, 1.8)
        dp, ang = self.strip_slip()
        print(f"[t={self.t:6.2f}s] 提板完成，夹持滑移 {dp:.1f}mm/{ang:.1f}°")

        # 空中转体 90° 到水平工作位（分两步走，避免大幅摆动）。
        # 目标姿态按"板水平"反解手的姿态：抓取时板在槽中有小倾角，
        # 若手摆标准姿态则板面残留倾斜，右爪水平指腹将只剩单点接触
        mid = np.array([-0.18, 0.02, 0.32])
        q_mid, e, a = self.left.solve(data, mid, axes=[(0, np.array([1.0, 0, -1.0])), (1, np.array([1.0, 0, 1.0]))],
                                      q_init=q_lift)
        self.move_joint(self.left, q_mid, 2.0)
        site_hold, axes_hold = self.left_pose_for_strip(STRIP_HOLD, np.eye(3))
        q_hold, e, a = self.left.solve(data, site_hold, axes=axes_hold, q_init=q_mid)
        print(f"IK 左臂持板位: 误差 {e*1000:.1f}mm/{a:.1f}°")
        self.move_joint(self.left, q_hold, 2.2)
        self.dwell(0.6)
        _, bR = self.strip_pose()
        tilt = np.degrees(np.arccos(np.clip(bR[2, 2], -1, 1)))
        print(f"[t={self.t:6.2f}s] 转体到工作位，板面倾角 {tilt:.1f}°")

    # ---------- 阶段 3-5：撕剪两格入盒 B ----------
    def tear_segments(self):
        model, data = self.model, self.data
        results = []
        for idx, (col, row) in enumerate(TEAR_TARGETS):
            seg = ts.seg_name(col, row)
            target_welds = [w for w in ts.weld_names_of(col, row) if w not in self.broken]
            watch = self.make_watcher(target_welds)
            print(f"—— 目标 {idx+1}: {seg}，需断开易撕线 {target_welds} ——")

            R_tear = axes_rot(AXES_TEAR)

            def site_target():
                mujoco.mj_forward(model, data)
                seg_pos = data.body(seg).xpos.copy()
                grasp = seg_pos + np.array([ts.SEG_HX - EDGE_OVERLAP, 0, 0])
                return grasp - R_tear @ PADS_LOCAL

            data.ctrl[self.right_grip] = GRIP_OPEN
            pre = site_target() + X * PREGRASP_BACKOFF
            q_pre, e, a = self.right.solve(data, pre, axes=AXES_TEAR,
                                           q_init=self.right.q_now(data))
            print(f"IK 右臂预抓取: 误差 {e*1000:.1f}mm/{a:.1f}°")
            self.move_joint(self.right, q_pre, 1.8)
            self.dwell(0.3)

            q_grasp, e, a = self.right.solve(data, site_target(), axes=AXES_TEAR, q_init=q_pre)
            print(f"IK 右臂抓取位: 误差 {e*1000:.1f}mm/{a:.1f}°")
            self.move_joint(self.right, q_grasp, 1.2)
            self.close_gripper(self.right_grip, GRIP_CLOSE_R)
            self.dwell(0.3)
            print(f"[t={self.t:6.2f}s] 闭爪后指间接触点: {self.grasp_contacts(seg)}")

            # 缓慢扭转撕剪，断裂即停；必要时补一段下拉
            wrist_act = model.actuator("right/wrist_rotate").id
            twist0 = data.ctrl[wrist_act]
            for k in range(int(3.0 * CTRL_HZ)):
                if all(w in self.broken for w in target_welds):
                    break
                s = minjerk((k + 1) / (3.0 * CTRL_HZ))
                data.ctrl[wrist_act] = twist0 + s * TWIST_RAD
                self.step_ctrl(watch)
            if not all(w in self.broken for w in target_welds):
                q_now = self.right.q_now(data)
                q_pull, _, _ = self.right.solve(
                    data, self.right.site_pos(data) + np.array([0.03, 0, -0.04]),
                    axes=None, q_init=q_now)
                self.move_joint(self.right, q_pull, 1.2, watch)
            torn = all(w in self.broken for w in target_welds)
            self.dwell(0.5)   # 断裂后的弹性释放让格在指间振荡，先稳住再运送
            print(f"[t={self.t:6.2f}s] {seg} 撕剪{'成功' if torn else '失败'}"
                  f"（断后指间接触点 {self.grasp_contacts(seg)}）")

            # 提起（保持姿态只平移，防甩落）→ 盒 B 上方指尖朝下松爪投放
            mujoco.mj_forward(model, data)
            lift = self.right.site_pos(data) + np.array([0.04, 0, 0.05])
            q_lift, _, _ = self.right.solve(data, lift, axes=AXES_TEAR,
                                            q_init=self.right.q_now(data))
            self.move_joint(self.right, q_lift, 1.0)
            drop = ts.BOX_B_CENTER + np.array([0, 0, 0.075])
            q_drop, e, a = self.right.solve(data, drop, axes=[(0, DOWN)], q_init=q_lift)
            print(f"IK 右臂投放位(指尖朝下): 误差 {e*1000:.1f}mm/{a:.1f}°")
            self.move_joint(self.right, q_drop, 2.4)
            self.dwell(0.3)
            print(f"[t={self.t:6.2f}s] 投放前指间接触点: {self.grasp_contacts(seg)}")
            data.ctrl[self.right_grip] = GRIP_OPEN
            self.dwell(0.4)
            shake_base = data.ctrl[wrist_act]
            for k in range(int(1.2 * CTRL_HZ)):
                data.ctrl[wrist_act] = shake_base + 0.28 * np.sin(2 * np.pi * 3.5 * k / CTRL_HZ)
                self.step_ctrl()
            data.ctrl[wrist_act] = shake_base
            self.dwell(0.5)

            p = data.body(seg).xpos
            ok = (torn and abs(p[0] - ts.BOX_B_CENTER[0]) < ts.BOX_B_HX
                  and abs(p[1] - ts.BOX_B_CENTER[1]) < ts.BOX_B_HY and p[2] < 0.05)
            results.append((seg, ok))
            print(f"{seg} 最终位置 {np.round(p, 3)} -> {'√ 入盒 B' if ok else '× 未入盒 B'}")

            q_home, _, _ = self.right.solve(
                data, np.array([0.22, 0.02, 0.26]), axes=AXES_TEAR,
                q_init=self.right.q_now(data))
            self.move_joint(self.right, q_home, 1.4)
        return results

    # ---------- 阶段 6：剩板放回盒 A ----------
    def return_board(self):
        model, data = self.model, self.data

        # 高位转回竖直（先撤到中途点再转，避免扫到盒 B）
        mid = np.array([-0.16, 0.02, 0.30])
        q_mid, e, a = self.left.solve(data, mid, axes=[(0, np.array([-0.5, 0, -1.0])), (1, X)],
                                      q_init=self.left.q_now(data))
        self.move_joint(self.left, q_mid, 2.2)

        R_vert = _quat_mat(ts.BOARD_UP_QUAT)
        strip_high = ts.BOARD_HOME + np.array([0, 0, 0.09])
        site_high, axes_vert = self.left_pose_for_strip(strip_high, R_vert)
        q_high, e, a = self.left.solve(data, site_high, axes=axes_vert, q_init=q_mid)
        print(f"IK 左臂回放高位: 误差 {e*1000:.1f}mm/{a:.1f}°")
        self.move_joint(self.left, q_high, 2.2)
        self.dwell(0.4)

        # 撕掉若干格后板变短：按剩余板长算下插深度（板底入槽即可，松爪后自行滑到底）
        remain_x = max((ts.seg_offset(c, r)[0] for c, r in ts.all_segments()
                        if not all(w in self.broken for w in ts.weld_names_of(c, r))),
                       default=0.0) + ts.SEG_HX
        strip_z_ins = 0.04 + remain_x   # 板底降到 z=0.04（低于隔板顶，已被槽壁约束）

        # 用实时板位误差修正后缓慢下插
        bp, _ = self.strip_pose()
        sp, _ = self.site_pose(self.left)
        corr = ts.BOARD_HOME[:2] - bp[:2]
        target = sp + np.array([corr[0], corr[1], strip_z_ins - bp[2]])
        q_ins, e, a = self.left.solve(data, target, axes=axes_vert,
                                      q_init=self.left.q_now(data))
        print(f"IK 左臂插槽位: 误差 {e*1000:.1f}mm/{a:.1f}°（水平修正 {np.round(corr*1000,1)}mm，"
              f"剩板长 {remain_x*1000:.0f}mm）")
        self.move_joint(self.left, q_ins, 2.4)
        self.dwell(0.3)

        if self.latch:
            ts.release_latch(model, data)
        data.ctrl[self.left_grip] = GRIP_OPEN
        self.dwell(0.6)
        sp, _ = self.site_pose(self.left)
        q_up, _, _ = self.left.solve(data, sp + np.array([0, 0, 0.14]),
                                     axes=axes_vert, q_init=self.left.q_now(data))
        self.move_joint(self.left, q_up, 1.6)
        self.dwell(0.8)

        bp, bR = self.strip_pose()
        ok = (np.linalg.norm(bp[:2] - ts.BOARD_HOME[:2]) < 0.025
              and bp[2] < ts.BOARD_HOME[2] + 0.015 and bR[2, 0] < -0.9)
        print(f"剩板最终位置 {np.round(bp, 3)}（目标 {np.round(ts.BOARD_HOME, 3)}）"
              f" -> {'√ 已插回盒 A' if ok else '× 未插回盒 A'}")
        return ok

    # ---------- 主流程 ----------
    def run(self):
        self.reset()
        self.cams.MAIN = "room"      # 行驶阶段用房间全景机位
        self.dwell(0.6)
        self.drive_to_work()
        self.cams.MAIN = "follow_pack"
        self.pick_board()
        results = self.tear_segments()
        returned = self.return_board()

        n_ok = sum(ok for _, ok in results)
        print(f"总结: 撕剪入盒 B {n_ok}/{len(results)}，剩板放回盒 A {'成功' if returned else '失败'}")

        self.writer.close()
        print(f"视频: {self.video_path}（{self.n_frames} 帧, {self.n_frames/FPS:.1f} s）")
        self.cams.close()
        if self.live is not None:
            self.live.close()
        self.plot_load()
        return n_ok, returned

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
        for w in sorted(set(self.load_log["weld"])):
            m = warr == w
            ax.plot(tarr[m], larr[m], lw=1.6, label=f"易撕线 {w}")
        for bt, w, load in self.breaks:
            ax.axvline(bt, ls="--", lw=1, alpha=0.5, color="gray")
            ax.annotate(f"{w} 断裂", (bt, load), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)
        ax.axhline(TEAR_LOAD, color="gray", ls=":", lw=1.2, label=f"断裂阈值 {TEAR_LOAD}")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("易撕线约束载荷（efc 合力）")
        ax.set_title("分药 v5（移动机器人 + 固定桌全流程）：易撕线载荷曲线与断裂事件")
        ax.legend(loc="upper left", fontsize=8, ncols=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(IMAGE_DIR / "pill_full_v5_load.png")
        print(f"载荷曲线: {IMAGE_DIR / 'pill_full_v5_load.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALOHA 双臂完整分药流程演示（三机位）")
    parser.add_argument("--live", action="store_true", help="弹窗实时显示三机位画面")
    parser.add_argument("--no-latch", action="store_true",
                        help="关闭左爪锁定，用纯摩擦夹持（撕剪反力矩下会滑移，供对照实验）")
    args = parser.parse_args()
    FullDemo(live=args.live, latch=not args.no_latch).run()
