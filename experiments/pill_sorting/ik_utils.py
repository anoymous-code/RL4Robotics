"""离线 IK 求解工具：在 mjData 副本上纯运动学迭代，不受伺服动力学干扰。"""

import mujoco
import numpy as np

ARM_JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")


class ArmKinematics:
    def __init__(self, model, side, site_name):
        self.model = model
        self.side = side
        self.site_id = model.site(site_name).id
        self.joint_ids = [model.joint(f"{side}/{j}").id for j in ARM_JOINTS]
        self.dof_ids = np.array([model.jnt_dofadr[j] for j in self.joint_ids])
        self.qpos_ids = np.array([model.jnt_qposadr[j] for j in self.joint_ids])
        self.act_ids = [model.actuator(f"{side}/{a}").id for a in ARM_JOINTS]
        self.jnt_range = model.jnt_range[self.joint_ids].copy()

    def solve(self, data_ref, target_pos, target_zaxis=None, q_init=None,
              iters=300, tol=1e-4, damping=0.05, step_scale=0.7, restarts=8, seed=0,
              local_axis=2, axes=None):
        """迭代求解 IK（带随机重启）。返回 (q_solution, pos_err_m, axis_err_deg)。

        姿态目标两种写法（二选一）：
          - target_zaxis + local_axis: 单轴对齐（0/1/2 = 站点局部 x/y/z）
          - axes=[(local_axis, world_dir), ...]: 多轴同时对齐
        在 data_ref 的副本上操作——其余关节（另一只手臂等）保持 data_ref 中的值。
        """
        if axes is None:
            axes = [] if target_zaxis is None else [(local_axis, target_zaxis)]
        rng = np.random.default_rng(seed)
        best = None
        inits = [q_init if q_init is not None else data_ref.qpos[self.qpos_ids].copy()]
        for _ in range(restarts - 1):
            lo, hi = self.jnt_range[:, 0], self.jnt_range[:, 1]
            inits.append(lo + (hi - lo) * rng.random(len(self.qpos_ids)))
        for q0 in inits:
            sol = self._solve_once(data_ref, target_pos, axes, q0,
                                   iters, tol, damping, step_scale)
            score = sol[1] + np.radians(sol[2]) * 0.1
            if best is None or score < best[0]:
                best = (score, sol)
            if sol[1] < tol * 10 and sol[2] < 3.0:
                break
        return best[1]

    def _solve_once(self, data_ref, target_pos, axes, q_init,
                    iters, tol, damping, step_scale):
        data = mujoco.MjData(self.model)
        data.qpos[:] = data_ref.qpos
        if q_init is not None:
            data.qpos[self.qpos_ids] = q_init

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(self.model, data)
            mujoco.mj_comPos(self.model, data)
            pos = data.site_xpos[self.site_id]
            rot = data.site_xmat[self.site_id].reshape(3, 3)

            rows = [target_pos - pos]
            mujoco.mj_jacSite(self.model, data, jacp, jacr, self.site_id)
            jac_rows = [jacp[:, self.dof_ids]]
            for ax_idx, world_dir in axes:
                tz = np.asarray(world_dir, dtype=float)
                tz = tz / np.linalg.norm(tz)
                cur_a = rot[:, ax_idx]
                # 误差与雅可比保持一致：误差 = 轴向量差，雅可比 = d(cur_a)/dq
                rows.append((tz - cur_a) * 0.35)
                jz = np.empty((3, len(self.dof_ids)))
                for k, dof in enumerate(self.dof_ids):
                    jz[:, k] = np.cross(jacr[:, dof], cur_a)
                jac_rows.append(jz * 0.35)

            err = np.concatenate(rows)
            if np.linalg.norm(rows[0]) < tol:
                break
            J = np.vstack(jac_rows)
            JJt = J @ J.T + damping**2 * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJt, err)
            q = data.qpos[self.qpos_ids] + step_scale * dq
            data.qpos[self.qpos_ids] = np.clip(q, self.jnt_range[:, 0], self.jnt_range[:, 1])

        mujoco.mj_kinematics(self.model, data)
        pos = data.site_xpos[self.site_id].copy()
        rot = data.site_xmat[self.site_id].reshape(3, 3)
        pos_err = float(np.linalg.norm(target_pos - pos))
        axis_err = 0.0
        for ax_idx, world_dir in axes:
            tz = np.asarray(world_dir, dtype=float)
            tz = tz / np.linalg.norm(tz)
            err = float(np.degrees(np.arccos(np.clip(rot[:, ax_idx] @ tz, -1, 1))))
            axis_err = max(axis_err, err)
        return data.qpos[self.qpos_ids].copy(), pos_err, axis_err

    def q_now(self, data):
        return data.qpos[self.qpos_ids].copy()

    def command(self, data, q_des):
        for act_id, q in zip(self.act_ids, q_des):
            data.ctrl[act_id] = q

    def site_pos(self, data):
        return data.site_xpos[self.site_id].copy()

    def site_zaxis(self, data):
        return data.site_xmat[self.site_id].reshape(3, 3)[:, 2].copy()
