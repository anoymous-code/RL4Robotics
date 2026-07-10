"""诊断 2：用离线 IK 检查双臂目标位姿是否可达，并渲染解出的构型。"""

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from ik_utils import ArmKinematics, ARM_JOINTS

HERE = Path(__file__).resolve().parent
OUT = HERE / "debug"
NEUTRAL_ARM = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0])
UP = np.array([0.0, 0.0, 1.0])

PACK_TARGET = np.array([0.06, 0.05, 0.15])


def main():
    model = mujoco.MjModel.from_xml_path(str(HERE / "pill_scene.xml"))
    data = mujoco.MjData(model)
    for side in ("left", "right"):
        for jname, q in zip(ARM_JOINTS, NEUTRAL_ARM):
            data.qpos[model.joint(f"{side}/{jname}").qposadr[0]] = q
    mujoco.mj_forward(model, data)

    left = ArmKinematics(model, "left", "pack_center")
    right = ArmKinematics(model, "right", "stylus_tip_site")

    print("=== 左臂: 药板中心 -> 杯上方, 药板法向朝上 ===")
    qL, eL, aL = left.solve(data, PACK_TARGET, target_zaxis=UP, q_init=NEUTRAL_ARM)
    print(f"位置误差 {eL*1000:.1f} mm, 法向偏角 {aL:.1f}°, q={np.round(qL,2)}")
    data.qpos[left.qpos_ids] = qL
    mujoco.mj_forward(model, data)

    # 右臂目标：压杆尖端到泡罩 1 上方 3cm，杆轴竖直
    pocket1 = data.site("pocket_1").xpos.copy()
    print(f"泡罩1 现位置: {np.round(pocket1, 3)}")
    for name, tgt in (
        ("悬停", pocket1 + np.array([0, 0, 0.03])),
        ("压下", pocket1 + np.array([0, 0, -0.005])),
    ):
        qR, eR, aR = right.solve(data, tgt, target_zaxis=UP, q_init=NEUTRAL_ARM)
        print(f"右臂[{name}] 位置误差 {eR*1000:.1f} mm, 轴偏角 {aR:.1f}°, q={np.round(qR,2)}")

    # 渲染悬停构型
    qR, _, _ = right.solve(data, pocket1 + np.array([0, 0, 0.03]), target_zaxis=UP, q_init=NEUTRAL_ARM)
    data.qpos[right.qpos_ids] = qR
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=720, width=1280)
    for cam in ("side", "follow_pack", "teleoperator_pov"):
        renderer.update_scene(data, camera=cam)
        imageio.imwrite(OUT / f"ik2_{cam}.png", renderer.render())
    renderer.close()
    print("已渲染 ik2_*.png")


if __name__ == "__main__":
    main()
