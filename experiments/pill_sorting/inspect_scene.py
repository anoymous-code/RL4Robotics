"""场景检查：加载分药场景，摆好药片，渲染静态图确认几何布局正确。

运行:
    .venv\\Scripts\\python.exe experiments\\pill_sorting\\inspect_scene.py
"""

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "debug"
OUT.mkdir(exist_ok=True)

NEUTRAL_ARM = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
GRIPPER_Q = 0.0084


def load():
    model = mujoco.MjModel.from_xml_path(str(HERE / "pill_scene.xml"))
    data = mujoco.MjData(model)
    return model, data


def set_neutral(model, data):
    for side in ("left", "right"):
        for jname, q in zip(
            ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"),
            NEUTRAL_ARM,
        ):
            j = model.joint(f"{side}/{jname}")
            data.qpos[j.qposadr[0]] = q
        for fname in ("left_finger", "right_finger"):
            j = model.joint(f"{side}/{fname}")
            data.qpos[j.qposadr[0]] = GRIPPER_Q
    for side in ("left", "right"):
        for aname, q in zip(
            ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"),
            NEUTRAL_ARM,
        ):
            data.ctrl[model.actuator(f"{side}/{aname}").id] = q
        data.ctrl[model.actuator(f"{side}/gripper").id] = GRIPPER_Q


def place_pills(model, data):
    """把药片放进药板的各个泡罩格（贴着铝膜顶面）。"""
    mujoco.mj_forward(model, data)
    pack = data.body("pill_pack")
    R = pack.xmat.reshape(3, 3)
    pocket_xs = model.numeric("pocket_xs").data
    for i, px in enumerate(pocket_xs):
        local = np.array([px, 0.0, -0.001])  # 铝膜顶面 -0.003 + 药片半高 0.002
        world = pack.xpos + R @ local
        j = model.joint(f"pill_{i}_joint")
        adr = j.qposadr[0]
        data.qpos[adr : adr + 3] = world
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, pack.xmat)
        data.qpos[adr + 3 : adr + 7] = quat
        vadr = j.dofadr[0]
        data.qvel[vadr : vadr + 6] = 0.0
    mujoco.mj_forward(model, data)


def report(model, data):
    print("=== 关键位姿（世界坐标） ===")
    for name in ("pill_pack", "stylus"):
        b = data.body(name)
        print(f"body {name:12s} pos={np.round(b.xpos, 4)}")
    for name in ("left/gripper", "right/gripper", "stylus_tip_site"):
        s = data.site(name)
        print(f"site {name:16s} pos={np.round(s.xpos, 4)}")
    pack = data.body("pill_pack")
    R = pack.xmat.reshape(3, 3)
    print(f"药板法向（局部 z 的世界方向）= {np.round(R[:, 2], 3)}")
    for i in range(3):
        print(f"pill_{i} pos={np.round(data.body(f'pill_{i}').xpos, 4)}")


def render_stills(model, data):
    renderer = mujoco.Renderer(model, height=720, width=1280)
    for cam in ("closeup", "side", "teleoperator_pov", "overhead_cam"):
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        path = OUT / f"still_{cam.replace('/', '_')}.png"
        imageio.imwrite(path, img)
        print(f"已渲染 {path}")
    renderer.close()


def main():
    model, data = load()
    print(f"模型加载成功: nq={model.nq}, nu={model.nu}, nbody={model.nbody}")
    set_neutral(model, data)
    place_pills(model, data)
    # 短暂物理沉降，验证药片会不会掉出来
    for _ in range(500):
        mujoco.mj_step(model, data)
    report(model, data)
    render_stills(model, data)


if __name__ == "__main__":
    main()
