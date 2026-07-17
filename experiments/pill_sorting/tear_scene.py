"""撕剪版分药场景构建：8 格铝塑板（2 行 x 4 列）+ 易撕线（可断裂焊接约束）+ 托盘。

直接运行本文件可做几何/坐标探测并渲染静帧：
    ../../.venv/Scripts/python.exe tear_scene.py
"""

from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent

# ---------- 板块布局（strip 局部坐标，单位 m） ----------
N_COLS, N_ROWS = 4, 2
SEG_HX, SEG_HY, SEG_HZ = 0.0115, 0.0115, 0.003    # 单格半尺寸（厚 6mm 便于夹持）
PITCH_X, PITCH_Y = 0.024, 0.025
COL_X0 = 0.0225            # 第 0 列中心相对 strip 中心的 x 偏移
ROW_Y = (-0.0125, 0.0125)  # 行中心 y（f=前排 -y，b=后排 +y）
ROW_TAG = ("f", "b")

PILL_COLORS = ["0.95 0.45 0.35", "0.98 0.80 0.30", "0.40 0.75 0.55", "0.45 0.60 0.90"]

TRAY_CENTER = np.array([0.13, -0.09, 0.0])
TRAY_HX, TRAY_HY, TRAY_WALL_H = 0.08, 0.06, 0.02


def seg_name(col, row):
    return f"seg_{col}{ROW_TAG[row]}"


def seg_offset(col, row):
    """段中心在 strip 局部坐标中的偏移。"""
    return np.array([COL_X0 + col * PITCH_X, ROW_Y[row], 0.0])


def all_segments():
    return [(c, r) for c in range(N_COLS) for r in range(N_ROWS)]


def weld_names_of(col, row):
    """返回连接该段的所有焊接（易撕线）名称。"""
    names = []
    if col == 0:
        names.append(f"w_strip_{ROW_TAG[row]}")
    else:
        names.append(f"w_col{col-1}{col}_{ROW_TAG[row]}")
    if col < N_COLS - 1:
        names.append(f"w_col{col}{col+1}_{ROW_TAG[row]}")
    names.append(f"w_row_c{col}")
    return names


def _strip_pose_at_qpos0():
    """用不含板块的基础场景计算 qpos0 时 strip 的世界位姿。"""
    xml = f"""
<mujoco model=\"probe\">
  <include file=\"{(HERE / 'scene_tear.xml').as_posix()}\"/>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    b = data.body("strip")
    return b.xpos.copy(), b.xquat.copy(), b.xmat.reshape(3, 3).copy()


def build_xml():
    """生成完整场景 XML 字符串（板块世界位姿与 qpos0 的 strip 对齐，焊接 relpose 取 qpos0）。"""
    pos0, quat0, R0 = _strip_pose_at_qpos0()
    assert np.allclose(R0, np.eye(3), atol=1e-6), "qpos0 时 strip 应与世界系对齐"

    segs, welds = [], []
    for c, r in all_segments():
        p = pos0 + R0 @ seg_offset(c, r)
        color = PILL_COLORS[c % len(PILL_COLORS)]
        segs.append(f"""
    <body name="{seg_name(c, r)}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}">
      <freejoint name="{seg_name(c, r)}_joint"/>
      <geom name="{seg_name(c, r)}_plate" type="box" size="{SEG_HX} {SEG_HY} {SEG_HZ}"
        friction="2 0.01 0.001" solref="0.004 1" rgba="0.82 0.83 0.87 1" mass="0.002"/>
      <geom name="{seg_name(c, r)}_dome" type="ellipsoid" size="0.0075 0.0075 0.0042" pos="0 0 0.005"
        contype="0" conaffinity="0" rgba="0.88 0.92 0.97 0.45" mass="0.0004"/>
      <geom name="{seg_name(c, r)}_pill" type="cylinder" size="0.005 0.0016" pos="0 0 0.004"
        contype="0" conaffinity="0" rgba="{color} 1" mass="0.0006"/>
    </body>""")

    # solref/solimp 硬化 + torquescale 放大，否则板会绕焊点像铰链一样下垂
    W = ('solref="0.0015 1" solimp="0.99 0.999 0.0001" torquescale="20"')
    for r in range(N_ROWS):
        welds.append(f'    <weld name="w_strip_{ROW_TAG[r]}" body1="strip" '
                     f'body2="{seg_name(0, r)}" anchor="-{SEG_HX} 0 0" {W}/>')
        for c in range(N_COLS - 1):
            welds.append(f'    <weld name="w_col{c}{c+1}_{ROW_TAG[r]}" body1="{seg_name(c, r)}" '
                         f'body2="{seg_name(c+1, r)}" anchor="-{SEG_HX} 0 0" {W}/>')
    for c in range(N_COLS):
        welds.append(f'    <weld name="w_row_c{c}" body1="{seg_name(c, 0)}" '
                     f'body2="{seg_name(c, 1)}" anchor="0 -{SEG_HY} 0" {W}/>')

    tray = []
    tx, ty = TRAY_CENTER[0], TRAY_CENTER[1]
    tray.append(f'    <geom name="tray_bottom" type="box" size="{TRAY_HX} {TRAY_HY} 0.0015" '
                f'pos="{tx} {ty} 0.0015" rgba="0.30 0.55 0.75 1"/>')
    for sx, sy, hx, hy in ((1, 0, 0.003, TRAY_HY), (-1, 0, 0.003, TRAY_HY),
                           (0, 1, TRAY_HX, 0.003), (0, -1, TRAY_HX, 0.003)):
        px = tx + sx * (TRAY_HX + 0.003)
        py = ty + sy * (TRAY_HY + 0.003)
        tray.append(f'    <geom type="box" size="{hx} {hy} {TRAY_WALL_H}" '
                    f'pos="{px:.4f} {py:.4f} {TRAY_WALL_H}" rgba="0.30 0.55 0.75 0.5"/>')

    return f"""
<mujoco model="aloha_pill_tear">
  <include file="{(HERE / 'scene_tear.xml').as_posix()}"/>

  <option timestep="0.001" integrator="implicitfast"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>

  <worldbody>
    <camera name="follow_pack" mode="targetbody" target="strip" pos="0.42 -0.40 0.38"/>
    <camera name="side" pos="-0.60 -0.28 0.34" xyaxes="0.5165 -0.8563 0.0000 0.2453 0.1480 0.9581"/>
{''.join(segs)}
{chr(10).join(tray)}
  </worldbody>

  <equality>
{chr(10).join(welds)}
  </equality>

  <contact>
    <exclude body1="strip" body2="left/left_finger_link"/>
    <exclude body1="strip" body2="left/right_finger_link"/>
    <exclude body1="strip" body2="left/gripper_base"/>
  </contact>
</mujoco>"""


def load_model():
    return mujoco.MjModel.from_xml_string(build_xml())


def place_segments(model, data):
    """把所有未撕下的板块按 strip 当前位姿刚性摆放（保持焊接约束满足）。"""
    mujoco.mj_forward(model, data)
    strip = data.body("strip")
    R = strip.xmat.reshape(3, 3)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, strip.xmat)
    for c, r in all_segments():
        p = strip.xpos + R @ seg_offset(c, r)
        adr = model.joint(f"{seg_name(c, r)}_joint").qposadr[0]
        data.qpos[adr: adr + 3] = p
        data.qpos[adr + 3: adr + 7] = quat
        vadr = model.joint(f"{seg_name(c, r)}_joint").dofadr[0]
        data.qvel[vadr: vadr + 6] = 0.0
    mujoco.mj_forward(model, data)


def weld_load(model, data, weld_name):
    """某条易撕线上的约束合力（efc 行绝对值之和，含力与力矩行）。"""
    eq_id = model.equality(weld_name).id
    total = 0.0
    for i in range(data.nefc):
        if (data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
                and data.efc_id[i] == eq_id):
            total += abs(data.efc_force[i])
    return total


if __name__ == "__main__":
    import imageio.v2 as imageio

    from ik_utils import ARM_JOINTS

    NEUTRAL_ARM = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0])
    model = load_model()
    data = mujoco.MjData(model)
    print(f"模型编译成功: nq={model.nq}, nbody={model.nbody}, neq={model.neq}")

    for side in ("left", "right"):
        for jname, q in zip(ARM_JOINTS, NEUTRAL_ARM):
            data.qpos[model.joint(f"{side}/{jname}").qposadr[0]] = q
            data.ctrl[model.actuator(f"{side}/{jname}").id] = q
    place_segments(model, data)

    # 探测：右夹爪手指开合方向（gripper_link 局部系）
    g = data.body("right/gripper_link")
    Rg = g.xmat.reshape(3, 3)
    sep_world = (data.site("right/left_finger").xpos - data.site("right/right_finger").xpos)
    sep_local = Rg.T @ (sep_world / np.linalg.norm(sep_world))
    print(f"右手指分离方向（世界系）: {np.round(sep_world/np.linalg.norm(sep_world), 3)}")
    print(f"右手指分离方向（gripper_link 局部系）: {np.round(sep_local, 3)}")
    print(f"右 gripper 站点位置: {np.round(data.site('right/gripper').xpos, 4)}")

    # 沉降测试：焊接是否稳住整板
    for _ in range(1000):
        mujoco.mj_step(model, data)
    for c, r in [(0, 0), (3, 1)]:
        print(f"{seg_name(c, r)} 沉降后位置: {np.round(data.body(seg_name(c, r)).xpos, 4)}")
    print(f"w_strip_f 静载: {weld_load(model, data, 'w_strip_f'):.3f}")

    renderer = mujoco.Renderer(model, height=720, width=1280)
    (HERE / "debug").mkdir(exist_ok=True)
    for cam in ("follow_pack", "side"):
        renderer.update_scene(data, camera=cam)
        imageio.imwrite(HERE / "debug" / f"tear_probe_{cam}.png", renderer.render())
    renderer.close()
    print("已渲染 debug/tear_probe_*.png")
