"""一次性生成撕剪版模型文件：
- aloha_tear.xml：把左手上的整块药板换成窄持板条（strip），其余与 v1 相同；
- scene_tear.xml：include aloha_tear.xml 的场景包装。
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---- aloha_tear.xml ----
src = (HERE / "aloha_nokey.xml").read_text(encoding="utf-8")
start = src.index('<body name="pill_pack"')
end = src.index('<site name="left/gripper"', start)
strip = """<body name="strip" pos="0.19 0 0">
                      <geom name="strip_tab" type="box" size="0.0075 0.012 0.0045"
                        pos="-0.0245 0 -0.0015" rgba="0.25 0.3 0.4 1" mass="0.004"/>
                      <geom name="strip_bar" type="box" size="0.010 0.027 0.0012" pos="0 0 0"
                        friction="1.5 0.01 0.001" rgba="0.80 0.81 0.85 1" mass="0.003"/>
                      <site name="strip_center" pos="0 0 0" group="5"/>
                    </body>
                    """
out = src[:start] + strip + src[end:]
(HERE / "aloha_tear.xml").write_text(out, encoding="utf-8")
print("aloha_tear.xml written,", len(out), "chars")

# ---- scene_tear.xml ----
src = (HERE / "scene_nokey.xml").read_text(encoding="utf-8")
out = (src.replace("aloha_scene_nokey", "aloha_scene_tear")
          .replace('<include file="aloha_nokey.xml"/>', '<include file="aloha_tear.xml"/>'))
(HERE / "scene_tear.xml").write_text(out, encoding="utf-8")
print("scene_tear.xml written")
