"""一次性生成撕剪版模型文件：
- aloha_tear.xml：v3 起左夹爪不再固连药板（药板是场景中的自由体，需真实抓取）；
- scene_tear.xml：include aloha_tear.xml 的场景包装。
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---- aloha_tear.xml ----
src = (HERE / "aloha_nokey.xml").read_text(encoding="utf-8")
start = src.index('<body name="pill_pack"')
end = src.index('<site name="left/gripper"', start)
out = src[:start] + src[end:]
(HERE / "aloha_tear.xml").write_text(out, encoding="utf-8")
print("aloha_tear.xml written,", len(out), "chars")

# ---- scene_tear.xml ----
src = (HERE / "scene_nokey.xml").read_text(encoding="utf-8")
out = (src.replace("aloha_scene_nokey", "aloha_scene_tear")
          .replace('<include file="aloha_nokey.xml"/>', '<include file="aloha_tear.xml"/>'))
(HERE / "scene_tear.xml").write_text(out, encoding="utf-8")
print("scene_tear.xml written")
