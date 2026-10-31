"""Renders the film's inset: the crabs' take as MuJoCo draws it, no outfits, no
boat, the same body poses the film plays at the same moment.

Run from the repository root with the project environment, after
render/scripts/export_jumper.py:
    .\\.venv\\Scripts\\python.exe render/scripts/sim_inset.py

Writes render/out/sim/NNNN.png, one 640x360 image per film frame 336-431
(14 to 18 s, past the sky passage's speed ramp), while the sailor waves and the
angler salutes; film.py lays it over the cove orbit in the 16:9 cut.
"""
import json
import math
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "third_party" / "jumper" / "assets" / "jumper" / "jumper.xml"
OUT = ROOT / "render" / "out"
WINDOW = range(336, 432)                # film frames, 14.0-18.0 s: after the passage's speed ramp
COVE_START = 252                        # film frame of the cove track's first frame
# Stations on the deck, as in scene.py: (x, y), heading.
CREW = {"angler": ((-2.40, -0.45), math.pi), "sailor": ((-2.38, 0.50), -math.pi / 2 - 0.45)}
WIDTH, HEIGHT = 640, 360


def placed(model, meta, track, k, at, heading):
    """MjData whose body and geom frames hold track frame k, moved to its station."""
    d = mujoco.MjData(model)
    turn = np.array([[math.cos(heading), -math.sin(heading), 0], [math.sin(heading), math.cos(heading), 0], [0, 0, 1]])
    shift = np.array([at[0], at[1], 0.0])
    for name, pose in zip(meta["bodies"], track[k]):
        b = model.body(name).id
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.array(pose[3:]))
        d.xpos[b] = turn @ np.array(pose[:3]) + shift
        d.xmat[b] = (turn @ rot.reshape(3, 3)).ravel()
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        body = d.xmat[b].reshape(3, 3)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[g])
        d.geom_xpos[g] = d.xpos[b] + body @ model.geom_pos[g]
        d.geom_xmat[g] = (body @ rot.reshape(3, 3)).ravel()
    return d


def deck(scene):
    """A plain floor under both stations."""
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.array([0.45, 0.75, 0.004]),
                        np.array([-2.39, 0.02, -0.004]), np.eye(3).ravel(), np.array([0.32, 0.34, 0.38, 1.0], dtype=np.float32))
    scene.ngeom += 1


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    meta = json.loads((OUT / "jumper.json").read_text(encoding="utf-8"))
    (OUT / "sim").mkdir(parents=True, exist_ok=True)
    for old in (OUT / "sim").glob("*.png"):              # film.py plays whatever is there
        old.unlink()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (-2.36, -0.06, 0.06)
    cam.distance, cam.azimuth, cam.elevation = 1.3, 25.0, -18.0          # from aft and starboard, facing both
    option = mujoco.MjvOption()
    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        for f in WINDOW:
            k = f - COVE_START
            first = True
            for crab, (at, heading) in CREW.items():
                d = placed(model, meta, meta["tracks"][f"{crab}_cove"], k, at, heading)
                if first:
                    renderer.update_scene(d, camera=cam, scene_option=option)
                    first = False
                else:
                    mujoco.mjv_addGeoms(model, d, option, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC,
                                        renderer.scene)
            deck(renderer.scene)
            pixels = renderer.render()
            write_png(OUT / "sim" / f"{f:04d}.png", pixels)
    print(f"wrote {len(WINDOW)} inset frames to render/out/sim/")


def write_png(path, pixels):
    """RGB uint8 array to PNG with the standard library only."""
    import struct
    import zlib
    h, w, _ = pixels.shape
    raw = b"".join(b"\x00" + pixels[y].tobytes() for y in range(h))
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


if __name__ == "__main__":
    main()
