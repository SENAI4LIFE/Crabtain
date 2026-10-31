"""Measures how much deck each Jumper behavior needs, from jumper's own data.

Run from the repository root with the project environment:
    .\\.venv\\Scripts\\python.exe simulation/scripts/motion_envelope.py

Only jumper's model and shipped motion clips are read; no jumper code is run.
Every geom of the robot is tracked through its bounding box in world space.

- stand: the default standing pose.
- gestures: the shipped salute, hello, bow and paw clips
  (tasks/jumper/gesture_*/out/example/*.motion.trajectory.json), replayed frame by
  frame in place, with the clip's body orientation and the feet on the deck.
- claw: the left-front arm's whole workspace. Its four joints are swept over
  their full ranges with the rest of the robot standing; poses that put any part
  of the arm below the deck are dropped. This bounds "strike with the claw
  extended" in every direction.

Prints per behavior the horizontal envelope relative to the body center (x
forward, y left), the turning radius (farthest horizontal point from the center,
i.e. the circle swept when turning in place) and the highest point, and writes
simulation/out/envelopes.json.
"""
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
JUMPER = ROOT / "third_party" / "jumper"
MODEL = JUMPER / "assets" / "jumper" / "jumper.xml"
GESTURES = ["salute", "hello", "bow", "paw"]
OUT = ROOT / "simulation" / "out" / "envelopes.json"
ARM = ["LF_J0_joint", "LF_J1_joint", "LF_J2_joint", "LF_J3_joint"]
ARM_STEPS = 12      # samples per arm joint over its range
FRAME_STRIDE = 2    # every second clip frame


class Robot:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0]
        signs = np.array(list(itertools.product((-1, 1), repeat=3)))
        # Bounding-box corners of every geom in its own frame.
        self.corners = np.stack([m.geom_aabb[g, :3] + signs * m.geom_aabb[g, 3:]
                                 for g in self.geoms])
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
        self.root = m.jnt_qposadr[free]

    def qadr(self, name):
        return self.model.jnt_qposadr[self.model.joint(name).id]

    def points(self, joints, quat=(1, 0, 0, 0)):
        """World bounding-box corners of all geoms, feet resting on z = 0, center at x = y = 0."""
        d = self.data
        d.qpos[:] = self.model.qpos0
        for name, value in joints.items():
            d.qpos[self.qadr(name)] = value
        d.qpos[self.root:self.root + 3] = 0
        d.qpos[self.root + 3:self.root + 7] = quat
        mujoco.mj_kinematics(self.model, d)
        pos = d.geom_xpos[self.geoms]
        rot = d.geom_xmat[self.geoms].reshape(-1, 3, 3)
        pts = pos[:, None, :] + np.einsum("gij,gkj->gki", rot, self.corners)
        pts[..., 2] -= pts[..., 2].min()
        return pts

    def arm_mask(self):
        """Geoms that move with the left-front arm."""
        m = self.model
        base = m.jnt_bodyid[m.joint(ARM[0]).id]
        def under(b):
            while b != 0:
                if b == base:
                    return True
                b = m.body_parentid[b]
            return False
        return np.array([under(m.geom_bodyid[g]) for g in self.geoms])


def envelope(pts):
    xy = pts[..., :2].reshape(-1, 2)
    return {
        "x": [float(xy[:, 0].min()), float(xy[:, 0].max())],
        "y": [float(xy[:, 1].min()), float(xy[:, 1].max())],
        "turn_radius": float(np.hypot(xy[:, 0], xy[:, 1]).max()),
        "height": float(pts[..., 2].max()),
    }


def load_clip(name):
    path = next((JUMPER / "tasks" / "jumper").glob(f"gesture_*/out/example/{name}.motion.trajectory.json"))
    clip = json.loads(path.read_text(encoding="utf-8"))
    return clip["joint_order"], clip["q"], clip["root_quat"], path.relative_to(JUMPER).as_posix()


def main():
    robot = Robot()
    order, q, _, _ = load_clip("salute")
    stand = dict(zip(order, q[0]))
    results = {"stand": envelope(robot.points(stand))}

    for name in GESTURES:
        order, q, quats, source = load_clip(name)
        pts = np.concatenate([robot.points(dict(zip(order, q[i])), quats[i])
                              for i in range(0, len(q), FRAME_STRIDE)])
        results[name] = envelope(pts) | {"source": source, "frames": len(q)}

    arm = robot.arm_mask()
    ranges = [robot.model.jnt_range[robot.model.joint(j).id] for j in ARM]
    kept = []
    for values in itertools.product(*(np.linspace(lo, hi, ARM_STEPS) for lo, hi in ranges)):
        joints = dict(stand) | dict(zip(ARM, values))
        pts = robot.points(joints)
        # The standing feet stay on the deck; poses that put the arm through it are dropped.
        pts[..., 2] -= pts[~arm][..., 2].min()
        if pts[arm][..., 2].min() >= -1e-3:
            kept.append(pts[arm])
    results["claw"] = envelope(np.concatenate(kept)) | {
        "samples": ARM_STEPS ** len(ARM), "kept": len(kept)}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"{'behavior':10} {'x min':>7} {'x max':>7} {'y min':>7} {'y max':>7} {'turn r':>7} {'height':>7}")
    for name, e in results.items():
        print(f"{name:10} {e['x'][0]:7.3f} {e['x'][1]:7.3f} {e['y'][0]:7.3f} {e['y'][1]:7.3f} "
              f"{e['turn_radius']:7.3f} {e['height']:7.3f}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
