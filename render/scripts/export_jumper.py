"""Exports Jumper as rigid body meshes plus the animation tracks the film plays.

Run from the repository root with the project environment:
    .\\.venv\\Scripts\\python.exe render/scripts/export_jumper.py

MuJoCo loads jumper.xml and computes every body's pose; no jumper code is run.

- render/out/jumper.obj: one object per body, its visual meshes in the body's
  own frame (metres, z up, robot facing +x).
- render/out/jumper.json: per body its group (shell, display, detail, limbs),
  the shell and display boxes in their body frames, the rod frame in the
  left-front palm, and the tracks. A track is a list of frames at FPS, each
  frame one [x, y, z, qw, qx, qy, qz] per body, in the crab's floor frame.

Every track keeps the feet planted: the root pose and the arm joints are
scripted, and the leg joints are solved by damped least squares so each foot
tip stays where it stands in the rest pose (or a widened stance).

Poses and clips:
- stand: the first frame of the shipped gesture_salute clip;
- carry: stand with the left-front arm in jumper.five_foot's carry pose and the
  claw closed on the 12 mm rod handle; the angler never opens it;
- salute (right-front arm) and hello (left-front arm): the shipped gesture
  clips, 50 Hz, resampled.
"""
import json
import math
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
JUMPER = ROOT / "third_party" / "jumper"
MODEL = JUMPER / "assets" / "jumper" / "jumper.xml"
CLIPS = JUMPER / "tasks" / "jumper"
OUT = ROOT / "render" / "out"
FPS = 24
CLIP_HZ = 50
STORM, COVE = 10.5, 14.5          # s of film each scene's track covers

SHELL, DISPLAY = "upper_shell_link", "display_module_link"
DETAIL = {"tof_sensor_link", "camera_link"}
LEGS = ("LM", "RM", "LR", "RR")
# jumper.five_foot carry pose (tasks/jumper/five_foot/out/example/layout.json); the
# finger at -0.14 rad leaves the mouth about 12 mm open, the rod handle's diameter.
CARRY = {"LF_J0_joint": -0.5236, "LF_J1_joint": -1.5708, "LF_J2_joint": -0.5236,
         "LF_J3_joint": -1.309, "LF_J4_joint": -0.14}
ROD_PITCH = math.radians(35)      # rod raised above the robot's forward axis


def clip(name):
    d = json.loads((CLIPS / f"gesture_{name}" / "out" / "example" / f"{name}.motion.trajectory.json")
                   .read_text(encoding="utf-8"))
    return d["joint_order"], np.array(d["q"])


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def quat(roll=0.0, pitch=0.0, yaw=0.0):
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([roll, pitch, yaw]), "xyz")
    return q


class Rig:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.bodies = [m.body(b).name for b in range(1, m.nbody)]
        order, q = clip("salute")
        self.stand = dict(zip(order, q[0]))
        self.set_joints(self.stand)
        self.data.qpos[:3], self.data.qpos[3:7] = 0, (1, 0, 0, 0)
        mujoco.mj_kinematics(m, self.data)
        self.height = -self.floor()            # root height that puts the lowest point on z = 0
        self.root(np.zeros(3), quat())
        self.feet = {leg: self.data.xpos[m.body(f"{leg}_foot_tip_link").id].copy() for leg in LEGS}
        self.leg_dofs = {leg: [m.jnt_dofadr[m.joint(f"{leg}_J{k}_joint").id] for k in range(3)] for leg in LEGS}
        self.leg_qpos = {leg: [m.jnt_qposadr[m.joint(f"{leg}_J{k}_joint").id] for k in range(3)] for leg in LEGS}

    def set_joints(self, joints):
        for joint, value in joints.items():
            self.data.qpos[self.model.jnt_qposadr[self.model.joint(joint).id]] = value

    def root(self, offset, q):
        self.data.qpos[:3] = (0, 0, getattr(self, "height", 0.0)) + np.asarray(offset)
        self.data.qpos[3:7] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)          # the Jacobians need it

    def geoms(self):
        m = self.model
        for g in range(m.ngeom):
            if m.geom_contype[g] or m.geom_conaffinity[g] or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            yield g, m.body(m.geom_bodyid[g]).name, m.geom_dataid[g]

    def floor(self):
        m, d = self.model, self.data
        z = math.inf
        for g, _, mesh in self.geoms():
            v = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
            z = min(z, (d.geom_xpos[g][2] + v @ d.geom_xmat[g].reshape(3, 3).T[:, 2]).min())
        return z

    def plant(self, targets, iters=30):
        """Leg joints that put each foot tip on its target, within the joint ranges."""
        m, d = self.model, self.data
        jac = np.zeros((3, m.nv))
        for _ in range(iters):
            worst = 0.0
            for leg in LEGS:
                body = m.body(f"{leg}_foot_tip_link").id
                err = targets[leg] - d.xpos[body]
                worst = max(worst, np.linalg.norm(err))
                mujoco.mj_jacBody(m, d, jac, None, body)
                J = jac[:, self.leg_dofs[leg]]
                step = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), err)
                for adr, (k, dq) in zip(self.leg_qpos[leg], enumerate(step)):
                    jid = m.joint(f"{leg}_J{k}_joint").id
                    d.qpos[adr] = np.clip(d.qpos[adr] + dq, *m.jnt_range[jid])
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            if worst < 2e-4:
                break
        return worst

    def frame(self, joints, offset=(0, 0, 0), q=None, spread=0.0):
        """Body poses for scripted arm joints and root pose, feet planted (moved
        outward by `spread` metres for a braced stance)."""
        self.set_joints(self.stand | joints)
        self.root(offset, quat() if q is None else q)
        targets = {}
        for leg, p in self.feet.items():
            out = np.array([p[0], p[1], 0.0])
            targets[leg] = p + spread * out / np.linalg.norm(out)
        miss = self.plant(targets)
        poses = [list(self.data.xpos[b]) + list(self.data.xquat[b]) for b in range(1, self.model.nbody)]
        return np.round(poses, 5).tolist(), miss

    def export_meshes(self):
        m = self.model
        lines, base, groups = ["# Jumper, body frames, metres, z up, facing +x"], 1, {}
        per_body = {}
        for g, body, mesh in self.geoms():
            v = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
            f = m.mesh_face[m.mesh_faceadr[mesh]:m.mesh_faceadr[mesh] + m.mesh_facenum[mesh]]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            per_body.setdefault(body, []).append((m.geom_pos[g] + v @ R.reshape(3, 3).T, f))
        for body, parts in per_body.items():
            groups[body] = ("shell" if body == SHELL else "display" if body == DISPLAY
                            else "detail" if body in DETAIL else "limbs")
            lines.append(f"o {body}")
            for verts, faces in parts:
                lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in verts]
                lines += [f"f {a + base} {b + base} {c + base}" for a, b, c in faces]
                base += len(verts)
        (OUT / "jumper.obj").write_text("\n".join(lines) + "\n", encoding="utf-8")
        boxes = {b: [np.concatenate([v for v, _ in per_body[b]]).min(0).round(4).tolist(),
                     np.concatenate([v for v, _ in per_body[b]]).max(0).round(4).tolist()]
                 for b in (SHELL, DISPLAY)}
        return groups, boxes

    def rod_frame(self):
        """Grip point, closing axis and rod axis in the left-front palm's frame, in the carry pose."""
        m, d = self.model, self.data
        self.frame(CARRY)
        palm = m.body("LF_palm_link").id
        R, p = d.xmat[palm].reshape(3, 3), d.xpos[palm]
        pad, tip = (d.xpos[m.body(n).id] for n in ("LF_palm_grip_insert_link", "LF_finger_grip_insert_link"))
        closing = (pad - tip) / np.linalg.norm(pad - tip)
        grip = (pad + tip) / 2
        want = np.array([math.cos(ROD_PITCH), 0, math.sin(ROD_PITCH)])
        axis = want - closing * want.dot(closing)
        axis /= np.linalg.norm(axis)
        local = lambda v: (R.T @ v).round(5).tolist()
        return {"body": "LF_palm_link", "grip": local(grip - p), "closing": local(closing), "axis": local(axis),
                "gap": round(float(np.linalg.norm(pad - tip)), 4)}


def angler_storm(t):
    """Braced at the stern, rod out; nibbles from 7.6 s, the strike at 9.0 s yanks the
    rod up and back and the crab leans into it."""
    lift = 0.0
    if 7.6 < t < 8.8:
        lift = 0.06 * math.sin(2 * math.pi * 3.0 * (t - 7.6)) * math.sin(math.pi * (t - 7.6) / 1.2)
    strike = ease((t - 9.0) / 0.18) - 0.35 * ease((t - 9.6) / 0.9)
    tremble = 0.02 * math.sin(2 * math.pi * 7 * t) * (t > 9.0)
    joints = dict(CARRY)
    joints["LF_J1_joint"] += -0.55 * strike + lift + tremble
    joints["LF_J3_joint"] += -0.30 * strike
    joints["RF_J1_joint"] = -1.30                   # right claw lifted clear of the deck
    sway = 0.06 * math.sin(2 * math.pi * t / 3.1)
    lean = math.radians(9) * strike
    return joints, (-0.01 * strike, 0, -0.004), quat(sway, lean), 0.012


def sailor_storm(t):
    """Wide braced stance, low, rolling against the deck; the left claw out for
    balance, snapping at each lurch, the right one shut on the mop's handle."""
    roll = math.sin(2 * math.pi * t / 3.1)
    snap = 0.5 + 0.5 * math.sin(2 * math.pi * t / 1.55)
    joints = {"LF_J1_joint": -1.40 - 0.18 * roll, "RF_J1_joint": -1.40,     # stand -1.07; lower raises
              "LF_J4_joint": -0.9 * snap, "RF_J4_joint": 0.0}
    return joints, (0, 0.006 * roll, -0.022), quat(-0.12 * roll, 0.03 * math.sin(2 * math.pi * t / 2.3)), 0.022


def gesture(name, arm, t, start):
    """The clip's `arm` joints at film time t, the clip starting at `start` s."""
    order, q = clip(name)
    x = (t - start) * CLIP_HZ
    i = int(np.clip(math.floor(x), 0, len(q) - 1))
    j, w = min(i + 1, len(q) - 1), float(np.clip(x - i, 0, 1))
    row = q[i] * (1 - w) + q[j] * w
    return {n: float(v) for n, v in zip(order, row) if n.startswith(arm)}


def angler_cove(t):
    joints = dict(CARRY) | gesture("salute", "RF", t, 3.0)
    joints["RF_J1_joint"] -= 0.12                   # keep the resting claw off the deck
    return joints, (0, 0, 0.003 * math.sin(2 * math.pi * t / 2.6)), quat(), 0.0


def sailor_cove(t):
    joints = gesture("hello", "LF", t, 2.4)
    return joints, (0, 0, 0.003 * math.sin(2 * math.pi * t / 2.4 + 1)), quat(0, 0, 0.1 * ease((t - 2.0) / 1.0)), 0.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rig = Rig()
    groups, boxes = rig.export_meshes()
    meta = {"fps": FPS, "bodies": rig.bodies, "groups": groups, "shell": boxes[SHELL],
            "display": boxes[DISPLAY], "rod": rig.rod_frame(), "tracks": {}}
    for name, script, length in (("angler_storm", angler_storm, STORM), ("sailor_storm", sailor_storm, STORM),
                                 ("angler_cove", angler_cove, COVE), ("sailor_cove", sailor_cove, COVE),
                                 ("stand", lambda t: ({}, (0, 0, 0), quat(), 0.0), 0.0),
                                 ("carry", lambda t: (dict(CARRY), (0, 0, 0), quat(), 0.0), 0.0)):
        frames, worst = [], 0.0
        for k in range(max(1, round(length * FPS))):
            joints, offset, q, spread = script(k / FPS)
            poses, miss = rig.frame(joints, offset, q, spread)
            frames.append(poses)
            worst = max(worst, miss)
        meta["tracks"][name] = frames
        print(f"{name:13} {len(frames):4} frames, worst foot miss {worst * 1000:.2f} mm")
    (OUT / "jumper.json").write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    print("wrote render/out/jumper.obj and jumper.json")


if __name__ == "__main__":
    main()
