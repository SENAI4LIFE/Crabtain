"""Finds the smallest uniform boat scale that gives both crabs room to work.

Run from the repository root, after motion_envelope.py and after
    blender --background --factory-startup --python-exit-code 1 \\
        --python assets/boat/import_benchy.py -- --deck-out simulation/out/deck.npz

Zones, from the measured envelopes (simulation/out/envelopes.json) plus MARGIN:

- Angler: faces the stern; standing, gestures and the whole claw workspace, as a
  rectangle. The rod (ROD long, held at the claw) sweeps ROD_SWEEP degrees
  centered on dead aft; the sweep must not cross the wheelhouse or the sailor's
  zone. It passes over the rail; the rod angle that needs is reported.
- Sailor: walks WALK in a straight line and turns in place anywhere along it,
  with gestures and the claw: a stadium of the largest turning radius plus
  MARGIN around a WALK-long segment, fore and aft or athwartships.

The deck raster is the Benchy deck at its exported length with the hold hatched
flush; a larger boat is the same raster with the zones shrunk by the scale
factor. Zones stay on free deck aft of the wheelhouse (AFT_ONLY), or may also use
the deck under the wheelhouse roof, where the headroom is checked.

Writes simulation/out/deck_layout.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
DECK = ROOT / "simulation" / "out" / "deck.npz"
ENVELOPES = ROOT / "simulation" / "out" / "envelopes.json"
OUT = ROOT / "simulation" / "out" / "deck_layout.json"

MARGIN = 0.20      # m, around every envelope
WALK = 1.00        # m, sailor's straight walk
ROD = 1.20         # m, rod length
ROD_SWEEP = 90.0   # degrees, centered on dead aft
STEP = 0.01        # scale factor increment
WHEELHOUSE = ("block_bridge_walls", "block_doorframe_port", "block_doorframe_starboard",
              "block_wheel", "block_chimney_body")


def union(envs, names):
    e = [envs[n] for n in names]
    return {"x": [min(v["x"][0] for v in e), max(v["x"][1] for v in e)],
            "y": [min(v["y"][0] for v in e), max(v["y"][1] for v in e)],
            "turn_radius": max(v["turn_radius"] for v in e),
            "height": max(v["height"] for v in e)}


class Deck:
    def __init__(self, path, aft_only):
        d = np.load(path)
        self.cell, self.length = float(d["cell"]), float(d["length"])
        self.lo = d["lo"]
        self.level = float(d["level"])
        self.wheelhouse_aft = float(d["wheelhouse_aft"])
        self.headroom = float(d["roof_underside"]) - self.level
        self.rail = float(d["rail_top"]) - self.level
        blocks = [d[k] for k in d.files if k.startswith("block_")]
        self.wheelhouse = np.any([d[k] for k in WHEELHOUSE if k in d.files], axis=0)
        self.deck, self.hatch = d["deck"], d["hatch"]
        self.free = self.deck & ~np.any(blocks, axis=0)
        self.xs = self.lo[0] + (np.arange(self.free.shape[0]) + 0.5) * self.cell
        self.ys = self.lo[1] + (np.arange(self.free.shape[1]) + 0.5) * self.cell
        self.under_roof = (self.xs >= self.wheelhouse_aft)[:, None] & np.ones_like(self.free)
        if aft_only:
            self.free &= ~self.under_roof
        self.X, self.Y = np.meshgrid(self.xs, self.ys, indexing="ij")

    def cells(self, metres):
        return max(0, int(math.ceil(metres / self.cell)))


def rect_struct(deck, x0, x1, y0, y1):
    """Structuring element for a rectangle relative to its anchor, odd-sized and centered."""
    r = deck.cells(max(abs(x0), abs(x1), abs(y0), abs(y1)))
    c = np.arange(-r, r + 1) * deck.cell
    gx, gy = np.meshgrid(c, c, indexing="ij")
    return (gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)


def stadium_struct(deck, radius, half, along_x):
    r = deck.cells(radius + half)
    c = np.arange(-r, r + 1) * deck.cell
    gx, gy = np.meshgrid(c, c, indexing="ij")
    u, v = (gx, gy) if along_x else (gy, gx)
    return np.hypot(np.clip(np.abs(u) - half, 0, None), v) <= radius


def place(deck, struct, center):
    """Mask of `struct` placed with its center at grid index `center`."""
    out = np.zeros_like(deck.free)
    r = struct.shape[0] // 2
    i, j = center
    i0, j0 = i - r, j - r
    a0, b0 = max(0, -i0), max(0, -j0)
    a1 = min(struct.shape[0], out.shape[0] - i0)
    b1 = min(struct.shape[1], out.shape[1] - j0)
    out[i0 + a0:i0 + a1, j0 + b0:j0 + b1] = struct[a0:a1, b0:b1]
    return out


def sector(deck, apex, radius):
    dx, dy = deck.X - apex[0], deck.Y - apex[1]
    ang = np.degrees(np.arctan2(dy, -dx))     # 0 = dead aft
    return (np.hypot(dx, dy) <= radius) & (np.abs(ang) <= ROD_SWEEP / 2)


def fit(deck, angler, sailor, k, max_angler=40):
    """First layout at scale factor k (boat length = k * deck.length), or None."""
    m = MARGIN / k
    # Angler faces aft: robot +x is world -x.
    ax = (-(angler["x"][1]) / k - m, -(angler["x"][0]) / k + m)
    ay = (-(angler["y"][1]) / k - m, -(angler["y"][0]) / k + m)
    a_struct = rect_struct(deck, *ax, *ay)
    a_ok = ndimage.binary_erosion(deck.free, a_struct, border_value=0)
    claw = angler["claw_reach"] / k
    rs = sailor["turn_radius"] / k + m
    half = WALK / k / 2
    candidates = np.argwhere(a_ok)
    if not len(candidates):
        return None
    # Most aft first, spread across the beam.
    candidates = candidates[np.lexsort((np.abs(candidates[:, 1] - deck.free.shape[1] / 2),
                                        candidates[:, 0]))]
    candidates = candidates[:: max(1, len(candidates) // max_angler)][:max_angler]
    for along_x in (True, False):
        s_struct = stadium_struct(deck, rs, half, along_x)
        s_ok = ndimage.binary_erosion(deck.free, s_struct, border_value=0)
        if not s_ok.any():
            continue
        for a in candidates:
            apex = (deck.xs[a[0]] - claw, deck.ys[a[1]])
            rod = sector(deck, apex, ROD / k)
            if (rod & deck.wheelhouse).any():
                continue
            taken = place(deck, a_struct, a) | rod
            # Sailor centers whose stadium touches the angler zone or the rod sweep.
            n = deck.cells(half)
            line = np.ones((2 * n + 1, 1) if along_x else (1, 2 * n + 1), bool)
            swept = ndimage.binary_dilation(taken, line)
            near = ndimage.distance_transform_edt(~swept) * deck.cell <= rs
            ok = s_ok & ~near
            if ok.any():
                # Farthest sailor center from the angler.
                idx = np.argwhere(ok)
                s = idx[np.argmax(np.hypot(*(idx - a).T))]
                return {"angler": a, "sailor": s, "along_x": along_x, "apex": apex,
                        "a_struct": a_struct, "s_struct": s_struct, "rod": rod}
    return None


def main(aft_only=True):
    envs = json.loads(ENVELOPES.read_text(encoding="utf-8"))
    gestures = ["stand", "salute", "hello", "bow", "paw"]
    angler = union(envs, gestures + ["claw"]) | {"claw_reach": envs["claw"]["x"][1]}
    sailor = union(envs, gestures + ["claw"])
    deck = Deck(DECK, aft_only)

    # Coarse steps of 10 * STEP up to the first fit, then STEP from the last miss.
    k, layout = 1.0, None
    while layout is None:
        if k > 4.0:
            sys.exit("no layout up to 4x")
        layout = fit(deck, angler, sailor, k)
        if layout is None:
            k = round(k + 10 * STEP, 4)
    lo = max(1.0, round(k - 10 * STEP, 4))
    for fine in np.arange(lo + STEP, k, STEP):
        found = fit(deck, angler, sailor, round(float(fine), 4))
        if found is not None:
            k, layout = round(float(fine), 4), found
            break

    length = deck.length * k
    a, s = layout["angler"], layout["sailor"]
    zones = place(deck, layout["a_struct"], a) | place(deck, layout["s_struct"], s)
    under = bool((zones & deck.under_roof).any())
    tallest = max(angler["height"], sailor["height"])
    # Rod over the stern rail: the claw holds the butt at most `height` above the deck.
    to_rail = layout["apex"][0] - deck.lo[0]
    rail_angle = math.degrees(math.atan2(max(0.0, deck.rail * k - angler["height"]), to_rail * k))
    result = {
        "scale_factor": k, "boat_length": round(length, 3),
        "benchy_scale": round(length / 0.060, 2),
        "aft_only": aft_only, "uses_deck_under_roof": under,
        "headroom_under_roof": round(deck.headroom * k, 3), "tallest_pose": round(tallest, 3),
        "rail_above_deck": round(deck.rail * k, 3),
        "rod_pitch_to_clear_stern_rail_deg": round(rail_angle, 1),
        "margin": MARGIN, "walk": WALK, "rod": ROD, "rod_sweep_deg": ROD_SWEEP,
        "angler_envelope": angler, "sailor_envelope": sailor,
        "angler_center": [round(deck.xs[a[0]] * k, 3), round(deck.ys[a[1]] * k, 3)],
        "angler_zone": {"x": [round((deck.xs[a[0]] * k - angler["x"][1] - MARGIN), 3),
                              round((deck.xs[a[0]] * k - angler["x"][0] + MARGIN), 3)],
                        "y": [round((deck.ys[a[1]] * k - angler["y"][1] - MARGIN), 3),
                              round((deck.ys[a[1]] * k - angler["y"][0] + MARGIN), 3)]},
        "rod_apex": [round(layout["apex"][0] * k, 3), round(layout["apex"][1] * k, 3)],
        "sailor_center": [round(deck.xs[s[0]] * k, 3), round(deck.ys[s[1]] * k, 3)],
        "sailor_walk_along": "x" if layout["along_x"] else "y",
        "sailor_radius": round(sailor["turn_radius"] + MARGIN, 3),
        "wheelhouse_aft": round(deck.wheelhouse_aft * k, 3),
        "stern": round(deck.lo[0] * k, 3),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for key, value in list(result.items()):
        if isinstance(value, list):
            result[key] = [float(v) for v in value]
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for key, value in result.items():
        if not isinstance(value, dict):
            print(f"{key:36} {value}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    if "--walk" in sys.argv:
        WALK = float(sys.argv[sys.argv.index("--walk") + 1])
    main(aft_only="--under-roof" not in sys.argv)
