"""Builds the boat baseline from the CC0 multi-part 3DBenchy and measures its aft deck.

Run from the repository root after scripts/setup.ps1:
    blender --background --factory-startup --python-exit-code 1 --python assets/boat/import_benchy.py

Imports the 17 part STLs from third_party/3dbenchy, names them, scales them
uniformly to the boat length (true Benchy proportions), puts the keel at Z = 0
with the bow towards +X, and saves assets/boat/source/boat.blend. Then it reports
whether two Jumper footprints fit on the deck. Geometry is not changed beyond
the uniform scale.

With `-- --deck-out <file.npz>` it also writes the deck rasters that
simulation/scripts/deck_layout.py fits the crews' working zones into.
"""
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[2]
PARTS_DIR = ROOT / "third_party" / "3dbenchy"
OUT = ROOT / "assets" / "boat" / "source" / "boat.blend"

LENGTH = 6.60     # m, boat length overall; sized by simulation/scripts/deck_layout.py
JUMPER = 0.40     # m, Jumper footprint, square
CLEARANCE = 0.20  # m, to rails, obstacles and the other Jumper
CELL = 0.01       # m, raster size for the deck measurement

PARTS = [
    "Hull", "Deck surface", "Gunwale", "Bridge walls", "Bridge roof", "Window",
    "Stern window", "Doorframe port", "Doorframe starboard", "Chimney body",
    "Chimney top", "Cargo box", "Fishing-rod-holder", "Wheel", "Hawsepipe port",
    "Hawsepipe starboard", "Stern name plate",
]
# Parts that stand on the deck and block a Jumper.
OBSTACLES = [
    "Gunwale", "Bridge walls", "Doorframe port", "Doorframe starboard",
    "Cargo box", "Fishing-rod-holder", "Wheel", "Chimney body",
]


def object_name(part):
    return "Benchy." + part.lower().replace(" ", "_").replace("-", "_")


def import_parts():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    boat = bpy.data.collections.new("Boat")
    bpy.context.scene.collection.children.link(boat)
    objects = {}
    for part in PARTS:
        path = PARTS_DIR / f"{part}.stl"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run scripts/setup.ps1")
        bpy.ops.wm.stl_import(filepath=str(path))
        obj = bpy.context.selected_objects[0]
        obj.name = obj.data.name = object_name(part)
        for c in obj.users_collection:
            c.objects.unlink(obj)
        boat.objects.link(obj)
        objects[part] = obj
    return objects


def bounds(objs):
    pts = np.array([o.matrix_world @ v.co for o in objs for v in o.data.vertices])
    return pts.min(axis=0), pts.max(axis=0)


def place(objects):
    """Uniform scale to LENGTH, bow to +X, centered in X and Y, keel at Z = 0."""
    lo, hi = bounds(objects.values())
    scale = LENGTH / (hi[0] - lo[0])
    offset = Vector((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
    matrix = Matrix.Scale(scale, 4) @ Matrix.Translation(offset)
    # The stern name plate sits at the stern; turn 180 degrees if the bow points to -X.
    if np.mean([v.co.x for v in objects["Stern name plate"].data.vertices]) + offset.x > 0:
        matrix = Matrix.Rotation(math.pi, 4, "Z") @ matrix
    for obj in objects.values():
        obj.data.transform(matrix)
        obj.data.update()
    return scale


def triangles(obj):
    mesh = obj.data
    mesh.calc_loop_triangles()
    co = np.array([v.co for v in mesh.vertices])
    idx = np.array([t.vertices for t in mesh.loop_triangles])
    return co[idx]  # (n, 3 corners, xyz)


def sample(tris, spacing):
    """Points covering each triangle, including triangles seen edge-on from above."""
    pts = []
    for t in tris:
        n = max(1, math.ceil(max(np.linalg.norm(t[1] - t[0]), np.linalg.norm(t[2] - t[0]),
                                 np.linalg.norm(t[2] - t[1])) / spacing))
        for i in range(n + 1):
            for j in range(n + 1 - i):
                a, b = i / n, j / n
                pts.append(t[0] + a * (t[1] - t[0]) + b * (t[2] - t[0]))
    return np.array(pts)


def erode(mask, r_cells, disk=True):
    out = mask.copy()
    for dx in range(-r_cells, r_cells + 1):
        for dy in range(-r_cells, r_cells + 1):
            if disk and dx * dx + dy * dy > r_cells * r_cells:
                continue
            shifted = np.zeros_like(mask)
            xs = slice(max(dx, 0), mask.shape[0] + min(dx, 0))
            xd = slice(max(-dx, 0), mask.shape[0] + min(-dx, 0))
            ys = slice(max(dy, 0), mask.shape[1] + min(dy, 0))
            yd = slice(max(-dy, 0), mask.shape[1] + min(-dy, 0))
            shifted[xd, yd] = mask[xs, ys]
            out &= shifted
    return out


class Deck:
    """Rasters of the main deck and of every obstacle standing on it."""

    def __init__(self, objects):
        deck = triangles(objects["Deck surface"])
        normals = np.cross(deck[:, 1] - deck[:, 0], deck[:, 2] - deck[:, 0])
        up = deck[normals[:, 2] > 0.9 * np.linalg.norm(normals, axis=1)]
        self.level = float(np.median(up[:, :, 2]))
        top = up[np.all(np.abs(up[:, :, 2] - self.level) < 0.02, axis=1)]
        self.lo, hi = bounds(objects.values())
        self.shape = (int((hi[0] - self.lo[0]) / CELL) + 1, int((hi[1] - self.lo[1]) / CELL) + 1)
        self.top = self.raster(sample(top, CELL / 2))
        self.blocks, self.footprints = {}, {}
        for part in OBSTACLES:
            pts = sample(triangles(objects[part]), CELL / 2)
            # Rectangle under the part: the cargo box is a coaming around an open hold,
            # so taking it off means covering the hold with a flush hatch.
            (i0, j0), (i1, j1) = (((p[:2] - self.lo[:2]) / CELL).astype(int)
                                  for p in (pts.min(axis=0), pts.max(axis=0)))
            self.footprints[part] = np.zeros(self.shape, bool)
            self.footprints[part][i0:i1 + 1, j0:j1 + 1] = True
            self.blocks[part] = self.raster(
                pts[(pts[:, 2] > self.level + 0.005) & (pts[:, 2] < self.level + 1.0)])
        self.wheelhouse_aft = bounds([objects["Bridge walls"]])[0][0]
        self.roof_underside = bounds([objects["Bridge roof"]])[0][2]
        # Stern rail: the gunwale's top over its aftmost 0.10 m (the bow rail is far higher).
        rail = np.array([v.co for v in objects["Gunwale"].data.vertices])
        self.rail_top = rail[rail[:, 0] < rail[:, 0].min() + 0.10][:, 2].max()
        self.xs = self.lo[0] + (np.arange(self.shape[0]) + 0.5) * CELL

    def export(self, path, removed):
        """Deck rasters with `removed` parts taken off, for simulation/scripts/deck_layout.py."""
        deck = self.top.copy()
        for part in removed:
            deck |= self.footprints[part]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, cell=CELL, length=LENGTH, lo=self.lo[:2], level=self.level,
            wheelhouse_aft=self.wheelhouse_aft, roof_underside=self.roof_underside,
            rail_top=self.rail_top, deck=deck, hatch=self.footprints["Cargo box"],
            **{"block_" + p.lower().replace(" ", "_").replace("-", "_"): self.blocks[p]
               for p in OBSTACLES if p not in removed})

    def raster(self, points):
        grid = np.zeros(self.shape, bool)
        if len(points):
            ij = ((points[:, :2] - self.lo[:2]) / CELL).astype(int)
            grid[ij[:, 0], ij[:, 1]] = True
        return grid

    def report(self, label, removed=(), aft_only=True):
        """Free deck with `removed` parts taken off (the rectangle under each becomes deck)."""
        free = self.top.copy()
        for part in removed:
            free |= self.footprints[part]
        for part in OBSTACLES:
            if part not in removed:
                free &= ~self.blocks[part]
        if aft_only:
            free &= (self.xs < self.wheelhouse_aft)[:, None]

        rows = np.nonzero(free.any(axis=1))[0]
        length = (rows.max() - rows.min() + 1) * CELL if len(rows) else 0.0
        width = free.sum(axis=1).max() * CELL if len(rows) else 0.0
        # Centers where a Jumper square keeps CLEARANCE to everything that is not free deck.
        clear = erode(free, round(CLEARANCE / CELL))
        centers = erode(clear, round(JUMPER / 2 / CELL), disk=False)
        ij = np.argwhere(centers)
        fits_two = len(ij) > 0 and np.ptp(ij, axis=0).max() * CELL >= JUMPER + CLEARANCE
        print(f"-- {label}")
        print(f"   {length:.2f} m long, up to {width:.2f} m wide, {free.sum() * CELL * CELL:.2f} m2 free;"
              f" one Jumper: {'yes' if len(ij) else 'no'}, two: {'yes' if fits_two else 'no'}")


def main():
    objects = import_parts()
    scale = place(objects)
    lo, hi = bounds(objects.values())
    roof = bounds([objects["Bridge roof"]])[1][2]
    print(f"scale             x{scale * 1000:.2f} (mm to m: {scale:.5f})")
    print(f"boat              {hi[0] - lo[0]:.2f} L x {hi[1] - lo[1]:.2f} W x {hi[2] - lo[2]:.2f} H m"
          f" (roof {roof:.2f} m)")
    deck = Deck(objects)
    print(f"deck level        {deck.level:.3f} m above keel")
    print(f"wheelhouse aft    x = {deck.wheelhouse_aft:.3f} m (stern at x = {lo[0]:.3f} m)")
    print(f"Jumper fit test   {JUMPER:.2f} m square, {CLEARANCE:.2f} m clearance to rails, "
          f"obstacles and each other")
    movable = ("Cargo box", "Fishing-rod-holder")
    deck.report("aft deck, as modeled")
    deck.report("aft deck, hold hatched flush, no rod holder", removed=movable)
    deck.report("main deck incl. under the wheelhouse roof, hold hatched flush, no rod holder",
                removed=movable, aft_only=False)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--deck-out" in argv:
        path = (ROOT / argv[argv.index("--deck-out") + 1]).resolve()
        deck.export(path, movable)
        print(f"deck rasters      {path.relative_to(ROOT)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT), compress=True)
    print(f"saved             {OUT.relative_to(ROOT)}")


main()
