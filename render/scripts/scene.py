"""Shared scene builder for the Crabtain renders: boat, crew, outfits, captain, shots.

Imported by the render scripts (crew.py, storm.py, cove.py, film.py), which run inside
Blender. Inputs, all generated locally:
- assets/boat/source/boat.blend from assets/boat/import_benchy.py;
- render/out/jumper.obj and jumper.json from render/scripts/export_jumper.py.

The boat is the plain CC0 3DBenchy at x110 with no added geometry; only its
materials change (plank wood, wear, wet sheen, a waterline grime band). The one
exception is a flat glass pane in each existing window opening, so the windows
can hold glass and the cabin's interior glow.

The crew are Jumper's rigid bodies keyed per frame from the exported tracks, so
the rod, parented to the angler's left-front palm, stays welded in its closed
claw. Garments are grown from the shell's own outer faces, pushed out along the
surface normals in layers, so nothing intersects the shell or each other;
check_clearance() measures it.
"""
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree

ROOT = Path(__file__).resolve().parents[2]
BOAT = ROOT / "assets" / "boat" / "source" / "boat.blend"
OUT = ROOT / "render" / "out"
REVIEW = ROOT / "review"
FPS = 24
# The sky passage between the storm and the bay: the film holds the time-lapse sky
# for PASSAGE frames after the cut, then plays the bay's frames RAMP a little fast,
# easing in and out, so from RAMP[1] on the film and the cove's own clock agree.
CUT = 252             # film frame where the cove's clock starts
PASSAGE = 24          # film frames
RAMP = (264, 336)     # cove frames, while its camera barely moves

WATERLINE = 0.45      # m above the keel
DECK = 0.94           # m, the aft cockpit sole
CAPTAIN = 0.75        # m, penguin height
ROD = 1.20            # m
CLEAR = 0.0012        # m, gap between the shell and the first cloth layer

# Crew stations on the aft cockpit, the free deck between the rod-holder post and
# the cargo hold (x -2.75 to -2.07 m): the angler to starboard facing aft, the
# sailor to port facing the angler. Each zone keeps 0.2 m from the other.
ANGLER_AT, ANGLER_HEADING = (-2.40, -0.45), math.pi
SAILOR_AT, SAILOR_HEADING = (-2.38, 0.50), -math.pi / 2 - 0.45

# Project palette
OAK, TAR, PINE, DARK_OAK, BRASS, IRON = "#4A3426", "#1B1A1F", "#A87A4C", "#6B4A33", "#C9A13B", "#3A3B3E"
BLACK, BONE, CANVAS, HEMP, AGED_WOOD = "#121212", "#EDE3CC", "#E4D6B5", "#B8955E", "#7A5232"
FLAME, LANTERN, GOLD, LAGOON = "#FFB85C", "#A8782F", "#D4A017", "#3E8E87"
OXBLOOD, WALNUT, SCARLET = "#8E2A22", "#3B2A20", "#B83A2E"
SEA_TEAL, SAND, MUSTARD = "#2C5F6F", "#C8B48A", "#C9962B"
INK, CREAM, ORANGE, NAVY, PITCH = "#15181C", "#F1EBDD", "#E8892B", "#1E2A44", "#1B1A1F"
GRIME, VERDIGRIS, RUST, LINEN, CAP = "#3A3F2A", "#4F7A6A", "#7A3E22", "#DCCFB2", "#A23A16"

CREW = {"angler": {"shell": OXBLOOD, "limbs": WALNUT, "detail": INK, "eyes": "#FFC65A"},
        "sailor": {"shell": SEA_TEAL, "limbs": SAND, "detail": INK, "eyes": "#8EEBFF"}}


def args():
    """Arguments after Blender's `--`."""
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


# Materials -------------------------------------------------------------------

def rgb(hex_color, alpha=1.0):
    """sRGB hex to linear RGBA."""
    c = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c] + [alpha]


def material(name, color, roughness=0.6, metallic=0.0, emission=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgb(color)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if emission:
        bsdf.inputs["Emission Color"].default_value = rgb(color)
        bsdf.inputs["Emission Strength"].default_value = emission
    return mat


class Nodes:
    """Small helper for building shader graphs."""

    def __init__(self, mat):
        self.tree = mat.node_tree
        self.bsdf = self.tree.nodes.get("Principled BSDF")

    def new(self, kind, **inputs):
        node = self.tree.nodes.new(kind)
        for key, value in inputs.items():
            if key in node.inputs:
                self.link(value, node.inputs[key]) if hasattr(value, "links") else \
                    setattr(node.inputs[key], "default_value", value)
            else:
                setattr(node, key, value)
        return node

    def link(self, a, b):
        self.tree.links.new(a, b)

    def math(self, op, a, b=0.0, clamp=False):
        node = self.tree.nodes.new("ShaderNodeMath")
        node.operation, node.use_clamp = op, clamp
        for socket, value in zip(node.inputs, (a, b)):
            self.link(value, socket) if hasattr(value, "links") else setattr(socket, "default_value", value)
        return node.outputs[0]

    def mix(self, factor, a, b, blend="MIX"):
        node = self.tree.nodes.new("ShaderNodeMix")
        node.data_type, node.blend_type = "RGBA", blend
        for socket, value in ((node.inputs["Factor"], factor), (node.inputs["A"], a), (node.inputs["B"], b)):
            self.link(value, socket) if hasattr(value, "links") else setattr(socket, "default_value", value)
        return node.outputs["Result"]

    def ramp(self, fac, stops, constant=False):
        node = self.tree.nodes.new("ShaderNodeValToRGB")
        if constant:
            node.color_ramp.interpolation = "CONSTANT"
        els = node.color_ramp.elements
        while len(els) < len(stops):
            els.new(0.5)
        for el, (pos, color) in zip(els, stops):
            el.position = pos
            el.color = color if len(color) == 4 else list(color) + [1]
        self.link(fac, node.inputs["Fac"])
        return node.outputs["Color"]

    def coord(self, kind="Object"):
        return self.new("ShaderNodeTexCoord").outputs[kind]

    def xyz(self, vector):
        node = self.new("ShaderNodeSeparateXYZ")
        self.link(vector, node.inputs[0])
        return node.outputs

    def scaled(self, vector, scale):
        node = self.new("ShaderNodeMapping")
        node.inputs["Scale"].default_value = scale
        self.link(vector, node.inputs["Vector"])
        return node.outputs[0]

    def noise(self, vector, scale, detail=4.0, rough=0.55):
        node = self.new("ShaderNodeTexNoise", Scale=scale, Detail=detail, Roughness=rough)
        self.link(vector, node.inputs["Vector"])
        return node.outputs["Fac"]


def wet_finish(n, color, roughness, wet):
    """Wet look: darker base, lower roughness and a clear water coat."""
    if wet <= 0:
        return color, roughness
    color = n.mix(0.35 * wet, color, [0, 0, 0, 1], "MULTIPLY") if hasattr(color, "links") else \
        [c * (1 - 0.3 * wet) for c in color[:3]] + [1]
    n.bsdf.inputs["Coat Weight"].default_value = 0.85 * wet
    n.bsdf.inputs["Coat Roughness"].default_value = 0.04
    roughness = n.math("MULTIPLY", roughness, 1 - 0.55 * wet) if hasattr(roughness, "links") else roughness * (1 - 0.55 * wet)
    return color, roughness


def wood(name, color, axis, width, grain="X", seam=WALNUT, wet=0.0, waterline=False):
    """Plank wood: boards `width` m wide stacked along `axis`, grain running along
    `grain`; each board its own shade, dark seams, grain streaks, lighter worn
    edges, grime in the crevices; optionally a tar boot-top and a waterline grime
    band; wet sheen when wet > 0."""
    mat = material(name, color, 0.72)
    n = Nodes(mat)
    co = n.coord()
    sep = n.xyz(co)
    along = {"X": 0, "Y": 1, "Z": 2}[axis]
    # Board index -> per-board shade.
    index = n.math("FLOOR", n.math("DIVIDE", sep[along], width))
    shade = n.new("ShaderNodeTexWhiteNoise", noise_dimensions="1D")
    n.link(index, shade.inputs["W"])
    base = n.mix(n.math("MULTIPLY", shade.outputs["Value"], 0.55), rgb(color),
                 [c * 1.35 for c in rgb(color)[:3]] + [1])
    base = n.mix(n.math("MULTIPLY", n.math("SUBTRACT", 1, shade.outputs["Value"]), 0.35), base,
                 [c * 0.7 for c in rgb(color)[:3]] + [1])
    # Grain: noise stretched along the grain, plus a fine ring pattern.
    stretch = {"X": (1.5, 45, 45), "Y": (45, 1.5, 45), "Z": (45, 45, 1.5)}[grain]
    streaks = n.noise(n.scaled(co, stretch), 1.0, 6.0, 0.6)
    base = n.mix(n.math("MULTIPLY", streaks, 0.45), base, [c * 0.62 for c in rgb(color)[:3]] + [1])
    # Seams between boards.
    bands = n.new("ShaderNodeTexWave", wave_type="BANDS", bands_direction=axis, wave_profile="SAW",
                  Scale=1.0 / width, Distortion=0.0)
    n.link(co, bands.inputs["Vector"])
    seam_mask = n.math("LESS_THAN", bands.outputs["Fac"], 0.03)
    base = n.mix(seam_mask, base, rgb(seam))
    # Crevice grime and worn edges.
    ao = n.new("ShaderNodeAmbientOcclusion", Distance=0.25, samples=8)
    n.link(base, ao.inputs["Color"])
    grime = n.math("SUBTRACT", 1, ao.outputs["AO"], clamp=True)
    base = n.mix(n.math("MULTIPLY", grime, 0.6, clamp=True), base, rgb(GRIME), "MIX")
    convex = n.new("ShaderNodeAmbientOcclusion", Distance=0.03, samples=6, inside=True)
    edge = n.math("MULTIPLY", n.math("SUBTRACT", 0.85, convex.outputs["AO"]), 4.0, clamp=True)
    scuff = n.noise(co, 9.0, 3.0)
    wear = n.math("MULTIPLY", edge, n.math("GREATER_THAN", scuff, 0.45), clamp=True)
    base = n.mix(n.math("MULTIPLY", wear, 0.55), base, [c * 1.7 for c in rgb(color)[:3]] + [1])
    rough = n.math("ADD", 0.62, n.math("MULTIPLY", streaks, 0.25))
    if waterline:
        z = sep[2]
        boot = n.math("LESS_THAN", z, WATERLINE + 0.06)
        drip = n.noise(n.scaled(co, (30, 30, 1.6)), 1.0, 5.0)
        band = n.math("SUBTRACT", 1, n.math("DIVIDE", n.math("SUBTRACT", z, WATERLINE + 0.04),
                                             n.math("ADD", 0.18, n.math("MULTIPLY", drip, 0.28))), clamp=True)
        band = n.math("MULTIPLY", band, n.math("GREATER_THAN", z, WATERLINE - 0.4))
        weed = n.mix(n.noise(co, 14.0), rgb(GRIME), rgb("#55603A"))
        base = n.mix(n.math("MULTIPLY", band, 0.9), base, weed)
        base = n.mix(boot, base, rgb(TAR))
        rough = n.math("SUBTRACT", rough, n.math("MULTIPLY", band, 0.2))
    color, rough = wet_finish(n, base, rough, wet)
    n.link(color, n.bsdf.inputs["Base Color"])
    n.link(rough, n.bsdf.inputs["Roughness"])
    bump = n.new("ShaderNodeBump", Strength=0.25, Distance=0.004)
    n.link(n.math("ADD", n.math("MULTIPLY", seam_mask, -1.0), n.math("MULTIPLY", streaks, 0.4)), bump.inputs["Height"])
    n.link(bump.outputs["Normal"], n.bsdf.inputs["Normal"])
    return mat


def metal(name, color, tarnish, wet=0.0, metallic=0.9, roughness=0.35):
    """Brass or iron with patchy tarnish (verdigris or rust) gathered in the crevices."""
    mat = material(name, color, roughness, metallic)
    n = Nodes(mat)
    co = n.coord()
    patches = n.noise(co, 6.0, 6.0)
    ao = n.new("ShaderNodeAmbientOcclusion", Distance=0.12, samples=8)
    crevice = n.math("SUBTRACT", 1, ao.outputs["AO"], clamp=True)
    amount = n.math("MULTIPLY", n.math("ADD", n.math("MULTIPLY", n.math("SUBTRACT", patches, 0.5), 2.2), crevice),
                    1.0, clamp=True)
    base = n.mix(amount, rgb(color), rgb(tarnish))
    rough = n.math("ADD", roughness, n.math("MULTIPLY", amount, 0.4))
    n.link(n.math("SUBTRACT", metallic, n.math("MULTIPLY", amount, metallic * 0.8)), n.bsdf.inputs["Metallic"])
    base, rough = wet_finish(n, base, rough, wet)
    n.link(base, n.bsdf.inputs["Base Color"])
    n.link(rough, n.bsdf.inputs["Roughness"])
    return mat


def paint(name, color, roughness=0.55, wet=0.0):
    """Painted or plain surface with soft dirt in the crevices."""
    mat = material(name, color, roughness)
    n = Nodes(mat)
    ao = n.new("ShaderNodeAmbientOcclusion", Distance=0.15, samples=8)
    base = n.mix(n.math("MULTIPLY", n.math("SUBTRACT", 1, ao.outputs["AO"], clamp=True), 0.5), rgb(color),
                 [c * 0.45 for c in rgb(color)[:3]] + [1])
    base, rough = wet_finish(n, base, roughness, wet)
    n.link(base, n.bsdf.inputs["Base Color"])
    if hasattr(rough, "links"):
        n.link(rough, n.bsdf.inputs["Roughness"])
    else:
        n.bsdf.inputs["Roughness"].default_value = rough
    return mat


def cloth(name, color, pattern=None, wet=0.0):
    """Fabric with a fine weave bump and sheen; pattern: None, ("dots", color, scale),
    ("ribs", color, scale) or ("stripes", color, scale)."""
    mat = material(name, color, 0.85)
    n = Nodes(mat)
    co = n.coord()
    base = rgb(color)
    if pattern:
        kind, other, scale = pattern
        if kind == "dots":
            vor = n.new("ShaderNodeTexVoronoi", Scale=scale, Randomness=0.0)
            n.link(co, vor.inputs["Vector"])
            base = n.mix(n.math("LESS_THAN", vor.outputs["Distance"], 0.22), base, rgb(other))
        else:
            wave = n.new("ShaderNodeTexWave", wave_type="BANDS", bands_direction="X" if kind == "ribs" else "Z",
                         wave_profile="SIN", Scale=scale)
            n.link(co, wave.inputs["Vector"])
            base = n.mix(n.math("MULTIPLY", wave.outputs["Fac"], 0.35 if kind == "ribs" else 1.0), base, rgb(other))
    weave = n.new("ShaderNodeTexWave", wave_type="BANDS", bands_direction="DIAGONAL", Scale=900.0)
    n.link(co, weave.inputs["Vector"])
    mottle = n.noise(co, 120.0, 2.0)
    base = n.mix(n.math("MULTIPLY", mottle, 0.25), base, [c * 0.75 for c in rgb(color)[:3]] + [1])
    n.bsdf.inputs["Sheen Weight"].default_value = 0.15
    base, rough = wet_finish(n, base, 0.85, wet * 0.6)
    n.link(base, n.bsdf.inputs["Base Color"])
    if hasattr(rough, "links"):
        n.link(rough, n.bsdf.inputs["Roughness"])
    else:
        n.bsdf.inputs["Roughness"].default_value = rough
    bump = n.new("ShaderNodeBump", Strength=0.3, Distance=0.0006)
    n.link(n.math("ADD", weave.outputs["Fac"], mottle), bump.inputs["Height"])
    n.link(bump.outputs["Normal"], n.bsdf.inputs["Normal"])
    return mat


def screen(name, eye_color, squint=1.0):
    """Jumper's display as a glossy black screen with two lit pill eyes, drawn in the
    display body's frame: no geometry. Returns the material and the eyelid value
    socket (1 open, 0 shut) to key blinks on; `squint` narrows the right eye."""
    mat = material(name, INK, 0.08)
    n = Nodes(mat)
    n.bsdf.inputs["Coat Weight"].default_value = 1.0
    n.bsdf.inputs["Coat Roughness"].default_value = 0.02
    co = n.xyz(n.coord())
    front = n.math("GREATER_THAN", n.xyz(n.coord("Normal"))[0], 0.5)
    lid = n.new("ShaderNodeValue")
    lid.name = "eyelid"
    lid.outputs[0].default_value = 1.0
    eyes = None
    for side, open_ in ((1, 1.0), (-1, squint)):
        dy = n.math("DIVIDE", n.math("SUBTRACT", co[1], side * 0.021), 0.0105)
        height = n.math("MULTIPLY", n.math("MULTIPLY", lid.outputs[0], open_), 0.0135)
        dz = n.math("DIVIDE", n.math("SUBTRACT", co[2], -0.0005), n.math("MAXIMUM", height, 0.0006))
        r = n.math("ADD", n.math("POWER", n.math("ABSOLUTE", dy), 2.4), n.math("POWER", n.math("ABSOLUTE", dz), 2.4))
        eye = n.math("SUBTRACT", 1, n.math("DIVIDE", n.math("SUBTRACT", r, 0.75), 0.25), clamp=True)
        eyes = eye if eyes is None else n.math("MAXIMUM", eyes, eye)
    mask = n.math("MULTIPLY", eyes, front)
    n.link(n.mix(mask, rgb(INK), rgb(eye_color)), n.bsdf.inputs["Base Color"])
    n.bsdf.inputs["Emission Color"].default_value = rgb(eye_color)
    n.link(n.math("MULTIPLY", mask, 9.0), n.bsdf.inputs["Emission Strength"])
    return mat, lid.outputs[0]


def glass(name, glow=FLAME):
    """Old window glass: clear, faintly green, a little uneven, catching the warm
    cabin light from behind."""
    mat = material(name, "#D8E4DA", 0.03)
    n = Nodes(mat)
    n.bsdf.inputs["Transmission Weight"].default_value = 1.0
    n.bsdf.inputs["IOR"].default_value = 1.5
    co = n.coord()
    bump = n.new("ShaderNodeBump", Strength=0.08)
    n.link(n.noise(co, 5.0, 2.0), bump.inputs["Height"])
    n.link(bump.outputs["Normal"], n.bsdf.inputs["Normal"])
    n.bsdf.inputs["Emission Color"].default_value = rgb(glow)
    n.link(n.math("MULTIPLY", n.noise(co, 2.0, 1.0), 0.6), n.bsdf.inputs["Emission Strength"])
    return mat


def assign(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)


# Geometry helpers ------------------------------------------------------------

def bounds(objs):
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    return (Vector([min(p[i] for p in pts) for i in range(3)]),
            Vector([max(p[i] for p in pts) for i in range(3)]))


def empty(name, parent=None, location=(0, 0, 0)):
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent, obj.location = parent, location
    return obj


def mesh_object(name, verts, faces, mat=None, parent=None, smooth=True):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    mesh.polygons.foreach_set("use_smooth", [smooth] * len(mesh.polygons))
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    if mat:
        mesh.materials.append(mat)
    obj.parent = parent
    return obj


def ellipsoid(name, loc, radii, mat, parent=None, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=40, ring_count=20, location=loc, rotation=rot)
    obj = bpy.context.active_object
    obj.name, obj.scale = name, radii
    bpy.ops.object.shade_smooth()
    assign(obj, mat)
    obj.parent = parent
    return obj


def cylinder(name, a, b, radius, mat, parent=None, vertices=24, radius2=None):
    """A cylinder (or a cone to radius2) from point a to point b."""
    a, b = Vector(a), Vector(b)
    if radius2 is None:
        bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=(b - a).length,
                                            location=(a + b) / 2)
    else:
        bpy.ops.mesh.primitive_cone_add(vertices=vertices, radius1=radius, radius2=radius2,
                                        depth=(b - a).length, location=(a + b) / 2)
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_euler = (b - a).to_track_quat("Z", "Y").to_euler()
    bpy.ops.object.shade_smooth()
    assign(obj, mat)
    obj.parent = parent
    return obj


def rope(name, points, radius, mat, parent=None):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions, curve.bevel_depth, curve.bevel_resolution = "3D", radius, 3
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for p, co in zip(spline.points, points):
        p.co = (*co, 1)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    obj.parent = parent
    return obj


def bake_keys(obj, path, frames, values, interpolation="LINEAR"):
    """Keys obj.<path> (all components) at `frames` from rows of `values`, fast."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    obj.keyframe_insert(path, frame=frames[0])
    action = obj.animation_data.action
    for index in range(values.shape[1]):
        fc = next(f for f in action.fcurves if f.data_path == path and f.array_index == index)
        fc.keyframe_points.clear()
        fc.keyframe_points.add(len(frames))
        co = np.column_stack([frames, values[:, index]]).ravel()
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation",
                                       [bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items[interpolation].value]
                                       * len(frames))
        fc.update()


# Boat ------------------------------------------------------------------------

def load_boat(wet=0.0):
    """Opens the plain Benchy at x110 and dresses its parts in plank wood, brass,
    iron and tar; adds glass to the windows and a warm light in the cabin."""
    bpy.ops.wm.open_mainfile(filepath=str(BOAT))
    _RIG.clear()                                  # the new file has no rig meshes yet
    boat = {o.name.split(".", 1)[1]: o for o in bpy.data.objects if o.name.startswith("Benchy.")}
    brass = metal("brass", BRASS, VERDIGRIS, wet)
    iron = metal("iron", IRON, RUST, wet, 0.7, 0.5)
    mats = {"hull": wood("hull", OAK, "Z", 0.22, "X", wet=wet, waterline=True),
            "gunwale": wood("gunwale", DARK_OAK, "Z", 0.12, "X", wet=wet),
            "deck_surface": wood("deck", PINE, "Y", 0.16, "X", wet=wet),
            "bridge_walls": wood("cabin", DARK_OAK, "Y", 0.14, "Z", wet=wet),
            "cargo_box": wood("hold", AGED_WOOD, "Z", 0.11, "X", wet=wet),
            "bridge_roof": paint("roof", TAR, 0.45, wet), "stern_name_plate": wood("name_plate", DARK_OAK, "Z", 0.5, "Y", wet=wet),
            "window": brass, "stern_window": brass, "doorframe_port": brass, "doorframe_starboard": brass,
            "wheel": brass, "hawsepipe_port": brass, "hawsepipe_starboard": brass,
            "chimney_body": iron, "chimney_top": iron, "fishing_rod_holder": iron}
    for part, obj in boat.items():
        assign(obj, mats[part])
        obj.data.polygons.foreach_set("use_smooth", [False] * len(obj.data.polygons))
    for part in ("window", "stern_window"):
        panes(boat[part])
    cabin = bpy.data.lights.new("cabin_glow", "POINT")
    cabin.energy, cabin.color, cabin.shadow_soft_size = 140, rgb(FLAME)[:3], 0.35
    light = bpy.data.objects.new("cabin_glow", cabin)
    bpy.context.scene.collection.objects.link(light)
    light.location = (0.3, 0.0, 3.2)
    return boat


def panes(frame, segments=48):
    """A flat pane in each opening of a window frame: per loose part of the frame,
    a disc in its best-fit plane whose outline runs a third of the way from the
    opening's edge into the frame, so the edge hides inside the frame."""
    bm = bmesh.new()
    bm.from_mesh(frame.data)
    bm.verts.ensure_lookup_table()
    seen, islands = set(), []
    for v in bm.verts:
        if v.index in seen:
            continue
        stack, island = [v], []
        seen.add(v.index)
        while stack:
            u = stack.pop()
            island.append(u.co.copy())
            for e in u.link_edges:
                w = e.other_vert(u)
                if w.index not in seen:
                    seen.add(w.index)
                    stack.append(w)
        islands.append(np.array(island))
    bm.free()
    mat = glass("window_glass")
    for k, pts in enumerate(islands):
        if len(pts) < 24:
            continue
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c)
        u, w = vt[0], vt[1]
        local = np.column_stack([(pts - c) @ u, (pts - c) @ w])
        ang = np.arctan2(local[:, 1], local[:, 0])
        rad = np.hypot(*local.T)
        ring = []
        for s in range(segments):
            a = -math.pi + 2 * math.pi * s / segments
            near = np.abs((ang - a + math.pi) % (2 * math.pi) - math.pi) < math.pi / segments * 1.5
            if not near.any():
                continue
            inner, outer = rad[near].min(), rad[near].max()
            r = inner + (outer - inner) * 0.33
            ring.append(c + (math.cos(a) * u + math.sin(a) * w) * r)
        if len(ring) < 8:
            continue
        pane = mesh_object(f"pane_{frame.name}_{k}", [Vector(p) for p in ring] + [Vector(c)],
                           [(i, (i + 1) % len(ring), len(ring)) for i in range(len(ring))], mat, smooth=False)
        pane.matrix_world = frame.matrix_world
        pane.visible_shadow = False


# Crew ------------------------------------------------------------------------

_RIG = {}


def rig():
    """Jumper's body meshes (imported once) and the export metadata."""
    if not _RIG:
        meta = json.loads((OUT / "jumper.json").read_text(encoding="utf-8"))
        before = set(bpy.data.objects)
        bpy.ops.wm.obj_import(filepath=str(OUT / "jumper.obj"), forward_axis="Y", up_axis="Z")
        meshes = {}
        for obj in set(bpy.data.objects) - before:
            body = obj.name.split(".")[0]
            mesh = obj.data
            mesh.materials.clear()
            mesh.materials.append(None)
            mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
            meshes[body] = mesh
            bpy.data.objects.remove(obj)
        _RIG.update(meshes=meshes, meta=meta)
    return _RIG["meshes"], _RIG["meta"]


def crab(name, track, location, heading, start=0, parent=None):
    """A Jumper keyed through `track` from frame `start`, in its crew colors and
    outfit; the angler holds the rod. Returns the root and its body objects."""
    meshes, meta = rig()
    colors = CREW[name]
    root = empty(name, parent, location)
    root.rotation_euler = (0, 0, heading)
    mats = {g: material(f"{name}_{g}", colors[g], 0.4 if g == "shell" else 0.55) for g in ("shell", "limbs", "detail")}
    for g in ("shell", "limbs"):
        n = Nodes(mats[g])
        n.bsdf.inputs["Coat Weight"].default_value = 0.3
    face, lid = screen(f"{name}_screen", colors["eyes"], squint=0.62 if name == "angler" else 1.0)
    mats["display"] = face
    frames = np.asarray(meta["tracks"][track])
    keys = start + np.arange(len(frames))
    bodies = {}
    for i, body in enumerate(meta["bodies"]):
        if body not in meshes:
            continue
        obj = bpy.data.objects.new(f"{name}.{body}", meshes[body])
        bpy.context.scene.collection.objects.link(obj)
        obj.parent = root
        obj.material_slots[0].link = "OBJECT"
        obj.material_slots[0].material = mats[meta["groups"][body]]
        obj.rotation_mode = "QUATERNION"
        obj.location = frames[0, i, :3]
        obj.rotation_quaternion = frames[0, i, 3:]
        if len(frames) > 1:
            bake_keys(obj, "location", keys, frames[:, i, :3])
            bake_keys(obj, "rotation_quaternion", keys, frames[:, i, 3:])
        bodies[body] = obj
    blinks(lid, start, len(frames), seed=3 if name == "angler" else 8)
    (angler_outfit if name == "angler" else sailor_outfit)(bodies, meta)
    if name == "angler":
        held_rod(bodies, meta)
    bpy.context.view_layer.update()
    return root, bodies


def blinks(lid, start, length, seed):
    """Eyelid keys: a quick blink every few seconds."""
    rng = np.random.default_rng(seed)
    t, keys = rng.uniform(0.6, 1.8), [(start, 1.0)]
    while t * FPS < length - 6:
        f = start + t * FPS
        keys += [(f - 1, 1.0), (f + 1, 0.0), (f + 3, 1.0)]
        t += rng.uniform(2.2, 4.0)
    node = lid.node
    for f, v in keys:
        node.outputs[0].default_value = v
        node.outputs[0].keyframe_insert("default_value", frame=f)


class Shell:
    """The shell's outer surface in its body frame, sampled on a fine grid by rays
    cast inward from all around (heading x elevation about a point under the
    shell). Garments are cells of this grid, so their edges follow clean lines and
    their offsets follow smooth normals. Rays that first hit the screen, sensors
    or the lower body mark cells no garment may use."""

    COLS, ROWS = 360, 300
    MARGIN = 0.003      # m kept bare around the screen and sensors
    PHI = (math.radians(-12), math.radians(89.5))

    def __init__(self, meshes, meta):
        rest = dict(zip(meta["bodies"], meta["tracks"]["stand"][0]))
        to_shell = self.frame(rest["upper_shell_link"]).inverted()
        verts, polys, owner = [], [], []
        for k, body in enumerate(("upper_shell_link", "display_module_link", "base_link",
                                  "tof_sensor_link", "camera_link")):
            m = to_shell @ self.frame(rest[body])
            base = len(verts)
            verts += [m @ v.co for v in meshes[body].vertices]
            polys += [[base + i for i in p.vertices] for p in meshes[body].polygons]
            owner += [k] * len(meshes[body].polygons)
        self.solid = BVHTree.FromPolygons(verts, polys)
        owner = np.array(owner)
        pick = lambda keep: BVHTree.FromPolygons(verts, [p for p, k in zip(polys, owner) if k in keep])
        self.skin = pick({0})
        self.face = pick({1, 3, 4})
        centre = Vector((0.0, 0.0, -0.03))
        heading = np.linspace(-math.pi, math.pi, self.COLS, endpoint=False)
        phi = np.linspace(*self.PHI, self.ROWS)
        P = np.zeros((self.ROWS, self.COLS, 3))
        kind = np.zeros((self.ROWS, self.COLS), int)       # 0 miss, 1 shell, 2 anything else
        for j, f in enumerate(phi):
            for i, h in enumerate(heading):
                d = Vector((math.cos(h) * math.cos(f), math.sin(h) * math.cos(f), math.sin(f)))
                loc, _, idx, _ = self.solid.ray_cast(centre + d * 0.4, -d, 0.45)
                if loc is not None:
                    P[j, i] = loc
                    kind[j, i] = 1 if owner[idx] == 0 else 2
        # Smooth outward normals from the grid itself.
        dh = np.roll(P, -1, 1) - np.roll(P, 1, 1)
        dv = np.zeros_like(P)
        dv[1:-1] = P[2:] - P[:-2]
        dv[0], dv[-1] = P[1] - P[0], P[-1] - P[-2]
        N = np.cross(dh, dv)
        N /= np.maximum(np.linalg.norm(N, axis=2, keepdims=True), 1e-12)
        N *= np.sign(np.sum(N * (P - np.array(centre)), axis=2, keepdims=True))
        # Arc length up each column from the shell's lower edge.
        step = np.zeros((self.ROWS, self.COLS))
        step[1:] = np.linalg.norm(P[1:] - P[:-1], axis=2)
        shell = kind == 1
        first = np.argmax(shell, axis=0)
        s = np.cumsum(step, axis=0)
        s -= s[first, np.arange(self.COLS)]
        # Keep a margin around the screen and the sensors.
        near = np.zeros_like(shell)
        for j, i in zip(*np.nonzero(shell)):
            near[j, i] = self.face.find_nearest(Vector(P[j, i]), self.MARGIN)[0] is not None
        self.P, self.N, self.kind, self.s, self.near, self.shell = P, N, kind, s, near, shell

    @staticmethod
    def frame(pose):
        return Matrix.Translation(pose[:3]) @ Quaternion(pose[3:]).to_matrix().to_4x4()

    def cells(self, select):
        """Mask of grid cells (by their first corner) whose four corners are on the
        shell, away from the screen, and whose centre passes
        select(|heading|, height above the edge, normal z, position)."""
        ok = self.shell & np.roll(self.shell, -1, 1)
        ok[:-1] &= ok[1:]
        ok[-1] = False
        ok[-2] &= ok[-3]
        up = lambda a: np.concatenate([a[1:], a[-1:]])
        c = lambda a: (a + np.roll(a, -1, 1) + up(a) + np.roll(up(a), -1, 1)) / 4
        p, nz, s = c(self.P), c(self.N[..., 2]), c(self.s)
        h = np.abs(np.arctan2(p[..., 1], p[..., 0]))
        near = self.near | np.roll(self.near, -1, 1) | up(self.near) | np.roll(up(self.near), -1, 1)
        return ok & ~near & select(h, s, nz, p)

    def surface(self, name, mask, offset, mat, parent):
        """The cells in `mask`, pushed out `offset` along the normals."""
        used, verts, faces = {}, [], []
        cols = self.P.shape[1]
        for j, i in zip(*np.nonzero(mask)):
            quad = []
            for jj, ii in ((j, i), (j, (i + 1) % cols), (j + 1, (i + 1) % cols), (j + 1, i)):
                if (jj, ii) not in used:
                    used[(jj, ii)] = len(verts)
                    verts.append(Vector(self.P[jj, ii] + self.N[jj, ii] * offset))
                quad.append(used[(jj, ii)])
            faces.append(quad)
        if mask[-2].all():                       # close the crown with a fan to its pole
            top = self.P.shape[0] - 1
            pole = len(verts)
            verts.append(Vector(self.P[top].mean(0) + np.array((0, 0, 1.0)) * offset))
            faces += [(used[(top, i)], used[(top, (i + 1) % cols)], pole) for i in range(cols)]
        obj = mesh_object(name, verts, faces, mat, parent)
        bm = bmesh.new()                          # the squeezed rows at the rim and pole
        bm.from_mesh(obj.data)                    # leave slivers; merge them away
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1.5e-4)
        bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=1.5e-4)
        bm.to_mesh(obj.data)
        bm.free()
        self.smooth_edges(obj, offset)
        return obj

    def smooth_edges(self, obj, offset, rounds=8):
        """Relaxes the stair-stepped outline of a cell region into a smooth curve,
        keeping it on the offset surface."""
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        edge = [v for v in bm.verts if v.is_boundary]
        for _ in range(rounds):
            moved = {}
            for v in edge:
                ring = [e.other_vert(v) for e in v.link_edges if e.is_boundary]
                if len(ring) != 2:
                    continue
                target = (v.co + ring[0].co + ring[1].co) / 3
                loc, nrm, _, _ = self.skin.find_nearest(target)
                if loc is not None:
                    nrm = nrm if nrm.dot(v.co - loc) >= 0 else -nrm
                    moved[v] = loc + nrm * offset
            for v, co in moved.items():
                v.co = co
        bm.to_mesh(obj.data)
        bm.free()

    def garment(self, name, select, offset, thickness, mat, parent, wrinkle=0.0):
        """Shell cells passing select, pushed out `offset` along the normals and
        given `thickness` outward with a rounded hem; `wrinkle` adds folds that
        only ever push outward."""
        obj = self.surface(name, self.cells(select), offset, mat, parent)
        if wrinkle:
            tex = bpy.data.textures.new(f"{name}_wrinkles", "CLOUDS")
            tex.noise_scale, tex.noise_depth = 0.01, 2
            disp = obj.modifiers.new("wrinkles", "DISPLACE")
            disp.texture, disp.strength, disp.mid_level, disp.texture_coords = tex, wrinkle, 0.0, "LOCAL"
        solid = obj.modifiers.new("cloth", "SOLIDIFY")
        solid.thickness, solid.offset, solid.use_rim = thickness, 1.0, True
        bevel = obj.modifiers.new("hem", "BEVEL")
        bevel.width, bevel.segments, bevel.limit_method = thickness * 0.45, 2, "ANGLE"
        return obj

    def point(self, heading, s):
        """Surface point and normal on the column nearest `heading`, `s` m above the edge."""
        i = int(round((heading + math.pi) / (2 * math.pi) * self.COLS)) % self.COLS
        col = np.nonzero(self.shell[:, i])[0]
        j = col[np.argmin(np.abs(self.s[col, i] - s))]
        return Vector(self.P[j, i]), Vector(self.N[j, i])

    def ring(self, name, s, out, radius, mat, parent):
        """A rope around the shell `s` m above its lower edge, `out` m clear of the
        surface plus its own radius, broken where it would pass the screen."""
        pts = []
        for i in range(0, self.COLS, 2):
            col = np.nonzero(self.shell[:, i])[0]
            if not len(col):
                continue
            j = col[np.argmin(np.abs(self.s[col, i] - s))]
            if self.near[j, i]:
                continue
            pts.append(Vector(self.P[j, i]) + Vector(self.N[j, i]) * (out + radius))
        if len(pts) == self.COLS // 2:
            pts.append(pts[0])
        return rope(name, pts, radius, mat, parent)


def angler_outfit(bodies, meta):
    """Buccaneer: scarlet polka-dot bandana knotted at the back with two tails, an
    open walnut coat over the sides, a gold sash and a leather belt with a brass
    buckle. The face stays a clean screen."""
    meshes, _ = rig()
    s = Shell(meshes, meta)
    shell = bodies["upper_shell_link"]
    bandana = cloth("bandana", SCARLET, ("dots", BONE, 140.0))
    layer2 = CLEAR + 0.0022 + 0.0008
    s.garment("bandana", lambda h, a, nz, p: (a > 0.031) & (p[..., 0] < 0.045), CLEAR, 0.0022, bandana, shell, 0.0007)
    s.garment("coat", lambda h, a, nz, p: (h > 0.95) & (a > 0.0072) & (a <= 0.029),
              CLEAR, 0.0024, cloth("coat", WALNUT), shell, 0.0005)
    s.garment("sash", lambda h, a, nz, p: a <= 0.0062,
              CLEAR, 0.0022, cloth("sash", GOLD, ("stripes", MUSTARD, 900.0)), shell, 0.0003)
    s.garment("belt", lambda h, a, nz, p: (a > 0.0012) & (a <= 0.005) & (h > 0.95),
              layer2, 0.0016, paint("belt", BLACK, 0.4), shell)
    s.garment("buckle", lambda h, a, nz, p: (a > 0.0004) & (a <= 0.0058) &
              (np.abs(np.arctan2(p[..., 1], p[..., 0]) - 1.6) < 0.075),
              layer2 + 0.0016 + 0.0006, 0.0014, metal("buckle", BRASS, VERDIGRIS), shell)
    # Knot and tails at the back of the bandana, clear of everything.
    p, n = s.point(math.pi, 0.036)
    knot_at = p + n * (CLEAR + 0.0022 + 0.0005 + 0.007)
    ellipsoid("bandana_knot", knot_at, (0.0085, 0.012, 0.007), bandana, shell, rot=n.to_track_quat("Z", "Y").to_euler())
    for side in (-1, 1):                       # tails streaming back, splayed
        ellipsoid("bandana_tail", knot_at + Vector((-0.021, side * 0.01, -0.002)), (0.016, 0.0055, 0.0022),
                  bandana, shell, rot=(side * 0.2, 0.3, side * 0.4))
    push_clear(shell, [o for o in shell.children if o.name.startswith(("bandana_knot", "bandana_tail"))],
               bodies, Vector((-1, 0, 0.3)).normalized())


def sailor_outfit(bodies, meta):
    """Pirate deckhand: loose linen shirt, open sea-teal vest, mustard knit cap with
    a turned-up brim, navy sash and a hemp rope belt. The face stays a clean screen."""
    meshes, _ = rig()
    s = Shell(meshes, meta)
    shell = bodies["upper_shell_link"]
    s.garment("shirt", lambda h, a, nz, p: a >= 0, CLEAR, 0.0016, cloth("linen", LINEN), shell, 0.0009)
    layer2 = CLEAR + 0.0016 + 0.0009 + 0.0004
    cap_mat = cloth("knit_cap", CAP, ("ribs", "#7A2E12", 260.0))
    cap = lambda a, p: (a > 0.034) & (p[..., 0] < 0.03)
    s.garment("knit_cap", lambda h, a, nz, p: cap(a, p), layer2, 0.0034, cap_mat, shell, 0.0005)
    s.garment("cap_brim", lambda h, a, nz, p: cap(a, p) & ((a < 0.039) | (p[..., 0] > 0.022)),
              layer2 + 0.0034 + 0.0005 + 0.0006, 0.0028, cap_mat, shell)
    s.garment("vest", lambda h, a, nz, p: (h > 0.9) & (a > 0.0068) & (a <= 0.032),
              layer2, 0.0022, cloth("vest", "#1F4A57", ("stripes", "#183A45", 500.0)), shell, 0.0004)
    s.garment("sash", lambda h, a, nz, p: (a > 0.0015) & (a <= 0.006),
              layer2, 0.0022, cloth("sash", NAVY, ("stripes", "#2E4066", 700.0)), shell, 0.0003)
    hemp = cloth("rope", HEMP, ("stripes", "#7E6438", 1600.0))
    s.ring("rope_belt", 0.003, layer2 + 0.0022 + 0.0003 + 0.0006, 0.0019, hemp, shell)


def push_clear(parent, objs, bodies, direction, step=0.0005, tries=40):
    """Moves `objs` along `direction` (parent frame) until none overlaps the crab."""
    for _ in range(tries):
        if not overlaps(objs, list(bodies.values()) + [o for o in parent.children if o not in objs]):
            return
        for o in objs:
            o.location += direction * step
        bpy.context.view_layer.update()
    print(f"warning: {[o.name for o in objs]} still overlap after {tries} steps")


def bvh(obj):
    if obj.type not in ("MESH", "CURVE"):
        return None
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    mesh = ev.to_mesh()
    verts = [obj.matrix_world @ v.co for v in mesh.vertices]
    tree = BVHTree.FromPolygons(verts, [p.vertices[:] for p in mesh.polygons]) if len(mesh.polygons) else None
    ev.to_mesh_clear()
    return tree


def overlaps(a_objs, b_objs, trees=None):
    """(a, b) name pairs whose meshes intersect; `trees` caches BVHs by object."""
    trees = {} if trees is None else trees
    tree = lambda o: trees[o] if o in trees else trees.setdefault(o, bvh(o))
    hits = []
    for a in a_objs:
        ta = tree(a)
        if ta is None:
            continue
        for b in b_objs:
            tb = tree(b)
            if tb is not None and b is not a and ta.overlap(tb):
                hits.append((a.name, b.name))
    return hits


def check_clearance(groups, frames, boat=()):
    """Overlap report: every pair of distinct groups, plus each group against the
    boat (which must not move), at the given frames. groups: {label: [objects]}."""
    report = {}
    scene = bpy.context.scene
    labels = list(groups)
    still = {}
    for f in frames:
        scene.frame_set(int(f))
        trees = dict(still)
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                for pair in overlaps(groups[a], groups[b], trees):
                    report.setdefault(f"{a} x {b}", set()).add(pair)
            if boat:
                for pair in overlaps(groups[a], boat, trees):
                    report.setdefault(f"{a} x boat", set()).add(pair)
        still.update({o: trees[o] for o in boat if o in trees})
    return report


def captain_rig(penguin):
    """Pivots on the captain: a neck for the head and hat, shoulders for the flippers."""
    neck = empty("captain_neck", penguin, (0.01, 0, 0.47))
    bpy.context.view_layer.update()
    shoulders = []
    for part in [c for c in penguin.children if c is not neck]:
        shoulder = None
        if part.name.startswith("flipper"):
            side = 1 if part.location.y > 0 else -1
            shoulder = empty("captain_shoulder", penguin, (-0.01, side * 0.17, 0.43))
            shoulders.append(shoulder)
            bpy.context.view_layer.update()
        pivot = shoulder or (neck if part.location.z >= 0.45 else None)
        if pivot:
            world = part.matrix_world.copy()
            part.parent = pivot
            part.matrix_world = world
    for pivot in [neck] + shoulders:
        pivot.rotation_mode = "XYZ"
    return neck, shoulders


def held_rod(bodies, meta):
    """Rod, cork grip and reel as one prop parented to the left-front palm: the grip
    sits between the closed claw's grip inserts, so the rod moves with the claw.
    The grip radius is the largest that clears the claw."""
    rod = meta["rod"]
    palm = bodies[rod["body"]]
    grip, axis, closing = Vector(rod["grip"]), Vector(rod["axis"]), Vector(rod["closing"])
    side = closing.cross(axis).normalized()
    claw = [o for b, o in bodies.items() if b.startswith("LF_") and b != "LF_shoulder_link"]
    wood_mat, cork, brass = paint("rod", DARK_OAK, 0.4), cloth("cork", HEMP), metal("reel", BRASS, VERDIGRIS)
    parts = []
    radius = 0.0068
    while radius > 0.0025:
        handle = cylinder("rod_grip", grip - axis * 0.05, grip + axis * 0.05, radius, cork, palm, 20)
        bpy.context.view_layer.update()
        if not overlaps([handle], claw):
            break
        bpy.data.objects.remove(handle)
        radius -= 0.0004
    parts.append(handle)
    parts.append(cylinder("rod_butt", grip - axis * 0.085, grip - axis * 0.05, radius * 0.85, wood_mat, palm, 16))
    parts.append(cylinder("rod_blank", grip + axis * 0.05, grip + axis * (ROD - 0.13), radius * 0.6, wood_mat, palm,
                          12, radius2=0.0012))
    reel_at = grip - axis * 0.07 - side * 0.019
    parts.append(cylinder("reel_foot", grip - axis * 0.07 - side * radius * 0.5, reel_at, 0.003, brass, palm, 8))
    reel = cylinder("reel", reel_at - closing * 0.009, reel_at + closing * 0.009, 0.014, brass, palm, 28)
    parts.append(reel)
    for k in (0.3, 0.55, 0.8):                                  # line guides
        at = grip + axis * (ROD - 0.13) * k
        parts.append(cylinder("guide", at, at - side * 0.008, 0.0012, brass, palm, 6))
    bpy.context.view_layer.update()
    hits = overlaps(parts, claw + [bodies[b] for b in bodies if not b.startswith("LF_")])
    for name, other in hits:
        print(f"rod part {name} touches {other}")
    palm["rod_tip"] = list(grip + axis * (ROD - 0.13))
    palm["rod_radius"] = radius
    print(f"rod grip radius {radius * 1000:.1f} mm, claw gap {rod['gap'] * 1000:.1f} mm")


def rod_tip(bodies):
    palm = bodies["LF_palm_link"]
    return palm.matrix_world @ Vector(palm["rod_tip"])


STERN_GUIDE = (-3.40, 1.45)     # m in the ship's frame: 14 cm past the transom, 27 cm over the rail


def fishing_line(bodies, water_z, frames, pull=None):
    """The line from the rod tip out over the stern to the water, keyed per frame.
    Slack, it runs through a guide point fixed to the boat past the transom and
    above the rail and hangs from there to the water; pull(frame) in 0..1 draws
    it taut into a straight run from the tip to the water further out."""
    scene = bpy.context.scene
    curve = bpy.data.curves.new("line", "CURVE")
    curve.dimensions, curve.bevel_depth = "3D", 0.0012
    spline = curve.splines.new("POLY")
    spline.points.add(4)
    line = bpy.data.objects.new("line", curve)
    scene.collection.objects.link(line)
    curve.materials.append(material("line", BONE, 0.4))
    ship = bodies["LF_palm_link"]
    while ship.parent:
        ship = ship.parent
    for f in frames:
        scene.frame_set(int(f))
        tip = rod_tip(bodies)
        to_ship = ship.matrix_world.inverted()
        local = to_ship @ tip
        guide = ship.matrix_world @ Vector((STERN_GUIDE[0], local.y, STERN_GUIDE[1]))
        taut = pull(f) if pull else 0.0
        out = (ship.matrix_world.to_3x3() @ Vector((-1.0, 0.0, 0.0))).normalized()
        end = guide + out * (0.15 + 0.5 * taut)
        end.z = water_z(end.x, end.y, f) if callable(water_z) else water_z
        slack = [tip, tip.lerp(guide, 0.5) - Vector((0, 0, 0.03)), guide, guide.lerp(end, 0.5) + out * 0.04, end]
        straight = [tip.lerp(end, k / 4) for k in range(5)]
        for sp, a_, b_ in zip(spline.points, slack, straight):
            sp.co = (*a_.lerp(b_, taut), 1)
            sp.keyframe_insert("co", frame=f)
    return line


def feathers(name, color, translucent=0.0):
    """Plumage: soft sheen, a fine streaked feather bump and roughness that varies
    across the body, so it reads as feathers rather than plastic; `translucent`
    adds a little subsurface (the belly)."""
    mat = material(name, color, 0.55)
    n = Nodes(mat)
    co = n.coord()
    n.link(n.math("ADD", 0.4, n.math("MULTIPLY", n.noise(co, 9.0, 3.0), 0.32)), n.bsdf.inputs["Roughness"])
    n.bsdf.inputs["Sheen Weight"].default_value = 0.55
    n.bsdf.inputs["Sheen Roughness"].default_value = 0.35
    if translucent:
        n.bsdf.inputs["Subsurface Weight"].default_value = translucent
        n.bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.7, 0.5)
        n.bsdf.inputs["Subsurface Scale"].default_value = 0.01
    streaks = n.noise(n.scaled(co, (1.0, 1.0, 3.5)), 70.0, 4.0, 0.6)
    bump = n.new("ShaderNodeBump", Strength=0.12, Distance=0.002)
    n.link(streaks, bump.inputs["Height"])
    n.link(bump.outputs["Normal"], n.bsdf.inputs["Normal"])
    return mat


def glossy(name, color, roughness):
    """Beak and feet: a glossy clear coat over the colour."""
    mat = material(name, color, roughness)
    Nodes(mat).bsdf.inputs["Coat Weight"].default_value = 0.45
    return mat


def captain(location, heading, parent=None):
    """The penguin captain, an original design, 0.75 m tall: stout body, cream belly,
    orange beak and feet, admiral-navy coat with brass buttons, black tricorn with a
    bone plume. Built from primitives so it resembles no existing penguin character."""
    root = empty("captain", parent)
    m = lambda name, color, r=0.55, metal_=0.0: material(name, color, r, metal_)
    ink, cream = feathers("plumage", INK), feathers("belly", CREAM, 0.12)
    ellipsoid("body", (0, 0, 0.27), (0.17, 0.15, 0.25), ink, root)
    ellipsoid("neck", (0.0, 0, 0.44), (0.12, 0.115, 0.1), ink, root)          # head flows into body
    ellipsoid("belly", (0.07, 0, 0.25), (0.11, 0.115, 0.2), cream, root)
    ellipsoid("coat", (-0.01, 0, 0.21), (0.175, 0.16, 0.2), cloth("navy", NAVY), root)
    # Brass buttons down the coat's front edges, on the navy beside the belly,
    # never on the feathers.
    for z in (0.16, 0.23, 0.30):
        for y in (-0.135, 0.135):
            x = 0.175 * math.sqrt(max(0.0, 1 - (y / 0.16) ** 2 - ((z - 0.21) / 0.2) ** 2)) - 0.01 + 0.006
            ellipsoid("button", (x, y, z), (0.011, 0.007, 0.011), m("brass", BRASS, 0.3, 0.9), root,
                      rot=(0, 0, math.copysign(0.7, y)))
    ellipsoid("head", (0.01, 0, 0.55), (0.11, 0.105, 0.105), ink, root)
    ellipsoid("face", (0.075, 0, 0.545), (0.055, 0.075, 0.065), cream, root)
    for y in (-0.035, 0.035):
        ellipsoid("eye", (0.115, y, 0.565), (0.013, 0.013, 0.016), glossy("eye", INK, 0.04), root)
    bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.028, depth=0.08,
                                    location=(0.15, 0, 0.53), rotation=(0, math.pi / 2, 0))
    beak = bpy.context.active_object
    bpy.ops.object.shade_smooth()
    assign(beak, glossy("beak", ORANGE, 0.3))
    beak.parent = root
    for y in (-0.06, 0.06):
        ellipsoid("foot", (0.05, y, 0.015), (0.07, 0.04, 0.015), glossy("feet", ORANGE, 0.35), root)
    for y in (-1, 1):
        ellipsoid("flipper", (-0.01, y * 0.17, 0.3), (0.045, 0.02, 0.15), cloth("navy", NAVY), root,
                  rot=(y * 0.25, 0, 0))
    bpy.ops.mesh.primitive_cylinder_add(vertices=3, radius=0.17, depth=0.03, location=(0, 0, 0.64))
    brim = bpy.context.active_object
    brim.rotation_euler = (0, 0, math.pi)
    brim.modifiers.new("bevel", "BEVEL").segments = 4
    brim.modifiers["bevel"].width = 0.03
    felt = cloth("felt", PITCH)
    assign(brim, felt)
    brim.parent = root
    ellipsoid("crown", (0, 0, 0.67), (0.085, 0.085, 0.06), felt, root)
    ellipsoid("plume", (-0.06, 0.05, 0.72), (0.03, 0.025, 0.08), cloth("plume", BONE), root,
              rot=(0.3, -0.5, 0))
    cylinder("cutlass", (0.02, -0.17, 0.2), (0.12, -0.2, -0.0), 0.008, m("steel", "#B7BCC2", 0.25, 1.0), root, 8)
    bpy.context.view_layer.update()
    top = max((o.matrix_world @ Vector(c)).z for o in root.children for c in o.bound_box)
    root.scale = (CAPTAIN / top,) * 3
    root.location = location
    root.rotation_euler = (0, 0, heading)
    bpy.context.view_layer.update()
    return root


def build(scene_name, start=0, wet=0.0, ship=None):
    """The boat, the crew at their stations playing their `scene_name` tracks
    ("storm", "cove", or a still pose "stand") from frame `start`, and the captain
    in the starboard door. Everything hangs under one `ship` empty."""
    boat = load_boat(wet)
    ship = ship or empty("ship")
    for obj in list(bpy.context.scene.objects):
        if obj.parent is None and obj is not ship:
            obj.parent = ship
    angler_track = f"angler_{scene_name}" if scene_name != "stand" else "carry"
    sailor_track = f"sailor_{scene_name}" if scene_name != "stand" else "stand"
    angler, angler_bodies = crab("angler", angler_track, (*ANGLER_AT, DECK + 0.001), ANGLER_HEADING, start, ship)
    sailor, sailor_bodies = crab("sailor", sailor_track, (*SAILOR_AT, DECK + 0.001), SAILOR_HEADING, start, ship)
    lo, hi = bounds([boat["doorframe_starboard"]])
    penguin = captain(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, DECK + 0.002), -math.pi / 2, ship)
    return {"boat": boat, "ship": ship, "angler": angler, "sailor": sailor, "captain": penguin,
            "angler_bodies": angler_bodies, "sailor_bodies": sailor_bodies}


def crew_groups(s):
    """Overlap groups for check_clearance: each crab's robot, its outfit and its
    props separately, and the captain."""
    groups = {}
    for name in ("angler", "sailor"):
        bodies = s[f"{name}_bodies"]
        groups[f"{name} robot"] = list(bodies.values())
        shell = bodies["upper_shell_link"]
        groups[f"{name} outfit"] = [o for o in shell.children if o.type in ("MESH", "CURVE")]
        if name == "angler":
            groups["rod"] = [o for o in bodies["LF_palm_link"].children if o.type == "MESH"]
    groups["captain"] = [o for o in s["captain"].children if o.type == "MESH"]
    return groups


# Shots -----------------------------------------------------------------------

def look(location, target, roll=0.0, prev=None):
    q = (Vector(target) - Vector(location)).to_track_quat("-Z", "Y")
    if roll:
        q = q @ Quaternion((0, 0, 1), roll)
    return q.to_euler("XYZ", prev) if prev is not None else q.to_euler("XYZ")


def shot(name, keys, parent=None, handheld=0.0, focus=None, fstop=4.0, seed=0):
    """A camera keyed through `keys`: (film seconds, location, target, lens[, roll]),
    in the parent's frame. handheld adds operator shake (radians); focus is an
    object for depth of field."""
    data = bpy.data.cameras.new(name)
    data.sensor_width = 36
    cam = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(cam)
    cam.parent = parent
    prev = None
    for key in keys:
        t, loc, target, lens = key[:4]
        roll = key[4] if len(key) > 4 else 0.0
        f = round(t * FPS)
        cam.location = loc
        prev = look(loc, target, roll, prev)
        cam.rotation_euler = prev
        data.lens = lens
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)
        data.keyframe_insert("lens", frame=f)
    if handheld:
        for fc in cam.animation_data.action.fcurves:
            mod = fc.modifiers.new("NOISE")
            if fc.data_path == "rotation_euler":
                mod.strength, mod.scale = handheld * (0.5 if fc.array_index == 1 else 1.0), 9.0
            else:
                mod.strength, mod.scale = handheld * 0.25, 14.0
            mod.phase = seed * 7.3 + fc.array_index * 13.1
    if focus is not None:
        data.dof.use_dof, data.dof.focus_object, data.dof.aperture_fstop = True, focus, fstop
    return cam


def cut(cam, t):
    """Switches to `cam` at film second t."""
    marker = bpy.context.scene.timeline_markers.new(cam.name, frame=round(t * FPS))
    marker.camera = cam


def disable_stamps(render):
    """Disable visible stamps and embedded render metadata."""
    render.use_stamp = False
    for attr in dir(render):
        if attr.startswith("use_stamp_"):
            try:
                setattr(render, attr, False)
            except (AttributeError, TypeError):
                pass


def setup_render(width, height, samples, transparent=False, blur=True):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("HIP", "OPTIX", "CUDA", "ONEAPI"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        if any(d.type == backend for d in prefs.devices):
            for d in prefs.devices:
                d.use = d.type == backend
            scene.cycles.device = "GPU"
            break
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.denoising_use_gpu = scene.cycles.device == "GPU"   # same image, denoised on the card
    scene.cycles.use_adaptive_sampling = True
    scene.render.fps = FPS
    scene.render.film_transparent = transparent
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.use_motion_blur = blur
    scene.render.use_persistent_data = True       # keep the scene on the device between frames
    scene.render.motion_blur_shutter = 0.5
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    disable_stamps(scene.render)
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"


def render(path):
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    print(f"rendered {path.relative_to(ROOT)}")


def cove_clock(f):
    """The cove's frame (fractional inside the ramp) that film frame `f` shows; the
    storm's frames are their own. Takes a number or an array."""
    f = np.asarray(f, dtype=float)
    k = np.clip(f - RAMP[0] - PASSAGE, 0, None)
    n = RAMP[1] - RAMP[0] - PASSAGE
    ramped = RAMP[0] + k + PASSAGE * (1 - np.cos(np.pi * k / n)) / 2      # speed 1 at both ends
    c = np.where(f < CUT, f, np.where(f < CUT + PASSAGE, CUT, np.where(f < RAMP[0] + PASSAGE, f - PASSAGE,
                                                                       np.where(f < RAMP[1], ramped, f))))
    return c if c.ndim else float(c)


def save_motion(name, frames, roll, pitch, heave):
    """The boat's motion, per frame on the scene's clock, for the soundtrack's creaks
    and slaps (render/out/motion-<name>.json)."""
    OUT.mkdir(parents=True, exist_ok=True)
    data = {"frames": [int(f) for f in frames], "roll": np.round(roll, 5).tolist(),
            "pitch": np.round(pitch, 5).tolist(), "heave": np.round(heave, 5).tolist()}
    (OUT / f"motion-{name}.json").write_text(json.dumps(data), encoding="utf-8")


def eyes(name, frames, scale):
    """Re-keys a crab's eyelid (its screen's `eyelid` value) on `frames` as its blinks
    times `scale` (an array, 1 as is, above 1 wide-eyed, below 1 squinting)."""
    tree = bpy.data.materials[f"{name}_screen"].node_tree
    value = tree.nodes["eyelid"].outputs[0]
    fc = [c for c in tree.animation_data.action.fcurves if "eyelid" in c.data_path][0]
    blink = [fc.evaluate(f) for f in frames]
    for k in reversed(range(len(fc.keyframe_points))):
        if frames[0] <= fc.keyframe_points[k].co.x <= frames[-1]:
            fc.keyframe_points.remove(fc.keyframe_points[k])
    for f, b, s in zip(frames, blink, scale):
        value.default_value = b * s
        value.keyframe_insert("default_value", frame=f)


def active_camera(frame):
    markers = sorted((m for m in bpy.context.scene.timeline_markers if m.camera), key=lambda m: m.frame)
    cam = None
    for m in markers:
        if m.frame <= frame:
            cam = m.camera
    return cam


def run(stills, first, last, prefix):
    """Command line: `--stills` renders the named review stills to review/;
    `--frames a b [c d ...]` renders film frames a..b (and c..d) to render/out/frames/,
    each at the scene time cove_clock() gives it; `--samples n` and `--percent p` override quality."""
    a = args()
    scene = bpy.context.scene
    if "--samples" in a:
        scene.cycles.samples = int(a[a.index("--samples") + 1])
    if "--percent" in a:
        scene.render.resolution_percentage = int(a[a.index("--percent") + 1])
    if "--frames" in a:
        i = a.index("--frames") + 1
        bounds = []
        while i < len(a) and a[i].lstrip("-").isdigit():
            bounds.append(int(a[i]))
            i += 1
        (OUT / "frames").mkdir(parents=True, exist_ok=True)
        frames = [f for lo, hi in zip(bounds[::2], bounds[1::2]) for f in range(max(lo, first), min(hi, last) + 1)]
        for f in frames:
            c = cove_clock(f)
            scene.frame_set(int(c), subframe=c - int(c))
            scene.camera = active_camera(c)
            render(OUT / "frames" / f"{f:04d}.png")
        return
    REVIEW.mkdir(exist_ok=True)
    only = a[a.index("--only") + 1].split(",") if "--only" in a else None
    for label, t in stills:
        if only and label not in only:
            continue
        f = round(t * FPS)
        scene.frame_set(f)
        scene.camera = active_camera(f)
        render(REVIEW / f"{prefix}-{label}.png")
