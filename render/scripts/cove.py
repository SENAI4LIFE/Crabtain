"""The hidden bay at sunset, film seconds 10.5 to 25: the boat at anchor in a
sheltered bay under rocky headlands, palms and torches, its mouth open to the
sea to the north-east, a low amber sun.

Run from the repository root, after assets/boat/import_benchy.py and
render/scripts/export_jumper.py:
    blender --background --factory-startup --python-exit-code 1 --python render/scripts/cove.py [-- options]

Options as in storm.py (scene.run).

Shots:
4. 10.5-19.0 s orbit: the whip from the storm settles out of the sky onto the
   boat, then a slow orbit closes from wide to a medium on the crew; the sailor
   waves, the angler salutes with the rod still in its claw.
5. 19.0-21.0 s close two-shot: the angler's face, the sailor waving behind.
6. 21.0-25.0 s the catch: a chest drops onto the deck between the crew and its
   lid creaks open on the storm inside; the camera dives into it until the
   storm fills the frame, landing on the frame before the film's first, so the
   film loops. The sailor ducks, hops at the slam and bounces for joy; the captain
   hurries aft to see it. Needs storm.py --preroll first.
Colors: amber to ember sky, bay water, sand, grey rock with warm strata, palms
and torches.
"""
import math
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector, noise

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene  # noqa: E402
from scene import DECK, FPS, WATERLINE, Nodes, material, rgb  # noqa: E402

AMBER, EMBER, DUSK, LAGOON, SHALLOWS = "#F2A65A", "#C8553D", "#4B3B5C", "#1F9DB0", "#5FD3C4"
DEEP_SEA = "#0E4C77"
SAND, WET_SAND, CLIFFS, OCHRE, BASALT, RED_ROCK = "#ECDDBC", "#A88A60", "#6E5A4B", "#9A7550", "#423A36", "#8A5040"
LIMESTONE, GREY_ROCK = "#CDBFA6", "#7B7570"
PALMS, MOSS, TORCHES, BARK = "#4F6B32", "#5D6B34", "#FFD27A", "#5A4636"
SEED = 11
SUN_ELEVATION, SUN_HEADING = math.radians(4.0), math.radians(205)
COVE = Vector((-4.0, 26.0))          # centre of the beach arc, behind the boat
MOUTH = math.radians(55)             # the bay opens to the sea this way from COVE
ANCHORAGE = Vector((-0.3, 0.0))      # under the boat's midships
FIRST, LAST = round(10.5 * FPS), round(25 * FPS) - 1
CHEST_AT = (-2.455, 0.02)          # m, on the aft deck between the crew, clear of the rod holder
CHEST = (0.19, 0.30, 0.13)         # m: depth (x), width (y), body height
PORTAL = 0.27                      # m wide (y), 16:9: the storm shows here
LANDS, OPENS, DIVES = 21.0, 21.8, 23.0       # s
PREROLL = 48                       # frames from storm.py --preroll, ending the frame before 0
STILLS = [("s4-whip", 10.62), ("s4-orbit", 13.0), ("s4-dolphins", 13.95), ("s4-turtle", 14.9), ("s4-life", 16.8), ("s4-medium", 17.4), ("s5-close", 20.0), ("s6-drop", 21.08), ("s6-walk", 21.5), ("s6-gasp", 22.3), ("s6-catch", 22.6), ("s6-peek", 23.4), ("s6-dive", 24.4), ("s6-loop", 24.92)]


def sky():
    """Physical sunset sky, graded from amber at the horizon through ember to dusk
    violet overhead, with a few lit cloud streaks."""
    world = bpy.data.worlds.new("sunset")
    world.use_nodes = True
    n = Nodes(world)
    tree = world.node_tree
    vec = n.new("ShaderNodeNewGeometry").outputs["Incoming"]
    view = n.new("ShaderNodeVectorMath", operation="SCALE")
    n.link(vec, view.inputs[0])
    view.inputs["Scale"].default_value = -1.0
    up = n.xyz(view.outputs[0])[2]
    t = n.math("POWER", n.math("MAXIMUM", up, 0.0), 0.5)
    grade = n.ramp(t, [(0.0, rgb("#FFC07A")), (0.12, rgb(AMBER)), (0.3, rgb(EMBER)), (0.55, rgb("#8A4E6E")),
                       (1.0, rgb("#2E3462"))])
    # A hot glow around the low sun, wide and soft.
    to_sun = Vector((math.cos(SUN_HEADING) * math.cos(SUN_ELEVATION), math.sin(SUN_HEADING) * math.cos(SUN_ELEVATION),
                     math.sin(SUN_ELEVATION)))
    dot = n.new("ShaderNodeVectorMath", operation="DOT_PRODUCT")
    n.link(view.outputs[0], dot.inputs[0])
    dot.inputs[1].default_value = to_sun
    glow = n.math("POWER", n.math("MAXIMUM", dot.outputs["Value"], 0.0), 12.0)
    graded = n.mix(n.math("MULTIPLY", glow, 0.9), grade, [c * 3.0 for c in rgb("#FFD08A")[:3]] + [1], "ADD")
    streaks = n.noise(n.scaled(vec, (1.0, 1.0, 9.0)), 2.5, 6.0, 0.6)
    near_horizon = n.math("SUBTRACT", 1, n.math("MULTIPLY", n.math("ABSOLUTE", n.math("SUBTRACT", up, 0.12)), 6.0),
                          clamp=True)
    band = n.math("MULTIPLY", near_horizon, n.math("MULTIPLY", n.math("SUBTRACT", streaks, 0.5), 3.0, clamp=True))
    lit = n.mix(n.math("MULTIPLY", band, 0.7), graded, [c * 0.8 for c in rgb("#FFB070")[:3]] + [1])
    n.link(lit, tree.nodes["Background"].inputs["Color"])
    tree.nodes["Background"].inputs["Strength"].default_value = 0.9
    bpy.context.scene.world = world
    data = bpy.data.lights.new("sun", "SUN")
    data.energy, data.angle, data.color = 3.4, math.radians(1.5), rgb("#FFB070")[:3]
    sun = bpy.data.objects.new("sun", data)
    bpy.context.scene.collection.objects.link(sun)
    sun.rotation_euler = (math.pi / 2 - SUN_ELEVATION - math.radians(6), 0, SUN_HEADING + math.pi / 2)


def shore_height(x, y):
    """Bay floor, a broad crescent beach around COVE with a dry sand terrace, then
    rounded green hills set back from the shore. To the south the bay reaches out
    to where the boat lies at anchor; toward MOUTH the beach and hills give way to
    a channel that stays under water out to the open sea."""
    d = (Vector((x, y)) - COVE).length
    ang = math.atan2(y - COVE.y, x - COVE.x)
    south = max(0.0, -math.sin(ang)) ** 3
    mouth = max(0.0, math.cos(ang - MOUTH)) ** 6
    opening = max(south, mouth)
    r = 22.0 + 18.0 * opening
    t = (d - (r - 18.0)) / 18.0
    beach = min(max(t, 0.0), 1.0)
    z = WATERLINE - 1.2 + 2.0 * beach ** 0.7 * (1 - mouth) - 1.5 * mouth * beach
    z += 0.5 * min(max((d - r) / 8.0, 0.0), 1.0) * (1 - mouth)       # dry sand terrace
    cliff = max(0.0, (d - r - 8.0) / 14.0)                           # hills set back, gentle
    z += 13.0 * (1 - math.exp(-cliff * 1.1)) * (1 - opening)
    z += 0.6 * (1 - mouth) * noise.noise(Vector((x * 0.08, y * 0.08, 0.3))) \
        + 2.2 * min(cliff, 1.0) * noise.noise(Vector((x * 0.15, y * 0.15, 1.7))) \
        + 3.0 * min(cliff, 1.0) * (1 - opening) * noise.noise(Vector((x * 0.05, y * 0.05, 4.1)))
    # The anchorage: the boat rides over a sandy hollow at least 1.5 m deep, eased
    # back into the bay floor so the nearest beach is some 10 m off.
    s = min(max(((Vector((x, y)) - ANCHORAGE).length - 6.0) / 6.0, 0.0), 1.0)
    return min(z, WATERLINE - 1.5 + (z - WATERLINE + 1.5) * s * s * (3 - 2 * s))


def lagoon(size=90, steps=180):
    """The water as a grid whose vertex color is the depth to the floor below, so
    the shallows glow turquoise over the sand and the deep water goes dark. A low
    swell rolls in from the mouth and dies away over the shallows, wind ripples
    drift across it and the foam at the shore churns."""
    verts, faces, depth = [], [], []
    for j in range(steps + 1):
        for i in range(steps + 1):
            x, y = -size / 2 + size * i / steps, -size / 2 + size * j / steps + 10
            verts.append((x, y, WATERLINE))
            depth.append(min(max((WATERLINE - shore_height(x, y)) / 2.0, 0.0), 1.0))
    for j in range(steps):
        for i in range(steps):
            a = j * (steps + 1) + i
            faces.append((a, a + 1, a + steps + 2, a + steps + 1))
    obj = scene.mesh_object("lagoon", verts, faces)
    attr = obj.data.attributes.new("depth", "FLOAT", "POINT")
    attr.data.foreach_set("value", depth)
    # Swell: a slow displacement drifting in from the mouth, weighted by depth.
    group = obj.vertex_groups.new(name="swell")
    for k, (w, v) in enumerate(zip(depth, verts)):
        rim = min(size / 2 - abs(v[0]), size / 2 - abs(v[1] - 10)) / 8.0   # flat at the grid's edge, as the open water
        group.add([k], w * min(max(rim, 0.0), 1.0), "REPLACE")
    drift = scene.empty("swell_drift")
    first, last = round(10.5 * FPS), round(25.0 * FPS)
    inward = Vector((-math.cos(MOUTH), -math.sin(MOUTH), 0)) * 1.1 * (last - first) / FPS   # 1.1 m/s
    for f, at in ((first, Vector()), (last, inward)):
        drift.location = at
        drift.keyframe_insert("location", frame=f)
    swell = bpy.data.textures.new("swell", "CLOUDS")
    swell.noise_scale, swell.noise_depth = 6.0, 1
    disp = obj.modifiers.new("swell", "DISPLACE")
    disp.texture, disp.texture_coords, disp.texture_coords_object = swell, "OBJECT", drift
    disp.strength, disp.mid_level, disp.vertex_group = 0.24, 0.5, "swell"
    bpy.ops.mesh.primitive_plane_add(size=800, location=(0, 0, WATERLINE - 0.002))
    far = bpy.context.active_object
    far.name = "open_water"
    for o, deep_only in ((obj, False), (far, True)):
        mat = material(f"lagoon_{o.name}", LAGOON, 0.03)
        n = Nodes(mat)
        slicks = n.noise(n.scaled(n.coord(), (0.14, 0.14, 0.14)), 1.0, 2.0)          # glassy wind slicks
        n.link(n.math("ADD", 0.02, n.math("MULTIPLY", slicks, 0.05)), n.bsdf.inputs["Roughness"])
        deep = rgb(DEEP_SEA)
        if deep_only:
            n.bsdf.inputs["Base Color"].default_value = deep
        else:
            d = n.new("ShaderNodeAttribute", attribute_name="depth").outputs["Fac"]
            water = n.ramp(d, [(0.0, rgb(SHALLOWS)), (0.55, rgb(LAGOON)), (1.0, deep)])
            # Foam where the water thins out over the shore, broken up.
            edge = n.math("SUBTRACT", 1, n.math("MULTIPLY", d, 22.0), clamp=True)
            froth = n.math("GREATER_THAN", n.noise(flowing(n, (0.9, 0.5)), 2.4, 5.0), 0.46)
            n.link(n.mix(n.math("MULTIPLY", edge, froth), water, rgb("#EEF2EE")), n.bsdf.inputs["Base Color"])
        # Three scales drifting with the breeze: slow chop, ripples, fine capillary texture.
        breeze = flowing(n, (2.6, 1.4))
        ripples = n.math("ADD", n.math("ADD", n.math("MULTIPLY", n.noise(breeze, 0.45, 3.0), 1.4),
                                         n.noise(n.scaled(breeze, (1, 2.5, 1)), 1.6, 5.0)),
                         n.math("MULTIPLY", n.noise(n.scaled(breeze, (1, 1.8, 1)), 11.0, 3.0), 0.35))
        bump = n.new("ShaderNodeBump", Strength=0.2)
        n.link(ripples, bump.inputs["Height"])
        n.link(bump.outputs["Normal"], n.bsdf.inputs["Normal"])
        scene.assign(o, mat)
        for fc in mat.node_tree.animation_data.action.fcurves:
            for k in fc.keyframe_points:
                k.interpolation = "LINEAR"
    for fc in drift.animation_data.action.fcurves:
        for k in fc.keyframe_points:
            k.interpolation = "LINEAR"


def flowing(n, travel):
    """Object coordinates sliding `travel` (x, y metres) over the cove's frames, so
    a texture fed from them drifts steadily."""
    move = n.new("ShaderNodeMapping")
    n.link(n.coord(), move.inputs["Vector"])
    for f, k in ((round(10.5 * FPS), 0.0), (round(25.0 * FPS), 1.0)):
        move.inputs["Location"].default_value = (travel[0] * k, travel[1] * k, 0.0)
        move.inputs["Location"].keyframe_insert("default_value", frame=f)
    return move.outputs[0]


def terrain(size=150, steps=240):
    verts, faces = [], []
    for j in range(steps + 1):
        for i in range(steps + 1):
            x, y = -size / 2 + size * i / steps, -size / 2 + size * j / steps + COVE.y
            verts.append((x, y, shore_height(x, y)))
    for j in range(steps):
        for i in range(steps):
            a = j * (steps + 1) + i
            faces.append((a, a + 1, a + steps + 2, a + steps + 1))
    obj = scene.mesh_object("terrain", verts, faces)
    obj.data.polygons.foreach_set("use_smooth", [True] * len(faces))
    scene.assign(obj, ground())
    return obj


def ground():
    """Sand to cliff: wet dark sand at the water, dry rippled sand above, then coastal
    rock, grey basalt and limestone with faint warm strata, cracked, with green on
    the ledges facing up."""
    mat = material("ground", SAND, 0.9)
    n = Nodes(mat)
    co = n.coord()
    sep = n.xyz(co)
    geo = n.new("ShaderNodeNewGeometry")
    nz = n.xyz(geo.outputs["Normal"])[2]
    z = sep[2]
    # Sand: wet band, dry ripples, scattered darker grains.
    wet = n.math("SUBTRACT", 1, n.math("MULTIPLY", n.math("SUBTRACT", z, WATERLINE + 0.05), 3.0), clamp=True)
    ripple = n.new("ShaderNodeTexWave", wave_type="BANDS", bands_direction="X", Scale=1.4, Distortion=6.0,
                   **{"Detail Scale": 2.0})
    n.link(co, ripple.inputs["Vector"])
    grains = n.noise(co, 40.0, 2.0)
    sand = n.mix(n.math("MULTIPLY", ripple.outputs["Fac"], 0.25), rgb(SAND), [c * 0.85 for c in rgb(SAND)[:3]] + [1])
    sand = n.mix(n.math("MULTIPLY", n.math("GREATER_THAN", grains, 0.68), 0.5), sand, rgb(OCHRE))
    sand = n.mix(wet, sand, rgb(WET_SAND))
    # Rock strata: bands up the cliff, tilted and warped.
    warp = n.noise(co, 0.35, 3.0)
    strata = n.new("ShaderNodeTexWave", wave_type="BANDS", bands_direction="Z", wave_profile="SIN", Scale=0.09,
                   Distortion=1.2, **{"Detail Scale": 1.0})
    tilt = n.new("ShaderNodeMapping")
    tilt.inputs["Rotation"].default_value = (math.radians(9), math.radians(-6), 0)
    n.link(co, tilt.inputs["Vector"])
    n.link(tilt.outputs[0], strata.inputs["Vector"])
    layer = n.math("ADD", n.math("MULTIPLY", strata.outputs["Fac"], 0.3), n.math("MULTIPLY", warp, 0.7))
    rock = n.ramp(layer, [(0.0, rgb(BASALT)), (0.3, rgb(GREY_ROCK)), (0.5, rgb(CLIFFS)), (0.62, rgb(GREY_ROCK)),
                          (0.78, rgb(OCHRE)), (0.9, rgb(LIMESTONE)), (1.0, rgb(GREY_ROCK))])
    grit = n.noise(co, 2.2, 6.0, 0.6)
    rock = n.mix(n.math("MULTIPLY", grit, 0.45), rock, [c * 0.6 for c in rgb(CLIFFS)[:3]] + [1], "MULTIPLY")
    cracks = n.noise(co, 3.0, 8.0, 0.7)
    rock = n.mix(n.math("MULTIPLY", n.math("LESS_THAN", cracks, 0.32), 0.7), rock, rgb(BASALT))
    moss = n.math("MULTIPLY", n.math("MULTIPLY", n.math("SUBTRACT", nz, 0.6), 5.0, clamp=True),
                  n.math("GREATER_THAN", n.noise(co, 1.5, 4.0), 0.4))
    rock = n.mix(n.math("MULTIPLY", moss, 0.85), rock, rgb(MOSS))
    is_rock = n.math("MAXIMUM", n.math("MULTIPLY", n.math("SUBTRACT", z, WATERLINE + 1.7), 0.9, clamp=True),
                     n.math("MULTIPLY", n.math("SUBTRACT", 0.8, nz), 5.0, clamp=True))
    n.link(n.mix(is_rock, sand, rock), n.bsdf.inputs["Base Color"])
    n.link(n.math("SUBTRACT", 0.92, n.math("MULTIPLY", wet, 0.55)), n.bsdf.inputs["Roughness"])
    bump = n.new("ShaderNodeBump", Strength=0.45, Distance=0.08)
    n.link(n.math("ADD", n.math("MULTIPLY", cracks, is_rock), n.math("MULTIPLY", ripple.outputs["Fac"], 0.15)),
           bump.inputs["Height"])
    n.link(bump.outputs["Normal"], n.bsdf.inputs["Normal"])
    return mat


def palm(base, height, lean, seed):
    rng = random.Random(seed)
    bark = material("bark", BARK, 0.8)
    leaf = material("frond", PALMS, 0.7)
    Nodes(leaf).bsdf.inputs["Subsurface Weight"].default_value = 0.0
    base = Vector(base)
    pts = [base + Vector((lean.x * u * u, lean.y * u * u, height * u)) for u in (k / 8 for k in range(9))]
    scene.rope("palm_trunk", pts, 0.16, bark)
    top = pts[-1]
    crown = scene.empty("palm_crown", None, top)
    for c in range(3):                                            # coconuts under the fronds
        ang = 2 * math.pi * c / 3 + rng.uniform(0, 1)
        scene.ellipsoid("coconut", Vector((0.18 * math.cos(ang), 0.18 * math.sin(ang), -0.22)),       # in the crown's frame
                        (0.11, 0.11, 0.12), material("coconut", "#5B4632", 0.7), crown)
    for k in range(9):
        heading = 2 * math.pi * k / 9 + rng.uniform(-0.2, 0.2)
        length = rng.uniform(2.2, 3.0)
        curve = bpy.data.curves.new("frond", "CURVE")
        curve.dimensions, curve.extrude, curve.twist_mode = "3D", 0.2, "MINIMUM"
        spline = curve.splines.new("POLY")
        spline.points.add(6)
        for s, p in enumerate(spline.points):
            u = s / 6
            p.co = (*(top + Vector((math.cos(heading) * length * u, math.sin(heading) * length * u,
                                    0.6 * math.sin(math.pi * u * 0.7) - 1.3 * u * u))), 1)
        frond = bpy.data.objects.new("frond", curve)
        bpy.context.scene.collection.objects.link(frond)
        curve.materials.append(leaf)
        frond.parent = crown
        frond.matrix_parent_inverse = crown.matrix_world.inverted()
    # Sway: an irregular drift of the whole crown in the breeze.
    crown.keyframe_insert("rotation_euler", frame=round(10.5 * FPS))
    for i, fc in enumerate(crown.animation_data.action.fcurves):
        mod = fc.modifiers.new("NOISE")
        mod.strength, mod.scale, mod.phase = (0.06, 0.06, 0.03)[i], 40.0 + 7 * i, seed * 1.7 + i


def torch(base):
    x, y, z = base
    scene.cylinder("torch_pole", (x, y, z - 0.2), (x, y, z + 1.5), 0.04, material("torch_pole", BARK, 0.8), vertices=8)
    scene.cylinder("torch_head", (x, y, z + 1.5), (x, y, z + 1.72), 0.08, material("pitch", "#1B1A1F", 0.7), vertices=12)
    scene.ellipsoid("torch_flame", (x, y, z + 1.86), (0.08, 0.08, 0.16), material("flame", TORCHES, 0.3, emission=12.0))
    data = bpy.data.lights.new("torch_light", "POINT")
    data.energy, data.color, data.shadow_soft_size = 220, rgb(TORCHES)[:3], 0.15
    light = bpy.data.objects.new("torch_light", data)
    bpy.context.scene.collection.objects.link(light)
    light.location = (x, y, z + 1.95)


def haze():
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, COVE.y, 8))
    vol = bpy.context.active_object
    vol.name, vol.scale = "haze", (200, 200, 18)
    mat = bpy.data.materials.new("haze")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.remove(nodes["Principled BSDF"])
    scatter = nodes.new("ShaderNodeVolumePrincipled")
    scatter.inputs["Color"].default_value = rgb(AMBER)
    scatter.inputs["Density"].default_value = 0.0016
    mat.node_tree.links.new(scatter.outputs["Volume"], nodes["Material Output"].inputs["Volume"])
    scene.assign(vol, mat)


def dress_beach():
    rng = random.Random(SEED)
    for k in range(9):
        ang = math.radians(rng.uniform(100, 200))
        d = rng.uniform(13.5, 17.5)
        x, y = COVE.x + d * math.cos(ang), COVE.y + d * math.sin(ang)
        palm((x, y, shore_height(x, y) - 0.1), rng.uniform(5.5, 8.0),
             Vector((rng.uniform(-1.2, 1.2), rng.uniform(-1.5, -0.4))), SEED + k)
    for k in range(5):
        ang = math.radians(105 + 17 * k)
        x, y = COVE.x + 9.8 * math.cos(ang), COVE.y + 9.8 * math.sin(ang)
        torch((x, y, shore_height(x, y)))


def camp():
    """The crew's camp on the east beach, which every cove shot sees across the
    water: a weathered jetty reaching into the bay, barrels, crates and a rope coil
    at its foot, a driftwood campfire with a flickering glow and drifting smoke,
    and coconut palms swaying behind it."""
    on_shore = lambda x, y: Vector((x, y, shore_height(x, y)))
    planks = scene.wood("jetty_wood", scene.AGED_WOOD, "X", 0.22, grain="Y")
    staves = scene.wood("barrel_wood", scene.DARK_OAK, "Z", 0.09, grain="Z")
    iron = scene.metal("hoop_iron", scene.IRON, scene.RUST, 0.0, 0.7, 0.5)
    # Jetty: posts down to the floor, a plank deck at a steady height.
    foot, far = Vector((14.5, 17.5)), Vector((7.6, 9.2))
    along = (far - foot).normalized()
    across = Vector((-along.y, along.x))
    deck = WATERLINE + 0.75
    length = (far - foot).length
    for k in range(int(length / 1.5) + 1):
        c = foot + along * k * 1.5
        for s in (-0.75, 0.75):
            p = c + across * s
            scene.cylinder("jetty_post", (p.x, p.y, shore_height(p.x, p.y) - 0.4), (p.x, p.y, deck + 0.05), 0.09,
                           planks, vertices=10)
    for k in range(int(length / 0.26)):
        c = foot + along * (k * 0.26 + 0.13)
        bpy.ops.mesh.primitive_cube_add(size=1, location=(c.x, c.y, deck + 0.09))
        board = bpy.context.active_object
        board.name, board.scale = "jetty_plank", (0.24, 1.8, 0.05)
        board.rotation_euler = (0.0, 0.0, math.atan2(along.y, along.x) + random.Random(k).uniform(-0.03, 0.03))
        scene.assign(board, planks)
    # Barrels, one on its side, crates stacked, a coil of rope.
    for x, y, lie in ((15.0, 21.5, False), (15.6, 20.9, False), (14.5, 20.6, True)):
        g = on_shore(x, y)
        a, b = (g, g + Vector((0, 0, 0.9))) if not lie else (g + Vector((-0.45, 0, 0.32)), g + Vector((0.45, 0, 0.32)))
        scene.cylinder("barrel", a, b, 0.32, staves, vertices=20)
        for u in (0.15, 0.85):
            h1, h2 = a.lerp(b, u - 0.03), a.lerp(b, u + 0.03)
            scene.cylinder("barrel_hoop", h1, h2, 0.333, iron, vertices=20)
    for x, y, size, z in ((16.6, 21.4, 0.62, 0.0), (16.65, 21.45, 0.44, 0.62), (17.3, 21.0, 0.5, 0.0)):
        g = on_shore(x, y)
        bpy.ops.mesh.primitive_cube_add(size=size, location=(g.x, g.y, g.z + z + size / 2))
        crate = bpy.context.active_object
        crate.name, crate.rotation_euler.z = "crate", random.Random(int(x * 10)).uniform(-0.4, 0.4)
        crate.modifiers.new("edges", "BEVEL").width = 0.02
        scene.assign(crate, planks)
    g = on_shore(15.7, 22.7)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.28, minor_radius=0.07, location=(g.x, g.y, g.z + 0.07))
    coil = bpy.context.active_object
    coil.name = "rope_coil"
    scene.assign(coil, scene.cloth("coil_rope", scene.HEMP, ("stripes", "#7E6438", 60.0)))
    # Campfire: crossed driftwood, a flickering flame and light, smoke drifting up.
    fire = on_shore(17.3, 26.0)
    for k in range(4):
        ang = k * math.pi / 4 + 0.3
        d = Vector((math.cos(ang), math.sin(ang), 0)) * 0.45
        scene.cylinder("firewood", fire - d + Vector((0, 0, 0.05)), fire + d + Vector((0, 0, 0.18)), 0.06,
                       material("driftwood", "#8C7B68", 0.85), vertices=8)
    flame_mat = material("campfire", TORCHES, 0.3, emission=9.0)
    for k, (dx, dy, h) in enumerate(((0, 0, 0.55), (0.12, 0.05, 0.4), (-0.1, -0.06, 0.45))):
        scene.ellipsoid("flame", fire + Vector((dx, dy, 0.25 + h / 2)), (0.13, 0.13, h / 2), flame_mat)
    strength = Nodes(flame_mat).bsdf.inputs["Emission Strength"]
    data = bpy.data.lights.new("campfire_light", "POINT")
    data.energy, data.color, data.shadow_soft_size = 420, rgb("#FFB25C")[:3], 0.3
    light = bpy.data.objects.new("campfire_light", data)
    bpy.context.scene.collection.objects.link(light)
    light.location = fire + Vector((0, 0, 0.7))
    strength.keyframe_insert("default_value", frame=round(10.5 * FPS))
    data.keyframe_insert("energy", frame=round(10.5 * FPS))
    for fc, value in ((flame_mat.node_tree.animation_data.action.fcurves[0], 9.0),
                      (data.animation_data.action.fcurves[0], 420.0)):
        mod = fc.modifiers.new("NOISE")
        mod.strength, mod.scale = value * 0.35, 2.0                     # irregular flicker
    bpy.ops.mesh.primitive_cube_add(size=1, location=fire + Vector((0.3, 0.2, 3.0)))
    smoke = bpy.context.active_object
    smoke.name, smoke.scale = "smoke", (1.6, 1.6, 5.0)
    mat = bpy.data.materials.new("smoke")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.remove(nodes["Principled BSDF"])
    s = Nodes(mat)
    vol = s.new("ShaderNodeVolumePrincipled")
    vol.inputs["Color"].default_value = rgb("#8D8A86")
    co = s.coord()
    drift = s.new("ShaderNodeMapping")
    s.link(co, drift.inputs["Vector"])
    for f, z in ((round(10.5 * FPS), 0.0), (round(25.0 * FPS), -1.6)):     # rises, leaning with the breeze
        drift.inputs["Location"].default_value = (0.35 * z, 0.18 * z, z)
        drift.inputs["Location"].keyframe_insert("default_value", frame=f)
    puffs = s.noise(drift.outputs[0], 2.2, 3.0)
    column = s.math("SUBTRACT", 1, s.math("MULTIPLY", s.math("ADD", s.xyz(co)[2], 0.5), 1.0), clamp=True)
    narrow = s.math("SUBTRACT", 1, s.math("MULTIPLY", s.math("ADD", s.math("POWER", s.xyz(co)[0], 2.0),
                                                                s.math("POWER", s.xyz(co)[1], 2.0)), 5.0), clamp=True)
    density = s.math("MULTIPLY", s.math("MULTIPLY", s.math("SUBTRACT", puffs, 0.45), 2.0, clamp=True),
                     s.math("MULTIPLY", column, narrow))
    s.link(s.math("MULTIPLY", density, 0.9), vol.inputs["Density"])
    s.link(vol.outputs["Volume"], nodes["Material Output"].inputs["Volume"])
    scene.assign(smoke, mat)
    for fc in mat.node_tree.animation_data.action.fcurves:
        for k in fc.keyframe_points:
            k.interpolation = "LINEAR"
    # Coconut palms behind the camp, leaning out over the sand.
    for k, (x, y) in enumerate(((19.0, 14.0), (20.5, 19.5), (22.5, 23.5), (21.0, 28.0), (18.6, 31.5), (17.6, 9.0))):
        rng = random.Random(SEED + 40 + k)
        palm(tuple(on_shore(x, y) - Vector((0, 0, 0.1))), rng.uniform(6.0, 8.5),
             Vector((rng.uniform(-1.8, -0.6), rng.uniform(-0.6, 0.6))), SEED + 40 + k)


def gulls():
    """Three gulls wheeling high over the bay, gliding with the odd wing beat."""
    feather = material("gull", "#E9ECEC", 0.6)
    tips = material("gull_tips", "#3A3D40", 0.6)
    for k, (radius, height, speed, phase) in enumerate(((7.0, 17.0, 0.32, 0.0), (9.5, 19.5, 0.27, 2.1),
                                                        (6.0, 15.5, 0.36, 4.0))):
        hub = scene.empty("gull_orbit", None, (5.0, 27.0, height))
        bird = scene.empty("gull", hub, (radius, 0, 0))
        bird.rotation_euler = (0.25, 0, math.pi / 2)                    # banked into the turn, flying tangent
        scene.ellipsoid("gull_body", (0, 0, 0), (0.22, 0.06, 0.06), feather, bird)
        for side in (-1, 1):
            wing = scene.empty("gull_shoulder", bird, (0, side * 0.04, 0.02))
            scene.ellipsoid("gull_wing", (-0.02, side * 0.38, 0), (0.09, 0.36, 0.015), feather, wing)
            scene.ellipsoid("gull_tip", (-0.05, side * 0.7, 0), (0.06, 0.08, 0.012), tips, wing)
            wing.rotation_euler = (side * 0.18, 0, 0)
            wing.keyframe_insert("rotation_euler", index=0, frame=round(10.5 * FPS))
            mod = wing.animation_data.action.fcurves[0].modifiers.new("NOISE")
            mod.strength, mod.scale, mod.phase = 0.35, 5.0, k * 3.1 + side
        for f, ang in ((round(10.5 * FPS), phase), (round(25.0 * FPS), phase + speed * 14.5)):
            hub.rotation_euler = (0, 0, ang)
            hub.keyframe_insert("rotation_euler", frame=f)
        for fc in hub.animation_data.action.fcurves:
            for key in fc.keyframe_points:
                key.interpolation = "LINEAR"


def dolphins():
    """Two dolphins arcing once in the bay's deepest open water (about 1.1 m, off the
    boat's port side) while the orbit passes, one just after the other: nose up out
    of the water and back in, then gone, never touching the bed or lingering at the
    surface."""
    skin = material("dolphin", "#6F7C86", 0.35)
    belly = material("dolphin_belly", "#C9CED2", 0.4)
    out = Vector((-1.5, 9.5))                                       # deepest water the orbit sees
    for k, (start, offset) in enumerate(((13.6, Vector((0, 0))), (14.05, Vector((2.2, 1.0))))):
        root = scene.empty("dolphin", None, (0, 0, -5))
        scene.ellipsoid("dolphin_body", (0, 0, 0), (1.1, 0.22, 0.24), skin, root)
        scene.ellipsoid("dolphin_belly", (0.1, 0, -0.08), (0.85, 0.16, 0.14), belly, root)
        bpy.ops.mesh.primitive_cone_add(vertices=8, radius1=0.16, depth=0.36, location=(-0.05, 0, 0.3),
                                        rotation=(0, -0.4, 0))
        fin = bpy.context.active_object
        fin.parent = root
        scene.assign(fin, skin)
        scene.ellipsoid("dolphin_flukes", (-1.15, 0, 0), (0.12, 0.42, 0.04), skin, root)
        base = out + offset
        heading = MOUTH + math.pi / 2                                    # swimming across the mouth
        ahead = Vector((math.cos(heading), math.sin(heading)))
        for u in [i / 12 for i in range(13)]:
            p = base + ahead * (u - 0.5) * 4.5
            # A leap about 1.2 m clear, from deep enough that nose and flukes are both
            # under the (opaque) surface at either end, where it appears and vanishes.
            z = WATERLINE - 1.5 + 2.7 * math.sin(math.pi * u)
            root.location = (p.x, p.y, z)
            root.rotation_euler = (0, -math.atan2(2.7 * math.pi * math.cos(math.pi * u), 4.5), heading)
            root.keyframe_insert("location", frame=(start + u * 0.9) * FPS)
            root.keyframe_insert("rotation_euler", frame=(start + u * 0.9) * FPS)
        # Only there for its leap: hidden before it surfaces and once it is back under.
        shown = (round(start * FPS) - 1, round((start + 0.9) * FPS) + 1)
        for o in [root] + list(root.children_recursive):
            for f, hidden in ((shown[0] - 1, True), (shown[0], False), (shown[1], True)):
                o.hide_render = hidden
                o.keyframe_insert("hide_render", frame=f)


def poly_haven():
    """Two CC0 Poly Haven models (fetched by scripts/setup.ps1): a cannon by the
    jetty's foot aimed out over the bay, and rocky outcrops at both ends of the
    beach, set into the sand."""
    root = scene.ROOT / "third_party" / "polyhaven"

    def place(model, at, heading, sink=0.0):
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(root / model / f"{model}_1k.gltf"))
        holder = scene.empty(model)
        for obj in set(bpy.data.objects) - before:
            if obj.parent is None and obj is not holder:
                obj.parent = holder
        holder.location = (at[0], at[1], shore_height(*at) - sink)
        holder.rotation_euler.z = heading
        return holder

    to_boat = math.atan2(-19.6, -13.6)
    place("cannon_01", (13.6, 19.6), to_boat - math.pi / 2)
    place("coast_land_rocks_03", (20.5, 37.0), 0.4, sink=0.45)
    place("coast_land_rocks_03", (11.5, -1.5), 2.1, sink=0.45)


def sea_life():
    """Life breaking the surface of the bay where the orbit and the medium shot see
    open water: small fish leaping at odd moments, and a sea turtle that surfaces
    to breathe and slips back under."""
    silver = material("fish", "#C9D3D8", 0.25, 0.6)
    for k, (x, y, at, heading) in enumerate(((4.0, 9.0, 12.4, 0.6), (7.5, 14.0, 15.6, 2.2), (2.5, 12.5, 15.9, 2.0),
                                             (-1.5, 7.0, 17.1, -0.4))):
        fish = scene.empty("fish", None, (x, y, WATERLINE - 0.5))
        scene.ellipsoid("fish_body", (0, 0, 0), (0.14, 0.035, 0.045), silver, fish)
        scene.ellipsoid("fish_tail", (-0.15, 0, 0), (0.04, 0.008, 0.045), silver, fish)
        ahead = Vector((math.cos(heading), math.sin(heading), 0))
        for u in [i / 8 for i in range(9)]:
            fish.location = Vector((x, y, 0)) + ahead * (u - 0.5) * 0.9 + Vector((0, 0, WATERLINE - 0.15 + 0.5 * math.sin(math.pi * u)))
            fish.rotation_euler = (0, -math.atan2(0.5 * math.pi * math.cos(math.pi * u), 0.9), heading)
            fish.keyframe_insert("location", frame=(at + u * 0.5) * FPS)
            fish.keyframe_insert("rotation_euler", frame=(at + u * 0.5) * FPS)
    shell = material("turtle_shell", "#4E5A3A", 0.5)
    skin = material("turtle_skin", "#7A7F5A", 0.6)
    turtle = scene.empty("turtle", None, (0.5, 13.0, WATERLINE - 0.6))
    scene.ellipsoid("turtle_shell", (0, 0, 0), (0.45, 0.36, 0.12), shell, turtle)
    scene.ellipsoid("turtle_head", (0.5, 0, 0.05), (0.12, 0.08, 0.07), skin, turtle)
    for side in (-1, 1):
        scene.ellipsoid("turtle_flipper", (0.15, side * 0.42, -0.02), (0.2, 0.08, 0.02), skin, turtle,
                        rot=(0, 0, side * 0.5))
    turtle.rotation_euler.z = 2.4
    for s, z in ((13.9, WATERLINE - 0.6), (14.4, WATERLINE - 0.04), (15.5, WATERLINE - 0.04), (16.1, WATERLINE - 0.6)):
        turtle.location.z = z
        turtle.keyframe_insert("location", index=2, frame=s * FPS)


def bob(ship):
    """The boat at anchor: a slow heave and a little roll."""
    frames = np.arange(FIRST - 12, LAST + 13)
    t = frames / FPS
    heave = 0.025 * np.sin(2 * math.pi * t / 4.3)
    roll = math.radians(1.2) * np.sin(2 * math.pi * t / 5.1 + 0.7)
    pitch = math.radians(0.5) * np.sin(2 * math.pi * t / 3.7 + 1.9)
    scene.bake_keys(ship, "location", frames, np.column_stack([0 * t, 0 * t, heave]), "BEZIER")
    scene.bake_keys(ship, "rotation_euler", frames, np.column_stack([roll, pitch, 0 * t]), "BEZIER")
    scene.save_motion("cove", frames, roll, pitch, heave)


def cameras(s, ship):
    crew = Vector((-2.42, 0.0, DECK + 0.14))
    orbit = lambda az, r, h: crew + Vector((math.cos(math.radians(az)) * r, math.sin(math.radians(az)) * r, h))
    keys = [(10.5, orbit(-150, 11.0, 3.2), crew + Vector((-6, -4, 14)), 28, 0.25),
            (10.9, orbit(-148, 10.6, 3.0), crew + Vector((0, 0, 0.6)), 28)]
    for k, t in enumerate(np.linspace(11.4, 19.0, 6)):
        u = k / 5
        e = u * u * (3 - 2 * u)
        keys.append((t, orbit(-148 + 50 * e, 10.4 - 9.1 * e, 2.9 - 2.5 * e), crew + Vector((0, 0, 0.5 - 0.48 * e)),
                     28 + 10 * e))
    c4 = scene.shot("s4_orbit", keys, focus=s["sailor_bodies"]["upper_shell_link"], fstop=4.0, handheld=0.002, seed=4)
    scene.cut(c4, 10.5)
    ax, ay = scene.ANGLER_AT
    sx, sy = scene.SAILOR_AT
    face = Vector((ax - 0.04, ay, DECK + 0.12)).lerp(Vector((sx, sy, DECK + 0.12)), 0.3)
    c5 = scene.shot("s5_close", [(19.0, (ax - 0.6, ay - 0.4, DECK + 0.23), face, 33),
                                 (21.0, (ax - 0.54, ay - 0.36, DECK + 0.22), face, 35)],
                    focus=s["angler_bodies"]["upper_shell_link"], fstop=2.8, handheld=0.003, seed=5)
    scene.cut(c5, 19.0)
    # 6. The catch, from aft and starboard at crab height, clear of the rod holder:
    # the angler near, the chest centred, the sailor and the captain behind; a slow push while the
    # lid opens, then the dive: an eased move to straight down onto the portal,
    # framing it edge to edge on the last frame. Parented to the boat, so the
    # bob never shifts the portal in frame.
    cx, cy = CHEST_AT
    lid_top = Vector((cx, cy, DECK + CHEST[2] + 0.04))
    portal = Vector((cx, cy, DECK + CHEST[2] - 0.02))
    above = portal + Vector((0.0, 0.0, 30 * PORTAL / 36))        # lens 30 on a 36 mm sensor: edge to edge
    look_at = lid_top + Vector((0.25, 0.2, 0.12))        # between the chest and the captain's head
    c6 = scene.shot("s6_catch", [(LANDS, (cx - 0.30, cy - 0.72, DECK + 0.42), look_at, 27),
                                 (DIVES, (cx - 0.26, cy - 0.64, DECK + 0.40), look_at, 28),
                                 (25.0 - 1 / FPS, above, portal + Vector((0.004, 0.0, 0.0)), 30)], parent=ship)
    scene.cut(c6, LANDS)


def chest(ship):
    """The catch: an iron-banded plank chest. It drops onto the deck at LANDS (swung
    aboard and unhooked, the line hanging over the stern again), its lid creaks open at
    OPENS, and inside it a 16:9 portal plays the opening shot's pre-roll
    (render/out/preroll/), timed so its last image is the frame before the film's
    first. Both crabs' eyes widen as it opens."""
    d, w, h = CHEST
    wall = 0.012
    planks = scene.wood("chest_wood", scene.AGED_WOOD, "Z", 0.032, grain="Y")
    iron = scene.metal("chest_iron", scene.IRON, scene.RUST, 0.0, 0.7, 0.5)
    root = scene.empty("chest", ship)

    def box(name, center, size, mat, parent):
        bpy.ops.mesh.primitive_cube_add(size=1)
        o = bpy.context.active_object
        o.name, o.parent, o.location, o.scale = name, parent, center, size
        bevel = o.modifiers.new("edges", "BEVEL")
        bevel.width, bevel.segments = min(min(size) * 0.3, 0.003), 2
        scene.assign(o, mat)
        return o

    box("chest_floor", (0, 0, wall / 2), (d, w, wall), planks, root)
    for x in (-1, 1):
        box("chest_wall", (x * (d - wall) / 2, 0, h / 2), (wall, w, h), planks, root)
    for y in (-1, 1):
        box("chest_wall", (0, y * (w - wall) / 2, h / 2), (d - 2 * wall, wall, h), planks, root)
        for x in (-1, 1):                                           # iron straps
            box("chest_strap", (x * (d / 2 + 0.0015), y * 0.09, h / 2), (0.003, 0.016, h + 0.002), iron, root)
    # Lid: a shallow barrel hinged on the back (+x) edge.
    hinge = scene.empty("chest_hinge", root, (d / 2, 0, h))
    arc = [(-d / 2 + d / 2 * math.cos(a), 0.045 * math.sin(a)) for a in np.linspace(0, math.pi, 13)]
    shell = lambda y0, y1, scale=1.0: [(x * scale, y, z * scale) for y in (y0, y1) for x, z in arc]
    strip = [(k, k + 1, 13 + k + 1, 13 + k) for k in range(12)]
    lid = scene.mesh_object("chest_lid", shell(-w / 2, w / 2), strip + [tuple(range(13))[::-1], tuple(range(13, 26))],
                            planks, hinge)
    solid = lid.modifiers.new("thickness", "SOLIDIFY")
    solid.thickness, solid.offset = 0.01, -1.0
    for y in (-0.09, 0.09):
        band = scene.mesh_object("chest_band", shell(y - 0.008, y + 0.008, 1.03), strip, iron, hinge)
        band.modifiers.new("thickness", "SOLIDIFY").thickness = 0.002
    box("chest_hasp", (-d / 2 - 0.004, 0, h - 0.012), (0.006, 0.026, 0.03), iron, root)
    # The portal, image top toward +x and right toward -y: until the pre-roll starts
    # it shows the pre-roll's first image, then the pre-roll plays, its last image
    # landing on the film's last frame. Two quads, swapped on that frame.
    ph, z = PORTAL * 9 / 16, h - 0.02
    play = round(25.0 * FPS) - PREROLL
    for name, sequence, shown in (("portal_still", False, (round(LANDS * FPS), play)),
                                  ("portal", True, (play, round(25.0 * FPS)))):
        quad = scene.mesh_object(name, [(ph / 2, PORTAL / 2, z), (ph / 2, -PORTAL / 2, z), (-ph / 2, -PORTAL / 2, z),
                                        (-ph / 2, PORTAL / 2, z)], [(0, 1, 2, 3)], None, root, smooth=False)
        uv = quad.data.uv_layers.new(name="UVMap")
        for loop, (u, v) in zip(uv.data, ((0, 1), (1, 1), (1, 0), (0, 0))):
            loop.uv = (u, v)
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.remove(nodes["Principled BSDF"])
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(scene.OUT / "preroll" / "0001.png"))
        if sequence:
            tex.image.source = "SEQUENCE"
            tex.image_user.frame_duration, tex.image_user.frame_start = PREROLL, play
            tex.image_user.use_auto_refresh = True
        glow = nodes.new("ShaderNodeEmission")
        mat.node_tree.links.new(tex.outputs["Color"], glow.inputs["Color"])
        for s, strength in ((LANDS, 2.2), (DIVES, 2.2), (24.6, 1.0)):     # glows in the chest, neutral by the end
            glow.inputs["Strength"].default_value = strength
            glow.inputs["Strength"].keyframe_insert("default_value", frame=s * FPS)
        mat.node_tree.links.new(glow.outputs[0], nodes["Material Output"].inputs["Surface"])
        quad.data.materials.append(mat)
        quad.visible_shadow = quad.visible_diffuse = quad.visible_glossy = False
        for f, hidden in ((shown[0] - 1, True), (shown[0], False), (shown[1], True)):
            quad.hide_render = hidden
            quad.keyframe_insert("hide_render", frame=f)
    # Hidden before LANDS, when it is still above the frame: swung in off the line from
    # the stern (clear of the crew either side), it falls in under gravity, slams down a
    # few frames later and bounces.
    root.rotation_mode = "XYZ"
    fall, high, out = 0.22, 0.9, 0.35          # s, m above the deck, m aft at the cut
    speed = (high - 4.9 * fall ** 2) / fall    # m/s down at the cut
    drop = [(LANDS + fall * k / 6, high - speed * fall * k / 6 - 4.9 * (fall * k / 6) ** 2, out * (1 - k / 6),
             0.15 - 0.18 * k / 6) for k in range(7)]
    for t, lift, side, tilt in drop + [(LANDS + 0.32, 0.012, 0.0, 0.01), (LANDS + 0.42, 0.0, 0.0, 0.0)]:
        root.location = (CHEST_AT[0] - side, CHEST_AT[1], DECK + 0.001 + max(lift, 0.0))
        root.rotation_euler = (tilt, tilt * 0.5, 0.0)
        root.keyframe_insert("location", frame=t * FPS)
        root.keyframe_insert("rotation_euler", frame=t * FPS)
    for o in [root] + [c for c in root.children_recursive if not c.name.startswith("portal")]:
        for f, hidden in ((round(LANDS * FPS) - 1, True), (round(LANDS * FPS), False)):
            o.hide_render = hidden
            o.keyframe_insert("hide_render", frame=f)
    hinge.rotation_mode = "XYZ"
    for t, deg in ((OPENS, 0.0), (OPENS + 0.55, 108.0), (OPENS + 0.8, 101.0)):
        hinge.rotation_euler = (0, math.radians(deg), 0)
        hinge.keyframe_insert("rotation_euler", frame=t * FPS)
    # The crew's surprise: eyes widen as the lid comes up, then ease back.
    for name in ("angler", "sailor"):
        tree = bpy.data.materials[f"{name}_screen"].node_tree
        value = tree.nodes["eyelid"].outputs[0]
        fc = [c for c in tree.animation_data.action.fcurves if "eyelid" in c.data_path][0]
        for k in reversed(range(len(fc.keyframe_points))):         # no blink inside the beat
            if LANDS * FPS <= fc.keyframe_points[k].co.x <= (OPENS + 1.4) * FPS:
                fc.keyframe_points.remove(fc.keyframe_points[k])
        # A flinch as the chest slams down, a wary half-squint, then the wide eyes.
        for t, v in ((LANDS + 0.18, 1.0), (LANDS + 0.24, 0.25), (LANDS + 0.36, 0.7), (LANDS + 0.62, 0.62),
                     (OPENS - 0.1, 0.8), (OPENS, 1.0), (OPENS + 0.35, 1.32), (OPENS + 1.0, 1.28), (OPENS + 1.4, 1.1)):
            value.default_value = v
            value.keyframe_insert("default_value", frame=t * FPS)
    return root


def captain_idle(penguin, neck, shoulders):
    """The captain in the door while the orbit closes in: his weight shifting, his
    head turning aft to the crew as the sailor waves, a flipper wave back, then
    settled by the end of the close shot."""
    heading = penguin.rotation_euler.z
    first, last = round(10.5 * FPS), round(LANDS * FPS) - 4
    for f in range(first, last + 1, 4):
        s = f / FPS
        settle = min(max((LANDS - 0.6 - s) / 0.8, 0.0), 1.0)
        penguin.rotation_euler = (0.035 * math.sin(2 * math.pi * s / 2.6) * settle, 0, heading)
        penguin.keyframe_insert("rotation_euler", frame=f)
    aft = -1                                     # facing out to starboard, aft is to his right (-y local)
    for s, yaw, roll in ((10.5, 0.0, 0.0), (12.6, 0.0, 0.0), (13.3, aft * 0.62, 0.1), (17.6, aft * 0.55, 0.06),
                         (18.6, aft * 0.25, 0.0), (LANDS - 0.2, 0.0, 0.0)):
        neck.rotation_euler = (roll, 0, yaw)
        neck.keyframe_insert("rotation_euler", frame=s * FPS)
    wave = [sh for sh in shoulders if sh.location.y < 0][0]                     # the aft flipper
    keys = [(10.5, 0.0), (14.6, 0.0), (15.0, -1.25)]
    keys += [(15.0 + 0.25 * k, -1.25 + (0.3 if k % 2 else 0.0)) for k in range(1, 6)]
    keys += [(16.9, 0.0), (LANDS - 0.2, 0.0)]
    for s, lift in keys:
        wave.rotation_euler = (lift, 0, 0)
        wave.keyframe_insert("rotation_euler", frame=s * FPS)


def captain_life(penguin, neck, walk, phase_of):
    """What keeps the captain alive in every cove frame: slow uneven breathing,
    squash and stretch on each waddle step and a stretch of surprise as the lid
    opens, blinks at irregular gaps, and a soft drift on his head and balance so
    nothing loops. `walk` is the hurry's time span; `phase_of(s)` its step phase."""
    base = penguin.scale.copy()
    for f in range(round(10.5 * FPS), round(25.0 * FPS)):
        s = f / FPS
        breath = 0.012 * math.sin(2 * math.pi * s / 3.1 + 0.6 * math.sin(s / 1.7))
        u = min(max((s - walk[0]) / (walk[1] - walk[0]), 0.0), 1.0)
        step = abs(math.sin(phase_of(s))) * (math.sin(math.pi * u) ** 0.5 if 0 < u < 1 else 0.0)
        squash = -0.035 * (1 - step)                                  # low on each foot plant
        d = s - (OPENS + 0.3)
        surprise = 0.06 * math.exp(-max(d, 0) * 3.0) * math.cos(9.0 * d) if d > 0 else 0.0
        z = 1 + breath + squash + surprise
        side = 1 / math.sqrt(z)                                       # volume kept
        penguin.scale = (base.x * side, base.y * side, base.z * z)
        penguin.keyframe_insert("scale", frame=f)
    eyes = [c for c in neck.children if c.name.startswith("eye")]
    rng = random.Random(SEED + 3)
    s = 10.5 + rng.uniform(0.6, 1.8)
    while s < 24.8:
        if not OPENS - 0.1 < s < OPENS + 1.3:                         # eyes wide through the surprise
            f = s * FPS
            for eye in eyes:
                open_ = eye.scale.z
                for df, k in ((-1, 1.0), (1, 0.12), (3, 1.0)):
                    eye.scale.z = open_ * k
                    eye.keyframe_insert("scale", index=2, frame=f + df)
                eye.scale.z = open_
        s += rng.uniform(2.0, 4.0)
    for obj, strength in ((neck, 0.035), (penguin, 0.012)):
        for fc in obj.animation_data.action.fcurves:
            if fc.data_path == "rotation_euler" and fc.array_index in (0, 2 if obj is neck else 0):
                mod = fc.modifiers.new("NOISE")
                mod.strength, mod.scale, mod.phase = strength, 28.0, 3.7 * fc.array_index + (1.3 if obj is neck else 0)


def sailor_reacts(root):
    """The sailor's comic beats in the catch: it ducks and squashes flat as the chest
    comes down beside it, jumps clean off the deck at the slam, lands squashed, then
    bounces three times for joy with a wiggle once the lid is open."""
    root.rotation_mode = "XYZ"
    home, heading = root.location.copy(), root.rotation_euler.z
    beats = [(LANDS - 0.1, 0.0, 1.0, 0.0), (LANDS + 0.05, 0.0, 0.86, 0.0), (LANDS + 0.2, 0.0, 0.84, 0.0),
             (LANDS + 0.3, 0.07, 1.08, 0.0), (LANDS + 0.42, 0.0, 0.88, 0.0), (LANDS + 0.56, 0.0, 1.0, 0.0)]
    beats.append((OPENS + 0.85, 0.0, 1.0, 0.0))                          # held still until then
    for k in range(3):                                                   # the celebration
        at = OPENS + 0.9 + 0.36 * k
        beats += [(at, 0.0, 0.92, 0.0), (at + 0.12, 0.045, 1.05, 0.22 * (-1) ** k), (at + 0.26, 0.0, 0.94, 0.0)]
    beats.append((OPENS + 2.1, 0.0, 1.0, 0.0))
    for t, lift, squash, wiggle in beats:
        root.location = home + Vector((0.0, 0.0, lift))
        root.scale = (1 / math.sqrt(squash), 1 / math.sqrt(squash), squash)
        root.rotation_euler = (0.0, 0.0, heading + wiggle)
        for path in ("location", "scale", "rotation_euler"):
            root.keyframe_insert(path, frame=t * FPS)


def captain_hurries(penguin, shoulders):
    """The captain in the last shot: off screen at the cut, he waddles aft along the
    port side deck, quick and rocking, flippers tucked in on the narrow deck, stops
    behind the sailor and turns to the chest; as the lid opens he jumps, flippers
    flapping, rocks back on his heels, then cocks his head. His keys start at LANDS."""
    scn = bpy.context.scene
    scn.frame_set(round(LANDS * FPS) - 1)
    neck = [c for c in penguin.children if c.name.startswith("captain_neck")][0]
    # Down the middle of the port side deck, 0.53 m between the cabin and the bulwark
    # (he is 0.42 m across), then out past the cabin's corner to stop behind the sailor.
    route = [Vector((-1.0, 0.885)), Vector((-2.2, 0.84))]
    legs = [(b - a).length for a, b in zip(route, route[1:])]
    stop = route[-1]
    to_chest = Vector(CHEST_AT) - stop
    turned = math.atan2(to_chest.y, to_chest.x) % (2 * math.pi)                # left turn from facing aft (pi)
    walk = (LANDS, OPENS + 0.5)
    penguin.rotation_mode = "XYZ"
    first = round(LANDS * FPS)
    penguin.keyframe_insert("location", frame=first - 1)
    penguin.keyframe_insert("rotation_euler", frame=first - 1)
    for f in range(first, round(25.0 * FPS)):
        s = f / FPS
        u = min(max((s - walk[0]) / (walk[1] - walk[0]), 0.0), 1.0)
        e = u * u * (3 - 2 * u)
        along = e * sum(legs)
        for k, leg in enumerate(legs):
            if along <= leg or k == len(legs) - 1:
                at = route[k].lerp(route[k + 1], min(along / leg, 1.0))
                break
            along -= leg
        stepping = math.sin(math.pi * u) ** 0.5 if u < 1 else 0.0                # waddle fades in and out
        phase = 2 * math.pi * 1.9 * (s - walk[0])
        turn = min(max((s - walk[1] + 0.15) / 0.35, 0.0), 1.0)
        turn = turn * turn * (3 - 2 * turn)
        lean = min(max((s - (OPENS + 0.9)) / 0.5, 0.0), 1.0)
        lean = lean * lean * (3 - 2 * lean)
        jump = 0.05 * math.exp(-((s - (OPENS + 0.42)) / 0.09) ** 2)               # the start at the lid
        penguin.location = (at.x, at.y, DECK + 0.01 + 0.012 * abs(math.sin(phase)) * stepping + jump)
        penguin.rotation_euler = (0.05 * math.sin(phase) * stepping, -0.1 * lean, math.pi + (turned - math.pi) * turn)
        penguin.keyframe_insert("location", frame=f)
        penguin.keyframe_insert("rotation_euler", frame=f)
    fc = penguin.animation_data.action.fcurves
    for c in fc:
        if c.data_path in ("location", "rotation_euler"):
            c.keyframe_points[0].interpolation = "CONSTANT"                      # no smear across the cut
    # His cutlass held in against his hip down the narrow deck, clear of the planking.
    cutlass = next(c for c in penguin.children if c.name.startswith("cutlass"))
    hip = cutlass.location.copy()
    for t_, inboard in ((first / FPS, 0.035), (walk[1], 0.035), (walk[1] + 0.3, 0.0)):
        cutlass.location = hip + Vector((0.0, inboard, 0.0))
        cutlass.keyframe_insert("location", frame=t_ * FPS)
    # Head: a cock to the side and a peek down once he has turned.
    for t_, roll, nod in ((OPENS + 0.6, 0.0, 0.0), (OPENS + 1.0, 0.28, 0.18), (OPENS + 2.2, 0.22, 0.24),
                          (25.0, 0.26, 0.22)):
        neck.rotation_euler = (roll, nod, 0)
        neck.keyframe_insert("rotation_euler", frame=t_ * FPS)
    # Flippers tucked in all the way down the narrow deck; a short flap as he jumps,
    # kept close to his sides (the bulwark is a hand's width off), then tucked again.
    for shoulder in shoulders:
        side = 1 if shoulder.location.y > 0 else -1
        for t_, lift in ((first / FPS, -0.4), (OPENS + 0.3, -0.4), (OPENS + 0.42, 0.25), (OPENS + 0.52, -0.15),
                         (OPENS + 0.62, 0.2), (OPENS + 0.8, -0.3)):
            shoulder.rotation_euler = (side * lift, 0, 0)
            shoulder.keyframe_insert("rotation_euler", frame=t_ * FPS)
    return walk


def fill(ship):
    """Warm bounce from the lagoon onto the aft cockpit, which the cabin shades;
    linked to the boat and crew only so it never lights the water itself."""
    data = bpy.data.lights.new("bounce", "AREA")
    data.energy, data.size, data.color = 120, 2.5, rgb("#FFB98A")[:3]
    light = bpy.data.objects.new("bounce", data)
    bpy.context.scene.collection.objects.link(light)
    light.location = (-4.2, -1.8, 1.6)
    light.rotation_euler = (Vector((-2.4, 0, DECK + 0.1)) - light.location).to_track_quat("-Z", "Y").to_euler()
    light.visible_glossy = False
    receivers = bpy.data.collections.new("bounce_receivers")
    for obj in ship.children_recursive:
        receivers.objects.link(obj)
    light.light_linking.receiver_collection = receivers


def main():
    s = scene.build("cove", start=FIRST, wet=0.0)
    ship = s["ship"]
    bob(ship)
    sky()
    lagoon()
    terrain()
    dress_beach()
    camp()
    gulls()
    dolphins()
    poly_haven()
    sea_life()
    haze()
    fill(ship)
    scene.fishing_line(s["angler_bodies"], WATERLINE, range(FIRST, LAST + 1))
    chest(ship)
    neck, shoulders = scene.captain_rig(s["captain"])
    captain_idle(s["captain"], neck, shoulders)
    walk = captain_hurries(s["captain"], shoulders)
    sailor_reacts(s["sailor"])
    captain_life(s["captain"], neck, walk, lambda t: 2 * math.pi * 1.9 * (t - walk[0]))
    cameras(s, ship)
    scene.setup_render(1920, 1080, 96)
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = FIRST, LAST
    sc.cycles.volume_bounces = 1
    scene.run(STILLS, FIRST, LAST, "cove")


main()
