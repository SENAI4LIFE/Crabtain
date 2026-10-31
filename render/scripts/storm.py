"""The open-sea storm, film seconds 0 to 10.5: the boat riding high, steep seas
at night under slate cloud, streaked rain, spray, branching lightning.

Run from the repository root, after assets/boat/import_benchy.py and
render/scripts/export_jumper.py:
    blender --background --factory-startup --python-exit-code 1 --python render/scripts/storm.py [-- options]

Options (scene.run): --stills (default) renders the review stills to review/,
--only a,b limits them; --frames a b renders film frames a..b to
render/out/frames/; --samples n and --percent p override quality. --preroll
renders the PREROLL frames before frame 0 (shot 1's camera holding before its
push) to render/out/preroll/, for the chest in cove.py's last shot, then any --frames.

Shots:
1. 0.0-3.0 s  hook: push in low over the waves on the boat's starboard bow as a
   bolt silhouettes it and the captain in the door.
2. 3.0-7.0 s  handheld medium on the sailor, skidding about the rolling deck.
3. 7.0-10.5 s handheld close on the angler: nibbles, the strike at 9.0 s with a
   bolt overhead, then a whip pan up the line into the lightning (cut to cove).
Colors: slate sky, deep teal water, foam, lightning and lantern from the palette.
"""
import math
import random
import sys
import zlib
from pathlib import Path

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene  # noqa: E402
from scene import DECK, FPS, WATERLINE, Nodes, material, rgb  # noqa: E402

SLATE, DEEP_TEAL, FOAM, LIGHTNING = "#2F3742", "#1B3340", "#DCE6E8", "#DDE8FF"
SEED = 7
FIRST, LAST = 0, round(10.5 * FPS) - 1
PREROLL = 48
SETTLE = 0.12                                      # m below the still-water line, for weight in the waves
KEEL_MARGIN = 0.06       # m: the hull bottom stays at least this far under the wave
DECK_MARGIN = 0.05       # m: and the sea at least this far under the cockpit sole
WIND = Vector((-0.55, 0.35, 0.0))            # spray and rain drift, from the starboard bow
STRIKES = [(1.0, (-38, 46, 0), 1.0), (5.4, (60, 70, 0), 0.5), (9.05, (-30, -26, 0), 1.3), (10.05, (-22, -10, 0), 1.0)]
STILLS = [("s1-open", 0.0), ("s1-early", 0.33), ("s1-hook", 1.08), ("s1-rise", 0.7), ("s1-pass", 2.6), ("s1-shadow", 2.4), ("s2-sailor", 5.0), ("s3-angler", 8.3), ("s3-strike", 9.12), ("s3-whip", 10.13)]


def ocean():
    """High, steep, wind-driven seas, tiled out to the fog; time runs with the film."""
    size, repeat = 40.0, 7
    bpy.ops.mesh.primitive_plane_add(size=2, location=(-size * (repeat - 1) / 2, -size * (repeat - 1) / 2, WATERLINE))
    obj = bpy.context.active_object
    obj.name = "ocean"
    mod = obj.modifiers.new("ocean", "OCEAN")
    mod.geometry_mode, mod.spatial_size = "GENERATE", int(size)
    mod.resolution = mod.viewport_resolution = 18
    mod.repeat_x = mod.repeat_y = repeat
    mod.spectrum, mod.wave_scale, mod.choppiness = "PIERSON_MOSKOWITZ", 2.0, 1.9   # about 2 m seas: the boat rides them
    mod.wind_velocity, mod.wave_alignment, mod.wave_direction = 20.0, 0.5, math.radians(150)
    mod.random_seed = SEED
    mod.use_foam, mod.foam_layer_name, mod.foam_coverage = True, "foam", 0.55
    for f in (FIRST - 60, LAST + 60):
        mod.time = 2.0 + f / FPS
        obj.keyframe_insert(f'modifiers["ocean"].time', frame=f)
    for fc in obj.animation_data.action.fcurves:
        for k in fc.keyframe_points:
            k.interpolation = "LINEAR"
    bpy.ops.object.shade_smooth()

    # Whitecaps: crest height and the modifier's foam, broken by streaky noise
    # along the wind; dark troughs; rough foam over glassy water.
    mat = material("storm_water", DEEP_TEAL, 0.05)
    n = Nodes(mat)
    co = n.coord()
    z = n.xyz(co)[2]
    crest = n.math("MULTIPLY", n.math("SUBTRACT", z, 0.9), 0.7, clamp=True)
    foam_attr = n.new("ShaderNodeAttribute", attribute_name="foam")
    streak = n.noise(n.scaled(co, (0.6, 2.2, 1.0)), 1.6, 8.0, 0.65)
    caps = n.math("MULTIPLY", n.math("ADD", crest, n.math("MULTIPLY", foam_attr.outputs["Fac"], 0.8)),
                  n.math("MULTIPLY", n.math("SUBTRACT", streak, 0.35), 3.0, clamp=True), clamp=True)
    lace = n.math("MULTIPLY", n.math("GREATER_THAN", n.noise(co, 9.0, 6.0), 0.62), n.math("MULTIPLY", crest, 1.5, clamp=True))
    foam = n.math("MAXIMUM", caps, lace)
    trough = n.math("MULTIPLY", n.math("SUBTRACT", 0.0, z), 0.4, clamp=True)
    water = n.mix(trough, rgb(DEEP_TEAL), [c * 0.35 for c in rgb(DEEP_TEAL)[:3]] + [1])
    # The serpent's shadow after the first strike: its body, an S-curve 26 m long
    # tapering head to tail, gliding under the boat and undulating as it swims,
    # darkening the sea and dulling its sheen along its length.
    pos = n.xyz(n.new("ShaderNodeNewGeometry").outputs["Position"])
    keyed = {}
    for name in ("cx", "cy", "phase", "present"):
        keyed[name] = n.new("ShaderNodeValue").outputs[0]
    for s, cx, cy, phase in ((1.6, 22.0, -16.0, 0.0), (3.2, -18.0, 14.0, -9.0)):
        for name, v in (("cx", cx), ("cy", cy), ("phase", phase)):
            keyed[name].default_value = v
            keyed[name].keyframe_insert("default_value", frame=s * FPS)
    for s, v in ((1.6, 0.0), (1.9, 1.0), (2.9, 1.0), (3.2, 0.0)):
        keyed["present"].default_value = v
        keyed["present"].keyframe_insert("default_value", frame=s * FPS)
    dx, dy = n.math("SUBTRACT", pos[0], keyed["cx"]), n.math("SUBTRACT", pos[1], keyed["cy"])
    along = n.math("ADD", n.math("MULTIPLY", dx, -0.8), n.math("MULTIPLY", dy, 0.6))     # heading (-0.8, 0.6)
    side = n.math("ADD", n.math("MULTIPLY", dx, 0.6), n.math("MULTIPLY", dy, 0.8))
    taper = n.math("SQRT", n.math("SUBTRACT", 1.0, n.math("POWER", n.math("DIVIDE", along, 13.0), 2.0), clamp=True))
    head = n.math("MULTIPLY", n.math("ADD", 1.0, n.math("MULTIPLY", n.math("DIVIDE", along, 13.0), 0.35)), taper)
    width = n.math("MULTIPLY", head, 2.3)
    curve = n.math("MULTIPLY", n.math("SINE", n.math("ADD", n.math("MULTIPLY", along, 2 * math.pi / 14.0),
                                                     keyed["phase"])), 2.2)
    off = n.math("ABSOLUTE", n.math("SUBTRACT", side, curve))
    shade = n.math("MULTIPLY", n.math("DIVIDE", n.math("SUBTRACT", width, off), 0.8, clamp=True), keyed["present"])
    water = n.mix(n.math("MULTIPLY", shade, 0.8), water, [0.004, 0.008, 0.01, 1])
    n.bsdf.inputs["Specular IOR Level"].default_value = 0.7
    # Over the body the sea loses its sheen too: blend to a dark matte surface.
    dull = n.new("ShaderNodeBsdfDiffuse")
    dull.inputs["Color"].default_value = (0.002, 0.004, 0.005, 1)
    hide = n.new("ShaderNodeMixShader")
    n.link(n.math("MULTIPLY", shade, 0.85), hide.inputs["Fac"])
    n.link(n.bsdf.outputs[0], hide.inputs[1])
    n.link(dull.outputs[0], hide.inputs[2])
    n.link(hide.outputs[0], mat.node_tree.nodes["Material Output"].inputs["Surface"])
    n.link(n.mix(foam, water, rgb(FOAM)), n.bsdf.inputs["Base Color"])
    patches = n.math("MULTIPLY", n.noise(n.scaled(co, (0.05, 0.05, 0.05)), 1.0, 2.0), 0.12)
    n.link(n.math("ADD", n.math("ADD", 0.02, patches), n.math("MULTIPLY", foam, 0.7)), n.bsdf.inputs["Roughness"])
    n.bsdf.inputs["Subsurface Weight"].default_value = 0.0
    # Texture on top of the swell: wind ripples and fine capillary chop drifting with
    # the wind, and a sheen that varies in patches.
    drift = n.new("ShaderNodeMapping")
    n.link(co, drift.inputs["Vector"])
    for f, k in ((FIRST - PREROLL, 0.0), (LAST, 1.0)):
        drift.inputs["Location"].default_value = (WIND.x * 14.0 * k, WIND.y * 14.0 * k, 0.0)
        drift.inputs["Location"].keyframe_insert("default_value", frame=f)
    ripples = n.math("ADD", n.noise(drift.outputs[0], 4.0, 8.0),
                     n.math("MULTIPLY", n.noise(n.scaled(drift.outputs[0], (1.0, 2.6, 1.0)), 22.0, 4.0), 0.45))
    bump = n.new("ShaderNodeBump", Strength=0.45, Distance=0.05)
    n.link(ripples, bump.inputs["Height"])
    n.link(bump.outputs["Normal"], n.bsdf.inputs["Normal"])
    scene.assign(obj, mat)
    return obj


class Surface:
    """Water height under any point at any film frame. The sea repeats every tile,
    so a single-tile copy of the ocean at a matching offset (hidden from render)
    is sampled, with lookups wrapped into it."""

    def __init__(self, ocean):
        self.probe = ocean.copy()
        self.probe.data = ocean.data.copy()
        self.probe.animation_data_clear()
        bpy.context.scene.collection.objects.link(self.probe)
        mod = self.probe.modifiers["ocean"]
        mod.repeat_x = mod.repeat_y = 1
        self.size = mod.spatial_size
        self.probe.location = (0, 0, WATERLINE)
        self.probe.hide_render = True
        self.time = lambda f: 2.0 + f / FPS
        self.cache = {}

    def tree(self, frame):
        if frame not in self.cache:
            self.probe.modifiers["ocean"].time = self.time(frame)
            dg = bpy.context.evaluated_depsgraph_get()
            ev = self.probe.evaluated_get(dg)
            mesh = ev.to_mesh()
            verts = [v.co.copy() for v in mesh.vertices]
            polys = [p.vertices[:] for p in mesh.polygons]
            ev.to_mesh_clear()
            self.cache[frame] = BVHTree.FromPolygons(verts, polys)
        return self.cache[frame]

    def height(self, x, y, frame):
        half = self.size / 2
        u = (x + half) % self.size - half
        v = (y + half) % self.size - half
        loc = self.tree(int(frame)).ray_cast(Vector((u, v, 50)), Vector((0, 0, -1)), 200)[0]
        return loc.z + WATERLINE if loc is not None else WATERLINE


def ride(ship, surface):
    """Keys the ship's heave, roll and pitch from the water under its whole bottom:
    the lowest point of the hull every 0.3 m along it, on the keel and both bilges,
    where it is under the still-water line (the bow and transom above may lift).
    Each frame a plane is fitted (least squares) to the water heights under those
    points: its height, fore-and-aft and athwartships slopes give heave, pitch and
    roll, the hull SETTLE deeper than its still-water line; all three are low-passed
    for the hull's inertia, so the bow rides up the crests and settles into the
    troughs. Where a stretch of bottom would still show over a trough, the hull sinks
    just that much more (plus KEEL_MARGIN, at most 0.3 m), eased in and out."""
    frames = list(range(FIRST - PREROLL - 12, LAST + 13))
    hull = next(o for o in ship.children_recursive if o.name == "Benchy.hull")
    rest = ship.matrix_world.inverted() @ hull.matrix_world
    keel = {}
    for co in (rest @ v.co for v in hull.data.vertices):
        key = (round(co.x / 0.3), -1 if co.y < -0.3 else 1 if co.y > 0.3 else 0)
        if key not in keel or co.z < keel[key].z:
            keel[key] = co
    keel = [v for v in keel.values() if v.z < WATERLINE - 0.1]      # the wetted bottom; bow and transom may lift
    fit = np.column_stack([np.ones(len(keel)), [v.x for v in keel], [v.y for v in keel]])
    plane = np.array([np.linalg.lstsq(fit, [surface.height(v.x, v.y, f) for v in keel], rcond=None)[0]
                      for f in frames])
    heave = plane[:, 0] - WATERLINE - SETTLE
    pitch = -np.arctan(plane[:, 1])
    roll = np.arctan(plane[:, 2]) * 1.2 + 0.05 * np.sin(np.array(frames) / FPS * 2 * math.pi / 3.1)
    kernel = np.hanning(9)
    kernel /= kernel.sum()
    smooth = lambda a: np.convolve(np.pad(a, 4, mode="edge"), kernel, "valid")
    heave, pitch, roll = smooth(heave), smooth(pitch), np.clip(smooth(roll), -0.32, 0.32)
    sink = np.zeros(len(frames))
    for i, f in enumerate(frames):
        pose = Euler((roll[i], pitch[i], 0.0)).to_matrix()
        worst = max((pose @ v).z + heave[i] - surface.height(*(pose @ v).xy, f) for v in keel)
        sink[i] = min(max(worst + KEEL_MARGIN, 0.0), 0.3)
    widen = lambda a: np.max([np.roll(np.pad(a, 6, mode="edge"), k)[6:-6] for k in range(-6, 7)], axis=0)
    heave -= smooth(widen(sink))
    sunk = smooth(widen(sink))
    # But the sea must never stand over the cockpit sole: wherever a crest would, the
    # hull rides up on it instead (eased over the neighbouring frames), its bow or
    # stern lifting clear of the trough beside it for a moment, as a small boat does.
    sole = next(o for o in ship.children_recursive if o.name == "Benchy.deck_surface")
    rest = ship.matrix_world.inverted() @ sole.matrix_world
    deck = [rest @ v.co for v in sole.data.vertices][::max(1, len(sole.data.vertices) // 80)]
    lift = np.zeros(len(frames))
    for i, f in enumerate(frames):
        pose = Euler((roll[i], pitch[i], 0.0)).to_matrix()
        lift[i] = max(max(surface.height(*(pose @ v).xy, f) - ((pose @ v).z + heave[i]) + DECK_MARGIN
                          for v in deck), 0.0)
    lifted = smooth(widen(lift))
    heave += lifted
    gaps = [max((Euler((roll[i], pitch[i], 0.0)).to_matrix() @ v).z + heave[i]
                - surface.height(*(Euler((roll[i], pitch[i], 0.0)).to_matrix() @ v).xy, f) for v in keel)
            for i, f in enumerate(frames)]
    print(f"ride: {len(keel)} bottom points, worst bottom-to-wave {max(gaps):+.3f} m, extra sink up to "
          f"{sunk.max():.2f} m; sea kept off the deck on {int((lifted > 0.001).sum())} frames, lifted up to "
          f"{lifted.max():.2f} m: {[f for f, l in zip(frames, lifted) if l > 0.001]}")
    ship.rotation_mode = "XYZ"
    scene.bake_keys(ship, "location", frames, np.column_stack([np.zeros_like(heave), np.zeros_like(heave), heave]),
                    "BEZIER")
    scene.bake_keys(ship, "rotation_euler", frames, np.column_stack([roll, pitch, np.zeros_like(roll)]), "BEZIER")
    scene.save_motion("storm", frames, roll, pitch, heave)
    print(f"ride: heave {heave.min():.2f}..{heave.max():.2f} m, roll {np.degrees(roll).min():.0f}.."
          f"{np.degrees(roll).max():.0f} deg, pitch {np.degrees(pitch).min():.0f}..{np.degrees(pitch).max():.0f} deg")


def sky():
    """Low slate cloud with billows, darker overhead; the lightning keys `flash`."""
    world = bpy.data.worlds.new("storm")
    world.use_nodes = True
    tree = world.node_tree
    n = Nodes(world)
    bg = tree.nodes["Background"]
    vec = n.new("ShaderNodeNewGeometry").outputs["Incoming"]
    up = n.math("MULTIPLY", n.xyz(vec)[2], -1.0)
    billow = n.noise(n.scaled(vec, (1.0, 1.0, 3.0)), 2.6, 10.0, 0.62)
    gradient = n.math("MULTIPLY", n.math("ADD", up, 0.08), 3.0, clamp=True)
    low = [c * 1.5 for c in rgb(SLATE)[:3]] + [1]
    high = [c * 0.25 for c in rgb(SLATE)[:3]] + [1]
    base = n.mix(gradient, low, high)
    clouds = n.mix(n.math("MULTIPLY", n.math("SUBTRACT", billow, 0.42), 2.5, clamp=True), base,
                   [c * 1.9 for c in rgb(SLATE)[:3]] + [1])
    flash = tree.nodes.new("ShaderNodeValue")
    flash.name = "flash"
    # The flash lights the clouds the camera sees in full, the scene only a quarter.
    camera = n.new("ShaderNodeLightPath").outputs["Is Camera Ray"]
    reach = n.math("ADD", 0.25, n.math("MULTIPLY", camera, 0.75))
    lit = n.mix(n.math("MULTIPLY", n.math("MULTIPLY", flash.outputs[0], billow), reach), clouds, rgb(LIGHTNING), "ADD")
    n.link(lit, bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.5
    bpy.context.scene.world = world
    return flash.outputs[0]


def bolt(start, end, rng, depth=0, width=0.09):
    """Points of a jagged bolt by midpoint displacement, plus side branches."""
    pts = [Vector(start), Vector(end)]
    rough = (Vector(end) - Vector(start)).length * 0.18
    for _ in range(6):
        out = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            m = (a + b) / 2 + Vector((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 0.3))) * rough
            out += [m, b]
        pts, rough = out, rough * 0.55
    branches = []
    if depth < 2:
        for _ in range(rng.randint(2, 4) if depth == 0 else rng.randint(0, 2)):
            i = rng.randint(len(pts) // 6, len(pts) * 2 // 3)
            a = pts[i]
            span = (Vector(end) - Vector(start)).length * rng.uniform(0.2, 0.45) * (0.6 if depth else 1)
            b = a + Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), -rng.uniform(0.6, 1.2))).normalized() * span
            branches += bolt(a, b, rng, depth + 1, width * 0.5)
    return [(pts, width)] + branches


def lightning(sky_flash):
    """For each strike: a branching bolt that flickers for a few frames, the sky and
    a broad cold light flashing with it."""
    rng = random.Random(SEED)
    glow = material("bolt", LIGHTNING, 0.5, emission=80.0)
    data = bpy.data.lights.new("flash", "AREA")
    data.size, data.color, data.energy = 40, rgb(LIGHTNING)[:3], 0
    flash = bpy.data.objects.new("flash", data)
    bpy.context.scene.collection.objects.link(flash)
    flash.visible_glossy = False
    energy, sky = [(FIRST - 1, 0.0)], [(FIRST - 1, 0.0)]
    for t, ground, power in STRIKES:
        f0 = round(t * FPS)
        g = Vector(ground) + Vector((0, 0, WATERLINE))
        top = g + Vector((rng.uniform(-8, 8), rng.uniform(-8, 8), 55))
        parts = bolt(top, g, rng)
        objs = [scene.rope(f"bolt_{f0}", pts, w * (1 + power), glow) for pts, w in parts]
        pattern = [(f0 - 1, 0), (f0, 1), (f0 + 1, 0.25), (f0 + 2, 1), (f0 + 3, 0.6), (f0 + 5, 0)]
        for o in objs:
            # Seen, and seen in reflections, but the keyed flash light does the lighting.
            o.visible_shadow = o.visible_diffuse = o.visible_volume_scatter = False
            for f, v in [(FIRST - 1, 0)] + pattern + [(LAST + 1, 0)]:
                o.hide_render = v < 0.2
                o.keyframe_insert("hide_render", frame=f)
        energy += [(f, v * 3e4 * power) for f, v in pattern]
        sky += [(f, v * 0.45 * power) for f, v in pattern]
        if t == STRIKES[0][0]:
            flash.location = (g + top) / 2
    for f, v in energy:
        data.energy = v
        data.keyframe_insert("energy", frame=f)
    node = sky_flash.node
    for f, v in sky:
        node.outputs[0].default_value = v
        node.outputs[0].keyframe_insert("default_value", frame=f)
    flash.location = (-20, 10, 40)
    flash.rotation_euler = (Vector((0, 0, 0)) - flash.location).to_track_quat("-Z", "Y").to_euler()


def drop(name, radius, mat):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=radius, location=(0, -500, -500))
    obj = bpy.context.active_object
    obj.name = name
    scene.assign(obj, mat)
    return obj


def particles(emitter, name, count, start, end, life, speed, normal, size, instance, random_v=0.0,
              gravity=1.0, emit_from="FACE"):
    mod = emitter.modifiers.new(name, "PARTICLE_SYSTEM")
    ps = mod.particle_system
    ps.seed = zlib.crc32(name.encode()) % 1000          # the same drops in every run
    s = ps.settings
    s.count, s.frame_start, s.frame_end, s.lifetime = count, start, end, life
    s.emit_from, s.use_emit_random = emit_from, True
    s.normal_factor = normal
    s.factor_random = random_v
    s.effector_weights.gravity = gravity
    s.render_type, s.instance_object = "OBJECT", instance
    s.particle_size, s.size_random = size, 0.5
    s.use_rotations = False
    emitter.show_instancer_for_render = not emitter.name.endswith("_sheet")   # hide the bare sheets
    return s


def weather(boat):
    """Rain from a sheet above the boat, wind spray off the crests, splashes on the
    deck and spray thrown off the bow; a wind field drives the airborne water."""
    rain_mat = material("rain", FOAM, 0.05, emission=1.0)
    Nodes(rain_mat).bsdf.inputs["Alpha"].default_value = 0.75
    spray_mat = material("spray", FOAM, 0.4, emission=0.15)
    Nodes(spray_mat).bsdf.inputs["Alpha"].default_value = 0.7
    drop_obj = drop("raindrop", 0.005, rain_mat)
    drop_obj.scale = (1, 1, 5.0)
    mist = drop("spraydrop", 0.006, spray_mat)

    bpy.ops.mesh.primitive_plane_add(size=1, location=(-1, 0, 16))
    sheet = bpy.context.active_object
    sheet.name, sheet.scale = "rain_sheet", (36, 30, 1)
    sheet.rotation_euler = (math.pi, 0, 0)                      # normal down
    particles(sheet, "rain", 170000, FIRST - 60, LAST + 1, 30, 0, 9.0, 1.0, drop_obj, 0.08, 0.6)

    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, WATERLINE + 0.6))
    low = bpy.context.active_object
    low.name, low.scale = "spray_sheet", (40, 40, 1)
    particles(low, "spindrift", 30000, FIRST - 30, LAST + 1, 18, 0, 2.5, 1.2, mist, 0.8, 0.5)

    particles(boat["deck_surface"], "deck_splash", 18000, FIRST - 10, LAST + 1, 4, 0, 1.3, 0.5, mist, 0.7, 1.0)
    particles(boat["gunwale"], "rail_spray", 14000, FIRST - 10, LAST + 1, 14, 0, 2.2, 1.0, mist, 0.9, 1.0)

    bpy.ops.object.effector_add(type="WIND", location=(0, 0, 4))
    wind = bpy.context.active_object
    wind.field.strength, wind.field.flow = 70.0, 1.2
    wind.rotation_euler = WIND.to_track_quat("Z", "Y").to_euler()
    bpy.ops.object.effector_add(type="TURBULENCE", location=(0, 0, 4))
    gust = bpy.context.active_object
    gust.field.strength, gust.field.size = 8.0, 3.0


def fog():
    """Volumetric fog, patchy and thicker toward the horizon."""
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 9))
    vol = bpy.context.active_object
    vol.name, vol.scale = "fog", (220, 220, 18)
    mat = bpy.data.materials.new("fog")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.remove(nodes["Principled BSDF"])
    n = Nodes(mat)
    scatter = n.new("ShaderNodeVolumePrincipled")
    scatter.inputs["Color"].default_value = rgb("#56606C")
    co = n.coord()
    patches = n.noise(n.scaled(co, (1, 1, 2.5)), 6.0, 3.0)
    density = n.math("MULTIPLY", n.math("ADD", 0.3, patches), 0.012)
    n.link(density, scatter.inputs["Density"])
    n.link(scatter.outputs["Volume"], nodes["Material Output"].inputs["Volume"])
    scene.assign(vol, mat)


def moon_key():
    """A cold key through the cloud and a warm lantern-like rim from the cabin side."""
    data = bpy.data.lights.new("cloud_key", "SUN")
    data.energy, data.angle, data.color = 1.8, math.radians(10), rgb("#9FB2C8")[:3]
    key = bpy.data.objects.new("cloud_key", data)
    bpy.context.scene.collection.objects.link(key)
    key.rotation_euler = Euler((math.radians(55), 0, math.radians(-130)))


def crew_rim(s, ship):
    """Cold backlights on each crab from the side away from its camera, linked to
    the crew alone so the boat and sea keep their light; they separate the dark
    shells from the night deck in the close shots, from the cut to shot 2 on."""
    for crab, at, source in (("sailor", scene.SAILOR_AT, (-0.15, 0.75, 0.55)),
                             ("angler", scene.ANGLER_AT, (0.6, 0.35, 0.5))):
        receivers = bpy.data.collections.new(f"{crab}_rim_receivers")
        for obj in [s[crab]] + list(s[crab].children_recursive):
            receivers.objects.link(obj)
        data = bpy.data.lights.new(f"{crab}_rim", "AREA")
        data.energy, data.size, data.color = 18.0, 0.35, rgb("#B7CCE3")[:3]
        light = bpy.data.objects.new(f"{crab}_rim", data)
        bpy.context.scene.collection.objects.link(light)
        light.parent = ship
        light.location = Vector((at[0] + source[0], at[1] + source[1], DECK + source[2]))
        target = Vector((at[0], at[1], DECK + 0.12))
        light.rotation_euler = (target - light.location).to_track_quat("-Z", "Y").to_euler()
        light.visible_glossy = False
        light.light_linking.receiver_collection = receivers
        for f, energy in ((round(3.0 * FPS) - 1, 0.0), (round(3.0 * FPS), 18.0)):
            data.energy = energy
            data.keyframe_insert("energy", frame=f)
        data.animation_data.action.fcurves[0].keyframe_points[0].interpolation = "CONSTANT"


SLIDE_X, SLIDE_Y = (-0.16, 0.08), (-0.19, 0.19)   # m about its spot: clear of the transom, cabin, angler, bulwark


def sailor_slides(s, ship):
    """The sailor on the wet deck, a comic turn: the boat's own roll and pitch tip
    it about, so it slides downhill (light and slippery, the slope doubled), skids
    against its feet's grip, claws its way back toward its spot, gets stopped short
    at the cockpit's limits with a bump, its feet bouncing as they scrabble and its
    body spinning a little with the slide; its eyes go wide on the long slides and
    squeeze shut at each bump. As the serpent rises it does a double take."""
    root = s["sailor"]
    home, heading = root.location.copy(), root.rotation_euler.z
    fc = {i: ship.animation_data.action.fcurves.find("rotation_euler", index=i) for i in (0, 1)}
    frames = list(range(FIRST - PREROLL - 12, LAST + 13))
    dt, g = 1 / FPS, 9.81
    pos, vel = np.zeros(2), np.zeros(2)
    track, bumps = [], []
    for f in frames:
        roll, pitch = fc[0].evaluate(f), fc[1].evaluate(f)
        slope = 2.0 * g * np.array([math.sin(pitch), -math.sin(roll)])   # downhill on the deck
        push = slope - 7.0 * pos - 1.6 * vel                               # clawing back, its legs' drag
        speed = np.linalg.norm(vel)
        if speed > 0.01:
            push -= 0.6 * vel / speed                                      # kinetic friction
        elif np.linalg.norm(push) < 0.5:
            push[:] = 0.0                                                  # it holds
        vel += push * dt
        pos += vel * dt
        for k, (lo, hi) in enumerate((SLIDE_X, SLIDE_Y)):
            if not lo <= pos[k] <= hi:
                pos[k] = min(max(pos[k], lo), hi)
                if abs(vel[k]) > 0.12:
                    bumps.append(f)
                vel[k] *= -0.25
        track.append((pos.copy(), vel.copy()))
    t = np.array(frames) / FPS
    speed = np.array([np.linalg.norm(v) for _, v in track])
    fast = np.clip(speed / 0.35, 0.0, 1.0)
    # The double take: a glance at the serpent's flash, back, then a hard second look.
    look = np.interp(t, [0.95, 1.1, 1.3, 1.45, 1.6, 2.4, 2.8], [0, 0.35, 0.0, 0.0, 0.7, 0.7, 0.0])
    keys = []
    for (p, v), f, tt, ff, lk in zip(track, frames, t, fast, look):
        hop = 0.014 * abs(math.sin(2 * math.pi * 6.5 * tt)) * ff                   # feet scrabbling
        local = Vector((p[0], p[1], hop))
        keys.append((f, home + local, heading + 0.9 * v[1] - 0.5 * v[0] + lk))
    root.rotation_mode = "XYZ"
    scene.bake_keys(root, "location", frames, np.array([list(k[1]) for k in keys]), "BEZIER")
    scene.bake_keys(root, "rotation_euler", frames, np.array([[0.0, 0.0, k[2]] for k in keys]), "BEZIER")
    squint = np.ones(len(frames))
    for b in bumps:
        squint *= 1 - 0.8 * np.exp(-((np.array(frames) - b - 1) / 2.0) ** 2)
    wide = 1 + 0.4 * fast + 0.45 * np.exp(-((t - 1.62) / 0.2) ** 2)
    scene.eyes("sailor", frames, np.clip(wide * squint, 0.1, 1.6))
    sailor_mop(s, frames, [v for _, v in track])
    print(f"sailor: slides x {min(p[0] for p, _ in track):+.2f}..{max(p[0] for p, _ in track):+.2f} m, "
          f"y {min(p[1] for p, _ in track):+.2f}..{max(p[1] for p, _ in track):+.2f} m, {len(bumps)} bumps")


def sailor_mop(s, frames, vel):
    """A swab mop brandished head-up in the sailor's shut right claw like a banner:
    the handle stub just below the claw, the shaft leaning up and out ahead and to
    its right, the swab at the top with its rope strands hanging round the shaft.
    The strands trail behind every slide (swung opposite the slide's velocity a few
    frames late) and swing back. Nothing of it comes near the deck: the claw rides
    low, so a swab carried at deck level would drag through the planks."""
    bodies, root = s["sailor_bodies"], s["sailor"]
    bpy.context.scene.frame_set(FIRST)
    pad, tip = (bodies[n].matrix_world.translation for n in ("RF_palm_grip_insert_link", "RF_finger_grip_insert_link"))
    grip = root.matrix_world.inverted() @ ((pad + tip) / 2)
    axis = Vector((0.4, -0.4, 1.0)).normalized()               # up, leaning out ahead and to its right
    palm = bodies["RF_palm_link"]
    local = lambda v: palm.matrix_world.inverted() @ (root.matrix_world @ v)
    wood, rope = scene.paint("mop_handle", "#7A5A3A", 0.5), scene.cloth("mop_rope", "#BDB29A")
    top = grip + axis * 0.30
    scene.cylinder("mop_handle", local(grip - axis * 0.015), local(top + axis * 0.01), 0.0055, wood, palm, 16)
    head = scene.empty("mop_head", palm)                      # rigid in the claw, strands hanging plumb
    head.rotation_mode = "XYZ"
    head.matrix_world = root.matrix_world @ Matrix.Translation(top)
    rest = head.rotation_euler.copy()
    for k in range(16):
        a = 2 * math.pi * k / 16
        ring = Vector((0.016 * math.cos(a), 0.016 * math.sin(a), 0.0))
        scene.cylinder("mop_strand", ring, ring + Vector((0.012 * math.cos(a), 0.012 * math.sin(a), -0.065)),
                       0.0028, rope, head, 6)
    lag = 3
    for i, f in enumerate(frames):
        v = vel[max(i - lag, 0)]
        head.rotation_euler = (rest.x + float(np.clip(-1.4 * v[1], -0.6, 0.6)),
                               rest.y + float(np.clip(1.4 * v[0], -0.6, 0.6)), rest.z)
        head.keyframe_insert("rotation_euler", frame=f)
    print(f"mop: grip {grip.z * 100:.1f} cm above the deck, butt {(grip - axis * 0.015).z * 100:.1f} cm")


def captain_braces(s):
    """The captain in the starboard door through the storm: he leans against the
    boat's roll and pitch (half of each, so he stays upright as the deck tips),
    grips the door frame with his forward flipper, breathes and blinks."""
    penguin, ship = s["captain"], s["ship"]
    neck, shoulders = scene.captain_rig(penguin)
    heading, base = penguin.rotation_euler.z, penguin.scale.copy()
    grip = [sh for sh in shoulders if sh.location.y > 0][0]              # toward the bow, on the frame
    grip.rotation_euler = (1.0, 0, 0)
    grip.keyframe_insert("rotation_euler", frame=FIRST - PREROLL)
    sc = bpy.context.scene
    rng = random.Random(SEED + 5)
    blinks = set()
    t = rng.uniform(0.5, 1.5)
    while t * FPS < LAST:
        blinks.add(round(t * FPS))
        t += rng.uniform(2.0, 4.0)
    for f in range(FIRST - PREROLL, LAST + 1, 2):
        sc.frame_set(f)
        roll, pitch = ship.rotation_euler.x, ship.rotation_euler.y
        penguin.rotation_euler = (0.5 * pitch, -0.5 * roll, heading)
        breath = 1 + 0.014 * math.sin(2 * math.pi * f / FPS / 2.4)
        penguin.scale = (base.x / math.sqrt(breath), base.y / math.sqrt(breath), base.z * breath)
        penguin.keyframe_insert("rotation_euler", frame=f)
        penguin.keyframe_insert("scale", frame=f)
    for eye in [c_ for c_ in neck.children if c_.name.startswith("eye")]:
        open_ = eye.scale.z
        for f in sorted(blinks):
            for df, k in ((-1, 1.0), (1, 0.12), (3, 1.0)):
                eye.scale.z = open_ * k
                eye.keyframe_insert("scale", index=2, frame=f + df)
        eye.scale.z = open_
    # He sees it first: as the serpent rises he flings his free flipper out at it and
    # his head follows it across, then he grabs back on for the next sea.
    point = [sh for sh in shoulders if sh.location.y < 0][0]
    for t, lift in ((FIRST - PREROLL) / FPS, 0.0), (0.85, 0.0), (1.15, -1.45), (1.9, -1.3), (2.5, -1.4), (2.95, -0.15):
        point.rotation_euler = (lift, 0, 0)
        point.keyframe_insert("rotation_euler", frame=round(t * FPS))
    for t, yaw in (((FIRST - PREROLL) / FPS, 0.0), (0.85, 0.0), (1.2, 0.3), (2.6, -0.35), (3.2, 0.0)):
        neck.rotation_euler = (0, 0, yaw)
        neck.keyframe_insert("rotation_euler", frame=round(t * FPS))
    # A warm lamp just inside the door lights his front, linked to him alone, so the
    # wide shot reads him against the glow while the storm keeps its dark.
    receivers = bpy.data.collections.new("captain_lamp_receivers")
    for obj in [penguin] + list(penguin.children_recursive):
        receivers.objects.link(obj)
    data = bpy.data.lights.new("captain_lamp", "POINT")
    data.energy, data.shadow_soft_size, data.color = 9.0, 0.08, rgb("#FFB070")[:3]
    lamp = bpy.data.objects.new("captain_lamp", data)
    bpy.context.scene.collection.objects.link(lamp)
    lamp.parent = penguin
    lamp.location = (0.35, 0.0, 0.95)                # out in front of him (his +x) and above
    lamp.visible_glossy = False
    lamp.light_linking.receiver_collection = receivers
    for fc in neck.animation_data.action.fcurves:
        mod = fc.modifiers.new("NOISE")
        mod.strength, mod.scale, mod.phase = 0.05, 22.0, 2.3 * fc.array_index


SERPENT_LENGTH, SERPENT_WAVE, SERPENT_RISE = 24.0, 8.0, 2.2      # m: body, hump spacing, hump height
SERPENT_SWIM = 3.5                                                # m/s, the humps' travel along the body


def sea_monster():
    """A glimpse in the opening: a huge sea serpent swimming through the swell behind
    the boat, between it and the first bolt, so the flash at 1 s shows its rolling
    humps, spined back and raised head in silhouette before it sinks into the dark;
    a moment later its spined back cuts the surface right behind the hull, low and
    fast, as it passes under the boat.
    The body is a tapered tube on a travelling wave, keyed per frame."""
    skin = material("serpent", "#1E2B2A", 0.28)
    Nodes(skin).bsdf.inputs["Coat Weight"].default_value = 0.5            # wet
    eye_mat = material("serpent_eye", "#C9B24A", 0.2, emission=0.6)
    look = Vector((-0.69, 0.73, 0))                                       # from the hook camera past the boat
    across = Vector((0.73, 0.69, 0))
    centre = look * 26.0
    n = 48
    curve = bpy.data.curves.new("serpent", "CURVE")
    curve.dimensions, curve.bevel_depth, curve.bevel_resolution = "3D", 1.0, 4
    curve.use_fill_caps = True
    spline = curve.splines.new("POLY")
    spline.points.add(n - 1)
    body = bpy.data.objects.new("serpent", curve)
    bpy.context.scene.collection.objects.link(body)
    curve.materials.append(skin)
    head = scene.empty("serpent_head")
    scene.ellipsoid("serpent_skull", (0, 0, 0), (1.1, 0.75, 0.6), skin, head)
    scene.ellipsoid("serpent_snout", (1.1, 0, -0.12), (0.9, 0.45, 0.32), skin, head)
    jaw = scene.empty("serpent_jaw_hinge", head, (-0.1, 0, -0.3))           # opens with the groan
    scene.ellipsoid("serpent_jaw", (0.9, 0, -0.12), (0.95, 0.4, 0.18), skin, jaw, rot=(0, 0.12, 0))
    jaw.rotation_mode = "XYZ"
    for s_, opening in ((0.75, 0.0), (1.05, 0.38), (1.3, 0.3), (1.6, 0.05), (1.8, 0.0)):
        jaw.rotation_euler = (0, opening, 0)
        jaw.keyframe_insert("rotation_euler", frame=s_ * FPS)
    for side in (-1, 1):
        scene.ellipsoid("serpent_eye", (0.55, side * 0.55, 0.22), (0.12, 0.08, 0.09), eye_mat, head)
        scene.cylinder("serpent_horn", (-0.3, side * 0.35, 0.45), (-1.2, side * 0.6, 1.05), 0.12, skin, head,
                       vertices=8, radius2=0.0)
    spines = [scene.empty("serpent_spine_pivot") for _ in range(9)]
    for sp in spines:
        scene.cylinder("serpent_spine", (0, 0, 0), (0, 0, 0.9), 0.22, skin, sp, vertices=6, radius2=0.0)
    first, last = round(0.2 * FPS), round(3.1 * FPS)
    for f in range(first, last + 1):
        s = f / FPS
        u = min(max((s - 0.2) / 0.6, 0.0), 1.0)
        rise = u * u * (3 - 2 * u)
        sink = min(max((s - 1.7) / 0.6, 0.0), 1.0)
        base = WATERLINE - 2.6 + 2.3 * rise - 3.5 * sink * sink                # surfaces for the strike, then dives
        here, height = centre, SERPENT_RISE
        if s >= 2.15:                                                     # the pass: back only, by the hull
            v = min(max((s - 2.15) / 0.9, 0.0), 1.0)
            here, height = Vector((1.5, -4.2, 0)) + across * (-6.0 + 12.0 * v), 0.6          # off the starboard side, in view
            base = WATERLINE - 1.15 + 0.35 * math.sin(math.pi * v)
        pts = []
        for k in range(n):
            x = -SERPENT_LENGTH / 2 + SERPENT_LENGTH * k / (n - 1)          # tail to head
            swell = 0.55 + 0.45 * (SERPENT_LENGTH / 2 - x) / SERPENT_LENGTH            # stronger toward the tail
            wave = height * swell * math.sin(2 * math.pi * (x - SERPENT_SWIM * s) / SERPENT_WAVE)
            lead = min(max((s - 1.45 - (SERPENT_LENGTH / 2 - x) / 80.0) / 0.4, 0.0), 1.0)  # head first, tail under by the pass
            dive = 0.0 if s >= 2.15 else -3.5 * (lead * lead - sink * sink)
            lift = 1.2 * max(0.0, (x - SERPENT_LENGTH / 2 + 3.0) / 3.0) * (height > 1)   # the head rides high
            p = here + across * (x + SERPENT_SWIM * (s - 1.0) * 0.4) + Vector((0, 0, base + wave + lift + dive))
            pts.append(p)
            sp = spline.points[k]
            sp.co = (*p, 1.0)
            sp.radius = 0.95 * max(0.08, math.sin(math.pi * min((k + 0.5) / n * 1.15, 1.0))) ** 0.6
            sp.keyframe_insert("co", frame=f)
            sp.keyframe_insert("radius", frame=f)
        tip, neck = pts[-1], pts[-4]
        head.location = tip + (tip - neck).normalized() * 0.8
        sway = Euler((0.0, 0.0, 0.14 * math.sin(2 * math.pi * s / 1.9)))      # the head never locks rigid
        head.rotation_euler = ((tip - neck).to_track_quat("X", "Z").to_matrix() @ sway.to_matrix()).to_euler()
        head.keyframe_insert("location", frame=f)
        head.keyframe_insert("rotation_euler", frame=f)
        for j, sp in enumerate(spines):                                   # a ridge down the back, behind the neck
            k = n - 8 - j * 4
            a_, b_ = pts[k], pts[k + 1]
            sp.location = a_ + Vector((0, 0, 0.75 * spline.points[k].radius))
            sp.rotation_euler = Vector((0, 0, 1)).rotation_difference(Vector((0, 0, 1)) + (b_ - a_).normalized() * -0.5).to_euler()
            sp.keyframe_insert("location", frame=f)
            sp.keyframe_insert("rotation_euler", frame=f)
    # Water sheeting off its head as it breaks the surface (particles need a mesh).
    skull = [c for c in head.children if c.name.startswith("serpent_skull")][0]
    particles(skull, "serpent_spray", 1800, round(0.5 * FPS), round(2.2 * FPS), 16, 0, 2.6, 1.1,
              bpy.data.objects["spraydrop"], 0.9, 1.0)


def cameras(s, ship):
    sail = s["sailor_bodies"]["upper_shell_link"]
    angl = s["angler_bodies"]["upper_shell_link"]
    # 1. Hook: world camera low over the swell, pushing in on the starboard bow.
    ship_at = lambda t, local: ship_matrix(ship, t) @ Vector(local)
    cab = (0.4, -0.7, 2.6)
    c1 = scene.shot("s1_hook", [
        (0.0, ship_at(0.0, (13.0, -14.0, 1.6)), ship_at(0.0, cab), 32),
        (1.5, ship_at(1.5, (9.5, -10.5, 1.9)), ship_at(1.5, cab), 36),
        (3.0, ship_at(3.0, (6.5, -7.6, 2.3)), ship_at(3.0, cab), 40)], handheld=0.004, seed=1)
    scene.cut(c1, 0.0)
    # 2. Handheld medium on the sailor, from starboard inside the cockpit at crab height.
    sx, sy = scene.SAILOR_AT
    c2 = scene.shot("s2_sailor", [
        (3.0, (sx + 0.18, sy - 0.86, DECK + 0.34), (sx, sy, DECK + 0.13), 32),
        (7.0, (sx + 0.14, sy - 0.7, DECK + 0.3), (sx, sy, DECK + 0.12), 35)],
        parent=ship, handheld=0.014, focus=sail, fstop=2.8, seed=2)
    scene.cut(c2, 3.0)
    # 3. Handheld close on the angler from astern; whip pan up the line at the strike.
    ax, ay = scene.ANGLER_AT
    face = (ax - 0.05, ay, DECK + 0.13)
    c3 = scene.shot("s3_angler", [
        (7.0, (ax - 0.52, ay - 0.22, DECK + 0.24), face, 40),
        (9.0, (ax - 0.46, ay - 0.2, DECK + 0.23), face, 42),
        (9.55, (ax - 0.5, ay - 0.24, DECK + 0.25), (ax - 0.05, ay, DECK + 0.16), 42),
        (10.1, (ax - 0.5, ay - 0.24, DECK + 0.3), (ax - 4.0, ay - 2.5, DECK + 5.5), 30, -0.3),
        (10.5, (ax - 0.5, ay - 0.24, DECK + 0.32), (ax - 4.5, ay - 2.0, DECK + 7.5), 28, -0.35)],
        parent=ship, handheld=0.012, focus=angl, fstop=2.4, seed=3)
    scene.cut(c3, 7.0)
    # The whip racks to deep focus so the bolt it lands on stays crisp.
    for t, fstop in ((9.55, 2.4), (10.1, 22.0)):
        c3.data.dof.aperture_fstop = fstop
        c3.data.dof.keyframe_insert("aperture_fstop", frame=round(t * FPS))


def ship_matrix(ship, t):
    bpy.context.scene.frame_set(round(t * FPS))
    return ship.matrix_world.copy()


def main():
    s = scene.build("storm", start=FIRST, wet=1.0)
    ship, boat = s["ship"], s["boat"]
    sea = ocean()
    surface = Surface(sea)
    ride(ship, surface)
    flash = sky()
    lightning(flash)
    weather(boat)
    fog()
    sea_monster()
    moon_key()
    crew_rim(s, ship)
    captain_braces(s)
    sailor_slides(s, ship)
    pull = lambda f: min(max((f / FPS - 9.0) / 0.25, 0.0), 1.0)
    scene.fishing_line(s["angler_bodies"], lambda x, y, f: surface.height(x, y, f), range(FIRST - PREROLL, LAST + 1), pull)
    cameras(s, ship)
    scene.setup_render(1920, 1080, 96)
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = FIRST, LAST
    sc.view_settings.exposure = 0.8                 # night, but readable on a phone
    sc.cycles.volume_bounces, sc.cycles.volume_step_rate = 1, 2.0
    sc.cycles.max_bounces, sc.cycles.transparent_max_bounces = 8, 16
    preroll = "--preroll" in scene.args()
    # Rain, spindrift and spray baked from before any of it starts, in both runs, so the
    # pre-roll's last frame and the film's frame 0 carry the very same drops (the loop seam).
    early = FIRST - PREROLL - 60
    for obj in sc.objects:
        for mod in obj.modifiers:
            if mod.type == "PARTICLE_SYSTEM":
                mod.particle_system.point_cache.frame_start = early
    sc.frame_start = early
    sc.frame_set(early)
    bpy.ops.ptcache.bake_all(bake=True)
    sc.frame_start = FIRST - PREROLL if preroll else FIRST
    if preroll:
        a = scene.args()
        if "--samples" in a:
            sc.cycles.samples = int(a[a.index("--samples") + 1])
        if "--percent" in a:
            sc.render.resolution_percentage = int(a[a.index("--percent") + 1])
        (scene.OUT / "preroll").mkdir(parents=True, exist_ok=True)
        sc.camera = scene.active_camera(FIRST)
        for k, f in enumerate(range(FIRST - PREROLL, FIRST), start=1):
            sc.frame_set(f)
            scene.render(scene.OUT / "preroll" / f"{k:04d}.png")
        if "--frames" not in scene.args():
            return
    scene.run(STILLS, FIRST, LAST, "storm")


main()
