"""Crew turnaround and clearance check.

Run from the repository root, after render/scripts/export_jumper.py:
    blender --background --factory-startup --python-exit-code 1 --python render/scripts/crew.py [-- --check]

Renders each crab front and back in a neutral studio on the deck to
review/crew-<crab>-<view>.png. With --check it also plays both scenes' tracks
and reports every overlap between robot, outfit, rod, captain and boat at every
fourth frame; it exits non-zero if any is found.
"""
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene  # noqa: E402
from scene import DECK, REVIEW, args  # noqa: E402


def studio():
    world = bpy.data.worlds.new("studio")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.6, 0.66, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.5
    bpy.context.scene.world = world
    for name, loc, energy in (("key", (-1.0, -3.5, 3.5), 260), ("fill", (-4.5, 2.5, 2.0), 90), ("rim", (-1.5, 3.0, 2.6), 160)):
        data = bpy.data.lights.new(name, "AREA")
        data.energy, data.size = energy, 1.5
        obj = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(obj)
        obj.location = loc
        obj.rotation_euler = (Vector((-2.4, 0, DECK)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()


def check():
    failed = False
    for name, frames in (("storm", range(0, 252, 4)), ("cove", range(0, 348, 4))):
        s = scene.build(name)
        boat = [o for o in s["boat"].values()]
        report = scene.check_clearance(scene.crew_groups(s), frames, boat)
        for pair, hits in report.items():
            failed = True
            print(f"OVERLAP {name}: {pair}: {sorted(hits)[:6]}")
        print(f"{name}: {len(report)} overlapping pairs over {len(frames)} frames")
    return failed


def main():
    if "--check" in args():
        sys.exit(1 if check() else 0)
    s = scene.build("stand")
    studio()
    scene.setup_render(1600, 1000, 96, blur=False)
    for name, at, heading in (("angler", scene.ANGLER_AT, scene.ANGLER_HEADING),
                              ("sailor", scene.SAILOR_AT, scene.SAILOR_HEADING)):
        centre = Vector((*at, DECK + 0.14))
        for view, turn, up in (("front", 0.55, 0.16), ("back", math.pi + 0.5, 0.42)):
            d = Vector((math.cos(heading + turn), math.sin(heading + turn), 0))
            cam = scene.shot(f"{name}_{view}", [(0, centre + d * 0.62 + Vector((0, 0, up)), centre, 55)],
                             focus=s[f"{name}_bodies"]["upper_shell_link"], fstop=5.6)
            bpy.context.scene.camera = cam
            scene.render(REVIEW / f"crew-{name}-{view}.png")


REVIEW.mkdir(exist_ok=True)
main()
