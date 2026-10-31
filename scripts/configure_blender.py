"""Sets the Blender preferences this project relies on and saves them.

Run with: blender --background --python-exit-code 1 --python scripts/configure_blender.py
Not with --offline-mode: that override makes use_online_access read-only.
"""
import bpy

prefs = bpy.context.preferences
# No extension repositories or update checks reach the network.
prefs.system.use_online_access = False
# .blend files never run embedded Python on open.
prefs.filepaths.use_scripts_auto_execute = False
bpy.ops.wm.save_userpref()

devices = []
cycles = prefs.addons.get("cycles")
if cycles:
    cprefs = cycles.preferences
    for backend in ("HIP", "OPTIX", "CUDA", "ONEAPI", "METAL"):
        try:
            cprefs.compute_device_type = backend
        except TypeError:
            continue
        cprefs.get_devices()
        gpus = [d for d in cprefs.devices if d.type == backend]
        if gpus:
            for d in cprefs.devices:
                d.use = d.type == backend
            devices = [f"{d.name} ({backend})" for d in gpus]
            bpy.ops.wm.save_userpref()
            break

print(f"blender {bpy.app.version_string}: online access off, script auto-run off, "
      f"cycles devices: {', '.join(devices) or 'CPU only'}")
