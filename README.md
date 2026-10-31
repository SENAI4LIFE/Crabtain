# Crabtain

Procedural film rendering with MuJoCo and Blender Cycles.

![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)
![Blender 4.5 LTS](https://img.shields.io/badge/Blender-4.5%20LTS-F5792A.svg?logo=blender&logoColor=white)
![Cycles](https://img.shields.io/badge/render-Cycles-F5792A.svg)
![MuJoCo](https://img.shields.io/badge/physics-MuJoCo-2F6FB3.svg)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)
![Windows](https://img.shields.io/badge/platform-Windows-0078D4.svg)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/readme-assets/figures/hero-dark.jpg">
  <img alt="The boat in the storm and the crew in the bay" src="docs/readme-assets/figures/hero-light.jpg">
</picture>

[![Download 16:9 MP4](https://img.shields.io/badge/Download-16%3A9%20MP4-2F6FB3.svg?style=for-the-badge)](https://github.com/SENAI4LIFE/Crabtain/releases/download/v1.0/crabtain-16x9.mp4)
[![Download 9:16 MP4](https://img.shields.io/badge/Download-9%3A16%20MP4-2F6FB3.svg?style=for-the-badge)](https://github.com/SENAI4LIFE/Crabtain/releases/download/v1.0/crabtain-9x16.mp4)

## Requirements

| Software | Version or use |
|---|---|
| Windows PowerShell and Git | Setup and source checkout |
| Blender | 4.5 LTS, Cycles and bundled FFmpeg |
| Python | 3.12, with the Windows `py` launcher |
| MuJoCo | 3.11 series, installed by setup |
| PyTorch CPU | 2.15.0; torchvision 0.29.2, pinned by setup |

Hardware: a Cycles-capable CPU or GPU, memory for the scenes and disk space for frames.

## Reproduction

Run from the repository root in Windows PowerShell, with Blender on `PATH`, in this order:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
. .\scripts\activate.ps1
blender --background --factory-startup --python-exit-code 1 --python assets/boat/import_benchy.py
python render/scripts/export_jumper.py
blender --background --factory-startup --python-exit-code 1 --python render/scripts/crew.py -- --check
blender --background --factory-startup --python-exit-code 1 --python render/scripts/storm.py -- --preroll --frames 0 251
blender --background --factory-startup --python-exit-code 1 --python render/scripts/cove.py -- --frames 252 599
python render/scripts/sim_inset.py
blender --background --factory-startup --python-exit-code 1 --python render/scripts/film.py
```

MuJoCo meshes and motion, inverse kinematics, boat motion fitted to waves and clearance checks feed procedural Cycles scenes.
`render/scripts/film.py` assembles the edit, recorded and synthesized sound, both formats and the loop.

## Security and privacy

Activation disables telemetry and online model access; setup disables Blender online access and embedded script auto-run. No personal data is required.
Assets use fixed URLs and SHA-256 checks; source and PyTorch versions are pinned. Other Python dependencies are installed without hash pinning.

## Credits

- Jumper, its models and its toolkit: [KingKong Robotics](https://github.com/KingKongRobotics/jumper), Apache License 2.0.
- Boat: [#3DBenchy](https://www.3dbenchy.com), CreativeTools, CC0 1.0.
- Bay props: [Cannon 01](https://polyhaven.com/a/cannon_01) and
  [Coast Land Rocks 03](https://polyhaven.com/a/coast_land_rocks_03), Poly Haven, CC0 1.0.
- Title: [Pirata One](https://fonts.google.com/specimen/Pirata+One), Google Fonts, SIL Open Font License 1.1;
  the inset label in [Inter](https://rsms.me/inter/), bundled with Blender, SIL Open Font License 1.1.
- Thunder: [Tonitrus](https://commons.wikimedia.org/wiki/File:Tonitrus.ogg) and
  [Rain and thunder](https://commons.wikimedia.org/wiki/File:Rain_and_thunder.ogg), Wikimedia Commons, public domain.
- Timber creaks: [Tree creaking](https://opengameart.org/content/tree-creaking), OpenGameArt, CC0 1.0.

## License

Apache-2.0. See [LICENSE](LICENSE).
