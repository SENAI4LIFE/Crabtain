"""Cuts the rendered frames into the film: a 1920x1080 master and a 1080x1920
crop, with the end-card title and a scored soundtrack.

Run from the repository root, after storm.py and cove.py rendered every film
frame (--frames 0 599) to render/out/frames/ and sim_inset.py wrote the end
card's MuJoCo inset to render/out/sim/:
    blender --background --factory-startup --python-exit-code 1 --python render/scripts/film.py

Over the storm-to-cove cut, where both shots look up at open sky, it lays a
sky plate it renders itself: clouds racing overhead and breaking up as the
slate storm light turns to dusk, a time-lapse that eases in and out. The bay's
first seconds then run a little fast and ease into real time (scene.cove_clock),
so the passage holds longer without moving any later frame.

Writes render/out/crabtain-16x9.mp4, render/out/crabtain-9x16.mp4 and the
soundtrack render/out/crabtain.wav, and the README heroes
docs/readme-assets/figures/hero-{dark,light}.jpg from two film frames. The sound combines noise and
sine tones with real thunder and timber creaks (public domain/CC0 recordings
fetched by scripts/setup.ps1): rain, wind and sea, a whoosh on the whip
pan, then the bay's surf, palms and gulls and a soft chord over the catch,
with a synthesized score (a storm drone, a squeezebox shanty over the bay, a
sting at the chest). After each flash a synthesized tear and a real thunder roll
arrive after a distance delay; the storm stays out of the bay until the final dive.
The boat creaks, thuds and knocks with its own motion, saved by storm.py
and cove.py. Servo whirs follow the crabs' limb speed in the exported tracks, on
the shots that show them.
"""
import json
import math
import sys
import wave
from pathlib import Path

import aud
import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene  # noqa: E402
from scene import CREAM, CUT, FPS, OUT, PASSAGE, ROOT, Nodes, cove_clock, rgb  # noqa: E402

LENGTH = 25.0                           # s
FRAMES = OUT / "frames"
RATE = 48000
STRIKES = [(1.0, 1.0), (5.4, 0.5), (9.05, 1.3), (10.05, 1.0)]      # storm.py: time, power
# Each strike's thunder: delay after the flash (distance / 343 m/s) and rumble length.
THUNDER = [(0.12, 4.0), (1.6, 5.0), (0.03, 4.5), (0.2, 3.5)]      # s
# Real thunder, public domain recordings from Wikimedia Commons (fetched by
# scripts/setup.ps1): per strike the recording and the stretch of it (s) to play.
SOUNDS = ROOT / "third_party" / "sounds"
TAKES = [("Rain_and_thunder.ogg", 2.6, 9.0), ("Tonitrus.ogg", 12.7, 18.0), ("Tonitrus.ogg", 18.8, 25.0),
         ("Tonitrus.ogg", 1.3, 6.5)]
# The title spans the loop: it surfaces in the chest's storm as the dive fills the
# frame, is fully up on the last frame and on the first, and fades as the storm opens.
TITLE_IN, TITLE_HOLD, FADE = 24.0, 0.6, 0.35     # s: it clears before the serpent's flash at 1 s
SEAM = 6                                # frames of pre-roll faded in over the end of the dive
HEROES = {"dark": 1.08, "light": 17.4}     # s: the storm hook, the crew in the cove
SERVO = [(0.0, 0.0), (3.0, 1.0), (10.5, 0.0), (15.0, 0.5), (19.0, 1.0), (21.0, 0.0)]   # s: whir gain per shot
# 9:16 crop: horizontal centre of interest as a fraction of the frame width, eased
# linearly within a shot and jumping on each cut (the key a frame before it).
# film frames: the sky plate fades in after the bolt, holds through the passage, dissolves
# out over the cove's first frames as they settle from the sky onto the boat
PLATE = (245, 249, CUT + PASSAGE + 4, CUT + PASSAGE + 14)
SLATE = "#2F3742"                       # storm.py's sky
DUSK = [(0.0, "#FFC07A"), (0.12, "#F2A65A"), (0.3, "#C8553D"), (0.55, "#8A4E6E"), (1.0, "#2E3462")]   # cove.py's
# In shot 2 it follows the sailor's glowing eyes as it skids about (tracked from the
# rendered frames), leaning toward the mop it holds up; in shot 6 it leans onto the chest's
# opening as the lid comes up and centres again for the dive into it.
CROP = [(0.0, 0.5), (2.96, 0.58), (3.0, 0.48), (3.5, 0.38), (3.83, 0.24), (4.17, 0.16), (4.5, 0.16),
        (4.83, 0.16), (5.17, 0.23), (5.5, 0.32), (5.83, 0.37), (6.17, 0.40), (6.5, 0.41), (6.96, 0.44),
        (7.0, 0.52), (10.46, 0.52), (10.5, 0.55), (16.0, 0.55), (18.96, 0.7), (19.0, 0.66), (20.96, 0.66),
        (21.0, 0.5), (22.2, 0.46), (23.2, 0.38), (24.2, 0.42), (24.75, 0.5)]


def band(n, lo, hi, rng, rolloff=None):
    """`n` samples of noise band-limited to lo..hi Hz, unit RMS, with optional treble roll-off."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / RATE)
    spec[(f < lo) | (f > hi)] = 0
    if rolloff is not None:
        spec *= np.cos(0.5 * np.pi * ramp(f, rolloff, hi)) ** 2
    x = np.fft.irfft(spec, n)
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12)


def ramp(t, a, b):
    return np.clip((t - a) / (b - a), 0.0, 1.0)


def swell(t, start, attack, decay):
    """A soft rise and fall, silent outside the phrase."""
    a = ramp(t, start, start + attack)
    d = 1 - ramp(t, start + attack, start + attack + decay)
    return a * a * (3 - 2 * a) * d * d * (3 - 2 * d)


def limb_speed(t):
    """Both crabs' summed limb-body speed (m/s) at times `t`, from jumper.json:
    the storm tracks cover film frames 0-251, the cove tracks the cove's clock from 252."""
    meta = json.loads((OUT / "jumper.json").read_text(encoding="utf-8"))
    limbs = [k for k, b in enumerate(meta["bodies"]) if b[:2] in ("LF", "RF", "LM", "RM", "LR", "RR")]
    speed = []
    for scene_name in ("storm", "cove"):
        total = 0
        for crab in ("angler", "sailor"):
            pos = np.array(meta["tracks"][f"{crab}_{scene_name}"])[:, limbs, :3]
            step = np.linalg.norm(np.diff(pos, axis=0, append=pos[-1:]), axis=2).sum(axis=1)
            total = total + step * meta["fps"]
        speed.append(total)
    speed = np.concatenate(speed)
    return np.interp(cove_clock(t * FPS), np.arange(len(speed)), speed)


def note(t, start, freq, length=0.9, partials=((1.0, 1.0), (3.9, 0.35), (9.2, 0.12))):
    """A struck, marimba-like note: a few inharmonic partials, each fading fast."""
    d = t - start
    on = d >= 0
    out = np.zeros_like(t)
    for ratio, gain in partials:
        out[on] += gain * np.sin(2 * np.pi * freq * ratio * d[on]) * np.exp(-d[on] * 3.2 * ratio ** 0.5 / length)
    return out


def reed(t, start, freq, length=0.55):
    """A squeezebox voice: two slightly detuned reeds rich in harmonics, a soft
    attack and a bellows swell, so the bay tune sounds like a shanty."""
    d = t - start
    on = (d >= 0) & (d < length + 0.25)
    out = np.zeros_like(t)
    env = np.minimum(d[on] / 0.04, 1.0) * np.exp(-np.maximum(d[on] - length, 0) / 0.08)
    env *= 0.85 + 0.15 * np.sin(2 * np.pi * 0.9 * d[on])
    for detune in (0.996, 1.004):
        for k in range(1, 9):
            out[on] += env * np.sin(2 * np.pi * freq * detune * k * d[on]) / k
    return out / 4


def thunder(t, at, power, delay, length, seed, rng):
    """One strike's thunder, `delay` s after its flash. A near strike tears first:
    a few sharp cracks over a quarter second. Then the rumble rolls on, a few swells
    as farther stretches of the channel arrive, losing its highs as it goes; a far
    strike only rumbles, darker. The timing comes from `seed`, the same in both
    ears; the noise from `rng`. Returns the sound and its envelope."""
    n = len(t)
    d = t - at - delay
    near = delay < 0.5
    timing = np.random.default_rng(seed)
    crack = np.zeros(n)
    if near:
        for k in range(5):
            off = timing.uniform(0.0, 0.22) if k else 0.0
            amp = timing.uniform(0.5, 1.0) * np.exp(-off / 0.2)
            crack += amp * np.where(d >= off, np.exp(-np.maximum(d - off, 0) / 0.025), 0)
        crack *= band(n, 180, 6000, rng) + 0.6 * band(n, 40, 180, rng)
    body = ramp(d, 0.0, 0.25 if near else 0.6) * np.exp(-np.maximum(d - 0.25, 0) / (length / 3))
    rolls = np.full(n, 0.45)
    for _ in range(int(length * 1.2)):
        centre, width = timing.uniform(0.3, 0.8 * length), timing.uniform(0.25, 0.7)
        rolls += timing.uniform(0.4, 1.0) * np.exp(-((d - centre) / width) ** 2)
    env = body * rolls
    tone = (band(n, 20, 110, rng) + 0.7 * band(n, 70, 300, rng) * np.exp(-np.maximum(d, 0) / 1.4)
            + (0.35 if near else 0.08) * band(n, 300, 1600, rng) * np.exp(-np.maximum(d, 0) / 0.5))
    fade = 1 - ramp(t, 11.0, 11.95)                                    # silent before the bay
    return power * (0.5 * crack + 0.7 * env * tone) * fade, power * env * fade


def recorded(t, at, power, delay, take, ear):
    """One strike's thunder from a recording: the take's stretch placed so its first
    swell lands `delay` s after the flash, levelled by the strike's power, its tail
    faded; a far strike (long delay) darker. Returns the sound and its envelope."""
    name, start, stop = take
    clip = aud.Sound(str(SOUNDS / name)).resample(RATE, False).data()
    clip = clip[:, min(ear, clip.shape[1] - 1)] if clip.ndim > 1 else clip
    clip = clip[int(start * RATE):int(stop * RATE)].astype(float)
    if delay > 0.5:                                                   # far: only the low roll carries
        spec = np.fft.rfft(clip)
        spec[np.fft.rfftfreq(len(clip), 1 / RATE) > 500] = 0
        clip = np.fft.irfft(spec, len(clip))
    blk = int(0.05 * RATE)
    level = np.sqrt(np.convolve(clip ** 2, np.ones(blk) / blk, "same"))
    onset = int(np.argmax(level > 0.3 * level.max()))
    clip = clip / (level.max() + 1e-9) * 0.5 * power
    clip[:int(0.01 * RATE)] *= np.linspace(0, 1, int(0.01 * RATE))
    clip[-int(0.8 * RATE):] *= np.linspace(1, 0, int(0.8 * RATE))
    out, env = np.zeros(len(t)), np.zeros(len(t))
    at0 = int((at + delay) * RATE) - onset
    lo, hi = max(at0, 0), min(at0 + len(clip), len(t))
    out[lo:hi] = clip[lo - at0:hi - at0]
    env[lo:hi] = level[lo - at0:hi - at0] / (level.max() + 1e-9) * power
    fade = 1 - ramp(t, 11.0, 11.95)                                    # including any recorded rain
    return out * fade, env * fade


def creak_grains():
    """Separate timber creaks at envelope rises, leaving out the quiet tail."""
    clip = aud.Sound(str(SOUNDS / "tree_creak.flac")).resample(RATE, False).data()
    clip = (clip.mean(axis=1) if clip.ndim > 1 else clip).astype(float)
    hop = int(0.01 * RATE)
    env = np.sqrt(np.mean(clip[:len(clip) // hop * hop].reshape(-1, hop) ** 2, axis=1))
    env = np.convolve(env, np.ones(3) / 3, "same")
    peak = env.max()
    rise = env - np.pad(env, (4, 0))[:-4]
    candidates = np.flatnonzero((rise > 0.12 * peak) & (env > 0.15 * peak)
                               & (rise >= np.roll(rise, 1)) & (rise > np.roll(rise, -1)))
    onsets = []
    for i in candidates:
        if onsets and i - onsets[-1] < 18:                            # one onset per timber stroke
            if rise[i] > rise[onsets[-1]]:
                onsets[-1] = i
        else:
            onsets.append(i)
    grains = []
    for i, following in zip(onsets, onsets[1:] + [len(env) + 4]):
        start, stop = max(i - 4, 0), min(following - 4, i + 86, len(env))
        active = np.flatnonzero(env[start:stop] > 0.08 * peak)
        if not len(active):
            continue
        stop = min(stop, start + active[-1] + 5)
        grain = clip[start * hop:stop * hop].copy() / (peak + 1e-9) * 0.5
        edge = min(int(0.03 * RATE), len(grain) // 2)
        grain[:edge] *= np.linspace(0, 1, edge)
        grain[-edge:] *= np.linspace(1, 0, edge)
        grains.append(grain)
    if not grains:
        raise ValueError("No creak onsets in tree_creak.flac")
    return grains


def boat_motion(t):
    """The boat's roll and pitch speed (rad/s) and heave velocity (m/s) at film
    times `t`, from the motion storm.py and cove.py saved, and which of the two
    scenes is on (1 storm, 0 cove; neither under the passage's sky), plus the
    storm's roll and pitch reversal times."""
    series, reversals = [], []
    for name in ("storm", "cove"):
        m = json.loads((OUT / f"motion-{name}.json").read_text(encoding="utf-8"))
        f = np.array(m["frames"], dtype=float)
        rates = [np.gradient(m[axis]) for axis in ("roll", "pitch")]
        turn = np.abs(rates).sum(axis=0)
        if name == "storm":
            for speed in rates:
                i = np.flatnonzero(np.diff(np.signbit(speed)))
                frames = f[i] - speed[i] * (f[i + 1] - f[i]) / (speed[i + 1] - speed[i])
                reversals.extend(frames[(frames >= 0) & (frames < CUT)] / FPS)
        series.append((f, turn * FPS, np.gradient(m["heave"]) * FPS))
    film = t * FPS
    storm = film < CUT
    clock = cove_clock(film)
    out = []
    for k in (1, 2):
        out.append(np.where(storm, np.interp(film, series[0][0], series[0][k]),
                            np.interp(clock, series[1][0], series[1][k])))
    shown = 1.0 - ramp(film, CUT, CUT + PASSAGE + 4) + ramp(film, CUT + PASSAGE + 4, CUT + PASSAGE + 14)
    return out[0] * shown, out[1] * shown, storm.astype(float), np.sort(reversals)


def wooden_ship(t, seed, rng):
    """The boat's own voice, driven by its motion: real timber creaks (a CC0
    recording) as it rolls and pitches, hard in the storm and only now and then at
    anchor; the bow thuds and splashes as it drops into a trough; water
    knocks along the hull as it heaves."""
    n = len(t)
    turn, lift, storm, reversals = boat_motion(t)
    level = np.where(storm > 0, np.clip(turn / (np.percentile(turn[storm > 0], 95) + 1e-9), 0, 1) ** 1.5,
                     0.35 * np.clip(turn / (np.percentile(turn[storm == 0], 95) + 1e-9), 0, 1) ** 2)
    out = np.zeros(n)
    grains = creak_grains()
    timing = np.random.default_rng(19)                                 # shared strokes and pitch in both ears
    events, last = [], -np.inf
    for at in reversals:
        i = int(at * RATE)
        if i >= n:
            continue
        strength = level[max(0, i - int(0.35 * RATE)):i + 1].max()      # strain approaching the reversal
        if strength < 0.05 or at - last < 1.4 - 1.1 * strength:
            continue
        events.append((at, strength))
        last = at
    at = 12.0 + timing.uniform(1.0, 3.0)
    while at < 23.6:
        events.append((at, 10 ** (-18 / 20)))                          # occasional creaks at anchor
        at += timing.uniform(5.0, 7.0)
    for at, strength in events:
        grain = grains[timing.integers(len(grains))]
        rate = timing.uniform(0.85, 1.15)
        stroke = np.interp(np.arange(0, len(grain) - 1, rate), np.arange(len(grain)), grain)
        edge = min(int(0.01 * RATE), len(stroke) // 2)
        stroke[:edge] *= np.linspace(0, 1, edge)
        stroke[-edge:] *= np.linspace(1, 0, edge)
        gain = 0.65 * strength * timing.uniform(0.8, 1.2)
        gain *= np.sqrt(1 + (2 * seed - 3) * timing.uniform(-0.2, 0.2))
        lo, hi = int(at * RATE), min(int(at * RATE) + len(stroke), n)
        if lo >= hi:
            continue
        fade = 1 - ramp(t[lo:hi], 11.5, 11.95) if at < 12 else 1 - ramp(t[lo:hi], 23.6, 25.0)
        out[lo:hi] += gain * stroke[:hi - lo] * fade
    # Slaps: where the hull stops dropping, as hard as it came down.
    falls = np.where((lift[:-1] < 0) & (lift[1:] >= 0))[0]
    thud, splash = np.zeros(n), np.zeros(n)
    for i in falls:
        speed = -lift[max(i - int(0.15 * RATE), 0):i + 1].min()
        if speed < 0.05:
            continue
        d = t - t[i]
        hit = min(speed / 0.6, 1.0) * storm[i]                           # at anchor the hull only creaks
        thud += hit * np.where(d >= 0, np.exp(-np.maximum(d, 0) / 0.07), 0)
        splash += hit * ramp(d, 0.0, 0.02) * np.exp(-np.maximum(d, 0) / 0.3)
    hull = 0.55 * thud * band(n, 40, 220, rng) + 0.16 * splash * band(n, 700, 5000, rng)
    knock = np.clip(np.abs(lift) / (np.percentile(np.abs(lift), 95) + 1e-9), 0, 1)
    hull += 0.05 * knock * band(n, 150, 700, rng) * storm
    out += hull * (1 - ramp(t, 11.7, 12.1))                             # no storm tails at anchor
    return out


def score(t):
    """The music, the same in both ears: a low D-minor drone and slow pulse under the
    storm with a low stinger on the strike; a D-major shanty over the bay on a
    squeezebox with plucked bass, stomps and claps; a held breath, then a rising
    arpeggio as the lid opens; and the drone again through the dive, so the loop
    returns to its start."""
    hz = lambda name: {"D2": 73.42, "A2": 110.0, "D3": 146.83, "A3": 220.0, "B3": 246.94, "D4": 293.66,
                       "E4": 329.63, "F#4": 369.99, "A4": 440.0, "B4": 493.88, "D5": 587.33, "F#5": 739.99}[name]
    storm = (1 - ramp(t, 10.0, 11.6)) + ramp(t, 23.6, 25.0)          # out over the passing sky
    drone = sum(np.sin(2 * np.pi * hz(n) * t) for n in ("D2", "A2", "D3")) / 3
    drone += 0.35 * sum(np.sin(2 * np.pi * hz(n) * t) for n in ("D4", "A4")) / 2      # heard on small speakers
    swell = 0.6 + 0.4 * np.sin(2 * np.pi * t / 5.0)
    pulse = np.zeros_like(t)
    for beat in np.arange(0.0, 10.5, 1.2):
        pulse += note(t, beat, hz("D2"), 0.5, ((1.0, 1.0), (2.0, 0.5), (3.0, 0.2)))
    out = storm * (0.16 * drone * swell + 0.12 * pulse * (1 - ramp(t, 9.0, 10.5)))
    stinger = sum(reed(t, 9.05, hz(n), 0.7) for n in ("D2", "A2", "D3"))
    d = t - 9.05
    sub = np.where(d >= 0, np.sin(2 * np.pi * 42 * d) * np.exp(-np.maximum(d, 0) / 0.35), 0)
    out += (0.22 * stinger + 0.25 * sub) * (1 - ramp(t, 11.6, 12.1))
    beat = 0.53                                                        # 113 bpm over the bay
    tune = ["D5", "F#5", "A4", "B4", "A4", "F#4", "E4", "D4", "E4", "F#4", "A4", "B4", "D5", "B4", "A4", "F#4"]
    bass = ["D3", "A2", "D3", "A2", "B3", "A3", "D3", "A2"]
    bay = np.zeros_like(t)
    start = 12.2                                                       # as the bay settles
    for k, name in enumerate(tune):
        bay += reed(t, start + k * beat, hz(name), 0.45)
    for k, name in enumerate(bass):
        bay += 0.7 * note(t, start + k * 2 * beat, hz(name), 1.2, ((1.0, 1.0), (2.0, 0.5), (3.0, 0.2)))
    rng = np.random.default_rng(7)
    hits = np.zeros_like(t)
    for k in range(16):                                                # stomp on 1 and 3, clap on 2 and 4
        at = start + k * beat
        d = t - at
        env = np.where((d >= 0) & (d < 0.12), np.exp(-np.maximum(d, 0) / (0.05 if k % 2 == 0 else 0.02)), 0)
        hits += env * (np.sin(2 * np.pi * 70 * d) if k % 2 == 0 else rng.standard_normal(len(t)) * 0.5)
    out += (0.16 * bay + 0.08 * hits) * ramp(t, 11.9, 12.3) * (1 - ramp(t, 20.6, 21.3))
    for k, name in enumerate(("D4", "F#4", "A4", "D5", "F#5")):          # the lid: ta-da
        out += 0.1 * note(t, 21.8 + k * 0.09, hz(name), 1.4)
        out += 0.06 * reed(t, 21.8 + k * 0.09, hz(name), 0.6 - k * 0.09)
    return out


def channel(seed):
    rng = np.random.default_rng(seed)
    n = int(LENGTH * RATE)
    t = np.arange(n) / RATE
    # The storm rises again out of the chest and is at full by the loop point.
    storm = 1 - ramp(t, 10.4, 11.4) + ramp(t, 23.6, 25.0) ** 2     # gone under the passing sky; back in the dive
    cove = ramp(t, 11.15, 11.9) * (1 - ramp(t, 23.6, 25.0))    # leads the bay's picture by about 8 frames
    out = 0.7 * storm * (0.12 * band(n, 2500, 9000, rng)                 # beds sit under the action
                   + 0.2 * band(n, 200, 900, rng) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.17 * t + seed))
                   + 0.28 * band(n, 25, 140, rng) * (0.7 + 0.3 * np.sin(2 * np.pi * 0.11 * t)))
    peals = np.zeros(n)
    for k, ((at, power), (delay, length), take) in enumerate(zip(STRIKES, THUNDER, TAKES)):
        crack, _ = thunder(t, at, power, delay, length, 100 + k, rng)      # the synthesized tear, under the real roll
        sound, env = recorded(t, at, power, delay, take, seed - 1)
        out += 0.35 * crack + sound
        peals = peals + env
    out += wooden_ship(t, seed, rng)
    # The serpent: a deep groan under the first thunder, then a surge as it passes under.
    groan = ramp(t, 0.7, 1.1) * (1 - ramp(t, 1.9, 2.6))
    pitch_hz = 48 - 14 * ramp(t, 0.8, 2.4)
    phase = 2 * np.pi * np.cumsum(pitch_hz) / RATE
    out += 0.22 * groan * (np.sin(phase) + 0.5 * np.sin(2 * phase + 0.3) + 0.25 * np.sin(3 * phase + seed))
    out += 0.18 * np.exp(-((t - 2.4) / 0.45) ** 2) * band(n, 60, 600, rng)
    up, down = np.exp(-((t - 9.85) / 0.2) ** 2), np.exp(-((t - 11.75) / 0.18) ** 2)    # the whip up, the whip down
    sky = 1 - ramp(t, 11.7, 12.1)
    out += sky * (0.22 * up + 0.14 * down) * band(n, 400, 3500, rng)
    out += sky * 0.12 * np.exp(-((t - 11.0) / 0.55) ** 2) * band(n, 150, 1400, rng)   # wind under the passing sky
    s = limb_speed(t)
    s = np.clip(s / (np.percentile(s, 95) + 1e-9), 0.0, 1.0)
    phase = 2 * np.pi * np.cumsum(650 + 700 * s) / RATE
    whir = np.sin(phase) + 0.35 * np.sin(2 * phase + seed) + 0.25 * band(n, 1800, 5000, rng)
    gain = np.interp(t, *zip(*SERVO))
    out += 0.07 * whir * s ** 1.5 * gain
    # The bay: spaced washes and short breezes, timed together in both ears.
    wash = np.zeros(n)
    for at, decay in ((12.1, 1.9), (17.0, 2.2), (22.4, 1.7)):
        wash += swell(t, at, 0.6, decay)
    rustle = sum(swell(t, at, 0.25, 0.65) for at in (13.0, 18.0, 23.0))
    water = band(n, 150, 1800, rng, rolloff=650)
    lap = band(n, 60, 180, rng)
    foam = band(n, 700, 1800, rng, rolloff=900)
    palms = band(n, 3500, 9000, rng)
    out += cove * (wash * (0.07 * water + 0.012 * wash * foam)
                   + 0.0015 * lap + 0.003 * rustle * palms)             # quiet gaps; palms well below the water
    pitch = np.zeros(n)
    voice = np.zeros(n)
    for call in (12.7, 15.1, 17.8, 21.5):
        for syllable in (0.0, 0.34):
            d = t - call - syllable
            on = (d >= 0) & (d < 0.28)
            pitch[on] = 2400 - 3000 * d[on]
            voice[on] = np.sin(np.pi * d[on] / 0.28)
    phase = 2 * np.pi * np.cumsum(pitch) / RATE
    out += 0.05 * voice * (np.sin(phase) + 0.4 * np.sin(2 * phase))
    chord = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.18, 329.63, 440.0)) / 4
    out += 0.3 * chord * ramp(t, 19.0, 21.5) * (1 - ramp(t, 23.4, 24.6))
    out += score(t) * (1 - 0.5 * np.clip(peals, 0, 1))                       # the music ducks under thunder
    # Spot effects: the reel whirring on the strike, the campfire crackling over the bay,
    # the captain's footsteps as he hurries aft, the sailor's blip at the reveal.
    whir = ramp(t, 9.05, 9.1) * (1 - ramp(t, 9.45, 9.6)) * (0.6 + 0.4 * np.sign(np.sin(2 * np.pi * 42 * t)))
    out += 0.07 * whir * band(n, 1500, 4500, rng)
    clicks = np.zeros(n)
    fire = np.random.default_rng(8)
    levels = rng.uniform(0.3, 1.0, 128)[64:73]                           # keep later effects on their original noise sequence
    for at, level in zip(12.4 + np.arange(9) + fire.uniform(-0.3, 0.3, 9), levels):
        k = int(at * RATE)
        clicks[k:k + 240] += np.exp(-np.arange(min(240, n - k)) / 40.0) * level
    out += 0.02 * np.convolve(clicks, np.ones(3) / 3, "same") * band(n, 1500, 7000, rng)
    for step in np.arange(21.0, 22.3, 1 / 3.8):
        d = t - step
        out += 0.06 * np.where((d >= 0) & (d < 0.12), np.exp(-np.maximum(d, 0) / 0.035), 0) * band(n, 60, 400, rng)
    d = t - 22.15
    blip = (d >= 0) & (d < 0.14)
    out[blip] += 0.05 * np.sin(2 * np.pi * (1200 * d[blip] + 2200 * d[blip] ** 2))
    # The catch: the chest lands, the lid creaks, coins shimmer, then the dive.
    d = t - 21.22
    out += 0.5 * np.where(d >= 0, np.exp(-np.maximum(d, 0) / 0.09), 0) * band(n, 40, 220, rng)
    creak = ramp(t, 21.8, 21.85) * (1 - ramp(t, 22.25, 22.4))
    grain = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * (14 + 10 * ramp(t, 21.8, 22.4)) * t))
    out += 0.16 * creak * grain * band(n, 350, 1400, rng)
    shimmer = sum(np.sin(2 * np.pi * f * t + seed) for f in (3520.0, 4186.0, 5274.0, 6272.0)) / 4
    out += 0.07 * shimmer * np.where(t >= 22.2, np.exp(-np.maximum(t - 22.2, 0) / 0.7), 0)
    d = t - 24.1
    out += 0.3 * (ramp(d, 0.0, 0.15) * np.exp(-np.maximum(d - 0.15, 0) / 1.2)) * band(n, 20, 160, rng)
    out *= 1 - 0.85 * ramp(t, 21.42, 21.5) * (1 - ramp(t, 21.74, 21.8))    # a held breath before the lid
    edge = int(0.004 * RATE)                                                  # click-free loop point
    out[:edge] *= np.linspace(0, 1, edge)
    out[-edge:] *= np.linspace(1, 0, edge)
    return out


def k_weighted(x):
    """ITU-R BS.1770 K-weighting at 48 kHz, applied in the frequency domain."""
    z = np.exp(-2j * np.pi * np.fft.rfftfreq(len(x), 1 / RATE) / RATE)
    biquad = lambda b, a: (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)
    h = biquad((1.53512485958697, -2.69169618940638, 1.19839281085285), (1.0, -1.69065929318241, 0.73248077421585))
    h *= biquad((1.0, -2.0, 1.0), (1.0, -1.99004745483398, 0.99007225036621))
    return np.fft.irfft(np.fft.rfft(x) * h, len(x))


def loudness(channels):
    """Integrated loudness in LUFS: 400 ms blocks at 75 % overlap, gated at
    -70 LUFS and then 10 LU under the ungated mean."""
    weighted = [k_weighted(c) for c in channels]
    size, hop = int(0.4 * RATE), int(0.1 * RATE)
    power = np.array([sum(np.mean(w[i:i + size] ** 2) for w in weighted)
                      for i in range(0, len(weighted[0]) - size + 1, hop)])
    lufs = lambda p: -0.691 + 10 * np.log10(np.mean(p) + 1e-12)
    gated = power[-0.691 + 10 * np.log10(power + 1e-12) > -70]
    return lufs(gated[-0.691 + 10 * np.log10(gated + 1e-12) > lufs(gated) - 10])


def true_peak(channels):
    """Sample peak of a 4x oversampled copy, in dBTP."""
    up = lambda x: np.fft.irfft(np.fft.rfft(x), len(x) * 4) * 4
    return 20 * np.log10(max(np.abs(up(c)).max() for c in channels))


def limit(channels, ceiling):
    """Smooth peak limiter: per 5 ms block the gain that keeps it under `ceiling`,
    held across neighbouring blocks and eased, so transients duck without clicks."""
    block = int(0.005 * RATE)
    n = len(channels[0]) // block * block
    peak = np.max([np.abs(c[:n]).reshape(-1, block).max(axis=1) for c in channels], axis=0)
    g = np.minimum(1.0, ceiling / (peak + 1e-12))
    g = np.min([np.roll(g, k) for k in range(-3, 4)], axis=0)
    g = np.convolve(np.pad(g, 3, mode="edge"), np.ones(7) / 7, "valid")
    g = np.interp(np.arange(len(channels[0])), np.arange(len(g)) * block + block / 2, g)
    return [c * g for c in channels]


def soundtrack(path):
    """-14 LUFS integrated with the true peak at most -1 dBTP: normalize, then
    limit the thunder transients, a few rounds."""
    chans = [channel(1), channel(2)]
    for _ in range(4):
        chans = [c * 10 ** ((-14.0 - loudness(chans)) / 20) for c in chans]
        chans = limit(chans, 10 ** (-1.6 / 20))
    if true_peak(chans) > -1.0:
        chans = [c * 10 ** ((-1.0 - true_peak(chans)) / 20) for c in chans]
    left, right = chans
    print(f"soundtrack: {loudness([left, right]):.1f} LUFS, {true_peak([left, right]):.1f} dBTP")
    pcm = (np.clip(np.column_stack([left, right]), -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


def strips(se):
    return se.strips if hasattr(se, "strips") else se.sequences


PIRATA = ROOT / "third_party" / "fonts" / "PirataOne-Regular.ttf"   # fetched by scripts/setup.ps1


STORM_LIGHT, SUNSET_RIM = "#929DA7", "#E0965A"     # sampled: the storm's highlights, the bay's warm rim
METALS = {"brass": ("#B98B3C", "#2E2414"), "gold": ("#D4A74E", "#3A2A12"),     # metal, patina
          "bronze": ("#9C6B3A", "#1F2A26")}
TITLE_W, TITLE_H, GLINT = 2000, 720, 14     # px; frames of the lightning glint sweeping the title


def title_card(style="gold", out=None, frames=None):
    """Renders the CRABTAIN title to render/out/title/ as an RGBA sequence: the word
    in Pirata One cut from salt-weathered brass (bevelled, dark patina in the
    recesses, pitted), on a twisted rope that runs out past the last letter and
    drops into a fishhook. Lit cold from above like the storm with a warm sunset
    rim behind, so it sits in both scenes; frame 0 is at rest, frames 1-GLINT are
    a lightning glint sweeping across it. Rendered on the CPU, small."""
    brass, patina = METALS[style]
    sc = bpy.context.scene                     # the startup scene, emptied
    for obj in list(sc.objects):
        bpy.data.objects.remove(obj)
    scene.setup_render(TITLE_W, TITLE_H, 48, transparent=True, blur=False)
    sc.cycles.device, sc.cycles.denoising_use_gpu = "CPU", False      # the GPU is busy with frames
    world = bpy.data.worlds.new("title_world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.012, 0.016, 0.02, 1)
    sc.world = world
    metal = bpy.data.materials.new("title_brass")
    metal.use_nodes = True
    n = Nodes(metal)
    pits = n.noise(n.scaled(n.coord(), (40.0, 40.0, 40.0)), 1.0, 6.0)
    crevice = n.new("ShaderNodeAmbientOcclusion", Distance=0.06, only_local=True).outputs["AO"]
    wear = n.math("MULTIPLY", n.math("POWER", crevice, 2.2), n.math("ADD", 0.75, n.math("MULTIPLY", pits, 0.3)), clamp=True)
    n.link(n.mix(wear, rgb(patina), rgb(brass)), n.bsdf.inputs["Base Color"])
    n.bsdf.inputs["Metallic"].default_value = 1.0
    n.link(n.math("ADD", 0.22, n.math("MULTIPLY", n.math("SUBTRACT", 1.0, wear), 0.45)), n.bsdf.inputs["Roughness"])
    bump = n.new("ShaderNodeBump", Strength=0.25)
    n.link(pits, bump.inputs["Height"])
    n.link(bump.outputs["Normal"], n.bsdf.inputs["Normal"])
    word = bpy.data.curves.new("title_word", "FONT")
    word.body, word.font = "CRABTAIN", bpy.data.fonts.load(str(PIRATA), check_existing=True)
    word.align_x, word.align_y, word.size, word.space_character = "CENTER", "CENTER", 1.0, 1.04
    word.extrude, word.bevel_depth, word.bevel_resolution = 0.05, 0.018, 3
    title = bpy.data.objects.new("title_word", word)
    sc.collection.objects.link(title)
    title.data.materials.append(metal)
    bpy.context.view_layer.update()
    half = title.dimensions.x / 2
    # A slow three-strand lay, tapered and whipped over a shallow sag.
    height, span = title.dimensions.y, 1.4 * half
    strands = []
    for k in range(5):
        pts, radii = [], []
        steps = 240 if k < 3 else 64
        for i in range(steps + 1):
            t = i / steps
            u = t if k < 3 else (0.012 + 0.04 * t) * height / span
            if k == 4:
                u = 1 - u
            taper = 0.44 + 0.56 * math.sin(math.pi * u)
            x, y = span * (u - 0.5), -0.44 - 0.06 * height * 4 * u * (1 - u)
            slope = 0.24 * height * (2 * u - 1)
            normal = math.hypot(span, slope)
            a = 2 * math.pi * (5 * u + k / 3 if k < 3 else 4 * t)
            r = height * (0.021 * taper if k < 3 else 0.0395 * taper + 0.003)
            pts.append((x - slope / normal * r * math.cos(a),
                        y + span / normal * r * math.cos(a), r * math.sin(a)))
            radii.append(taper if k < 3 else 1.0)
        name = f"title_rope{k}" if k < 3 else f"title_whipping{k - 3}"
        strands.append((name, pts, radii, height * (0.0185 if k < 3 else 0.0045)))
    # An open hook: eye, straight shank, round bend and a tapered point.
    h = 0.22 * height
    end_x, end_y, z = span / 2, -0.44, 0.012 * height
    hook = [(end_x, end_y - 0.28 * h, z)]
    hook += [(end_x + 0.25 * h * (1 + math.cos(a)),
              end_y - 0.66 * h + 0.25 * h * math.sin(a), z)
             for a in (math.pi + 1.2 * math.pi * i / 32 for i in range(33))]
    hook.append((end_x + 0.39 * h, end_y - 0.43 * h, z))
    radii = [1.0] * len(hook)
    radii[-3:] = [0.85, 0.55, 0.06]
    strands.append(("title_hook", hook, radii, 0.045 * h))
    for name, pts, radii, depth in strands:
        curve = bpy.data.curves.new(name, "CURVE")
        curve.dimensions, curve.bevel_depth, curve.bevel_resolution = "3D", depth, 3
        curve.use_fill_caps = True
        spline = curve.splines.new("POLY")
        spline.points.add(len(pts) - 1)
        for pt, co, radius in zip(spline.points, pts, radii):
            pt.co, pt.radius = (*co, 1.0), radius
        rope = bpy.data.objects.new(name, curve)
        sc.collection.objects.link(rope)
        rope.data.materials.append(metal)
    # The eye overlaps the bound end and joins the top of the shank.
    spline = curve.splines.new("POLY")
    spline.points.add(31)
    spline.use_cyclic_u = True
    for i, pt in enumerate(spline.points):
        a = 2 * math.pi * i / 32
        pt.co = (end_x + 0.14 * h * math.cos(a),
                 end_y - 0.14 * h + 0.14 * h * math.sin(a), z, 1.0)
    barb = bpy.data.curves.new("title_barb", "CURVE")
    barb.dimensions, barb.bevel_depth, barb.bevel_resolution = "3D", 0.03 * h, 3
    barb.use_fill_caps = True
    spline = barb.splines.new("POLY")
    spline.points.add(1)
    spline.points[0].co = (*hook[-2], 1.0)
    spline.points[1].co = (end_x + 0.28 * h, end_y - 0.57 * h, z, 1.0)
    spline.points[1].radius = 0.06
    barb_obj = bpy.data.objects.new("title_barb", barb)
    sc.collection.objects.link(barb_obj)
    barb_obj.data.materials.append(metal)
    cam = bpy.data.objects.new("title_cam", bpy.data.cameras.new("title_cam"))
    cam.data.type, cam.data.ortho_scale = "ORTHO", 2 * half * 1.12
    cam.location, cam.rotation_euler = (0.08, -0.08, 5.0), (0, 0, 0)
    sc.collection.objects.link(cam)
    sc.camera = cam
    # A shadow catcher just behind the letters: a soft dark drop shadow, baked into the
    # card's alpha, keeps it legible over the bright sea and the sunset.
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, -0.12))
    catcher = bpy.context.active_object
    catcher.is_shadow_catcher = True
    for name, color, energy, at, size in (("key", STORM_LIGHT, 2400.0, (-3.0, 4.0, 6.0), 3.0),
                                          ("rim", SUNSET_RIM, 1800.0, (3.5, -2.5, -1.0), 2.0),
                                          ("fill", "#5E7580", 380.0, (0.0, -6.0, 3.0), 6.0)):
        light = bpy.data.objects.new(name, bpy.data.lights.new(name, "AREA"))
        light.data.color, light.data.energy, light.data.size = rgb(color)[:3], energy, size
        light.location = at
        light.rotation_euler = (Vector((0, 0, 0)) - Vector(at)).to_track_quat("-Z", "Y").to_euler()
        sc.collection.objects.link(light)
    glint = bpy.data.objects.new("glint", bpy.data.lights.new("glint", "AREA"))
    glint.data.shape, glint.data.size, glint.data.size_y = "RECTANGLE", 0.08, 6.0
    glint.data.color, glint.data.energy = (0.85, 0.93, 1.0), 0.0
    glint.rotation_euler = (0, 0, math.radians(18))
    sc.collection.objects.link(glint)
    lit = bpy.data.collections.new("title_glint_receivers")      # the glint on the metal only, never the shadow
    for obj in sc.objects:
        if obj.name.startswith(("title_", "title")) and obj.type in ("FONT", "CURVE"):
            lit.objects.link(obj)
    glint.light_linking.receiver_collection = lit
    folder = OUT / ("title" if out is None else out)
    folder.mkdir(parents=True, exist_ok=True)
    for k in (range(GLINT + 1) if frames is None else frames):
        u = k / GLINT
        glint.location = (-half * 1.3 + 2.6 * half * u, 0.0, 1.2)
        glint.data.energy = 0.0 if k in (0, GLINT) else 260.0 * math.sin(math.pi * u)
        sc.render.filepath = str(folder / f"{k:04d}.png")
        bpy.ops.render.render(write_still=True, scene=sc.name)
    print(f"wrote the title card ({style}), {GLINT + 1} frames")


def card(se, name, start, end, width, height, centre, across, keys, glint=False, grow=None):
    """The title card (render/out/title/) over [start, end): `across` of the frame
    wide, centred at `centre` (fractions of the frame), its opacity keyed at
    (seconds, alpha). It bobs gently on a 2.5 s swell, which divides the film, so it
    matches across the loop; `glint` plays the lightning glint from the start;
    `grow` = (from s, to s, start scale) swells it up to full size."""
    folder = OUT / "title"
    s = strips(se).new_image(name, str(folder / "0000.png"), 9, start, fit_method="ORIGINAL")
    for k in range(1, end - start):
        s.elements.append(f"{k:04d}.png" if glint and k <= GLINT else "0000.png")
    s.frame_final_duration = end - start
    full = across * width / TITLE_W
    for f in range(start, end):
        sec = f / FPS
        u = 1.0 if not grow else min(max((sec - grow[0]) / (grow[1] - grow[0]), 0.0), 1.0)
        scale = full * (1.0 if not grow else grow[2] + (1 - grow[2]) * u * u * (3 - 2 * u))
        s.transform.scale_x = s.transform.scale_y = scale
        s.transform.offset_x = (centre[0] - 0.5) * width
        s.transform.offset_y = (centre[1] - 0.5) * height + 0.004 * height * math.sin(2 * math.pi * sec / 2.5)
        for path in ("scale_x", "scale_y", "offset_x", "offset_y"):
            s.transform.keyframe_insert(path, frame=f)
    for t_, v in keys:
        s.blend_alpha = v
        s.keyframe_insert("blend_alpha", frame=round(t_ * FPS))


def sky_plate():
    """Renders the transition's sky to render/out/sky/: a world that turns from
    the storm's slate gradient to the cove's dusk under a cloud deck that drifts,
    races and slows (a moving plane, so Cycles blurs it like real motion) while
    it thins out; the camera eases from the storm's last tilt toward the cove's."""
    sc = bpy.context.scene                     # the startup scene, emptied
    for obj in list(sc.objects):
        bpy.data.objects.remove(obj)
    scene.setup_render(1920, 1080, 32)          # the film's device, colour and motion blur
    blend = {}                                  # node sockets keyed through the move
    world = bpy.data.worlds.new("passage")
    world.use_nodes = True
    w = Nodes(world)
    up = w.math("MAXIMUM", w.math("MULTIPLY", w.xyz(w.new("ShaderNodeNewGeometry").outputs["Incoming"])[2], -1.0), 0.0)
    storm = w.mix(w.math("MULTIPLY", up, 3.0, clamp=True), [c * 1.5 for c in rgb(SLATE)[:3]] + [1],
                  [c * 0.25 for c in rgb(SLATE)[:3]] + [1])
    dusk = w.ramp(w.math("POWER", up, 0.5), [(x, rgb(c)) for x, c in DUSK])
    blend["sky"] = w.new("ShaderNodeValue").outputs[0]
    w.link(w.mix(blend["sky"], storm, dusk), world.node_tree.nodes["Background"].inputs["Color"])
    blend["strength"] = world.node_tree.nodes["Background"].inputs["Strength"]
    sc.world = world
    mesh = bpy.data.meshes.new("cloud_deck")
    mesh.from_pydata([(-3000, -3000, 0), (3000, -3000, 0), (3000, 3000, 0), (-3000, 3000, 0)], [], [(0, 1, 2, 3)])
    deck = bpy.data.objects.new("cloud_deck", mesh)
    sc.collection.objects.link(deck)
    deck.location.z = 150.0
    mat = bpy.data.materials.new("clouds")
    mat.use_nodes = True
    c = Nodes(mat)
    nodes = mat.node_tree.nodes
    nodes.remove(nodes["Principled BSDF"])
    billow = c.noise(c.scaled(c.coord(), (1.0, 1.6, 1.0)), 0.006, 8.0, 0.62)
    blend["cover"] = c.new("ShaderNodeValue").outputs[0]
    density = c.math("MULTIPLY", c.math("SUBTRACT", billow, blend["cover"]), 3.5, clamp=True)
    blend["tint"] = c.new("ShaderNodeValue").outputs[0]
    glow = c.new("ShaderNodeEmission")
    c.link(c.mix(blend["tint"], [v * 1.9 for v in rgb(SLATE)[:3]] + [1], [v * 0.8 for v in rgb("#FFB070")[:3]] + [1]),
           glow.inputs["Color"])
    shader = c.new("ShaderNodeMixShader")
    c.link(density, shader.inputs["Fac"])
    c.link(c.new("ShaderNodeBsdfTransparent").outputs[0], shader.inputs[1])
    c.link(glow.outputs[0], shader.inputs[2])
    c.link(shader.outputs[0], nodes["Material Output"].inputs["Surface"])
    deck.data.materials.append(mat)
    cam = bpy.data.objects.new("sky", bpy.data.cameras.new("sky"))
    cam.data.lens = 29
    sc.collection.objects.link(cam)
    sc.camera = cam
    for f in range(PLATE[0] - 2, PLATE[-1] + 1):
        u = min(max((f - PLATE[0]) / (PLATE[-1] - PLATE[0]), 0.0), 1.0)
        e = u * u * (3 - 2 * u)                         # slow, racing, slow
        values = {"sky": e, "tint": e, "cover": 0.42 + 0.2 * e, "strength": 0.5 + 0.4 * e}
        for key, socket in blend.items():
            socket.default_value = values[key]
            socket.keyframe_insert("default_value", frame=f)
        sc.view_settings.exposure = -0.45 * (1 - e) + 0.95 * e    # from the storm's dark sky to the bay's bright dusk
        sc.view_settings.keyframe_insert("exposure", frame=f)
        tilt = math.radians(59 + 11 * e)               # storm.py ends 59 deg up, cove.py opens 70 deg up
        cam.rotation_euler = scene.look((0, 0, 0), (math.cos(tilt), 0, math.sin(tilt)), -0.3 + 0.5 * e)
        cam.keyframe_insert("rotation_euler", frame=f)
        # The clouds run with the storm's wind (toward 147.5 deg) and swing round to the
        # bay's evening breeze (toward 28 deg) as time passes.
        heading = math.radians(147.5 + (28.0 - 147.5) * e)
        deck.location.x, deck.location.y = 900.0 * e * math.cos(heading), 900.0 * e * math.sin(heading)
        deck.keyframe_insert("location", frame=f)
    (OUT / "sky").mkdir(parents=True, exist_ok=True)
    for f in range(PLATE[0], PLATE[-1] + 1):
        sc.frame_set(f)
        sc.render.filepath = str(OUT / "sky" / f"{f:04d}.png")
        bpy.ops.render.render(write_still=True, scene=sc.name)
    print(f"wrote the sky plate, frames {PLATE[0]}-{PLATE[-1]}")


def plate(se, scale):
    """The sky plate over the cut, faded in and out where both shots show sky."""
    frames = sorted((OUT / "sky").glob("[0-9][0-9][0-9][0-9].png"))
    s = strips(se).new_image("sky", str(frames[0]), 7, int(frames[0].stem), fit_method="ORIGINAL")
    for f in frames[1:]:
        s.elements.append(f.name)
    s.transform.scale_x = s.transform.scale_y = scale     # the plate is even across, so no crop pan
    for f, v in zip(PLATE, (0.0, 1.0, 1.0, 0.0)):
        s.blend_alpha = v
        s.keyframe_insert("blend_alpha", frame=f)


def inset(se, width, height):
    """The MuJoCo view of the same moment, lower left in the 16:9 cut while the
    orbit closes in on the crew, labelled, fading in and out. The 9:16 crop has
    no room beside the boat for it."""
    if width < height:
        return
    frames = sorted((OUT / "sim").glob("[0-9][0-9][0-9][0-9].png"))
    start, end = int(frames[0].stem), int(frames[-1].stem) + 1
    s = strips(se).new_image("mujoco", str(frames[0]), 5, start, fit_method="ORIGINAL")
    for f in frames[1:]:
        s.elements.append(f.name)
    scale, cx, cy = 0.7, 96 + 224, 54 + 126       # centre, px from bottom left, inside the 5 % safe margin
    s.transform.scale_x = s.transform.scale_y = scale
    s.transform.offset_x, s.transform.offset_y = cx - width / 2, cy - height / 2
    label = strips(se).new_effect("mujoco_label", "TEXT", 6, frame_start=start, frame_end=end)
    label.text, label.font_size = "MuJoCo, same take", 26
    label.font = bpy.data.fonts.load(str(Path(bpy.utils.system_resource("DATAFILES", path="fonts")) / "Inter.woff2"),
                                     check_existing=True)
    label.location = ((cx - 640 * scale / 2) / width, (cy + 360 * scale / 2 + 10) / height)
    label.anchor_x = label.alignment_x = "LEFT"
    label.color = rgb(CREAM)
    for strip in (s, label):
        top = 0.9 if strip is s else 1.0
        for f, v in ((start, 0.0), (start + 12, top), (end - 12, top), (end - 1, 0.0)):
            strip.blend_alpha = v
            strip.keyframe_insert("blend_alpha", frame=f)


def seam(se, scale, width, height):
    """The loop's seam: the dive ends on the portal filling the frame; over its last
    frames the film's own first frame fades in at full frame, so the last frame is
    the first and the loop restarts without a jump (shot 1's camera is at rest there)."""
    start = round(LENGTH * FPS) - SEAM
    s = strips(se).new_image("seam", str(FRAMES / "0000.png"), 8, start, fit_method="ORIGINAL")
    s.frame_final_duration = SEAM
    s.transform.scale_x = s.transform.scale_y = scale
    if width < height:
        s.transform.offset_x = float(np.clip((0.5 - CROP[0][1]) * 1920 * scale, -(1920 * scale - width) / 2,
                                             (1920 * scale - width) / 2))
    for k in range(SEAM):
        u = (k + 1) / SEAM
        s.blend_alpha = u * u * (3 - 2 * u)
        s.keyframe_insert("blend_alpha", frame=start + k)


def cut(width, height, path, wav):
    sc = bpy.data.scenes.new(path.stem)
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = width, height, 100
    sc.render.fps, sc.frame_start, sc.frame_end = FPS, 0, round(LENGTH * FPS) - 1
    sc.view_settings.view_transform = "Standard"     # the frames are already tone-mapped
    se = sc.sequence_editor_create()
    frames = sorted(FRAMES.glob("[0-9][0-9][0-9][0-9].png"))
    film = strips(se).new_image("film", str(frames[0]), 1, 0, fit_method="ORIGINAL")
    for f in frames[1:]:
        film.elements.append(f.name)
    strips(se).new_sound("sound", str(wav), 2, 0)
    scale = max(width / 1920, height / 1080)
    film.transform.scale_x = film.transform.scale_y = scale
    if width < height:                   # pan the crop onto each shot's subject
        reach = (1920 * scale - width) / 2
        for t, u in CROP:
            film.transform.offset_x = float(np.clip((0.5 - u) * 1920 * scale, -reach, reach))
            film.transform.keyframe_insert("offset_x", frame=round(t * FPS))
        fc = [c for c in sc.animation_data.action.fcurves if "offset_x" in c.data_path][0]
        for k in fc.keyframe_points:
            k.interpolation = "LINEAR"
    plate(se, scale)
    # The passage carries on after the sky: the bay opens dim and cool, as the plate
    # ends, and warms into its full sunset over about 1.6 s.
    grade = film.modifiers.new("passage", "COLOR_BALANCE")
    for f, gain in ((CUT + PASSAGE - 1, (1.0, 1.0, 1.0)), (CUT + PASSAGE, (0.72, 0.8, 0.95)),
                    (CUT + PASSAGE + 38, (1.0, 1.0, 1.0))):
        grade.color_balance.gain = gain
        grade.color_balance.keyframe_insert("gain", frame=f)
    inset(se, width, height)
    seam(se, scale, width, height)
    # In the lower third, over the water below the boat; in 9:16 above the bottom UI. The end title surfaces in
    # the chest's storm and is full on the last frame, as on the first.
    centre, across = ((0.5, 0.17), 0.42) if width > height else ((0.5, 0.195), 0.8)
    end_in, start_out = round(TITLE_IN * FPS), round((TITLE_HOLD + FADE) * FPS)
    card(se, "title_end", end_in, round(LENGTH * FPS), width, height, centre, across,
         ((TITLE_IN, 0.0), (TITLE_IN + 0.5, 1.0), (LENGTH, 1.0)), grow=(TITLE_IN, LENGTH - 1 / FPS, 0.45))
    card(se, "title_start", 0, start_out, width, height, centre, across,
         ((0.0, 1.0), (TITLE_HOLD, 1.0), (TITLE_HOLD + FADE, 0.0)), glint=True)
    r = sc.render
    scene.disable_stamps(r)
    r.use_sequencer = True
    r.image_settings.file_format = "FFMPEG"
    r.ffmpeg.format, r.ffmpeg.codec = "MPEG4", "H264"
    r.ffmpeg.constant_rate_factor, r.ffmpeg.ffmpeg_preset = "NONE", "GOOD"
    r.ffmpeg.video_bitrate = r.ffmpeg.maxrate = 18000          # kbit/s
    r.ffmpeg.minrate, r.ffmpeg.buffersize, r.ffmpeg.gopsize = 0, 36000, FPS
    r.ffmpeg.audio_codec, r.ffmpeg.audio_bitrate, r.ffmpeg.audio_channels = "AAC", 320, "STEREO"
    r.ffmpeg.audio_mixrate = RATE
    r.filepath = str(path)
    bpy.ops.render.render(animation=True, scene=sc.name)
    print(f"wrote {path}")


def heroes():
    """README heroes: 1600 px JPEGs of two film frames, no metadata."""
    sc = bpy.data.scenes.new("heroes")
    scene.disable_stamps(sc.render)
    sc.view_settings.view_transform = "Standard"
    s = sc.render.image_settings
    s.file_format, s.quality, s.color_mode = "JPEG", 88, "RGB"
    figures = ROOT / "docs" / "readme-assets" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for theme, t in HEROES.items():
        img = bpy.data.images.load(str(FRAMES / f"{round(t * FPS):04d}.png"))
        img.scale(1600, 900)
        img.save_render(str(figures / f"hero-{theme}.jpg"), scene=sc)
        print(f"wrote hero-{theme}.jpg")


def main():
    frames = list(FRAMES.glob("[0-9][0-9][0-9][0-9].png"))
    if len(frames) != round(LENGTH * FPS):
        sys.exit(f"expected {round(LENGTH * FPS)} frames in {FRAMES}, found {len(frames)}")
    wav = OUT / "crabtain.wav"
    soundtrack(wav)
    title_card()
    sky_plate()
    cut(1920, 1080, OUT / "crabtain-16x9.mp4", wav)
    cut(1080, 1920, OUT / "crabtain-9x16.mp4", wav)
    heroes()


main()
