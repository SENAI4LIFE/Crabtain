<#
.SYNOPSIS
    Prepares the simulation environment: the pinned jumper checkout, a CPU
    Python environment and the Blender preferences this project relies on.
.DESCRIPTION
    Network is used only to fetch the pinned jumper commit, the pinned 3DBenchy
    mesh, two CC0 Poly Haven models, the title font, three soundtrack
    recordings (public domain and CC0) and the Python packages. Every step is idempotent; rerun it after a failure.
#>
[CmdletBinding()]
param(
    # Path to blender.exe; found under Program Files when omitted.
    [string]$Blender
)
$ErrorActionPreference = 'Stop'
$Root      = Split-Path -Parent $PSScriptRoot
$JumperUrl = 'https://github.com/KingKongRobotics/jumper.git'
$JumperSha = 'b2e58669848a3ba4f209597cd6bebdfe9d0533a4'
$Jumper    = Join-Path $Root 'third_party\jumper'
$BenchyUrl = 'https://raw.githubusercontent.com/CreativeTools/3DBenchy/b542785f9642ab39db8a778a0ccc934b3ebd829a/Multi-part'
$Benchy    = Join-Path $Root 'third_party\3dbenchy'
# The 17 single-part STLs of the multi-part 3DBenchy, by part name, with SHA-256.
$BenchyParts = [ordered]@{
    'Bridge roof'         = '9a5e4a1dee5c32b925c6186b06c44cfaef5f3e504dc02623bb5baf5f3791418a'
    'Bridge walls'        = 'f459d2468f107aa928e3e1f92c056f62631751fd4bd707fb2f2f0bd0aff77a88'
    'Cargo box'           = '262989714519c0199185a07dda526731084403bafc9530964cb1bb3054faf9d2'
    'Chimney body'        = 'c17c7905a8d405d4eb4f224f025f2d84eccb43ff09718f78f9451c8de32b46ea'
    'Chimney top'         = 'c70d7fa5b82f237ce1e1515c305de33dc05f13bc53c446790c5283d1b6bc3c03'
    'Deck surface'        = 'ba8f11d078385dfac125dae74547ae164618d2e352f9e8f4d667fc82ae92d7ac'
    'Doorframe port'      = 'ce4a01732552ba8b8fc21c6d5586658df83a3885633b73b5d4e814da919ef66c'
    'Doorframe starboard' = 'd37674c3abc224b2e1efe4b3bd5a740d1699c422aa40bb4cb8d7110cb1b86aea'
    'Fishing-rod-holder'  = '884ef5902d72709b818a743eb574a4bf9ede21e54b4031a1de2b20e65cb609db'
    'Gunwale'             = 'd6439d063a20708d733642f8f922fbfbaad9d2bd3b3004cf1fbc895b0cea175d'
    'Hawsepipe port'      = '6f143cffa0b1e9ccee990caf25516e332a612be868470dfdd69acae17bf167e2'
    'Hawsepipe starboard' = 'ccebc7327d12770862426c828be0d5700b27e7262cc5702fb12568019e6b4ea0'
    'Hull'                = '6cc572a7fcfcf7717286887430f6fd5126a9f216bae69d1edaca4d098c3c63a0'
    'Stern name plate'    = 'c1fd99127bae8e20deed50afa5b64f6ab5e59368a1c97eb707679af104e8e94d'
    'Stern window'        = 'edb429f92294de24994048c82a539ae252639d590efa1c890d852e7e180954aa'
    'Wheel'               = 'd229ff677f3e5ac8bf56c5364b10441ff9bec8e738983aa4da120d2840a73f9e'
    'Window'              = '60adae0cd5bc4645c5f88066964b726aad602ef26d7f6b888de3dd0d1557c431'
}
# CC0 models from Poly Haven for the bay (1k glTF): file under each model, its source folder, SHA-256.
$PolyHavenUrl = 'https://dl.polyhaven.org/file/ph-assets/Models'
$PolyHaven    = Join-Path $Root 'third_party\polyhaven'
$PolyHavenFiles = [ordered]@{
    'cannon_01/cannon_01_1k.gltf' = @('gltf/1k', 'd46d014e0fa1818c51222260fe6cab3bc34b7568b6bf19f196e14b24406168ee')
    'cannon_01/cannon_01.bin' = @('gltf/8k', '4f035b40eecfd21cf29c7c2c13a85bb91804b0b486e11f6072a77c583f3a49e9')
    'cannon_01/textures/cannon_01_diff_1k.jpg' = @('jpg/1k', '5ea48e1e18f8a9156b9f6464d8a8679db169d6aa07e479cec54e3b61bb015fac')
    'cannon_01/textures/cannon_01_arm_1k.jpg' = @('jpg/1k', '7a7afd1a036964272219b514f9deaa29e512e9792ac6df9d33dcfd9ea909a881')
    'cannon_01/textures/cannon_01_nor_gl_1k.jpg' = @('jpg/1k', 'f041013490c5549c797485abf8479adbd8f4977438daa75eb42ff234c0175b4c')
    'coast_land_rocks_03/coast_land_rocks_03_1k.gltf' = @('gltf/1k', 'b71ea37f8dd72785c0a90b99d52190f7bee3731439bb1d53750a4482fa1b938c')
    'coast_land_rocks_03/coast_land_rocks_03.bin' = @('gltf/8k', '69692370dbf2f19ee39eea284397663ce99373c153bb477fea3cbe087f9fec38')
    'coast_land_rocks_03/textures/coast_land_rocks_03_diff_1k.jpg' = @('jpg/1k', '958ff17d27b19d831acb83d6990b3e51908c762d3f14ff4d209427b1ba07dd38')
    'coast_land_rocks_03/textures/coast_land_rocks_03_arm_1k.jpg' = @('jpg/1k', 'b28392c9e7befc67e70d39a2e71e37539029a7c8eb91451bd9eb134b696d00cc')
    'coast_land_rocks_03/textures/coast_land_rocks_03_nor_gl_1k.jpg' = @('jpg/1k', 'a290423f001ab8029cf1dc9d88ea159eef21c882e6efc7018f753fa7d06d2d33')
}
# Pirata One (SIL OFL 1.1) for the film's title, from Google Fonts at a pinned commit.
$FontUrl = 'https://raw.githubusercontent.com/google/fonts/90abd17b4f97671435798b6147b698aa9087612f/ofl/pirataone/PirataOne-Regular.ttf'
$FontSha = '5347a2e155589ecf667d4b766613c8ee003edde9f83717fd24c09599a4b1ecc0'
$Font    = Join-Path $Root 'third_party\fonts\PirataOne-Regular.ttf'
# Recordings for the soundtrack, with SHA-256: thunder from Wikimedia Commons (public
# domain) and timber creaks from OpenGameArt (CC0).
$Sounds = [ordered]@{
    'Tonitrus.ogg'         = @('https://upload.wikimedia.org/wikipedia/commons/c/c8/Tonitrus.ogg',
                               '75c4d3911eedd1370db6aa30c90aee9e2eee3d18b429862eee6db221bcd0f421')
    'Rain_and_thunder.ogg' = @('https://upload.wikimedia.org/wikipedia/commons/4/42/Rain_and_thunder.ogg',
                               'cbfd7b7504bc4e53d6e56ac8d933ba56f97cc28f15a46800c74c2d8eccb3fa89')
    'tree_creak.flac'      = @('https://opengameart.org/sites/default/files/tree_creak.flac',
                               '9bab7c7f68850136c43fb91216c7c1760ac2bda22b2a31659905c559a464d422')
}
$Venv      = Join-Path $Root '.venv'
$Python    = Join-Path $Venv 'Scripts\python.exe'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'

function Step([string]$Text) { Write-Host "==> $Text" }
function Need([string]$Command, [string]$Install) {
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        throw "$Command not found. Install it with: $Install"
    }
}
function Run {
    & $args[0] $args[1..($args.Count - 1)]
    if ($LASTEXITCODE -ne 0) { throw "Failed ($LASTEXITCODE): $($args -join ' ')" }
}

Need git 'winget install --id Git.Git -e'
Need py   'winget install --id Python.Python.3.12 -e'

Step "jumper at $($JumperSha.Substring(0, 7))"
if (-not (Test-Path (Join-Path $Jumper '.git'))) {
    New-Item -ItemType Directory -Force $Jumper | Out-Null
    Run git -C $Jumper init -q
    Run git -C $Jumper remote add origin $JumperUrl
}
$head = (git -C $Jumper rev-parse --verify -q HEAD) 2>$null
if ($head -ne $JumperSha) {
    Run git -C $Jumper fetch -q --depth 1 origin $JumperSha
    Run git -C $Jumper checkout -q --detach FETCH_HEAD
}

Step '3DBenchy multi-part STLs (boat baseline)'
function Get-Sha256([string]$Path) { (Get-FileHash $Path -Algorithm SHA256).Hash.ToLower() }
New-Item -ItemType Directory -Force $Benchy | Out-Null
$ProgressPreference = 'SilentlyContinue'
foreach ($part in $BenchyParts.Keys) {
    $file = Join-Path $Benchy "$part.stl"
    if ((Test-Path $file) -and (Get-Sha256 $file) -eq $BenchyParts[$part]) { continue }
    $name = [uri]::EscapeDataString("#3DBenchy - Multi-part - Single - $part - (3DBenchy.com).stl")
    Invoke-WebRequest -UseBasicParsing -Uri "$BenchyUrl/$name" -OutFile $file
    if ((Get-Sha256 $file) -ne $BenchyParts[$part]) {
        Remove-Item $file
        throw "3DBenchy '$part' hash mismatch"
    }
}

Step 'Poly Haven CC0 models (bay props)'
foreach ($rel in $PolyHavenFiles.Keys) {
    $file = Join-Path $PolyHaven $rel
    $kind, $sha = $PolyHavenFiles[$rel]
    if ((Test-Path $file) -and (Get-Sha256 $file) -eq $sha) { continue }
    New-Item -ItemType Directory -Force (Split-Path -Parent $file) | Out-Null
    $id, $name = $rel.Split('/')[0], ($rel -split '/')[-1]
    Invoke-WebRequest -UseBasicParsing -Uri "$PolyHavenUrl/$kind/$id/$name" -OutFile $file
    if ((Get-Sha256 $file) -ne $sha) {
        Remove-Item $file
        throw "Poly Haven '$rel' hash mismatch"
    }
}

Step 'Title font (Pirata One)'
if (-not ((Test-Path $Font) -and (Get-Sha256 $Font) -eq $FontSha)) {
    New-Item -ItemType Directory -Force (Split-Path -Parent $Font) | Out-Null
    Invoke-WebRequest -UseBasicParsing -Uri $FontUrl -OutFile $Font
    if ((Get-Sha256 $Font) -ne $FontSha) {
        Remove-Item $Font
        throw 'Title font hash mismatch'
    }
}

Step 'Soundtrack recordings (public domain and CC0)'
foreach ($name in $Sounds.Keys) {
    $url, $sha = $Sounds[$name]
    $file = Join-Path $Root "third_party\sounds\$name"
    if ((Test-Path $file) -and (Get-Sha256 $file) -eq $sha) { continue }
    New-Item -ItemType Directory -Force (Split-Path -Parent $file) | Out-Null
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $file
    if ((Get-Sha256 $file) -ne $sha) {
        Remove-Item $file
        throw "Recording '$name' hash mismatch"
    }
}

Step 'Python environment (.venv, CPU build)'
if (-not (Test-Path $Python)) { Run py -3.12 -m venv $Venv }
Run $Python -m pip install -q torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
Run $Python -m pip install -q -e $Jumper

Step 'Checking that MuJoCo loads Jumper'
# Single quotes only: Windows PowerShell 5.1 drops embedded double quotes in native arguments.
$check = @'
import sys, mujoco
m = mujoco.MjModel.from_xml_path(sys.argv[1])
d = mujoco.MjData(m)
mujoco.mj_step(m, d)
print(f'mujoco {mujoco.__version__}: {m.nbody} bodies, {m.njnt} joints, {m.nu} actuators')
'@
Run $Python -c $check (Join-Path $Jumper 'assets\jumper\jumper.xml')

Step 'Blender preferences'
if (-not $Blender) {
    $Blender = Get-ChildItem "$env:ProgramFiles\Blender Foundation\*\blender.exe" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $Blender) { throw 'Blender not found. Install the current LTS from https://www.blender.org/download/lts/ or pass -Blender <path>.' }
# No --offline-mode: it makes use_online_access read-only. A fresh install starts offline anyway.
Run $Blender --background --python-exit-code 1 --python (Join-Path $PSScriptRoot 'configure_blender.py')

Step 'Done. Load the environment with: . .\scripts\activate.ps1'
