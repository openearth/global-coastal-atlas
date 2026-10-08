param(
    [string]$Notebook = "17_seabed_litho.ipynb",
    [string]$Variable = "seabed_litho",
    [string]$PDir = "P:\11209117-041-ipdc-egypt-swi\data\IntDeltaPlatform_QGA\Data_Vivian",
    [string]$Mirror = "C:\Ocean\Work\Projects\2026\Africa\Data\Data_Vivian"
)

# Docker cannot see the P: drive: copy the input to a local mirror, run the notebook,
# copy the stac/ output back to P:, verify by hash, then remove the mirror.
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

New-Item -ItemType Directory -Force $Mirror | Out-Null
Copy-Item "$PDir\$Variable.tif" $Mirror -Force

Push-Location $here
try {
    docker compose run --rm --no-deps stac-notebooks jupyter nbconvert --to notebook --execute --inplace "/workspace/global-coastal-atlas/STAC/data/notebooks/$Notebook"
    if ($LASTEXITCODE -ne 0) { throw "Notebook execution failed (mirror kept at $Mirror)" }
} finally { Pop-Location }

$out = Join-Path $Mirror "stac"
robocopy $out "$PDir\stac" /E /NFL /NDL /NJH /NJS | Out-Null
if ($LASTEXITCODE -ge 8) { throw "Copy to P: failed (mirror kept at $Mirror)" }

$allOk = $true
foreach ($f in Get-ChildItem $out -Recurse -File) {
    $rel = $f.FullName.Substring($out.Length + 1)
    if ((Get-FileHash $f.FullName).Hash -ne (Get-FileHash "$PDir\stac\$rel").Hash) {
        Write-Warning "Hash mismatch: $rel"; $allOk = $false
    }
}
if (-not $allOk) { throw "Verification failed, mirror kept at $Mirror" }

Remove-Item $Mirror -Recurse -Force
Write-Host "Done. Output copied to $PDir\stac and local mirror removed."
