# Packs the Windows release archive: scaleup-<version>-<target>.zip with both executables, every DLL
# that `ort` copied next to them (DirectML.dll, onnxruntime_providers_*.dll), LICENSE, README.md and
# models\licenses\.
# Usage: powershell -File packaging\zip_windows.ps1 -Version 0.1.0 -Target x86_64-pc-windows-msvc
param(
  [Parameter(Mandatory = $true)] [string] $Version,
  [Parameter(Mandatory = $true)] [string] $Target,
  [string] $BinDir = "target\$Target\release",
  [string] $OutDir = "dist"
)
$ErrorActionPreference = "Stop"
$name = "scaleup-$Version-$Target"
$stage = Join-Path ([System.IO.Path]::GetTempPath()) $name
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path (Join-Path $stage "models\licenses") | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

foreach ($exe in @("scaleup.exe", "scaleup-app.exe")) {
  Copy-Item (Join-Path $BinDir $exe) $stage
}
# Symlinks (ort copy-dylibs) are followed: Copy-Item copies the target's bytes.
Get-ChildItem -Path $BinDir -Filter "*.dll" | ForEach-Object {
  Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $stage $_.Name)
}
Copy-Item LICENSE, README.md $stage
Copy-Item "models\licenses\*.txt" (Join-Path $stage "models\licenses")

$zip = Join-Path (Resolve-Path $OutDir) "$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $stage -DestinationPath $zip -CompressionLevel Optimal
Remove-Item -Recurse -Force $stage
Write-Host "wrote $zip"
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($zip)
try {
  $archive.Entries | ForEach-Object { Write-Host ("  {0,12}  {1}" -f $_.Length, $_.FullName) }
} finally {
  $archive.Dispose()
}
