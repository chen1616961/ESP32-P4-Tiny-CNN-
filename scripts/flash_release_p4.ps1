param(
    [string]$Port = "COM3",
    [int]$Baud = 460800,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Firmware = Join-Path $Here "..\firmware"

$Bootloader = Join-Path $Firmware "bootloader.bin"
$Partitions = Join-Path $Firmware "partition-table.bin"
$App = Join-Path $Firmware "esp32p4_buoy_vision_lab.bin"

if (-not (Test-Path $Bootloader)) { throw "missing $Bootloader" }
if (-not (Test-Path $Partitions)) { throw "missing $Partitions" }
if (-not (Test-Path $App)) { throw "missing $App" }

& $Python -m esptool --chip esp32p4 -p $Port -b $Baud `
    --before default-reset --after hard-reset write-flash `
    --flash-mode dio --flash-size 16MB --flash-freq 80m `
    0x2000 $Bootloader `
    0x8000 $Partitions `
    0x10000 $App
