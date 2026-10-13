# Copyright (c) 2026 jesus luque.
#
# The Windows installer: installer-windows.iss over package.ps1's folder, with
# Inno Setup's compiler (winget install JRSoftware.InnoSetup).
#
#   scripts\installer-windows.ps1 <package folder> [out dir]
param([Parameter(Mandatory = $true)][string]$Package, [string]$Out = "")
$ErrorActionPreference = "Stop"
$Package = (Resolve-Path $Package).Path
if (-not $Out) { $Out = Split-Path $Package -Parent }
$version = ((Split-Path $Package -Leaf) -replace '^athenea-([0-9.]+)-.*$', '$1')
$iscc = @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
          "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw "Inno Setup 6 not found: winget install JRSoftware.InnoSetup" }
& $iscc /Q "/DSource=$Package" "/DVersion=$version" "/O$Out" (Join-Path $PSScriptRoot "installer-windows.iss")
if ($LASTEXITCODE -ne 0) { throw "iscc failed" }
Join-Path $Out "athenea-$version-windows-x64-setup.exe"
