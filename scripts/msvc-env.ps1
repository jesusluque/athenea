# Copyright (c) 2026 jesus luque.
#
# The MSVC x64 environment for the Windows build scripts (build-usd.ps1,
# build-oidn.ps1, build-ocio.ps1), dot-sourced by them: the newest Visual
# Studio with the C++ tools, found by vswhere, entered through its DevShell
# module so cl, link and the Windows SDK are on the path of this process.
# Already inside a developer prompt (cl on the path), nothing is done.

if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return }
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere not found: install Visual Studio 2022 or its Build Tools with the C++ workload" }
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "no Visual Studio with the C++ x64 tools" }
Import-Module (Join-Path $vs "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
Enter-VsDevShell -VsInstallPath $vs -DevCmdArguments "-arch=x64 -host_arch=x64" -SkipAutomaticLocation | Out-Null
Write-Host "MSVC: $vs"
