# Copyright (c) 2026 jesus luque.
#
# athenea for one Windows user, from the unpacked package; no administrator
# rights and nothing to install first. Run from the package folder:
#
#   powershell -ExecutionPolicy Bypass -File install.ps1 [-Prefix <folder>] [-Blender <blender.exe>] [-NoBlender]
#
# It copies the package to the prefix (default %LOCALAPPDATA%\athenea), sets
# the user's environment -- bin\ on PATH, ofx\ on OFX_PLUGIN_PATH, ATHENEA_MCP
# to bin\athenea-mcp.exe, and plugin\usd on PXR_PLUGINPATH_NAME with
# -UsdPluginPath --, installs the Blender add-on into Blender 5.3's user
# extensions when a Blender 5.3 is found, and runs a check. What it did is
# written to <prefix>\install.json, which uninstall.ps1 reads to undo it.
# Run again, it replaces the previous install in the same prefix; programs
# started before it see the new environment only once restarted.
#
# -EnvironmentKey writes the variables under another HKCU key instead of
# HKCU\Environment, for testing the script without touching the user's
# environment.
param(
    [string]$Prefix = (Join-Path $env:LOCALAPPDATA "athenea"),
    [string]$Blender = "",
    [switch]$NoBlender,
    [switch]$UsdPluginPath,
    [switch]$NoCheck,
    [switch]$Force,
    [string]$EnvironmentKey = "Environment"
)
$ErrorActionPreference = "Stop"
$source = $PSScriptRoot
$marker = "install.json"

function Say($text) { Write-Host "athenea install: $text" }
# A program's standard error is lines to show, not a failure: its exit code says that.
function Invoke-Native([scriptblock]$Block) {
    $ErrorActionPreference = "Continue"
    & $Block 2>&1 | ForEach-Object { "$_" }
}

if (-not (Test-Path (Join-Path $source "bin\athenea.exe"))) {
    throw "no bin\athenea.exe beside install.ps1: run it from the unpacked package"
}
$Prefix = [System.IO.Path]::GetFullPath($Prefix)
$inPlace = ([System.IO.Path]::GetFullPath($source).TrimEnd('\') -ieq $Prefix.TrimEnd('\'))

# --- the user's environment, read and written raw (PATH keeps its %VARIABLES%) ----------
function Open-EnvKey([bool]$write) {
    $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey($EnvironmentKey, $write)
    if (-not $key -and $write) { $key = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey($EnvironmentKey) }
    return $key
}
function Get-UserVar([string]$name) {
    $key = Open-EnvKey $false
    if (-not $key) { return $null }
    try { return $key.GetValue($name, $null, [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames) }
    finally { $key.Close() }
}
function Set-UserVar([string]$name, $value) {
    $key = Open-EnvKey $true
    try {
        if ($null -eq $value -or $value -eq "") { $key.DeleteValue($name, $false) }
        else {
            $kind = if ($value -match '%') { [Microsoft.Win32.RegistryValueKind]::ExpandString } else { [Microsoft.Win32.RegistryValueKind]::String }
            $key.SetValue($name, $value, $kind)
        }
    } finally { $key.Close() }
}
function Add-ToList([string]$name, [string]$entry) {
    $current = Get-UserVar $name
    $items = @()
    if ($current) { $items = $current -split ';' | Where-Object { $_ -ne "" } }
    if ($items | Where-Object { $_.TrimEnd('\') -ieq $entry.TrimEnd('\') }) { return $false }
    Set-UserVar $name ((@($items) + $entry) -join ';')
    return $true
}
function Remove-FromList([string]$name, [string]$entry) {
    $current = Get-UserVar $name
    if (-not $current) { return }
    $items = $current -split ';' | Where-Object { $_ -ne "" -and $_.TrimEnd('\') -ine $entry.TrimEnd('\') }
    Set-UserVar $name ($items -join ';')
}
function Send-EnvironmentChange {
    if ($EnvironmentKey -ne "Environment") { return }
    if (-not ("AtheneaInstall.Native" -as [type])) {
        Add-Type -Namespace AtheneaInstall -Name Native -MemberDefinition @'
[DllImport("user32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
public static extern System.IntPtr SendMessageTimeout(System.IntPtr hWnd, uint Msg, System.UIntPtr wParam,
    string lParam, uint fuFlags, uint uTimeout, out System.UIntPtr lpdwResult);
'@
    }
    $result = [System.UIntPtr]::Zero
    [void][AtheneaInstall.Native]::SendMessageTimeout([System.IntPtr]0xffff, 0x1A, [System.UIntPtr]::Zero,
        "Environment", 2, 5000, [ref]$result)
}

# --- Blender 5.3 ---------------------------------------------------------------------------
function Find-Blender {
    if ($Blender) {
        $candidate = if ((Test-Path $Blender -PathType Container)) { Join-Path $Blender "blender.exe" } else { $Blender }
        if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
        throw "no blender.exe at $Blender"
    }
    $found = @()
    $onPath = Get-Command blender.exe -ErrorAction SilentlyContinue
    if ($onPath) { $found += $onPath.Source }
    foreach ($root in @($env:ProgramFiles, ${env:ProgramFiles(x86)}, $env:LOCALAPPDATA)) {
        if ($root) { $found += Get-ChildItem (Join-Path $root "Blender Foundation\Blender 5.3*\blender.exe") -ErrorAction SilentlyContinue | ForEach-Object FullName }
    }
    foreach ($exe in $found) {
        $version = (Invoke-Native { & $exe --version } | Select-Object -First 1)
        if ($version -match '^Blender 5\.3') { return $exe }
    }
    return $null
}

# --- a previous install --------------------------------------------------------------------
$previous = $null
$markerPath = Join-Path $Prefix $marker
if (Test-Path $markerPath) {
    $previous = Get-Content $markerPath -Raw | ConvertFrom-Json
} elseif (-not $inPlace -and (Test-Path $Prefix) -and (Get-ChildItem $Prefix -Force | Select-Object -First 1) -and -not $Force) {
    throw "$Prefix exists, is not empty and holds no athenea install: choose another -Prefix, or -Force to replace it"
}

# --- the files -----------------------------------------------------------------------------
if (-not $inPlace) {
    Say "copying the package to $Prefix"
    New-Item -ItemType Directory -Force $Prefix | Out-Null
    # /MIR: the prefix becomes the package, so files a previous version had and this one does not go.
    & robocopy $source $Prefix /MIR /NFL /NDL /NJH /NJS /NP /R:2 /W:1 /XF $marker | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "copying to $Prefix failed (robocopy $LASTEXITCODE)" }
}
Get-ChildItem $Prefix -Recurse -File | Unblock-File -ErrorAction SilentlyContinue

# --- the environment -----------------------------------------------------------------------
$bin = Join-Path $Prefix "bin"
$ofx = Join-Path $Prefix "ofx"
$usdPlugins = Join-Path $Prefix "plugin\usd"
$mcp = Join-Path $bin "athenea-mcp.exe"
$record = [ordered]@{
    version = ((Invoke-Native { & (Join-Path $Prefix "bin\athenea.exe") --version }) -join " ")
    prefix = $Prefix
    environmentKey = $EnvironmentKey
    added = [ordered]@{}
    athenea_mcp_before = $null
    blender = $null
}
# Entries an earlier install added stay recorded, so uninstall takes them back.
$added = @{}
if ($previous -and $previous.added) {
    foreach ($p in $previous.added.PSObject.Properties) { $added[$p.Name] = $p.Value }
}
foreach ($pair in @(@("Path", $bin), @("OFX_PLUGIN_PATH", $ofx))) {
    if (Add-ToList $pair[0] $pair[1]) { Say "$($pair[0]) += $($pair[1])"; $added[$pair[0]] = $pair[1] }
}
if ($UsdPluginPath) {
    if (Add-ToList "PXR_PLUGINPATH_NAME" $usdPlugins) { Say "PXR_PLUGINPATH_NAME += $usdPlugins"; $added["PXR_PLUGINPATH_NAME"] = $usdPlugins }
} elseif ($added.ContainsKey("PXR_PLUGINPATH_NAME")) {
    Remove-FromList "PXR_PLUGINPATH_NAME" $added["PXR_PLUGINPATH_NAME"]
    $added.Remove("PXR_PLUGINPATH_NAME")
}
foreach ($k in $added.Keys) { $record.added[$k] = $added[$k] }
$mcpBefore = Get-UserVar "ATHENEA_MCP"
if ($previous) { $record.athenea_mcp_before = $previous.athenea_mcp_before }
elseif ($mcpBefore -and $mcpBefore -ine $mcp) { $record.athenea_mcp_before = $mcpBefore }
Set-UserVar "ATHENEA_MCP" $mcp
Say "ATHENEA_MCP = $mcp"
Send-EnvironmentChange

# --- the Blender add-on --------------------------------------------------------------------
$addon = Get-ChildItem (Join-Path $Prefix "blender\athenea_hydra-*.zip") -ErrorAction SilentlyContinue | Select-Object -First 1
if ($previous -and $previous.blender) { $record.blender = $previous.blender }
if ($addon -and -not $NoBlender) {
    $exe = Find-Blender
    if ($exe) {
        Say "installing the add-on into Blender 5.3 ($exe)"
        # Installed, then enabled from a Python session and saved in the user's preferences:
        # `install-file --enable` does not enable it in a profile whose extension repository has
        # not been loaded yet.
        $zipPath = $addon.FullName
        Invoke-Native { & $exe --command extension install-file --repo user_default $zipPath } | ForEach-Object { "  $_" } | Write-Host
        $installed = ($LASTEXITCODE -eq 0)
        if ($installed) {
            $enable = "import addon_utils, bpy; m = addon_utils.enable('bl_ext.user_default.athenea_hydra', default_set=True); " +
                      "bpy.ops.wm.save_userpref(); print('athenea_hydra enabled' if m else 'athenea_hydra NOT enabled')"
            $out = Invoke-Native { & $exe -b --python-exit-code 1 --python-expr $enable }
            $out | Select-String "athenea_hydra" | ForEach-Object { "  $($_.Line)" } | Write-Host
            $installed = ($out -match "athenea_hydra enabled").Count -gt 0
        }
        if ($installed) { $record.blender = $exe }
        else { Write-Warning "Blender did not install the add-on (exit $LASTEXITCODE): install $($addon.FullName) from Blender's Preferences > Get Extensions > Install from Disk" }
    } else {
        Say "no Blender 5.3 found: install $($addon.FullName) from Blender's Preferences > Get Extensions > Install from Disk, or run again with -Blender <blender.exe>"
    }
}

$record | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $markerPath

# --- the check -----------------------------------------------------------------------------
if (-not $NoCheck) {
    $exe = Join-Path $bin "athenea.exe"
    $version = Invoke-Native { & $exe --version }
    if ($LASTEXITCODE -ne 0) { throw "athenea --version failed ($LASTEXITCODE): $version" }
    Say "$version"
    $info = Invoke-Native { & $exe info }
    $code = $LASTEXITCODE
    $info | Select-Object -First 12 | ForEach-Object { "  $_" } | Write-Host
    if ($code -ne 0) { Write-Warning "athenea info exited ${code}: the programs are installed, but no GPU device opened (NVIDIA driver?)" }
}
Say "installed in $Prefix. Open a new terminal (or restart Nuke, Natron, Blender) for the environment to apply."
