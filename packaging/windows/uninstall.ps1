# Copyright (c) 2026 jesus luque.
#
# Takes back what install.ps1 did, from the record it left (<prefix>\install.json):
# the entries it added to PATH, OFX_PLUGIN_PATH and PXR_PLUGINPATH_NAME,
# ATHENEA_MCP (the value before the install comes back, if there was one), the
# Blender add-on, and the prefix folder itself.
#
#   powershell -ExecutionPolicy Bypass -File "%LOCALAPPDATA%\athenea\uninstall.ps1" [-Prefix <folder>] [-KeepFiles]
param(
    [string]$Prefix = "",
    [switch]$KeepFiles
)
$ErrorActionPreference = "Stop"
if (-not $Prefix) {
    $Prefix = if (Test-Path (Join-Path $PSScriptRoot "install.json")) { $PSScriptRoot } else { Join-Path $env:LOCALAPPDATA "athenea" }
}
$Prefix = [System.IO.Path]::GetFullPath($Prefix)
$markerPath = Join-Path $Prefix "install.json"
if (-not (Test-Path $markerPath)) { throw "no athenea install in $Prefix (no install.json)" }
$record = Get-Content $markerPath -Raw | ConvertFrom-Json
$EnvironmentKey = if ($record.environmentKey) { $record.environmentKey } else { "Environment" }

function Say($text) { Write-Host "athenea uninstall: $text" }
function Invoke-Native([scriptblock]$Block) {
    $ErrorActionPreference = "Continue"
    & $Block 2>&1 | ForEach-Object { "$_" }
}
function Open-EnvKey([bool]$write) { return [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey($EnvironmentKey, $write) }
function Get-UserVar([string]$name) {
    $key = Open-EnvKey $false
    if (-not $key) { return $null }
    try { return $key.GetValue($name, $null, [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames) }
    finally { $key.Close() }
}
function Set-UserVar([string]$name, $value) {
    $key = Open-EnvKey $true
    if (-not $key) { return }
    try {
        if ($null -eq $value -or $value -eq "") { $key.DeleteValue($name, $false) }
        else {
            $kind = if ($value -match '%') { [Microsoft.Win32.RegistryValueKind]::ExpandString } else { [Microsoft.Win32.RegistryValueKind]::String }
            $key.SetValue($name, $value, $kind)
        }
    } finally { $key.Close() }
}
function Remove-FromList([string]$name, [string]$entry) {
    $current = Get-UserVar $name
    if (-not $current) { return }
    $items = $current -split ';' | Where-Object { $_ -ne "" -and $_.TrimEnd('\') -ine $entry.TrimEnd('\') }
    Set-UserVar $name ($items -join ';')
    Say "$name -= $entry"
}

# --- the Blender add-on --------------------------------------------------------------------
if ($record.blender -and (Test-Path $record.blender)) {
    Say "removing the add-on from Blender ($($record.blender))"
    $exe = $record.blender
    Invoke-Native { & $exe --command extension remove user_default.athenea_hydra } | ForEach-Object { "  $_" } | Write-Host
    if ($LASTEXITCODE -ne 0) { Write-Warning "Blender did not remove the add-on (exit $LASTEXITCODE): remove Athenea in Preferences > Get Extensions" }
}

# --- the environment -----------------------------------------------------------------------
if ($record.added) {
    foreach ($p in $record.added.PSObject.Properties) { Remove-FromList $p.Name $p.Value }
}
$mcp = Join-Path $Prefix "bin\athenea-mcp.exe"
$current = Get-UserVar "ATHENEA_MCP"
if ($current -and $current -ieq $mcp) {
    Set-UserVar "ATHENEA_MCP" $record.athenea_mcp_before
    Say ("ATHENEA_MCP " + $(if ($record.athenea_mcp_before) { "= $($record.athenea_mcp_before)" } else { "removed" }))
}
if ($EnvironmentKey -eq "Environment") {
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

# --- the files -----------------------------------------------------------------------------
if ($KeepFiles) {
    Remove-Item $markerPath
    Say "environment restored; the files stay in $Prefix"
} else {
    Set-Location ([System.IO.Path]::GetTempPath())
    Remove-Item -Recurse -Force $Prefix
    Say "removed $Prefix"
}
