; Copyright (c) 2026 jesus luque.
;
; The Windows installer: package.ps1's folder installed to Program Files\athenea,
; its bin\ on the machine's PATH, and an uninstaller that takes both back.
; Not signed: SmartScreen asks once. Built by scripts\installer-windows.ps1:
;   iscc /DSource=<package folder> /DVersion=<x.y.z> /O<out dir> installer-windows.iss

#ifndef Source
  #error Source (the package folder) is required
#endif
#ifndef Version
  #error Version is required
#endif

[Setup]
AppId={{6C0B8C1E-5A0E-4E7B-9C3F-2A4E1D9B7A10}
AppName=athenea
AppVersion={#Version}
AppPublisher=jesus luque
AppPublisherURL=https://github.com/jesusluque/athenea
DefaultDirName={autopf}\athenea
DefaultGroupName=athenea
DisableProgramGroupPage=yes
OutputBaseFilename=athenea-{#Version}-windows-x64-setup
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
ChangesEnvironment=yes
Compression=lzma2/max
SolidCompression=yes
LicenseFile={#Source}\share\doc\athenea\LICENSE
PrivilegesRequired=admin

[Files]
Source: "{#Source}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Registry]
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; ValueType: expandsz; \
    ValueName: "Path"; ValueData: "{olddata};{app}\bin"; Check: NeedsAddPath(ExpandConstant('{app}\bin'))

[Code]
function NeedsAddPath(Dir: string): Boolean;
var
  Paths: string;
begin
  if not RegQueryStringValue(HKLM, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', Paths) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Paths) + ';') = 0;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Paths, Dir: string;
  P: Integer;
begin
  if CurUninstallStep <> usPostUninstall then exit;
  if not RegQueryStringValue(HKLM, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', Paths) then exit;
  Dir := ';' + ExpandConstant('{app}\bin');
  P := Pos(Uppercase(Dir), Uppercase(Paths));
  if P > 0 then
  begin
    Delete(Paths, P, Length(Dir));
    RegWriteExpandStringValue(HKLM, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', Paths);
  end;
end;
