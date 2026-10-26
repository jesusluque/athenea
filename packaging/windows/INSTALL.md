<!-- Copyright (c) 2026 jesus luque. -->
# athenea for Windows — install

*Español más abajo.*

## Requirements

- Windows 10 or 11, x64.
- An NVIDIA GPU and its driver, version 560.76 or later (the package's CUDA
  runtime is 12.6). Nothing else: no CUDA toolkit, no OptiX, no Visual C++
  redistributable — the package carries the CUDA compiler runtime (NVRTC,
  nvJitLink) and the MSVC runtime beside the programs.
- About 450 MB of disk (the ZIP is about 190 MB).
- For the Blender add-on: Blender 5.3.

No administrator rights are needed; everything is per user.

## Install

1. Unpack `athenea-<version>-windows-x64.zip` anywhere (for example your
   Downloads folder).
2. In the unpacked `athenea-<version>-windows-x64` folder, run in PowerShell:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\install.ps1
   ```

   Options: `-Prefix <folder>` installs somewhere other than
   `%LOCALAPPDATA%\athenea`; `-Blender <path to blender.exe>` names the
   Blender to install the add-on into (otherwise one is looked for on `PATH`
   and in `Program Files\Blender Foundation\Blender 5.3*`); `-NoBlender`
   skips the add-on; `-UsdPluginPath` also puts the Hydra plugin on
   `PXR_PLUGINPATH_NAME` (see below); `-NoCheck` skips the final check;
   `-Force` installs into a non-empty folder that holds no previous install
   (its contents are replaced).
3. Open a new terminal and run `athenea --version` and `athenea info`.

Running `install.ps1` again (the same or a newer package) replaces the
previous install in the same folder.

## What it changes

| Where | What |
|---|---|
| `%LOCALAPPDATA%\athenea` (or `-Prefix`) | the package's files, and `install.json`, the record of what follows |
| user `PATH` | `+ <prefix>\bin` |
| user `OFX_PLUGIN_PATH` | `+ <prefix>\ofx` (holds `Athenea.ofx.bundle`) |
| user `ATHENEA_MCP` | `<prefix>\bin\athenea-mcp.exe`, the server the OpenFX plugin starts |
| user `PXR_PLUGINPATH_NAME` | `+ <prefix>\plugin\usd`, only with `-UsdPluginPath` |
| Blender 5.3 | the `athenea_hydra` extension in its `user_default` repository, enabled |

`PXR_PLUGINPATH_NAME` is left alone by default: `athenea` finds its own
plugin without it, and every USD application reads it — one built against a
different OpenUSD (a DCC's own) would be offered an `hdAthenea` it cannot
load. Set it only for a USD application built against the package's OpenUSD
(26.08). Blender is unaffected either way: its add-on registers its own
plugin first.

## Uninstall

```powershell
powershell -ExecutionPolicy Bypass -File "$env:LOCALAPPDATA\athenea\uninstall.ps1"
```

It removes the add-on from Blender, takes the entries it added out of the
variables above (a previous `ATHENEA_MCP` comes back), and deletes the
folder. `-KeepFiles` restores the environment and leaves the files.

## Using it from other applications

- **Nuke, Natron** (OpenFX): after the install, start the application from
  a new session; it finds `Athenea.ofx.bundle` through `OFX_PLUGIN_PATH`.
  The node is *Athenea* under the plugin's group. To use another folder,
  copy `<prefix>\ofx\Athenea.ofx.bundle` into the host's own plugin
  folder (`C:\Program Files\Common Files\OFX\Plugins` needs administrator
  rights) — the plugin still starts `<prefix>\bin\athenea-mcp.exe` through
  `ATHENEA_MCP`.
- **Blender 5.3**: the installer enables the add-on. Otherwise:
  *Edit > Preferences > Get Extensions > Install from Disk* and choose
  `<prefix>\blender\athenea_hydra-<version>-windows-x64.zip`. The render
  engine is *Athenea*.
- **The command line**: `athenea render`, `athenea stage`, `athenea
  mesh2splat`… — the operations manual is in
  `<prefix>\share\doc\athenea\operations.md` (Spanish:
  `operations.es.md`).

---

# athenea para Windows — instalación

## Requisitos

- Windows 10 u 11, x64.
- Una GPU de NVIDIA y su driver, versión 560.76 o posterior (el runtime de
  CUDA del paquete es el 12.6). Nada más: ni toolkit de CUDA, ni OptiX, ni
  redistribuible de Visual C++; el paquete lleva el runtime del compilador de
  CUDA (NVRTC, nvJitLink) y el de MSVC junto a los programas.
- Unos 450 MB de disco (el ZIP ocupa unos 190 MB).
- Para el add-on de Blender: Blender 5.3.

No hacen falta permisos de administrador; todo es por usuario.

## Instalar

1. Descomprime `athenea-<versión>-windows-x64.zip` donde quieras (por
   ejemplo, en Descargas).
2. En la carpeta `athenea-<versión>-windows-x64` descomprimida, ejecuta en
   PowerShell:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\install.ps1
   ```

   Opciones: `-Prefix <carpeta>` instala en otro sitio que
   `%LOCALAPPDATA%\athenea`; `-Blender <ruta a blender.exe>` indica en qué
   Blender instalar el add-on (si no, se busca uno en el `PATH` y en
   `Program Files\Blender Foundation\Blender 5.3*`); `-NoBlender` se salta
   el add-on; `-UsdPluginPath` pone además el plugin de Hydra en
   `PXR_PLUGINPATH_NAME` (ver abajo); `-NoCheck` se salta la comprobación
   final; `-Force` instala en una carpeta no vacía que no tiene una
   instalación anterior (su contenido se sustituye).
3. Abre un terminal nuevo y ejecuta `athenea --version` y `athenea info`.

Volver a ejecutar `install.ps1` (con el mismo paquete o uno más nuevo)
sustituye la instalación anterior en la misma carpeta.

## Qué cambia

| Dónde | Qué |
|---|---|
| `%LOCALAPPDATA%\athenea` (o `-Prefix`) | los ficheros del paquete, e `install.json`, el registro de lo que sigue |
| `PATH` del usuario | `+ <prefijo>\bin` |
| `OFX_PLUGIN_PATH` del usuario | `+ <prefijo>\ofx` (contiene `Athenea.ofx.bundle`) |
| `ATHENEA_MCP` del usuario | `<prefijo>\bin\athenea-mcp.exe`, el servidor que arranca el plugin de OpenFX |
| `PXR_PLUGINPATH_NAME` del usuario | `+ <prefijo>\plugin\usd`, solo con `-UsdPluginPath` |
| Blender 5.3 | la extensión `athenea_hydra` en su repositorio `user_default`, activada |

`PXR_PLUGINPATH_NAME` no se toca por defecto: `athenea` encuentra su propio
plugin sin ella, y la leen todas las aplicaciones de USD; a una compilada
contra otro OpenUSD (el propio de una aplicación DCC) se le ofrecería un
`hdAthenea` que no puede cargar. Ponla solo para una aplicación de USD
compilada contra el OpenUSD del paquete (26.08). A Blender no le afecta en
ningún caso: su add-on registra antes su propio plugin.

## Desinstalar

```powershell
powershell -ExecutionPolicy Bypass -File "$env:LOCALAPPDATA\athenea\uninstall.ps1"
```

Quita el add-on de Blender, saca de las variables de arriba las entradas que
añadió (vuelve un `ATHENEA_MCP` anterior, si lo había) y borra la carpeta.
`-KeepFiles` restaura el entorno y deja los ficheros.

## Usarlo desde otras aplicaciones

- **Nuke, Natron** (OpenFX): después de instalar, arranca la aplicación
  desde una sesión nueva; encuentra `Athenea.ofx.bundle` por
  `OFX_PLUGIN_PATH`. El nodo es *Athenea*, en el grupo del plugin. Para usar
  otra carpeta, copia `<prefijo>\ofx\Athenea.ofx.bundle` a la carpeta de
  plugins del host (`C:\Program Files\Common Files\OFX\Plugins` pide
  permisos de administrador); el plugin sigue arrancando
  `<prefijo>\bin\athenea-mcp.exe` por `ATHENEA_MCP`.
- **Blender 5.3**: el instalador activa el add-on. Si no:
  *Edit > Preferences > Get Extensions > Install from Disk* y elige
  `<prefijo>\blender\athenea_hydra-<versión>-windows-x64.zip`. El motor de
  render es *Athenea*.
- **La línea de comandos**: `athenea render`, `athenea stage`, `athenea
  mesh2splat`…; el manual de operaciones está en
  `<prefijo>\share\doc\athenea\operations.es.md` (inglés: `operations.md`).
