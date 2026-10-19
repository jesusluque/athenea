# OpenFX (Nuke, Natron)

`Athenea.ofx.bundle` es un plugin OpenFX con un nodo, **Athenea 3DGS** (grupo *Athenea*). Es un generador sin entrada: dibuja una nube de Gaussians a través de la cámara del nodo, al formato del proyecto y en el tiempo del frame, con el rasterizador de athenea, y saca floats RGBA lineales y premultiplicados y las demás capas del frame (profundidad, posición en el mundo, Cryptomatte). Una nube que admite relight (un transfer) la ilumina un cielo que el artista elige en el nodo.

El código está en `integrations/ofx`. El manual de operación lo describe en la sección 6.1 de [`docs/operations.es.md`](https://github.com/jesusluque/athenea/blob/main/docs/operations.es.md), y el diseño y las medidas están en la sección *The OpenFX plugin* de [`docs/decisions.md`](https://github.com/jesusluque/athenea/blob/main/docs/decisions.md).

## El motor fuera de proceso

<figure markdown>
  ![Natron con el nodo Athenea 3DGS: el grafo de nodos (Athenea3DGS hacia el Viewer), sus parámetros (nube, cámara, cielo, salida) y el peón bajo el atardecer de Venecia.](assets/natron-ui.jpg)
  <figcaption>Natron con el nodo Athenea 3DGS: el grafo de nodos (Athenea3DGS hacia el Viewer), sus parámetros (nube, cámara, cielo, salida) y el peón bajo el atardecer de Venecia.</figcaption>
</figure>

El plugin no carga nada del motor. Un compositor lleva su propio OpenUSD y su propio TBB, y el motor trae uno de cada; dos de cualquiera de ellos en un proceso comparten símbolos. Por eso el plugin enlaza solo la librería de soporte de OpenFX y un parser de JSON, y arranca `athenea-mcp`, el servidor MCP del motor, como proceso hijo la primera vez que renderiza.

- El servidor mantiene abiertos el dispositivo y la nube entre frames.
- Cada frame vuelve como números en un fichero del directorio temporal, nunca como imagen: floats RGBA, después la profundidad, después las capas pedidas, con la fila de abajo primero y premultiplicados, como está una imagen de OpenFX.
- Hay un servidor para todo el proceso del host, haga el host las instancias del nodo que haga (Natron hace una por hilo de render), así que una nube se carga una vez. Los renders le llegan de uno en uno.
- Un dispositivo perdido o un fallo de un kernel terminan el servidor, no el host; el frame siguiente lo arranca otra vez.

## Compilación

El plugin se compila con el motor cuando `ATHENEA_BUILD_OFX` está activado, que es lo predeterminado. La build de CMake lo compila en macOS y Linux, y no en Windows. Se ha ejecutado en Natron sobre macOS (ver *Medidas*). Necesita el submódulo `third_party/openfx`, que es el OpenFX de Natron: su librería de soporte lleva la extensión multiplano (los planos de Foundry y los planos propios de Natron).

```sh
cmake --preset macos-arm64-release          # o linux-x86_64-release
cmake --build --preset macos-arm64-release
```

El bundle se escribe en `<build>/ofx/Athenea.ofx.bundle`, organizado como OpenFX organiza uno:

| Ruta dentro del bundle | Qué contiene |
|---|---|
| `Contents/MacOS/Athenea.ofx` (macOS), `Contents/Linux-x86-64/Athenea.ofx` (Linux) | el plugin |
| `Contents/Info.plist` (macOS) | la descripción del bundle |
| `Contents/Resources/bin/athenea-mcp` | el servidor que arranca el plugin |

## Instalación

Se copia o enlaza el bundle en un directorio donde el host busque plugins OpenFX: en macOS `/Library/OFX/Plugins`, o cualquier directorio de `OFX_PLUGIN_PATH`.

| Variable | Qué hace |
|---|---|
| `ATHENEA_MCP` | el servidor que arranca el plugin, en lugar del `athenea-mcp` de `Contents/Resources/bin` y de uno del `PATH` |
| `ATHENEA_OFX_DEBUG` | con cualquier valor: el plugin imprime por la salida de error la petición de cada frame y la respuesta del servidor (su tiempo) |

## Parámetros

!!! tip "El peón de ejemplo"
    El peón de [athenea-example-pawn.zip](https://github.com/jesusluque/athenea/releases/download/v0.3.0/athenea-example-pawn.zip) en el nodo: `cloud` = `pawn_gs.usda`, translate `0 0.048 0.18`, distancia focal 50, apertura horizontal 36, near 0.01, y cualquier `sky` (su carpeta `hdri/` trae dos). Es el frame de las imágenes de esta página.

<figure markdown>
  ![El cielo en el nodo: la misma nube bajo un cielo, el mismo cielo girado 120 grados, y el interior de un taller.](assets/ofx-skies.jpg)
  <figcaption>El cielo en el nodo: la misma nube bajo un cielo, el mismo cielo girado 120 grados, y el interior de un taller.</figcaption>
</figure>

| Parámetro | Valor | Por defecto | Notas |
|---|---|---|---|
| `cloud` | fichero | ninguno | un `.usdc` o `.usda` con la nube (un `ParticleField3DGaussianSplat`). Vacío no dibuja nada |
| `translate` | 3 números | `0 0 5` | dónde está la cámara, en las unidades de la nube |
| `rotate` | 3 números, grados | `0 0 0` | alrededor de x, y y z |
| `rot_order` | `XYZ` … `ZYX` | `ZXY` | el orden en que se aplican las rotaciones, la primera letra primero; ZXY es el de Nuke por defecto |
| `focal` | mm | `50` | |
| `haperture` | mm | `24.576` | |
| `vaperture` | mm | `0` | 0 se ajusta al aspecto del formato, contando el aspecto de píxel, como Nuke ajusta una cámara a su formato |
| `near`, `far` | unidades de la escena | `0.1`, `10000` | los planos de recorte |
| `time_offset` | frames | `0` | se suma al frame para dar el tiempo USD de la nube, para una nube animada (con skinning) |
| `sky` | fichero | ninguno | una imagen HDR (`.hdr` o `.exr` lat-long) en todos los domos del stage; vacío conserva el del propio stage; un stage sin domo recibe uno |
| `sky_rotation` | grados | `0` | alrededor del eje vertical del stage |
| `sky_visible` | sí/no | sí | apagado deja el fondo transparente, para componer sobre una plancha; el cielo sigue iluminando |
| `output` | `rgba`, `depth`, `P`, `CryptoObject00` … `02` | `rgba` | la capa que muestra en RGBA un host que no admite planos; `depth` es la z de vista en todos los canales |

**La cámara.** La matriz se construye como Nuke construye la de un nodo Camera: las rotaciones en el orden elegido y luego la traslación, mirando hacia −z con +y arriba. Los knobs de un nodo Camera (`translate`, `rotate`, `rot_order`, `focal`, `haperture`) se enlazan a estos por expresión, así que la nube se mueve con la cámara de la plancha. El nodo renderiza frames enteros, no tiles.

**El cielo.** El cielo solo se cambia cuando cambia, porque cada cambio es una edición de sesión que Hydra sincroniza. El primer frame después de un cambio prepara el cielo nuevo (sus armónicos y su cadena prefiltrada); los frames siguientes cuestan lo que costaban.

## Capas

<figure markdown>
  ![Lo que da el nodo, de un solo render: RGBA (reiluminado, premultiplicado), profundidad, P (posición en el mundo) y Cryptomatte (un id por objeto: el cuerpo y la cabeza de vidrio).](assets/ofx-outputs.jpg)
  <figcaption>Lo que da el nodo, de un solo render: RGBA (reiluminado, premultiplicado), profundidad, P (posición en el mundo) y Cryptomatte (un id por objeto: el cuerpo y la cabeza de vidrio).</figcaption>
</figure>

| Capa | Canales | Contenido |
|---|---|---|
| color | `R G B A` | lineal, premultiplicado |
| `depth` | `Z` | la z de vista, 0 donde no se dibujó nada |
| `P` | `X Y Z A` | la posición en el mundo, calculada en el dispositivo a partir de la profundidad; `A` 1 donde se dibujó algo |
| `CryptoObject00` … `CryptoObject02` | `R G B A` | las capas del Cryptomatte: id, cobertura, id, cobertura |

Las Gaussians escriben el color, la profundidad y el Cryptomatte; `P` se deriva de la profundidad. Para las Gaussians no se escriben normales ni vectores de movimiento.

**Natron**, un host que admite planos, recibe todas las capas como planos de la salida del nodo, diga lo que diga `output`, dibujadas en una sola llamada; solo se dibujan los planos que pide el host. El writer de Natron usa por defecto half floats, que redondearían un id de Cryptomatte en otro: el matte se escribe con profundidad de bits float de 32 bits.

**Nuke** solo admite de un nodo OpenFX sus propios planos, así que allí `output` elige la capa que muestra RGBA; un nodo por capa da todas.

El manifest del Cryptomatte no llega al host, porque OpenFX no lleva metadatos: un matte se elige por id y no por nombre.

## Medidas

De la sección *The OpenFX plugin* de `docs/decisions.md`. Los tiempos del servidor son los que imprime `ATHENEA_OFX_DEBUG`.

| Caso | Resultado |
|---|---|
| Esfera unidad, 96k Gaussians, 640 × 360, por el host de prueba de OpenFX (`tests/ofx`), frente al render propio de `athenea-mcp` con la misma cámara | idéntico (relMSE 0); un frame en caliente le cuesta al servidor 16 ms |
| `P` frente a la profundidad, nube del gorrión, 480 × 270 | la distancia a lo largo del eje de vista, dentro de 7e-8 relativo; cada píxel cubierto se reproyecta sobre su propio centro |
| Nube con transfer del peón, 1024 × 1024, cielo cambiado | unos 0,4 s el primer frame tras el cambio, 50 ms los siguientes; devolver el cielo del stage dibuja un frame idéntico al del stage (relMSE 0) |
| Peón en Natron, 1920 × 1080, todos los planos, float de 32 bits | un EXR de 21 canales; color y `P` idénticos a los frames del host de prueba con la misma cámara (relMSE 0; `P` dentro de 1e-6) |
| Nube con transfer del Corvette (3 M Gaussians, 2,9 GB), 24 frames en Natron, 1920 × 1080, todos los planos | primer frame 7,5 s (el stage abierto y subido); 120–170 ms por frame con el cielo quieto; 190–220 ms con el cielo girando en cada frame; 14 s desde el primer render hasta la última escritura |

## Pendiente

- El orden de las rotaciones de Nuke está tomado de su documentación y aún no se ha comprobado en Nuke.
- La entrada de cámara propia de Nuke (su extensión de cámara de OpenFX) en lugar de knobs enlazados.
- Normales y vectores de movimiento de las Gaussians, que el rasterizador no escribe.
- El cielo como entrada de imagen del nodo en lugar de un fichero.
- Abrir `.ply` y `.spz` directamente en lugar de convertirlos antes.
- Builds de Linux y Windows, y un bundle firmado. La build de CMake compila el plugin en Linux, pero el registro da la build de Linux como pendiente; en Windows el plugin no se compila.
