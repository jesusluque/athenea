[English](operations.md) · Español

# Ejecutar athenea

Este documento es para quien ejecuta el motor: qué poner en el entorno, qué
admite cada subcomando, cómo se hace el trabajo para el que el motor existe,
qué le puede decir una escena USD, y qué hacer cuando se niega.

No explica cómo está hecho por dentro. Eso lo reparten cuatro documentos:

| Documento | Responde de |
|---|---|
| [`README.md`](../README.md) | por qué existe el motor, y una invocación de cada cosa |
| **este** | cómo se ejecuta, y cómo se autoriza una escena para él |
| [`development.es.md`](development.es.md) | cómo está hecho, y cómo se cambia |
| [`decisions.md`](decisions.md) | por qué es así, contra qué, y cuánto midió |

Este documento no enlaza nunca al código: enlaza al README para compilar, al
de desarrollo para los interiores, y llega al registro de decisiones solo a
través de él. Mantenerlo al día: una opción, un `athenea:*`, una variable de
entorno o un error impreso que cambien, cambian este fichero y su versión
inglesa en el mismo commit.

**Cómo se leen las tablas de opciones.** `athenea --help` imprime una descripción
de cada opción y ningún valor por defecto, así que las tablas de aquí llevan
lo que él no puede: el valor, el defecto, la unidad, y la restricción que se
comprueba mientras el comando corre y no mientras analiza la línea. Una opción
repetible va marcada ‹repetible›. Un flag no lleva valor; donde un flag se
escribe `!--no-x`, el comportamiento está activo y el flag lo apaga.

## 1. Ponerlo en marcha

### 1.1 Un binario

La sección *Building* del README tiene los requisitos con sus versiones y
prefijos, los tres scripts de dependencias, los submódulos y los cuatro
presets de CMake. Aquí no se repite. Dos presets más, `macos-arm64-webgpu` y
`linux-x86_64-webgpu`, son la build release con el backend WebGPU junto al de
la plataforma (`ATHENEA_WEBGPU=ON`, §1.4); necesitan Dawn en
`~/tools/dawn-138.0.7204.168` (`scripts/build-dawn.sh`, o `ATHENEA_DAWN_ROOT`)
y dejan `libdawn` junto a los binarios en `bin/`. Lo que importa después es el entorno en
el que corre el binario.

### 1.2 El entorno de ejecución

| Variable | Qué hace |
|---|---|
| `ATHENEA_SHADER_DIR` | de dónde se leen los shaders de Slang. Sin ella el motor busca un directorio `shaders` que contenga `athenea/` junto a la imagen desde la que se cargó su código, o hasta tres directorios por encima -- `<exe>/../shaders` para un programa, `<build>/shaders` para el plugin de Hydra cargado por usdview o Blender --, luego junto al ejecutable, y luego en el directorio con el que se compiló. `athenea info` imprime el que está usando. |
| `PXR_PLUGINPATH_NAME` | apunta una aplicación USD a `<build>/plugin/usd`, donde están el delegate de Hydra y los schemas codeless. La necesita cualquier host que no sea `athenea`. `athenea` registra por su cuenta `<su binario>/../plugin/usd` al arrancar, así que los schemas que aplica una nube convertida se escriben esté o no definida. |
| `ATHENEA_MATERIALX_ROOT` | un directorio que contiene los `libraries/` de MaterialX, que el compilador de materiales de hdAthenea lee en lugar de las bibliotecas que cargó el USD del host. Sin definir por defecto: se usan las del host. Para un host cuyo MaterialX es anterior al generador de Slang (Blender 5.3 trae 1.39.4: sin implementaciones `genslang` y con definiciones de nodo anteriores), se apunta a las de 1.39.5. Se lee una vez, cuando compila el primer material; los renderers propios del host conservan las suyas. |
| `AOFX_PLUGIN_PATH` | directorios extra de bundles AOFX, buscados antes que la ruta del sistema y antes que `--path`. |
| `ATHENEA_BACKEND` | qué dispositivo abrir, como un orden separado por comas: `metal,cuda,vulkan,d3d12,webgpu`. Las palabras desconocidas avisan y se saltan. `webgpu` solo se abre si se nombra, y solo en una build con `ATHENEA_WEBGPU` (si no, se informa como `not built in`). |
| `ATHENEA_WEBGPU_WEB_LIMITS` | sujeta un dispositivo WebGPU a los límites de un navegador en vez de a los de su adaptador. Sin definir o `0`: los del adaptador. `1` o `default`: los valores por defecto de la especificación de WebGPU, que todo navegador concede -- 8 storage buffers por etapa de shader, 16384 bytes de memoria de workgroup, 128 MiB por binding de storage, 256 MiB por buffer. `10` o `typical`: lo que informa el 98-99 % de los adaptadores -- 10 buffers, 32768 bytes, 256 MiB por binding, 256 MiB por buffer (este último no está medido, así que se queda en el valor por defecto). `S,W,B,F`: storage buffers, bytes de workgroup, MiB por binding, MiB por buffer. Un límite solo se baja, nunca se sube por encima del del adaptador. Un kernel que pase un límite falla al crear su pipeline, nombrando el límite. El dispositivo imprime los límites a los que quedó sujeto. Solo lo lee el backend WebGPU. |
| `ATHENEA_GPU_BUDGET` | la memoria del dispositivo que puede ocupar esta ejecución, en MiB. Sin ella el presupuesto es el working set recomendado de Metal (`recommendedMaxWorkingSetSize`); en CUDA y Vulkan no hay ninguno salvo que esta lo ponga. El dispositivo imprime el que usa (`GPU memory budget: N MiB`). Una reserva que lo pase falla como `OutOfMemory` antes de hacerse, los presupuestos de streaming y las sombras de splats se dimensionan con lo que deja (§3.3, §9), y es como se mantiene una ejecución por debajo de lo que dejan otros trabajos en la misma GPU. En Apple silicon una reserva además debe caber en la memoria física que el sistema tiene libre menos 1,5 GiB guardados para el resto de la máquina -- contando como libre lo que devuelve cuando se le pide (páginas inactivas, purgables y especulativas, y la caché de ficheros) -- diga lo que diga el presupuesto: pasado eso la máquina manda a swap la memoria de la GPU y deja de dibujar sus ventanas. |
| `ATHENEA_SHADER_CACHE` | dónde se cachean los shaders compilados entre ejecuciones. Por defecto, un directorio bajo el de caché de la plataforma. Borrarla cuesta un primer frame lento. |

Un binario compilado sin `ATHENEA_BUILD_VIEW` no tiene el subcomando `athenea view`:
es la forma esperada en un nodo de render sin ventana.

### 1.3 Qué se pierde con cada dependencia opcional que falte

Todo esto se busca al configurar, y se nota al ejecutar. `athenea info` informa de
lo que ve.

| Falta | Qué se pierde |
|---|---|
| OpenColorIO | el view transform de OCIO. AgX y ACES 2.0 siguen funcionando, y `--ocio-*` se rechaza. |
| Open Image Denoise | `--denoise` y `athenea:denoise` no hacen nada; un frame trazado se queda como lo reunió. |
| libwebp | las nubes `.sog` se rechazan al leerlas. |
| OpenVDB | los campos de volumen `.vdb` se rechazan al leerlos. |
| Cabeceras de OptiX (CUDA) | el ray tracing por pipeline en CUDA; los ray queries en línea siguen, así que casi todo el motor también. |
| zstd | las nubes `.spz` se rechazan, con `this build reads no .spz`. |
| Dawn (`ATHENEA_WEBGPU` apagado, por defecto) | el backend WebGPU: `ATHENEA_BACKEND=webgpu` informa `WebGPU (not built in)`. Nada más. |

### 1.4 Metal y CUDA

El motor abre un dispositivo y lo conserva. En macOS es Metal; en Linux es
CUDA donde hay un dispositivo CUDA, y Vulkan si no. `ATHENEA_BACKEND` manda sobre
ese orden.

Lo que cambia entre ellos, en la práctica:

- **Ray tracing.** Metal tiene ray queries en línea y estructuras de
  aceleración pero no pipelines de rayos, así que la ruta que necesita un
  pipeline no está disponible y `athenea info` lo dice. CUDA tiene las dos, con
  las cabeceras de OptiX.
- **El denoiser.** En Metal, Open Image Denoise corre en la cola del propio
  motor. En Vulkan corre a través de CUDA con memoria importada.
- **Media precisión.** `athenea info` dice si la hay; donde no la hay el motor se
  queda con buffers float y no cambia nada más.
- **WebGPU** (una build `*-webgpu`, `ATHENEA_BACKEND=webgpu`) es Dawn, el
  WebGPU de Chrome, sobre Metal o Vulkan por debajo. Está para ejecutar en
  nativo el backend del navegador: sin ray queries, sin estructuras de
  aceleración, sin denoiser, sin operaciones de wave, y con los shaders
  compilados a WGSL. Hoy los kernels del raster de splats compilan a WGSL; con
  `ATHENEA_WEBGPU_WEB_LIMITS` la proyección y la decodificación aún no caben en
  el límite de storage buffers de un navegador (docs/decisions.md, «WebGPU»).

### 1.5 Códigos de salida, y dónde se imprime un error

Cada subcomando imprime sus errores en la salida de error y termina con `1`,
salvo cuando la GPU se quedó sin memoria, que termina con `3` y dice de qué
pedir menos: el único fallo con el que un script puede hacer algo, volviendo a
intentarlo más tarde o más pequeño. Una ejecución correcta termina con `0`. No
hay más códigos: un pipeline debe mirar el estado de salida y leer stderr, no
analizar stdout, que lleva el informe — tiempos, cuentas, la ruta escrita.

La falta de memoria se atiende antes de informarse. Un frame que falla por
ella devuelve lo que el motor vuelve a hacer cuando se le pide (los buffers
crecidos del rasterizador, las estructuras de los trazadores, el denoiser),
renuncia a una cosa más, y se intenta otra vez, una: primero las sombras de
splats, después un nivel de detalle cada vez (el siguiente nivel más grueso de
un grupo LOD, el corte de un asset en streaming al doble de píxeles y con la
mitad de su presupuesto), hasta cuatro. Cada paso es un aviso que nombra a qué
renunció. Solo un frame que sigue fallando es el error del comando.

`-v` (o `--verbose`) antes del subcomando enciende el log de depuración, que
también va a stderr.

## 2. La línea de comandos

Una invocación es `athenea [-v] <subcomando> [opciones]`. Cada subcomando está
abajo con su tabla completa.

### 2.1 `athenea info` — el dispositivo, y qué sabe hacer

Su primera línea es la versión, tal como la imprime `athenea --version`: la del
proyecto, y la etiqueta `vX.Y.Z` de la que salió el build cuando salió de una
versión publicada (`CHANGELOG.md`).

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `--backend` | `metal` \| `cuda` \| `vulkan` \| `webgpu` | la preferencia de la plataforma | un backend, no una lista; `webgpu` solo en una build con `ATHENEA_WEBGPU` |

Imprime el backend y la tarjeta, si rasteriza, si traza rayos por pipeline y
por ray query, si tiene timestamps, medias y memoria unificada, la versión de
Slang, el directorio de shaders en uso, la versión de MaterialX, el denoiser,
si lleva OpenColorIO dentro, y cuántas librerías TBB hay cargadas en el
proceso. Esa última línea es un test en sí misma: dos TBB en un proceso es un
defecto.

```sh
athenea info
```

### 2.2 `athenea render` — ficheros de splats a un EXR

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `--splats` | rutas ‹repetible› | — | `.ply`, `.splat`, `.spz`, `.sog`, `.athc` |
| `--points` | rutas ‹repetible› | — | `.ply`, `.xyz`, `.txt`, `.pts`, `.csv`, `points3D.txt`/`.bin` de COLMAP |
| `--technique` | `raster` \| `rt` \| `rt-hw` \| `rt-bvh` \| `reference` \| `reference-rt` | `raster` | las `reference*` son la verdad de referencia en GPU, lentas a propósito |
| `--point-route` | `raster` \| `discs` | `raster` | |
| `--point-size` | número | `0.01` | unidades de mundo, o píxeles con el flag siguiente |
| `--point-pixels` | flag | apagado | leer `--point-size` como píxeles |
| `--edl` | número | `0` | fuerza del eye-dome lighting, ruta raster |
| `--surface` | número | `0` | holgura en profundidad del surface splatting, ruta raster |
| `--eye` | 3 números | el target más 1.2 de la extensión de la escena en z | |
| `--target` | 3 números | el centro de la caja de todas las nubes | |
| `--up` | 3 números | `0 1 0` | |
| `--rotate-x` | 1 número | ninguno | grados, a todas las nubes; las de COLMAP quieren `180` |
| `--scale` | 3 números | ninguno | a todas las nubes |
| `--focal` | número | `35` | mm, con apertura de 24.576 mm |
| `--size` | `ANCHOxALTO` | `1920x1080` | |
| `--near` | número | `0.01` | unidades de escena |
| `--degree` | 0..3 | `3` | tope de armónicos evaluados |
| `--lod` | número | `0` | píxeles que puede abarcar una celda fundida; 0 dibuja todos los splats |
| `--stream-budget` | entero | `0` | `.athc`: splats que se quedan en el dispositivo; 0 lee el fichero entero |
| `--no-antialias` | flag | apagado | quita la compensación de filtro 2D de Mip-Splatting |
| `-o`, `--output` | ruta | `out.exr` | RGBA en media más un canal `Z` |

Lo que se comprueba al ejecutar, no al analizar:

- un `.athc` necesita `--lod`; sin él, se rechaza el fichero;
- `--lod` necesita `--technique raster` y ningún punto;
- las técnicas que no son raster rechazan los puntos;
- hay que dar algo: `--splats` o `--points`.

```sh
athenea render --splats capture.ply --size 1920x1080 -o shot.exr
```

### 2.3 `athenea bench` — el mismo frame, cronometrado

Admite todas las opciones de `athenea render` menos `-o`, y añade:

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `--repeat` | entero | `20` | frames dibujados |
| `--stages` | flag | apagado | espera tras cada etapa para cronometrarlas por separado; más lento, y por eso sus números no son los de un frame real |

No escribe imagen. Con `--stages` informa por separado de la proyección, el
sort por profundidad, las cuentas, el emit, el sort por tile y el blend.

### 2.4 `athenea convert` — un fichero de splats a USD, o a un `.athc`

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `input` | ruta, obligatoria | — | `.ply`, `.splat`, `.spz`, `.sog`, `meta.json` |
| `output` | ruta, obligatoria | — | `.usda`, `.usdc`, `.usd` o `.athc` |
| `--chunk-splats` | entero | `65536` | `.athc`: splats por chunk |
| `--max-group-fraction` | número | `0.5` | `.athc`: grupos por splat que puede tener el nivel fundido más fino |
| `--degree` | 0..3 | `3` | armónicos que se conservan |
| `--rotate-x` | número | `0` | grados; las nubes de COLMAP quieren `180` |
| `--no-camera` | flag | apagado | no añadir `/World/Camera` a la escena escrita |

Un `.athc` de salida rechaza un `--rotate-x` distinto de cero: el contenedor
guarda la nube como está, y el giro pertenece al prim que lo referencia.

Una escena escrita aquí dice `metersPerUnit = 1`: ninguno de los formatos de
entrada registra una unidad, y la escala de una captura se toma como metros.
Una nube en otra unidad se escala donde se referencia, o se edita el
`metersPerUnit` de la escena.

```sh
athenea convert capture.ply scene.usda
athenea convert capture.ply capture.athc --chunk-splats 131072
```

### 2.4.1 `athenea decimate` — menos gaussianas, conservadas donde importan

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `input` | ruta, obligatoria | — | una escena (`.usd`, `.usda`, `.usdc`: su primer ParticleField gaussiano), o `.ply`, `.splat`, `.spz`, `.sog` |
| `output` | ruta, obligatoria | — | `.usda`, `.usdc`, `.usd` |
| `--prim` | ruta | el primero | el ParticleField que leer, donde la escena tiene más de uno |
| `--colour-tolerance` | 0..1 | `0.05` | cuánto puede alejarse el color o la opacidad de una gaussiana de la que la sustituye antes de contar como distinta. El color se compara en el espacio propio de la nube: sRGB para una captura, luz lineal para una nube que dice `primvars:athenea:splat:linear` -- donde el mismo número es un paso más grueso en las sombras y más fino en las luces |
| `--outliers` | 0..1 | `0.05` | la fracción de las gaussianas que representa una fusión que puede ser distinta |
| `--flat-tolerance` | 0..1 | `0.08` | donde las gaussianas son discos, cuánto más gruesa que ellos puede ser una fusión, frente a su anchura |
| `--reach` | 0.1..10 | `1.5` | a cuántas de sus desviaciones típicas puede estar una gaussiana de la fusión que la sustituye |
| `--degree` | 0..3 | `3` | armónicos que se conservan |
| `--no-camera` | flag | apagado | no añadir `/World/Camera` |

Una escena sale como entró, con menos gaussianas: se copia su capa raíz -- el
rig y su animación, las luces, la cámara, el manifiesto Cryptomatte, las
variantes, cada primvar constante --, las rutas de assets relativas se anclan a
donde estaban, y cada array de una gaussiana de largo se funde sobre lo que
representa cada gaussiana conservada. El log nombra cada uno:

```
decimate: carries primvars:athenea:splat:cryptoObject (1 a gaussian)
decimate: carries primvars:skel:jointIndices (4 a gaussian)
```

Cómo se funde cada uno depende de lo que es: `skel:jointIndices` con
`skel:jointWeights` como un rig (el peso de cada articulación sumado, las cuatro
más pesadas conservadas); un array cuyo nombre acaba en `shadowBits` bit a bit;
cualquier otro entero -- un id, una parte, una lámina -- como lo que no se
puede fundir entre sí; `primvars:athenea:splat:normal` como una dirección (la
media ponderada hecha de nuevo un vector unitario); cualquier otro float --
metallic, roughness, un transfer, `primvars:athenea:splat:emission` -- como una
media. Metallic, roughness y transmisión también se comparan
como el color. Un array muestreado en el tiempo se funde muestra a muestra. La
copia conserva el `metersPerUnit` y el `upAxis` de la fuente. Un fichero de
splats se escribe como una escena nueva, como la escribe `athenea convert`, en
metros.

```sh
athenea decimate car_gs.usdc car_fewer.usdc
athenea decimate capture.ply capture_fewer.usdc --colour-tolerance 0.1
```

### 2.5 `athenea stage` — una escena USD por el delegate de Hydra

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `stage` | ruta, obligatoria | — | `.usd`, `.usda`, `.usdc` |
| `--camera` | ruta de prim | la primera cámara de la escena | |
| `--time` | número | `0` | time code de USD |
| `--size` | `ANCHOxALTO` | `1920x1080` | |
| `--technique` | `raster` \| `rt` | `raster` | el `athenea:technique` del delegate |
| `--visibility` | `automatic` \| `raster` \| `rays` \| `bvh` | `automatic` | cómo se ven las mallas |
| `--path-samples` | entero | `1` | rt: caminos por píxel en cada pasada |
| `--path-bounces` | entero | `1` | rt: rebotes tras el primer impacto; atravesar un vidrio (entrar en él, salir de él, dentro de él) no es uno, hasta 8 por camino |
| `--path-total` | entero | `1` | rt: caminos por píxel hasta los que se dibuja |
| `--denoise` | flag | apagado | rt: denoise cuando alcanza el total |
| `--default-lights` | flag | apagado | un dome y un sol en la capa de sesión, para una escena sin luces |
| `--shutter` | `ABRE:CIERRA` | sin poner | en frames; solo para una cámara hecha con `--eye` |
| `--variant` | `/World{set=valor}` ‹repetible› | ninguna | una selección de variante antes del primer frame |
| `--motion-buckets` | 1..8 | `4` | rt: rodajas del obturador |
| `--splat-override` | `PRIM=M,R,T[,R,G,B[,REPLACE]]` ‹repetible› | ninguno | metallic, roughness, transmission y un tinte para las gaussianas que vinieron de `PRIM` (una ruta del manifest Cryptomatte de la nube, `*` para todas); `-1` deja el de la gaussiana; `REPLACE` 1 hace de R,G,B el color en vez de un factor sobre él. Sólo una nube convertida con ids tiene manifest. Falla con un prim que ninguno nombra |
| `--refine` | entero | `0` | niveles de subdivisión; 0 dibuja la malla de control |
| `--light-samples` | entero | `1` | muestras por luz y píxel |
| `--no-transfer-indirect` | flag | se suma | dibujar una nube con transfer sin su mitad rebotada |
| `--splat-reflections` | flag | apagado | rt: una gaussiana refleja la nube a la que pertenece, no sólo el cielo |
| `--splat-shadows` | flag | apagado | rt: una nube relit se sombrea a sí misma, un rayo por splat |
| `--no-antialias` | flag | antialias encendido | |
| `--no-cloud-shadows` | flag | sombras de nube encendidas | |
| `--cloud-shadow-texels` | entero | `1024` | por lado, por luz |
| `--cloud-shadow-density` | número | `1.0` | multiplicador de la profundidad óptica de la nube |
| `--cloud-shadow-terms` | `0`, `1`, `3`, `5`, `7` | `0` | 1 es solo el total, el resto añaden pares de Fourier; 0 deja decidir a quien recibe |
| `--eye` | 3 números | ninguno | una cámara propia, en vez de una de la escena |
| `--target` | 3 números | ninguno | adónde mira esa cámara |
| `--up` | 3 números | `0 1 0` | |
| `--frame-all` | flag | apagado, e implícito si la escena no tiene cámara | encuadra todo |
| `--focal` | número | `35` | mm, apertura 24.576 mm |
| `--fstop` | número | `0` | 0 es estenopeica |
| `--focus` | número | `0` | la profundidad enfocada, unidades de escena |
| `--near` | número | `0.1` | |
| `--far` | número | `100000` | |
| `-o`, `--output` | ruta | `out.exr` | RGBA en media más `Z` |
| `--render-settings` | ruta de prim | sin poner | renderiza los productos de ese prim y para |
| `--frames` | entero | `1` | repite, e informa del primero, la mediana y el más rápido |

```sh
athenea stage shot.usda --camera /World/Camera -o shot.exr
athenea stage shot.usda --technique rt --path-total 256 --denoise -o shot.exr
athenea stage shot.usda --render-settings /Render/Settings
```

### 2.6 `athenea view` — una ventana sobre una escena

Solo existe en una compilación con el viewer. Los controles de la ventana
están en §5.

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `stage` | ruta, obligatoria | — | |
| `--camera` | ruta de prim | una cámara libre que encuadra la escena | |
| `--technique` | `raster` \| `rt` | `raster` | |
| `--visibility` | `automatic` \| `raster` \| `rays` \| `bvh` | `automatic` | |
| `--size` | `ANCHOxALTO` | `1600x900` | en puntos, no en píxeles |
| `--frames` | entero | `0` | cierra tras tantos frames e imprime sus tiempos; 0 corre hasta que se cierre la ventana |
| `--light-samples` | entero | `1` | 1 es interactivo |
| `--choose-lights` | flag | apagado | una luz por muestra, elegida por potencia |
| `--path-samples` | entero | `1` | rt: caminos por píxel y frame |
| `--path-bounces` | entero | `4` | atravesar un vidrio no es un rebote, hasta 8 por camino |
| `--path-total` | entero | `64` | dónde el frame cuenta como convergido, y se denoisea |
| `--denoise` | flag | apagado | |
| `--no-default-lights` | flag | luces por defecto encendidas | una escena sin luces se queda a oscuras |
| `--edr` | flag | apagado | rango extendido: superficie float y ACES 2.0 hasta el pico de la pantalla |
| `--ocio-config` | ruta o URI | ninguno; `ocio://studio-config-latest` si se da display o view | |
| `--ocio-display` | nombre | el del config | |
| `--ocio-view` | nombre | el del display | |
| `--snapshot` | ruta | ninguna | con `--frames`: el último frame tal como se ve, paneles incluidos, a ese EXR |
| `--capture` | directorio | ninguno | cada frame tal como se ve, paneles incluidos, un PNG por frame (`frame_00000.png` y sucesivos): la grabación del viewer reproduciendo |
| `--isolate` | ruta de prim | ninguna | enseña solo la matte de ese prim; implica la salida Cryptomatte |
| `--aov` | `color`, `depth`, `primId`, `instanceId`, `elementId`, `Neye`, `normal`, `cryptomatte`, `CryptoObject00`..`02` | `color` | con qué salida arranca la ventana |
| `--fstop` | número | `0` | el diafragma de la cámara libre; una de la escena trae el suyo |
| `--focus` | número | `0` | |
| `--variant` | `/World{set=valor}` ‹repetible› | ninguna | |
| `--hdri` | directorio ‹repetible› | la carpeta donde está la imagen de cada domo | cielos que ofrece el combo **Sky** para los domos de la escena (`.hdr`, `.exr`) |
| `--play` | flag | apagado | arranca con la línea de tiempo reproduciendo |
| `--every-frame` | flag | apagado | un time code por frame dibujado, en vez de por el reloj |
| `--shutter` | número | `0` | frames que el obturador está abierto; 0.5 es un obturador de 180 grados |

```sh
athenea view shot.usda
athenea view car_gs.usdc --aov cryptomatte
athenea view car_gs.usdc --isolate /root/Kapoot/Object_57 --frames 10 --snapshot matte.exr
```

### 2.7 `athenea live` — una escena sobre un reloj

Cada frame se dibuja para su instante ST 2059-1, libre sobre el reloj de esta
máquina o siguiendo a un maestro PTP.

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `stage` | ruta, obligatoria | — | |
| `--camera` | ruta de prim | la primera | |
| `--size` | `ANCHOxALTO` | `1920x1080` | |
| `--technique` | `raster` \| `rt` | `raster` | |
| `--rate` | `25`, `50`, `29.97`, `59.94`, `23.976` o `N/D` | `25` | frames por segundo |
| `--ptp` | host | vacío, el reloj de esta máquina | seguir a este maestro PTP |
| `--port` | entero | el de PTP | el 319 pide privilegios; vale cualquiera en el que coincidan los dos lados |
| `--domain` | entero | `0` | dominio PTP |
| `--lock-timeout` | segundos | `10` | cuánto se espera al enganche |
| `--frames` | entero | `25` | frames escritos |
| `--start` | número | el inicio de la escena | el tiempo USD que suena primero |
| `--at` | `HH:MM:SS:FF` | sin poner | el instante UTC de reloj de pared del primer frame |
| `--tai-utc` | segundos | el del alineamiento | TAI menos UTC, para los timecodes |
| `-o`, `--output` | ruta | `live.####.exr` | las `#` se vuelven el número de frame |

Cada frame escrito lleva su timecode y su cadencia como atributos EXR, junto
al instante TAI, el índice de frame, el tiempo USD y cuánto se retrasó el
despertar.

### 2.8 `athenea mesh2splat` — un modelo en gaussianas

La conversión, y el bake de luz que viene después. Qué hace cada etapa está en
[development.es.md §6](development.es.md#6-hornear-una-gaussiana-entero); la
receta es §3.1.

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `stage` | ruta, obligatoria | — | una escena con mallas |
| `-o`, `--output` | ruta | `splats.usda` | `.usda`, `.usdc`, `.usd`, o `.athc` con niveles de detalle: solo las gaussianas y sus normales de sombreado (sin metallic/roughness/transmission, ids Cryptomatte, índice del vidrio, eje vertical ni unidad); con él se rechazan `--skinned`, `--transfer` y `--lod-levels` |
| `--prim` | ruta de prim | todas las mallas | solo las que cuelgan de esa ruta |
| `--hide` | ruta de prim, repetible | ninguna | se deja fuera con todo lo que cuelga de ella, como invisible (opinión de sesión; el fichero no cambia) |
| `--thin-glass` | ruta o nombre de material, repetible | ninguno | un ajuste a mano: un material que transmite leído como de pared delgada aunque ni él ni su malla lo digan. Un vidrio ya se lee de pared delgada donde su material lo dice o donde su malla es una lámina o una losa -- más de una doscientosava parte de sus aristas abiertas, o alguna compartida por más de dos, o el doble de su volumen sobre su área por debajo de cuatro celdas o de una cincuentava parte de su tamaño, lo que la conversión mide en el dispositivo e imprime por malla; esto nombra uno que la malla no muestra. Un vidrio sólido cubre `--glass-opacity` de lo que hay detrás; una pared delgada cubre lo que refleja de frente (menos de una décima), así que el habitáculo se ve |
| `--solid-glass` | ruta o nombre de material, repetible | ninguno | el ajuste en el otro sentido: un material que transmite que se queda sólido mida lo que mida su malla |
| `--resolution` | entero | `512` | celdas a lo largo del lado largo de la caja sobre la que se mide la densidad |
| `--lod-levels` | entero | `1` | niveles de detalle: la conversión otra vez a la mitad de resolución cada vez; `-o` pasa a ser la escena que los dibuja como una nube, cada nivel un `<nombre>_lod<n>.usdc` a su lado |
| `--density` | `per-model` \| `per-mesh` | `per-model` | qué caja es esa |
| `--cell-min` | número | `0`, derivado | unidades de mundo; por malla, lo más fina que puede ser una celda |
| `--cell-max` | número | `0`, derivado | unidades de mundo; lo más gruesa |
| `--max-splats` | entero | `2000000` | el presupuesto, de toda la escena, repartido entre las mallas en proporción a lo que quiere cada una |
| `--cell-from-camera` | ruta de un prim cámara | ninguna | la celda de cada malla es lo que cubre un píxel de esa cámara donde la caja de la malla le queda más cerca (como poco el plano cercano), en `--time`; sustituye a `--density`. `--cell-min`/`--cell-max` la acotan, si se dan; no se deriva nada |
| `--camera-pixels` | entero, de 1 a 65536 | `1920` | con `--cell-from-camera`: píxeles a lo ancho de la apertura horizontal de la cámara |
| `--sigma` | número | `1.0` | anchura de la gaussiana en celdas; la de mesh2splat es 0.65 |
| `--flatness` | número | `0.1` | el tercer tamaño como fracción del menor de los otros dos |
| `--opacity` | número, de 0 a 1 | `1.0` | cobertura: cuánto de lo que hay detrás cubre la superficie convertida, multiplicado por la opacidad propia del material. Toda opacidad es cobertura -- esta, la constante del material, el valor de un mapa, lo que conserva un vidrio -- y cada gaussiana toma lo que necesita una de las varias que hay sobre un punto, así que 0.5 cubre la mitad a cualquier tamaño |
| `--glass-opacity` | número, de 0 a 1 | 1 con `--transfer` (TX), si no `0.6` | cobertura que conserva un sólido que transmite del todo. El fotograma de un transfer TX dibuja la imagen propia de la lente -- el cielo doblado por las dos caras, la lejana estimada por la curvatura, y el campo donde el paso está cerrado --, así que su vidrio cubre del todo lo que hay detrás; a 0,6 la cabeza del peón dejaba pasar la sala recta y sin tinte. Un vidrio de pared fina (y una opacidad de UsdPreviewSurface menor que uno en su modo `transparent` por defecto) cubre en cambio lo que la lámina refleja con su índice |
| `--opacity-cut` | número, de 0 a 1 | `0.5` | donde la opacidad de un material es un mapa sin umbral propio (el `opacity` de UsdPreviewSurface, el `opacity` de standard_surface, el `geometry_opacity` de OpenPBR, el `alpha` de glTF en BLEND): por debajo de esto no se escribe ninguna gaussiana; por encima, la superficie cubre lo que lee el mapa. El umbral propio del material (`opacityThreshold`, el `alpha_cutoff` de glTF en MASK) se usa en su lugar, y lo que conserva queda entero |
| `--max-cells` | entero | `262144` | celdas como mucho que recorre un triángulo |
| `--texture-size` | entero | `1024` | un mapa se lee no mayor que esto; 0 lo lee a su tamaño |
| `--no-textures` | flag | apagado | ignorar los mapas; los materiales se quedan con sus valores constantes |
| `--normal-map-turns` | flag | apagado | el mapa de normales gira la gaussiana, no solo su sombreado. La normal de sombreado se escribe en ambos casos (`primvars:athenea:splat:normal`) |
| `--no-displacement` | flag | apagado | ignorar el displacement de los materiales: toda gaussiana se queda sobre la malla plana |
| `--displace-refine` | entero, 1 a 64 | `8` | donde el relieve estira una celda, partirla en como mucho este número de gaussianas en cada uno de sus dos ejes |
| `--simplify` | número, 0 a 1 | `0` (apagado) | un bloque de celdas cuyo color, metallic, roughness, emisión, recorte y normales varían no más que esto -- en el bloque y en un bloque más allá de cada lado, todo dentro de un triángulo -- se convierte en una gaussiana de su tamaño. Colores y recorte van de 0 a 1; la emisión se compara en sus propias unidades, así que una brillante se funde menos; las normales se comparan por la longitud de su diferencia, más o menos el ángulo en radianes |
| `--simplify-levels` | entero, 1 a 5 | `3` | el bloque más grande que puede fundir `--simplify` tiene 2^esto celdas de lado |
| `--no-camera` | flag | cámara añadida | |
| `--no-bake` | flag | bake encendido | llevar el material para ser relit, en vez de hornear la luz |
| `--bake-samples` | entero | `128` | caminos que toma primero cada gaussiana; una transferencia toma estos más `--bake-extra` |
| `--bake-extra` | entero | `128` | caminos por gaussiana en promedio añadidos tras la primera pasada, repartidos por sqrt(varianza relativa / coste) de lo que vio la primera; 0 no traza ninguno |
| `--bake-pass-samples` | entero, 1 a 4096 | `64` | caminos que da cada pasada añadida a las gaussianas para las que es; una gaussiana recibe como mucho 16 de ellas |
| `--bake-filter` | entero, 0 a 8 | `3` | iteraciones à-trous del filtro de bake de splats (el bundle `SplatBakeFilter`) sobre la luz horneada, y sobre las mitades rebotadas de un transfer TX (su mitad indirecta y su campo reflejado); 0 no filtra nada. La primera alcanza una celda de 1.5 gaussianas, y cada una la dobla |
| `--bake-filter-luminance` | número | `4` | el borde del filtro: una vecina cuya luz difiere en este número de desviaciones típicas del ruido de la gaussiana cuenta e^-1 veces. Más alto suaviza más y conserva menos un borde tenue |
| `--bake-filter-indirect-only` | flag | desactivado | filtra solo la luz indirecta y deja la directa como se trazó |
| `--bake-bounces` | entero | `3` | tras el primer impacto |
| `--bake-degree` | 0..3 | `2` | armónicos ajustados; 0 es un color |
| `--transfer` | flag | apagado | hornear cuánto cielo llega a cada gaussiana, en vez de la luz que llegó |
| `--indirect` / `--no-indirect` | flag | encendido | con `--transfer`: guardar también la mitad que rebotó; un transfer zonal guarda sólo la mitad directa |
| `--transfer-degree` | 2 o 3 | 3 | con `--transfer`: el grado de los armónicos, 16 coeficientes directos y 48 indirectos con 3, los 9 y 27 del primer transfer con 2 |
| `--transfer-cells` | 0, 16 o 32 | 16 | con `--transfer`: celdas por lado de la rejilla de direcciones abiertas sobre la esfera entera (256 o 1024 bits por gaussiana); 0 escribe la de 8 x 8 del primer transfer sobre la mitad a la que mira la gaussiana |
| `--transfer-slice` | entero | `0` | con `--transfer`: gaussianas horneadas a la vez. 0 toma tantas como caben en una respuesta de 1,5 GB, y como mucho un millón donde se filtran las mitades rebotadas; el lote propio del horneado (524288) es lo menos que toma una porción salvo que esto pida menos. El filtro ve los vecinos dentro de una porción |
| `--specular-filter` | 0 a 4 | 1 con `--transfer`, si no 0 | cuánto ensancha la rugosidad de una gaussiana y la de su capa el giro de la superficie bajo ella: lo que se separan las normales de las esquinas, sobre el ancho de la gaussiana, sumado a la varianza de las pendientes (Toksvig). 0 deja la rugosidad del material; las conversiones relit y horneadas siguen en 0 salvo que se pida |
| `--validate` | directorio | — | medir la conversión material a material frente a la escena trazada, en este directorio (abajo) |
| `--validate-camera` | ruta de prim | `--cell-from-camera`, si no la primera cámara de la escena | la cámara de los fotogramas |
| `--validate-size` | W H | `960 540` | los fotogramas, en píxeles |
| `--validate-paths` | entero | `512` | caminos por píxel que reúne el GT |
| `--validate-bounces` | entero | `6` | rebotes de los caminos del GT |
| `--validate-material` | ruta de prim o nombre | todos los materiales | sólo este (repetible) |
| `--validate-sky` | `white` o un fichero de imagen | las luces de la escena | cada fotograma bajo otro cielo -- una constante de radiancia uno, o esa imagen en los domos de la escena -- con sus demás luces apagadas; el GT se guarda como `gt_<cielo>.exr` |
| `--transfer-lobes` | 0 a 2 | `0` | con `--transfer`: guardarlo como este número de lóbulos zonales en el marco propio de cada gaussiana (el bundle `SplatTransferZonal`); 0 es dos lóbulos con `--skinned` y nueve armónicos en el mundo en otro caso |
| `--skinned` | flag | apagado | llevar el esqueleto; obliga a `--no-bake`, guarda un `--transfer` como lóbulos zonales |
| `--range` | `INICIO:FIN[:PASO]` | el rango de la escena | time codes que guarda una nube con esqueleto |
| `--default-lights` | flag | apagado | un dome y un sol para el bake, en una escena sin luces |
| `--time` | número | `0` | el instante en que se posa la escena y traza el bake |
| `--path` | directorio ‹repetible› | ninguno | directorios extra de bundles AOFX |

`--skinned` y un bake se rechazan juntos: una nube que se mueve no puede
llevar luz horneada en una pose, así que la conversión lo dice y conserva el
material. `--skinned` y `--transfer` van juntos: el transfer se guarda como
lóbulos zonales en el marco de cada gaussiana, que giran con ella (abajo).

Una malla cuyos GeomSubsets (familia `materialBind`) enlazan materiales
propios se convierte un subset cada vez, cada uno con su material, y las caras
que no reclama ningún subset con el de la malla; el log nombra el prim de cada
subset. Sus gaussianas conservan el id Cryptomatte de la malla.

La salida se escribe entera o no se escribe: como `.<nombre>.partial-<pid>.<ext>`
en el mismo directorio, y renombrada a `-o` cuando está completa. Una
conversión que falla no deja nada en `-o` -- o deja el fichero que ya había,
tal como estaba -- y borra su fichero parcial. Con `--lod-levels`, cada nivel
y la escena que los dibuja se escriben así.

### 2.9 `athenea visibility` — lo que proyecta una nube con esqueleto, por partes

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `stage` | ruta, obligatoria | — | la escena de la nube |
| `--prim` | ruta de prim | `/World/Splats` | el ParticleField |
| `--skeleton-stage` | ruta, obligatoria | — | la escena origen, por la jerarquía de joints |
| `--skeleton-prim` | ruta de prim, obligatoria | — | el Skeleton en ella |
| `-o`, `--output` | ruta | ninguna, edita en el sitio | escribir una copia en su lugar |
| `--parts` | entero | `12` | en cuántas partes se corta el rig |
| `--min-joints` | entero | `6` | un subárbol menor que esto se queda con la parte de su padre |
| `--grid` | entero | `24` | sondas por eje de la caja de una parte |
| `--octave` | entero | `16` | direcciones por lado del mapa octaédrico |
| `--cut` | número | `0.001` | transmitancia bajo la cual un rayo del bake se detiene |
| `--time` | número | `0` | el instante en que se compromete la escena |

### 2.10 `athenea aofx` — los plugins de efectos

Dos subsubcomandos.

`athenea aofx list` imprime cada bundle encontrado, cargado o rechazado, y qué
declara: sus efectos, sus entradas y sus parámetros con sus defectos.

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `--path` | directorio ‹repetible› | ninguno | se busca después de `$AOFX_PLUGIN_PATH` |

`athenea aofx run` corre un efecto sobre ficheros EXR.

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `effect` | identificador, obligatorio | — | por ejemplo `tv.mediapro.aofx.invert` |
| `inputs` | rutas, obligatorias ‹repetible› | — | en el orden de entradas del efecto, o `Clip=ruta` |
| `-o`, `--output` | ruta | `out.exr` | |
| `--param` | `nombre=valor` o `nombre=v1,v2,...` ‹repetible› | los del efecto | |
| `--time` | número | `0` | tiempo de frame |
| `--path` | directorio ‹repetible› | ninguno | |

```sh
athenea aofx list
athenea aofx run tv.mediapro.aofx.invert shot.exr -o inverted.exr
```

### 2.11 `athenea compare` — lo que contiene una imagen, medido en la GPU

Con una imagen imprime la media y el valor más alto de cada canal. Con dos, eso
mismo para ambas y después cuánto se aleja la primera de la segunda, que se
toma como referencia: el error HDR relativo (`relMSE`, p99 y mayor diferencia
relativa) y la distribución en valores de código sRGB de 8 bits (p99, máximo,
píxeles por encima de 2). La CPU solo lee los ficheros; cada número es de un
kernel. Los kernels son los del efecto Measure (§7.1), corridos por el host
AOFX como los corre un compositor, así que el bundle tiene que estar en la
ruta de búsqueda: uno compilado en este árbol siempre lo está.

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `image` | ruta, obligatoria | — | un EXR |
| `reference` | ruta | ninguna | un EXR del mismo tamaño |
| `--window` | `X0 Y0 X1 Y1` | la imagen entera | píxeles en [X0, X1) × [Y0, Y1), filas contadas desde abajo; solo las medias, las diferencias son de toda la imagen. Un X1 o Y1 de 0, o más allá del borde, es el borde |
| `--heatmap` | ruta | ninguna | escribe además la imagen del efecto, un EXR del tamaño de `image` |
| `--show` | `source`, `difference`, `relative`, `codes` | `codes` | lo que dibuja `--heatmap`: el `mode` del efecto Measure (§7.1) |
| `--gain` | número ≥ 0 | `1` | por cuánto se multiplica `--heatmap`; los números no dependen de él |
| `--path` | directorio ‹repetible› | ninguno | directorios de bundles buscados después de `$AOFX_PLUGIN_PATH` |

Una media conserva el signo, y una diferencia no: un horno blanco que no debe
devolver más de 1, o un plano convertido que debe cubrir todos sus píxeles, es
una media.

```sh
athenea compare cloud.exr mesh.exr
athenea compare furnace.exr --window 192 192 320 320
athenea compare render.exr golden.exr --heatmap where.exr --show relative --gain 4
```

### 2.12 `athenea migrate` — los ficheros de lucabRTrender con los nombres de athenea

athenea es lucabRTrender con otro nombre, y un fichero escrito antes del cambio
no nombra nada de lo que este motor lee: se ignoran sus esquemas, sus primvars,
sus settings y sus nubes `.lrtc`. `migrate` escribe una copia con los nombres
nuevos; nunca escribe en la entrada.

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `input` | ruta, obligatoria | — | `.usda`, `.usdc`, `.usd`, `.usdz` o `.lrtc` |
| `-o`, `--output` | ruta, obligatoria | — | la misma clase de fichero: una capa como `.usda`, `.usdc` o `.usd` (un `.usd` conserva la codificación de la entrada), un paquete como `.usdz`, un `.lrtc` como `.athc`; nunca la entrada |
| `-r`, `--recursive` | flag | no | migra también cada capa, paquete y `.lrtc` que el fichero nombra -- sublayers, references, payloads, value clips, atributos de tipo asset -- y que esté bajo `--root`, cada uno al mismo sitio bajo el directorio de la salida |
| `--root` | directorio | el de la entrada | lo que `--recursive` puede copiar; debe contener la entrada, y no puede ser el directorio de la salida |
| `-q`, `--quiet` | flag | no | imprime solo los avisos y los totales |

Qué se renombra, capa a capa, sin componer la escena (cada capa conserva sus
propias opiniones, variantes incluidas):

| Antes | Después |
|---|---|
| `LrtSplatEditAPI`, `LrtSplatLightingAPI`, `LrtSplatSkinningAPI`, `LrtPointStyleAPI`, `LrtStreamedAssetAPI`, `LrtSplatVisibilityAPI`, `LrtSplatCryptomatteAPI`, `LrtVolumeAPI` en `apiSchemas` | `Athenea…API` |
| toda propiedad cuyo nombre tenga un componente `lrt`: `primvars:lrt:splat:*`, los render settings `lrt:*`, `outputs:lrt:*` | lo mismo con `athenea`; se conservan valor, metadatos, time samples y conexiones |
| un destino de conexión o de relación que nombra una de esas propiedades | la propiedad renombrada |
| `hydra:rendererName` `lrt`, `HdLrtRendererPlugin` | `athenea`, `HdAtheneaRendererPlugin` |
| claves de `customData` y `customLayerData` con un componente `lrt` | lo mismo con `athenea` |
| una ruta de asset que acaba en `.lrtc` | `.athc` |
| un `.lrtc` (`LRTC`, versión 1) | un `.athc` (`ATHC`, versión 1, sin normales); el contenido se copia tal cual |

Rutas de asset. Una ruta relativa a un fichero que no se copia (una textura,
una capa sin `--recursive`, una fuera de `--root`) se hace absoluta cuando la
salida está en otro directorio, para que siga resolviendo (`anchored` en el
informe). Con `--recursive`, una ruta a una copia migrada nombra la copia:
relativa como lo era, o absoluta hacia donde está la copia. Dentro de un
`.usdz` toda ruta sigue siendo relativa y un `.lrtc` se convierte y se
renombra dentro del paquete; los ficheros conservan su orden, así que el
primero sigue siendo la capa raíz.

Cada renombrado se imprime, uno por línea (`schema`, `property`, `target`,
`value`, `metadata`, `asset`, `anchored`, `file`, `warning`), y un total. Un
fichero ya migrado se escribe sin cambios y no informa de ningún renombrado.
Un `warning` es algo que se dejó como estaba: una propiedad cuyo nombre nuevo
ya está escrito a su lado (gana la nueva), un `.lrtc` que la copia nombra y
que aún no existe (ejecute `migrate` sobre él), un `.lrtc` en una expresión o
en una plantilla de clips. USD no conserva los comentarios `#` de un `.usda`.

```sh
athenea migrate old/shot.usda -o new/shot.usda
athenea migrate ~/assets/Sparrow/FilmGs.usda -o ~/migrated/FilmGs.usda --recursive
athenea migrate cloud.lrtc -o cloud.athc
```

## 3. Tareas

### 3.1 Un modelo en una nube

`athenea mesh2splat` convierte las mallas de una escena en una nube de gaussianas
y la escribe como un `UsdVolParticleField3DGaussianSplat`. Los valores por
defecto convierten y hornean la luz dentro; lo que sigue son las decisiones
que vale la pena tomar a mano.
Se convierte lo que el renderizador dibuja: una malla invisible, o bajo un
prim invisible, no da gaussianas, y el prototipo de un PointInstancer se
convierte una vez por instancia, donde está cada una.

**Cuántas gaussianas, y dónde.** `--resolution` son celdas a lo largo del lado
largo de una caja, y `--density` dice qué caja: `per-model` mide todo lo que
se convierte junto, `per-mesh` mide cada malla por su cuenta. Per-model es lo
que hace el algoritmo original, y significa que un objeto grande de la escena
diluye a uno pequeño — un suelo de dieciséis unidades bajo un coche de dos le
da al coche un octavo de las celdas que tiene convertido solo. Per-mesh le da
a cada malla las mismas celdas sobre su propio lado largo, con la celda
acotada en unidades de mundo entre `--cell-min` y `--cell-max`, para que una
insignia no salga más fina de lo que el ojo puede usar ni un suelo más grueso
de lo que el frame puede enseñar. A cero, esos límites se derivan del modelo:
el más grueso es la celda del modelo, y el más fino un octavo de ella.

**Qué compra una celda, además de detalle.** Es también la nitidez de todo
reflejo que lleve la nube: un rayo cruza gaussianas repartidas sobre un par de
celdas de superficie, sus normales difieren en ese arco, y la mezcla promedia
sus direcciones de espejo sobre él. Medido sobre una esfera de oro de radio
uno que pide `specular_roughness` 0.08, contra esa misma esfera path traced
como malla: una celda de 0.0281 se lee como roughness **0.34**, 0.0141 como
**0.20**, 0.0070 como **0.14**. Una nube se lee como la malla a **`r + 9c/R`**,
con `c` la celda y `R` el radio de curvatura, así que un espejo de roughness
`r` quiere una celda por debajo de `r/9` de ese radio. Quince veces las
gaussianas costaron un 36 % más de tiempo por frame —13.4 ms a 18.2 ms a
900 x 340 y 64 paths— y dieciséis veces el disco, de 10 MB a 157 MB. Lo que
ata es `--max-splats` y el fichero, no el tracer.

La escena hero del Mustang a `--resolution 512`, con un suelo de dieciséis
unidades dentro:

| Densidad | Total | Parachoques | Capó | Suelo | Mallas corridas dos veces |
|---|---|---|---|---|---|
| `per-model` | 375 199 | 762 | 2 326 | 262 144 | 1 |
| `per-mesh` | 934 338 | 3 082 | 20 279 | 262 144 | 5 |

El suelo está en su presupuesto en los dos casos: son dos triángulos, y lo
decide el techo de celdas, no la densidad.

**El presupuesto.** `--max-splats` es un techo de toda la escena. Primero se
cuenta cada malla (cada GeomSubset de una) -- una pasada del efecto con sitio
para una gaussiana, que lo cuenta todo y no escribe nada -- y cuando quieren
más que el presupuesto se reparte en proporción: cada una recibe
`presupuesto × quiere / total` (y una como poco), y recorre una celda
`sqrt(quiere / parte)` veces más gruesa para querer más o menos eso. Es un
solo suelo de densidad para toda la escena: todas las mallas pierden densidad
por igual y no se tira ninguna. El aviso dice cuántas se querían, y la línea
de log de cada malla cuánto más gruesa recorrió. Una malla que aún quiere un
poco más que su parte tras engrosarla conserva las gaussianas de sus primeros
triángulos, en el orden de la malla. Contar cuesta una pasada corta por malla.

**Desde una cámara.** `--cell-from-camera` da a cada malla la celda de la
cámara que la va a mirar (la regla de Mesh2GS): `z × (apertura / píxeles) /
focal`, con `z` la distancia de la cámara al punto más cercano de la caja de
la malla (como poco su plano cercano, y cero dentro de la caja), así que una
celda cubre más o menos un píxel donde la malla está más cerca. Cada
triángulo de la malla recorre exactamente esa celda. Una cámara libre que no
está en la escena no la conoce la conversión, así que es una opción y no lo
de por defecto.

**Texturas.** Cada mapa viaja al dispositivo como float4, dieciséis bytes por
texel, así que un mapa de 4k son 268 MB y un coche con quince no cabe. La
conversión muestrea un mapa una vez por celda, y a `--resolution 512` el
modelo tiene 512 celdas de lado, así que casi todo un mapa de 4k se tira antes
de leerlo. `--texture-size 1024` es el techo por defecto; `0` los lee a su
tamaño, y eso es para un primer plano de un objeto.

**Displacement.** La altura de un material mueve sus gaussianas, y ahí una
nube es más barata que una malla: una gaussiana es un punto, así que subirla no
cuesta nada, mientras que una malla hay que cortarla más fina que el relieve
antes de poder moverla. La altura se lee del `displacement` de
UsdPreviewSurface (a través del `scale` y el `bias` de UsdUVTexture, en el
canal conectado), o de un nodo `displacement` de MaterialX
(`ND_displacement_float`, por su `scale`) al que apunte el terminal de
displacement del material; una constante sin mapa mueve toda la superficie.
Está en las unidades de la propia malla, así que una malla escalada por dos
desplaza el doble. Un displacement vectorial no se lee, y el log lo dice.

Cada gaussiana se pone sobre el relieve y se gira hacia él. Donde el relieve
es empinado, una celda de la superficie plana se convierte en un trozo más
largo de relieve, y la celda se parte en tantas gaussianas por eje como pida
ese estiramiento, hasta `--displace-refine`; una celda que quería más se queda
más fina, y el log las cuenta:

```
mesh2splat: 312 cells of relief wanted more than 8 gaussians along an axis and were left thinner (--displace-refine)
```

El recuento es el coste. Un suelo y una bola de adoquines a `--resolution
768`: 533 029 gaussianas en plano, 1 505 305 con displacement, casi todas en la
bola, cuyo relieve es tan alto como anchos son sus adoquines.

El bake de luz sale del punto de la superficie plana que hay bajo cada
gaussiana -- el trazador tiene la malla plana -- y la sombrea como mira el
relieve, así que el relieve se ilumina según está girado. Lo que no lleva es la
sombra del relieve sobre sí mismo: en el trazador no hay nada donde está el
relieve. `--no-displacement` convierte la superficie plana. La ruta de mallas
de Hydra ignora el displacement.

**Lo que emite.** La emisión de un material se lleva, gaussiana a gaussiana,
como la radiancia lineal con que se renderiza su malla: `emission` por
`emission_color` de standard_surface, `emission_luminance` por
`emission_color` de OpenPBR (nits, multiplicados tal cual, que es lo que hace
su grafo), `emissive` por `emissive_strength` de glTF, y `emissiveColor` de
UsdPreviewSurface. Un mapa sobre el color es el color y el peso lo multiplica;
un mapa sobre el peso se lee en un canal y el color lo multiplica. La línea de
log de una malla que emite luz lo dice:

```
mesh2splat: /World/Quad uses /World/Looks/Screen (colour 0.50 0.50 0.50, ..., emission 2.000000 2.000000 2.000000 x '.../emission_gradient.png')
```

Se escribe como `primvars:athenea:splat:emission` (§4.3) sólo donde algún
material de la escena emite. Las nubes reiluminadas (`--no-bake`) y con
transfer la suman al dibujarse; una horneada ya la lleva en sus colores. Lo que
no hace es iluminar nada: los triángulos emisivos de la malla son una luz para
el path tracer, las gaussianas de la nube no. Una capa de coat de OpenPBR sobre
la emisión, que en la malla la tiñe y la atenúa, no se lleva.

**Lo que pone sobre su base.** Lo que un material pone sobre su color, su
metalness y su roughness se lleva como constantes del material: el peso y el
tinte del reflejo dieléctrico (`specular_weight` y `specular_color` de
OpenPBR, `specular` y `specular_color` de standard_surface, `specular` y
`specular_color` de glTF; el tinte es también el color del borde de un metal)
al índice del specular (`specular_ior`, `specular_IOR`, `ior`); una capa de
coat transparente (`coat_weight`, `coat`, `clearcoat` de UsdPreviewSurface,
`clearcoat` de glTF, con su roughness y su índice; el coat de
UsdPreviewSurface está a su propio `ior`), con el `coat_darkening` de OpenPBR
(1 por defecto allí, ninguno en los otros vocabularios); y un sheen, su color
por su peso (el fuzz de OpenPBR, `fuzz_weight` x `fuzz_color`; `sheen` x
`sheen_color` de standard_surface; `sheen_color` de glTF) con su roughness. Un UsdPreviewSurface en su flujo specular
(`useSpecularWorkflow` 1) se lleva como el índice cuya reflectividad de frente
es el canal más brillante de su `specularColor`, teñido por el color dividido
por él, y sin metal. Un mapa sobre el peso o el color del specular, el peso o
la roughness del coat, o el color, el peso o la roughness del sheen se
muestrea en cada gaussiana, como los mapas de la base, en lugar de la
constante de la entrada -- los tres primeros mapas que tenga un material; más
allá de ellos, y sobre un índice o `coat_darkening`, queda la constante y el
log lo dice. Donde no hay nada escrito se toman
los valores por defecto de cada superficie (un coat de OpenPBR a 1.6, uno de
standard_surface de roughness 0.1 a 1.5). La línea de log de una malla cuyo
material pone algo encima dice qué:

```
mesh2splat: /World/Ball layers specular 1.00 x (1.00 1.00 1.00) at 1.500, coat 1.00 rough 0.00 at 1.450, sheen (0.00 0.00 0.00) rough 0.30
```

Se escriben (los nueve primvars de §4.3 desde `specularWeight`) sólo donde
algún material de la escena difiere del specular simple -- peso uno, blanco,
índice 1.5, sin coat, sin sheen -- y entonces para cada gaussiana. Las nubes
reiluminadas, con transfer y horneadas reflejan con ellos al dibujarse: un
bake guarda el cuerpo y deja los reflejos al frame. El bake distingue un metal
de un pulido por la metalness del material, así que un metal oscuro bajo una
laca (la pintura de un coche, una base de 0.05) se hornea como el metal que es
y no como nada.

**Menos gaussianas donde la superficie es igual.** Una gaussiana por celda es lo
que cuesta la superficie esté donde esté, y casi toda una superficie -- un panel
pintado, una pared, un suelo -- es igual de una celda a la siguiente.
`--simplify` recorre cada triángulo como un árbol de bloques de 8, 4 y 2 celdas
de lado, y convierte un bloque en una gaussiana de su tamaño donde lo que dicen
los mapas es igual en todo él, dentro de la tolerancia. La gaussiana de un
bloque es tan ancha como el bloque y su cola llega un bloque más allá de cada
lado, así que ese alcance también tiene que coincidir y tiene que caer dentro
del mismo triángulo: un borde de la textura conserva sus celdas, y el borde de
una malla no echa flecos. Por eso rinde en triángulos grandes -- suelos,
paredes, paneles -- y no en una malla más fina que sus bloques.

Un suelo de adoquines a `--resolution 768`, en plano: 262 145 gaussianas,
99 413 con `--simplify 0.02`. La bola de al lado, de 2304 triángulos, es más
fina que sus bloques y se queda en 270 884. El relieve está curvado en todas
partes y funde poco: de 349 701 a 309 972 con 0.02, 272 664 con 0.1. El
recorrido cuesta tiempo de conversión -- de 1.3 s a 6.2 s sin bake en esa
escena -- y luego el bake tiene menos gaussianas que trazar. Al simplificar, el
bake de cada gaussiana se reparte por su huella en vez de tomarse en su centro;
sube `--bake-samples` con él, o un brillo que cazan unos pocos caminos se
reparte por un bloque entero.

**Menos gaussianas después, en cualquier nube, con todo lo que lleva.** `--simplify` sólo funde dentro
de triángulos grandes. `athenea decimate` trabaja sobre la nube misma, la haya
hecho quien la haya hecho -- una conversión, una captura: construye los niveles
de detalle y se queda, para siempre, con la fusión más gruesa que representa lo
que tiene debajo, y con cada splat donde ninguna lo hace. Una fusión representa
a sus splats cuando caen dentro de `--reach` de ella, cuando, si son discos, no
es más gruesa de lo que su superficie permite, y cuando pocos de ellos
(`--outliers`) difieren de ella en color más que `--colour-tolerance`; y la cola
de una fusión, que llega más allá de sus splats, no puede caer sobre nada de
otro color. Una fusión se escribe más ancha de lo que la hacen sus momentos,
para que se solape con sus vecinas como se solapaban sus splats.

Medido en los adoquines convertidos a `--resolution 768` y horneados a 64
caminos, dibujados en raster a 1200 x 800 frente a la nube sin diezmar:

| | Se conserva | RMS de imagen |
|---|---|---|
| por defecto | 82.9 % | 0.0020 |
| `--colour-tolerance 0.1` | 37.9 % | 0.011 |
| horneada a 256 caminos, por defecto | 53.9 % | |
| la misma conversión sin bake, por defecto | 57.8 % | |

Una nube horneada lleva el ruido de su bake, una gaussiana distinta de la
siguiente en unos pocos por ciento, y la tolerancia tiene que superarlo: a
256 caminos los valores por defecto conservan un 53.9 % donde a 64 conservan un
82.9 %. Tarda segundos: 533 007 gaussianas en 5 s.

**El bake de luz.** Por defecto la conversión traza la luz de la escena hasta
cada gaussiana y la ajusta a armónicos, así que la nube queda bajo la luz en
la que se convirtió y no necesita luces para dibujarse. `--bake-samples` y
`--bake-bounces` son la calidad; `--bake-degree` es cuánta dirección conserva
el resultado — 0 es un color, 2 es donde un brillo empieza a parecer un
brillo. `--no-bake` conserva el material, y la nube la ilumina la escena
donde se ponga.

El bake se toma en dos mitades, la luz directa y la indirecta, como sumas:
`--bake-samples` caminos en cada gaussiana, después `--bake-extra` más en
promedio donde la primera pasada fue más ruidosa para lo que costaron sus
caminos, y después la luz filtrada entre gaussianas vecinas de un mismo prim
por el bundle `SplatBakeFilter`, que tiene que estar en la ruta de búsqueda
AOFX mientras `--bake-filter` sea mayor que 0. En el peón bajo un HDRI los
valores por defecto tardan un 10 % más que 256 caminos por gaussiana y
llevan la mitad de su error frente a un bake de 4096 caminos
(docs/decisions.md, "The bake's grain").

**Una nube que lleva el cielo en vez de la luz.** `--transfer` hornea un
transfer vector: para cada gaussiana, cuánto de un entorno le llega desde cada
dirección, que es geometría y no tiene ni color ni cielo dentro. El frame lo
combina con el cielo bajo el que está la nube, así que el mismo fichero es
correcto bajo cualquier entorno, donde una nube horneada lo es bajo uno solo.
Lo que cuesta es el fichero: nueve floats por gaussiana la mitad directa, y
veintisiete más la que llegó después de rebotar en la escena, que
`--no-indirect` deja fuera. Los rayos son los mismos, así que esa segunda
mitad no cuesta bake, y `athenea:splatTransferIndirect` la apaga al renderizar sin
volver a hornear. Un transfer y un light bake son excluyentes: uno es lo que
hizo la luz, el otro lo que haría cualquiera.

**Un transfer que gira con la gaussiana.** Nueve armónicos están en el mundo y
se quedan allí cuando un esqueleto gira la gaussiana. `--transfer-lobes 1` o
`2` -- y `--skinned`, donde dos es el valor por defecto -- guarda en cambio la
mitad directa como uno o dos lóbulos zonales por gaussiana, cada uno un eje
escrito en el marco propio de la gaussiana y tres coeficientes: diez floats por
gaussiana frente a nueve, sin mitad indirecta, y los bits de sombra dispuestos
sobre el marco de la gaussiana. Cada frame gira los ejes con el marco que tenga
la gaussiana, así que el transfer sigue al ala. Una nube con esqueleto se posa
en `--time` antes de que el bake la trace, porque la escena que traza está
posada ahí; lo que guardan los lóbulos es lo que esa pose dejó pasar alrededor
de cada gaussiana, así que la oclusión de otra extremidad en otra pose no está
en ellos. El ajuste es el bundle `SplatTransferZonal`, que tiene que estar en
la ruta de búsqueda AOFX, y el log dice su error relativo frente a los nueve
armónicos que sustituye, `|f - g| / |f|` sobre la esfera, como una mediana, un
percentil 90 y un percentil 99 (cada uno el borde superior de un cuarto de
octava):

```
mesh2splat: transfer kept as 2 zonal lobes in each gaussian's frame for <n> gaussians in <t> ms; relative error against the nine harmonics: median <a>, p90 <b>, p99 <c>
```

**Una nube que se mueve.** `--skinned` construye las gaussianas en la pose de
bind y le da a cada una los joints que la llevan, así que la nube se deforma
al renderizar con el Skeleton al que está atada. Un bake se rechaza con él,
porque la luz horneada en una pose está mal en todas las demás. Cada gaussiana
guarda además cómo cambian sus pesos a lo largo de ella
(`jointWeightGradients`, doce bytes por gaussiana), que es lo que la estira a
través de un pliegue como se estira su triángulo; una nube convertida antes de
que se escribieran sigue moviéndose, solo con la mezcla de sus joints, y los
gana al convertirla de nuevo.

```sh
athenea mesh2splat car.usda --density per-mesh --resolution 512 \
    --max-splats 20000000 -o car_gs.usdc
athenea mesh2splat bird.usda --skinned --resolution 1100 -o bird_gs.usdc
athenea visibility bird_gs.usdc --skeleton-stage bird.usda --skeleton-prim /World/Skel
```

La última línea es el tercer horneado: lo que la nube proyecta sobre sí misma,
por partes, para que un ala sombree el cuerpo en cualquier pose sin un rayo.
Edita el fichero de la nube en el sitio salvo que `-o` nombre otro.

**Una conversión medida por el motor.** `--validate DIR` convierte la escena
una vez por material, dejando como malla todas las demás, y mide cada una
frente a la escena trazada, sobre los píxeles de ese material -- la máscara es
el Cryptomatte del fotograma de mallas, así que una malla de dos materiales
(GeomSubsets) cuenta en los dos. El GT se traza una vez en `DIR/gt.exr` y la
siguiente ejecución del mismo tamaño lo vuelve a leer; bórralo para trazarlo de
nuevo. Todas las demás opciones son las de la conversión, así que lo que se
mide es `--transfer`, `--no-bake` y el resto. `--hide` deja sus prims fuera
de todo -- el GT, cada fotograma y el horneado -- mediante `DIR/hidden.usda`,
una capa que los desactiva; un GT trazado antes con otros prims ocultos se
vuelve a leer igual, así que da a cada `--hide` su propio directorio. Lo que
escribe:

| fichero | qué |
|---|---|
| `validate.json` | por material: mallas, gaussianas, la parte del fotograma, relMSE, p99, media y media del GT sobre sus píxeles, y el relMSE y la media de la malla rasterizada |
| `<material>.png`, `.exr` | el GT, la malla rasterizada y la nube, uno junto a otro, sobre la caja del material |
| `<material>_gs.exr` | el fotograma entero de la nube |
| `gt.exr`, `mesh.exr` | la escena trazada, y rasterizada como mallas |
| `clouds/<material>.usdc`, `.usda` | la nube, y la escena en la que se dibujó |

```sh
athenea mesh2splat car.usda --transfer --cell-from-camera /World/Camera \
    --validate car_validate --validate-size 1920 1080 --validate-paths 512
```

### 3.2 Renderizar una escena

`athenea stage` dibuja por el mismo delegate de Hydra que carga un DCC, así que lo
que dibuja es lo que ve un host.

**Dos rutas.** `--technique raster` rasteriza la visibilidad de las mallas y
los splats de la nube y los compone por profundidad; es la ruta interactiva y
la del viewer. `--technique rt` traza las superficies y compone la nube
encima. Un frame de solo splats lo traza entero el ray tracer de gaussianas.

**Convergencia.** `--path-samples` es lo que reúne cada pasada y `--path-total`
donde la imagen está terminada; `athenea stage` dibuja hasta alcanzarlo.
`--denoise` pasa Open Image Denoise cuando lo alcanza.

**Un dome ilumina una nube con su imagen.** Una nube que pidió ser relit
(`primvars:athenea:splat:relight`) toma la dirección del cielo de un dome que
lleve imagen: el cuerpo de la irradiancia del cielo y el reflejo del cielo
convolucionado a su roughness. No se autoriza nada para ello; el frame lo
prepara cuando el dome cambia. Se preparan cuatro domes, y un quinto se dibuja
como su color.

**Luces.** `--light-samples` son muestras por luz y píxel, y una es un frame
interactivo. `--default-lights` pone un dome y un sol en la capa de sesión
para una escena que no declara ninguna, que es lo que hace visible un asset
sin editarlo.

**Una `DistantLight` está en las unidades de UsdLux.** `intensity` (por
`2^exposure` y `color`) es la radiancia del disco del sol, en nits; con
`normalize = 1` se divide por el ángulo sólido proyectado del disco,
`pi sin^2(angle / 2)`, y pasa a ser la iluminancia sobre una superficie
que mira a la luz, en lux. Un `angle` de 0 es una luz paralela cuya
irradiancia es la intensidad en ambos casos. Así, un sol escrito sin
`normalize` deja `intensity * pi sin^2(angle / 2)`: el sol por defecto de
UsdLux (50000 a 0.53 grados) deja unos 3.4, y uno de intensidad 3 con el
ángulo por defecto queda casi negro -- escribe `normalize = 1` para un sol
cuya intensidad sea aquello con lo que ilumina. El sol por defecto está
normalizado. Una `DomeLight` ignora `normalize`, como dice el esquema.

**Sombras.** Las mallas sombrean por rayo en la ruta trazada. Una nube
proyecta a través de un mapa de transmitancia en cada luz, sin rayo ninguno:
`--cloud-shadow-texels`, `--cloud-shadow-density` y `--cloud-shadow-terms` lo
gobiernan, y `--no-cloud-shadows` lo apaga. Una `DomeLight` también
proyecta: en los huecos del mapa que dejan las luces (ocho en total), seis
mapas a lo largo de su cenit y de un anillo a cuarenta grados de altura, y una
muestra del domo en una malla lee el más cercano a su dirección.
`--splat-shadows` es la otra
dirección en la ruta trazada: una nube relit sombreándose a sí misma, un rayo
por splat.

**Una cámara, o una hecha aquí.** `--camera` toma una de la escena, con su
obturador, su diafragma y su distorsión. `--eye`/`--target`/`--up` hacen una,
y entonces la describen `--focal`, `--fstop`, `--focus`, `--near`, `--far` y
`--shutter`.

**Salidas.** `-o` escribe un EXR con el color y un canal `Z`.
`--render-settings` renderiza en su lugar los productos de un prim
`UsdRenderSettings`: cada producto es un EXR cuyos canales son sus render
vars. El `sourceName` de una var decide qué lee — `Ci` es el color, `z` la
profundidad, un `primvar` lee un primvar, y una expresión de camino de luz de
la forma `C.*<L.'NOMBRE'>` lee ese grupo de luz. `CryptoObject00`, `01` y `02`
son las capas Cryptomatte, y un producto que las lleve carga además los cuatro
atributos estándar que las nombran, con el manifest que devuelve un id a su
ruta de prim.

```sh
athenea stage shot.usda --technique rt --path-samples 8 --path-total 512 \
    --denoise --light-samples 4 -o shot.exr
```

### 3.3 Una nube grande: niveles de detalle y streaming

Un `.athc` guarda una nube ordenada por código de Morton, cortada en chunks,
con niveles fundidos encima: un grupo de splats que a esta distancia no se
distinguen se dibuja como la única gaussiana que los representa. `athenea convert`
escribe el contenedor; `--chunk-splats` dice cuánto viaja de una vez y
`--max-group-fraction` cuán agresivo es el nivel fundido más fino.

Al renderizar, `--lod` es el corte: cuántos píxeles puede abarcar una celda
fundida antes de que el renderer tome el nivel más fino. Más pequeño es más
fino y más caro. `--stream-budget` limita cuántos splats se quedan en el
dispositivo, y el resto se leen según los pide la vista; `0` lee el fichero
entero.

En una escena, esos dos son `primvars:athenea:lod:threshold` y
`primvars:athenea:stream:budget` en el prim que nombra el asset con
`primvars:athenea:asset`.

Un `.athc` funde celdas, y una nube que lleva un esqueleto no puede: una celda
que juntara ala y cuerpo no sabría con cuál moverse. Esa nube tiene niveles de
detalle convirtiéndola otra vez, más gruesa (`athenea mesh2splat --lod-levels`):
cada nivel es un ParticleField propio, con su propio rig, y los prims de un
mismo `primvars:athenea:lod:group` son una nube. Una vista dibuja el nivel más
grueso cuyo `primvars:athenea:lod:cell` (en sus propias unidades) no ocupa más de
`primvars:athenea:lod:threshold` píxeles donde la nube está más cerca -- el más fino
si ninguno lo cumple --, y sólo ese nivel se posa.

Una nube que guarda normales de sombreado (las de una conversión,
`primvars:athenea:splat:normal`) las guarda en su `.athc`: cuatro bytes más por
gaussiana, y en los niveles fundidos la media ponderada de lo que representan
hecha unitaria de nuevo. Es la versión 2 del formato; un fichero de la versión
1, que no tiene, se sigue leyendo. La misma versión guarda si los colores son
luz lineal (`primvars:athenea:splat:linear`) en los flags de su cabecera (bit
1, junto al bit 0 de las normales); un fichero escrito antes lo tiene a cero y
se lee como una captura, sRGB. Una nube que emite luz
(`primvars:athenea:splat:emission`) también la guarda, cuatro bytes más por
gaussiana (una palabra RGB9E5, tras las normales donde están las dos), y en los
niveles fundidos la media ponderada de lo que representan; es el bit 2 de los
`flags` de la cabecera (el bit 0 son las normales), así que un fichero sin ella
se lee como antes. Una nube sin normales, sin emisión y en sRGB se sigue escribiendo como versión
1, de modo que un lector que solo conoce la versión 1 la abre: la versión 2 se
escribe solo donde los `flags` no son cero. Una nube con material (`pbr`, y
sus capas donde las tiene) lo guarda bajo el bit 4, una palabra por elemento
y tres más para las capas; una con transfer lo guarda bajo el bit 5, sus
valores como pares f16 y sus direcciones abiertas detrás -- los números en una
cabecera de treinta y dos bytes tras la primera, y en los niveles fundidos el
transfer de un grupo promediado por opacidad (uno zonal, cuyos ejes no se
promedian, el de una gaussiana), sus bits puestos donde los tiene la mitad del
peso, su material el de una gaussiana. El bit 3 se guarda para la propuesta
009. Un bit de los `flags` que esta compilación no conoce (el bit 3, o pasado
el bit 5) se rechaza, nombrando el fichero: un bit posterior puede añadir un bloque, y un lector que lo saltara
leería cada bloque detrás de él desde el sitio equivocado.

Cómo se ve un presupuesto corto: los grupos cuyos chunks no han llegado
dibujan su gaussiana fundida, así que la nube está pero roma, y se afina según
aterrizan.

El presupuesto se ciñe al del dispositivo (`ATHENEA_GPU_BUDGET`, §1.2): un
stream se abre con como mucho la mitad de lo que le queda al presupuesto del
dispositivo, a 132 bytes por splat, y un asset al que se pidió leer entero
cuyo fichero ocuparía más que esa mitad se lee en streaming. Ambas cosas se
imprimen cuando pasan (`streaming budget N splats (~M MiB; asked ...)`). `athenea stage` espera a que los streams se asienten antes de una
imagen fija, así que un frame renderizado nunca está a medio llegar.

### 3.4 Color

Dentro del motor todo es lineal. Un EXR escrito por `athenea render`, `athenea stage`
o `athenea live` es RGBA lineal premultiplicado, en media por defecto, con `Z` en
unidades de vista; `athenea:exrHalf` en un prim de render settings elige media o
float para sus productos, y una capa Cryptomatte se queda en float diga lo que
diga, porque un id redondeado a media es el nombre de otro id.

El display transform se aplica solo donde se enseña una imagen: el viewer, y
la vista previa de MCP. Ofrece AgX, ACES 2.0, y OpenColorIO donde la
compilación lo tenga — `--ocio-config`, `--ocio-display` y `--ocio-view`
compilan el display y el view de ese config dentro del kernel. `--edr` pide
una superficie float y lleva ACES 2.0 hasta el pico de la pantalla, que en una
pantalla normal es la misma imagen que sin él.

Una textura se lee en el espacio de color que nombra su material o su luz
-- el `colorspace` de MaterialX, el `sourceColorSpace` de un UsdUVTexture, el
`colorSpace` de USD en la entrada de archivo, el `colorSpace` de un domo en
`inputs:texture:file` -- y se lleva al espacio de trabajo (Rec.709 lineal) en
el dispositivo. Los nombres son los del studio config de OpenColorIO
(`srgb_texture`, `lin_rec709`, `acescg`, `g22_rec709`, ...), sus alias y
roles, los de USD (`lin_ap1_scene`, `srgb_rec709_scene`, ...) y los de
UsdUVTexture (`sRGB`, `raw`, `auto`). `raw`, `data`, `Non-Color`, `none`,
`identity` y `Utility - Raw` leen el archivo como datos. Sin nombre, o con
`auto`: una imagen de 8 bits es sRGB salvo que el archivo diga otra cosa, y
cualquier otra, lineal. Un nombre que nadie conoce se avisa una vez y el
archivo se lee como él dice. Sin OpenColorIO en la compilación solo se
conocen sRGB, Rec.709 lineal y datos.

`athenea view --snapshot` escribe el frame **tal como se ve**, codificado para
pantalla y con los paneles dentro. Es una captura de pantalla, no una salida
de render: para eso está `athenea stage`.

### 3.5 Una secuencia sobre un reloj

`athenea live` dibuja cada frame para el instante que le toca, en vez de tan
rápido como puede. `--rate` es la cadencia, drop-frame incluidas. Sin `--ptp`
el reloj es el de esta máquina; con él, el motor sigue a un maestro PTP
(`genlock-cli master` al otro lado) en `--port` y `--domain`, y espera hasta
`--lock-timeout` al enganche. `--at` nombra el instante UTC de reloj de pared
del primer frame, `--start` el tiempo USD que suena primero.

Cada EXR lleva su timecode, su cadencia como racional, el instante TAI, el
índice de frame, el tiempo USD y cuánto se retrasó el despertar, así que un
frame se coloca en una línea de tiempo por lo que lleva dentro.

### 3.6 El visor web

El visor propio de athenea sobre el WebGPU de una página web, por módulos
(docs/decisions.md, "The web viewer"): su raster de splats, sus niveles de
detalle, los armónicos y, después, las features reiluminadas, como un
directorio que un sitio sirve tal cual. Se construye en la CPU, sin GPU:

```sh
scripts/web-kernels.py --out build/web/athenea          # necesita slangc; Naga y Dawn para sus comprobaciones
node build/web/athenea/check.mjs build/web/athenea [x.spz]   # el motor contra sus kernels, en la CPU
```

La construcción falla si un kernel no compila, si Naga o Tint lo rechazan, o si
pide más que los valores por defecto de la especificación: 8 storage buffers
por etapa, 16384 bytes de memoria de workgroup.

| Ruta | Qué es |
|---|---|
| `index.html`, `viewer.js`, `viewer.css` | la página del visor |
| `build.json` | el commit desde el que se construyó el directorio (`commit`, `short`, `branch`, `dirty`), que la página muestra en su cabecera |
| `athenea-webgpu.js` | el mismo motor como renderer del sitio athenea-web: `createRenderer`, `RENDERER_INFO` |
| `lib/engine.js` | W-host: `Engine`, la API de la página de abajo |
| `lib/gpu.js`, `lib/probe.js`, `lib/orbit.js`, `lib/math.js` | el ejecutor de kernels, la sonda de tier, la órbita, las matrices de la cámara |
| `lib/loaders/` | `ply.js`, `spz.js`, y `cloud.js`, que comparten |
| `lib/modules/` | `core-raster.js`, `sh.js`, `lod.js`: un manifiesto y los ganchos de cada uno |
| `manifest.json` | por kernel: su fichero WGSL, el tamaño de workgroup, cada parámetro por nombre con su `@binding` y su tipo, los desplazamientos de los campos de un uniform, sus constantes override por `@id`, y lo que usa de los límites. `format` `athenea-webgpu-kernels`, `version` 1 |
| `kernels/<entry>.wgsl` | un módulo WGSL por kernel, compilado desde `shaders/`; nunca se edita |
| `check.mjs` | la comprobación en CPU de arriba |

Todas las rutas son relativas y todos los imports se resuelven desde
`import.meta.url`, sin ningún import por nombre suelto: el directorio se sirve
tal cual, en cualquier sitio, y se carga con `import(url)`. Sin hilos, así que
sin cabeceras COOP/COEP.

**El visor** (`index.html`): el directorio se sirve tal cual, en cualquier ruta
(el `/viewer/` del sitio). Un fichero se suelta en la página, se abre con el
botón, o se da en la dirección:

| Parámetro | Recibe | Por defecto |
|---|---|---|
| `s` | una escena del catálogo del sitio: `<assets>/<s>/scene.json` da el fichero, su transformación (la del fichero, o la de la escena), la primera cámara y el fondo | ninguno |
| `f` | con `s`: el formato que abrir, `spz` o `ply` | el primero de `spz`, `ply` que tenga la escena |
| `assets` | dónde están las escenas | `https://athenea-assets.lucab.co.uk/scenes/` |
| `url` | un `.ply` o `.spz`, relativo a la página o absoluto (con CORS) | ninguno: la página espera un fichero |
| `tier` | `T1`, `T2`, `T3` | el de la sonda (abajo) |
| `transform` | JSON, como lo da un scene.json, sobre el de la escena: `{"position": [x,y,z], "rotation": [x,y,z] (grados, XYZ), "scale": s}` | ninguno |
| `eye`, `target`, `fov` | `x,y,z`, `x,y,z`, grados verticales | la nube encuadrada, 50 |
| `modules` | ids, separados por comas | `core-raster,sh,lod` |
| `features` | JSON, las opciones de abajo: `{"linear": true, "lodThreshold": 2}` | el valor por defecto de cada opción |
| `shot` | un nombre: se descarga `<nombre>.webp` 4 s después de dibujarse la nube, y las estadísticas se escriben en la consola | ninguno |

La órbita: arrastre izquierdo gira alrededor del objetivo, arrastre derecho o
con mayúsculas desplaza, la rueda acerca. El panel lista las opciones de cada
módulo; la que el tier no puede tener sale en gris con el porqué. La línea de
estadísticas: tier, píxeles, fps, ms por cuadro; splats, dibujados, visibles,
pares; el umbral del LOD, los niveles y los slots.

**Tiers**, de la sonda del adaptador, o `tier=`:

| Tier | Cuándo | Dibujados como mucho | Suelo de celda | Resolución | Objetivo de cuadro |
|---|---|---|---|---|---|
| T1 | WebGPU con los límites por defecto, o cualquier móvil, tableta o visor | 1,25 M | 1,5 px | 1,5 × CSS | 33,3 ms |
| T2 | 32 KiB de memoria de workgroup y 644 MB por vinculación | 2,5 M | 1 px | 2 × | 16,7 ms |
| T3 | T2, 1,25 GB por vinculación y texturas BC | 4,5 M | 1 px | la de la pantalla | 16,7 ms |

Sin WebGPU (T0), la página lo dice.

**Los módulos y sus opciones** (`engine.setFeatures({id: valor})`):

| Módulo | `id` de la opción | Recibe | Por defecto |
|---|---|---|---|
| core-raster | `antialias` | la compensación de energía del filtro de Mip-Splatting | true |
| | `linear` | mezclar en luz lineal y codificar al final, como `athenea render`; false mezcla en el sRGB de pantalla, como los visores 3DGS estándar | false |
| | `exposure` | un multiplicador, 0,25-4 | 1 |
| sh | `shDegree` | 0-3; un pipeline por grado, sin mover datos | 3 |
| lod | `lod` | niveles de detalle sí o no | true |
| | `lodAuto` | el controlador sostiene el tiempo de cuadro y el presupuesto de dibujados del tier | true |
| | `lodThreshold` | píxeles que puede abarcar una celda fusionada, 0,5-8; fijarlo apaga `lodAuto` | el inicial del tier (2 px en T1, 1 px en T2-T3) |

**Niveles de detalle.** Se construyen en la GPU cuando la nube ha cargado (con
4096 splats o más): el método del `LodBuilder` nativo -- orden Morton, niveles
de un octree, una gaussiana fusionada por celda, hasta celdas de dos splats.
En cada cuadro el corte dibuja fusionada una celda que abarca como mucho el
umbral, en una lista que la proyección recorre sobre un número fijo de slots
-- el presupuesto de dibujados del tier --, así que lo que el corte deja fuera
no cuesta nada después del corte. El controlador sube el umbral ×1,25 cuando
el corte quiere más que los slots, o cuando la décima parte de los cuadros del
último medio segundo pasa de 1,5 × el objetivo; lo baja tras tres segundos por
debajo de 0,7 ×, y no en los diez segundos siguientes a subirlo; del suelo del
tier a 8 px. Un pool con los splats y todos los niveles ocupa unas 1,6 veces la
memoria de la nube.

**El motor** (`lib/engine.js`), lo que una página maneja directamente:

| Llamada | Recibe, devuelve |
|---|---|
| `Engine.create({canvas, base, tier, modules, device})` | `base`: el directorio de `manifest.json` (por defecto, el que está encima de `lib/`). `device`: uno que ya tenga la página, o se pide uno con `maxStorageBufferBindingSize` y `maxBufferSize` del adaptador y el resto de límites por defecto. Configura el canvas `rgba8unorm` con `STORAGE_BINDING` |
| `load({url o blob, format, onProgress, transform})` | `format` `ply` o `spz`, por el nombre si no se da. Se resuelve en `{count, restPerColour, bounds}` cuando la nube está en el dispositivo y los módulos han construido lo que guardan |
| `setCamera({position, target, up, fov})`, `getCamera()` | espacio del mundo, Y arriba; `fov` vertical, en grados; `up` por defecto `[0, 1, 0]` |
| `setSky({color, hdriUrl, exposure})` | `color` sRGB `#rrggbb` o `[r, g, b]`, por defecto `#0b0b0f`; aún no se muestra cielo: devuelve `{hdri: false}` |
| `setLightState({[grupo]: {on, intensity}})` | se guarda para el módulo de grupos de luces; devuelve `{applied: false}` |
| `setFeatures({id: valor})` | las opciones de arriba; devuelve, por id, si un módulo la aceptó |
| `options()` | las opciones de todos los módulos: `{id, label, kind, range, step, effect, default, module, value, available, why}` |
| `frame()` | dibuja en el canvas a su tamaño en píxeles; se resuelve en `{total, drawn, visible, pairs, pairsDropped, ms, counters, modules}` |
| `bounds()`, `dispose()` | la caja en el mundo; libera todos los buffers, y el dispositivo si lo creó |

**El renderer del sitio** (`athenea-webgpu.js`) sigue `renderers/index.js` de
athenea-web: `createRenderer({canvas, scene, file, url, onProgress})`
(`file.format`; `file.transform` o `scene.transform`; el tier de la sonda);
`bounds()`; `setCamera`/`getCamera`; `setBackground({color, hdriUrl, showHdri,
exposure, blur})` a `setSky` (`blur` se ignora); `setDetail(scale)`: el umbral
del LOD es el inicial del tier entre `scale`, y `scale` 1 devuelve el mando al
controlador; `hasLod` true; `hasRoi`, `roi` false; `setLightState`,
`setFeatures`; `stats()` `{total, rendered (dibujados), fps, extra, debug}`,
donde `extra` es una línea (`<tier> · <visibles> visible · <pares> pairs · <ms>
ms`, o `error: <mensaje>`) y `debug` los números del cuadro; `snapshot()` un
webp tras el siguiente cuadro; `dispose()`. `RENDERER_INFO` es `{label:
"athenea · WebGPU", formats: ["ply", "spz"], needsWebGPU: true}`.

**A qué apunta `url`.**
- Un PLY de 3DGS: `binary_little_endian`, un elemento `vertex` sin nada antes,
  todas las propiedades `float`, con `x y z opacity scale_0-2 rot_0-3
  f_dc_0-2` y `f_rest_*` para 0, 9, 24 o 45 armónicos. Se recibe en streaming y
  se decodifica por trozos de como mucho 64 MiB según llega; `content-length`
  hace del progreso una fracción.
- Un SPZ, v2 o v3 (gzip): se lee entero, lo descomprime el navegador y sus
  streams se decodifican en la GPU. v1 (posiciones float16) y v4 (zstd) se
  rechazan. Sus coordenadas son right-up-back y se giran como las gira el
  lector nativo.

**Límites.** Cada stream es una vinculación. El mayor de un splat es su
registro proyectado, 48 bytes, así que una nube tiene como mucho
`maxStorageBufferBindingSize / 48` splats (2,8 M con los 128 MiB por defecto);
los armónicos se quitan de grado en grado hasta que 1,6 veces la nube cabe en
una vinculación. Los pares son como mucho `min(vinculación / 4, 65535 × 256)`;
a partir de ahí el cuadro descarta el resto y dice cuántos en `pairsDropped`.

**Color.** Por defecto la mezcla es en el sRGB de pantalla, como mezclan los
visores 3DGS estándar (Spark, three.js), y el resultado se escribe tal cual;
`linear` mezcla en luz lineal y codifica al final, como el raster nativo. El
color de fondo es sRGB en ambos casos.

**Errores**, que lanzan `Engine.create`, `load` y `createRenderer`: `this
browser has no WebGPU`; `WebGPU: no adapter`; `<url>: HTTP <status>`; `format
'<f>': this page reads ply, spz`; `not a PLY file`; `PLY format '<f>'; splat
clouds are binary_little_endian`; `PLY: a property that is not float32 (this
module reads float32 records only)`; `PLY: no '<name>' -- this does not look
like a trained splat cloud`; `PLY: <n> harmonic coefficients is not a whole
degree`; `not an SPZ file (no NGSP header)`; `SPZ v1: float16 positions (a
format never released)`; `SPZ v<n>: this page reads v2 and v3`; `<name>: not a
gzipped SPZ (v4's zstd is not read here)`; `<n> splats: past this device's <b>
bytes a binding`; `<label>: <n> bytes is past the device's binding size <b>`;
`<name>: <k> of <n> records arrived`. Un cuadro que falla para el bucle del
visor y muestra su mensaje.

## 4. Autorizar para este motor en USD

### 4.1 Apuntar una aplicación al plugin

```sh
export PXR_PLUGINPATH_NAME=<build>/plugin/usd
```

Ese directorio tiene el delegate de Hydra `hdAthenea` y los schemas codeless
juntos, así que una variable encuentra los dos. El host ofrece entonces el
renderer con su nombre, y los ajustes de abajo aparecen en su panel de render
settings.

### 4.2 Render settings

Se autorizan en el namespace `athenea:` de un prim `UsdRenderSettings`, o los pone
un host a través del delegate. El primer grupo lo declara el delegate y sale
en un panel; el segundo se lee donde se encuentre y no sale.

| Ajuste | Tipo | Por defecto | Qué significa |
|---|---|---|---|
| `athenea:technique` | token | `raster` | `raster` o `rt` |
| `athenea:visibility` | token | `automatic` | cómo se ven las mallas: `automatic`, `raster`, `rays`, `bvh` |
| `athenea:settleStreams` | bool | `false` | esperar a los assets en streaming antes de dibujar; una imagen fija lo pone |
| `athenea:lightSamples` | int | `1` | muestras por luz |
| `athenea:chooseLights` | bool | `false` | una luz por muestra, elegida por potencia |
| `athenea:pathSamples` | int | `1` | rt: caminos por píxel y pasada |
| `athenea:pathBounces` | int | `1` | rt: rebotes tras el primer impacto; hasta 8 cruces de un vidrio por camino no cuentan |
| `athenea:pathTotal` | int | `1` | rt: caminos por píxel hasta converger |
| `athenea:pathAdaptive` | bool | `false` | parar un píxel cuando su error baja lo suficiente |
| `athenea:pathError` | float | `0.02` | el error estándar relativo en que para |
| `athenea:pathMis` | bool | `true` | muestreo por importancia múltiple |
| `athenea:denoise` | bool | `false` | denoise al terminar de reunir |
| `athenea:motionBuckets` | int | `4` | rt: rodajas del obturador, de 1 a 8 |
| `athenea:antialias` | bool | `true` | un desplazamiento subpíxel por pasada |
| `athenea:splatTransferIndirect` | bool | `true` | una nube con transfer suma su mitad rebotada |
| `athenea:splatReflections` | bool | `false` | rt: una gaussiana refleja la nube a la que pertenece, un rayo cada una |
| `athenea:splatShadows` | bool | `false` | rt: una nube relit se sombrea a sí misma |
| `athenea:cloudShadows` | bool | `true` | el mapa de transmitancia de una nube en cada luz |
| `athenea:cloudShadowResolution` | int | `1024` | texels por lado, por luz |
| `athenea:cloudShadowTerms` | int | `0` | 1 es solo el total; 3, 5, 7 añaden pares de Fourier; 0 deja decidir a quien recibe |
| `athenea:cloudShadowDensity` | float | `1.0` | multiplicador de la profundidad óptica de la nube |

Leídos pero no declarados:

| Ajuste | Tipo | Dónde | Qué significa |
|---|---|---|---|
| `athenea:shutter` | double2 | prim de settings | apertura y cierre, en frames |
| `athenea:lens` | double2 | cámara | radio de apertura y distancia de enfoque |
| `athenea:exrHalf` | bool | prim de settings | escribir los productos en media en vez de float |
| `athenea:disableMotionBlur` | bool | settings o producto | para ese producto |
| `athenea:disableDepthOfField` | bool | settings o producto | para ese producto |
| `athenea:lightGroup` | string | un prim de luz | el grupo bajo el que se reúne su aportación |

### 4.3 Los schemas de API

Ocho schemas codeless de API, que se aplican a un prim como cualquier otro.
Todos los atributos son primvars, así que heredan hacia abajo.

**`AtheneaSplatLightingAPI`** — si la nube la ilumina la escena en vez de enseñar
la radiancia que lleva.

| Atributo | Tipo | Por defecto |
|---|---|---|
| `primvars:athenea:splat:relight` | bool | `false` |
| `primvars:athenea:splat:litBody` | bool | `false` |
| `primvars:athenea:splat:linear` | bool | `false` |
| `primvars:athenea:splat:metallic` | float[] | — |
| `primvars:athenea:splat:roughness` | float[] | — |
| `primvars:athenea:splat:transmission` | float[] | — |
| `primvars:athenea:splat:ior` | float | `0` |
| `primvars:athenea:splat:transferDirect` | float[] ‹9 o 16 por gaussiana› | — |
| `primvars:athenea:splat:transferIndirect` | float[] ‹27 o 48 por gaussiana› | — |
| `primvars:athenea:splat:transferReflected` | float[] ‹48 por gaussiana› | — |
| `primvars:athenea:splat:transferZonal` | float[] ‹10 por gaussiana› | — |
| `primvars:athenea:splat:shadowBits` | int[] ‹2, 8 o 32 por gaussiana› | — |
| `primvars:athenea:splat:thinWalled` | int[] ‹1 por gaussiana› | — |
| `primvars:athenea:splat:curvature` | float[] ‹3 por gaussiana› | — |
| `primvars:athenea:splat:schlickMetal` | int[] ‹1 por gaussiana› | — |
| `primvars:athenea:splat:normal` | normal3f[] ‹1 por gaussiana› | — |
| `primvars:athenea:splat:emission` | color3f[] ‹1 por gaussiana› | — |
| `primvars:athenea:splat:specularWeight` | float[] ‹1 por gaussiana, 0 a 1› | `1` |
| `primvars:athenea:splat:specularColor` | color3f[] ‹1 por gaussiana, 0 a 1› | `(1, 1, 1)` |
| `primvars:athenea:splat:specularIor` | float[] ‹1 por gaussiana, 1 a 2.99› | `1.5` |
| `primvars:athenea:splat:coatWeight` | float[] ‹1 por gaussiana, 0 a 1› | `0` |
| `primvars:athenea:splat:coatRoughness` | float[] ‹1 por gaussiana, 0 a 1› | `0` |
| `primvars:athenea:splat:coatIor` | float[] ‹1 por gaussiana, 1 a 2.98› | `1.5` |
| `primvars:athenea:splat:sheenColor` | color3f[] ‹1 por gaussiana, 0 a 1› | `(0, 0, 0)` |
| `primvars:athenea:splat:sheenRoughness` | float[] ‹1 por gaussiana, 0 a 1› | `0` |
| `primvars:athenea:splat:coatDarkening` | float[] ‹1 por gaussiana, 0 a 1, activo desde 0.5› | `0` |

`relight` dice que los colores son un albedo que las luces de la escena tienen
que iluminar. `litBody` dice que ya son la luz sobre el cuerpo del material,
así que lo que añade un frame es el reflejo — que es lo que escribe una
conversión horneada.
`ior` es el índice con el que sus gaussianas transmisivas doblan el cielo. A 0
la mitad transmitida es la media de lo que hay detrás, que es translucidez; por
encima de uno es el cielo en la dirección que da Snell sobre la normal de cada
gaussiana, que es lo que hace que una bola de cristal enseñe la sala girada.
La ruta trazada dobla dos veces: recorre el árbol de la propia nube hasta la
cara de salida del objeto y vuelve a doblar al salir, que es lo que hace de una
bola una lente y no una ventana teñida, y donde el rayo se encuentra luego con
gaussianas de la nube enseña ésas. El rasterizador dobla una sola vez, en la
cara por la que entra, y enseña el cielo. Ninguno de los dos enseña una
**malla** a través del cristal.
Los dos arrays de transfer son lo que escribe `--transfer` en su lugar: cuánto
de cualquier cielo llega a la gaussiana, directo y tras un rebote, que el
frame combina con el cielo que hay. Una nube que los lleva no necesita
`litBody`, y no hay ningún atributo que lo diga: llevarlos es lo que lo
dice. Dieciséis y cuarenta y ocho por gaussiana son grado 3
(`--transfer-degree 3`, el valor por omisión), multiplicados por dieciséis
armónicos del cielo; nueve y veintisiete son los del primer transfer, que se
leen como siempre.
`transferZonal` es la mitad directa como dos lóbulos zonales en el marco propio
de cada gaussiana, escrita en lugar de `transferDirect` (`--transfer-lobes`,
`--skinned`): por cada lóbulo, el (u, v) del cuadrado octaédrico de su eje en el
marco que da la orientación de la gaussiana, y luego sus coeficientes zonales de
las bandas 0, 1 y 2 (un ajuste de un lóbulo escribe el segundo como ceros).
Donde están los dos, es el que se lee. Un lector que no lo conoce dibuja la nube
reiluminada sin transfer, y eso es todo su versionado: es un primvar nuevo, no
un significado nuevo de uno viejo, y un `.athc` lleva cualquiera de las dos bajo su bit 5.
`shadowBits` se escribe a su lado: sesenta y cuatro bits por gaussiana, uno por
celda de una rejilla octaédrica de 8 x 8 sobre la esfera en el espacio propio
de la nube -- sobre el marco propio de la gaussiana junto a `transferZonal` --,
puesto donde el rayo del bake en esa dirección salió de la escena. Es lo que
sombrea el sol que un frame saca del cielo; sólo se lee en una nube que lleva
también `transferDirect` o `transferZonal`, y sin él el transfer sombrea el sol
de forma suave.
Ocho o treinta y dos enteros por gaussiana son los del transfer TX
(`--transfer-cells` 16 o 32): una rejilla de 16 x 16 o 32 x 32 sobre la esfera
entera, trazada también la mitad de detrás de la superficie. El número es lo
que dice cuál; nada más lo dice. Con ellos un frame sombrea el sol y cualquier
luz que sea una dirección, y estrecha cada reflejo -- el de la base y el del
coat -- por la parte de su propio lóbulo que los bits dejan abierta, en vez de
por un número para todo el hemisferio.
`transferReflected` es lo que muestran las direcciones cerradas: la luz que
llega a la gaussiana por dirección después de encontrarse con la escena, bajo
un cielo blanco de radiancia uno, como dieciséis armónicos rgb (grado 3). Un
frame la escala al cielo que tiene -- por cuánto más recoge la mitad
indirecta bajo ese cielo que bajo el blanco, contado el rebote del sol -- y un
reflejo la muestra donde los bits dicen que el cielo no llega: la carrocería
en el cromo, el suelo en la pintura. Se escribe con las celdas y la mitad
indirecta, y sólo se lee junto a ellas; `athenea:splatTransferIndirect` la
apaga con la mitad indirecta. Una conversión la rellena sobre las direcciones
cerradas: a lo largo de cada una guarda la media de lo que muestran las
direcciones cerradas en torno a ella, y a lo largo de una abierta la media de
las cerradas, de modo que una dirección cerrada junto a una abierta no se lee
diluida. Una nube convertida antes guarda la proyección tal como la hizo el
horneado, y se lee igual.
Una luz que no es el cielo -- distante, esfera, disco, rectángulo -- ilumina
una nube con las celdas por los mismos lóbulos, sombreada por los bits sobre
el cono que la luz subtiende desde cada gaussiana (su penumbra), y su rebote
llega al cuerpo y a los reflejos como el del sol. Donde un frame midió
además una sombra para ella (`--splat-shadows`, o el mapa de sombra de la
nube), se queda la más oscura de las dos. Los bits dicen qué direcciones salen de la escena, no cuáles llegan a
una lámpara: un objeto más allá de una lámpara que está entre cosas también
la sombrea.
Una gaussiana que transmite, en un transfer TX, guarda también la mitad de
atrás: su campo se hornea sobre la esfera entera, así que el rasterizador
muestra a través de un parabrisas el cielo donde los bits de detrás están
abiertos y lo que guarda el campo -- el habitáculo -- donde no, a lo largo de
la dirección que sigue recta, como la envía una lámina. Donde la nube lleva
las capas, una gaussiana que transmite se dobla con su propio índice
(`specularIor`) siempre que el `ior` de la nube diga que se dobla; el `ior`
único de la nube es el del primer vidrio que encontró la conversión.
`curvature` es cómo gira la superficie bajo la gaussiana: su operador de
forma en los dos primeros ejes de la propia gaussiana (uu, uv, vv), a partir
de las normales de las esquinas de la malla. `athenea mesh2splat --transfer`
lo escribe (no con niveles de detalle ni con `--skinned`), y un fotograma
gira entonces el reflejo a lo largo de la huella de cada gaussiana -- el
brillo que muestra la laca de un coche se mueve por la gaussiana en vez de
quedarse plano sobre ella.
`schlickMetal` es distinto de cero donde el metal de la gaussiana es un
Schlick -- el de OpenPBR y el de glTF, de su color de frente a su color
especular rasante -- y no el conductor de índice artístico que son el de
standard_surface y el de UsdPreviewSurface; `athenea mesh2splat` lo escribe
según el vocabulario del material. Para un metal oscuro los dos se separan al
doble a sesenta grados.
`thinWalled` es distinto de cero donde la gaussiana vino de un vidrio de pared
fina (`geometry_thin_walled` de OpenPBR): la conversión la hizo tan
transparente como la lámina (una tarjeta de ellas detiene `2R/(1+R)`, 0,077
con índice 1,5) y el frame sombrea sólo su reflejo.
`normal` es la normal de sombreado, aparte del marco de la gaussiana: hacia
dónde miraba la superficie una vez que el normal map de la malla la giró, en el
espacio propio del field, como las posiciones. Una gaussiana reiluminada se
ilumina con ella -- puesta del lado de su disco en que está el ojo, porque un
disco se ve por las dos caras -- en lugar de con su eje más corto, que es la
normal de la propia cara y no sabe nada del mapa; el marco sigue respondiendo
a todo lo geométrico (la huella, dónde corta un rayo al disco). `athenea
mesh2splat` la escribe siempre (doce bytes por gaussiana en el fichero, cuatro
en el dispositivo); un esqueleto que lleva la nube la gira igual que gira el
marco; una captura no tiene.
`linear` dice que los colores (los armónicos, de todo grado) son luz lineal,
Rec.709 lineal -- el espacio de trabajo en el que se mezcla toda gaussiana -- y
se dibujan tal cual. Sin él se toman por los de una captura: el sRGB que ajusta
todo entrenador, hecho lineal gaussiana a gaussiana al evaluar los armónicos,
antes de la mezcla. El sRGB no aparece en ningún otro sitio hasta que se
enseña una imagen (la transformada de vista, OpenColorIO). `athenea
mesh2splat` lo escribe en toda nube que hace -- un albedo, el albedo de un
transfer, un bake -- y una nube que se vuelve a escribir (`athenea decimate`,
un export) lo conserva. El aspecto de una captura se mueve un poco: donde se
solapan splats, la media de su luz es más clara que la luz de su media
(decisions.md tiene la medida).

`emission` es la luz que cada gaussiana emite por sí misma, radiancia lineal en
las unidades de la escena, sin tope: lo que era la emisión de su material donde
estaba (arriba). Una gaussiana reiluminada la suma a lo que refleja, con
transfer o sin él, sin sombra y igual por las dos caras de su disco; una cuyos
colores son `litBody` no, porque el bake que los escribió se encontró la
emisión y ya la lleva. No ilumina nada más -- una gaussiana emisiva no es una
luz, y una malla a su lado no recibe luz de ella. `athenea mesh2splat` la
escribe sólo donde algún material de la escena emite luz (doce bytes por
gaussiana en el fichero, cuatro en el dispositivo como RGB9E5: tres mantisas de
9 bits bajo un exponente compartido, hasta 65408, cada canal a 1/512 del más
brillante); una captura no tiene.

Los nueve de `specularWeight` a `coatDarkening` son lo que el material puso
sobre su base, en las unidades de OpenPBR: el peso, el tinte y el índice del
reflejo dieléctrico (el tinte es también el color del borde de un metal), un
coat transparente sobre todo -- un dieléctrico GGX de su propia roughness e
índice -- y un sheen (su color por su peso, el lóbulo de Imageworks). Cada capa
le quita a lo que tiene debajo la parte que refleja hacia el ojo, como hace el
`layer` de MaterialX, así que un coat oscurece en rasante el cuerpo que cubre y
devuelve esa luz como reflejo propio; con `coatDarkening` la base bajo el coat
es aún más oscura, por lo que la cara interior del coat le devuelve (el
`(1 - K) / (1 - E K)` de OpenPBR). Una nube los lleva donde alguno está
escrito y lee uno que falte con el valor por defecto de la tabla; una sin
ninguno refleja con el specular simple, exactamente como antes de que
existieran. Se recortan a sus rangos, y en el dispositivo son tres palabras por
gaussiana, un byte por valor (un índice en pasos de 1/128, el del coat en pasos
de 1/64 junto al bit de su oscurecimiento). `athenea mesh2splat` escribe los
nueve donde algún material de la escena pone algo encima (52 bytes por
gaussiana en el fichero, doce en el dispositivo). Los niveles de detalle y
un `.athc` los llevan con el material (bit 4), tomando un grupo fundido los de
una gaussiana.
 — los joints que llevan una nube.

| Atributo | Tipo | Nota |
|---|---|---|
| `primvars:athenea:splat:jointIndices` | int[] | cuatro por gaussiana |
| `primvars:athenea:splat:jointWeights` | float[] | cuatro por gaussiana |
| `primvars:athenea:splat:geomBindTransform` | matrix4d | |
| `primvars:athenea:splat:skinningXforms` | matrix4d[] | una por joint, lo único que cambia con el tiempo |
| `primvars:athenea:splat:jointWeightGradients` | half[] | seis por gaussiana: los gradientes de peso de los tres primeros joints a lo largo de sus dos ejes de reposo, por unidad del espacio de la nube; el del cuarto es menos su suma. Opcional |
| `primvars:athenea:splat:skeleton` | string | de dónde vino |

**`AtheneaSplatVisibilityAPI`** — lo que proyecta una nube con esqueleto, horneado
por partes.

| Atributo | Tipo | Nota |
|---|---|---|
| `primvars:athenea:splat:visibilityParts` | float[] | doce floats por parte |
| `primvars:athenea:splat:visibilityTexels` | int[] | dos medias por palabra |
| `primvars:athenea:splat:visibilityAmbient` | int[] | la media de una sonda, para los domes |
| `primvars:athenea:splat:visibilityPartOf` | int[] | la parte a la que pertenece cada gaussiana |

**`AtheneaSplatCryptomatteAPI`** — de qué prim vino cada gaussiana.

| Atributo | Tipo | Nota |
|---|---|---|
| `primvars:athenea:splat:cryptoObject` | int[] | un id por gaussiana |
| `primvars:athenea:splat:cryptoManifest` | string | `{"<ruta>":"<ocho dígitos hex>", ...}` |

**`AtheneaSplatEditAPI`** — una edición no destructiva sobre las nubes que cuelgan
de un prim: conservar lo que está dentro de un volumen, quitarlo, o graduarlo.

| Atributo | Tipo | Por defecto |
|---|---|---|
| `primvars:athenea:edit:active` | bool | `false` |
| `primvars:athenea:edit:shape` | token | `box`, o `sphere` |
| `primvars:athenea:edit:mode` | token | `grade`, o `keep`, o `remove` |
| `primvars:athenea:edit:centre` | float3 | `(0, 0, 0)`, en el espacio de la nube |
| `primvars:athenea:edit:size` | float3 | `(1, 1, 1)`: semiejes de la caja, o el radio de la esfera en x |
| `primvars:athenea:edit:tint` | color3f | `(1, 1, 1)` |
| `primvars:athenea:edit:saturation` | float | `1` |
| `primvars:athenea:edit:brightness` | float | `1` |
| `primvars:athenea:edit:opacity` | float | `1` |
| `primvars:athenea:edit:minOpacity` | float | `0` |
| `primvars:athenea:edit:maxScale` | float | `0` |
| `primvars:athenea:edit:invert` | bool | `false` |

Un grade (`tint`, `saturation`, `brightness`) trabaja sobre el color de cada
splat en luz lineal, como la mezcla: un brillo de 2 dobla la luz, y una
captura graduada aquí no coincide con los mismos números aplicados a su sRGB.

**`AtheneaStreamedAssetAPI`** — una nube dibujada desde un `.athc`.

| Atributo | Tipo | Por defecto |
|---|---|---|
| `primvars:athenea:asset` | asset | — |
| `primvars:athenea:lod:threshold` | float | `1` |
| `primvars:athenea:stream:budget` | int64 | `0`, lee entero |

**Niveles de detalle de un ParticleField** — la misma nube convertida con
varias celdas (`athenea mesh2splat --lod-levels`).

| Atributo | Tipo | Por defecto |
|---|---|---|
| `primvars:athenea:lod:group` | string | —, se dibuja tal cual |
| `primvars:athenea:lod:cell` | float ‹sus propias unidades› | — |
| `primvars:athenea:lod:threshold` | float ‹píxeles› | `1` |

**`AtheneaPointStyleAPI`** — cómo se dibuja un `UsdGeomPoints`.

| Atributo | Tipo | Por defecto |
|---|---|---|
| `primvars:athenea:sizeInPixels` | float | `0` |
| `primvars:athenea:edl` | float | `0` |
| `primvars:athenea:surfaceOffset` | float | `0` |

**Los Gaussian splats de Blender como `UsdGeomPoints`** — no es un esquema:
los atributos de un PointCloud de Blender de tipo Gaussian splat, tal como los
escribe la exportación USD de Blender. Un prim `Points` que lleva una
`rotation` cuaternión y una `scale`, y `radiance:base` o `radiance:sh_0`, se
dibuja como una nube de splats, no como puntos; `widths` y los estilos de
punto se ignoran entonces. Los valores son los que guarda un ParticleField (el
importador de Blender los copia sin cambiarlos): los colores son los de una
captura, sRGB.

| Primvar | Tipo | Significado |
|---|---|---|
| `primvars:rotation` | quatf[] o quath[] | la orientación de la gaussiana |
| `primvars:scale` | float3[] o half3[] | sus tres desviaciones típicas, lineales, en las unidades del prim |
| `primvars:radiance:base` | float4[] o half4[] | el coeficiente DC (rgb) y la opacidad (lineal, de 0 a 1). La exportación propia de Blender lo pierde; lo escribe el add-on `athenea_hydra`. Sin él la nube es opaca y su DC es 0 (gris), con un aviso |
| `primvars:radiance:sh_N` | float3[] o half3[] | el coeficiente N + 1 (rgb), N desde 0; 3, 8 o 15 de ellos hacen grado 1, 2 o 3, y el resto incompleto de un grado no se lee |

**`AtheneaVolumeAPI`** — cómo dispersa un Volume de `UsdVol`, cuando no hay
Material que lo diga.

| Atributo | Tipo | Por defecto |
|---|---|---|
| `primvars:athenea:densityScale` | float | `1` |
| `primvars:athenea:albedo` | color3f | `(0.8, 0.8, 0.8)` |
| `primvars:athenea:anisotropy` | float | `0` |

Un Material con un terminal `volume` es la manera estándar de decir lo mismo,
y donde hay uno atado, gana.

### 4.4 Qué ignora el motor, y qué rechaza

**Ignora** aquello para lo que no tiene significado: un primvar que no lee, un
render setting fuera del namespace `athenea:`, un schema que no conoce. No avisa
de nada, porque una escena lleva lo que necesitan otros renderers.

**Rechaza**, con un mensaje, lo que se le pide y no puede hacer: un `.spz` en
una compilación sin zstd, un `.sog` sin libwebp, un `.vdb` sin OpenVDB, la
visibilidad de mallas por rayos en un dispositivo sin ray tracing, un `.athc`
dibujado sin `--lod`, un producto de render sin resolución o sin vars.

## 5. El viewer

`athenea view` abre una ventana sobre una escena y mantiene cada frame en el
dispositivo: el display transform escribe la superficie de la ventana y Dear
ImGui dibuja encima. No vuelve nada salvo el píxel picado y un snapshot.

### 5.1 Ratón y teclado

| Entrada | Qué hace |
|---|---|
| arrastrar con el izquierdo | orbitar |
| mayúsculas + izquierdo, o el central | desplazar |
| arrastrar con el derecho, o la rueda | acercar y alejar |
| clic izquierdo sin arrastrar | picar lo que está bajo el cursor |
| `F` | encuadrar toda la escena, y volver a la cámara libre |
| `Escape` | cerrar la ventana |

### 5.2 Los paneles

Los paneles, contados por su papel y no widget a widget, porque se mueven
según crece el motor.

**View** lleva el frame: qué cámara (la libre, o cualquiera de la escena) y su
focal, diafragma y enfoque; la técnica y, bajo `rt`, los caminos por frame,
los rebotes, el denoiser y cuántos caminos por píxel lleva reunidos; las
sombras de nube y su densidad; las luces por defecto, ofrecidas solo donde la
escena no tiene; la visibilidad de mallas; el combo **Output**; el view
transform y la codificación de pantalla; la exposición; el obturador; la
escala de render, que dibuja a una fracción de la ventana y cuesta en
proporción; la línea de tiempo con play, pausa, paso y *every frame*; los
conjuntos de variantes que lleva la escena, con un filtro cuando uno tiene más
de una docena; el bloque **Lights**; el bloque **Sky**; y abajo, el
dispositivo, los tiempos del
frame, lo que tenía el último frame, y qué se picó.

**El sol, sacado del cielo.** Al preparar un domo se busca su fuente más
brillante: una búsqueda octaédrica de 4096 direcciones, luego el perfil radial
de la propia fuente, y sólo se extrae un disco cuando su pico supera ocho veces
la media del cielo y su brillo vuelve al cielo antes de diez grados. Lo que se
encuentra sale de los nueve armónicos —que no pueden sostener un disco de medio
grado— y se enciende como luz direccional, con un coseno exacto. La energía se
conserva al medio por ciento: los dos lados suman los mismos téxeles.

Cada domo preparado dice qué se encontró, que es lo único de un cielo que no se
ve mirándolo:

```
athenea [info] sky 0: a sun at (-0.539, 0.183, 0.822), 0.0247 sr, irradiance
           0.093 0.080 0.061; it is 4 degrees wide
athenea [info] sky 0: no sun (20 degrees of bright sky, which nine coefficients
           hold well enough)
```

Un nublado y un horizonte ancho y brillante se dejan en paz a propósito: sacarle
el núcleo a una fuente amplia deja un anillo en los armónicos y pone un
terminador duro donde va uno suave. El sol llega sólo al cuerpo de la nube —el
mapa prefiltrado se queda su disco para los reflejos, así que se cuenta una vez.

**Sky** es un combo por luz de domo, con un slider *Turn* al lado. El combo
lista todos los `.hdr` y `.exr` que encuentra en la carpeta donde está la
imagen de ese domo, más lo que añada `--hdri`, de modo que una escena cuyo
cielo salió de una librería ofrece la librería entera sin que haya que
decírselo. Elegir uno reilumina el frame sin tocar nada más de la escena;
*(the stage's own)* devuelve lo que pedía el fichero. Los dos se escriben en el
session layer, así que la escena en disco no se edita nunca.

Cambiar cualquiera de los dos cambia el record del domo, que es la clave del
cielo preparado: el frame siguiente reconstruye sus nueve armónicos y su cadena
prefiltrada, unos 180 ms con una imagen de 4k, y los frames posteriores no
cuestan nada más. Arrastrar *Turn* paga esa reconstrucción por frame.

**Lights** es un checkbox por cada luz que la escena define, con su schema al
lado, incluidas las que el fichero dejó inactivas. Desmarcar una desactiva el
prim en el session layer, así que Hydra quita la luz como si nunca se hubiera
escrito; marcarla borra esa opinión, y sólo afirma `active = true` para una luz
que el propio fichero dejó apagada. La escena en disco no se edita nunca. Es
como se juzga un cielo solo: la `DistantLight` propia de una escena sigue
encendida sea cual sea la imagen que el combo **Sky** ponga en el domo, y deja
un brillo que ningún cielo explica. Un domo apagado se lleva su cielo, fondo
incluido, y sale del bloque **Sky** hasta que se vuelve a encender; encenderlo
cuesta la reconstrucción que cuesta un cielo nuevo.

**Stage** es el árbol de prims, y lo picado queda seleccionado en él.

**Picked** es en qué resultó ser un píxel, y se abre con la ventana en vez de
esperar a que lo encuentres: el prim y la instancia que nombra Hydra, la matte
que nombra una nube, y de qué están hechas las gaussianas de ese prim.

**Gaussians** es qué son los splats en pantalla y qué hizo el frame con ellos,
plegable sección a sección, con los números separados por miles. Sobre una
escena sin nubes dice *no gaussians in this stage* y nada más. Se describe una
vez en `modules/ui` (`ui::gaussianPanel`) y se dibuja después del frame, así que
su primera fila es el frame en pantalla.

Lleva dos clases de número, y el panel dice a qué frame pertenece cada una. Lo
que recibió el frame -- las nubes de la escena, lo que conservó el nivel de
detalle, lo que lleva y ocupa cada una -- es exacto para el frame en pantalla.
Lo que contó el dispositivo se lee sin esperar, así que pertenece al frame que
nombra la fila **Counted**, que puede ir uno o dos frames por detrás (bajo la
ruta raster hoy es el mismo frame, porque el rasterizador ya se espera a sí
mismo al final).

| Fila | Qué es | Unidad, y cuándo se muestra |
|---|---|---|
| Frame | la cuenta de frames dibujados del motor, y la ruta: *rasterised*, *splats traced* o *meshes traced, splats rasterised* | siempre, en una escena con nubes |
| Counted | el frame al que pertenecen las cuentas del dispositivo de abajo, y cuánto va por detrás | rutas raster |
| In the stage | las gaussianas de todas las nubes, dibujadas o no | gaussianas |
| Submitted | entregadas al renderer: tras el nivel de detalle, los prims ocultos y los niveles de variante no elegidos; la parte de *In the stage* | gaussianas |
| Visible | conservadas por la proyección, el tamaño del orden por profundidad; la parte de lo entregado en el frame contado | gaussianas, rutas raster |
| Culled, y una fila por razón | *Removed by an edit*, *Outside near/far*, *No area*, *Too faint* (bajo 1/255 una vez repartida por su huella: en lo que se convierte una gaussiana demasiado pequeña), *Off screen* (fuera de los lados del frustum), *Touch no tile*; una razón es fila solo donde descartó algo | gaussianas, rutas raster |
| Tile pairs | pares (tile, gaussiana), el tamaño del orden por tile, y la media por gaussiana visible | pares |
| Most tiles | el mayor número de tiles que tocó una gaussiana | tiles |
| Sort sizes | las claves del orden por profundidad y del orden por tile | claves |
| Traced | lo que dibujó el trazador de rayos: no descarta nada que contar | gaussianas, `rt` solo |
| Time each stage | un interruptor: cada etapa del rasterizador espera entonces al dispositivo, y el frame va más lento por esas esperas | apagado por defecto |
| Project ... Total | las etapas del rasterizador | ms, mientras *Time each stage* está encendido |
| Structures, Build, Trace, Total | el trazador de rayos: si sus estructuras se reconstruyeron o se conservaron, su ruta, y sus tiempos | ms, `rt` solo |
| Clouds (memoria) | los arrays de todas las nubes en el dispositivo, incluidas una copia posada y su esqueleto | bytes |
| Levels of detail | los assets de los que se toman los cortes, y los almacenes de streaming | bytes, donde los hay |
| una fila por nube | el prim; sus gaussianas, cuántas se entregaron, y las visibles y los pares que contó el dispositivo para ella; su nivel (*level 1 of 3 in 'bird'*, o *cut* con sus gaussianas propias y fusionadas); para un `.athc` en streaming, los chunks en el dispositivo, pedidos, ausentes y cargando; lo que lleva (grado SH, lineal o sRGB de captura, reiluminada, lit body, transfer, con esqueleto, normales, emisión, PBR, ids, visibilidad horneada, ior); lo que ocupa | hasta 24 nubes; el resto entra en los totales |

Las cuentas cuestan cuatro dispatches pequeños y una copia por frame, y solo se
toman mientras el panel está abierto: plegar su ventana las detiene.

**GPU memory** aparece abajo en la ventana cuando al dispositivo le faltó
memoria: a qué renunció el motor (las sombras de splats, un nivel de detalle) y
cuántos niveles más grueso de lo pedido dibuja ahora, o, donde no quedaba nada
que dar, que el frame se saltó y el siguiente lo vuelve a intentar. La ventana
sigue; cerrar el panel quita el mensaje hasta la siguiente vez.

### 5.2.1 Cambiar de qué está hecho el prim picado

El rasterizador guarda un Cryptomatte en todos los frames que dibuja, sea o no
la matte lo que enseña la ventana, porque una gaussiana no escribe `primId` y
la matte es el único nombre que tiene un píxel de splats. No le cuesta nada al
color y al frame le cuesta como una décima parte: de 13.7 ms a 15.5 ms con una
nube de 730 000 gaussianas a 1600 x 900. La ruta trazada no escribe matte, así
que bajo `rt` una nube no se puede picar por id y el panel lo dice, con un
botón para volver a Raster.

Picar un píxel de nube responde con un id de Cryptomatte, y todas las
gaussianas que llevan ese id vinieron de un mismo prim. El panel las cuenta y
dice de qué están hechas -- `12 400 gaussians carry it`, y luego el metallic,
la roughness y la transmission, con el rango al lado de la media allí donde un
mapa le dio a cada gaussiana la suya.

Bajo **Say otherwise** están esos tres como sliders, más un tinte. Mover uno
pone una fila en la tabla del frame, con ese id como clave: todas las
gaussianas de ese prim la toman a la vez, en las dos rutas. Lo que no has
movido se queda como lo escribió el fichero, así que un tinte no aplana un
mapa de roughness. **Put the file back** quita la fila de ese prim y **Put
every prim back** las quita todas.

Es la opinión del frame y nada más: la nube en disco no se toca y no hay nada
que guardar. Una nube sin Cryptomatte -- una captura, que no vino de ningún
prim -- no tiene nada que direccionar, y el panel lo dice.

### 5.3 Salidas, y aislar una matte

El combo Output elige qué enseña la ventana: el color, la profundidad, los ids
de prim, de instancia y de elemento, las normales en ojo y en mundo, el
Cryptomatte previsualizado con un color por id, o cualquiera de sus tres capas
en crudo. `--aov` arranca en una de ellas.

Picar un píxel lo nombra dos veces. El pick de Hydra da el prim de la
superficie que hay detrás; la matte da lo que más cubre el píxel, que para una
nube es el único nombre que tiene, porque una gaussiana no escribe id de prim.
Donde una nube está sobre una malla los dos difieren a propósito, y el panel
enseña los dos.

Con una matte picada, **Isolate this matte** enseña ese id solo, blanco sobre
negro — la comprobación de que el id del capó es el del capó. `--isolate
<prim>` hace lo mismo desde la línea de comandos, y enciende la salida
Cryptomatte por su cuenta.

### 5.4 Snapshots

`--frames N --snapshot salida.exr` cierra la ventana tras N frames y escribe
el último **tal como se ve**: codificado para pantalla, con los paneles
dentro. Así se hacen las imágenes de la documentación de este repositorio, y
no es una salida de render: para eso, `athenea stage`.

## 6. El servidor MCP

`athenea-mcp` es el motor como servidor de Model Context Protocol: JSON-RPC 2.0
delimitado por saltos de línea sobre stdin y stdout. Mantiene abiertos el
dispositivo y la escena entre llamadas, así que el segundo render de una
escena cuesta lo que debe costar un segundo render.

```sh
claude mcp add athenea -- <build>/bin/athenea-mcp
```

La salida estándar lleva solo protocolo; el log va a la de error.

| Herramienta | Qué hace |
|---|---|
| `open_stage` | abre una escena y la conserva; responde con sus cámaras, su eje vertical y su rango de tiempo |
| `stage_tree` | qué hay bajo una ruta de prim: nombre, tipo, si tiene hijos |
| `variants` | los conjuntos de variantes de la escena, y hace una selección |
| `device_info` | la GPU abierta y qué sabe hacer |
| `render` | renderiza la escena abierta; responde con una vista previa y los tiempos. Los ajustes se pegan entre llamadas |
| `render_products` | renderiza los productos de un prim `UsdRenderSettings` a EXR |
| `pick` | qué dibujó el último frame en un píxel |
| `bounds` | dónde está, en mundo, lo que dibujó el último frame |
| `convert` | una captura de splats a una escena USD |
| `timings` | varios frames de la escena abierta, y su mediana |
| `settings` | qué guarda la sesión: la escena, el último frame, qué se le puede pedir a un render |

**El `output` de `render` escribe dos ficheros distintos.** Un nombre que
acaba en `.exr` recibe los números del beauty, lineales. Cualquier otro nombre
recibe el frame **tal como se ve**: la AOV que pidió `aov`, pasada por el view
transform, en PNG. Es la única forma de conservar los colores de un
Cryptomatte, de una normal o de una profundidad, porque ninguno de ellos es un
color y los números del beauty no son los suyos; y es con lo que se rueda una
secuencia de una AOV, una llamada por frame sobre una escena que se queda
abierta.

Los argumentos de cada herramienta se declaran en el protocolo y los enseña el
cliente, así que aquí no se repiten. La forma de una sesión es: `open_stage`,
luego `stage_tree` para encontrar una cámara o un prim, luego `render`, luego
`pick` sobre algo de la imagen. `settings` dice qué guarda la sesión en ese
momento.

## 7. Efectos AOFX

El motor hospeda plugins AOFX, y `athenea mesh2splat` es uno de ellos corriendo
por ese host. Los bundles se buscan en este orden, y un duplicado encontrado
dos veces se carga una:

1. los directorios de `$AOFX_PLUGIN_PATH`;
2. la ruta del sistema — `/Library/AOFX/Plugins` en macOS, `C:/Program
   Files/Common Files/AOFX/Plugins` en Windows, `/usr/AOFX/Plugins` en el
   resto;
3. los directorios dados con `--path`;
4. el directorio de bundles que esta compilación lleva dentro.

`athenea aofx list` imprime qué se encontró, qué se rechazó y por qué, y qué
declara cada bundle. `athenea aofx run` corre un efecto sobre ficheros EXR, con
`--param nombre=valor` para lo que declare.

### 7.1 Measure — `rt.sparrow.aofx.measure`

Un nodo de QC: `Source` frente a `Reference` (opcional), medidas en el
dispositivo donde ya están, con los números colgados de la salida. Es la
implementación de `athenea compare`, y el mismo bundle carga en openFXplayer.

| Parámetro | Valor | Por defecto | Notas |
|---|---|---|---|
| `mode` | choice: `source`, `difference`, `relative`, `codes` | `codes` | la imagen de salida. `source` deja pasar Source; `difference` es \|S − R\| × gain, alfa 1; `relative` es una rampa (negro, azul, cian, verde, amarillo, rojo) de \|S − R\| / max(\|R\|, 0.001) del canal mayor, rojo en 1 / gain; `codes` colorea los píxeles cuya diferencia de 8 bits pasa de `threshold`, rojo en 32 / gain, y deja el resto en negro. Sin Reference la imagen es Source |
| `gain` | número ≥ 0 | `1` | solo el mapa de calor |
| `threshold` | entero 0–255 | `2` | valores de código de 8 bits que un píxel puede diferir sin contar como por encima |
| `window` | cuatro números, píxeles | `0 0 0 0` | X0 Y0 X1 Y1 en píxeles de Source, filas desde abajo; un X1 o Y1 de 0 es el borde. Solo las medias y los máximos |

Lo que adjunta, todo floats. Una cuenta son dos floats, `alto × 2^24 + bajo`,
cada uno exacto (alto es 0 por debajo de 16 777 216):

| Id | Valores |
|---|---|
| `source` | media R G B A, máximo R G B A, suma R G B A, los píxeles de la ventana (alto, bajo) |
| `reference` | lo mismo de Reference; solo si está conectada |
| `hdr` | relMSE, diferencia relativa p99, mayor diferencia relativa, la suma del error cuadrático relativo, píxeles (alto, bajo) |
| `codes` | p99 de 8 bits, máximo de 8 bits, píxeles por encima de `threshold` (alto, bajo), `threshold`, píxeles (alto, bajo) |

La diferencia relativa va en bins de un octavo de octava desde 2^-16, así que
su p99 y su máximo son la cota superior de un bin (en torno al 9 %). Las sumas
están para un host que divide en double, como hace `athenea compare`. Dos
imágenes de tamaños distintos se rechazan (`measure compares pictures of one
size`), y también una ventana sin nada dentro (`measure: an empty window`).

## 8. Referencia

### 8.1 Variables de entorno

Las que cambian cómo se comporta una ejecución están en §1.2. El resto sirven
para mirar dentro de una ejecución, y son de quien trabaja en el motor más que
de quien lo ejecuta: están en
[development.es.md §3](development.es.md#3-trabajar-en-el-árbol).

### 8.2 Formatos

**Nubes que se leen.** `.ply` (3DGS), `.splat`, `.spz` (Niantic, necesita
zstd), `.sog` o un `meta.json` suelto (PlayCanvas, necesita libwebp), `.athc`
(el contenedor por chunks de este motor).

**Puntos que se leen.** `.ply`, `.xyz`, `.txt`, `.pts`, `.csv`, y los
`points3D.txt` y `points3D.bin` de COLMAP.

**Otras entradas.** Campos de volumen `.vdb` por OpenVDB; perfiles
fotométricos IESNA LM-63; y las texturas de materiales por los plugins de
imagen de OpenUSD, así que lo que lea esa compilación.

**Imágenes que se escriben.** OpenEXR, lineal premultiplicado, con la fila de
abajo primero dentro del motor y escrito con la de arriba primero como quiere
el formato. Un frame va en media por defecto con un canal `Z`; los canales de
un producto siguen a sus vars, y una capa Cryptomatte va siempre en float. PNG
solo se escribe como la vista previa de MCP.

### 8.3 Qué escribe cada comando

| Comando | Escribe |
|---|---|
| `athenea info` | un informe por la salida estándar |
| `athenea render` | un EXR: RGBA en media y `Z` |
| `athenea bench` | nada; los tiempos por la salida estándar |
| `athenea convert` | una escena USD, o un `.athc` |
| `athenea stage` | un EXR; o, con `--render-settings`, uno por producto |
| `athenea view` | nada, o un EXR con `--snapshot` |
| `athenea live` | un EXR por frame, numerado, cada uno con su timecode |
| `athenea mesh2splat` | una escena USD con la nube |
| `athenea visibility` | el fichero de la nube, editado en el sitio o copiado |
| `athenea aofx run` | un EXR |
| `athenea migrate` | una copia de la escena, el paquete o la nube; con `--recursive`, también de lo que nombra |

### 8.4 Los scripts

En `scripts/` está lo que compila las dependencias y lo que reproduce los
assets que enseña el README.

| Script | Qué hace |
|---|---|
| `build-usd.sh [versión\|dev]` | compila OpenUSD con MaterialX y OpenVDB en su prefijo, sin Python |
| `build-oidn.sh [versión]` | compila Open Image Denoise, solo dispositivos GPU |
| `build-ocio.sh [versión]` | compila OpenColorIO con sus dependencias enlazadas estáticamente |
| `build-dawn.sh` | deja Dawn 138.0.7204.168, precompilado, en `~/tools/dawn-<versión>` (`ATHENEA_DAWN_ROOT` lo cambia), comprobado contra su SHA-256 |
| `wgsl-report.py [--markdown] [--only E,...] [--out DIR]` | compila a WGSL con `slangc` los kernels del raster de splats y el efecto Measure, e imprime por kernel sus storage buffers, sus bytes de workgroup y lo que falla; con Naga (`cargo install naga-cli`) y Dawn presentes, también sus veredictos. Solo CPU. El WGSL va a `--out` (por defecto `$TMPDIR/wgsl`) |
| `wgsl-tint.cpp` | Tint, por el backend nulo de Dawn, sobre un fichero WGSL: el módulo, y después su pipeline con los límites por defecto de la web. Lo compila `wgsl-report.py` |
| `web-kernels.py --out DIR [--no-tint]` | el visor web (3.6): los kernels de sus módulos como WGSL, `manifest.json`, y `web/` al lado; falla con un kernel que no compila, que Naga o Tint rechazan, o que pasa de 8 storage buffers o de 16384 bytes de workgroup. Solo CPU |
| `fetch-fox.sh [dir]` | el zorro de Khronos, por Blender, para una conversión con esqueleto |
| `sketchfab-to-usd.sh <zip> [nombre]` | un archivo de Sketchfab a un asset USD |
| `readme-images.sh [salida]` | las imágenes del README, desde el asset del gorrión |
| `remote-test.sh [user@host] [preset]` | compila y corre la suite en otra máquina y trae el log |
| `scripts/film/` | la película del gorrión: sus frames, su cámara y su sombra |
| `bmw-to-usd.py`, `sparrow-*.py` | ayudas de autoría para esos assets |

La cabecera de cada script dice qué necesita y dónde deja las cosas.

## 9. Cuando algo falla

| Qué se imprime | Qué significa | Qué hacer |
|---|---|---|
| `no GPU device` (los tests se saltan) | no se pudo abrir dispositivo | mira `athenea info`; en Linux pon `ATHENEA_BACKEND` |
| `colour: no colour space '<nombre>' in <config>; read as the file says` | una textura nombra un espacio de color que no conocen ni el config ni el studio config | corrige el nombre (`athenea info` dice si OpenColorIO está compilado); la textura se lee como si no se hubiera dado espacio de color |
| `no Measure bundle on the AOFX search path` | `athenea compare` no encontró `rt.sparrow.aofx.measure` | compila el target `athenea_aofx_measure`, o da su directorio con `--path` |
| `measure: an empty window` | `athenea compare --window` (o el `window` del efecto) no contiene ningún píxel | da X0 < X1 e Y0 < Y1 dentro de la imagen |
| un error de compilación de shader con una ruta | los shaders del disco no son los del binario | recompila, o apunta `ATHENEA_SHADER_DIR` al `shaders` de esta compilación |
| `this build reads no .spz` | faltaba zstd cuando se compiló este binario | recompila con zstd, o convierte la captura en otro sitio |
| un `.sog` rechazado | faltaba libwebp | instálalo y recompila |
| `a .athc is drawn through its levels of detail: give --lod` | se le dio un contenedor a un render normal | añade `--lod`, y usa `--technique raster` |
| `mesh visibility by rays: the device has no ray tracing` | `--visibility rays` en un dispositivo sin ello | usa `automatic`, que elige lo que el dispositivo tiene |
| un host no ofrece el renderer | no encontró el plugin | pon `PXR_PLUGINPATH_NAME` a `<build>/plugin/usd` |
| `render product '<ruta>' has no resolution` / `no vars` | el prim de settings está incompleto | dale al producto resolución y vars ordenadas |
| una nube convertida sale más gruesa de lo pedido, y `warning: the meshes want N splats` | `--max-splats` estaba por debajo de lo que querían las mallas, y todas se engrosaron por igual para repartirlo | sube `--max-splats`, o baja `--resolution` para elegir tú lo grueso |
| `warning: the budget is exhausted` o `the budget ran out before N mesh(es)` | `--max-splats` era menor que lo que querían las mallas | sube `--max-splats`, baja `--resolution`, o con `--density per-mesh` sube `--cell-min` |
| `warning: N cells lay past --max-cells` | un triángulo quería más celdas de las que puede recorrer uno; el resto queda desnudo | sube `--max-cells`, o baja `--resolution` |
| los reflejos de una nube salen más planos que los de la malla | no lleva normal de sombreado (`primvars:athenea:splat:normal`): se convirtió antes de que las conversiones la escribieran | conviértela de nuevo; `--normal-map-turns` además gira los propios discos |
| `cells of relief wanted more than N gaussians`, y el relieve muestra huecos en sus pendientes más fuertes | el relieve estiró esas celdas más de lo que permite la partición | sube `--displace-refine`; un polo de las coordenadas de textura estira sin límite y deja unas pocas sea cual sea el valor |
| los reflejos de una nube salen más blandos que los de la malla | la celda de la conversión es el kernel de desenfoque: una nube se lee como la malla a `r + 9c/R`, con `c` la celda y `R` el radio de curvatura | convierte con `--resolution` más fina: un espejo de roughness `r` quiere una celda por debajo de `r/9` de ese radio. Lo paga el fichero, no el frame -- quince veces las gaussianas fueron un 36 % más de tiempo por frame y dieciséis veces el disco |
| una bola de cristal enseña la sala pero no la dobla | la nube no tiene índice | `athenea mesh2splat` escribe el IOR del material de cristal; en una nube de otro origen pon `primvars:athenea:splat:ior` (1.5 es cristal). Una nube guarda un solo índice: con dos cristales de IOR distinto se queda el primero y la conversión lo avisa; una nube que lleva las capas dobla cada gaussiana por su propio índice especular |
| una nube convertida sale negra | el bake no encontró luz | dale luces a la escena, o `--default-lights`, o `--no-bake` |
| `<file>: not a readable .athc (unknown flag bits N; this reads bits 0 (normals), 1 (linear), 2 (emission), 4 (material) and 5 (transfer))` | el `.athc` lo escribió un motor más nuevo, con algo en sus bloques que esta compilación no sabe dónde buscar | léelo con ese motor, o actualiza este |
| una nube sale roma y luego se afina | todavía están llegando chunks | sube `--stream-budget`, o espera; una imagen fija se asienta antes |
| `OutOfMemory: ... does not fit in the GPU's memory budget`, código de salida 3 | el frame necesitaba más que el presupuesto del dispositivo, aun después de que el motor devolviera lo que pudo y bajara de nivel | cierra lo que más ocupe la GPU, renderiza más pequeño, da a un asset en streaming un presupuesto menor; `ATHENEA_GPU_BUDGET` sube o baja el presupuesto |
| `OutOfMemory: ... does not fit in the memory the system has free` | en Apple silicon, la memoria libre de la máquina menos su reserva de 1,5 GiB no cabría la reserva: otros procesos ocupan el resto | cierra lo que más corra, o renderiza más pequeño; la reserva no se configura |
| `the GPU ran out of memory running ...` | un command buffer falló en el propio dispositivo (`Insufficient Memory` de Metal), sin llegar aún al presupuesto -- otros trabajos ocupan el resto | lo mismo; bajar `ATHENEA_GPU_BUDGET` mantiene esta ejecución por debajo de lo que dejan |
| `trying again with ...; levels of detail N coarser than asked` | al dispositivo le faltó memoria y el motor bajó de nivel; los frames siguen | nada, o lo mismo que arriba para recuperar el detalle (una ejecución nueva empieza como se pidió) |
| `splat shadows skipped: the proxies of N gaussians need ~M MiB` | se pidieron sombras de splats para una nube cuyos proxies ocuparían más de la mitad de lo que le queda al presupuesto (unos 1,7 KB por gaussiana) | una nube más pequeña o sus niveles de detalle; volver a pedirlas (apagar y encender `athenea:splatShadows`) lo vuelve a intentar |
| fallan los tests del oráculo de Storm | `HDX_MSAA_SAMPLE_COUNT` no es 1 | ctest lo pone; ponlo a mano si corres el binario directamente |

## 10. Glosario

| Término | English | Qué significa aquí |
|---|---|---|
| gaussiana | splat, gaussian | una gaussiana anisótropa: posición, tres tamaños, rotación, opacidad y sus armónicos |
| nube | cloud | un conjunto de gaussianas dibujado como un prim |
| captura | capture | una nube entrenada a partir de fotografías; no tiene ascendencia en un modelo |
| conversión | conversion | una nube hecha de mallas por `athenea mesh2splat` |
| horneado | bake | escribir dentro de la nube lo que si no habría que calcular cada frame: la luz, o lo que ocluye una parte |
| corte | cut | qué nivel de detalle toma un frame, decidido por cuántos píxeles abarca una celda fundida |
| chunk | chunk | la unidad que un `.athc` transmite: un tramo de gaussianas en orden de Morton |
| grupo | group | las gaussianas que un nivel fundido sustituye por una |
| parte | part | lo que lleva un joint, y la unidad en que se hornea la visibilidad |
| matte | matte | una capa Cryptomatte: qué prims cubrieron un píxel, y cuánto |
| manifest | manifest | el mapa de un id a la ruta de prim que representa |
| producto, var | product, var | un `UsdRenderProduct` y su `UsdRenderVar`: un fichero, y una capa dentro |
| registro de instancia | instance record | lo que guarda el dispositivo de una instancia dibujada de un prim |
| lobe | lobe | un término de un material: un difuso, un especular, un conductor |
