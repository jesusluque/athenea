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
prefijos, los tres scripts de dependencias, los submódulos y los
presets de CMake. Aquí no se repite. Lo que importa después es el entorno en
el que corre el binario.

Una release en GitHub trae un instalador y un archivo para cada sistema. Cada
uno lleva los programas con todas las bibliotecas que cargan (OpenUSD con
MaterialX, OpenVDB, TBB y Boost, OpenColorIO, Slang, OIDN y, en Linux y
Windows, el runtime redistribuible de CUDA: NVRTC, nvJitLink, cudart); el
driver de la GPU es el de la máquina.

| Sistema | Instalador | Instala en | Necesita |
|---|---|---|---|
| macOS (Apple silicon) | `athenea-<v>-macos-arm64.pkg` | `/usr/local/athenea`, programas enlazados en `/usr/local/bin` | macOS 15 o posterior. Sin firma de Developer ID: se abre con clic derecho, Abrir. |
| Linux (x86_64) | `athenea_<v>_amd64.deb` | `/opt/athenea`, programas enlazados en `/usr/bin` | glibc 2.39 (Ubuntu 24.04 o posterior) y un driver de NVIDIA con OptiX para hornear (570 o posterior). |
| Windows (x64) | `athenea-<v>-windows-x64-setup.exe` | `Program Files\athenea`, `bin\` añadido al `PATH` de la máquina | un driver de NVIDIA con OptiX (570 o posterior). Sin firma: SmartScreen pregunta una vez. |

Los archivos (`.zip`, `.tar.gz`) llevan la misma carpeta y funcionan desde
donde se descompriman: los programas encuentran los shaders, los bundles AOFX
y el plugin de Hydra con rutas relativas a ellos. Bajo WSL2 un horneado se
rechaza: WSL2 no carga OptiX.

En Windows el paquete lleva además las cabeceras con las que compila NVRTC
(`include\`), y el motor apunta `CUDA_PATH` al paquete antes de abrir un
dispositivo CUDA, así que la máquina no necesita el toolkit de CUDA. Donde hay
uno instalado, su `CUDA_PATH` se deja de lado para el proceso: el runtime del
paquete es con el que se probó, y NVRTC de una versión con los builtins de
otra no abre un dispositivo.

Un frame trazado por caminos o un horneado en CUDA compila además contra las
cabeceras de OptiX, que la licencia de NVIDIA no deja llevar al paquete. Una
máquina las descarga ella misma (`include\` de github.com/NVIDIA/optix-dev, tag
`v9.0.0`) en el `include\` del paquete, o las nombra con
`ATHENEA_OPTIX_INCLUDE`. Sin ellas un pipeline de trazado de rayos falla con
`Failed to locate OptiX headers (optix.h)`.

### 1.2 El entorno de ejecución

| Variable | Qué hace |
|---|---|
| `ATHENEA_SHADER_DIR` | de dónde se leen los shaders de Slang. Sin ella el motor busca un directorio `shaders` que contenga `athenea/` junto a la imagen desde la que se cargó su código, o hasta tres directorios por encima -- `<exe>/../shaders` para un programa, `<build>/shaders` para el plugin de Hydra cargado por usdview o Blender --, luego junto al ejecutable, y luego en el directorio con el que se compiló. `athenea info` imprime el que está usando. |
| `PXR_PLUGINPATH_NAME` | apunta una aplicación USD a `<build>/plugin/usd`, donde están el delegate de Hydra y los schemas codeless. La necesita cualquier host que no sea `athenea`. `athenea` registra por su cuenta `<su binario>/../plugin/usd` al arrancar, así que los schemas que aplica una nube convertida se escriben esté o no definida. |
| `ATHENEA_MATERIALX_ROOT` | un directorio que contiene los `libraries/` de MaterialX, que el compilador de materiales de hdAthenea lee en lugar de las bibliotecas que cargó el USD del host. Sin definir por defecto: se usan las del host. Para un host cuyo MaterialX es anterior al generador de Slang (Blender 5.3 trae 1.39.4: sin implementaciones `genslang` y con definiciones de nodo anteriores), se apunta a las de 1.39.5. Se lee una vez, cuando compila el primer material; los renderers propios del host conservan las suyas. |
| `AOFX_PLUGIN_PATH` | directorios extra de bundles AOFX, buscados antes que la ruta del sistema y antes que `--path`. Después van `aofx/` junto a la carpeta del ejecutable (la disposición de una release) y los bundles del árbol de compilación. |
| `ATHENEA_MCP` | el servidor que arranca el plugin OpenFX (§6.1), en lugar del `athenea-mcp` que lleva su bundle en `Contents/Resources/bin` y del que haya en el `PATH`. |
| `ATHENEA_OFX_DEBUG` | con cualquier valor: el plugin OpenFX escribe en la salida de error la petición de cada frame y la respuesta del servidor (su tiempo). |
| `ATHENEA_STREAM_TRACE` | con cualquier valor: cada pool en vivo registra, en cada actualización, la época mostrada, las páginas colocadas, las épocas que esperan y las que se construyen (páginas que faltan), las cargas en cola y en curso, el staging libre y los contadores de su fuente. |
| `ATHENEA_BACKEND` | qué dispositivo abrir, como un orden separado por comas: `metal,cuda,vulkan,d3d12`. Las palabras desconocidas avisan y se saltan. |
| `ATHENEA_GPU_VALIDATION` | cualquier valor: las capas de validación y depuración de la API gráfica (la de D3D12 necesita las Graphics Tools de Windows), para un fallo del que el driver no da motivo, como un pipeline que no puede crear. Lento. |
| `ATHENEA_GPU_BUDGET` | la memoria del dispositivo que puede ocupar esta ejecución, en MiB. Sin ella el presupuesto es el working set recomendado de Metal (`recommendedMaxWorkingSetSize`); en CUDA y Vulkan no hay ninguno salvo que esta lo ponga. El dispositivo imprime el que usa (`GPU memory budget: N MiB`). Una reserva que lo pase falla como `OutOfMemory` antes de hacerse, los presupuestos de streaming y las sombras de splats se dimensionan con lo que deja (§3.3, §9), y es como se mantiene una ejecución por debajo de lo que dejan otros trabajos en la misma GPU. En Apple silicon una reserva además debe caber en la memoria física que el sistema tiene libre menos 1,5 GiB guardados para el resto de la máquina -- contando como libre lo que devuelve cuando se le pide (páginas inactivas, purgables y especulativas, y la caché de ficheros) -- diga lo que diga el presupuesto: pasado eso la máquina manda a swap la memoria de la GPU y deja de dibujar sus ventanas. |
| `ATHENEA_VOLUME_PLY` | una ruta: cada frame que dibuja un medio de gaussianas (los medios de §3.3, `primvars:athenea:splat:volume`) escribe además ese medio como un PLY de 3DGS con la luz que el frame horneó para él: la radiancia de cada gaussiana (emisión, el single scattering de las luces a través del cloud shadow map, el domo) pasada por Reinhard (x / (1 + x) por canal) y la curva sRGB como su color base, ln sigma y la temperatura en `f_rest_0` y `f_rest_1`, las posiciones en el espacio propio de la nube (sin la transformación del prim), un registro por cada registro del stage, uno muerto o descartado una mota en el origen con ln sigma -10 -- la disposición de `vdb2splat --cloud`, así que `video-encode --volume` lo toma. Un segundo medio se escribe como `<stem>_1.ply`, y así sucesivamente. Se sobrescribe en cada frame: pensado para un frame por ejecución (`athenea stage`). |
| `ATHENEA_VOLUME_PLY_EXPOSURE` | pasos de exposición sobre la radiancia antes de la curva, para `ATHENEA_VOLUME_PLY`; por defecto 0. |
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
| OpenVDB | los campos de volumen `.vdb` se rechazan al leerlos. El path tracer sigue necesitando `nanovdb/PNanoVDB.h` de NanoVDB, solo cabecera: un prefijo de USD que no la tiene nombra un directorio que la contiene con `-DATHENEA_NANOVDB_INCLUDE=<dir>`, o el configure avisa de que los kernels del path tracer no compilarán. |
| Cabeceras de OptiX (CUDA) | el ray tracing por pipeline en CUDA; los ray queries en línea siguen, así que casi todo el motor también. |
| zstd | las nubes `.spz` se rechazan, con `this build reads no .spz`. |

### 1.4 Metal y CUDA

El motor abre un dispositivo y lo conserva. En macOS es Metal; en Linux es
CUDA donde hay un dispositivo CUDA, y Vulkan si no (y en Windows CUDA, luego
D3D12, luego Vulkan). Un build cuyo gpe es el de Vulkan
(`GPE_BACKEND_WANTED=Vulkan`) prueba Vulkan primero, porque los efectos AOFX
-- `mesh2splat` entre ellos -- sólo corren en el dispositivo que gpe comparte.
`ATHENEA_BACKEND` manda sobre ese orden.

Lo que cambia entre ellos, en la práctica:

- **Ray tracing.** Metal tiene ray queries en línea y estructuras de
  aceleración pero no pipelines de rayos, así que la ruta que necesita un
  pipeline no está disponible y `athenea info` lo dice. CUDA tiene el pipeline
  (OptiX, con sus cabeceras y un driver que lo cargue) y no tiene ray queries.
  Un horneado necesita uno de los dos: donde no hay ninguno -- CUDA bajo WSL2,
  que no carga OptiX -- `athenea mesh2splat` lo rechaza con "has no ray
  tracing".
- **El denoiser.** En Metal, Open Image Denoise corre en la cola del propio
  motor. En Vulkan corre a través de CUDA con memoria importada.
- **Media precisión.** `athenea info` dice si la hay; donde no la hay el motor se
  queda con buffers float y no cambia nada más.

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
| `--backend` | `metal` \| `cuda` \| `vulkan` | la preferencia de la plataforma | un backend, no una lista |

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
| `--no-antialias` | flag | apagado | quita la compensación de filtro 2D de Mip-Splatting a todas las nubes, por encima de lo que diga un `.spz` |
| `--neural` | ruta | — | un `.athn` (§2.3.2, §8.2): los colores de la nube desde sus latentes y su red en vez de desde sus armónicos. Un solo fichero en `--splats`, aquel al que se ajustó el `.athn`, y sin `--lod`; solo raster |
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
| `--features` | 8, 16 o 32 | apagado | con `--neural`: las features de la nube mezcladas y decodificadas una vez por píxel en vez de su color decodificado una vez por gaussiana, para cronometrarlo; la red del píxel no está entrenada, así que el frame no es para mirarlo |
| `--trace` | flag | apagado | el frame guarda su trace -- las cuatro gaussianas de más peso de cada píxel, 44 bytes por píxel -- para cronometrar lo que cuesta; solo raster |

No escribe imagen. Con `--stages` informa por separado de la proyección, el
sort por profundidad, las cuentas, el emit, el sort por tile y el blend.

### 2.3.1 `athenea lift` — una máscara sobre los frames de unas cámaras, como el matte que ve otra

Una máscara hecha sobre uno o más frames de una nube se lleva a sus gaussianas, y
el frame de otra cámara se escribe con ese key: lo que cubren allí las gaussianas
de dentro de la máscara, y nada más. No intervienen ni malla ni croma; los propios
pesos del blend de la nube dicen de qué gaussianas estaba hecho cada píxel de la máscara.

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `--splats` | rutas, obligatorio | — | `.ply`, `.splat`, `.spz`, `.sog`; varias son una sola escena |
| `--from` | `x y z`, repetible | — | una cámara sobre la que se hace la máscara; al menos una |
| `--mask` | rutas | — | un EXR por cada `--from`, del tamaño del frame: su canal rojo, 1 dentro de la máscara |
| `--rect` | `left bottom right top` | — | en lugar de `--mask`: el mismo rectángulo en el frame de cada `--from`, en fracciones del frame desde su esquina inferior izquierda |
| `--eye` | `x y z`, obligatorio | — | la cámara para la que se dibuja el matte |
| `--target` | `x y z` | el centro de las nubes | adonde miran todas las cámaras |
| `--up` | `x y z` | `0 1 0` | |
| `--focal` | mm | `35` | la de todas las cámaras |
| `--size` | `WIDTHxHEIGHT` | `1920x1080` | el de todos los frames |
| `--degree` | 0..3 | `3` | armónicos conservados |
| `--hard` | flag | apagado | una gaussiana está dentro de la máscara o fuera, en vez de estarlo en la proporción de su peso que cubrieron las máscaras |
| `--threshold` | 0..1 | `0.5` | `--hard`: esa proporción |
| `--matte-by-trace` | flag | apagado | lee el matte de la trace del frame de `--eye` (las cuatro gaussianas de más peso de un píxel, y una suposición para el resto) en vez de que lo sume el blend del frame sobre todas las gaussianas, que es lo que se hace por defecto y no guarda trace |
| `--by-trace` | flag | apagado | hace el lift por la trace de cada frame (las cuatro gaussianas de más peso de un píxel) en vez de durante el blend (todas) |
| `--turn` | entero | `0` | además este número de frames de la cámara `--eye` dando una vuelta al target (sobre y), cada uno con el key del mismo lift: `<output>_NNN.exr` |
| `-o`, `--output` | ruta | `lift.exr` | el frame de `--eye`, premultiplicado por el matte, con el matte como alfa, y su profundidad |

Tras escribir el frame, el comando dice cuánto parpadea el key al pasar la
cámara: tres frames a medio grado alrededor del target, el central
contra la media de sus vecinos, para los frames con key y para los frames
completos (relMSE, p99 y la mayor diferencia en code values de 8 bits).

Una gaussiana que ninguna cámara `--from` vio no tiene opinión, y se toma como dentro
de la máscara en la medida en que lo están las demás de su píxel: una máscara hecha
desde un lado deja el lado contrario a esa suposición, y más cámaras `--from` son lo que la quita.

### 2.3.2 `athenea neural-fit` — los armónicos de una nube aprendidos por un latente por gaussiana y una red

Ajusta una apariencia neural a una nube y dice cuánto se acercó: la nube dibujada
por sus armónicos, desde cámaras repartidas por una esfera a su alrededor, es hacia
lo que se mueve la nube dibujada con once números por gaussiana y una red de 1008 pesos.
La geometría no se toca. `--save` escribe la apariencia ajustada; `-o`, tres
frames para mirar.

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `--splats` | ruta | — | un `.ply`, `.splat`, `.spz` o `.sog`; esto o `--pictures` |
| `--pictures` | ruta | — | en lugar de `--splats`: un manifiesto de una malla horneada en las mismas gaussianas bajo varios soles, y un fit iluminado a esas nubes (abajo) |
| `--steps` | entero | `3000` | una cámara cada uno; el error y la medida de las cámaras reservadas se imprimen diez veces por el camino |
| `--size` | `WIDTHxHEIGHT` | `960x540` | el de los frames ajustados y el de los medidos |
| `--target` | `x y z` | el centro de la nube | adonde miran todas las cámaras |
| `--up` | `x y z` | `0 1 0` | y o z arriba |
| `--distance` | número | 1,4 veces el lado mayor de la nube | la distancia de cada cámara al target, en las unidades de la nube |
| `--focal` | mm | `35` | |
| `--degree` | 0..3 | `3` | los armónicos que se aprenden |
| `--held-out` | entero | `8` | cámaras que el fit nunca usa, medidas antes y después |
| `--rate-features` | número | `0.01` | el rate de Adam para las features |
| `--rate-base` | número | `0.002` | para los colores base |
| `--rate-weights` | número | `0.004` | para la red |
| `--rate-decay` | número | `1` | a cuánto han caído los tres rates en el último paso, como factor, paso a paso; `0.1` es lo que pide un fit largo |
| `--view-smooth` | número | `0` | con cuánta fuerza se sujeta el color de cada gaussiana vista un grado más allá a su color, como múltiplo de lo que pesa el error de un píxel; `0` no lo sujeta y ahorra la segunda evaluación. `64` bajó el shimmer del color 1,6 veces en el peón sin coste medido de calidad |
| `--split` | flag | off | dos redes de la misma forma en vez de una: a la primera nunca se le dice la vista, así que lo que dice de una gaussiana es lo mismo desde cualquier cámara, y la segunda da lo que añade la vista. 2016 pesos donde una red son 1008, dos evaluaciones por gaussiana, y se guarda así en un `.athn` (`--save`); `--load` toma una apariencia como se guardó |
| `--reached` | `ray` o `map` | — | `--shadows`: a la nube neural se le da además lo que llega a cada gaussiana desde la luz, y su color es su respuesta por eso. `ray`: la transmitancia que encuentra el shadow ray del motor, así que cada frame dibujado con ella traza esos rayos. `map`: leída del shadow map de la nube desde el sol, construido sin ningún rayo; lo que la nube debe ser sigue siendo lo de los rayos |
| `--map-side` | entero | `2048` | `--reached map`: los texels por lado del shadow map |
| `--map-coefficients` | 1, 3, 5 o 7 | `7` | `--reached map`: la profundidad óptica total del mapa y sus pares de Fourier |
| `--map-bias` | unidades de mundo | `0` | `--reached map`: cuánto por detrás del caster más cercano de un texel sigue un receptor estando sobre él; 0 es el 2% del slab |
| `--map-spread` | número | `0` | `--reached map`: cada gaussiana lee el map en su centro y en cuatro puntos a este número de sus propias desviaciones típicas a lo largo de sus dos ejes más largos, cada uno a su profundidad, y recibe la media (el centro cuenta dos veces); `0` lee solo el centro. `3` es donde el error del peón contra el path tracer fue menor |
| `--load` | ruta | — | un `.athn` del que partir; con `--steps 0` solo se mide |
| `--save` | ruta | — | escribe la apariencia ajustada como un `.athn` (§8.2), con el que `athenea render --neural` dibuja la nube |
| `--relight` | flag | apagado | ajusta en su lugar una nube neural iluminada: lo que debe ser es el relight del propio motor sobre la nube (sus colores como aquello de lo que está hecha, sin sombra) bajo un sol que viene de una dirección distinta en cada paso, y las cámaras reservadas se miden bajo soles que el fit nunca usó |
| `--sun` | número | `3` | `--relight`: la intensidad del sol |
| `--turn` | entero | `0` | `--relight` con `-o`: este número de frames de un sol que da una vuelta a la nube, a cuarenta grados de altura, desde la primera cámara reservada, como `<prefijo>_turn_engine_NNN.exr` y `_turn_latents_NNN.exr` |
| `--shadows` | flag | apagado | `--relight`: en lo que la nube debe ser, el sol arroja la sombra de la nube sobre sí misma (un rayo por gaussiana, a través de las gaussianas de la propia nube); necesita un dispositivo que trace rayos. A la nube neural se le sigue diciendo solo de dónde viene la luz |
| `-o`, `--output` | prefijo | — | escribe `<prefijo>_harmonics.exr`, `_base.exr` y `_latents.exr`: el frame de la primera cámara reservada con cada uno; con `--relight`, `<prefijo>_engine_sun0..2.exr` y `_latents_sun0..2.exr`, esa cámara bajo tres soles reservados |

Las medidas son las de `athenea compare` contra el frame de los armónicos, la
media sobre las cámaras reservadas: relMSE, el percentil 99 en code values de 8 bits,
y la mayor diferencia de cualquiera de ellas. Con `--relight` las mismas
medidas se toman además bajo tres soles de tres colores a la vez, contra el
relight del motor bajo esos tres (y `-o` escribe `_engine_three.exr` y
`_latents_three.exr`).

**`--pictures`: ajustado a lo que horneó el path tracer.** El manifiesto es un fichero
de texto, una línea por nube: `train` o `held` (o `base`, abajo), el giro del sol sobre y y luego sobre
x en grados (una luz distante que alumbra hacia su propio -z, como lo escriben
`xformOp:rotateY`, `xformOp:rotateX`), y un fichero USD junto al
manifiesto -- la misma malla convertida por `athenea mesh2splat` bajo ese sol, de modo
que cada fichero tiene las mismas gaussianas con la luz de ese sol, rebotes y
sombras, en sus armónicos. A una nube neural iluminada de esas gaussianas se le dice cada
sol `train` y se mueve hacia su nube, y se mide contra las nubes `held`,
cuyos soles nunca se le enseñaron, desde las cámaras reservadas. `--reached
map` le da el shadow map de la nube desde cada sol; aquí no hay rayos, así que
`--reached ray` y `--shadows` no se aplican. Una línea `base` (sus dos ángulos
no se leen) nombra las mismas gaussianas convertidas con `--no-bake`: los colores
base de la nube neural arrancan entonces como los del propio material y no como los de la
primera nube, que es la foto que ese sol hizo de ellas -- cuando las dos tienen las mismas
gaussianas, cosa que no ocurre si el bake descartó alguna, y entonces la línea se rechaza. `-o` escribe
`<prefijo>_picture_sun0..2.exr` y `_latents_sun0..2.exr`.

### 2.3.3 `athenea wall-fit` — una nube a la que se enseña lo que le hace una pared de luz

Una pared LED es una luz cuya imagen se conoce en cada frame. Cada gaussiana aprende
dónde de la pared lee esa imagen, con qué anchura y cuánto -- `--lobes`
lecturas de seis números cada una -- y queda iluminada por lo que la pared muestre con
una lectura filtrada por lobe, sin muestrear ninguna luz y sin ningún rayo. De lo que aprende es
del path tracer: la misma malla convertida en las mismas gaussianas con un solo texel de
la pared encendido cada vez. La primera línea del manifest es la pared, en el espacio de la
nube, y luego una línea por texel:

```
wall W H  cx cy cz  ux uy uz  vx vy vz  halfU halfV
patch <texel> <file.usda>
```

`W` x `H` texels; el centro; los ejes unitarios a lo largo de u y v de la imagen; la mitad
del tamaño de la pared en cada uno. El texel `y * W + x`, la fila de abajo primero, es la
nube que `athenea mesh2splat` hizo bajo un `RectLight` puesto donde está ese
texel. Cada texel pide su nube, y todas las nubes las mismas gaussianas.

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `--patches` | ruta | — | el manifest; obligatorio |
| `--steps` | entero | `4000` | una cámara cada uno |
| `--size` | `WIDTHxHEIGHT` | `960x540` | de los frames ajustados y medidos |
| `--target` | `x y z` | el centro de la nube | adonde mira cada cámara |
| `--distance` | número | 1,4 veces el lado mayor de la nube | en las unidades de la nube |
| `--focal` | mm | `35` | |
| `--up` | `x y z` | `0 1 0` | `0 0 1` para un stage cuyo z es arriba |
| `--lowest` | número | `-0.25` | lo más bajo que se pone una cámara: su altura sobre el target como fracción de su distancia, negativa por debajo. Una nube sobre un suelo lo quiere por encima de cero; lo más alto es `0.9` |
| `--held-out` | entero | `4` | cámaras que el fit nunca usa |
| `--lobes` | entero | `1` | lecturas que guarda una gaussiana; seis números cada una |
| `--start-level` | número | el nivel más ancho menos uno | con qué anchura empieza una lectura, como nivel del mip chain de la pared |
| `--start-gain` | número | `0.05` | con qué fuerza empiezan las lecturas de una gaussiana, juntas |
| `--rate-uv` | número | `0.002` | el rate de Adam para dónde está una lectura, en paredes por step |
| `--rate-level` | número | `0.01` | para su anchura |
| `--rate-gain` | número | `0.01` | para el logaritmo de su ganancia |
| `--rate-decay` | número | `0.1` | a cuánto han caído los rates en el último step |
| `--whole-every` | entero | `1` | uno de cada tantos steps se ajusta a una imagen completa -- un color por texel, un color por bloque, unos pocos texels encendidos -- cuyo frame es la nube de cada texel dibujada y sumada, así que cuesta un frame por texel; los demás a un texel solo. `0`: solo texels sueltos |
| `--grid` | entero | — | en vez de lecturas, un peso por cada texel de este nivel de la pared y una ganancia por canal: `0` los texels de la propia pared, `1` la mitad por lado. Lo que pide una cosa mate; `--lobes` entonces no se lee |
| `--show` | ruta | — | un EXR que la pared ajustada muestra después, los texels de la pared por una potencia de dos por lado (una pared de 8 x 4 muestra 128 x 64): las lecturas o el grid quedan como se ajustaron y leen una imagen más fina. Se mide como las demás, y con `-o` se escribe como `<prefijo>_shown0..2_picture.exr` y `_wall.exr` |
| `--frozen` | flag | off | cada lectura es la media de toda la pared y solo se aprende su ganancia: la pared como una luz sin imagen, para comparar |
| `-o`, `--output` | prefijo | — | `<prefijo>_<imagen>_picture.exr` y `_wall.exr` para ramp, checker, random, window y white, desde la primera cámara held-out |

Las medidas son las de `athenea compare`, sobre cinco imágenes completas que el fit nunca
ve (una rampa, un checker, un color por texel, una ventana brillante en una pared oscura, todo
blanco) desde las cámaras held-out, contra las nubes sumadas por los texels de cada
imagen. Las últimas líneas son el tiempo de un frame con y sin la pared,
y el de un frame quieto: la cámara y la nube como estaban y la pared mostrando
otra imagen, que conserva la geometría del frame anterior y rehace solo colores y
blend, comparado con el frame de esa misma imagen dibujado entero.

### 2.3.4 `athenea neural-scene` — una escena guardada por piezas

Una escena demasiado grande para convertirla, ajustarla o tenerla como una sola nube se guarda como un directorio,
`<nombre>.aths` (§8.2): un manifest y, por pieza, sus gaussianas y la apariencia
neural con la que se dibujan. Una pieza es lo que se convierte y se ajusta por
su cuenta -- una planta de un edificio, un objeto -- y cada una tiene su propia red.

```sh
athenea mesh2splat stage.usda --prim /root/floor1 -o floor1_gs.usda --no-camera
athenea neural-scene add    hall.aths --id floor1 --cloud floor1_gs.usda --prim /root/floor1 --up-axis Z
athenea neural-scene camera hall.aths --name door --eye 8 0 1.7 --target 0 0 2
athenea neural-scene fit    hall.aths --steps 3000
athenea neural-scene render hall.aths --camera door --compare -o door.exr
athenea neural-scene info   hall.aths

# Un render cuyo resultado es la escena, encontrada mientras reúne caminos; la imagen se hace de ella después.
athenea neural-scene shoot  stage.usda hall.aths --views views.txt --focal 20 --samples 64 --denoise
athenea neural-scene render hall.aths --camera door -o door.png          # o .exr
athenea neural-scene render hall.aths --camera door --at 0 -o then.png   # la escena como estaba
```

| Subcomando | Qué hace |
|---|---|
| `add` | copia la nube de una pieza dentro de la escena (`pieces/<id>.<ext>`), la sube una vez para contarla y tomar sus bounds, y escribe el manifest; una pieza con el mismo `--id` se sustituye, su revision sube y su apariencia se descarta, por ser de otras gaussianas. El primer `add` crea el directorio |
| `camera` | una vista con nombre en el manifest; una con el mismo nombre se sustituye |
| `fit` | los harmonics de cada pieza aprendidos por un latente por gaussiana y la red propia de la pieza, una pieza cada vez; escribe `pieces/<id>.athn` y las medidas en el manifest tras cada pieza, así que un fit que se para conserva las piezas que terminó |
| `render` | todas las piezas desde una vista, cada una con su apariencia donde la tiene y con sus harmonics donde no. Imprime cuánto de la vista tiene la escena -- la cobertura media del frame -- y el `A` del EXR es esa cobertura por píxel: la máscara de lo que se conoce desde ahí, para quien renderice o rellene el resto |
| `look` | la mirada de una cámara en un tiempo: se dibuja el frame y cada gaussiana que le dio algo queda encontrada, sellada con esta mirada donde ninguna la había visto. Escribe `pieces/<id>.seen` y la mirada en los `snapshots` del manifest, e imprime cuántas gaussianas vio, cuántas eran nuevas y cuántas tenía ya la escena |
| `discover` | una pieza hecha de un frame: el frame de cualquier renderer por una cámara conocida, un EXR con `R`, `G`, `B` y su profundidad de vista como `Z`, se convierte en una gaussiana por píxel -- un disco en la superficie del píxel, en el mundo -- por cada píxel cuya superficie la escena, dibujada por esa cámara, no tiene ya. Escribe `pieces/<id>.ply`, la pieza (`origin` discovered) y el snapshot de la mirada; imprime cuánto del frame tenía ya la escena, y la escena por esa cámara contra el frame. Crea la escena donde no la hay. Antes de añadir, retira: donde la escena tiene una superficie más cerca que la del frame, o algo donde el frame no tiene nada, lo que había allí se ha movido o se ha ido, y las gaussianas cuyo peso cayó en su mayor parte sobre esos píxeles quedan selladas como idas (`pieces/<id>.gone`) y ya no las dibuja nada |
| `shoot` | un stage USD path traced hacia dentro de la escena: el resultado del render son gaussianas y ninguna imagen. Cada una es tan ancha como su píxel a su distancia (al doble de lejos, el doble de ancha), un disco en su superficie estirado solo en la dirección en que la superficie se aparta del ojo, y del material de su superficie: algo opaco lleva su fracción de metal y su roughness; una superficie que deja pasar la vista da dos gaussianas -- el vidrio mismo, de poca opacidad, con su transmisión, su índice y si es una lámina, y donde acaba el rayo refractado lo que se ve a través, con la luz que envió antes de que el vidrio la tiñera. Se encuentran mientras el render reúne caminos: una primera mirada en celdas de `--coarsest` píxeles de lado en cuanto cada celda tiene los caminos con los que acabará un píxel, luego miradas más finas, cada una retirando las gaussianas más gruesas que sustituye; la última espera a que el frame esté entero (y con denoise). Cada mirada es un snapshot con su tiempo, sus caminos y su huella, y sus gaussianas una pieza `pieces/<vista>_t<tiempo>_f<huella>_s<snapshot>.athg`. Lo que una mirada ve a través -- algo que se movió -- queda sellado como ido igual que en `discover`, y nada sale del disco. Cada frame se almacena según se hace (`frames[]` en el manifest), y al terminar una vista la escena sale del dispositivo: la siguiente vista vuelve a leer solo las piezas cuyas cajas toca su frustum, así que una escena no está limitada por la memoria del dispositivo, que además tiene cargado el stage |
| `info` | el manifest, leído; sin dispositivo |

| Opción | De | Valor | Por defecto | Notas |
|---|---|---|---|---|
| `--id` | `add` | nombre | — | letras, dígitos, `_` y `-`: es el nombre de un fichero |
| `--cloud` | `add` | ruta | — | un ParticleField de USD (`athenea mesh2splat`) o un `.ply`, `.splat`, `.spz`, `.sog` |
| `--prim` | `add` | ruta | — | de dónde salió la pieza en el stage de origen; se anota, no se lee |
| `--name`, `--up-axis`, `--meters-per-unit`, `--source`, `--licence` | `add` | | el nombre del directorio, `Y`, `1`, —, — | los de la escena; cualquier `add` puede fijarlos |
| `--origin` | `add` | `converted`, `discovered` o `generated` | `converted` | de dónde salió la pieza: geometría convertida entera, lo que vio una cámara, o lo que rellenó otro sistema |
| `--found-by`, `--time` | `add` | nombre, número | —, — | la cámara que encontró la pieza y cuándo; con `--time` la pieza entra en el snapshot de la escena de ese tiempo y esa cámara |
| `--name`, `--eye`, `--target`, `--up`, `--focal` | `camera` | | —, —, —, el up de la escena, `35` | |
| `--piece` | `fit` | id | todas las piezas | repetible |
| `--steps` | `fit` | entero | `3000` | por pieza |
| `--size` | `fit`, `render` | `WIDTHxHEIGHT` | `960x540` | |
| `--held-out` | `fit` | entero | `8` | cámaras desde las que se mide una pieza y nunca se ajusta |
| `--rate-decay` | `fit` | número | `0.1` | |
| `--view-smooth`, `--split` | `fit` | | `0`, off | como los de `athenea neural-fit` (§2.3.2) |
| `--camera` | `render` | nombre | la primera del manifest | o `--eye` y `--target` (con `--up`, `--focal`) para una vista propia |
| `--harmonics` | `render` | flag | off | todas las piezas con sus harmonics |
| `--views` | `render` | ruta | — | un recorrido: un fichero de vistas, `NAME EX EY EZ TX TY TZ [UX UY UZ]` por línea (el de `athenea stage --cameras`), cada una escrita como `<-o>_NAME.exr` con la escena cargada una vez; `--focal` es la de todas. Imprime el tiempo de un frame, el menor y el medio, con el reloj esperando al dispositivo. Sin `-o` no se escribe nada y el recorrido solo se cronometra: un frame de 3840 x 2160 son unos 60 MB, y un recorrido de ellos llena un disco |
| `--discovered` | `render` | flag | off | solo las gaussianas que han encontrado las miradas: la escena tal como se conoce. Una pieza de la que no se ha tomado ninguna mirada no se dibuja |
| `--up-to` | `render` | entero | `0` | `--discovered`: solo las miradas hasta esta, contadas desde uno en el orden de `snapshots`; `0` son todas |
| `--at`, `--snapshot` | `render` | tiempo, entero | —, — | la escena como estaba: en un tiempo, el último snapshot tomado hasta entonces; o un snapshot del manifest, contado desde uno. Las piezas que habían entrado hasta él, menos las gaussianas que se habían ido hasta él. Sin ninguno, la escena como está ahora |
| `--frame` | `render` | entero | — | un frame que la escena almacenó (`shoot`), contado desde uno: se dibuja de nuevo por su cámara, a su tamaño, con la escena como estaba cuando entró su última mirada. En lugar de `--camera`, `--size` y `--snapshot` |
| `--exposure` | `render` | stops | `0` | `-o` con un `.png`: sobre la luz de la escena, antes del view transform estándar |
| `stage`, `scene` | `shoot` | rutas | — | el stage USD, y el directorio de la escena: se crea, o se le añade |
| `--views`; `--eye`, `--target`, `--up`, `--camera` | `shoot` | | — | las vistas, un fichero de `NAME EX EY EZ TX TY TZ [UX UY UZ]` por línea; o una vista y el nombre con el que se guarda (`camera`) |
| `--focal`, `--size` | `shoot` | mm, `WIDTHxHEIGHT` | `35`, `960x540` | sobre una apertura de 24.576 mm, como la de `athenea stage` |
| `--time`, `--times` | `shoot` | time codes | `0` | el tiempo del stage; varios se disparan uno tras otro por todas las vistas, cada mirada encontrando lo que dejó la anterior y retirando lo que se movió |
| `--time-step` | `shoot` | time codes | `0` | `--views`: cada vista del fichero es esto más tarde que la anterior, desde `--time`: una secuencia, un frame por línea, con la cámara y el stage moviéndose |
| `--samples`, `--pass-samples`, `--bounces`, `--denoise`, `--ray-margin` | `shoot` | | `64`, `1`, `4`, apagado, el de `athenea:rayMargin` | los del path tracer: caminos por píxel en total y por pasada, rebotes, si el frame entero pasa por denoise antes de su última mirada, y cuánto se separa un rayo de una superficie (§3) |
| `--coarsest` | `shoot` | píxeles | `8` | el lado de las celdas de la primera mirada, una potencia de dos; las miradas siguen a la mitad hasta un píxel. Una mirada en celdas de `f` espera a `samples / f²` caminos por píxel, así que las gaussianas de cada mirada reúnen los mismos caminos. `1`: una sola mirada, cuando el frame está entero |
| `--spread`, `--tolerance`, `--refine`, `--repaint` | `shoot` | | `1`, `0.06`, `1.5`, `0` | como los de `discover`, el ancho de un disco en celdas de la mirada |
| `--turned` | `shoot` | 0..1 | `0.5` | lo mínimo que se toma como orientación de un disco hacia el ojo: un disco se estira en la dirección en que su superficie se aparta del ojo por uno entre esa orientación, y no más que uno entre este valor |
| `--glass-opacity` | `shoot` | 0..1 | `0.05` | lo mínimo que un vidrio cubre de lo que tiene detrás, por claro que sea. Un vidrio cubre lo que su canal más oscuro quita a la luz que lo atraviesa, y se colorea con lo que los otros conservan por encima de eso, de modo que mezclado sobre la gaussiana de detrás devuelve el píxel de la mirada |
| `--name`, `--up-axis`, `--meters-per-unit`, `--source`, `--licence` | `shoot` | | el stem del directorio, el del stage, `1`, el stage, — | los de la escena |
| `--camera`, `--time` | `look` | nombre, número | —, — | una vista del manifest y cuándo; el par es el snapshot de la mirada, y una mirada repetida es la misma |
| `--views`, `--frames` | `discover` | ruta, directorio | —, junto al fichero | varias miradas en una ejecución, con la escena en el dispositivo de una a la siguiente: un fichero de vistas, `NAME EX EY EZ TX TY TZ [UX UY UZ]` por línea (el de `athenea stage --cameras`), cada una con su frame `NAME.exr` y convertida en la pieza `look_NAME`, en tiempos uno tras otro desde `--time` (0 donde no se da). En lugar de `--frame`, `--id` y `--camera` |
| `--frame` | `discover` | ruta | — | el frame: un EXR con `R`, `G`, `B` (luz lineal) y `Z`, la profundidad a lo largo de la vista; su tamaño es el de la mirada |
| `--id`, `--camera`, `--time` | `discover` | | —, —, — | el nombre de la pieza nueva, la vista por la que se dibujó el frame y cuándo. La vista es una del manifest, o `--eye`, `--target`, `--up`, `--focal` guardados con ese nombre; la focal es sobre una apertura de 24,576 mm, como la de `athenea stage` |
| `--tolerance` | `discover` | fracción | `0.06` | la superficie de un píxel ya está donde la profundidad de la escena por esa cámara queda dentro de esta fracción de la del frame |
| `--repaint` | `discover` | fracción | `0` | una superficie que la escena tiene a la profundidad del frame y cuyo color se aparta del del frame en más de esta fracción del más brillante de los dos se retira y se vuelve a crear desde el frame; `0` conserva una superficie solo por su profundidad. `0.3` queda por encima de lo que una escena descubierta se aparta de su propio frame |
| `--refine` | `discover` | razón | `1.5` | una superficie que la escena tiene con gaussianas más de tantas veces más anchas que el disco que esta mirada crearía ahí -- encontrada de lejos, vista ahora de cerca -- se retira y se vuelve a crear, más fina; `0` conserva una superficie por gruesa que sea |
| `--spread` | `discover` | píxeles | `1` | la desviación típica de una gaussiana, en píxeles a su propia profundidad; la mitad en el borde de una superficie. Más ancha cierra los huecos entre gaussianas, pinta más allá de las siluetas y queda más blanda desde una cámara que no tomó ninguna mirada: `1` es donde las vistas que ninguna mirada tomó quedan más cerca, `1.3` donde el propio frame de una mirada vuelve más parecido |
| `--name`, `--up-axis`, `--meters-per-unit`, `--source`, `--licence` | `discover` | | | los de la escena, como los de `add` |
| `--threshold` | `look` | píxeles | `0.05` | cuánto da una gaussiana al frame para contar como vista: su peso de blend sumado sobre sus píxeles |
| `--compare` | `render` | flag | off | de las dos maneras: los latentes contra los harmonics (las medidas de `athenea compare`) y el tiempo de un frame con cada uno, el menor de doce |
| `-o`, `--output` | `render` | ruta | — | un EXR (lineal, con su profundidad), o un `.png`: la imagen, por el view transform estándar, para un display sRGB; con `--compare`, `<stem>_harmonics.exr` y `<stem>_latents.exr`; con `--views`, un prefijo, y uno acabado en `.png` escribe `<stem>_NAME.png` |

Una pieza se ajusta desde dentro de la escena: el ojo de cada cámara en algún punto de la
caja de la escena, una décima hacia dentro de sus paredes, mirando a algún punto de la caja
de la pieza con una lente de 20 mm. Desde ahí se ve un edificio; un objeto sobre una
mesa se ajusta mejor con `athenea neural-fit`, cuyas cámaras se ponen a su alrededor.

### 2.4 `athenea convert` — un fichero de splats a USD, o a un `.athc`

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `input` | ruta, obligatoria | — | `.ply`, `.splat`, `.spz`, `.sog`, `meta.json`, `.asg`; o un ParticleField de USD (`.usd`, `.usda`, `.usdc`) a un `.athc` |
| `--prim` | ruta de prim | el primero de la escena | una entrada USD: el ParticleField que se lee |
| `output` | ruta, obligatoria | — | `.usda`, `.usdc`, `.usd` o `.athc` |
| `--chunk-splats` | entero | `65536` | `.athc`: splats por chunk |
| `--max-group-fraction` | número | `0.5` | `.athc`: grupos por splat que puede tener el nivel fundido más fino |
| `--merged-opacity` | 0.01..5 | `5` | `.athc`: la opacidad máxima de una gaussiana fundida; por encima de 1 se dibuja con el núcleo más duro |
| `--degree` | 0..3 | `3` | armónicos que se conservan |
| `--rotate-x` | número | `0` | grados; las nubes de COLMAP quieren `180` |
| `--no-camera` | flag | apagado | no añadir `/World/Camera` a la escena escrita |

Un `.athc` de salida rechaza un `--rotate-x` distinto de cero: el contenedor
guarda la nube como está, y el giro pertenece al prim que lo referencia.

Un ParticleField de USD pasa a un `.athc` con todo lo que lleva -- su material
y sus capas, su transfer y sus direcciones abiertas, su curvatura (empaquetada,
bit 6), sus normales de sombreado, su emisión y sus ids de Cryptomatte -- en su
propio espacio, en el tiempo por defecto. No se convierte nada otra vez: es la
misma nube con sus niveles de detalle. Uno que lleva un esqueleto se rechaza.
Lo que queda en el prim (`relight`, `ior`) se escribe en el prim que nombra el
asset. Un árbol de fusiones (`primvars:athenea:splat:lodTree`, `athenea
decimate --tree`) da solo sus hojas -- las gaussianas cuyo `lodSize` es 1,
compactadas en el dispositivo --, la nube de la que se construyó el árbol; sus
fusiones se dibujarían encima de ellas.

Un `.asg` (gaussianas surfel ancladas, el bitstream v1 de 3Dspace) se
decodifica en el dispositivo, un hilo por gaussiana, y se escribe con todo lo
que lleva: el material de la tabla de su anchor (metallic, transmisión, pared
fina), la rugosidad, la normal de sombreado, los ids de Cryptomatte, los
bloques directo e indirecto del transfer y las direcciones abiertas; `relight`
se pone donde lleva transfer. Las gaussianas se escriben en el orden del
fichero, agrupadas por anchor. Lo que la escena llevaba junto a ellas sale de
la cabecera (revisión 2): su `metersPerUnit`, el `ior`, si es lineal y
reiluminada, y el manifiesto de Cryptomatte; un fichero sin ellos se escribe en
metros, lineal, reiluminado donde lleva transfer, sin índice y sin
manifiesto. La línea del log
`asg: N gaussians from A anchors decoded in X ms` es solo la decodificación,
medida en su segunda ejecución.

Una escena escrita aquí dice `metersPerUnit = 1`: ninguno de los formatos de
entrada registra una unidad, y la escala de una captura se toma como metros.
Una nube en otra unidad se escala donde se referencia, o se edita el
`metersPerUnit` de la escena.

```sh
athenea convert capture.ply scene.usda
athenea convert capture.ply capture.athc --chunk-splats 131072
athenea convert car_tx.usdc car_tx.athc
athenea convert pawn.asg pawn.usdc --no-camera
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
| `--ratio` | 1..10000 | apagado | fundir por pares en vez de cortar el octree, hasta tener estas veces menos gaussianas; entonces no se leen las cuatro tolerancias de arriba |
| `--edge` | 0..2 | `0.15` | `--ratio`: una diferencia de color, codificada en sRGB, que nunca se funde |
| `--max-angle` | grados, 0..180 | `60` | `--ratio`: normales más separadas no se funden nunca |
| `--merge-growth` | factor | `0`: ninguno | `--ratio`: ninguna fusión más ancha (la raíz de la traza de su covarianza) que este factor por la más ancha de las dos que fusiona; la proporción es entonces lo más lejos que llega |
| `--max-aspect` | 1..1000 | `1000` (apagado) | `--ratio`: se rechaza una fusión cuyo plano sale más largo que esto frente a su anchura (o que 1.25 veces lo que ya eran sus miembros) -- una cadena de discos a lo largo de un embellecedor se fundiría en una aguja que emborrona |
| `--merge-glass` | flag | apagado | `--ratio`: las gaussianas que transmiten (vidrio) también se funden entre sí; apagado, el vidrio se conserva tal cual |
| `--no-surfels` | flag | apagado | `--ratio`: una fusión plana sigue siendo una gaussiana fina en vez de un surfel |
| `--surfel-flatness` | 0..1 | `0.12` | `--ratio`: una fusión de discos cuyo eje más fino está por debajo de esta fracción del mediano se escribe como surfel |
| `--highlight-cone` | 0..1000 | `1` | `--ratio`: en una superficie brillante (alpha GGX por debajo de 0.1), normales más separadas que esto por sqrt(alpha) radianes no se funden nunca -- la fusión movería el brillo; grande deja decidir solo a la proporción |
| `--tree` | ruta | — | `--ratio`: además, el árbol de fusiones entero aquí: las hojas en orden de profundidad y luego cada fusión, con `primvars:athenea:splat:lodParent`, `lodCost`, `lodStart` y `lodSize` (su tramo de hojas) y la constante `lodTree` |
| `--camera` | ruta de prim | `/World/Camera` donde la hay | `--ratio`: la cámara para la que se hace la nube; una fusión de menos de medio píxel en ella no cuesta nada de su forma |
| `--camera-pixels` | 1..65536 | `1920` | `--ratio`: píxeles de ancho de la imagen de esa cámara |
| `--max-cost` | > 0 | `0` (apagado) | `--ratio`: ninguna fusión cuesta más que esto; `--ratio` es entonces lo más lejos que llega, y donde cada par que queda cuesta más se para antes (en las piezas del Corvette 0.25 dejó una puerta a 6x y una rejilla a 2x) |
| `--max-pixels` | > 0 | `0` (apagado) | `--ratio`: ninguna fusión más ancha que estos píxeles en la pantalla de `--camera`: una así no sostiene un borde, y sus colas sobresalen del panel como pelillos. En el Corvette (horneado a una gaussiana por píxel de su cámara) 3 dejó 2.14M de los 2.90M vistos, a 46 dB de la nube completa y sin pelillos; sin él, los 1.07M de 5x los tenían |
| `--merge-opacity` | `mean`, `mass`, `mass-tight`, `falloff` | `mean` | `--ratio`: la opacidad de una fusión: la de sus miembros, ponderada por área, con la fusión ensanchada tanto como se solapaban; o su masa entre esa área ensanchada; o su masa entre el área que dan sus momentos, sin ensanchar; o esa masa sin tope hasta 5 -- el kernel más duro de Spark, donde una opacidad D mayor que 1 se dibuja como 1 - (1 - g)^A, A = exp((D² - 1)/e), una meseta tan sólida como lo eran los miembros. Medido en las piezas del Corvette: `mass` es `mean` en superficies opacas, `mass-tight` deja huecos; en el coche entero, ver `--hardness` |
| `--prune-share` | 0 a 1 | `0.5` | `mix`: la parte del exceso que se descarta antes de las fusiones |
| `--prune-score` | `area` o `edge` | `area` | `prune`, `mix`: por qué se conserva una gaussiana -- opacidad x el área de sus dos ejes más anchos, u opacidad x la raíz de esa área x su eje más ancho entre el intermedio, para que las finas (un borde, la barba de una pluma) no sean las primeras en irse |
| `--merge-aspect` | proporción | `1000`: ninguna | `pairs`, `mix`: ninguna fusión más larga que esto respecto a su anchura (o 1,25 veces la de sus miembros) |
| `--hardness` | flag | apagado | `--ratio`: una fusión escrita como surfel se dibuja exp(-(\|u\|^p + \|v\|^p)/2) en sus propios ejes, con p a partir de la curtosis de sus miembros (una zona uniforme, una caja; una concentrada, la gaussiana), escrito como `primvars:athenea:splat:hardness`; conserva sus momentos, sin ensanchar, así que un lector que no conoce p dibuja la gaussiana que dan. Con `--merge-opacity falloff --max-pixels 4` el Corvette quedó en 1.85M de 5.38M sin pelillos, a 42.8 / 44.7 / 46.7 dB (día frente, día detrás, noche), donde las fusiones gaussianas necesitaban 3 px y 2.14M; a 5 px (1.24M) el paragolpes delantero se deshilacha |
| `--max-hardness` | 2 a 32 | `16` | `--hardness`: la p más alta |
| `--no-light-bands` | flag | apagado | `--ratio`: fundir a través de un cambio de cuatro veces en la luz de un grupo de lámparas (`primvars:athenea:splat:lightLayer:<grupo>`); apagado, una gaussiana iluminada y una sin iluminar se mantienen separadas, para que un disco fundido no extienda un faro de noche |
| `--drop-hidden` | flag | apagado | primero dejar fuera lo que ninguna cámara alrededor de la nube ve: la nube dibujada por raster desde `--hidden-views` direcciones, cada una a 2.5, 1.6 y 1.15 veces el radio de su esfera envolvente, 90 grados de ancho, y una gaussiana que dio a todas juntas menos de `--hidden-pixels` de píxel se quita. Sin `--ratio`, solo eso |
| `--hidden-views` | 1.. | `256` | `--drop-hidden`: direcciones, una espiral de Fibonacci sobre la esfera |
| `--hidden-size` | 16.. | `1024` | `--drop-hidden`: píxeles de ancho de cada vista, cuadrada |
| `--hidden-pixels` | >= 0 | `0.02` | `--drop-hidden`: lo que una gaussiana tiene que dar a todas las vistas juntas para quedarse, en píxeles (transmitancia por alpha sobre sus píxeles) |
| `--hidden-below` | grados, -90..89 | `-90` (todo alrededor) | `--drop-hidden`: la elevación más baja desde la que se toma una vista, sobre el horizonte de la escena; `-10` para un coche en una carretera, que nunca se ve desde abajo |
| `--prune-error` | > 1 | apagado | primero estas veces menos quitando, en `--prune-rounds` rondas, lo que deja el menor error cuadrático en una órbita de `--prune-views` direcciones a `--prune-size` píxeles (la puntuación de GaussianPOP). Medido en las piezas del Corvette: huecos en toda superficie opaca, peor que fundir salvo en una insignia; se mantiene para nubes que no son superficies |
| `--prune-rounds`, `--prune-views`, `--prune-size` | 1.., 1.., 16.. | `8`, `64`, `512` | rondas, direcciones y píxeles de `--prune-error` |
| `--passes` | 1.. | `60` | `--ratio`: como mucho estas pasadas por pares; un árbol de muchas piezas pequeñas necesita más para llegar a sus raíces (el Corvette entero: 4000) |
| `--tree-coverage` | flag | apagado | la entrada es un árbol de merges: la opacidad de cada merge pasa a ser lo que sus hojas cubren de verdad, compuestas en su plano (`primvars:athenea:splat:coverage`), y su derrame, la fracción de lo que pinta que sus hojas no pintan (`primvars:athenea:splat:lodSpill`, 0 en una hoja); se escribe el árbol entero (con `--ratio 1`) |
| `--tree-pages` | gaussianas | apagado | la entrada es un árbol de merges: sus páginas por error de estas gaussianas (`lodPage`, `lodPageInfo`, `lodPageBounds`); se escribe el árbol entero |
| `--cut-error` | píxeles | apagado | la entrada es un árbol de merges: se queda el corte cuyo error, proyectado en la pantalla de `--camera`, no pasa de esto mientras el de su padre sí |
| `--cut-bounds`, `--cut-guard` | flag, > 0 | apagado, `4` | `--cut-error`: el error de un nodo acota también la dispersión de color y el cono de normales de sus hojas; un merge más allá de las guardas (un pliegue, un borde de color) cuenta estas veces su error |
| `--cut-cull-back` | flag | apagado | `--cut-bounds`: deja fuera los nodos cuyo cono de normales entero mira en sentido contrario a `--camera` en una pieza cerrada (`primvars:athenea:splat:closed`; nunca el cristal) |
| `--cut-spill`, `--cut-spill-floor` | >= 0, 0..1 | `0`, `0` | `--cut-error`: el derrame de un merge por encima del umbral, por este peso, sumado a su error. En la parrilla del Corvette un peso de 2 no tapó sus huecos más que las hojas pero guardó un 82-99 % más de nodos, y ningún umbral separa los merges que tapan huecos de los de un panel: apagado |
| `--occlusion-mask` | entradas | apagado | la entrada es un árbol de merges con bits de sombra de 16 x 16: la máscara de occlusion de cada nodo como índice en un codebook de estas entradas; se escribe el árbol entero |
| `--occlusion-group` | rutas | -- | `--occlusion-mask`: más árboles que comparten un codebook con la entrada, cada uno escrito junto a la salida con su propio nombre; un árbol cuya salida es su misma ruta se escribe junto a él y se renombra encima |
| `--cut-occlusion` | flag | apagado | `--cut-error`: deja fuera los nodos cuya máscara de occlusion está cerrada hacia `--camera` |
| `--mark-closed` | ruta | -- | la entrada es un árbol de merges: un fichero de rutas de prim, una por línea, cuyos nodos son cascos cerrados (`primvars:athenea:splat:closed`, por `cryptoObject`) |
| `--mark-material-roughness` | ruta | -- | la entrada es un árbol de merges: un fichero de líneas `ruta-de-prim rugosidad`; cada nodo recibe la rugosidad del material de su pieza (`primvars:athenea:splat:materialRoughness`, por `cryptoObject`; -1 para una pieza no nombrada) |

**Por pares** (`--ratio`). En cada pasada, cada gaussiana -- o fusión hasta
entonces -- pesa una fusión con sus 16 vecinas a lo largo de dos órdenes de
Morton de los centros, y los pares que se han elegido mutuamente se funden, los
más baratos primero, como mucho la mitad de lo que queda por pasada, hasta
llegar a la cuenta o hasta que nada más pueda fundirse (el log dice cuál, y
cuántas pasadas). El coste: cuánto más ancha es la fusión que lo que sustituye,
sin escala y ponderado por lo ancha que es en la pantalla de la cámara (nada
por debajo de medio píxel, todo a partir de dos); cuánto más gruesa que sus
discos; el giro de las normales, más en un espejo (mueve el brillo); el color,
codificado en sRGB; metallic, roughness y transmisión. Nunca se funde: una
clave (ids, articulaciones, láminas, paredes finas, metales Schlick), vidrio
con lo que no lo es, un espejo (roughness por debajo de 0.1) con lo que no lo
es, normales más allá de `--max-angle`, un color más allá de `--edge`. Una
fusión de discos plana hasta 0.12 de su anchura se escribe como surfel (una
escala exactamente 0, `primvars:athenea:splat:surfel`); su opacidad es su masa
entre su área, topada en 0.99, y la masa entre el área sin tope se escribe como
`primvars:athenea:splat:coverage`. Con `--hardness`, la p de ese surfel se
escribe como `primvars:athenea:splat:hardness` (2, una gaussiana), y con
`--merge-opacity falloff` su opacidad puede pasar de 1 (hasta 5): el propio
kernel más duro de Spark, que este raster dibuja como Spark y cualquier otro
lector topa en 1. Cada array se funde sobre los registros que
representa una gaussiana conservada, como abajo; `transferZonal` y
`curvature`, que están en el marco propio de cada gaussiana, a través del marco
de cada una al mundo y al de la conservada.

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
athenea decimate car_tx.usdc car_tx_10x.usdc --ratio 10
athenea decimate car_tx.usdc car_tx_fewer.usdc --drop-hidden --hidden-below -10 --ratio 2.7
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
| `--ray-margin` | float | el de `athenea:rayMargin` | rt: cuánto se separa un rayo de una superficie, como fracción de su distancia al origen (`athenea:rayMargin`) |
| `--denoise` | flag | apagado | rt: denoise cuando alcanza el total |
| `--default-lights` | flag | apagado | un dome y un sol en la capa de sesión, para una escena sin luces |
| `--shutter` | `ABRE:CIERRA` | sin poner | en frames; solo para una cámara hecha con `--eye` |
| `--variant` | `/World{set=valor}` ‹repetible› | ninguna | una selección de variante antes del primer frame |
| `--motion-buckets` | 1..8 | `4` | rt: rodajas del obturador |
| `--splat-override` | `PRIM=M,R,T[,R,G,B[,REPLACE]]` ‹repetible› | ninguno | metallic, roughness, transmission y un tinte para las gaussianas que vinieron de `PRIM` (una ruta del manifest Cryptomatte de la nube, `*` para todas); `-1` deja el de la gaussiana; `REPLACE` 1 hace de R,G,B el color en vez de un factor sobre él. Sólo una nube convertida con ids tiene manifest. Falla con un prim que ninguno nombra |
| `--light-state` | nombre | ninguno | el estado de luces de la escena cuyos niveles toman sus grupos de lámparas (`athenea:lightState`, §4.2); sin él cada grupo queda en su `initialOn` |
| `--light-level` | `GRUPO=NIVEL` ‹repetible› | ninguno | el nivel de un grupo de lámparas sobre el del estado (`athenea:lightLevel:GRUPO`): 0 apagado, 1 su radiancia, más para una lámpara que comparten dos funciones (un freno a 4 veces su piloto). Falla con un grupo que la escena no lleva |
| `--refine` | entero | `0` | niveles de subdivisión; 0 dibuja la malla de control |
| `--light-samples` | entero | `1` | muestras por luz y píxel |
| `--mesh-light-samples` | 0..64 | `0` | raster: muestras por píxel de las luces de malla; 0 ilumina por ellas en forma cerrada (`athenea:meshLightSamples`) |
| `--volume-filter` | `trilinear`, `nearest`, `stochastic` | `trilinear` | rt: la densidad de un volumen entre vóxeles (`athenea:volumeFilter`) |
| `--no-mesh-lights` | flag | luces de malla encendidas | las mallas emisoras no se agrupan en luces (`athenea:meshLights` apagado) |
| `--no-transfer-indirect` | flag | se suma | dibujar una nube con transfer sin su mitad rebotada |
| `--splat-display-blend` | flag | mezcla lineal | raster: los splats se mezclan como los mezcla un visor estándar, sumando valores sRGB (`athenea:splatDisplayBlend`) |
| `--splat-reflections` | flag | apagado | rt: una gaussiana refleja la nube a la que pertenece, no sólo el cielo |
| `--splat-shadows` | flag | apagado | rt: una nube relit se sombrea a sí misma, un rayo por splat |
| `--no-antialias` | flag | antialias encendido | |
| `--no-cloud-shadows` | flag | sombras de nube encendidas | |
| `--no-dome-prefilter` | flag | prefiltrado | raster: un domo ilumina una malla con una muestra por píxel y un rayo de sombra |
| `--screen-space-reflections` | flag | apagado | raster: `athenea:screenSpaceReflections` |
| `--mesh-antialias` | `0`, `4`, `8`, `16` | `4` | raster: `athenea:meshAntialias`, muestras por píxel de las que se resuelven los bordes de las mallas; 0 es una muestra por píxel |
| `--cloud-shadow-texels` | entero | `1024` | por lado, por luz |
| `--cloud-shadow-density` | número | `1.0` | multiplicador de la profundidad óptica de la nube |
| `--cloud-shadow-terms` | `0`, `1`, `3`, `5`, `7` | `0` | 1 es solo el total, el resto añaden pares de Fourier; 0 deja decidir a quien recibe |
| `--eye` | 3 números | ninguno | una cámara propia, en vez de una de la escena |
| `--target` | 3 números | ninguno | adónde mira esa cámara |
| `--up` | 3 números | `0 1 0` | |
| `--frame-all` | flag | apagado, e implícito si la escena no tiene cámara | encuadra todo: mallas, nubes, puntos y volúmenes (la caja de un volumen es la de las hojas de su grid, con cualquier técnica); se registran la caja y el tiempo que llevó (`framed (...) - (...) in N ms`) |
| `--focal` | número | `35` | mm, apertura 24.576 mm |
| `--fstop` | número | `0` | 0 es estenopeica |
| `--focus` | número | `0` | la profundidad enfocada, unidades de escena |
| `--near` | número | `0.1` | |
| `--far` | número | `100000` | |
| `-o`, `--output` | ruta | `out.exr` | RGBA en media más `Z` |
| `--render-settings` | ruta de prim | sin poner | renderiza los productos de ese prim y para |
| `--frames` | entero | `1` | repite, e informa del primero, la mediana y el más rápido |
| `--no-readback` | flag | apagado | con `--frames`: todos los frames salvo el último se dibujan y se esperan, sin leerlos de vuelta -- lo que paga un host que deja la imagen en el dispositivo; el último se lee para `-o` |
| `--drift` | número | `0` | con `--frames` y una cámara propia (`--eye`, `--frame-all`): unidades de escena que se mueve por su propio eje x en cada frame, a un lado y de vuelta, para medir una cámara en movimiento |
| `--cameras` | ruta | ninguna | un fichero de vistas, una por línea: `NOMBRE EX EY EZ TX TY TZ [UX UY UZ]` (`#` empieza un comentario; el vector arriba es `--up` donde la línea no da ninguno). El stage se carga una vez y cada vista se escribe en el directorio de `-o` como `NOMBRE.exr`, con la lente, el tamaño y la técnica de esta orden; `-o` es un directorio, o un fichero del que se toma su directorio |
| `--turntable` | entero | `0` | tantas vistas alrededor de `--target`, la primera desde `--eye`, cada una girada 360/N grados alrededor de `--up`, escritas en el directorio de `-o` como `f_0000.exr`, `f_0001.exr`, ... con una sola carga del stage. Imprime el tiempo de cada vista y, al final, el de la primera y la mediana de las demás |

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
| `--mesh-antialias` | `0`, `4`, `8`, `16` | `4` | raster: `athenea:meshAntialias`, muestras por píxel de las que se resuelven los bordes de las mallas; 0 es una muestra por píxel |
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
máquina o siguiendo a un maestro PTP. El subcomando existe solo en un build con
genlock (`ATHENEA_WITH_GENLOCK`, activo cuando `third_party/genlock` está descargado).

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

### 2.7.1 `athenea stream` — una secuencia de páginas a un stream en vivo

Reproduce las páginas SPZ que escribió `athenea flatten --range` en el stream
en vivo que nombra un stage (`AtheneaLiveSourceAPI`, §4.3) y dibuja el stage
mientras suena, en el mismo proceso: un hilo productor empuja las páginas de
cada frame como una época, con su tiempo PTP, y el frame dibuja la última
época cuyas páginas están todas en el dispositivo.

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `stage` | ruta, obligatoria | — | un stage con un ParticleField bajo `AtheneaLiveSourceAPI`; sus `chunks` al menos las páginas por frame de la secuencia, su `chunkSplats` al menos `--chunk-splats`, `degree` 3 |
| `pages` | directorio o `.athv`, obligatorio | — | contiene `stream.txt` y las páginas; o un vídeo que escribió `athenea video-encode` (§2.7.2), cada frame decodificado por el decodificador hardware al empujarlo |
| `--stream` | nombre | el que nombra el stage | |
| `--camera` | ruta de prim | la primera del stage | |
| `--size` | `ANCHOxALTO` | `960x540` | |
| `--rate` | épocas por segundo | la de la secuencia (`stream.txt`, los time codes por segundo del stage) | |
| `--loops` | entero | `1` | la secuencia otra vez, con las épocas siguiendo la cuenta |
| `--lockstep` | flag | por el reloj | cada frame empuja su época y dibuja hasta que el stage la muestra: cada época dibujada una vez |
| `--reference` | ruta de stage | ninguna | con `--lockstep`: se dibuja en el time code de cada frame y se compara con el frame en el dispositivo (p99, max, píxeles a más de 2 valores de código) |
| `--frame-producer` | flag | un hilo productor | por el reloj: el hilo del frame empuja las épocas que tocan antes de cada frame -- para una ejecución en prioridad de fondo, donde un hilo productor dormido despierta tarde y empuja a ráfagas |
| `--splat-display-blend` | flag | blend lineal | la nube en vivo mezclada como la mezcla un visor estándar, que es para lo que se ajustan los colores de flatten |
| `-o`, `--output` | directorio | ninguno | cada frame mostrado como `frame_NNNNN.exr` (y `reference_NNNNN.exr`), con `athenea:timecode`, `athenea:ptpTime` (segundos) y `athenea:epoch` |

Por el reloj imprime los frames dibujados, las épocas mostradas y saltadas,
con qué retraso se mostró una época respecto a cuándo tocaba, y si el ritmo
se sostuvo (cada época dibujada). El productor está en este proceso: las colas
con nombre son las del propio motor (`lod::PageQueue::named`).

Un `.athv` se reproduce en el mismo stage: un frame de vídeo es una época,
sus chunks las páginas, y los `chunks` del stage tienen que ser al menos los
del frame más grande del vídeo (`chunkSplats` 32768, `degree` 3). La imagen
decodificada se queda donde la escribió el decodificador; el dispositivo la
lee allí y escribe las gaussianas en el almacén. La ejecución imprime además
cuánto tardó el decodificador hardware por frame (mediana y p95) y cuántas
páginas se colocaron. Por ahora solo en macOS.

### 2.7.2 `athenea video-encode` — frames de gaussianas como vídeo hardware

Coloca las gaussianas de cada frame en planos en el dispositivo -- ordenadas
por código de Morton, cada atributo una tesela de luma de un mosaico,
cuantizado sobre el rango de su grupo de imágenes
(`shaders/athenea/scene/video_planes_encode.slang`) -- y hace que el
codificador HEVC hardware codifique el mosaico. Las posiciones tienen el
doble de `--bits`: la mitad alta va junto al vídeo, comprimida con deflate y
exacta, la baja por el códec. La croma se queda en gris medio, así que el
4:2:0 que acepta cualquier decodificador no pierde nada. El fichero es un
`.athv` (§8.2).

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `input` | directorio, o ficheros de splats, obligatorio | — | un directorio de páginas (`stream.txt`, todos sus frames; una página `f<frame>_c<página>.spz`, o `.ply` donde no hay `.spz`), o ficheros leídos como un frame |
| `-o`, `--output` | ruta, obligatoria | — | el `.athv`; se escribe con un nombre parcial y se renombra al completarse |
| `--bits` | `8` o `10` | `8` | HEVC Main o Main 10; una posición tiene 16 o 20 bits |
| `--range` | `PRIMERO:ÚLTIMO` | todos los frames | los frames de la secuencia por índice desde 0, ambos incluidos |
| `--quality` | 0 a 1 | `0.5` | la calidad constante del codificador: la escala propia de VideoToolbox; en NVENC un mismo QP para toda imagen, round(51 x (1 - calidad)) (0.5 es QP 26) |
| `--megabits` | Mbit/s | por calidad | una tasa media en su lugar (NVENC: tasa variable, con un pico del doble de la media) |
| `--keyint` | imágenes | el ritmo (un segundo) | entre fotogramas clave: donde puede empezar un reproductor |
| `--intra` | flag | no | cada imagen un fotograma clave |
| `--hi` | `side` o `video` | `side` | las partes altas de la posición junto al vídeo (exactas) o como planos propios (por el códec, y no exactas: `docs/decisions.md`, "Video planes") |
| `--order` | `frame`, `first` o `stable` | `frame` | cada frame ordenado por código de Morton, o el orden del primer frame mientras el número sea el mismo; `stable` es `first` para frames que listan las mismas gaussianas en el mismo orden (`flatten --keep-order`), con el mismo número en todos, y lo dice en el fichero (`stableOrder`): el índice i es una misma gaussiana de principio a fin |
| `--range-group` | frames | el intervalo entre claves | frames cuyos rangos de atributo son una sola unión: la de toda la secuencia mantiene en su paso cuantizado un valor que no se mueve |
| `--hold-boxes` | flag | no | con `--order stable`, `--position-block` y rangos por grupo: la caja de cada bloque es la unión sobre su grupo de rangos, para que sus pasos de posición no salten |
| `--position-target` | metros | `0`: no | con `--position-block`: precisión adaptativa -- cada bloque con sus propios bits altos (un byte tras las cajas del chunk): ninguno donde su caja no es más ancha que esto, si no los que hagan falta para que un paso no sea más ancho que esto entre 255 |
| `--geometry` | flag | no | posición, escala y rotación exactas junto al vídeo (posiciones de 16 bits, escalas logarítmicas de 10 bits sobre los rangos de la secuencia, la palabra de rotación del almacén), y los planos solo opacidad y color |
| `--geometry-key` | frames | `0`: todos los frames absolutos | con `--geometry`: frames de una clave de geometría a la siguiente; los frames intermedios envían la diferencia de cada gaussiana respecto al frame anterior (necesita `--order stable` y un `--range-group` de toda la secuencia) |
| `--geometry-stagger` | flag | no | con `--geometry-key`: sin frames clave completos; bloques de 256 gaussianas refrescados por turnos, cada uno una vez por intervalo |
| `--geometry-position-bits` | 10 a 16 | `16` | con `--geometry`: un eje de posición sobre el rango de la secuencia (los 0,23 m del gorrión: 12 bits son 56 um) |
| `--geometry-scale-bits` | 6 a 10 | `10` | con `--geometry`: una escala logarítmica sobre su rango |
| `--geometry-rotation-bits` | 5 a 10 | `10` | con `--geometry`: cada componente de la rotación en tres menores del almacén, redondeada (la palabra conserva 10 bits, los bajos a cero) |
| `--geometry-predict` | 1, 2 o 3 | `1` | con `--geometry-key`: el residuo respecto al frame anterior, respecto a su movimiento prolongado (2 b - c), o 3, movimiento: el movimiento rígido de cada tramo de 256 gaussianas ajustado desde lo que tiene el decodificador (se envía, 32 bytes por tramo), la rotación enviada como un giro en el marco propio de la gaussiana en pasos ensanchados según su forma (ninguno donde su extensión no puede mostrarlo), y claves que envían lo que el decodificador tendría, de modo que una clave no salta. 3 recorre el bucle dos veces antes de escribir para que su último frame lleve al primero |
| `--geometry-hold` | P,S,R, cada uno de 0 a 64 | desactivado | con `--geometry-key` y `--geometry-predict` 1 o 3: entre claves, un eje de posición a no más de P de sus pasos del valor que tiene el decodificador (con 3: de su predicción), una escala logarítmica a no más de S, o una rotación cuyas tres componentes estén cada una a no más de R (con el mismo índice) conserva el valor que se tiene y se envía como sin cambio. El formato del fichero no cambia: un decodificador no necesita ninguna opción. Un valor conservado nunca está a más de P, S o R pasos de la fuente |
| `--cells` | número > 0, unidades del mundo | ninguno | con `--order stable`: frames cuyas gaussianas no comparten identidad (ajustes independientes) dispuestos por espacio -- la caja de la secuencia (leída una vez sobre todos los frames) cortada en cubos de este lado, cada uno dueño de tantos huecos como el máximo de gaussianas que tiene en algún frame, las gaussianas de un frame en orden de Morton dentro de su cubo y los demás huecos del cubo muertos (sin extinción, sin opacidad, en el centro del cubo); un lado mayor que la caja es un solo cubo: cada frame en el orden de Morton de la caja de la secuencia, rellenado hasta su frame mayor. Deja predecir a la capa de geometría; medido en ajustes independientes no compensa (decisions.md, las explosiones) |
| `--slots` | entero > 0, huecos | ninguno | con `--order stable` (no con `--cells` ni `--gaussians`): cada frame dispuesto como este número de huecos, sus propias gaussianas primero en su orden y los huecos tras ellas muertos (una mota en el origen, ln sigma -10, como escribe `vdb2splat --cloud` un hueco muerto) -- para una secuencia cuyos huecos conservan sus gaussianas pero cuyos frames listan menos, como un ajuste en caliente cuyo pool aún se llena; un frame con más es un error |
| `--volume` | número > 0, densidad a extinción por unidad de longitud | ninguno | los splats son de un medio (`vdb2splat --cloud`, `ATHENEA_VOLUME_PLY`): bit 8 de los flags de la cabecera y esta escala de densidad en el offset 88, para que un reproductor dibuje cada splat con el alfa 1 − exp(−τ) de su profundidad óptica, σ = exp(`f_rest_0`) por ella. Necesita `--sh-degree 1` o más, que lleva `f_rest_0` |
| `--geometry-video` | flag | no | posición, escala logarítmica y rotación como planos de luma del vídeo, sin nada junto a él salvo los rangos del registro: un plano grueso cada una (8 bits sobre el rango de la secuencia) y, para un grupo plegado, dos planos finos (FORMAT.md §8e). La rotación va como x, y, z de su forma canónica (w >= 1/2 se reconstruye). Solo 8 bits; no con `--geometry`, `--position-block` ni `--hi video`; pensado con `--order stable`, un `--range-group` de toda la secuencia y un `--keyint` (las imágenes predichas mantienen quieto de frame a frame el error del códec en la geometría) |
| `--geometry-fold` | P,S,R, cada uno de 0 a 255 | `16,16,16` | con `--geometry-video`: el pliegue K de posición, escala logarítmica y rotación. Un atributo plegado se envía con 255 x 255 / K niveles (K 16: 4064, 60 um por nivel sobre los 0,245 m del gorrión) como dos ondas triangulares a un cuarto de periodo, que el plano grueso sitúa mientras su error de códec esté por debajo de K / 4 de sus códigos; 0 envía solo el plano grueso. Cada grupo plegado añade seis planos |
| `--ranges` | `group` o `frame` | `group` | el rango de cada atributo sobre un grupo de imágenes (de un fotograma clave al siguiente), o el de cada frame |
| `--sh-degree` | 0 a 3 | `3` | como mucho el de la fuente |
| `--columns` | teselas por fila | el mosaico más cuadrado | |
| `--max-picture` | `ANCHOxALTO` | 16384 de lado | la imagen más grande, para un decodificador que admite menos (el de un teléfono); un frame que necesita más va en porciones |
| `--position-block` | gaussianas | `0`: el rango del frame | una potencia de dos de 64 a 32768: cada tramo de tantas gaussianas del orden de Morton se cuantiza sobre su propia caja (seis halves al principio de los datos laterales de cada chunk) |
| `--position-high-bits` | 0 a `--bits` | `--bits` | los bits de una posición por encima de la muestra, enviados exactos junto al vídeo; 0 no envía ninguno (con `--position-block`, la caja es lo que la mantiene cerca) |
| `--chroma` | flag | no | colores como YCoCg: la Y de cada color un plano de luma, su Co y su Cg el Cb y el Cr de esa tesela, una muestra por bloque de 2 x 2 de cuatro gaussianas consecutivas en el orden de Morton; solo 8 bits |
| `--chroma-weight` | `mean` o `area` | `mean` | el croma de un bloque: la media simple de sus gaussianas, o ponderada por opacidad x el área de sus dos ejes mayores |
| `--reduce` | `pairs`, `prune`, `mix`, `cut`, `keep` | `pairs` | con `--gaussians`: fusionadas por pares; las que menos aportan (opacidad x el área de los dos ejes más anchos) descartadas; la mitad del exceso descartada y después fusionadas; o el corte del octree por error, con sus tolerancias escaladas hasta la cuenta. Lo que una fusión limitada deja por encima de la cuenta se descarta. `keep`: las de mayor puntuación en el primer frame, los mismos registros en todos (lo que necesita `--order stable`) |
| `--merge-growth` | factor | `0`: ninguno | `pairs`, `mix`: ninguna fusión más ancha (la raíz de la traza de su covarianza) que este factor por la más ancha de las dos |
| `--merge-opacity` | 0 a 3 | `0` | `pairs`, `mix`: 0 la media de sus miembros, la fusión ensanchada, por debajo de 0,99; 1 su masa sobre el área ensanchada; 2 su masa, sin ensanchar; 3 su masa, sin ensanchar, hasta 5 (dibujada con el núcleo más duro de Spark) |
| `--gaussians` | número | las del frame | unas tantas por frame: cada frame fusionado por pares donde fusionar pierde menos (la fusión NanoGS de la decimación, `docs/decisions.md`), nunca descartadas; sin surfels |
| `--rendition` | 0..65535 | `0` | el id que lleva cada registro, uno por rendición de una escalera (`athv_stream.py`) |
| `--rate` | frames por segundo | el de la secuencia, o 30 | se guarda en el fichero; `athenea stream` lo reproduce a ese ritmo |
| `--hevc` | ruta | ninguna | el vídeo solo como Annex B, para `ffprobe`; sus imágenes son planos, no una imagen |
| `--check` | flag | no | cada imagen se decodifica otra vez con el decodificador hardware, vuelve a gaussianas en el dispositivo y se mide contra su fuente. Necesita un decodificador hardware (VideoToolbox, NVDEC): donde no lo hay (sin driver de NVIDIA en Linux o Windows) se rechaza diciéndolo; allí se comprueba con `--units` |
| `--units` | `.athv` | ninguno | las unidades del vídeo tomadas de este fichero -- el codificador de otra máquina, los mismos frames y las mismas opciones -- en lugar del codificador de esta máquina; con `--check`, el flujo de ese codificador por el decodificador de esta máquina contra la misma fuente. Un fichero cuyas imágenes se disponen de otro modo (bits, colores, tiles) se rechaza |
| `--each-a-frame` | flag | no | los ficheros de splats dados son una secuencia, cada fichero un frame en el orden dado; sin él son las páginas de un frame |

**Tamaños.** Una imagen tiene como mucho 8192 x 8192 de área y 16384 de lado
(el codificador hardware rechazó 10240 x 9600). Un frame cuyos planos
necesitan más se corta en porciones -- tramos de su orden de Morton, una
imagen cada una, todas fotogramas clave -- y un frame sigue siendo una época.
Todas las imágenes tienen el tamaño de las del primer frame; un frame
necesita una tesela de 512 x 64 filas por cada 32768 gaussianas, y un plano
por atributo: 59 en grado 3 (62 con `--hi video`). Con `--max-picture` la imagen se limita
además a ese tamaño: 1920 x 1080 admite tres columnas de teselas, así que un
frame de 300847 gaussianas va en cuatro imágenes de 1536 x 960 en grado 0 (14
planos) y en cinco de 1536 x 1024 en grado 1 (23); el grado 3 (59 planos) no
cabe en 1080 filas. El nivel que el codificador escribe en el SPS sigue al
tamaño de la imagen.

**Qué imprime.** El mosaico y las porciones; MB por frame (el vídeo y los
datos laterales aparte) y Mbit/s a 24 y 30 por segundo; el tamaño de una
imagen clave frente al de una predicha. Con `--check`, por cada grupo de
atributos (posición, opacidad, escala logarítmica, rotación, color base,
resto) el error mayor y el medio en pasos de su propia cuantización (los de
una posición, de la de 16 o 20 bits), si todas las partes altas de la
posición volvieron exactas, y la decodificación hardware y el kernel de
planos -> gaussianas en ms (medianas, el kernel enviado y esperado).

**Dónde codifica.** macOS: VideoToolbox. Linux y Windows: NVENC, mediante la
biblioteca propia del driver de NVIDIA (`libnvidia-encode.so.1`,
`nvEncodeAPI64.dll`) abierta en tiempo de ejecución -- API de NVENC 12.2, un
driver 550.54.14 (Linux) o 551.76 (Windows) o posterior; sin él el
codificador se rechaza (`Unsupported`, con el motivo). En un dispositivo CUDA
(el backend por defecto allí) la sesión se abre sobre el propio contexto del
motor y toma los planos donde los dejaron los kernels, así que no se leen de
vuelta: la línea de cabecera dice `encoder hardware, from device memory`.
Ambos escriben HEVC Main (Main 10 con `--bits 10`), 4:2:0, rango completo en
la VUI, solo imágenes I y P.

**Dónde decodifica** (`--check`, `video-decode`, `stream`). macOS:
VideoToolbox. Linux y Windows: NVDEC, mediante la biblioteca propia del driver
(`libnvcuvid.so.1`, `nvcuvid.dll`) abierta en tiempo de ejecución, el mismo
driver que pide NVENC, sobre el propio contexto del dispositivo CUDA; la imagen
se copia en el dispositivo a las texturas de los planos, nunca a la CPU. Sin un
dispositivo CUDA (otro backend) el decodificador se abre pero sus frames no
pueden hacerse texturas, y eso se rechaza (`Unsupported`). Ambos rechazan un
flujo cuyo SPS dice rango de vídeo (16-235).

### 2.7.3 `athenea video-decode` — los frames de un vídeo, decodificados y cronometrados

Cada imagen de un `.athv` por el decodificador hardware y el kernel de planos
-> gaussianas (`video_planes_decode.slang`), a una nube en el dispositivo, sin
dibujar nada. Imprime, por imagen, las medianas de la lectura del registro, la
decodificación hardware (y su p95) y el kernel, enviado y esperado como
cronometra un kernel `ATHENEA_STAGES`.

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `input` | `.athv`, obligatorio | — | |
| `--frames` | imágenes | todas | |
| `--frame` | frame de origen, desde 0 | `0` | con `--dump-compare` o `--dump-state` |
| `--dump-compare` | ruta CSV | ninguna | las gaussianas de `--frame` según otro decodificador, una fila por gaussiana en el orden del frame (`i,x,y,z,opacity,lnsx,lnsy,lnsz,r,g,b,...`, con una línea de cabecera): la mayor diferencia y la media de cada grupo respecto a esta decodificación y las filas que pasan una tolerancia (1 mm, 0,02, 0,1, 0,02), y la opacidad y la mayor escala logarítmica de esta decodificación |
| `--dump-state` | ruta | ninguna | el estado de la capa de geometría en `--frame` tal como lo tiene este decodificador (el stream decodificado desde el frame 0): cuatro u32 little-endian por gaussiana en el orden del frame (las palabras de §8c), sin cabecera. Las palabras de otro decodificador (un reproductor web) se comparan con estas palabra por palabra. Un fichero sin capa de geometría se rechaza |
| `--luma-histogram` | flag | no | el plano 0 de cada imagen tal como lo dejó el decodificador: la muestra más baja y la más alta, y la proporción en 128, por debajo de 16 y por encima de 235 (una imagen de rango completo cubre 0-255) |

### 2.7.4 `athenea video-flicker` — cuánto se mueve el error de una decodificación

Dos directorios de `frame_NNNNN.exr` (`athenea stream -o`): frames dibujados
a partir de un vídeo y los mismos frames dibujados a partir de su fuente. Por
píxel, el mayor canal de |(video_k - fuente_k) - (video_k-1 - fuente_k-1)| en
valores de código sRGB -- el cambio del error de un frame al siguiente, que es
lo que se ve como parpadeo; un error que no se mueve no suma nada. Imprime la
media, el p99 y el p99,9 sobre todos los píxeles de todos los pares (un
kernel; el host lee su histograma).

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `video` | directorio, obligatorio | — | |
| `source` | directorio, obligatorio | — | los mismos números de frame y el mismo tamaño |
| `--first`, `--last` | números de frame | `1`, `61` | |
| `--rect` | `X0 Y0 X1 Y1` | toda la imagen | solo estos píxeles, fila de arriba 0, los extremos pasado el último (una parte del frame: una cola, un ala) |

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
| `--flatness` | número | `0.1` | el tercer tamaño como fracción del menor de los otros dos; `0` hace de cada gaussiana un surfel -- el disco de los otros dos, una escala exactamente 0, dibujado por el rayo de cada píxel (gaussian splatting 2D) -- y marca la nube con `primvars:athenea:splat:surfel` |
| `--opacity` | número, de 0 a 1 | `1.0` | cobertura: cuánto de lo que hay detrás cubre la superficie convertida, multiplicado por la opacidad propia del material. Toda opacidad es cobertura -- esta, la constante del material, el valor de un mapa, lo que conserva un vidrio -- y cada gaussiana toma lo que necesita una de las varias que hay sobre un punto, así que 0.5 cubre la mitad a cualquier tamaño |
| `--glass-opacity` | número, de 0 a 1 | `1` en un stage iluminado solo por cúpulas, `0.6` donde hay una lámpara o un sol | cobertura que conserva un sólido que transmite del todo. El fotograma de un transfer TX dibuja la imagen propia de la lente -- el cielo doblado por las dos caras, la lejana estimada por la curvatura, y el campo donde el paso está cerrado. Iluminado solo por cúpulas, la cara cercana cubre del todo: por debajo, la luz que dejaba pasar llegaba a las gaussianas de la cara lejana, que miran hacia atrás y no devuelven nada, y una bola transparente salía un 23 % oscura. Esa imagen sale solo de las cúpulas, así que bajo una lámpara lo que el resto deja pasar recto es la otra luz de la escena: a 1 una bola de vidrio bajo una lámpara perdía el suelo iluminado que hay detrás. Un vidrio de pared fina (y una opacidad de UsdPreviewSurface menor que uno en su modo `transparent` por defecto) cubre en cambio lo que la lámina refleja con su índice |
| `--opacity-cut` | número, de 0 a 1 | `0.5` | donde la opacidad de un material es un mapa sin umbral propio (el `opacity` de UsdPreviewSurface, el `opacity` de standard_surface, el `geometry_opacity` de OpenPBR, el `alpha` de glTF en BLEND): por debajo de esto no se escribe ninguna gaussiana; por encima, la superficie cubre lo que lee el mapa. El umbral propio del material (`opacityThreshold`, el `alpha_cutoff` de glTF en MASK) se usa en su lugar, y lo que conserva queda entero |
| `--max-cells` | entero | `262144` | celdas como mucho que recorre un triángulo |
| `--texture-size` | entero | `1024` | un mapa se lee no mayor que esto; 0 lo lee a su tamaño |
| `--no-textures` | flag | apagado | ignorar los mapas; los materiales se quedan con sus valores constantes |
| `--normal-map-turns` | flag | apagado | el mapa de normales gira la gaussiana, no solo su sombreado. La normal de sombreado se escribe en ambos casos (`primvars:athenea:splat:normal`) |
| `--edge-reach` | número, 0 a 8 | `1.5` | cuánto puede asomar una gaussiana por un borde donde su superficie termina (una arista abierta) o se pliega (dos triángulos que giran más de 40°), en sus propios tamaños: su tamaño en perpendicular a ese borde se limita a su distancia a él dividida por esto. Evita que la pestaña que un panel pliega hacia su junta asome sobre la pintura como una línea oscura, y que la pintura invada la junta. `0` no limita nada |
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
| `--skip` | ruta de prim ‹repetible› | ninguno | no se convierte, con todo lo que cuelga de ella; a diferencia de `--hide`, la prim sigue en la stage que traza el horneado, así que lo que refleja, sombrea y rebota sigue en la luz de las demás |
| `--detail` | `PRIM=FACTOR` ‹repetible› | ninguno | cuando se reparte el presupuesto, las mallas en `PRIM` o por debajo toman FACTOR² veces su parte proporcional, así que su celda sale FACTOR veces más fina que la del resto; nunca más de lo que piden, y lo que no pueden tomar vuelve a las demás |
| `--bake-toward` | 3 números, unidades del mundo | ninguno | hornea lo que ve una cámara situada ahí, reflejos incluidos, como un color por gaussiana (grado 0): para una cámara que solo gira en torno a ese punto. Una gaussiana que mira hacia otro lado hornea su cuerpo sobre su hemisferio |
| `--bake-toward-radius` | unidades del mundo | `0.15` | `--bake-toward`: la bola alrededor del punto en la que puede estar la cámara; cada camino mira desde un punto de ella |
| `--light-groups` | `GRUPO[,GRUPO...]` ‹repetible›, como mucho 8 | ninguno | grupos de emisión (`primvars:athenea:lightGroup`) cuya luz indirecta se hornea, todos en una pasada de `--bake-samples` + `--bake-extra` caminos y `--bake-bounces`, como una capa cada uno, `primvars:athenea:splat:lightLayer:emission:<grupo>`: lo que los emisores del grupo ponen en cada gaussiana a nivel 1 tras un rebote. La mitad directa es del propio frame (mesh lights sobre gaussianas reiluminadas); un frame suma la capa al nivel del grupo. No con `--skinned` |
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
| `--validate-material` | ruta de prim o nombre | todos los materiales | sólo se mide este (repetible); los fotogramas siguen dejando fuera de la conversión las mallas de los demás materiales, y `flatten` sigue dibujando su fichero con todas las mallas del stage apagadas |
| `--validate-sky` | `white` o un fichero de imagen | las luces de la escena | cada fotograma bajo otro cielo -- una constante de radiancia uno, o esa imagen en los domos de la escena -- con sus demás luces apagadas; el GT se guarda como `gt_<cielo>.exr` |
| `--shadow-catcher` | flag | apagado | convertir la sombra que `--prim` arroja sobre su suelo en vez de `--prim`: un parche de gaussianas sobre el suelo bajo él y alrededor, horneado como transfer TX (el objeto y el suelo son lo que encuentran sus rayos), escrito con `primvars:athenea:splat:catcher` y dibujado negro, cubriendo lo que el objeto quita de la luz. Implica `--transfer`, una sola celda en todas partes y `--transfer-cells 32` salvo que se dé (el borde de la sombra de un sol bajo el objeto es más estrecho que 16 x 16 celdas) |
| `--catcher-ground` | ruta de prim | encontrado | con `--shadow-catcher` o `--light-catcher`: el suelo; se encuentra como la malla plana más grande fuera de `--prim` cuya cara superior está en su base y que llega por debajo de él |
| `--catcher-margin` | número | `1.5` | con `--shadow-catcher` o `--light-catcher`: hasta dónde pasa el parche de la huella del objeto, en alturas del objeto |
| `--catcher-reach` | número | `3` | con `--shadow-catcher`: hasta dónde pasa de la huella del objeto un anillo de celdas del doble de la del catcher, en alturas del objeto, para la sombra larga de una luz baja sobre el horizonte (más allá de `--catcher-margin`, con un cuarto de las gaussianas que tomaría allí la celda del parche). 0, o no más que `--catcher-margin`, es sin anillo; el catcher es entonces sólo el parche |
| `--catcher-cell` | unidades del mundo | `0` | con `--shadow-catcher` o `--light-catcher`: la celda del parche; 0 es una centésima de la altura del objeto, una treintava para un light catcher |
| `--light-catcher` | flag | apagado | convertir la luz que las lámparas de la escena arrojan sobre el suelo de `--prim` en vez de `--prim`: el parche del shadow catcher (bajo `/AtheneaLightCatcher`), que llega `--light-reach` por delante y por detrás, horneado como la radiancia que emite bajo las luces de la escena, escrito con `primvars:athenea:splat:lightCatcher` y dibujado sumado sobre lo que haya detrás, sólo las capas de sus lámparas (§3.6). Implica el bake de radiancia y una sola celda en todas partes; se rechaza con `--transfer` y con `--shadow-catcher` |
| `--light-reach` | número | `3` | con `--light-catcher`: hasta dónde pasa el parche de los extremos del objeto, a lo largo de su eje horizontal más largo, en longitudes del objeto; nunca menos de `--catcher-margin` alturas |
| `--transfer-lobes` | 0 a 2 | `0` | con `--transfer`: guardarlo como este número de lóbulos zonales en el marco propio de cada gaussiana (el bundle `SplatTransferZonal`); 0 es dos lóbulos con `--skinned` y nueve armónicos en el mundo en otro caso |
| `--skinned` | flag | apagado | llevar el esqueleto; obliga a `--no-bake`, guarda un `--transfer` como lóbulos zonales, trazado en la pose de bind (los esqueletos de la escena se fijan ahí en la capa de sesión) diga lo que diga `--time` |
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

Una entrada del material que no cabe en una constante ni en un mapa simple --
una que calcula un grafo (un mix, un ruido, un teselado hexagonal), un
`tiledimage` repetido o desplazado, un shader de superficie que no sea
`standard_surface`, OpenPBR, el de glTF o UsdPreviewSurface, o un color de
coat -- se nombra en el log, una vez por material y entrada:

```
mesh2splat: '/World/Looks/Materials/M_BrickPattern': 'base_color' is computed by a graph (ND_clamp_color3); a TX bake (--transfer) reads what it can of it back from the material on the device; otherwise the value read here stands
```

Con `--transfer` el bake lee esos campos del propio material en cada
gaussiana -- el color base (por el color del coat que tiene encima), la
rugosidad del especular y la del coat, la metalicidad y la emisión -- y dice en
cuántas gaussianas: `mesh2splat: the material read back on the device into N
gaussians`. Sin él conservan la constante que dio el stage. Un material que
transmite conserva su color base tal como se leyó.

La salida se escribe entera o no se escribe: como `.<nombre>.partial-<pid>.<ext>`
en el mismo directorio, y renombrada a `-o` cuando está completa. Una
conversión que falla no deja nada en `-o` -- o deja el fichero que ya había,
tal como estaba -- y borra su fichero parcial. Con `--lod-levels`, cada nivel
y la escena que los dibuja se escriben así.

### 2.8.1 `athenea flatten` — una nube TX como ficheros estándar de Gaussian Splatting

Las nubes TX (`mesh2splat --transfer`) y los receptores de sombra de una
escena, bajo sus propias luces o bajo otro cielo, escritas como los armónicos
de grado 3 que lee un visor estándar: sin volver a hornear y sin malla. Cada
gaussiana se sombrea desde `--directions` direcciones exactamente como la
sombrea el rasterizador, se ajusta lo que un visor mostraría ahí, y el ajuste
se escribe en cada formato pedido. La escena se compone con sus mallas
apagadas (`<output>_flatten_stage.usda`, que se queda junto a los ficheros).

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `stage` | ruta, obligatoria | — | una escena con nubes TX y una cúpula: la propia del bake, o una nube con `--sky` |
| `-o`, `--output` | ruta | `flat` | sin extensión, cada formato de `--format` uno al lado de otro; con una de ellas, ese formato solo |
| `--format` | lista separada por comas de `ply`, `spz`, `glb`, `usdc`, `usda` | `ply,spz,glb` | |
| `--sky` | ruta de imagen \| `white` | las luces de la escena | las cúpulas de la escena toman la imagen y sus demás luces se apagan; una escena sin cúpula recibe una, girada hacia su eje vertical |
| `--hide` | ruta de prim ‹repetible› | ninguna | se deja fuera, como si estuviera inactiva |
| `--directions` | entero, al menos 16 | `256` | direcciones desde las que se mira cada gaussiana; 1024 para metales más nítidos |
| `--exposure` | pasos | `0` | sobre la luz, antes de codificarla |
| `--referred` | `display` \| `scene` | `display` | `display` ajusta el color sRGB que muestra un visor; `scene` ajusta luz lineal, solo para `ply` y USD |
| `--no-roughness-floor` | flag | suelo activo | reflejos tan nítidos como el material; anillos donde el grado 3 no puede sostenerlos |
| `--roughness-floor-share` | 0..1 | `0.05` | la parte de la energía de un lóbulo de reflexión que el suelo deja por encima de la banda 3 |
| `--fit-lambda` | número | `1e-4` | la regularización del ajuste, por l²(l+1)² |
| `--back-weight` | número | `0.01` | lo que pesa en el ajuste el hemisferio trasero de una superficie; una pared fina pesa en toda la esfera |
| `--baked` | flag | no | todas las nubes de la stage, con la luz horneada (`mesh2splat` sin `--transfer`), escritas como las muestra el frame; no solo las nubes TX reiluminadas |
| `--facing` | 3 números, unidades del mundo | ninguno | una cámara que solo gira en torno a este punto (una cabeza PTZ): toda gaussiana cuya superficie mira hacia otro lado se escribe transparente, porque esa cámara nunca la ve; una pared fina se conserva |
| `--view-box` | 6 números, unidades del mundo | ninguno | la zona de la cámara, `MINX MINY MINZ MAXX MAXY MAXZ`: cada gaussiana se ajusta a las direcciones desde las que se ve allí, y el resto de la esfera pesa `--view-box-outside` de ellas |
| `--view-box-samples` | 16..4096 | `256` | `--view-box`: puntos de la zona desde los que se mira cada gaussiana |
| `--view-box-outside` | 0..1 | `0.05` | `--view-box`: lo que pesan las `--directions` de la esfera frente a las de la zona |
| `--spz-version` | `3` \| `4` | `3` | 3 es gzip y lo que lee Spark; 4 es ZSTD y necesita ZSTD en la compilación |
| `--spz-sh1-bits`, `--spz-sh-rest-bits` | 1..8 | `8`, `8` | bits que SPZ guarda de la banda 1, y de las bandas 2 y 3 |
| `--time` | número | `0` | time code de USD |
| `--range` | `INICIO:FIN[:PASO]` | sin fijar | una secuencia: cada time code aplanado y cortado en páginas SPZ, una época de un stream en vivo por frame (§3.7); fuerza `--format spz` y hace de `-o` un directorio |
| `--chunk-splats` | entero | `65536` | con `--range`: las gaussianas que caben como mucho en una página; las de un frame se ordenan sobre una curva de Morton de su caja y se cortan en tramos de este tamaño, así que una página es una parte del modelo |
| `--keep-order` | flag | no | con `--range`: cada frame se corta en páginas en el orden en que el stage lista sus gaussianas, no a lo largo de una curva de Morton: una gaussiana de una nube con esqueleto conserva su índice en todos los frames (lo que necesita `video-encode --order stable`) |
| `--validate` | directorio | ninguno | medir un fichero escrito contra las mallas trazadas, material a material (§3.1, *Una conversión medida*) |
| `--validate-stage` | ruta | la de entrada | la escena de mallas desde la que se traza el GT |
| `--validate-format` | `ply` \| `spz` | `spz` | qué fichero se vuelve a leer y se mide |
| `--validate-camera`, `--validate-size`, `--validate-paths`, `--validate-bounces`, `--validate-material` | | como en `mesh2splat --validate` | |
| `--path` | directorio ‹repetible› | ninguno | directorios extra de bundles AOFX (el efecto Measure) |

**Los ficheros.** Y arriba, metros, sea cual sea el eje y la unidad de la
escena: lo que three.js, Spark, SuperSplat y glTF toman tal cual. Las mismas
coordenadas en todos los formatos. Los colores son lo que muestra un visor,
codificados en sRGB, `0.5 + SH·C0` en el término constante, y el visor
recorta los negativos; el ajuste mantiene cada dirección delante de una
gaussiana de superficie (en toda la esfera para una pared fina) en cero o
por encima, encogiendo sus bandas altas.

- **PLY**: la disposición de 3DGS (la de INRIA): `x y z nx ny nz f_dc_0..2
  f_rest_0..44 opacity scale_0..2 rot_0..3`, float32, la opacidad como
  logit, las escalas como ln σ, la rotación w x y z; 248 bytes por gaussiana.
  Ninguna propiedad más.
- **SPZ**: el de Niantic, grado 3, armónicos a 8 bits (5 y 4 es lo que pone
  su empaquetador por defecto; los reflejos aplanados viven en las bandas
  altas), la bandera antialiased puesta, el punto fijo elegido a partir de la
  caja (como mucho 20 bits). Los armónicos que se salen del [-1, 1] de SPZ se
  escalan enteros, no se recortan.
- **glTF** (`.glb`): una primitiva `POINTS` con `KHR_gaussian_splatting`
  (`kernel` `ellipse`, `colorSpace` `srgb_rec709_display`), `POSITION`,
  `ROTATION` (x y z w), `SCALE` lineal, `OPACITY` lineal y
  `SH_DEGREE_l_COEF_n` en float, solo en `extensionsUsed`. Aún no se ha
  abierto en un visor.
- **USD** (`.usdc`, `.usda`): un `ParticleField3DGaussianSplat` como el que
  escribe `athenea convert`, eje Y arriba, un metro por unidad.

**Suelos.** Ningún eje de un PLY o un glTF baja de ln σ = -12 (el rango de
Spark); el suelo propio de SPZ es -10. Se imprime la fracción que se lleva al
suelo, igual que la de gaussianas cuyas bandas se encogieron y la de las que
SPZ escaló para que cupieran.

**Lo que imprime.** Cada nube y su número de gaussianas, el suelo de
rugosidad encontrado, la caja, las pérdidas de arriba, el tamaño de cada
fichero y los bytes por gaussiana, y lo que tardaron el sombreado y el
ajuste, y el empaquetado y la escritura.

### 2.8.2 `athenea lightmap` — la luz de un stage en texturas sobre sus propias coordenadas

Cada malla que lleva las coordenadas de lightmap (`--uv`) se traza con
caminos en cada texel que cubren sus triángulos en su atlas -- la luz que
emite su superficie, con emisores y rebotes --, y los atlas se escriben junto
a un stage que los muestra: el original como sublayer, cada malla horneada
sin luz (`primvars:athenea:unlit`, §4.3) y ligada a un material cuya primera
textura es su atlas, leída en las mismas coordenadas.

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `stage` | ruta, obligatoria | — | |
| `-o`, `--output` | directorio | `lightmap` | `lightmap_<atlas>.exr` (half, lineal) y `lightmapped.usda` |
| `--uv` | nombre de primvar | `st_lightmap` | una malla sin él se queda como está |
| `--size` | texels, de 16 a 16384 | `1024` | el lado de un atlas |
| `--samples` | de 1 a 65536 | `64` | caminos por texel |
| `--bounces` | de 0 a 16 | `3` | después del primer impacto |
| `--dilate` | anillos | `8` | cuánto crece la luz de una isla en el margen que la rodea, antes del denoiser y después |
| `--extra-samples` | caminos | `0` | la media de caminos por texel que se añade después de la primera pasada, donde la luz es más ruidosa para lo que cuesta |
| `--pass-samples` | caminos | `32` | `--extra-samples`: caminos por texel en cada pasada extra |
| `--max-passes` | entero | `16` | `--extra-samples`: pasadas extra por texel como mucho |
| `--no-denoise` | flag | con denoise | deja el atlas tal como se trazó |
| `--planar` | flag | apagado | un atlas sobre el rectángulo horizontal que cubren las mallas (con `--prim`, un suelo), tengan las coordenadas que tengan, y `<name>.json` al lado; no se escribe stage |
| `--name` | texto | `lightmap_<atlas>` | el nombre de los ficheros, sin extensión |
| `--bake-toward` | X Y Z, unidades del mundo | ninguno | cada texel se hornea como lo ve una cámara en ese punto: el material entero, con su pulido y los reflejos que muestra |
| `--bake-toward-radius` | unidades del mundo | `0` | `--bake-toward`: la bola alrededor del punto en la que puede estar la cámara |
| `--prim` | ruta de prim | todas las mallas | |
| `--time` | número | `0` | time code de USD |
| `--default-lights` | flag | apagado | un dome y un sol para un stage que no tiene luces |
| `--light-groups` | `GRUPO[,GRUPO...]` ‹repetible›, como mucho 8 | ninguno | grupos de emisión (`primvars:athenea:lightGroup`) sacados de cada atlas (horneado a nivel 0) y horneados en uno propio cada uno, todos en una pasada: `<nombre>_emission_<grupo>.exr`, la luz entera del grupo a nivel 1, directa e indirecta, ya que una malla con lightmap no se ilumina. `lightmapped.usda` enlaza solo el atlas principal; la suma a los niveles de los grupos es del host |

**Atlas.** Una malla va en el atlas que dice su `userProperties:lightmap_atlas`
(un int o un float) -- en la malla, o en el prim más cercano por encima que lo
tenga, que es donde Blender escribe las propiedades de un objeto --, el 0 si
ninguno lo dice; un fichero por atlas. Un texel es
del primer triángulo que llega a él; uno cuyo centro queda a menos de medio
texel del borde de un triángulo es de ese triángulo. El horneado es el del
propio trazador de caminos (el de `mesh2splat`), de grado 0: lo que emite la
superficie, igual en todas las direcciones -- un lightmap no guarda luz que
dependa de la vista, y sus reflejos son de otra pasada
(`athenea:screenSpaceReflections`).

**Visto desde un sitio.** Con `--bake-toward` un texel no es la luz que su
superficie manda hacia todos lados sino lo que ve ahí una cámara en ese
punto -- a menos de `--bake-toward-radius` de él --, por los mismos caminos
que traza `mesh2splat --bake-toward`: el brillo de un suelo y lo que refleja
quedan en la textura, bien para una cámara que gira alrededor de ese punto y
sólo para ella. Un texel cuya superficie no mira al punto se hornea como
siempre.

**Planar.** Con `--planar` los texels se reparten sobre el rectángulo que
cubren las cajas de las mallas en el mundo, sobre los dos ejes horizontales
del stage: u por +x, v por +y con Z arriba (por -z con Y arriba), uv (0, 0)
en la esquina menor, el atlas entero para ese rectángulo -- para un suelo que
un host dibuja como un quad. `<name>.json` dice dónde: las cuatro esquinas en
el marco del stage y en uno con Y arriba (x, z, -y), la altura (lo más alto
de las cajas), los ejes, y los texels por metro en u y en v.

**Con denoise.** Si el build lleva OIDN, el atlas pasa por el denoiser en su
propio espacio, guiado por el albedo que el horneado lee en cada texel y por
hacia dónde mira, cada guía crecida en los márgenes igual que la luz; después
los texels propios crecen otra vez, para que un margen tenga la luz de su
isla y no lo que el filtro extendió por el borde.

**Lo que no hace.** Una malla instanciada no se hornea; el binding propio de
un GeomSubset es más fuerte que el que se escribe en su malla, así que una
malla con subsets muestra sus materiales, no su atlas.

### 2.8.3 `athenea vdb2splat` — un volumen en gaussianas volumétricas, y de vuelta

Un volumen de niebla `.vdb` ajustado por una suma de gaussianas 3D,
μ_t(x) = Σ σ_i G_i(x) con cada G_i de pico 1 (propuesta 103): la extinción de
la grid de densidad y, si el fichero las tiene, las grids de temperatura y de
Cd que la acompañan, como medias ponderadas por la densidad. El ajuste es el
efecto Vdb2Splat (§7.2) y cada paso suyo corre en el dispositivo; cómo lo
cuenta development.es.md §6.13. Las gaussianas se escriben como un
ParticleField (abajo). `--back` escribe la mezcla evaluada en los vóxeles
activos de la propia grid de densidad de la entrada: un `.vdb` que el path
tracer dibuja como dibuja cualquier volumen.

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `input` | ruta, obligatoria | — | un `.vdb`; con `--make-fixture`, el fichero que se escribe |
| `-o`, `--output` | ruta | ninguna | `.usda`, `.usdc`, `.usd`; sin ella no se escribe nada |
| `--back` | ruta | ninguna | un `.vdb` de `density` (y `temperature`, donde la entrada la tiene) sobre la topología de la entrada |
| `--density` | nombre de grid | `density` | una grid float; el ajuste rechaza un fichero sin ella |
| `--temperature` | nombre de grid | `temperature` | una grid float, leída donde el fichero la tiene; sus vóxeles se leen en los de la densidad. Los frames ajustados de una secuencia también la leen, en frío y en caliente, cada uno la suya (`flames` para un fichero cuyo fuego es una grid de llamas) |
| `--colour` | nombre de grid | `Cd` | una grid vec3, leída donde el fichero la tiene, recortada a [0, 1] y guardada con 8 bits por canal |
| `-n`, `--target` | entero | `20000` | cuántas gaussianas se escriben como mucho |
| `--em-iterations` | entero | `40` | pasos de EM sobre la geometría, primero: cada gaussiana se lleva a los momentos de la densidad de la que es responsable |
| `--iterations` | entero | `100` | pasos de Adam sobre la geometría, después de ellos |
| `--attribute-iterations` | entero | `20` | pasos de Jacobi sobre temperatura y Cd, con la geometría fija; ninguno sin una de las dos grids |
| `--max-block` | `8`, `16`, `32`, `64`, `128` | `64` | vóxeles de lado del bloque mayor del que arranca una gaussiana; un tercio de él es la desviación mayor que puede tomar una gaussiana |
| `--min-std` | 0.05 a 10 | `0.5` | vóxeles: la desviación menor que puede tomar una gaussiana. Por debajo de un vóxel, más o menos, la mezcla puede coincidir con la rejilla solo en los centros de los vóxeles y diferir entre ellos, lo que muestra un frame en caliente llevado por el flujo fuera de la red (véase decisions.md, «la desviación menor») |
| `--emission-weight` | número ≥ 0 | `1` | lo que cuenta la dispersión de la temperatura en un bloque junto al error de la densidad, con la temperatura escalada al rango de la densidad |
| `--init-fraction` | de 0,01 a 1 | `0.6` | la parte del presupuesto que tiene el primer corte; el resto nace donde el residuo es mayor |
| `--densify-every` | entero | `20` | iteraciones entre rondas de densificar y podar; ninguna después del 70 % de ellas |
| `--densify-rounds` | 0 a 64 | `0` | las rondas de densificar y podar como un número: la ronda k tras k / (n + 3) de las iteraciones (EM y Adam), así que el calendario escala con `--iterations` (4 sobre las 140 por defecto: cada 20, como `--densify-every 20`; sobre 560: cada 80); 0 usa `--densify-every`. Los frames en caliente de una secuencia siguen con `--warm-densify-every` |
| `--learning-rate` | número | `0.1` | el paso de Adam, en unidades de log para el pico y la diagonal del factor y en desviaciones para el centro; decae cien veces a lo largo de los pasos de Adam |
| `--split-share` | de 0 a 1 | `0.5` | la parte de los huecos nuevos de una ronda de densificación que parte las gaussianas responsables de más residuo, cada una en dos a lo largo de su eje más largo; el resto nace en los peores vóxeles |
| `--no-bins` | flag | apagado | la pasada hacia delante sobre todas las gaussianas para cada hoja, descartadas por su caja, en vez de la lista de cada hoja, y el gradiente un grupo por gaussiana: el ajuste como era antes del binning, para medir tiempos |
| `--bin-slack` | número ≥ 0, vóxeles | `0` | cuánto se ensancha la caja de una gaussiana en las listas de las hojas, que se conservan entonces de una iteración a otra hasta que una caja sale de las celdas de hoja sobre las que se repartió; `0` las rehace en cada iteración. El resultado no depende de ello; en smoke2 un margen hizo el ajuste más lento |
| `--checkerboard` | flag | apagado | el gradiente de cada paso de Adam sumado sobre los vóxeles de una paridad, alternando de un paso a otro, contados dos veces: la mitad de los vóxeles por paso (smoke2 con 50k y 300 pasos: 3,98 s frente a 5,21, el error de densidad 0,0395 frente a 0,0377) |
| `--quantise` | `half` \| `byte` | ninguno | los registros del ajuste cuantizados en el registro compacto y todo lo que sigue (`--back`, `--measure`, `-o`) hecho de lo que este decodifica: el centro sobre la caja de la nube y ln σ sobre su rango en 16 bits cada uno, el factor de la precisión como los logaritmos de su diagonal y sus fuera de la diagonal sobre la media geométrica de las diagonales a su lado en medios floats (`half`, 24 bytes por gaussiana) o en bytes sobre sus rangos (`byte`, 20 bytes), Cd en rgb8 y la temperatura en 8 bits sobre su rango |
| `--compact` | ruta | ninguna | con `--quantise`: los registros compactos como fichero: una cabecera de 64 bytes (`ATVG`, versión 1, número, modo 0 half o 1 byte, y doce rangos float: la caja de los centros, baja y alta, en espacio de índices más medio vóxel, el de ln σ, el de la diagonal en logaritmo, el de la temperatura), el mapa de índice a mundo como 16 doubles (por columnas) y después 6 o 5 palabras por gaussiana |
| `--from-compact` | ruta | ninguna | las gaussianas de un fichero de `--compact`, sus palabras decodificadas en registros en el dispositivo, en vez de un ajuste; `input` es la grid sobre la que se devuelven y contra la que se juzgan, y `--back`, `--measure` y `-o` siguen como tras un ajuste |
| `--against` | `.vdb` | ninguno | mide la densidad de esta grid frente a la de `input` sobre los vóxeles activos y se detiene; las dos deben compartir topología (una cuantización NanoVDB de `input` devuelta a floats la comparte) |
| `--device-mb` | entero, MB | `2048` | el techo del pool del dispositivo en memoria unificada (una tarjeta discreta toma tres cuartos de lo libre cuando es más): de él salen los búferes de trabajo del ajuste, unos 60 bytes por vóxel de las hojas y las listas de las hojas; bunny_cloud (85 525 hojas) necesita más que el valor por defecto |
| `--profile` | flag | apagado | el procesador espera a cada kernel e imprime el tiempo de cada uno, el mayor primero: un desglose, más lento que el ajuste que mide |
| `--density-scale` | número | `1` | de densidad a extinción por unidad del mundo: para los rayos de `--measure` y la opacidad que se escribe para un visor estándar |
| `--measure` | flag | apagado | imprime el error L2 relativo de la vuelta sobre los vóxeles activos y su mayor diferencia, y la transmitancia en forma cerrada frente al ratio tracking a través de la entrada a lo largo de `--rays` segmentos aleatorios |
| `--rays` | entero | `4096` | `--measure` |
| `--ray-samples` | entero | `64` | estimaciones de ratio tracking por rayo |
| `--make-fixture` | `mixture` \| `blob` \| `blob-sequence` | ninguno | escribe `input` y termina: una caja de `--fixture-size` vóxeles de lado, todos activos, rellena en el dispositivo con una mezcla conocida de `--fixture-gaussians` gaussianas aleatorias o con una bola de ruido suave, con una grid de temperatura |
| `--fixture-size` | entero | `96` | vóxeles de lado |
| `--fixture-gaussians` | entero | `64` | `mixture` |
| `--voxel-size` | número | `0.05` | el vóxel del fixture, en unidades del mundo |
| `--seed` | entero | `1` | el del fixture y el de los rayos |
| `--path` | directorio ‹repetible› | ninguno | directorios de bundles que se buscan después de `$AOFX_PLUGIN_PATH` |
| `--sequence` | flag | apagado | `input` (y `--back`) nombran los frames con una racha de `#`, con ceros hasta la longitud de la racha; con `--frames` |
| `--frames` | `PRIMERO ÚLTIMO` | `0 0` | los frames de una secuencia, los dos incluidos; también los de `--make-fixture blob-sequence` |
| `--independent` | flag | apagado | cada frame un ajuste completo propio; si no, cada frame tras el primero arranca de las gaussianas del anterior en sus mismos huecos, llevadas por su `vel` donde el fichero la tiene |
| `--warm-em` | entero | `0` | pasos de EM de un frame en caliente (EM después de los pasos de Adam mueve mucho las gaussianas: apagado por defecto) |
| `--warm-iterations` | entero | `30` | pasos de Adam de un frame en caliente |
| `--warm-learning-rate` | número | `0.03` | el paso de Adam en un frame en caliente |
| `--prune-mass` | 0 a 1 | `1e-6` | en una ronda de densificar, una gaussiana por debajo de esta fracción de la masa total se va y su hueco se vuelve a dar |
| `--warm-prune-mass` | 0 a 1 | `1e-6` | `--prune-mass` en los frames en caliente de una secuencia, cuyas rondas solo tienen para la densidad nueva los huecos que esto libera |
| `--warm-densify-every` | entero | `20` | pasos entre las rondas de densificar y podar de un frame en caliente (ninguna después del 70 % de ellos; 30 pasos: una ronda): el calendario de un ajuste completo no llega a un frame en caliente |
| `--no-advect` | flag | apagado | un frame en caliente arranca donde el anterior dejó sus gaussianas |
| `--dt` | número | `1` | frames sobre los que el flujo lleva las gaussianas, de un frame al siguiente |
| `--cloud` | ruta con una tira de `#` | ninguna | una secuencia: las gaussianas de cada frame, todos los huecos (uno muerto es una mota de ln σ −30 en el origen, para que los huecos conserven su sitio), como PLY de 3DGS: posición, desviaciones y rotación en el mundo, el albedo como color base, ln σ en `f_rest_0` y la temperatura en `f_rest_1` (los dos primeros coeficientes de resto del rojo), la opacidad como la escribe `-o` para un visor. Lo que toma `video-encode --each-a-frame` |
| `--from-athv` | `.athv` | ninguno | una secuencia: las gaussianas de cada frame decodificadas de este vídeo de splats de `--cloud` (en orden, con el decodificador hardware) en vez de ajustadas, y luego devueltas y juzgadas como un ajuste: el error de densidad frente al frame de `input`, el parpadeo, `--back`. `input` puede ser las grids que devolvió un ajuste, para juzgar el vídeo frente a la nube f32 |
| `--velocity` | nombre de grid | `vel` | la grid del flujo (vec3), leída de cada frame; el flujo del frame anterior se pone sobre los vóxeles activos del actual. Una grid escalonada (clase `staggered`, una grid MAC como `velocity` de Mantaflow) se promedia de las dos caras de cada vóxel a su centro |
| `--velocity-units` | `world`, `mantaflow` | `world` | `world`: unidades del mundo por frame (divididas por el vóxel). `mantaflow`: celdas por unidad de tiempo de Mantaflow, de las que un frame es 0,1 x 25 / `--fps` (0,10417 a 24) |
| `--fps` | número | `24` | con `--velocity-units mantaflow`: frames por segundo de la secuencia |
| `--tile-dir` | directorio | ninguno | ajusta la grid por ladrillos (abajo) y guarda aquí el plan y el resultado de cada ladrillo; la misma orden lanzada otra vez sigue desde el primer ladrillo que no había terminado. Un directorio guarda un plan: otra entrada, otra grid, otro margen, otro `--tile-cells`, otro `-n` u otro `--tile-weight` se rechazan |
| `--tile-cells` | entero | `0` | las celdas del tamaño de una hoja (8³ vóxeles: una hoja, o la parte de un tile activo) que lee como mucho un ladrillo, con su margen; `0` toma `--device-mb` × 1024 / 48, unos 48 KB del pool por celda (los 60 bytes por vóxel del ajuste, la imagen de las hojas y la grid, las hojas que añade el alcance del medio) |
| `--tile-margin` | 0 a 256, vóxeles | `32` | cuánto se lee y se ajusta alrededor del núcleo de un ladrillo; solo se quedan las gaussianas cuyo centro cae en el núcleo |
| `--tile-weight` | `mass`, `energy` | `mass` | qué es la parte de `-n` de un ladrillo: la densidad de su núcleo sumada, o su cuadrado sumado |
| `--tile-refine` | entero | `30` | pasos de Adam de una segunda pasada por cada ladrillo, en caliente desde las gaussianas de toda la mezcla que lo alcanzan (sin rondas de densificar ni de podar), quedándose las que tienen el centro en su núcleo: el margen de un ladrillo tiene entonces las gaussianas de sus vecinos; `0` se la salta (en la nube a media resolución la costura sin ella dejaba una línea y un error de densidad de 0,199 frente a 0,152 sin ladrillos; con 30 pasos, 0,152) |
| `--tile-seconds` | número, s | `0` | no se empieza ningún ladrillo pasados estos segundos (0: sin límite); la orden dice "run the same command again to go on" y sale con 0 |
| `--back-grid` | `.vdb` | la entrada | con `--tile-dir`: la grid sobre cuya topología de densidad se escribe `--back`, leída entera -- el mismo volumen a otra resolución (con su propia transformación: las gaussianas pasan por el mundo) -- y la mezcla medida allí contra su densidad |

Una secuencia imprime, por frame: las gaussianas vivas, el error L2 relativo de la densidad devuelta, el tiempo del ajuste y, desde el segundo frame, el parpadeo -- sqrt(Σ((b₁ − b₀) − (o₁ − o₀))² / Σ(o₁ − o₀)²) sobre los vóxeles activos, lo que el cambio de un frame al siguiente de la grid devuelta tiene de mal frente al del original -- y cuántas de las gaussianas del frame anterior siguen vivas en el mismo hueco, y cuánto se movieron frente a su desviación media. `--measure` imprime también el error de la temperatura, ponderado por la densidad (la medida del propio ajuste: donde no hay densidad la temperatura de una gaussiana no dice nada) y sin ponderar. `--make-fixture blob-sequence` escribe `--frames` de la bola de ruido llevada por un flujo conocido -- un giro alrededor de su vertical de 2π/120 por frame y una subida de 0,3 vóxeles por frame --, cada uno con su `density` y su `vel` (unidades del mundo por frame).

**Ladrillos** (`--tile-dir`): una grid mayor que el dispositivo, o que la
memoria del host que pide leerla entera, se corta en ladrillos y se ajusta de
uno en uno. El plan se hace solo con la estructura del árbol (sin descomprimir
ningún valor): sus celdas del tamaño de una hoja se cortan con un k-d por la
celda mediana a lo largo del lado más largo hasta que un ladrillo, con su
margen, tiene como mucho `--tile-cells`. Un ladrillo se lee solo -- la carga
diferida de OpenVDB descomprime solo las hojas de dentro -- y pasa pasada a
pasada por el dispositivo: sus sumas (la masa de su núcleo, la de lo que lee, la
energía de su núcleo); su ajuste, de `-n` × la parte de su peso × lo que lee
entre la masa de su núcleo, del que se quedan las gaussianas cuyo centro cae en
el núcleo; la pasada de las costuras (`--tile-refine`); y con `--measure` su núcleo devuelto desde cada gaussiana cuya caja
lo toca, con las sumas de los cuadrados sumándose al L2 relativo de toda la
grid (impreso ladrillo a ladrillo y en total). Las opciones del ajuste valen
para cada ladrillo; `--temperature`, `--colour`, `--quantise`, `--rays` y las
secuencias no se leen con ladrillos (solo se ajusta la densidad). Ficheros del
directorio: `plan.txt`, `sums_NNNN.txt`, `fit_NNNN.rec`, `refine_NNNN.rec`
(una cabecera de 20 bytes -- `BRCK`, cuántas se quedan, el presupuesto o los
pasos de refinado, cuántas se escribieron, los segundos del ajuste como float
-- y luego 16 floats por gaussiana, los registros del efecto en el marco de
índice de la entrada) y `measure*_NNNN.txt`.

Lo que guarda el stage en su `UsdVolParticleField3DGaussianSplat`, además de
las posiciones, orientaciones, escalas y `radiance:base` del esquema:

| Primvar | Tipo | Significado |
|---|---|---|
| `primvars:athenea:splat:volume` | bool | `true`: las gaussianas son un medio, no una superficie |
| `primvars:athenea:volume:sigma` | half[] ‹1 por gaussiana› | el pico de la extinción, en las unidades de densidad de la grid (el `densityScale` del Volume lo multiplica) |
| `primvars:athenea:volume:albedo` | color3h[] | Cd, 1 donde el fichero no lo tiene |
| `primvars:athenea:volume:temperature` | half[] | la temperatura, donde el fichero la tiene |

Las posiciones y las escalas están en el mundo de la grid (su mapa de índice
a mundo). `radiance:base` guarda Cd y, como opacidad, 1 − exp(−τ) a través
del centro de la gaussiana a lo largo de su desviación media,
τ = σ · `--density-scale` · √(2π) · s, nunca por debajo de 1/254 (un cargador
descarta lo que es más tenue que 1/255): lo que dibuja un visor estándar. El
raster de este motor dibuja el medio mismo (§4.3, un medio de gaussianas) y no
lee ninguna opacidad. El
presupuesto cuenta las gaussianas que se escriben; se escriben menos donde los
bloques más gruesos que tiene el volumen ya son más que `--init-fraction` de
él.

```sh
athenea vdb2splat smoke.vdb -n 50000 -o smoke_gs.usdc --back smoke_back.vdb --measure
athenea vdb2splat blob.vdb --make-fixture blob --fixture-size 128
athenea vdb2splat cloud.vdb -n 200000 --tile-dir cloud_bricks --device-mb 14336 --tile-refine 30 --tile-seconds 250 --measure -o cloud_gs.usdc --back cloud_half_back.vdb --back-grid cloud_half.vdb
```

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

### 2.9.2 `athenea splat-transfer` — un transfer TX horneado en una nube que no tiene malla

`mesh2splat --transfer` hornea un transfer trazando caminos sobre una malla.
Una nube hecha sin ella -- surfels sobre el level set de la opacidad de una
captura, una nube hecha a partir de renders -- se hornea aquí contra sí misma y
las demás nubes del stage, con las estructuras del ray tracer de splats (la ruta
por hardware: un dispositivo con ray queries). Se escribe como mesh2splat
escribe un transfer TX (16 valores directos, 48 indirectos, 48 del campo
reflejado, las direcciones abiertas), así que la nube se reilumina del mismo
modo.

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `stage` | ruta, obligatoria | — | `.usd`, `.usda`, `.usdc`; se traza cada `ParticleField3DGaussianSplat` que haya en ella, en su propio espacio |
| `--prim` | ruta de prim, obligatoria | — | el ParticleField que se hornea |
| `-o`, `--output` | ruta, obligatoria | — | la escena escrita, aplanada, con el transfer en `--prim` |
| `--cells` | 16 o 32 | `16` | celdas por lado de las direcciones abiertas sobre toda la esfera (8 o 32 enteros por gaussiana) |
| `--grid` | entero, múltiplo de `--cells`, como máximo 128 | `32` | rayos por lado de la parrilla octaédrica sobre la que se suman las mitades: `grid²` rayos por gaussiana y pasada |
| `--offset` | número | `1` | un rayo parte a este número de veces el sigma más ancho de su gaussiana fuera de la superficie, por su propio lado: menos deja que los rayos rocen las gaussianas vecinas de la misma superficie |
| `--cut` | número | `0.001` | transmitancia bajo la cual un rayo se detiene |

- **Normales.** `primvars:athenea:splat:normal` donde la nube la conserva, y si
  no el eje más fino de cada gaussiana. Deben mirar hacia fuera de la
  superficie: el bake imprime qué parte de las direcciones abiertas está del
  lado al que miran, y lo avisa cuando es menos de un tercio.
- **Un cristal sólido** (`primvars:athenea:splat:ior` mayor que 1, constante):
  como lo hornea el path tracer -- sus direcciones abiertas detrás de él
  cerradas por su propia cara lejana, y su campo sobre toda la esfera, una
  dirección detrás de su superficie doblada hacia dentro por el índice y
  enviada más allá de sus propias gaussianas, que lleva el throughput del
  cristal (Fresnel en las dos caras, su color como tinte dos veces) por lo que
  encuentra y el cielo blanco más allá: lo que la lente del raster lee a
  través de él. El rayo doblado sale por la cara lejana del propio cristal --
  su gaussiana más cercana allí --, doblado de nuevo hacia fuera por el índice
  a través de la normal de esa gaussiana, o reflejado una vez donde la
  reflexión total interna lo retiene.
- **La lente de un cristal sólido.** La lente del raster toma la cara lejana
  de un cristal de una esfera de la curvatura media de cada gaussiana, y da la
  vuelta a la sala como una bola entera con cualquier curvatura mayor que
  cero; con ninguna la deja recta, como una losa. El bake escribe
  `primvars:athenea:splat:curvature` encima de la del cristal con la de las
  dos con la que coinciden sus caras lejanas -- la normal de cada cara lejana
  frente a la de la bola y la de la losa, sumadas sobre los caminos a través:
  el operador de forma del propio prim donde gana la bola y es convexo, y si
  no una esfera cuyo diámetro es el grosor del cristal de frente
  (`2 / grosor`, en uu y vv); ceros donde gana la losa. Imprime cuántas
  gaussianas tomaron cada una.
- **Un prim por ejecución.** Un segundo prim se hornea a partir de la salida de
  la primera ejecución: `athenea splat-transfer a.usdc --prim /A -o b.usdc`, y
  luego `athenea splat-transfer b.usdc --prim /B -o c.usdc`.
- **Gaussianas descartadas al cargar** (un registro degenerado) conservan ceros
  en los arrays, que tienen la longitud del fichero.
- **Qué calcula:** la mitad directa y las direcciones abiertas por cuadratura
  sobre las celdas octaédricas, como las calcula con celdas el transfer del path
  tracer; la mitad indirecta y el campo reflejado, a partir de lo que encuentra
  cada rayo, un rebote sobre las nubes, con cada gaussiana encontrada aportando
  su albedo (su color base en luz lineal) por su propia mitad directa. Los
  rebotes especulares no están: un coat o un metal encontrado envía solo su
  parte difusa.

```sh
athenea splat-transfer dragon.usdc --prim /Splat/Ior_1_55 -o dragon_glass.usdc
athenea splat-transfer dragon_glass.usdc --prim /Splat/Opaque -o dragon_tx.usdc
```

### 2.9.1 `athenea lights merge` — una nube con sus lámparas

La nube base (todas las lámparas apagadas) y una nube por grupo de lámparas,
en la base con una capa de luz por grupo, que un frame suma por el peso de cada
grupo (§3.6). Cada nube de capa tiene que ser la misma conversión de la misma
geometría que la base, con sólo ese grupo encendido a radiancia uno.

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `base` | ruta, obligatoria | — | `.usd`, `.usda`, `.usdc`: la nube con todas las lámparas apagadas |
| `--layer` | `GRUPO=NUBE` ‹repetible›, obligatoria | — | la nube de un grupo; `GRUPO` pasa a ser `primvars:athenea:splat:lightLayer:GRUPO` y tiene que ser un identificador válido. Se toma su término constante (`0.5 + SH0 dc`, lineal donde la nube dice `primvars:athenea:splat:linear`, si no a través de la curva sRGB); sus otros armónicos no |
| `--lights` | ruta | ninguna | un sidecar `.lights.usda` cuyos Scopes `LightGroups`, `LightStates` y `LightRules` se copian bajo el prim de la nube, estén donde estén en el sidecar |
| `--prim` | ruta de prim | el primer ParticleField | el field de la base; el de una capa es su primero |
| `--tolerance` | número | `1e-4` | cuánto puede apartarse una gaussiana de una capa de la de la base, por coordenada, relativo a la coordenada (absoluto por debajo de 1) |
| `--drop-below` | número | `0` | una capa ninguna de cuyas gaussianas emite más que esto por unidad de radiancia (en algún canal) se descarta, y se dice |
| `-o`, `--output` | ruta, obligatoria | — | el root layer de la base, con sus capas escritas en su field |

Imprime una línea por grupo (`lights merge: cruce: 412093 of 3061200
gaussians lit`, `, dropped` para una descartada), una línea por Scope copiado,
y `lights merge: wrote RUTA`. Se niega, y no escribe nada, cuando una capa
tiene otro número de gaussianas:

```
lights merge: 'layer_drl.usdc' (drl): 3061199 gaussians where the base 'base_tx.usdc' has 3061200; a layer must be the same conversion of the same geometry
```

o cuando alguna de sus gaussianas está en otro sitio:

```
lights merge: 'layer_drl.usdc' (drl): 12 of its 3061200 gaussians are not where the base's are (tolerance 0.0001); a layer must be the same conversion of the same geometry
```

Una capa que no dice que sus colores son lineales se lee como sRGB, con un
aviso. La comparación y la conversión son un kernel por capa; vuelven dos
contadores y los floats de la capa.

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
relativa), la distribución en valores de código sRGB de 8 bits (p99, máximo,
píxeles por encima de 2) y lo que se ve: el PSNR del color codificado en sRGB y
la SSIM media de su luminancia en una caja de 7 x 7 (`display`). La CPU solo lee
los ficheros; cada número es de un
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
| `--min-alpha` | de 0 a 1 | `0`, todos los píxeles | las diferencias (relMSE, las distribuciones, PSNR, SSIM) solo donde el alfa de la referencia llega a esto; el alfa de un volumen trazado es 1 − su transmitancia, así que 0,01 es donde deja pasar menos del 99 %. Las medias no se enmascaran |
| `--mask` | ruta | ninguna | un EXR del mismo tamaño que decide los píxeles comparados en lugar del alfa de la referencia (`--mask-mode`) |
| `--mask-mode` | `alpha` \| `darker` | `alpha` | `alpha`: donde el alfa de la máscara llega a `--min-alpha`; `darker`: donde la luminancia de la referencia está por debajo de `--mask-ratio` de la de la máscara -- una sombra sobre un suelo, la máscara el suelo trazado sin lo que la proyecta |
| `--mask-ratio` | número ≥ 0 | `0.9` | la proporción de `darker` |
| `--exclude` | ruta | ninguna | un EXR del mismo tamaño cuyos píxeles de alfa ≥ `--min-alpha` se dejan fuera (lo que proyecta la sombra, trazado solo) |
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

**Una malla sin material** se convierte como la dibuja el render: su
`displayColor` donde tiene un solo color, si no el gris de Hydra (0,18), con
los valores por defecto del dieléctrico simple (roughness 0,5, sin metal). Un
color por vértice no se lee; se queda el gris.

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
muestra del domo en una malla lee el más cercano a su dirección. En la ruta
raster un domo ilumina una malla **prefiltrado** por defecto: sin muestra y
sin rayo, así que sin grano -- los lóbulos difusos toman la irradiancia de
los armónicos del cielo más el sol sacado de ellos, los brillantes la imagen
del domo desenfocada a su rugosidad, y la sombra de la nube son esos seis
mapas suavizados por el ancho de cielo que representa cada uno. Lo que no ve
es una malla sombreando a otra desde el domo, y una superficie que transmite
(vidrio) sigue muestreando el domo. Donde todas las luces del frame son
domos así, no se traza ningún rayo de sombra: el vidrio muestrea sus domos
sombreados solo por las nubes. `--no-dome-prefilter` muestrea los domos en
todas partes, como antes.
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

Una gaussiana fundida conserva la masa de sus miembros: donde cubrían su celda
más de una vez, su opacidad pasa de 1 y se dibuja con el núcleo más duro, tan
sólida como ellos hasta el borde de la celda. Una región cuyos chunks aún no
están en el dispositivo dibuja su nivel fundido más fino, y esas fusiones, una
junto a otra, la cubren como la cubrían sus splats. `--merged-opacity 0.99`
escribe los niveles por debajo de opacidad uno, como los ficheros escritos
antes; esos niveles dejan ver lo que hay detrás entre fusión y fusión.

Los prims en streaming de una escena comparten un solo almacén: su tamaño es la
suma de sus presupuestos, y en cada fotograma sus huecos van a los chunks que
la vista más pide, sea cual sea su prim -- un panel cerca de la cámara toma
huecos que uno lejano no necesita. Un `.athc` con otra disposición (otro tamaño
de chunk u otras cargas) tiene un almacén propio, del tamaño de su presupuesto.

Los niveles de una nube TX mantienen aparte lo que una fusión estropearía: un
grupo fundido reúne splats cuyas normales de sombreado se inclinan hacia el
mismo lado del mismo eje -- las dos caras de un panel nunca son un grupo -- y
el vidrio y las láminas (un material que transmite más de lo que refleja, o
una pared delgada que transmite) nunca se dibujan fundidos, se dibujan sus
splats. Las dos cosas se leen de la nube; nada se fija por asset. Un fichero
así tiene más grupos, y sus chunks son tramos de un solo tipo de splat cada
vez. Leído entero, el material, las capas, los ids, el transfer, las
direcciones abiertas y la curvatura de un `.athc` se quedan donde los guardan
sus niveles, y un corte los lee ahí en vez de copiarlos en cada fotograma.

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
peso, su material el de una gaussiana. Una con curvatura
(`primvars:athenea:splat:curvature`, lo que gira los lóbulos nítidos de un
transfer TX) la guarda bajo el bit 6, una palabra por elemento tras las
direcciones abiertas: los bytes 0 a 2 son uu, uv y vv del operador de forma,
cada uno con su signo en el bit 7 y en los bits 0 a 6 un paso q de 16/127 de
octava sobre las 16 octavas por debajo del byte 3, que guarda el log2 del
mayor redondeado hacia arriba como byte con signo (el tamaño 2^(top - 16 +
q 16/127); q 0 es plano); la quinta palabra de la cabecera extra es 1, y un
grupo fundido de más de una es plano (cero). El bit 3 se guarda para la propuesta
009. Un bit de los `flags` que esta compilación no conoce (el bit 3, o pasado
el bit 6) se rechaza, nombrando el fichero: un bit posterior puede añadir un bloque, y un lector que lo saltara
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

### 3.6 Las lámparas de un coche

Un coche cuyas lámparas se encienden en tiempo de ejecución es la nube base y
una capa de luz por grupo de lámparas (docs/decisions.md, light groups). Cada
una sale de la misma conversión de la misma geometría: la base con todas las
lámparas apagadas, y una por grupo con sólo ese grupo encendido a radiancia uno
y el domo negro. Después las capas se fusionan sobre la base, con los grupos y
estados del `.lights.usda`:

```sh
athenea mesh2splat base.usda --prim /World/Corvette --transfer ... -o base_tx.usdc
athenea mesh2splat group_cruce.usda --prim /World/Corvette ... -o layer_cruce.usdc   # y cada grupo
athenea lights merge base_tx.usdc --layer cruce=layer_cruce.usdc --layer drl=layer_drl.usdc \
    --lights corvette.lights.usda -o corvette_lights.usdc
athenea stage night.usda --light-state noche_ciudad -o night.exr
athenea stage night.usda --light-state aparcado --light-level drl=0.5 -o drl.exr
```

La luz que las lámparas arrojan sobre el suelo es un light catcher
(`mesh2splat --light-catcher`), convertido igual -- una base y uno por grupo,
sobre las mismas escenas -- y fusionado igual; los mismos pesos encienden el
coche y su charco de luz. Un frame dibuja, para cada grupo, su nivel por su
`emissionLuminance` por su `emissionColor` por su capa, sumado a lo que
muestra la nube; el nivel de un grupo es `athenea:lightLevel:<grupo>` donde
esté puesto, si no el objetivo del estado (`athenea:lightState:target:<grupo>`,
0 donde el estado no lo nombra), si no 1 con `initialOn` y 0 sin él. Una capa
es luz por unidad de radiancia del grupo: el material del grupo a
`emission_luminance` 1 cuando se horneó. Encender una lámpara no sube nada.
Las reglas y las curvas en el tiempo no se evalúan (un estado y niveles es lo
que toma un frame), y sólo el rasterizador suma las capas.

### 3.7 Una animación sobre un stream en vivo

Una nube con esqueleto reproducida como un stream en vivo de páginas: cada
frame posado, aplanado bajo la luz del stage y cortado en páginas que quedan
juntas, y empujado una época por frame al stream que nombra un stage. Lo que
recibe un visor es SPZ estándar, un frame cada vez; el stage solo muestra
frames enteros.

```sh
athenea mesh2splat SparrowBird.usda --skinned --transfer --resolution 256 -o sparrow_tx.usdc
# sparrow.usda: la nube, una cámara y un domo (la luz que flatten hornea)
athenea flatten sparrow.usda --range 1:61 --chunk-splats 32768 --spz-version 4 -o pages
# live.usda: un ParticleField bajo AtheneaLiveSourceAPI, stream "sparrow",
# chunks = los del manifiesto, chunkSplats 32768, degree 3, girado 270 grados
# sobre x en un stage Z arriba (las páginas son Y arriba, y los ejes de SPZ van girados)
athenea stream live.usda pages --lockstep --reference sparrow.usda --splat-display-blend -o frames
athenea stream live.usda pages --rate 30 --loops 3 --splat-display-blend
# los mismos frames como vídeo hardware: una quinta parte de los bytes de las páginas a calidad 0.5
athenea video-encode pages -o sparrow.athv --check
athenea stream live.usda sparrow.athv --rate 30 --loops 3 --splat-display-blend --frame-producer
```

`pages/stream.txt` lista el ritmo, las páginas por frame como mucho
(`chunks`), `chunkSplats` y una línea por frame: índice, time code, páginas,
gaussianas, bytes; las páginas son `f<índice>_c<página>.spz`. Un frame con
menos páginas que el máximo quita las demás. `--reference` mide el camino
entero -- el ajuste a grado 3, la cuantización de SPZ, el stream e Hydra --
contra la nube con esqueleto mientras se mueve.

## 4. Autorizar para este motor en USD

### 4.1 Apuntar una aplicación al plugin

```sh
export PXR_PLUGINPATH_NAME=<build>/plugin/usd
```

Ese directorio tiene el delegate de Hydra `hdAthenea` y los schemas codeless
juntos, así que una variable encuentra los dos. El host ofrece entonces el
renderer con su nombre, y los ajustes de abajo aparecen en su panel de render
settings.

### 4.1.1 Dentro de Blender

El objetivo es la compilación de desarrollo de Blender (la diaria de la 5.3
alfa; la del 4 de octubre de 2026 lleva OpenUSD 26.08 en su `libusd_ms`,
espacio de nombres `pxrBlender_v26_08`). Blender dibuja con su propio USD, así
que hdAthenea para Blender es una compilación aparte contra él
(`macos-arm64-blender`, ver el manual de desarrollo). El add-on de
`integrations/blender/athenea_hydra` es una extensión de Blender: registra
el motor de render **Athenea** (`bl_delegate_id = HdAtheneaRendererPlugin`),
entrega el plugin a USD con `pxr.Plug` al activarse, y se niega cuando
`pxr.Usd.GetVersion()` no es el USD contra el que se compiló el plugin (el
panel de render y el terminal dicen cuál, y qué recompilar).

| Variable | Qué hace |
|---|---|
| `ATHENEA_HYDRA_PLUGIN_DIR` | el directorio `plugin/usd` de una compilación `macos-arm64-blender`. Sin ella el add-on mira en su propio `plugin/usd` y luego en su preferencia. |
| `ATHENEA_MATERIALX_ROOT` | el add-on la pone en el `plugin/materialx` de la compilación (las bibliotecas de MaterialX 1.39.5) si no está ya definida. |

```sh
ATHENEA_HYDRA_PLUGIN_DIR=<build>/plugin/usd \
  /Applications/Blender.app/Contents/MacOS/Blender -b scene.blend \
  --python-expr "import sys; sys.path.insert(0, '<repo>/integrations/blender'); import athenea_hydra; athenea_hydra.register()" \
  -E ATHENEA_HYDRA -f 1
```

**El paquete.** `scripts/package-blender-addon.sh [build] [salida]` en macOS
y Linux, `scripts\package-blender-addon.ps1 [build] [salida]` en Windows (por
defecto `build/<so>-blender` y `build/blender-addon`), hace el directorio de
la extensión y `athenea_hydra-<versión>-<plataforma>.zip`, uno por sistema
(`macos-arm64`, `linux-x64`, `windows-x64`: el `platforms` del manifiesto
nombra solo ese), que instala
*Edit > Preferences > Get Extensions > Install from Disk* (o `blender
--command extension install-file -r user_default -e <zip>`). Lleva el Python,
`blender_manifest.toml`, `plugin/usd` (hdAthenea con libslang, libzstd,
libwebp y libsharpyuv al lado, nombradas `@loader_path` y firmadas ad hoc),
`plugin/materialx`, `plugin/aofx` (Mesh2Splat, SplatBakeFilter,
SplatTransferZonal, Measure) y `shaders`, que encuentra desde su propio
directorio sin definir nada. Junto a `LICENSE`, `NOTICE` y
`THIRD_PARTY_NOTICES.md` de athenea lleva `THIRD_PARTY_LICENSES.md` y
`licenses/<componente>/`: los ficheros de licencia de lo que lleva (Slang,
MaterialX 1.39.5, zstd, libwebp, OpenVDB cuando los shaders contienen
`PNanoVDB.h`) y de aquello contra lo que se compila pero deja a Blender
(OpenUSD, oneTBB, OpenColorIO, OIDN). El script se para si falta alguno.
En Linux las bibliotecas al lado de hdAthenea (Slang, zstd, libwebp) se
encuentran por un runpath `$ORIGIN`, y las de Blender ya están en el proceso;
`plugin/usd/lib` y `plugin/usd/include` tienen NVRTC de CUDA (como
`libnvrtc.so`, el nombre que pide slang-rhi), sus builtins, nvJitLink y las
cabeceras con las que compila NVRTC, de `ATHENEA_CUDA_RUNTIME` (una carpeta
con `lib/` e `include/`). Cada fichero es una copia, ninguno un enlace
simbólico: Blender descomprime una extensión con el `zipfile` de Python, que
no conserva ninguno. En Windows la carpeta al lado de `hdAthenea.dll`
tiene cada DLL que carga que no es de Blender ni de Windows (halladas con
`dumpbin`), los módulos GLSL de Slang y DXC (`ATHENEA_DXC`), y
`plugin\usd\bin` y `plugin\usd\include` tienen el runtime y las cabeceras
de CUDA (`ATHENEA_CUDA_RUNTIME`, como `package.ps1`), a los que el motor
apunta `CUDA_PATH`, así que CUDA abre solo con el driver. El script de
Windows necesita `dumpbin` (`scripts\msvc-env.ps1`) y `BLENDER_DIR` como lo
tenía el build.
`blender --command extension validate <zip>` comprueba el manifiesto.

**El motor de render.** El método de exportación **Hydra** de Blender no
entrega ninguna nube de puntos a un delegate ni ejecuta ningún hook de
exportación; con **USD**, el Empty de una nube convertida y una nube de
puntos de Gaussian splats llegan a hdAthenea (véase *Los Gaussian splats de
Blender como `UsdGeomPoints`* en 4.3). Convertir pone el método de la escena
en USD; el panel de render lo avisa cuando no lo es. El world llega al
delegate como una `DomeLight` (la conversión propia de Blender: un world de
*Environment Texture* es su imagen; uno procedural no se traduce), las luces
como luces UsdLux. Los ajustes que el motor entrega al delegate:

| Panel | Propiedad | Ajuste | Por defecto | Se aplica a |
|---|---|---|---|---|
| Athenea | *Final render* | `athenea:technique` | Raster | F12 (el viewport es siempre el raster) |
| Athenea | *Mesh antialias* | `athenea:meshAntialias` | 4 | viewport, F12 raster |
| Raster | *Light samples*, *One light a sample* | `athenea:lightSamples`, `athenea:chooseLights` | 1, apagado | todo frame |
| Raster | *Screen-space reflections* | `athenea:screenSpaceReflections` | apagado | viewport, F12 raster |
| Raster | *Prefiltered world* | `athenea:domePrefiltered` | activado | viewport, F12 raster |
| Raster | *Mesh lights*, *Mesh light samples* | `athenea:meshLights`, `athenea:meshLightSamples` | activado, 0 | todo frame; las muestras solo en raster |
| Path Tracing | *Paths*, *Paths a pass*, *Bounces* | `athenea:pathTotal`, `athenea:pathSamples`, `athenea:pathBounces` | 256, 16, 6 | F12 path traced |
| Path Tracing | *Denoise*, *Adaptive*, *Error* | `athenea:denoise`, `athenea:pathAdaptive`, `athenea:pathError` | activado, apagado, 0.02 | F12 path traced |
| Path Tracing | *Multiple importance*, *Antialias*, *Shadows through glass* | `athenea:pathMis`, `athenea:antialias`, `athenea:glassShadows` | activado, activado, activado | F12 path traced |
| Path Tracing | *Motion slices* | `athenea:motionBuckets` | 4 | F12 path traced |
| Path Tracing | *Clouds shadow themselves*, *Clouds reflect themselves* | `athenea:splatShadows`, `athenea:splatReflections` | apagado, apagado | F12 path traced |
| Gaussians | *Bounced light* | `athenea:splatTransferIndirect` | activado | todo frame |
| Gaussians | *Viewer blending* | `athenea:splatDisplayBlend` | apagado | raster |
| Gaussians | *Show* | `athenea:splatView` (Light, Albedo, ... Material roughness) | Light | raster |
| Gaussians | *Wait for streamed data* | `athenea:settleStreams` | apagado (F12 siempre espera) | viewport |
| Cloud Shadows | (cabecera), *Resolution*, *Terms*, *Density* | `athenea:cloudShadows`, `athenea:cloudShadowResolution`, `athenea:cloudShadowTerms`, `athenea:cloudShadowDensity` | activado, 1024, por receptor, 1 | todo frame |
| Volumes | *Lookup*, *Grid cache (MB)* | `athenea:volumeFilter`, `athenea:volumeCacheMB` | trilinear, 2048 | F12 path traced |
| Advanced | *Mesh visibility* | `athenea:visibility` | Automatic | todo frame |
| -- | *Film > Transparent* | `athenea:domeVisible` apagado | -- | todo frame (el world ilumina el frame y no se dibuja detrás) |
| -- | -- | `athenea:colourHalf` | -- | viewport (Blender copia el frame por memoria del host para mostrarlo; en media son la mitad de bytes) |

Los pases de F12: Combined (`color`), Depth (`depth`), Normal (`normal`) y
DiffCol (`albedo`), cada uno donde la view layer lo pide. Los volúmenes
(`.vdb`) están desactivados en esta compilación. El panel de render
(*Render Properties > Athenea*) tiene esos ajustes, en los subpaneles que
nombra la tabla, y **Compare with Cycles**: el frame de la cámara renderizado con Cycles -- las mallas de las
que se hizo una nube, mostradas otra vez; el suelo al que sustituye un
catcher, hecho shadow catcher de Cycles; film transparente -- y con Athenea
tal como está la escena, medido por `athenea compare` dentro de Blender
(`athenea_compare`, el efecto Measure): relMSE y el p99 en 8 bits en el
panel, y las imágenes `Athenea Cycles`, `Athenea Render` y `Athenea
Difference` (el mapa de calor) en Blender, escritas en `//splats/compare/`.
La escena se deja como estaba. Necesita un .blend guardado y una cámara.

**Mallas a gaussianas.** *View3D > Sidebar > Athenea > Gaussian Splats*
convierte los objetos malla seleccionados con `athenea mesh2splat
--transfer` (el transfer TX, 2.8), ejecutado dentro de Blender a través de
hdAthenea (`athenea_mesh2splat`, en la GPU; un plugin compilado sin
`ATHENEA_HYDRA_COMMANDS` se rechaza, y el panel lo dice). El .blend tiene que
estar guardado: todo se escribe a su lado. Lo que es de cada asset -- vidrio
lámina, losa o macizo, los metales, el suelo -- lo encuentra la conversión
sola.

| Opción | Por defecto | Qué es |
|---|---|---|
| Quality | Final | Preview: `--resolution 256 --max-splats 1000000 --bake-samples 64 --bake-extra 0`. Final: `--density per-mesh --resolution 512 --max-splats 8000000 --bake-samples 256` |
| Shadow catcher | activado | una segunda conversión con `--shadow-catcher`: la sombra sobre el suelo que encuentra la conversión (la malla plana más grande en la base de la selección) como gaussianas dibujadas en negro. Si no encuentra suelo lo notifica y la nube del objeto se queda |
| Result | Relit cloud | un Empty por nube en una colección `<malla>_splats`, cuyo `athenea_cloud` nombra el archivo; el hook de exportación lo referencia bajo el Empty y hdAthenea lo reilumina cada frame. O Blender splats: las nubes aplanadas bajo el world de la escena (`athenea flatten --format usdc`, abajo) e importadas como la nube de puntos de Gaussian splats propia de Blender, iluminada una vez, que dibuja cualquier motor |
| Directory | `//splats/` | donde se escriben los archivos (abajo) |
| Hide the meshes | activado | las mallas convertidas, y el suelo al que sustituye un catcher, se ocultan en el viewport y en los renders |

El subpanel *Overrides* (cerrado) tiene lo que si no decide el motor:
*Gaussians* (`--max-splats`, 0 el de la calidad), *Surroundings* (activado:
las demás mallas visibles de la escena se exportan como lo que encuentran los
rayos del bake; desactivado, la selección sola y sin catcher), *Thin glass* y
*Solid glass* (listas separadas por comas, `--thin-glass` / `--solid-glass`
un material cada uno) y *Skinned* (`--skinned`, los armatures exportados con
su animación; el transfer guardado en el marco de cada gaussiana; no con
Blender splats).

Los archivos, para una selección llamada `<malla>` en `<blend>.blend`:
`<blend>_<malla>_object.usdc` (la selección, bajo `/object`),
`_surroundings.usdc` (las demás mallas, bajo `/surroundings`), `_mesh.usda`
(el stage que subcapa ambos, con el eje vertical y la unidad del primero),
`.usdc` (la nube), `_catcher.usdc` y `_catcher_stage.usda` (el catcher), y
con Blender splats `_flatten_stage.usdc` y `_flat.usdc`. Donde el archivo de
una nube dibujada sigue abierto en el USD de Blender, la conversión siguiente
se llama `<malla>_2` y así.

La conversión corre en un hilo; el panel, la barra de estado y el cursor de
progreso la siguen -- la parte entre paréntesis de las líneas `transfer N of
M gaussians (P%)` de mesh2splat mientras se hornea el transfer, sus fases si
no -- y la ventana sigue viva. **Esc** la para: mesh2splat mira entre los
trozos del transfer y entre niveles de detalle, y devuelve 4. Sus avisos se
notifican y sus líneas se imprimen en el terminal; un error se notifica con
la última línea del comando (la salida 3, la GPU sin memoria, lo dice). En
segundo plano (`blender -b`, un script) `bpy.ops.athenea.mesh_to_splats()`
corre hasta el final antes de volver.

**Gaussianas hacia fuera.** *File > Export > Gaussian Splats (Athenea)* y el
*Export Gaussian Splats* de la barra lateral escriben las nubes visibles de
la escena (sus Empties; *Selected only* para las seleccionadas) bajo su world
y sus luces con `athenea flatten` (2.8.1) dentro de Blender
(`athenea_flatten`): SPZ (versión 3), PLY o glTF (`.glb`), *Exposure* como
`--exposure`. El stage que lee es la exportación USD de Blender de los
Empties, las luces visibles y el world, `<archivo>_flatten_stage.usdc` junto
al archivo.

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
| `athenea:meshLights` | bool | `true` | las mallas emisoras y las prims con `MeshLightAPI` son luces que el frame muestrea a través de un árbol de clusters (§4.3, *Luces de malla*); apagado, el path tracer las toma de una tabla plana y el raster muestra su brillo |
| `athenea:meshLightSamples` | int | `0` | raster: 0 ilumina las mallas por las luces de malla en forma cerrada; de 1 a 64, por tantas muestras por píxel a través de su árbol, cada una con un rayo de sombra (§4.3, *Luces de malla*) |
| `athenea:domeVisible` | bool | `true` | toda técnica: se dibuja un dome detrás del frame; apagado, los píxeles que nada cubre quedan transparentes (alfa 0), como pide la película transparente de un host, y el dome sigue iluminando el frame |
| `athenea:screenSpaceReflections` | bool | `false` | raster: las mallas que reflejan muestran lo que tienen las mallas del frame a lo largo de su reflejo (§4.3, *Reflejos en espacio de pantalla*) |
| `athenea:meshAntialias` | int | `0` | raster: los bordes de las mallas se resuelven de 4, 8 o 16 muestras por píxel, un número intermedio redondeado hacia arriba; 0 y 1 dibujan una muestra por píxel (§4.3, *Bordes de malla antialiasados*) |
| `athenea:splatView` | int | `0` | el raster: un parámetro de las gaussianas dibujado en lugar de su luz, mezclado por su opacidad -- 1 albedo, 2 roughness (la que lee el especular), 3 metallic, 4 transmission, 5 ior (`ior - 1`, un gris), 6 normal (del mundo, `n / 2 + 1/2`), 7 opacidad, 8 tamaño (el eje mayor, una rampa de azul a 10 µm a rojo a 10 cm, una década cada cuarto), 9 visibilidad del cielo (el cielo visto por el transfer ponderado por el coseno, 1 sin nada delante), 10 rebote (la mitad indirecta bajo un cielo blanco), 11 celdas abiertas (la fracción de direcciones de un transfer TX que salen de la escena), 12 emisión, 13 campo reflejado (su media bajo un cielo blanco), 14 la luz que lleva la nube, 15 la roughness propia del material (0 donde no la lleva); una nube que no se reilumina muestra su luz, su opacidad y su tamaño y nada más. 0 dibuja la luz |
| `athenea:glassShadows` | bool | `true` | rt y sus bakes: el rayo de sombra de una luz sigue a través de hasta 4 caras de vidrio cuya transmisión es una delta (una pared fina; un sólido de rugosidad especular constante por debajo de 0.03), teñido por cada una; apagado, el vidrio lo detiene |
| `athenea:rayMargin` | float | `1e-3` | rt y sus bakes: cuánto se separa de su superficie un rayo que no cruza nada, como fracción de la distancia de la posición al origen (a lo largo de la normal y a lo largo del rayo, con un TMin igual de largo), de 1e-6 a 0.1. Una escena en centímetros o lejos de su origen pide menos: con el valor por defecto, a 2000 unidades, la luz pasa por toda junta cóncava a menos de 2 unidades (Sponza: `2e-5`) |
| `athenea:denoise` | bool | `false` | denoise al terminar de reunir |
| `athenea:motionBuckets` | int | `4` | rt: rodajas del obturador, de 1 a 8 |
| `athenea:antialias` | bool | `true` | un desplazamiento subpíxel por pasada |
| `athenea:splatTransferIndirect` | bool | `true` | una nube con transfer suma su mitad rebotada |
| `athenea:splatDisplayBlend` | bool | `false` | raster: cada splat se codifica en sRGB antes de la mezcla y el píxel se decodifica después, como dibujan Spark, SuperSplat y el rasterizador de 3DGS |
| `athenea:splatReflections` | bool | `false` | rt: una gaussiana refleja la nube a la que pertenece, un rayo cada una |
| `athenea:splatShadows` | bool | `false` | rt: una nube relit se sombrea a sí misma |
| `athenea:cloudShadows` | bool | `true` | el mapa de transmitancia de una nube en cada luz |
| `athenea:domePrefiltered` | bool | `true` | raster: un domo ilumina una malla prefiltrado, sin muestra ni rayo |
| `athenea:cloudShadowResolution` | int | `1024` | texels por lado, por luz |
| `athenea:cloudShadowTerms` | int | `0` | 1 es solo el total; 3, 5, 7 añaden pares de Fourier; 0 deja decidir a quien recibe |
| `athenea:cloudShadowDensity` | float | `1.0` | multiplicador de la profundidad óptica de la nube |
| `athenea:volumeCacheMB` | int o float, MB | `2048` | rt: las rejillas de volumen que se guardan leídas en el host; sale primero la dibujada hace más tiempo. Las rejillas que dibuja el frame se guardan diga lo que diga, y 0 guarda solo esas (§4.3, *Volúmenes*) |
| `athenea:volumeFilter` | token | `trilinear` | rt: cómo consulta el medio su densidad entre vóxeles: `trilinear`, `nearest` o `stochastic` (un vóxel sorteado con los pesos trilineales); otro nombre se dice una vez y se usa `trilinear` (§4.3, *Volúmenes*) |

Leídos pero no declarados:

| Ajuste | Tipo | Dónde | Qué significa |
|---|---|---|---|
| `athenea:shutter` | double2 | prim de settings | apertura y cierre, en frames |
| `athenea:lens` | double2 | cámara | radio de apertura y distancia de enfoque |
| `athenea:exrHalf` | bool | prim de settings | escribir los productos en media en vez de float |
| `athenea:colourHalf` | bool (o un int, distinto de cero) | el host, a través del delegate | el AOV `color` es de media precisión (`Float16Vec4`) en vez de float; para un host que copia el color por memoria del host para mostrarlo (el viewport de Blender). Se lee cuando el host pide el formato del AOV, así que se pone antes del primer frame. Los grupos de luz y las demás salidas siguen en float |
| `athenea:disableMotionBlur` | bool | settings o producto | para ese producto |
| `athenea:disableDepthOfField` | bool | settings o producto | para ese producto |
| `athenea:lightGroup` | string | un prim de luz; o una malla, como `primvars:athenea:lightGroup` (constante) | el grupo bajo el que se reúne su aportación; en una malla, su grupo de emisión: lo que emite, visto e iluminando (§4.3, *Luces de malla*), 31 grupos así por frame |
| `athenea:lightState` | string | prim de settings, o un host | el estado de luces cuyos objetivos toman los grupos de lámparas de una escena; vacío: cada uno en su `initialOn` (§3.6). Lo lee `StageRenderer`, no hdAthenea bajo otro host |
| `athenea:lightLevel:<grupo>` | float | prim de settings, o un host | el nivel de ese grupo de lámparas sobre el del estado: 0 apagado, 1 su radiancia; para un grupo de emisión, lo que emiten sus emisores se multiplica por él, visto e iluminando |

### 4.3 Los schemas de API

Ocho schemas codeless de API, que se aplican a un prim como cualquier otro.
Todos los atributos son primvars, así que heredan hacia abajo.

**`AtheneaSplatLightingAPI`** — si la nube la ilumina la escena en vez de enseñar
la radiancia que lleva.

| Atributo | Tipo | Por defecto |
|---|---|---|
| `primvars:athenea:splat:relight` | bool | `false` |
| `primvars:athenea:splat:litBody` | bool | `false` |
| `primvars:athenea:splat:catcher` | bool | `false` |
| `primvars:athenea:splat:linear` | bool | `false` |
| `primvars:athenea:splat:surfel` | bool | `false` |
| `primvars:athenea:splat:coverage` | float[] ‹1 por gaussiana› | — |
| `primvars:athenea:splat:hardness` | float[] ‹1 por gaussiana›, 2 a 32 | — (2) |
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
| `primvars:athenea:splat:materialRoughness` | float[] ‹1 por gaussiana, de 0 a 1, o -1› | — |
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
`materialRoughness` es la rugosidad del propio material de la gaussiana, antes
de que `--specular-filter` ensanchara la `roughness` guardada por la
dispersión de normales bajo la gaussiana; -1 donde el material la lee de un
mapa. `athenea mesh2splat` la escribe. Donde vale al menos 0.3, el campo
reflejado de una transferencia TX responde de lo que el propio lóbulo del
pulido encuentra cerrado -- una bola muestra su suelo --, y por debajo, o en
una nube sin él, esa parte queda oscura.
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

**`AtheneaLiveSourceAPI`** — una nube dibujada desde un stream en vivo de
páginas SPZ que productores del mismo proceso empujan, una época cada vez, al
stream de este nombre (`lod::PageQueue::named`). Un frame dibuja la última
época cuyas páginas están todas en el dispositivo, nunca media; una época sin
sitio junto a la que se muestra, o con una página que no se descomprime, se
descarta con un aviso. Sólo la dibuja el rasterizador, sin niveles de detalle. Los prims que nombran
un mismo stream lo comparten y muestran la misma época. Con `latest`, las
épocas que esperan se juntan en la más nueva (la última página de cada chunk)
y la siguiente se carga mientras se coloca una. Un stream que nadie lee guarda
solo su época completa más nueva: una escena que lo nombra después empieza
desde ahí. Una época está completa cuando tiene una página para tantos chunks
distintos como dice; cada página se dibuja con el antialiasing que dice su
propio SPZ.

Con `sync = clock` un frame muestra la época completa más nueva cuyo `ptpTime`
(nanosegundos) ha llegado en el instante del frame -- el reloj con genlock del
anfitrión (`StageRenderer::setLiveClock`), o el reloj del sistema si no lo hay
-- y las épocas completas posteriores esperan, ocho como mucho: pasado eso se
descarta la más antigua con un aviso. Toda época que ya toca se aplica en
orden, así que un reloj que salta hacia delante muestra la más nueva que toca.
Uno que salta hacia atrás deja la época mostrada como está: una época es un
cambio sobre la anterior y no se deshace; la siguiente espera a que el reloj
vuelva a llegar a ella. Las épocas se colocan en el dispositivo antes de su
instante según haya sitio en el almacén (`primvars:athenea:stream:budget`: el
valor por defecto, dos épocas, guarda una por delante), así que cada una se
muestra en el mismo frame en que llega su instante; una que no se pudo colocar
a tiempo se muestra en cuanto lo está, y se cuenta como tardía.

| Atributo | Tipo | Por defecto |
|---|---|---|
| `primvars:athenea:live:stream` | string | —, no se dibuja nada |
| `primvars:athenea:live:chunks` | int | `64`; los ids de chunk de las páginas van de 0 a este menos uno |
| `primvars:athenea:live:chunkSplats` | int | `65536`; las gaussianas que como mucho tiene una página |
| `primvars:athenea:live:degree` | int 0..3 | `3`; armónicos que se guardan |
| `primvars:athenea:live:sync` | token `latest` \| `clock` | sin autorar: lo que pidió `stream_open`, si no `latest` |
| `primvars:athenea:live:loaders` | int 1..64 | `4`; hilos que leen y descomprimen las páginas |
| `primvars:athenea:stream:budget` | int64 ‹gaussianas› | `0`: el almacén crece con lo que se envía (sitio para la época mostrada y dos más), hasta tres veces `chunks` × `chunkSplats` |
| `extent` | float3[2] | —; los límites del stream hasta que se muestra una época, cuyos límites de página son entonces los suyos |

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
captura, sRGB, salvo que el prim diga otra cosa con
`primvars:athenea:splat:linear` (bool, constante), como hace un
ParticleField -- que el add-on escribe en una nube que trajo una conversión.

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

**Volúmenes**: se trazan con path tracing (`rt`); el raster no dibuja ningún
prim Volume, pero dibuja un medio de gaussianas (abajo). Cada rejilla que nombra un field se lee una vez por fichero y nombre de
rejilla y se guarda leída en el host, en una caché de `athenea:volumeCacheMB`
(2048 MB por defecto): pasado ese tamaño, las rejillas que el frame no dibuja
salen, primero la dibujada hace más tiempo, y una que se vuelve a dibujar se
vuelve a leer. Un `filePath` muestreado en el tiempo, un fichero por frame,
guarda lo que permite el presupuesto y no todos los frames mostrados. El
dispositivo guarda solo las rejillas del frame. Un Volume bajo un
`PointInstancer` o una instancia nativa se dibuja en cada instancia, y todas
leen la misma rejilla, dispuesta una sola vez. El light linking vale en un
medio como en una superficie: una luz cuya `collection:lightLink` deja fuera
un Volume no ilumina nada de su medio; su sombra no se enlaza (un volumen
atenúa toda luz que proyecta sombras). Un level set (una rejilla de OpenVDB
de clase `level set`, como una superficie convertida en volumen) se lee como
una niebla: densidad 0 fuera, 1 desde la mitad del ancho de la banda hacia
dentro, y una rampa lineal entre ambas -- `clamp(-distancia / semiancho, 0,
1)`, siendo el semiancho el valor de fondo de la rejilla -- con su interior
lleno. Una rejilla se lee en su propio tipo de valor: float (también una que
el fichero guarda en half), double, int32, vec3f, vec3d y máscara. Leída como
densidad, una rejilla vectorial es su longitud, una de enteros su valor y una
máscara 1 donde está activa. Las rejillas bool, int64 y de vectores enteros
se rechazan con un error que nombra el tipo, y un level set ha de ser de
floats.
El `fieldName` de un `OpenVDBAsset` nombra las rejillas de las que lee, en
el orden del fichero; `fieldIndex` cuenta entre las rejillas de ese nombre,
desde 0; `fieldDataType` (half, float, double, int, half3, float3, double3,
mask) ha de coincidir con la rejilla que elige el índice o, sin índice,
elige la primera de ese nombre que lo tenga; `vectorDataRoleHint` (None,
Point, Normal, Vector, Color) dice qué son los valores de un field
vectorial, por encima del tipo vectorial del propio fichero. Un field que no
nombra ninguna rejilla del fichero, o una de otro tipo, se rechaza con el
motivo, dicho una vez, y su Volume no se dibuja si es la densidad. Un
`Field3DAsset` se acepta y se rechaza al leerlo -- dicho una vez -- porque
ninguna compilación de este motor lee todavía Field3D. Un Volume guarda
todos los fields que nombra, hasta ocho (la densidad primero, luego en el
orden de Hydra; el que pasa del octavo se dice y no se lee), cada uno por su
nombre en el dispositivo; el medio se sombrea solo con su densidad, y los
demás fields esperan al sombreado que los lea.
La densidad se consulta entre vóxeles según dice `athenea:volumeFilter`,
con los valores de los vóxeles en coordenadas de índice enteras, como en
OpenVDB: `trilinear`, por defecto; `nearest`; o `stochastic`, uno de los
ocho vóxeles alrededor del punto sorteado con los pesos trilineales -- más
ruidoso a igual número de caminos, el valor trilineal en esperanza, una
consulta cada vez.

**Un medio de gaussianas** es un ParticleField con
`primvars:athenea:splat:volume = true` y un `primvars:athenea:volume:sigma`
por gaussiana (lo que escribe `athenea vdb2splat`, §2.8.3). El raster lo dibuja
como un medio participante: el alfa de cada gaussiana en un píxel es
1 − exp(−τ), con τ la profundidad óptica exacta del rayo del píxel a través de
ella (la cuerda que cortan sus 3,5 desviaciones, recortada en la cámara y en
la capa opaca bajo los splats), y su color la radiancia que un frame hornea
para ella. Sus gaussianas se ordenan y se mezclan con todos los demás splats.
Las `opacities` del fichero no se leen. Las constantes del medio son primvars
del field, con los nombres del Volume donde un Volume tiene lo mismo:

| Atributo | Tipo | Por defecto | Significado |
|---|---|---|---|
| `primvars:athenea:densityScale` | float | `1` | de densidad a extinción por unidad de mundo: lo que multiplica la sigma de cada gaussiana |
| `primvars:athenea:albedo` | color3f | `(0.8, 0.8, 0.8)` | el albedo de scattering, por el de cada gaussiana (`primvars:athenea:volume:albedo`, 1 donde el fichero no tiene) |
| `primvars:athenea:anisotropy` | float | `0` | la g de Henyey-Greenstein |
| `primvars:athenea:volume:emission` | color3f | `(0, 0, 0)` | radiancia de la parte que absorbe: el medio emite (1 − albedo) por ella, como el `uniform_edf` de un Volume |
| `primvars:athenea:volume:emissionScale` | float | `0` (sin fuego) | una gaussiana brilla como un cuerpo negro a su temperatura por esto |
| `primvars:athenea:volume:temperatureScale` | float | `1` | kelvin por unidad de la temperatura del fichero |
| `primvars:athenea:volume:temperatureOffset` | float | `0` | kelvin que se le suman |
| `primvars:athenea:volume:octaves` | int, de 1 a 16 | `4` | las octavas del scattering múltiple, la primera el simple; 1 es solo scattering simple |

El cuerpo negro es el color CIE de la ley de Planck a luminancia uno por
σ_SB T⁴ / π · 10⁻⁶ (la convención del Principled Volume de Cycles): unos 0,09
a 1500 K y 0,7 a 2500 K. Un frame hornea la radiancia de cada gaussiana: la
emisión; el scattering simple de cada luz, con la transmitancia hacia la luz
leída del mapa de sombras de nubes en el centro de la gaussiana (el medio
proyecta en el mapa y lo lee, con tres pares de Fourier; el mapa es el que
leen las mallas y las nubes TX, así que el medio les hace sombra y ellas a
él); el color de una cúpula por la transmitancia media de las direcciones de
cúpula del mapa; y `octaves` de scattering múltiple (Wrenninge: la extinción
hacia la luz, la energía y la anisotropía a la mitad en cada una). La
transmitancia de una luz distante es la media de tres lecturas a lo largo de
ella a través de la gaussiana.
Una cúpula con textura ilumina un medio con su color, no con su imagen. El
medio se dibuja solo a través de una cámara pinhole (una ortográfica lo deja
fuera, dicho una vez), y un frame trazado solo de splats (`rt` sin mallas ni
Volume) lo deja fuera, dicho una vez; con mallas, `rt` lo compone como compone
cualquier nube.

**Grupos de luces**, que todavía no declara ningún schema. Las lámparas de una
nube son un primvar por grupo en su field, y sus grupos y estados son
atributos en prims Scope -- hijos de un Scope `LightGroups` y de uno
`LightStates` en cualquier lugar de la escena, ganando el primero de cada
nombre (§3.6):

| Atributo | En | Tipo | Por defecto |
|---|---|---|---|
| `primvars:athenea:splat:lightLayer:<grupo>` | el field | color3f[] ‹1 por gaussiana, lineal, por unidad de radiancia› | — |
| `primvars:athenea:splat:lightLayer:emission:<grupo>` | el field | color3f[] ‹1 por gaussiana, lineal, al nivel 1 del grupo› | — |
| `primvars:athenea:splat:lightCatcher` | el field | bool | `false` |
| `athenea:lightGroup:emissionLuminance` | un grupo | float | `1` (`:radiance` donde falta este) |
| `athenea:lightGroup:emissionColor` | un grupo | color3f | `(1, 1, 1)` (`:color` donde falta este) |
| `athenea:lightGroup:initialOn` | un grupo | bool | `false` |
| `athenea:lightState:target:<grupo>` | un estado | double | un grupo no nombrado está a `0` |

Una nube lleva como mucho 16 capas; una capa que no es tres floats por
gaussiana se deja fuera con un aviso. Un light catcher dibuja sólo sus capas,
sumadas sobre lo que haya detrás, y no proyecta nada.

**Placas**, que tampoco declara ningún schema: una malla con
`primvars:athenea:plate` es una imagen puesta sobre el frame terminado, no
una superficie. Sus píxeles son los que cubren sus triángulos; muestra ahí la
primera textura de su material, en su `st`, y pone su color sobre lo que tiene
el frame según el alfa de la textura -- una mezcla, no un recorte -- donde
ninguna superficie de la capa de mallas queda más cerca. Así una imagen de
cámara keyeada deja ver el plató a través de su alfa y queda detrás de lo que
tiene delante.

| Atributo | En | Tipo | Por defecto |
|---|---|---|---|
| `primvars:athenea:plate` | una malla | int, constante ‹1 alfa directo, 2 color premultiplicado por él› | `0`, una malla normal |

El material sólo dice dónde está la imagen: su primera textura (también un
slot `aofx://`, que un host actualiza cada frame), leída en el espacio de color
y con el wrap que dé. No se sombrea nada de él, la exposición no toca la
placa, y cuesta una lectura de textura por píxel cubierto (0.28 ms sobre un
frame entero de 1920 x 1080 en una RTX 3070). Sólo la dibuja la ruta de
raster, después de todo lo demás, una placa sobre otra en el
orden de sus rutas; no sabe nada de los splats, que no guardan profundidad, así que tapa
una nube allí donde queda delante de las mallas; no escribe ningún AOV ni
profundidad; una placa instanciada no se dibuja, con un aviso, ni tampoco una
cuyo material no lee ninguna textura.

**Superficies sin luz**: una malla con `primvars:athenea:unlit` se dibuja y
queda detrás de lo que tenga delante como cualquier malla -- tapa, escribe
profundidad --, pero su sombreado es una lectura de textura: la primera de su
material, en su primvar `st_lightmap` si la tiene y en `st` si no, sin luz,
sin lóbulos y sin dome. Una superficie cuya luz se horneó en una textura (un
lightmap) se muestra tal como se horneó. Un material que no lee ninguna
textura muestra el `displayColor` de la malla.

| Atributo | En | Tipo | Por defecto |
|---|---|---|---|
| `primvars:athenea:unlit` | una malla | int (o bool), constante ‹distinto de 0: sin luz› | `0`, sombreada |
| `primvars:st_lightmap` | una malla | texCoord2f[] | ninguno: `st` |

La textura se lee en el nivel que pide la huella del píxel sobre esas
coordenadas, de la cadena de mips que recibe todo fichero de textura, y por lo
demás tal cual: no se aplican la escala, el bias ni otros nodos del material.
Los kernels de sombreado dejan esos píxeles: ni la lectura prefiltrada del
dome ni los grupos de luz los tocan. Un frame en el que todas las mallas van
sin luz no ejecuta ninguno de los kernels con luz -- ni lóbulos, ni rayos de
sombra, ni el kernel de sombreado, ni el dome --, tenga el stage las luces que
tenga; `ATHENEA_STAGES` los muestra a 0. Sólo en la ruta de raster.

**Reflejos en espacio de pantalla** (`athenea:screenSpaceReflections`, un
render setting, apagado por defecto; `StageRenderer::setScreenSpaceReflections`;
`athenea stage --screen-space-reflections`): después del sombreado y de los
domes, una superficie que refleja recorre su dirección de espejo por la
profundidad de la capa de mallas y suma lo que la capa tiene donde el
recorrido pasa por detrás de una superficie -- pesado por cuánto refleja y
atenuado por su rugosidad (nada a partir de 0.6), por el borde de la pantalla
y por la distancia (12 unidades de escena como mucho). Qué refleja: una
superficie sin luz, por sus primvars constantes; una sombreada, por el albedo
especular y la rugosidad que miden para ella los domes prefiltrados (un frame
iluminado sólo por domes).

| Atributo | En | Tipo | Por defecto |
|---|---|---|---|
| `primvars:athenea:reflectance` | una malla sin luz | float, constante ‹0..1, F0 (de Schlick)› | `0`, sin reflejo |
| `primvars:athenea:roughness` | una malla sin luz | float, constante ‹0..1› | `0.3` |

Lo que la pantalla no tiene no se refleja: lo que está detrás de la cámara o
pasado el borde del frame se desvanece, y los splats no se reflejan. Sólo en
la ruta de raster.

**Bordes de malla antialiasados** (`athenea:meshAntialias`, un render
setting, 0 -- apagado -- salvo que lo ponga un host; `athenea stage`, `athenea
view` y el add-on de Blender ponen 4, porque los AOV y los grupos de luz siguen
con una muestra por píxel y solo así suman la beauty en un borde; `StageRenderer::setMeshAntialias`;
`athenea stage --mesh-antialias`): la capa de mallas se sigue viendo y
sombreando con una muestra por píxel, y después un píxel junto al que se vio
otro triángulo se resuelve de 4, 8 o 16 submuestras (las posiciones
estándar de multisample): cada una toma el más cercano de los triángulos que
vio su vecindad de 3x3 que su rayo encuentra y el color ya sombreado en el píxel que lo vio,
el fondo donde no encuentra ninguno. Todas las rutas de visibilidad resuelven
igual, con luz o sin ella, y los ids, la profundidad y las demás salidas (AOVs,
Cryptomatte) siguen con una muestra por píxel. Lo que no puede resolver: un
triángulo que no vio ningún centro de píxel a su alrededor (un cable más fino
que un píxel), el borde del agujero de un recorte, y los splats compuestos
sobre la capa, cuya profundidad es la del centro del píxel. Sólo la técnica
raster; el path tracer antialiasa con sus propias pasadas
(`athenea:antialias`).

**Luces de malla** (`athenea:meshLights`, un render setting, encendido por
defecto; `StageRenderer::setMeshLights`): una malla cuyo material emite luz, o
una malla con la `MeshLightAPI` de UsdLux, ilumina la escena como lo hace una
luz, trazada y en el raster. Sus triángulos emisores se agrupan en clusters (por grupo de emisión, por
qué caras emiten, por el octante al que apunta su normal y por un cubo de una
rejilla de 16^3 sobre los límites de los emisores) y el path tracer elige un
triángulo a través de un árbol sobre los clusters, según cuánto puede importar
cada uno en el punto. Un triángulo es una luz cuando la luminancia de su
emisión en su centro, leída sobre toda su huella de textura, supera 1e-4; uno
por debajo lo siguen encontrando los rayos del material. Los clusters se
rehacen cuando cambian las mallas, sus materiales o cuáles de ellas emiten --
al cargar una escena, al editar un material -- y mientras un frame se
reproduce solo se mueven y se reponderan. Apagado, el path tracer toma sus
triángulos emisores de una tabla plana por potencia, como antes, y el raster
muestra solo su brillo. De una forma u otra, en un frame trazado con volúmenes
una colisión en un medio los muestrea como lo hace una superficie, y su luz le
llega a través de los medios que haya entre ambos.

El raster ilumina las mallas por ellas en forma cerrada
(`athenea:meshLightSamples` 0, el valor por defecto): cada tile de 16 x 16 del
frame toma un corte de hasta 16 entradas del árbol; un cluster se integra
sobre el rectángulo que lo representa -- su factor de forma para un lóbulo
difuso, el coseno transformado linealmente ajustado a GGX para un reflejo, el
de Zeltner para un sheen -- y una entrada que representa a muchos clusters
ilumina los lóbulos difusos como un punto de su potencia. La sombra son dos
rayos por píxel hacia los clusters, su cociente de la luz que llegó entre la
luz que había, filtrado sobre 9 x 9 píxeles de la misma profundidad y
orientación. Donde el dispositivo no tiene rayos inline (CUDA) no hay sombra
de luces de malla. Con 1 a 64 muestras por píxel, en cambio, cada una es una
muestra con sombra de un triángulo, como la dibuja la estimación del
siguiente evento del path tracer: con más ruido, y lo que la forma cerrada
aproxima. Ninguna de las dos ilumina la transmisión: es del path tracer.

Una nube reiluminada (`AtheneaSplatLightingAPI`) también se ilumina con ellas,
en el raster: cada celda de una rejilla de 32^3 sobre las nubes reiluminadas
toma un corte de hasta 4 clusters, y cada gaussiana integra los rectángulos de
esos clusters en forma cerrada -- su cuerpo por el factor de forma, su reflejo
y su coat por el LTC, su sheen como un cuerpo -- con la sombra de los bits de
su propia transferencia TX y sumando el rebote de la transferencia. Una nube
cuyos colores ya son luz (`primvars:athenea:splat:litBody`) se queda como
está. Una nube no proyecta sombra de una luz de malla sobre una malla.

Una malla en un grupo de emisión (`primvars:athenea:lightGroup`) emite su luz,
vista e iluminando, por el nivel de su grupo (`athenea:lightLevel:<grupo>`;
donde no hay ninguno, el peso que el rig de lámparas da a un grupo de ese
nombre, §3.6), y su luz va a la salida del grupo (`lightGroup:<grupo>`),
incluido lo que se ve del propio emisor. Un nivel cambiado vuelve a dibujar el
frame con los mismos clusters: ningún rebuild, unos pocos números subidos.

La luz indirecta de un grupo de emisión -- lo que sus emisores ponen en una
gaussiana tras un rebote, incluido el color que sube de un suelo rojo -- no la
calcula un frame; `athenea mesh2splat --light-groups` la hornea, todos los
grupos nombrados en una pasada, como una capa cada uno
(`primvars:athenea:splat:lightLayer:emission:<grupo>`, a nivel 1), y un frame
suma cada capa por el nivel de su grupo -- el mismo peso que lleva la luz
directa del grupo, así que un grupo atenuado atenúa las dos.

Una prim con `MeshLightAPI` emite la radiancia de su LightAPI --
`inputs:color` por `inputs:intensity`, `inputs:exposure`, y el cuerpo negro de
`inputs:colorTemperature` donde `inputs:enableColorTemperature` -- desde sus
caras frontales, o desde ambas donde la gprim es de doble cara, igual al verse
que al iluminar:

| `light:materialSyncMode` | Qué emite |
|---|---|
| `materialGlowTintsLight` (por defecto) | la radiancia de la LightAPI por la emisión del material enlazado donde el material emite; la radiancia sola donde no |
| `independent` | la radiancia de la LightAPI más la emisión del material |
| `noMaterialResponse` | la radiancia de la LightAPI sola |

`inputs:normalize` aún no se aplica: la radiancia no se divide por el área de
la malla. Una escena admite como mucho 127 juegos distintos de estos
controles; las prims que pasan de ahí toman el último juego, y se avisa una
vez.

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
de una docena; el bloque **Lights**; el bloque **Lamps**, donde la escena
lleva grupos de luces; el bloque **Sky**; y abajo, el
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

**Lamps** son los grupos de luces de la escena (§3.6): un combo *State* --
*(as each starts)* y cada estado que lleva la escena; elegir uno borra los
niveles puestos sobre el anterior -- y por cada grupo un interruptor y un
slider *Level*, de 0 a 4. Cada uno es un render setting (`athenea:lightState`,
`athenea:lightLevel:<grupo>`), que el siguiente frame convierte en los pesos de
los grupos: no se recarga ni se vuelve a hornear nada. Al pasar por encima de
un grupo dice su prim, su `emissionLuminance` y su `emissionColor`.

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
| `stream_open` | un stream en vivo que alimenta otro proceso; responde `{"ring", "bytes"}`, la memoria compartida donde escribir las páginas |
| `stream_commit` | las páginas de una época, cada una nombrada por dónde está en el anillo |
| `stream_close` | cierra el stream y quita su anillo |

**Un stream en vivo desde otro proceso.** Una escena nombra el stream en un
prim `AtheneaLiveSourceAPI` (§4.3); un productor que no está en el proceso del
servidor -- un script de Python, una aplicación anfitriona -- lo alimenta con
tres herramientas:
- `stream_open` `{name, chunks, ringBytes, sync}` crea un anillo de memoria
  compartida (memoria compartida POSIX en macOS y Linux, un file mapping con
  nombre en Windows; 64 MiB por defecto) y responde su nombre, que
  `multiprocessing.shared_memory.SharedMemory(name=...)` de Python abre tal
  cual. `sync` (`latest`, `clock`) es lo que muestra una escena que no lo
  dice.
- El productor escribe ficheros SPZ en el anillo y luego `stream_commit`
  `{name, epoch, epochPages, pages: [{chunk, op: add|replace|remove, offset, bytes, bounds, ptpTime}]}`:
  `offset` y `bytes` dicen dónde está cada página; `epochPages` son las
  páginas de la época en total (las de esta llamada por defecto), así que una
  época puede llegar en varias llamadas. Las páginas se copian del anillo
  antes de la respuesta: el anillo se puede sobrescribir en cuanto llega. Se
  rechaza la llamada entera por una página más allá de `chunks` o fuera del
  anillo, un `epoch` que no sea un entero desde 1 (como mucho 2^62), una op
  que no sea add, replace o remove, un chunk dos veces, o un `epochPages`
  menor que las páginas enviadas.
- Una sola sesión alimenta un stream: un segundo `stream_open` de un nombre
  abierto en otro sitio del proceso del servidor se rechaza. Un anillo se
  llama `al-<hash>-<pid>-<aleatorio>`, así que el anillo de un servidor caído
  nunca bloquea uno nuevo; en Linux los anillos de procesos que ya no existen
  se borran al abrir un stream.
- `bounds` (mín xyz, máx xyz) es lo que cubre la página; los límites del
  stream son la unión de los de sus páginas mostradas -- y, antes de mostrar
  ninguna época, el `extent` del prim.
- Un frame dibuja la última época cuyas páginas están todas en el
  dispositivo: el render que sigue a un commit puede mostrar aún la anterior.

`scripts/examples/live_producer.py` lo hace todo con la biblioteca estándar:
`live_producer.py --mcp <build>/bin/athenea-mcp --epochs 6 --out <dir> a.spz b.spz`
escribe una escena que nombra el stream (o toma `--stage`), hace commit de los
ficheros como chunks época tras época, y renderiza un PNG después de cada una.

**El `output` de `render` escribe dos ficheros distintos.** Un nombre que
acaba en `.exr` recibe los números del beauty, lineales. Cualquier otro nombre
recibe el frame **tal como se ve**: la AOV que pidió `aov`, pasada por el view
transform, en PNG. Es la única forma de conservar los colores de un
Cryptomatte, de una normal o de una profundidad, porque ninguno de ellos es un
color y los números del beauty no son los suyos; y es con lo que se rueda una
secuencia de una AOV, una llamada por frame sobre una escena que se queda
abierta.

**`render` para un programa que compone el frame.** `xform` es una cámara
propia tal como la guarda un compositor: de cámara a mundo, dieciséis números
fila a fila, mirando hacia −z con +y arriba, con `focal`, `haperture` y
`vaperture` en milímetros (la apertura vertical es por defecto la horizontal
al aspecto de la imagen) y `near` y `far` en unidades de la escena. `raw`
nombra un fichero al que van los números del frame en lugar de una imagen: el
color como cuatro floats de 32 bits por píxel y luego la profundidad (z de
vista, 0 donde no se dibujó nada) como uno, las dos con la fila de abajo
primero, lineales y premultiplicadas. La respuesta es entonces solo texto. Es
lo que pide el plugin OpenFX. `layers` añade, después de la profundidad y en
el orden dado, cuatro floats por píxel de cada una de: `P`, dónde está cada
píxel en el mundo (x, y, z, 1 donde se dibujó algo, ceros en el resto; pide
una cámara propia de la herramienta, `xform` o `eye`), y de `CryptoObject00` a
`CryptoObject02`, las capas del Cryptomatte como (id, cobertura, id,
cobertura). Un frame con matte responde una línea `manifest {...}`: los
nombres a los que corresponden sus ids, como los guardan los metadatos
Cryptomatte de un EXR.

**El cielo.** `sky` pone una imagen (un `.hdr` o `.exr` lat-long) en todas las
luces de domo de la escena, y una escena sin ninguna recibe antes el domo y el
sol por defecto; vacío devuelve la de la propia escena. `skyRotation` gira
todos los domos esos grados alrededor del eje vertical de la escena, y
`skyVisible` a false deja el fondo transparente mientras el cielo sigue
iluminando. Una nube reiluminable (un transfer) se ilumina con él. El primer
frame tras un cambio prepara el cielo nuevo (alrededor de medio segundo para
una imagen 4K); los frames siguientes cuestan lo mismo que antes. Una nube
girada por su prim lee su transfer en su propio sistema de referencia, con el
cielo girado a él una vez por frame, así que se reilumina como sin girar.

**Un parámetro en lugar de la luz.** `splatView` dibuja uno de los parámetros
de las gaussianas en lugar de su luz (`athenea:splatView`, por su nombre:
`albedo`, `roughness`, `metallic`, `transmission`, `ior`, `normal`,
`opacity`, `size`, `skyVisibility`, `bounce`, `openCells`, `emission`,
`field`, `baked`, `materialRoughness`; `light` la devuelve), cada uno un frame
de ese canal visto por la cámara, en un EXR como números o en un PNG como se
ve.

Los argumentos de cada herramienta se declaran en el protocolo y los enseña el
cliente, así que aquí no se repiten. La forma de una sesión es: `open_stage`,
luego `stage_tree` para encontrar una cámara o un prim, luego `render`, luego
`pick` sobre algo de la imagen. `settings` dice qué guarda la sesión en ese
momento.

### 6.1 El plugin OpenFX — Athenea 3DGS en Nuke y Natron

`Athenea.ofx.bundle` es una nube de gaussianas dibujada por el rasterizador de
athenea como generador OpenFX, para un compositor: un nodo **Athenea 3DGS**
(grupo *Athenea*) sin entrada, que dibuja una nube a través de su cámara al
formato del proyecto y en el tiempo del frame, como floats RGBA lineales y
premultiplicados, y sus capas. Se construye sobre el OpenFX de Natron
(`third_party/openfx`), cuya librería de soporte lleva la extensión
multiplano. Se compila con `ATHENEA_BUILD_OFX` (activado por defecto;
macOS y Linux) en `<build>/ofx/Athenea.ofx.bundle`.

El plugin no carga nada del motor. Arranca `athenea-mcp` (§6) como proceso
hijo la primera vez que renderiza y lo conserva, con la nube abierta, para los
frames siguientes; cada frame vuelve por un fichero del directorio temporal,
nunca como imagen. Un servidor que muere se arranca otra vez para el frame
siguiente. El bundle lleva el servidor en `Contents/Resources/bin` y los
shaders que compila en `Contents/Resources/shaders`, así que un bundle copiado
fuera del árbol de build funciona tal cual; `ATHENEA_MCP` nombra otro
servidor. Hay un servidor para todo el proceso del host, haga
el host las instancias del nodo que haga (Natron hace una por hilo de render),
así que una nube se carga una vez.

Para instalarlo, se copia o enlaza el bundle en un directorio que el host
busque: en macOS `/Library/OFX/Plugins`, o un directorio de `OFX_PLUGIN_PATH`.

| Parámetro | Valor | Por defecto | Notas |
|---|---|---|---|
| `cloud` | fichero | ninguno | un `.usdc` o `.usda` con la nube (un `ParticleField3DGaussianSplat`, como lo escriben `athenea convert`, `athenea mesh2splat` y los exports). Vacío no dibuja nada |
| `translate` | 3 números | `0 0 5` | dónde está la cámara, en las unidades de la nube |
| `rotate` | 3 números, grados | `0 0 0` | alrededor de x, y y z |
| `rot_order` | `XYZ` … `ZYX` | `ZXY` | el orden en que se aplican las rotaciones, la primera letra primero; ZXY es el de Nuke por defecto |
| `focal` | mm | `50` | |
| `haperture` | mm | `24.576` | |
| `vaperture` | mm | `0` | 0 se ajusta al aspecto del formato, contando el aspecto de píxel, como Nuke ajusta una cámara a su formato |
| `near`, `far` | unidades de la escena | `0.1`, `10000` | los planos de recorte |
| `time_offset` | frames | `0` | se suma al frame para dar el tiempo USD de la nube, para una nube animada (con skinning) |
| `sky` | fichero | ninguno | una imagen HDR (`.hdr` o `.exr` lat-long) en todos los domos de la escena; vacío conserva el de la propia escena |
| `sky_rotation` | grados | `0` | alrededor del eje vertical de la escena |
| `sky_visible` | sí/no | sí | apagado deja el fondo transparente, para componer sobre una plancha; el cielo sigue iluminando |
| `output` | `rgba`, `depth`, `P`, `CryptoObject00` … `02` | `rgba` | la capa que muestra en RGBA un host que no admite planos; `depth` es la z de vista en todos los canales |

La cámara se construye como Nuke construye la de un nodo Camera: las
rotaciones en el orden elegido y luego la traslación, mirando hacia −z con +y
arriba. Los knobs de un nodo Camera se enlazan a estos por expresión
(`translate`, `rotate`, `rot_order`, `focal`, `haperture`), así que la nube se
mueve con la cámara de la plancha. El nodo dibuja frames enteros: no renderiza
por tiles, y a su servidor va un render de una instancia cada vez.

**Capas.** Un host que admite planos (Natron) recibe todas las capas como
planos de la salida del nodo, diga lo que diga `output`: el color, `depth` (un
canal, `Z`: la z de vista, 0 donde no se dibujó nada), `P` (`X Y Z A`:
posición en el mundo, `A` 1 donde se dibujó algo) y de `CryptoObject00` a
`CryptoObject02` (`R G B A`: id, cobertura, id, cobertura). Solo se dibujan
los planos que pide el host. Nuke no admite de un nodo OpenFX planos que no
sean los suyos, así que allí `output` elige la capa que muestra RGBA; un nodo
por capa da todas. El manifest del Cryptomatte no llega al host (OpenFX no
lleva metadatos), así que un matte se elige por id y no por nombre.

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
| `minAlpha` | número 0–1 | `0` | las diferencias solo sobre los píxeles cuyo alfa de Reference llega a esto; 0 es todos. Va en la palabra de relleno del bloque de uniforms, así que un host que no lo envía compara como antes |
| `maskMode` | entero 1–2 | `1` | con un clip `Mask` conectado (opcional, como `Exclude`): 1 los píxeles cuyo alfa de Mask llega a `minAlpha`, 2 aquellos donde la luminancia de Reference está por debajo de `maskRatio` de la de Mask. Los píxeles cuyo alfa de `Exclude` llega a `minAlpha` no se comparan nunca |
| `maskRatio` | número ≥ 0 | `0.9` | la proporción de `maskMode` 2 |

Lo que adjunta, todo floats. Una cuenta son dos floats, `alto × 2^24 + bajo`,
cada uno exacto (alto es 0 por debajo de 16 777 216):

| Id | Valores |
|---|---|
| `source` | media R G B A, máximo R G B A, suma R G B A, los píxeles de la ventana (alto, bajo) |
| `reference` | lo mismo de Reference; solo si está conectada |
| `hdr` | relMSE, diferencia relativa p99, mayor diferencia relativa, la suma del error cuadrático relativo, píxeles (alto, bajo) |
| `codes` | p99 de 8 bits, máximo de 8 bits, píxeles por encima de `threshold` (alto, bajo), `threshold`, píxeles (alto, bajo) |
| `display` | PSNR del color codificado en sRGB (dB, 999 si son iguales), SSIM media de la luminancia codificada en una caja de 7 × 7, el error cuadrático medio del color codificado |

La diferencia relativa va en bins de un octavo de octava desde 2^-16, así que
su p99 y su máximo son la cota superior de un bin (en torno al 9 %). Las sumas
están para un host que divide en double, como hace `athenea compare`. Dos
imágenes de tamaños distintos se rechazan (`measure compares pictures of one
size`), y también una ventana sin nada dentro (`measure: an empty window`).

### 7.2 Vdb2Splat — `rt.sparrow.aofx.vdb2splat`

La extinción de un volumen disperso ajustada por gaussianas: lo que ejecuta
`athenea vdb2splat` (§2.8.3). Su único clip, `Grid`, no es una imagen de
color sino las hojas de la grid dispuestas por el host, 513 entradas por
hoja: el origen de la hoja en espacio de índices y luego sus 512 vóxeles en
el orden de NanoVDB (x el más lento, z el más rápido), cada uno (densidad,
temperatura, Cd como rgba8 en los bits de un float, activo 0 o 1). Sus filas
no pueden llevar relleno, ni las de la salida. La salida guarda cuatro
entradas por gaussiana en el espacio de índices de la grid desplazado medio
vóxel (el valor de un vóxel en ijk + 0,5; los de OpenVDB y del path tracer
están en ijk) -- (centro, σ),
(A00, A10, A11, A20), (A21, A22, temperatura, Cd rgba8), (las semiextensiones
de su caja a 3,5 desviaciones, 1) --, donde A es el factor de Cholesky
inferior de la precisión, G(x) = exp(−|Aᵀ(x − μ)|² / 2); su tamaño es el hueco
del presupuesto, un cuarto de sus entradas.

| Parámetro | Valor | Por defecto | Notas |
|---|---|---|---|
| `leaves` | entero | `0` | cuántas hojas tiene `Grid`; lo pone el host, y 0 se rechaza |
| `target` | entero | `20000` | el presupuesto, acotado por el hueco de la salida |
| `emIterations`, `iterations`, `attributeIterations`, `maxBlock`, `emissionWeight`, `initFraction`, `densifyEvery`, `densifyRounds`, `learningRate` | | como en §2.8.3 | las opciones del mismo nombre |
| `minStd` | número, vóxeles | `0.5` | la desviación menor que puede tomar una gaussiana; por debajo de medio vóxel, más o menos, una gaussiana muestreada en los centros de los vóxeles ya no tiene la masa que dice su integral, que es lo que suponen los pasos de EM |
| `pruneMass` | número | `1e-6` | se va una gaussiana cuya masa es menor que esta parte del total |
| `hasTemperature`, `hasColour` | booleano | `0` | si `Grid` las lleva (las pone el host) |
| `binned` | booleano | `1` | lo contrario de `--no-bins` |
| `binSlack` | número ≥ 0, vóxeles | `0` | `--bin-slack` |
| `checkerboard` | booleano | `0` | `--checkerboard` |
| `profile` | booleano | `0` | `--profile`: se espera a cada kernel y su tiempo se adjunta como `profile:<kernel>` (ms) |
| `splitShare` | de 0 a 1 | `0.5` | `--split-share` |
| `keepSlots` | booleano | `0` | cada hueco escrito en su sitio, los muertos a cero, para que el siguiente frame de una secuencia arranque de ellos; `gaussians` dice entonces la capacidad como escritas y las vivas en noveno lugar |
| `dt` | número | `1` | con los clips `Previous` y `Velocity` (opcionales): frames sobre los que el flujo lleva las gaussianas. `Previous` es la salida del frame anterior (`keepSlots`); `Velocity`, el flujo en los vóxeles de las hojas dispuesto como `Grid`, (u, activo) en vóxeles por frame |

Adjunta `gaussians`: las escritas, el presupuesto, las hojas, el error L2
relativo del ajuste sobre los vóxeles activos, las gaussianas que tenía el
primer corte, las iteraciones, el nivel del bloque más grueso (2^nivel
vóxeles de lado) y las menos gaussianas que permiten los bloques más gruesos.
Y `syncs`: las lecturas que esperó el procesador, la carga de los kernels en
ms y cuántas veces se construyeron las listas de las hojas.

## 8. Referencia

### 8.1 Variables de entorno

Las que cambian cómo se comporta una ejecución están en §1.2. El resto sirven
para mirar dentro de una ejecución, y son de quien trabaja en el motor más que
de quien lo ejecuta: están en
[development.es.md §3](development.es.md#3-trabajar-en-el-árbol).

### 8.2 Formatos

**Nubes que se leen.** `.ply` (3DGS), `.splat`, `.spz` (Niantic, necesita
zstd), `.sog` o un `meta.json` suelto (PlayCanvas, necesita libwebp), `.asg`
(anchored surfel gaussians, el bitstream v1 de 3Dspace, §2.4), `.athc` (el
contenedor por chunks de este motor), y el `ParticleField3DGaussianSplat` de
USD con las primvars `athenea:splat:*` (§4.3). §8.2.1 dice qué tipos de
gaussiana contienen.

Un `.spz` (versiones 2 a 4) es el contrato de entrada de splats del motor:
- **Ejes.** La extensión de sistema de coordenadas del fichero dice en cuál
  están guardados sus datos; sin ella es right-up-back, el del formato. Se
  leen right-up-back y right-down-front; cualquier otro sistema se rechaza.
- **Antialiasing.** El flag `antialiased` de la cabecera decide si la nube se
  dibuja con la compensación del filtro de Mip-Splatting, sea cual sea el
  valor por defecto del frame; los formatos que no lo dicen toman el valor por
  defecto (encendido). `--no-antialias` lo apaga para todas las nubes.
- **Memoria.** Los seis streams cuantizados siguen empaquetados hasta que el
  dispositivo los convierte en registros (65 bytes por splat en grado 3,
  frente a 236 en floats).

**Puntos que se leen.** `.ply`, `.xyz`, `.txt`, `.pts`, `.csv`, y los
`points3D.txt` y `points3D.bin` de COLMAP.

**Otras entradas.** Campos de volumen `.vdb` por OpenVDB; perfiles
fotométricos IESNA LM-63; y las texturas de materiales por los plugins de
imagen de OpenUSD, así que lo que lea esa compilación.

**Nubes que se escriben.** `athenea flatten` escribe `.ply` de 3DGS, `.spz`
(versiones 3 y 4; la 4 necesita zstd; cada uno lleva la extensión de sistema de
coordenadas diciendo right-up-back), `.glb` de glTF con
`KHR_gaussian_splatting`, y USD (§2.8.1). `athenea vdb2splat` escribe USD (§2.8.3).

**Volúmenes que se escriben.** `athenea vdb2splat --back` y `--make-fixture`
escriben grids float `.vdb`, volúmenes de niebla, por OpenVDB.

**Páginas que se escriben.** `athenea flatten --range` escribe un directorio de
páginas SPZ y su `stream.txt` (§3.7), que lee `athenea stream`.

**Vídeo que se escribe.** `athenea video-encode` escribe un `.athv`,
versión 2: una cabecera de 128 bytes (códec, bits, coeficientes de resto,
tamaño de chunk, columnas del mosaico, las gaussianas que caben como mucho en
una imagen, si las partes altas de la posición son planos, el antialiasing de
la fuente, si el color va en el croma, el id de la rendición, el ritmo y el
ritmo como fracción, los frames de origen, la calidad, el bloque y los bits
altos de la posición; el bit 8 de los flags un medio, su escala de densidad un
f32 en el offset 88, `--volume`) y un registro por imagen, cada uno completo en
sí mismo. Un registro empieza con un prefijo fijo -- magic `ATVF`, tipo de
registro (u16, 0), rendición (u16), frame de presentación (u32), tiempo PTP
(u64 en ns en el offset 12), los bytes del registro, la versión y los flags
(bit 0 imagen clave) --, así que un relé enruta con sus primeros 28 bytes;
después su índice, el frame de origen y qué porción suya es, dónde empiezan
sus gaussianas en el frame, cuántas son, el time code y la etiqueta de
timecode, los rangos (dos floats por atributo), los datos laterales con
deflate (zlib: las cajas de los bloques de cada chunk y después sus partes
altas de la posición), los conjuntos de parámetros HEVC en una imagen
clave y la unidad de acceso como unidades NAL precedidas de su longitud.
Little endian; los ficheros de la versión 1 se siguen leyendo.
`modules/video/include/athenea/video/Athv.h` es su especificación.

**Vídeo enviado como stream.** `scripts/examples/athv_stream.py` (biblioteca
estándar de Python) lee un `.athv` por rendición -- los mismos frames a otros
tamaños y calidades, cada uno codificado con su propio `--rendition` -- y
escribe sus registros en la salida estándar al ritmo de los ficheros, en
bucle, para que los lleve un transporte (un servidor WebSocket que reenvía
stdout, o `srt-live-transmit file://con "srt://HOST:9000?mode=caller&latency=120"`).
Cada mensaje es una longitud u32 little endian y esos bytes: un manifiesto
(magic `ATVM`, tipo de registro 1, rendición 0xFFFF, el mismo prefijo fijo, y
después una longitud u32 y JSON: el ritmo como fracción, los frames del bucle
y, por rendición, su id, gaussianas, grado SH, calidad, imágenes por frame y
su tamaño, Mbit/s medios, cadena de códec y hvcC), primero y después una vez
por segundo; y cada registro, byte a byte como lo guarda el fichero salvo el
frame de presentación (offset 8: sigue contando entre bucles, el mismo para
el mismo instante en todas las rendiciones) y el tiempo PTP (offset 12:
`CLOCK_REALTIME` en ns al enviarlo); y, con `--anc` o `--env`, datos
auxiliares (magic `ATVA`, tipo de registro 2, rendición 0xFFFF, la
disposición del manifiesto: JSON como el entorno bajo el que se dibujaron
los frames) justo después de cada manifiesto. El frame k de todas las rendiciones va
antes que el frame k + 1, en inicio + k / ritmo según el reloj monótono, así
que el ritmo no deriva.

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `athv` | rutas, obligatorias | — | una por rendición; los mismos frames al mismo ritmo, ids de rendición distintos |
| `--rate` | frames por segundo | el de los ficheros | |
| `--loops` | número | `0`: siempre | |
| `--manifest-every` | frames | el ritmo (un segundo) | |
| `--anc` | ruta JSON | ninguna | datos auxiliares (registro de tipo 2, magic `ATVA`, la disposición del manifiesto): se envían justo después de cada manifiesto y otra vez cuando cambia el fichero |
| `--env` | URL | ninguna | un bloque de entorno que nombra esa URL: el domo del gorrión, su latlong girado al espacio de la nube (`toWorld`) |
| `--spin-ms` | ms | `12` | cuánto antes del tiempo de un frame deja de dormir y espera activamente: un proceso dormido despierta hasta ~10 ms tarde (3 ms dejaron 7 ms de jitter, 12 ms 0,5 ms) |
| `--stats` | flag | no | una línea por segundo en stderr: frames por segundo, Mbit/s, con qué retraso salió el último frame |

`scripts/examples/athv_read.py` es el otro extremo: lee los mensajes de la
entrada estándar durante `--seconds` (10 por defecto) e imprime frames por
segundo, Mbit/s en total y por rendición, los huecos entre frames tal como
llegan y tal como los marcó el emisor (mediana y la mayor desviación respecto
al ritmo) y los frames que faltan; un mensaje con otro magic se cuenta y el
stream se resincroniza en el siguiente `ATVF`. Con `--info FICHERO` describe
en su lugar un `.athv` (versión 2): la cabecera y el mosaico que implica, los
tamaños de los registros, el primer registro, el perfil, el tier y el nivel
del SPS, y la cadena de códec de WebCodecs y el registro `hvcC` construido con
el VPS, el SPS y el PPS de ese registro (`--hvcc SALIDA` lo escribe).

**Una apariencia neural, `.athn`.** Lo que escribe `athenea neural-fit --save` y
lee `athenea render --neural`: los colores de una nube como un latente por gaussiana
y una red, junto al fichero de la nube y de ella gaussiana por gaussiana, en
su orden (no nombra ninguna nube; se comprueba el número). Little endian: `ATHN`, seis
uint32 -- versión (1), gaussianas, números por gaussiana (11), palabras por gaussiana (6),
pesos (1008, o 2016 para dos redes), flags (bit 0: iluminada; bit 1: split, los pesos de
dos redes, a la primera nunca se le dice la vista) -- luego los latentes como
halves, dos por uint32 y por tanto 24 bytes por gaussiana, y luego los pesos como float32.

**Una escena neural, `.aths`.** Un directorio: lo que escribe y lee `athenea neural-scene`
(§2.3.4). `scene.json` es el manifest y todas sus rutas son
relativas al directorio, así que una escena se mueve como se mueve un directorio.

```
<name>.aths/
  scene.json               el manifest
  pieces/<id>.usda|…       las gaussianas de una pieza: un ParticleField de USD o un fichero de splats
  pieces/<id>.athn         su apariencia neural, una vez ajustada
  pieces/<id>.seen         qué mirada vio primero cada gaussiana
  pieces/<id>.gone         qué mirada encontró ida cada gaussiana
  pieces/<id>.athg         las gaussianas que dejó un render del propio motor (neural-scene shoot)
```

Un `.seen` es un uint32 por gaussiana, little endian y nada más: el puesto en
`snapshots`, contado desde uno, de la mirada que la vio primero, 0 donde ninguna.
Un `.gone` es lo mismo para la mirada que vio a través de donde estaba la gaussiana, 0
mientras sigue en pie; todos los comandos dibujan una pieza sin las gaussianas que su `.gone`
sella.

Un `.athg` son dieciséis bytes -- `ATHG`, una versión (1), cuántas gaussianas y
cuántos floats cada una, uint32 little endian -- y después los registros,
float32. El registro de un render son 26 floats: la posición (3), el logit de
la opacidad, los logaritmos de las escalas (3), la rotación (w x y z), el primer
armónico como guarda un color una captura (3), la normal de la superficie en el
mundo (3), la fracción de metal, la roughness, la transmisión, y después la fila
del material en el frame que la creó más uno (0 para lo visto a través de
vidrio), su índice de refracción, 1 para una lámina sin interior, y lo que deja
pasar en un cruce (r g b).

La escena se conserva, no se sustituye: una mirada añade piezas y sella lo que
encontró ido, y nada se quita del disco. Así la escena como estaba en un
snapshot son las piezas cuyo `origin.snapshot` no es posterior, menos las
gaussianas cuyo sello en `.gone` no es posterior (`neural-scene render --at`,
`--snapshot`).

| Clave de `scene.json` | Valor |
|---|---|
| `format`, `version` | `"athenea-neural-scene"`, `2`; la versión 1 se lee tal cual (la 2 añadió `origin.snapshot`, `origin.footprint`, en un snapshot `samples`, `footprint`, `taken`, y `frames`), y un lector rechaza cualquier otra |
| `name`, `upAxis`, `metersPerUnit` | el nombre de la escena; `"Y"` o `"Z"`; metros por unidad |
| `source` | `stage`: de qué se convirtieron las piezas; `licence`: qué pide a quien lo use |
| `bounds` | `min`, `max`: la caja que envuelve las de todas las piezas, escrita y no leída |
| `pieces[]` | `id`; `prim` (de dónde salió); `cloud` (su fichero); `gaussians`; `bounds`; `transform` (objeto a mundo, dieciséis números por filas); `revision`, `movedAt`, `paintedAt`; `static`; `origin`; y, una vez ajustada, `appearance` |
| `pieces[].gone`, `removed` | su `.gone` y cuántas gaussianas vieron a través miradas posteriores; ausentes hasta que una lo hace. Retirar gaussianas sube la `revision` y el `movedAt` de la pieza |
| `pieces[].seen`, `discovered` | su `.seen` y cuántas gaussianas han encontrado las miradas; ausentes hasta que se toma una mirada |
| `pieces[].origin` | `kind`: `"converted"` (de geometría, entera), `"discovered"` (las gaussianas que vio una cámara) o `"generated"` (rellenada por otro sistema); `camera` y `time`: qué la encontró y cuándo; `snapshot`: el snapshot en que entró, contado desde uno (ausente: estaba antes de cualquier mirada); `footprint`: el lado en píxeles de la celda de la que se hizo cada una de sus gaussianas (las de un render) |
| `pieces[].appearance` | `file` (su `.athn`); `lit`; `split`; `fit`: `steps`, `against` (`"harmonics"`), `relMSE`, `p99`, `max` como los midió el dispositivo desde las cámaras held-out |
| `lights[]` | `name`, `kind` (`"distant"` o `"dome"`), `towards`, `radiance`: bajo qué se hicieron las piezas; se anota, y todavía no lo lee nada |
| `cameras[]` | `name`, `eye`, `target`, `up`, `focal` (mm) |
| `frames[]` | un frame que un render metió en la escena: `time`, `camera`, `width`, `height`, `samples` (caminos por píxel), `firstSnapshot` y `lastSnapshot` (sus miradas, contadas desde uno), `seconds`, `taken`. `neural-scene render --frame` lo dibuja de nuevo |
| `snapshots[]` | `time`, `camera`, `pieces`: una mirada a la escena y las piezas que añadió o cambió; la escena en un tiempo son las piezas de los snapshots hasta él. Una mirada tomada mientras su render reunía caminos tiene además `samples` (los caminos por píxel que tenía), `footprint` (el lado de sus celdas) y `taken` (cuándo, según el reloj, ISO 8601) |

`revision` sube cada vez que se escriben las gaussianas de una pieza o su apariencia,
y `static` dice que sus gaussianas no se mueven de un frame a otro: lo que un
renderer guarda de una pieza lo guarda mientras la revision se mantenga. `movedAt` y
`paintedAt` son las revisions en las que se movió por última vez y en la que cambió por última vez
aquello de lo que está hecha: un fit sube solo `paintedAt`, unas gaussianas nuevas suben las dos.
Un manifest sin estas claves se lee como convertido, movido y pintado en su
revision, y sin snapshots.

**Imágenes que se escriben.** OpenEXR, lineal premultiplicado, con la fila de
abajo primero dentro del motor y escrito con la de arriba primero como quiere
el formato. Un frame va en media por defecto con un canal `Z`; los canales de
un producto siguen a sus vars, y una capa Cryptomatte va siempre en float. PNG
se escribe como la vista previa de MCP y con
`athenea neural-scene render -o x.png`.

### 8.2.1 Tipos de gaussianas

Toda nube es el mismo registro en el dispositivo -- una posición, tres
tamaños, una rotación, una opacidad y sus armónicos (`GpuSplats`) -- y todos
los renderers la toman. Lo que distingue un tipo de otro es un flag o una
primvar en la prim (§4.3), y cada uno es independiente de los demás: una nube
TX con skinning, de surfels y en niveles de detalle es una sola nube.

**Por forma: qué cubre una gaussiana.**

| Tipo | Qué es | Lo hace | Lo marca | Se dibuja |
|---|---|---|---|---|
| Gaussiana 3D (3DGS) | un elipsoide: tres tamaños y una rotación | una captura; `athenea mesh2splat` (`--flatness` mayor que 0) | el valor por defecto | proyectada y mezclada por tiles (raster), o alcanzada por un rayo (path tracer, horneado de transfer) |
| Surfel (2DGS) | un disco plano: un tamaño exactamente 0 | `mesh2splat --flatness 0`, las fusiones planas de `decimate`, `splat-transfer` sobre el level set de una captura, un `.asg` | `primvars:athenea:splat:surfel` | el rayo de cada píxel corta su plano (2D gaussian splatting), así que no se hincha de canto |
| Surfel duro | un surfel cuya caída es exp(-(\|u\|^p + \|v\|^p)/2), p de 2 (una gaussiana) a 32 (casi una caja) | `decimate --hardness` | `primvars:athenea:splat:hardness` | como un surfel; un lector que no conoce p dibuja la gaussiana que dan sus momentos |
| Gaussiana volumétrica | un grumo de densidad en un medio, no una superficie | `athenea vdb2splat` (§2.8.3), el efecto Vdb2Splat | `primvars:athenea:splat:volume` | como un medio participante: absorbida y dispersada a lo largo de un rayo, sombreada por el mapa de sombras de nube |

**Por luz: qué significa su color.**

| Tipo | Qué guarda | Lo marca | Bajo una luz nueva |
|---|---|---|---|
| Radiancia | la luz con la que se capturó u horneó, en armónicos esféricos (grado 0 a 3); sRGB salvo `primvars:athenea:splat:linear` | el valor por defecto (una captura, un fichero 3DGS, la salida de `flatten`) | muestra lo que guarda; las luces del stage no lo cambian |
| Reiluminada (PBR) | un material por gaussiana: color base, metallic, roughness, transmisión e IOR, una normal, coat, sheen y specular (§4.3) | `primvars:athenea:splat:relight` | sombreada en cada frame por las luces y el domo del stage, como una malla con ese material |
| TX (transfer) | cómo responde el asset a la luz, trazado una vez: `transferDirect`, `transferIndirect`, `transferReflected`, `transferZonal`, con `shadowBits` para las salidas abiertas, y `curvature` y `thinWalled` para el vidrio | `mesh2splat --transfer`, `splat-transfer` | reiluminada en tiempo real por cualquier cielo, con las sombras e interreflexiones que trazó el horneado; el modo del producto |
| Emisiva | la luz que emite, un color por gaussiana; grupos de lámparas como capas (`lightLayer:<grupo>`) | `primvars:athenea:splat:emission`; `athenea lights merge` | suma su propia luz, escalada por su grupo |
| Shadow catcher | un parche en el suelo con la sombra (o la luz de las lámparas) que proyecta un objeto | `mesh2splat --shadow-catcher` / `--light-catcher`; `primvars:athenea:splat:catcher`, `lightCatcher` | oscurece (o ilumina) lo que tiene debajo, sin malla de suelo |

**Por movimiento.**

| Tipo | Qué se mueve | Lo marca |
|---|---|---|
| Estática | nada | -- |
| Con skinning | las gaussianas siguen un esqueleto; lo que ocluye cada parte se hornea por parte | `skel:skeleton`, `primvars:athenea:splat:jointIndices` / `jointWeights`; `athenea visibility` (§2.9) |
| Con muestras de tiempo | los puntos y atributos de la prim, frame a frame, como los autora USD | muestras de tiempo en la prim |
| Secuencia | una página por frame: páginas SPZ en stream (`athenea stream`), o frames como vídeo HEVC (`.athv`, §2.7.2) | un `stream.txt`, un `.athv` |

**Por tamaño: cuánto está en el dispositivo.**

| Tipo | Qué es | Lo marca |
|---|---|---|
| Entera | todas las gaussianas subidas | -- |
| Niveles de detalle | niveles fusionados sobre un orden Morton, cortados en cada frame por los píxeles que abarca una celda; los chunks entran y salen en stream (§3.3) | un `.athc`; `primvars:athenea:splat:lodTree`, `lodParent` de `decimate` |

### 8.3 Qué escribe cada comando

| Comando | Escribe |
|---|---|
| `athenea info` | un informe por la salida estándar |
| `athenea render` | un EXR: RGBA en media y `Z` |
| `athenea bench` | nada; los tiempos por la salida estándar |
| `athenea lift` | el EXR de `-o`: el frame de `--eye` con el key del matte, y su profundidad |
| `athenea neural-fit` | las medidas por la salida estándar; con `--save`, un `.athn`; con `-o`, frames EXR |
| `athenea neural-scene` | un directorio `.aths`: `scene.json`, las nubes de las piezas copiadas dentro, un `.athn` por pieza ajustada, un `.athg` por mirada de `shoot`; `render -o`, frames EXR o imágenes PNG |
| `athenea convert` | una escena USD, o un `.athc` |
| `athenea stage` | un EXR; o, con `--render-settings`, uno por producto |
| `athenea view` | nada, o un EXR con `--snapshot` |
| `athenea live` | un EXR por frame, numerado, cada uno con su timecode |
| `athenea stream` | nada, o con `-o` un EXR por frame mostrado |
| `athenea video-encode` | un `.athv`, y con `--hevc` el vídeo solo como Annex B |
| `athenea video-decode` | nada; los tiempos por la salida estándar |
| `athenea mesh2splat` | una escena USD con la nube |
| `athenea vdb2splat` | una escena USD con las gaussianas; con `--back`, un `.vdb`; con `--make-fixture`, un `.vdb` y nada más |
| `athenea flatten` | ficheros `.ply`, `.spz`, `.glb` y USD de las nubes aplanadas, la escena compuesta junto a ellos, y con `--validate` lo que escribe `mesh2splat --validate` |
| `athenea lightmap` | `lightmap_<atlas>.exr` (half, lineal, la fila de abajo primero) y `lightmapped.usda`, que pone el original como sublayer; sus materiales van bajo el prim raíz que comparten todas las mallas horneadas (`/<raíz>/AtheneaLightmaps`), así que referenciar ese prim los conserva |
| `athenea visibility` | el fichero de la nube, editado en el sitio o copiado |
| `athenea splat-transfer` | la escena aplanada en `--output`, con el transfer en `--prim` |
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
| `fetch-fox.sh [dir]` | el zorro de Khronos, por Blender, para una conversión con esqueleto |
| `sketchfab-to-usd.sh <zip> [nombre]` | un archivo de Sketchfab a un asset USD |
| `readme-images.sh [salida]` | las imágenes del README, desde el asset del gorrión |
| `remote-test.sh user@host [preset]` | compila y corre la suite en otra máquina y trae el log |
| `package-blender-addon.sh [build] [salida]` | la extensión de Blender para macOS o Linux como directorio y zip, con zstd y libwebp dentro (y el runtime de CUDA en Linux) y los ficheros de licencia de cada tercero (4.1.1) |
| `package-blender-addon.ps1 [build] [salida]` | lo mismo para Windows, con las DLL que carga hdAthenea, DXC y el runtime de CUDA (4.1.1) |
| `scripts/film/` | la película del gorrión: sus frames, su cámara y su sombra (`SPARROW` la carpeta de las escenas, `ATHENEA` el binario) |
| `bmw-to-usd.py`, `sparrow-*.py` | ayudas de autoría para esos assets |

La cabecera de cada script dice qué necesita y dónde deja las cosas.

## 9. Cuando algo falla

| Qué se imprime | Qué significa | Qué hacer |
|---|---|---|
| `no GPU device` (los tests se saltan) | no se pudo abrir dispositivo | mira `athenea info`; en Linux pon `ATHENEA_BACKEND` |
| `<prim> carries lucabRTrender's names (lrt:…), which athenea does not read` | una nube escrita antes del cambio de nombre: su rig, su relight y su campo de visibilidad están en `primvars:lrt:*`, así que se dibuja quieta y sin iluminar | `athenea migrate <fichero> -o <copia>` (`--recursive` para las capas que nombra), §2.12 |
| `colour: no colour space '<nombre>' in <config>; read as the file says` | una textura nombra un espacio de color que no conocen ni el config ni el studio config | corrige el nombre (`athenea info` dice si OpenColorIO está compilado); la textura se lee como si no se hubiera dado espacio de color |
| `no Measure bundle on the AOFX search path` | `athenea compare` no encontró `rt.sparrow.aofx.measure` | compila el target `athenea_aofx_measure`, o da su directorio con `--path` |
| `measure: an empty window` | `athenea compare --window` (o el `window` del efecto) no contiene ningún píxel | da X0 < X1 e Y0 < Y1 dentro de la imagen |
| un error de compilación de shader con una ruta | los shaders del disco no son los del binario | recompila, o apunta `ATHENEA_SHADER_DIR` al `shaders` de esta compilación |
| `this build reads no .spz` | faltaba zstd cuando se compiló este binario | recompila con zstd, o convierte la captura en otro sitio |
| `video: the stream's SPS says video range (16-235); video planes are full range` | se dio al decodificador de planos un stream HEVC cuyo VUI no marca rango completo | codifícalo con `athenea video-encode` (rango completo, '420f'), o vuelve a codificarlo en rango completo |
| `SPZ data stored in coordinate system N; this reader reads right-up-back (4) and right-down-front (6)` | la extensión de sistema de coordenadas del fichero nombra otro sistema | reexpórtalo en right-up-back o right-down-front |
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
