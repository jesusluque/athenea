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
presets de CMake. Aquí no se repite. Lo que importa después es el entorno en
el que corre el binario.

### 1.2 El entorno de ejecución

| Variable | Qué hace |
|---|---|
| `ATHENEA_SHADER_DIR` | de dónde se leen los shaders de Slang. Sin ella el motor busca un directorio `shaders` que contenga `athenea/` junto a la imagen desde la que se cargó su código, o hasta tres directorios por encima -- `<exe>/../shaders` para un programa, `<build>/shaders` para el plugin de Hydra cargado por usdview o Blender --, luego junto al ejecutable, y luego en el directorio con el que se compiló. `athenea info` imprime el que está usando. |
| `PXR_PLUGINPATH_NAME` | apunta una aplicación USD a `<build>/plugin/usd`, donde están el delegate de Hydra y los schemas codeless. La necesita cualquier host que no sea `athenea`. `athenea` registra por su cuenta `<su binario>/../plugin/usd` al arrancar, así que los schemas que aplica una nube convertida se escriben esté o no definida. |
| `ATHENEA_MATERIALX_ROOT` | un directorio que contiene los `libraries/` de MaterialX, que el compilador de materiales de hdAthenea lee en lugar de las bibliotecas que cargó el USD del host. Sin definir por defecto: se usan las del host. Para un host cuyo MaterialX es anterior al generador de Slang (Blender 5.3 trae 1.39.4: sin implementaciones `genslang` y con definiciones de nodo anteriores), se apunta a las de 1.39.5. Se lee una vez, cuando compila el primer material; los renderers propios del host conservan las suyas. |
| `AOFX_PLUGIN_PATH` | directorios extra de bundles AOFX, buscados antes que la ruta del sistema y antes que `--path`. |
| `ATHENEA_BACKEND` | qué dispositivo abrir, como un orden separado por comas: `metal,cuda,vulkan,d3d12`. Las palabras desconocidas avisan y se saltan. |
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

### 1.5 Códigos de salida, y dónde se imprime un error

Cada subcomando imprime sus errores en la salida de error y termina con `1`.
Una ejecución correcta termina con `0`. No hay más códigos: un pipeline debe
mirar el estado de salida y leer stderr, no analizar stdout, que lleva el
informe — tiempos, cuentas, la ruta escrita.

`-v` (o `--verbose`) antes del subcomando enciende el log de depuración, que
también va a stderr.

## 2. La línea de comandos

Una invocación es `athenea [-v] <subcomando> [opciones]`. Cada subcomando está
abajo con su tabla completa.

### 2.1 `athenea info` — el dispositivo, y qué sabe hacer

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
| `--colour-tolerance` | 0..1 | `0.05` | cuánto puede alejarse el color o la opacidad de una gaussiana de la que la sustituye antes de contar como distinta |
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
metallic, roughness, un transfer -- como una media. Metallic, roughness y transmisión también se comparan
como el color. Un array muestreado en el tiempo se funde muestra a muestra. Un
fichero de splats se escribe como una escena nueva, como la escribe
`athenea convert`.

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
| `--path-bounces` | entero | `1` | rt: rebotes tras el primer impacto |
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
| `--path-bounces` | entero | `4` | |
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
| `-o`, `--output` | ruta | `splats.usda` | `.usda`, `.usdc`, `.usd` |
| `--prim` | ruta de prim | todas las mallas | solo las que cuelgan de esa ruta |
| `--hide` | ruta de prim, repetible | ninguna | se deja fuera con todo lo que cuelga de ella, como invisible (opinión de sesión; el fichero no cambia) |
| `--resolution` | entero | `512` | celdas a lo largo del lado largo de la caja sobre la que se mide la densidad |
| `--lod-levels` | entero | `1` | niveles de detalle: la conversión otra vez a la mitad de resolución cada vez; `-o` pasa a ser la escena que los dibuja como una nube, cada nivel un `<nombre>_lod<n>.usdc` a su lado |
| `--density` | `per-model` \| `per-mesh` | `per-model` | qué caja es esa |
| `--cell-min` | número | `0`, derivado | unidades de mundo; por malla, lo más fina que puede ser una celda |
| `--cell-max` | número | `0`, derivado | unidades de mundo; lo más gruesa |
| `--max-splats` | entero | `2000000` | el presupuesto, de toda la escena |
| `--sigma` | número | `1.0` | anchura de la gaussiana en celdas; la de mesh2splat es 0.65 |
| `--flatness` | número | `0.1` | el tercer tamaño como fracción del menor de los otros dos |
| `--opacity` | número | `1.0` | de donde arranca cada gaussiana |
| `--glass-opacity` | número | `0.6` | lo que sigue parando un material que transmite del todo |
| `--opacity-cut` | número | `0.5` | por debajo de esto, un mapa de recorte dice que no hay superficie: el `opacity` de UsdPreviewSurface, el `opacity` de standard_surface o el `geometry_opacity` de OpenPBR conectado a una imagen |
| `--max-cells` | entero | `262144` | celdas como mucho que recorre un triángulo |
| `--texture-size` | entero | `1024` | un mapa se lee no mayor que esto; 0 lo lee a su tamaño |
| `--no-textures` | flag | apagado | ignorar los mapas; los materiales se quedan con sus valores constantes |
| `--normal-map-turns` | flag | apagado | el mapa de normales gira la gaussiana, no solo su sombreado. La normal de sombreado se escribe en ambos casos (`primvars:athenea:splat:normal`) |
| `--no-displacement` | flag | apagado | ignorar el displacement de los materiales: toda gaussiana se queda sobre la malla plana |
| `--displace-refine` | entero, 1 a 64 | `8` | donde el relieve estira una celda, partirla en como mucho este número de gaussianas en cada uno de sus dos ejes |
| `--simplify` | número, 0 a 1 | `0` (apagado) | un bloque de celdas cuyo color, metallic, roughness, recorte y normales varían no más que esto -- en el bloque y en un bloque más allá de cada lado, todo dentro de un triángulo -- se convierte en una gaussiana de su tamaño. Colores y recorte van de 0 a 1; las normales se comparan por la longitud de su diferencia, más o menos el ángulo en radianes |
| `--simplify-levels` | entero, 1 a 5 | `3` | el bloque más grande que puede fundir `--simplify` tiene 2^esto celdas de lado |
| `--no-camera` | flag | cámara añadida | |
| `--no-bake` | flag | bake encendido | llevar el material para ser relit, en vez de hornear la luz |
| `--bake-samples` | entero | `64` | caminos por gaussiana |
| `--bake-bounces` | entero | `3` | tras el primer impacto |
| `--bake-degree` | 0..3 | `2` | armónicos ajustados; 0 es un color |
| `--transfer` | flag | apagado | hornear cuánto cielo llega a cada gaussiana, en vez de la luz que llegó |
| `--indirect` / `--no-indirect` | flag | encendido | con `--transfer`: guardar también la mitad que rebotó |
| `--skinned` | flag | apagado | llevar el esqueleto; obliga a `--no-bake` |
| `--range` | `INICIO:FIN[:PASO]` | el rango de la escena | time codes que guarda una nube con esqueleto |
| `--default-lights` | flag | apagado | un dome y un sol para el bake, en una escena sin luces |
| `--time` | número | `0` | el instante en que se posa la escena y traza el bake |
| `--path` | directorio ‹repetible› | ninguno | directorios extra de bundles AOFX |

`--skinned` y un bake se rechazan juntos: una nube que se mueve no puede
llevar luz horneada en una pose, así que la conversión lo dice y conserva el
material.

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
kernel.

| Opción | Valor | Por defecto | Notas |
|---|---|---|---|
| `image` | ruta, obligatoria | — | un EXR |
| `reference` | ruta | ninguna | un EXR del mismo tamaño |
| `--window` | `X0 Y0 X1 Y1` | la imagen entera | píxeles en [X0, X1) × [Y0, Y1), filas contadas desde abajo; solo las medias, las diferencias son de toda la imagen |

Una media conserva el signo, y una diferencia no: un horno blanco que no debe
devolver más de 1, o un plano convertido que debe cubrir todos sus píxeles, es
una media.

```sh
athenea compare cloud.exr mesh.exr
athenea compare furnace.exr --window 192 192 320 320
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

**El presupuesto.** `--max-splats` es un techo de toda la escena, tomado en
orden de malla, así que un presupuesto corto conserva enteras las primeras
mallas y tira las últimas del todo. El log dice cuántas querían más de lo que
recibieron. Sube el presupuesto, o con `--density per-mesh` sube `--cell-min`
para que cada malla cueste menos.

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

**Una nube que se mueve.** `--skinned` construye las gaussianas en la pose de
bind y le da a cada una los joints que la llevan, así que la nube se deforma
al renderizar con el Skeleton al que está atada. Un bake se rechaza con él,
porque la luz horneada en una pose está mal en todas las demás.

```sh
athenea mesh2splat car.usda --density per-mesh --resolution 512 \
    --max-splats 20000000 -o car_gs.usdc
athenea mesh2splat bird.usda --skinned --resolution 1100 -o bird_gs.usdc
athenea visibility bird_gs.usdc --skeleton-stage bird.usda --skeleton-prim /World/Skel
```

La última línea es el tercer horneado: lo que la nube proyecta sobre sí misma,
por partes, para que un ala sombree el cuerpo en cualquier pose sin un rayo.
Edita el fichero de la nube en el sitio salvo que `-o` nombre otro.

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

**Sombras.** Las mallas sombrean por rayo en la ruta trazada. Una nube
proyecta a través de un mapa de transmitancia en cada luz, sin rayo ninguno:
`--cloud-shadow-texels`, `--cloud-shadow-density` y `--cloud-shadow-terms` lo
gobiernan, y `--no-cloud-shadows` lo apaga. `--splat-shadows` es la otra
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
1, que no tiene, se sigue leyendo.

Cómo se ve un presupuesto corto: los grupos cuyos chunks no han llegado
dibujan su gaussiana fundida, así que la nube está pero roma, y se afina según
aterrizan. `athenea stage` espera a que los streams se asienten antes de una
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
| `athenea:pathBounces` | int | `1` | rt: rebotes tras el primer impacto |
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
| `primvars:athenea:splat:metallic` | float[] | — |
| `primvars:athenea:splat:roughness` | float[] | — |
| `primvars:athenea:splat:transmission` | float[] | — |
| `primvars:athenea:splat:ior` | float | `0` |
| `primvars:athenea:splat:transferDirect` | float[] ‹9 por gaussiana› | — |
| `primvars:athenea:splat:transferIndirect` | float[] ‹27 por gaussiana› | — |
| `primvars:athenea:splat:shadowBits` | int[] ‹2 por gaussiana› | — |
| `primvars:athenea:splat:thinWalled` | int[] ‹1 por gaussiana› | — |
| `primvars:athenea:splat:normal` | normal3f[] ‹1 por gaussiana› | — |

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
dice.
`shadowBits` se escribe a su lado: sesenta y cuatro bits por gaussiana, uno por
celda de una rejilla octaédrica de 8 x 8 sobre la esfera en el espacio propio
de la nube, puesto donde el rayo del bake en esa dirección salió de la escena.
Es lo que sombrea el sol que un frame saca del cielo; sólo se lee en una nube
que lleva también `transferDirect`, y sin él el transfer sombrea el sol de
forma suave.
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

**`AtheneaSplatSkinningAPI`** — los joints que llevan una nube.

| Atributo | Tipo | Nota |
|---|---|---|
| `primvars:athenea:splat:jointIndices` | int[] | cuatro por gaussiana |
| `primvars:athenea:splat:jointWeights` | float[] | cuatro por gaussiana |
| `primvars:athenea:splat:geomBindTransform` | matrix4d | |
| `primvars:athenea:splat:skinningXforms` | matrix4d[] | una por joint, lo único que cambia con el tiempo |
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

Dos paneles, contados por su papel y no widget a widget, porque se mueven
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
| `remote-test.sh [user@host] [preset]` | compila y corre la suite en otra máquina y trae el log |
| `scripts/film/` | la película del gorrión: sus frames, su cámara y su sombra |
| `bmw-to-usd.py`, `sparrow-*.py` | ayudas de autoría para esos assets |

La cabecera de cada script dice qué necesita y dónde deja las cosas.

## 9. Cuando algo falla

| Qué se imprime | Qué significa | Qué hacer |
|---|---|---|
| `no GPU device` (los tests se saltan) | no se pudo abrir dispositivo | mira `athenea info`; en Linux pon `ATHENEA_BACKEND` |
| un error de compilación de shader con una ruta | los shaders del disco no son los del binario | recompila, o apunta `ATHENEA_SHADER_DIR` al `shaders` de esta compilación |
| `this build reads no .spz` | faltaba zstd cuando se compiló este binario | recompila con zstd, o convierte la captura en otro sitio |
| un `.sog` rechazado | faltaba libwebp | instálalo y recompila |
| `a .athc is drawn through its levels of detail: give --lod` | se le dio un contenedor a un render normal | añade `--lod`, y usa `--technique raster` |
| `mesh visibility by rays: the device has no ray tracing` | `--visibility rays` en un dispositivo sin ello | usa `automatic`, que elige lo que el dispositivo tiene |
| un host no ofrece el renderer | no encontró el plugin | pon `PXR_PLUGINPATH_NAME` a `<build>/plugin/usd` |
| `render product '<ruta>' has no resolution` / `no vars` | el prim de settings está incompleto | dale al producto resolución y vars ordenadas |
| a una nube convertida le faltan las últimas mallas | el presupuesto se acabó en orden de malla | sube `--max-splats`, o con `--density per-mesh` sube `--cell-min` |
| los reflejos de una nube salen más planos que los de la malla | no lleva normal de sombreado (`primvars:athenea:splat:normal`): se convirtió antes de que las conversiones la escribieran | conviértela de nuevo; `--normal-map-turns` además gira los propios discos |
| `cells of relief wanted more than N gaussians`, y el relieve muestra huecos en sus pendientes más fuertes | el relieve estiró esas celdas más de lo que permite la partición | sube `--displace-refine`; un polo de las coordenadas de textura estira sin límite y deja unas pocas sea cual sea el valor |
| los reflejos de una nube salen más blandos que los de la malla | la celda de la conversión es el kernel de desenfoque: una nube se lee como la malla a `r + 9c/R`, con `c` la celda y `R` el radio de curvatura | convierte con `--resolution` más fina: un espejo de roughness `r` quiere una celda por debajo de `r/9` de ese radio. Lo paga el fichero, no el frame -- quince veces las gaussianas fueron un 36 % más de tiempo por frame y dieciséis veces el disco |
| una bola de cristal enseña la sala pero no la dobla | la nube no tiene índice | `athenea mesh2splat` escribe el IOR del material de cristal; en una nube de otro origen pon `primvars:athenea:splat:ior` (1.5 es cristal). Una nube guarda un solo índice: con dos cristales de IOR distinto se queda el primero y la conversión lo avisa |
| una nube convertida sale negra | el bake no encontró luz | dale luces a la escena, o `--default-lights`, o `--no-bake` |
| una nube sale roma y luego se afina | todavía están llegando chunks | sube `--stream-budget`, o espera; una imagen fija se asienta antes |
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
