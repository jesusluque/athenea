# Arquitectura

athenea es un motor USD escrito en C++ y Slang. Lee un stage a través de Hydra, y cada paso que toca datos de la escena es un kernel de compute en la GPU. Sobre ese motor hace el bake de assets USD de producción a Gaussian splats que admiten relight: cada mesh se convierte en Gaussians anisótropas, y el path tracer del motor guarda en cada Gaussian cómo le llega cualquier cielo, de modo que la nube se reilumina al dibujarla bajo cielos y luces con los que nunca se hizo el bake.

Esta página es una visión de conjunto. El detalle está en [`docs/development.es.md`](https://github.com/jesusluque/athenea/blob/main/docs/development.es.md) (cómo está hecho el motor y cómo se cambia) y en [`docs/decisions.md`](https://github.com/jesusluque/athenea/blob/main/docs/decisions.md) (por qué es así, y qué se midió).

## Dos productos, dos procesos

El trabajo son dos procesos con objetivos distintos, descritos en el [paper](paper.md) como Parte I y Parte II.

| | El master de producción | El formato web |
|---|---|---|
| Proceso | Parte I, el bake | Parte II, derivado del master en un paso aparte |
| Fichero | un USDC: un `ParticleField3DGaussianSplat` por nube, con primvars de athenea (`primvars:athenea:splat:`) | `.athc` versión 3, comprimido y servido por streaming en pages |
| Contiene | todos los parámetros a precisión completa | los parámetros del master codificados: PCA por clusters del transfer, direcciones abiertas cuantizadas por vectores, un registro compacto de 40 bytes |
| Lo leen | el motor (`athenea stage`, `athenea view`, el delegate hdAthenea), el nodo OpenFX en Nuke y Natron, el add-on de Blender | un player WebGPU en un navegador, de escritorio y de móvil |
| Objetivo | fidelidad | tamaño, memoria y tiempo de frame |
| Se juzga por | su imagen con relight frente a un path trace del mesh de origen bajo el mismo cielo y las mismas luces | su pérdida frente al relight sin comprimir del master |

**El master.** Sus atributos estándar (posiciones, orientaciones, escalas, opacidades, armónicos de radiancia) los lee cualquier aplicación USD; las primvars de athenea llevan el resto: el material por Gaussian, el transfer del bake y, para un asset animado, su rig y su visibilidad. Lo producen estos pasos, cada uno un conjunto de kernels de GPU:

1. **Conversión** (`athenea mesh2splat`): cada mesh se convierte en Gaussians, una por cell de una red dimensionada a partir del footprint de píxel de una cámara, con el material por capas capturado en cada Gaussian.
2. **El bake del transfer** (`--transfer`): el path tracer traza el hemisferio de cada Gaussian y guarda cómo le llega cualquier cielo.
3. **El rig y la visibilidad** de un asset animado: las influencias del skinning por Gaussian y los campos de visibilidad por partes (`athenea visibility`).
4. **LOD**: merge trees cuyo cut por error de pantalla está acotado.

**El formato web.** Empieza donde termina el master y no cambia nada de él. Una pestaña de móvil tiene un budget de memoria (unos 50–60 MB de datos del asset mientras la cámara se mueve, unos 100 MB en reposo), una descarga que debe empezar a dibujar en segundos, y un tiempo de frame; el formato se hace para cumplirlos. Las herramientas que escriben `.athc` versión 3 y el player WebGPU que lo lee no están en este repositorio. El módulo `lod` del motor tiene su propio lector y escritor de `.athc` (`lod/Athc.h`), que usan `athenea convert` y el streaming del motor.

## Módulos

`modules/<nombre>` es una librería estática `athenea::<nombre>`. Los módulos se listan en `modules/CMakeLists.txt` en orden de dependencia, y cada uno enlaza solo con los de arriba.

| # | Módulo | Qué contiene |
|---|---|---|
| 1 | core | `Result` (errores sin excepciones), logging, `Platform` (toda la superficie del sistema operativo) |
| 2 | ui | los paneles del viewer como descripción, que dibujan `athenea view` y la app de iOS |
| 3 | sched | `FrameClock`: genlock PTP y alineación ST 2059-1, timecode; solo se compila con genlock |
| 4 | image | `Image`: RGBA float32 premultiplicado, origen abajo a la izquierda; lo que intercambian gpe y aofx |
| 5 | io | lectores de ficheros en CPU (cabecera PLY, SPZ, SOG zip/WebP, OpenVDB), EXR con atributos, los ficheros estándar que escribe `athenea flatten` |
| 6 | gpu | el dispositivo de slang-rhi, `ShaderLibrary`, `ComputeKernel` (parámetros enlazados por nombre), `CommandBatch`, `Buffer`, `AsyncReadback`; `gpu/algo` para prefix sum y radix sort |
| 7 | gpu_host | gpe adoptando el dispositivo de slang-rhi: un único `MTLDevice`, contexto CUDA o `VkDevice`, buffers compartidos sin copias |
| 8 | colour | OpenColorIO como compilador de funciones Slang y LUTs |
| 9 | scene | `CloudLoader`: registros en bruto subidos y decodificados en la GPU a `GpuSplats` / `GpuPoints`, el layout que leen todos los renderers |
| 10 | render | `TileRasterizer`, `GaussianRayTracer`, `PointRasterizer`, `ReferenceRenderer`, `Camera` / `Projection`, `Flatten` |
| 11 | geom | `MeshBuilder` (meshes de Hydra triangulados y sus primvars expandidas en la GPU), `Skinner`, `Subdivision` |
| 12 | material | `TextureStore` (decodificación, mips, UDIM), `MaterialCompiler` (grafos MaterialX a Slang), la librería de lobes |
| 13 | light | luces UsdLux en el dispositivo: un registro por luz, y cómo las muestrea un punto de shading |
| 14 | world | `GpuScene` (pools de vértices, índices y primvars, registros de instancia), `Instancing`, `RayTracingScene` (BLAS/TLAS), `BvhScene` (LBVH en compute) |
| 15 | technique | cómo se dibuja un frame: las rutas de visibilidad, el shading de materiales, `PathTracer`, `AovShading`, `Denoiser` (OIDN), `DisplayTransform`, `SplatVisibility` |
| 16 | lod | `LodBuilder`, `CutSelector`, `Athc.h` (el lector y escritor de `.athc`), `StreamingPool` |
| 17 | usd | `Engine`, `StageRenderer`, `Export`; el plugin de Hydra `hdAthenea`; los schemas codeless de `modules/usd/schemas` |
| 18 | mcp | el motor como servidor MCP: JSON-RPC 2.0 (`Server`) y las herramientas sobre un stage abierto |
| 19 | aofx | el lado de este motor del host AOFX (el `host/` de aopenfx): capacidades, `renderEffect` |
| 20 | view | `athenea view`: ventana GLFW, Dear ImGui dibujado en el dispositivo del motor |

Algunos módulos son condicionales: `sched` solo se compila con `ATHENEA_WITH_GENLOCK`; `gpu_host`, `mcp` y `aofx` no se compilan para iOS; `view`, solo con `ATHENEA_BUILD_VIEW`. Quien integra el motor enlaza `athenea::engine`, un target de interfaz que contiene el renderer sin el viewer, el servidor MCP ni `athenea::usd`.

## Shaders

Los shaders de Slang están en `shaders/athenea/<área>` y reflejan los módulos. La build los copia a `build/<preset>/shaders` y se compilan en tiempo de ejecución.

| Directorio | Qué contiene |
|---|---|
| `common` | lo que comparten todas las áreas; `packing.slang` es el contrato entre módulos: cómo se empaquetan la opacidad, la escala, el cuaternión y el color de un splat |
| `algo` | prefix sums, radix sorts, cadenas de mips |
| `scene` | la decodificación y validación de nubes en el dispositivo |
| `splat` | el pipeline del rasterizador por tiles (`frame.slang` nombra sus etapas), flatten, el cielo girado al marco de una nube |
| `rt` | el ray tracer de Gaussians y su BVH en compute; las dos rutas comparten `rt_integrate.slang` |
| `points` | el rasterizador de puntos |
| `reference` | el renderer de referencia y las comparaciones de imágenes (estadísticas, comparación HDR y de 8 bits) |
| `lod` | la decimación, el cut, los bloques de `.athc` |
| `geom` | topología y normales de meshes, expansión de primvars, skinning, curvas |
| `world` | instancing, el BVH en compute de la escena, movimiento |
| `material` | la librería de lobes, el runtime de materiales, el soporte de MaterialX (`mx`) |
| `light` | luces, entornos, el BVH de luces, perfiles IES |
| `technique` | visibilidad, shading, AOVs, ids de Cryptomatte, transformaciones de display |
| `volume` | volúmenes y NanoVDB |
| `usd` | los kernels del módulo `usd`: conversión de AOVs, posición en el mundo, el gather y el resolve del bake, la validación de mesh2splat |
| `view` | el dibujo de ImGui del viewer |
| `test` | los kernels de comprobación que las pruebas usan como oráculos en la GPU |

## Programas

`apps/athenea` es la línea de comandos (CLI11). Cada subcomando está documentado en [`docs/operations.es.md`](https://github.com/jesusluque/athenea/blob/main/docs/operations.es.md), sección 2.

| Subcomando | Qué hace |
|---|---|
| `info` | el dispositivo, lo que puede hacer, y el toolchain |
| `render`, `bench` | ficheros de splats a un EXR; el mismo frame cronometrado |
| `convert` | un fichero de splats a un stage USD `ParticleField`, o a un `.athc` con LOD |
| `decimate` | menos Gaussians, conservadas donde importan |
| `stage` | un stage USD renderizado por el delegate de Hydra del motor |
| `view` | una ventana sobre un stage: cámara libre, cámaras del stage, salidas, picking |
| `live` | un stage con reloj, cada frame en su instante ST 2059-1; solo en builds con genlock |
| `mesh2splat` | un stage USD de meshes a Gaussians, opcionalmente con el bake del transfer y skinning |
| `flatten` | las nubes con transfer de un stage bajo un cielo, como ficheros estándar de Gaussian splatting (PLY, SPZ, glTF, USD) |
| `visibility` | la visibilidad por partes de una nube con skinning, con bake en su fichero |
| `lights merge` | una nube base y una nube por grupo de faros, fusionadas en capas de luz |
| `aofx list`, `aofx run` | los bundles AOFX encontrados; un efecto aplicado a ficheros EXR |
| `compare` | lo que contienen una o dos imágenes EXR y cuánto se separan, medido en la GPU |
| `migrate` | ficheros escritos con los nombres anteriores del motor, reescritos con los de athenea |

`apps/athenea-mcp` es el servidor MCP por stdio: JSON-RPC 2.0 con un dispositivo y un stage abiertos, y herramientas como `open_stage`, `stage_tree`, `render`, `pick` y `settings`. El nodo OpenFX lo maneja como proceso hijo.

## Plugins AOFX

Los bundles de `plugins/` se compilan con el SDK de aopenfx (`third_party/aopenfx`), uno por directorio, se recogen sin registro y quedan en `build/<preset>/aofx`. Los pasos de procesado de datos del bake son efectos AOFX, de modo que los mismos binarios funcionan en cualquier host AOFX.

| Bundle | Identificador | Qué hace |
|---|---|---|
| Mesh2Splat | `rt.sparrow.aofx.mesh2splat` | un mesh a Gaussian splats; la conversión que ejecuta `athenea mesh2splat` |
| SplatBakeFilter | `rt.sparrow.aofx.splatbakefilter` | la luz indirecta del bake de una nube, promediada entre Gaussians vecinas sobre su superficie |
| SplatTransferZonal | `rt.sparrow.aofx.splattransferzonal` | el transfer de un bake ajustado a uno o dos lobes zonales en el marco propio de cada Gaussian |
| Measure | `rt.sparrow.aofx.measure` | dos imágenes medidas una contra otra en el dispositivo; lo que ejecuta `athenea compare` |
| Invert | `tv.mediapro.aofx.invert` | uno menos la imagen, en la GPU |

## Integraciones

| Directorio | Qué es |
|---|---|
| `integrations/ofx` | `Athenea.ofx.bundle`, un generador OpenFX para Nuke y Natron que dibuja una nube fuera de proceso; ver [OpenFX (Nuke, Natron)](ofx.md) |
| `integrations/blender` | `athenea_hydra`, una extensión de Blender que registra hdAthenea como motor de render, convierte meshes a Gaussians y los exporta; ver [Gaussian splats en Blender](tutorials/blender.md) |

## Cómo se hace un frame

**Una GPU, un hilo.** El `Sync` de Hydra corre en el hilo que le dé el host, y puede correr en varios a la vez. Nunca toca el dispositivo: toma un lock y entrega al motor registros del lado de la CPU. Después, el hilo del render pass los confirma (subidas, apertura de ficheros `.athc`, construcción de estructuras de aceleración) y dibuja. El dispositivo tiene un solo llamante.

**Un frame rasterizado.**

1. Los meshes se escriben en un visibility buffer, `(instancia + 1, triángulo)` por píxel. Tres rutas escriben los mismos ids: `VisibilityRaster`, `VisibilityTrace` y `VisibilityBvh`.
2. El shading y los AOVs reconstruyen la superficie a partir de ese buffer y la sombrean, de modo que las tres rutas sombrean igual.
3. Los puntos, si los hay, se rasterizan en una capa propia y se componen con los meshes por la z de vista.
4. Esa capa opaca se entrega a `TileRasterizer` como su capa `under`; los splats se proyectan, se ordenan, se reparten en tiles y se mezclan encima en luz lineal.
5. Los domos se pintan detrás y se aplica la exposición. `DisplayTransform` convierte el frame en algo que mirar, y solo lo llaman un viewer o una previsualización.

**Un frame con path tracing.** `PathTracer` integra a partir del mismo visibility buffer, así que las rutas de raster y de path tracing sombrean la misma superficie y se pueden comparar píxel a píxel. Un frame solo de splats lo traza entero `GaussianRayTracer`, cuyas dos rutas (estructuras de aceleración por hardware, BVH en compute) comparten un integrador.

**Materiales.** Un grafo MaterialX se compila a un módulo Slang nombrado por un hash de su código, de modo que los materiales que solo difieren en valores comparten un módulo.

**Los render buffers de Hydra** van con la fila de abajo primero, como los de Storm, y solo se convierten cuando un host los mapea. `athenea view` mantiene los frames en el dispositivo.
