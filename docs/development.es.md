[English](development.md) · Español

# Trabajar en athenea

Este documento es para quien cambia el motor: qué forma tiene, qué reglas
sigue y qué evita cada una, cómo se trabaja en el árbol, cómo se añaden las
cosas que se añaden a menudo, cómo se hornea una gaussiana, y cómo se
comprueba todo eso.

| Documento | Responde de |
|---|---|
| [`README.md`](../README.md) | por qué existe el motor, y una invocación de cada cosa |
| [`operations.es.md`](operations.es.md) | cómo se ejecuta, y cómo se autoriza una escena para él |
| **este** | cómo está hecho, y cómo se cambia |
| [`decisions.md`](decisions.md) | por qué es así, contra qué, y cuánto midió |

Este documento enlaza libremente al código y al registro de decisiones. Cita
una decisión por el título de su sección, nunca por número de línea, y cita
una especificación que ya vive en una cabecera en vez de copiarla. Las reglas
están en [`CLAUDE.md`](../CLAUDE.md), que es la copia que manda; el §2 de aquí
dice para qué está cada una. Mantenerlo al día: un módulo, una regla, una
manera de añadir algo o un paso del horneado que cambien, cambian este fichero
y su versión inglesa en el mismo commit.

## 1. La forma del motor

### 1.1 Los módulos, y la cabecera que especifica cada uno

`modules/<nombre>` es una librería estática `athenea::<nombre>`. Están listados en
`modules/CMakeLists.txt` en orden de dependencia, y cada uno enlaza solo con
los de arriba. `CLAUDE.md` tiene la tabla de qué guarda cada uno; lo que sigue
es la otra mitad — por cada módulo, la cabecera cuyo comentario es la
especificación real de ese subsistema. Cuando una de ellas y este documento no
coinciden, la cabecera tiene razón.

| # | Módulo | Léete esto primero |
|---|---|---|
| 1 | core | `core/Result.h` (la convención de errores), `core/Platform.h` (toda la superficie del sistema operativo), `core/Hash.h` |
| 2 | sched | `sched/FrameClock.h` — genlock, PTP y alineamiento ST 2059-1 |
| 3 | image | `image/Image.h` — la imagen de host que el host AOFX pasa de un lado a otro |
| 4 | io | `io/Sog.h`, `io/Exr.h`, `io/Vdb.h` — uno por formato, cada uno con su layout |
| 5 | gpu | `gpu/Device.h` (backends, el directorio de shaders, la caché), `gpu/ComputeKernel.h` (enlace por nombre) |
| 6 | gpu_host | `gpu_host/Context.h` — un dispositivo, dos runtimes encima, un hilo que le habla |
| 7 | scene | `scene/GpuClouds.h` — el layout de nube que leen todos los renderers |
| 8 | render | `render/TileRasterizer.h` y `shaders/athenea/splat/frame.slang` (el pipeline), `render/GaussianRayTracer.h` |
| 9 | geom | `geom/Skinner.h`, `geom/Subdivision.h` |
| 10 | material | `material/MaterialCompiler.h` — MaterialX a Slang |
| 11 | light | `light/LightTable.h` — una luz en el dispositivo |
| 12 | world | `world/GpuScene.h` — la escena tal como la lee cada técnica |
| 13 | technique | `technique/PathTracer.h`, `technique/SplatVisibility.h`, `technique/Environment.h`, `technique/DisplayTransform.h`, `technique/MaterialPrograms.h` |
| 14 | lod | `lod/Athc.h` — **la única especificación del formato `.athc`**, como un mapa de páginas; `shaders/athenea/lod/lod_decimate.slang` para lo que conserva un diezmado, `lod_attributes.slang` para lo que lleva (`usd::decimateStage` es todo el proceso) |
| 15 | usd | `usd/MeshStage.h` (leer una escena sin Hydra), `src/Engine.h` (el frame), `usd/Migrate.h` (en qué se convirtieron los nombres de lucabRTrender, y `athenea migrate`) |
| 16 | mcp | `mcp/Server.h` — el transporte JSON-RPC y qué es una herramienta |
| 17 | aofx | `aofx/Features.h`, `aofx/Version.h` — el ABI, copiado literal de su propio repositorio |
| 18 | view | `view/Viewer.h` — las opciones de la ventana |

Una cosa que parece una violación y no lo es: `render` (8) enlaza `aofx::aofx`
(17). Ese target es solo cabeceras, una librería `INTERFACE`, y lo que `render`
le toma son `Mat4` y `Vec3`. La regla de orden es sobre librerías compiladas.

### 1.2 Una GPU, un hilo

El `Sync` de Hydra corre en el hilo que le dé el host, y puede correr en
varios a la vez. No toca el dispositivo nunca: coge el lock y le entrega al
motor registros de CPU. El hilo del render pass los compromete después —
subidas, abrir `.athc`, construir estructuras de aceleración — y dibuja. Así
el dispositivo tiene exactamente un interlocutor, y todo lo relativo al orden
se sigue de ahí.

`gpu_host::Context` es donde encaja el otro runtime: gpe adopta el dispositivo
que abrió slang-rhi, así que un `MTLDevice` o un contexto CUDA sirve a los dos
y los buffers se comparten sin copias. El cruce en sí -- adoptar, una vista
slang-rhi de un buffer gpe, un handle gpe sobre un buffer slang-rhi -- son
tres funciones libres en `gpu_host/Views.h`, sobre el dispositivo y el pool
desnudos; los métodos del contexto las llaman, y también un programa que ya
tiene hilo de GPU y pool propios y sólo quiere el cruce.

### 1.3 Un frame, rasterizado

1. `Engine::render` toma la proyección, los ajustes y la petición de AOVs.
2. Mallas: `world::GpuScene` guarda los pools y los registros de instancia;
   una ruta de visibilidad escribe `(instancia + 1, triángulo)` por píxel en
   el buffer de visibilidad. Las tres rutas escriben lo mismo —
   `VisibilityRaster`, `VisibilityTrace`, `VisibilityBvh` — y un test las
   obliga a ello.
3. `technique::MaterialShading` (o `HeadlightShading`) reconstruye la
   superficie desde ese buffer y la sombrea. `AovShading` reconstruye la misma
   superficie para los ids, las normales y los primvars; `CryptoShading` para
   el plano de ids de la matte.
4. Los puntos, donde los haya, se rasterizan en una capa propia y se componen
   con las mallas por z de vista (`layers_nearest.slang`).
5. Esa capa opaca se le entrega a `render::TileRasterizer` como su `under`, y
   los splats se proyectan, ordenan, reparten por tiles y mezclan encima. Las
   etapas del pipeline están nombradas en `shaders/athenea/splat/frame.slang`.
6. Los domes se pintan detrás, se aplica la exposición y el frame está. Lo que
   lo convierte en algo que mirar es `technique::DisplayTransform`, y solo lo
   llaman un viewer o una vista previa.

### 1.4 Un frame, trazado

La diferencia empieza en el paso 3: `technique::PathTracer` integra desde el
mismo buffer de visibilidad, así que las dos rutas sombrean la misma
superficie construida igual, y se pueden comparar píxel a píxel. Los splats
los sigue componiendo el rasterizador sobre lo que produjo el trazador. Un
frame de solo splats lo traza entero `render::GaussianRayTracer`, cuyas dos
rutas (estructuras de aceleración por hardware, BVH por compute) comparten
`shaders/athenea/rt/rt_integrate.slang`.

### 1.5 La capa de shaders

`shaders/athenea/<área>` refleja los módulos: `common`, `algo`, `scene`, `splat`,
`rt`, `points`, `reference`, `lod`, `geom`, `world`, `material`, `light`,
`technique`, `volume`, `usd`, `view`, y `test` para los kernels de
comprobación.

- **El contrato entre módulos es `common/packing.slang`**: cómo se empaquetan
  en cuatro palabras la opacidad, la escala, el cuaternión y el color DC de un
  splat, y en una la normal de sombreado opcional (`packNormal`). Todo lo que
  escribe una nube y todo lo que la lee pasa por ahí. Un buffer opcional de
  `GpuSplats` (`pbr`, `normals`) se enlaza tenga o no la nube -- la forma en
  su lugar -- y un flag en los parámetros dice cuál.
- **El índice de un splat no es el de su registro.** La validación descarta
  lo que no se puede dibujar; `GpuSplats::origin` dice de qué registro vino
  cada splat conservado. Cualquier otra cosa que un fichero guarde por
  gaussiana y un kernel lea por splat pasa por `CloudLoader::keptOnly`, y de
  vuelta por `toRecords` para escribirse.
- Un fichero declara `module <nombre>;` igual que su nombre de fichero. Un
  hermano se importa a secas (`import frame;`), otra área por su ruta con
  puntos (`import athenea.common.packing;`). Todo lo que use otro módulo va
  `public`, buffers incluidos.
- En C++ un módulo se nombra por su ruta con barras:
  `ComputeKernel::create(library, "athenea/technique/crypto_ids", "cryptoIdsPass")`.
- Un kernel que evalúa materiales tiene que recorrer sus píxeles en orden de
  quad (`atheneaQuadPixel`), porque el bump toma sus derivadas de pantalla del
  quad del hilo. Uno que los recorra en orden de raster saca el bump plano y
  nadie avisa.

## 2. Las reglas, y qué evita cada una

Las reglas están en [`CLAUDE.md`](../CLAUDE.md). Aquí está el fallo del que
salió cada una, porque una regla de la que se olvida el motivo es una regla
que se discute.

**Nada de aritmética en CPU sobre datos de escena.** No es un defecto sino un
diseño: un camino por CPU que existe es un camino por CPU que se usa, y
entonces dos implementaciones se separan y la lenta se convierte en el
oráculo. Así que `SLANG_RHI_ENABLE_CPU` está apagado, la verdad de referencia
es un renderer de GPU (`render::ReferenceRenderer`), la comparación es un
kernel (`compareImages`), y un test se lee contadores, no píxeles. La CPU lee
ficheros, analiza cabeceras, descomprime y lleva cuentas.

**Los parámetros de shader se enlazan por nombre.** Enlazar por slot es contar
slots, y contar slots es como salieron los bugs de la D30 de openFXplayer: un
parámetro añadido en medio de una estructura mueve a todos los de después, y
nada deja de compilar. `cursor["nombre"].setBinding(...)` falla en voz alta.

**`Result<T>` y `ATHENEA_TRY`, sin excepciones.** Un renderer que lanza a través
de un command buffer deja el dispositivo en un estado que nadie sabe
describir. Donde una dependencia lanza — OpenUSD lo hace — la excepción se
captura en el borde de ese módulo y se convierte en un `Error`.

**El sistema operativo vive en `core/Platform`.** Cada llamada al sistema que
hace el motor fuera de sus dependencias está detrás de una cabecera, así que
el port a Windows tiene un fichero por el que empezar y no una búsqueda. Un
fichero que un comando escribe para que otro paso lo lea se escribe con
`platform::writeAtomically`: con un nombre parcial a su lado, renombrado al
completarse, así que un fallo no deja medio fichero con el nombre pedido.

**aofx solo cambia de manera aditiva.** Los bundles de openFXplayer tienen que
seguir cargando, así que las cabeceras del SDK se copian literales de su
propio repositorio y las congela `aofx_sdk_manifest`, que calcula el hash de
cada una y falla ante cualquier edición, alta o baja. Un parámetro se añade
apilando un `ParamDesc`, nunca cambiando uno.

**Un Slang, un slang-rhi, un TBB.** Dos TBB en un proceso son dos pools de
hilos peleándose por los mismos núcleos, y es invisible hasta que algo va
misteriosamente lento. `athenea info` los cuenta y el test `single_tbb` afirma la
cuenta.

**`docs/decisions.md` se actualiza con el cambio.** El registro es lo que hace
discutible una decisión un año después: qué se decidió, contra qué, cuánto
midió, qué no está hecho.

## 3. Trabajar en el árbol

### 3.1 Compilar y probar

```sh
cmake --preset macos-arm64-debug && cmake --build --preset macos-arm64-debug
ctest --preset macos-arm64-debug                     # todos los tests, uno a uno
ctest --test-dir build/macos-arm64-debug -R lod      # los que casan con el nombre
build/macos-arm64-debug/bin/athenea_lod_tests "chunks*"  # un caso, o una [etiqueta]
cmake --build build/macos-arm64-debug --target athenea_render_tests
```

Los requisitos, sus versiones y prefijos, y los cuatro presets están en el
*Building* del README. Los tests corren uno a uno porque comparten una GPU:
cada preset de test pone `jobs: 1`, y correr dos suites a la vez es como un
tiempo deja de significar nada.

### 3.2 Un cambio de solo shader

Los shaders se compilan en ejecución, no van dentro del binario, así que un
cambio en un `.slang` que ya existe necesita el paso de copia que hace
cualquier build y ninguna recompilación de C++. Un fichero de shader **nuevo**
necesita reconfigurar CMake, porque el glob que los copia es
`CONFIGURE_DEPENDS`.

Para comprobar que un shader compila sin compilar nada más:

```sh
~/tools/slang/bin/slangc shaders/athenea/<dir>/<fichero>.slang -I shaders \
    -target metal -entry <entry> -stage compute -o /dev/null
```

Los shaders generados — los materiales, los kernels de sombreado y de path
tracing — los escribe `ATHENEA_SHADER_DUMP=<dir>`, y compilan con `slangc -I
shaders -I <dir>`, que es como se mide el tiempo de compilación de un kernel
fuera del proceso.

### 3.3 Mirar dentro de una ejecución

| Variable | Qué hace |
|---|---|
| `ATHENEA_SHADER_DUMP` | escribe cada módulo Slang generado en ese directorio |
| `ATHENEA_MTLX_DUMP` | escribe el documento MaterialX que se le da al generador, uno por material |
| `ATHENEA_SHADOW_DEBUG` | loguea los mapas de sombra de nube: emisores, luces, texels, transmitancia media |
| `ATHENEA_VISIBILITY_DEBUG` | loguea los factores de visibilidad por partes |
| `ATHENEA_CUDA_LDG` | `=1` deja en su sitio el camino de caché de solo lectura de CUDA, reproduciendo un defecto que el prelude normalmente esquiva |
| `ATHENEA_OPTIX_INCLUDE` | las cabeceras de OptiX que se le dan a nvrtc, por encima de la ruta de compilación |
| `ATHENEA_TEST_DUMP` | un directorio para las imágenes que vuelcan los tests; sin él no vuelcan nada |
| `ATHENEA_VIEW_SWITCH_AT` | `=N`: el viewer cambia de técnica en el frame N, como haría un clic, para ejecuciones `--frames` reproducibles |
| `ATHENEA_VIEW_ORBIT` | `=R`: la cámara libre del viewer gira R radianes alrededor de su objetivo en cada frame, como haría un arrastre, para medir una cámara en movimiento con `--frames` |
| `ATHENEA_STAGES` | `=1`: una línea por frame rasterizado de splats diciendo adónde fue -- commit, visibilidad por partes, proyección, recuentos, orden por profundidad, emisión, orden por tile, blend --, esperando cada etapa, así que el frame es más lento |
| `ATHENEA_PORTABLE_SORT` | `=1`: todo radix sort toma las pasadas por trozos, como antes de la ruta por tiles; para apartar la ruta por tiles de un backend |
| `ATHENEA_ORACLE_*` | las entradas del oráculo de Storm en paralelo; son siete, documentadas donde el test las lee |
| `HDX_MSAA_SAMPLE_COUNT` | tiene que ser `1` para el oráculo de Storm; ctest lo pone, y el test falla explicándolo si no |

`ATHENEA_BACKEND` corre toda la suite sobre un backend: `cuda`, `vulkan`, `metal`,
`d3d12`.

### 3.4 Otra máquina

`scripts/remote-test.sh [user@host] [preset] [rama]` empuja la rama, compila y
corre la suite allí, y trae `LastTest.log` de vuelta a
`build/remote/<preset>/`. Extrae del log los mensajes de skip, porque un skip
no es un pass.

## 4. Cómo se añade algo

### 4.1 Un kernel

1. Escribe `shaders/athenea/<área>/<nombre>.slang`: `module <nombre>;`, los
   buffers que declara, un `ConstantBuffer<Params>`, y `[shader("compute")]
   [numthreads(...)] void <entry>(uint3 tid: SV_DispatchThreadID)`. El entry
   no puede llamarse `main`; Metal lo reserva.
2. `gpu::ComputeKernel::create(library, "athenea/<área>/<nombre>", "<entry>")`.
3. Despáchalo con **hilos**, no con grupos — la cuenta de grupos sale del
   `[numthreads]` reflejado, así que el kernel se comprueba los límites él
   mismo:

```cpp
kernel.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
    cursor["positions"].setBinding(cloud.positions.rhi());
    cursor["params"]["count"].setData(count);
});
ATHENEA_TRY(batch.submit(true));
```

Todo nombre que declare el shader tiene que estar enlazado, lo lea o no esta
vez. Para eso están los buffers de relleno del rasterizador.

### 4.2 Un test

Añade fuentes a un `athenea_test(...)` de `tests/CMakeLists.txt`, o crea uno. El
helper enlaza Catch2, depende de la copia de shaders, pone `ATHENEA_SHADER_DIR` y
registra los casos. Una etiqueta por binario: `gpu` (la de por defecto), `gpe`
(necesita el dispositivo de gpe), `display` (necesita ventana), para que una
máquina que no tenga una los excluya por etiqueta en vez de leer sus skips
como aprobados.

En el cuerpo del test:

```cpp
ATHENEA_REQUIRE_GPU(gpu);                        // o SKIP("no GPU device")
gpu::Buffer stats = test::uintBuffer(*gpu->device, 8, "check.stats");
gpu::ComputeKernel check = test::kernel(*gpu, "athenea/test/thing_check");
// despacha, envía, y lee contadores — nunca píxeles
std::array<uint32_t, 8> row = test::reduceStats(*gpu, stats, rows);
```

`test::dumpPpm` y los volcados a EXR escriben solo cuando `ATHENEA_TEST_DUMP`
nombra un directorio, y no se afirma nada sobre ellos: son para que los mire
una persona.

La primera etiqueta es el área (`[render]`, `[usd]`, `[technique]`), y el
resto la estrechan (`[crypto]`, `[mis]`, `[chi2]`).

### 4.3 Una AOV

Cinco sitios, y un nombre que falte en cualquiera de ellos falla en silencio:

| Dónde | Qué añadir |
|---|---|
| `modules/usd/src/Engine.h` | un valor en `enum class AovKind` |
| `modules/usd/src/Engine.cpp`, `Engine::aovView` | dónde vive en el dispositivo: buffer, tipo de fuente, stride, offset |
| `modules/usd/src/RenderPass.cpp` | del token de Hydra (o su prefijo) a `AovSource`, y qué tiene que calcular el frame en `AovRequest` |
| `modules/usd/src/RenderDelegate.cpp`, `GetDefaultAovDescriptor` | el formato que reserva Hydra y su valor de limpieza. Si falta aquí, el buffer no se reserva nunca |
| `modules/usd/src/StageRenderer.cpp`, `displaySource` | del nombre a un `DisplaySource::Kind`, para `athenea view --aov` y la vista previa de MCP |

Otros dos sitios llevan la lista como prosa y envejecerán sin avisar: la
cadena de ayuda de la herramienta MCP en `modules/mcp/src/Tools.cpp`, y el
comentario sobre `StageRenderer::displaySource`.

### 4.4 Un schema codeless

Dos ficheros escritos a mano, sin C++ generado:

- `modules/usd/schemas/generatedSchema.usda` — la clase, en la forma en que la
  escribe `usdGenSchema`. Cada propiedad es un `primvars:athenea:*` para que
  herede hacia abajo y el delegate la lea como primvar.
- `modules/usd/schemas/plugInfo.json` — una entrada en `Types` con
  `schemaKind: "singleApplyAPI"` y `bases: ["UsdAPISchemaBase"]`.

Los dos se copian junto a `hdAthenea` en
`<build>/plugin/usd/atheneaSchemas/resources/`, así que un `PXR_PLUGINPATH_NAME`
encuentra el delegate y los schemas juntos. El adaptador de prim que lee los
primvars nuevos va en `modules/usd/src/`.

### 4.5 Un módulo

Añádelo a `modules/CMakeLists.txt` **en orden de dependencia** y dale
`athenea_add_module(<nombre> SOURCES ...)`. Sus cabeceras públicas van en
`modules/<nombre>/include/athenea/<nombre>/`, las privadas junto a sus fuentes.
Escribe el comentario de cabecera que va a ser su especificación: la tabla del
§1.1 de este documento es el índice de esas cabeceras, y un módulo sin una es
un módulo que nadie puede leer.

### 4.6 Un subcomando

`apps/athenea/src/Cmd<Nombre>.cpp` con un `void add<Nombre>(CLI::App&)`, declarado
en `Commands.h` y llamado desde `main.cpp`. Las opciones se registran en el
subcomando y se leen en su callback; un error se imprime en stderr y el
callback lanza `CLI::RuntimeError(1)`. Y luego se documenta: una opción que no
esté en [`operations.es.md`](operations.es.md) no existe para quien ejecuta el
motor.

### 4.7 Un plugin aofx

Un bundle bajo `plugins/<nombre>`, compilado con el helper de CMake del SDK.
El efecto declara sus entradas y sus parámetros y recibe imágenes y números —
no ve USD, ni una escena, ni una API de dispositivo, que es justo por lo que
el mismo binario corre en openFXplayer. `athenea mesh2splat` es el ejemplo más
grande del árbol, y el §6 de este documento es lo que hace.

## 5. El banco de pruebas

Dieciocho binarios, uno por área, todos bajo `tests/`.

| Binario | Cubre |
|---|---|
| `athenea_core_tests` | el hash con el que Cryptomatte nombra las cosas, y su manifest |
| `athenea_gpu_tests` | prefix sum, radix sort, texturas, uniforms |
| `athenea_scene_tests` | cargar y decodificar nubes |
| `athenea_render_tests` | el rasterizador contra la referencia, puntos, ray tracing, movimiento, lente, SPZ, SOG |
| `athenea_geom_tests` | mallas, skinning, curvas, subdivisión |
| `athenea_material_tests` | el texture store, los lobes, el compilador de MaterialX |
| `athenea_technique_tests` | visibilidad, display, materiales, luces, sombras y visibilidad de splats |
| `athenea_lod_tests` | construir niveles, elegir el corte, `.athc` |
| `athenea_volume_tests` | volúmenes |
| `athenea_usd_tests` | el delegate de punta a punta |
| `athenea_storm_oracle_tests` | Storm como oráculo de geometría; su propio proceso y dispositivo, y `HDX_MSAA_SAMPLE_COUNT=1` |
| `athenea_mcp_tests` | el servidor JSON-RPC y sus herramientas |
| `athenea_host_tests` | el plugin por `UsdImagingGLEngine`; **no enlaza `athenea::usd` a propósito** |
| `athenea_view_tests` | el viewer; etiqueta `display` |
| `athenea_gpu_host_tests` | gpe adoptando el dispositivo de slang-rhi; etiqueta `gpe` |
| `athenea_aofx_tests` | el SDK, el host y el efecto mesh2splat; etiqueta `gpe` |
| `athenea_sched_tests` | el reloj de frame y PTP |

Y cuatro tests que no son de Catch2: `aofx_sdk_manifest` (los hashes del SDK),
`single_tbb` (un TBB en el proceso) y los dos `mesh2splat_density_*`, que
corren el CLI de verdad y afirman sobre la línea que imprime.

**Por qué `athenea_host_tests` no enlaza nada.** Conduce el plugin como lo hace un
host, por su nombre. Si además enlazara `athenea::usd`, una plantilla instanciada
en los dos — el `make_shared` del render pass — se ataría a la copia del
ejecutable, y el pass dejaría de reconocer los render buffers del plugin. El
test existe para atrapar justo eso.

**El patrón de oráculo.** Un test genera su entrada con un kernel, comprueba
el resultado con un kernel, y se lee un puñado de números. Un kernel de
comprobación vive en `shaders/athenea/test/`, recibe los buffers que comprueba más
uno de `stats`, y hace `InterlockedAdd` sobre un contador por cada propiedad
que prueba — así un fallo dice cuántos píxeles estaban mal, no solo que algo
lo estaba.

Un tiempo que valga la pena guardar va a `decisions.md` con el hardware y el
preset en el pie, no en un comentario.

## 6. Hornear una gaussiana, entero

Aquí se llaman horneado tres cosas distintas, y pasan en este orden: la
**conversión** de una malla en gaussianas, el **bake de luz** que llena esas
gaussianas con la radiancia que sale de la superficie, y el **horneado de
visibilidad** que registra lo que una nube con esqueleto proyecta sobre sí
misma. Las dos primeras son `athenea mesh2splat`; la tercera es `athenea visibility`.
Lo que sigue son las tres, desde la frontera hacia dentro.

### 6.1 La frontera: un host, y un efecto que no ha oído hablar de USD

La conversión es un efecto AOFX. Recibe una imagen de triángulos, hasta tres
mapas y un puñado de números, y escribe registros de gaussiana en la imagen
que le dan. Eso es todo lo que se le puede dar a un efecto AOFX — y es
exactamente por lo que el mismo binario corre sin cambios dentro de
openFXplayer.

Así que la escena es trabajo del host, en `apps/athenea/src/CmdMesh2Splat.cpp`:
abrirla, leer sus mallas, triangularlas en el dispositivo, empaquetar los
triángulos en una imagen, convertir cada textura que nombra un material en
filas de float4 lineal, correr el efecto una vez por malla, y escribir lo que
vuelve como un `UsdVolParticleField3DGaussianSplat`.

Las imágenes viven en el almacenamiento de imágenes del propio host AOFX, que
en un dispositivo de memoria unificada es la memoria que lee un kernel. Los
triángulos se escriben donde los va a leer el efecto y no se copia nada. Los
registros se quedan en el dispositivo desde ahí hasta el fichero: la imagen de
cada pasada se coloca en los buffers de la propia nube con
`athenea/usd/mesh2splat_gather` (registros, los rayos del bake, las
articulaciones), el desplazamiento de los rayos sale de la caja de la nube con
`mesh2splat_span`, el bake responde en un buffer del dispositivo que
`mesh2splat_bake` escribe en los registros, y la exportación los decodifica
donde están (`usd::DeviceSplatRecords`). Lo que cruza al procesador son
cuentas y, al final, los valores que guarda un array de USD — que son asunto
del procesador por definición.

### 6.2 La vida de una gaussiana, en quince líneas

1. Una malla se triangula en el dispositivo y se empaqueta en una imagen: seis
   `float4` por entrada — posición con `u`, normal con `v`, tres veces.
2. Su caja se reduce en el dispositivo, y de ahí sale el tamaño de celda.
3. Un hilo coge un triángulo. Lo proyecta sobre los dos ejes hacia los que
   menos apunta su normal, y recorre las celdas de esa proyección que cubre su
   caja.
4. Una celda cuyo centro cae dentro del triángulo, y que un mapa de recorte no
   borra, es una gaussiana.
5. Un escaneo por prefijo sobre los triángulos convierte esas cuentas en el
   slot que ocuparán las gaussianas de cada triángulo.
6. El emit recorre otra vez las mismas celdas. En cada una interpola la
   posición, la normal y los dos juegos de coordenadas de textura, y muestrea
   allí los mapas.
7. Los dos tamaños de la gaussiana sobre la superficie son de una celda de
   ancho, escalados por `--sigma`; el tercero es una fracción de ellos.
8. Su marco es la arista más larga del triángulo, la normal de la superficie y
   el producto vectorial de las dos; ese marco se vuelve un cuaternión.
9. Su opacidad es la del material por lo que lea el mapa de recorte.
10. Su color es el color del material por el mapa de albedo, teñido hacia el
    color de transmisión.
11. Su metallic y su roughness son los del material por los del mapa.
12. Su id es el hash Cryptomatte del prim del que vino.
13. Se escribe el registro, y el slot avanza.
14. Después, salvo `--no-bake`, se traza un rayo desde la gaussiana a lo largo
    de su normal de sombreado y la luz que vuelve se ajusta a armónicos.
15. La nube entera se escribe en una escena USD, con los primvars y los
    schemas que dicen qué lleva.

### 6.3 Leer la escena, sin Hydra

`usd::MeshStage` abre la escena directamente. Hydra es la interfaz de un
renderer a una escena, y esto no es renderizar: quiere las mallas tal como se
autorizaron, sus materiales estrechados a lo que puede llevar una gaussiana, y
nada compuesto para un frame.

`geom::MeshBuilder` triangula en el dispositivo, en el orden de `HdMeshUtil`,
así que las caras de una nube convertida se corresponden con lo que habría
dibujado Hydra.

La imagen de triángulos (`shaders/athenea/usd/mesh_pack.slang`) son seis `float4`
por entrada: por cada una de las tres esquinas, la posición con la primera
coordenada de textura en `w`, y luego la normal con la segunda en `w`. Un
segundo juego de uv, donde el material use uno, viaja en una imagen propia.

Las texturas se vuelven filas de `float4` lineal por `material::TextureStore`.
Un mapa viaja a dieciséis bytes por texel donde el fichero guarda uno, así que
un mapa de 4k son 268 MB en el dispositivo y un coche con quince no cabe. La
conversión muestrea un mapa una vez por celda, y a `--resolution 512` el
modelo tiene 512 celdas de lado, así que casi todo un mapa de 4k se tira antes
de mirarlo: `--texture-size` los limita a 1024 por defecto, y `0` los lee tal
cual.

El displacement de un material se lee como una altura: el `displacement` de
UsdPreviewSurface a través del `scale` y el `bias` de UsdUVTexture en el canal
conectado (`StageTexture::scale`, `bias`), o un `ND_displacement_float` de
MaterialX al que apunte el terminal de displacement del material, por su
`scale`. El resultado es `StageMaterial::displacementMap`,
`displacementScale` y `displacementBias`; `StageMesh::displacementUnit` es la
raíz cúbica del volumen de la transformación, porque una altura se escribe en
las unidades de la propia malla.

### 6.4 Un solo kernel: contar, escanear, emitir

`plugins/mesh2splat/mesh2splat.slang` es un port de la conversión mesh2splat
de Electronic Arts, de su pipeline de OpenGL a un kernel de compute. El suyo
es un geometry shader y un fragment shader; aquí no hay rasterizador, así que
los fragmentos se recorren — las mismas muestras que habría producido el
rasterizador, a la misma densidad, conservando el append atómico que hace que
el orden de la salida no sea asunto de nadie.

**La proyección.** Un triángulo se proyecta sobre los dos ejes hacia los que
menos apunta su normal — su elección triplanar —, con la posición tomada
relativa a la caja y dividida por un rango. Eso es lo que rasteriza su render
target, así que una celda suya es `rango / resolution` del mundo.

**La jacobiana.** `J = V (O)^-1`, el mapa de esa proyección al espacio. Sus
columnas son cuánto se mueve en el mundo un paso de una unidad de proyección;
un paso de una celda es eso partido por la resolución, y una gaussiana mide
`--sigma` de una celda.

**Qué caja, y cómo de grande es una celda.** El `rango` es donde difieren las
dos densidades. Por modelo es el mayor de los dos extents del plano de
proyección, lo que hace que una celda dependa de hacia dónde mira un
triángulo: en un coche de 0.77 de ancho, 1.92 de largo y 0.60 de alto, un
panel mirando al frente recorría una celda 2.5 veces más fina que el capó. Por
malla, `cellByLongest` mide cada triángulo contra el lado largo de su propia
malla, así que una malla tiene una celda haga lo que haga cada triángulo.
Luego la celda se acota en unidades de mundo entre `--cell-min` y
`--cell-max`, y `perCell` se recalcula **solo donde un límite muerde**, así
que una ejecución sin acotar conserva `1 / resolution` exacto y sus centros de
celda donde estaban.

`decisions.md: "The density is a mesh's own, not the stage's"` tiene las
medidas: un suelo de dieciséis unidades le dio a un coche de 1.9 un octavo de
las celdas que tiene convertido solo.

**Una gaussiana no puede ser mayor que el triángulo sobre el que está.** Las
columnas de la jacobiana llevan `1 / determinante`, y una astilla vista casi
de canto por su propia proyección tiene un determinante justo por encima del
suelo, así que las columnas se disparan: en el coche bmw27, 1859 triángulos
así dibujaron púas blancas de metros a lo ancho del frame. Los dos tamaños se
recortan a la arista más larga del triángulo, que es la cota siempre cierta,
hiciera lo que hiciera la proyección.

**Contar.** Un hilo por triángulo, sobre las celdas de su caja en la rejilla
de proyección, probando el centro de cada una contra el triángulo y contra el
mapa de recorte. `--max-cells` limita cuántas celdas puede recorrer un
triángulo; lo que no pudo recorrer se informa, así que un techo que muerde se
ve.

**Escanear.** Una suma por prefijo sobre los triángulos convierte las cuentas
en offsets, en un workgroup de 256 hilos y tres fases: cada hilo suma su
tramo, el hilo cero recorre los totales de tramo en una suma corrida, y cada
hilo recorre su tramo otra vez dejando la suya. La otra opción era un escaneo
en serie en un hilo; el peón tiene 42 892 triángulos y el zorro tiene más.

El escaneo es también donde se aplica el presupuesto: los slots son un prefijo
en orden de malla, así que un presupuesto corto conserva enteros los primeros
triángulos y corta en el límite de uno, nunca a la mitad. El primer triángulo
en el que corta queda registrado, y el siguiente tramo del host arranca ahí.

**Tramos.** Una malla cuyos triángulos quieren más de lo que permiten el
presupuesto o el techo se convierte por tramos: el host vuelve a correr el
efecto desde el triángulo que registró el corte, hasta que todo quepa o no
quepa más. Y cuando una ejecución quiere más gaussianas de las que le dio la
primera estimación, el host vuelve a correr esa malla con la cuenta exacta que
pidió — el log dice cuántas mallas hubo que correr dos veces, que es el
indicador más barato de que una estimación está mal afinada.

**Emitir.** El mismo recorrido otra vez, escribiendo los registros. Un
triángulo que no atrapó ninguna celda — más pequeño que una celda, o mal
colocado — se lleva igualmente una gaussiana en su centroide: una superficie
que existe tiene que dibujarse, y la cuenta y el emit tienen que estar de
acuerdo en eso o los slots que se le dieron a un triángulo no son los que
rellena.

Siguen siendo dos kernels, y el compilador contrae cada uno a su manera: el
centro de una celda justo sobre la arista que comparten dos triángulos puede
quedar dentro para uno y fuera para el otro. Así que el emit escribe
exactamente los slots que le dio la cuenta — desde su inicio hasta el del
triángulo siguiente —, para en el último, y escribe un slot para el que no
tiene nada como una gaussiana transparente en el centro del triángulo.

**Relieve.** Donde el material desplaza, cada celda mide cuánto la estira el
relieve en cada eje de la proyección — la tangente de la superficie elevada
frente a la de la plana, por diferencias centrales de media celda — y la
cuenta y el emit la parten en `ceil(0.8 * stretch)` gaussianas por eje, con
`--displace-refine` como tope. Una celda que quería más se cuenta en el sexto
contador. El relieve se lee en el punto de la propia subcelda, también fuera
del triángulo: el plano, las normales y las coordenadas de textura del
triángulo siguen más allá de su arista, y recortada hacia dentro una subcelda
leía la altura de un punto en el que no estaba.

**Bloques.** Con `simplify`, la cuenta y el emit recorren el triángulo con
`m2sWalkBlocks`: un árbol de bloques alineado a la rejilla de celdas, de
2^levels celdas de lado arriba, en profundidad sobre una pila de dieciséis (así
que cinco niveles como mucho). Un bloque es una sola gaussiana
(`m2sBlockIsOne`) cuando él y un bloque más allá de cada lado caen dentro del
triángulo, con un margen en las esquinas, y todos los centros de celda de ese
alcance coinciden dentro de la tolerancia en color, metallic y roughness,
recorte (por encima del corte en todas partes), normal de sombreado y normal
del relieve, sin ninguna celda que el relieve partiría. El bloque se lee antes
que su entorno: es donde más a menudo falla. Si no, se apilan sus cuatro
hijos, y una celda sola es lo que siempre fue. Un triángulo en el que el
recorrido no encuentra nada cae en la única gaussiana de su centro.

### 6.5 Qué lleva una gaussiana

**La posición** es la mezcla baricéntrica de las esquinas del triángulo en el
centro de la celda.

**Los tres tamaños.** Dos son las columnas de la jacobiana por `--sigma`
partido por la resolución — la gaussiana es tan ancha como una celda, medida
en el mundo. El tercero es `--flatness` por el menor de esos dos, y que sea
una *fracción* y no una longitud es justo el punto: mesh2splat escribe `1e-7`
ahí, un número en las unidades del propio modelo, así que lo fina que es una
gaussiana dependería de lo grande que resulte ser el modelo. El peón de
ajedrez mide 66 mm y se trazaba bien; el zorro de Khronos mide cien unidades,
lo que hace el mismo `1e-7` mil quinientas veces más extremo, y el ray tracer
— que integra densidad a lo largo del rayo en vez de proyectar una elipse —
veía un fantasma donde el rasterizador veía un zorro.

**El marco.** La tangente es la arista más larga del triángulo, el eje corto
es la normal de la superficie (o la del mapa de normales, con
`--normal-map-turns`), y el tercero es perpendicular a los dos. No las
columnas de la jacobiana: la gaussiana es un disco en el plano del triángulo,
se hubiera parametrizado ese plano como se hubiera parametrizado. El marco se
vuelve un cuaternión, y un marco que colapsa no escribe gaussiana.

**La opacidad** es `--opacity` por lo que lea el mapa de recorte, y el valor
del mapa **es** la opacidad y no un sí o un no. La barba de una pluma es un
texel de alfa 0.3, y una gaussiana de opacidad 0.3 mezcla como mezclaba la
tarjeta; cortada a la mitad, cada barba era una gaussiana entera o nada.
`--opacity-cut` es donde el mapa deja de significar *fino* y empieza a
significar *ausente*.

**El color** es el color del material por el mapa de albedo, y luego llevado
hacia el color de transmisión por la transmisión del material. Su conversión
no tiene canal ninguno para la transmisión, así que el cristal salía como una
gaussiana blanca opaca.

**Metallic y roughness** son los valores del material por los del mapa, azul y
verde como los empaqueta glTF. Los suyos venían por defecto a (0.1, 0.5) e
ignoraban el material, así que toda conversión salía del mismo plástico.

**El mapa de normales**, donde lo hay, se lee en el marco tangente y se vuelve
la normal de sombreado de la gaussiana — y, con `--normal-map-turns`, también
su eje corto. La normal de sombreado se escribe diga lo que diga el flag
(`primvars:athenea:splat:normal`, tres floats más por registro): una nube
reiluminada se ilumina con ella y conserva el relieve, mientras el disco sigue
sobre la cara.

**La transmisión** va a un canal propio en vez de bajar la opacidad. Un
material translúcido no es uno transparente, y bajar aquí la opacidad diría
que sí lo es.

**Joints y pesos**, con `--skinned`: cuatro de cada por gaussiana, mezclados
desde las esquinas del triángulo, para que la nube se deforme con el esqueleto
que llevaba la malla.

**El id Cryptomatte** es el hash de la ruta del prim origen, heredado por cada
gaussiana que la conversión hace de los triángulos de ese prim. Es lo que
permite nombrar un coche convertido pieza a pieza en una matte en vez de como
una nube; `decisions.md: "A pixel says which prims it saw: Cryptomatte"`.

**El relieve**, donde el material desplaza: la gaussiana está en el punto
plano más la normal por la altura, su eje corto es la normal del relieve (el
producto vectorial de las dos tangentes elevadas) y su primer eje la tangente
elevada en u, y sus dos tamaños son los de la celda plana por el estiramiento
entre la partición — no más de lo que valen 1.25 celdas donde la partición
llegó al tope, porque el polo de las coordenadas de textura de una esfera
estira sin límite y, dimensionada por eso, una gaussiana era un pincho que
cruzaba el frame. Tres entradas más por registro llevan lo que necesitan el
bake y un movimiento posterior del relieve: el punto plano con la altura, la
normal plana y la normal del relieve.

### 6.6 El bake de luz

Salvo `--no-bake`, a la conversión le sigue un path trace cuya cámara es una
lista de rayos en vez de un frame.

**Dónde arranca un rayo.** Desde la posición de la gaussiana, a lo largo de la
normal de sombreado, desplazado `1e-4` de la diagonal **del modelo**. Tomado
como fracción de la unidad de la escena — una milésima, con suelo de uno — el
peón de ajedrez, de 66 mm en una escena cuya unidad es el metro, arrancaba sus
rayos a un milímetro de la superficie: más grueso que el anillo de oro bajo la
bola de cristal y lo bastante alto para arrancar dentro de la bola, así que el
rayo bajaba a la superficie equivocada y el anillo horneó gris, el color del
cuerpo de mármol, donde la malla lee oro (`decisions.md: "The bake's ray
started a millimetre off the model"`).

**Sobre una huella.** La `w` de la normal de un punto, donde no es cero, es el
ancho de la gaussiana, y cada una de sus muestras baja sobre un punto de un
disco de la mitad de ese ancho en vez de sobre el centro. La conversión la pone
cuando `--simplify` está activo, así que la gaussiana de un bloque lleva la luz
del bloque.

**Quién los traza.** `StageRenderer::bakePoints` toma dos `float4` por punto —
el punto con su desplazamiento, y luego su normal — y despacha el path tracer
sobre ellos como si fueran píxeles. `bakePointsOnDevice` es lo mismo con los
rayos ya en el dispositivo en la disposición de tres `float4` del kernel y la
respuesta dejada allí, las entradas de un punto juntas
(`athenea/usd/bake_gather`); la conversión abre el renderer en su propio
dispositivo y usa esa. Es el mismo integrador que usa un frame,
compilado con su constante de bake en cierto: no una segunda implementación.

**Una gaussiana elevada** se hornea desde el punto plano que tiene debajo,
bajando por la normal plana — un rayo desde donde está empezaría bajo la
superficie allí donde el relieve la hundió — con una tercera entrada, su
orientación, junto a las dos. La normal de sombreado del impacto pasa a ser
esa orientación, así que su material, normal map incluido, se ilumina según lo
gira el relieve, y las direcciones que se proyectan son las del hemisferio de
esa orientación. Un rebote desde ese primer vértice por debajo de la
superficie plana se cierra: encontraría la malla plana, iluminada, donde está
el relieve. La sombra del relieve sobre sí mismo no se hornea; en el trazador
no hay nada ahí.

**Las direcciones se estratifican, no se sortean.** La radiancia que sale de
una superficie brillante oscila en órdenes de magnitud por el hemisferio, así
que unas direcciones tomadas al azar dejan una gaussiana en el espejo del sol
y a su vecina en ninguna parte — sal y pimienta que más caminos apenas tocan.
Una rejilla con un jitter en cada celda lo cubre por igual, y la misma cuenta
responde entonces a otra pregunta.

**Qué se queda, y qué se tira.** En el primer vértice la pila de lobes del
material se reduce a su *cuerpo*: los lobes difusos, que llevan la textura y
la luz que le llegó, lo que el material transmite, y el reflejo de un
conductor — un metal no tiene más cuerpo que ese, y tirarlo dejaría el oro
negro. Lo que se tira es el pulido dieléctrico y el sheen, que el renderer
devuelve en el frame a partir del metallic y el roughness que lleva la
gaussiana, y los devuelve *con una dirección dentro*. Un reflejo es justo la
parte de una superficie que un color no puede guardar: horneado, es el mismo
desde todas las direcciones, y el mármol del ajedrez volvió como plástico gris
liso con su veteado desaparecido, la textura enterrada bajo un tres por ciento
de especular.

Un metal no siempre es un `conductor_bsdf`. MaterialX escribe el metal de
OpenPBR — y el de Standard Surface — como un Schlick generalizado, el mismo
reflejo bajo otro nodo. Tirado como pulido, un metal escrito así hornea a
nada: cada una de las 27 016 gaussianas cromadas de un Mustang guardó el DC
que decodifica a negro, contra el 0.19 de la malla. Lo que distingue a los dos
es la reflectividad a incidencia normal: la de un dieléctrico es la que da su
índice de refracción, 0.04 a 1.5 y 0.17 al 2.42 del diamante, mientras que la
de un conductor es la mitad de la luz o más. Así que un Schlick cuyo F0 pase
de un quinto es el metal que representa.

**El ajuste.** El resultado se proyecta sobre armónicos esféricos del grado
que pida `--bake-degree`, sobre la mitad de la esfera a la que mira la
superficie. La base es ortonormal sobre la esfera entera y sobre nada más, así
que ajustada coeficiente a coeficiente sobre un hemisferio cada uno explica la
misma luz otra vez y su suma se pasa — medido, el peón volvió diez veces
demasiado brillante. El ajuste es en su lugar un sistema lineal pequeño, con
un suelo bajo el pivote de Cholesky para que la dirección que los datos no
pudieron ver quede acotada en vez de amplificada. `decisions.md: "The bake
fits the harmonics, and no longer projects them"` tiene la aritmética y lo que
midió cada intento.

**Una gaussiana bajo la que el bake no encontró nada** se queda con opacidad
cero, no con un color cero. Los coeficientes vuelven a cero, y cero no es «sin
color»: el término constante se guarda desplazado a donde lo entrena 3DGS, así
que un cero ahí decodifica a negro. Una nube salida de mesh2splat es casi toda
discos, así que uno de ellos visto de canto en una silueta es una astilla
negra — que es lo que el anillo de oro del peón tenía por fleco, cuarenta y
cuatro de ellas en 729 073.

**El fit se hace en luz lineal**, y sólo el resultado ajustado se lleva al
espacio en el que se blendea una nube: el término constante por la curva sRGB,
y las bandas que le dan forma por su pendiente ahí. Se hizo sample a sample
durante un tiempo, que es otra cantidad — la curva es cóncava, así que la
media de los samples codificados es más oscura que la codificación de su media
allí donde los samples difieren. Donde coinciden, que es un cielo abierto, las
dos son la misma, y por eso pasaban todos los tests del bake. Donde un
occluder los hace diferir, un punto junto a un muro con medio hemisferio
tapado se ajustaba en 0.2098 contra los 0.309 de luz que realmente tiene.

**Un bake que mide el cielo en vez de la luz.** `--transfer` lanza los mismos
rayos con una tercera variante del kernel, `kTransfer`. El primer vértice no
es el del material: la superficie se toma como un Lambert blanco, la dirección
es la del sample estratificado, y lo que vale un camino es `2 cos(theta)` — un
hemisferio uniforme de densidad `1/2pi` contra un estimador que quiere
`cos/pi`. Un camino que escapa paga la base leída donde salió, `Y_k(omega)`;
uno que no escapa no paga nada. No hay fit, porque la acumulación **es** la
proyección.

De los mismos rayos salen dos mitades: un camino que escapó desde el primer
vértice es el transfer directo, que es geometría y no tiene color, y uno que
escapó tras rebotar lleva el color de aquello en lo que rebotó, que es la
global illumination estática de la nube. Nueve escalares y veintisiete floats
por gaussiana, en f16, escritos como dos primvars. Lo que hace luego un frame
es `<T, L_SH>` con los nueve coeficientes del cielo bajo el que está la nube
(`technique::Environment`), por el albedo; y un metal no tiene cuerpo ninguno
y se va entero al reflejo prefiltrado con `f0 = albedo`. Ese reflejo lee una
mip chain del domo: ocho niveles octaédricos cuya base sigue la resolución del
propio cielo -- 2048 de lado para una imagen de 4k, 256 para un domo que sólo
es un color -- y que la roughness recorre como su raíz cuadrada, de modo que
los téxels de un nivel son del ancho del lóbulo que guarda.

**El sol.** `technique::Environment` busca además la fuente más brillante de
cada domo y la entrega como dirección e irradiancia (`env_sun.slang`),
dejándola fuera de los nueve coeficientes (`env_project` se salta su cono) y
sumándola de vuelta sobre el cuerpo en `relitByDome`. Los dos lados suman los
mismos téxeles lat-long con la misma medida, que es lo que hace el cambio
neutro en energía. Su visibilidad es `splatSunOpen`: donde la nube lleva `shadowBits`
-- sesenta y cuatro bits por gaussiana que `kTransfer` traza una vez, en la
primera muestra, un rayo por celda de una rejilla octaédrica de 8 x 8, escritos
como un plano después de la cobertura -- las cuatro celdas alrededor del sol
pesadas bilinealmente, dejando fuera las que quedan bajo el horizonte de la
gaussiana; si no, `splatSunShare`, el transfer leído en esa dirección sobre la
misma truncatura de un hemisferio abierto, exactamente uno donde no ocluye
nada. La polish conserva el sol del propio mapa y le quita la parte que no pasa
como un lóbulo GGX analítico, recortado en cero, así que un metal también
queda sombreado. Una nube sin transfer recibe el sol de vuelta sin sombra.

**Decir otra cosa de un prim.** El id de Cryptomatte que lleva una gaussiana
es también una selección -- todo lo que vino de un mismo prim -- así que
`render::SplatOverride` es una fila con ese id como clave: metallic,
roughness, transmission y un tinte, en un buffer que recorren los dos kernels
de sombreado (`shaders/athenea/common/splat_override.slang`). Un valor negativo
deja lo que lleve la gaussiana, y `replaceColour` hace del tinte el color en
vez de un factor sobre él; `athenea stage --splat-override` escribe filas por ruta
de prim a través del manifest. `render::measureSplatId` responde en el otro
sentido, con un kernel que cuenta las gaussianas de un id y los límites de lo
que llevan. Ninguno toca el fichero: la tabla es del frame, y vaciarla
devuelve la nube.

**Una pared fina.** Un dieléctrico bajo un `surface` con `thin_walled`
activado (el `geometry_thin_walled` de OpenPBR) es una lámina:
`kFlagThinWalled`, que pone `atheneaPushLobe` a partir de `gAtheneaThinWalled`, que a
su vez fija el constructor de superficie compilado antes de su BSDF. Refleja
`2R / (1 + R)` y deja pasar el resto en línea recta, como una delta. Una gaussiana convertida de
una (`thinWalled`, el bit 24 de `pbr`) es tan transparente como la lámina:
`scene::thinWallOpacity` fija su opacidad, y el frame sombrea sólo su reflejo,
escalado por `1/R0`.

**Una nube bajo un obturador.** Dos cosas mueven una gaussiana durante él: un
esqueleto (`motion`, de `splat_skin.slang`) y objeto-a-vista (`viewStep`). Lo
segundo es `V_close M_close - V_open M_open`: `ParticleField` muestrea el
transform del prim a lo largo del obturador y le da el paso a `Engine`, y los
extremos de la cámara vienen de la proyección. Los dos llegan al único término
de rango uno del rasterizador; el trazador dibuja una nube en el instante del
frame.

**Qué es entonces la nube.** Horneada, sus colores son la luz sobre el cuerpo
del material y el frame añade el pulido: eso es `relight` con `litBody`. Sin
hornear, son un albedo y el frame los ilumina enteros: `relight` solo. En los
dos casos la nube se escribe como relit, porque una malla convertida no es una
captura y sus colores no fueron nunca radiancia que alguien fotografió.

### 6.7 El esqueleto

`--skinned` construye las gaussianas en la pose de bind y le da a cada una los
cuatro joints y pesos mezclados del triángulo sobre el que está. Una malla que
no lleva nadie ocupa igualmente su sitio en el rig: las mallas con esqueleto
de una escena rara vez son todas, y las influencias tienen que quedar una a
una con las gaussianas o la nube y su rig no se ponen de acuerdo sobre quién
es quién. Esas gaussianas se llevan cuatro joints de peso cero, que el skinner
lee como *deja esta donde la puso la pose de bind*.

El fichero lleva el rig, no los frames: las transformaciones de los joints son
una matriz por joint, muestreadas sobre el rango que pida `--range`, y todo lo
demás es estático. En un pájaro de 4 269 858 gaussianas solo ese array tiene
muestras de tiempo, y son 609 matrices — que es lo que hace que una nube
animada cueste kilobytes por frame en vez de decenas de megabytes.

El skinner gira la normal de sombreado con el marco, con la misma mezcla y
como una normal (`(M a) x (M b)` para dos direcciones `a`, `b` de su superficie,
que es la inversa traspuesta salvo escala), así que el relieve de un miembro
se dobla con él.

Un bake se rechaza con `--skinned`, porque la luz horneada en una pose está
mal en todas las demás.

### 6.8 El horneado de visibilidad

Una nube con esqueleto no puede llevar una visibilidad horneada, porque el ala
se mueve y se lleva su sombra. Pero una parte de un cuerpo cambia muy poco de
forma entre poses, así que `athenea visibility` le da a cada parte un campo
propio: sobre una rejilla de sondas (`--grid` por lado), en cada dirección (un
mapa octaédrico de `--octave` por lado), cuánto de un rayo que sale de esa
sonda detienen las gaussianas de la parte, horneado en la pose en que se ató
la nube. Un rayo se detiene cuando la transmitancia baja de `--cut`.

Las partes salen del rig: la jerarquía de joints se corta en `--parts`
subárboles, y un subárbol con menos de `--min-joints` joints se queda con el
de su padre. Cada gaussiana pertenece a la parte cuyo joint más la lleva.

Al renderizar, una gaussiana le pregunta a cada parte, en el marco actual de
esa parte, y multiplica las respuestas: una lectura de tabla por parte, por
luz y por gaussiana, y ningún rayo. Lo que se cede es que una parte se toma
como rígida y que las partes ocluyen independientemente. `decisions.md: "A
cloud shadows itself by part: baked once, read every frame, no ray"`.

### 6.9 Qué se escribe en el fichero

Un `UsdVolParticleField3DGaussianSplat` en `/World/Splats`, con los atributos
estándar — posiciones, orientaciones, escalas, opacidades, el grado de
armónicos y sus coeficientes, el extent — y una cámara que lo encuadra salvo
`--no-camera`.

Al lado, los primvars que dicen lo que necesita este motor y los schemas que
los declaran: `AtheneaSplatLightingAPI` (`relight`, `litBody`, y metallic,
roughness, transmission y la normal de sombreado por gaussiana), `AtheneaSplatSkinningAPI` donde la nube
tiene esqueleto, `AtheneaSplatCryptomatteAPI` con un id por gaussiana y el
manifest que los nombra, y `AtheneaSplatVisibilityAPI` una vez que ha corrido `athenea
visibility`. La referencia de atributos es
[`operations.es.md §4.3`](operations.es.md#43-los-schemas-de-api).

Cada slot se escribe, sobreviviera o no una gaussiana en él: uno vacío lleva
opacidad cero y no estira el extent de la nube hasta donde esté. Conservar los
slots es lo que mantiene los arrays por gaussiana índice a índice, que es lo
que permite leer las influencias del esqueleto, los ids y los armónicos con el
mismo índice.

### 6.10 Cómo se comprueba cada fase, y qué midió

| Fase | Lo comprueba | Qué afirma |
|---|---|---|
| la proyección y la celda | `athenea_aofx_tests "[mesh2splat]"` | un cuadrado unidad a una resolución conocida da una cuenta conocida, y cada gaussiana tiene un ancho conocido |
| la celda acotada | los mismos, casos `[cell]` | un límite que muerde cambia la cuenta, y uno que no la deja bit a bit |
| densidad por malla | `ctest -R mesh2splat_density` | el plano pequeño de una escena de dos mallas saca decenas de gaussianas por modelo y centenares por malla |
| el mapa de recorte | `athenea_aofx_tests` | un alfa de 0.3 se vuelve una opacidad de 0.3, no una gaussiana o nada |
| el bake de luz | casos de bake de `athenea_usd_tests` | un plano lambertiano vuelve con la radiancia que dice la aritmética, y un metal no sale negro |
| el ajuste de armónicos | los mismos | el ajuste reproduce una función direccional conocida dentro de tolerancia |
| los ids | `athenea_usd_tests "[crypto]"` | los ids que nombra la matte de un frame son exactamente los prims que leyó la conversión |
| el horneado de visibilidad | `athenea_technique_tests` | un factor horneado coincide con uno trazado dentro de tolerancia |
| el relieve | `athenea_aofx_tests "[displacement]"` | bajo una tienda de altura cada gaussiana está a la altura que lee el mapa, mira según la pendiente y tiene el tamaño de su parte de una celda partida en dos; una altura constante mueve la superficie y no parte nada |
| la lectura de la altura | `athenea_usd_tests "[displacement]"` | el scale y el bias de UsdUVTexture en el canal conectado, el nodo de MaterialX y su scale, una constante, y una malla escalada por dos |
| el bake del relieve | los mismos | puntos de un plano llano orientados 40 grados hornean lo que hornea un plano girado de verdad 40, dentro de un 3 % |
| los bloques | `athenea_aofx_tests "[simplify]"` | un cuadrado de un color a 64 celdas saca 2144 gaussianas frente a 4160, todas sobre él, de ese color y de una, dos, cuatro u ocho celdas de ancho; bajo un damero de casillas de cuatro celdas, ni una menos |

Medido, y registrado en `decisions.md` con el hardware en el pie:

| Qué | Número |
|---|---|
| el peón de ajedrez, convertido | 729 073 gaussianas, 25 s a 64 caminos (debug) |
| su cuerpo contra el de la malla | 0.099 / 0.097 / 0.085 contra 0.085 / 0.099 / 0.098 |
| horneado entero, contra cuerpo más pulido, contra relit | 0.137, 0.110, 0.044 — la malla lee 0.085 |
| 64 caminos contra 256 | indistinguibles |
| el Mustang, por modelo contra por malla | 375 199 contra 934 338 gaussianas |

### 6.11 Los fallos que le dieron forma

Cada uno es una línea aquí y una sección allí; el registro tiene el síntoma,
la causa y la medida.

- Los programas del bake se compilaron ciegos a la variante de bake, y una
  gaussiana de 729 073 volvió iluminada.
- Un bake no tiene cámara, así que no puede tener headlight: lo que quedaría
  horneado es una lámpara puesta donde estuviera la cámara de un píxel.
- Horneado a lo largo de la normal, el mármol volvió como plástico pulido — el
  motivo de que exista `bakeBody`.
- Direcciones sorteadas al azar daban sal y pimienta que cuadruplicar los
  caminos no tocó; estratificadas, la misma cuenta salió limpia.
- Un rayo que arrancaba en la superficie horneaba negro a ángulo rasante.
- Un rayo que arrancaba a un milímetro del modelo horneó gris el anillo de oro
  del peón.
- La conversión leía la pose de reposo donde la escena estaba posada, así que
  una nube con esqueleto se construía de un pájaro que ya había salido de su
  bind pose.
- Un slot sin nada dentro se conserva, para que los arrays queden índice a
  índice.
- Un USDZ llega gris, porque nadie declaró el binding que espera.
- La cuenta y el emit, dos kernels, discreparon sobre el centro de una celda
  justo en la diagonal de un cuadrado en cuanto el emit cambió de forma: tres
  slots dados, dos escritos, y el tercero una gaussiana de tamaño cero en el
  origen.
- La red de material perdía el espacio de color de una textura y el albedo
  volvía lineal donde era sRGB.
- La opacidad es cobertura, no cristal: el valor de un mapa de recorte es la
  opacidad de la gaussiana.
- La densidad es de la malla, no de la escena — y hizo falta un suelo bajo un
  coche para verlo.
- El fit promediaba sus samples en el espacio codificado, así que toda zona de
  contacto de toda nube horneada estaba un tercio oscura — cosa que sólo podía
  enseñar un occluder, y la enseñó la esquina de un transfer.
- `--glass-opacity` se enviaba a la conversión con el nombre de un edit, así
  que el número no llegaba a ninguna parte y todo cristal salía opaco.

## 7. Dependencias y toolchain

**Obligatorio.** Slang en `SLANG_ROOT` (un error fatal nombra la release si
falta), OpenUSD con MaterialX y su generador de Slang en `ATHENEA_USD_ROOT`, zlib,
y un compilador de C++20 con Ninja y CMake.

**Traído al configurar.** slang-rhi (fijado a un commit y parcheado), CLI11,
nlohmann_json, tinyexr, Catch2 bajo `BUILD_TESTING`, y GLFW con Dear ImGui
para el viewer.

**Opcional, y qué se pierde sin cada uno**: OpenColorIO (su view transform),
Open Image Denoise (el denoising), libwebp (`.sog`), OpenVDB (`.vdb`), zstd
(`.spz`), las cabeceras de OptiX (los pipelines de rayos en CUDA). Todos se
buscan solo en su propio prefijo, y cada uno imprime una línea de estado al
configurar diciendo si se encontró. Esa línea es el diagnóstico.

**Los cuatro parches a slang-rhi** están en `cmake/patches/`, aplicados al
traerlo por un script que se salta los ya aplicados, porque FetchContent
vuelve a correr el comando sobre un árbol que puede llevarlos ya:

| Parche | Qué arregla |
|---|---|
| longitud del array de render targets en Metal | un array de render target cuya longitud Metal informaba mal |
| formato de vista de textura en Metal | una vista creada con el formato equivocado |
| símbolos del driver de CUDA | un símbolo del driver tapado por el del runtime |
| estructuras de aceleración en Metal | su manejo en Metal |

**Submódulos.** `third_party/gpe` sigue la rama `lrt-fixes`,
`third_party/genlock` sigue `main`. Un cambio en gpe se commitea en el
submódulo, no aquí.

## 8. El registro de diseño

`docs/decisions.md` es el registro: una sección por subsistema o por trabajo,
añadida al final, nunca reordenada. Una sección dice qué se decidió y contra
qué alternativa, qué lo comprueba, qué se midió (con el hardware y el preset),
y qué no está hecho. Los hitos M0 a M11 son también secciones suyas, y el
trabajo posterior se pliega dentro del hito que posee el subsistema en vez de
añadirse como uno nuevo.

Qué va ahí y no aquí: un número, una derrota, una alternativa rechazada, una
medida. Qué va aquí y no ahí: cómo funciona la cosa ahora, y cómo se cambia.

## 9. Deudas conocidas

- **Windows.** El port no ha empezado. `core/Platform.h` es el fichero por el
  que empieza, y los `#ifdef` de ahí nombran lo que falta.
- **El ABI en un comentario.** El comentario de cabecera del `CMakeLists.txt`
  raíz todavía dice que aofx está en el ABI 23; el código va por el 25 desde
  hace tiempo. La regla no puede atrapar un número escrito en prosa.
- **Un módulo que falta en una tabla.** `image` es un módulo real del orden de
  dependencia y no está en la tabla de `CLAUDE.md`.
- **La lista de AOVs, tres veces.** `StageRenderer::displaySource` es la
  verdad; el comentario que tiene encima y la cadena de ayuda de la
  herramienta MCP la repiten, y las dos van ya con retraso respecto a las
  salidas Cryptomatte.
- **Los huecos de Cryptomatte**, escritos donde toca: la ruta trazada de
  splats solos no escribe matte, los puntos no llevan id, y una silueta de
  malla no tiene cobertura subpíxel porque la visibilidad es una muestra.

## 10. Glosario

Los mismos términos que [`operations.es.md`](operations.es.md#10-glosario),
con los que solo importan por dentro:

| Término | English | Qué significa aquí |
|---|---|---|
| buffer de visibilidad | visibility buffer | `(instancia + 1, triángulo)` por píxel, lo que escriben las tres rutas de malla |
| pila de lobes | lobe stack | a lo que evalúa un material en un punto: una lista de términos con peso |
| kernel de comprobación | check kernel | un oráculo de GPU en `shaders/athenea/test`, que cuenta lo que está mal |
| registro | record | un splat tal como lo guarda un fichero, antes de decodificar |
| slot | slot | el índice de un splat proyectado dentro del buffer proj de un frame |
| par | pair | una entrada (tile, splat) que recorre el blend |

## Construir dentro de otro proyecto

El motor se construye como subdirectorio de otro proyecto CMake -- el
compositor para el que se escribió lo construye dentro de su propio árbol --
y el mismo `CMakeLists.txt` sirve a las dos formas. Lo que el padre pone,
antes de `add_subdirectory`:

| variable | significado | valor por defecto standalone |
|---|---|---|
| `ATHENEA_EMBEDDED` | el padre aporta gpe, genlock, aopenfx y spz; el motor deja sus directorios de salida fuera de los del padre | `OFF` |
| `ATHENEA_BUILD_APPS` | `athenea` y `athenea-mcp` | `ON` |
| `ATHENEA_BUILD_PLUGINS` | los bundles de `plugins/` | `ON` |
| `ATHENEA_BUILD_VIEW` | `athenea view` (GLFW, Dear ImGui) | `ON` |
| `ATHENEA_SHADER_OUTPUT_DIR` | dónde se copian los shaders para el compilador en tiempo de ejecución | `<build>/shaders` |
| `AOFX_BUNDLE_DIR` | dónde deja un bundle `aofx_add_bundle` | `<build>/aofx` |
| `AOFX_BUNDLE_ID_PREFIX` | el prefijo DNS inverso del `Info.plist` de un bundle | `rt.sparrow.aofxp.` |

Todo target que un padre pueda tener ya -- `gpe`, `genlock::genlock`,
`aofx::aofx`, `aofx::host`, `slang-rhi`, `nlohmann_json::nlohmann_json`,
`OpenColorIO::OpenColorIO`, `Catch2::Catch2WithMain` -- se busca por nombre
antes de crearlo, y `ofxp::spz` sustituye a la copia propia de spz del motor.
Las rutas propias del motor pasan por `ATHENEA_ROOT` y `ATHENEA_BUILD`, nunca por
`CMAKE_SOURCE_DIR` ni `CMAKE_BINARY_DIR`, que son las del padre; y los módulos
CMake propios se incluyen por ruta, porque un padre bien puede tener un
`cmake/Warnings.cmake` propio antes en la ruta de módulos. `ATHENEA_USD_ROOT` es
ahora variable de caché, así que un build fuera de los presets puede decir
dónde está OpenUSD. Una cosa que saber de un directorio de build que ya
configuró una vez: `find_package` guarda `TBB_DIR` en la caché, y una ruta de
prefijos que cambie después no se consulta -- `cmake -UTBB_DIR` cuando
`athenea info` cuenta dos.

`athenea::engine` es lo que enlaza quien embebe: el renderizador entero, sin el
visor, el servidor MCP, los plugins del host -- ni `athenea::usd`, que quien
embebe y lee stages enlaza al lado, con un USD construido contra el TBB que
tenga el resto de su proceso. `athenea::io` toma el `TBB::tbb` que el padre haya
encontrado antes de buscar el propio de USD, por la misma razón: un TBB.

Quien embebe y ya maneja la GPU le pasa su dispositivo a un stage:
`usd::StageRenderer::open(path, device)` construye el delegado de render y el
motor sobre ese `gpu::Device` en vez de abrir un segundo, y todas las llamadas
al renderizador salen entonces del único hilo de GPU de quien embebe. Listar
los prims de un stage no necesita renderizador: `usd::outline(path, prim)` y
`usd::stageCameras(path)` leen sólo el stage -- sin Hydra, sin dispositivo --
para una vista en árbol en el hilo que la dibuje. `setPrimVisible(prim,
visible)` oculta un prim con una opinión de sesión y, para mostrarlo, borra esa
opinión en vez de forzar `inherited`. `scripts/build-usd.sh` con
`ATHENEA_USD_TBB=<prefijo>` enlaza en el prefijo de USD un TBB que el proceso ya
tiene, en vez de construir otro (el OpenImageIO del compositor trae el de
Homebrew); OpenVDB 10.1 se construye entonces con `tbb/version.h` incluido a
la fuerza, que oneTBB 2021.12 y posteriores necesitan. `drawImage(camera,
time, width, height, technique)` es `render` sin la lectura -- los streams
asentados, un fotograma trazado reunido -- para quien embebe y lleva el
fotograma más allá desde `displaySource` por su cuenta.
`setShaderTexture(shader, input, file)` apunta la entrada de fichero de un
shader a otra textura como opinión de sesión (vacío devuelve la del stage).
Una textura llamada `aofx://<nombre>` la rellena quien embebe desde un buffer
del dispositivo en cada fotograma (`updateExternalTexture`); nunca se lee de
disco.
`usd::registerPlugins(directory)` hace lo que `PXR_PLUGINPATH_NAME`, para un
programa que no puede ponerla antes de que USD arranque (los esquemas de athenea,
desde `<build>/plugin/usd/atheneaSchemas/resources`).
