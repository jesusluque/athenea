<!-- Copyright (c) 2026 jesus luque. -->

# 2. El motor y el renderer athenea

athenea es un renderer de USD construido para que todo paso que toca datos de la escena se ejecute en la GPU.
Dibuja tres clases de primitiva: meshes, con materiales MaterialX y luces UsdLux; puntos; y Gaussian splats 3D.
Ofrece dos técnicas, un raster interactivo y un path tracer que sirve de verdad de referencia. Hydra alimenta
ambas a través del delegado de render `hdAthenea`. Esta sección cubre la arquitectura del motor, los algoritmos que
ejecuta cada subsistema, cómo se establece la corrección sin un CPU oracle y lo que se ha medido. Salvo que se
indique otra cosa, cada número se cita de `docs/decisions.md`, con su unidad y sus condiciones, y se midió en un
Apple M5 Pro con Metal.

## 2.1 Reglas de diseño

El código se atiene a un número reducido de reglas. Donde es posible, una prueba hace cumplir cada una.

1. **Ninguna aritmética de CPU sobre los datos.** athenea no tiene renderer de referencia en CPU, ni alternativa
   en CPU, ni CPU test oracle. El host lee ficheros, analiza cabeceras, descomprime y lleva las cuentas:
   recuentos, ranuras y colas. La decodificación, el sort, la fusión, el descarte, la generación de mips, los
   prefijos de potencia de las luces y la comparación de imágenes son todos kernels de cómputo. Los pasos que se
   ejecutaban en el host se llevaron al device; son ejemplos el entrelazado
   por elemento de los arrays de `ParticleField`, la conversión de half a float, la conversión por píxel de los
   buffers de render de Hydra y la parte acumulada de cada luz en la potencia del frame (`light_prefix.slang`). La
   única excepción documentada es la teselación de Dear ImGui, que procesa la interfaz del viewer y ningún dato de la
   escena.
2. **Enlace por nombre.** Todo parámetro de shader se enlaza por reflexión (`cursor["name"].setBinding(...)`,
   `ComputeKernel::dispatch`) y nunca por ranura, lo que elimina la clase de errores en que un recuento de
   ranuras se aparta de la declaración del shader.
3. **Errores como valores.** Los errores se devuelven como `Result<T>` con `ATHENEA_TRY`, y nada lanza excepciones.
   Las llamadas al sistema operativo viven solo en `modules/core/Platform`, que es también donde empezaría un port a
   Windows.
4. **Un Slang, un slang-rhi, un TBB por proceso.** OpenUSD, OIDN y el motor comparten un único TBB, y la prueba
   `single_tbb` comprueba el recuento. Por la misma razón OIDN se compila desde el código fuente: sus binarios
   publicados traen un segundo `libtbb.12`.
5. **Solo devices GPU, en todas partes.** OIDN se compila solo para devices GPU, porque un denoiser que recurre a la CPU es una alternativa en CPU. Una prueba cuya capacidad falta en un device se omite y
   nombra la capacidad que falta.

Los backends son los de slang-rhi: Metal (macOS, iOS), Vulkan y CUDA (Linux, NVIDIA L4). Todos los kernels están
escritos en Slang.

## 2.2 Arquitectura de módulos

Cada módulo bajo `modules/<name>` es una biblioteca estática, `athenea::<name>`. Los módulos se listan en orden de
dependencia y cada uno enlaza solo con los anteriores (`modules/CMakeLists.txt`) (figura 2).

| Módulo | Contenido |
|---|---|
| core | `Result`, registro, `Platform` (todas las llamadas al SO) |
| ui | los paneles del viewer como descripción (`Controls.h`, `GaussianPanel.h`), traversals por `athenea view` y la app de iOS |
| sched | `FrameClock`: genlock PTP y alineación ST 2059-1, timecode |
| image | `Image`: float32 RGBA premultiplicado, origen abajo a la izquierda; el tipo que intercambian gpe y AOFX |
| io | lectores de ficheros en CPU (cabecera PLY, SPZ, SOG zip/WebP, IES), EXR con atributos |
| gpu | device slang-rhi, `ShaderLibrary`, `ComputeKernel`, `CommandBatch`, `Buffer`, `AsyncReadback`; `gpu/algo` (PrefixSum, RadixSort) |
| gpu_host | gpe adoptando el device de slang-rhi: un `MTLDevice` o contexto CUDA, buffers compartidos sin copias |
| colour | `ColourCompiler` (OpenColorIO como compilador a Slang), `ColourNames` |
| scene | `CloudLoader`: registros en bruto subidos y decodificados en la GPU en `GpuSplats` / `GpuPoints` |
| render | `TileRasterizer`, `GaussianRayTracer`, `PointRasterizer`, `ReferenceRenderer`, `Camera`/`Projection`, `SplatEdit` |
| geom | `MeshBuilder`: meshes de Hydra trianguladas, con normales suaves y primvars expandidos en la GPU |
| material | `TextureStore`, `MaterialCompiler` (MaterialX → Slang), la biblioteca de lobes |
| light | luces UsdLux en el device: un registro por luz, muestreo, densidades, BVH de luces |
| world | `GpuScene`, `Instancing`, `RayTracingScene` (BLAS/TLAS), `BvhScene` (LBVH de cómputo en dos niveles) |
| technique | `VisibilityRaster/Trace/Bvh`, shading, `AovShading`, path tracer, `Denoiser`, `DisplayTransform`, `SplatVisibility` |
| lod | `LodBuilder`, `CutSelector`, el lector/escritor de `.athc` y `StreamingPool` (Sección 6) |
| usd | `Engine`, `StageRenderer`, `Export`, el plugin `hdAthenea`, esquemas sin código |
| mcp | el motor como servidor MCP (JSON-RPC 2.0) |
| aofx | el lado de este motor del host AOFX (`host/` de aopenfx), `renderEffect`, los bundles |
| view | `athenea view`: ventana GLFW, Dear ImGui en el device del motor |

![Figura 2. Módulos del motor en orden de dependencia, con sus clases principales. Cada biblioteca estática enlaza solo los módulos de debajo.](figures/fig-modules.es.svg)

El tree de shaders `shaders/athenea/` refleja los módulos (common, algo, scene, splat, rt, points, reference, lod,
geom, world, material, light, technique, usd, volume, view). Los shaders se copian al tree de compilación y se
compilan en tiempo de ejecución.

## 2.3 Fundamentos de GPU

**Device y kernels.** `gpu::Device` envuelve un device slang-rhi. `ComputeKernel` es un punto de entrada de
cómputo compilado como pipeline. Se despacha en hilos, no en grupos: el número de grupos se redondea hacia arriba en
todos los backends y cada kernel comprueba sus propios límites. `RasterKernel` es un pipeline de vértices y
fragmentos que toma sus datos de `StructuredBuffer`s, sin buffers de vértices ni layouts de entrada.
`RayTracingKernel` cubre los pipelines de OptiX y de Vulkan RT. En Metal, slang-rhi no ofrece pipelines de ray
tracing, así que el trazado usa `RayQuery` en línea dentro de kernels de cómputo.

**ShaderLibrary.** La biblioteca carga módulos Slang. Un módulo puede declarar constantes de enlace
(`extern static const uint kName;`), y entonces se enlaza contra un módulo de exportaciones generado, lo que da un
programa por cada conjunto distinto de valores. También compila módulos que solo existen como código generado, como
los materiales y las funciones de color; se rechaza cargar un nombre existente con otro código. Una
`DiskShaderCache` persistente escribe un fichero por programa compilado, con la clave del hash que slang-rhi da al
programa enlazado. Con la caché caliente, la batería de pruebas completa pasó de 89 s a 26 s.

**Primitivas paralelas.** `PrefixSum` y `RadixSort` son las únicas ordenaciones y barridos del motor. La ruta radix
por bloques (`radix_block.slang`) es una forma reducir-y-luego-barrer de la pasada LSD estable. Un grupo de 256 hilos
cuenta un bloque de 1024 claves en memoria de grupo, una suma prefija ordenada por dígito sitúa la parte de cada
bloque en cada dígito, y el grupo ordena su bloque con ocho divisiones de un bit. Ordena 10 M de pares de 32 bits en
**12.4 ms**, frente a 39.9 ms de las pasadas por trozos a las que sustituyó a partir de 65 536 elementos. No usa
operaciones de onda ni atómicas de device.

**Mips.** slang-rhi no genera cadenas de mips, así que `MipGenerator` es un kernel. Cada texel de un nivel es la media
ponderada por área de los texels de encima. Un borde impar $2n+1$ se pliega en $n$ con pesos
$(n-x,\,n,\,x+1)/(2n+1)$, de modo que cada texel de origen da exactamente $n/(2n+1)$ de sí al nivel inferior y la
cadena conserva la media del nivel 0. En tamaños 64², 37×23, 1×9 y 128×5, las medias coinciden a 2·10⁻⁶.

**La memoria como error de primera clase.** Un command buffer de Metal que se quedaba sin memoria provocaba una
aserción dentro de slang-rhi en el hilo de finalización de Metal, que terminaba el proceso. Un parche registra ahora
el error en la cola (`SLANG_E_OUT_OF_MEMORY`) y lo devuelve en el siguiente `submit()` o `waitOnHost()`.
`Device::admit` rechaza toda reserva que exceda el budget: `ATHENEA_GPU_BUDGET` o, en su defecto, el
`recommendedMaxWorkingSetSize` de Metal, que es 18 186 MiB en el M5 Pro. Una reserva debe caber además en la memoria
física libre menos 1.5 GiB. En memoria unificada, varios procesos, cada uno dentro de la cifra de Metal, pueden agotar
juntos la memoria física y obligar al sistema a paginar. Cuando un frame se queda
sin memoria, `Engine::relieveMemory` baja escalones: primero suelta las sombras de splats, después pasa hasta cuatro
niveles de LOD más gruesos y por último reabre los streams a la mitad de su budget.

## 2.4 La representación en Gaussians

Todos los renderers, los cargadores y el LOD comparten un único layout en GPU, `GpuSplats`. Cada splat ocupa
**32 bytes antes de sus SH**, definidos en `common/packing.slang`:

| Campo | Codificación | Razón |
|---|---|---|
| posición, opacidad | 4 × float32 | posiciones en fp16 saltarían dos píxeles a 4K de cerca |
| rotación | cuaternión smallest-three, 3 × 10 bits + índice omitido | dentro de una décima de grado |
| escala | f16 de $\log s$ por eje | 0.1 % de precisión relativa a cualquier tamaño |
| color base | 3 × f16 | puede salir de [0, 1] |
| resto de SH | `shWords` × u32, f16 por coeficiente | con signo |
| normal (opcional) | octaédrica 2 × 16 bits | shading normal separada del marco |
| emisión (opcional) | RGB9E5 | radiancia que emite la superficie |

`CloudLoader` lee PLY, SPZ, SOG, `.splat` y prims `ParticleField` de USD. Sube los registros en bruto y los decodifica
en la GPU. Para SOG, el único trabajo del host es descomprimir el zip y decodificar las imágenes WebP.

**Luz lineal.** Toda Gaussian se mezcla en Rec.709 lineal, y el sRGB solo aparece donde se muestra una imagen. Una
captura conserva los SH sRGB con los que se entrenó: cada proyección los evalúa, los recorta en cero y los
convierte a lineal (`cloudLight`) antes de mezclar. Una nube escrita por athenea declara que contiene luz
(`primvars:athenea:splat:linear`) y se usa tal como está almacenada. Sobre dos discos semiopacos solapados, el
rasterizer, la referencia y el ray tracer coinciden cada uno con el operador over analítico a p99 0. Con la
mezcla codificada anterior, los tres leían p99 26.

## 2.5 El tile rasterizer

`render::TileRasterizer` es la ruta interactiva para splats. Es un tile rasterizer determinista, basado en
sort, de la familia de 3DGS. Siete etapas se ejecutan en el device (figura 3):

| Etapa | Trabajo |
|---|---|
| project | por splat: covarianza, cónica, color, clave de profundidad de 24 bits, footprint, número exacto de tiles tocadas |
| sort | splats por profundidad (RadixSort) |
| gather | recuentos de tiles tocadas en orden de profundidad, suma prefija |
| emit | un par (tile, splat) por tile tocada, en orden de profundidad |
| sort | pares por tile, de forma estable, solo con los bits de tile |
| ranges | inicio y fin de los pares de cada tile |
| blend | por píxel: los pares de su tile, de delante hacia atrás |

El motor ordena los splats por profundidad una sola vez (N claves de 32 bits) y después ordena los pares de forma
estable por tile (P claves de unos 15 bits). Así cada tile recibe sus splats en orden de profundidad, con los
empates resueltos por índice de splat, exactamente como lo haría una única sort de 64 bits por
(tile, profundidad). La construcción es determinista. Las tiles son de 16 × 16 píxeles.

![Figura 3. El tile rasterizer: siete etapas de compute, tres por splat y cuatro por par (tile, splat).](figures/fig-raster.es.svg)

### 2.5.1 Proyección

Un splat con rotación $R$ y escalas $S=\mathrm{diag}(s)$ tiene la covarianza 3D $\Sigma = R S S^\top R^\top$,
construida en el espacio propio de la nube. Con $W$ la parte lineal de objeto a vista y $J$ el Jacobian de la
aplicación perspectiva en el centro en espacio de vista $t$, la covarianza en pantalla es la aproximación afín local
EWA

$$\Sigma_{2D} = (J W M)(J W M)^\top,\qquad M = R S,\qquad
J=\begin{pmatrix} f_x/t_z & 0 & -f_x t_x/t_z^2\\ 0 & f_y/t_z & -f_y t_y/t_z^2\end{pmatrix}.$$

Antes de construir $J$, $t_x/t_z$ y $t_y/t_z$ se acotan al frustum ensanchado a $[-0.15, 1.15]$ de la imagen. Un
splat muy fuera de pantalla se extendería si no por todo el frame, y 3DGS acota de la misma manera. Las cámaras
ortográficas usan $J=\mathrm{diag}(f_x,f_y)$.

**Shutter y lente, como convoluciones.** Una Gaussian arrastrada a lo largo de un camino recto en pantalla $d$ bajo
un shutter uniforme es esa Gaussian convolucionada con un segmento. Igualar segundos momentos convierte esto en una
actualización de rango uno. Una lente delgada convoluciona con el disco del círculo de confusión:

$$\Sigma' = \Sigma + \tfrac{1}{12} d\,d^\top,\qquad
\Sigma' = \Sigma + \tfrac{R^2}{4} I,\quad R = f\,A\,\left|\tfrac1z - \tfrac1{z_f}\right|.$$

El desplazamiento $d$ tiene dos fuentes: lo que un rig movió el splat durante el shutter (un buffer por
splat) y un paso en espacio de vista que combina el movimiento de la cámara y el del prim (`viewStep`). Ambos se
proyectan con las mismas filas $JW$ que la covarianza. La actualización se comprueba frente al mismo rasterizer
ejecutado en N instantes y promediado en el device (§2.5.3).

**Filtro antialiasing y la opacidad que devuelve.** A continuación la covarianza se dilata en 0.3 px², como en el
filtro 2D de Mip-Splatting. Cada convolución ensancha el splat, así que su pico debe bajar en
$k = \sqrt{\det\Sigma_{\text{still}}/\det\Sigma'}$ para conservar su energía. Una sola raíz cuadrada cubre a la vez el
filtro, el shutter y la lente. Escalar sin más pierde masa, sin embargo, porque un splat solo se dibuja hasta donde
cae por debajo de 1/255, y el ensanchamiento agrandó esa área en $1/k$. Un plano convertido que debía cubrir 0.25
cubría 0.18 a una vigésima de la pantalla. Por eso la opacidad se devuelve como

$$\alpha' = \begin{cases} k\alpha + (1-k)/255 & k\alpha \ge 1/255\\ k\alpha & \text{en otro caso.}\end{cases}$$

Esto conserva el área cubierta del splat nítido (`paidOpacity`). Devolverla a todos los splats dibujaría también los
que el filtro había dejado demasiado tenues para conservarse. Eso subía un peón convertido lejano de 8.7 a 29.8 ms
(compilación debug), y la regla de arriba cuesta allí 11.0 ms.

**Footprint y tiles.** Un splat se corta en el radio de Mahalanobis al cuadrado

$$r^2_{\max}(\alpha) = \min\big(2\ln(255\,\alpha),\ 9\big),$$

que es donde su propia contribución cae por debajo de 1/255, y nunca más allá de 3σ. El recuento de tiles es exacto
en lugar de tomarse de una caja envolvente. Para cada tile de la extensión alineada con los ejes, el mínimo de la
forma cuadrática $q(\delta)=\delta^\top \Sigma'^{-1}\delta$ sobre la caja de los centros de píxel de la tile se
calcula en forma cerrada: como $q$ es convexa, es 0 o está en una arista, donde es una cuadrática 1D acotada. La
etapa emit repite la misma prueba. Como la mezcla corta en el mismo radio, ningún píxel fuera de una tile contada
pudo ser tocado, así que los bordes de las tiles no dejan costuras. Una ranura descartada escribe el motivo del
descarte en su palabra de clave de profundidad sin usar (edit, profundidad, sin área, tenue, fuera de pantalla, sin
tile), y el panel Gaussians del viewer cuenta esos motivos sin añadir un contador a la proyección.

**Clave de profundidad.** La clave es `bit_cast<uint>(z) >> 7`, que conserva el orden para floats positivos y tiene
24 bits.

### 2.5.2 Mezcla

Un grupo de 16 × 16 hilos por tile carga los registros de la tile en tandas de 256 en memoria de grupo, y cada
píxel los recorre desde ahí. Para el píxel $p$ y el splat $i$ con cónica $Q_i=\Sigma_i'^{-1}$, centro $\mu_i$ y color
$c_i$:

$$\alpha_i(p) = \min\!\Big(0.99,\ \alpha_i' \exp\!\big(-\tfrac12 (p-\mu_i)^\top Q_i (p-\mu_i)\big)\Big),\qquad
C = \sum_i c_i\,\alpha_i \prod_{j<i}(1-\alpha_j) + T\,C_{\text{behind}}.$$

Las contribuciones por debajo de 1/255 se saltan. Un píxel se detiene en cuanto $T=\prod(1-\alpha_j) < 10^{-4}$, y el
grupo deja de cargar cuando sus 256 píxeles han terminado. $C_{\text{behind}}$ es el fondo o la capa opaca *under* de
meshes y puntos, compuesta por z de vista: la traversal termina en el primer registro detrás de la profundidad de la
capa. La profundidad se escribe como la media ponderada por alfa o como el primer cruce de un umbral de coverage. El
misma traversal acumula también un Cryptomatte de seis rangos, por coverage de id, y, para la poda, la contribución
por splat $T\alpha$ o el error cuadrático que dejaría quitar el splat (la puntuación de GaussianPOP). Puntos de entrada
especializados eliminan en compilación las ramas de composición y de mate, porque dejarlas vivas en un frame que
no tiene ninguna de las dos costaba a la mezcla la mitad más.

**Surfels y núcleos más duros** (rama athenea-pieces). Una Gaussian con una escala exactamente 0 es un *surfel*.
El rasterizer lo dibuja intersecando el rayo de cada píxel con el plano del disco (Huang et al. 2024): los ejes del
disco se guardan en coordenadas homogéneas de pantalla y se llevan de vuelta a través de los dos planos del píxel. El
filtro de pantalla es un máximo, no una convolución: el mayor entre el disco y una Gaussian redonda de varianza
½ px². Su opacidad se devuelve por área,

$$\alpha' = \alpha\cdot\min\!\Big(1,\ \frac{\sqrt{\lambda_1\lambda_2}}{\sqrt{\lambda_1\lambda_2}+\tfrac12-\sqrt{\min(\lambda_1,\tfrac12)\min(\lambda_2,\tfrac12)}}\Big),$$

donde $\lambda_{1,2}$ son los autovalores de la covarianza en pantalla sin dilatar. Una Gaussian merged puede
llevar una opacidad $D>1$. Ese valor selecciona el núcleo más duro de Spark, $1-(1-g)^A$ con $A=\exp((D^2-1)/e)$ y
$D$ limitado a 5. Es la composición de $A$ Gaussians coincidentes, y su corte se ensancha en $0.7(D-1)\sigma$. Un
*surfel duro* usa $\exp(-(|u|^p+|v|^p)/2)$, con $p$ ajustado a la curtosis de sus miembros. La Sección 6 cubre ambos,
junto con los merges que los producen.

### 2.5.3 Medidas

**Coverage de superficies convertidas** (`athenea_coverage_tests`, 256 × 256, alfa de planos convertidos por
mesh2splat, frente al alfa del mesh; los tamaños son fracciones de la pantalla completa):

| Plano | Mesh | Raster 1× / 0.25× / 0.05× | Ray tracer 1× / 0.25× / 0.05× |
|---|---|---|---|
| constante 0.25 | 0.250 | 0.251 / 0.250 / 0.248 | 0.251 / 0.251 / 0.251 |
| constante 0.5 | 0.500 | 0.501 / 0.499 / 0.493 | 0.501 / 0.501 / 0.500 |
| constante 0.75 | 0.750 | 0.752 / 0.749 / 0.736 | 0.752 / 0.752 / 0.752 |
| opaco | 1.000 | 1.000 / 1.000 / 0.999 | 1.000 / 1.000 / 1.000 |

Sin `paidOpacity`, el raster leía 0.18 / 0.455 / 0.73 a 0.05×.

**Blur de movimiento y lente** (`[motion]`, `[lens]`; verdad = el mismo rasterizer en N instantes, o en 64
posiciones de lente, promediado en el device):

| Caso | Resultado |
|---|---|
| paso de shutter 0.10 (muchos anchos de splat) | p99 8, máx 24 |
| paso de shutter 0.03 | p99 2, máx 9; razón de energía 1.0000 |
| paso de shutter 0.9 | razón de energía 0.9030 (colas bajo el corte de 1/255) |
| shutter nulo frente a still frame | máx 0, bit a bit |
| f/2, 12.7 px de confusión | p99 17 (sin desenfocar: p99 113) |
| f/11, 2.3 px de confusión | p99 9 (sin desenfocar: p99 33) |
| a la distancia de enfoque | máx 0, bit a bit |

**Coste del frame por etapa** (`ATHENEA_STAGES=1`, esperando a cada etapa; los clips del gorrión, 5 887 323
Gaussians a 1600 × 1000, en reproducción):

| Etapa | Antes | Después | Cambio |
|---|---|---|---|
| depth sort (24 bits) | 11.3 ms | 6.6 ms | radix por bloques |
| tile sort (8.8 M pares) | 8.4 ms | 6.0 ms | radix por bloques |
| mezcla | 13–25 ms | 8–13 ms | registros de la tile compartidos en memoria de grupo |
| proyección, emisión, recuentos, commit | 13.6 ms | 13.4 ms | – |
| frame, en reproducción | 102 ms | 49 ms | la nube ya no se resube en cada paso |

Un frame de seis millones de Gaussians gasta su tiempo en las Gaussians y no en los píxeles: cambiar la escala
de render de 800 × 500 a 3200 × 2000 deja el tiempo igual.

## 2.6 El ray tracer de Gaussians

`render::GaussianRayTracer` evalúa cada partícula en 3D en su respuesta máxima a lo largo del rayo, sin aproximación
EWA y sin la dilatación de 0.3 px. Las partículas se mezclan de delante hacia atrás en el orden de su pico a lo largo
del rayo. En el marco blanqueado de la partícula (origen $o$, dirección $d$, con $\Sigma^{-1/2}$ aplicado), el pico y
la entrada en el elipsoide de corte son

$$t^\ast = -\frac{o\cdot d}{d\cdot d},\qquad \rho = \lVert o + t^\ast d\rVert^2,\qquad
t_{\text{entry}} = t^\ast - \sqrt{(r^2_{\max}-\rho)/(d\cdot d)}.$$

El discriminante de libro $b^2-ac$ se cancela catastróficamente en partículas planas y perdía 176 de 400 capas finas,
por eso se usa la forma del pico. El color es el del rasterizer: los SH evaluados una vez por frame en
la dirección del ojo al centro.

**Dos rutas, un integrador.** Ambas rutas alimentan el mismo `Integrator` (`rt_integrate.slang`). La ruta *Hardware*
da a cada partícula un proxy de icosaedro estirado (el de 3DGRT) en una BVH del device consultada con `RayQuery`
en línea. La ruta *ComputeBvh* es una LBVH de Karras construida en la GPU (códigos de Morton, radix sort, jerarquía,
reajuste hasta que una pasada no cambia nada) y recorrida en cómputo. Cada traversal registra hasta 256 entradas sin
ordenar, que se evalúan y ordenan tras la traversal con una counting sort en 128 cubetas y después una
sort por inserción sobre una permutación de índices. Cuando un rayo encuentra más proxies que esos, el segmento
se cierra en la entrada descartada más cercana y la siguiente traversal se reanuda un relativo 10⁻⁵ antes de ella.
Las partículas cuyo pico cae más allá del límite se arrastran al siguiente segmento; solo cuando se arrastran más de
64 se mezclan antes las más cercanas, que es la única aproximación de la ruta.

| train_7k (742 k splats), 1080p | Frame | Construcción |
|---|---|---|
| Hardware (Metal) | 435 ms | 731 ms |
| ComputeBvh (Metal) | 343 ms | 168 ms |
| tile rasterizer, mismo frame | 13 ms | – |

En Metal, cada candidato no opaco cuesta un viaje de ida y vuelta fuera de la traversal por hardware. Una consulta
opaca de impacto más cercano sobre los mismos 15 M de triángulos proxy tarda 6 ms, pero enumerar todos los candidatos
tarda 255 ms, así que `Auto` elige ComputeBvh en Metal. La historia de optimización de este frame fue de 2.52 s
(un k-buffer de 16 entradas al estilo de 3DGRT) a 343 ms. Los pasos mayores fueron registrar entradas sin ordenar y
evaluar tras la traversal (1.94 s → 658 ms) y el sort por cubetas (−85 ms); el tamaño de registro de 256 se
eligió de un barrido de 64 a 512.

**Consultas de sombra y secundarias.** La transmitancia es un producto, así que `rt_shadow.slang` toma los proxies en
el orden en que llegan, sin k-buffer, y se detiene cuando el producto cae por debajo de su corte. No descarta caras
traseras, porque un rayo de sombra nacido dentro de un proxy debe seguir viendo la partícula en la que empieza. Tres
formas cerradas lo comprueban: $1-\alpha$ a través de un centro, medido 0.50000; $(1-\alpha)^4$ a través de cuatro,
medido 0.06250; y la terminación temprana tras tres de cuatro. El integrador responde a una consulta
`rtTraceSegment(ray, [a, b))` en lugar de escribir un píxel, y `segmentOver` compone dos segmentos: una nube dispersa
partida a cualquier profundidad es exacta bit a bit con la consulta única.

## 2.7 Puntos

`PointRasterizer` dibuja nubes de puntos con un pipeline raster con z-buffer y sin sort, seguido opcionalmente
de surface splatting y de eye-dome lighting. En devices sin rasterización (CUDA), los puntos se dibujan como
discos en el pipeline de splats. En ese caso un `colour.w > 0.5` en un registro lo marca como disco opaco, que termina
la traversal exactamente como lo hace la capa under.

## 2.8 Meshes: geometría, escena y visibilidad

**En el device, en el orden de Hydra.** `geom::MeshBuilder` triangula la topología de Hydra en el orden de
abanico de `HdMeshUtil`, con agujeros y orientación levógira incluidos. Calcula normales suaves con la fórmula de
`Hd_SmoothNormals`, como productos vectoriales por esquina dispersados a los puntos mediante un radix sort, y expande
primvars de toda interpolación por esquina, resolviendo los primvars indexados y decodificando los dobles en el
device. Las normales de un campo de alturas quedan a 1.1·10⁻⁷ rad de una suma por fuerza bruta.
`world::GpuScene` agrupa posiciones, índices, esquinas, caras y valores de primvars, con un registro de 64 bytes por
mesh y un `InstanceRecord` de 176 bytes por copia dibujada. `world::Instancing` compone las cadenas de instanciadores
exactamente como Storm, $\text{instancer}\cdot T\cdot R\cdot S\cdot\text{instanceTransform}$, anidadas como
$\text{parent}[i]\cdot\text{level}[j]$, con un único dispatch para todos los conjuntos instanciados. En los 1462
conjuntos de Kitchen_set_instanced esto llevó la grabación de comandos de 32 ms a 0.4 ms.

**Tres rutas, un buffer de visibilidad.** Cada ruta escribe (instancia + 1, triángulo) por píxel. El shading y los
AOVs reconstruyen el impacto a partir de esos dos números con Möller–Trumbore en espacio de vista
(`technique/surface.slang`), así que todas las rutas sombrean igual.

- `VisibilityRaster` usa Z infinita invertida (profundidad $= \text{near}/z$, D32Float), un draw por mesh con
  `instanceCount` y un único objeto raíz compartido.
- `VisibilityTrace` construye una BLAS por mesh y una TLAS por frame. Un kernel escribe los descriptores de
  instancia en el layout de cada backend (64 B genérico, 80 B OptiX, 68 B Metal).
- `VisibilityBvh` construye una LBVH de Karras por mesh y otra sobre las instancias, para devices sin hardware
  de RT.

El descarte de cara única coincide con Storm en las tres rutas.

**Storm como oracle** (Kitchen_set, 480 × 270; ids comparados como segmentación, porque Storm numera los prims de
otra forma):

| Ruta | Coverage | Segmentación | Profundidad, peor | Neye > 6/255 |
|---|---|---|---|---|
| raster | 0 difieren | 18 de 57 219 | 1.2·10⁻⁶ | 17 |
| rays | 4 difieren | 41 de 57 236 | 6.3·10⁻⁶ | 17 |
| bvh | 4 difieren | 36 de 57 233 | 6.3·10⁻⁶ | 17 |

**Tiempos de visibilidad** (`athenea stage --frames 40`, mediana de los frames tras el primero, shading con luz
de cámara, lectura incluida):

| Escena | Tamaño | raster | rays | bvh |
|---|---|---|---|---|
| Kitchen_set (1788 meshes) | 480 × 270 | 18.0 ms | 4.8 ms | 8.9 ms |
| Kitchen_set | 1920 × 1080 | 31.7 ms | 19.2 ms | 29.2 ms |
| Kitchen_set_instanced (1462 conjuntos) | 1920 × 1080 | 29.1 ms | 19.2 ms | 28.6 ms |

El raster pierde sobre todo en el lado del host: grabar unos 1800 draws cuesta unos 8 ms, porque slang-rhi escribe el
render state por draw. Por eso el ajuste automático prefiere rayos dondequiera que el device tenga consultas
de rayos.

## 2.9 Materiales

**Texturas.** El `TextureStore` usa Hio solo para leer los bytes de un fichero. Los bytes se suben en bruto y un
kernel los decodifica al espacio de trabajo. Los mips son un kernel que decodifica el sRGB, filtra en luz y vuelve a
codificar. UDIM es una tabla de indirección. Todas las texturas están en un único `ParameterBlock` de 1024 ranuras,
que Metal toma como argument buffer. El filtrado se elige por destino con `__target_switch`: `SampleGrad` donde el
cómputo lo permite, y un nivel explícito a partir del lado mayor del footprint en otro caso (CUDA). En Metal, donde
existen ambos, eligen los mismos niveles. Una textura decodificada es el fichero hasta el último bit (0 de 3404
componentes difieren). Los espacios de color de entrada se resuelven con `colour::ColourNames`, y las
transformaciones las genera OpenColorIO (§2.12).

**De MaterialX a Slang.** El generador deriva del `SlangShaderGenerator` de MaterialX 1.39.5. Cada nodo conserva su
implementación genglsl o genslang salvo cuatro clases cuyo significado de rasterizer no puede valer aquí:

- el nodo de superficie, que aquí no tiene bucle de luces: su grafo de BSDF se ejecuta una vez, empujando lobes;
- los nodos BSDF y EDF, que empujan lobes en lugar de responder a una luz;
- los nodos de imagen, que muestrean la tabla de texturas;
- `heighttonormal`, que necesita derivadas de pantalla.

Un valor de BSDF es un *peso por lobe construido*, así que `mix`, `layer`, `add` y `multiply` combinan pesos. Los
uniforms no se fijan en el código: cada entrada se lee de un blob de floats, y el módulo se nombra por un hash de su
código, de modo que los materiales que solo difieren en valores comparten un módulo compilado. Entre los tamaños
generados están UsdPreviewSurface con 580 líneas y 23 palabras de blob, standard_surface con 821 y 59, y OpenPBR con
945 y 55. `technique::MaterialPrograms` genera un módulo que importa todos los materiales compilados y despacha según
la fila de material. Un kernel que evalúa materiales recorre sus píxeles en orden de quad (`atheneaQuadPixel`),
porque en Metal los lanes van a lo largo de las filas, y en orden de filas todas las derivadas verticales son
erróneas (15 939 de 15 939 hilos).

**La biblioteca de lobes.** Contiene Oren–Nayar y EON con compensación de energía, Burley, translúcido, dieléctrico
(reflexión, transmisión, ambas), conductor, Schlick generalizado con tinte F82, y sheen en sus formas de Imageworks y
de Zeltner. Cada lobe tiene `eval`, `sample` y `pdf`. Los lobes de microfacetas muestrean la distribución de
normales visibles y la transmisión sigue a Walter. Los lobes de un material forman una `LobeStack`, que se muestrea
con MIS de una muestra según los albedos direccionales ponderados de los lobes.

**Comprobaciones contra el propio MaterialX.** Los mismos grafos se compilan una segunda vez en una variante de
referencia cuyos closures son las propias respuestas genglsl de MaterialX, y un kernel evalúa ambas para 65 536
direcciones de luz. La peor diferencia por componente sobre once grafos es **3.2·10⁻⁵**. En el device, los
lobes pasan una prueba χ² de las direcciones muestreadas frente a la pdf (393.2 con 399 grados de libertad para los
lobes difusos) y un horno blanco que coincide dentro del 3 %.

**Recortes y bump.** La visibilidad evalúa `opacityThreshold`: el raster usa un fragment shader generado que
descarta, y las rutas de rayos llevan el rayo más allá de una muestra recortada hasta dieciséis veces. Solo las filas
marcadas como recortes pagan por ello. Un cuadrado recortado muestra exactamente lo que hay detrás (8464 píxeles,
0 discrepancias) en las tres rutas.

## 2.10 Luces

**Registros y unidades.** Cada luz UsdLux se convierte en un registro en espacio de mundo, en las unidades que el
stage escribió. La exposición, el cuerpo negro de una temperatura de color (ajuste de Krystek, normalizado a luminancia
1) y `normalize` se derivan en el shader. Las clases admitidas son esfera, disco, rectángulo, distante con ángulo,
dome y cilindro, además de conos de shaping y perfiles IES. Un perfil IES se lee tal como está escrito, se muestrea
bilinealmente con la simetría que declara y modula la radiancia pero nunca la densidad. Las luces bajo instanciadores
se componen en el device como los meshes.

**Muestreo donde está la luz.** Una esfera o un sol se muestrean uniformemente sobre el cono que subtienden. Un disco,
un rectángulo o un cilindro se muestrean uniformemente por área y se pasan a ángulo sólido con
$p_\omega = d^2/(\cos\theta_l\,A)$. Un dome se muestrea con peso coseno alrededor de la shading normal o
deformando su cadena de mips por luminancia. La densidad de la deformación no necesita tablas:

$$p(\omega) = \frac{L(u,v)}{\bar L\; 2\pi^2 \sin\theta},$$

donde $\bar L$ es la media del nivel de mip de 1 × 1. La variación de la imagen elige entre las dos estrategias: con
un cielo plano la deformación se desvía un 20 % donde el coseno se desvía un 0.08 %. `lightPdf` devuelve la densidad
de cualquier dirección y `lightHit` responde a la pregunta inversa, qué luz encuentra un rayo y con qué radiancia.
Juntas son lo que necesita el MIS. Con `athenea:chooseLights`, una muestra toma una luz elegida mediante una **BVH de
luces** (`light_bvh.slang`). Cada nodo es una caja, un cono de orientación (Conty y Kulla 2018) y la potencia que hay
debajo, y la BVH se construye con los mismos kernels de LBVH de cómputo que la geometría.

**Comprobaciones.** Cada luz se contrasta con una forma cerrada que no comparte código con el renderer: un plano
lambertiano bajo una esfera, un disco, un rectángulo (fórmula de aristas de Lambert), un sol y un dome. En cada
caso 0 de 8281 píxeles superan el 2 %; el peor es un 1.6 % para el disco. Cada luz ejecuta además una prueba χ² con
10⁶ muestras, agrupadas en el soporte propio de la luz, frente a `lightPdf` integrada por intervalo: esfera
z = −0.23, disco 0.14, rectángulo −0.27, sol −0.29, dome −1.61, y toda pdf integra 1.0000. La deformación del
dome necesitó dos correcciones, ambas halladas con χ² en espacio de imagen. El error llegó a z = 61 543 y después a
1 164 589, frente a 0.84 tras las correcciones. La comprobación de consistencia por muestra había pasado todo el
tiempo, porque compara una densidad consigo misma.

**Medidas** (Kitchen_set con cuatro luces escritas, 1600 × 900, `athenea view --frames 200`, mediana de dibujo):

| Configuración | Dibujo |
|---|---|
| solo luz de cámara, rays | 10.61 ms |
| cuatro luces, 1 muestra, rays | 27.77 ms |
| lo mismo, BVH de cómputo | 35.14 ms |
| lo mismo, raster | 40.94 ms |
| cuatro luces, 16 muestras, rays | 275.40 ms |

## 2.11 El path tracer

El path tracer es la verdad de referencia del motor. Sus paths parten del mismo buffer de visibilidad que lee
el shading raster, así que las dos técnicas difieren exactamente en una cosa: los bounces. Con una lente, el
path tracer lanza además sus propios rayos primarios. Un impacto lo reconstruye `material_surface.slang`, su material se
evalúa en la misma pila de lobes, y la luz directa se recoge con estimación del siguiente evento (NEE). El bounce
muestrea la pila, y el throughput del camino se multiplica por `LobeSample.weight` $= f|\cos\theta|/p$ (figura 4).

![Figura 4. Un vértice del path tracer: el hit se reconstruye desde el visibility buffer, la luz directa se recoge por next-event estimation, el lobe stack se muestrea para el siguiente bounce y las dos estimaciones se combinan por multiple importance sampling.](figures/fig-pathtracer.es.svg)

**MIS.** En cada vértice que continúa el camino muestreando su material, la muestra de NEE de la luz $k$ y los
impactos en luces de la muestra del material (`lightHit`, incluidos los rayos que escapan y encuentran domes y
soles) se ponderan con la heurística de potencia,

$$w_{\text{light}} = \frac{p_l^2}{p_l^2 + p_m^2},\qquad p_l = P(k\mid x)\; p_k(\omega),$$

donde $P(k\mid x)$ es la probabilidad de elección de la luz (por potencia, o mediante la BVH de luces) y $p_m$ es
`stackPdf`. Las luces delta, los lobes delta y el último vértice conservan peso 1. **Los triángulos emisores**
forman una tabla de luces (`technique::EmissiveTable`): la emisión de cada fila de material se sondea una vez, los
triángulos se ponderan por área en mundo por esa luminancia, y la elección entre luces y triángulos se hace por
potencia, con la densidad de cada lado llevando la parte del otro. Sobre un suelo de roughness 0.2 a 32 paths, el
MIS redujo el error frente a un frame profundo 4.5× bajo un rectángulo, 57.7× bajo un dome sin imagen y 2.4×
bajo una esfera. Un quad emisor y un rectángulo UsdLux equivalente coinciden a relMSE 4.1·10⁻⁷ con 8192 paths.

**Acumulación, muestreador, adaptatividad.** Un frame es una media móvil de color × opacidad. Una implementación
anterior tomaba el producto de las medias, que solo es insesgado mientras toda muestra es opaca. El muestreador pliega
píxel, flujo, índice absoluto de camino, bounce y dimensión mediante un hash PCG, así que 256 paths en una pasada y
4 × 64 difieren en relMSE 3.9·10⁻¹⁵. Entre pasadas la cámara se desplaza dentro del píxel con un desplazamiento de
Halton (2, 3); la primera pasada no toma desplazamiento. Es antialiasing sin coste. El muestreo adaptativo detiene un
píxel en cuanto

$$\frac{\sqrt{(\mathbb E[l^2]-\mathbb E[l]^2)/N}}{\mathbb E[l]} < \varepsilon$$

tras un número mínimo de paths. La varianza usada es la mayor entre la del propio píxel y la media de la de sus
vecinos 3 × 3, lo que redujo los píxeles detenidos a más de tres de sus propias σ de 526 a 100 de 4212. **La lente
delgada y la distorsión radial** vienen de `UsdGeomCamera`: el radio de la lente es $f/(2N)$, y los rayos se escalan
por $1+k_1r^2+k_2r^4$.

**Verificación.** Cada comprobación es un kernel de GPU frente a una forma cerrada o frente a un segundo estimador
independiente:

| Comprobación | Resultado |
|---|---|
| esfera emisora cerrada, $E(1+\rho+\dots+\rho^N)$, N = 0, 1, 2, 3, 6 | 0 de 3072 píxeles por encima de 10⁻⁴, peor 1.7·10⁻⁷ |
| horno blanco bajo un dome sin imagen frente a NEE sin ponderar | p99 relativo 0.0000, máx 0.0003 |
| un bounce frente a la luz directa del raster (nada en que rebotar) | p99 1, máx 1 (códigos de 8 bits), 4096 paths |
| error frente a N, pares de estimaciones independientes, 64–1024 paths | exponente −0.499 (ventana −0.42 a −0.58) |
| enfocado frente a estenopeica (radio de lente 0.2) | relMSE 0, bit a bit |
| desenfocado frente al perfil de segmento circular | 0 de 4235 píxeles por encima del 4 % |
| OIDN, 16 paths frente a referencia de 4096 paths | relMSE 3.65·10⁻⁴ → 7.17·10⁻⁵ |

El horno destapó un peso de MIS de un solo lado. Con un dome frente a un lobe lambertiano las dos densidades son
la misma función, así que el peso valía exactamente ½, y el horno leía 0.5453 en lugar de 1. La esfera destapó una
doble división, $\rho\pi/\cos$ en lugar de $\rho$, cuyo segundo momento infinito había mantenido el exponente de
convergencia en −0.43 a −0.45.

**Denoiser.** `technique::Denoiser` ejecuta el filtro `RT` de OIDN 2.5.1 en la propia cola del motor,
guiado por el albedo y la normal del primer impacto. En Metal, OIDN solo comparte buffers con seguimiento de riesgos,
que slang-rhi prohíbe. Por eso el motor copia a través de buffers intermedios con seguimiento mediante un kernel, ya
que el blit de slang-rhi no hacía nada, en silencio, con un buffer envuelto (4095 de 4096 palabras intactas). En CUDA
los buffers del motor se comparten directamente. En Vulkan, OIDN se ejecuta a través de CUDA sobre memoria importada
(`oidnNewSharedBufferFromFD`).

**Medidas** (Kitchen_set, 1920 × 1080, `athenea stage --frames 11`, mediana, release):

| Stage | raster | rt (1 camino, 1 bounce) |
|---|---|---|
| sin luces (luz de cámara) | 43.1 ms | 333.3 ms |
| iluminado (dome, rectángulo, dos esferas) | 96.0 ms | 358.9 ms |

A través de MCP, el OpenChessSet en path trace a 640 × 360 con 64 paths, un bounce y ACES 2.0 tarda 25.7 s.

## 2.12 Visualización y color

`technique::DisplayTransform` escribe directamente la textura de superficie de la ventana. Admite Standard; AgX, en el
ajuste analítico de Wrensch (log2 sobre [−12.47, 4.03] pasos, una sigmoide de sexto orden); y ACES 2.0. La
transformación de salida ACES 2.0 (`aces2.slang`) está portada función a función desde el CTL de la Academia,
incluidos el modelo de apariencia de Hellwig 2022, la escala tonal y la compresión de gama. Las tablas que la
referencia calcula al iniciar (362 entradas por matiz) las construyen en el device tres kernels. Comprobaciones:
256 grises sobre dieciséis pasos permanecen neutros a 5·10⁻⁶; el gris del 18 % lee 0.10000 del blanco de referencia
a 100 nits; inversa y después directa sobre 8192 colores vuelven a 6·10⁻⁶ del pico. Las fórmulas de visualización,
derivadas de nuevo en un segundo kernel, coinciden a 3.5·10⁻⁶. OpenColorIO se usa **como compilador**:
`colour::ColourCompiler` pide a OCIO el HLSL, lo carga como un módulo Slang `athenea_cs_<hash>` y llena sus LUTs en el
device. El host calcula el texto del shader y los valores de las LUTs, pero nada por píxel.

## 2.13 USD e Hydra

El plugin `hdAthenea` implementa `HdRenderDelegate` con rprims para meshes, basis curves, puntos, `ParticleField`
(Gaussian splats) y volúmenes; sprims para cámaras, materiales, los seis tipos de luz UsdLux, ext computations y
sistemas de coordenadas; y bprims para buffers de render y render settings. Las superficies implícitas (esfera, cubo,
cono, cilindro, cápsula, plano) llegan al delegado como meshes, convertidas por los scene indices de hdsi. **Hilos:**
`Sync`, que Hydra puede ejecutar en cualquier hilo, solo entrega registros de CPU a `Engine` bajo un cerrojo. El hilo
del render pass sube los datos, abre los ficheros `.athc` y compila los materiales, y después renderiza, así que el
device tiene un solo llamador. Una nube con skinning cuyos arrays no llevan muestras de tiempo se conserva entre
pasos (`_held`) y no se resube; esto llevó el frame en reproducción del gorrión de 102.4 a 71.7 ms antes del
trabajo de sort y mezcla de §2.5.3. **Salidas:** color, profundidad, primId, instanceId, elementId, Neye,
normal, `primvars:*`, Cryptomatte y grupos de luces. Se convierten en el device (`usd/aov_convert.slang`) solo
cuando un host mapea un buffer, y se guardan con la fila inferior primero, como las guarda Storm. `StageRenderer` es el
punto de entrada programático y dibuja hasta que un frame de path trace alcanza su total.

El motor lee esquemas sin código propios (`AtheneaSplatLightingAPI`, `AtheneaStreamedAssetAPI` y otros). Los ajustes
de render llegan como ajustes `athenea:*`: técnica, ruta de visibilidad, muestras de camino, bounces, totales, MIS,
denoise, antialiasing y lente.

## 2.14 Hosts y front ends

- **`athenea view`** mantiene los frames en el device: `StageRenderer::draw`, después `DisplayTransform`
  escribe la textura de superficie y Dear ImGui dibuja en el mismo device. Medido con materiales a 1600 × 900,
  Kitchen_set se dibuja en 9.27 ms con rays. Los paneles se describen una vez en `modules/ui`, así que la app de iOS
  dibuja los mismos.
- **Host AOFX.** El SDK y el host de AOFX son los de aopenfx (`third_party/aopenfx`), y el motor solo declara lo que
  aporta (`Host.cpp`). Cambian solo de forma aditiva bajo la regla de la ABI, así que los bundles del compositor
  cargan sin cambios. `aofx_sdk_manifest` falla ante cualquier cambio de cabecera. Los pasos de procesamiento de datos
  como mesh2splat (Sección 3) y `athenea compare` son bundles AOFX.
- **Servidor MCP** (`athenea-mcp`): JSON-RPC 2.0 sobre stdio, con el device y el stage calientes. Las
  herramientas son `open_stage`, `render` (que devuelve un bloque de imagen PNG), `pick`, `bounds`, `convert`,
  `timings` y `settings`. Una batería de pruebas de protocolo comprueba el saludo inicial, las notificaciones y los
  códigos de error. Para un programa que compone el frame, `render` acepta una cámara tal como la guarda un compositor
  (de cámara a mundo y una lente en milímetros) y escribe los números del propio frame en un fichero: RGBA
  premultiplicado, la profundidad de vista y, si se piden, la posición en el mundo (hecha en el device a partir de
  la profundidad) y las tres capas del Cryptomatte, con el manifest del matte en la respuesta. `sky`, `skyRotation` y
  `skyVisible` ponen otra imagen en todos los domos del stage, la giran y ocultan el fondo mientras sigue iluminando.
- **Nodo OpenFX** (`Athenea.ofx.bundle`, construido sobre el OpenFX de Natron). Un generador que dibuja una nube a
  través de su cámara al formato del proyecto y en el tiempo del frame. No carga nada del motor: arranca `athenea-mcp`
  como proceso hijo, uno para el proceso del host, y lee cada frame de un fichero, así que el USD y el TBB del host nunca
  se cruzan con los del motor y un fallo del device termina el servidor, no el host. Sus knobs son los de un nodo
  Camera (translate, rotate en un orden, focal, aperturas), el cielo, su rotación y su visibilidad, y la capa que muestra
  en RGBA un host sin planos. A través de un host OpenFX mínimo, su frame es idéntico bit a bit al render del propio
  servidor con la misma cámara (relMSE 0). En Natron, que admite planos propios, un peón a 1920 × 1080 escribe un EXR de
  32 bits con 21 canales (color, `depth.Z`, `P`, `CryptoObject00`–`02`) cuyos color y posición coinciden con los de ese
  host con un error menor de $10^{-6}$. La nube reiluminable del Corvette (3 M de Gaussians, 2.9 GB) a 1920 × 1080 con
  todos los planos tarda 7.5 s en el primer frame, que abre el stage y lo sube, y después 120–170 ms por frame con el
  cielo quieto y 190–220 ms con el cielo girando en cada frame. Nuke solo acepta de un nodo OpenFX sus propios planos,
  así que allí se ve una capa por nodo.
- **Add-on de Blender** (`athenea_hydra`). Un motor de render Hydra para Blender 5.3 sobre el delegado del motor: el
  viewport y el render final por athenea, la conversión de mesh a Gaussians de los objetos elegidos, las nubes
  referenciadas como Empties y el entorno del World como domo.

## 2.15 Verificación sin CPU oracle

Toda comparación de imágenes es un kernel. `ReferenceRenderer` es la verdad de referencia para splats: cada píxel
considera cada splat, ordena exactamente lo que le llega (profundidad, luego índice, la regla de desempate del
rasterizer) y mezcla de delante hacia atrás, sin tiles, rangos ni pares. Una comparación solo es válida cuando
ningún píxel desborda sus 1024 contribuyentes. `renderPeaks` es la referencia del ray tracer, que ordena
exactamente por pico. Las métricas que se leen de vuelta en el host son `compareImages` (p99, máximo y número de
píxeles por encima de 2, en códigos sRGB de 8 bits), `compareHdr` (relMSE $=\overline{(a-b)^2/(b^2+10^{-2})}$ y el
p99 relativo) y `countDifferent` para buffers de ids. El ray tracer se mantiene frente a su referencia a
p99 ≤ 1 con como mucho un 0.05 % de píxeles por encima de 2. Frente al rasterizer, la tolerancia solo se cumple donde
las dos definiciones coinciden: p99 2 en escenas ortográficas dispersas, donde EWA es exacto, y 4 en escenas en
perspectiva con partículas de pocos píxeles. Las escenas densas difieren en decenas de códigos, lo que refleja el
orden y no un error.

**Tres devices** (cada máquina calcula ella misma la misma métrica de GPU, y ningún píxel viaja
entre máquinas):

| | Metal (M5 Pro) | Vulkan (L4) | CUDA (L4, OptiX 9, sin rayos en línea) |
|---|---|---|---|
| Casos | 188 | 188 | 188 |
| Superados | 188 | 180 | 96 |
| Fallidos | 0 | 0 | 1 |
| Omitidos (capacidad nombrada) | 0 | 8 | 91 |

## 2.16 Limitaciones

- Los splats aún no están dentro de los rayos secundarios del path tracer. Un stage solo de splats lo traza
  entero `GaussianRayTracer`. Con meshes, los splats se componen por profundidad sobre las superficies del path
  trace; proyectan sombras sobre los meshes, pero un rayo de bounce no los ve.
- El blur de movimiento y de lente del rasterizer son Gaussians con momentos igualados, no mesetas ni discos
  de bokeh: hasta un 38 % demasiado picudas en estelas largas.
- Las nubes cuyas Gaussians miden unas centésimas de píxel desaparecen del raster (5·10⁻⁶ del frame frente a
  3·10⁻⁴ del mesh y del ray tracer); el LOD es el remedio (Sección 6).
- La visibilidad raster de meshes paga unos 8 ms de tiempo de host por cada 1800 draws.
- La compilación de materiales es síncrona.
- La TLAS se reconstruye en cada frame.
- No se aplica la distorsión de lente más allá de $k_1, k_2$.
