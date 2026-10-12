<!-- Copyright (c) 2026 jesus luque. -->

# 1. Introducción

## 1.1 El problema

Los assets 3D de producción se crean como meshes: polígonos, materiales físicos por capas, texturas, luces y, en los
personajes, esqueletos, todo compuesto en USD. Un path tracer los renderiza con fidelidad, pero es demasiado lento para
el uso interactivo; un rasterizer los renderiza de forma interactiva, pero necesita residentes toda la geometría y todas
las texturas del asset y aun así aproxima su transporte de luz. El Gaussian splatting 3D (3DGS) ofrece una tercera
vía: una nube de Gaussians anisótropas, ordenadas y mezcladas en el espacio de pantalla, dibuja detalle fotográfico a
ritmo interactivo en hardware modesto. Sin embargo, una nube de Gaussians normalmente se *entrena* a partir de
fotografías de una escena real. Contiene la luz de esa captura, no admite relight, y no tiene material, ni esqueleto,
ni relación con un asset creado por un artista.

Este artículo aborda el problema inverso: un **baker de relight** para Gaussian splatting. Dado un asset USD
fotorrealista (el caso que se sigue de principio a fin es un Chevrolet Corvette C7 de 192 meshes con pintura de coche
OpenPBR, cromo, vidrio, goma y faros), el baker produce una representación en Gaussians que

1. **admite relight**: cada Gaussian lleva el transporte de luz del asset en su bake, de modo que la nube responde al
   dibujar a cualquier cielo y cualquier luz, no solo a la iluminación con la que se hizo;
2. es **fiel a un path trace** del mesh de origen bajo esa iluminación, juzgada imagen contra imagen y región por
   región;
3. conserva el rig cuando el asset lo tiene, de modo que un asset animado siga moviéndose y su luz gire con él;
4. escala de las vistas lejanas a los primeros planos mediante LOD.

Ninguna de las vías evidentes cumple esto. Entrenar una nube a partir de vistas del asset con path tracing da una
iluminación fija y gasta horas de optimización por asset. Rasterizar el mesh con un modelo de sombreado aproximado
renuncia al transporte de luz del path trace. La vía que se sigue aquí convierte el mesh directamente en Gaussians y
hace el bake del transporte de luz del asset en cada Gaussian con el propio path tracer de GPU del renderer; la
nube con relight se compara después con el path trace del mesh. Reducir, comprimir y hacer streaming de la nube con su
bake hasta un player WebGPU en un móvil es una aplicación paralela de los mismos datos (§7).

## 1.2 Objetivos y restricciones

El requisito central es la fidelidad de las Gaussians con relight a la referencia del path trace. Los requisitos de
diseño de abajo se mantienen en todo momento y determinan casi todas las decisiones de diseño de las secciones 2–7.

| requisito | enunciado | dónde actúa |
|---|---|---|
| Fidelidad al path trace | La referencia es el path trace del mesh de origen bajo la misma iluminación. Donde la calidad y el frame rate chocan, prevalece la calidad. | §3.10, §4.10, §8 |
| Aritmética solo en GPU | Ninguna aritmética sobre datos de la escena se ejecuta en la CPU: ni renderer, ni alternativa, ni oráculo de pruebas en CPU. El host lee ficheros y lleva las cuentas (recuentos, slots, colas). | Todas las secciones; §2.1 enuncia la regla |
| 3DGS puro al dibujar | Lo que se dibuja son solo Gaussians. El suelo no es un mesh: su interacción con el objeto se convierte en Gaussians (un shadow catcher). Un mesh solo es una entrada del bake. | §3.9, §7.9 |
| Automático, no manual | Las decisiones por asset se detectan a partir de los datos: el tamaño de cell a partir de una cámara, el vidrio sólido frente al fino, las cáscaras cerradas, el suelo para el shadow catcher. Las opciones solo lo sobrescriben. | §3.3, §3.5, §3.8, §3.9 |
| Las piezas pequeñas, con el mismo criterio | Emblemas, grilles, molduras y faros se juzgan con el mismo cuidado que los paneles, tanto de cerca como de lejos. | §3.11, §6.9, §7.12 |

## 1.3 Contribuciones

Las contribuciones que describe este artículo son las siguientes.

1. **Un transfer con relight en el bake de cada Gaussian** (TX, sección 4), el núcleo del baker: spherical harmonics
   de grado 3 del transfer directo e indirecto, un campo reflejado normalizado sobre las direcciones cerradas y acoplado
   al cielo al dibujar, 256 bits de direcciones abiertas que cada lóbulo de reflexión lee como conos, y las marcas del
   material; sharp lobes por píxel guiados por un operador de forma almacenado, con el ensanche del bake deshecho y un
   tubo distinguido de un fold.
2. **Una conversión de mesh a Gaussians** (mesh2splat, sección 3) que prepara la entrada del baker, ejecutada como
   efecto AOFX en el dispositivo, con cells derivadas del footprint de un píxel de una cámara, una opacidad que
   reproduce la coverage de una superficie, bordes duros y cáscaras cerradas detectados en el dispositivo, ensanche de
   Toksvig de la roughness por la dispersión de las normales, captura de OpenPBR por capas y clasificación automática
   del vidrio.
3. **Un renderer USD solo en GPU** (athenea, sección 2) sobre el que corre el baker, que dibuja meshes, puntos y
   splats de Gaussians a través de Hydra, con un rasterizer por tiles y un ray tracer de Gaussians que comparten una
   única disposición de Gaussians, un compilador de materiales de MaterialX a Slang comprobado frente al propio
   MaterialX, un path tracer con muestreo por importancia múltiple usado como verdad de referencia y como integrador
   del bake, y una disciplina de verificación en la que toda comparación de imágenes es a su vez un kernel de GPU.
4. **Nubes de Gaussians con skinning** (sección 5) que llevan el rig y no los frames: un kernel de skinning que usa el
   jacobiano completo, incluidos los gradientes de los pesos, lóbulos zonales de transfer que giran con la Gaussian,
   campos de visibilidad por parte leídos como un producto de consultas a tablas, y una decimación que conserva el rig.
5. **Merge trees por pieza con un cut por error acotado** (sección 6): merge por pares bajo guardas que respetan piezas
   y materiales, una opacidad de coverage por mínimos cuadrados para los merges, un error hecho monótono con esferas
   envolventes para que una sola prueba paralela seleccione el cut, cull de caras traseras de cáscaras cerradas, un
   codebook de oclusión compartido y conservador, y pages ordenadas por el error del padre.
6. **Un relato medido** (sección 8) de lo que aporta cada paso del bake frente al path trace, sobre un asset de
   producción, a través de las leaves con relight, los cuts y los bakes de 640, 1280 y 5120 px.
7. **Una aplicación a la reproducción en móvil** (sección 7): un contenedor `.athc` v3 con codificaciones por planos de
   bytes, PCA por clusters del transfer mantenido comprimido en la GPU, bits de direcciones abiertas cuantizados
   vectorialmente, un registro compacto de 40 bytes, registros de recorrido cuantizados de 5–11 bytes, y una política
   de tiempo de frame y de still frame para un player WebGPU en un iPhone.

## 1.4 Organización

La sección 2 describe el motor y el renderer sobre los que corre el baker. La sección 3 describe la conversión de
meshes USD a Gaussians, y la sección 4 el transporte de luz en su bake y cómo se hace su relight. La sección 5 trata
los assets animados: el skinning, el gorrión y el peón. La sección 6 trata el LOD. La sección 7 describe la aplicación
a móvil: la compresión y un player WebGPU. La sección 8 consolida la evaluación de las secciones 2–7, la sección 9
reúne las limitaciones y el trabajo pendiente, y la sección 10 enumera las referencias, seguidas de un apéndice sobre
los repositorios y los comandos (figura 1).

![Figura 1. El pipeline. El baker de relight convierte un stage USD en Gaussians (§3) y hace el bake del transporte de luz en cada una con el path tracer (§4); la nube con relight se compara con un path trace del mesh de origen. Los merge trees (§6) y los assets animados (§5) parten de la nube con bake; el player WebGPU (§7) es una aplicación.](figures/fig-overview.es.svg)

## 1.5 Trabajo relacionado

**Gaussian splatting.** Kerbl et al. (2023) introdujeron el Gaussian splatting 3D: una escena como Gaussians
anisótropas con color en spherical harmonics, optimizadas a partir de fotografías y dibujadas por un rasterizer por
tiles ordenado en profundidad. Su proyección es el splatting EWA de Zwicker et al. (2001, 2002), una aproximación afín
local de la proyección perspectiva. Mip-Splatting (Yu et al., 2024) añade un filtro de suavizado 3D y un filtro de
pantalla 2D contra el aliasing; la dilatación de 0.3 px² de athenea, con su compensación de opacidad, pertenece a esa
familia (§2.5.1). El Gaussian splatting 2D (Huang et al., 2024) aplana cada Gaussian en un disco orientado que se
evalúa por intersección rayo–plano, en la línea de los surfels (Pfister et al., 2000); los surfels de athenea lo siguen
(§2.5.2). El ray tracing de Gaussians 3D (Moënne-Loccoz et al., 2024) traza partículas a través de proxies envolventes
en una BVH, y la ruta por hardware de athenea lo sigue (§2.6).

**De meshes a Gaussians.** El mesh2splat de Electronic Arts convierte un mesh en Gaussians rasterizando cada triángulo
en el plano de su eje dominante, una Gaussian por texel cubierto; la conversión de athenea parte de él (§3.1). Mesh2GS
(arXiv 2606.21898) deriva el tamaño de una Gaussian del footprint de un píxel de una cámara, y `--cell-from-camera` de
athenea lo sigue (§3.3).

**Transporte de luz precalculado.** El precomputed radiance transfer (Sloan, Kautz y Snyder, 2002) guarda, por punto,
la proyección de la visibilidad por el coseno sobre spherical harmonics, de modo que el relight es un producto escalar
con los coeficientes del entorno; Ramamoorthi y Hanrahan (2001) mostraron que nueve coeficientes representan bien la
irradiancia. El PCA por clusters (Sloan et al., 2003) comprime esos transfers agrupándolos y con una base principal por
cluster, y athenea lo usa tanto en el fichero como en la GPU del móvil (§7.2). Los armónicos zonales, que giran
girando un eje, se ajustan según Sloan (2008) (§4.4.5, §5.7). El muestreo por importancia filtrado (Křivánek y
Colbert, 2008) prefiltra el entorno (§4.3), y Lagarde y de Rousiers (2014) dan el ajuste de oclusión especular que
usaba el primer transfer (§4.6.1).

**Antialiasing especular.** Toksvig (2005) ensancha una distribución de microfacetas por la varianza de las normales
que promedia; athenea aplica la misma idea en la conversión, a la dispersión de normales bajo una Gaussian (§3.7,
§4.9).

**LOD para Gaussians.** Octree-GS (Ren et al., 2024) organiza anclas en un octree seleccionado por la distancia de la
vista. Las Gaussians 3D jerárquicas (Kerbl et al., 2024) construyen una jerarquía de merge a partir de los momentos de
las Gaussians y la cortan por tamaño en pantalla; el LOD original de athenea sigue sus momentos (§6.1). NanoGS (Xiong
et al., 2026) hace merge por pares de vecinos más cercanos mutuos, y los trees por pieza de athenea lo ejecutan en el
dispositivo (§6.2). El coste usa la cota de Runnalls (2007) para la reducción de mezclas de Gaussians, y las pages por
bandas de error siguen el uso que hace Nanite (Karis et al., 2021) de un error monótono para un cut paralelo (§6.3,
§6.8).

**Motores y formatos.** El motor está construido sobre OpenUSD e Hydra, materiales MaterialX y OpenPBR, shaders Slang
compilados mediante slang-rhi a Metal, Vulkan y CUDA, Intel Open Image Denoise y OpenColorIO. El player es un fork de
Spark con backend WebGPU.

## 1.6 Convenciones

Salvo que se diga otra cosa, las medidas se tomaron en un Apple M5 Pro con Metal. Las imágenes se comparan en la GPU
(§2.15). El **PSNR** está en dB sobre imágenes codificadas para pantalla; el **relMSE** es el error cuadrático medio
relativo sobre valores lineales. **El path trace** (PT) del Corvette es el path trace del mesh desde `/World/Camera` a
1920 × 1080 bajo el cielo `autoshop_01_4k`. El **recorte del coche** es el PSNR sobre el recorte de 900 × 420 del
coche, el **frame** sobre el frame entero; un **cociente por clase** es la media de una región sobre la del path trace.
El arnés validate-corvette del player mide otras regiones (coche, puerta, umbral, parachoques, capó), que no son el
recorte del coche (§7.12.3). Las palabras *Gaussian* y *splat* se usan indistintamente; una *leaf* es una Gaussian con
su bake, un *merge* un nodo de un merge tree, y un *cut* el conjunto de nodos que se dibujan para una vista.
*Transfer* es el transporte de luz del bake (§4), *TX* su forma actual.
