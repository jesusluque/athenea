<!-- Copyright (c) 2026 jesus luque. -->

::: {#front}

# athenea: un baker de relight para Gaussian splatting a partir de assets USD de producción {.unnumbered .unlisted .paper-title}

<p class="byline">jesus luque</p>

<p class="date">19 de octubre de 2026</p>

## Resumen {.unnumbered .unlisted}

Se describe athenea, un baker de relight para Gaussian splatting 3D: convierte meshes USD de producción y sus
materiales en Gaussians y hace el bake, en cada Gaussian, del transporte de luz necesario para su relight al dibujar,
siempre con un path trace del mesh de origen como referencia. Un mesh se convierte en el device en Gaussians
anisótropas, una por cell de una lattice derivada de una cámara, con opacidad que respeta la coverage, bordes sujetos
donde una superficie termina o forma un fold, materiales OpenPBR por capas, vidrio clasificado como sólido o como pared
fina, y roughness ensanchada por la dispersión de normales bajo cada Gaussian. El propio path tracer de GPU del
renderer hace después el bake de un transfer en cada Gaussian: spherical harmonics (SH) de grado 3 de la luz
directa e indirecta, un reflected field, 256 bits de open directions y las marcas del material. Al dibujar, estos
datos hacen el relight de la nube bajo cualquier cielo y cualquier luz, con reflejos nítidos girados por píxel mediante
un shape operator almacenado. En un Chevrolet Corvette C7 de 192 meshes, doblar la resolución del bake de 640 a
1280 px subió el PSNR del coche con relight frente al path trace de 16.55 a 17.88 dB en su recorte y de 22.18 a
23.90 dB en el frame, y para las vistas de cerca se hizo el bake de un close-up level a 5120 px de 74.5 millones de
leaves en 24 trees. El baker se apoya en un renderer de USD e Hydra en el que cada paso que toca datos de la escena
se ejecuta como un kernel de cómputo en la GPU. Los assets con rig conservan su esqueleto, con un Jacobian que incluye
los gradientes de los pesos, lobes zonales de transfer que giran con un miembro y campos de visibilidad por parte.
Merge trees por pieza con un error de pantalla monótono y acotado, culls de caras traseras y de oclusión, y pages
ordenadas por error dan el LOD con un coste de unos 0.1 dB. Un transfer se lee en el sistema de referencia de su propia
nube, con el cielo girado a él una vez por instancia, así que una nube que la escena gira hace el relight igual que sin
girar (relMSE $9\times10^{-6}$). Cada parámetro que lleva una Gaussian, desde su albedo y su normal hasta la visibilidad
del cielo, el bounce y el reflected field del transfer, se puede dibujar a través de una cámara como un canal de imagen
propio. La salida del bake es un master: un fichero USD con cada parámetro a precisión completa, juzgado frente al path
trace, que leen tal cual el motor, un nodo OpenFX en un compositor (con capas de profundidad, posición en el mundo y
Cryptomatte y un cielo que elige el artista) y un add-on de Blender. Un proceso aparte, con su propio objetivo, deriva
del master un formato web para un móvil: un PCA por clusters del transfer, la visibilidad cuantizada vectorialmente y un
registro compacto de 40 bytes llevan el coche con relight a un player WebGPU en un iPhone, con su pérdida medida frente
al master.

**Palabras clave:** relight baking; Gaussian splatting 3D; precomputed radiance transfer; spherical harmonics; USD;
Hydra; de mesh a Gaussians; path tracing; LOD; PCA por clusters; cadenas solo en GPU; WebGPU; OpenFX; composición;
canales de imagen.

## Índice {.unnumbered .unlisted}

::: {#contents}

| 1. Introducción
| 2. El motor y el renderer athenea
| 3. De meshes USD a Gaussians (mesh2splat)
| 4. Transporte de luz en el bake de las Gaussians (transfer) y relight
| 5. Assets animados: skeletons, skinning y visibilidad por partes (el peón y el gorrión)
| 6. LOD: merge trees, cuts por error, pages y oclusión
| 7. Aplicación: compresión y un player WebGPU en móvil
| 8. Evaluación
| 9. Limitaciones y trabajo futuro
| 10. Referencias
| Apéndice A. Repositorios y comandos

:::

:::
