<!-- Copyright (c) 2026 jesus luque. -->

::: {#front}

# athenea: assets USD fotorrealistas como gaussianas reiluminables, de la malla al móvil {.unnumbered .unlisted .paper-title}

<p class="byline">jesus luque</p>

<p class="date">12 de octubre de 2026</p>

## Resumen {.unnumbered .unlisted}

Describimos athenea, un renderizador de USD e Hydra en el que cada paso que toca datos de la escena se ejecuta como un
kernel de cómputo en la GPU, y la cadena construida sobre él que convierte mallas USD de producción en splats de
gaussianas 3D que conservan su luz. Una malla se convierte en el dispositivo en gaussianas anisótropas, una por celda
de una retícula derivada de una cámara, con opacidad que respeta la cobertura, bordes sujetos donde una superficie
termina o se pliega, materiales OpenPBR por capas, vidrio clasificado como sólido o como pared fina, y rugosidad
ensanchada por la dispersión de normales bajo cada gaussiana. El propio path tracer de la escena hornea después un
transfer en cada gaussiana: armónicos esféricos de grado 3 de la luz directa e indirecta, un campo reflejado y 256
bits de direcciones abiertas. Al dibujar, estos reiluminan la nube bajo cualquier cielo y cualquier luz, con reflejos
nítidos girados por píxel mediante un operador de forma almacenado. Los assets con rig conservan su esqueleto, con un
jacobiano que incluye los gradientes de los pesos, lóbulos zonales de transfer que giran con un miembro y campos de
visibilidad por parte. Árboles de fusión por pieza con un error de pantalla monótono y acotado, culls de caras
traseras y de oclusión, y páginas ordenadas por error dan los niveles de detalle. Un PCA por clusters, la visibilidad
cuantizada vectorialmente y un registro compacto de 40 bytes llevan el Chevrolet Corvette C7 entero a un reproductor
WebGPU en un iPhone. En todo momento, la referencia es un path trace de la malla de origen. Doblar la resolución del
horneado de 640 a 1280 px subió el PSNR del coche de 16.55 a 17.88 dB en su recorte y de 22.18 a 23.90 dB en el
fotograma. El nivel de detalle cuesta unos 0.1 dB. En el móvil, los datos del coche se mantuvieron en 31–44 MB en
movimiento. Para las vistas de cerca se construyó un nivel de primer plano a 5120 px de 74.5 millones de hojas en 24
árboles.

**Palabras clave:** splatting de gaussianas 3D; USD; Hydra; transferencia de radiancia precalculada; armónicos
esféricos; de malla a gaussianas; nivel de detalle; PCA por clusters; WebGPU; render en móvil; path tracing; cadenas
solo en GPU.

## Índice {.unnumbered .unlisted}

::: {#contents}

| 1. Introducción
| 2. El motor y el renderizador athenea
| 3. De mallas USD a gaussianas (mesh2splat)
| 4. Transporte de luz horneado en gaussianas (transfer) y reiluminación
| 5. Assets animados: esqueletos, skinning y visibilidad por partes (el peón y el gorrión)
| 6. Niveles de detalle: árboles de fusión, cortes por error, páginas y oclusión
| 7. Compresión y el reproductor WebGPU en el iPhone
| 8. Evaluación
| 9. Limitaciones y trabajo futuro
| 10. Referencias
| Apéndice A. Repositorios, comandos y figuras

:::

:::
