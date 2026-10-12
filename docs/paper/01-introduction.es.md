<!-- Copyright (c) 2026 jesus luque. -->

# 1. Introducción

## 1.1 El problema

Los assets 3D de producción se crean como mallas: polígonos, materiales físicos por capas, texturas, luces y, en los
personajes, esqueletos, todo compuesto en USD. Un path tracer los renderiza con fidelidad, pero es demasiado lento para
un móvil; un rasterizador los renderiza de forma interactiva, pero necesita residentes toda la geometría y todas las
texturas del asset y aun así aproxima su transporte de luz. El splatting de gaussianas 3D (3DGS) ofrece una tercera
vía: una nube de gaussianas anisótropas, ordenadas y mezcladas en el espacio de pantalla, dibuja detalle fotográfico a
ritmo interactivo en hardware modesto. Sin embargo, una nube de gaussianas normalmente se *entrena* a partir de
fotografías de una escena real. Contiene la luz de esa captura, no se puede reiluminar, y no tiene material, ni
esqueleto, ni relación con un asset creado por un artista.

Este artículo aborda el problema inverso. Dado un asset USD fotorrealista (el caso que se sigue de principio a fin es
un Chevrolet Corvette C7 de 192 mallas con pintura de coche OpenPBR, cromo, vidrio, goma y faros), producir una
representación en gaussianas que

1. sea **fiel a un path trace** de la malla de origen, juzgada imagen contra imagen y región por región;
2. **se reilumine**: que responda a cualquier cielo y cualquier luz, no solo a la iluminación con la que se hizo;
3. conserve el rig cuando el asset lo tiene, de modo que un asset animado siga moviéndose;
4. **se reproduzca en un móvil** dentro de una página web, con un presupuesto de memoria fijo, tanto de cerca como de
   lejos.

Ninguna de las vías evidentes cumple las cuatro. Entrenar una nube a partir de vistas del asset con path tracing da una
iluminación fija y gasta horas de optimización por asset. Rasterizar la malla en el móvil está limitado por la memoria
y por el sombreado por píxel que puede permitirse una GPU móvil. La vía que se sigue aquí convierte la malla
directamente en gaussianas, hornea el transporte de luz del asset en cada gaussiana con el propio path tracer del
renderizador, y después reduce, comprime y transmite el resultado.

## 1.2 Objetivos y restricciones

El trabajo se hizo bajo un pequeño conjunto de restricciones, fijadas por el usuario y mantenidas en todo momento.
Determinan casi todas las decisiones de diseño de las secciones 2–7.

| restricción | enunciado | dónde actúa |
|---|---|---|
| Aritmética solo en GPU | Ninguna aritmética sobre datos de la escena se ejecuta en la CPU: ni renderizador, ni alternativa, ni oráculo de pruebas en CPU. El host lee ficheros y lleva las cuentas (recuentos, huecos, colas). | Todas las secciones; §2.1 enuncia la regla |
| 3DGS puro en la reproducción | Lo que dibuja el reproductor son solo gaussianas. El suelo no es una malla: su interacción con el objeto se convierte en gaussianas (un shadow catcher). Una malla solo es una entrada del horneado. | §3.9, §7.9 |
| Fidelidad ante todo | La referencia es el path trace de la malla. Donde la calidad y el frame rate chocan, gana la calidad («aunque empeore el rendimiento quiero calidad», 12 de octubre de 2026). | §4.10, §7.10 |
| Presupuesto de memoria | 50–60 MB de datos del coche residentes mientras la cámara se mueve (unos 55 MB en el reproductor actual), unos 100 MB en reposo; el estado de render se cuenta aparte. | §6.8, §7.10, §7.11 |
| Automático, no manual | Las decisiones por asset se detectan a partir de los datos: el tamaño de celda a partir de una cámara, el vidrio sólido frente al fino, las cáscaras cerradas, el suelo para el shadow catcher. Las opciones solo lo sobrescriben. | §3.3, §3.5, §3.8, §3.9 |
| Las piezas pequeñas deben verse bien | Emblemas, rejillas, molduras y faros se juzgan con el mismo cuidado que los paneles, tanto de cerca como de lejos. | §3.11, §6.9, §7.12 |

## 1.3 Contribuciones

Las contribuciones que describe este artículo son las siguientes.

1. **Un renderizador USD solo en GPU** (athenea, sección 2) que dibuja mallas, puntos y splats de gaussianas a través
   de Hydra, con un rasterizador por teselas y un trazador de rayos de gaussianas que comparten una única disposición
   de gaussianas, un compilador de materiales de MaterialX a Slang comprobado frente al propio MaterialX, un path
   tracer con muestreo por importancia múltiple usado como verdad de referencia, y una disciplina de verificación en
   la que toda comparación de imágenes es a su vez un kernel de GPU.
2. **Una conversión de malla a gaussianas** (mesh2splat, sección 3), ejecutada como efecto AOFX en el dispositivo,
   con celdas derivadas de la huella de un píxel de una cámara, una opacidad que reproduce la cobertura de una
   superficie, bordes duros y cáscaras cerradas detectados en el dispositivo, ensanche de Toksvig de la rugosidad por la
   dispersión de las normales, captura de OpenPBR por capas y clasificación automática del vidrio.
3. **Un transfer reiluminable horneado en cada gaussiana** (TX, sección 4): armónicos esféricos de grado 3 del
   transfer directo e indirecto, un campo reflejado normalizado sobre las direcciones cerradas y acoplado al cielo al
   dibujar, y 256 bits de direcciones abiertas que cada lóbulo de reflexión lee como conos; lóbulos nítidos por píxel
   guiados por un operador de forma almacenado, con el ensanche del horneado deshecho y un tubo distinguido de un
   pliegue.
4. **Nubes de gaussianas con skinning** (sección 5) que llevan el rig y no los fotogramas: un kernel de skinning que usa
   el jacobiano completo, incluidos los gradientes de los pesos, lóbulos zonales de transfer que giran con la
   gaussiana, campos de visibilidad por parte leídos como un producto de consultas a tablas, y una decimación que
   conserva el rig.
5. **Árboles de fusión por pieza con un corte por error acotado** (sección 6): fusión por pares bajo guardas que
   respetan piezas y materiales, una opacidad de cobertura por mínimos cuadrados para las fusiones, un error hecho
   monótono con esferas envolventes para que una sola prueba paralela seleccione el corte, cull de caras traseras de
   cáscaras cerradas, un codebook de oclusión compartido y conservador, y páginas ordenadas por el error del padre.
6. **Un camino de compresión y streaming hasta un móvil** (sección 7): un contenedor `.athc` v3 con codificaciones por
   planos de bytes, PCA por clusters del transfer mantenido comprimido en la GPU, bits de direcciones abiertas
   cuantizados vectorialmente, un registro compacto de 40 bytes, registros de recorrido cuantizados de 5–11 bytes, y una
   política de tiempo de frame y de fotograma quieto que mantiene los datos del coche dentro de su presupuesto.
7. **Un relato medido** (sección 8) de lo que aporta cada paso frente al path trace, sobre un asset de producción,
   desde el horneado hasta un iPhone real.

## 1.4 Organización

La sección 2 describe el motor y el renderizador. La sección 3 describe la conversión de mallas USD a gaussianas, y la
sección 4 el transporte de luz horneado en ellas y cómo se reiluminan. La sección 5 trata los assets animados: el
skinning, el gorrión y el peón. La sección 6 trata los niveles de detalle, y la sección 7 la compresión y el
reproductor WebGPU en el iPhone. La sección 8 consolida la evaluación de las secciones 2–7, la sección 9 reúne las
limitaciones y el trabajo pendiente, y la sección 10 enumera las referencias, seguidas de un apéndice sobre los
repositorios, los comandos y las figuras.

## 1.5 Trabajo relacionado

**Splatting de gaussianas.** Kerbl et al. (2023) introdujeron el splatting de gaussianas 3D: una escena como
gaussianas anisótropas con color en armónicos esféricos, optimizadas a partir de fotografías y dibujadas por un
rasterizador por teselas ordenado en profundidad. Su proyección es el splatting EWA de Zwicker et al. (2001, 2002),
una aproximación afín local de la proyección perspectiva. Mip-Splatting (Yu et al., 2024) añade un filtro de suavizado
3D y un filtro de pantalla 2D contra el aliasing; la dilatación de 0.3 px² de athenea, con su compensación de
opacidad, pertenece a esa familia (§2.5.1). El splatting de gaussianas 2D (Huang et al., 2024) aplana cada gaussiana en
un disco orientado que se evalúa por intersección rayo–plano, en la línea de los surfels (Pfister et al., 2000); los
surfels de athenea lo siguen (§2.5.2). El trazado de rayos de gaussianas 3D (Moënne-Loccoz et al., 2024) traza
partículas a través de proxies envolventes en una BVH, y la ruta por hardware de athenea lo sigue (§2.6).

**De mallas a gaussianas.** El mesh2splat de Electronic Arts convierte una malla en gaussianas rasterizando cada
triángulo en el plano de su eje dominante, una gaussiana por texel cubierto; la conversión de athenea parte de él
(§3.1). Mesh2GS (arXiv 2606.21898) deriva el tamaño de una gaussiana de la huella de un píxel de una cámara, y
`--cell-from-camera` de athenea lo sigue (§3.3).

**Transporte de luz precalculado.** La transferencia de radiancia precalculada (Sloan, Kautz y Snyder, 2002) guarda,
por punto, la proyección de la visibilidad por el coseno sobre armónicos esféricos, de modo que reiluminar es un
producto escalar con los coeficientes del entorno; Ramamoorthi y Hanrahan (2001) mostraron que nueve coeficientes
representan bien la irradiancia. El PCA por clusters (Sloan et al., 2003) comprime esos transfers agrupándolos y con
una base principal por cluster, y athenea lo usa tanto en el fichero como en la GPU del móvil (§7.2). Los armónicos
zonales, que giran girando un eje, se ajustan según Sloan (2008) (§4.4.5, §5.7). El muestreo por importancia filtrado
(Křivánek y Colbert, 2008) prefiltra el entorno (§4.3), y Lagarde y de Rousiers (2014) dan el ajuste de oclusión
especular que usaba el primer transfer (§4.6.1).

**Antialiasing especular.** Toksvig (2005) ensancha una distribución de microfacetas por la varianza de las normales
que promedia; athenea aplica la misma idea en la conversión, a la dispersión de normales bajo una gaussiana (§3.7,
§4.9).

**Niveles de detalle para gaussianas.** Octree-GS (Ren et al., 2024) organiza anclas en un octree seleccionado por la
distancia de la vista. Las gaussianas 3D jerárquicas (Kerbl et al., 2024) construyen una jerarquía de fusión a partir
de los momentos de las gaussianas y la cortan por tamaño en pantalla; el nivel de detalle original de athenea sigue sus
momentos (§6.1). NanoGS (Xiong et al., 2026) fusiona por pares vecinos más cercanos mutuos, y los árboles por pieza de
athenea lo ejecutan en el dispositivo (§6.2). El coste usa la cota de Runnalls (2007) para la reducción de mezclas de
gaussianas, y las páginas por bandas de error siguen el uso que hace Nanite (Karis et al., 2021) de un error monótono
para un corte paralelo (§6.3, §6.8).

**Motores y formatos.** El motor está construido sobre OpenUSD e Hydra, materiales MaterialX y OpenPBR, shaders Slang
compilados mediante slang-rhi a Metal, Vulkan y CUDA, Intel Open Image Denoise y OpenColorIO. El reproductor es un fork
de Spark con backend WebGPU.

## 1.6 Convenciones

Salvo que se diga otra cosa, las medidas se tomaron en un Apple M5 Pro con Metal. Las imágenes se comparan en la GPU
(§2.15). El **PSNR** está en dB sobre imágenes codificadas para pantalla; el **relMSE** es el error cuadrático medio
relativo sobre valores lineales. **El path trace** (PT) del Corvette es la malla trazada por caminos desde
`/World/Camera` a 1920 × 1080 bajo el cielo `autoshop_01_4k`. El **recorte del coche** es el PSNR sobre el recorte de
900 × 420 del coche, el **fotograma** sobre el fotograma entero; un **cociente por clase** es la media de una región
sobre la del path trace. El arnés validate-corvette del reproductor mide otras regiones (coche, puerta, umbral,
parachoques, capó), que no son el recorte del coche (§7.12.3). Los identificadores de prueba (T15–T75) son los del
registro del trabajo de pieces, y los hashes de commit son los del repositorio que se nombra (apéndice A). Las palabras
*gaussiana* y *splat* se usan indistintamente; una *hoja* es una gaussiana horneada, una *fusión* (merge) un nodo de un
árbol de fusión, y un *corte* el conjunto de nodos que se dibujan para una vista. *Transfer* (transferencia) es el
transporte de luz horneado (§4), *TX* su forma actual. Las figuras que las secciones nombran como rutas de fichero se
enumeran en el apéndice A.3.
