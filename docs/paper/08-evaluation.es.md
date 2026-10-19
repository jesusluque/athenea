<!-- Copyright (c) 2026 jesus luque. -->

# 8. Evaluación

Esta sección reúne las medidas de las secciones 2–7 en un solo relato. Empieza por el bake, validado haciendo el
relight de las Gaussians frente al path trace del mesh de origen (§8.2, §8.3); siguen el motor (§8.4), el LOD (§8.5) y
los costes del bake (§8.6), y la aplicación a móvil (§8.7–§8.9) va al final. No introduce cifras nuevas: cada cifra se
cita de la sección que se indica a su lado. Donde dos fuentes dan cifras distintas para la misma magnitud, la §8.10
dice cuál es cuál.

## 8.1 Marco

Las medidas de escritorio se tomaron en un Apple M5 Pro con Metal; los resultados de Vulkan y CUDA proceden de una
NVIDIA L4 (§2.15). La referencia del Corvette es el path trace del mesh desde `/World/Camera` a 1920 × 1080 (§1.6). El
player se midió de tres maneras que no son intercambiables (§7): marcas de tiempo de GPU de Dawn sin ventana en el
M5 Pro, el sitio publicado en Chrome de escritorio en su modo móvil a 393 × 852 (el *iPhone emulado*, más o menos el
doble de rápido que el real), y un iPhone real (iOS 18.7, Safari 26.6.1, 393 × 852 @3).

## 8.2 La conversión frente al path trace

| asset o prueba | medida | resultado | § |
|---|---|---|---|
| cuadrilátero unidad iluminado de frente (píxel del mesh 0.5032) | píxel convertido con $\sigma$ = 1.0 / con los valores del material | 0.456 / 0.5009 | 3.4, 3.6 |
| Corvette, 640 px, bordes sujetos frente al control | relMSE / PSNR / SSIM, frame de día | 0.0887 / 21.93 dB / 0.8567 frente a 0.0902 / 21.86 dB / 0.8554 | 3.5 |
| shadow catcher del Corvette frente al path trace | bajo el coche / contacto del parachoques / suelo abierto | 0.192 / 0.030 / 0.220 frente a 0.182 / 0.043 / 0.217 | 3.9 |
| Corvette, bake de 640 → 1280 px | recorte del coche / frame | 16.55 → 17.88 dB / 22.18 → 23.90 dB | 3.11 |
| peón blanco, cuerpo / cabeza de vidrio (2048²) | relMSE | 0.0218 / 0.1418 (ejecuciones posteriores: vidrio 0.149, 0.081, 0.043) | 5.11 |
| gorrión (conversión web) frente al path trace del mesh | relMSE, raster | 0.0025 | 5.10 |
| ala del gorrión, raster / trazada | relMSE con la coverage siguiendo a la opacidad | 0.1635 / 0.1247 | 5.10 |

## 8.3 El coche entero a lo largo de la cadena

El mismo frame, con relight bajo el cielo del path trace y medido a medida que se añadía cada paso del bake y del
dibujo, en el orden en que se añadieron. Filas de la §4.10.3 y la §6.10; los nodos son Gaussians dibujadas.

| bake | configuración | nodos | recorte del coche (dB) | frame (dB) |
|---|---|---|---|---|
| 640 px | todas las leaves, antes de los pasos posteriores de la §4 | 5.38 M | 16.37 | 21.92 |
| 640 px | el mismo bake, cut por error de 1 px, ambos culls | 675 760 | 16.28 | 21.80 |
| 640 px | cut por error de 0.25 px, cull de caras traseras | 2 313 541 | 16.29 | 21.81 |
| 640 px | leaves, sharp lobes sin ensanchar | 5.38 M | 16.58 | 22.15 |
| 640 px | cut, + coat sujeto en los folds | — | 16.51 | 22.15 |
| 640 px | cut, + campo del polish en materiales rugosos | — | 16.55 | 22.18 |
| 640 px | leaves (0.01 px), sin cull / con cull | 4.35 M / 3.70 M | 16.73 / 16.75 | 22.40 / 22.43 |
| 640 px | grupo 0 con derrame $w_s$ = 2 | 561 242 (grupo 0) | 16.79 | 22.38 |
| 640 px | giro limitado / fold del coat solo en la pintura | — | 16.57 / 16.57 | 22.20 / 22.20 |
| 1280 px | cut 0.25 px, cull de caras traseras | 4 998 976 | **17.88** | **23.90** |
| 1280 px | sin el límite / polish nítido condicionado | — | 17.88 / 17.88 | 23.91 / 23.90 |
| 1280 px | condición + separación tubo/fold | — | 17.90 | 23.93 |
| 1280 px | separación tubo/fold (retenida) | — | 17.88 | 23.91 |

De aquí salen tres conclusiones. El LOD cuesta unos 0.1 dB frente al path trace (todas las leaves de 640 px frente al
cut de 1 px con ambos culls, 16.37 → 16.28 con un octavo de las leaves; §6.10). Los pasos de sombreado de la §4
movieron el coche de 640 px unos 0.2–0.4 dB. La resolución del bake lo movió 1.3 dB en el recorte y 1.7 dB en el frame
(el cut de 640 px con el campo del polish frente al cut de 1280 px). El PSNR sobre un recorte se mueve poco, así que
los cocientes por clase de abajo muestran dónde actúan los cambios.

| región o clase (media sobre la del path trace) | antes | después | condición | § |
|---|---|---|---|---|
| puerta, espejo al suelo a < 3 m | 0.78 | 1.02 | leaves de 640 px → cut de 640 px con campo del polish | 4.10.4 |
| puerta, espejo al suelo a 3–30 m | 0.69 | 0.88 | leaves de 640 px → cut de 640 px con campo del polish | 4.10.4 |
| puerta, espejo al HDRI (abierto) | 0.92 | 1.13 | leaves de 640 px → cut de 640 px con campo del polish | 4.10.4 |
| juntas de la puerta | 0.69 | 1.04 | leaves de 640 px → cut de 640 px con campo del polish | 4.10.4 |
| reflejos abiertos de la puerta, contraste local ($\sigma$ 1.5 / 4 px) | 0.35 / 0.53 | 0.56 / 0.74 | cut de 640 px con campo del polish → cut de 1280 px | 4.10.4 |
| brillos del capó, desplazamiento $p_{50}$ | 3.2 px | 2.0 px | cut de 640 px con campo del polish → cut de 1280 px | 4.10.5 |
| media del capó | 0.89 | 0.98 | 640 → 1280 px | 4.10.5 |
| huecos de la grille en el cut grueso | 2.24 | 1.04 | 640 → 1280 px | 3.11, 6.5 |
| pelos del umbral (píxeles) | 987 | 459 | 640 → 1280 px | 3.11 |
| umbral, bajo el coche (móvil) | 1.66 | 1.08 | sin → con la sombra de contacto | 7.9 |
| paso alto de la tira cromada / PT | 0.41 | 0.41 | cut de 1280 px → separación tubo/fold (polish nítido condicionado: 1.12) | 4.10.6 |
| motas brillantes en el fold del umbral | 10 | 1 | cut de 1280 px → separación tubo/fold | 4.10.6 |

| material (relMSE frente al path trace) | cromo | metal rugoso | pintura | § |
|---|---|---|---|---|
| primer transfer | 0.536 | 0.0896 | 0.705 | 4.10.1 |
| TX, cells y campo en todas partes | 1.64 | 0.221 | 0.393 | 4.10.1 |
| TX con la condición del polish | 0.591 | 0.0835 | 0.341 | 4.10.1 |
| + coat nítido y polish nítido por píxel | 0.545 | 0.083 | 0.281 | 4.10.1 |

## 8.4 La corrección y la velocidad del motor

Cada comprobación de abajo es un kernel de GPU frente a una forma cerrada, un estimador independiente, MaterialX o
Storm.

| comprobación | resultado | § |
|---|---|---|
| dos discos semiopacos, raster / referencia / ray tracer frente al operador over analítico | p99 0 (mezcla codificada: p99 26) | 2.4 |
| planos convertidos de coverage 0.25 / 0.5 / 0.75, raster a tamaño completo | 0.251 / 0.501 / 0.752 | 2.5.3 |
| paso de obturador 0.03 frente a N instantes | p99 2, cociente de energía 1.0000 | 2.5.3 |
| ray tracer frente a su referencia de orden exacto | p99 ≤ 1, como mucho el 0.05 % de los píxeles por encima de 2 | 2.15 |
| visibilidad de meshes frente a Storm (Kitchen_set), raster / rayos / bvh | segmentación: 18 / 41 / 36 píxeles distintos de ~57 k | 2.8 |
| grafos MaterialX frente al genglsl del propio MaterialX, 65 536 direcciones | peor 3.2·10⁻⁵ en once grafos | 2.9 |
| luces frente a formas cerradas; χ² frente a `lightPdf` | 0 de 8281 píxeles por encima del 2 %; $\lvert z\rvert \le 1.61$ | 2.10 |
| path tracer: cáscara emisora cerrada; exponente de convergencia | 0 de 3072 por encima de 10⁻⁴; −0.499 | 2.11 |
| OIDN, 16 caminos frente a una referencia de 4096 caminos | relMSE 3.65·10⁻⁴ → 7.17·10⁻⁵ | 2.11 |
| batería de pruebas en tres dispositivos (188 casos) | Metal 188 pasan; Vulkan 180 pasan, 8 omitidas; CUDA 96 pasan, 1 falla, 91 omitidas | 2.15 |
| una nube TX girada 60° con su cámara y su cielo frente a sin girar | peón relMSE 9·10⁻⁶ (61 dB); bolas de prueba 1–2·10⁻⁴, 1.5·10⁻²–0.13 bajo un cielo sin girar | 4.3 |
| nodo OpenFX frente al render del propio servidor, misma cámara | relMSE 0; en Natron, color y posición en el mundo a menos de 10⁻⁶ de un host OpenFX simple | 2.14 |

| carga | cifra | § |
|---|---|---|
| radix sort, 10 M pares de 32 bits | 12.4 ms (39.9 ms antes) | 2.3 |
| clips del gorrión, 5 887 323 Gaussians a 1600 × 1000, reproduciendo | 102 → 49 ms por frame | 2.5.3 |
| train_7k (742 k), 1080p: raster por tiles / ComputeBvh / ray tracer por hardware | 13 / 343 / 435 ms | 2.6 |
| visibilidad de Kitchen_set a 1920 × 1080: raster / rayos / bvh | 31.7 / 19.2 / 29.2 ms | 2.8 |
| Kitchen_set iluminado a 1920 × 1080: raster / path tracer (1 camino, 1 rebote) | 96.0 / 358.9 ms | 2.11 |
| Corvette TX entero (14.7 M Gaussians) a 1920 × 1080, rasterizer de escritorio | 158 ms (unos 83 ms para una nube con relight de su tamaño) | 4.10.2 |
| Corvette reiluminable (3 M) por el nodo OpenFX en Natron, 1920 × 1080, todos los planos | primer frame 7.5 s (abre el stage); luego 120–170 ms, 190–220 ms con el cielo girando | 2.14 |
| un canal de parámetros, peón a 768 × 768 | 20–60 ms | 4.11 |

## 8.5 LOD

| nivel | leaves | merges | raíces | pages | codebook de oclusión | cells reabiertas | § |
|---|---|---|---|---|---|---|---|
| coche a 640 px | 5.38 M | — | 17 133 | 3 582 | 8192 | 11.9 % | 6.2.5, 6.7 |
| coche a 1280 px | 8.69 M | 8.67 M | 20 723 | 5 789 | 8187 | 12.3 % | 6.7, 6.10 |
| close-up level a 5120 px | 74 536 126 | 74 328 031 | 208 095 | 49 634 | 8 182 | 0–23.6 % por tree | 6.7, 6.9 |

El close-up level son 24 trees y 148 864 157 nodos, 87.4 GB en disco, con su oclusión compartida escrita en el sitio
(645 s, 10.2 GB de pico residente; §6.9).

| cut, a 393 px de ancho, frente a las leaves a 4× | nodos de cerca | dB de cerca | nodos de lejos | dB de lejos | § |
|---|---|---|---|---|---|
| error 1 px, cotas | 666 585 | 33.29 | 343 541 | 39.15 | 6.10 |
| + cull de caras traseras de cáscaras cerradas | 522 931 | 33.63 | — | — | 6.10 |
| + oclusión | 360 333 | 33.58 | 195 049 | 39.53 | 6.10 |
| ambos culls, regla $O<2\sqrt{E}$ | 378 028 | 33.53 | — | — | 6.10 |

Cada cull baja el número de nodos, y ningún cut con cull da menos frente a las leaves que el cut por error solo
(§6.10). En el player, con todos los culls activos se dibujan un 51 % menos de nodos que con todos apagados, a 36.9 dB
entre las dos imágenes, y el codebook de oclusión alcanza el 84 % del ideal (§7.5.1). La residencia es el punto débil:
un recorrido necesita unas 2.2× el cut, y las pages tienen 3.6–4.5× el cut sea cual sea su tamaño (§6.8), y por eso el
player separa el recorrido de la carga útil (§7.5.2).

## 8.6 Costes

| paso | cifra | § |
|---|---|---|
| coche a 1280 px, 8 grupos, transfer de cada uno | 212–304 s, unos 34 min en total | 3.11 |
| shadow catcher, 309 k Gaussians | 79 s | 3.9 |
| campos de visibilidad por parte del gorrión (12 partes) | 22 s | 5.8 |
| decimación del gorrión conservando el rig (5.89 M) | 28 s | 5.9 |
| codebook de oclusión compartido sobre el close-up level | 645 s | 6.9 |
| construcción del LOD original, train_7k | 47 ms | 6.1 |

## 8.7 Compresión

| paso | antes | después | coste en la imagen | § |
|---|---|---|---|---|
| planos de bytes (sin pérdida), Corvette ligero / HD | 68.1 / 288.7 MB | 60.9 / 248.6 MB | ninguno | 7.1 |
| codificación 3 del transfer (CPCA en el fichero), set ligero del Corvette / page del peón | 67.9 / 281 MB | 49.5 / 84 MB | relMSE por splat $4\times10^{-8}$–$3\times10^{-5}$ | 7.2.1 |
| CPCA conjunto en la GPU, base del coche entero (256 clusters) | 112 valores | 1 id + 24 pesos int4 | relMSE del transfer 1.03·10⁻²; imagen a relMSE 10⁻³ | 7.2.2 |
| base de 512 clusters (prueba v8), fine3_0 | 0.0181 | 0.0136 | el mismo tiempo de shade en el M5 Pro | 7.2.2 |
| bits de direcciones abiertas por VQ, 1024 centroides por pieza | 256 bits | índice u16 | 62.9–65.4 dB | 7.3 |
| registro compacto (residente) | 308 B por Gaussian + 29 MiB | 42.4 B + 16 B por splat dibujado + ~3 MiB | 40.2–41.9 dB frente al relight completo | 7.4 |
| registro de recorrido | 24 B (Spark) | 5–11 B por clase | el mismo cut | 7.5.3 |

## 8.8 El móvil

**Frente al path trace, por versión de datos** (regiones de validate-corvette, PSNR en dB a 1920 × 1080; §7.12.3):

| versión | coche | puerta | umbral | parachoques | capó |
|---|---:|---:|---:|---:|---:|
| v6 (trees de 640 px) | 19.33 | 33.77 | 31.87 | 20.21 | 14.16 |
| v7 + dispersión del píxel, $\tau$ 1 (1280 px) | — | 35.53 | 32.36 | relMSE 0.199 | 14.46 |
| v7, valores por defecto actuales | 19.03 | 35.52 | 32.36 | 22.28 | 14.68 |
| v8 (close-up level a 5120 px) | — | — | — | — | — |

v8 no se había convertido para el player en el momento de escribir esto y no se ha medido. En el recorte del coche, el
sharp lobe por píxel del camino compacto llevó el coche de 16.49 a 17.46 dB (§7.12.3).

**Velocidad** (§7.12.1, §7.12.2):

| dispositivo | datos | dibujados | resultado |
|---|---|---|---|
| iPhone real, página de estrés | 3DGS simple | 200 k / ~1 M | 58.8 / 26.3–27 fps |
| iPhone real, página de estrés | compacto, base real | 200 k / 400 k / ~1 M | 27 / 47 / 108 ms |
| iPhone real, modelo | compacto | — | ~115 ms por millón + 4 ms: ~250 k a 30 fps, ~110 k a 60 fps |
| iPhone real, primera prueba en el dispositivo | v7, vista por defecto | 106 k | 53 fps ($\tau$ 8.00) |
| iPhone real, primera prueba en el dispositivo | v7, de cerca | 160 k | 32 fps, logo y grille blandos |
| iPhone emulado | v5 / v7 / v7 tras un ajuste posterior del tiempo de frame | 226 k / 216 k / 234 k | 60 fps con $\tau$ 3 / 2.22 / 1.95 |
| iPhone emulado, still frame | v7, de cerca | 529 k | un frame con $\tau$ 1.23 |
| Dawn, M5 Pro, pasada compacta | 393 × 852 $\tau$ 3 / 1920 $\tau$ 1 | — | 4.0 / 17.2 ms |

Según el modelo, el cut de LOD de lejos con ambos culls (195 049 nodos, §8.5) cabe en las ~250 k que el móvil dibuja a
30 fps, y el de cerca es más o menos 1.4× eso (§6.10). Una segunda prueba en el dispositivo encontró la vista de cerca
todavía blanda. Lo blando se atribuyó a la nube de 1280 px, no al player, y eso llevó al close-up level a 5120 px
(§7.12.1, §6.9). Para el gorrión, el cut móvil (488 656 splats, 31.7 MB) iba a 60 fps sin ventana en modo móvil,
animado (§5.12).

## 8.9 Memoria

| magnitud | cifra | § |
|---|---|---|
| datos del coche, v6 tras el atlas, $\tau$ 3: Camera / de cerca / de lejos | 39.0 / 43.8 / 33.0 MB | 7.11 |
| render state, mismas vistas | 24.1 / 26.0 / 21.7 MB | 7.11 |
| página (heap de JS, three.js, domes) | ~39 → 25 → ~17 MB | 7.11 |
| datos del coche, v7, $\tau$ 3 | 31–36 MB | 7.11 |
| datos del coche, still frame de cerca (presupuesto ~100 MB) | 95.6 MB | 7.10, 7.11 |
| por Gaussian, relight anterior / camino compacto | 308 B + 29 MiB / 42.4 B + 16 B por splat dibujado + ~3 MiB | 7.4, 7.11 |
| con 1 M splats, anterior / compacto | 326 / 58.6 MiB | 7.4 |
| escritura de un tree, puerta de 4.08 M leaves (tras el arreglo) | 10.5 GB de pico residente | 6.9 |
| oclusión compartida sobre 24 trees | 10.2 GB de pico residente | 6.9 |

El objetivo de 50–60 MB del player se aplica solo a los datos del coche; el render state se cuenta aparte (§7.11). En
movimiento, los datos del coche eran 49.1–59.0 MB en v6 antes del atlas y 31–44 MB en v6 después de él y en v7, dentro
del objetivo (§7.11).

## 8.10 Donde las fuentes difieren

- **Leaves a 1280 px.** Los ocho bakes escribieron 8 703 429 Gaussians (§3.11, «8.70 M»); los trees tienen 8.69 M
  leaves (§6.10). La primera cifra es el recuento de la conversión, la segunda la de los trees.
- **Front_Bumper_001.** El bake del transfer da 5 009 059 Gaussians, el tree 5 008 986 leaves (§3.11); los totales
  usan la cifra del tree.
- **El tamaño en disco del close-up level.** 87.4 GB para los 24 trees; «unos 86 GB» era la estimación antes de
  escribir el último tree (§6.9).
- **El total de oclusión del close-up level.** Un primer recuento marcó un 132.4 % reabierto; era un desbordamiento de
  32 bits del total de cells cerradas, ya corregido. Las cifras por tree (0–23.6 %) y el codebook eran correctos
  (§6.7).
- **El still frame.** Una medida da 529 k dibujados con $\tau$ 1.23 y 95.6 MB de datos del coche; otra da 531 k con
  $\tau$ 1.22 en 118 ms (§7.10). Son dos ejecuciones de la misma política.
- **Juntas de la puerta a 1280 px.** La §6.10 da las juntas subiendo de 0.90 a 1.02 entre el cut de 640 px con el campo
  del polish y el cut de 1280 px, mientras que la tabla por clases de la §4.10.4 da las juntas del primero en 1.04
  (0.90 es el de las leaves de 640 px sin cull). Las dos cifras vienen de ejecuciones de medida distintas; ambas dejan
  las juntas a 1280 px a un 2–4 % del path trace.
- **Recorte del coche en el móvil.** Los 17.46 dB del camino compacto (§7.12.3) y los 17.88 dB del cut de 1280 px en el
  rasterizer de escritorio (§4.10.3) son medidas separadas sobre caminos y datos distintos, y aquí no se comparan.
- **Velocidad real frente a emulada.** Las cifras emuladas van más o menos al doble que las del iPhone real; los
  dibujados se juzgan en el dispositivo (§7).
