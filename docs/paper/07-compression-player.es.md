<!-- Copyright (c) 2026 jesus luque. -->

# 7. Aplicación: compresión y un player WebGPU en móvil

Esta sección describe una aplicación de las nubes con relight de §3–§6, desarrollada como una línea paralela: la reproducción de un asset entero en un móvil. Sus requisitos de diseño son propios: el Corvette entero, con su transfer de luz completo, dibujado en un iPhone dentro de una página web y juzgado frente al path trace del mesh; un budget de unos 50–60 MB de datos del coche en movimiento y unos 100 MB en reposo; piezas pequeñas que se vean bien. El player es un fork de Spark con backend WebGPU (sparkwebGPU, rama `webgpu`); athenea escribe las nubes, los merge trees, las pages y la oclusión. Esta sección describe los formatos, el streaming, el dibujado y el relight en el móvil, y los resultados medidos. El formato `.athc` v3 está especificado en el repositorio del player (`spark/docs/docs/athc-v3.md`).

**Aparecen tres tipos de medida**, y no son intercambiables:

- **Dawn / M5 Pro**: marcas de tiempo de GPU de Dawn sin interfaz (Metal) en un Apple M5 Pro, a la resolución CSS del móvil (393 × 852) o a 1920 × 1080. Son un sustituto de la GPU del móvil.
- **iPhone emulado**: el sitio publicado en Chrome de escritorio en el mismo M5 Pro, con la página en su modo móvil a 393 × 852. Los tamaños de pool, el cut y la memoria son los del móvil; la velocidad no. Las tasas de frames emuladas son más o menos el doble que las del iPhone real, así que los dibujados se juzgan en el device.
- **iPhone real**: iOS 18.7, Safari 26.6.1, pantalla de 393 × 852 @3 (la clase iPhone 14 Pro–16), canvas de 393 × 695 con pixel ratio 1. Son las rampas de la página de estrés y dos pasadas del coche entero en el device.

La calidad de imagen se da como PSNR en dB, o como relMSE ($\sum(x-y)^2/\sum y^2$; cuando el PSNR se escribe como $-10\log_{10}\mathrm{relMSE}$ se indica). La referencia es el path trace en la `/World/Camera` de athenea (1920 × 1080) o el propio camino sin comprimir del player. En cada caso se dice cuál.

## 7.1 El contenedor: `.athc` versión 3

El `.athc` v2 de athenea es un array plano de palabras por splat. El player web lo reescribe como **versión 3**, organizada por secciones y tiers. Dentro de cada bloque, los datos que necesita un tier son un prefijo, así que cada par (bloque, tier) es una sola petición HTTP Range:

| Tier | Secciones | Contenido |
|---|---|---|
| 1 | `CORE`, `SHRS` y, opcionalmente, `LODS`, `LODE`, `OCCL`, `LODM`, `TREE`, `SKIN` | geometría y color del splat; metadatos de LoD |
| 2 | `MATL` | normales, emisión, pbr, lobes |
| 3 | `SHAD`, `CURV`, `TXDI`, `TXIN`, `TXFD` | bits de open directions, curvatura, transfer (directo, indirecto, reflected field) |
| 4 | `CMPC` | el registro compacto para el móvil (§7.4) |

Un navegador ofrece gzip y nada mejor (`DecompressionStream`), así que la palanca sin pérdida es lo que se le da a gzip. La **codificación 1** transpone cada array en planos de bytes: el byte 0 de todos los elementos, luego el byte 1, y así sucesivamente. El byte alto de un f16 o un f32 cambia poco de un splat al siguiente en orden de Morton, así que forma tramos que gzip encuentra. La **codificación 2** hace lo mismo tras sustituir cada carril de 16 bits por su diferencia (con desbordamiento) respecto al elemento anterior. `athc-convert --gzip --planes` elige la codificación más pequeña por sección. Medido en las nubes publicadas: Corvette ligero 68.1 → 60.9 MB (−10.6 %), Corvette HD 288.7 → 248.6 MB (−13.9 %), su pintura −23 %, piezas del peón −17 a −33 %. Los bits de sombra son aleatorios y nunca ganan.

## 7.2 Compresión del transfer por PCA agrupada

El transfer es el 30–70 % de los bytes de una nube con relight, y los transfers vecinos se parecen. Se usan dos codificaciones por PCA agrupada: una codificación con pérdida del fichero (codificación 3) y una base conjunta que el móvil mantiene residente.

### 7.2.1 Codificación 3 (en el fichero)

Siguiendo a Sloan et al. (2003), una sección de transfer de un bloque se guarda como hasta $K$ clusters (k-means sobre una muestra; $K \le n/128$). Cada cluster $c$ tiene una media $\mu_c$ y $m_c \le M$ direcciones principales $\mathbf{d}_{c,j}$ en f16. Cada elemento $i$ tiene un id de cluster y $m_c$ coeficientes cuantizados:

$$
\hat{\mathbf{t}}_i = \mu_{c(i)} + \sum_{j<m_{c(i)}} \left(q_{ij}\,\Delta_{c(i)}\right)\mathbf{d}_{c(i),j},
\qquad q_{ij}\in\mathbb{Z},
$$

donde cada producto y cada suma se redondean a f32 en ese orden y el resultado se redondea al half más cercano. Eso hace que los decodificadores de Rust/WASM y de TypeScript sean idénticos al bit. El paso $\Delta_c$ es proporcional al RMS del cluster, lo que da el mismo error relativo en clusters oscuros que en claros. Los coeficientes son enteros zig-zag de 1–3 bytes, guardados como planos de bytes. $m_c$ es el menor número cuyo truncado consume como mucho el 40 % del budget de error. El budget es relativo por bloque:

$$
E_{\text{block}} = \frac{\sum_i \lVert \hat{\mathbf t}_i - \mathbf t_i\rVert^2}{\sum_i \lVert \mathbf t_i\rVert^2} \le E, \qquad E = 10^{-5}\ \text{por defecto},
$$

y el paso se reduce hasta que se cumple. Como los coeficientes SH del transfer son ortonormales, $E$ es el error $L^2$ sobre la esfera. Un bloque que no cumple $E$, o que los planos de bytes guardan más pequeño, conserva sus valores exactos. Medido: las secciones de transfer ocupan entre 4 y 7 veces menos (el TX 112 de la tapa del peón pasó de 157 a 28 MB); un splat relight se mueve un relMSE de $4\times10^{-8}$ a $3\times10^{-5}$; el set ligero del Corvette pasó de 67.9 a 49.5 MB y la página del peón de 281 a 84 MB.

### 7.2.2 La base conjunta residente en la GPU

En el móvil el transfer nunca se expande. La CPCA conjunta de athenea divide los 112 valores TX en tres bloques $b \in \{\text{directo } (16), \text{indirecto } (48), \text{reflejado } (48)\}$. Usa $C$ clusters compartidos por los tres bloques y $K = 8$ direcciones por bloque. Una Gaussian guarda un id de cluster $c$ y $3 \times 8$ pesos:

$$
\mathbf t^{(b)}_i \;\approx\; \mu^{(b)}_{c} + \sum_{k=1}^{8} q^{(b)}_{ik}\,\frac{s^{(b)}_{c,k}}{127}\,\mathbf B^{(b)}_{c,k},
$$

donde $\mathbf B$ son los ejes de cada cluster y $s$ sus escalas. En la GPU, cada dirección se guarda premultiplicada por $s/127$ en f16, 504 palabras por cluster. Los **pesos** son int8 o, en el móvil, int4: se codifican a 7 niveles y se decodifican como $q_8 = \mathrm{round}(q_4 \cdot 127/7)$. En el móvil, int4 cuesta $+0.7\times10^{-4}$ de relMSE del color por splat y $0.1$–$4.6\times10^{-5}$ de relMSE de imagen; domina el propio truncado de la CPCA.

**La ruta proyectada.** Bajo una sola cúpula, solo el bloque reflejado depende del ojo. En cada relight, un hilo por cluster proyecta la media y los 8 ejes de los bloques directo e indirecto sobre la luz (los 16 harmonics de la cúpula, la dirección del sol, el acoplamiento de cielo blanco): 144 palabras por cluster. Los términos independientes de la vista ("conservados") de una Gaussian son entonces productos escalares de sus pesos con esas proyecciones. En cada frame solo se reconstruyen, en registros, los 48 valores reflejados. Con datos reales esta ruta reproduce la reconstrucción completa con un relMSE de $3\times10^{-11}$.

**Evolución de la base.**

| base | clusters | ajustada sobre | relMSE del transfer | notas |
|---|---|---|---|---|
| base de athenea | 256 | la muestra de `base_tx` de athenea | 1.69e-2 (coche entero) | fichero f32 de 1.17 MB; la asignación de clusters del player coincide con la de athenea al 100 % |
| ajuste del player, seis piezas | 256 | seis piezas del coche | directo 2.7e-4, indirecto 2.5e-3, reflejado 1.1e-3 | dentro de la muestra; piezas excluidas entre 10 y 400 veces peor |
| ajuste del player, coche entero | 256 | uno de cada dos nodos del coche | 1.03e-2 (directo 1.6e-3, indirecto 8.0e-3, reflejado 1.06e-2) | imagen a relMSE 1e-3 del transfer sin comprimir |
| ajuste del player, 512 clusters (datos v8) | 512 | car1280 + car5120 | fine3_0: 0.0181 → 0.0136 | todos los grupos de paneles de 1280, −20 a −30 % |

El ajuste (`fit-compact-basis`) normaliza la energía por bloque, aplica k-means (semillas k-means++, Lloyd sobre todos los nodos) a los vectores conjuntos, luego una PCA por cluster de cada bloque, y elige la escala de cada dirección con el menor error a través del código int4. La **base de 512 clusters** mantiene el tamaño del registro: el 9.º bit del id de cluster va en el bit 15 libre del índice del diccionario de sombras, lo que limita el diccionario a 32 768 entradas (el del coche tiene 4096). Los ficheros de 256 clusters se leen igual que antes. Coste: base 516 KB → 1.03 MB (f16), proyecciones 147 → 295 KB, tier 4 +1.6–1.8 %, y el mismo tiempo de shade en el M5 Pro (2.82–2.88 ms con 400k dibujados). No se ha medido en el móvil real.

## 7.3 Los bits de open directions por cuantización vectorial

Las 16 × 16 cells de open directions de cada Gaussian (256 bits) se sustituyen por un índice a un diccionario (`athc-convert --shadow-vq`). El diccionario es k-means en espacio de Hamming, cada centroide la mayoría bit a bit de sus miembros con los empates abiertos, 12 rondas, con semilla en el miembro $\lfloor c\,n/C\rfloor$. Con 1024 centroides por pieza, el error de imagen es de 62.9–65.4 dB (cifra de athenea: 62 dB), y la distancia de Hamming media es de 2.0–2.5 sobre 256 (p99 14–16). Las máscaras en bruto como diccionario ocuparían 15.3 MB a 1M splats, frente a 1.51 MB con la VQ. Sobre seis piezas, un diccionario de asset de 4096 entradas (131 KB, distancia media 1.07) gana en bytes a seis diccionarios de 1024 (197 KB, distancia media 1.00) con casi el mismo error, así que el diccionario compartido es el de por defecto.

## 7.4 El registro compacto

El camino compacto guarda una Gaussian como planos de bytes en un único storage buffer. Cada plano está alineado a 64 bytes y se lee con `fieldAt(plane, width, i)`:

| plano | bytes | codificación |
|---|---:|---|
| posición | 6 | $3\times$u16 en la caja de la page: $x = \ell + q\,(h-\ell)/65535$ |
| albedo | 3 | u8 sRGB |
| opacidad | 1 | $q\le191$: $\alpha=(q/191)^2$; $q\ge192$: $\alpha = 1 + (q-192)/63$ (opacidad de LoD de Spark, $>1$) |
| escala | 3 | $s = \exp(\ell + (q-1)\,\delta)$, $\delta = (h-\ell)/254$; $q=0$ significa exactamente 0 (surfels) |
| rotación | 4 | palabra de cuaternión: eje octaédrico de 10+10 bits, ángulo de 12 |
| normal | 3 | octaédrica, 12 bits por eje |
| curvatura | 3 | $s8$: $\kappa = \pm f\,(2^{|q|R/127}-1)$, log con signo en el rango de la page |
| material | 2 | u16 a una tabla exacta (pbr + lobes, 4 palabras) |
| cluster | 1 | u8 (el 9.º bit en el índice de sombra) |
| pesos | 12 | $3\times8$ int4 |
| sombra | 2 | u16 al diccionario |
| **total en GPU** | **40** | |

Los planos son los que impusieron los datos. Una normal de 8+8 bits movía los reflejos del clear coat un relMSE de $5.2\times10^{-2}$; con 12 bits el error es de $2.7\times10^{-4}$. Un código lineal de opacidad destruía los alfas de 0.01 de las láminas de cristal. El coche tiene 17–22k materiales distintos, así que un índice u8 no los cabía. En una ablación, posición, opacidad, albedo, escala y curvatura aportan cada una menos de $10^{-6}$ de error de imagen. Frente al relight completo sobre la misma nube, el color por splat del camino compacto queda a un relMSE de $6.3$–$7.3\times10^{-4}$. La imagen queda a 40.2–41.9 dB con 0.4–1.0M splats.

Frente a los 308 B por Gaussian del relight anterior, este camino mantiene residentes **42.4 B** (planos más tablas) y **16 B por splat dibujado** (los buffers del sort reutilizados como registros de dibujado). El coste fijo baja de 29 MiB a unos 3 MiB. Con 1M splats son 58.6 MiB, frente a 326 MiB antes (Dawn, tamaños de buffer).

## 7.5 Pages por bandas de error y el tier compacto

### 7.5.1 El cut por error

Cada nodo de merge $n$ lleva las cotas $\varepsilon_n$ (error) y $\rho_n$ (radio de la esfera) en f16 (`LODE`). Con $p$ los píxeles por unidad a distancia unidad, la traversal refina $n$ cuando

$$
P_n = \frac{\varepsilon_n\, p}{\max(d_n - \rho_n,\ \text{near})} > \tau .
$$

$\tau$ está en píxeles. athenea pagina su merge tree por **bandas de error**, como Nanite. Un nodo hace falta exactamente cuando el error proyectado de su padre supera $\tau$, así que cada nodo va a la banda (una octava) del error de su padre. Las bandas se ordenan de gruesa a fina, cada una en orden de leaves, y el orden se trocea en pages de $P = 3000$ nodos. Una fila de la tabla de pages guarda una caja alrededor de las esferas de los padres y el error de padre máximo y mínimo. Una page hace falta cuando

$$
\frac{\varepsilon^{\max}_{\text{page}}\, p}{\max(\operatorname{dist}(\text{eye}, \text{box}),\ \text{near})} > \tau \quad(\text{la page 0 siempre}).
$$

Como la caja acota las esferas de los padres, este test es conservador respecto a la traversal. En un conjunto de piezas del coche, el cut paginado ya asentado fue idéntico al cut con todo residente; lo residente fue 2.27–2.53 veces el cut.

En la traversal se aplican dos culls:

- **Caras traseras.** Un nodo de una pieza cerrada se poda con su subárbol cuando $\mathbf n\cdot\mathbf v < -\sin(\theta + \arcsin(\rho/d))$, con $\theta$ el semiángulo de su cono de normales.
- **Oclusión.** Con el ojo fuera de la caja del asset, un nodo se poda cuando las cuatro cells bilineales hacia el ojo ($\mathrm{oct}(\mathbf v)\cdot16-0.5$) están todas cerradas en su entrada del codebook de oclusión.

El codebook de athenea para el coche tiene 8192 entradas (el 84 % del ideal, sin falsos positivos), y el de v8 tiene 8182. Con centros cuantizados, la oclusión se prueba sobre la caja de centros posibles (partida por octantes, donde las dos coordenadas octaédricas son monótonas). Sobre los 10.7M nodos del coche, en las cámaras de athenea y 24 ojos aleatorios, esto ocultó 0 nodos que el centro exacto conserva, a costa de conservar como mucho el 0.06 % de los ocultos. Con todos los culls activos, los dibujados bajan un −51 % frente a todos apagados, con 36.9 dB entre las dos imágenes.

### 7.5.2 El tier compacto y sus versiones

La sección `CMPC` del tier 4 guarda los 40 bytes de GPU más los bytes de la traversal, así que un móvil no lee nada de los tiers 1–3. Las tablas que comparte una pieza (`CMPT`: diccionario, materiales, base) y un asset (`.cmpa`: base, un diccionario, una tabla de materiales, y el codebook de oclusión con la caja del asset; identificado por FNV-1a 64) se leen una sola vez.

| `CMPC` | bytes por nodo (fichero) | traversal | cambio |
|---|---|---|---|
| v1 | 64 = 40 + 24 | las 4 palabras del tree LoD + las 2 de `LODE` | primeras pages compactas; paginado = residente bit a bit |
| v2 | 56 = 40 + 16 | 16 B en el fichero, 14 B en WASM | `CMPT` v2 y la tabla de asset `.cmpa`; mismo cut nodo a nodo; descarga de cerca 23.7 → 16.8 MB, tablas 3.62 → 0.97 MB |
| v3 | 64 = 40 + 24 | + palabra de cono, índice de oclusión de 13 bits; 30 B de entrada, 20 B en WASM | back-cull y oclusión activos por defecto; solo se planifican las pages que pide la traversal |
| v4 | contenedor `CMP4` | registros cuantizados de 12 bytes | carga en **tramos raw-deflate de 64 nodos**, plano tras plano; pool de nodos; tier 4 +10 % (314 → 344 MB) |

La **page v4** es un contenedor: una cabecera, el final de cada tramo, los planos de la traversal en gzip, y luego cada tramo de 64 nodos comprimido con raw deflate. Los tramos se pueden pedir, guardar y desalojar uno a uno, así que la CPU guarda la carga de una page comprimida y la GPU solo guarda registros del cut **dibujado**. Lo motivó el censo del pool: con pages enteras residentes, solo se dibujaba un ~26 % de los nodos de cada page, y las pages más pequeñas no ayudaban (residente/cut 4.54× con $P=3000$, 4.46× con $P=500$) (figura 17).

![Figura 17. El formato compacto: una tabla del asset, ficheros `.athc` v3 cuyo tier 4 contiene pages v4 con traversal records y registros de payload de 40 bytes, y la base de clustered PCA del transfer residente en la GPU.](figures/fig-compact.es.svg)

### 7.5.3 Registros de la traversal

El coste por nodo de la traversal en WASM bajó por pasos, todos con el mismo cut:

| etapa | bytes por nodo | cómo |
|---|---:|---|
| `LodSplat` de Spark | 24 | tree completo |
| `CMPC` v2 / v3 en WASM | 14 / 20 | centro f16 como paso en orden f16 desde la posición compacta (exacto), $\varepsilon,\rho$ f16, hijos |
| `QuantNode` de v4 | 12 | centro $3\times10$ bits en la caja de centros de la page; $\varepsilon$ 8 y $\rho$ 7 bits de código log sobre el rango de la page; $\theta$ 6; eje 5+5; oclusión 13; hijos como posición + delta de page |
| registros por clase | 5, 7, 9 u 11 | 2 bits eligen leaf/merge × con/sin cono; centros de 9 bits; **índice de 9 bits a la paleta de oclusión de la page** (511: ninguna); un offset de byte cada 32 nodos |

El redondeo solo refina más y cullea menos: $\varepsilon,\rho,\theta$ se redondean hacia arriba, el error del eje de un merge se suma a su $\theta$, y el mayor error de centro de la page se suma a cada esfera. En el coche entero, 1.15M nodos residentes ocupan 10.4 MB como registros por clase, frente a 13.8 MB como palabras de 12 bytes. El budget de pages supone 8 B por nodo. Otro paso sin pérdida hizo que los merges de cada page apunten a la page de sus hijos a través de una **paleta por page** de 9 bits. Eso dejó 436 nodos en forma completa en vez de 181k en el coche de 1280, y quitó un mapa WASM de 2.5 MB.

### 7.5.4 Raíces sintéticas de varios niveles

Una ranura de page 0 cabe 3072 nodos, pero los bosques car5120 de athenea conservan a propósito miles de raíces sueltas (los merges nunca cruzan piezas ni las guardas de athenea): fine5 tiene 29 213 y fine6 21 065. Pasado un fan-out (2048), el packer corta las raíces, en orden de Morton de sus centros, en tramos, cada uno hijo de un nodo sintético, y repite hasta que la page 0 cabe el resto. Cada nodo sintético tiene error infinito (siempre se refina), una esfera alrededor de sus hijos, sin cono ni índice de oclusión, así que el cut es el del bosque plano (probado con 5 ojos × 5 valores de $\tau$). fine5 pasó de 10 ficheros a 3 y fine6 de 8 a 1.

## 7.6 Streaming y el cut de cada frame

El lector (`CompactBandStreamer`) solo lee las tablas de pages al abrir. Tras cada traversal:

1. El **conjunto necesario** sale solo de la tabla de pages de cada tree, con el ojo en el espacio de objeto del tree (el giro del coche desde el escenario Z-up de athenea) y con foveación.
2. Las **pages que faltan** se piden por orden del error que quita una page por cada byte que cuesta; en caso de empate, la más cercana. Las pages consecutivas se juntan en una sola petición Range (hasta 8 MiB), dentro del hueco libre del pool.
3. El pool **desaloja** las pages no necesarias desde la banda más fina hacia arriba. La page 0 y cualquier page de la que dependa una residente nunca se desalojan.
4. **$\tau$** sube hasta que el conjunto necesario cabe (`tauToFit`). La versión actual busca por bisección el $\tau$ más bajo que cabe, lo que quitó la oscilación de cerca (4.61–5.76 → 5.16).

Un nodo cuyos hijos no están se dibuja como leaf, o como su padre o sus hijos mientras se decodifica un tramo. Con los tramos de v4, los tramos del propio cut y los de sus padres e hijos se guardan siempre, y el resto se desaloja primero bajo un budget de tramos. También se recorre una cámara extrapolada (2.5 actualizaciones por delante), para precargar sus tramos. Hay hasta 12 peticiones Range en vuelo.

El trabajo se reparte entre el hilo principal, la traversal en WASM y dos workers (figura 18):

| dónde | qué | medido |
|---|---|---|
| worker de carga (WASM) | traversal de todos los trees, con el cut enviado por trozos; pila en vez de heap cuando el budget no limita | Camera, $\tau=3$: 128 → 40 ms (mismo cut) |
| worker de nodos (`NodeCutter`) | cut a ranuras, inflado de tramos (~0.3 µs por nodo), empaquetado de registros, lista de dibujado con sustitutos, precarga | empaquetado JS 32.8 → ~9 ms; máx / p99 en la órbita de 31–35 / 26–31 a 17.5–19.0 / 17.4–18.7 ms |
| hilo principal | peticiones Range, escritura de registros y listas de dibujado (adoptadas, sin GC) | — |
| compute de GPU | movimiento de registros cuando el pool encoge (un dispatch, un hilo por palabra) | primer frame tras un still frame 88–100 → 34 ms |

![Figura 18. Los hilos del player: peticiones Range y el pool de pages en el hilo principal, la traversal en un worker WASM, el cut empaquetado en slots en un segundo worker y cinco pasadas de GPU por frame; el tiempo de GPU fija el suelo de τ.](figures/fig-player.es.svg)

En una órbita rápida de 120° a 1° por frame en el móvil emulado, la parte del cut que falta en un frame bajó del 26 % con el primer streamer de v4 al 4.3–6.4 %, que es el 0.24–0.30 % del área de pantalla. Los frames se mantuvieron a 60 fps con un máximo de 17.3–19.6 ms. El cut se asentó 2.6–3.1 s después de la carga.

## 7.7 El dibujado

El camino compacto (`CompactRelitPath`, `compact.slang`) ejecuta cinco pases por frame:

1. **Proyección** (un hilo por cluster, solo cuando cambia la luz): las proyecciones por cluster de §7.2.2.
2. **Cull** (un hilo por nodo de la lista): test de frustum; una clave de distancia radial de 24 bits y el índice añadidos con un atómico.
3. **Sort**: radix sort en GPU de las claves dibujadas, de atrás adelante (6 pasadas sobre 24 bits).
4. **Shade** (un hilo por ranura dibujada, agrupado por cluster): decodificar, relight (§7.8), proyectar. Se escribe un registro de 16 bytes en los propios buffers del sort: centro como u16 a 1/16 px, $\sigma_1$ (half) con el ángulo del eje mayor, $\sigma_2$ (half) con $\alpha$ (half; el signo negativo marca opacidad de LoD), y color en RGB9E5.
5. **Dibujado**: quads instanciados, premultiplicados sobre un target rgba16float.

**Proyección (EWA).** Con $W$ la rotación de la vista, $R,S$ la rotación y la escala del splat, y $J$ el Jacobian de la perspectiva en el centro en espacio de vista $(x,y,z)$:

$$
J = \begin{pmatrix} f_x/z & 0 & -f_x x/z^2\\ 0 & f_y/z & -f_y y/z^2\end{pmatrix},\qquad
\Sigma_{2D} = J\,W R S\,(J W R S)^{\!\top} = \begin{pmatrix} a & b\\ b & d\end{pmatrix}.
$$

**Compensación de opacidad.** El blur de anti-aliasing suma $\beta = 0.3\ \text{px}^2$ a la diagonal, y la opacidad se escala para conservar la masa del splat:

$$
\Sigma' = \Sigma_{2D} + \beta I,\qquad \alpha' = \alpha_0 \sqrt{\det\Sigma_{2D}/\det\Sigma'} .
$$

Un nodo de merge con opacidad de LoD $\alpha_0 > 1$ se dibuja con $\alpha_0 \leftarrow \min(4\alpha_0 - 3, 5)$, y su quad se ensancha $0.7(\alpha_0-1)$ desviaciones. Un splat tenue ($\alpha' < 4\,\alpha_{\min}$) conserva su masa $\alpha_0\sqrt{\det\Sigma_{2D}}$: se elige su blur para que su opacidad llegue al objetivo tenue, o se dibuja a ese objetivo con probabilidad $\alpha'/(4\alpha_{\min})$ mediante un dither de hash estable. El suelo de 0.1 de una lámina fina de cristal se aplica **después** del blur y se atenúa por su área en pantalla. Aplicado antes del blur, había dejado la tapa del faro 1.63 veces demasiado clara. El fragmento evalúa $\alpha = \alpha'\,e^{-z^2/2}$ dentro del radio de corte, descarta por debajo de $\alpha_{\min}$ y suma los sharp lobes por píxel (§7.8).

Un hilo por ranura dibujada reutiliza los buffers del sort como registros, así que no hay nada más por splat. El camino compacto no tiene registros relit (84 B), ni registros de píxel (48 B), ni acumulador (32 B), ni splats ext (32 B). En el móvil tampoco tiene textura de profundidad: nadie la lee. Con la vista quieta solo se repite el dibujado. Escribir directamente en 8 bits no ahorró tiempo de GPU (Apple mezcla en memoria de tile) y cambiaba la imagen hasta unos 100 códigos en el 1–3 % de los canales, así que el target sigue siendo rgba16float.

## 7.8 Relight en el móvil

El shade evalúa `relitByDome` de athenea (rama TX) para una cúpula. El código es una copia literal con las lecturas del campo tomadas de registros, porque WebGPU no puede enlazar un mismo buffer como lectura y escritura en un dispatch. Los términos conservados salen de las proyecciones; el reflected field (16 coeficientes × RGB) se lee a lo largo del espejo. Se portaron o añadieron los siguientes refinamientos, y cada uno se midió.

**El reflected field.** Es el 42 % del shade, y el coste es aritmético, no de memoria. Dos cambios lo reducen: el shade agrupado por cluster (un counting sort de las ranuras por cluster, con histogramas por workgroup), −6 a −9 %; y el campo limitado a grado 1 (4 coeficientes) cuando un footprint mide menos de 3 px, −23 % del shade con 215k dibujados. Frente al path trace, el cambio en puerta, umbral, parachoques y capó es menor de 0.01 dB.

**Sharp lobes por píxel.** Un clear coat o un polish por debajo de los umbrales de roughness de los sharp lobes no se responde una sola vez en el centro. El shade resta la lectura del cielo en el centro y le pasa al dibujado un registro de 5 palabras: la dirección del espejo en el centro (unorm16 × 2), sus pasos un píxel a la derecha y un píxel abajo (pares f16), el peso del clear coat y las dos roughnesses como raíces cuadradas, y el peso del polish en RGB9E5. El fragmento lee entonces el cielo a lo largo de $\mathrm{oct}^{-1}(\mathbf a_0 + \mathbf s_x\,\delta x + \mathbf s_y\,\delta y)$. La roughness con que se lee se construye en tres pasos:

- *Desensanchado.* El bake ensanchó la roughness guardada por Toksvig, $r'^4 = r^4 + 2s^2$ con $s$ la dispersión de normales. Como el giro por píxel ya cuenta esa dispersión, el lobe lee $r = \max(r'^4 - 2s^2, 0)^{1/4}$.
- *Regla del fold.* Con $t = 2s$ el giro sobre $\pm2\sigma$ y $w = \mathrm{smoothstep}(0.7, 1.5, t)$, el giro del píxel se escala por $1-w$ y el lobe se ensancha, $\alpha' = \sqrt{\alpha^2 + w\,(t/2)^2}$, por debajo de 0.5.
- *Dispersión del píxel.* La normal gira $\mathbf d_x,\mathbf d_y$ a lo ancho de un píxel. Esa caja de pendientes, el doble para el espejo, ensancha el $\alpha = r^2$ de GGX:
$$
\alpha^2 \leftarrow \alpha^2 + \frac{2\left(\lVert\mathbf d_x\rVert^2+\lVert\mathbf d_y\rVert^2\right)}{12} = \alpha^2 + \frac{\lVert\mathbf d_x\rVert^2+\lVert\mathbf d_y\rVert^2}{6}.
$$

Sin la dispersión del píxel, v7 mostraba motas brillantes en los folds. Con $\tau$ 1 / 3, las motas pasaron de 24 / 27 a 6 / 2 en el umbral y de 19 / 25 a 2 / 1 en el fold de la puerta; la puerta pasó de 34.21 a 35.53 dB. Un lobe cuyo espejo se queda dentro de 2 texels de su nivel de cielo a lo largo de $3\sigma$ se lee una vez en el centro (el dibujado −10 a −12 % de cerca, sin cambio frente al path trace dentro de 0.001 dB).

**El entorno como atlas filtrable.** Al principio el fragmento decodificaba dos niveles de cuatro texels cada uno desde un storage buffer. Ahora los ocho niveles prefiltrados de la cúpula se copian en un único atlas rgba16float: el nivel 0 en el origen, los demás en una columna a su derecha (0.9 MB con un lado base de 256, 3.1 MB con 512). El atlas se muestrea con un sampler bilineal de hardware, dos muestras por lobe. La imagen queda a 77 dB de las lecturas del storage. El pase compacto a 393 × 852, $\tau=3$ (M5 Pro, tiempo de reloj) pasó de 11.8 a 4.0 ms con sharp lobes desde 1 px; a 1920, $\tau=1$, de 75 a 17.2 ms. El shade también lee sus propios texels del entorno desde el atlas, lo que liberó el buffer del entorno (−2.67 MB).

**polishField y coatFold.** Cada material lleva su propia roughness en los bits 26–31 de pbr, sin coste por nodo. Donde esa roughness es $\ge 0.3$, el polish consulta las cells de su propio lobe, y el reflected field responde lo que cierran (el cociente de la puerta frente al path trace pasó de 0.786 a 0.857). Donde la superficie gira deprisa, el clear coat se sujeta: $r_{\text{coat}} = \max(\mathrm{unwidened}(c,s),\ \min(c, 0.3))$. Solo se aplica donde la roughness del material es $\ge 0.3$. Aplicado en todas partes, iluminaba cromo, cristal y radios de las llantas (ruedas +63 % de relMSE). En el bake de 1280 el clear coat sale a 0, así que el término ya no cambia el coche.

## 7.9 La contact shadow

Un set compacto se dibuja sin el renderer de Spark, así que la contact shadow es un pase propio (`ground-shadow.js`), y usa solo Gaussians (sin mesh y sin AO). El shadow catcher de athenea son Gaussians negras sobre el plano del suelo, cada una con sus cells de open directions. Se splatean sobre la cúpula antes del coche: EWA con el mismo blur de 0.3 px² y la misma compensación de opacidad, negras con opacidad $\alpha_i = o_i\,\kappa_i$. Negro sobre negro no necesita sort. El oscurecimiento es

$$
\kappa_i = \frac{-\ln r_i}{2\pi},\qquad
r_i = \frac{\displaystyle\sum_{\text{cells abiertas } j} L_{\text{dome}}(\boldsymbol\omega_j)\cos\theta_j\,\Delta\Omega_j \;+\; \text{sol a través de las cells}}{\text{irradiancia de todo el cielo sobre el suelo}},
$$

recalculado en la CPU a partir de los 16 harmonics de la cúpula cada vez que cambia la cúpula. El catcher de día de athenea (309k Gaussians a 1.26 cm) cuesta 9.9 MB de GPU dibujado entero, y aclarado a 1 de cada 4 muestra un grano ondulado. En su lugar, `floor_shadow --merge 2` ajusta por momentos grupos de 2 × 2 vecinos (centro y covarianza ponderados por opacidad × área, masa conservada, cells por mayoría). El resultado son 78k Gaussians, 0.65 MB en gzip, 2.5 MB de GPU, y es el más cercano al path trace en todas las regiones:

| región (factor de sombra) | PT | aclarado 1 de cada 4, 77k | completo 309k | merge 2 × 2, 78k |
|---|---:|---:|---:|---:|
| rueda delantera | 0.607 | 0.565 | 0.568 | **0.594** |
| rueda trasera | 0.713 | 0.617 | 0.618 | **0.638** |
| suelo delantero | 0.211 | 0.168 | 0.164 | **0.224** |
| bajo el umbral | 0.437 | 0.390 | 0.387 | **0.430** |

Sin la sombra el coche flotaba, y el umbral salía 1.66 veces el path trace: era el suelo de la nave en el infinito visto por el borde del coche. Con la sombra el umbral sale 1.08 (escritorio: 1.07).

## 7.10 Control por tiempo de frame y el still frame

El objetivo de dibujado ha tenido cuatro diseños:

1. **Objetivo de dibujados.** Empezaba en 260k dibujados y bajaba en el acto cuando la mediana de GPU pasaba de 16.5 ms. Subía un 6 % por segundo mientras estaba por debajo de 14 ms.
2. **~227k dibujados.** Con el ojo quieto, el suelo de $\tau$ se buscaba por bisección hasta dibujar unos 227k (el cut a 60 fps de v6).
3. **Suelo de $\tau$ por tiempo de frame, más un still frame**, con un objetivo de 16.5 ms.
4. **Primero la calidad**: la calidad de imagen tiene prioridad sobre la tasa de frames, con un objetivo de 33 ms. Es la versión actual.

En la versión actual, el tiempo de GPU de un frame se mide desde su inicio hasta `onSubmittedWorkDone`. Cada 0.5 s, con $\bar t$ la mediana de al menos 8 frames en movimiento y $T = 33$ ms:

$$
\tau_{\text{floor}} \leftarrow
\begin{cases}
\tau_{\text{floor}}\cdot\min\!\left(1.6,\ \sqrt{\bar t / (0.91\,T)}\right) & \bar t > T,\\[2pt]
\tau_{\text{floor}}/1.04 & \bar t < 0.85\,T \text{ durante dos periodos},\\
\tau_{\text{floor}} & \text{en otro caso},
\end{cases}
\qquad \tau_{\text{floor}} \in [1.5,\ 8]\ \text{px}.
$$

La raíz cuadrada sale de que el número de dibujados escala más o menos como $1/\tau^2$.

**El still frame.** Tras 300 ms sin entrada ni cambio de cámara, el cut en movimiento se fija en el pool de nodos, $\tau$ baja a 0.5 px y el límite de pages sube del pool en movimiento (14 MB de traversal) al pool del still frame (28 MB). Se dibuja un cut fino cueste lo que cueste, y se vuelve a dibujar según llegan más pages. Después no se redibuja nada hasta un toque, que trae de vuelta el cut en movimiento fijado en ese mismo frame y desaloja hasta el budget en movimiento a 32 pages por actualización. El budget es de unos 100 MB de datos del coche en reposo y unos 55 MB en movimiento. Medido en el móvil emulado, de cerca y quieto: 529k dibujados con $\tau$ 1.23 y 95.6 MB de datos del coche. Una segunda medida dio 531k con $\tau$ 1.22 en 118 ms, a 190 ms por millón simulados, con el primer frame tras un toque en 39–47 ms.

## 7.11 Contabilidad de memoria

El budget de 50–60 MB se aplica solo a los **datos del coche**. Los datos del coche son las pages y los tramos residentes, la traversal en WASM, el heap y los buffers del worker de nodos, y los registros de la GPU. El **estado de render** se cuenta aparte: registros de sharp lobes, atlas, sombra, targets, sort. Es lo que da la calidad de imagen, y una pestaña del iPhone 15 llegó a unos 1.2 GB en las sondas de estrés. La **página** en sí (heap de JS, three.js, cúpulas) es una tercera cifra.

| estado | datos | datos del coche | estado de render | página |
|---|---|---:|---:|---:|
| seis piezas, $\tau$ 1, emulado | CMPC v1 | — | GPU 53 MB en total (pico 59.6): datos 24.7, fx 9, target + cúpula 7.9, env 2.7, sort 4.6, canvas 3.8 | — |
| coche entero, $\tau$ 3, primer streamer de v4 | v5 | ≈60 MB de CPU + GPU (pool de pages al mismo $\tau$: ≈87) | — | — |
| coche entero, $\tau$ 3, registros de traversal por clase | v6 | Camera 55.1, cerca 59.0, lejos 49.1 (todo junto) | | |
| coche entero, $\tau$ 3, tras el atlas | v6 | Camera **39.0**, cerca **43.8**, lejos **33.0** | 24.1 / 26.0 / 21.7 | ~39 → 25 → ~17 (dos reducciones) |
| coche entero, $\tau$ 3 | v7 | 31–36 | — | — |
| still frame, de cerca | v7 | **95.6** (529k, $\tau$ 1.23) | — | — |

Varias reducciones fueron idénticas al byte en la imagen:

- los floats del cielo del relight se reconstruyen desde el RGBE al subirlo (1.6 MB retenidos en vez de 8 MB para una cúpula de 1k);
- la cúpula visible en rg11b10ufloat en el móvil;
- un único codebook de oclusión compartido (antes, 4.3 MB de arrays de JS);
- los registros de sharp lobes dimensionados a las ranuras dibujadas (−0.3 a −2.2 MB);
- el WASM servido como fichero aparte, con el layout parseado en un worker.

La contabilidad anterior por Gaussian explica por qué hacía falta el camino compacto. El relight previo guardaba 308 B por Gaussian más 29 MiB fijos: 147 / 206 / 323 MiB con 0.4 / 0.6 / 1.0M. El camino compacto guarda 42.4 B + 16 B por splat dibujado + unos 3 MiB.

## 7.12 Resultados medidos

### 7.12.1 El iPhone real

**Página de estrés, 10–11 de octubre** (rampas sintéticas sobre splats reales del Corvette; medianas):

| camino | 200k | 400k | 600k | 800k | ~1M | notas |
|---|---:|---:|---:|---:|---:|---|
| 3DGS plano, fps | 58.8 | 58.8 | 45.5 | 34.5 | 26.3–27 | móvil frío; ~37 ms por millón |
| transfer, camino "by" anterior, fps | 38.5 | 12.3 | 7.2 | 5.2 | 3.7 | 362–469 MB de GPU |
| compacto, base sintética, ms (fps) | 30 (33.3) | 63 (15.9) | 101 (9.9) | 137 (7.3) | 154 (6.5) | 20.8–79.8 MB de GPU |
| compacto, base ajustada sobre datos del coche, shade optimizado, ms (fps) | 27 (37) | 47 (21.3) | 70 (14.3) | 95 (10.5) | 108 (9.3) | 35 MB de GPU con 400k |

Modelo del shade optimizado: unos 115 ms por millón dibujado + 4 ms, así que 30 fps aguantan unos 250k dibujados y 60 fps unos 110k. El primer segundo de cada escalón corría 2–3 veces más rápido que el ritmo sostenido (reloj o estado térmico), y una pasada hecha tras cuatro pasadas en 20 minutos fue la mitad de rápida a partir de 600k. Para planificar se usan las cifras sostenidas.

**Pasadas del coche entero en el device** (datos v7):

| pasada | vista | dibujados | $\tau$ | fps | observación |
|---|---|---:|---:|---:|---|
| primera | por defecto | 106k | 8.00 | 53 | fluido; el $\tau$ 8.00 era la "respiración" vieja por tiempo de frame atascada en su techo (arreglado) |
| primera | de cerca | 160k | — | 32 | logo y grille blandos |
| segunda | de cerca | — | — | — | aún blando |

El coste de shade estimado en ese iPhone es de unos 190 ms por millón. Lo blando de cerca viene de la nube, no del player: el logo con bake a 1280 px sale borroso incluso con todas sus leaves, y nítido a 5120 px. El close-up level car5120 (v8) lo resuelve.

### 7.12.2 El iPhone emulado

| datos | vista | dibujados / residentes | $\tau$ | descarga | fps |
|---|---|---|---:|---:|---:|
| seis piezas, paginado | por defecto | 174k / 384k | 1.55 | 25 MB | 61 |
| seis piezas, CMPC v1 | por defecto | 199k / 555k | 1 | 24.6 MB | 60 |
| el mismo, CMPC v2 + `.cmpa` | por defecto | 199k / 555k | 1 | 17 MB | 60 |
| coche entero, v3 (pool de 24 MB, fov = h) | por defecto | 147k / 580k | 6.75 | 20 MB | 60 |
| coche entero, v5 (CMPC v4) | por defecto | 226k / 1016k | 3 | 36 MB leídos | 60 |
| coche entero, v7 | por defecto | 216k / 1318k | 2.22 | — | 60 |
| coche entero, v7, revisiones posteriores del streaming | por defecto | 234k | 1.95 | — | 60 |
| coche entero, v7, still frame | de cerca | 529k | 1.23 | — | (un frame) |

### 7.12.3 Frente al path trace, por versión de datos

Las versiones de datos de `car-compact-tx` son distintas de las versiones del formato `CMPC`:

| set | contenido |
|---|---|
| v3 | occl8k, CMPC v3 |
| v4 | trees closed2 de athenea |
| v5 | tramos de CMPC v4 |
| v6 | trees mrough de athenea, roughness por material, 1.35 GB |
| v7 | car1280, 1.81 GB |
| v8 | car5120: 24 trees, 74.5M leaves, 148.9M nodos, oclusión compartida de 8182 entradas (§6.9) |
| v8.2b (final) | car5120 rehecho (§6.2.1): 149.0M nodos, oclusión de 8181 entradas, una base compartida de 512 clusters reajustada con los datos de 5120 y cuatro bases de 256 propias del asset (Part_2, Infinite_Black, dos partes de Part_3 y Front_Bumper_001), 83 ficheros, 12.77 GB; relMSE del transfer 0.0054 sobre el coche |

Salvo que se indique, las cifras de abajo son PSNR en la cámara de athenea a 1920 × 1080, por las regiones del harness validate-corvette.

| versión | coche | puerta | umbral | parachoques | capó | fuente / condición |
|---|---:|---:|---:|---:|---:|---|
| v6 | 19.33 | 33.77 | 31.87 | 20.21 | 14.16 | camino compacto |
| v6, desensanchado de Toksvig | 19.68 → 20.58 | +1.0 | — | — | +1.3 | con detalle por píxel |
| v7 (primera conversión) | +0.5 frente a v6 | −2.3 | −2.7 | — | +0.9 | motas en los folds |
| v7 + dispersión del píxel, $\tau$ 1 | — | **35.53** | **32.36** | relMSE 0.199 | 14.46 | con dispersión del píxel |
| v7, valores por defecto actuales | 19.03 | 35.52 | 32.36 | 22.28 | 14.68 | campo de grado 1 |
| v8.1 | 19.22 | 35.92 | — | — | — | pool 3000, $\tau$ 1; dibuja 4.0 M |
| v8.2b | 19.27 | 35.53 | — | — | — | pool 3000, $\tau$ 1; dibuja 2.26 M; logo desde una cámara cercana 18.12 dB (v8.1 17.75) |

La métrica car crop de §1.6 (recorte de 900 × 420) es otra región. Con ella, el sharp lobe por píxel en el camino compacto llevó el coche de 16.49 a 17.46 dB (17.41 con sharp lobes desde 4 px), que es el nivel del camino de escritorio. En el móvil, $\tau$ 3 frente a $\tau$ 1 difiere en 40 dB, así que los defectos que quedan (pelos, rayas, manchas) vienen de la nube, no del cut. Los pasos sin pérdida se comprobaron frente a la imagen anterior: los registros de sharp lobes en 5 palabras, 60–80 dB; el shade agrupado, 84 dB (ruido entre ejecuciones 78 dB); los términos solo de cristal, 188 dB.

## 7.13 Limitaciones y trabajo pendiente

- **v8.2b**, el conjunto de datos final (car5120 rehecho, §6.2.1), está publicado: frente al path trace mejora en el coche, la parrilla, el cromo, el paragolpes y el logo de cerca, y pierde 0.39 dB en la puerta frente a v8.1 dibujando la mitad de Gaussians. Su descarga y sus tiempos de frame en el móvil real están sin medir.
- Solo hay tiempos del device real para las rampas de estrés y las dos pasadas del coche entero. La velocidad del móvil emulado no es la del móvil.
- La tira cromada del parachoques delantero sale un 38 % más clara que el path trace. Las lámparas no tienen emisión en el bake. Los reflejos abiertos de la puerta (1.13) vienen de cells de 16 × 16 de 11°.
- La traversal compacta no guarda el tamaño de LoD de un merge, así que el cut combinado tamaño/error no tiene con qué protegerse. El cut por error por sí solo no ganó al cut por tamaño en todas las vistas con estos datos.
