<!-- Copyright (c) 2026 jesus luque. -->

# 4. Transporte de luz horneado en gaussianas (transfer) y reiluminación

Una nube convertida desde una malla puede llevar su luz de dos maneras. Un *horneado de radiancia* guarda, por
gaussiana, la luz que llegó bajo la iluminación del propio stage; es exacto para esa iluminación y erróneo bajo
cualquier otra. Un *transfer* guarda en cambio cómo el entorno de cada gaussiana le llevaría un entorno arbitrario, y un
fotograma combina esa geometría con el cielo y las luces que tenga. Esta sección describe el transfer tal como athenea
lo hornea y lo evalúa en su forma TX («tarea TX»: un transfer correcto bajo cualquier cielo y cualquier luz), el modelo
de reflexión construido encima, el tratamiento por píxel de los lóbulos nítidos, y lo que se midió que aporta cada pieza
frente a un path trace de la malla de origen. Todo lo descrito corre en la GPU: el horneado es el path tracer de la
propia escena en un modo transfer, y la evaluación son los kernels de proyección y mezcla de splats
(`shaders/athenea/splat/splat_relight.slang`, `splat_project.slang`, `splat_blend.slang`).

## 4.1 Notación

Una gaussiana tiene una posición $\mathbf{p}$, una normal unitaria $\mathbf{n}$ (su eje más corto, o una normal de
sombreado guardada, girada hacia el ojo), un albedo $\rho$, una metalicidad $m$, una rugosidad perceptual $r$ con
parámetro GGX $\alpha = r^2$ y, opcionalmente, un clear coat (peso $w_c$, rugosidad $r_c$, índice $\eta_c$), un sheen y
una transmisión. $\boldsymbol{\omega}_o$ es la dirección hacia el ojo y $\mathbf{m} = \operatorname{reflect}(-\boldsymbol{\omega}_o,
\mathbf{n})$ la dirección especular. $Y_k$ son los armónicos esféricos reales, con índice $k = l^2 + l + m$; el grado 2
tiene 9 coeficientes y el grado 3 tiene 16. Un cielo tiene coeficientes de radiancia $L_k$ (rgb), un sol de dirección
$\mathbf{s}$ e irradiancia $E_\odot$ extraído de él, y un mapa prefiltrado $\tilde L(\boldsymbol{\omega}, r)$.

## 4.2 Qué guarda una gaussiana

El transfer es una tira de valores por gaussiana, f16 en el dispositivo. No hay token de modo: la cuenta es el formato
(`common/transfer_layout.slang`).

| valores | formato | origen |
|---|---|---|
| 9 | grado 2, directa | primer transfer |
| 36 | grado 2, directa + indirecta (27) | primer transfer |
| 84 | grado 2, ambas + campo reflejado (48) | TX con `--transfer-degree 2` |
| 16 | grado 3, directa | |
| 64 | grado 3, directa + indirecta (48) | |
| 112 | grado 3, ambas + campo reflejado (48) | **TX por defecto** |
| 10 | dos lóbulos zonales en el marco de la gaussiana | nubes con esqueleto |

Junto a ella:

- **Direcciones abiertas** (`primvars:athenea:splat:shadowBits`): 2 palabras (64 bits, una rejilla octaédrica de
  8 x 8; primer transfer), 8 palabras (16 x 16 celdas sobre la esfera entera; TX por defecto) o 32 palabras (32 x 32).
  Un bit se pone donde el rayo del horneado en esa celda salió de la escena.
- **Curvatura** (`primvars:athenea:splat:curvature`): el operador de forma en los dos ejes del plano de la gaussiana,
  $(S_{uu}, S_{uv}, S_{vv})$, tres floats, escrita con `--transfer`.
- **Rugosidad del material** (`primvars:athenea:splat:materialRoughness`): la rugosidad propia del material antes del
  ensanche del horneado (sección 4.9), -1 bajo un mapa de textura. En el dispositivo va en cinco bits (27-31) de la
  palabra pbr, marcada por el bit 26.

La mitad directa es un escalar por coeficiente (solo geometría); la indirecta es rgb, coloreada por aquello en lo que
rebotó la luz; el campo reflejado es rgb. El formato TX cuesta, por gaussiana:

| | primer transfer | TX |
|---|---|---|
| directa | 9 | 16 |
| indirecta | 27 | 48 |
| campo reflejado | -- | 48 |
| visibilidad | 2 ints (64 bits) | 8 ints (256 bits) |
| fichero, bytes | 152 | 480 |
| dispositivo, bytes | 80 | 256 |

Para el Corvette entero (unos 10 M de gaussianas entonces) son 4.8 GB de fichero y 2.6 GB de dispositivo solo para el
transfer; los códecs que lo llevan a un teléfono (un PCA por clusters del transfer, un codebook k-means de Hamming de los
bits de sombra) se describen en la sección 7.

## 4.3 El entorno en el momento de dibujar

`technique::Environment` prepara cada domo en tres formas, todas buffers leídos a través de un único módulo
(`light/environment.slang`) por el rasterizador y por el trazador.

**Armónicos esféricos de radiancia.** `env_project` suma los téxeles de un nivel de la cadena de mips del domo, cada uno
pesado por el ángulo sólido lat-long $2\pi^2 \sin\theta\,\Delta u\,\Delta v$ (la medida que usa el muestreo por
importancia del domo):

$$L_k = \sum_{\text{téxeles}} L(\boldsymbol{\omega}_t)\, Y_k(\boldsymbol{\omega}_t)\, 2\pi^2 \sin\theta_t\, \Delta u\, \Delta v .$$

Se proyectan dieciséis coeficientes (grado 3); la irradiancia de un cuerpo sin transfer lee los nueve primeros,
convolucionados al leer con los pesos zonales del coseno truncado $\{\pi, 2\pi/3, \pi/4\}$. Un cielo constante vuelve a
dar $\pi L$ de irradiancia en toda normal: 6.2841 frente a 6.2832 (0.089 %) sobre 64 direcciones.

**El sol, sacado.** Un disco mil veces más brillante que el cielo no es una función de grado 2; proyectado, se emborrona
en un lóbulo de sesenta grados. `env_sun` encuentra la región más brillante cuya luz está tan concentrada como un disco
de su ángulo sólido (ángulo cuadrático medio pesado por la luz alrededor de su centro como mucho el doble que el de un
disco, $\theta_{\text{cone}}^2/2$), saca sus téxeles tanto de los armónicos como de la fuente del prefiltrado, y devuelve
una dirección $\mathbf{s}$, una irradiancia $E_\odot$ y un semiángulo. La irradiancia se suma exactamente sobre los
téxeles que los armónicos omiten, al mismo nivel y con la misma medida, de modo que lo que sale de los coeficientes llega
como luz (forma cerrada 0.1119 frente a 0.1113).

**El mapa prefiltrado.** El cielo sin su sol, convolucionado con un lóbulo GGX en una cadena de mips octaédrica de ocho
niveles, el nivel $k$ para la rugosidad $(k/7)^2$ (0, 0.020, 0.082, 0.184, 0.327, 0.510, 0.735, 1). El lado de la base
iguala el téxel del propio domo, $S = \sqrt{2WH/\pi}$, redondeado hacia abajo a una potencia de dos en $[256, 2048]$. El
prefiltrado usa muestreo por importancia filtrado (Krivanek y Colbert): cada muestra GGX lee la fuente un nivel más
basto de lo que pide su propio ángulo sólido, nunca más estrecho que el téxel que escribe, con la secuencia girada por
un hash por téxel, 64 muestras en el nivel 1 que se doblan hasta 1024. Los mips lat-long de la fuente pesan cada fila
por $\cos\theta_0 - \cos\theta_1$, de modo que cada nivel conserva la media por ángulo sólido del cielo. Sin esto, un sol
de 1.5 grados en un nivel rugoso era una dispersión de manchas del tamaño de un téxel (la banda de 4-16 px de una bola
de pintura a 2.13 veces el path trace).

## 4.4 El horneado

### 4.4.1 La integral de transfer y su estimador

El transfer es la proyección de la visibilidad por el coseno truncado, con el albedo tomado como uno:

$$T_k = \frac{1}{\pi} \int_{S^2} V(\boldsymbol{\omega})\, \max(\mathbf{n}\cdot\boldsymbol{\omega}, 0)\, Y_k(\boldsymbol{\omega})\, d\boldsymbol{\omega},$$

y un fotograma lo recombina con el cielo como $\langle T, L \rangle = \sum_k T_k L_k$. Lo recoge una tercera variante del
kernel de path tracing, `kTransfer`, junto al horneado de radiancia y al camino de cámara. En el primer vértice de una
gaussiana el material se sustituye por la propia dirección muestreada: las direcciones están estratificadas (una
rejilla $\sqrt{N}\times\sqrt{N}$ con jitter) y son uniformes sobre el hemisferio, densidad $1/2\pi$, frente a un
estimador que quiere $\cos\theta/\pi$, así que una muestra que escapa vale $2\cos\theta\, Y_k(\boldsymbol{\omega})$ y
una bloqueada nada. Uniforme en lugar de coseno, y sobre el hemisferio medido como una esfera con la mitad lejana vacía,
porque los armónicos son ortonormales solo sobre la esfera; ajustados coeficiente a coeficiente sobre un hemisferio se
pasan (el peón salía diez veces demasiado brillante). La acumulación *es* la proyección; no se resuelve ningún ajuste.

Los mismos caminos dan dos mitades sin ningún rayo más: un camino que escapa justo después del primer vértice contribuye
a la mitad directa (escalar); uno que escapa tras rebotar lleva el color de lo que encontró y contribuye a la mitad
indirecta $T^{\text{ind}}_k$ (rgb). Comprobado frente a formas cerradas: un punto sin oclusión transfiere el coseno
truncado, $(0.2820, 0.3258, 0.1581)$ frente al exacto $(0.2821, 0.3257, 0.1577)$ para las bandas 0-2, y un cielo
constante de 0.7 lee 0.6998. Una esquina de suelo y pared, transfer recombinado frente a un horneado de radiancia de los
mismos puntos:

| | transfer | horneado de radiancia | peor punto |
|---|---|---|---|
| sin pared, 1 rebote | 0.6000 | 0.6000 | 0.0 % |
| pared negra, 1 rebote | 0.3091 | 0.3039 | 2.5 % |
| pared blanca, 3 rebotes | 0.5095 | 0.5325 | 4.3 % |

### 4.4.2 Direcciones abiertas y la mitad directa por cuadratura

En la primera muestra de cada punto TX traza un rayo any-hit por celda de una rejilla octaédrica de $S_c \times S_c$
($S_c = 16$ o 32) sobre la esfera *entera*: también la mitad lejana, empezando desde la cara de atrás de la superficie,
de modo que una dirección detrás de un sólido encuentra el sólido y una detrás de una lámina sale. El bit de una celda
se pone donde pasa al menos la mitad de la luz: el rayo de oclusión atraviesa hasta ocho superficies cuyo material
transmite, pesadas por la transmisión por la luminancia del color menos el Fresnel a ese ángulo (una novena cierra el
paso). Las celdas se trazan desde cuatro orígenes sobre la huella de la gaussiana (la celda $c$ desde el origen
$(c + \lfloor c/S_c \rfloor) \bmod 4$: el centro y tres puntos llevados a la superficie dentro del disco de la
gaussiana), de modo que una ranura más estrecha que una gaussiana ya no se lee como una línea discontinua de sombras
enteras.

Los mismos rayos dan entonces la mitad directa por cuadratura (sin grano), cada celda pesada por su ángulo sólido
octaédrico:

$$T_k^{\text{dir}} = \sum_{c} \tau_c\, \frac{\max(\mathbf{n}\cdot\boldsymbol{\omega}_c,0)}{\pi}\, \Delta\Omega_c\, Y_k(\boldsymbol{\omega}_c),
\qquad \Delta\Omega_c = \frac{4}{S_c^2}\,\lVert\boldsymbol{\omega}_c\rVert_1^3,$$

con $\tau_c$ la fracción transmitida a lo largo del rayo de la celda. Las palabras se escriben de cuatro en cuatro a
medida que se trazan cada 128 celdas, de modo que el kernel no guarda ningún array del tamaño de la rejilla.

### 4.4.3 El campo reflejado

Una reflexión necesita la luz rebotada *por dirección*, no integrada. Para cada camino que sale de la gaussiana,
encuentra la escena y luego escapa, el throughput pasado el primer vértice es la radiancia que llega por su primera
dirección $\boldsymbol{\omega}_1$ bajo un cielo blanco de radiancia uno. Se proyecta sobre el grado 3 de
$\boldsymbol{\omega}_1$, uniformemente sobre el dominio del que se sacó:

$$F_j \approx \frac{1}{N} \sum_{i:\ \text{escapó tras un rebote}} \Omega_{\text{meas}}\, \frac{\beta_i}{w_i}\, Y_j(\boldsymbol{\omega}_{1,i}),$$

con $\beta_i$ el throughput, $w_i$ el peso del primer vértice y $\Omega_{\text{meas}} = 2\pi$ (o $4\pi$ sobre la esfera).
Son 16 coeficientes rgb, `primvars:athenea:splat:transferReflected`. No cuesta ningún rayo. Como $F$ es cero en las
direcciones abiertas, dieciséis armónicos leen una dirección cerrada junto a una abierta a una fracción de lo que hay
allí (la parte cerrada de la pintura era 0.0014 de 0.115). Así que el horneado proyecta también el indicador de las
direcciones cerradas $C_j$ sobre las mismas muestras, y la conversión divide ambos, una convolución normalizada que da
la radiancia media de las direcciones cerradas alrededor de cada dirección, cae a la media de las direcciones cerradas
donde su proporción está por debajo de 0.1 (mezclada hasta 0.4), y vuelve a proyectar el resultado sobre una rejilla de
16 x 16 por el ángulo sólido de cada celda (`m2sFieldOverClosed`).

Para una gaussiana que transmite, la primera dirección se saca sobre la esfera entera (`bakeSphereDirection`: la altura
del estrato estirada a $[-1,1]$ y la mitad inferior reflejada en el plano tangente). Las muestras de delante conservan
el estimador del transfer ($4\cos\theta$ con densidad $1/4\pi$); las de detrás alimentan solo el campo, tras doblarse
hacia dentro de un dieléctrico sólido por su índice (`refractInto`), de modo que el campo a lo largo de
$-\boldsymbol{\omega}_o$ contiene lo que el rayo del ojo encuentra a través de ambas caras. Sin doblar, una bola de
vidrio validaba a la mitad del path trace (0.265 frente a 0.541 bajo un domo blanco).

### 4.4.4 Rebanadas, filtrado y el horneado partido

El Corvette TX entero pedía la respuesta de su horneado de una vez: 35 float4 por gaussiana, 7.8 GB, con 5 GB libres.
`Converter::transfer` hornea por tanto en rebanadas cuya respuesta es como mucho 1.5 GB (nunca por debajo del lote
propio del horneado, $2^{19}$), y como mucho $2^{20}$ gaussianas donde corre el filtro. Los rayos de cada rebanada se
copian en el dispositivo, se hornean (`StageRenderer::bakePointsOnDevice` en modo transfer), se filtran sus mitades
rebotadas, se escriben con `m2sTransferInto` y se leen de vuelta a los arrays del fichero antes de la siguiente. El
horneado de radiancia usa el emparentado `StageRenderer::bakeSplitOnDevice`, que guarda las mitades directa e indirecta
como sumas separadas para que la indirecta pueda filtrarse antes de que `combineBake` ajuste ambas. El transfer sigue el
mismo principio: su mitad indirecta y su campo (los valores más ruidosos, a 64 caminos por gaussiana en los horneados
del Corvette) pasan por el filtro de horneado de splats, un filtro à-trous de `--bake-filter` iteraciones (3), cuyos
pesos son la posición en el plano tangente, la orientación y el id de prim; la mitad directa, que contiene las sombras,
no se filtra nunca.

### 4.4.5 Lóbulos zonales para nubes con esqueleto

Nueve armónicos están en el mundo; un esqueleto gira la gaussiana debajo de ellos. Rotarlos cuesta una 9 x 9 por
gaussiana. Un lóbulo zonal rota rotando su eje, así que una nube con esqueleto guarda dos lóbulos, cada uno con el
$(u, v)$ octaédrico de su eje en el marco propio de la gaussiana y los coeficientes zonales $z_0, z_1, z_2$: diez valores
(`primvars:athenea:splat:transferZonal`). Como armónicos, un lóbulo alrededor de $\mathbf{a}$ es
$g_{lm} = z_l\, k_l\, Y_{lm}(\mathbf{a})$ con $k_l = \sqrt{4\pi/(2l+1)}$. Para un eje fijo los mejores coeficientes son la
proyección $z_l = k_l \sum_m f_{lm} Y_{lm}(\mathbf{a})$, que deja un error $\lVert f\rVert^2 - \sum_l z_l^2$, así que el
ajuste (un efecto AOFX, `splattransferzonal`) busca el eje en el dispositivo (la dirección de la banda 1, la normal, 64
direcciones de Fibonacci y luego un paseo que divide el paso a la mitad), ajusta el segundo lóbulo al residuo y reajusta
el par dos veces (Sloan 2008). Al dibujar, `splatTransferFrame` gira cada eje con el marco deformado y reconstruye nueve
armónicos del mundo, que todo lo de después lee sin cambios. El horneado posa la nube en `--time` con el propio skinner
del motor, de modo que el marco contra el que se ajustan los lóbulos es el cuaternión empaquetado que lee un renderer.
Los bits de sombra siguen siendo dos palabras, reagrupados en el marco de la gaussiana. La mitad indirecta no se guarda
zonal.

## 4.5 El cuerpo

Para una gaussiana con transfer, la parte que no depende del ojo es

$$\mathbf{B} = \sum_{k<16} L_k\,\big(T_k^{\text{dir}} + T_k^{\text{ind}}\big) \;+\; E_\odot\,\frac{\max(\mathbf{n}\cdot\mathbf{s},0)}{\pi}\, O(\mathbf{s}) \;+\; E_\odot\, T^{\text{ind}}(\mathbf{s}),
\qquad T^{\text{ind}}(\mathbf{s}) = \max\Big(\sum_k T_k^{\text{ind}} Y_k(\mathbf{s}), 0\Big),$$

siendo el último término el rebote del sol, que el primer transfer perdió con el sol que sacó de los armónicos.
$O(\mathbf{s})$ es la parte del sol que llega: para TX la lectura bilineal de las cuatro celdas alrededor de
$\mathbf{s}$, dejando fuera las celdas detrás de la normal; para los 64 bits lo mismo en una rejilla de 8 x 8; sin bits,
el cociente de grado 2

$$O_{\text{SH}}(\mathbf{s}) = \frac{\sum_k T_k Y_k(\mathbf{s})}{\sum_k \hat A_l\, Y_k(\mathbf{n})\, Y_k(\mathbf{s})}, \qquad \hat A_l = \{1, 2/3, 1/4\},$$

con numerador y denominador compartiendo el truncamiento, de modo que lee uno donde nada ocluye. El cuerpo difuso es
entonces $U\,\rho\,(1-m)\,\big[(1-t)\,\mathbf{B} + t\,(\ldots)\big]$ con $U$ el throughput de las capas (sección 4.6.4) y
$t$ la transmisión. $\mathbf{B}$, la parte del sol y el acoplamiento del campo (abajo) se guardan por gaussiana mientras
el cielo y las luces no cambian (`transferViewless`, el `txCache`); un fotograma calcula solo lo que cambia el ojo.

El término constante del transfer es además una oclusión ambiental: sobre un hemisferio abierto
$T_0 = A_0 Y_0 / \pi = 0.28209$, así que $o = \operatorname{sat}(T_0 / 0.28209)$ es la fracción del cielo que llega a la
gaussiana.

**Luces que no son el cielo.** Una luz distante, esférica, de disco o rectangular se evalúa con una muestra en su centro
a través de los mismos lóbulos. Con las celdas TX, su sombra son las celdas leídas sobre el cono que la luz subtiende
($\tan^{-1}$ de su radio sobre la distancia, o la mitad del diámetro angular de una luz distante), la más oscura entre
esa y cualquier sombra medida con rayos; su rebote es $T^{\text{ind}}$ leído a lo largo de ella, por el $L/p$ de la
muestra, sobre el cuerpo y el sheen, y acopla el campo reflejado igual que el sol (`splatLightBounce`).

## 4.6 Reflexión

### 4.6.1 Qué parte de un lóbulo ve el cielo

El primer transfer estrechaba toda reflexión con el ajuste de Lagarde y de Rousiers del término constante,

$$O_{\text{L}}(\mathbf{n}\cdot\boldsymbol{\omega}_o, o, r) = \operatorname{sat}\!\Big( (\mathbf{n}\cdot\boldsymbol{\omega}_o + o)^{2^{-16r-1}} - 1 + o \Big),$$

un número para todo el hemisferio: una puerta reflejaba el cielo abierto donde el suelo está en su espejo, y las llantas
del Corvette leían 2.4 veces la malla con path trace, `Metal_rough` 6.5 veces, los discos de freno cien. Con las celdas
TX se pregunta al propio lóbulo (`splatLobeOpen`, `splatConeOpen`): la lectura de celdas $o_c(\cdot)$ (bilineal sobre
cuatro celdas, excluidas las que quedan detrás de $\mathbf{n}$) en el espejo y en un anillo de seis direcciones a su
alrededor con ángulo

$$\theta_r = \max\!\big(\min(1.4\,\tan^{-1}\alpha,\ 1.2),\ 2\cdot\tfrac{\pi}{2S_c}\big),$$

más o menos el ángulo dentro del cual GGX refleja la mitad de su energía, nunca más estrecho que una celda; y

$$O(\mathbf{m}, \alpha) = \frac{\tfrac13 o_c(\mathbf{m}) + \tfrac19 \sum_{k=1}^{6} o_c(\mathbf{d}_k)}{\tfrac13 + \tfrac69}$$

sobre las lecturas que pesaron algo. Un cono dentro de una celda leía el único bit de esa celda y salpicaba la pintura
del Corvette (relMSE 0.393 frente al path trace; 0.298 con el cono limitado a una celda).

### 4.6.2 Qué muestra la parte cerrada: el campo acoplado al cielo

El campo se horneó bajo un cielo blanco. Bajo uno real su patrón se mantiene y su brillo se escala por canal según
cuánto más rebota el entorno bajo ese cielo que bajo el blanco (`splatFieldCoupling`):

$$\kappa = \frac{\sum_k T_k^{\text{ind}} L_k + E_\odot\, T^{\text{ind}}(\mathbf{s})}{\sqrt{4\pi}\; T_0^{\text{ind}}},$$

cayendo a la radiancia media del cielo donde la gaussiana no rebota nada. El acoplamiento es de rango uno: exacto para
una gaussiana rodeada por una superficie iluminada por un cielo, y pone el color de un suelo iluminado, no su forma, bajo
un cielo iluminado por un lado. Leído a lo largo de una dirección a la rugosidad de un lóbulo, las bandas del campo se
estrechan como un GGX reflejado tomado como una von Mises-Fisher de concentración $1/2\alpha^2$:

$$F(\mathbf{d}; \alpha) = \max\Big(\sum_{j<16} F_j\, e^{-l_j(l_j+1)\alpha^2}\, Y_j(\mathbf{d}),\ 0\Big).$$

Las dos rugosidades que necesita una gaussiana (polish y coat) se sacan de una sola lectura de los 48 valores
(`splatFieldAlongPair`). Las alternativas que se sopesaron fueron una matriz de transfer de 9 x 9 por gaussiana (243
números, 5 GB para el coche, y aún grado 2) y un índice por dirección dentro de la nube (necesita una caché de radiancia
y no ve nada de un suelo de malla).

### 4.6.3 Polish y coat

La reflexión de la base («polish») y el coat son

$$\mathbf{R}_{\text{polish}} = U\,g\,P(\mathbf{n}\cdot\boldsymbol{\omega}_o)\,\big[O\,\tilde L(\mathbf{m}, r) + (1-O)\,\chi\,\kappa\,F(\mathbf{m};\alpha)\big] + U\,g\,P\,E_\odot\, O(\mathbf{s})\, p_\odot(\mathbf{m}, \mathbf{s}),$$
$$\mathbf{R}_{\text{coat}} = w_c E_c\,\big[O_c\,\tilde L(\mathbf{m}, r_c) + (1-O_c)\,\kappa\,F(\mathbf{m};\alpha_c)\big] + w_c E_c\, E_\odot\, O(\mathbf{s})\, p_\odot(\mathbf{m},\mathbf{s}),$$

donde $P$ es el albedo de entorno del polish, $g$ la ganancia de pared fina (sección 4.7), $E_c$ el albedo del coat y
$\chi$ la puerta del campo del polish:

- **La apertura del polish y su puerta.** $\chi = 1$ y $O = O(\mathbf{m}, \alpha)$ solo donde la rugosidad propia del
  material es al menos `kPolishFieldRoughness` = 0.3; en el resto $\chi = 0$ y $O = O_{\text{L}}$. En el cromo (0) y el
  metal rugoso (0.22) del Corvette, la apertura por celdas leía un relMSE de 1.64 y 0.221 frente al path trace donde la
  de los armónicos leía 1.05 y 0.128 para la misma media, y el campo leía los metales un tercio más claros (cromo 0.261
  donde el path trace lee 0.197). Un lóbulo de 0.3 o más ancho es más ancho de lo que resuelve el grado del campo, así
  que lo que lo cierra (el suelo de una bola) es lo que dice el campo: bolas verdes y blancas a 0.3421 sobre un suelo
  gris reflejaban el cielo en su mitad inferior sin esto. La puerta lee la rugosidad *del material*, no la guardada,
  porque el ensanche del horneado lleva la rugosidad guardada de un tubo fino de cromo por encima de 0.3. El coat usa
  siempre las celdas y el campo.
- **El sol analítico.** El mapa no contiene sol, así que el sol se añade a través de la distribución de direcciones
  reflejadas de cada lóbulo con el ojo a lo largo de la normal, $D(\mathbf{h})/4$ para el vector medio entre espejo y
  sol, con el semiángulo del disco $\theta_\odot$ incorporado a la anchura:

  $$p_\odot(\mathbf{m}, \mathbf{s}) = \frac{\tilde\alpha^2}{4\pi\,d^2},\qquad d = \cos^2\!\tfrac{\theta}{2}\,(\tilde\alpha^2-1)+1,\qquad \tilde\alpha^2 = \alpha^2 + \theta_\odot^2,\qquad \cos\theta = \mathbf{m}\cdot\mathbf{s}.$$

  Integra uno, así que peso por $E_\odot$ por $p_\odot$ es la reflexión entera del sol. El split sum leía un sol
  compacto una cuarta parte corto en todo nivel del mapa (0.57-0.74 del path trace en una bola de metal flotante; parte
  de esa diferencia se atribuyó después a la densidad del domo del propio path tracer, ya corregida).

### 4.6.4 Capas, Fresnel, metales, sheen

**Albedo de entorno.** $P$ es el `ggxDirectionalAlbedo` de MaterialX con compensación de dispersión múltiple, no el
ajuste de Lazarov, que difería hasta un 76 % en rasante; un dieléctrico usa el Fresnel exacto frente a $F_0$, un
conductor su color en ambos extremos, mezclados por $m$. Donde el metal del material es un Schlick generalizado
(OpenPBR, glTF), va de su color de frente al color especular en rasante (`ggxEnvSchlickMetal`); para un metal oscuro un
conductor de índice artístico y un Schlick difieren al doble a sesenta grados, y la pintura del coche se leía mate y
clara como conductor.

**El Fresnel del coat.** Schlick se queda bajo en el medio de la curva de un dieléctrico (0.070 a sesenta grados donde
una laca a 1.45 refleja 0.081); un coat sobre negro leía 0.084 frente a 0.099 del path trace. El albedo del coat es el
ajuste escalado por exacto sobre Schlick en el ojo, desvaneciéndose con la rugosidad:

$$E_c = E_{\text{fit}}(r_c, \mathbf{n}\cdot\boldsymbol{\omega}_o, \eta_c)\cdot \operatorname{lerp}\!\Big(\frac{F(\mathbf{n}\cdot\boldsymbol{\omega}_o,\eta_c)}{F_{\text{Schlick}}},\ 1,\ \operatorname{sat}(\alpha_c)\Big),$$

y exactamente $F$ por debajo de $\alpha_c = 10^{-3}$. Frente a una cuadratura del lóbulo queda dentro del 3 % con
$\alpha$ 0.3 y 0.6, donde el ajuste se quedaba un 8-16 % por debajo.

**Capas.** Como el nodo `layer` de MaterialX: el coat sobre el sheen sobre la base, cada capa quitando de lo que hay
debajo la parte que refleja en el ojo. Con $E_{\text{fit}}$ el albedo ajustado del coat, $E_s$ el albedo direccional del
sheen y $w_s$ el mayor canal del color del sheen,

$$U = \big(1 - w_c E_{\text{fit}}\big)\big(1 - w_s E_s\big)\cdot \operatorname{lerp}\!\Big(1,\ \frac{1-K}{1-E_b K},\ w_c\Big),\qquad K = 1 - \frac{1-F_0(\eta_c)}{\eta_c^2},$$

siendo el último factor el oscurecimiento del coat de OpenPBR (la luz que vuelve de la base se refleja de nuevo hacia
abajo en la cara interior del coat), con $E_b$ el albedo de la base; la bola de pintura leía 1.5 veces la malla en su
centro sin él. El sheen (Imageworks, Conty y Kulla) es un lóbulo tan ancho como un cuerpo, así que se responde como uno:
su albedo direccional por $\mathbf{B}$, bajo el coat.

## 4.7 Vidrio, transmisión y paredes finas

Una gaussiana no puede refractar. Lo que transmite es la luz que le llega por detrás: $(1-t)$ del cuerpo mira hacia el
lado del ojo, $t$ mira hacia el otro, y lo que hay detrás se ve a través de la opacidad de la gaussiana.

- **Pasa lo que no se refleja**: la mitad transmitida se pesa por $1 - w_{\text{spec}} E_{\text{fit}}(r,\eta)$, con
  $w_{\text{spec}}$ el peso especular; un Fresnel por cara, como lo paga el lóbulo de transmisión del path tracer; antes
  una bola salía la mitad más brillante, con un borde 2.5 veces demasiado caliente.
- **El vidrio TX rasterizado** ve a través a lo largo de $-\boldsymbol{\omega}_o$: el cielo donde las celdas de detrás
  están abiertas (`splatConeOpen` alrededor de $-\mathbf{n}$) y $\kappa F$ donde están cerradas, de modo que un parabrisas
  muestra el habitáculo, emborronado a grado 3. **Un vidrio sólido es una lente**: su propia cara lejana cierra todas las
  celdas de detrás, así que el valor del campo bajo cielo blanco a lo largo de $-\boldsymbol{\omega}_o$ decide la
  apertura ($\operatorname{smoothstep}(0.5, 0.85)$ de su luminancia) y la parte abierta lee el cielo nítido a lo largo de
  la salida de una esfera estimada a partir de la curvatura media guardada (`lensExit`: doblado al entrar, a través de
  la cuerda, doblado al salir), con el Fresnel y el segundo tinte de la cara lejana (`lensExitThrough`). Llenar la lente
  con el campo dibujaba la cabeza transparente del peón lechosa y plana (0.58 frente a la malla con path trace).
- **Rugosidad a través del vidrio**: una inclinación de microfaceta $\theta$ gira un rayo especular $2\theta$ pero uno
  transmitido mucho menos, así que el cielo a través de ambas caras se lee a
  $r_t = r\sqrt{\operatorname{sat}((\eta - 1)/\sqrt2)}$; la cabeza del peón pasó de un relMSE de 0.081 a 0.048.
- **La reflexión de la cara lejana** devuelve $(1-R)\,R\,(1-R)$ por el tinte de la luz a lo largo del espejo (el color
  de la gaussiana lleva el primer tinte); una bola transparente bajo un cielo uniforme leía 0.952 sin ella, 0.9998 con
  ella (0.92 a través de ambas caras + 0.04 de polish cercano + 0.037 esto).
- **Paredes finas.** Una placa clasificada como fina (aristas abiertas, o un grosor $2V/A$ por debajo de cuatro celdas o
  de un cincuentavo de su propio tamaño) se convierte con opacidad $2R_0/(1+R_0)$, $R_0 = ((\eta-1)/(\eta+1))^2$, y su
  reflexión se escala por la ganancia $g = 1/R_0$ a su propio índice ($R_0$ con un mínimo de $10^{-3}$; 1.5 solo donde
  no se guardó índice), de modo que la mezcla deja la reflexión entera. La tapa del faro del Corvette a 1.15 recibía
  0.0097 de opacidad y, con el antiguo suelo de 1.5, una ganancia de 25 donde 205 la deja entera. Las gaussianas de una
  lámina se dibujan con una opacidad de al menos 0.1 con su color reducido en la misma proporción (y su rampa con él):
  relMSE del parabrisas de 2.26 a 0.89 bajo el taller, de 0.31 a 0.067 bajo goegap.

## 4.8 La reflexión a lo largo de la huella

La investigación midió el error de la pintura como nitidez, no energía: frente al path trace emborronado un píxel su
relMSE bajaba de 0.80 a 0.28. Una gaussiana mostraba la reflexión de su centro plana sobre toda su huella.

### 4.8.1 El giro de la normal por píxel

La conversión escribe el operador de forma del triángulo del centro en los ejes $(\mathbf{u}, \mathbf{v})$ de la
gaussiana, $S = \operatorname{sym}\big([\Delta\mathbf{n}_1\ \Delta\mathbf{n}_2]\,[\Delta\mathbf{p}_1\ \Delta\mathbf{p}_2]^{-1}\big)$
proyectado sobre $(\mathbf{u},\mathbf{v})$ (`m2sShape`). La proyección invierte su propia linealización en el plano del
splat: con $J$ la aplicación $2\times2$ de coordenadas del plano a píxeles, un píxel a la derecha y uno abajo son
$\mathbf{e}_x = J^{-1}(1,0)$, $\mathbf{e}_y = J^{-1}(0,1)$, y la normal a un píxel de distancia es

$$\mathbf{n}_x = \operatorname{normalize}\big(\mathbf{n} + \mathbf{h}_x\big),\quad \mathbf{h}_x = \operatorname{clamp}_{0.5}\!\big((S\,\mathbf{e}_x)_u\,\mathbf{u} + (S\,\mathbf{e}_x)_v\,\mathbf{v}\big),$$

con el giro limitado a media unidad para que un splat visto de canto no barra el cielo.

### 4.8.2 La pendiente, y el lóbulo nítido

Para una reflexión rugosa la proyección vuelve a leer el cielo a lo largo de
$\operatorname{reflect}(-\boldsymbol{\omega}_o, \mathbf{n}_x)$ y $\mathbf{n}_y$ y guarda el cambio del color por píxel,
$\mathbf{g}_x, \mathbf{g}_y$ (`kSlopeMark`). La mezcla dibuja $\max(\mathbf{c} + \mathbf{g}_x d_x + \mathbf{g}_y d_y, 0)$
en el desplazamiento $\mathbf{d}$. La mayor caída de la rampa sobre la elipse de tres sigma se limita al color por canal,
$3\sqrt{\mathbf{g}^\top\Sigma\,\mathbf{g}} \le \mathbf{c}$, de modo que el sombreado nunca llega a cero dentro de la
huella (antes, una línea negra discontinua a lo largo del anillo dorado de un peón). La pendiente de un lado ganó a la
diferencia central y a minmod en el Corvette (pintura 0.393 frente a 0.584 y 0.564 de relMSE).

Una rampa lineal no puede dibujar un punto. Un coat más liso que `kSharpRoughness` (0.2) y un polish de base más liso
que `kSharpPolishRoughness` (0.5) se pasan en cambio a la mezcla (`kSharpMark`): la proyección resta del color la lectura
del centro (cielo y sol analítico), sin recortar, y escribe la coordenada octaédrica del espejo $\mathbf{a}_0$ y sus
pasos $\Delta\mathbf{a}_x, \Delta\mathbf{a}_y$ (puestos a cero a través del pliegue del octaedro), los pesos de los
lóbulos (peso del coat por $O_c$; peso rgb del polish por $O$), su rugosidad y sus pesos de sol. Por píxel la mezcla
decodifica $\mathbf{m}(\mathbf{d}) = \operatorname{octDecode}(\mathbf{a}_0 + \Delta\mathbf{a}_x d_x + \Delta\mathbf{a}_y d_y)$
y suma

$$w_c\,\tilde L(\mathbf{m}(\mathbf{d}), r_c') + \mathbf{w}_p\,\tilde L(\mathbf{m}(\mathbf{d}), r_p') + E_\odot\big(\mathbf{w}_{p\odot}\,p_\odot(\mathbf{m}(\mathbf{d}),\mathbf{s}; r_p') + w_{c\odot}\,p_\odot(\mathbf{m}(\mathbf{d}),\mathbf{s}; r_c')\big),$$

recortando después. Un sol de 1.5 grados reflejado en un coat curvo se dibuja donde cae dentro de la gaussiana y no como
una mancha del tamaño de la gaussiana. Coste: una lectura del cielo por píxel por registro nítido, y la memoria de grupo
de la pendiente en 17 412 de los 17 416 bytes de WebGPU.

### 4.8.3 La dispersión propia del píxel

La lectura nítida es una muestra puntual en el centro del píxel, mientras que la normal gira $\mathbf{h}_x, \mathbf{h}_y$
a través del píxel: una caja de pendientes cuya varianza, doblada para el espejo, ensancha el lóbulo:

$$\alpha'^2 = \alpha^2 + 2\cdot\frac{\lVert\mathbf{h}_x\rVert^2 + \lVert\mathbf{h}_y\rVert^2}{12},$$

con $\alpha' \le 1$, aplicada después del des-ensanche y del ensanche de pliegue de la sección 4.9. El labio enrollado de
un panel ya no centellea. El port del lóbulo nítido al teléfono no tenía este término y dibujaba motas brillantes en los
pliegues de la puerta y del umbral: con él, 27/25 motas pasaron a 2/1 (en el coche de 1280 px el teléfono leyó entonces
la puerta a 35.53 dB y el umbral a 32.36 frente al path trace, desde 33.77 y 31.87 de la versión anterior, que además
tenía el horneado de 640 px).

## 4.9 Deshacer el ensanche del horneado

### 4.9.1 Toksvig en la conversión

Una gaussiana tiene una normal. Donde una tira fina y curva de cromo gira la reflexión a lo largo de una gaussiana, el
píxel con path trace promedia un brillo que se mueve en una línea fina, y una gaussiana dibujaba una cuenta de su propio
tamaño a pleno brillo (el cromo del Corvette leía 1.6 veces el path trace). La conversión ensancha la rugosidad, y la
del coat, por la dispersión de normales bajo la gaussiana (Toksvig):

$$s = \lambda\, \max_{\text{aristas } kj} \frac{\angle(\mathbf{n}_k, \mathbf{n}_j)}{\lVert\mathbf{p}_j - \mathbf{p}_k\rVert}\, \max(\sigma_u, \sigma_v),\qquad \alpha'^2 = \alpha^2 + 2 s^2,$$

con $\lambda$ = `--specular-filter` (1 con `--transfer`) y $r' = \operatorname{sat}(\alpha'^{1/2})$.

### 4.9.2 Lóbulos nítidos sin ensanchar

Los lóbulos nítidos ya giran el espejo por píxel; leídos a la rugosidad ensanchada, el giro se cuenta dos veces. La laca
de una puerta, un espejo ($r_c = 0$), salía mate: guardada a $p_{50} = 0.112$ en la puerta y 0.080 en el capó
(investigación t63b), saturando a 1 en los pliegues. La proyección recupera la dispersión a partir de la curvatura
guardada,

$$s \approx \kappa_{\max}\, \max(\sigma_u, \sigma_v),\qquad \kappa_{\max} = \Big|\frac{S_{uu}+S_{vv}}{2}\Big| + \sqrt{\Big(\frac{S_{uu}-S_{vv}}{2}\Big)^2 + S_{uv}^2},$$

y lee el coat y el polish nítidos a

$$r = \big(r'^4 - 2 s^2\big)^{1/4}\quad (\text{limitado a } 0).$$

### 4.9.3 El coat del pliegue

En un pliegue la laca ensanchada llegaba casi a 1 y una junta dibujaba una raya oscura (juntas de la puerta a 0.69 del
path trace). El coat se limita:

$$r_c \leftarrow \max\big(\operatorname{unwidened}(r_c, s),\ \min(r_c,\ 0.3)\big)\quad\text{si } r_{\text{mat}} \ge 0.3,$$

`kCoatFoldRoughness` = 0.3. Aplicado en todas partes, iluminaba el cromo, el vidrio y los radios de una llanta que el
path trace muestra oscuros (cromo -1.4 dB, radios -0.5 dB por región en el visor web), de ahí la puerta solo para
pintura. A 1280 px el horneado ya no ensancha el coat (sale exactamente a 0), así que en el coche re-horneado este
término no cambia nada.

### 4.9.4 Un tubo gira, un pliegue se ensancha

El giro por píxel extrapola el operador de forma del triángulo del centro; pasada una arista deja de valer. La normal y
la curvatura guardadas son exactas en el centro (normal a $p_{90}$ 0.009 grados de la normal interpolada de la malla),
pero el 64 % de las huellas $\pm2\sigma$ de un capó cruzan una arista, y el error RMS de la normal extrapolada llegaba a
$p_{90}$ 46 y $p_{99}$ 198 grados. Sea $t = 2s$ el giro a lo largo de $\pm2\sigma$. En la nube de 1280 px el cuerpo de un
tubo de cromo está en $t \in [0.3, 1]$, coherente con sus vecinas (2 % de signo opuesto); las normales suavizadas a
través de un pliegue dentro de triángulos más estrechos que la gaussiana están por encima de 1 (de 5 a 9 rad en un
umbral). Con

$$w = \operatorname{smoothstep}(0.7,\ 1.5,\ t),\qquad S \leftarrow (1-w)\,S,\qquad \alpha' = \min\!\Big(\sqrt{\alpha^2 + w\,(t/2)^2},\ 0.5^2\Big)$$

(`kFoldTurnLow` 0.7, `kFoldTurnHigh` 1.5, `foldWidened` limitado a `kFoldRoughness` 0.5 en $r$, nunca más estrecho que
antes), un tubo gira sus píxeles del todo y dibuja su línea fina de luz, mientras que un pliegue se integra como lo
integra el path tracer dentro de un píxel. El smoothstep evita que una cámara en movimiento vea un salto.

## 4.10 Medidas frente al path trace

Todas las cifras del coche son frente al path trace de la malla `simplify3/pt/day_pt_Camera.exr` (`/World/Camera`,
1920 x 1080, `autoshop_01_4k`). «Recorte del coche» es el PSNR sobre el recorte de 900 x 420 del coche, «fotograma» sobre
el fotograma entero; los cocientes por clase son la media de una región sobre la del path trace (investigación
`proto/door/*.py`).

### 4.10.1 La puerta TX, por material (relMSE)

| estado | cromo | metal rugoso | pintura |
|---|---|---|---|
| primer transfer | 0.536 | 0.0896 | 0.705 |
| TX, celdas y campo en todo | 1.64 | 0.221 | 0.393 |
| polish abierto por armónicos, cono ≥ celda, campo solo en el coat | 0.591 | 0.0835 | 0.341 |
| después: + coat nítido y polish nítido por píxel | 0.545 | 0.083 | 0.281 |

### 4.10.2 Coste de reproducción

El Corvette TX entero (14.7 M de gaussianas, sin suelo, 1920 x 1080) se dibujaba en 158 ms frente a unos 83 de una nube
reiluminada de su tamaño; de los 84.5 ms de la proyección, el campo y su acoplamiento se llevaban 24, las celdas leídas
por los lóbulos 16.5, la parte del sol y su rebote 12, el transfer del cuerpo 7.5. Leer el campo una sola vez para
ambos lóbulos y guardar en caché los términos que no dependen del ojo atacan el primero y el último.

### 4.10.3 El coche entero, T63-T75

| prueba | qué | recorte del coche (dB) | fotograma (dB) |
|---|---|---|---|
| T63b | hojas a 640 px (5.38 M), antes de los pasos de esta sección | 16.37 | 21.92 |
| T64 | hojas, lóbulos nítidos sin ensanchar (b26d276) | 16.58 | 22.15 |
| T63c | corte a 0.25 px con culling de caras traseras | 16.29 | -- |
| T65 cut_a | + sin ensanchar + coat limitado en pliegues | 16.51 | 22.15 |
| T65 cut_b | + campo del polish en materiales rugosos (35ae3cb) | 16.55 | 22.18 |
| T66 | hojas (corte a 0.01 px), sin cull / con cull | 16.73 / 16.75 | 22.40 / 22.43 |
| T69 / T70 | + giro limitado a 0.2 rad / + coat de pliegue solo en pintura | 16.57 / 16.57 | 22.20 / 22.20 |
| T71 | **horneado a 1280 px**, corte a 0.25 px, culling de caras traseras (5.0 M) | **17.88** | **23.90** |
| T72 | T71 sin el límite | 17.88 | 23.91 |
| T73 | T71 + polish nítido con puerta en la rugosidad sin ensanchar | 17.88 | 23.90 |
| T74 | T73 + separación tubo/pliegue | 17.90 | 23.93 |
| T75 | T71 + separación tubo/pliegue (la que queda) | 17.88 | 23.91 |

El PSNR sobre un recorte se mueve poco; los cocientes por clase de abajo muestran dónde actúan los cambios.

### 4.10.4 La puerta y el capó, por clase (investigación t63b-t66)

La pintura es una base de metal oscuro (rugosidad 0.3421) bajo un coat de rugosidad 0 a 1.45. Cada píxel se clasifica
por dónde cae su espejo: el suelo (cerrado; responde el campo) o el HDRI (abierto). En t63b, el 86 % del déficit de la
puerta estaba en píxeles que reflejan suelo. Media sobre el path trace:

| clase de la puerta | t63b hojas | t63c corte+cull | t65 cut_a | t65 cut_b | t66 sin cull | t66 con cull |
|---|---|---|---|---|---|---|
| espejo hacia el suelo < 3 m | 0.78 | 0.89 | 0.92 | **1.02** | 0.90 | 1.02 |
| espejo hacia el suelo 3-30 m | 0.69 | 0.77 | 0.76 | 0.88 | 0.79 | 0.88 |
| espejo hacia el HDRI (abierto) | 0.92 | 1.04 | 1.03 | 1.13 | 1.01 | 1.13 |
| juntas | 0.69 | 0.83 | 0.95 | **1.04** | 0.90 | 1.02 |

Deshacer el ensanche (t64) dejaba las clases de suelo en 0.78/0.67 y las juntas en 0.69; el campo del polish (cut_b)
subió el suelo cercano a 1.02 y el coat de pliegue las juntas a 1.04, como predecía una estimación analítica del
término que faltaba (bajo-coat 0.94 x albedo del polish 0.078 x suelo reflejado 0.11 = 0.008 frente a un déficit medido
de 0.0056-0.0085). El culling de caras traseras sube toda la puerta un 12 % aproximadamente (t66). El 1.13 que queda en
la clase abierta se reparte según si el espejo encuentra el propio coche: el 68 % del exceso está en píxeles «coche»
(2.04 del path trace), donde las celdas de 16 x 16, de unos 11 grados, dejan un 25 % de cielo abierto detrás de un
retrovisor o un pilar.

Contraste local (desviación típica del paso alto sobre la media, $\sigma$ = 1.5 / 4 px), relativo al path trace:

| | t63b | t64 | t65 cut_b | T71 (1280 px) |
|---|---|---|---|---|
| puerta, reflejos abiertos | 0.12 / 0.26 | 0.40 / 0.61 | 0.35 / 0.53 | 0.56 / 0.74 |
| capó, reflejos abiertos | 0.27 / 0.38 | 0.41 / 0.52 | 0.41 / 0.51 | -- |

Una simulación de la reflexión del coat con el espejo del path trace sobre el HDRI prefiltrado mostró que la rugosidad
guardada del coat explica por sí sola el emborronado: con $r_c$ = 0.11 la puerta conserva 0.24 / 0.31 del contraste de
$r_c$ = 0, con 0.08 el capó 0.21 / 0.28.

### 4.10.5 Los brillos del capó

Teselas de 24 px con brillo (168), alineadas por correlación del paso alto dentro de $\pm$6 px:

| render | desplazamiento $p_{50}$ | $\sigma$ que iguala | pico (PT 5.8) |
|---|---|---|---|
| t63b (ensanchado) | 4.1 px | 2.5 | 1.9 |
| t68 sin giro | 4.1 px | 1.5 | 2.1 |
| t68 sin pendiente | 4.2 px | 2.5 | 1.8 |
| t64 / t66 (hojas) | 3.2 px | 1.5 | 3.0 |
| t65 cut_b | 3.2 px | 1.5 | 2.6 |
| T71 (1280 px) | 2.0 px | 0.75 | 3.3 |

La media del capó pasó de 0.89 a 0.98 del path trace a 1280 px. El error de extrapolación de la normal escala con el
cuadrado de la celda (mediana de 0.146 grados a 640 px, 0.060 a 960, 0.032 a 1280).

### 4.10.6 La normal extrapolada, y la separación tubo/pliegue

Error RMS de la normal extrapolada en el capó (grados):

| variante | $p_{50}$ | $p_{90}$ | $p_{99}$ |
|---|---|---|---|
| sin giro | 0.575 | 22.7 | 35.8 |
| operador de forma del centro | 0.146 | 46 | 198 |
| + límite $\lVert S\rVert\,2\sigma \le 0.2$ rad | 0.147 | 20.2 | 32.0 |
| mínimos cuadrados con muestras de la malla | 0.088 | 9.8 | 14.6 |

El límite (T69) movió el recorte del coche de 16.55 a 16.57 dB, pero empeoró las juntas de la llanta delantera (1.48 a
1.54); T72 mostró que era irrelevante a 1280 px. La separación tubo/pliegue lo sustituyó:

| | T71 | T73 (puerta) | T74 (puerta + separación) | T75 (separación) |
|---|---|---|---|---|
| motas brillantes en el pliegue del umbral | 10 | 11 | 1 | 1 |
| tira cromada, RMS | 0.0358 | 0.0535 | 0.0415 | 0.0359 |
| tira cromada, paso alto / PT | 0.41 | 1.12 | 0.58 | 0.41 |

T75 llevó además las motas de la puerta de 405 a 389 y las de la llanta de 204 a 190, sin cambiar fotograma ni recorte.
A 1280 px el horneado ya no ensanchaba el coat, y las gaussianas del pliegue del umbral reflejaban como espejos con
normales que giraban, rayas brillantes sobre la ranura oscura del path trace; la separación las quita sin tocar la tira.
Poner en cambio una puerta al polish nítido según la rugosidad sin ensanchar (T73) triplicó el error de paso alto de la
tira y se descartó, igual que `--specular-filter 0` (una línea oscura discontinua en el pliegue del umbral, cromo que
centellea): el polish necesita ahí su ensanche.

<!-- Figuras:
/Users/muriel/luc/athenea-renders/pieces/car/t63b/research_coat_roughness.png  (rugosidad guardada del coat, horneado de 640 px)
/Users/muriel/luc/athenea-renders/pieces/car/t63b/mos.png                      (path trace frente a hojas frente a corte)
/Users/muriel/luc/athenea-renders/pieces/car/t65/mos.png                       (cut_a / cut_b frente al path trace)
/Users/muriel/luc/athenea-renders/pieces/car/t66/research_open_split.png       (clase abierta de la puerta: coche / bits / cielo)
/Users/muriel/luc/athenea-renders/pieces/car/t71/mos.png                       (horneado de 1280 px frente al path trace)
/Users/muriel/luc/athenea-renders/pieces/car/t71/research_sill.png             (rayas en el pliegue del umbral)
/Users/muriel/luc/athenea-renders/pieces/car/t73/bump_mos.png                  (polish nítido con puerta en la tira cromada)
/Users/muriel/luc/athenea-renders/pieces/car/t74/sill_mos.png                  (separación tubo/pliegue en el umbral)
-->

## 4.11 Limitaciones

- **Campo de rango uno.** El acoplamiento pone el color de un suelo iluminado, no su forma, en una reflexión bajo un
  cielo iluminado por un lado; a grado 3 el campo resuelve un horizonte a unos veinte grados y mezcla la sombra del
  coche en el suelo con el suelo lejano (en simulación: correlación 0.56 a grado 3, 0.63 a grado 5, 0.85 con un mapa
  octaédrico de 32 x 32).
- **Resolución de las celdas.** Celdas de 16 x 16, de unos 11 grados, dejan pasar cielo donde un coat nítido refleja el
  propio coche (la clase abierta de la puerta a 1.13); celdas de 32 x 32 (+96 B por gaussiana) lo reducirían a la
  mitad.
- **Un sol por domo**; las lámparas dentro de una escena se sombrean con bits que dicen qué direcciones salen de la
  escena, no cuáles llegan a la lámpara. El horneado no tiene emisión de lámparas: los faros salen oscuros frente al
  path trace.
- **Defectos que quedan** en el coche de 1280 px: la tira cromada del parachoques delantero un 38 % más clara que el
  path trace y con el pico más bajo; la reflexión del suelo cercano en la llanta a 0.72; los brillos del capó desplazados
  2.0 px, que una curvatura ajustada sobre la huella a partir de la malla reduciría más o menos a la mitad.
- **Los transfers zonales** guardan solo la mitad directa, y conocen solo la pose que se trazó; la oclusión entre
  miembros en otras poses no se multiplica.
