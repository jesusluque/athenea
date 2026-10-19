<!-- Copyright (c) 2026 jesus luque. -->

# 4. Transporte de luz en el bake de las Gaussians (transfer) y relight

Una nube convertida desde un mesh puede llevar su luz de dos maneras. Un *bake de radiancia* guarda, por Gaussian, la
luz que llegó bajo la iluminación del propio stage; es exacto para esa iluminación y erróneo bajo cualquier otra. Un
*transfer* guarda en cambio cómo el entorno de cada Gaussian le llevaría un entorno arbitrario, y un frame combina esa
geometría con el cielo y las luces que tenga. Esta sección describe el núcleo del trabajo, el baker: el transfer que el path
tracer de la escena calcula en el bake de cada Gaussian, y el relight que lo evalúa al dibujar, en su forma TX (un transfer que vale bajo cualquier cielo y cualquier luz), el modelo de reflexión
construido encima, el tratamiento por píxel de los sharp lobes, y lo que se midió que aporta cada pieza frente a un path
trace del mesh de origen. Todo lo descrito corre en la GPU: el bake es el path tracer de la propia escena en un modo
transfer, y la evaluación son los kernels de proyección y blend de splats (`shaders/athenea/splat/splat_relight.slang`,
`splat_project.slang`, `splat_blend.slang`).

## 4.1 Notación

Una Gaussian tiene una posición $\mathbf{p}$, una normal unitaria $\mathbf{n}$ (su eje más corto, o una normal de
shading guardada, girada hacia el ojo), un albedo $\rho$, una metalicidad $m$, una roughness perceptual $r$ con
parámetro GGX $\alpha = r^2$ y, opcionalmente, un clear coat (peso $w_c$, roughness $r_c$, índice $\eta_c$), un sheen y
una transmisión. $\boldsymbol{\omega}_o$ es la dirección hacia el ojo y $\mathbf{m} = \operatorname{reflect}(-\boldsymbol{\omega}_o,
\mathbf{n})$ la dirección especular. $Y_k$ son los spherical harmonics reales, con índice $k = l^2 + l + m$; el grado 2
tiene 9 coeficientes y el grado 3 tiene 16. Un cielo tiene coeficientes de radiancia $L_k$ (rgb), un sol de dirección
$\mathbf{s}$ e irradiancia $E_\odot$ extraído de él, y un prefiltered map $\tilde L(\boldsymbol{\omega}, r)$.

## 4.2 Qué guarda una Gaussian

El transfer es una tira de valores por Gaussian, f16 en el device. No hay token de modo: la cuenta es el formato
(`common/transfer_layout.slang`) (figura 11).

| valores | formato | origen |
|---|---|---|
| 9 | grado 2, directa | primer transfer |
| 36 | grado 2, directa + indirecta (27) | primer transfer |
| 84 | grado 2, ambas + reflected field (48) | TX con `--transfer-degree 2` |
| 16 | grado 3, directa | |
| 64 | grado 3, directa + indirecta (48) | |
| 112 | grado 3, ambas + reflected field (48) | **TX por defecto** |
| 10 | dos lobes zonales en el marco de la Gaussian | nubes con skinning |

![Figura 11. Qué guarda el bake por Gaussian y cómo lo combina el dibujado con el entorno: el body a partir del SH directo e indirecto, polish y clear coat a partir del reflected field y de las cells de open directions, y el sharp lobe evaluado por píxel.](figures/fig-transfer.es.svg)

Junto a ella:

- **Open directions** (`primvars:athenea:splat:shadowBits`): 2 palabras (64 bits, un grid octaédrico de 8 x 8;
  primer transfer), 8 palabras (16 x 16 cells sobre la esfera entera; TX por defecto) o 32 palabras (32 x 32). Un bit se
  pone donde el rayo del bake en esa cell salió de la escena.
- **Curvatura** (`primvars:athenea:splat:curvature`): el shape operator en los dos ejes del plano de la Gaussian,
  $(S_{uu}, S_{uv}, S_{vv})$, tres floats, escrita con `--transfer`.
- **Roughness del material** (`primvars:athenea:splat:materialRoughness`): la roughness propia del material antes del
  ensanche del bake (sección 4.9), -1 bajo un mapa de textura. En el device va en cinco bits (27-31) de la palabra
  pbr, marcada por el bit 26.

La direct half es un escalar por coeficiente (solo geometría); la indirecta es rgb, coloreada por aquello en lo que
rebotó la luz; el reflected field es rgb. El formato TX cuesta, por Gaussian:

| | primer transfer | TX |
|---|---|---|
| directa | 9 | 16 |
| indirecta | 27 | 48 |
| reflected field | -- | 48 |
| visibilidad | 2 ints (64 bits) | 8 ints (256 bits) |
| fichero, bytes | 152 | 480 |
| device, bytes | 80 | 256 |

Para el Corvette entero (unos 10 M de Gaussians entonces) son 4.8 GB de fichero y 2.6 GB de device solo para el
transfer; los códecs que reducen su almacenamiento y su ancho de banda (un PCA por clusters del transfer, un codebook k-means de Hamming de los
bits de sombra) se describen en la sección 7, donde sirven a una aplicación, un player WebGPU.

## 4.3 El entorno en el momento de dibujar

`technique::Environment` prepara cada domo en tres formas, todas buffers leídos a través de un único módulo
(`light/environment.slang`) por el rasterizer y por el ray tracer.

**Spherical harmonics de radiancia.** `env_project` suma los texels de un nivel de la cadena de mips del domo, cada uno
pesado por el ángulo sólido lat-long $2\pi^2 \sin\theta\,\Delta u\,\Delta v$ (la medida que usa el muestreo por
importancia del domo):

$$L_k = \sum_{\text{texels}} L(\boldsymbol{\omega}_t)\, Y_k(\boldsymbol{\omega}_t)\, 2\pi^2 \sin\theta_t\, \Delta u\, \Delta v .$$

Se proyectan dieciséis coeficientes (grado 3); la irradiancia de un cuerpo sin transfer lee los nueve primeros,
convolucionados al leer con los pesos zonales del coseno truncado $\{\pi, 2\pi/3, \pi/4\}$. Un cielo constante vuelve a
dar $\pi L$ de irradiancia en toda normal: 6.2841 frente a 6.2832 (0.089 %) sobre 64 direcciones.

**El sol, sacado.** Un disco mil veces más brillante que el cielo no es una función de grado 2; proyectado, se emborrona
en un lobe de sesenta grados. `env_sun` encuentra la región más brillante cuya luz está tan concentrada como un disco
de su ángulo sólido (ángulo cuadrático medio pesado por la luz alrededor de su centro como mucho el doble que el de un
disco, $\theta_{\text{cone}}^2/2$), saca sus texels tanto de los harmonics como de la fuente del prefiltrado, y devuelve
una dirección $\mathbf{s}$, una irradiancia $E_\odot$ y un semiángulo. La irradiancia se suma exactamente sobre los
texels que los harmonics omiten, al mismo nivel y con la misma medida, de modo que lo que sale de los coeficientes llega
como luz (forma cerrada 0.1119 frente a 0.1113).

**El prefiltered map.** El cielo sin su sol, convolucionado con un lobe GGX en una cadena de mips octaédrica de ocho
niveles, el nivel $k$ para la roughness $(k/7)^2$ (0, 0.020, 0.082, 0.184, 0.327, 0.510, 0.735, 1). El lado de la base
iguala el texel del propio domo, $S = \sqrt{2WH/\pi}$, redondeado hacia abajo a una potencia de dos en $[256, 2048]$. El
prefiltrado usa filtered importance sampling (Krivanek y Colbert): cada muestra GGX lee la fuente un nivel más
basto de lo que pide su propio ángulo sólido, nunca más estrecho que el texel que escribe, con la secuencia girada por
un hash por texel, 64 muestras en el nivel 1 que se doblan hasta 1024. Los mips lat-long de la fuente pesan cada fila
por $\cos\theta_0 - \cos\theta_1$, de modo que cada nivel conserva la media por ángulo sólido del cielo. Sin esto, un sol
de 1.5 grados en un nivel rugoso era una dispersión de manchas del tamaño de un texel (la banda de 4-16 px de una bola
de pintura a 2.13 veces el path trace).

**Una nube que la escena gira.** Todo lo que una Gaussian guarda de su transfer (§4.2) -- los harmonics de las dos
mitades, el reflected field y los bits de open directions -- es un conjunto de direcciones de la nube, medidas en el
sistema de referencia del fichero en el que se hizo su bake. Un prim que gira la nube (un asset Y-up colocado en una
escena Z-up, una pieza girada a su sitio) no cambia ninguna. Por eso el dibujo lee el transfer en el sistema de la nube.
Sea $R$ la rotación de la transformación de la instancia, su parte lineal con las columnas hechas unitarias. Cada
dirección del mundo por la que pregunta el dibujo -- el espejo de la vista, el camino a través de un vidrio, el sol, una
luz, el cono de celdas de un lobe -- se gira por $R^\top$ antes de consultarla, y los dieciséis coeficientes del cielo
se giran al mismo sistema una vez por instancia y frame,

$$L'_k = \int_{S^2} Y_k(\boldsymbol{\omega})\, L(R\,\boldsymbol{\omega})\, d\boldsymbol{\omega},$$

con una regla producto exacta para los productos de grado 6 que integra (cuatro nodos de Gauss–Legendre en $\cos\theta$
por ocho pasos iguales en $\varphi$). El cuerpo hace entonces el producto del transfer con $L'_k$; lo que se evalúa en
el mundo (el prefiltered map, el fondo) conserva $L_k$. Un transfer zonal (§4.4.5) ya gira con cada Gaussian y se deja
como está. Con el peón de §5.11 girado 60 grados alrededor de su eje vertical junto con su cámara y su cielo, el frame con
relight coincide con el del peón sin girar con relMSE $9\times10^{-6}$ (61 dB) a 768 × 768; girado al revés, 11 dB.
Leído en cambio en el sistema del mundo, el peón levantado de Y-up a una escena Z-up dibujaba su cabeza de vidrio en
fragmentos (figura 12).

![Figura 12. La cabeza de vidrio del peón en una escena Z-up, con sus nubes levantadas por un cuarto de vuelta. Izquierda: el transfer leído en el sistema del mundo. Centro: leído en el sistema de la nube con el cielo girado a él. Derecha: la nube sin girar en su propio stage Y-up, cuyo cielo queda en otro acimut.](figures/img-turned-glass.jpg)

## 4.4 El bake

### 4.4.1 La integral de transfer y su estimador

El transfer es la proyección de la visibilidad por el coseno truncado, con el albedo tomado como uno:

$$T_k = \frac{1}{\pi} \int_{S^2} V(\boldsymbol{\omega})\, \max(\mathbf{n}\cdot\boldsymbol{\omega}, 0)\, Y_k(\boldsymbol{\omega})\, d\boldsymbol{\omega},$$

y un frame lo recombina con el cielo como $\langle T, L \rangle = \sum_k T_k L_k$. Lo recoge una tercera variante del
kernel de path tracing, `kTransfer`, junto al bake de radiancia y al camino de cámara. En el primer vértice de una
Gaussian el material se sustituye por la propia dirección muestreada: las direcciones están estratificadas (un grid
$\sqrt{N}\times\sqrt{N}$ con jitter) y son uniformes sobre el hemisferio, densidad $1/2\pi$, frente a un estimador que
quiere $\cos\theta/\pi$, así que una muestra que escapa vale $2\cos\theta\, Y_k(\boldsymbol{\omega})$ y una bloqueada
nada. Uniforme en lugar de coseno, y sobre el hemisferio medido como una esfera con la mitad lejana vacía, porque los
harmonics son ortonormales solo sobre la esfera; ajustados coeficiente a coeficiente sobre un hemisferio se pasan (el
peón salía diez veces demasiado brillante). La acumulación *es* la proyección; no se resuelve ningún ajuste.

Los mismos paths dan dos mitades sin ningún rayo más: un camino que escapa justo después del primer vértice contribuye
a la direct half (escalar); uno que escapa tras rebotar lleva el color de lo que encontró y contribuye a la mitad
indirecta $T^{\text{ind}}_k$ (rgb). Comprobado frente a formas cerradas: un punto sin oclusión transfiere el coseno
truncado, $(0.2820, 0.3258, 0.1581)$ frente al exacto $(0.2821, 0.3257, 0.1577)$ para las bandas 0-2, y un cielo
constante de 0.7 lee 0.6998. Una esquina de suelo y pared, transfer recombinado frente a un bake de radiancia de los
mismos puntos:

| | transfer | bake de radiancia | peor punto |
|---|---|---|---|
| sin pared, 1 bounce | 0.6000 | 0.6000 | 0.0 % |
| pared negra, 1 bounce | 0.3091 | 0.3039 | 2.5 % |
| pared blanca, 3 bounces | 0.5095 | 0.5325 | 4.3 % |

### 4.4.2 Open directions y la direct half por cuadratura

En la primera muestra de cada punto TX traza un rayo any-hit por cell de un grid octaédrico de $S_c \times S_c$
($S_c = 16$ o 32) sobre la esfera *entera*: también la mitad lejana, empezando desde la cara de atrás de la superficie,
de modo que una dirección detrás de un sólido encuentra el sólido y una detrás de una lámina sale. El bit de una cell
se pone donde pasa al menos la mitad de la luz: el rayo de oclusión atraviesa hasta ocho superficies cuyo material
transmite, pesadas por la transmisión por la luminancia del color menos el Fresnel a ese ángulo (una novena cierra el
paso). Las cells se trazan desde cuatro orígenes sobre el footprint de la Gaussian (la cell $c$ desde el origen
$(c + \lfloor c/S_c \rfloor) \bmod 4$: el centro y tres puntos llevados a la superficie dentro del disco de la
Gaussian), de modo que una ranura más estrecha que una Gaussian ya no se lee como una línea discontinua de sombras
enteras.

Los mismos rayos dan entonces la direct half por cuadratura (sin grano), cada cell pesada por su ángulo sólido
octaédrico:

$$T_k^{\text{dir}} = \sum_{c} \tau_c\, \frac{\max(\mathbf{n}\cdot\boldsymbol{\omega}_c,0)}{\pi}\, \Delta\Omega_c\, Y_k(\boldsymbol{\omega}_c),
\qquad \Delta\Omega_c = \frac{4}{S_c^2}\,\lVert\boldsymbol{\omega}_c\rVert_1^3,$$

con $\tau_c$ la fracción transmitida a lo largo del rayo de la cell. Las palabras se escriben de cuatro en cuatro a
medida que se trazan cada 128 cells, de modo que el kernel no guarda ningún array del tamaño del grid.

### 4.4.3 El reflected field

Una reflexión necesita la luz rebotada *por dirección*, no integrada. Para cada camino que sale de la Gaussian,
encuentra la escena y luego escapa, el throughput pasado el primer vértice es la radiancia que llega por su primera
dirección $\boldsymbol{\omega}_1$ bajo un cielo blanco de radiancia uno. Se proyecta sobre el grado 3 de
$\boldsymbol{\omega}_1$, uniformemente sobre el dominio del que se sacó:

$$F_j \approx \frac{1}{N} \sum_{i:\ \text{escapó tras un bounce}} \Omega_{\text{meas}}\, \frac{\beta_i}{w_i}\, Y_j(\boldsymbol{\omega}_{1,i}),$$

con $\beta_i$ el throughput, $w_i$ el peso del primer vértice y $\Omega_{\text{meas}} = 2\pi$ (o $4\pi$ sobre la esfera).
Son 16 coeficientes rgb, `primvars:athenea:splat:transferReflected`. No cuesta ningún rayo. Como $F$ es cero en las
open directions, dieciséis harmonics leen una dirección cerrada junto a una abierta a una fracción de lo que hay
allí (la parte cerrada de la pintura era 0.0014 de 0.115). Así que el bake proyecta también el indicador de las
direcciones cerradas $C_j$ sobre las mismas muestras, y la conversión divide ambos, una convolución normalizada que da
la radiancia media de las direcciones cerradas alrededor de cada dirección, cae a la media de las direcciones cerradas
donde su proporción está por debajo de 0.1 (mezclada hasta 0.4), y vuelve a proyectar el resultado sobre un grid de
16 x 16 por el ángulo sólido de cada cell (`m2sFieldOverClosed`).

Para una Gaussian que transmite, la primera dirección se saca sobre la esfera entera (`bakeSphereDirection`: la altura
del estrato estirada a $[-1,1]$ y la mitad inferior reflejada en el plano tangente). Las muestras de delante conservan
el estimador del transfer ($4\cos\theta$ con densidad $1/4\pi$); las de detrás alimentan solo el campo, tras doblarse
hacia dentro de un dieléctrico sólido por su índice (`refractInto`), de modo que el campo a lo largo de
$-\boldsymbol{\omega}_o$ contiene lo que el rayo del ojo encuentra a través de ambas caras. Sin doblar, una bola de
vidrio validaba a la mitad del path trace (0.265 frente a 0.541 bajo un domo blanco).

### 4.4.4 Rebanadas, filtrado y el bake partido

El Corvette TX entero pedía la respuesta de su bake de una vez: 35 float4 por Gaussian, 7.8 GB, con 5 GB libres.
`Converter::transfer` hace por tanto el bake en rebanadas cuya respuesta es como mucho 1.5 GB (nunca por debajo del lote
propio del bake, $2^{19}$), y como mucho $2^{20}$ Gaussians donde corre el filtro. Los rayos de cada rebanada se copian
en el device, pasan por el bake (`StageRenderer::bakePointsOnDevice` en modo transfer), se filtran sus mitades
rebotadas, se escriben con `m2sTransferInto` y se leen de vuelta a los arrays del fichero antes de la siguiente. El bake
de radiancia usa el emparentado `StageRenderer::bakeSplitOnDevice`, que guarda las mitades directa e indirecta como
sumas separadas para que la indirecta pueda filtrarse antes de que `combineBake` ajuste ambas. El transfer sigue el
mismo principio: su indirect half y su campo (los valores más ruidosos, a 64 paths por Gaussian en los bakes del
Corvette) pasan por el filtro de bake de splats, un filtro à-trous de `--bake-filter` iteraciones (3), cuyos pesos son
la posición en el plano tangente, la orientación y el id de prim; la direct half, que contiene las sombras, no se
filtra nunca.

### 4.4.5 Lobes zonales para nubes con skinning

Nueve harmonics están en el mundo; un esqueleto gira la Gaussian debajo de ellos. Rotarlos cuesta una 9 x 9 por
Gaussian. Un lobe zonal rota rotando su eje, así que una nube con skinning guarda dos lobes, cada uno con el
$(u, v)$ octaédrico de su eje en el marco propio de la Gaussian y los coeficientes zonales $z_0, z_1, z_2$: diez valores
(`primvars:athenea:splat:transferZonal`). Como harmonics, un lobe alrededor de $\mathbf{a}$ es
$g_{lm} = z_l\, k_l\, Y_{lm}(\mathbf{a})$ con $k_l = \sqrt{4\pi/(2l+1)}$. Para un eje fijo los mejores coeficientes son la
proyección $z_l = k_l \sum_m f_{lm} Y_{lm}(\mathbf{a})$, que deja un error $\lVert f\rVert^2 - \sum_l z_l^2$, así que el
ajuste (un efecto AOFX, `splattransferzonal`) busca el eje en el device (la dirección de la banda 1, la normal, 64
direcciones de Fibonacci y luego un paseo que divide el paso a la mitad), ajusta el segundo lobe al residuo y reajusta
el par dos veces (Sloan 2008). Al dibujar, `splatTransferFrame` gira cada eje con el marco deformado y reconstruye nueve
harmonics del mundo, que todo lo de después lee sin cambios. El bake posa la nube en `--time` con el propio skinner
del motor, de modo que el marco contra el que se ajustan los lobes es el cuaternión empaquetado que lee un renderer.
Los bits de sombra siguen siendo dos palabras, reagrupados en el marco de la Gaussian. La indirect half no se guarda
zonal.

## 4.5 El cuerpo

Para una Gaussian con transfer, la parte que no depende del ojo es

$$\mathbf{B} = \sum_{k<16} L_k\,\big(T_k^{\text{dir}} + T_k^{\text{ind}}\big) \;+\; E_\odot\,\frac{\max(\mathbf{n}\cdot\mathbf{s},0)}{\pi}\, O(\mathbf{s}) \;+\; E_\odot\, T^{\text{ind}}(\mathbf{s}),
\qquad T^{\text{ind}}(\mathbf{s}) = \max\Big(\sum_k T_k^{\text{ind}} Y_k(\mathbf{s}), 0\Big),$$

siendo el último término el bounce del sol, que el primer transfer perdió con el sol que sacó de los harmonics.
$O(\mathbf{s})$ es la parte del sol que llega: para TX la lectura bilineal de las cuatro cells alrededor de
$\mathbf{s}$, dejando fuera las cells detrás de la normal; para los 64 bits lo mismo en un grid de 8 x 8; sin bits, el
cociente de grado 2

$$O_{\text{SH}}(\mathbf{s}) = \frac{\sum_k T_k Y_k(\mathbf{s})}{\sum_k \hat A_l\, Y_k(\mathbf{n})\, Y_k(\mathbf{s})}, \qquad \hat A_l = \{1, 2/3, 1/4\},$$

con numerador y denominador compartiendo el truncamiento, de modo que lee uno donde nada ocluye. El cuerpo difuso es
entonces $U\,\rho\,(1-m)\,\big[(1-t)\,\mathbf{B} + t\,(\ldots)\big]$ con $U$ el throughput de las capas (sección 4.6.4) y
$t$ la transmisión. $\mathbf{B}$, la parte del sol y el acoplamiento del campo (abajo) se guardan por Gaussian mientras
el cielo y las luces no cambian (`transferViewless`, el `txCache`); un frame calcula solo lo que cambia el ojo.

El término constante del transfer es además una oclusión ambiental: sobre un hemisferio abierto
$T_0 = A_0 Y_0 / \pi = 0.28209$, así que $o = \operatorname{sat}(T_0 / 0.28209)$ es la fracción del cielo que llega a la
Gaussian.

**Luces que no son el cielo.** Una luz distante, esférica, de disco o rectangular se evalúa con una muestra en su centro
a través de los mismos lobes. Con las cells TX, su sombra son las cells leídas sobre el cono que la luz subtiende
($\tan^{-1}$ de su radio sobre la distancia, o la mitad del diámetro angular de una luz distante), la más oscura entre
esa y cualquier sombra medida con rayos; su bounce es $T^{\text{ind}}$ leído a lo largo de ella, por el $L/p$ de la
muestra, sobre el cuerpo y el sheen, y acopla el reflected field igual que el sol (`splatLightBounce`).

## 4.6 Reflexión

### 4.6.1 Qué parte de un lobe ve el cielo

El primer transfer estrechaba toda reflexión con el ajuste de Lagarde y de Rousiers del término constante,

$$O_{\text{L}}(\mathbf{n}\cdot\boldsymbol{\omega}_o, o, r) = \operatorname{sat}\!\Big( (\mathbf{n}\cdot\boldsymbol{\omega}_o + o)^{2^{-16r-1}} - 1 + o \Big),$$

un número para todo el hemisferio: una puerta reflejaba el cielo abierto donde el suelo está en su espejo, y las llantas
del Corvette leían 2.4 veces el path trace del mesh, `Metal_rough` 6.5 veces, los discos de freno cien. Con las cells
TX se pregunta al propio lobe (`splatLobeOpen`, `splatConeOpen`): la lectura de cells $o_c(\cdot)$ (bilineal sobre
cuatro cells, excluidas las que quedan detrás de $\mathbf{n}$) en el espejo y en un anillo de seis direcciones a su
alrededor con ángulo

$$\theta_r = \max\!\big(\min(1.4\,\tan^{-1}\alpha,\ 1.2),\ 2\cdot\tfrac{\pi}{2S_c}\big),$$

más o menos el ángulo dentro del cual GGX refleja la mitad de su energía, nunca más estrecho que una cell; y

$$O(\mathbf{m}, \alpha) = \frac{\tfrac13 o_c(\mathbf{m}) + \tfrac19 \sum_{k=1}^{6} o_c(\mathbf{d}_k)}{\tfrac13 + \tfrac69}$$

sobre las lecturas que pesaron algo. Un cono dentro de una cell leía el único bit de esa cell y salpicaba la pintura
del Corvette (relMSE 0.393 frente al path trace; 0.298 con el cono limitado a una cell).

### 4.6.2 Qué muestra la parte cerrada: el campo acoplado al cielo

El campo se calculó en el bake bajo un cielo blanco. Bajo uno real su patrón se mantiene y su brillo se escala por
canal según cuánto más rebota el entorno bajo ese cielo que bajo el blanco (`splatFieldCoupling`):

$$\kappa = \frac{\sum_k T_k^{\text{ind}} L_k + E_\odot\, T^{\text{ind}}(\mathbf{s})}{\sqrt{4\pi}\; T_0^{\text{ind}}},$$

cayendo a la radiancia media del cielo donde la Gaussian no rebota nada. El acoplamiento es de rango uno: exacto para
una Gaussian rodeada por una superficie iluminada por un cielo, y pone el color de un suelo iluminado, no su forma, bajo
un cielo iluminado por un lado. Leído a lo largo de una dirección a la roughness de un lobe, las bandas del campo se
estrechan como un GGX reflejado tomado como una von Mises-Fisher de concentración $1/2\alpha^2$:

$$F(\mathbf{d}; \alpha) = \max\Big(\sum_{j<16} F_j\, e^{-l_j(l_j+1)\alpha^2}\, Y_j(\mathbf{d}),\ 0\Big).$$

Las dos roughness que necesita una Gaussian (polish y coat) se sacan de una sola lectura de los 48 valores
(`splatFieldAlongPair`). Las alternativas consideradas fueron una matriz de transfer de 9 x 9 por Gaussian (243
números, 5 GB para el coche, y aún grado 2) y un índice por dirección dentro de la nube (necesita una caché de radiancia
y no ve nada de un suelo de mesh).

### 4.6.3 Polish y coat

La reflexión de la base (polish) y el coat son

$$\mathbf{R}_{\text{polish}} = U\,g\,P(\mathbf{n}\cdot\boldsymbol{\omega}_o)\,\big[O\,\tilde L(\mathbf{m}, r) + (1-O)\,\chi\,\kappa\,F(\mathbf{m};\alpha)\big] + U\,g\,P\,E_\odot\, O(\mathbf{s})\, p_\odot(\mathbf{m}, \mathbf{s}),$$
$$\mathbf{R}_{\text{coat}} = w_c E_c\,\big[O_c\,\tilde L(\mathbf{m}, r_c) + (1-O_c)\,\kappa\,F(\mathbf{m};\alpha_c)\big] + w_c E_c\, E_\odot\, O(\mathbf{s})\, p_\odot(\mathbf{m},\mathbf{s}),$$

donde $P$ es el albedo de entorno del polish, $g$ la ganancia de pared fina (sección 4.7), $E_c$ el albedo del coat y
$\chi$ la puerta del campo del polish:

- **La apertura del polish y su puerta.** $\chi = 1$ y $O = O(\mathbf{m}, \alpha)$ solo donde la roughness propia del
  material es al menos `kPolishFieldRoughness` = 0.3; en el resto $\chi = 0$ y $O = O_{\text{L}}$. En el cromo (0) y el
  metal rugoso (0.22) del Corvette, la apertura por cells leía un relMSE de 1.64 y 0.221 frente al path trace donde la
  de los harmonics leía 1.05 y 0.128 para la misma media, y el campo leía los metales un tercio más claros (cromo 0.261
  donde el path trace lee 0.197). Un lobe de 0.3 o más ancho es más ancho de lo que resuelve el grado del campo, así
  que lo que lo cierra (el suelo de una bola) es lo que dice el campo: bolas verdes y blancas a 0.3421 sobre un suelo
  gris reflejaban el cielo en su mitad inferior sin esto. La puerta lee la roughness *del material*, no la guardada,
  porque el ensanche del bake lleva la roughness guardada de un tubo fino de cromo por encima de 0.3. El coat usa
  siempre las cells y el campo.
- **El sol analítico.** El mapa no contiene sol, así que el sol se añade a través de la distribución de direcciones
  reflejadas de cada lobe con el ojo a lo largo de la normal, $D(\mathbf{h})/4$ para el vector medio entre espejo y
  sol, con el semiángulo del disco $\theta_\odot$ incorporado a la anchura:

  $$p_\odot(\mathbf{m}, \mathbf{s}) = \frac{\tilde\alpha^2}{4\pi\,d^2},\qquad d = \cos^2\!\tfrac{\theta}{2}\,(\tilde\alpha^2-1)+1,\qquad \tilde\alpha^2 = \alpha^2 + \theta_\odot^2,\qquad \cos\theta = \mathbf{m}\cdot\mathbf{s}.$$

  Integra uno, así que peso por $E_\odot$ por $p_\odot$ es la reflexión entera del sol. El split sum leía un sol
  compacto una cuarta parte corto en todo nivel del mapa (0.57-0.74 del path trace en una bola de metal flotante; parte
  de esa diferencia venía de la densidad de muestreo del domo del propio path tracer, ya corregida).

### 4.6.4 Capas, Fresnel, metales, sheen

**Albedo de entorno.** $P$ es el `ggxDirectionalAlbedo` de MaterialX con compensación de dispersión múltiple, no el
ajuste de Lazarov, que difería hasta un 76 % en rasante; un dieléctrico usa el Fresnel exacto frente a $F_0$, un
conductor su color en ambos extremos, mezclados por $m$. Donde el metal del material es un Schlick generalizado
(OpenPBR, glTF), va de su color de frente al color especular en rasante (`ggxEnvSchlickMetal`); para un metal oscuro un
conductor de índice artístico y un Schlick difieren al doble a sesenta grados, y la pintura del coche se leía mate y
clara como conductor.

**El Fresnel del coat.** Schlick se queda bajo en el medio de la curva de un dieléctrico (0.070 a sesenta grados donde
una laca a 1.45 refleja 0.081); un coat sobre negro leía 0.084 frente a 0.099 del path trace. El albedo del coat es el
ajuste escalado por exacto sobre Schlick en el ojo, desvaneciéndose con la roughness:

$$E_c = E_{\text{fit}}(r_c, \mathbf{n}\cdot\boldsymbol{\omega}_o, \eta_c)\cdot \operatorname{lerp}\!\Big(\frac{F(\mathbf{n}\cdot\boldsymbol{\omega}_o,\eta_c)}{F_{\text{Schlick}}},\ 1,\ \operatorname{sat}(\alpha_c)\Big),$$

y exactamente $F$ por debajo de $\alpha_c = 10^{-3}$. Frente a una cuadratura del lobe queda dentro del 3 % con
$\alpha$ 0.3 y 0.6, donde el ajuste se quedaba un 8-16 % por debajo.

**Capas.** Como el nodo `layer` de MaterialX: el coat sobre el sheen sobre la base, cada capa quitando de lo que hay
debajo la parte que refleja en el ojo. Con $E_{\text{fit}}$ el albedo ajustado del coat, $E_s$ el albedo direccional del
sheen y $w_s$ el mayor canal del color del sheen,

$$U = \big(1 - w_c E_{\text{fit}}\big)\big(1 - w_s E_s\big)\cdot \operatorname{lerp}\!\Big(1,\ \frac{1-K}{1-E_b K},\ w_c\Big),\qquad K = 1 - \frac{1-F_0(\eta_c)}{\eta_c^2},$$

siendo el último factor el oscurecimiento del coat de OpenPBR (la luz que vuelve de la base se refleja de nuevo hacia
abajo en la cara interior del coat), con $E_b$ el albedo de la base; la bola de pintura leía 1.5 veces el mesh en su
centro sin él. El sheen (Imageworks, Conty y Kulla) es un lobe tan ancho como un cuerpo, así que se responde como uno:
su albedo direccional por $\mathbf{B}$, bajo el coat.

## 4.7 Vidrio, transmisión y paredes finas

Una Gaussian no puede refractar. Lo que transmite es la luz que le llega por detrás: $(1-t)$ del cuerpo mira hacia el
lado del ojo, $t$ mira hacia el otro, y lo que hay detrás se ve a través de la opacidad de la Gaussian.

- **Pasa lo que no se refleja**: la mitad transmitida se pesa por $1 - w_{\text{spec}} E_{\text{fit}}(r,\eta)$, con
  $w_{\text{spec}}$ el peso especular; un Fresnel por cara, como lo paga el lobe de transmisión del path tracer; sin
  él una bola sale la mitad más brillante, con un borde 2.5 veces demasiado caliente.
- **El vidrio TX rasterizado** ve a través a lo largo de $-\boldsymbol{\omega}_o$: el cielo donde las cells de detrás
  están abiertas (`splatConeOpen` alrededor de $-\mathbf{n}$) y $\kappa F$ donde están cerradas, de modo que un parabrisas
  muestra el habitáculo, emborronado a grado 3. **Un vidrio sólido es una lente**: su propia cara lejana cierra todas las
  cells de detrás, así que el valor del campo bajo cielo blanco a lo largo de $-\boldsymbol{\omega}_o$ decide la
  apertura ($\operatorname{smoothstep}(0.5, 0.85)$ de su luminancia) y la parte abierta lee el cielo nítido a lo largo de
  la salida de una esfera estimada a partir de la curvatura media guardada (`lensExit`: doblado al entrar, a través de
  la cuerda, doblado al salir), con el Fresnel y el segundo tinte de la cara lejana (`lensExitThrough`). Llenar la lente
  con el campo dibujaba la cabeza transparente del peón lechosa y plana (0.58 frente al path trace del mesh).
- **Roughness a través del vidrio**: una inclinación de microfaceta $\theta$ gira un rayo especular $2\theta$ pero uno
  transmitido mucho menos, así que el cielo a través de ambas caras se lee a
  $r_t = r\sqrt{\operatorname{sat}((\eta - 1)/\sqrt2)}$; la cabeza del peón pasó de un relMSE de 0.081 a 0.048.
- **La reflexión de la cara lejana** devuelve $(1-R)\,R\,(1-R)$ por el tinte de la luz a lo largo del espejo (el color
  de la Gaussian lleva el primer tinte); una bola transparente bajo un cielo uniforme leía 0.952 sin ella, 0.9998 con
  ella (0.92 a través de ambas caras + 0.04 de polish cercano + 0.037 esto).
- **Paredes finas.** Una placa clasificada como fina (aristas abiertas, o un grosor $2V/A$ por debajo de cuatro cells o
  de un cincuentavo de su propio tamaño) se convierte con opacidad $2R_0/(1+R_0)$, $R_0 = ((\eta-1)/(\eta+1))^2$, y su
  reflexión se escala por la ganancia $g = 1/R_0$ a su propio índice ($R_0$ con un mínimo de $10^{-3}$; 1.5 solo donde
  no se guardó índice), de modo que el blend deja la reflexión entera. La tapa del faro del Corvette a 1.15 recibía
  0.0097 de opacidad y, con el antiguo suelo de 1.5, una ganancia de 25 donde 205 la deja entera. Las Gaussians de una
  lámina se dibujan con una opacidad de al menos 0.1 con su color reducido en la misma proporción (y su rampa con él):
  relMSE del parabrisas de 2.26 a 0.89 bajo el taller, de 0.31 a 0.067 bajo goegap.

## 4.8 La reflexión a lo largo del footprint

El error de la pintura es de nitidez, no de energía: frente al path trace emborronado un píxel su relMSE baja de 0.80 a
0.28. Una Gaussian mostraba la reflexión de su centro plana sobre todo su footprint.

### 4.8.1 El giro de la normal por píxel

La conversión escribe el shape operator del triángulo del centro en los ejes $(\mathbf{u}, \mathbf{v})$ de la
Gaussian, $S = \operatorname{sym}\big([\Delta\mathbf{n}_1\ \Delta\mathbf{n}_2]\,[\Delta\mathbf{p}_1\ \Delta\mathbf{p}_2]^{-1}\big)$
proyectado sobre $(\mathbf{u},\mathbf{v})$ (`m2sShape`). La proyección invierte su propia linealización en el plano del
splat: con $J$ la aplicación $2\times2$ de coordenadas del plano a píxeles, un píxel a la derecha y uno abajo son
$\mathbf{e}_x = J^{-1}(1,0)$, $\mathbf{e}_y = J^{-1}(0,1)$, y la normal a un píxel de distancia es

$$\mathbf{n}_x = \operatorname{normalize}\big(\mathbf{n} + \mathbf{h}_x\big),\quad \mathbf{h}_x = \operatorname{clamp}_{0.5}\!\big((S\,\mathbf{e}_x)_u\,\mathbf{u} + (S\,\mathbf{e}_x)_v\,\mathbf{v}\big),$$

con el giro limitado a media unidad para que un splat visto de canto no barra el cielo.

### 4.8.2 La pendiente, y el sharp lobe

Para una reflexión rugosa la proyección vuelve a leer el cielo a lo largo de
$\operatorname{reflect}(-\boldsymbol{\omega}_o, \mathbf{n}_x)$ y $\mathbf{n}_y$ y guarda el cambio del color por píxel,
$\mathbf{g}_x, \mathbf{g}_y$ (`kSlopeMark`). El blend dibuja $\max(\mathbf{c} + \mathbf{g}_x d_x + \mathbf{g}_y d_y, 0)$
en el desplazamiento $\mathbf{d}$. La mayor caída de la rampa sobre la elipse de tres sigma se limita al color por canal,
$3\sqrt{\mathbf{g}^\top\Sigma\,\mathbf{g}} \le \mathbf{c}$, de modo que el shading nunca llega a cero dentro del
footprint (sin el límite, una línea negra discontinua a lo largo del anillo dorado de un peón). La pendiente de un lado
gana a la diferencia central y a minmod en el Corvette (pintura 0.393 frente a 0.584 y 0.564 de relMSE).

Una rampa lineal no puede dibujar un punto. Un coat más liso que `kSharpRoughness` (0.2) y un polish de base más liso
que `kSharpPolishRoughness` (0.5) se pasan en cambio al blend (`kSharpMark`): la proyección resta del color la lectura
del centro (cielo y sol analítico), sin recortar, y escribe la coordenada octaédrica del espejo $\mathbf{a}_0$ y sus
pasos $\Delta\mathbf{a}_x, \Delta\mathbf{a}_y$ (puestos a cero a través del fold del octaedro), los pesos de los
lobes (peso del coat por $O_c$; peso rgb del polish por $O$), su roughness y sus pesos de sol. Por píxel el blend
decodifica $\mathbf{m}(\mathbf{d}) = \operatorname{octDecode}(\mathbf{a}_0 + \Delta\mathbf{a}_x d_x + \Delta\mathbf{a}_y d_y)$
y suma

$$w_c\,\tilde L(\mathbf{m}(\mathbf{d}), r_c') + \mathbf{w}_p\,\tilde L(\mathbf{m}(\mathbf{d}), r_p') + E_\odot\big(\mathbf{w}_{p\odot}\,p_\odot(\mathbf{m}(\mathbf{d}),\mathbf{s}; r_p') + w_{c\odot}\,p_\odot(\mathbf{m}(\mathbf{d}),\mathbf{s}; r_c')\big),$$

recortando después. Un sol de 1.5 grados reflejado en un coat curvo se dibuja donde cae dentro de la Gaussian y no como
una mancha del tamaño de la Gaussian. Coste: una lectura del cielo por píxel por registro de sharp lobe, y la memoria de
grupo de la pendiente en 17 412 de los 17 416 bytes de WebGPU.

### 4.8.3 La dispersión propia del píxel

La lectura del sharp lobe es una muestra puntual en el centro del píxel, mientras que la normal gira
$\mathbf{h}_x, \mathbf{h}_y$ a través del píxel: una caja de pendientes cuya varianza, doblada para el espejo, ensancha
el lobe:

$$\alpha'^2 = \alpha^2 + 2\cdot\frac{\lVert\mathbf{h}_x\rVert^2 + \lVert\mathbf{h}_y\rVert^2}{12},$$

con $\alpha' \le 1$, aplicada después del des-ensanche y del ensanche de fold de la sección 4.9. El labio enrollado de
un panel ya no centellea. En el sharp lobe del player WebGPU (§7), sin este término, los folds de la puerta y del umbral muestran
motas brillantes: con él, 27/25 motas pasan a 2/1 (en el coche de 1280 px el player lee la puerta a 35.53 dB y el
umbral a 32.36 frente al path trace, frente a 33.77 y 31.87 sin el término y con el bake de 640 px).

## 4.9 Deshacer el ensanche del bake

### 4.9.1 Toksvig en la conversión

Una Gaussian tiene una normal. Donde una tira fina y curva de cromo gira la reflexión a lo largo de una Gaussian, el
píxel del path trace promedia un brillo que se mueve en una línea fina, y una Gaussian dibujaba una cuenta de su propio
tamaño a pleno brillo (el cromo del Corvette leía 1.6 veces el path trace). La conversión ensancha la roughness, y la
del coat, por la dispersión de normales bajo la Gaussian (Toksvig):

$$s = \lambda\, \max_{\text{aristas } kj} \frac{\angle(\mathbf{n}_k, \mathbf{n}_j)}{\lVert\mathbf{p}_j - \mathbf{p}_k\rVert}\, \max(\sigma_u, \sigma_v),\qquad \alpha'^2 = \alpha^2 + 2 s^2,$$

con $\lambda$ = `--specular-filter` (1 con `--transfer`) y $r' = \operatorname{sat}(\alpha'^{1/2})$.

### 4.9.2 Sharp lobes sin ensanchar

Los sharp lobes ya giran el espejo por píxel; leídos a la roughness ensanchada, el giro se cuenta dos veces. La laca
de una puerta, un espejo ($r_c = 0$), salía mate: guardada a $p_{50} = 0.112$ en la puerta y 0.080 en el capó (bake de
640 px), saturando a 1 en los folds. La proyección recupera la dispersión a partir de la curvatura guardada,

$$s \approx \kappa_{\max}\, \max(\sigma_u, \sigma_v),\qquad \kappa_{\max} = \Big|\frac{S_{uu}+S_{vv}}{2}\Big| + \sqrt{\Big(\frac{S_{uu}-S_{vv}}{2}\Big)^2 + S_{uv}^2},$$

y lee el coat y el polish de sharp lobe a

$$r = \big(r'^4 - 2 s^2\big)^{1/4}\quad (\text{limitado a } 0).$$

### 4.9.3 El coat del fold

En un fold la laca ensanchada llegaba casi a 1 y un joint dibujaba una raya oscura (joints de la puerta a 0.69 del
path trace). El coat se limita:

$$r_c \leftarrow \max\big(\operatorname{unwidened}(r_c, s),\ \min(r_c,\ 0.3)\big)\quad\text{si } r_{\text{mat}} \ge 0.3,$$

`kCoatFoldRoughness` = 0.3. Aplicado en todas partes, ilumina el cromo, el vidrio y los radios de una llanta que el
path trace muestra oscuros (cromo -1.4 dB, radios -0.5 dB por región en el viewer web), de ahí la puerta solo para
pintura. A 1280 px el bake ya no ensancha el coat (sale exactamente a 0), así que en el coche con el bake de 1280 px
este término no cambia nada.

### 4.9.4 Un tubo gira, un fold se ensancha

El giro por píxel extrapola el shape operator del triángulo del centro; pasada una arista deja de valer. La normal y
la curvatura guardadas son exactas en el centro (normal a $p_{90}$ 0.009 grados de la normal interpolada del mesh),
pero el 64 % de los footprints $\pm2\sigma$ de un capó cruzan una arista, y el error RMS de la normal extrapolada llegaba
a $p_{90}$ 46 y $p_{99}$ 198 grados. Sea $t = 2s$ el giro a lo largo de $\pm2\sigma$. En la nube de 1280 px el cuerpo de
un tubo de cromo está en $t \in [0.3, 1]$, coherente con sus vecinas (2 % de signo opuesto); las normales suavizadas a
través de un fold dentro de triángulos más estrechos que la Gaussian están por encima de 1 (de 5 a 9 rad en un
umbral). Con

$$w = \operatorname{smoothstep}(0.7,\ 1.5,\ t),\qquad S \leftarrow (1-w)\,S,\qquad \alpha' = \min\!\Big(\sqrt{\alpha^2 + w\,(t/2)^2},\ 0.5^2\Big)$$

(`kFoldTurnLow` 0.7, `kFoldTurnHigh` 1.5, `foldWidened` limitado a `kFoldRoughness` 0.5 en $r$, nunca más estrecho que
antes), un tubo gira sus píxeles del todo y dibuja su línea fina de luz, mientras que un fold se integra como lo
integra el path tracer dentro de un píxel. El smoothstep evita que una cámara en movimiento vea un salto.

## 4.10 Medidas frente al path trace

Todas las cifras del coche son frente al path trace del mesh (`/World/Camera`, 1920 x 1080, `autoshop_01_4k`).
«Recorte del coche» es el PSNR sobre el recorte de 900 x 420 del coche, «frame» sobre el frame entero; los cocientes por
clase son la media de una región sobre la del path trace. «Leaves» es la nube entera del bake dibujada sin cut; «cut»
es el cut por error de §6.3 a 0.25 px; «cull» es el back-face culling de §6.6. Las configuraciones *ensanchadas* leen los
sharp lobes a la roughness guardada por el bake (§4.9.1); las *sin ensanchar* aplican §4.9.2.

### 4.10.1 La puerta TX, por material (relMSE)

| estado | cromo | metal rugoso | pintura |
|---|---|---|---|
| primer transfer | 0.536 | 0.0896 | 0.705 |
| TX, cells y campo en todo | 1.64 | 0.221 | 0.393 |
| polish abierto por harmonics, cono ≥ cell, campo solo en el coat | 0.591 | 0.0835 | 0.341 |
| después: + coat y polish de sharp lobe por píxel | 0.545 | 0.083 | 0.281 |

### 4.10.2 Coste de reproducción

El Corvette TX entero (14.7 M de Gaussians, sin suelo, 1920 x 1080) se dibujaba en 158 ms frente a unos 83 de una nube
con relight de su tamaño; de los 84.5 ms de la proyección, el campo y su acoplamiento se llevaban 24, las cells leídas
por los lobes 16.5, la parte del sol y su bounce 12, el transfer del cuerpo 7.5. Leer el campo una sola vez para
ambos lobes (§4.6.2) y guardar en caché los términos que no dependen del ojo (§4.5) atacan el primero y el último.

### 4.10.3 El coche entero, por bake y configuración

| bake | configuración | recorte del coche (dB) | frame (dB) |
|---|---|---|---|
| 640 px | leaves (5.38 M), ensanchadas | 16.37 | 21.92 |
| 640 px | leaves, sharp lobes sin ensanchar | 16.58 | 22.15 |
| 640 px | cut + cull, ensanchado | 16.29 | -- |
| 640 px | cut, sin ensanchar + coat limitado en folds (*cut A*) | 16.51 | 22.15 |
| 640 px | *cut A* + campo del polish en materiales rugosos (*cut B*) | 16.55 | 22.18 |
| 640 px | leaves (cut a 0.01 px) con todos los términos de *cut B*, sin cull / con cull | 16.73 / 16.75 | 22.40 / 22.43 |
| 640 px | *cut B* + giro limitado a 0.2 rad / + coat de fold solo en pintura | 16.57 / 16.57 | 22.20 / 22.20 |
| 1280 px | **cut + cull, giro limitado, coat de fold solo en pintura (5.0 M)** | **17.88** | **23.90** |
| 1280 px | lo mismo sin el límite | 17.88 | 23.91 |
| 1280 px | limitado + puerta del polish de sharp lobe en la roughness sin ensanchar | 17.88 | 23.90 |
| 1280 px | limitado + puerta + separación tubo/fold | 17.90 | 23.93 |
| 1280 px | limitado + separación tubo/fold (la que se conserva) | 17.88 | 23.91 |

El PSNR sobre un recorte se mueve poco; los cocientes por clase de abajo muestran dónde actúan los cambios.

### 4.10.4 La puerta y el capó, por clase

La pintura es una base de metal oscuro (roughness 0.3421) bajo un coat de roughness 0 a 1.45. Cada píxel se clasifica
por dónde cae su espejo: el suelo (cerrado; responde el campo) o el HDRI (abierto). Con las leaves de 640 px y los
lobes ensanchados, el 86 % del déficit de la puerta estaba en píxeles que reflejan suelo. Media sobre el path trace
(bake de 640 px):

| clase de la puerta | leaves, ensanchadas | cut + cull, ensanchado | *cut A* | *cut B* | leaves, sin cull | leaves, con cull |
|---|---|---|---|---|---|---|
| espejo hacia el suelo < 3 m | 0.78 | 0.89 | 0.92 | **1.02** | 0.90 | 1.02 |
| espejo hacia el suelo 3-30 m | 0.69 | 0.77 | 0.76 | 0.88 | 0.79 | 0.88 |
| espejo hacia el HDRI (abierto) | 0.92 | 1.04 | 1.03 | 1.13 | 1.01 | 1.13 |
| joints | 0.69 | 0.83 | 0.95 | **1.04** | 0.90 | 1.02 |

Deshacer el ensanche por sí solo deja las clases de suelo en 0.78/0.67 y los joints en 0.69; el campo del polish
(*cut B*) sube el suelo cercano a 1.02 y el coat de fold los joints a 1.04, como predice una estimación analítica del
término que falta (bajo-coat 0.94 x albedo del polish 0.078 x suelo reflejado 0.11 = 0.008 frente a un déficit medido
de 0.0056-0.0085). El back-face culling sube toda la puerta un 12 % aproximadamente. El 1.13 que queda en la clase
abierta se reparte según si el espejo encuentra el propio coche: el 68 % del exceso está en píxeles «coche» (2.04 del
path trace), donde las cells de 16 x 16, de unos 11 grados, dejan un 25 % de cielo abierto detrás de un retrovisor o un
pilar.

Contraste local (desviación típica del paso alto sobre la media, $\sigma$ = 1.5 / 4 px), relativo al path trace:

| | leaves de 640 px, ensanchadas | leaves de 640 px, sin ensanchar | *cut B* de 640 px | cut + cull de 1280 px |
|---|---|---|---|---|
| puerta, reflejos abiertos | 0.12 / 0.26 | 0.40 / 0.61 | 0.35 / 0.53 | 0.56 / 0.74 |
| capó, reflejos abiertos | 0.27 / 0.38 | 0.41 / 0.52 | 0.41 / 0.51 | -- |

Una simulación de la reflexión del coat con el espejo del path trace sobre el HDRI prefiltrado muestra que la roughness
guardada del coat explica por sí sola el emborronado: con $r_c$ = 0.11 la puerta conserva 0.24 / 0.31 del contraste de
$r_c$ = 0, con 0.08 el capó 0.21 / 0.28.

### 4.10.5 Los brillos del capó

Tiles de 24 px con brillo (168), alineados por correlación del paso alto dentro de $\pm$6 px:

| render | desplazamiento $p_{50}$ | $\sigma$ que iguala | pico (PT 5.8) |
|---|---|---|---|
| leaves de 640 px, ensanchadas | 4.1 px | 2.5 | 1.9 |
| 640 px, sin giro por píxel | 4.1 px | 1.5 | 2.1 |
| 640 px, sin pendiente | 4.2 px | 2.5 | 1.8 |
| leaves de 640 px, sin ensanchar | 3.2 px | 1.5 | 3.0 |
| *cut B* de 640 px | 3.2 px | 1.5 | 2.6 |
| cut + cull de 1280 px | 2.0 px | 0.75 | 3.3 |

La media del capó pasa de 0.89 a 0.98 del path trace a 1280 px. El error de extrapolación de la normal escala con el
cuadrado de la cell (mediana de 0.146 grados a 640 px, 0.060 a 960, 0.032 a 1280).

### 4.10.6 La normal extrapolada, y la separación tubo/fold

Error RMS de la normal extrapolada en el capó (grados):

| variante | $p_{50}$ | $p_{90}$ | $p_{99}$ |
|---|---|---|---|
| sin giro | 0.575 | 22.7 | 35.8 |
| shape operator del centro | 0.146 | 46 | 198 |
| + límite $\lVert S\rVert\,2\sigma \le 0.2$ rad | 0.147 | 20.2 | 32.0 |
| mínimos cuadrados con muestras del mesh | 0.088 | 9.8 | 14.6 |

A 640 px el límite mueve el recorte del coche de 16.55 a 16.57 dB pero empeora los joints de la llanta delantera (1.48 a
1.54); a 1280 px quitarlo no cambia nada (§4.10.3). La separación tubo/fold de §4.9.4 lo sustituye. Sobre el cut de
1280 px:

| | giro limitado | + puerta del polish de sharp lobe | + puerta + separación | + separación (la que se conserva) |
|---|---|---|---|---|
| motas brillantes en el fold del umbral | 10 | 11 | 1 | 1 |
| tira cromada, RMS | 0.0358 | 0.0535 | 0.0415 | 0.0359 |
| tira cromada, paso alto / PT | 0.41 | 1.12 | 0.58 | 0.41 |

La separación que se conserva lleva además las motas de la puerta de 405 a 389 y las de la llanta de 204 a 190, sin
cambiar frame ni recorte. A 1280 px el bake ya no ensancha el coat, y sin la separación las Gaussians del fold del
umbral reflejan como espejos con normales que giran, rayas brillantes sobre la ranura oscura del path trace; la
separación las quita sin tocar la tira. Alternativas consideradas: poner una puerta al polish de sharp lobe según la
roughness sin ensanchar triplica el error de paso alto de la tira; `--specular-filter 0` dibuja una línea oscura
discontinua en el fold del umbral y cromo que centellea. El polish necesita ahí su ensanche.

## 4.11 Los parámetros como canales de imagen

Cada Gaussian lleva los parámetros que lee el relight -- su material, su forma y su transporte de luz precalculado -- y
cualquiera de ellos se puede dibujar a través de una cámara en lugar de la luz (`athenea:splatView`). El splat se mezcla
por su opacidad como la luz, así que un frame de un parámetro es ese parámetro tal como la cámara ve la nube: el mismo
orden, las mismas footprints y la misma coverage que el frame con luz, y por tanto alineado con él píxel a píxel. Los valores
que no son colores se ponen en una escala para verlos, y se guardan como números en un EXR. La figura 13 muestra el peón
de §5.11.

| canal | qué es | escala |
|---|---|---|
| light | el frame con relight bajo el cielo elegido | radiancia lineal |
| baked | la luz que lleva la nube, capturada o de su bake, sin relight | radiancia lineal |
| albedo | el color de la superficie sin luz | lineal |
| roughness | la roughness que lee el especular, ensanchada por la dispersión de normales bajo la Gaussian (§3.7) | 0–1 |
| materialRoughness | la roughness propia del material antes de ese ensanche; 0 donde la nube no la guarda | 0–1 |
| metallic, transmission | los pesos de metal y de transmisión del material | 0–1 |
| ior | el índice de refracción del especular menos uno | gris |
| normal | la orientación que usa el shading, en el mundo | $\mathbf{n}/2 + 1/2$ |
| opacity | la opacidad máxima de la Gaussian | 0–1 |
| size | el eje mayor de la Gaussian | rampa logarítmica, 10 µm (azul) a 10 cm (rojo) |
| skyVisibility | el cielo visto por el transfer ponderado por el coseno, $T_0 / Y_0$: 1 donde nada ocluye | 0–1 |
| bounce | la indirect half bajo un cielo blanco de radiancia uno: la luz que llegó a la Gaussian tras un bounce, con el color de lo que la rebotó | lineal |
| openCells | la fracción de las 256 direcciones que el bake encontró saliendo de la escena | 0–1 |
| field | la media del reflected field bajo un cielo blanco: lo que la Gaussian ve a su alrededor | lineal |
| emission | la luz que emite la propia superficie | lineal |

Los canales forman tres grupos. El material y la geometría (albedo, roughness, metallic, transmission, ior, normal, con
la profundidad y la posición en el mundo de §2.14) son entradas independientes. Los canales del transfer
(skyVisibility, bounce, openCells, field) son el transporte de luz que el bake precalculó en 3D: dependen de toda la
escena, incluido lo que ninguna cámara ve, y son la parte cara del asset. La luz es una función fija de ambos grupos y del
cielo: el relight de §4.5–§4.7 la calcula, así que la luz de un frame está determinada por sus canales y el HDRI. Algunos
canales se siguen de otros. La posición en el mundo es la profundidad desproyectada por la cámara, de forma exacta; la
normal se sigue de forma aproximada del gradiente de la posición en pantalla, perdiendo detalle y rompiéndose en las
siluetas; skyVisibility es, hasta el límite de banda del transfer, la media ponderada por el coseno de las direcciones
abiertas, que llevan más que ella; y la roughness es la del material ensanchada por la dispersión de las normales, que
una nube que guardase esa dispersión recalcularía. Otros no: los canales del transfer no se pueden recuperar de lo que ve
una cámara, porque una oclusión ambiental en espacio de pantalla pierde lo que queda fuera y un bounce o un reflejo
dependen de vecinos que pueden no estar en el frame; el albedo a partir de la luz es inverse rendering, mal planteado; y
la opacidad y el tamaño son propios de las Gaussians.

Dibujada así, una nube con relight bajo un conjunto de cielos y cámaras da frames alineados de entradas (material,
geometría), transporte precalculado y objetivos (la luz por cielo) -- el material de un dataset para aprender a predecir
el transporte, o la luz con relight, a partir de entradas que no lo llevan. En el peón a 768 × 768 un canal cuesta
20–60 ms. Una nube sin relight solo tiene su luz, su opacidad y su tamaño que mostrar.

![Figura 13. Los canales del peón a través de su cámara, 768 × 768: el frame con luz bajo el cielo del bake y quince parámetros dibujados en lugar de la luz. La cabeza de vidrio es la única parte que transmite y el anillo dorado el único metal; el canal de tamaño muestra las Gaussians fusionadas más grandes del LOD de la cabeza; el bounce es oscuro porque el peón está solo.](figures/img-channels.jpg)

## 4.12 Limitaciones

- **Campo de rango uno.** El acoplamiento pone el color de un suelo iluminado, no su forma, en una reflexión bajo un
  cielo iluminado por un lado; a grado 3 el campo resuelve un horizonte a unos veinte grados y mezcla la sombra del
  coche en el suelo con el suelo lejano (en simulación: correlación 0.56 a grado 3, 0.63 a grado 5, 0.85 con un mapa
  octaédrico de 32 x 32).
- **Resolución de las cells.** Cells de 16 x 16, de unos 11 grados, dejan pasar cielo donde un coat de sharp lobe
  refleja el propio coche (la clase abierta de la puerta a 1.13); cells de 32 x 32 (+96 B por Gaussian) lo reducirían a
  la mitad.
- **Un sol por domo**; las lámparas dentro de una escena se sombrean con bits que dicen qué direcciones salen de la
  escena, no cuáles llegan a la lámpara. El bake no tiene emisión de lámparas: los faros salen oscuros frente al path
  trace.
- **Defectos que quedan** en el coche de 1280 px: la tira cromada del parachoques delantero un 38 % más clara que el
  path trace y con el pico más bajo; la reflexión del suelo cercano en la llanta a 0.72; los brillos del capó desplazados
  2.0 px, que una curvatura ajustada sobre el footprint a partir del mesh reduciría más o menos a la mitad.
- **Los transfers zonales** guardan solo la direct half, y conocen solo la pose que se trazó; la oclusión entre
  miembros en otras poses no se multiplica.
- **Una nube girada en la ruta trazada.** La ruta con ray tracing gira sus direcciones al sistema de la nube pero
  hace el producto de los harmonics del transfer con el cielo del mundo, así que allí el cuerpo de una nube girada se
  ilumina con un cielo mal orientado.
