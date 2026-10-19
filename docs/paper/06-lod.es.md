<!-- Copyright (c) 2026 jesus luque. -->

# 6. LOD: merge trees, cuts por error, pages y oclusión

Una nube obtenida por bake desde un mesh a 1280 px contiene 8.7 millones de Gaussians para un solo coche. El close-up
level, con bake a 5120 px, contiene 74.5 millones. Las vistas van desde el asset entero a lo lejos hasta un emblema de
cerca, y una nube de ese tamaño no puede dibujarse ni mantenerse residente entera: lo que se dibuja y lo que está
residente debe seguir a la vista, bajo un presupuesto de nodos y uno de memoria, con los datos cargados por streaming
en unidades. Esta sección describe primero el LOD original del motor (§6.1), hecho para escenas capturadas, y explica
por qué no escala a esas nubes. Después trata la estructura que lo sustituye: merge trees por pares por pieza (§6.2),
un único cut por un error de pantalla acotado (§6.3), la opacidad que recibe un merge (§6.4) y una métrica de spill que
se evaluó y no se adoptó (§6.5), dos culls dependientes de la vista (§6.6, §6.7), pages ordenadas por error (§6.8), el
close-up level a 5120 px (§6.9) y las medidas (§6.10). Todo kernel nombrado aquí es un kernel de cómputo en Slang. La
CPU solo lee ficheros y lleva la contabilidad (§2).

Notación. Un nodo $n$ es una leaf (una Gaussian del bake) o un merge. $\mu_n$ es su centro y $\Sigma_n$ su
covarianza, con escalas $s_n = (s_x, s_y, s_z)$ y opacidad $o_n$. $\mathrm{par}(n)$ es su padre. $\tau$ es el umbral
en píxeles, $f$ es la focal en píxeles a distancia unidad (`pixelsPerUnit`) y $e$ es el ojo.

## 6.1 El LOD original del motor

**Octree por código de Morton.** `LodBuilder` (`modules/lod`, `shaders/athenea/lod/lod_common.slang`) ordena los
splats por un código de Morton de 30 bits. La cell del octree de un splat en el nivel $r$ son los $3r$ bits altos de
su código, así que las cells de un nivel son tramos de prefijos iguales. Una pasada de fronteras y una suma prefija
las encuentran, sin recorrer ningún árbol. Cada cell lleva momentos, siguiendo a Kerbl et al. 2024. Con
$w = o \cdot A_2(s)$, donde $A_2$ es el área de los dos ejes más largos,

$$W=\sum w,\quad S_1=\sum w\,\mu,\quad S_2=\sum w\,(\Sigma+\mu\mu^\top),\quad C=\sum w\,c .$$

Los momentos se suman, así que cada nivel más grueso se suma desde sus hijos. Una cell se convierte en una Gaussian
con $\mu=S_1/W$, $\Sigma=S_2/W-\mu\mu^\top$ (barridos de Jacobi en el shader), $c=C/W$ y $o=\min(W/A_2,0.99)$. Los
niveles guardados van desde `coarsestLevel` hasta el nivel más profundo cuyo número de cells es como mucho la mitad
del de splats.

**Cut.** `CutSelector` dibuja un grupo cuando la arista de su cell, proyectada a la distancia más cercana de la cell,
es como mucho $\tau$ y la proyección de la cell padre supera $\tau$. Una cell hija está dentro de su padre, así que la
prueba es monótona a lo largo de cualquier línea de la raíz a una leaf, y cada lugar se dibuja en exactamente un
nivel. Lo único que se lee de vuelta es un recuento por nivel y una lectura por instancia y frame.

**Chunks, `.athc` y streaming.** Los splats se dividen en chunks de 65 536 splats consecutivos en orden de Morton. Los
niveles fusionados se quedan en el dispositivo, y los chunks se cargan por streaming. El formato `.athc` (`.lrtc` en
el árbol de lucabRTrender) tiene una cabecera de 104 B, tablas de niveles y chunks, y bloques alineados a 4 KiB para
que un chunk pueda mapearse y traerse por fallo de página por sí solo. `StreamingPool` mapea el fichero. Dos hilos de
carga copian los chunks pedidos, los más pedidos primero. Cada ranura se toma de una ranura libre, del chunk pedido
hace más tiempo o de un chunk pedido menos de la mitad. Un grupo del nivel más fino al que le faltan chunks dibuja su
propia Gaussian fusionada, así que la imagen nunca tiene un agujero. En train_7k (742 k splats, M5 Pro), la
construcción tardó 47 ms y produjo 146 k Gaussians fusionadas. A 8 px el cut dibujó 83 k splats (11 %), con un render
de 3.3 ms frente a 12.6 ms y un error absoluto medio de $3\cdot10^{-3}$. En train_30k, el `.athc` cargado por
streaming en 4 de 16 ranuras escribió EXR idénticos byte a byte y se asentó en 2 cuts.

**Por qué no escala a assets con bake.** Cuatro propiedades de esta estructura no encajan con nubes de decenas de
millones de Gaussians con material bajo un presupuesto compartido.

1. El umbral se fija por instancia y no es un presupuesto. Cada instancia cuesta dos envíos bloqueantes y una lectura
   por frame, y no se comparte ningún presupuesto entre instancias.
2. Las cells del octree ignoran el objeto. Una cell fusiona lo que caiga dentro, incluido un emblema con el panel
   bajo él y el vidrio con la pintura.
3. Un splat por streaming cuesta 132 B, frente a ~55 B del registro compacto (§7.4).
4. Solo los splats van en chunks. Los niveles fusionados, hasta uno con hasta la mitad de cells que splats, están
   siempre residentes, y para millones de leaves ellos solos superan el presupuesto.

## 6.2 Merge trees por pares por pieza

`athenea decimate --tree` construye un merge tree binario por merge de pares de vecinos más cercanos mutuos. Es NanoGS
(Xiong et al. 2026) en el dispositivo (`lod_pairs.slang`, `lod_pairs_final.slang`, `Lod.cpp`), con guardas adecuadas
a nubes de bake con material (figura 16).

![Figura 16. Un merge tree y su cut. Los nodos son sumas de momentos; el error ε es monótono hacia arriba y la esfera ρ de cada nodo cubre las de sus hijos. Los nodos se agrupan en pages según el error del padre y llevan un índice a un codebook de oclusión conservador; el recorrido de cada frame prueba pages, back faces y oclusión.](figures/fig-lod.es.svg)

### 6.2.1 Los nodos como momentos

Un nodo es un vector de 24 floats: $W$, $W\mu$, $W(\Sigma+(\mu-\bar\mu)(\mu-\bar\mu)^\top)$ (6), $W\cdot\mathrm{sRGB}(c)$, $W n$,
$W\alpha$, $W\cdot$metálico, $W\cdot$transmisión, $W\cdot$especular, la clave, el número de registros, la razón de
grosor ponderada y el índice del nodo en el árbol. Aquí $w = o\cdot\max(s_xs_y, s_ys_z, s_xs_z)$ (`discArea`), y
$\alpha$ es la roughness GGX al cuadrado. Un merge suma los dos vectores. Como las sumas son asociativas, los momentos
de un nodo son los de todas sus leaves sea cual sea el orden de los merges. Todos los demás atributos (transfer, bits
de sombra, material, joints) se fusionan una sola vez al final, ponderados por masa sobre el tramo de leaves del nodo
(`lodMergeAttribute`).

El segundo momento se toma respecto al propio centro del grupo $\bar\mu$, no al origen: el de un grupo de leaves a partir
del desplazamiento de cada Gaussian a la media del grupo, el de un grupo más basto a partir de los de sus hijos con el
desplazamiento de la media de cada hijo (el teorema de los ejes paralelos), y el merge de un par igual. Respecto al
origen, $\Sigma = S_2/W - \mu\mu^\top$ es la diferencia de dos números del tamaño de $\mu\mu^\top$, que un float guarda
con unos $5\times10^{-7}$ m² a dos metros del origen -- toda la varianza de una Gaussian de un milímetro de ancho. Las
agujas de los bakes del close-up level sobre triángulos largos y finos (0.93 × 0.043 × 0.004 mm en el paragolpes
delantero, con varianzas transversales de $2\times10^{-9}$ m²) se fusionaban tan finas como las dejaba el redondeo, y el
coste leído de esos mismos momentos las daba casi gratis, así que un corte las elegía y dibujaba sus huecos como grano
brillante. En la esquina delantera izquierda del paragolpes con su bake aislado a 5120 px (301 k Gaussians), el corte a
0.01 px desde una cámara cercana pasó de relMSE $8.8\times10^{-4}$ a $1.3\times10^{-5}$ frente al bake, y el corte a
0.25 px desde `/World/Camera` dibuja 42 719 Gaussians en lugar de 94 351 con algo menos de error ($3.7$ frente a
$3.9\times10^{-4}$).

La Gaussian fusionada (`pairFinalize`) toma $\mu$ y $\Sigma$ ajustadas por momentos, descompuestas en autovectores
por Jacobi con el eje más delgado en tercer lugar. Fusionar por momentos conserva el segundo momento. Un bloque de
$2\times2$ discos tan anchos como su separación se fusiona por tanto en un disco de 0.58 de esa separación, lo que deja
agujeros visibles de cerca. A lo largo de cada eje ancho $u$, el merge se ensancha así. Sea $v_u$ la varianza
ponderada por área de los centros de los miembros, $c_u$ su anchura media y $d^2 = 12\sqrt{v_uv_v}/(n-1)$ la
separación de una cuadrícula de $n$ miembros sobre esa dispersión. Entonces

$$s_u' = \mathrm{clamp}\!\left(\frac{c_u}{d}\sqrt{12v_u+d^2},\; s_u,\; 2s_u\right).$$

La opacidad, con la regla por defecto, es la media de los miembros ponderada por área, limitada a 0.99. La coverage
de masa $W/A(s')$ se guarda como `coverageMass`. Un merge de discos cuyo eje más delgado es menor que 0.12 de su eje
medio se escribe como surfel (escala más delgada exactamente 0).

### 6.2.2 El coste

Para los nodos $a, b$ con $\pi = W_a/(W_a+W_b)$ y $\Sigma$ fusionada, el término de forma es la cota de Runnalls de la
divergencia KL por unidad de masa. El área del disco sustituye al volumen, que un surfel no tiene:

$$c_{\text{shape}} = \max\!\Big(0,\ \tfrac12\big[\log I_2(\Sigma) - \pi\log I_2(\Sigma_a) - (1-\pi)\log I_2(\Sigma_b)\big]\Big),$$

donde $I_2$ es el segundo invariante (la suma de los productos de autovalores dos a dos, calculada sin autovalores).
El coste completo es

$$c = \sigma_{\text{px}}\big(c_{\text{shape}} + w_f\,\phi^2 + w_n(1-\cos\theta_{ab})(1+F_0/\alpha)\big) + w_c\,\lVert\Delta c_{\text{sRGB}}\rVert^2 + w_m\,\Delta_{\text{mat}},$$

donde $\phi$ es el grosor que el merge añade sobre su miembro más grueso. El término $\sigma_{\text{px}} =
\mathrm{smoothstep}(0.5, 2, \text{px})$ anula los términos de forma de los merges que miden menos de medio píxel en la
cámara de la conversión. Valores por defecto: $w_c=20$, $w_n=1$, $w_m=4$, $w_f=4$.

### 6.2.3 Guardas

Un par cuyos miembros difieren en cualquiera de estas formas cuesta $\infty$ (`kNoMerge`):

| guarda | regla | por defecto |
|---|---|---|
| pieza | la clave difiere (prim, joint, lámina, pared fina, metal Schlick) | siempre |
| vidrio | transmisión $>0.5$ en un solo lado | siempre |
| vidrio con vidrio | solo con `--merge-glass` | no |
| espejo | $\alpha<0.01$ en un solo lado | siempre |
| fold | $\cos\theta_{ab} < \cos 60^\circ$ | 60° |
| brillo | $\alpha<0.1$ y $1-\cos\theta_{ab} > \tfrac12\kappa^2\alpha$ | $\kappa=1$ |
| borde de color | $\lVert\Delta c_{\text{sRGB}}\rVert > 0.15$ | 0.15 |
| aguja | aspecto del plano por encima de `maxAspect` y de 1.25× el de los miembros | 1000 (desactivada) |
| trazo | merge más ancho que `maxPixels` en la cámara | no |

### 6.2.4 Pasadas y la cima relajada

En cada pasada, cada nodo calcula el coste de fusionarse con los 8 vecinos a cada lado a lo largo de un orden de
Morton de los centros. Después repite esto a lo largo de un segundo orden desplazado media cell fina, y se queda con
la pareja más barata. Los pares que se eligieron mutuamente se reparten en un histograma de $\log_{10}$ del coste, y
el host elige el umbral de coste que fusiona tantos pares como permite la pasada, que es como mucho la mitad de los
nodos. El merge se detiene en `treeRatio` (4096×) o cuando ya nada puede fusionarse. Con `relaxTop`, el merge sigue
después con el borde de color (2.0), el ángulo, la guarda de vidrio entre sí, el aspecto y el límite de píxeles
levantados. Las guardas de clave, de vidrio con no vidrio y de espejo se mantienen. Cada uno de esos merges guarda su
coste **negado**, lo que lo marca como merge más allá de las guardas. Esto da a una vista lejana niveles gruesos a los
que cortar: una tapa de faro pasa de 76 486 raíces a 123.

**Disposición.** El árbol se escribe como un ParticleField. Las leaves van primero, en orden de profundidad, de modo
que cada nodo es un tramo contiguo de leaves. Los merges siguen en el orden en que se hicieron. Cada nodo lleva
`lodParent` (una raíz es su propio padre), `lodStart` (su primera leaf), `lodSize` (su número de leaves) y `lodCost`.

### 6.2.5 Muchas piezas: `--passes`, `--merge-glass`, `--object-groups`

En el Corvette entero (5.38 M leaves), el árbol puede construirse de una vez, pero escribir sus atributos supera la
memoria del dispositivo. `--object-groups K --object-group i` conserva las Gaussians cuyo `cryptoObject` módulo $K$ es
$i$. Los merges nunca cruzan piezas, así que el árbol de cada grupo es exactamente el subárbol que el asset entero
tendría para esas piezas. Cada pasada fusiona como mucho la mitad de lo que queda, y un grupo de muchas piezas
pequeñas fusiona pocos pares a la vez. El tope por defecto de 60 pasadas deja un grupo con 116 549 raíces de 323 613
leaves. Con `--passes 1000` llega a 1 298 raíces en 310 pasadas. `--tree-report` muestra dos causas de las raíces
restantes: el vidrio macizo nunca se fusiona (raíces igual a leaves), y los neumáticos fusionan un goteo por pasada
hasta el tope.

| grupo (coche a 640 px) | raíces antes | con `--merge-glass --passes 4000` |
|---|---|---|
| grupo 2 | 190 490 | 4 237 |
| grupo 5 | 36 508 | 817 |
| coche entero (8 grupos) | — | 17 133 raíces, 5.38 M leaves, 3 582 pages |

## 6.3 El cut por error, con cotas

**Alternativa considerada: el cut por tamaño.** `--cut-pixels` conserva un nodo cuando su anchura en pantalla
$2\max(s)f/d$ es como mucho $\tau$ y la de ningún antepasado lo es. Los merges más allá de las guardas solo pasan a
$0.25\tau$. Ese cut sigue disponible. El cut por error que sigue, a partir de Nanite, da más PSNR con menos nodos
(tabla al final de esta sección).

**Error propio.** Sea $r_n=\lVert s_n\rVert$ (la raíz cuadrada de $\mathrm{tr}\,\Sigma_n$) y $\alpha_n =
\max(\text{roughness},0.02)^2$. El error propio de un merge, en metros, es

$$\varepsilon_0(n) = r_n\Big(|\mathrm{cost}_n| + w_c\,s_c(n) + w_n\,\frac{\theta(n)^2}{\alpha_n} + w_s\max(E_n-E_0,0)\Big),\qquad w_c=10,\ w_n=1,$$

y $\varepsilon_0=0$ para una leaf. Los merges más allá de las guardas ($\mathrm{cost}_n<0$) no se miden por su coste,
que no ve el moiré que causan en el vidrio ni el color que arrastran a través de un borde. Por eso se acotan a

$$\varepsilon_0(n) \leftarrow \max\big(g\,\varepsilon_0(n),\ 2r_n\big),\qquad g=4\ (\texttt{--cut-guard}).$$

El término de spill $w_s$ es 0 por defecto (§6.5).

**Cotas.** El coste de un merge solo cubre el último par que unió. Una cadena de merges, cada uno dentro del borde de
color, puede derivar a través de ese borde. Por eso una dispersión de color $s_c$ y un cono de normales $\theta$ se
suben por el árbol con la desigualdad triangular (`treeBoundsUp`), empezando en 0 en las leaves:

$$s_c(p) = \max_{k\in\mathrm{ch}(p)}\big(s_c(k)+\lVert c_k-c_p\rVert\big),\qquad \theta(p) = \max_{k}\big(\theta(k)+\angle(n_k,n_p)\big).$$

Ambas son por tanto cotas sobre cada leaf bajo $p$.

**Error monótono y esferas que cubren.** El error se hace monótono, y cada nodo recibe una esfera que contiene las
esferas de sus hijos:

$$\varepsilon(n) = \max\big(\varepsilon_0(n),\ \max_{k}\varepsilon(k)\big),\qquad \rho(n)=\max_k\big(\lVert\mu_n-\mu_k\rVert+\rho(k)\big),\quad \rho(\text{leaf})=2\max(s).$$

Cada subida es un kernel lanzado sobre todos los nodos, en el que cada nodo eleva a su padre con `InterlockedMax` sobre
los bits del float. El kernel se relanza hasta que un contador lee cero, lo que da la profundidad del árbol en rondas.
El error se proyecta en el punto más cercano de la esfera:

$$P(n) = \frac{\varepsilon(n)\,f}{\max(\lVert\mu_n-e\rVert-\rho(n),\ 10^{-3})}.$$

**Proposición.** Si $P(n)\le\tau<P(\mathrm{par}(n))$ selecciona $n$ (a una raíz le basta la primera desigualdad),
entonces cada leaf se dibuja exactamente una vez. *Demostración.* $\varepsilon$ no decrece al subir. La esfera del
padre contiene la del hijo, así que la distancia a la esfera decrece al subir. Por tanto $P$ no decrece a lo largo de
ningún camino de una leaf a la raíz. Las leaves tienen $P=0\le\tau$, así que el predicado «$P\le\tau$» se cumple en un
prefijo del camino que empieza en la leaf, y exactamente un nodo es el último de ese prefijo. $\square$

Así que el cut es una sola prueba paralela por nodo (`treeErrorKeep`), sin subir por los antepasados. `--cut-budget
N` biseca $\log\tau$ en $[10^{-4},10^4]$ en 40 pasos para hallar el umbral más pequeño cuyo cut tiene como mucho $N$
nodos.

| configuración | escena | cut por tamaño | cut por error |
|---|---|---|---|
| lejos, 393 px, 0.25 px | seis piezas | 83 k, 34.4 dB | 34 k, 34.9 dB |
| cerca, 426 k nodos | seis piezas | 42.92 / 31.22 dB (frame / grille) | 43.61 / 34.50 dB (prototipo de referencia 43.61 / 34.47) |

## 6.4 La opacidad de un merge: coverage por mínimos cuadrados

La masa sobre la elipse de momentos cuenta dos veces las colas solapadas de las leaves. El exceso mide un 27 % en la
mediana y 2.3× en el p10. En la grille a 393 px, los merges salen más oscuros que sus leaves. `--cut-coverage` (al
hacer el cut) y `--tree-coverage` (al escribir el árbol) fijan la opacidad de cada merge a partir de lo que sus leaves
cubren de verdad, así:

1. El plano del nodo lo generan sus dos ejes más anchos $u,v$, con escalas $s_u,s_v$. Una cuadrícula de $8\times8$
   cubre el núcleo $[-1.25,1.25]^2$ en unidades de $(s_u,s_v)$.
2. Cada leaf $l$ bajo el nodo (su tramo, hasta 4096 leaves al escribir el árbol y 2048 en un cut) se proyecta sobre
   el plano. Su covarianza en el plano es $(u^\top\Sigma_lu,\ u^\top\Sigma_lv,\ v^\top\Sigma_lv)$ y el desplazamiento
   de su centro es $m_l$. Un tramo más largo conserva su opacidad basada en la masa. Las leaves se componen como lo
   haría el rasterizer: $C(x)=1-\prod_l\big(1-\min(o_l,0.99)\,G_l(x)\big)$.
3. El footprint propio del nodo en la cuadrícula es $G(x)=\exp(-\tfrac12\lVert x\rVert^2)$. La opacidad es el ajuste
   por mínimos cuadrados de $aG$ a $C$:

$$a^\star = \frac{\sum_x G(x)\,C(x)}{\sum_x G(x)^2},\qquad o_n = \mathrm{clamp}(a^\star,0,0.99).$$

Se midieron dos alternativas. Normalizar por $\sum G$ en vez de $\sum G^2$ da 0.5 para un merge tan ancho como una
leaf opaca. Un ajuste sobre $\pm2\sigma$ en vez de $\pm1.25\sigma$ da 0.84 a los merges de un panel, porque el borde es
donde los vecinos llevan la superficie, y un guardabarros sale moteado. Las leaves de canto respecto al plano, como
los alambres de una grille vistos de frente, no cubren nada, así que el merge va a 0. Un cargador descarta todo lo que
esté por debajo de 1/255, lo que rompería el árbol, así que un merge se escribe como mínimo a

$$o_n \ge 2/255\quad(\texttt{treeCoverageFloor}).$$

El árbol se reescribe en su propio orden de fichero, y solo cambian las opacidades de los merges
(`treeCoverageToFile`), porque `lodParent`/`lodStart` son índices del fichero que una reescritura a través del almacén
en orden de Morton revolvería. El ajuste lee los ejes de un Blob como filas. Leídos como columnas dan la rotación
inversa y giran mal cada Gaussian: los merges de un guardabarros salen entonces a 0.59 de su masa donde un ajuste de
referencia da 1.00.

| configuración | qué | coverage | masa |
|---|---|---|---|
| ajuste al hacer el cut (48 k) | grille, 393 px | 39.79 dB (desde 33.87) | — ; leaves 41.59 |
| árbol reajustado (49.4 k) | grille, 393 px | 41.28 dB | 39.79 con 48.1 k (`--cut-coverage`) |
| 1 px, cotas, ejes leídos como columnas | seis piezas cerca a 1920 | 40.96 frame / 32.09 grille | árbol antiguo 41.11 / 30.09 |
| ejes leídos como filas | seis piezas cerca a 1920 | **43.86** | 42.55 |
| ejes leídos como filas | seis piezas lejos a 393 | 39.95 | 39.66 |
| ejes leídos como filas | grille cerca a 393 | 39.71 | **41.16** |

La grille es la excepción. Los merges vecinos se solapan y se componen como $1-\prod$, así que un poco más de
opacidad oscurece todo el parche. El player web (Spark) dibuja la masa de cada nodo
($\min(\text{coverageMass},0.99)$), y el rasterizer del motor conserva la coverage.

## 6.5 La métrica de spill, y por qué falló en la grille

En un cut grueso los agujeros de la grille se cierran: en el cut de 0.25 px del coche a 640 px a 1920 px con back-face
culling, la región de suelo cercano lee 2.24× el path trace, frente a 1.18 en las leaves. La causa son merges de dos
leaves de las tiras subpíxel de `Chrome_Part_Grill`, vistas de frente y de 3–6 px de ancho a $\tau=0.25$. Su coste es
~0 porque la coverage de un par, medida en su propio plano, no cambia. Se define un spill sobre la propia cuadrícula
del ajuste de coverage:

$$E_n = \frac{\sum_x \max\big(a G(x) - C(x),\,0\big)}{\sum_x aG(x)}\in[0,1],$$

es decir, la parte de lo que pinta el merge que sus leaves no pintan. Coincide con una implementación de referencia
en NumPy con una correlación de 0.994, y se escribe como `lodSpill`. El cut suma $r\,w_s\max(E-E_0,0)$ al error
propio.

| coche a 640 px, grupo 0, 0.25 px | nodos | recorte del coche | frame |
|---|---|---|---|
| $w_s=0$ | 307 665 | 16.55 dB | 22.18 dB |
| $w_s=2$ | 561 242 (+82 %) | 16.79 | 22.38 |
| $w_s=5$ | 599 983 | 16.79 | 22.38 |
| $w_s=10$ | 612 963 (+99 %) | 16.79 | 22.38 |

El spill cierra los agujeros: de 1.90 a 1.09 de las leaves en el render, y de 4.66× a 1.0 en un indicador de
agujeros en espacio de imagen. El precio es un 82–99 % más de nodos. Todo merge derrama alrededor de 0.16, y los
merges que llenan los agujeros no derraman más que los de un panel, así que ningún suelo $E_0$ separa los dos grupos.
El agujero es una propiedad de la **proyección**: el merge no cubre en 3D nada que sus vecinos no cubran, y el hueco
solo se abre donde los footprints en pantalla de las tiras se separan. Ninguna medida por nodo independiente de la
vista lo ve. El valor por defecto es $w_s=0$. El bake a 1280 px (§6.10) resuelve el problema en los datos: los
agujeros de la grille en el cut pasan de 2.24 a 1.04.

## 6.6 Back-face culling en closed shells

A tamaños de salida pequeños, un cut muestra las caras traseras en negro. En la grille a 393 px, el culling de caras
traseras sube las leaves sin cut de 38.2 a 41.6 dB. `--cut-cull-back` descarta un nodo cuyo cono de normales entero
mira hacia atrás:

$$\theta(n)<\tfrac{\pi}{2}\ \wedge\ n_n\cdot\frac{e-\mu_n}{\lVert e-\mu_n\rVert} < -\sin\theta(n).$$

El cono acota cada leaf bajo el nodo (§6.3), así que todo el subárbol del nodo mira también hacia atrás. Dos medidas
fijan dónde se aplica. Aplicado a las seis piezas de prueba, con la tapa de vidrio del faro entre ellas, cuesta 6 dB a
1920: el vidrio se ve a través, así que sus caras traseras son visibles. En el coche entero a 393 px con el cut por
error de 1 px, las puertas y paneles, que son dos pieles, muestran una red de grietas oscuras: los merges de la piel
interior, mirando hacia atrás e iluminados por el interior oscuro, se ordenan por parches delante de la piel exterior.
La red es la misma con opacidad de masa o de coverage, y `--cut-guard 1000` no la cambia. En un closed shell, lo que
hay detrás de una cara que mira hacia atrás es el interior de la propia pieza, así que hacerle culling es seguro y
elimina la red. Por eso el culling se aplica, a leaves y merges por igual, solo en **closed shells que no son vidrio**
(`primvars:athenea:splat:closed`, transmisión $\le 0.5$).

**La regla de aristas abiertas.** `mesh2splat` cuenta las aristas no compartidas por exactamente dos triángulos, $O$.
Se consideraron dos reglas más simples: $O=0$, y una proporción de aristas abiertas por debajo del 2 % (111 de 192
meshes cerrados). El borde de una lámina abierta crece como la raíz cuadrada de su área, así que un tubo fino pasa la
regla de proporción: el barril de una llanta, al 1 %, recibe culling, y se ve luz entre los radios. La regla que se
usa es

$$\text{closed} \iff O < 2\sqrt{E},\qquad E = 1.5\,T,$$

para un mesh de $T$ triángulos. Con ella, las puertas (0), los logos (1.4) y las pinzas de freno (1.5) son cerrados, y
las llantas (4.0), los neumáticos (3.1), el capó (3.1) y el parachoques (4.4) no, lo que da 102 de los 192 meshes del
Corvette. Con todas las leaves a 1920 px frente al path trace, el culling sube toda la puerta un ~12 % de forma
uniforme (puerta 1.01, suelo cercano 1.02 del path trace), con 4.35 M → 3.70 M Gaussians y el recorte del coche
16.73 → 16.75 dB.

## 6.7 Occlusion masks

Un transfer TX guarda 256 bits de sombra por Gaussian, uno por cell de una cuadrícula octaédrica de $16\times16$ sobre
la esfera del mundo, activado donde un rayo desde la Gaussian llegó al cielo. Una dirección cuyas cells están todas
cerradas es una desde la que un ojo fuera del asset no puede ver la Gaussian. Los bits del sombreado no pueden
reutilizarse para esto: un merge guarda los bits de sus leaves por mayoría, y el codebook del player (`shadow_vq`,
§7.3) también es una mayoría, así que ambos cerrarían direcciones desde las que se ve alguna leaf.
`--occlusion-mask N` (`occlusion_mask.slang`, `Occlusion.cpp`) construye una máscara conservadora por nodo.

**Máscaras.** La máscara $M_l$ de una leaf son sus cells abiertas dilatadas una cell ($3\times3$), reflejadas a
través del borde del cuadrado octaédrico en el fold. Una cell son ~11°, y sin el margen el borde de un oclusor cierra
direcciones desde las que el cromo de una grille aún se ve. La máscara está toda abierta en el vidrio (transmisión
$>0.5$, porque lo que hay detrás del vidrio se ve a través de él) y en una leaf sin bits. La máscara de un merge es el
OR de las de sus hijos, subido hasta que ninguna máscara cambia:

$$M_p = \bigvee_{k\in\mathrm{ch}(p)} M_k .$$

**Codebook conservador.** Las máscaras se sustituyen por un índice a $N$ entradas de 256 bits cada una. La entrada 0
está toda abierta. El **valor** de un nodo es el número de cells cerradas entre las *relevantes*, las direcciones
con $d\cdot\hat{up} > -0.25$ (un ojo hasta ~15° bajo el horizonte). Los nodos de los intervalos de menor
$\log_2$-valor, que juntos suman menos del 2 % del valor total, toman la entrada 0 y quedan fuera del ajuste. Las
demás entradas se siembran con nodos ajustados repartidos por el fichero. La ronda 1 asigna cada nodo a la entrada más
cercana en distancia de Hamming sobre las cells relevantes. Las rondas 2–8 asignan un nodo solo a una entrada que
**contiene** su conjunto abierto, eligiendo la que tiene menos cells relevantes abiertas. Tras cada ronda una entrada
pasa a ser el OR de sus miembros. Cada entrada es por tanto un superconjunto del conjunto abierto de cada miembro, y el
codebook nunca puede cerrar una dirección que un nodo tiene abierta. El coste del codebook son las cells que reabre
($M_{\text{entry}}\wedge\lnot M_n$) como fracción de todas las cells cerradas.

**Repartido por necesidad.** Los árboles de un asset comparten un codebook (`--occlusion-group`). Cada unidad se
ajusta sola al tamaño completo y se cuentan las entradas que usa. Después el codebook se reparte como
$\text{share}_u = 16 + \lfloor\text{spare}\cdot \text{used}_u/\sum\text{used}\rfloor$, y cada unidad se reajusta
dentro de su parte. Repartir por calidad de ajuste y por pieza (`--occlusion-by-part`) se midió, y es peor en ambos
casos:

| configuración | árboles | codebook | reabierto | culling frente al ideal |
|---|---|---|---|---|
| un codebook por árbol | 6 piezas | 256 | 3.2–10 % | — |
| un ajuste sobre todos los árboles | 6 piezas | 4096 / 1024 | 3.2 % / 6.5 % | — |
| + valor, solo cells relevantes | 6 piezas | 4096 / 1024 | 21.9 % / 24.7 % | — |
| valor = cells cerradas, no leaves | 6 piezas | 4096 | 14.7 % | — |
| repartido por necesidad | 6 piezas | 4096 | 7.2 % (grille 2.8 %, 1 249 entradas) | 86 % cerca / 82 % lejos |
| 8 grupos por necesidad | coche 640 | 4096 | 16.2 % | — |
| 8 grupos por necesidad | coche 640 | 8192 | 11.9 % | 84 % |
| por pieza (192) | coche 640 | 4096 / 8192 | — | 66 % / 74 % |
| 8 grupos por necesidad | coche 1280 | 8187 | 12.3 % | — |
| 24 árboles por necesidad | coche 5120 | 8182 | 0–23.6 % por árbol | — |

(Desde la tercera fila, el ajuste cuenta solo las cells relevantes, así que el total reabierto sobre *todas* las cells
sube. Esos porcentajes no son comparables con los de las dos primeras filas.) En el close-up level, la suma de cells
cerradas sobre ~150 M nodos supera un contador de 32 bits, así que los totales se suman en 64 bits. Los resultados por
árbol son: Windshield 0.0 % (vidrio, 1 entrada usada), Part_2 5.3 %, Hood 7.0 %, la puerta piloto 7.7 %, los grupos
finos 8.9–15.9 % y el grupo de paneles pequeños pan2 23.6 %.

**La regla del player** (y `occludedFrom` en el cut). Esta regla se aplica solo cuando el ojo está fuera de la caja
envolvente del asset. Sea $d=\mathrm{normalize}(e-\mu_n)$ y $q=16\,\mathrm{oct}(d)-0.5$. Las cuatro cells
$\lfloor q\rfloor+\{0,1\}^2$, limitadas a $[0,15]^2$, son el footprint bilineal de $2\times2$ hacia el ojo. El nodo
recibe culling cuando las cuatro están cerradas en su entrada. Con el coche como oclusor (nunca en una escena
parcial), la regla quita un 12 % del cut cercano y un 8 % del lejano por 0.06 dB. En el coche entero a 393 px, con
back-face culling, la oclusión lleva el cut cercano de 522 931 a 360 333 nodos.

## 6.8 Pages por error

Un player no puede tener un árbol entero. Necesita unidades que traiga enteras y que pruebe sin leer sus nodos.

**Alternativa considerada: pages por nivel.** Las pages tomadas como cimas en anchura de subárboles hermanos siguen
necesitando 114 de las 263 pages del capó para un cut lejano. Las cotas conservan nodos pequeños dondequiera que gira
un brillo o un color, así que el cut abarca nodos desde leaves sueltas hasta $2^{11}$ leaves.

**Pages por error.** Un nodo se necesita exactamente cuando es una raíz o el error proyectado de su padre supera
$\tau$. Así que la disposición sigue el error del padre (`lod_pages.slang`). La **banda** de un nodo es

$$b(n) = \mathrm{clamp}\big(\lfloor\log_2\varepsilon(\mathrm{par}(n))\rfloor+40,\ 0,\ 63\big),\qquad b(\text{root}) = 63 .$$

La clave de ordenación, en el orden propio de las pages, es (banda más gruesa primero, después primera leaf, después
el nodo más grande primero):

$$\text{hi} = (63-b)\ll22 \;|\; \text{bucket}\ll19 \;|\; \text{start}\gg6,\qquad \text{lo} = (\text{start}\,\&\,63)\ll26 \;|\; (2^{26}-1-\text{leaves}).$$

Una ordenación radix en la GPU ordena los nodos, y la page $p$ contiene las posiciones ordenadas $[pP,(p+1)P)$. Las
pages cruzan los límites de banda, para que una banda gruesa de pocos nodos no sea una page por sí sola. La banda de
un padre nunca es más fina que la de su hijo, su primera leaf nunca es posterior, y cuando ambas empatan el padre es
el nodo más grande. Por tanto un padre nunca se coloca después de su hijo. (Una disposición que recorre al revés las
leaves de cada banda pone 92 padres después de sus hijos en el capó.) `--tree-pages P` escribe, para cada nodo,
`lodPage`, y para cada page:

- `lodPageInfo` = (banda más gruesa, nodos, primer nodo en orden de page), y
- `lodPageBounds` = la caja que contiene las esferas de los **padres** (la necesidad de un nodo se mide desde su
  padre) y el error de padre mayor y menor.

El player proyecta el error mayor desde la caja y carga la page cuando esa proyección supera $\tau$. Los bits de
cubeta (`--pages-by-facing`: eje dominante de la normal y su signo) empeoran la residencia, 4.45× frente a 3.98× con
$\tau=2.5$, y están desactivados.

**Residencia.** Un recorrido necesita el cut y todos sus antepasados, alrededor de 2× el cut. Las pages contienen
alrededor de 4× el cut sea cual sea su tamaño u orden, porque los nodos necesarios y los descartados comparten cada
page mientras que el culling depende de la vista.

| configuración | escena | $\tau$ | cut | nodos necesarios | pages contienen (× cut) |
|---|---|---|---|---|---|
| P 3000 | seis piezas, cerca 1920 | 1 | 244 114 | — | 2.0–2.6× |
| P 3000 | seis piezas, lejos 393 | — | — | — | 2.2–4.9× (capó: 2 pages) |
| P 3000 | coche 640, cerca 393 | 2 | 293 629 | 647 707 (2.20×) | 1 170 000 (3.98×) |
| P 3000 | coche 640, cerca 393 | 2.5 | 263 918 | 580 997 (2.20×) | 1 050 000 (3.97×) |
| P 3000 | coche 640, cerca 393 | 3 | 198 702 | 445 195 (2.24×) | 957 000 (4.81×) |
| P 3000 | coche 640, cerca 393 | 6.75 | 92 987 | 214 827 (2.31×) | 423 000 (4.54×) |
| P 1000 / 500 | coche 640, cerca 393 | 2 | 293 629 | — | 3.91× / 3.87× |
| P 1000 / 500 | coche 640, cerca 393 | 2.5 | 263 918 | — | 3.77× / 3.60× |

`treeCulledMark` marca los nodos que los culls dejan fuera. El subárbol de un nodo descartado también queda
descartado, así que un recorrido se detiene ahí. Eso deja los nodos necesarios en 2.2× (un recorrido dentro de
subárboles descartados necesita ~4×), pero las pages siguen conteniendo 3.6–4.5×. Un player con streaming puede por
tanto guardar el recorrido a 12 B por nodo para lo que se necesita y la carga útil en tramos deflate de 64 para lo
que se dibuja; en el player WebGPU de §7, de cerca con $\tau=3$, eso supone ~60 MB y 226 k dibujadas a 60 fps
(emulado).

## 6.9 El close-up level

De cerca, el logo y la grille siguen blandos con el bake a 1280 px. Una cámara cercana llega a unos 0.5 mm por píxel
de salida, mientras que las leaves a 1280 px miden 3–4 mm. Un bake de logo y grille desde una cámara cercana a
5120 px muestra nítidos la bandera a cuadros, la pajarita y la malla metálica de la grille (108 527 → 404 594
Gaussians). El close-up level (car5120) vuelve a hacer el bake del coche por grupos (figura 9):

- **Piezas detalladas.** Las 66 piezas a las que el bake a 1280 px da como mucho 1.5 Gaussians por triángulo
  (emblemas, grilles, faros, pinzas, retrovisores, ruedas, limpiaparabrisas, molduras), a 5120 px, en 7 grupos
  (`fine0–6`).
- **Paneles.** Cada pieza a 5120 px, o tan fina como la mantenga bajo un tope de 4.5 M leaves (un árbol de 5.01 M
  leaves está en el límite de la memoria del dispositivo al escribirse), con las pequeñas agrupadas hasta 4 M. Hace
  falta `--max-cells 1048576`: a 4736 px, Infinite_Black pierde 359 628 cells más allá de las 262 144 por triángulo
  por defecto, lo que deja agujeros.

Cada grupo se convierte con todas las demás piezas ocultas, pero el bake traza el escenario entero, así que una pieza
conserva las sombras de las otras. Los árboles son por pieza, así que el close-up level tiene el mismo formato que
car1280: las raíces gruesas se quedan, y los niveles más profundos solo se paginan de cerca.

| grupos | leaves | resolución |
|---|---|---|
| fine0–6 (66 piezas detalladas) | 18 610 627 | 5120 |
| puerta (piloto) | 4 084 170 | 5120 |
| Bottom | 4 897 428 | 3392 |
| Part_3 | 4 370 459 | 4480 |
| Infinite_Black | 4 481 738 | 4736 |
| Panel_under_Door_001, Windshield, Front_Fender, Hood, Front_Bumper, Part_2, Part_23 | 15 994 566 | 5120 |
| pan0–4 (paneles pequeños) | 17 088 152 | 5120 |
| Front_Bumper_001 | 5 008 986 | 4480 |
| **24 árboles** | **74 536 126 leaves, 74 328 031 merges, 208 095 raíces** | |

Los 24 árboles contienen 148 864 157 nodos en 49 634 pages de 3000, y 87.4 GB en disco. Rehechos a partir de sus
leaves con los segundos momentos respecto al propio centro de cada grupo (§6.2.1), y con un codebook de oclusión
compartido de 8181 entradas (el 11.9 % de las celdas reabiertas), el corte del coche a 0.25 px dibuja 11.7 M de Gaussians
en lugar de 16.9 M y da 24.00 dB en el frame y 18.31 dB en el coche frente al path trace (antes 23.62 y 17.91). Algunos árboles conservan
muchas raíces sueltas (fine5: 29 213; Infinite_Black: 92 773), porque los merges nunca cruzan piezas ni guardas, por
diseño. El empaquetador del player (§7.5.4) añade encima una raíz sintética de varios niveles, ya que un fichero
contiene unas 3 000 raíces.

**Memoria para árboles de 4–5 M leaves.** Una rueda delantera con 4.45 M leaves y 8.88 M nodos (solo el neumático
2.56 M) construye su árbol dentro de la memoria del dispositivo, pero escribirlo requiere dos cambios.

1. `mergeAttribute` necesita un buffer de pesos solo para los joints de un rig. Reservado tan grande como cada
   atributo, cuesta 1.7 GB junto a cada uno de los arrays de 48 floats de un transfer sobre 8.9 M nodos. Es una
   palabra para cualquier otro atributo.
2. La escritura no debe tener cada array tres veces en memoria unificada (lo leído, una copia, y los arrays
   fusionados empezados como otra copia). Los arrays fusionados empiezan solo desde los metadatos, y la escritura del
   árbol, el último paso, libera cada array en cuanto está en el dispositivo.

Con estos cambios, una puerta de 4.08 M leaves tiene un pico de 10.5 GB residentes y se escriben 4.90 M leaves.
Front_Bumper_001 (5.01 M) se escribe con 2.23 GB de memoria libre en el punto más bajo; el tope de 4.5 M para los
paneles mantiene ese margen.

**Oclusión escrita en el sitio.** Con 87.4 GB para los 24 árboles, una copia de cada árbol no cabe junto a ellos en
disco. `occlusionMasksShared` escribe cada árbol en `<tree>.occlusion.usdc` a su lado y lo renombra sobre el original.
El codebook compartido (8 182 entradas) tarda 645 s, con un pico de conjunto residente de 10.2 GB; por árbol reabre
del 0 al 23.6 % de las cells cerradas (§6.7).

## 6.10 Resultados medidos

Todos los renders son del rasterizer del motor. «Leaves a 4×» significa todas las leaves dibujadas a 4× de resolución
y reducidas por caja, la referencia para cuts lejanos (un bake a 1920 px dibujado a 393 px se transparenta). «PT» es
el path trace del mesh a 1920×1080 desde `/World/Camera`, comparado en el frame y en el recorte de 900×420 del coche.

**El coche a 640 px a 393 px, frente a las leaves a 4×.**

| configuración | cut | nodos cerca | dB cerca | nodos lejos | dB lejos |
|---|---|---|---|---|---|
| árboles con coverage | error 1 px, cotas | 666 585 | 33.29 | 343 541 | 39.15 |
| árboles con masa | error 1 px, cotas | 666 585 | 33.27 | — | — |
| árboles con coverage | $\tau=0.5$ | 930 119 | 33.17 | — | — |
| + back-face culling | closed shells por la regla del 2 % | 522 931 | 33.63 | — | — |
| + oclusión | 8192 repartidas por grupo | 360 333 | 33.58 | 195 049 | 39.53 |
| ambos culls | regla $O<2\sqrt{E}$ (102 piezas) | 378 028 | 33.53 | — | — |

El cut lejano (195 k) cabe en un presupuesto de ~250 k nodos, el que el player móvil de §7 dibuja a 30 fps. El cut
cercano es ~1.4× eso.

**Frente al path trace (1920 px).**

| bake | qué | nodos | recorte del coche | frame |
|---|---|---|---|---|
| 640 px | todas las leaves | 5.38 M | 16.37 | 21.92 |
| 640 px | error 1 px, ambos culls | 675 760 | 16.28 (31.91 frente a leaves) | 21.80 |
| 640 px | error 0.25 px, back-face culling | 2 313 541 | 16.29 (34.01 frente a leaves) | 21.81 |
| 640 px | leaves, sharp lobes sin ensanchar | 5.38 M | 16.58 | 22.15 |
| 640 px | 0.25 px, back-face culling, correcciones de sombreado | — | 16.55 | 22.18 |
| 640 px | leaves (0.01 px), sin cull / con cull | 4.35 M / 3.70 M | 16.73 / 16.75 | 22.40 / 22.43 |
| 640 px | grupo 0 con spill $w_s=2$ | 561 242 (g0) | 16.79 | 22.38 |
| **1280 px** | 0.25 px, back-face culling | 4 998 976 | **17.88** | **23.90** |
| 1280 px | 0.25 px, back-face culling, separación tubo/fold (§4.9.4) | — | 17.88 | 23.91 |

El LOD cuesta alrededor de 0.1 dB frente al path trace (16.37 → 16.28 con 1/8 de las leaves). El bake explica el
resto: su resolución, sus faros y la nitidez de sus reflejos. El nuevo bake a 1280 px (8.69 M leaves, 8.67 M merges,
20 723 raíces, 5 789 pages, la oclusión reabriendo un 12.3 %) gana 1.3 dB en el coche. Frente al cut de 0.25 px a
640 px con las correcciones de sombreado, el cut a 1280 px lleva los agujeros de la grille de 2.24 a 1.04 del path
trace, los brillos del capó de 3.2 a 2.0 px de desvío, y las juntas de 0.90 a 1.02 del path trace.

## 6.11 Límites y cuestiones abiertas

- **Agujeros solo de proyección.** Un cut grueso de tiras subpíxel (la grille) llena agujeros que ninguna medida por
  nodo e independiente de la vista puede ver (§6.5). En la práctica se resuelve con leaves más finas, no con el cut.
- **Ponderación de la oclusión.** El codebook se reparte por entradas usadas, no por la frecuencia con que una vista
  descarta un nodo. Ponderar por una órbita de vistas en el bake debería subir el culling por encima del 84 % del
  ideal.
- **Residencia en ~4×.** La limitan pages que mezclan nodos necesarios y nodos descartados por la vista. Un player con
  streaming separa el recorrido de la carga útil en vez de encoger las pages.
- **Merges de vidrio.** Cada Gaussian de lámina fina se eleva al suelo de lámina de 0.1, así que fusionar en menos
  láminas más grandes hace variar la transmisión de cell a cell. Mientras tanto Spark declara los merges de lámina a
  4× su tamaño, y queda pendiente un suelo por píxel en lugar de por Gaussian.
- **Las molduras finas y lejanas** siguen convirtiéndose en trazos gruesos en todos los cuts. Las cotas no ven ese
  error.
- **Sin portar.** La regla de caras traseras en closed shells y el cut por error viven en `athenea-pieces`. El
  `CutSelector` del motor principal no ha cambiado.
