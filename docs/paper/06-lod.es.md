<!-- Copyright (c) 2026 jesus luque. -->

# 6. Niveles de detalle: árboles de fusión, cortes por error, páginas y oclusión

Una nube horneada desde una malla a 1280 px contiene 8.7 millones de gaussianas para un solo coche. El nivel de
primer plano, horneado a 5120 px, contiene 74.5 millones. El presupuesto del teléfono, fijado por el usuario, es de
50–60 MB residentes y unas 250 k gaussianas dibujadas a 30 fps. Cada cifra entre esos límites depende de la
estructura de niveles de detalle. Esta sección describe primero el nivel de detalle original del motor (§6.1), hecho
para escenas capturadas, y explica por qué no cumple ese presupuesto. Después trata lo que lo sustituyó en el trabajo
por piezas: árboles de fusión por pares por pieza (§6.2), un único corte por un error de pantalla acotado (§6.3), la
opacidad que recibe una fusión (§6.4) y un intento fallido de mejorarla (§6.5), dos descartes dependientes de la
vista (§6.6, §6.7), páginas ordenadas por error (§6.8), el nivel de primer plano a 5120 px (§6.9) y las medidas
(§6.10). Todo kernel nombrado aquí es un kernel de cómputo en Slang. La CPU solo lee ficheros y lleva la contabilidad
(§2).

Notación. Un nodo $n$ es una hoja (una gaussiana horneada) o una fusión. $\mu_n$ es su centro y $\Sigma_n$ su
covarianza, con escalas $s_n = (s_x, s_y, s_z)$ y opacidad $o_n$. $\mathrm{par}(n)$ es su padre. $\tau$ es el umbral
en píxeles, $f$ es la focal en píxeles a distancia unidad (`pixelsPerUnit`) y $e$ es el ojo.

## 6.1 El nivel de detalle original del motor

**Octree por código de Morton.** `LodBuilder` (`modules/lod`, `shaders/athenea/lod/lod_common.slang`) ordena los
splats por un código de Morton de 30 bits. La celda del octree de un splat en el nivel $r$ son los $3r$ bits altos de
su código, así que las celdas de un nivel son tramos de prefijos iguales. Una pasada de fronteras y una suma prefija
las encuentran, sin recorrer ningún árbol. Cada celda lleva momentos, siguiendo a Kerbl et al. 2024. Con
$w = o \cdot A_2(s)$, donde $A_2$ es el área de los dos ejes más largos,

$$W=\sum w,\quad S_1=\sum w\,\mu,\quad S_2=\sum w\,(\Sigma+\mu\mu^\top),\quad C=\sum w\,c .$$

Los momentos se suman, así que cada nivel más grueso se suma desde sus hijos. Una celda se convierte en una gaussiana
con $\mu=S_1/W$, $\Sigma=S_2/W-\mu\mu^\top$ (barridos de Jacobi en el shader), $c=C/W$ y $o=\min(W/A_2,0.99)$. Los
niveles guardados van desde `coarsestLevel` hasta el nivel más profundo cuyo número de celdas es como mucho la mitad
del de splats.

**Corte.** `CutSelector` dibuja un grupo cuando la arista de su celda, proyectada a la distancia más cercana de la
celda, es como mucho $\tau$ y la proyección de la celda padre supera $\tau$. Una celda hija está dentro de su padre,
así que la prueba es monótona a lo largo de cualquier línea de la raíz a una hoja, y cada lugar se dibuja en
exactamente un nivel. Lo único que se lee de vuelta es un recuento por nivel y una lectura por instancia y fotograma.

**Chunks, `.athc` y streaming.** Los splats se cortan en chunks de 65 536 splats consecutivos en orden de Morton. Los
niveles fusionados se quedan en el dispositivo, y los chunks se cargan en streaming. El formato `.athc` (`.lrtc` en el
árbol de lucabRTrender) tiene una cabecera de 104 B, tablas de niveles y chunks, y bloques alineados a 4 KiB para que
un chunk pueda mapearse y traerse por fallo de página por sí solo. `StreamingPool` mapea el fichero. Dos hilos de carga
copian los chunks pedidos, los más pedidos primero. Cada ranura se toma de una ranura libre, del chunk pedido hace más
tiempo o de un chunk pedido menos de la mitad. Un grupo del nivel más fino al que le faltan chunks dibuja su propia
gaussiana fusionada, así que la imagen nunca tiene un agujero. En train_7k (742 k splats, M5 Pro), la construcción
tardó 47 ms y produjo 146 k gaussianas fusionadas. A 8 px el corte dibujó 83 k splats (11 %), con un render de 3.3 ms
frente a 12.6 ms y un error absoluto medio de $3\cdot10^{-3}$. En train_30k, el `.athc` cargado en streaming en 4 de
16 ranuras escribió EXR idénticos byte a byte y se asentó en 2 cortes.

**Por qué se sustituyó para el teléfono.** La estructura se mapeó el 10 oct (`PIECES-PLAN.md`), y cuatro propiedades
no encajaban con el objetivo.

1. El umbral se fija por instancia y no es un presupuesto. Cada instancia cuesta dos envíos bloqueantes y una lectura
   por fotograma, y no se comparte ningún presupuesto entre instancias.
2. Las celdas del octree ignoran el objeto. Una celda fusiona lo que caiga dentro, incluido un emblema con el panel
   bajo él y el vidrio con la pintura.
3. Un splat en streaming cuesta 132 B, frente al objetivo de ~55 B.
4. Solo los splats van en chunks. Los niveles fusionados, hasta uno con hasta la mitad de celdas que splats, están
   siempre residentes, y para millones de hojas ellos solos superan el presupuesto.

## 6.2 Árboles de fusión por pares por pieza

`athenea decimate --tree` construye un árbol binario de fusiones por fusión de pares de vecinos más cercanos mutuos.
Es NanoGS (Xiong et al. 2026) en el dispositivo (`lod_pairs.slang`, `lod_pairs_final.slang`, `Lod.cpp`), con guardas
adecuadas a nubes horneadas con material.

### 6.2.1 Los nodos como momentos

Un nodo es un vector de 24 floats: $W$, $W\mu$, $W(\Sigma+\mu\mu^\top)$ (6), $W\cdot\mathrm{sRGB}(c)$, $W n$,
$W\alpha$, $W\cdot$metálico, $W\cdot$transmisión, $W\cdot$especular, la clave, el número de registros, la razón de
grosor ponderada y el índice del nodo en el árbol. Aquí $w = o\cdot\max(s_xs_y, s_ys_z, s_xs_z)$ (`discArea`), y
$\alpha$ es la rugosidad GGX al cuadrado. Una fusión suma los dos vectores. Como las sumas son asociativas, los
momentos de un nodo son los de todas sus hojas sea cual sea el orden de las fusiones. Todos los demás atributos
(transfer, bits de sombra, material, articulaciones) se fusionan una sola vez al final, ponderados por masa sobre el
tramo de hojas del nodo (`lodMergeAttribute`).

La gaussiana fusionada (`pairFinalize`) toma $\mu$ y $\Sigma$ ajustadas por momentos, descompuestas en autovectores
por Jacobi con el eje más delgado en tercer lugar. Fusionar por momentos conserva el segundo momento. Un bloque de
$2\times2$ discos tan anchos como su separación se fusiona por tanto en un disco de 0.58 de esa separación, lo que deja
agujeros visibles de cerca. A lo largo de cada eje ancho $u$, la fusión se ensancha así. Sea $v_u$ la varianza
ponderada por área de los centros de los miembros, $c_u$ su anchura media y $d^2 = 12\sqrt{v_uv_v}/(n-1)$ la
separación de una rejilla de $n$ miembros sobre esa dispersión. Entonces

$$s_u' = \mathrm{clamp}\!\left(\frac{c_u}{d}\sqrt{12v_u+d^2},\; s_u,\; 2s_u\right).$$

La opacidad, con la regla por defecto, es la media de los miembros ponderada por área, limitada a 0.99. La cobertura
de masa $W/A(s')$ se guarda como `coverageMass`. Una fusión de discos cuyo eje más delgado es menor que 0.12 de su eje
medio se escribe como surfel (escala más delgada exactamente 0).

### 6.2.2 El coste

Para los nodos $a, b$ con $\pi = W_a/(W_a+W_b)$ y $\Sigma$ fusionada, el término de forma es la cota de Runnalls de la
divergencia KL por unidad de masa. El área del disco sustituye al volumen, que un surfel no tiene:

$$c_{\text{shape}} = \max\!\Big(0,\ \tfrac12\big[\log I_2(\Sigma) - \pi\log I_2(\Sigma_a) - (1-\pi)\log I_2(\Sigma_b)\big]\Big),$$

donde $I_2$ es el segundo invariante (la suma de los productos de autovalores dos a dos, calculada sin autovalores).
El coste completo es

$$c = \sigma_{\text{px}}\big(c_{\text{shape}} + w_f\,\phi^2 + w_n(1-\cos\theta_{ab})(1+F_0/\alpha)\big) + w_c\,\lVert\Delta c_{\text{sRGB}}\rVert^2 + w_m\,\Delta_{\text{mat}},$$

donde $\phi$ es el grosor que la fusión añade sobre su miembro más grueso. El término $\sigma_{\text{px}} =
\mathrm{smoothstep}(0.5, 2, \text{px})$ anula los términos de forma de las fusiones que miden menos de medio píxel en
la cámara de la conversión. Valores por defecto: $w_c=20$, $w_n=1$, $w_m=4$, $w_f=4$.

### 6.2.3 Guardas

Un par cuyos miembros difieren en cualquiera de estas formas cuesta $\infty$ (`kNoMerge`):

| guarda | regla | por defecto |
|---|---|---|
| pieza | la clave difiere (prim, articulación, lámina, pared fina, metal Schlick) | siempre |
| vidrio | transmisión $>0.5$ en un solo lado | siempre |
| vidrio con vidrio | solo con `--merge-glass` | no |
| espejo | $\alpha<0.01$ en un solo lado | siempre |
| pliegue | $\cos\theta_{ab} < \cos 60^\circ$ | 60° |
| brillo | $\alpha<0.1$ y $1-\cos\theta_{ab} > \tfrac12\kappa^2\alpha$ | $\kappa=1$ |
| borde de color | $\lVert\Delta c_{\text{sRGB}}\rVert > 0.15$ | 0.15 |
| aguja | aspecto del plano por encima de `maxAspect` y de 1.25× el de los miembros | 1000 (desactivada) |
| trazo | fusión más ancha que `maxPixels` en la cámara | no |

### 6.2.4 Pasadas y la cima relajada

En cada pasada, cada nodo calcula el coste de fusionarse con los 8 vecinos a cada lado a lo largo de un orden de
Morton de los centros. Después repite esto a lo largo de un segundo orden desplazado media celda fina, y se queda con
la pareja más barata. Los pares que se eligieron mutuamente se reparten en un histograma de $\log_{10}$ del coste, y
el host elige el umbral de coste que fusiona tantos pares como permite la pasada, que es como mucho la mitad de los
nodos. La fusión se detiene en `treeRatio` (4096×) o cuando ya nada puede fusionarse. Con `relaxTop`, la fusión sigue
después con el borde de color (2.0), el ángulo, la guarda de vidrio entre sí, el aspecto y el límite de píxeles
levantados. Las guardas de clave, de vidrio con no vidrio y de espejo se mantienen. Cada una de esas fusiones guarda
su coste **negado**, lo que la marca como fusión más allá de las guardas. Esto da a una vista lejana niveles gruesos
a los que cortar: una tapa de faro pasó de 76 486 raíces a 123 (3d7ee66).

**Disposición.** El árbol se escribe como un ParticleField. Las hojas van primero, en orden de profundidad, de modo
que cada nodo es un tramo contiguo de hojas. Las fusiones siguen en el orden en que se hicieron. Cada nodo lleva
`lodParent` (una raíz es su propio padre), `lodStart` (su primera hoja), `lodSize` (su número de hojas) y `lodCost`.

### 6.2.5 Muchas piezas: `--passes`, `--merge-glass`, `--object-groups`

En el Corvette entero (5.38 M hojas), construir el árbol de una vez funcionó, pero escribir sus atributos agotó la
memoria del dispositivo. `--object-groups K --object-group i` conserva las gaussianas cuyo `cryptoObject` módulo $K$
es $i$. Las fusiones nunca cruzan piezas, así que el árbol de cada grupo es exactamente el subárbol que el asset
entero tendría para esas piezas. Cada pasada fusiona como mucho la mitad de lo que queda, y un grupo de muchas piezas
pequeñas fusiona pocos pares a la vez. El tope por defecto de 60 pasadas dejó un grupo con 116 549 raíces de 323 613
hojas. Con `--passes 1000` llegó a 1 298 raíces en 310 pasadas (94fa9f1). `--tree-report` mostró dos causas de las
raíces restantes: el vidrio macizo nunca se fusionaba (raíces igual a hojas), y los neumáticos seguían fusionando un
goteo por pasada hasta el tope.

| grupo (coche a 640 px) | raíces antes | con `--merge-glass --passes 4000` |
|---|---|---|
| grupo 2 | 190 490 | 4 237 |
| grupo 5 | 36 508 | 817 |
| coche entero (8 grupos) | — | 17 133 raíces, 5.38 M hojas, 3 582 páginas |

## 6.3 El corte por error, con cotas

El primer corte (`--cut-pixels`, 655c027) conservaba un nodo cuando su anchura en pantalla $2\max(s)f/d$ era como
mucho $\tau$ y la de ningún antepasado lo era. Las fusiones más allá de las guardas solo pasaban a $0.25\tau$. Ese
corte sigue disponible, pero lo sustituyó un corte por error (2bb1b80, e13b08b; investigación 100, a partir de
Nanite).

**Error propio.** Sea $r_n=\lVert s_n\rVert$ (la raíz cuadrada de $\mathrm{tr}\,\Sigma_n$) y $\alpha_n =
\max(\text{rugosidad},0.02)^2$. El error propio de una fusión, en metros, es

$$\varepsilon_0(n) = r_n\Big(|\mathrm{cost}_n| + w_c\,s_c(n) + w_n\,\frac{\theta(n)^2}{\alpha_n} + w_s\max(E_n-E_0,0)\Big),\qquad w_c=10,\ w_n=1,$$

y $\varepsilon_0=0$ para una hoja. Las fusiones más allá de las guardas ($\mathrm{cost}_n<0$) no se miden por su
coste, que no ve el moiré que causan en el vidrio ni el color que arrastran a través de un borde. Por eso se acotan a

$$\varepsilon_0(n) \leftarrow \max\big(g\,\varepsilon_0(n),\ 2r_n\big),\qquad g=4\ (\texttt{--cut-guard}).$$

El término de derrame $w_s$ es 0 por defecto (§6.5).

**Cotas.** El coste de una fusión solo cubre el último par que unió. Una cadena de fusiones, cada una dentro del borde
de color, puede derivar a través de ese borde. Por eso una dispersión de color $s_c$ y un cono de normales $\theta$
se suben por el árbol con la desigualdad triangular (`treeBoundsUp`), empezando en 0 en las hojas:

$$s_c(p) = \max_{k\in\mathrm{ch}(p)}\big(s_c(k)+\lVert c_k-c_p\rVert\big),\qquad \theta(p) = \max_{k}\big(\theta(k)+\angle(n_k,n_p)\big).$$

Ambas son por tanto cotas sobre cada hoja bajo $p$.

**Error monótono y esferas que cubren.** El error se hace monótono, y cada nodo recibe una esfera que contiene las
esferas de sus hijos:

$$\varepsilon(n) = \max\big(\varepsilon_0(n),\ \max_{k}\varepsilon(k)\big),\qquad \rho(n)=\max_k\big(\lVert\mu_n-\mu_k\rVert+\rho(k)\big),\quad \rho(\text{leaf})=2\max(s).$$

Cada subida es un kernel lanzado sobre todos los nodos, en el que cada nodo eleva a su padre con `InterlockedMax` sobre
los bits del float. El kernel se relanza hasta que un contador lee cero, lo que da la profundidad del árbol en rondas.
El error se proyecta en el punto más cercano de la esfera:

$$P(n) = \frac{\varepsilon(n)\,f}{\max(\lVert\mu_n-e\rVert-\rho(n),\ 10^{-3})}.$$

**Proposición.** Si $P(n)\le\tau<P(\mathrm{par}(n))$ selecciona $n$ (a una raíz le basta la primera desigualdad),
entonces cada hoja se dibuja exactamente una vez. *Demostración.* $\varepsilon$ no decrece al subir. La esfera del
padre contiene la del hijo, así que la distancia a la esfera decrece al subir. Por tanto $P$ no decrece a lo largo de
ningún camino de una hoja a la raíz. Las hojas tienen $P=0\le\tau$, así que el predicado «$P\le\tau$» se cumple en un
prefijo del camino que empieza en la hoja, y exactamente un nodo es el último de ese prefijo. $\square$

Así que el corte es una sola prueba paralela por nodo (`treeErrorKeep`), sin subir por los antepasados. `--cut-budget
N` biseca $\log\tau$ en $[10^{-4},10^4]$ en 40 pasos para hallar el umbral más pequeño cuyo corte tiene como mucho $N$
nodos.

| prueba | escena | corte por tamaño | corte por error |
|---|---|---|---|
| T15 | seis piezas, lejos, 393 px, 0.25 px | 83 k, 34.4 dB | 34 k, 34.9 dB |
| T18 | seis piezas, cerca, las 426 k de investigación | 42.92 / 31.22 dB (fotograma / rejilla) | 43.61 / 34.50 dB (prototipo 43.61 / 34.47) |

## 6.4 La opacidad de una fusión: cobertura por mínimos cuadrados

La masa sobre la elipse de momentos cuenta dos veces las colas solapadas de las hojas. Investigación midió el efecto en
un 27 % en la mediana y 2.3× en el p10. En la rejilla a 393 px, las fusiones salían más oscuras que sus hojas.
`--cut-coverage` (0ab577e) y después `--tree-coverage` (73c0ec2, f980e42) fijan la opacidad de cada fusión a partir de
lo que sus hojas cubren de verdad, así:

1. El plano del nodo lo generan sus dos ejes más anchos $u,v$, con escalas $s_u,s_v$. Una rejilla de $8\times8$ cubre
   el núcleo $[-1.25,1.25]^2$ en unidades de $(s_u,s_v)$.
2. Cada hoja $l$ bajo el nodo (su tramo, hasta 4096 hojas al escribir el árbol y 2048 en un corte) se proyecta sobre
   el plano. Su covarianza en el plano es $(u^\top\Sigma_lu,\ u^\top\Sigma_lv,\ v^\top\Sigma_lv)$ y el desplazamiento
   de su centro es $m_l$. Un tramo más largo conserva su opacidad basada en la masa. Las hojas se componen como lo
   haría el rasterizador: $C(x)=1-\prod_l\big(1-\min(o_l,0.99)\,G_l(x)\big)$.
3. La huella propia del nodo en la rejilla es $G(x)=\exp(-\tfrac12\lVert x\rVert^2)$. La opacidad es el ajuste por
   mínimos cuadrados de $aG$ a $C$:

$$a^\star = \frac{\sum_x G(x)\,C(x)}{\sum_x G(x)^2},\qquad o_n = \mathrm{clamp}(a^\star,0,0.99).$$

Se midieron dos alternativas antes de quedarse con esta forma. Normalizar por $\sum G$ en vez de $\sum G^2$ daba 0.5
para una fusión tan ancha como una hoja opaca. Un ajuste sobre $\pm2\sigma$ en vez de $\pm1.25\sigma$ daba 0.84 a las
fusiones de un panel, porque el borde es donde los vecinos llevan la superficie, y un guardabarros salía moteado. Las
hojas de canto respecto al plano, como los alambres de una rejilla vistos de frente, no cubren nada, así que la
fusión va a 0. Un cargador descarta todo lo que esté por debajo de 1/255, lo que rompería el árbol, así que una fusión
se escribe como mínimo a

$$o_n \ge 2/255\quad(\texttt{treeCoverageFloor}).$$

El árbol se reescribe en su propio orden de fichero. Solo cambian las opacidades de las fusiones
(`treeCoverageToFile`), porque `lodParent`/`lodStart` son índices del fichero y una reescritura anterior a través del
almacén en orden de Morton los había revuelto. Investigación encontró un error: el ajuste leía los ejes de un Blob
como columnas, la rotación inversa, y así giraba mal cada gaussiana. Las fusiones de un guardabarros salían a 0.59 de
su masa donde el ajuste de investigación daba 1.00 (4dce865).

| prueba | qué | cobertura | masa |
|---|---|---|---|
| T20 | rejilla, 393 px, corte al ajustar (48 k) | 39.79 dB (desde 33.87) | — ; hojas 41.59 |
| T21 | rejilla, 393 px, árbol reajustado (49.4 k) | 41.28 dB | 39.79 con 48.1 k (`--cut-coverage`) |
| T24f | seis piezas cerca a 1920, 1 px, cotas | 40.96 fotograma / 32.09 rejilla | árbol antiguo 41.11 / 30.09 |
| T47 | seis piezas cerca a 1920 (tras corregir los ejes) | **43.86** | 42.55 |
| T47 | seis piezas lejos a 393 | 39.95 | 39.66 |
| T47 | rejilla cerca a 393 | 39.71 | **41.16** |

La rejilla es la excepción. Las fusiones vecinas se solapan y se componen como $1-\prod$, así que un poco más de
opacidad oscurece todo el parche. El reproductor web (Spark) dibuja la masa de cada nodo
($\min(\text{coverageMass},0.99)$), y nuestro rasterizador conserva la cobertura.

## 6.5 La métrica de derrame, y por qué falló en la rejilla

En un corte grueso los agujeros de la rejilla se cerraban: en T66 la región de suelo cercano leía 2.24× el path trace
en el corte de 0.25 px, frente a 1.18 en las hojas. Investigación lo atribuyó a fusiones de dos hojas de las tiras
subpíxel de `Chrome_Part_Grill`, vistas de frente y de 3–6 px de ancho a $\tau=0.25$. Su coste es ~0 porque la
cobertura de un par, medida en su propio plano, no cambia. Se definió un derrame sobre la propia rejilla del ajuste de
cobertura (1bb79f4):

$$E_n = \frac{\sum_x \max\big(a G(x) - C(x),\,0\big)}{\sum_x aG(x)}\in[0,1],$$

es decir, la parte de lo que pinta la fusión que sus hojas no pintan. Coincidió con el numpy de investigación con una
correlación de 0.994, y se escribe como `lodSpill`. El corte suma $r\,w_s\max(E-E_0,0)$ al error propio.

| T67, grupo 0, 0.25 px | nodos | recorte del coche | fotograma |
|---|---|---|---|
| $w_s=0$ | 307 665 | 16.55 dB | 22.18 dB |
| $w_s=2$ | 561 242 (+82 %) | 16.79 | 22.38 |
| $w_s=5$ | 599 983 | 16.79 | 22.38 |
| $w_s=10$ | 612 963 (+99 %) | 16.79 | 22.38 |

El derrame sí cerró los agujeros: de 1.90 a 1.09 de las hojas en el render, y de 4.66× a 1.0 en el indicador de
investigación. El precio fue un 82–99 % más de nodos. Toda fusión derrama alrededor de 0.16, y las fusiones que llenan
los agujeros no derraman más que las de un panel, así que ningún suelo $E_0$ separa los dos grupos. El agujero es una
propiedad de la **proyección**: la fusión no cubre en 3D nada que sus vecinas no cubran, y el hueco solo se abre donde
las huellas en pantalla de las tiras se separan. Ninguna medida por nodo independiente de la vista lo ve. El valor por
defecto es $w_s=0$. El horneado a 1280 px (§6.10) resolvió el problema en los datos: los agujeros de la rejilla en el
corte pasaron de 2.24 a 1.04.

## 6.6 Descarte de caras traseras en cascos cerrados

Al tamaño del teléfono, un corte mostraba las caras traseras en negro. En la rejilla a 393 px, descartar las caras
traseras subió las hojas sin cortar de 38.2 a 41.6 dB (investigación). `--cut-cull-back` descarta un nodo cuyo cono
de normales entero mira hacia atrás:

$$\theta(n)<\tfrac{\pi}{2}\ \wedge\ n_n\cdot\frac{e-\mu_n}{\lVert e-\mu_n\rVert} < -\sin\theta(n).$$

El cono acota cada hoja bajo el nodo (§6.3), así que todo el subárbol del nodo mira también hacia atrás. Dos
observaciones fijan dónde se aplica. Aplicado a las seis piezas de prueba, con la tapa de vidrio del faro entre ellas,
costó 6 dB a 1920: el vidrio se ve a través, así que sus caras traseras son visibles. En el coche entero a 393 px
(T50), las puertas y paneles, que son dos pieles, mostraron una red de grietas oscuras: las fusiones de la piel
interior, mirando hacia atrás e iluminadas por el interior oscuro, se ordenaban por parches delante de la piel
exterior (T51: lo mismo con o sin cobertura; `--cut-guard 1000` no cambió nada). En un casco cerrado, lo que hay
detrás de una cara que mira hacia atrás es el interior de la propia pieza, así que descartarla es seguro y elimina la
red (T56). Por eso el descarte se aplica, a hojas y fusiones por igual, solo en **cascos cerrados que no son vidrio**
(`primvars:athenea:splat:closed`, transmisión $\le 0.5$).

**La regla de aristas abiertas.** `mesh2splat` cuenta las aristas no compartidas por exactamente dos triángulos, $O$.
Primero marcó una malla como cerrada cuando $O=0$ (40a7927) y después por debajo del 2 % de sus aristas (2b05ebb, 111
de 192 mallas). El borde de una lámina abierta crece como la raíz cuadrada de su área, así que un tubo fino pasaba la
regla de proporción: el barril de una llanta, al 1 %, se descartaba, y en el teléfono se veía luz entre los radios. La
regla que se mantiene es

$$\text{closed} \iff O < 2\sqrt{E},\qquad E = 1.5\,T,$$

para una malla de $T$ triángulos (f52b6b4). Con ella, las puertas (0), los logos (1.4) y las pinzas de freno (1.5) son
cerradas, y las llantas (4.0), los neumáticos (3.1), el capó (3.1) y el parachoques (4.4) no, lo que da 102 de las 192
mallas del Corvette. T66 (todas las hojas, 1920 px, frente al path trace): el descarte sube toda la puerta un ~12 % de
forma uniforme (puerta 1.01, suelo cercano 1.02 del path trace), con 4.35 M → 3.70 M gaussianas y el recorte del
coche 16.73 → 16.75 dB.

## 6.7 Máscaras de oclusión

Un transfer TX guarda 256 bits de sombra por gaussiana, uno por celda de una rejilla octaédrica de $16\times16$ sobre
la esfera del mundo, activado donde un rayo desde la gaussiana llegó al cielo. Una dirección cuyas celdas están todas
cerradas es una desde la que un ojo fuera del asset no puede ver la gaussiana. Los bits del sombreado no pueden
reutilizarse para esto: una fusión guarda los bits de sus hojas por mayoría, y el codebook del teléfono (`shadow_vq`)
también es una mayoría, así que ambos cerrarían direcciones desde las que se ve alguna hoja. `--occlusion-mask N`
(1000bdf, `occlusion_mask.slang`, `Occlusion.cpp`) construye una máscara conservadora por nodo.

**Máscaras.** La máscara $M_l$ de una hoja son sus celdas abiertas dilatadas una celda ($3\times3$), reflejadas a
través del borde del cuadrado octaédrico en el pliegue. Una celda son ~11°, y sin el margen el borde de un oclusor
cerraba direcciones desde las que el cromo de una rejilla aún se veía. La máscara está toda abierta en el vidrio
(transmisión $>0.5$, porque lo que hay detrás del vidrio se ve a través de él) y en una hoja sin bits. La máscara de
una fusión es el OR de las de sus hijos, subido hasta que ninguna máscara cambia:

$$M_p = \bigvee_{k\in\mathrm{ch}(p)} M_k .$$

**Codebook conservador.** Las máscaras se sustituyen por un índice a $N$ entradas de 256 bits cada una. La entrada 0
está toda abierta. El **valor** de un nodo es el número de celdas cerradas entre las *relevantes*, las direcciones
con $d\cdot\hat{up} > -0.25$ (un ojo hasta ~15° bajo el horizonte). Los nodos de los intervalos de menor
$\log_2$-valor, que juntos suman menos del 2 % del valor total, toman la entrada 0 y quedan fuera del ajuste. Las
demás entradas se siembran con nodos ajustados repartidos por el fichero. La ronda 1 asigna cada nodo a la entrada más
cercana en distancia de Hamming sobre las celdas relevantes. Las rondas 2–8 asignan un nodo solo a una entrada que
**contiene** su conjunto abierto, eligiendo la que tiene menos celdas relevantes abiertas. Tras cada ronda una entrada
pasa a ser el OR de sus miembros. Cada entrada es por tanto un superconjunto del conjunto abierto de cada miembro, y el
codebook nunca puede cerrar una dirección que un nodo tiene abierta. El coste del codebook son las celdas que reabre
($M_{\text{entry}}\wedge\lnot M_n$) como fracción de todas las celdas cerradas.

**Repartido por necesidad.** Los árboles de un asset comparten un codebook (`--occlusion-group`, 5be8710). Cada unidad
se ajusta sola al tamaño completo y se cuentan las entradas que usa. Después el codebook se reparte como
$\text{share}_u = 16 + \lfloor\text{spare}\cdot \text{used}_u/\sum\text{used}\rfloor$, y cada unidad se reajusta
dentro de su parte (1103f9f). Repartir por calidad de ajuste (T41) y por pieza (`--occlusion-by-part`, 0fde773) fue
peor en ambos casos:

| prueba | árboles | codebook | reabierto | descarte frente al ideal |
|---|---|---|---|---|
| T27 | 6 piezas, un codebook cada una | 256 | 3.2–10 % | — |
| T30 | 6 piezas, un ajuste sobre todo | 4096 / 1024 | 3.2 % / 6.5 % | — |
| T32 | + valor, solo celdas relevantes | 4096 / 1024 | 21.9 % / 24.7 % | — |
| T39 | valor = celdas cerradas, no hojas | 4096 | 14.7 % | — |
| T40 | repartido por necesidad | 4096 | 7.2 % (rejilla 2.8 %, 1 249 entradas) | 86 % cerca / 82 % lejos |
| coche 640 | 8 grupos por necesidad | 4096 | 16.2 % | — |
| coche 640 | 8 grupos por necesidad | 8192 | 11.9 % | 84 % |
| coche 640 | por pieza (192) | 4096 / 8192 | — | 66 % / 74 % |
| coche 1280 | 8 grupos por necesidad | 8187 | 12.3 % | — |
| coche 5120 | 24 árboles por necesidad | 8182 | 0–23.6 % por árbol | — |

(Desde T32, el ajuste cuenta solo las celdas relevantes, así que el total reabierto sobre *todas* las celdas sube.
Esos porcentajes no son comparables con T27–T30.) En el nivel de primer plano, el total informó «132.4 %» porque la
suma de celdas cerradas sobre ~150 M nodos desbordó un contador de 32 bits. La cifra de cada árbol y el propio
codebook eran correctos, y los totales se suman ahora en 64 bits (4a8f8c2). Los resultados por árbol fueron:
Windshield 0.0 % (vidrio, 1 entrada usada), Part_2 5.3 %, Hood 7.0 %, la puerta piloto 7.7 %, los grupos finos
8.9–15.9 % y el grupo de paneles pequeños pan2 23.6 %.

**La regla del reproductor** (y `occludedFrom` en el corte). Esta regla se aplica solo cuando el ojo está fuera de la
caja envolvente del asset. Sea $d=\mathrm{normalize}(e-\mu_n)$ y $q=16\,\mathrm{oct}(d)-0.5$. Las cuatro celdas
$\lfloor q\rfloor+\{0,1\}^2$, limitadas a $[0,15]^2$, son la huella bilineal de $2\times2$ hacia el ojo. El nodo se
descarta cuando las cuatro están cerradas en su entrada. Investigación midió −12 % cerca y −8 % lejos en el corte por
0.06 dB, con el coche como oclusor (nunca en una escena parcial). En el coche entero a 393 px, con descarte de caras
traseras, la oclusión llevó el corte cercano de 522 931 a 360 333 nodos.

## 6.8 Páginas por error

Un reproductor no puede tener un árbol entero. Necesita unidades que traiga enteras y que pruebe sin leer sus nodos.

**Por nivel falla.** Las páginas tomadas como cimas en anchura de subárboles hermanos seguían necesitando 114 de las
263 páginas del capó para un corte lejano. Las cotas conservan nodos pequeños dondequiera que gira un brillo o un
color, así que el corte abarca nodos desde hojas sueltas hasta $2^{11}$ hojas (e562899).

**Por error funciona.** Un nodo se necesita exactamente cuando es una raíz o el error proyectado de su padre supera
$\tau$. Así que la disposición sigue el error del padre (`lod_pages.slang`). La **banda** de un nodo es

$$b(n) = \mathrm{clamp}\big(\lfloor\log_2\varepsilon(\mathrm{par}(n))\rfloor+40,\ 0,\ 63\big),\qquad b(\text{root}) = 63 .$$

La clave de ordenación, en el orden propio de las páginas, es (banda más gruesa primero, después primera hoja,
después el nodo más grande primero):

$$\text{hi} = (63-b)\ll22 \;|\; \text{bucket}\ll19 \;|\; \text{start}\gg6,\qquad \text{lo} = (\text{start}\,\&\,63)\ll26 \;|\; (2^{26}-1-\text{leaves}).$$

Una ordenación radix en la GPU ordena los nodos, y la página $p$ contiene las posiciones ordenadas $[pP,(p+1)P)$. Las
páginas cruzan los límites de banda, para que una banda gruesa de pocos nodos no sea una página por sí sola. La banda
de un padre nunca es más fina que la de su hijo, su primera hoja nunca es posterior, y cuando ambas empatan el padre es
el nodo más grande. Por tanto un padre nunca se coloca después de su hijo. Una disposición anterior recorría al revés
las hojas de cada banda y ponía 92 padres después de sus hijos en el capó (e2c2046). `--tree-pages P` escribe, para
cada nodo, `lodPage`, y para cada página:

- `lodPageInfo` = (banda más gruesa, nodos, primer nodo en orden de página), y
- `lodPageBounds` = la caja que contiene las esferas de los **padres** (la necesidad de un nodo se mide desde su
  padre) y el error de padre mayor y menor.

El reproductor proyecta el error mayor desde la caja y carga la página cuando esa proyección supera $\tau$. Los bits
de cubeta (`--pages-by-facing`: eje dominante de la normal y su signo) empeoraron la residencia, 4.45× frente a 3.98×
con $\tau=2.5$, y están desactivados.

**Residencia.** Un recorrido necesita el corte y todos sus antepasados, alrededor de 2× el corte. Las páginas
contienen alrededor de 4× el corte sea cual sea su tamaño u orden, porque los nodos necesarios y los descartados
comparten cada página mientras que el descarte depende de la vista.

| prueba | escena | $\tau$ | corte | nodos necesarios | páginas contienen (× corte) |
|---|---|---|---|---|---|
| T38/T42 | seis piezas, cerca 1920, P 3000 | 1 | 244 114 | — | 2.0–2.6× |
| T38/T42 | seis piezas, lejos 393 | — | — | — | 2.2–4.9× (capó: 2 páginas) |
| T60 | coche 640, cerca 393, P 3000 | 2 | 293 629 | 647 707 (2.20×) | 1 170 000 (3.98×) |
| T60 | | 2.5 | 263 918 | 580 997 (2.20×) | 1 050 000 (3.97×) |
| T60 | | 3 | 198 702 | 445 195 (2.24×) | 957 000 (4.81×) |
| T60 | | 6.75 | 92 987 | 214 827 (2.31×) | 423 000 (4.54×) |
| T59 | igual, $\tau=2$, P 1000 / 500 | 2 | 293 629 | — | 3.91× / 3.87× |
| T61 | igual, $\tau=2.5$, P 1000 / 500 | 2.5 | 263 918 | — | 3.77× / 3.60× |

`treeCulledMark` marca los nodos que los descartes dejan fuera. El subárbol de un nodo descartado también queda
descartado, así que un recorrido se detiene ahí. Eso deja los nodos necesarios en 2.2× (un recorrido dentro de
subárboles descartados necesitaba ~4×), pero las páginas siguen conteniendo 3.6–4.5× (a16da43). Por eso el
reproductor del teléfono (v5, sparkwebGPU) guarda el recorrido a 12 B por nodo para lo que se necesita y la carga útil
en tramos deflate de 64 para lo que se dibuja. De cerca, con $\tau=3$, eso supuso ~60 MB y 226 k dibujadas a 60 fps
(emulado).

## 6.9 El nivel de primer plano

En el iPhone de cerca, el logo y la rejilla seguían blandos. La cámara llegaba a unos 0.5 mm por píxel del teléfono,
mientras que las hojas a 1280 px medían 3–4 mm. Una prueba de logo y rejilla horneada desde una cámara cercana a
5120 px mostró nítidos la bandera a cuadros, la pajarita y la malla de la rejilla (108 527 → 404 594 gaussianas). El
nivel de primer plano (car5120) vuelve a hornear el coche por grupos:

- **Piezas detalladas.** Las 66 piezas a las que el horneado a 1280 px dio como mucho 1.5 gaussianas por triángulo
  (emblemas, rejillas, faros, pinzas, retrovisores, ruedas, limpiaparabrisas, molduras), a 5120 px, en 7 grupos
  (`fine0–6`).
- **Paneles.** Cada pieza a 5120 px, o tan fina como la mantenga bajo un tope de hojas (5 M, luego 4.5 M después de
  que un árbol de 5.01 M no cupiera), con las pequeñas agrupadas hasta 4 M. Hizo falta `--max-cells 1048576`: a
  4736 px, Infinite_Black perdió 359 628 celdas más allá de las 262 144 por triángulo por defecto (agujeros) y se
  volvió a hornear.

Cada grupo se convierte con todas las demás piezas ocultas, pero el horneado traza el escenario entero, así que una
pieza conserva las sombras de las otras. Los árboles son por pieza, así que el nivel de primer plano tiene el mismo
formato que car1280: las raíces gruesas se quedan, y los niveles más profundos solo se paginan de cerca.

| grupos | hojas | resolución |
|---|---|---|
| fine0–6 (66 piezas detalladas) | 18 610 627 | 5120 |
| puerta (piloto) | 4 084 170 | 5120 |
| Bottom | 4 897 428 | 3392 |
| Part_3 | 4 370 459 | 4480 |
| Infinite_Black | 4 481 738 | 4736 |
| Panel_under_Door_001, Windshield, Front_Fender, Hood, Front_Bumper, Part_2, Part_23 | 15 994 566 | 5120 |
| pan0–4 (paneles pequeños) | 17 088 152 | 5120 |
| Front_Bumper_001 | 5 008 986 | 4480 |
| **24 árboles** | **74 536 126 hojas, 74 328 031 fusiones, 208 095 raíces** | |

Los 24 árboles contienen 148 864 157 nodos en 49 634 páginas de 3000, y 87.4 GB en disco. Algunos árboles conservan
muchas raíces sueltas (fine5: 29 213; Infinite_Black: 92 773), porque las fusiones nunca cruzan piezas ni guardas, a
propósito. El empaquetador del teléfono añade encima una raíz sintética de varios niveles, ya que un fichero contiene
unas 3 000 raíces.

**Memoria para árboles de 4–5 M hojas** (5ed0b6a). El primer grupo, una rueda delantera con 4.45 M hojas y 8.88 M
nodos (solo el neumático 2.56 M), construyó su árbol y después agotó la memoria del dispositivo al escribirlo. Se
encontraron dos causas.

1. `mergeAttribute` reservaba un búfer de pesos tan grande como cada atributo, aunque solo las articulaciones de un rig
   lo escriben. Eso eran 1.7 GB junto a cada uno de los arrays de 48 floats de un transfer sobre 8.9 M nodos. Ahora es
   una palabra en los demás casos.
2. La escritura tenía cada array tres veces en memoria unificada: lo leído, una copia, y los arrays fusionados
   empezaban como otra copia. Los arrays fusionados empiezan ahora solo desde los metadatos, y la escritura del árbol,
   el último paso, libera cada array en cuanto está en el dispositivo.

Tras la corrección, una puerta de 4.08 M hojas tuvo un pico de 10.5 GB residentes. Pasaron 4.90 M hojas. El primer
intento con Front_Bumper_001 (5.01 M) agotó la memoria, lo que llevó al tope de 4.5 M para los paneles posteriores.
Una ejecución posterior lo escribió, con 2.23 GB como la memoria libre más baja bajo `memguard`.

**Oclusión escrita en el sitio.** Los 24 árboles ocupan 87.4 GB (unos 86 GB era la estimación antes del último árbol), y una copia de cada uno no
cabría junto a ellos en
disco. `occlusionMasksShared` escribe cada árbol en `<tree>.occlusion.usdc` a su lado y lo renombra sobre el original.
El codebook compartido (8 182 entradas) tardó 645 s, con un pico de conjunto residente de 10.2 GB; por árbol reabre del 0 al 23.6 % de las celdas cerradas (§6.7).

## 6.10 Resultados medidos

Todos los renders son de nuestro rasterizador. «Hojas a 4×» significa todas las hojas dibujadas a 4× de resolución y
reducidas por caja, la referencia para cortes lejanos (un horneado a 1920 px dibujado a 393 px se transparenta). «PT»
es el path trace de la malla a 1920×1080 desde `/World/Camera`, comparado en el fotograma y en el recorte de 900×420
del coche.

**El coche a 640 px al tamaño del teléfono (393 px), frente a las hojas a 4×.**

| prueba | corte | nodos cerca | dB cerca | nodos lejos | dB lejos |
|---|---|---|---|---|---|
| T50 | error 1 px, cotas | 666 585 | 33.29 | 343 541 | 39.15 |
| T51 | igual, árboles antes de la cobertura (masa) | 666 585 | 33.27 | — | — |
| T52 | $\tau=0.5$ | 930 119 | 33.17 | — | — |
| T56 | + descarte de caras traseras en cascos cerrados (regla del 2 %) | 522 931 | 33.63 | — | — |
| T57 | + oclusión (8192 repartidas por grupo) | 360 333 | 33.58 | 195 049 | 39.53 |
| T58 | ambos descartes, regla $O<2\sqrt{E}$ (102 piezas) | 378 028 | 33.53 | — | — |

El corte lejano (195 k) cabe en las ~250 k del teléfono a 30 fps. El corte cercano es ~1.4× eso.

**Frente al path trace (1920 px).**

| prueba | qué | nodos | recorte del coche | fotograma |
|---|---|---|---|---|
| T63b | 640 px, todas las hojas | 5.38 M | 16.37 | 21.92 |
| T63b | error 1 px, ambos descartes | 675 760 | 16.28 (31.91 frente a hojas) | 21.80 |
| T63c | error 0.25 px, descarte de caras traseras | 2 313 541 | 16.29 (34.01 frente a hojas) | 21.81 |
| T64 | hojas, lóbulos nítidos sin ensanchar | 5.38 M | 16.58 | 22.15 |
| T65 cut_b | 0.25 px + correcciones de sombreado | — | 16.55 | 22.18 |
| T66 | hojas (0.01 px), sin descarte / con descarte | 4.35 M / 3.70 M | 16.73 / 16.75 | 22.40 / 22.43 |
| T67 | grupo 0 con derrame $w_s=2$ | 561 242 (g0) | 16.79 | 22.38 |
| T71 | **horneado a 1280 px**, 0.25 px, descarte de caras traseras | 4 998 976 | **17.88** | **23.90** |
| T75 | T71 + la separación tubo/pliegue (§4.9.4) | — | 17.88 | 23.91 |

El nivel de detalle cuesta alrededor de 0.1 dB frente al path trace (T63b: 16.37 → 16.28 con 1/8 de las hojas). El
horneado explica el resto: su resolución, sus faros y la nitidez de sus reflejos. El rehorneado a 1280 px (8.69 M
hojas, 8.67 M fusiones, 20 723 raíces, 5 789 páginas, la oclusión reabriendo un 12.3 %) ganó 1.3 dB en el coche.
Investigación, comparando T71 con cut_b: los agujeros de la rejilla en el corte pasaron de 2.24 a 1.04, los brillos
del capó de 3.2 a 2.0 px de desvío, y las juntas de 0.90 a 1.02 del path trace.

<!-- Figuras: /Users/muriel/luc/athenea-renders/pieces/car/t50/cut_Camera.png (T50 cerca, la red de grietas),
/Users/muriel/luc/athenea-renders/pieces/car/t50/cull_Camera.png (con descarte de caras traseras),
/Users/muriel/luc/athenea-renders/pieces/close/r_1280.png y r_5120.png (prueba del logo, 1280 frente a 5120 px),
/Users/muriel/luc/athenea-renders/pieces/close/close_mos.png -->

## 6.11 Límites y cuestiones abiertas

- **Agujeros solo de proyección.** Un corte grueso de tiras subpíxel (la rejilla) llena agujeros que ninguna medida
  por nodo e independiente de la vista puede ver (§6.5). En la práctica se resuelve con hojas más finas, no con el
  corte.
- **Ponderación de la oclusión.** El codebook se reparte por entradas usadas, no por la frecuencia con que una vista
  descarta un nodo. Ponderar por una órbita de vistas en el horneado debería subir el descarte por encima del 84 % del
  ideal.
- **Residencia en ~4×.** La limitan páginas que mezclan nodos necesarios y nodos descartados por la vista. El teléfono
  separa el recorrido de la carga útil en vez de encoger las páginas.
- **Fusiones de vidrio.** Cada gaussiana de lámina fina se eleva al suelo de lámina de 0.1, así que fusionar en menos
  láminas más grandes hace variar la transmisión de celda a celda. Mientras tanto Spark declara las fusiones de lámina
  a 4× su tamaño, y queda pendiente un suelo por píxel en lugar de por gaussiana.
- **Las molduras finas y lejanas** siguen convirtiéndose en trazos gruesos en todos los cortes. Las cotas no ven ese
  error.
- **Sin portar.** La regla de caras traseras en cascos cerrados y el corte por error viven en `athenea-pieces`. El
  `CutSelector` del motor principal no ha cambiado.
