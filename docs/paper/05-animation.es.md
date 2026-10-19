<!-- Copyright (c) 2026 jesus luque. -->

# 5. Assets animados: skeletons, skinning y visibilidad por partes (el peón y el gorrión)

Esta sección describe cómo athenea convierte un asset USD con rig (UsdSkel) en
una nube de Gaussians 3D que un skeleton sigue moviendo. Trata cómo se
convierte la nube, cómo se posa en la GPU en cada frame, cómo su luz y su
autosombra siguen siendo válidas bajo la deformación, cómo la tratan los
LOD y, como aplicación, cómo se reproduce en un navegador y en
un teléfono. Dos
assets recorren la sección. El gorrión molinero (`SparrowBird.usda`, 609
joints, 71 clips de animación) es el caso de prueba animado. El peón de The
Open Chess Set es el control rígido. Todas las cifras proceden de
`docs/decisions.md`, de los ficheros fuente que se nombran junto a ellas o de
los ficheros de medida del player web (`sparkwebGPU`). Las cifras que no se
midieron se indican como no medidas.

::: {.nota}
**UsdSkel** — el dominio de OpenUSD [43] que describe personajes articulados. Sus piezas: un prim `SkelRoot`, que delimita el subárbol donde se resuelve el skinning (fuera de un `SkelRoot` nada se deforma); un prim `Skeleton`, con el array `joints` (tokens que son rutas, p. ej. `Root/Spine/Chest`, de modo que la jerarquía está codificada en el propio nombre y no en prims hijos), `bindTransforms` (la transformación de mundo de cada joint en el instante en que la malla se ligó) y `restTransforms` (transformaciones *locales* de una pose de reposo, que no tiene por qué coincidir con la de bind); un prim `SkelAnimation`, con `translations`, `rotations`, `scales` por joint (locales, muestreados en el tiempo) y `blendShapeWeights`; y el esquema aplicado `SkelBindingAPI`, que liga una geometría a un skeleton con `primvars:skel:jointIndices`, `primvars:skel:jointWeights`, `primvars:skel:geomBindTransform`, `skel:joints`, `skel:skeleton` y `skel:skinningMethod`.

**Rig, skeleton, joint** — un *rig* es el conjunto de controles que deforma un modelo; en UsdSkel se reduce a un *skeleton*: un árbol de *joints* (articulaciones o «huesos»), cada uno con una transformación relativa a su padre. Componer las transformaciones locales desde la raíz da la transformación de mundo de cada joint en cada instante.

**Blendshapes** (morph targets) — la otra forma de deformación de UsdSkel: desplazamientos por vértice que se suman a la geometría de reposo ponderados por `blendShapeWeights`. Se usan para caras y correcciones finas. El motor los aplica a meshes (`geom::Skinner`, `shaders/athenea/geom/skinning.slang`, `skinPoints`), pero esta sección no los aplica a nubes: una nube solo lleva skinning por joints.

**Los dos assets** — el gorrión es un control *animado* con 609 joints (un número alto: cada pluma de ala y cola tiene su cadena de joints), lo que pone a prueba la escala del rig; el peón no tiene skeleton y sirve de control *rígido*: cualquier cambio de la ruta de animación debe dejarlo igual.

*Referencias:* [43] del §10; especificación UsdSkel: openusd.org/release/api/usd_skel_page_front.html; código: `modules/usd/src/MeshStage.cpp`, `modules/geom/include/athenea/geom/Skinner.h`; `docs/decisions.md` (secciones "Skinning and blend shapes, on the device", "A cloud carried by a skeleton, on the device").
:::

## 5.1 Principio de diseño: llevar el rig, no los frames

Hay dos maneras de animar una nube de Gaussians. La primera guarda la
posición, la orientación y la escala de cada Gaussian en cada frame. La
segunda guarda las Gaussians una vez, en la bind pose, junto con los joints
que mueven a cada una, más una transformación por joint y por frame. athenea
usa la segunda, por tres razones, todas medidas:

::: {.nota}
**Bind pose** — la pose del skeleton en la que se ligó la geometría: la malla está modelada en esa pose, y la matriz de skinning de un joint, $X_j(t)=W_j(t)\,B_j^{-1}$ (transformación de mundo del joint en $t$ por la inversa de su transformación de bind), es la identidad en ella. Por eso «guardar las Gaussians en la bind pose» significa guardarlas en el espacio donde las matrices de skinning se aplican sin más.

**Las dos alternativas** — (a) *vertex/point cache*: cada frame es una nube completa (lo que haría un Alembic de puntos o un PLY por frame, o los métodos 4D-GS que almacenan o predicen atributos por instante); (b) *rig*: una nube estática más una función paramétrica de deformación (linear blend skinning) controlada por pocas matrices por frame. Es la misma dicotomía que en producción entre exportar caché de geometría o exportar el rig/animación; aquí se elige (b) por los tres motivos que siguen.

*Referencias:* UsdSkel, "Schemas: Skeleton, bindTransforms"; código: `shaders/athenea/scene/splat_skin.slang` (cabecera, "What it is for"); `docs/decisions.md` (sección "The file carries the rig, not the frames").
:::

1. **Tamaño.** Para un personaje verosímil de 150 k Gaussians en 120 frames,
   los arrays por frame cuestan 960 MB en media precisión (1.2 GB en float).
   El rig (cuatro joints por Gaussian, 60 joints por frame) cuesta 2.9 MB. Se
   midió, tal como se escribe, un tubo sintético (200 anillos × 64 segmentos,
   40 joints, 184 320 Gaussians). Su fichero ocupa 9.27 MB con 25 instantes y
   9.50 MB con 121, frente a unos 713 MB para el mismo rango como arrays por
   frame. Cinco veces los frames cuestan 230 kB.
2. **Exactitud entre muestras.** Un rig es exacto en cualquier instante del
   rango. Una nube muestreada solo es tan buena como la interpolación del
   lector.
3. **Identidad estable.** La nube posada se escribe en buffers que la
   sobreviven, así que un renderer ve la misma nube de un frame al siguiente.
   Eso es lo que permite al ray tracer hacer *refit* de su estructura de
   aceleración en lugar de reconstruirla (§5.6).

::: {.nota}
**Las cifras de tamaño** — con $150\,000$ Gaussians y 120 frames hay $1.8\times10^7$ pares (Gaussian, frame). El coste del rig cuadra con índices u16 y pesos f16: $4\times(2+2)=16$ B por Gaussian dan $2.4$ MB, y $60\times120$ matrices $4\times4$ en float ($64$ B) dan $0.46$ MB: total $\approx2.9$ MB. Los 960 MB / 1.2 GB de los arrays por frame son la estimación de `docs/decisions.md`, que no detalla qué atributos cuenta; la cifra medida del tubo sí se deja reconstruir: $713\ \text{MB}/(184\,320\times121)\approx32$ B por Gaussian y frame, unos 16 halves (posición, rotación, escala y relleno). «Cinco veces los frames cuestan 230 kB»: pasar de 25 a 121 instantes añade 96 frames $\times$ 40 joints $\times$ 64 B $=245\,760$ B $\approx 240$ kB, en línea con los 230 kB medidos.

**Exactitud entre muestras** — LBS es una función continua de las matrices de joint; si `skinningXforms` se muestrea en el tiempo, USD interpola linealmente las matrices entre muestras (o el lector puede pedir cualquier instante del `SkelAnimation`), y la nube se evalúa *exactamente* con esas matrices. Una caché por frame obliga a interpolar posiciones y, sobre todo, rotaciones y covarianzas Gaussian a Gaussian, con artefactos (encogimiento al interpolar linealmente rotaciones, saltos de orden si la nube cambia).

**Identidad estable y refit** — un BVH (bounding volume hierarchy) construido sobre la nube conserva su topología si el número y orden de las primitivas no cambian; basta recalcular las cajas de abajo arriba (*refit*), algo mucho más barato que reconstruir (ordenar por Morton, construir la jerarquía). La técnica es la de Wald et al. (2007), "Ray Tracing Deformable Scenes Using Dynamic Bounding Volume Hierarchies", y Lauterbach et al. (2006), "RT-DEFORM".

*Referencias:* [32] del §10 (LBVH de Karras); código: `modules/render/src/GaussianRayTracer.cpp` (`GaussianRayTracer::sync`); `docs/decisions.md` (secciones "The file carries the rig, not the frames", "What a rigged cloud actually costs", "A posed cloud refits its ray tracing structure").
:::

El coste de esta elección es que todo lo que se hizo bake en espacio de mundo
(radiancia, transfer en spherical harmonics, visibilidad) deja de ser válido
en cuanto se mueve un miembro. De esto se ocupan §5.7 y §5.8.

::: {.nota}
**Bake en espacio de mundo** — la radiancia horneada (color ya iluminado), el transfer PRT en 9 spherical harmonics (bandas $l=0..2$, $9=\sum_{l=0}^{2}(2l+1)$ coeficientes) y la visibilidad se expresan en coordenadas de mundo. Si un ala gira $90^\circ$, sus Gaussians siguen «viendo» el cielo y las oclusiones de la pose de conversión: la luz queda pegada al espacio, no al miembro. Las dos soluciones que siguen separan el problema en dos: lo que gira *con* la Gaussian (transfer en su marco local, §5.7) y lo que depende de la posición relativa entre miembros (visibilidad por partes, §5.8).

*Referencias:* [14] (PRT, Sloan, Kautz, Snyder 2002) y [17] del §10; `docs/decisions.md` (sección "A skinned cloud keeps its transfer: zonal lobes in each gaussian's frame").
:::

## 5.2 Lectura de un stage con rig

El conversor (`athenea mesh2splat`) no pasa por Hydra. Lee los meshes con su
propio `MeshStage`, lo que deja dos maneras de tratar un mesh con skinning:

- **Posado.** Sin `--skinned`, el stage se posa antes de leerlo.
  `UsdSkelBakeSkinning` escribe los puntos posados como muestras de tiempo en
  una capa de override en memoria sobre la capa raíz, sobre
  `GfInterval(t, t)`. Esto cuesta una pose, deja intacto el fichero en disco
  y deja sin tocar un stage sin `SkelRoot`. Prueba: un cuadrado movido por un
  joint que se desliza (1.2, 0.4, 0) entre t = 0 y t = 1 se convierte en unos
  límites, plegados en el device, que difieren exactamente en ese
  deslizamiento.
- **Llevado.** Con `--skinned` se lee la **bind** pose y se omite
  `UsdSkelBakeSkinning`. Posar primero aplicaría el skinning dos veces a la
  geometría. Para cada mesh, `MeshStage` resuelve su skel binding con un
  `UsdSkelCache` sobre cada `SkelRoot` y un `UsdSkelSkinningQuery` por prim
  con skinning. Devuelve las influencias de los joints, `geomBindTransform`,
  la ruta del skeleton y el orden de joints del skeleton.

::: {.nota}
**`MeshStage` y por qué no Hydra** — el conversor lee `UsdGeomMesh` directamente con su propio lector (`modules/usd/src/MeshStage.cpp`, `modules/usd/include/athenea/usd/MeshStage.h`). Hydra resolvería el skinning (usdSkelImaging lo convierte en ext computations), pero el conversor necesita la geometría en una pose concreta, no un frame de render.

**`UsdSkelBakeSkinning`** — utilidad de UsdSkel que evalúa el skinning y escribe los puntos deformados como time samples en los propios meshes. La función `poseStage` lo llama dentro de un `UsdEditContext` sobre la **session layer** del stage (la «capa de override en memoria»: la capa más fuerte, que nunca se guarda en disco) y sobre el intervalo degenerado `GfInterval(t, t)`, de modo que solo se calcula un instante. Si el stage no tiene `SkelRoot`, la llamada no hace nada.

**`UsdSkelCache`, `UsdSkelSkinningQuery`, skel binding** — `UsdSkelCache` se puebla recorriendo cada `SkelRoot` y responde, para cada prim con `SkelBindingAPI`, un `UsdSkelSkinningQuery`: el objeto que sabe calcular las influencias (`ComputeVaryingJointInfluences`), su número por punto (`GetNumInfluencesPerComponent`), la `geomBindTransform` (`GetGeomBindTransform`) y el método (`GetSkinningMethod`). La **geomBindTransform** es la transformación de mundo de la malla en el momento del bind: lleva los puntos del espacio del mesh al espacio de bind en el que actúan las matrices de skinning.

**Por qué la bind pose con `--skinned`** — el skinner del motor aplica $X_j$ a puntos de bind. Si el conversor leyera puntos ya posados en $t$, el motor aplicaría $X_j(t)$ sobre ellos y la deformación se aplicaría dos veces.

**La prueba del cuadrado** — un joint que se traslada $(1.2,0.4,0)$ entre $t=0$ y $t=1$ desplaza rígidamente todo lo que lleva; los límites (AABB) de la nube convertida en $t=0$ y $t=1$, plegados con una reducción en la GPU, deben diferir exactamente en ese vector. Comprueba la ruta posada sin comparar imágenes.

*Referencias:* [43] del §10; API: `UsdSkelBakeSkinning`, `UsdSkelCache`, `UsdSkelSkinningQuery` (openusd.org/release/api/usd_skel_page_front.html); código: `modules/usd/src/MeshStage.cpp` (`poseStage`, `skinningOf`, `resolveSkinning`), `tests/usd/test_usd.cpp`; `docs/decisions.md` (secciones "The conversion reads a pose, not a rest", "A gaussian carries the joints its triangle carries").
:::

**Orden de los joints.** Un mesh puede enumerar su propio subconjunto de
joints en `skel:joints`. El mapper de UsdSkel solo traduce del orden del
skeleton al del mesh, así que se aplica sobre la identidad para saber el
índice de skeleton de cada índice de mesh. Las influencias se escriben siempre
en el orden del skeleton, de modo que una nube tiene un solo orden de joints,
sea cual sea el número de meshes de los que se convirtió. Un mesh de un stage
con skinning al que no mueve ningún joint (el stage del gorrión lleva un
cilindro y un plano junto al pájaro) recibe igualmente un registro de
influencias por Gaussian: cuatro joints de peso cero, que el skinner lee como
«quédate en la bind pose». Así el rig y la nube se mantienen alineados uno a
uno.

::: {.nota}
**`skel:joints` y el mapper** — un mesh puede declarar su propia lista de joints (un subconjunto del skeleton, en otro orden), y entonces sus `jointIndices` indexan esa lista local. UsdSkel expone un `UsdSkelAnimMapper` que traduce arrays *del orden del skeleton al del mesh*. Para obtener la dirección inversa (índice local → índice de skeleton), `skinningOf` remapea el array identidad $[0,1,\dots,n-1]$: el resultado en la posición $i$ local es el índice de skeleton de ese joint. Es un truco de reutilización de la API sin reimplementar la resolución de nombres.

**Un solo orden por nube** — como una nube puede salir de muchos meshes, cada uno con su subconjunto, escribir siempre en el orden del skeleton garantiza que un índice $j$ en la nube significa siempre lo mismo y que `skinningXforms` (una matriz por joint del skeleton) se indexa directamente.

**Pesos cero = bind pose** — en `splat_skin.slang`, si la suma de pesos es $\le 10^{-5}$ la Gaussian se trata como «llevada por nada» y se queda en $G\,p$, transformada por las mismas matrices de mundo. Mantener un registro por Gaussian aunque no se mueva evita arrays dispersos y mantiene la correspondencia 1:1 entre nube y rig.

*Referencias:* API `UsdSkelAnimMapper::Remap`; código: `modules/usd/src/MeshStage.cpp` (`skinningOf`), `shaders/athenea/scene/splat_skin.slang` (`splatSkin`, rama `!(total > kEps)`); `docs/decisions.md` (sección "The conversion's order is the mesh's, and a slot with nothing in it is kept").
:::

**`--skinned` fuerza `--no-bake`.** Un bake de radiancia guarda el entorno y
los bounces de la escena (el suelo bajo una pata), y llevarlo hacia arriba con
un miembro levantado equivale a girar un lightmap. Una nube con skinning lleva
su material y se hace relight en cada frame. Su transfer de luz, cuando se
pide, se guarda de una forma que gira con la Gaussian (§5.7).

::: {.nota}
**Por qué `--skinned` implica `--no-bake`** — el bake de radiancia de §3–4 escribe en cada Gaussian la luz que le llega en la escena *tal como está*: el suelo que rebota hacia la pata, el cielo visible, la sombra de otro objeto. Es un *lightmap* en sentido clásico. Al levantar el miembro, esa luz viaja con la Gaussian aunque su entorno haya cambiado: rotar un lightmap no lo vuelve correcto. Por eso la nube con skinning conserva sus parámetros de material (albedo, roughness, metallic…) y se ilumina (*relight*) en cada frame con las luces actuales. Solo la parte que depende de la geometría propia (el transfer de autosombra) se guarda, y en una forma que gira con la Gaussian.

*Referencias:* [14] del §10; código: `apps/athenea/src/CmdMesh2Splat.cpp` (opciones `--skinned`, `--transfer`); `docs/decisions.md` (sección "A skinned cloud keeps its transfer: zonal lobes in each gaussian's frame").
:::

## 5.3 Influencias por Gaussian, y sus gradientes

`mesh_pack.slang` escribe las cuatro influencias más pesadas de cada esquina
de triángulo en una segunda imagen con exactamente la misma disposición que
la del mesh, para que el efecto AOFX direccione las dos igual. El efecto sabe
dónde está cada Gaussian dentro del triángulo, así que mezcla allí las
esquinas (`blendInfluences`, `mesh2splat.slang`). Con coordenadas
baricéntricas $\lambda_a,\lambda_b,\lambda_c$ y pesos de esquina $w_{cj}$, el
peso bruto del joint $j$ es

$$\tilde w_j = \sum_{c\in\{a,b,c\}} w_{cj}\,\lambda_c .$$

Entran doce pares (joint, peso) y salen cuatro. Un joint ya presente suma el
peso, un hueco vacío recibe un joint nuevo y, si no, el nuevo sustituye al más
ligero. Elegidos los cuatro joints, cada peso conservado se recalcula como su
mezcla completa y los cuatro se renormalizan, $w_k = \tilde w_k / W$ con
$W=\sum_{i\in K}\tilde w_i$. Por tanto, una Gaussian siempre queda llevada por
completo.

::: {.nota}
**Skinning weights (pesos de influencia)** — cada vértice tiene una lista de pares (joint, peso) con pesos no negativos que suman uno; describen cuánto sigue el vértice a cada joint. En UsdSkel el número por vértice es el `elementSize` de `primvars:skel:jointIndices`/`jointWeights` (25 en el ala del gorrión).

**Coordenadas baricéntricas** — un punto $x$ del triángulo $(a,b,c)$ se escribe $x=\lambda_a a+\lambda_b b+\lambda_c c$ con $\lambda_a+\lambda_b+\lambda_c=1$. Interpolar los pesos de esquina con ellas es interpolar linealmente sobre el triángulo, exactamente lo que haría la rasterización de un atributo por vértice. Así el peso de una Gaussian coincide con el que tendría el punto de superficie del mesh en ese lugar.

**La reducción de doce a cuatro** — cada esquina aporta hasta cuatro pares (el `mesh_pack.slang` ya conservó los cuatro más pesados por esquina), es decir doce candidatos. `blendInfluences` los recorre: si el joint ya está en un hueco, suma; si hay hueco libre (peso 0), lo ocupa; si no, sustituye al más ligero solo si el candidato pesa más. Esta selección es voraz y dependiente del orden; por eso, al final, cada peso conservado se **recalcula** como su mezcla completa $\sum_c w_{cj}\lambda_c$ (no lo acumulado desde que entró en el hueco), y luego se renormaliza: $w_k=\tilde w_k/W$. La renormalización redistribuye proporcionalmente el peso de los joints descartados, de modo que la Gaussian se mueve «entera» aunque haya perdido influencias (el error resultante es de *posición*, ver Truncamiento).

**Por qué «la misma disposición que el mesh»** — la imagen de influencias tiene una entrada por esquina de triángulo, alineada con la imagen de triángulos que lee el efecto AOFX; así el efecto direcciona ambas con el mismo índice sin tablas auxiliares.

*Referencias:* [12] del §10 (mesh2splat de EA SEED); código: `shaders/athenea/usd/mesh_pack.slang` (campo `perPoint`, buffer `influences`), `plugins/mesh2splat/mesh2splat.slang` (`blendInfluences`); `docs/decisions.md` (sección "A gaussian carries the joints its triangle carries").
:::

**Gradientes.** $\tilde w_j$ es lineal sobre el triángulo, con
$\nabla\tilde w_j=\sum_c w_{cj}\nabla\lambda_c$, donde
$\nabla\lambda_b = (e_2\times n)/|n|^2$,
$\nabla\lambda_c = (n\times e_1)/|n|^2$ y
$\nabla\lambda_a = -\nabla\lambda_b-\nabla\lambda_c$. El gradiente del peso
renormalizado sale de la regla del cociente:

$$\nabla w_k = \frac{\nabla\tilde w_k - w_k\sum_{i\in K}\nabla\tilde w_i}{W}.$$

Es exacto en el centro de la Gaussian. Sobre un disco solo actúan las dos
componentes a lo largo de los ejes de reposo de la Gaussian $(\hat u,\hat v)$,
y los gradientes suman cero, así que solo se guardan los de los tres primeros
joints. Son tres palabras de dos halves, **12 bytes por Gaussian**, en
`primvars:athenea:splat:jointWeightGradients` (`half[]`, `elementSize` 6).
El cuarto gradiente es menos la suma de los otros tres. Con los gradientes, el
registro del efecto para una Gaussian llevada crece a nueve entradas.

::: {.nota}
**Por qué hacen falta gradientes** — la covarianza de una Gaussian deformada depende de la derivada del mapa de deformación (§5.4.2), y esa derivada incluye cómo cambian los pesos en el espacio. Un peso constante por Gaussian (lo habitual en los avatares Gaussian que aplican LBS) ignora ese término.

**Gradiente de las baricéntricas** — con $e_1=b-a$, $e_2=c-a$ y $n=e_1\times e_2$ (normal no normalizada, $|n|=2\cdot$área), $\lambda_b$ y $\lambda_c$ son funciones lineales cuyos gradientes en el plano del triángulo son $\nabla\lambda_b=(e_2\times n)/|n|^2$ y $\nabla\lambda_c=(n\times e_1)/|n|^2$. Se comprueba: $\nabla\lambda_b\cdot e_1=(e_2\times n)\cdot e_1/|n|^2=n\cdot(e_1\times e_2)/|n|^2=1$ y $\nabla\lambda_b\cdot e_2=0$, como debe (moverse a lo largo de $e_1$ aumenta $\lambda_b$ en 1). Como suman uno, $\nabla\lambda_a=-\nabla\lambda_b-\nabla\lambda_c$. El código protege los triángulos degenerados (si $|n|^2\le10^{-30}$, gradiente 0).

**Regla del cociente** — $w_k=\tilde w_k/W$ con $W=\sum_{i\in K}\tilde w_i$, luego $\nabla w_k=(\nabla\tilde w_k\,W-\tilde w_k\nabla W)/W^2=(\nabla\tilde w_k-w_k\nabla W)/W$. Es exacto en el punto evaluado (centro de la Gaussian) pero no constante: $w_k$ es un cociente de funciones lineales, por tanto no lineal sobre el triángulo.

**Qué se guarda y por qué 12 bytes** — (1) Solo las componentes tangentes: una Gaussian de conversión es un disco casi plano (tercer eje mínimo), así que solo importan las derivadas a lo largo de sus ejes de reposo $\hat u,\hat v$: $(\nabla w_k\cdot\hat u,\ \nabla w_k\cdot\hat v)$. (2) Como $\sum_k w_k=1$, $\sum_k\nabla w_k=0$: basta guardar tres gradientes y el cuarto es $-(g_1+g_2+g_3)$. Tres gradientes $\times$ dos componentes $=6$ halves $=12$ bytes (`elementSize` 6). Los halves se recortan a $\pm65504$ (`m2sPackHalves`) para que una transición muy abrupta no se convierta en infinito.

**Nueve entradas** — el registro que el efecto AOFX devuelve por Gaussian (float4 por entrada) pasa de ocho a nueve: la novena lleva los gradientes empaquetados. El SDK de AOFX no cambia; el host lee la longitud del registro del propio efecto.

*Referencias:* cálculo vectorial estándar de funciones de forma lineales (elementos finitos P1); código: `plugins/mesh2splat/mesh2splat.slang` (`blendInfluences`, `m2sPackHalves`), `shaders/athenea/scene/splat_skin.slang` (lectura de `weightGradients`, `lastGradient`); `docs/decisions.md` (sección "The whole Jacobian of a skinned gaussian", párrafos "What a gaussian keeps" y "Where it comes from").
:::

**Medido.** Un cuadrado ligado por entero a un joint se convierte en 1056
Gaussians, todas con ese joint y todo el peso. A través del efecto, 272
Gaussians de un quad cuyas esquinas nombran todas el joint 3 salen en el joint
3, con los demás huecos vacíos y los pesos sumando uno. Con un solo joint en
todas partes, el gradiente guardado es menor que $10^{-3}$.

::: {.nota}
**Qué prueban estas medidas** — son pruebas de invariantes, no de calidad visual: (1) un mesh ligado a un solo joint con peso 1 debe dar Gaussians con ese joint y peso 1 (la reducción y la renormalización no inventan influencias); (2) si las tres esquinas nombran el mismo joint, la mezcla debe colapsar a un hueco con peso total 1 y tres huecos vacíos; (3) un peso constante tiene gradiente nulo, y el valor guardado debe ser ruido de redondeo ($<10^{-3}$, del orden de la precisión half relativa a la escala del triángulo). Las 1056 y 272 Gaussians son simplemente lo que produce la resolución de prueba para esa geometría.

*Referencias:* código: pruebas de mesh2splat en `tests/aofx/` y `tests/usd/test_usd.cpp`; `docs/decisions.md` (secciones "A gaussian carries the joints its triangle carries", "The whole Jacobian of a skinned gaussian").
:::

**Truncamiento.** Cuatro influencias es un truncamiento. En el gorrión, el
27 % de los vértices del ala tiene más de cuatro (6686 vértices,
`elementSize` 25), y una Gaussian toma la unión de los joints de tres
esquinas, así que la fracción de Gaussians truncadas es al menos igual de
grande. Esa fracción no se ha medido.

::: {.nota}
**Truncamiento a cuatro influencias** — es la convención de los motores de tiempo real (cuatro pares caben en dos registros vec4, y la mayoría de rigs de juego se limitan a 4). Los rigs de producción usan muchas más: el ala del gorrión llega con `elementSize` 25. Descartar influencias y renormalizar desplaza el punto respecto de donde lo pondría el mesh con todas las suyas: es un error de **posición** (y de marco), máximo donde muchos joints con pesos parecidos compiten, como en las puntas de las plumas. Como una Gaussian une los joints de tres esquinas (hasta 12 distintos), la proporción de Gaussians truncadas es al menos la de vértices con más de cuatro (27 %).

**Consecuencia medida en otra parte** — en §5.7 este mismo truncamiento hacía que el rayo del bake no encontrara la superficie del mesh (566 603 Gaussians transparentes) cuando el bake se hacía en una pose animada.

*Referencias:* `docs/decisions.md` (secciones "The whole Jacobian of a skinned gaussian", párrafo "Not done here", y "A skinned transfer is traced at the bind pose", que menciona el caso de más de cuatro influencias).
:::

## 5.4 Skinning de una Gaussian en el device

`scene::SplatSkinner` lanza `shaders/athenea/scene/splat_skin.slang`, un hilo
por Gaussian. Sus entradas son la nube de reposo, los pares (joint, peso), las
transformaciones de skinning de los joints en el instante actual y tres
matrices: la transformación de bind $G$, `skelToWorld` y `worldToPrim`. Las
matrices de USD son row-major y actúan sobre vectores fila, así que el host
las traspone como hace `geom::Skinner` con los meshes (figura 14).

::: {.nota}
**`scene::SplatSkinner`** — la clase C++ que prepara y lanza el kernel (`modules/scene/include/athenea/scene/SplatSkinner.h`, `modules/scene/src/SplatSkinner.cpp`, `SplatSkinner::skin`). Un hilo por Gaussian (`[numthreads(256,1,1)]`): no hay dependencias entre Gaussians, es un mapa puro.

**Las tres matrices** — $G$ (`geomBind`): espacio de la nube → espacio de bind del skeleton. Atención a una sutileza: la conversión escribe las Gaussians en espacio de mundo, mientras que la `geomBindTransform` de UsdSkel parte del espacio del mesh; `skinningOf` compone ambas una vez, `toWorld^{-1}·geomBindTransform`, para que el fichero lleve la única matriz que necesita un renderer. `skelToWorld`: del espacio del skeleton al mundo. `worldToPrim`: del mundo al espacio local del prim de la nube, porque el renderer espera la nube en su propio espacio y le aplica después su transformación de prim.

**Row-major y vectores fila** — USD (Gf) escribe matrices para multiplicar vectores fila por la izquierda ($v' = v\,M$, traslación en la última fila). Los kernels multiplican matriz por vector columna (`mul(m, v)`), así que el host traspone (`transposed` en `SplatSkinner.cpp`; lo mismo en `Engine::carryCloud` para las matrices de joint), igual que `geom::Skinner` para meshes.

*Referencias:* código: `modules/scene/src/SplatSkinner.cpp` (`transposed`, `setRows`), `modules/usd/src/MeshStage.cpp` (`skinningOf`, comentario "OUT OF THE CLOUD'S SPACE"), `modules/usd/src/Engine.cpp` (`Engine::carryCloud`); `docs/decisions.md` (secciones "A cloud carried by a skeleton, on the device", "And the engine puts the cloud where the skeleton is", párrafo "Where the spaces meet").
:::

![Figura 14. Skinning en el device: la nube en bind pose, los pesos de joint por Gaussian y sus gradientes, y los transforms de joint del instante actual dan la media y la covarianza en pose; las tablas de visibilidad por parte, con bake único en bind pose, dan la sombra de la nube en pose.](figures/fig-skinning.es.svg)

### 5.4.1 La media

Sea $q = G\,p$ el punto de reposo en espacio de bind, $X_j q = A_j q + t_j$ la
transformación de skinning del joint $j$, y
$M=\texttt{worldToPrim}\cdot\texttt{skelToWorld}$. La media posada es linear
blend skinning, con exactamente la aritmética que usa el skinner de meshes:

$$\mu' = M \sum_j w_j\,X_j\,(G\,\mu).$$

::: {.nota}
**Linear blend skinning (LBS)** — también llamado *skeletal subspace deformation* (Magnenat-Thalmann et al., 1988): el punto deformado es la combinación convexa de los puntos que daría cada joint por separado, $p'=\sum_j w_j X_j q$. Con $X_j=(A_j,t_j)$ afín, $\sum_j w_jX_j$ es otra matriz afín: se mezclan las *matrices*, no los puntos, y por eso el kernel acumula una sola matriz `blended` que sirve para la posición y (su parte lineal) para el marco.

**Matriz de skinning** — en UsdSkel, `ComputeSkinningTransforms` devuelve $X_j(t)=W_j(t)\,B_j^{-1}$ en espacio del skeleton: primero se deshace la transformación de bind del joint, después se aplica la actual. En la bind pose $X_j=I$.

**Defecto conocido de LBS** — mezclar linealmente rotaciones no da una rotación: cerca de una torsión de $180^\circ$ la mezcla al 50 % colapsa (*candy wrapper*) y en flexiones fuertes pierde volumen. Las alternativas son el **dual quaternion skinning** (Kavan, Collins, Žára, O'Sullivan, 2007/2008: mezcla cuaterniones duales, que sí representan transformaciones rígidas, y normaliza), el *optimized centers of rotation* (Le y Hodgins, 2016) o las correcciones con blendshapes (Pose Space Deformation, Lewis et al., 2000). athenea elige LBS por ser exactamente lo que hace el skinner de meshes (para que nube y mesh coincidan) y porque su derivada es sencilla (§5.4.2). El DQS existe para meshes (`SkinningMethod::DualQuaternion` en `geom::Skinner`), no para nubes.

*Referencias:* Magnenat-Thalmann, Laperrière, Thalmann (1988), "Joint-dependent local deformations for hand animation and object grasping"; Kavan et al. (2008), "Geometric Skinning with Approximate Dual Quaternion Blending", ACM TOG; código: `shaders/athenea/scene/splat_skin.slang` (`splatSkin`, `moved`, `blended`), `shaders/athenea/geom/skinning.slang`; `docs/decisions.md` (sección "Skinning and blend shapes, on the device").
:::

### 5.4.2 El marco: el Jacobian entero

La covarianza posada debería ser la covarianza de reposo empujada por la
derivada del mapa de posiciones. Cuando los pesos varían en el espacio, esa
derivada es

$$J = L(M)\Big[\sum_j w_j A_j\,L(G) \;+\; \sum_j (X_j q)\,\nabla_p w_j^{\mathsf T}\Big],$$

donde $L(\cdot)$ es la parte lineal. El primer término es el habitual «mezclar
las partes lineales de los joints». El segundo, el término *elástico*, es el
que estira una Gaussian a través de una flexión donde dos joints no coinciden
en adónde va su punto. Con dos joints separados por un ángulo relativo
$\theta$ y una transición de pesos de longitud $L$, su tamaño relativo al
primer término es de unos $2\tan(\theta/2)\,d/L$, que es del orden de uno en
ángulo recto.

::: {.nota}
**Jacobian de la deformación** — si $\phi(p)$ es el mapa de posiciones, la Gaussian de reposo $\mathcal N(\mu,\Sigma)$ se aproxima tras deformarla por la linealización local $\phi(x)\approx\phi(\mu)+J(x-\mu)$, con $J=\partial\phi/\partial p$ en $\mu$; la imagen de una Gaussian por un mapa afín es otra Gaussian de covarianza $J\Sigma J^{\mathsf T}$. Es la misma aproximación local afín del EWA splatting de Zwicker et al. [3], que 3DGS [1] usa para proyectar a pantalla ($\Sigma_{2D}=JW\Sigma W^{\mathsf T}J^{\mathsf T}$).

**Derivación** — $\phi(p)=M\sum_j w_j(p)\,X_j(Gp)$. Por la regla del producto: $\partial\phi/\partial p=L(M)\big[\sum_j w_j\,A_j\,L(G)+\sum_j (X_jq)\,(\nabla_p w_j)^{\mathsf T}\big]$; el primer sumando deriva $X_j(Gp)$ con peso fijo (parte lineal $A_jL(G)$); el segundo deriva el peso con el punto fijo (producto exterior de un vector columna, el punto llevado por el joint, por el gradiente fila). $L(\cdot)$ toma el bloque $3\times3$ de una matriz afín.

**El término elástico, intuición** — si dos joints mandan el mismo punto a lugares distintos, la Gaussian que está en la transición de pesos se *estira* entre ambos destinos, como una goma. Estimación: con dos joints, $\nabla w_2=-\nabla w_1$ y $|\nabla w|\approx1/L$; el término vale $(X_2q-X_1q)\nabla w_2^{\mathsf T}$. Si los joints difieren en un giro $\theta$ alrededor de un pivote a distancia $d$, $|X_2q-X_1q|=2d\sin(\theta/2)$; frente al primer término (de norma $\approx1$) el cociente es $\sim 2d\sin(\theta/2)/L$, que el texto escribe con $\tan$ (mismo orden a ángulos moderados). Con $\theta=90^\circ$ y $d\approx L$, $\approx1.4$: «del orden de uno». No es una corrección menor en codos, muñecas o la mano del ala.

*Referencias:* [1], [3] del §10; código: `shaders/athenea/scene/splat_skin.slang` (cabecera "THE JACOBIAN, WHOLE"); `docs/decisions.md` (sección "The whole Jacobian of a skinned gaussian").
:::

El kernel construye el término elástico en el mismo bucle que forma $X_k q$.
Acumula $\sum_k (X_k q)\,g_k$ sobre cada eje de reposo, *relativo al punto
mezclado*. Los gradientes suman cero, así que es la misma suma, pero se
calcula como diferencia de puntos cercanos, lo que se mantiene exacto en
float lejos del origen. Un joint de peso cero pero con gradiente no nulo
también contribuye.

::: {.nota}
**Cómo se implementa sin coste extra** — en el bucle por joint el kernel ya calcula `carried` $=X_kq$; con el gradiente $g_k=(g_{k,u},g_{k,v})$ leído acumula `alongU` $\mathrel{+}= X_kq\,g_{k,u}$ y `alongV` igual. Estas son las columnas del término elástico aplicado a $\hat u$ y $\hat v$ (no hace falta la matriz $3\times3$ completa, solo su acción sobre los dos ejes del disco).

**Relativo al punto mezclado (cancelación numérica)** — como $\sum_kg_k=0$, $\sum_k(X_kq)g_k=\sum_k(X_kq-y)g_k$ para cualquier $y$. Si la nube está lejos del origen (p. ej. coordenadas de 100 m), $X_kq$ son números grandes cuya combinación con pesos de signo opuesto cancela catastróficamente en float32 (unos 7 dígitos). Restando $y=\mu_{\text{mezclado}}$ cada término es una diferencia de puntos cercanos y se conserva la precisión. Como los gradientes se leen en half, su suma no es exactamente cero; el código resta $y\cdot$`gradientSum`, que hace la identidad exacta con los valores leídos.

**Joints de peso cero con gradiente** — en el borde donde la contribución de una esquina se anula, el peso es 0 pero su derivada no: el joint «tira» aunque no «lleve». El código lo contempla (`pulls`).

*Referencias:* Goldberg (1991), "What Every Computer Scientist Should Know About Floating-Point Arithmetic" (cancelación); código: `shaders/athenea/scene/splat_skin.slang` (`alongU`, `alongV`, `gradientSum`, `pulls`).
:::

La covarianza posada en el plano de la Gaussian es

$$\Sigma' = J\,E\,S^2E^{\mathsf T}J^{\mathsf T},\qquad E=[\hat u\ \hat v],\ S=\mathrm{diag}(s_u,s_v).$$

El kernel toma una base ortonormal $(e_1,e_2)$ del plano generado por
$a=J\hat u\,s_u$ y $b=J\hat v\,s_v$, escribe $[a\ b]=[e_1\ e_2]K$ con $K$
triangular superior, y diagonaliza $C=KK^{\mathsf T}$ en forma cerrada. El
autovalor mayor es $m+r$, con $m=\tfrac12(c_{00}+c_{11})$ y
$r=\sqrt{\tfrac14(c_{00}-c_{11})^2+c_{01}^2}$. El menor se calcula como
$(k_{00}k_{11})^2/(m+r)$ en lugar de $m-r$, que en float se cancelaría a nada
para una Gaussian fina. Los nuevos ejes son los autovectores y los nuevos
tamaños las raíces cuadradas de los autovalores. El primer eje es el
autovector más cercano al primer eje llevado, y donde $c_{01}$ es puro
redondeo la base se conserva exactamente, para que una Gaussian redonda no
gire. El tercer eje es la normal del plano, y el tercer tamaño (el grosor de un
disco) queda como se convirtió. La normal de shading gira con el cofactor del
$J$ entero, porque $(Ja)\times(Jb)=\mathrm{cof}(J)(a\times b)$.

::: {.nota}
**Covarianza de un disco** — una Gaussian de conversión es casi 2D (como en 2DGS [5] o los surfels [6]): $\Sigma=R\,S^2R^{\mathsf T}$ con ejes $\hat u,\hat v,\hat n$ y escalas $s_u,s_v$ y un grosor mínimo. Restringida al plano, $\Sigma_\parallel=E S^2E^{\mathsf T}$ con $E=[\hat u\ \hat v]$ ($3\times2$). La posada es $\Sigma'=JES^2E^{\mathsf T}J^{\mathsf T}=[a\ b][a\ b]^{\mathsf T}$, con $a=J\hat u\,s_u$ y $b=J\hat v\,s_v$, los ejes «llevados».

**Por qué no Gram–Schmidt** — la versión anterior ortonormalizaba $a,b$ conservando la dirección de $a$: eso tira la *cizalla* (un rombo se vuelve rectángulo con los lados de $a$), y el término elástico es sobre todo cizalla. Diagonalizar $\Sigma'$ da el elipse correcto: ejes = autovectores, tamaños = raíces de autovalores.

**Diagonalización cerrada $2\times2$** — se toma la base ortonormal $e_1=a/|a|$, $e_2\propto b-(e_1\cdot b)e_1$; en ella $[a\ b]=[e_1\ e_2]K$ con $K=\begin{pmatrix}k_{00}&k_{01}\\0&k_{11}\end{pmatrix}$ (factorización QR). $C=KK^{\mathsf T}$ es simétrica: $c_{00}=k_{00}^2+k_{01}^2$, $c_{01}=k_{01}k_{11}$, $c_{11}=k_{11}^2$. Sus autovalores son $\lambda_\pm=m\pm r$. Como $\lambda_+\lambda_-=\det C=(k_{00}k_{11})^2$, el menor se calcula como $\det C/\lambda_+$: en una Gaussian fina ($\lambda_-\ll\lambda_+$), $m-r$ resta dos números casi iguales y en float da 0 o negativo; el cociente conserva todos los dígitos significativos. El ángulo es $\tfrac12\operatorname{atan2}(2c_{01},c_{00}-c_{11})$.

**Estabilidad de identidad** — si $|c_{01}|$ es solo redondeo (pose rígida, Gaussian casi redonda), se conserva la base $(e_1,e_2)$: sin esto una Gaussian redonda giraría arbitrariamente frame a frame (el autovector de un autovalor doble está indeterminado). Además se elige como primer eje el autovector más cercano a $a$ (intercambiando tamaños si el ángulo pasa de $45^\circ$), para no permutar ejes entre frames.

**La normal por el cofactor** — para vectores $a,b$ y matriz $J$ invertible, $(Ja)\times(Jb)=\det(J)\,J^{-\mathsf T}(a\times b)=\operatorname{cof}(J)(a\times b)$. Es la regla de transformación de normales (la «inversa traspuesta» de los motores de raster), escrita sin invertir: el kernel construye dos tangentes de la normal de shading, las lleva con el $J$ completo y hace su producto cruz.

*Referencias:* [1], [5], [6] del §10; Golub y Van Loan, *Matrix Computations* (QR y problemas simétricos $2\times2$); código: `shaders/athenea/scene/splat_skin.slang` (bloque "THE FRAME", `k00`…`minor`, `angle`, bloque "THE SHADING NORMAL"); `docs/decisions.md` (sección "The whole Jacobian of a skinned gaussian", párrafo "The frame, exact"; sección "A gaussian keeps its shading normal apart from its frame").
:::

El kernel solo escribe la posición, la rotación y las dos escalas del plano.
La opacidad, el color, los harmonics, los canales PBR y la palabra que guarda
el tercer tamaño no se tocan. Un frame cuesta, por tanto, un kernel sobre la
nube, y no se vuelve a decodificar nada.

::: {.nota}
**Qué se reescribe** — el layout de `GpuSplats` empaqueta cuatro palabras de forma por Gaussian (`shape`): [0] cuaternión smallest-three (10+10+10 bits y el índice descartado), [1] $\log s_x$ y $\log s_y$ en f16, [2] $\log s_z$ y el rojo del color base, [3] verde y azul del color base. El kernel reescribe la palabra 0 (rotación, `encodeQuaternion`) y la 1 (log de $s_u,s_v$), copia la 2 y la 3 tal cual, y escribe `positions`. Opacidad (en `positions.w`), SH y canales PBR no dependen de la pose. Por eso `Engine::carryCloud` crea un `GpuSplats` posado que **comparte** los buffers de harmonics y PBR con la nube de reposo: el coste por frame es un solo dispatch sobre la nube, sin re-decodificar ni subir nada.

*Referencias:* código: `shaders/athenea/common/packing.slang` (layout de `shape`), `shaders/athenea/scene/splat_skin.slang` (`shape[row]`…`shape[row + 3]`), `modules/usd/src/Engine.cpp` (`Engine::carryCloud`).
:::

### 5.4.3 Medido

La prueba de la tira, en `tests/scene/test_loading.cpp`, usa 4096 Gaussians
sobre una rampa lineal de pesos, dobladas un cuarto de vuelta. Compara la
covarianza posada decodificada de cada Gaussian con la analítica
$J=(1-w)I+wR+(Rp-p)\nabla w^{\mathsf T}$, y cuenta los errores relativos de
Frobenius por encima del 3 %. La tolerancia viene del formato: un cuaternión
smallest-three de diez bits no puede fijar un eje con más precisión que unos
$\sqrt2/1023$.

| caso | solo la mezcla, Gram–Schmidt (antes) | Jacobian entero, marco propio (después) |
|---|---|---|
| doblada, ejes a 0.1 rad de la rampa | 2048 de 4096 | 0 |
| doblada, ejes girados 30° | 2048 | 0 |
| giro rígido y desplazamiento (control) | 0 | 0 |
| un joint que cizalla, pesos constantes | 4096 | 0 |
| doblada, sin gradientes guardados | 2048 | 1984 |

::: {.nota}
**La prueba de la tira** — 4096 Gaussians sobre una tira con un peso que crece linealmente ($w$ de 0 a 1 a lo largo de la tira), dos joints: uno quieto ($I$) y otro que gira un cuarto de vuelta ($R$). Para este caso el Jacobian tiene forma cerrada: $\phi(p)=(1-w)p+wRp$, luego $J=(1-w)I+wR+(Rp-p)\nabla w^{\mathsf T}$, que es la fórmula general con $X_1=I$, $X_2=R$ y $\nabla w_2=-\nabla w_1=\nabla w$. El *oráculo* es analítico y se evalúa en un kernel de prueba (`shaders/athenea/test/splat_skin_check.slang`), no en la CPU, y solo se lee un contador.

**Error relativo de Frobenius** — $\|\Sigma'_{\text{kernel}}-\Sigma'_{\text{analítica}}\|_F/\|\Sigma'_{\text{analítica}}\|_F$, con $\|A\|_F=\sqrt{\sum_{ij}a_{ij}^2}$. Umbral 3 %.

**Por qué 3 %** — el cuaternión se guarda en *smallest-three* de 10 bits: se descarta la componente de mayor módulo (recuperable por normalización) y las otras tres, acotadas a $[-1/\sqrt2,1/\sqrt2]$, se cuantizan en 1024 niveles. El paso es $\sqrt2/1023\approx1.4\times10^{-3}$ en componentes del cuaternión, del orden de una décima de grado en el eje. Sumado a las escalas en half, un umbral del 3 % separa el error del formato del error del método.

**Lectura de la tabla** — «antes» es el kernel anterior (solo el primer término y Gram–Schmidt): falla en la mitad de la tira (2048), justo donde la rampa de pesos y la flexión crean cizalla; falla en toda la tira con un joint que cizalla con pesos constantes, porque Gram–Schmidt elimina la cizalla aunque no haya término elástico. «Después» no falla en ningún caso con gradientes. La fila del control rígido muestra que ninguno de los dos estropea lo trivial. La última fila (kernel nuevo, fichero sin gradientes) aísla la contribución de cada pieza: el marco propio arregla la cizalla, pero sin el término elástico siguen fallando 1984 de 4096.

*Referencias:* código: `tests/scene/test_loading.cpp` (caso "a skinned gaussian follows the whole Jacobian of its blend"), `shaders/athenea/common/packing.slang` (cabecera: "rotation in 10-bit smallest-three"); `docs/decisions.md` (sección "The whole Jacobian of a skinned gaussian", párrafo "Measured").
:::

Una prueba anterior colocó 4096 Gaussians sobre una hélice, cada una con su
propio marco y tamaño. En reposo, cada Gaussian se queda donde estaba. Bajo
un giro rígido de 1.2 rad alrededor de un eje inclinado más un desplazamiento,
cada Gaussian se lleva rígidamente. Solo se leen cuatro contadores.

::: {.nota}
**La hélice** — prueba de regresión de la versión inicial: Gaussians con marcos y tamaños distintos (una hélice garantiza que cada una tenga una orientación diferente) bajo una transformación rígida (giro de 1.2 rad sobre un eje inclinado más traslación). Debe cumplirse $\mu'=R\mu+t$, $R'=R\,R_{\text{reposo}}$, escalas iguales. Se leen cuatro contadores (p. ej. Gaussians fuera de tolerancia en posición, rotación, escala, y en reposo), siguiendo la regla de no leer datos de vuelta a la CPU.

*Referencias:* código: `tests/scene/test_loading.cpp` (caso "a skinned cloud follows its joints, and nothing else moves"); `CLAUDE.md` del repositorio, regla "No CPU arithmetic on data".
:::

En el clip `Flight` del gorrión (61 frames, 1500 triángulos del ala, el mesh
posado como referencia), el término de mezcla solo se aparta del mapa propio
del mesh una mediana del 4–6 %, un percentil 90 del 14–19 % y un 99 del
37–50 %, en la mano del ala y los abanicos de plumas. Con el término elástico,
los percentiles 90 y 99 bajan al 5–6.5 % y al 11–13 %. En imágenes de path
trace (1 125 022 Gaussians, 1024×768, frames 5, 31, 36) los dos kernels
difieren en unos pocos niveles de 8 bits en cerca de una décima parte de los
píxeles (p99 14–16), y las alas se ven iguales. El relMSE de la imagen entera
frente al mesh empeora un 2–3 %, pero una ablación muestra que es cosa de la
métrica: los gradientes negados, intercambiados, a la mitad y al doble
(0.0900, 0.0907, 0.0887, 0.0983 frente a 0.0904) se ordenan por el tamaño del
cambio, no por su corrección, porque domina el error de conversión. La
corrección se apoya en la prueba de la tira y en un diagnóstico sobre 174 305
Gaussians del ala: entre vecinas con los mismos cuatro joints, los gradientes
guardados predicen el cambio de los pesos guardados con un residuo RMS del
12.5 %, frente al 158 % con los ejes intercambiados y el 100 % sin gradiente.

::: {.nota}
**Las cifras del gorrión** — en el clip `Flight` se compara, para 1500 triángulos del ala, el mapa local del mesh posado (el $J$ que el propio mesh induce al deformar cada triángulo) con el $J$ del kernel. Solo el primer término: error mediano 4–6 %, p90 14–19 %, p99 37–50 %, concentrado en la mano del ala y los abanicos de plumas, donde las transiciones de peso son cortas ($L$ pequeño) y los ángulos grandes. Con el término elástico, p90 5–6.5 % y p99 11–13 %; lo que queda es la curvatura de la mezcla dentro del triángulo (LBS no es afín por trozos: el $J$ cambia dentro del triángulo y la linealización en el centro no lo capta).

**relMSE** — error cuadrático medio relativo, $\frac1N\sum_i (x_i-y_i)^2/(y_i^2+10^{-2})$, promediado por canal (la definición de `shaders/athenea/reference/image_compare_hdr.slang`, la misma de la literatura de denoising); penaliza igual errores relativos en zonas oscuras y claras.

**La ablación como lección metodológica** — negar, intercambiar, dividir por dos o duplicar los gradientes produce relMSE ordenados por la *magnitud* del cambio (0.0887 a la mitad < 0.0900 negados < 0.0907 intercambiados < 0.0983 doble), no por si el cambio es correcto. Cuando el error dominante es otro (la conversión: tarjetas de plumas difuminadas, sin relieve), cualquier perturbación se puntúa como «más diferencia». Por eso la corrección se demuestra con dos pruebas que sí son sensibles al signo y a los ejes: la tira analítica y un diagnóstico de consistencia entre vecinas, donde el gradiente guardado predice $\Delta w\approx\nabla w\cdot\Delta x$ con residuo RMS del 12.5 % (frente a 158 % con ejes intercambiados y 100 % sin gradiente, que es la línea base trivial: predecir cambio cero).

**Los píxeles** — p99 14–16 en niveles de 8 bits sobre cerca de un 10 % de los píxeles: diferencias pequeñas y localizadas en la mano del ala, visualmente iguales a esa distancia; la ganancia se espera en Gaussians grandes respecto a su separación (LOD gruesos, primeros planos de una flexión).

*Referencias:* código: `apps/athenea/src/CmdCompare.cpp` (`athenea compare`), `shaders/athenea/reference/image_compare_hdr.slang`; `docs/decisions.md` (sección "The whole Jacobian of a skinned gaussian", párrafos "On the sparrow" y "The 2-3 % is the metric's").
:::

## 5.5 El fichero: el rig como USD

Una nube con skinning es un `ParticleField3DGaussianSplat`. El rig se
escribió primero bajo `AtheneaSplatSkinningAPI`, y ahora se escribe con los
nombres propios de UsdSkel. `mesh2splat --skinned` aplica `SkelBindingAPI` y
escribe `skel:jointIndices` y `skel:jointWeights` (`elementSize` 4),
`skel:geomBindTransform`, `skel:joints` y `skel:skeleton`. También escribe
`primvars:athenea:splat:skinningXforms`: una matriz por joint, en el orden del
skeleton, **muestreada en el tiempo**. Es la única propiedad de una nube
animada que cambia de un frame a otro. Las transformaciones son
`UsdSkelSkeletonQuery::ComputeSkinningTransforms` en cada instante de
`--range START:END[:STEP]`.

::: {.nota}
**`ParticleField3DGaussianSplat`** — el esquema de OpenUSD para nubes de Gaussians 3D (familia `ParticleField` del dominio de volúmenes/partículas, resuelto en Hydra por `usdVolImaging`), con posiciones, orientaciones, escalas, opacidades y coeficientes SH como atributos. athenea añade sus primvars `primvars:athenea:splat:*` (codeless schemas en `modules/usd/schemas`).

**De esquema propio a UsdSkel** — al principio el rig se guardaba con un esquema aplicado propio, `AtheneaSplatSkinningAPI`, porque UsdSkel no contempla aplicar `SkelBindingAPI` a algo que no sea `UsdGeomPointBased`. Ahora se usan los nombres estándar (`UsdSkelBindingAPI::Apply`, `CreateJointIndicesPrimvar(false, 4)`, etc.), de modo que cualquier lector UsdSkel reconoce el prim como «skinneable»; el lector propio sigue aceptando los nombres antiguos.

**`skinningXforms` muestreado en el tiempo** — un array de `matrix4f` por instante, con las matrices $X_j(t)$ de `UsdSkelSkeletonQuery::ComputeSkinningTransforms` (joint en mundo por inversa de bind, en espacio del skeleton). Es el único dato animado; todo lo demás (Gaussians, influencias, gradientes) es constante, lo que permite el reaprovechamiento por identidad de §5.6. `--range START:END[:STEP]` fija los instantes (por defecto el rango del stage, paso de un time code).

*Referencias:* [43] del §10; código: `modules/usd/src/Export.cpp` (bloque "THE BINDING IS UsdSkel'S" y "AND THE JOINTS' TRANSFORMS, WRITTEN DOWN"), `modules/usd/src/MeshStage.cpp` (`MeshStage::skeletonTransforms`); `docs/decisions.md` (secciones "The file carries the rig, not the frames", "Step 3: a cloud is bound to its Skeleton the way the specification binds anything").
:::

**Ligado al Skeleton.** usdSkelImaging construye ext computations solo para
meshes, curvas y puntos. Sí resuelve, en cambio, los `skinningTransforms` de
cualquier Skeleton en cualquier instante. `HdAtheneaSkelSplatSceneIndex`
(`modules/usd/src/SkelSplat.cpp`) pone esas transformaciones resueltas en la
nube como `athenea:splat:skinningXforms`, remapeadas por `UsdSkelAnimMapper`
cuando la nube nombra un subconjunto de joints, y reenvía a sus nubes los
avisos de cambio de un Skeleton. El primvar que escribe la conversión pasa a
ser una **caché**: un stage que compone la nube sin su Skeleton sigue
moviéndose, y donde están los dos gana la caché. El skinning por cuaterniones
duales (`skel:skinningMethod`) no está implementado para nubes; un prim así se
anima por linear blend, con un aviso que se imprime una vez.

::: {.nota}
**usdSkelImaging y ext computations** — en Hydra, UsdSkel se resuelve con *scene indices*: `UsdSkelImagingPointsResolvingSceneIndex` añade bajo cada mesh/curva/puntos dos `HdExtComputation` (un agregador con lo constante y la computación con lo animado) cuya salida son los `points` deformados. Una `ParticleField` no es `PointBased`, así que no recibe computación; pero el `Skeleton` sí queda resuelto (*resolved skeleton*) con sus `skinningTransforms` en cualquier instante.

**`HdAtheneaSkelSplatSceneIndex`** — un *filtering scene index* (`HdSingleInputFilteringSceneIndexBase`) que se intercala en la cadena: para cada nube ligada a un Skeleton, inyecta en el prim el primvar `athenea:splat:skinningXforms` con las transformaciones resueltas, remapeadas con `UsdSkelAnimMapper` si la nube nombra un subconjunto de joints, y propaga los avisos de *dirty* del Skeleton a sus nubes (`_PrimsDirtied`). Es el «half que falta» de usdSkelImaging para nubes.

**La caché y la prioridad** — el primvar escrito por la conversión es una caché: permite que un stage que compone la nube sin el Skeleton siga animado, y cuando están ambos gana la caché (lo autorado explícitamente). La consecuencia práctica es que una animación nueva del Skeleton no se ve si la nube trae caché (la prueba de la siguiente nota lo verifica).

**Dual quaternion no implementado** — si el binding pide `skel:skinningMethod = dualQuaternion`, `ParticleField.cpp` imprime un aviso una vez y usa LBS. Para una nube, DQS requeriría además derivar el Jacobian de la mezcla de cuaterniones duales normalizada, bastante más complejo que el de LBS.

*Referencias:* [43] del §10; Kavan et al. (2008) para DQS; código: `modules/usd/src/SkelSplat.h`, `modules/usd/src/SkelSplat.cpp`, `modules/usd/src/RenderDelegate.cpp` (registro del filtro), `modules/usd/src/ParticleField.cpp` (aviso de `dualQuaternion`); `docs/decisions.md` (sección "Aligning with the specification, step by step", subsección "Step 3: a cloud is bound to its Skeleton the way the specification binds anything").
:::

**Medido.** 512 Gaussians ligadas a un Skeleton de un joint cuya animación
pone el joint en t = 1 donde la caché lo tiene en t = 2 dibujan, con la caché
bloqueada, el frame t = 2 de la caché en t = 1 píxel a píxel (p99 0, máx. 0).
En el gorrión, los 609 joints que recibe el motor a través del Skeleton
coinciden con los de la capa de rig; las transformaciones propias de USD
frente a la caché escrita por Blender difieren como mucho $5\times10^{-4}$, y
los frames difieren en una media de $8\times10^{-6}$.

::: {.nota}
**La prueba de prioridad** — se construye un caso donde Skeleton y caché discrepan a propósito (el Skeleton pone el joint en $t=1$ donde la caché lo pone en $t=2$). Con la caché bloqueada (presente y autorada), el frame $t=1$ debe ser idéntico píxel a píxel al frame $t=2$ de la caché: p99 0 y máx. 0 en `compareImages`. Demuestra que la regla «gana la caché» se cumple de forma determinista.

**En el gorrión** — los 609 joints que llegan por el Skeleton coinciden con los de la capa de rig, y las matrices calculadas por USD frente a las escritas desde Blender difieren como mucho $5\times10^{-4}$ (cuantización de la exportación: traslaciones y rotaciones con seis decimales, escalas en `half3`), lo que se traduce en una diferencia media de imagen de $8\times10^{-6}$.

*Referencias:* código: `tests/usd/test_usd.cpp`; `docs/decisions.md` (sección "Aligning with the specification, step by step", subsección "Step 3").
:::

**Escala de tiempo.** Sin `timeCodesPerSecond`, la nube se leía a 24 bajo un
stage de 30 fps y se retrasaba respecto al mesh (error medio 0.031 en el
frame 16, 0.049 en el 61); con la tasa escrita, $1.4\times10^{-5}$ y
$1.1\times10^{-5}$.

::: {.nota}
**`timeCodesPerSecond`** — metadato de capa que dice cuántos time codes hay por segundo. USD compone capas con tasas distintas reescalando sus muestras (*layer offset* implícito): una capa sin el metadato se lee con el valor por defecto, 24. La nube muestreada a 30 y compuesta bajo un stage a 30 se estiraba por $30/24=1.25$ y se retrasaba de forma acumulativa (de ahí que el error crezca de 0.031 en el frame 16 a 0.049 en el 61). Escribirlo (`MeshStage::timeCodesPerSecond()` → `SplatSkinning` → exportación) deja el error en la cuantización, $\sim10^{-5}$.

*Referencias:* documentación de USD, "Time Codes and Time Scaling" (`timeCodesPerSecond`, `framesPerSecond`); código: `modules/usd/src/MeshStage.cpp`, `modules/usd/src/Export.cpp`, `scripts/sparrow-anim.py`; `docs/decisions.md` (sección "One cloud, seventy-one clips", subsección "And the cloud was playing slow").
:::

**Clips.** El gorrión llega como cinco ficheros FBX con 71 clips. Cada clip
solo cambia `skinningXforms` (609 matrices por frame), así que un clip es una
capa sobre una nube compartida: 377 MB para los 71, frente a 22 GB para 71
nubes convertidas. `SparrowClips.usda` los ofrece como un **variant set** de
USD. La nube entra como payload, porque la opinión local de una subcapa
ganaría a la variante, y cada clip entra como referencia dentro de su
variante. El motor sigue siendo genérico: lista los variant sets, selecciona
uno en una capa de override en memoria e informa de `animationRange()`, el
intervalo de las muestras escritas (por ejemplo 1..25 para `land_hop_1`,
1..33 para `air_fly_A0`). Elegir un clip lee unos 3 MB, no 213 MB.

::: {.nota}
**Clip como capa** — como el clip solo cambia `skinningXforms`, cada clip es una capa con 609 matrices por frame. 71 clips = 377 MB (unos 5.3 MB de media) frente a $71\times313$ MB $\approx22$ GB si se convirtiera la nube para cada uno.

**Variant set** — mecanismo de composición de USD: un prim lleva un conjunto con nombre (p. ej. `clip`) con variantes; una *selección* decide cuál compone. El motor no sabe nada de gorriones: lista los variant sets, escribe la selección en la session layer (el fichero no cambia) y devuelve `animationRange()`, el intervalo real de las muestras escritas, en lugar del `startTimeCode`/`endTimeCode` de la capa raíz (que vale para el clip más largo, 1..81, y congelaría un clip corto).

**Payload frente a subcapa y LIVRPS** — el orden de fuerza de la composición de USD es *Local, Inherits, VariantSets, References, Payloads, Specializes*. Una opinión de una subcapa es *local* y ganaría a cualquier variante: el valor por defecto de `skinningXforms` de la nube taparía las muestras de cada clip. Entrando la nube como **payload** (lo más débil salvo specializes) y cada clip como **referencia dentro de su variante**, las muestras del clip ganan. Además, USD solo abre la referencia de la variante seleccionada: elegir un clip lee unos 3 MB, no los 213 MB de la nube y todo lo demás.

*Referencias:* [43] del §10; documentación de USD, "LIVRPS Strength Ordering" y "VariantSets"; código: `scripts/sparrow-clips.py`, `scripts/sparrow-clip-variants.py`, `modules/usd/src/StageRenderer.cpp` (`animationRange`, selección de variantes), `modules/mcp/src/Tools.cpp`; `docs/decisions.md` (secciones "One cloud, seventy-one clips", "Seventy-one animations, and not one line about sparrows", subsecciones "Why the timeline stopped asking the stage" y "Packaging: a payload for the cloud, a reference a clip").
:::

| asset (`mesh2splat --skinned`) | Gaussians | joints | instantes | fichero |
|---|---|---|---|---|
| tubo con rig, resolución 512 | 184 320 | 40 | 25 / 121 | 9.27 / 9.50 MB |
| zorro de Khronos, resolución 512 | 204 517 | 24 | 28 | 13.35 MB |
| gorrión, resolución 1400 (primera conversión) | 4 991 908 | 609 | 33 | — |

::: {.nota}
**La tabla** — el tubo y el zorro (Fox de Khronos glTF-Sample-Assets, CC0/CC-BY) son assets de validación de tamaño medio; el gorrión a resolución 1400 ilustra la escala real: casi 5 M de Gaussians y 609 joints. Su tamaño de fichero no se dio porque esa primera conversión no se conservó tal cual. «Resolución» es el número de cells de la rejilla de conversión a lo largo del lado mayor de la caja (§3). Con dos influencias por vértice el fichero del tubo sería 12.55 MB en vez de 9.5: la mezcla baricéntrica da a la mayoría de Gaussians cuatro joints en vez de uno, y joints + pesos ocupan 32 B por Gaussian en float.

*Referencias:* `docs/decisions.md` (secciones "What a rigged cloud actually costs", "The fox").
:::

## 5.6 El motor: pose, límites, refit y motion blur

`Engine::carryCloud` empareja los dos arrays del fichero en la disposición
(joint, peso), conserva la nube subida como bind pose y ejecuta el skinner
sobre un `GpuSplats` aparte que comparte los buffers de harmonics y PBR, ya
que el skinner no escribe ninguno de los dos. Al medirlo aparecieron dos
errores. Primero, la extensión de una nube posada no es la de la bind pose,
así que ahora se vuelve a plegar en el device después del skinning.
Segundo, `Engine::bounds()` plegaba la nube subida en lugar de la dibujada.

::: {.nota}
**`Engine::carryCloud`** — hace tres cosas: (1) entrelaza los dos arrays del fichero (`jointIndices`, `jointWeights`) en pares (joint, peso), el formato que leen todos los skinners del motor, y traspone las matrices; (2) conserva la nube subida como bind pose y escribe la posada en buffers propios, de modo que lo que ve un renderer no cambia de identidad entre frames; (3) lanza `scene::SplatSkinner` y sube la `revision` de la nube para que las estructuras construidas sobre la pose anterior sepan que deben actualizarse.

**Los dos errores** — una nube posada sale de su caja de reposo (el ala levantada sobresale), así que todo lo que hace culling, encuadre u ordena por esa caja miraba al sitio equivocado; ahora la caja se vuelve a reducir en la GPU tras el skinning. Y `Engine::bounds()` plegaba `entry.gpu->bounds` (la nube subida) en vez de la dibujada.

*Referencias:* código: `modules/usd/src/Engine.cpp` (`Engine::carryCloud`, `Engine::bounds`), `modules/usd/src/Engine.h`; `docs/decisions.md` (sección "And the engine puts the cloud where the skeleton is").
:::

**Un paso de tiempo no es una nube nueva.** Al principio, cada cambio de time
code volvía a decodificar todos los arrays y reconstruía en la CPU los pares
de influencias (137 MB en el gorrión de 4.27 M Gaussians). Ahora los arrays
se comparan por identidad (excepto `skinningXforms`), los pares solo se
reconstruyen al subir, y los arrays sin muestras de tiempo se retienen
(`_held`), porque los arrays enteros comprimidos del fichero crate llegan en
buffers nuevos en cada lectura. Viewer, 400 frames a 1280×720:

| | antes | después |
|---|---|---|
| subidas de la nube en 400 frames | 400 | **1** |
| raster, mediana / en caliente | 134.96 / 31.56 ms | **56.87 / 13.40 ms** |
| `rt`, mediana / en caliente | 268.66 / 55.31 ms | **178.99** / 55.05 ms |

::: {.nota}
**Por qué cada time code «era» una nube nueva** — `usdVolImaging` marca la `ParticleField` como variable en el tiempo, así que cada paso del timeline llegaba a `Sync` con `DirtyPoints | DirtyPrimvar`. `VtArray` es copy-on-write: un `Get` en otro instante de un atributo sin muestras devuelve *el mismo buffer*; comparar las direcciones y tamaños de los arrays (diez punteros más el grado SH, `Engine::identityOf`) dice si algo de lo que depende el decode ha cambiado. `skinningXforms` se excluye a propósito.

**`_held`** — en ficheros crate (`.usdc`), los arrays enteros comprimidos se descomprimen en un buffer nuevo en cada lectura, de modo que la comparación por identidad fallaría; `HdAtheneaParticleField` retiene (`_held`) la primera lectura de los arrays sin time samples y la reutiliza.

**137 MB** — $4\,269\,858$ Gaussians $\times$ 4 pares $\times$ 8 B (dos floats) $\approx136.6$ MB de pares reconstruidos en la CPU en cada frame; ahora solo al subir.

**La tabla** — «mediana» incluye frames con trabajo extra (primer frame, recompilación); «en caliente» es el frame estable. El raster baja de 31.56 a 13.40 ms en caliente (unos 75 fps con 4.27 M Gaussians con skinning). El `rt` mejora en mediana pero no en caliente: su coste restante estaba en los proxies y la estructura de aceleración, que es lo que aborda el refit.

*Referencias:* código: `modules/usd/src/ParticleField.cpp`, `modules/usd/src/ParticleField.h` (`_held`), `modules/usd/src/Engine.cpp` (`Engine::identityOf`, `CloudIdentity`); `docs/decisions.md` (sección "A time change is not a new cloud").
:::

**Refit, no reconstrucción.** Una pose no cambia ni el número de partículas
ni su orden, así que `GaussianRayTracer::sync` hace refit. En la ruta
hardware recalcula los proxies y actualiza en sitio el BLAS de cada trozo. En
la ruta de BVH por compute hace refit del tree una altura cada vez, de abajo
arriba, sin lectura de vuelta. Cada `refitsPerRebuild` = 240 refits se hace
una reconstrucción completa. Frente a reconstruir en cada pose, las imágenes
coinciden con p99 0, máx. 1. Un tree sin refit da p99 255. Gorrión a
resolución 256 (300 862 Gaussians, BVH por compute): 140.9 / 143.1 /
135.6 ms por frame antes, 84.7 / 83.9 / 83.7 ms después (−40 %).

::: {.nota}
**Refit en las dos rutas** — *hardware*: el trazador construye por cada Gaussian proxies triangulares (un icosaedro envolvente, al estilo de 3DGRT [7]) y un BLAS (*bottom-level acceleration structure*) por trozo; al posar, recalcula los proxies y actualiza el BLAS en sitio (`BuildMode::Update`, construido con `AllowUpdate`; en Metal, `refitAccelerationStructure`). *BVH por compute*: el LBVH de Karras [32] se reajusta altura a altura, de abajo arriba (`bvh_refit_level.slang`), un dispatch por altura. Como el número de alturas se conoce al construir, el número de dispatches es fijo y no hace falta leer ningún contador a la CPU (la construcción original iteraba «hasta que nada cambie», lo que exigía lecturas).

**Por qué reconstruir cada 240** — el refit conserva la topología del árbol: si las primitivas se desplazan mucho respecto a la pose de construcción, las cajas crecen y se solapan y el trazado se ralentiza. Una reconstrucción periódica restaura la calidad; 240 se midió como el punto en que el trazado aún no se ha degradado, y una reconstrucción cuesta un tirón de unos 50 ms.

**Las cifras** — p99 0, máx. 1 frente a reconstruir cada pose: diferencias de un nivel por recorrer el árbol en otro orden (sumas de transmitancia en float con otro orden). Un árbol *sin* refit (cajas de la pose vieja) da p99 255: los rayos no encuentran las Gaussians. Gorrión a 256 (300 862 Gaussians): de ~140 ms a ~84 ms por frame, −40 %.

*Referencias:* [7], [32] del §10; Wald, Boulos, Shirley (2007), "Ray Tracing Deformable Scenes Using Dynamic Bounding Volume Hierarchies", ACM TOG; código: `modules/render/include/athenea/render/GaussianRayTracer.h` (`refitsPerRebuild = 240`), `modules/render/src/GaussianRayTracer.cpp` (`GaussianRayTracer::sync`), `shaders/athenea/rt/bvh_refit_level.slang`, `tests/render/test_splat_render.cpp`; `docs/decisions.md` (sección "A posed cloud refits its ray tracing structure").
:::

**Motion blur desde el skeleton.** `ParticleField` muestrea `skinningXforms`
a lo largo del shutter. El skinner mezcla la segunda pose en el mismo bucle
(solo los puntos, cerca de un tercio de kernel) y escribe el desplazamiento de
cada Gaussian como dos palabras de halves. El rasterizer convierte ese
desplazamiento en un término de covarianza de rango uno. Hydra toma las
muestras que abarcan el shutter en lugar de recortarlas, así que el
desplazamiento se escala por $(t_c-t_o)/(t_1-t_0)$; sin ese factor las alas
se desenfocan el doble. En el gorrión de 5.9 M a 1280×800, un frame tarda
90.0 ms con shutter 0 y 100.5 ms con shutter 0.5 (+11 %).

::: {.nota}
**Motion blur de una Gaussian como término de rango uno** — la imagen de una Gaussian arrastrada uniformemente a lo largo de un segmento $d$ (en pantalla) durante el shutter es su convolución con ese segmento. Igualando segundos momentos, un segmento uniforme de longitud $|d|$ tiene varianza $|d|^2/12$ a lo largo de sí mismo, así que $\Sigma'_{2D}=\Sigma_{2D}+dd^{\mathsf T}/12$: una actualización de rango uno de la covarianza 2D. La convolución con una medida normalizada conserva la integral, por lo que el pico debe bajar por $\sqrt{\det\Sigma/\det\Sigma'}$, la misma raíz que ya aplica el filtro de Mip-Splatting [4]. Alternativa descartada: promediar varias submuestras de la media (*Gaussian Splatting on the Move*, ECCV 2024, usa cinco).

**El desplazamiento desde el skeleton** — `ParticleField` muestrea `skinningXforms` en las dos muestras que abarcan el shutter. El kernel `splatSkin` reutiliza el bucle de joints para mezclar la segunda pose (solo el punto, no el marco: `movedEnd`), lo que cuesta «un tercio de kernel». El desplazamiento se guarda en el espacio de la nube como dos palabras de halves (`motion`).

**El factor de escala** — Hydra entrega las muestras $t_0,t_1$ que *abarcan* el intervalo del shutter $[t_o,t_c]$, no recortadas a él. El desplazamiento medido corresponde a $t_1-t_0$, mientras que el blur debe corresponder a $t_c-t_o$: se multiplica por $(t_c-t_o)/(t_1-t_0)$. Con shutter 0.5 entre frames, sin el factor el desplazamiento es el doble.

**Coste** — +11 % (90.0 → 100.5 ms) por la segunda pose y el término en la proyección.

*Referencias:* [3], [4] del §10; Seiskari et al. (2024), "Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation", ECCV; código: `shaders/athenea/scene/splat_skin.slang` (`skinningXformsEnd`, `motion`), `shaders/athenea/splat/splat_project.slang` (`d.x * d.x * (1.0 / 12.0)`), `modules/usd/src/ParticleField.cpp` (muestras que abarcan el shutter); `docs/decisions.md` (sección "A splat under a shutter: one rank-one term, not five passes").
:::

## 5.7 Luz que gira con un miembro: lobes zonales de transfer

Una nube estática puede llevar un transfer precalculado de nueve spherical
harmonics en espacio de mundo. Bajo un skeleton esos harmonics conservarían
el cielo de la pose de conversión, y girarlos cuesta una matriz $9\times9$ por
Gaussian. athenea guarda en su lugar uno o dos **lobes zonales** por
Gaussian, con el eje de cada lobe expresado en el marco propio de la
Gaussian. Una función zonal gira al girar su eje, una sola $3\times3$, así
que cualquier marco que produzca el skinner arrastra los lobes consigo, y
no hace falta asignar ninguna Gaussian a una parte.

::: {.nota}
**Transfer precalculado (PRT)** — la función $T(\omega)=V(\omega)\max(0,n\cdot\omega)$ (visibilidad por coseno) proyectada sobre la base de spherical harmonics; con la luz del entorno también proyectada, la irradiancia es un producto escalar de coeficientes [14]. Con tres bandas ($l=0,1,2$) son nueve coeficientes, suficientes para iluminación difusa de baja frecuencia [17].

**Rotar SH** — rotar una función expresada en SH exige una matriz por banda (bloques de $1\times1$, $3\times3$ y $5\times5$: una $9\times9$ diagonal por bloques), cuyos elementos son las matrices de Wigner-D; calcularlas por Gaussian y por frame es caro (Ivanic y Ruedenberg, 1996, dan una recurrencia). Alternativa: rotar por *parte* (una matriz por hueso), pero entonces una Gaussian en el borde entre dos partes no sabe cuál usar.

**Zonal harmonics (ZH) y lobes zonales** — una función con simetría de revolución alrededor de un eje $a$ solo tiene, en el sistema donde $a$ es el polo, componentes $m=0$: tres números $z_0,z_1,z_2$ para tres bandas. Rotarla es rotar su eje: una $3\times3$ aplicada a $a$. Por eso, si el eje se guarda en el *marco propio* de la Gaussian, el marco que produce el skinner (§5.4.2) lo arrastra automáticamente. Un transfer real no es simétrico, así que se aproxima por la suma de uno o dos lobes zonales (como en Sloan [16]). La misma idea aparece en *Relightable Full-Body Gaussian Codec Avatars* (arXiv 2501.14726), que guarda el transfer en ZH por la misma razón.

**Sin asignar partes** — a diferencia de la rotación por parte, aquí cada Gaussian lleva su propio marco, así que no hay frontera entre partes.

*Referencias:* [14], [16], [17] del §10; Ivanic, Ruedenberg (1996), "Rotation Matrices for Real Spherical Harmonics. Direct Determination by Recursion", J. Phys. Chem.; Green (2003), "Spherical Harmonic Lighting: The Gritty Details", GDC; código: `plugins/splattransferzonal/splattransferzonal.slang` (cabecera), `shaders/athenea/splat/splat_relight.slang` (`splatTransferFrame`); `docs/decisions.md` (sección "A skinned cloud keeps its transfer: zonal lobes in each gaussian's frame").
:::

Un lobe de eje $a$ y coeficientes $z_l$ es, en harmonics,
$g_{lm}=z_l\,k_l\,Y_{lm}(a)$ con $k_l=\sqrt{4\pi/(2l+1)}$. Para un eje fijo,
los mejores coeficientes son la proyección
$z_l = k_l\sum_m f_{lm}Y_{lm}(a)$. Por el teorema de adición, el residuo es

$$\|f-g\|^2=\|f\|^2-\sum_{l=0}^{2} z_l^2,$$

así que el eje se elige para maximizar $\sum_l z_l^2$. En el device, la
búsqueda prueba la dirección de la banda 1, la normal y 64 direcciones de
Fibonacci sobre un hemisferio, y después refina con un paseo que divide su
paso por dos. El segundo lobe se ajusta al residuo del primero, y los dos se
reajustan uno contra otro dos veces (el ajuste ZH de Sloan). Los datos son
diez floats por Gaussian, `primvars:athenea:splat:transferZonal`
(`elementSize` 10): por lobe, el $(u,v)$ octaédrico del eje en el marco y
$z_0,z_1,z_2$. En el device es el buffer `transfer` de siempre con
`transferCount` = 10. Al leer, `splatTransferFrame` lleva cada eje al mundo con
la rotación actual de la Gaussian y reconstruye los nueve harmonics.

::: {.nota}
**El lobe en SH** — una función zonal de coeficientes $z_l$ con eje $a$ tiene, en la base SH, coeficientes $g_{lm}=z_l\,k_l\,Y_{lm}(a)$ con $k_l=\sqrt{4\pi/(2l+1)}$. Sale del **teorema de adición** de los armónicos esféricos: $P_l(a\cdot\omega)=\frac{4\pi}{2l+1}\sum_m Y_{lm}(a)Y_{lm}(\omega)$, con $P_l$ el polinomio de Legendre. El factor $k_l$ normaliza para que $\sum_m g_{lm}^2=z_l^2$ (la energía de la banda es $z_l^2$ con independencia del eje).

**Mejores coeficientes para un eje y residuo** — proyectar $f$ sobre el subespacio de funciones zonales con eje $a$ es tomar $z_l=k_l\sum_m f_{lm}Y_{lm}(a)$ (producto escalar de $f$ con el lobe unitario de esa banda). Como la proyección es ortogonal, $\|f-g\|^2=\|f\|^2-\|g\|^2=\|f\|^2-\sum_l z_l^2$: minimizar el error es maximizar la energía capturada $E(a)=\sum_l z_l^2$. $E(a)=E(-a)$ (cambiar el eje de signo cambia el signo de $z_1$ pero no su cuadrado), así que basta buscar en un hemisferio.

**La búsqueda** — no lineal y con óptimos locales, por eso se siembra con candidatos razonables: la dirección de la banda 1 (el vector $(f_{1,1},f_{1,-1},f_{1,0})$ en orden $x,y,z$, la «dirección dominante» del transfer), la normal de la Gaussian y 64 direcciones de una espiral de Fibonacci (puntos casi uniformes sobre el hemisferio); después, un paseo local que divide el paso por dos al no mejorar. Con dos lobes: se ajusta el primero, se ajusta el segundo al residuo $f-g_1$, y se reajustan alternadamente dos veces (descenso por coordenadas por bloques, el «fast ZH approximation» de Sloan [16]).

**Almacenamiento** — 10 floats: por lobe, el eje en coordenadas octaédricas $(u,v)$ (mapa del octaedro desplegado a un cuadrado: dos números para un vector unitario sin singularidades de polo) y $z_0,z_1,z_2$; $2\times(2+3)=10$. En el device va en el buffer `transfer` de siempre como pares f16, y es `transferCount` = 10 lo que lo distingue de nueve SH (9) o de las dos mitades (36): sin flag nuevo. `splatTransferFrame` decodifica cada eje, lo lleva al mundo con la rotación actual de la Gaussian y reconstruye los nueve coeficientes con la fórmula anterior, de modo que el resto del shading (`transferValue`) no cambia.

**Error del ajuste** — el efecto AOFX adjunta un histograma de $\|f-g\|/\|f\|$ por cuartos de octava, que la conversión imprime como mediana, p90 y p99.

*Referencias:* [16] del §10; Cigolle et al. (2014), "A Survey of Efficient Representations for Independent Unit Vectors", JCGT (codificación octaédrica); González (2010), "Measurement of Areas on a Sphere Using Fibonacci and Latitude–Longitude Lattices", Math. Geosci.; código: `plugins/splattransferzonal/splattransferzonal.slang` (`zonalAbout`, `kept`, `takeAway`, `bestAxis`, `tzFit`), `plugins/splattransferzonal/SplatTransferZonal.cpp`, `shaders/athenea/splat/splat_relight.slang` (`splatTransferFrame`), `shaders/athenea/usd/transfer_zonal_io.slang`; `docs/decisions.md` (sección "A skinned cloud keeps its transfer: zonal lobes in each gaussian's frame", párrafos "What is stored" y "The fit").
:::

**La pose del bake.** Al principio, el bake posaba la nube en `--time` con
cuatro influencias por Gaussian, mientras Hydra animaba el mesh con todas las
suyas (diez por vértice en las plumas, 25 en las alas). Donde no coincidían,
el rayo del bake no encontraba su superficie, y 566 603 de 1 504 160
Gaussians salieron transparentes. Ahora el bake mantiene el skeleton en su
bind pose (`StageRenderer::holdBindPose`) y posa la nube con transformaciones
identidad: 1 504 156 de 1 504 160 encuentran su superficie. El transfer guarda
por tanto la autosombra de la bind pose (alas extendidas). La oclusión que
proyecta otro miembro en otra pose es cosa de §5.8.

::: {.nota}
**Por qué el bake fallaba en una pose animada** — el bake lanza, desde cada Gaussian, rayos contra el *mesh* posado (Hydra lo deforma con todas sus influencias: 10 por vértice en las plumas, 25 en las alas), mientras que la Gaussian se había posado con solo cuatro. Donde ambas posiciones discrepan más que la tolerancia del rayo, el rayo no encuentra su superficie y el transfer queda vacío; un transfer vacío se escribe como transparente: 566 603 de 1 504 160 Gaussians (37.7 %) con opacidad 0, plumaje visto como «puntos sueltos».

**La solución** — en la bind pose no hay deformación y mesh y nube coinciden exactamente: `StageRenderer::holdBindPose` escribe en la session layer los skeletons sin animación (rest transforms = bind transforms hechas locales) y la nube se posa con transformaciones identidad. Resultado: 1 504 156 de 1 504 160 (99.9997 %). Como los lobes viven en el marco de la Gaussian, un transfer horneado en la bind pose gira con el hueso igual que uno horneado en cualquier otra pose; lo que cambia es *qué* autosombra contiene: la de las alas extendidas.

*Referencias:* código: `modules/usd/include/athenea/usd/StageRenderer.h`, `modules/usd/src/StageRenderer.cpp` (`holdBindPose`), `apps/athenea/src/CmdMesh2Splat.cpp`; `docs/decisions.md` (sección "A skinned transfer is traced at the bind pose").
:::

**Comprobado.** En `[zonal]`, un transfer de dos lobes sobrevive al ajuste y
al almacenamiento en f16 dentro del 0.5 %. Girar el marco de una Gaussian por
$R$ y leer a lo largo de $Rw$ da lo que la Gaussian sin girar da a lo largo de
$w$. En una prueba USD, 512 Gaussians llevadas por un joint que gira 60°
coinciden con la misma nube bajo un Xform de esa rotación, con relMSE por
debajo de $10^{-4}$ en las rutas raster y trazada. El error del ajuste en el
gorrión no se ha medido. La indirect half del transfer no se guarda zonal.

::: {.nota}
**Las comprobaciones** — (1) *ida y vuelta*: un transfer que ya es suma de dos lobes debe recuperarse tras el ajuste y la cuantización f16 dentro del 0.5 %; (2) *equivariancia*: girar el marco por $R$ y leer en $Rw$ debe dar lo mismo que sin girar en $w$, es decir, $T_{R\cdot}(Rw)=T(w)$, que es la propiedad que justifica el diseño; (3) *extremo a extremo en USD*: 512 Gaussians llevadas por un joint que gira $60^\circ$ alrededor de $(1,2,0.5)$ frente a la misma nube quieta bajo un `Xform` con esa rotación, bajo un cielo distinto en cada dirección, en raster y en trazado: relMSE $<10^{-4}$.

**La mitad indirecta** — el transfer de athenea separa luz directa e indirecta (las «dos mitades», §4); la indirecta no se guarda en forma zonal, así que una nube con skinning solo tiene transfer directo, lo que explica que el cuerpo salga algo más oscuro en §5.10.

*Referencias:* código: `tests/aofx/test_transfer_zonal.cpp` (caso "a zonal transfer is fitted in the gaussian's frame and turns with it"), `shaders/athenea/test/transfer_zonal_check.slang`, `tests/usd/test_usd.cpp` (etiqueta `[zonal]`); `docs/decisions.md` (sección "A skinned cloud keeps its transfer: zonal lobes in each gaussian's frame", subsección "How it is checked").
:::

## 5.8 Campos de visibilidad por partes: bake una vez, un producto de lecturas de tabla por frame

Una nube con relight también tiene que hacerse sombra a sí misma, y las
medidas descartaron las rutas obvias. En la ruta trazada `--splat-shadows` no
cambiaba nada, porque un splat trazado no tenía pasada de sombra. En la ruta
raster costaba 36× el frame (49.58 → 1810.24 ms) y aplastaba el pájaro a
negro. Barrer el desplazamiento propio de 1 a 800 sigmas propias movía la
imagen media de forma monótona del 0 % al 99 % del valor sin sombra, sin
meseta. La literatura coincide en que la visibilidad con bake no sobrevive al
movimiento articulado salvo si se factoriza por parte del cuerpo: Lin et al.
usan MLP direccionales por parte, y DNF-Avatar un grid escalar de AO por
hueso. athenea combina las dos ideas: una **tabla direccional por parte**.

::: {.nota}
**Por qué fallan las rutas obvias** — en la ruta trazada, el shading de un splat no tenía pasada de sombra propia (`rt_shade.slang` declaraba el parámetro sin usarlo), así que `--splat-shadows` no cambiaba nada. En raster, la pasada de sombra traza desde cada splat hacia la luz y empieza el rayo a un desplazamiento de `shadowOffset` sigmas propias. Ese valor (3 sigmas) está pensado para capturas fotogramétricas de splats gordos y dispersos; en una nube convertida, los splats miden $\sim1.6\times10^{-4}$ y el plumaje apila muchas capas en medio milímetro: la transmitancia, un producto $\prod(1-\alpha_i)$, cae a casi cero tras atravesar unas pocas vecinas (con $\alpha\approx0.91$ por capa, tres capas dejan $0.09^3\approx7\times10^{-4}$). El barrido de 1 a 800 sigmas sin meseta demuestra que no existe un desplazamiento correcto: cualquier valor es a la vez demasiado corto para unas Gaussians y demasiado largo para otras.

**La literatura** — la visibilidad horneada sólo sobrevive al movimiento articulado si se factoriza por parte del cuerpo: Lin et al. [41] entrenan un MLP de visibilidad direccional por parte (*part-wise light visibility*, en avatares neuronales relightables); DNF-Avatar [42] usa un grid escalar de oclusión ambiental por hueso. athenea combina ambas: tablas *direccionales* (como [41]) en *grids* explícitos por parte (como [42]), sin red neuronal.

*Referencias:* [41], [42] del §10 (probablemente Lin et al., "Relightable and Animatable Neural Avatars from Videos", AAAI 2024, y Jiang et al., "DNF-Avatar: Distilling Neural Fields for Real-time Animatable Avatar Relighting", 2025; el artículo los cita sin datos completos); código: `shaders/athenea/rt/rt_shade.slang`, `shaders/athenea/splat/splat_project.slang`; `docs/decisions.md` (secciones "A cloud does not shadow itself, and the offset cannot be tuned to fix it" y "A cloud shadows itself by part: baked once, read every frame, no ray").
:::

**Partición.** `partitionJoints` toma el subárbol más grande que aún no es una
parte, con al menos `minJoints` (6) joints, y lo convierte en parte, hasta que
hay `--parts` (12). En el gorrión salen raíz, centro, pecho, pelvis, cabeza, y
cada ala en tres segmentos. Cada Gaussian pertenece a la parte de su joint
dominante.

::: {.nota}
**`partitionJoints`** — el skeleton es un árbol (la jerarquía se deduce de las rutas de los joints: el padre de `A/B/C` es `A/B`). Se calcula el tamaño de cada subárbol; se toma como parte la raíz más grande, y luego, repetidamente, el subárbol más grande que aún no es cabeza de parte y tiene al menos `minJoints` (6) joints, hasta tener `--parts` (12). Cada joint pertenece a la parte de su antepasado-cabeza más cercano. Elegir primero lo más grande separa un ala del pecho antes que un dedo del pie: las particiones grandes son las que proyectan sombras grandes. En el gorrión resultan: raíz, centro, pecho, pelvis, cabeza y cada ala en tres segmentos (dos partes, raíz y centro, no llevan ninguna Gaussian).

**Joint dominante** — cada Gaussian se asigna a la parte de su joint de mayor peso (`visibilityPartOf`). Una Gaussian en un límite entre partes queda en una sola, aproximación aceptable porque las partes son grandes.

*Referencias:* código: `modules/technique/src/SplatVisibility.cpp` (`partitionJoints`), `modules/technique/include/athenea/technique/SplatVisibility.h`, `apps/athenea/src/CmdVisibility.cpp` (`--parts`, `--min-joints`, valores por defecto 12 y 6), `tests/technique/test_splat_visibility.cpp` (caso "a rig becomes parts by its largest subtrees, small ones staying with their parent"); `docs/decisions.md` (sección "A cloud shadows itself by part: baked once, read every frame, no ray", subsección "What it is").
:::

**Bake** (`athenea visibility`, `splat_visibility_bake.slang`). Se hace una
vez, en la bind pose. Para la parte $p$, un grid de $N^3$ sondas cubre la caja
de la nube más un margen del 5 %. Cada sonda guarda un mapa octaédrico de
$O^2$ direcciones. Un texel guarda la transmitancia de un rayo que sale de la
sonda a través solo de las Gaussians de la parte $p$. Las Gaussians de la
parte se reúnen en una nube propia y se trazan con rayos inline, con el mismo
producto de $(1-\alpha)$ que usan los kernels de sombra. Cada sonda guarda
además un término ambiente, su media sobre las direcciones, que se usa para
los domes.

::: {.nota}
**Campo de visibilidad por parte** — una función $F_p(x,\omega)\in[0,1]$: la transmitancia desde el punto $x$ en la dirección $\omega$ a través *solo* de la parte $p$, en la bind pose. Se discretiza como un grid $N^3$ de sondas (*probes*, como las *irradiance probes* o los *light field probes* de McGuire et al., 2017) y, en cada sonda, un mapa octaédrico de $O\times O$ direcciones: el octaedro desplegado cubre toda la esfera con un cuadrado, sin distorsión extrema ni costuras de polo.

**El bake** — `visBake`: un hilo por (sonda, dirección), $N^3O^2$ hilos por parte. La parte se reúne en una nube propia sobre la que el trazador construye proxies (icosaedros) y un BLAS; cada hilo lanza un `RayQuery` inline (sin shaders de hit) y acumula la transmitancia con `shadowAlpha`, la misma función que usan los kernels de sombra: el $\alpha$ máximo de la Gaussian a lo largo del rayo, y $T=\prod(1-\alpha_i)$, cortando cuando $T$ cae bajo un umbral. El rayo empieza a una celda de sonda de distancia (`start`).

**Término ambiente** — la media de la transmitancia de la sonda sobre todas sus direcciones, una sola cifra que aproxima la fracción de cielo no tapada por la parte: se usa para luces dome.

*Referencias:* McGuire et al. (2017), "Real-Time Global Illumination using Precomputed Light Field Probes", I3D; Cigolle et al. (2014) para el mapa octaédrico; [7] del §10 (trazado de Gaussians); código: `shaders/athenea/splat/splat_visibility_bake.slang` (`visBake`), `shaders/athenea/splat/splat_visibility_common.slang`, `modules/technique/src/SplatVisibility.cpp` (`SplatVisibility::bake`), `apps/athenea/src/CmdVisibility.cpp`; `docs/decisions.md` (sección "A cloud shadows itself by part: baked once, read every frame, no ray").
:::

**Lectura** (`splat_visibility_read.slang`). En cada frame, `visPartFrames`
calcula un mapa afín por parte desde el espacio posado de vuelta al espacio
de bind donde vive el campo:

$$x_{\text{rest}} = U A_{k_p}^{-1}\,(x' - t_{k_p}) - U g,\qquad U=L(G)^{-1},$$

donde $k_p$ es el joint cabeza de la parte, $X_{k_p}=(A_{k_p},t_{k_p})$ su
transformación de skinning y $g$ la traslación de $G$; a este mapa se le
llama $T_p^{-1}$. Antes de precalcularlo costaba dos inversas $3\times3$ por
Gaussian, por parte y por luz, unos 260 millones por frame en el gorrión;
precalcularlo bajó la lectura de 12.7 a 8.0 ms (5 887 323 Gaussians,
1600×1000). El factor de la Gaussian $i$ para la dirección de luz
$\omega_\ell$ es entonces el producto sobre las demás partes:

$$V_{i\ell}=\prod_{p\ne \pi(i)} F_p\!\left(T_p^{-1}x_i,\;L(T_p)^{-1}\omega_\ell\right),$$

donde $\pi(i)$ es la parte propia de la Gaussian, $F_p$ se lee de forma
trilineal sobre las sondas y bilineal sobre el mapa octaédrico, y vale 1 fuera
de la caja de la parte. Un dome usa el término ambiente. Los factores van al
mismo `shadowFactors` que leen las dos rutas, así que la ruta trazada, que no
tenía ninguna sombra de splats, también los recibe. No se traza ningún rayo
por frame.

::: {.nota}
**El mapa inverso por parte** — la posición posada es $x'=X_k\,G\,x_{\text{rest}}$ (con la parte tratada como rígida, gobernada por su joint cabeza $k_p$). Invertir: $Gx_{\text{rest}}=A_k^{-1}(x'-t_k)$ y, con $G=(L(G),g)$, $x_{\text{rest}}=L(G)^{-1}\big(A_k^{-1}(x'-t_k)-g\big)=UA_k^{-1}(x'-t_k)-Ug$. Es una transformación afín $T_p^{-1}$, idéntica para todas las Gaussians de un frame.

**Por qué precalcular** — el kernel antiguo invertía dos matrices $3\times3$ (la del joint y la de bind) por Gaussian, por parte y por luz: $\approx5.9\times10^6\times11\times2\times2\approx2.6\times10^8$ inversas por frame. `visPartFrames` las calcula una vez por parte (12 hilos) y escribe tres filas `float4` por parte en `partFrames`; `visFactors` solo aplica $3\times4$. La lectura bajó de 12.7 a 8.0 ms.

**El producto de visibilidades** — se supone que las partes ocluyen de forma independiente, así que la transmitancia total hacia la luz es el producto de las de cada parte: $V=\prod_p F_p$. Es exacto si los segmentos de rayo que atraviesan cada parte no se solapan en profundidad con otras (la transmitancia es multiplicativa a lo largo del rayo: $T_{AB}=T_AT_B$); se aproxima cuando dos partes se interpenetran. Cada $F_p$ se lee en el espacio de bind de *esa* parte: posición por $T_p^{-1}$ y dirección por la parte lineal inversa (una dirección no se traslada).

**Interpolación** — trilineal entre las 8 sondas que rodean el punto y, en cada una, bilineal entre 4 texels del mapa octaédrico: 32 lecturas por parte y luz (`fieldAt`). Fuera de la caja del campo vale 1 (no ocluye). Para una luz no dome se toma *una* dirección por luz (la muestra central de `sampleLight`), así que una luz de área da sombra de una sola dirección; un dome usa el término ambiente.

**`shadowFactors`** — el buffer de factores por (Gaussian, luz) que ya leían la ruta raster (`splat_project.slang`) y la trazada (`rt_shade.slang`): la ruta trazada recibe por primera vez sombra entre splats. Cero rayos por frame: todo es lectura de tablas.

*Referencias:* Max (1995), "Optical Models for Direct Volume Rendering", IEEE TVCG (transmitancia multiplicativa); código: `shaders/athenea/splat/splat_visibility_read.slang` (`visPartFrames`, `partFrames`, `inverse3`, `fieldAt`, `ambientAt`, `visFactors`), `modules/technique/src/SplatVisibility.cpp` (`SplatVisibility::factors`), `shaders/athenea/splat/splat_relight.slang`, `shaders/athenea/rt/rt_common.slang`, `shaders/athenea/rt/rt_shade.slang`; `docs/decisions.md` (sección "A cloud shadows itself by part: baked once, read every frame, no ray").
:::

**Hallado al construirlo.** Los texels sin escribir de una parte vacía se
leían como sombra total y dejaban el pájaro negro; ahora un grid de 0 la marca
y se salta. El bake traza rayos inline, que el destino CUDA no tiene, así que
es un módulo aparte y los devices CUDA leen campos con bake hecho en otro
device. Una parte no se hace sombra a sí misma: su campo, leído en una
Gaussian suya, es la multitud en la que esa Gaussian está. Un mapa de
$8\times8$ (unos 22° por texel) no resuelve el borde de un ala; 16 y 32 son
exactos fuera de una banda de una cell de sonda.

::: {.nota}
**Parte vacía** — si una parte no tiene Gaussians, su bake no escribe nada y los texels quedan a 0, que como transmitancia es «negro desde todas direcciones». Un grid de tamaño 0 marca la parte y la lectura la salta (`field.grid == 0u`).

**CUDA y el inline ray** — `RayQuery`/`TraceRayInline` no existe en el destino CUDA de Slang; si el bake estuviera en el mismo módulo que la lectura, la lectura tampoco compilaría. Separados, un device CUDA (la L4) lee campos horneados en Metal: los campos viajan en el fichero (`primvars:athenea:splat:visibilityParts`, `visibilityTexels`, `visibilityAmbient`, `visibilityPartOf`). Metal tenía además un límite de 31 buffers por pipeline que hizo separar el módulo de lectura.

**Una parte no se sombrea a sí misma** — leer el campo de la propia parte en una Gaussian suya mide la «multitud» en la que está inmersa: el mismo problema que volvía negra la sombra por rayo, un nivel más arriba. Se pierde la autosombra local (pluma sobre pluma dentro de un segmento de ala).

**Resolución angular** — un mapa $8\times8$ tiene 64 direcciones para $4\pi$ sr: $\approx0.2$ sr por texel, unos $180^\circ/8\approx22^\circ$ de lado. La penumbra que da a la altura del ala es más ancha que el ala. Con 16 y 32 la respuesta es exacta salvo en una banda de una celda de sonda (más la altura del ala por la tangente de la inclinación de la luz).

*Referencias:* documentación de Slang (capacidades por destino: `RayQuery` no disponible en CUDA); código: `shaders/athenea/splat/splat_visibility_bake.slang` (comentario de cabecera), `shaders/athenea/splat/splat_visibility_read.slang`, `tests/technique/test_splat_visibility.cpp` (caso "a wing's field shadows the body under it, and moves with the wing's joint"); `docs/decisions.md` (sección "A cloud shadows itself by part…", subsección "What was found building it").
:::

**Medido** en el gorrión: 4 269 858 Gaussians, 609 joints, 12 partes, grid 24,
octava 16.

| | |
|---|---|
| bake (M5 Pro) | 22 s, 3 538 944 texels por parte ($24^3\times16^2$) |
| campos en el fichero | 81 MB (dos f16 por palabra) |
| factores por debajo de ½, frame 31, sol + dome | 21.3 % |
| viewer raster, 400 frames, mediana / en caliente | 71.82 / 9.69 ms (56.87 / 13.40 sin ellos) |
| viewer `rt`, mediana / en caliente | 193.88 / 51.65 ms (178.99 / 55.05 sin ellos) |
| L4 (CUDA) raster, 8 frames 1280×720 | 73.04 ms (67.48 sin ellos) |
| L4 (CUDA) `rt` | 128.77 ms (121.48 sin ellos) |

::: {.nota}
**Las cifras** — $24^3\times16^2=13\,824\times256=3\,538\,944$ texels por parte; con 12 partes, $42\,467\,328$ texels; en f16 (2 B), $84.9\times10^6$ B $=81.0$ MiB: los «81 MB» son MiB. Bake de 22 s en el M5 Pro para los 12 campos. El 21.3 % de factores por debajo de ½ indica cuánta superficie queda notablemente sombreada en el frame 31 con sol y dome.

**Coste en frame** — en caliente el raster *baja* (13.40 → 9.69 ms), señal de que el coste de la lectura queda por debajo de la variación entre ejecuciones; en mediana sube (56.87 → 71.82 ms). En la L4 la lectura cuesta 5–7 ms por frame (73.04 frente a 67.48 ms en raster).

*Referencias:* `docs/decisions.md` (sección "A cloud shadows itself by part…", subsección "What was measured").
:::

En la L4 los campos cuestan 5–7 ms por frame para
4.27 M Gaussians × 2 luces × 11 partes de consultas. La sombra cae donde cae
la del mesh (el ala lejana, el flanco bajo el ala levantada, la raíz de la
cola) y sigue a las alas de un frame a otro. Con todas las tarjetas de plumas
(9.5 M Gaussians a resolución 1400), el bake se quedó sin memoria en la parte
más grande tanto en el M5 Pro (Metal) como en una L4 a través de Vulkan (un
BLAS de 1.17 GB para unos 3.3 M Gaussians). Por eso el bake del pájaro se hace
a resolución 1100.

::: {.nota}
**Escala de las consultas** — $4.27\times10^6$ Gaussians $\times$ 2 luces $\times$ 11 partes (todas menos la propia) $\approx9.4\times10^7$ lecturas de campo por frame, cada una de 32 texels: 5–7 ms en una L4. La sombra resultante coincide en lugar con la del mesh (ala lejana, flanco bajo el ala levantada, raíz de la cola).

**El límite de memoria del bake** — el bake de una parte construye un BLAS sobre sus Gaussians; con todas las tarjetas de plumas (9.5 M Gaussians a resolución 1400) la parte más grande tiene unos 3.3 M Gaussians y el BLAS ocupa 1.17 GB (con proxies icosaédricos: $3.3\times10^6\times20$ triángulos, unos 18 B por triángulo), lo que agota la memoria tanto en Metal como en Vulkan. Por eso el pájaro se hornea a resolución 1100. Hornear por trozos de parte (varios BLAS) es lo que falta (§5.13).

*Referencias:* `docs/decisions.md` (sección "A cloud shadows itself by part…"; sección "A mesh past the ceiling is converted in slices, and a barb keeps its alpha").
:::

**Límites.** Se deja fuera la autosombra local de una parte (pluma sobre
pluma). El término ambiente es una media sobre la esfera, no un lobe coseno.
Se supone que las partes son rígidas y que ocluyen de forma independiente. El
suelo bajo el pájaro se sombrea con un mapa aparte en el espacio de la luz,
que se estabiliza frente a la «respiración» de una caja posada (dimensionado a
partir de la esfera de reposo, encajado en el grid de texels del mundo). Una
alternativa con un grid de densidad no está implementada.

::: {.nota}
**Las cuatro aproximaciones** — (1) sin autosombra local de la parte; (2) ambiente como media uniforme sobre la esfera, no ponderada por coseno respecto a la normal del receptor (lo correcto para irradiancia difusa sería $\frac1\pi\int V(\omega)\max(0,n\cdot\omega)\,d\omega$); (3) partes rígidas: dentro de una parte no se tiene en cuenta la deformación por LBS; (4) independencia de oclusión (el producto). La sombra del suelo es otro mecanismo: un mapa de sombra en espacio de la luz cuyo tamaño se fija desde la esfera de reposo y se encaja en el grid de texels del mundo, para que no «respire» con la caja posada (que cambia con cada aleteo y haría temblar la sombra).

*Referencias:* `docs/decisions.md` (secciones "A cloud shadows itself by part…", subsección "What it is not"; "A cloud's shadow that does not breathe with the wings"; "A cloud's shadow, measured from the light, with no ray and no sort").
:::

## 5.9 LOD y diezmado de nubes con skinning

El octree de `.athc` hace merge de cells en orden de Morton. Eso no sirve para
una nube con skinning, porque una cell que tomara ala y cuerpo juntos no
sabría a qué joint seguir. Dos mecanismos lo sustituyen.

::: {.nota}
**Octree en orden de Morton** — el LOD de `.athc` (§6) ordena las Gaussians por código de Morton (entrelazado de bits de las coordenadas cuantizadas, que convierte la proximidad 3D en proximidad 1D) y cada celda del octree es un tramo contiguo de ese orden; hacer merge de una celda es sustituir sus Gaussians por una que conserve momentos. El orden se calcula sobre posiciones fijas. En una nube con skinning, una celda podría mezclar Gaussians de ala y de cuerpo: la Gaussian resultante no tendría un rig coherente (¿sigue al ala o al cuerpo?) y, al animar, se desgarraría.

*Referencias:* [2], [8], [32], [34] del §10; Morton (1966), "A Computer Oriented Geodetic Data Base and a New Technique in File Sequencing"; código: `modules/lod/include/athenea/lod/Lod.h`, `modules/lod/src/Lod.cpp`; `docs/decisions.md` (sección "The sun comes out of the sky and is lit as a light", subsección "Levels of detail for a cloud that moves").
:::

**Niveles discretos por reconversión.** `mesh2splat --lod-levels N` convierte
a la resolución pedida, después a la mitad, y así sucesivamente, cada nivel
una nube con su propio rig. `Engine::lodLevelsFor` dibuja el nivel más grueso
cuya cell no abarca más de `athenea:lod:threshold` píxeles (1 por defecto) en
el punto más cercano de sus límites *posados*. Los niveles no dibujados no se
posan (`deferredPose`). Clips del gorrión a 1100 / 550 / 275 cells
(5.9 M / 1.5 M / 0.4 M Gaussians), 1600×1000: 49.6 → 51.5 ms cuando el
pájaro llena el frame; **71.3 → 24.7 ms** a cuatro veces la distancia. Todos
los niveles quedan residentes con sus propios campos de visibilidad (81 MB
cada uno), y el cambio de nivel es un salto, no un fundido.

::: {.nota}
**LOD discreto por reconversión** — cada nivel es una conversión independiente a la mitad de resolución, con su propio rig (influencias recalculadas desde el mesh, no promediadas). `usd::writeLodAssembly` escribe un stage que referencia los niveles (`<nombre>_lod<n>.usdc`) con `athenea:lod:group` y `athenea:lod:cell` (tamaño de celda).

**Selección** — `Engine::lodLevelsFor` proyecta la celda de cada nivel en el punto más cercano de los límites *posados* y dibuja el más grueso cuya celda ocupe como mucho `athenea:lod:threshold` píxeles (1 por defecto): criterio de error en pantalla, como en los LOD clásicos de Nanite [35] o de splats jerárquicos [2]. Los niveles no dibujados no se posan (`deferredPose`: se guarda la pose pendiente y se aplica cuando una vista lo pide), así que no cuestan kernel de skinning.

**Las cifras** — llenando el frame, el nivel elegido es el más fino (su celda ya ocupa unos 2 px) y el tiempo apenas cambia (49.6 → 51.5 ms); a cuatro veces la distancia se elige el nivel 1 (1.5 M) y el frame baja de 71.3 a 24.7 ms. El coste: memoria (todos los niveles residentes, cada uno con sus campos de visibilidad de 81 MB) y salto, no fundido, al cambiar de nivel.

*Referencias:* [2], [35] del §10; código: `modules/usd/src/Engine.cpp` (`Engine::lodLevelsFor`, `deferredPose`), `modules/usd/include/athenea/usd/Export.h` (`writeLodAssembly`); `docs/decisions.md` (subsección "Levels of detail for a cloud that moves").
:::

**Diezmado que conserva el rig.** `usd::decimateStage` encuentra cada array
por Gaussian y hace merge de cada uno según su tipo (`lod_attributes.slang`).
Cuando `skel:jointIndices` tiene un `skel:jointWeights` a juego, el par se
combina como rig (modo 3). Sobre el tramo de registros $R$ al que representa
una Gaussian conservada, cada uno con peso de merge $\omega_r$,

$$c_j=\sum_{r\in R}\omega_r\sum_k w_{rk}\,[\,j_{rk}=j\,],$$

reuniendo hasta 16 joints distintos. Se conservan los `width` más pesados y
se renormalizan, $w'_j = \min(c_j/\sum_{\text{kept}} c,\,1)$: una división en
el device no está correctamente redondeada, y un merge entero sobre un
joint salía a $1 + 10^{-7}$. El joint dominante de un
registro se pliega además en la clave de merge, así que un merge nunca cruza
el borde de un joint. Comprobado: un suelo cuyas mitades pertenecen a dos
joints pasa de 4096 a 256 Gaussians, cada Gaussian conservada entera en el
joint de su lado. El nivel más fino del gorrión (5 887 323 Gaussians, con ids,
material, partes de visibilidad y rig) conserva el 87.7 % en 28 s. Posado en
el tiempo 20, se dibuja como la nube sin diezmar con un RMS de imagen de
0.002.

::: {.nota}
**`usd::decimateStage`** — reduce una nube por merge de Gaussians (§6) y, para cada array por Gaussian del fichero, elige cómo combinarlo (`lod::AttributeMerge`): media ponderada (modo 0), «el primero» para lo que no se puede mezclar (ids, partes; modo 1), bits por mayoría (2), rig (3), dirección renormalizada (4), lobes zonales (5). El kernel es `lodMergeAttribute` en `lod_attributes.slang`.

**Modo rig** — para cada Gaussian conservada, que representa un tramo $R$ de registros originales con pesos de merge $\omega_r$ (los mismos que ponderan sus momentos), se acumula cuánto peso lleva cada joint: $c_j=\sum_r\omega_r\sum_kw_{rk}[j_{rk}=j]$ (el corchete de Iverson vale 1 si el $k$-ésimo joint del registro es $j$). Se reúnen hasta 16 joints distintos (`kMostJoints`), se conservan los `width` (4) más pesados y se renormalizan.

**El $\min(\cdot,1)$** — la división en GPU no está correctamente redondeada (Metal y CUDA usan aproximaciones de recíproco con hasta un par de ulp de error), y $c/c$ podía dar $1+10^{-7}$; el recorte evita pesos ligeramente mayores que uno, que romperían invariantes de pruebas y lectores estrictos.

**El joint dominante en la clave** — la clave de merge (lo que decide qué registros pueden fusionarse) incluye el joint dominante, de modo que un merge nunca cruza la frontera entre joints: el problema de la celda ala+cuerpo de §5.9 queda excluido por construcción. Prueba: suelo de dos mitades con joints distintos, 4096 → 256 Gaussians, cada una entera en el joint de su lado.

**El gorrión** — 5 887 323 Gaussians con ids, material, partes de visibilidad y rig se reducen al 87.7 % en 28 s; posado en $t=20$, RMS de imagen 0.002 frente a la nube sin diezmar.

*Referencias:* [9], [34] del §10; código: `modules/usd/src/Decimate.cpp` (planes `lod::AttributeMerge::Rig`), `modules/usd/include/athenea/usd/Export.h` (`decimateStage`), `shaders/athenea/lod/lod_attributes.slang` (`lodMergeAttribute`, `kMostJoints`), `tests/usd/test_usd.cpp`; `docs/decisions.md` (secciones "Decimation: the levels of detail, cut once by what a merge loses" y "A decimation keeps everything a cloud carries").
:::

## 5.10 El gorrión

El gorrión y un BMW fueron los dos primeros assets convertidos que no se
crearon para este renderer, y la mayor parte de lo que destapó estaba fuera
de las matemáticas del skinning (figura 15).

::: {.nota}
**Assets «no hechos para este renderer»** — hasta el gorrión y un BMW, los assets de prueba estaban construidos a mano o venían de Pixar (Kitchen Set, etc.) y respetaban las convenciones del motor. Un asset de producción importado (aquí, de FBX a Blender a USD) trae convenciones ajenas: rest y bind distintas, mapas de normales premultiplicados, alfa en un segundo juego de UV, metálico mal asignado. La mayoría de los defectos encontrados no tienen que ver con la matemática del skinning sino con la cadena de exportación y lectura.

*Referencias:* `docs/decisions.md` (secciones "Two assets out of Blender, and the two bugs they found", "What the sparrow needed on Blender's side").
:::

- **Exportación de la animación.** El exportador de Blender no escribía
  muestras de joints para este rig importado de FBX, así que
  `scripts/sparrow-anim.py` muestrea la pose evaluada. Resolver un cambio de
  base frente a `restTransforms` destrozaba la cabeza, el pico y los dedos,
  porque reposo y bind son poses distintas en este pájaro (peor 1.989 en
  `Toe.L.010`). Ahora escribe $W(t)=P(t)\,L^{-1}B$, de modo que $W(t)B^{-1}$
  es la propia matriz de skinning de Blender; lo que queda es cuantización,
  $4.9\times10^{-5}$–$7.1\times10^{-5}$ en los frames 1, 31, 61.
- **Stage corregido.** `SparrowBird.usda` lleva los arreglos del asset: nada
  de metálico en plumas ni ojos, un mapa de normales de pluma sin
  premultiplicar (`UV2_treesparrow_normal_fixed.png`) y el segundo juego de
  UV (`UVMap_001`) para el alfa de las plumas. Todas las medidas posteriores lo
  usan.
- **Recortes y budget.** Respetar la `opacity` conectada al alfa de las
  plumas bajó su budget de 15 378 695 a 5 898 220 Gaussians a
  resolución 1400. Los meshes por encima del techo de 2 M por pasada se
  convierten por trozos; con todas las tarjetas, el pájaro tiene 9.5 M
  Gaussians.
- **Referencia de mesh.** Hacer que coincidieran los renders raster y de path
  trace del mesh bajó el relMSE del raster frente al path tracer de 7.28 a
  3.67 (960×540).

::: {.nota}
**Rest frente a bind** — en UsdSkel, `restTransforms` (locales) describen la pose de reposo y `bindTransforms` (de mundo) la pose en la que se ligó la malla; Blender escribe la primera desde la pose de reposo de la armature y la segunda desde la pose del bind. UsdSkel calcula la matriz de skinning como $W(t)\,B^{-1}$. El script resolvía un cambio de base contra *rest*, que se cancela contra rest pero no contra bind: los joints donde ambas difieren (dedos, cabeza, pico; peor discrepancia 1.989 en `Toe.L.010`, cerca de media vuelta) quedaban con una rotación espuria. La «comprobación de ida y vuelta» de la versión anterior no lo detectó porque la base estaba *definida* para que la ida y vuelta de rest funcionase: un test tautológico.

**La fórmula correcta** — con $P(t)$ la matriz de mundo evaluada del hueso en Blender (`pose.matrix`), $L$ su matriz de reposo en el espacio de la armature (`matrix_local`) y $B$ la `bindTransform`, escribir $W(t)=P(t)L^{-1}B$ hace que $W(t)B^{-1}=P(t)L^{-1}$, exactamente la matriz de skinning de Blender, sin depender de la convención de ejes de hueso de ninguno de los dos lados. El error residual ($4.9$–$7.1\times10^{-5}$) es la cuantización del fichero (seis decimales, escalas en `half3`). Además, los nombres de joint de USD son los de los huesos con los caracteres no válidos en identificadores cambiados por `_` (p. ej. `Spine.001_Pelvis` → `Spine_001_Pelvis`).

**Stage corregido** — `SparrowBird.usda` (no `SparrowFlight.usda`) aplica los arreglos de material: plumas y ojos sin metálico, mapa de normales «sin premultiplicar» (`UV2_treesparrow_normal_fixed.png`, con los canales RGB no multiplicados por alfa, lo que sesgaba las normales en bordes) y la opacidad de plumas leída por el segundo juego de UV (`UVMap_001`).

**Recortes y budget** — respetar la `opacity` conectada al alfa (las tarjetas de pluma son rectángulos con alfa recortado) evita sembrar Gaussians en las zonas transparentes: de 15.4 M a 5.9 M a resolución 1400. Los meshes que superarían el techo de 2 M de Gaussians por pasada de conversión se procesan por trozos.

**Referencia de mesh** — para comparar la nube con el mesh primero hay que asegurarse de que raster y path trace del *mesh* coinciden: hacerlos coincidir bajó el relMSE entre ambos de 7.28 a 3.67.

*Referencias:* código: `scripts/sparrow-anim.py`; `docs/decisions.md` (secciones "The sparrow's animation, and a skinning matrix that has to be true", "A feather is a hole in a rectangle", "The feathers read their shape by a second set of coordinates", "A mesh past the ceiling is converted in slices, and a barb keeps its alpha", "The sparrow's mesh, the same bird on both routes").
:::

Medido nube frente a mesh (`--no-bake`, cámara del ala, alfa sin dome,
relMSE con él), después de que la coverage pasara a seguir la opacidad:

| | alfa antes → después (mesh) | relMSE antes → después |
|---|---|---|
| ala del gorrión, raster | 0.2395 → 0.2367 (0.2253) | 0.1742 → 0.1635 |
| ala del gorrión, trazada | 0.2385 → 0.2358 (0.2462) | 0.1331 → 0.1247 |

::: {.nota}
**Coverage y opacidad** — la *coverage* (alfa acumulado de la imagen) mide qué fracción del píxel tapa el objeto. Con `--no-bake` (sin luz horneada) y cámara al ala, se compara el alfa medio de la nube con el del mesh (entre paréntesis), y el relMSE del color con dome. El cambio «la coverage sigue a la opacidad» hace que una Gaussian convertida cubra exactamente lo que dice la opacidad del material (y no trate la opacidad como transmisión de vidrio). Mejora ambas rutas: el relMSE baja un 6 % y el alfa se acerca al del mesh en la trazada. Que el alfa raster quede por encima del mesh (0.2367 frente a 0.2253) y el trazado por debajo (0.2358 frente a 0.2462) refleja que cada ruta integra las Gaussians de forma distinta (elipse proyectada frente a densidad a lo largo del rayo).

*Referencias:* `docs/decisions.md` (secciones "Opacity is coverage, not glass", "A converted surface covers what its opacity says").
:::

![Figura 15. El gorrión: el mesh con path trace (izquierda) y las Gaussians convertidas, rasterizadas (derecha).](figures/img-sparrow.jpg)

Para la web, el pájaro se convirtió con `--skinned --transfer
--resolution 512`, con bake de lobes zonales a 256 paths y 3 bounces en la
bind pose y la visibilidad de 12 partes añadida (`sparrow_tx_vis.usdc`:
1 504 160 Gaussians, 1 504 096 con contenido). Frente al path trace del mesh
a 1024 paths (1024², `air_fly_A0`, frame 16), el render raster de la nube da
relMSE 0.0025. El cuerpo de la nube sale algo más oscuro y marrón, porque el
transfer zonal solo guarda la luz directa.

::: {.nota}
**La conversión para la web** — `--skinned --transfer --resolution 512`: nube con rig y lobes zonales horneados con 256 paths por Gaussian y 3 bounces, en la bind pose (§5.7), más los campos de 12 partes (§5.8). 1 504 160 Gaussians, de las que 1 504 096 tienen contenido (las 64 restantes quedan sin contenido; el texto no detalla la causa). Frente al path trace del mesh a 1024 spp: relMSE 0.0025, muy bajo; el sesgo restante (cuerpo más oscuro y marrón) se explica porque el transfer zonal solo guarda la mitad *directa*: falta la luz que rebota entre plumas y desde el suelo.

*Referencias:* `docs/decisions.md` (secciones "A skinned cloud keeps its transfer: zonal lobes in each gaussian's frame", "A skinned transfer is traced at the bind pose").
:::

## 5.11 El peón: el control rígido

El peón de ajedrez (The Open Chess Set, ASWF, CC BY 4.0) **no tiene
skeleton**. Son dos meshes: un cuerpo de mármol negro `M_Pawn_Body_B` y una
cabeza `M_Pawn_Top_B` con `transmission = 1`, una bola de cristal sobre un
anillo de oro. Está en esta sección porque es el asset que todo cambio
relacionado con la animación tenía que dejar igual:

- Cuando `MeshStageOptions` ganó un tiempo, el peón se convirtió en un fichero
  idéntico byte a byte al de antes.
- Cuando el eje plano pasó a ser
  $\min(\text{along}_U,\text{along}_V)\times0.1$ para corregir el fantasma del
  zorro bajo el ray tracer, el mármol del peón pasó de
  0.0644/0.0798/0.0778 a 0.0645/0.0799/0.0780.
- Toda ruta con skinning, con transformaciones identidad, se reduce a la
  rígida, y el peón es la referencia de la rígida.

::: {.nota}
**The Open Chess Set** — conjunto de piezas de ajedrez de la Academy Software Foundation (ASWF), publicado como referencia de USD/MaterialX, con licencia CC BY 4.0. El peón negro tiene dos materiales: mármol (cuerpo) y una cabeza de vidrio (`transmission = 1`, bola de cristal sobre anillo de oro), lo que lo convierte también en prueba dura de vidrio.

**Control rígido** — tres invariantes: (1) añadir un tiempo a `MeshStageOptions` no cambia un asset sin skeleton (fichero idéntico byte a byte: la mejor prueba de no regresión); (2) el cambio del eje plano: el grosor de una Gaussian era $10^{-7}$ en unidades del modelo, mientras las otras dos escalas son en celdas, así que en el zorro (cien unidades de largo) el disco era 1500 veces más extremo que en el peón (66 mm) y el ray tracer, que integra densidad a lo largo del rayo en vez de proyectar una elipse, veía un fantasma. Hacerlo relativo, $\min(\text{along}_U,\text{along}_V)\times0.1$, arregla el zorro y apenas mueve el peón (cambios en la cuarta cifra decimal de la media de color del mármol); (3) la ruta con skinning con identidades se reduce exactamente a la rígida.

*Referencias:* Academy Software Foundation, "The Open Chess Set" (CC BY 4.0); código: `modules/usd/include/athenea/usd/MeshStage.h` (`MeshStageOptions`), `plugins/mesh2splat/mesh2splat.slang`; `docs/decisions.md` (sección "The fox", tercer punto "The flat axis is a fraction, not a length").
:::

Su primera conversión de principio a fin produjo 729 073 Gaussians a
resolución 512. Sus resultados de material y transfer pertenecen a las
secciones 3 y 4. La validación de alta calidad del peón blanco
(`pawn_mesh.usda`, 2048², 2048 paths de referencia) da, por material:

| material | Gaussians | relMSE | p99 | media (nube) | media (path trace del mesh) |
|---|---|---|---|---|---|
| `M_Pawn_Body_W` | 1 568 499 | 0.0218 | 0.273 | 0.167 / 0.165 / 0.133 | 0.164 / 0.162 / 0.132 |
| `M_Pawn_Top_W` (cristal) | 307 296 | 0.1418 | 0.354 | 0.236 / 0.260 / 0.221 | 0.365 / 0.397 / 0.279 |

::: {.nota}
**La tabla del peón blanco** — a 2048² y con una referencia de 2048 paths por píxel, el cuerpo (1.57 M Gaussians) da relMSE 0.0218 y medias RGB a menos de un 2 % de las del mesh (0.167 frente a 0.164 en rojo); la cabeza de cristal (0.31 M) da 0.1418 y una media claramente más oscura (0.236 frente a 0.365): el vidrio, que depende de refracción y reflexión de lo que hay detrás, es lo más difícil de capturar en Gaussians con transfer de baja frecuencia. p99 es el percentil 99 del error por píxel.

*Referencias:* [24], [25], [30] del §10 (vidrio y microfacetas); `docs/decisions.md` (secciones "A transparent surface passes what it does not reflect", "A ray that went into a glass meets its far face").
:::

Tres conversiones posteriores del mismo peón, medidas a 1024², dan un relMSE
del cristal de 0.149, 0.081 y 0.043, con el cuerpo sin cambios en 0.025. El
cuerpo queda a unos pocos por ciento del path trace del mesh. El error se
concentra en la cabeza de cristal (figura 16).

::: {.nota}
**Evolución del cristal** — tres conversiones posteriores bajaron el relMSE del cristal de 0.149 a 0.081 y a 0.043 (a 1024²), con el cuerpo estable en 0.025: el progreso de §4 en vidrio (Fresnel, cara posterior, rayos de bake que salen de una lente) se concentra donde estaba el error.

*Referencias:* `docs/decisions.md` (secciones "A transmission pays its Fresnel once", "A glass gaussian's colour is its transmission's own").
:::

![Figura 16. El peón blanco a 2048²: el path trace del mesh con 2048 paths por píxel (izquierda) y las Gaussians convertidas (derecha).](figures/img-pawn.jpg)

## 5.12 Aplicación: reproducción en la web y en un teléfono

El player web (`sparkwebGPU`, un fork WebGPU de Spark) reutiliza los kernels
de athenea compilados a WGSL. El escritor de `.athc` de athenea rechaza
`--skinned`, así que el lado web lee la nube USD (`usd-athc`, sobre el crate
`openusd`) y escribe un `.athc` v3. Ese fichero añade una sección `SKIN` por
elemento, con las influencias empaquetadas como joint u16 | peso unorm16 y los
gradientes como halves, y un blob de skeleton `ATSK` con los nombres de los
joints, la transformación de bind y el primer clip. Cada uno de los demás
clips es un fichero `ATCL` que se descarga al reproducirlo. Los nodos LoD
resultado de un merge llevan la mezcla de las influencias de sus splats
ponderada por opacidad (las cuatro más pesadas, renormalizadas) y ningún
gradiente.

::: {.nota}
**sparkwebGPU y `.athc` v3** — el player es un fork con backend WebGPU de Spark [11] (World Labs), que ejecuta kernels de athenea compilados de Slang a WGSL [44], [46]. El escritor de `.athc` del motor no acepta `--skinned`, así que la herramienta `usd-athc` del player (Rust, sobre el crate `openusd`) lee la nube USD y escribe el `.athc` v3 con: una sección `SKIN` por elemento (influencias empaquetadas en 32 bits, índice de joint u16 | peso unorm16, y gradientes en halves), un blob `ATSK` con nombres de joints, transformación de bind y el primer clip, y ficheros `ATCL` por clip que se descargan al reproducirlos.

**unorm16** — entero sin signo de 16 bits interpretado como $x/65535\in[0,1]$: precisión de $1.5\times10^{-5}$, sobrada para pesos.

**Nodos LoD** — en el árbol LoD del player, un nodo resultado de merge lleva la mezcla de influencias de sus splats ponderada por opacidad (las cuatro más pesadas, renormalizadas, como el modo rig de §5.9) y sin gradientes (el Jacobian se reduce al término de mezcla).

*Referencias:* [11], [44], [46] del §10; player sparkwebGPU (demos públicas en https://sparkwebgpu.lucab.co.uk).
:::

El kernel de skinning (`athenea_adapter/skin.slang`) reescribe `splatSkin`
línea a línea, porque el original enlaza once storage buffers y Chrome en un
Mac permite diez por etapa. Una prueba de GPU ejecuta el kernel propio de
athenea sobre los mismos splats y compara los dos. Frente al raster de
athenea, el resultado es un relMSE de 0.0015–0.0053 con una razón de brillo
de 1.01. No se escribe movimiento de shutter.

::: {.nota}
**Por qué reescribir el kernel** — WebGPU fija límites por etapa de shader; el de storage buffers (`maxStorageBuffersPerShaderStage`) es 8 por defecto y Chrome en Mac expone 10. `splatSkin` enlaza 11 (incluidos los de la segunda pose del shutter y el buffer `motion`). La versión web omite el motion blur y además no toma `skelToWorld` ni `worldToPrim`, porque las transformaciones de la caché ya llevan el espacio del skeleton al de la nube. Una prueba de GPU ejecuta el kernel original de athenea sobre los mismos splats (en un device con los límites del adaptador) y compara: relMSE 0.0015–0.0053 frente al raster de athenea, razón de brillo 1.01.

*Referencias:* [46] del §10 (WebGPU, límites del adaptador); athenea: `shaders/athenea/scene/splat_skin.slang`.
:::

| página | contenido | medido |
|---|---|---|
| `/pawn` (publicada 2026-10-06) | cuerpo 727 061 + cabeza 673 711 splats, transfer, 6 HDRI CC0 | 60 fps en Chrome; conjunto de datos v8: descarga 281.3 → 82.7 MB; memoria de GPU en escritorio 1127 → 800 MB |
| `/sparrow` (publicada 2026-10-07) | 1 504 096 splats, 125.7 MB, 12 clips (6.4 MB gzip), transfer zonal, HDRI | memoria de GPU en escritorio 1084 → 684 MB |
| `/sparrow` en teléfonos | `sparrow-mobile`: 488 656 splats, 31.7 MB, cortado del tree LoD (`--keep-splats 600000`) | 60 fps headless en modo móvil, animado |
| sparrow-v2 (candidato) | 1 100 334 splats (−26.8 %), 75.9 MB; móvil 371 138, 26.1 MB | mismo error que el publicado (4.46e-3) |

::: {.nota}
**La tabla** — `/pawn`: dos nubes (cuerpo y cabeza) con transfer y 6 HDRI CC0 intercambiables; el «conjunto de datos v8» redujo la descarga de 281.3 a 82.7 MB y la memoria de GPU de 1127 a 800 MB (compresión de §7). `/sparrow`: 1.5 M splats en 125.7 MB, 12 clips que ocupan solo 6.4 MB comprimidos (el argumento de §5.1: los clips son matrices). La versión móvil es un *cut* del árbol LoD limitado a 600 000 splats (`--keep-splats`), 31.7 MB. sparrow-v2 aplica la poda de la última nota (−26.8 % de splats) con el mismo error medido (4.46e-3) que el publicado.

*Referencias:* §7 del artículo; demos públicas `/pawn` y `/sparrow` en https://sparkwebgpu.lucab.co.uk.
:::

**La ruta compacta.** Una nube con skinning también se reproduce por la ruta
compacta de §7: cada nodo lleva su skin en el registro compacto, el dibujo le
aplica el skinning, la caja de una page es la unión de sus cajas en todas las
poses, y los culls de oclusión y de caras traseras se apagan en lo que se mueve.
El merge tree del gorrión se parte en dos por raíces con escalas zonales
comunes, dos ficheros de unos 162 MiB con los clips y sus propios domos, 347 MB
en total (`sparrow-compact-tx`, publicado el 2026-10-18); en el escritorio dibuja
795 k Gaussians a $\tau$ 1.5 y 60 fps en vuelo, y el móvil emulado 232 k.

::: {.nota}
**La ruta compacta con skinning** — la ruta compacta de §7 guarda cada nodo del árbol en un registro reducido y lo pagina. Para una nube animada: (1) el skin va en el propio registro compacto, y el dibujo aplica el skinning; (2) la caja de una page (unidad de streaming y de culling) es la unión de sus cajas en *todas* las poses, para que el culling por frustum siga siendo conservador al moverse; (3) el cull de oclusión y el de caras traseras se desactivan para lo que se mueve (lo oculto o de espaldas en una pose puede no estarlo en otra). $\tau$ es el umbral de error en pantalla del cut (más alto = menos Gaussians).

*Referencias:* §7 del artículo.
:::

**Antialiasing temporal.** Se probó en el player una resolución temporal (el
esquema de Karis: jitter de Halton, reprojection, clamp en YCoCg y una media
exponencial), con vectores de movimiento por Gaussian, los de las alas a partir
de las matrices de hueso del frame anterior. Los vectores son correctos (error de
reprojection 0.05 → 0.015–0.028 en las alas, 0.018 → 0.003 en el coche), pero en
movimiento el frame no mejora al frame sin ella en el coche, el gorrión ni la
ruta compacta: muestra más error, parpadeo y estelas, porque los brillos son HDR
y dependen de la vista y el jitter no se promedia. Las variantes (historia
Catmull–Rom, el recorte por varianza de Salvi, un $\alpha$ según la velocidad,
afilado adaptativo al contraste) no lo cambian. Con la cámara quieta iguala una
acumulación de 16 frames. Cuesta 3 ms a 1080p y queda como opción.

::: {.nota}
**Antialiasing temporal (TAA)** — el esquema de Karis (2014): cada frame la cámara se desplaza un subpíxel siguiendo una secuencia de Halton [48]; el frame anterior (la *historia*) se reproyecta con vectores de movimiento; la historia se *recorta* (clamp) a la caja de colores del vecindario actual en el espacio YCoCg (luma/croma, donde la caja se ajusta mejor) para evitar fantasmas; y se mezcla con media exponencial (típicamente 10 % del frame nuevo). Variantes probadas: muestreo de la historia con filtro Catmull–Rom (más nítido que el bilineal), *variance clipping* de Salvi (2016: caja a $\mu\pm\gamma\sigma$ del vecindario en vez de min/max), $\alpha$ dependiente de la velocidad y afilado adaptativo al contraste (tipo AMD CAS).

**Por qué no gana aquí** — los vectores de movimiento son correctos (error de reproyección muy bajo, también en las alas con las matrices de hueso del frame anterior), pero los brillos de las Gaussians relit son HDR y dependientes de la vista: el clamp los recorta o los deja parpadear, y el jitter no se promedia en movimiento. Con la cámara quieta converge a una acumulación de 16 frames. Queda como opción (3 ms a 1080p).

*Referencias:* [48] del §10 (Halton); Karis (2014), "High Quality Temporal Supersampling", SIGGRAPH Advances in Real-Time Rendering; Salvi (2016), "An Excursion in Temporal Supersampling", GDC.
:::

**Podar un asset que se mueve.** Las Gaussians ocultas en la bind pose no se
pueden quitar sin más, porque otras poses las muestran. Medido sobre 12
clips × 4 muestras = 48 poses más la bind pose, con 61 vistas cada una: el
61.2 % del gorrión está oculto en la bind pose, pero solo el **32.5 %** lo
está en todas las poses. Quitar lo que oculta la bind pose cuesta un relMSE
de hasta $4.5\times10^{-2}$ en vistas cercanas de poses no muestreadas. Quitar
lo que ocultan todas las poses cuesta como mucho $2.9\times10^{-4}$.
sparrow-v2 es lo segundo. Como comparación, el peón estático solo tiene un
2.7 % oculto.

::: {.nota}
**Podar con poses** — una Gaussian totalmente oculta (tapada por otras desde cualquier vista) no contribuye a la imagen y se puede quitar, pero en un asset animado debe estar oculta en *todas* las poses. Muestreo: 12 clips $\times$ 4 instantes + bind = 49 poses, 61 vistas cada una. En la bind pose el 61.2 % está oculto (el interior del cuerpo, las plumas bajo otras plumas), pero solo el 32.5 % lo está en todas las poses: un ala que se abre descubre lo que tapaba. Quitar lo oculto solo en bind pose cuesta relMSE hasta $4.5\times10^{-2}$ en vistas cercanas de poses no muestreadas; quitar lo oculto en todas, como mucho $2.9\times10^{-4}$. El peón estático, sin interior ocluido apenas, solo tiene un 2.7 % oculto.

**Riesgo** — es una poda *empírica*: una pose no muestreada podría revelar algo quitado; por eso se mide el error en poses que no participaron en la selección.

*Referencias:* [10] del §10 (GaussianPOP, poda por error).
:::

## 5.13 Lo que no está hecho

- Más de cuatro influencias por Gaussian. Esto limita la precisión en las
  puntas de las alas del gorrión, tanto en la animación como en la
  coincidencia del bake con el mesh.
- El skinning por cuaterniones duales para nubes. El retargeting y la mezcla
  de clips, que una nube con transformaciones de bake no puede hacer.
- Varios meshes con skinning con transformaciones de prim distintas (solo se
  conserva la del primero), y skeletons con transformación animada propia.
- La ruta trazada posa una vez por frame, así que no aplica motion blur a una
  nube con skinning. Las instancias LoD no llevan movimiento.
- El término de autosombra propio de una parte. La oclusión entre partes en
  poses distintas de la bind pose, dentro del transfer zonal. Los bakes de
  visibilidad por encima de unos 3 M Gaussians por parte.
- El error del ajuste zonal en el gorrión, y una medida geométrica de la
  coverage en una flexión vista de cerca, que es lo que mostraría la ganancia
  del término elástico del Jacobian.

::: {.nota}
**Lo que falta, agrupado** — (1) *Fidelidad del rig*: más de cuatro influencias, DQS para nubes, varios meshes con transformaciones de prim distintas, skeletons con transformación animada propia. (2) *Animación*: retargeting y mezcla de clips (una nube con transformaciones horneadas en el primvar no tiene joints locales que mezclar; habría que pasar por el Skeleton). (3) *Movimiento*: la ruta trazada posa una vez por frame (sin motion blur de skinning) y las instancias LoD no llevan desplazamiento. (4) *Sombra*: autosombra intra-parte (la literatura propone AO canónico por hueso [42] o un trazado corto), oclusión entre partes dentro del transfer zonal en poses distintas, y bakes por encima de unos 3 M Gaussians por parte (límite de memoria del BLAS). (5) *Medidas*: el error del ajuste zonal en el gorrión y una medida geométrica de coverage en una flexión de cerca, que es donde se vería la ganancia del término elástico.

*Referencias:* [42] del §10; `docs/decisions.md` (párrafos "Not done" de las secciones "And the engine puts the cloud where the skeleton is", "The whole Jacobian of a skinned gaussian", "A skinned transfer is traced at the bind pose", "A cloud shadows itself by part…").
:::
