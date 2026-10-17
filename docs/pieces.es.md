<!-- Copyright (c) 2026 jesus luque. -->

# El Corvette en un iPhone: lo aplicado (11-12 oct 2026)

El Corvette entero, con su transfer, dibujado en un iPhone y juzgado contra el path trace de la malla
(`athenea-renders/simplify3/pt/day_pt_Camera.exr`, `/World/Camera`, 1920 x 1080), en el fotograma entero y en recortes
de las piezas pequeñas. Este documento recoge lo aplicado en `athenea-pieces` (rama `pieces`) y lo decidido con el
visor web (sparkwebGPU, sesión `sparkwebgpu-ce`), con lo que se midió. El registro en curso es `PIECES-PLAN.md`; la
versión en inglés de este documento es `docs/pieces.md`.

Las cifras marcadas «recorte del coche» son PSNR en el recorte de 900 x 420 del coche (`--cut 900x420+200+420`)
contra el path trace; «fotograma» es el fotograma entero de 1920 x 1080. Los cocientes como 1.04 son la media de una
región sobre la del path trace (las cajas de research, `athenea-research/proto/door/*.py`).

## 1. Sombreado

### 1.1 Los lóbulos nítidos leídos a la rugosidad anterior al ensanche de la cocción (b26d276)

La cocción ensancha la rugosidad guardada de una gaussiana por la dispersión de normales bajo ella (Toksvig,
`--specular-filter`): alpha'^2 = alpha^2 + 2 s^2, con s = kMax max(scale) del operador de forma. Los lóbulos nítidos
por píxel (`SplatSlope`) ya giran la normal a lo largo de la gaussiana, así que leerlos a la rugosidad ensanchada
contaba la dispersión dos veces. `unwidened(r, s) = (r^4 - 2 s^2)^(1/4)` (acotado en 0) la deshace para los lóbulos
nítidos del coat y del polish; `slope.bakeSpread` lleva s.

Research (t64): el contraste de los reflejos abiertos de la puerta pasó de 0.12/0.26 a 0.40/0.61 del del path trace, y
el del capó de 0.27 a 0.41; la oscuridad de la clase suelo (0.67-0.78) y de las juntas (0.69) se quedó.

### 1.2 El polish ve su suelo (35ae3cb)

Portado de `athenea` (tx-transfer-complete, 4c7dc40): la nube lleva la rugosidad propia del material,
`primvars:athenea:splat:materialRoughness` (antes de `--specular-filter`; -1 bajo un mapa). Donde es al menos
`kPolishFieldRoughness` (0.3), el polish consulta las celdas de su propio lóbulo y el campo reflejado acoplado al
cielo responde lo que cierran; por debajo, o sin el array, como antes. La pintura es 0.34; el cromo 0 y el metal
rugoso 0.22 mantienen oscura la parte cerrada.

El valor va en la palabra pbr: el bit 26 dice que está y los bits 27-31 lo guardan en 5 bits
(`kMaterialRoughnessMark`, 8.0 por paso sobre la transmisión). El kernel de streams ya enlaza todos los buffers que
Metal da a un kernel, así que la marca se añade después en una pasada propia, `splatMaterialRoughness` en
`scene/splat_hardness.slang`, junto a la de la dureza.

### 1.3 El coat de un pliegue acotado a 0.3, solo en pintura (35ae3cb, 5d0c530)

En un pliegue la cocción ensanchaba la laca hasta casi 1, y una junta dibujaba una raya oscura. `kCoatFoldRoughness`
(0.3) la acota: coat = max(unwidened(c, s), min(c, 0.3)). Medido por regiones en el visor web, iluminaba el cromo, el
cristal y los radios de una llanta que el path trace muestra oscuros (cromo -1.4 dB, radios -0.5 dB), así que solo se
aplica donde la rugosidad propia del material es al menos `kPolishFieldRoughness`. A 1280 px la cocción ya no
ensancha el coat (sale a 0), así que este término no cambia nada en el coche re-cocido.

### 1.4 Un tubo gira, un pliegue se ensancha (f9c9a2b)

El giro por píxel extrapola la normal de una gaussiana a partir del operador de forma de su triángulo central; más
allá de una arista deja de valer. La normal y la curvatura guardadas son exactas en el centro (research, hoodnormal:
p90 0.009 grados, correlación 1.000); el 64 % de las huellas de un capó cruzan una arista.

t, el giro a lo largo de +-2 sigma, es el doble de la dispersión de la cocción. Research (tubecrease) halló que separa
los dos casos en la nube de 1280 px: el cuerpo de un tubo cromado está en t 0.3-1, coherente con sus vecinas; las
normales suavizadas a través de un pliegue dentro de triángulos más estrechos que la gaussiana están por encima de 1
(5-9 rad en un umbral).

| t | la normal del píxel | los lóbulos nítidos |
|---|---|---|
| bajo `kFoldTurnLow` (0.7) | gira entera | como están guardados |
| entre medias | smoothstep | smoothstep |
| por encima de `kFoldTurnHigh` (1.5) | no gira | ensanchados: alpha' = sqrt(alpha^2 + (t/2)^2), acotados bajo `kFoldRoughness` (0.5) (`foldWidened`) |

### 1.5 Lo que se probó y se dejó fuera

| cambio | prueba | resultado |
|---|---|---|
| el giro acotado a 0.2 rad sobre la huella (`kMaxFootprintTurn`) | T69 | recorte del coche 16.55 -> 16.57; el capó no se movió (3.16 px); las juntas de la llanta delantera 1.48 -> 1.54. Sustituido por 1.4 |
| el polish nítido decidido por la rugosidad sin ensanchar | T73 | el error de paso alto de la tira cromada se triplicó (0.41 -> 1.12 del del path trace) |
| un suelo de coat, o huella 0, en el móvil | CW | las manchas de la puerta no cambiaron: eran de la nube |
| `--specular-filter 0` | prueba de cocción B, C | una línea oscura discontinua en el pliegue del umbral, y el cromo centellea. El polish necesita su ensanche ahí |
| `--edge-reach 1.0` | prueba de cocción C | elongación y pelos idénticos a B |

### 1.6 Resultados frente al path trace

| prueba | qué | recorte del coche | fotograma |
|---|---|---|---|
| T63c | 640 px, corte 0.25 px con cull de espalda | 16.29 | -- |
| T65 cut_a | + 1.1 y 1.3 | 16.51 | 22.15 |
| T65 cut_b | + 1.2 | 16.55 | 22.18 |
| T66 | hojas (corte 0.01 px), sin cull / con cull | 16.73 / 16.75 | 22.40 / 22.43 |
| T69 / T70 | + la cota de 0.2 rad / + coat solo en pintura | 16.57 / 16.57 | 22.20 / 22.20 |
| T71 | **cocción a 1280 px**, corte 0.25 px con cull de espalda (5.0M) | **17.88** | **23.90** |
| T72 | T71 sin la cota | 17.88 | 23.91 |
| T73 | T71 + el criterio de 1.5 | 17.88 | 23.90 |
| T74 | T73 + 1.4 | 17.90 | 23.93 |
| T75 | T71 + 1.4 (commiteado) | 17.88 | 23.91 |

T66 (research t66): el cull de espalda es correcto. Sube toda la puerta un 12 % aproximadamente, por igual (puerta
1.01, suelo cercano 1.02 del path trace); los reflejos abiertos de la puerta a 1.13 son el coat dejando pasar cielo
donde el espejo da contra el propio coche (celdas de 16 x 16 de 11 grados). T71 frente a cut_b (research): los brillos
del capó desplazados 3.2 -> 2.0 px, media 0.89 -> 0.98, juntas 0.90 -> 1.02; el contraste de los reflejos abiertos de
la puerta 0.35/0.53 -> 0.56/0.74; los huecos de la parrilla en el corte 2.24 -> 1.04. T75 frente a T71: motas
brillantes en el pliegue del umbral 10 -> 1, la puerta 405 -> 389, la llanta 204 -> 190.

## 2. Decimate y el árbol de merges

### 2.1 Opciones nuevas

`--mark-material-roughness`, `--cut-spill`, `--cut-spill-floor` (ver `docs/operations.es.md`, 2.4.1).
`--tree-coverage` ahora escribe también `primvars:athenea:splat:lodSpill`.

### 2.2 El derrame de un merge (1bb79f4)

E = sum max(a G - C, 0) / sum a G sobre la propia rejilla del ajuste de cobertura (8 x 8 sobre +-1.25 sigma en el
plano del nodo, C el compuesto de las hojas, a la cobertura ajustada): la fracción de lo que pinta un merge que sus
hojas no pintan. Coincide con el numpy de research con correlación 0.994. El corte suma r w_s max(E - E0, 0) al error
de un nodo.

T67 en el grupo de la parrilla: w_s 2 no tapa sus huecos más que las hojas (1.90 -> 1.09 de ellas; recorte del coche
16.55 -> 16.79), pero guarda un 82-99 % más de nodos. Research (spill_*.py): todo merge derrama alrededor de 0.16, y
los merges de dos hojas de las tiras subpíxel de Chrome_Part_Grill que tapan los huecos no derraman más que los de un
panel; ningún umbral los separa, ni ninguna medida independiente de la vista. **Por defecto 0.** A 1280 px los huecos
están a 1.04 sin él.

### 2.3 Árboles de millones de hojas (5ed0b6a)

Una rueda a 5120 px (4.45M hojas, 8.88M nodos) construyó su árbol y se quedó sin memoria del dispositivo al
escribirlo.

- `mergeAttribute` reservaba un buffer de pesos tan grande como cada atributo, aunque solo lo escriben las juntas de
  un rig (modo 3): 1.7 GB junto a cada uno de los arrays de 48 floats de un transfer. En los demás casos es una
  palabra.
- La escritura de decimate guardaba los arrays tres veces en la memoria que comparte la GPU: una copia de lo leído, y
  los arrays fusionados empezaban como otra copia. Ahora empiezan solo con los metadatos, y la escritura del árbol, la
  última, libera cada array en cuanto está en el dispositivo.

Medido después: una puerta de 4.08M hojas, 10.5 GB de memoria residente máxima; 4.90M pasó y 5.01M no al principio (sección 3.4);
una ejecución posterior lo escribió, con 2.23 GB como la memoria libre más baja.

### 2.4 La occlusion escrita sobre cada árbol

`occlusionMasksShared` escribe un árbol cuya salida es su entrada junto a él (`<árbol>.occlusion.usdc`) y lo renombra
encima: los árboles del nivel de cerca ocupan 87.4 GB juntos, y una copia de cada uno no cabría a su lado. Ejecutada
sobre los 24 árboles de car5120: un codebook compartido de 8,182 entradas en 645 s, 10.2 GB de memoria residente
máxima; por árbol se reabre del 0 al 23.6 % de las celdas cerradas (Windshield 0.0 %, pan2 23.6 %). El primer total
marcó «132.4 %»: el total de celdas cerradas sobre ~150M nodos desbordó un contador de 32 bits; las cifras por árbol y
el codebook eran correctos, y los totales se suman ahora en 64 bits (4a8f8c2).

## 3. Cocciones

### 3.1 El coche entero a 1280 px (car1280)

Las hojas de la cocción de 640 px medían 3-4 mm en el capó; las rayas de la puerta y los pelos del umbral estaban en
la nube (las hojas los mostraban; el path trace no). El coche se re-coció a `--camera-pixels 1280` con los ajustes de
la cocción de 640, en 8 grupos de piezas de primer nivel equilibrados por splats. Cada grupo se convierte con todas
las demás piezas ocultas con `--hide`: `--hide` solo afecta a la conversión, y la cocción traza el stage entero, así
que una pieza conserva las sombras de las demás. Los nombres de las piezas salen del manifiesto de cryptomatte (las
líneas entrelazadas del log desfiguraban algunos).

| | 640 px | 1280 px |
|---|---|---|
| hojas | 5.38M | 8.70M |
| un panel (capó / puerta) | 50k / 79k | 151k / 266k |
| cristal, neumáticos | una por triángulo | una por triángulo (apenas crecen) |
| occlusion 8192 compartida, celdas reabiertas | 11.9 % | 12.3 % |
| T71 corte 0.25 px: recorte del coche / fotograma | 16.55 / 22.18 | 17.88 / 23.90 |

El build de pieces escribe `closed` y `materialRoughness` por gaussiana, cada material de una malla el suyo.

### 3.2 La prueba de cocción (8 piezas a 1280 px)

Door, Door_001, Panel_under_Door, Panel_under_Door_001, Front_Fender, Chrome_Part_Grill, Grill_Rim, Hood; 931,017
gaussianas cada una. Research (attr.py): a 1280 px la rugosidad del coat sale exactamente a 0 (la cocción de 640 la
tenía a 0.17-0.24 con un 5-14 % de paso alto en los pliegues); los pelos del umbral bajan de 987 a 459 px solo por la
resolución.

| variante | ajustes | resultado |
|---|---|---|
| **A** | como la cocción de 640 | **elegida**: sin pelos, la puerta lisa |
| B | `--specular-filter 0` | línea discontinua en el pliegue del umbral, el cromo centellea |
| C | B + `--edge-reach 1.0` | como B |

### 3.3 El nivel de cerca (car5120)

En el iPhone de cerca, el logo y la parrilla seguían blandos: la cámara estaba mucho más cerca que `/World/Camera`,
con unos 0.5 mm por píxel del teléfono frente a hojas de 3-4 mm. Una prueba de logo y parrilla a 1280 y 5120 px desde
una cámara cercana mostró nítidas las hojas de 5120 px (la bandera de cuadros, la pajarita, la malla de la parrilla);
108,527 -> 404,594 gaussianas.

- **Piezas con detalle**: las 66 piezas a las que la cocción de 1280 px ya daba como mucho 1.5 gaussianas por
  triángulo (emblemas, parrillas, faros, pinzas, retrovisores, ruedas, limpiaparabrisas, molduras), re-cocidas a
  5120 px en 7 grupos.
- **Paneles**: el usuario pidió también las piezas grandes. Cada pieza a 5120 px, o tan fina como la deje bajo un tope
  de hojas (5M, luego 4.5M después de que un árbol de 5.01M no cupiera), y las pequeñas agrupadas hasta 4M.
- `--max-cells 1048576`: a 4736 px, Infinite_Black perdió 359,628 celdas por encima de las 262,144 por triángulo del
  valor por defecto (huecos); se re-coció entera.

| grupos | hojas | resolución |
|---|---|---|
| fine0-6 (66 piezas con detalle) | 18,610,627 | 5120 |
| puerta (piloto) | 4,084,170 | 5120 |
| Bottom | 4,897,428 | 3392 |
| Part_3 | 4,370,459 | 4480 |
| Infinite_Black | 4,481,738 | 4736 |
| Panel_under_Door_001, Windshield, Front_Fender, Hood, Front_Bumper, Part_2, Part_23 | 15,994,566 | 5120 |
| pan0-4 (paneles pequeños) | 17,088,152 | 5120 |
| Front_Bumper_001 | 5,008,986 | 4480 |
| **24 árboles** | **74,536,126** | |

Los árboles de merges de los grupos son por pieza, así que el nivel de cerca tiene el mismo formato que car1280: las
raíces gruesas se quedan, y los niveles más profundos solo se paginan de cerca. Cada grupo toma el lock de GPU
compartido por su cuenta. Los 24 árboles
tienen 74,536,126 hojas, 74,328,031 merges y 208,095 raíces: 148,864,157 nodos en 49,634 páginas, 87.4 GB en disco,
con la occlusion compartida escrita en el sitio (2.4).

### 3.4 Límites encontrados, y cómo se resolvieron

| límite | dónde | resuelto con |
|---|---|---|
| disco (5 GB libres en un momento) | las cocciones de 1280 y 5120 | intermedios viejos borrados (con el visto bueno del usuario); la cocción de un grupo se borra en cuanto su árbol está escrito; occlusion en el sitio (2.4) |
| memoria del dispositivo al escribir un árbol | rueda de 4.45M hojas | 5ed0b6a (2.3) |
| memoria del dispositivo por encima de ~5M hojas | Front_Bumper_001 (5.01M) | un tope de 4.5M hojas por pieza |
| celdas por triángulo | Infinite_Black | `--max-cells 1048576` |
| la lista de grupos de 1280 px sacada de un log desfigurado | car1280 | nombres de pieza del manifiesto de cryptomatte |

## 4. El móvil (sparkwebGPU; decisiones, su código vive allí)

| qué | decidido porque |
|---|---|
| una sombra de contacto del shadow catcher (`pieces/shadow/shadow.usdc`, 309k, fusionado 2 x 2 a 78k, 2.5 MB de GPU) | el coche flotaba; el x1.6 del umbral era el suelo de la nave en el infinito visto a través del borde del coche. Umbral 1.66 -> 1.08 del path trace (escritorio 1.07). 3DGS puro: solo gaussianas, sin malla ni AO |
| el lóbulo nítido por píxel en el camino compacto, con la dispersión del píxel alpha^2 += (\|dx\|^2 + \|dy\|^2)/6 | recorte del coche 16.49 -> 17.46 dB (el del escritorio); sin la dispersión del píxel, v7 tenía motas brillantes en los pliegues (27/25 -> 2/1) |
| el entorno como atlas rgba16float filtrable | el fragmento nítido 11.8 -> 4.0 ms en el móvil |
| 512 clusters de transfer (el 9º bit en el bit 15 libre del índice de sombra) | un fichero del detalle a 5120 a 0.0181 de relMSE; con 512 a 0.0136, todos los demás grupos un 20-30 % mejor, el mismo tiempo de shade |
| una raíz sintética de varios niveles en el packer | un fichero admite unas 3000 raíces; los árboles de 5120 tienen miles de raíces sueltas (los merges nunca cruzan piezas ni guardas, a propósito) |
| v7 = car1280 | puerta 33.77 -> 35.53 dB, umbral 31.87 -> 32.36, capó 14.16 -> 14.46 |
| objetivo de dibujo por tiempo de frame; calidad antes que frame rate (usuario: «aunque empeore el rendimiento quiero calidad») | el iPhone real se asentó en 106k dibujadas, tau 8, 53 fps; de cerca 160k, tau 8, 32 fps |
| un fotograma quieto: tras 300 ms quieta, un fotograma a tau ~0.5-1.2 sin límite de tiempo, y nada más hasta un toque | de cerca 531k a tau 1.22 en 118 ms (emulado a 190 ms por millón); el primer fotograma tras un toque 39-47 ms |
| datos del coche ~100 MB en reposo, ~55 MB en movimiento (usuario) | las páginas más finas del fotograma quieto |
| v8 = car5120 (el nivel de cerca) | sus 24 árboles y la occlusion compartida están escritos; la conversión del reproductor estaba en curso |

Las cifras del móvil emulado son aproximadamente el doble de las del iPhone real; el número de gaussianas dibujadas se
juzga en el dispositivo.

## 5. Operación

Los trabajos de GPU se ejecutan de uno en uno en la máquina a través de `~/luc/athenea-gpu/shared.sh`, que toma un
directorio de lock compartido con los trabajos del visor web (`mkdir`) y luego ejecuta el trabajo bajo `guard.sh`
(detiene el trabajo con presión térmica, SIGSTOP y SIGCONT). Cada paso se ejecuta bajo `memguard.sh` (mata el trabajo
cuando la memoria libre baja de 1.5 GB). Un trabajo espera 25 s antes de pedir el lock, para que los trabajos que
consultan cada 10 s entren entre medias, y espera mientras exista `~/luc/athenea-gpu/hold` (un hueco largo cedido a
otra sesión). Una necesidad corta de GPU de otra sesión se atiende pausando la cocción en curso (SIGSTOP al binario y
a su `memguard.sh`, SIGCONT después). Un directorio de build que usa un trabajo en curso o en cola no se recompila.

## 6. Pendiente

- v8 (car5120) en el reproductor: su conversión, y después su calidad frente al path trace y su comportamiento en el
  iPhone real.
- Lámparas: la cocción no tiene la emisión de las lámparas (los faros salen oscuros frente al path trace).
- La tira cromada del parachoques delantero: un 38 % más clara que el path trace, con el pico más bajo (un tubo de
  unos pocos píxeles de ancho, cada gaussiana leyendo su espejo).
- El reflejo del suelo cercano en la llanta a 0.72 del path trace.
- Los reflejos abiertos de la puerta a 1.13: el coat deja pasar cielo por celdas de 16 x 16 de 11 grados donde el
  espejo da contra el coche; celdas de 32 x 32 (+96 B por gaussiana) lo reducirían a la mitad.
- El labio inferior del parachoques delantero, deshilachado; los brillos del capó desplazados 2.0 px (un ajuste de
  curvatura sobre la huella a partir de la malla reduciría a la mitad el error mediano).
- Los huecos de la parrilla en un corte grueso: ninguna medida por nodo los separa; a 1280 px y más fino están a 1.04.
