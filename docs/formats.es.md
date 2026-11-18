[English](formats.md) · Español

# Los formatos de fichero de athenea

Esta es la especificación de los ficheros que athenea escribe en formatos
propios: `.athc` (una nube con sus niveles de detalle, leída chunk a chunk),
`.athz` (una nube paginada y con codificación de entropía, que escribe
`athenea athz` en la rama `athz`), `.athv` (un vídeo de gaussianas, y los
mensajes que lleva un stream suyo) y `.aths` (una escena neural: un
directorio de piezas, sellos y capas, que escribe `athenea neural-scene` en
la rama `neural`). Está escrita para que baste a un lector
independiente. El código que describe se cita en cada estructura; donde los
dos discrepan, el código tiene razón y este fichero está equivocado.

`docs/operations.md` dice qué comando escribe cada fichero y con qué opciones
(§2.4 `convert`, §8.2 Formatos); `docs/decisions.md` dice por qué cada campo
es como es. El schema de USD y sus primvars `athenea:splat:*` se especifican
en `docs/operations.md` §4.3, no aquí.

## 1. Convenciones

- **Orden de bytes.** Little endian en todo, salvo donde un campo diga otra
  cosa (las longitudes de las unidades NAL de HEVC de `.athv`, que son big
  endian como las tiene el propio encapsulado `hvcC` de HEVC).
- **Tipos.** `u8`, `u16`, `u32`, `u64` sin signo; `i32` en complemento a dos;
  `f16` IEEE 754 binary16; `f32` binary32; `f64` binary64. Una *palabra* es
  un `u32`. Un *par de medias* es una palabra cuyos 16 bits bajos son el
  primer `f16` y cuyos 16 bits altos son el segundo.
- **Arrays.** Los bloques se disponen como estructuras de arrays: cada
  atributo de todos los elementos de un bloque, y después el atributo
  siguiente.
- **Los offsets** están en bytes desde el inicio del fichero salvo que un
  campo diga desde dónde.
- **Rechazo.** Un lector rechaza un fichero cuyo magic, versión, cuentas u
  offsets no cuadren, y un bit de flag que no conozca: un bit posterior puede
  añadir un array, y un lector que se lo saltara leería todos los arrays que
  le siguen desde el sitio equivocado.
- **Escritura.** Todo escritor escribe con un nombre temporal junto al
  destino (`<name>.partial`) y lo renombra cuando el fichero está completo,
  así que un fichero con su propio nombre nunca está escrito a medias.

## 2. Las palabras de una gaussiana

`.athc` conserva el empaquetado propio del dispositivo
(`shaders/athenea/common/packing.slang`,
`shaders/athenea/scene/splat_encoding.slang`, `GpuSplats` en
`modules/scene/include/athenea/scene/GpuClouds.h`); `.athz` se decodifica en
él. Cada elemento de una nube son las palabras siguientes.

### 2.1 Posición: `f32` × 4, 16 bytes

| Offset | Tipo | Campo |
|---|---|---|
| 0 | f32 × 3 | x, y, z: el centro en el espacio propio de la nube, en las unidades de su fuente (metros en todos los escritores de athenea) |
| 12 | f32 | opacidad, lineal, de 0 a 1 (la sigmoide ya aplicada) |

No se registra ningún eje vertical. La opacidad de un grupo fundido es como
mucho 0.99.

### 2.2 Forma: `u32` × 4, 16 bytes

| Palabra | Bits | Campo |
|---|---|---|
| 0 | todos | rotación, las tres menores (abajo) |
| 1 | 0–15, 16–31 | `f16` ln(escala x), `f16` ln(escala y) |
| 2 | 0–15, 16–31 | `f16` ln(escala z), `f16` color base r |
| 3 | 0–15, 16–31 | `f16` color base g, `f16` color base b |

**Rotación.** Un cuaternio unitario (x, y, z, w). Los bits 0–1 son el índice
de su componente mayor en magnitud (0 x, 1 y, 2 z, 3 w); el cuaternio se
niega antes de empaquetarlo si esa componente es negativa, así que se
sobrentiende positiva. Las otras tres, por orden de índice, están en los bits
22–31, 12–21 y 2–11, cada una `round(1023 · (q / √½ · 0.5 + 0.5))` limitada a
0..1023. Para decodificar: cada componente pequeña es `(v / 1023 · 2 − 1) ·
√½`, y la mayor es `√max(0, 1 − Σ small²)`. La rotación lleva los ejes
propios de la gaussiana (las columnas de su matriz de rotación) al espacio de
la nube; las escalas son sus desviaciones típicas a lo largo de esos ejes.

**Escalas.** Logaritmos naturales en `f16`, limitados a ±65000 antes de la
conversión. Una escala cuyo logaritmo está por debajo de −100 es exactamente
cero: la gaussiana es un **surfel** (una gaussiana 2D, un disco plano). Los
escritores de athenea ponen el eje cero en tercer lugar, así que los dos
primeros abarcan el disco y el tercero es su normal. El tercer `f16` v de un
surfel lleva además su **dureza** p, la caída exp(−(|u|^p + |v|^p)/2) en sus
propios ejes: p = 2 + (−v − 1024) / 64 donde −60000 < v ≤ −1024 (los
escritores mantienen p ≤ 32, v ≥ −2944), y p = 2, una gaussiana, en otro caso
(los escritores usan −65000). Sus dos primeras escalas son entonces las de
momentos igualados, así que un lector que ignora p dibuja la gaussiana con
los mismos momentos.

**Color base.** El término DC como color: `0.5 + 0.28209479 · f_dc` para un
coeficiente DC de armónicos esféricos de 3DGS `f_dc`. Puede salirse de 0..1.
Está codificado para pantalla (sRGB tal como lo tiene una captura) salvo que
el fichero diga que sus colores son luz lineal (bit 1 de los flags de
`.athc`).

### 2.3 Armónicos: `shWords` palabras

Los coeficientes de resto (grado 1 y superiores), `f16`, rojo, verde y azul
de la base 0, después de la base 1, y así sucesivamente, dos por palabra, el
primero en la mitad baja; una cuenta impar deja a cero la última mitad alta.
Las bases están en el orden de 3DGS (grado 1: y, z, x; grado 2: xy, yz,
3z²−1, xz, x²−y²; grado 3 con la misma convención). `restPerColour` es 0, 3,
8 o 15 (grado 0 a 3), y `shWords = (3 · restPerColour + 1) / 2`: 0, 5, 12 o
23. Un escritor puede conservar una palabra a cero en grado 0.

### 2.4 Palabras opcionales

Cada una es un array del bloque, presente donde lo diga el fichero.

| Array | Palabras | Codificación |
|---|---|---|
| normales | 1 | la normal de sombreado, octaédrica (abajo): x en los bits 0–15, y en los bits 16–31, cada una `round(65535 · (p · 0.5 + 0.5))` |
| emisión | 1 | la radiancia que emite la gaussiana, RGB9E5 lineal (abajo) |
| pbr | 1 | el material (abajo) |
| capas | 3 | las capas sobre la base (abajo) |
| transfer | `transferWords` | los `transferCount` valores del transfer como pares de medias (abajo) |
| direcciones abiertas | `shadowWords` | un bit por celda de una rejilla octaédrica (abajo) |
| curvatura | 1 | el operador de forma (abajo) |

**Mapa octaédrico** (`shaders/athenea/common/octahedral.slang`). Una
dirección n se divide por |x| + |y| + |z|; donde z < 0, (x, y) pasa a ser
`(1 − |y|, 1 − |x|)` con los signos de x y de y; el resultado en −1..1 se
lleva a 0..1 como `p · 0.5 + 0.5`. Decodificación: f = 2u − 1, n = (f.x, f.y,
1 − |f.x| − |f.y|), t = max(0, −n.z), después x e y se acercan cada una t
hacia cero, y se normaliza. Una normal cero se empaqueta como +z.

**RGB9E5.** Bits 0–8, 9–17, 18–26 las mantisas m de rojo, verde y azul, bits
27–31 el exponente e: cada canal es `m · 2^(e − 24)`. Sin signo, hasta 65408;
los negativos y NaN se escriben como cero.

**pbr.** Bits 0–7 metallic, 8–15 roughness, 16–23 transmisión, cada uno
`round(255 · v)`. Bit 24: una pared fina (una lámina cuya transmisión es su
propia transparencia). Bit 25: un metal de Schlick (el conductor de OpenPBR y
de glTF). Bit 26: los bits 27–31 guardan la roughness propia del material,
antes de que la conversión la ensanchara con la dispersión de las normales
bajo la gaussiana, como `bits / 31`.

**Capas.** Bytes, cada uno `round(255 · v)` salvo que se diga otra cosa:

| Palabra | Bits 0–7 | 8–15 | 16–23 | 24–31 |
|---|---|---|---|---|
| 0 | color specular r | g | b | peso specular |
| 1 | peso del coat | roughness del coat | bits 0–6 IOR del coat como `1 + b / 64`; bit 7 oscurecimiento del coat | IOR specular como `1 + b / 128` |
| 2 | color del sheen r | g | b | roughness del sheen |

Una gaussiana sin capas tiene peso specular 1, color blanco, IOR 1.5, ni coat
ni sheen.

**Transfer** (`shaders/athenea/common/transfer_layout.slang`). Cómo responde
la gaussiana a la luz, trazado por path tracing una vez. La cuenta de valores
es la disposición:

| `transferCount` | Directo | Indirecto | Campo |
|---|---|---|---|
| 9 | 9 | — | — |
| 36 | 9 | 27 | — |
| 84 | 9 | 27 | 48 |
| 16 | 16 | — | — |
| 64 | 16 | 48 | — |
| 112 | 16 | 48 | 48 |
| 10 | zonal (abajo) | | |

La mitad directa es un valor por base armónica (grado 2: 9, grado 3: 16), en
el orden SH real estándar. La mitad indirecta son tres por base, rojo, verde
y azul de la base 0 primero. El campo reflejado son 16 coeficientes rgb, del
mismo modo. Los valores son `f16`; el valor 2k es la mitad baja de la palabra
k. `transferWords = (transferCount + 1) / 2`. Un transfer zonal (10) son dos
lóbulos, cada uno su eje en el marco propio de la gaussiana (u, v) y tres
coeficientes zonales (z0, z1, z2), en ese orden, con dos palabras de
direcciones abiertas en el mismo marco.

**Direcciones abiertas.** Por qué lados la salida de la gaussiana encontró
abierta el horneado, un bit por celda de una rejilla octaédrica de lado ×
lado sobre toda la esfera: lado 8 en 2 palabras (el primer transfer), 16 en 8
y 32 en 32 (un transfer TX). La celda (x, y) es el bit `(y · side + x) & 31`
de la palabra `(y · side + x) >> 5`; su dirección es la decodificación
octaédrica de `((x + 0.5) / side, (y + 0.5) / side)`. Un bit a uno es
abierto.

**Curvatura.** El operador de forma (uu, uv, vv por unidad de longitud) en
los ejes propios de la gaussiana. Los bits 24–31 son `top`, un `i8`: ⌈log2⌉
de la componente mayor. Los bits 0–7, 8–15, 16–23 guardan uu, uv, vv: el bit
7 el signo, los bits 0–6 q, el tamaño `2^(top − 16 + q · 16 / 127)`; q = 0 es
cero. Un grupo fundido de más de una gaussiana es plano (la palabra es cero).

## 3. `.athc`: una nube con sus niveles de detalle

Lo escriben `athenea convert` (§2.4 de `docs/operations.md`) y `writeAthc`
(`modules/lod/src/Athc.cpp`); lo lee `readAthc`, y chunk a chunk
`StreamingPool` (`modules/lod/include/athenea/lod/Athc.h`).

### 3.1 Disposición

```
offset 0      cabecera (104 bytes), y después la cabecera extra (32 bytes) donde está el bit 4, 5 o 6 de los flags,
              y después las palabras de origen (32 bytes) en el 136, en todos los ficheros
página 1      la tabla de niveles, y después la tabla de chunks
alineado      los inicios de los grupos del nivel más fino
alineado      cada nivel, el más grueso primero: un bloque
alineado      cada chunk, en orden: un bloque
```

Cada bloque, y la tabla de inicios, empieza en un límite de 4096 bytes, así
que un chunk se mapea y se carga en memoria solo; los huecos son cero, y la
longitud del fichero es un múltiplo de 4096.

Los splats están en orden Morton (§3.4), cortados en **chunks** de
`chunkSplats` splats consecutivos (el último lleva el resto). Un **nivel** l
lleva una gaussiana fundida (un **grupo**) por cada celda del octree a
profundidad `level` que contiene splats.

### 3.2 Cabecera

| Offset | Tamaño | Tipo | Campo | Significado |
|---|---|---|---|---|
| 0 | 4 | char[4] | magic | `ATHC` |
| 4 | 4 | u32 | version | 1 o 2 |
| 8 | 4 | u32 | count | splats, > 0 |
| 12 | 4 | u32 | restPerColour | 0, 3, 8, 15 |
| 16 | 4 | u32 | shWords | palabras de armónicos por elemento, de 1 a 64 |
| 20 | 4 | u32 | levels | niveles fundidos, ≥ 1 |
| 24 | 4 | u32 | chunkSplats | splats por chunk, > 0 |
| 28 | 4 | u32 | chunks | `⌈count / chunkSplats⌉` |
| 32 | 4 | u32 | finestGroups | grupos del nivel más fino |
| 36 | 12 | f32 × 3 | boundsLo | la esquina inferior del octree |
| 48 | 4 | f32 | extent | la arista del octree: la arista mayor de la caja |
| 52 | 12 | f32 × 3 | boundsMin | la caja de los splats |
| 64 | 12 | f32 × 3 | boundsMax | |
| 76 | 4 | u32 | flags | versión 2; cero (padding) en la versión 1 |
| 80 | 8 | u64 | levelTable | offset de la tabla de niveles: 4096 |
| 88 | 8 | u64 | chunkTable | offset de la tabla de chunks |
| 96 | 8 | u64 | starts | offset de la tabla de inicios |

Un fichero se escribe como versión 2 si y solo si `flags` no es cero; un
lector de la versión 1 trata `flags` como cero.

**Flags.**

| Bit | Valor | Significado |
|---|---|---|
| 0 | 1 | cada bloque lleva normales |
| 1 | 2 | los colores son luz lineal (sin array) |
| 2 | 4 | cada bloque lleva emisión |
| 3 | 8 | reservado (propuesta 009); se rechaza |
| 4 | 16 | cada bloque lleva el material: pbr y, según la cabecera extra, capas |
| 5 | 32 | cada bloque lleva un transfer y, según la cabecera extra, direcciones abiertas |
| 6 | 64 | cada bloque lleva curvatura |

Cualquier otro bit se rechaza.

**Cabecera extra** (en el offset 104, solo donde está el bit 4, 5 o 6):

| Offset | Tipo | Campo | Restricción |
|---|---|---|---|
| 104 | u32 | pbrWords | 1 si y solo si el bit 4 |
| 108 | u32 | lobesWords | 0 o 3 |
| 112 | u32 | transferCount | 9, 36, 84, 16, 64, 112 o 10 |
| 116 | u32 | transferWords | `(transferCount + 1) / 2`, ≤ 64; distinto de cero si y solo si el bit 5 |
| 120 | u32 | shadowWords | 0, 2, 8 o 32 |
| 124 | u32 | curvatureWords | 1 si y solo si el bit 6 |
| 128 | u32 × 2 | pad | cero |

**Palabras de origen** (en el offset 136, en todos los ficheros; ceros en uno
escrito antes, que se leen como «sin decir»): lo que el origen de la nube decía
de sí misma y no lleva ningún bloque. Son el padding de la página 0, así que un
lector que no las conoce se las salta; ningún bit de flags dice que están.

| Offset | Tipo | Campo | Restricción |
|---|---|---|---|
| 136 | u32 | antialiased | 0 sin decir, 1 ajustada sin el filtro de Mip-Splatting, 2 con él (el flag de SPZ); un lector rechaza cualquier otro valor |
| 140 | u32 × 7 | pad | cero |

### 3.3 Tablas

**Tabla de niveles** (`levels` entradas, 16 bytes cada una, en `levelTable`),
la más gruesa primero:

| Offset | Tipo | Campo | Restricción |
|---|---|---|---|
| 0 | u32 | level | profundidad del octree, de 1 a 10; cada entrada una más que la anterior |
| 4 | u32 | groups | > 0; el de la última entrada es `finestGroups` |
| 8 | u64 | offset | el bloque del nivel |

**Tabla de chunks** (`chunks` entradas, 16 bytes cada una, justo después de
la tabla de niveles):

| Offset | Tipo | Campo | Restricción |
|---|---|---|---|
| 0 | u64 | offset | el bloque del chunk |
| 8 | u32 | count | `min(chunkSplats, count − c · chunkSplats)` |
| 12 | u32 | pad | cero |

**Tabla de inicios** (`finestGroups` palabras en `starts`): para cada grupo
del nivel más fino, el índice de su primer splat en el orden Morton. Los
splats de un grupo son el tramo hasta el inicio del grupo siguiente.

### 3.4 Bloques

Un bloque de n elementos son estos arrays, cada uno empezando donde termina
el anterior:

| Array | Bytes | Presente |
|---|---|---|
| posiciones | 16 n | siempre (§2.1) |
| forma | 16 n | siempre (§2.2) |
| armónicos | 4 · shWords · n | siempre (§2.3) |
| cola | 4 n | siempre: en el bloque de un nivel, el código de celda de cada grupo; en el bloque de un chunk, el grupo del nivel más fino de cada splat |
| normales | 4 n | bit 0 de los flags |
| emisión | 4 n | bit 2 de los flags |
| pbr | 4 n | `pbrWords` = 1 |
| capas | 12 n | `lobesWords` = 3 |
| transfer | 4 · transferWords · n | `transferWords` > 0 |
| direcciones abiertas | 4 · shadowWords · n | `shadowWords` > 0 |
| curvatura | 4 n | `curvatureWords` = 1 |

Los bytes de un elemento son por tanto `16 + 16 + 4·shWords + 4 +
4·(normals) + 4·(emission) + 4·(pbrWords + lobesWords + transferWords +
shadowWords + curvatureWords)`, y un lector comprueba que cada bloque cabe en
el fichero.

**Orden Morton y códigos de celda** (`shaders/athenea/common/morton.slang`).
La posición unitaria de un punto es `(p − boundsLo) / extent`, limitada a
0..1; cada eje se cuantiza a 10 bits, `min(⌊1024 · u⌋, 1023)`, y los tres se
entrelazan en un código de 30 bits con x en el bit más alto de cada terna (el
bit 29 es el bit alto de x, el 28 el de y, el 27 el de z). La celda del
octree a profundidad r que contiene un punto son los 3r bits altos de su
código, `code >> (30 − 3r)`; esa es la palabra de cola del bloque de un
nivel. Los splats se ordenan por una clave cuya parte baja es el código de su
centro y cuya parte alta es una clase que una fusión no puede mezclar
(`shaders/athenea/lod/lod_class.slang`: 0–5 el eje y el signo hacia los que
más se inclina la normal de sombreado, 6 vidrio y láminas, que el corte nunca
dibuja fundidos). Un grupo es una clase en una celda, y sus splats son un
tramo del orden. La clase no se guarda: un lector toma los grupos de las
colas de los bloques de nivel, el grupo más fino de cada splat de las colas
de los bloques de chunk, y los tramos de la tabla de inicios. Dos grupos de
un nivel pueden compartir código de celda.

**Grupos fundidos** (`shaders/athenea/lod/lod_common.slang`). Un grupo es una
gaussiana con los momentos de los splats que tiene debajo, cada uno pesado
por w = opacidad × el área de sus dos ejes más anchos: media = Σ w μ / W,
covarianza = Σ w (Σ + (μ − m)(μ − m)ᵀ) / W (descompuesta en autovalores en la
rotación y las escalas), color = Σ w c / W, opacidad = min(W / su propia área
de dos ejes, 0.99). Un lector no necesita nada de esto para dibujar un
fichero; dice qué son los elementos de un nivel.

**Dibujar un corte.** Un grupo se dibuja cuando el tamaño proyectado de su
celda es como mucho el umbral y el de su padre no lo es; un splat se dibuja
cuando la celda del nivel más fino que lo contiene no lo es. A lo largo de
cualquier camino hacia abajo en el árbol la prueba cambia una vez, así que
cada lugar se dibuja exactamente en un nivel. El umbral 0 dibuja todos los
splats y ningún grupo.

### 3.5 Versiones

- **Versión 1**: sin flags (la palabra es padding).
- **Versión 2**: `flags` y, con los bits 4–6, la cabecera extra.
- **Palabras de origen** (13 nov 2026, sin cambio de versión): el
  antialiasing en el offset 136. Los ficheros escritos antes tienen ceros ahí
  y se leen como sin decir.
- **`.lrtc`** (lucabRTrender, el nombre anterior del motor): la versión 1 de
  esta disposición con el magic `LRTC`. `athenea` lo lee como un `.athc` y
  `migrateLrtc` reescribe solo la cabecera.
- **La versión 3** (`ATH3`) no es de athenea: la escribe sparkwebGPU para la
  web (§7).

## 4. `.athz`: una nube paginada y con codificación de entropía

Un fichero `.athz` guarda una nube con sus niveles de detalle: los grupos de
cada nivel fundido y los splats propios de la nube. Decodificado, cada
elemento es el registro que guarda un bloque de chunk de `.athc` (§3, §2),
bit a bit, seguido donde lo hay de la curvatura y de un rig de skinning; solo
el transfer codificado como CPCA (§4.12) es con pérdidas. Los elementos se
codifican en páginas, cada página en hasta cuatro capas independientes, cada
capa como planos de bytes codificados en crudo (empaquetados en bits) o por
rANS entrelazado de 32 carriles, tras una transformación entera por palabra.

Fuentes: `modules/lod/include/athenea/lod/Athz.h`, `modules/lod/src/Athz.cpp`,
`shaders/athenea/lod/athz_common.slang`, `athz_decode.slangh`,
`athz_encode.slang`, `athz_tables.slang`, `athz_cpca.slang`,
`athz_interleave.slang` (rama `athz`). Los shaders `lod_pages`,
`transfer_codec`, `geometry_quantize`, `shadow_vq` y `splat_compact` no los
usa `.athz`; `athz_cpca.slang` reimplementa el códec CPCA de
`transfer_codec.slang`.

### 4.1 Convenciones

- Todos los enteros son little endian, sin signo salvo que se diga. `f16` es
  IEEE 754 binary16, `f32` binary32, `f64` binary64, todos little endian.
- Los campos de varios bytes de la tabla del asset están empaquetados sin
  padding de alineación (un `u64` puede estar en un offset que no es múltiplo
  de 8).
- Una «palabra» es un `u32`. El byte `i` de una palabra (i = 0..3) son los
  bits `8i .. 8i+7`, es decir, el byte en la dirección `4w + i` de un array de
  palabras little endian.
- Un par de «medias» en una palabra: la mitad baja son los bits 0..15, la
  mitad alta los bits 16..31.
- `aligned(x, a) = ceil(x / a) * a`.
- CRC-32 es el `crc32` de zlib: el polinomio de IEEE 802.3 0x04C11DB7 (forma
  reflejada 0xEDB88320), valor inicial 0xFFFFFFFF, XOR final 0xFFFFFFFF, es
  decir, `crc32(0, bytes, n)`. Un CRC sobre varios rangos de bytes es el CRC
  de su concatenación.
- `zigzag(d) = (d << 1) XOR (0 - (d >> 31))` sobre 32 bits;
  `unzigzag(z) = (z >> 1) XOR (0 - (z & 1))`.
  `zigzag16`/`unzigzag16` son lo mismo sobre 16 bits, con los resultados
  enmascarados a 16 bits.
- `sub16x2(a, b)` / `add16x2(a, b)`: resta / suma de cada mitad de 16 bits
  por separado, módulo 2^16.
  `zigzag16x2` / `unzigzag16x2`: `zigzag16` / `unzigzag16` sobre cada mitad.

### 4.2 Disposición del fichero

Fuente: `Athz.cpp` (`writeAthz`, `prefixPages`).

| Región | Offset | Tamaño | Alineación |
|---|---|---|---|
| Cabecera | 0 | 64 | — |
| Extensiones (TLV) | `tlvOffset` (64) | hasta 4032 | 8 por registro |
| Padding a cero | | hasta 4096 | |
| Tabla del asset | `assetOffset` (4096) | `assetBytes`, un múltiplo de 64 | 4096 |
| Tabla de páginas | `pageTableOffset` = `aligned(4096 + assetBytes, 64)` | `pages * 64` | 64 |
| Inicios del nivel más fino | `startsOffset` | `finestGroups * 4` | 4096 |
| Base CPCA (si hay CPCA) | `basisOffset` | `basisBytes` | 4096 |
| Frames de la capa K (si hay capa K) | `temporalOffset` | `temporalBytes` | 4096 |
| Sección G | `sectionOffset[0]` | `sectionBytes[0]` | 4096 |
| Sección A (si está) | `sectionOffset[1]` | `sectionBytes[1]` | 4096 |
| Sección T (si está) | `sectionOffset[2]` | `sectionBytes[2]` | 4096 |
| Sección K (si está) | `sectionOffset[3]` | `sectionBytes[3]` | 4096 |

Todo hueco se llena de ceros, y el fichero se rellena con ceros hasta un
múltiplo de 4096. El escritor dispone las regiones en el orden de arriba, cada
región posterior en `aligned(end of previous, 4096)`. Un lector toma todos
los offsets de la cabecera y de la tabla del asset y no debe suponer la
colocación del escritor más allá de lo que se dice comprobado abajo.

### 4.3 Cabecera

Fuente: `Athz.cpp` (`encodeHeader`, `decodeHeader`).

| Offset | Tamaño | Tipo | Campo | Significado |
|---|---|---|---|---|
| 0 | 4 | char[4] | magic | `'A' 'T' 'H' 'Z'` |
| 4 | 2 | u16 | version | de 1 a 4; el escritor escribe 4 |
| 6 | 2 | u16 | headerBytes | 64 |
| 8 | 4 | u32 | flags | conjunto de bits, abajo |
| 12 | 4 | u32 | layerMask | bit L a uno cuando la capa L está en el fichero (1 G, 2 A, 4 T, 8 K) |
| 16 | 8 | u64 | gaussians | los splats propios de la nube; igual al `count` de la tabla del asset |
| 24 | 4 | u32 | pages | entradas de la tabla de páginas (menos que el total del asset en un prefijo) |
| 28 | 2 | u16 | runGaussians | R, elementos por tramo: de 32 a 256, múltiplo de 32 |
| 30 | 2 | u16 | pageRuns | tramos por página, al menos 1 |
| 32 | 8 | u64 | assetOffset | el offset de la tabla del asset; el escritor escribe 4096 |
| 40 | 4 | u32 | assetBytes | el tamaño de la tabla del asset, padding incluido |
| 44 | 8 | u64 | pageTableOffset | el offset de la tabla de páginas |
| 52 | 2 | u16 | entryBytes | 64 |
| 54 | 2 | u16 | reserved | 0 |
| 56 | 4 | u32 | tlvOffset | el offset de la primera extensión (64), 0 si no hay ninguna |
| 60 | 4 | u32 | headerCrc | CRC-32 de los bytes 0 a 59 |

Flags:

| Bit | Valor | Nombre | Significado |
|---|---|---|---|
| 0 | 1 | Transfer | la capa T está presente |
| 1 | 2 | Harmonics | `restPerColour > 0` (armónicos más allá del DC en la capa A) |
| 2 | 4 | Temporal | la capa K está presente |
| 3 | 8 | Linear | los colores están en luz lineal (el `linear` de la nube decodificada) |
| 4 | 16 | Predicted | al menos una entrada de plano (de cualquier tipo) tiene una transformación ≥ 4 (codificada frente al padre) |
| 5 | 32 | Emission | los elementos llevan una palabra de emisión RGB9E5 |

Solo el bit 3 cambia cómo se interpreta la nube decodificada. Los bits 0, 1,
2, 4 y 5 son descriptivos; el decodificador toma la disposición del elemento
de la tabla del asset y las transformaciones de las entradas de plano. Un bit
de flag fuera de 0x3F se rechaza.

### 4.4 Extensiones

Fuente: `Athz.cpp` (`writeAthz`, `parse`).

Una secuencia de registros que empieza en `tlvOffset`, cada uno:

| Offset | Tamaño | Tipo | Campo | Significado |
|---|---|---|---|---|
| 0 | 2 | u16 | type | 0 termina la lista; 1 es el nombre de la fuente |
| 2 | 2 | u16 | flags | se escribe 0; no se lee |
| 4 | 4 | u32 | length | bytes de la carga |
| 8 | length | u8[] | payload | |
| 8 + length | | | padding | ceros hasta `aligned(length, 8)` |

El registro siguiente empieza en `8 + aligned(length, 8)` desde este. La lista
termina con un registro de tipo 0 (8 bytes a cero; su longitud no se usa). Un
registro que pasa de `assetOffset` se rechaza; el recorrido se detiene en
`assetOffset` si no encuentra un terminador. Los tipos desconocidos se
saltan.

- Tipo 1 (fuente): el nombre del fichero desde el que se convirtió el asset
  (el escritor guarda el nombre de fichero de la entrada, como mucho 2048
  bytes, sin terminador). Su codificación de caracteres es la que tuviera el
  nombre en el sistema de ficheros; el código lo trata como bytes.

### 4.5 Tabla del asset

Fuente: `Athz.cpp` (`encodeAsset`, `decodeAsset`).

En `assetOffset`, de `assetBytes` de largo. Los offsets de abajo son de la
versión 4; véase §4.15 para la 1 a la 3.

| Offset | Tamaño | Tipo | Campo | Significado |
|---|---|---|---|---|
| 0 | 4 | u32 | bytes | el tamaño de la tabla, padding incluido; debe ser igual al `assetBytes` de la cabecera |
| 4 | 4 | u32 | kinds | 2 (tipo 0 grupos fundidos, tipo 1 los splats de la nube) |
| 8 | 4 | u32 | planes | P, planos de bytes por elemento: 4 × W (W, las palabras del elemento codificado) |
| 12 | 4 | u32 | tableLog | n, log2 del tamaño de una tabla rANS: de 9 a 12 |
| 16 | 4 | u32 | tables | T, tablas rANS en el fichero |
| 20 | 4 | u32 | levels | niveles fundidos: de 1 a 30 |
| 24 | 4 | u32 | count | los splats de la nube (> 0) |
| 28 | 4 | u32 | finestGroups | grupos del nivel fundido más fino |
| 32 | 4 | u32 | chunkSplats | elementos por página: debe ser igual a R × pageRuns, como mucho 65535 |
| 36 | 4 | u32 | restPerColour | coeficientes de armónicos más allá del DC por color: 0, 3, 8, 15 |
| 40 | 4 | u32 | shWords | palabras de armónicos por elemento: de 1 a 64 (una palabra de relleno en grado 0) |
| 44 | 4 | u32 | normals | 1: una palabra de normal por elemento |
| 48 | 4 | u32 | emission | 1: una palabra de emisión RGB9E5 por elemento |
| 52 | 4 | u32 | pbrWords | 0 o 1 |
| 56 | 4 | u32 | lobesWords | 0 o 3 |
| 60 | 4 | u32 | transferCount | valores del transfer decodificado (112 para un transfer TX de grado 3) |
| 64 | 4 | u32 | transferWords | palabras del transfer decodificado por elemento (pares f16) |
| 68 | 4 | u32 | shadowWords | palabras de direcciones abiertas por elemento |
| 72 | 4 | u32 | curvatureWords | 0 o 3 (v3+) |
| 76 | 4 | u32 | codedTransferWords | 0: transfer sin pérdidas; 4: transfer codificado como CPCA (v3+) |
| 80 | 4 | u32 | cpcaClusters | C, de 1 a 256 donde hay CPCA (el escritor escribe 256) (v3+) |
| 84 | 4 | u32 | cpcaComponents | K, ejes por bloque: 8 donde hay CPCA (v3+) |
| 88 | 4 | f32 | cpcaLevels | pasos de cuantización por lado (el escritor escribe 7.0) (v3+) |
| 92 | 8 | u64 | basisOffset | el offset de la base CPCA (v3+) |
| 100 | 8 | u64 | basisBytes | su tamaño (v3+) |
| 108 | 4 | u32 | influenceWords | 0, u 8 con la capa K (v4) |
| 112 | 4 | u32 | gradientWords | 0, o 3 con la capa K y gradientes (v4) |
| 116 | 4 | u32 | joints | J, las articulaciones del rig (v4) |
| 120 | 4 | u32 | frames | F, frames autorados; distinto de cero exactamente cuando lo es influenceWords (v4) |
| 124 | 8 | u64 | temporalOffset | el offset del bloque de frames de la capa K (v4) |
| 132 | 8 | u64 | temporalBytes | su tamaño: 64 + 8F + 24FJ cuando F ≠ 0 (v4) |
| 140 | 12 | f32[3] | boundsLo | la esquina inferior de la raíz del octree |
| 152 | 4 | f32 | extent | la arista de la raíz del octree (la arista mayor de la caja) |
| 156 | 12 | f32[3] | boundsMin | la caja alineada con los ejes de la nube, mínimo |
| 168 | 12 | f32[3] | boundsMax | máximo |
| 180 | 8 | u64 | assetId | FNV-1a 64 de esta tabla con este campo a cero (abajo) |
| 188 | 32 | u64[4] | sectionOffset | por capa G, A, T, K: el offset de la sección; 0 para una capa ausente |
| 220 | 32 | u64[4] | sectionBytes | por capa: el tamaño de la sección |
| 252 | 8 | u64 | startsOffset | los inicios del nivel más fino |
| 260 | 8 | u64 | startsBytes | debe ser igual a finestGroups × 4 |
| 268 | 16 × levels | LevelEntry[] | levels | el más grueso primero |
| … | 16 × 2P | PlaneEntry[2][P] | planeEntries | las P entradas del tipo 0, después las del tipo 1 |
| … | 512 × T | u16[T][256] | frequencies | las 256 frecuencias de símbolo de cada tabla rANS |
| … | | | padding | ceros hasta un múltiplo de 64 |

`assetId`: FNV-1a de 64 bits (base de offset 0xCBF29CE484222325, primo
0x100000001B3, aplicado byte a byte) sobre toda la tabla codificada, padding
incluido, con el campo `assetId` a 0 y todos los demás campos con su valor
final. Un prefijo (§4.14) conserva el id del fichero del que se cortó sin
recalcularlo. Ningún lector del código lo comprueba.

La cuenta de palabras W del elemento codificado es

    W = 4 + 4 + shWords + 1 + normals + emission + pbrWords + lobesWords
        + transferWordsCoded + shadowWords + curvatureWords
        + influenceWords + gradientWords

donde `transferWordsCoded` es `codedTransferWords` (4) con CPCA y
`transferWords` en otro caso. `planes` debe ser igual a 4W.

#### LevelEntry (16 bytes)

| Offset | Tamaño | Tipo | Campo | Significado |
|---|---|---|---|---|
| 0 | 4 | u32 | level | nivel del octree: celdas de arista `extent / 2^level` |
| 4 | 4 | u32 | groups | los grupos fundidos del nivel (> 0) |
| 8 | 4 | u32 | firstPage | su primera página en la tabla de páginas |
| 12 | 4 | u32 | pages | `ceil(groups / chunkSplats)` |

Se comprueba: la entrada 0 tiene `firstPage` 0; el `firstPage` de cada
entrada es el `firstPage + pages` de la anterior; cada `level` es el de la
anterior + 1; el `groups` de la última entrada es igual a `finestGroups`.

#### PlaneEntry (16 bytes, cuatro u32)

Fuente: `athz_common.slang` (`athzEncoding` … `athzTransform`).

| Offset | Tamaño | Tipo | Campo | Significado |
|---|---|---|---|---|
| 0 | 4 | u32 | coding | bits 0–7 codificación (0 crudo, 1 rANS); 8–15 anchura (de 0 a 8, bits en crudo por símbolo); 16–23 base (de 0 a 255); 24–31 capa (0 G, 1 A, 2 T, 3 K) |
| 4 | 4 | u32 | table | codificación 1: el índice de la tabla rANS (< T); codificación 0: 0xFFFFFFFF |
| 8 | 4 | u32 | rawStart | codificación 0: dónde empiezan las palabras en crudo del plano en la parte en crudo de un tramo, en palabras; codificación 1: 0 |
| 12 | 4 | u32 | transform | v2+: bits 0–7 la transformación de la palabra (de 0 a 8; la de la palabra de cola de 0 a 3). v1: una estimación del codificador, leída como transformación 0 |

El plano `b` (0 ≤ b < P) es el byte `b & 3` de la palabra `b >> 2` del
elemento. Los cuatro planos de una palabra llevan la misma transformación; el
decodificador la lee del último plano de la palabra (`b & 3 == 3`). `width` y
`base` están también en los planos rANS, pero solo se usan en los planos en
crudo.

Los índices de tabla se asignan en orden: los planos del tipo 0 b = 0..P−1,
después los del tipo 1, y cada plano rANS toma el índice siguiente. Cada tipo
tiene por tanto su propia tabla por plano rANS.

`rawStart`, por tipo y por capa: una suma acumulada sobre los planos en crudo
de esa capa en `b` creciente, cada uno sumando `ceil(R × width / 32)`
palabras (0 para anchura 0).

#### Frecuencias

Fuente: `athz_encode.slang` (`athzNormalize`), `athz_tables.slang`.

La tabla t son 256 frecuencias u16 `f[0..255]` en `frequencies + 512t`.
Suman exactamente `2^n`; una tabla que no lo hace se rechaza. Un símbolo con
frecuencia 0 nunca aparece. La tabla de ranuras del decodificador se deriva:
con `cum[s] = f[0] + … + f[s−1]`, la ranura `x` en `[0, 2^n)` corresponde al
único símbolo s con `cum[s] ≤ x < cum[s] + f[s]`. El escritor solo usa rANS
en planos con al menos dos símbolos distintos, así que todo `f[s] < 2^n`; el
motor empaqueta una ranura como `s | f << 8 | cum << 20` (12 bits cada uno
para f y cum).

Normalización que usa el escritor (informativa; un lector solo necesita las
frecuencias): sobre todos los elementos del tipo, se cuenta `c[s]` de cada
valor de byte residual; `f[s] = 0` si `c[s] = 0`, si no `max(1, floor(c[s] ×
2^n / total))`. Si la suma está por debajo de `2^n`, el déficit se suma al
símbolo más frecuente (el de menor valor entre cuentas iguales). Mientras la
suma pase de `2^n`, el símbolo con mayor `f` (el de menor valor en caso de
empate) pierde `min(f − 1, sum − 2^n)`. Un plano se codifica por rANS cuando
tiene al menos dos símbolos y `Σ c[s] (n − log2 f[s]) / 8 < 0.97 × total ×
width / 8`; si no, en crudo, con `base` el menor símbolo y `width` la
longitud en bits de `max − min` (0 cuando aparece un solo símbolo).

### 4.6 El elemento

Fuente: `athz_common.slang` (constantes de campo, `athzField`), `Athz.cpp`
(`fieldWords`, `layerOfByte`, `formatOfCloud`).

Un elemento son W palabras: los arrays de abajo en este orden, cada uno
ocupando su cuenta de palabras (0 donde no está).

| Campo | Palabras | Contenido | Capa |
|---|---|---|---|
| 0 positions | 4 | palabras de posición de `.athc` (§3, §2) | G |
| 1 shape | 4 | palabras de forma de `.athc` (§3, §2) | palabras 0, 1 y bytes 0–1 de la palabra 2: G; bytes 2–3 de la palabra 2 y palabra 3 (el color base): A |
| 2 sh | shWords | palabras de armónicos de `.athc` | A |
| 3 tail | 1 | grupo fundido: su código de celda en su nivel; splat: el índice de su grupo fundido más fino | G |
| 4 normal | normals | palabra de normal de `.athc` | G |
| 5 emission | emission | RGB9E5 | A |
| 6 pbr | pbrWords | palabra de material | G |
| 7 lobes | lobesWords | capas del material | G |
| 8 transfer | transferWords, o 4 con CPCA | pares f16, o el código CPCA (§4.12) | T |
| 9 shadow | shadowWords | bits de direcciones abiertas | T |
| 10 curvature | curvatureWords | 3 f32: operador de forma uu, uv, vv | G |
| 11 influences | influenceWords | 4 × (f32 índice de articulación, f32 peso) | K |
| 12 gradients | gradientWords | 3 palabras, cada una dos f16: baja d w_k/du, alta d w_k/dv, k = 0..2 | K |

El índice de la palabra de cola es `tailWord = 8 + shWords`. Los códigos de
celda de los grupos fundidos dentro de un nivel están en orden ascendente, y
el padre de una celda en el nivel de arriba tiene el código `cell >> 3`. Los
grupos fundidos llevan ceros en los campos 11 y 12 (nada les aplica
skinning).

`layerMask` es el OR de las capas de todos los bytes del elemento codificado:
G está siempre; A está en toda disposición de elemento (palabra de forma 3);
T donde existen los campos 8 o 9; K donde existen los campos 11 o 12.

El registro decodificado (lo que produce un lector) es el elemento codificado
con el campo 8 sustituido por el transfer decodificado de `transferWords`
palabras donde se usa CPCA.

### 4.7 Bandas y páginas

Fuente: `Athz.cpp` (`writeAthz`, `parse`), `athz_encode.slang` (`athzPageInfo`).

Una banda es un nivel fundido (bandas 0 a `levels − 1`, la más gruesa
primero) o los splats de la nube (banda `levels`). Los elementos de cada
banda se cortan en páginas de PG = `chunkSplats` = R × pageRuns elementos; la
última página de una banda lleva el resto. Los índices de página son globales
y van en orden de banda:

- la banda de nivel `l` contiene las páginas `levels[l].firstPage .. +
  levels[l].pages − 1`, y su página k contiene los grupos del nivel
  `[k·PG, min((k+1)·PG, groups))`;
- la banda de splats empieza en `leafFirstPage = levels[last].firstPage +
  levels[last].pages` y tiene `leafPages = ceil(count / PG)` páginas, y su
  página k contiene los splats `[k·PG, min((k+1)·PG, count))` en el orden del
  almacén (Morton).

El total es `leafFirstPage + leafPages`; el `pages` de la cabecera puede ser
menor (§4.14), nunca mayor.

Una página son `pageRuns` tramos de R elementos: el tramo r contiene los
elementos de la página `[r·R, r·R + here)`, `here = clamp(count − r·R, 0,
R)`. Toda página tiene sus `pageRuns` tramos en cada capa presente, incluidos
los tramos con `here = 0`.

#### Entrada de página (64 bytes)

| Offset | Tamaño | Tipo | Campo | Significado |
|---|---|---|---|---|
| 0 | 4 | u32 | info | bits 0–15 cuenta (elementos); 16–23 banda; 24–31 nivel del octree (0xFF para una página de splats) |
| 4 | 4 | u32 | parentPage | página fundida: la página del nivel de arriba que contiene el grupo padre de su primer elemento; página de splats: la página del nivel más fino que contiene el grupo de su primer splat; 0xFFFFFFFF para la banda 0 |
| 8 | 4 | u32 | box0 | f16 lo.x (mitad baja), f16 lo.y (mitad alta) |
| 12 | 4 | u32 | box1 | f16 lo.z, f16 hi.x |
| 16 | 4 | u32 | box2 | f16 hi.y, f16 hi.z |
| 20 | 4 | u32 | error | f16 eps (mitad baja), 0 (mitad alta) |
| 24 | 16 | u32[4] | offset | por capa G, A, T, K: el offset de los bytes de la página dentro de la sección de esa capa |
| 40 | 16 | u32[4] | bytes | por capa: los bytes de la página en esa sección; 0 para una capa ausente |
| 56 | 4 | u32 | crc | CRC-32 de los bytes de la página de todas las capas presentes, en orden G, A, T, K, concatenados |
| 60 | 4 | u32 | reserved | 0 |

- `count` debe ser igual a la cuenta de elementos de la banda para esa página
  y `band` a la banda en la que cae el índice de página.
- La caja: el mínimo y el máximo de las palabras de posición 0 a 2 de los
  elementos de la página (como f32), cada uno convertido a f16 y después
  movido un paso de f16 hacia fuera (hacia abajo para lo, hacia arriba para
  hi) donde la conversión fue hacia dentro, así que la caja contiene todas
  las posiciones.
- `eps`: `extent / 2^min(level, 30)` como f16 redondeado hacia arriba; 0 para
  las páginas de splats.
- Página padre de una página fundida en la banda l > 0: con `cell =
  tail(first element) >> 3`, `a` = el primer índice del nivel l−1 cuyo código
  de celda es ≥ `cell` (limitado a `groups − 1`); `parentPage =
  levels[l−1].firstPage + a / PG`. De una página de splats:
  `levels[last].firstPage + min(tail(first splat), finestGroups − 1) / PG`.
- La caja, `eps` y `parentPage` no los lee el motor.

Las páginas se guardan contiguas en orden de página en cada sección: los
bytes de la página i en la sección L empiezan donde terminan los de la página
i−1. Para una capa presente, el `bytes[L] ≥ 4 × pageRuns` de cada página y
`offset[L] + bytes[L] ≤ sectionBytes[L]`.

### 4.8 La capa de una página

Fuente: `athz_encode.slang` (`athzEncodeRuns`, `athzPlace`),
`athz_decode.slangh`.

Los bytes de la capa L de una página:

| Offset | Tamaño | Tipo | Campo | Significado |
|---|---|---|---|---|
| 0 | 4 × pageRuns | u32[pageRuns] | runEnd | runEnd[r]: el final del tramo r, en bytes desde el inicio de los bytes de la capa de la página |
| 4 × pageRuns | | | runs | el tramo 0 empieza en `4 × pageRuns`, el tramo r > 0 en `runEnd[r−1]` |

`runEnd[pageRuns − 1]` es igual al `bytes[L]` de la página.

Un tramo de la capa L, para el tipo k de la página (tipo 0 para las bandas de
nivel, 1 para la banda de splats). `hasRans(k, L)` es verdadero cuando alguna
entrada de plano del tipo k con capa L tiene codificación 1; es una propiedad
del asset, así que todo tramo de ese tipo y capa tiene la misma estructura.

| Presente cuando | Tamaño | Tipo | Campo | Significado |
|---|---|---|---|---|
| hasRans | 4 | u32 | words | N, palabras de 16 bits en el stream |
| hasRans | 128 | u32[32] | states | el estado final del codificador de cada carril, el estado inicial del decodificador |
| hasRans | 4 × ceil(N/2) | u16[N] + pad | stream | las palabras de 16 bits; la palabra j en el byte 2j; un u16 a cero rellena una N impar |
| siempre | 4 × rawWords(k, L) | u32[] | raw | los planos en crudo de la capa L |

`rawWords(k, L)` = Σ sobre los planos en crudo del tipo k en la capa L de
`ceil(R × width / 32)` — siempre calculado con la R completa, aunque el tramo
contenga menos elementos. El bloque en crudo de un plano está en la palabra
`rawStart` de la parte en crudo.

#### Carriles y orden de los símbolos

`steps = R / 32`. En cada plano, el elemento g del tramo (0 ≤ g < R)
pertenece al carril `g / steps` como su símbolo `k = g % steps`: cada carril
posee un bloque de `steps` elementos consecutivos. Los elementos con `g ≥
here` no llevan símbolo (ningún paso de rANS; bits en crudo 0).

#### Orden de codificación de los planos

Los planos se codifican y decodifican en este orden de i = 0..P−1:

    planeAt(i) = 4·tailWord + i                               if i < 4
                 4·w' + (i & 3), w' = (i−4)/4 + (1 if (i−4)/4 ≥ tailWord)   otherwise

es decir, primero los cuatro bytes de la palabra de cola, después todas las
demás palabras en orden de elemento, los bytes 0 a 3 de cada una. Los cuatro
planos de una palabra son siempre consecutivos.

#### Planos en crudo

Para el elemento g del tramo, con la `width` w > 0 del plano, los w bits en
la posición de bit `g·w` del bloque del plano (el bit j del bloque es el bit
`j % 32` de la palabra de bloque `j / 32`; un valor puede ocupar dos
palabras, la parte baja primero) son `v`; el byte es `(v + base) & 0xFF`. Con
`width = 0`, todos los bytes son `base`.

#### rANS

Fuente: `athz_common.slang` (comentario de cabecera), `athz_decode.slangh`,
`athz_encode.slang` (`athzEncodeRuns`).

Parámetros: el estado `x` es de 32 bits, en `[2^16, 2^32)` entre símbolos;
cota inferior `Lo = 2^16`; renormalización por palabras de 16 bits; precisión
de frecuencia `n = tableLog` (de 9 a 12). Cada tramo y capa tiene un stream
compartido por 32 carriles.

Decodificación de la capa L de un tramo (toda la aritmética sin signo de 32
bits):

    x[0..31] = states; ptr = 0
    for i in 0 .. P−1:
        b = planeAt(i); if layer(b) != L or encoding(b) != 1: continue
        f[], cum[] = table(b)
        for k in 0 .. steps−1:
            for lane in 0 .. 31:                       # todos los carriles, el mismo paso
                g = lane·steps + k
                need[lane] = false
                if g < here:
                    slot = x[lane] & (2^n − 1)
                    s = symbol of slot; byte(g, b) = s
                    x[lane] = f[s] · (x[lane] >> n) + slot − cum[s]
                    need[lane] = x[lane] < 2^16
            for lane in 0 .. 31 in increasing order:   # renormalizar
                if need[lane]:
                    x[lane] = (x[lane] << 16) | stream[ptr]; ptr += 1
    valid iff ptr == N and x[lane] == 2^16 for all 32 lanes

Un índice de palabra `≥ N` (leer más allá del stream), un estado final
distinto de 2^16 en cualquier carril, o `ptr ≠ N` hacen corrupto el tramo; el
motor cuenta esos carriles y hace fallar la lectura.

Codificación (informativa): cada carril empieza en `x = 2^16` y los planos y
pasos se recorren al revés; para un símbolo vivo s, si `x ≥ f[s] << (32 −
n)` el carril emite `x & 0xFFFF` y hace `x >>= 16`, y después `x = ((x /
f[s]) << n) + (x % f[s]) + cum[s]`. Las palabras emitidas en un paso se
colocan en orden creciente de carril, y los bloques de los pasos sucesivos
(al revés) se colocan desde el final del stream hacia su inicio, así que el
decodificador lo lee hacia delante.

Los streams rANS de capas distintas son independientes (sus propios estados y
palabras); un lector que no quiere una capa se salta sus bytes.

### 4.9 Palabras y transformaciones

Fuente: `athz_common.slang` (`athzForward`, `athzStep`, `athzCombine`,
`athzWithin`, `athzUnwithin`, `athzPreviousWord`), `athz_decode.slangh`.

Los cuatro bytes de la palabra w de un elemento (planos 4w .. 4w+3) forman el
residuo `r = b0 | b1 << 8 | b2 << 16 | b3 << 24`. La transformación t de la
palabra (entrada de plano del plano 4w+3, bits 0–7) dice cómo se recupera el
valor v. Las transformaciones son por palabra y por tipo, para todo el asset.

| t | Nombre | Residuo que codificó el escritor | Decodificación |
|---:|---|---|---|
| 0 | none | `v` | `v = r` |
| 1 | delta | `zigzag(v − v_prev)` | `v_g = Σ_{j ≤ g} unzigzag(r_j)` mod 2^32 a lo largo del tramo |
| 2 | delta16 | `zigzag16x2(sub16x2(v, v_prev))` | cada mitad: `Σ_{j ≤ g} unzigzag16(r_j half)` mod 2^16 |
| 3 | xor | `v XOR v_prev` | `v_g = r_0 XOR … XOR r_g` |
| 4 | parent | `zigzag(v − p)` | `v = p + unzigzag(r)` mod 2^32 |
| 5 | parent16 | `zigzag16x2(sub16x2(v, p))` | `v = add16x2(p, unzigzag16x2(r))` |
| 6 | xorParent | `v XOR p` | `v = p XOR r` |
| 7 | within | `within(v, q)` | `v = unwithin(r, q)` |
| 8 | parentWithin | `within(sub16x2(v, p), sub16x2(q, p_q))` | `v = add16x2(unwithin(r, sub16x2(q, p_q)), p)` |

- `v_prev` es la misma palabra del elemento anterior del mismo tramo; el
  primer elemento del tramo (g = 0) se predice desde 0. Los tramos son
  independientes.
- `p` es la palabra w del padre del elemento (§4.9), 0 cuando el elemento no
  tiene padre (banda 0). Para t = 8 sin padre, la palabra se decodifica como
  t = 7.
- `q` es la palabra anterior del propio elemento en orden de codificación,
  `pw = tailWord` para w = 0, `pw = tailWord − 1` para w = tailWord + 1, y
  en otro caso `pw = w − 1`; `p_q` es la palabra pw del padre.
- `within(v, q) = zigzag16((v.lo − q.hi) mod 2^16) | zigzag16((v.hi − v.lo)
  mod 2^16) << 16`, donde `.lo`/`.hi` son las mitades baja/alta.
  `unwithin(r, q)`: `lo = (unzigzag16(r.lo) + q.hi) mod 2^16`,
  `hi = (unzigzag16(r.hi) + lo) mod 2^16`, `v = lo | hi << 16`.
- La palabra de cola solo admite las transformaciones 0 a 3 (un fichero con
  una mayor se rechaza): se decodifica primero e identifica el padre de las
  palabras restantes.

El escritor solo da a una palabra una transformación con la que cada byte
depende de bytes de la capa G o de su propia capa (informativo; las
transformaciones 0, 3, 6 se permiten siempre; 1, 4 necesitan que todo byte de
la palabra pueda depender de los bytes por debajo de él; 2, 5 del byte bajo
de cada mitad; 7, 8 además de la mitad alta de la palabra anterior). Un
lector que deja fuera capas puede por tanto decodificar todos los bytes de
las capas que conserva.

#### Padres

Fuente: `athz_decode.slangh`, `athz_encode.slang` (`athzParents`).

Tras decodificar la palabra de cola de un elemento:

- Splat (banda `levels`): el padre es el grupo `min(tail, finestGroups − 1)`
  del nivel fundido más fino.
- Grupo fundido de la banda l > 0: con `cell = tail >> 3`, el padre es el
  grupo `min(a, groups(l−1) − 1)` del nivel l−1, donde `a` es el primer
  índice cuyo código de celda (palabra de cola) en el nivel l−1 es `≥ cell`
  (cota inferior sobre los códigos de celda ascendentes).
- Banda 0: sin padre.

Las palabras del padre son sus palabras de elemento codificado ya
decodificadas: para el campo 8 con CPCA, las palabras del código CPCA del
padre, no su transfer decodificado; para los campos 11 y 12, 0. Los niveles
fundidos deben decodificarse por tanto del más grueso al más fino, y los
splats después del nivel más fino.

### 4.10 Decodificar una página (resumen)

1. Comprobar la entrada de página (cuenta, banda, tamaños de capa) y su CRC
   sobre los bytes de las capas presentes.
2. Para cada capa L que decodificar, para cada tramo r: localizar el tramo
   (`runEnd`), leer `N`, los 32 estados y el stream donde `hasRans(k, L)`, y
   la parte en crudo.
3. Para i = 0..P−1 en orden de codificación: obtener el byte `b = planeAt(i)`
   de cada elemento del tramo del stream rANS o del bloque en crudo de su
   capa. Tras el cuarto plano de cada palabra, deshacer la transformación de
   la palabra para todos los elementos del tramo; tras la palabra de cola,
   determinar el padre de cada elemento.
4. El elemento g del tramo r de la página es el elemento `firstOfPage + r·R +
   g` de su banda.
5. Con CPCA, convertir el campo 8 (el código) en el transfer (§4.12).

Una capa no decodificada deja sus bytes a cero en el motor; G se decodifica
siempre.

### 4.11 Inicios del nivel más fino

En `startsOffset`, `finestGroups` u32: el primer splat (índice del almacén)
de cada grupo del nivel fundido más fino. Guardados en crudo.

### 4.12 La capa T como CPCA

Fuente: `athz_cpca.slang` (`athzCpca`, `athzCpcaPack`, `athzCpcaUnpack`),
`Athz.cpp` (`bindCpca`, `writeAthz`, `decodeAsset`).

Donde `codedTransferWords = 4` (versión 3+), el campo 8 de cada elemento (de
los dos tipos) es un código de 4 palabras en lugar del transfer, codificado a
través de los mismos planos y transformaciones (su predicción por el padre
usa el código del padre). Restricciones que comprueba el lector:
`codedTransferWords = 4`, `transferCount = 112`, `1 ≤ cpcaClusters ≤ 256`,
`cpcaComponents = 8`, la base dentro del fichero. El escritor solo acepta una
base de exactamente C = 256 clústeres y K = 8 (1 171 468 bytes); el lector no
comprueba `basisBytes` frente a C y K.

Disposición del código (4 palabras, 128 bits, el bit j es el bit `j % 32` de
la palabra `j / 32`):

| Bits | Contenido |
|---|---|
| 0–7 | id del clúster |
| 8 + 4i .. 11 + 4i, i = 0..23 | peso i = b·K + a (bloque b = 0..2, eje a = 0..7) como `q + levels`, de 0 a 2·levels |
| 104–127 | 0 |

Así, la palabra 0 guarda el id y los pesos 0–5, la palabra 1 los pesos 6–13,
la palabra 2 los pesos 14–21, y los bits 0–7 de la palabra 3 los pesos 22–23.

Base (array f32 en `basisOffset`; C clústeres, K ejes, bloques de
`D = (16, 48, 48)` coeficientes, 112 en total):

| Índice de float | Cuenta | Contenido |
|---|---|---|
| 0 | 112·C | centroide c en `112c`, en coeficientes normalizados (solo el escritor) |
| 112·C | 3 | norma por bloque (solo el escritor) |
| `meanAt[b]` | C·D_b | las medias del bloque b: clúster c, coeficiente d en `meanAt[b] + c·D_b + d` |
| `meanAt[b] + C·D_b` | C·D_b·K | ejes: `+ c·D_b·K + d·K + a` |
| `meanAt[b] + C·D_b + C·D_b·K` | C·K | escalas: `+ c·K + a` |

`meanAt[0] = 112C + 3`; `meanAt[b+1] = meanAt[b] + C·D_b + C·D_b·K + C·K`.

Decodificar el código de un elemento en el transfer (`transferWords`
palabras, aritmética f32):

    id = min(code & 0xFF, C − 1)
    for b in 0..2, with j running over 0..111 (block 0: j = d; 1: 16 + d; 2: 64 + d):
        for a in 0..7:
            nib = (code >> (8 + 4(bK + a))) & 0xF
            w[a] = (nib − levels) · scale[b][id][a] / levels
        for d in 0..D_b−1:
            v = mean[b][id][d]
            for a in 0..7: v += axes[b][id][d][a] · w[a]
            h[j] = f16(clamp(v, −65000, 65000))
    transfer word m = h[2m] | h[2m+1] << 16,   m = 0..55

El codificador (informativo): el id es el centroide más cercano en distancia
al cuadrado a `coefficient_j / norm[b]`; peso `w_a = Σ_d axes[d][a]
(coefficient_d − mean_d)`, `q = clamp(round(w_a / scale_a · levels),
−levels, levels)`.

### 4.13 La capa K

Fuente: `Athz.cpp` (`writeAthz`, `readAthzSkinned`), `athz_interleave.slang`
(`athzToHalves`, `athzFromHalves`), `SplatSkinner.h`.

Versión 4. Presente cuando `influenceWords = 8` (flag 4 de la cabecera, bit 8
de layerMask). Por elemento, los campos 11 y 12 (§4.6) se codifican en las
páginas como cualquier otro campo, en la capa K; los splats están en el orden
del almacén, así que el rig del elemento i es el del splat i. Los grupos
fundidos llevan ceros ahí.

El bloque de frames en `temporalOffset`, `temporalBytes = 64 + 8F + 24FJ`:

| Offset | Tamaño | Tipo | Campo | Significado |
|---|---|---|---|---|
| 0 | 64 | f32[16] | geomBind | la transformación de bind de la geometría, 16 floats como los guarda `SplatSkinning::geomBindTransform` (por filas, como el GfMatrix4d de USD) |
| 64 | 8F | f64[F] | times | el time code de cada frame |
| 64 + 8F | 24FJ | u32[F][J][6] | joints | primero por frame, después por articulación: 12 f16, medias `h[0..11]`, palabra k = `h[2k] | h[2k+1] << 16` |

Las 12 medias de una articulación son las filas 0 a 2 de la matriz 4×4 que
toma el skinner (la matriz de articulación de USD traspuesta: la traslación
en la última columna), el elemento `(row, col)` en la media `4·row + col`; la
fila 3 es `0 0 0 1` y no se guarda. Decodificación: cada media a f32, y la
fila 3 restaurada. Los time codes están en las unidades del stage; el fichero
no guarda time codes por segundo.

### 4.14 Prefijos

Fuente: `Athz.cpp` (`prefixPages`, `writeAthzPrefix`, `writeAthzPrefixBytes`,
`readAthz`, `readAthzCut`, `AthzSource`).

Un prefijo guarda las primeras `keep` páginas de un fichero (`keep` ≤ el
total, cualquier número): el `pages = keep` de la cabecera, cada sección
cortada a los bytes de sus primeras `keep` páginas (`sectionBytes[L]` = el
`offset[L]` original de la página `keep`), y la tabla de páginas cortada a
`keep` entradas. Todo lo demás es como en la fuente: las entradas de nivel,
`count`, `gaussians`, las entradas de plano, las tablas, los inicios, la
base, los frames, las extensiones (los bytes 64–4095 copiados tal cual) y
`assetId`. Las regiones se vuelven a disponer (§4.2) y la tabla del asset se
reescribe en la disposición de la versión 4 con la versión 4 en la cabecera,
sea cual sea la versión de la fuente.

Un lector distingue:

- `pages = leafFirstPage + leafPages`: el asset entero.
- `leafFirstPage ≤ pages < total`: todos los niveles fundidos están enteros;
  las páginas de splats de `pages − leafFirstPage` en adelante faltan. El
  motor lo lee como una nube cuyos chunks ausentes no están residentes (se
  dibujan como sus grupos fundidos).
- `pages < leafFirstPage`: un corte fijo. Solo es utilizable la banda más
  fina cuyas páginas están todas presentes, decodificada con los niveles de
  encima como padres; el motor lo rechaza como nube para streaming, y lo
  rechaza del todo si ni siquiera el nivel 0 está entero.

Un fichero con skinning se lee con la capa K solo cuando está entero.

### 4.15 Versiones

Fuente: `Athz.cpp` (`kVersion`, `decodeAsset`).

| Versión | Tabla del asset | Palabra 3 de la entrada de plano | Añade |
|---|---|---|---|
| 1 | offsets 0–71, después desde `boundsLo` en adelante (en 72) | estimación del codificador, leída como transformación 0 | — |
| 2 | como la versión 1 | transformación | transformaciones de palabra |
| 3 | offsets 0–107, después desde `boundsLo` en adelante (en 108) | transformación | curvatura, CPCA |
| 4 | como en la tabla de arriba (`boundsLo` en 140) | transformación | capa K |

Los campos ausentes en una versión se leen como 0. En las tablas de las
versiones 1 y 2 todo campo desde `boundsLo` en adelante está 68 bytes antes
que en la versión 4; en la versión 3, 32 bytes antes.

### 4.16 Qué debe rechazar un lector

Fuente: `Athz.cpp` (`decodeHeader`, `decodeAsset`, `parse`, `checkPage`),
`athz_decode.slangh`.

- Un fichero de menos de 4096 bytes; sin el magic `ATHZ`; una versión fuera
  de 1 a 4; `headerBytes ≠ 64`; un CRC de cabecera que no coincide.
- Un bit de flag fuera de 0x3F; un `layerMask` sin el bit 1 o con bits fuera
  de 0xF; R = 0, no múltiplo de 32 o > 256; `pageRuns = 0`;
  `entryBytes ≠ 64`.
- Una tabla del asset más allá del final del fichero o más corta que sus
  cuentas; `bytes ≠ assetBytes`; `kinds ≠ 2`; `levels` 0 o > 30;
  `count = 0`; `shWords` 0 o > 64; `planes ≠ 4W`; `tableLog` fuera de 9 a 12;
  `chunkSplats ≠ R × pageRuns`; `gaussians ≠ count`.
- CPCA: `codedTransferWords ≠ 0` con `codedTransferWords ≠ 4`,
  `transferCount ≠ 112`, `cpcaClusters` 0 o > 256, `cpcaComponents ≠ 8`, o
  una base fuera del fichero.
- Una tabla rANS cuyas frecuencias no suman `2^tableLog`.
- `curvatureWords` que no es 0 ni 3.
- Capa K: `influenceWords` que no es 0 ni 8; `gradientWords` que no es 0 ni
  3; `influenceWords ≠ 0` y `frames ≠ 0` en desacuerdo; un bloque de frames
  fuera del fichero; `temporalBytes ≠ 64 + 8F + 24FJ` donde F ≠ 0.
- Una entrada de plano con transformación > 8, o > 3 en los planos de la
  palabra de cola; codificación > 1; anchura > 8; capa > 3; un índice de
  tabla rANS ≥ `tables`; una capa que no está en `layerMask`.
- Una entrada de nivel que incumple las comprobaciones de §4.5;
  `startsBytes ≠ 4 × finestGroups`; inicios fuera del fichero; una sección
  fuera del fichero.
- `pages` mayor que el total del asset; una tabla de páginas fuera del
  fichero.
- Una página cuya banda o cuenta no es la que implica su índice; bytes
  distintos de cero en una capa ausente; para una capa presente,
  `offset + bytes > sectionBytes` o `bytes < 4 × pageRuns`.
- Un registro de extensión que pasa de `assetOffset`.
- Una página cuyo CRC no coincide, antes de decodificar nada de ella.
- Un tramo que se decodifica como corrupto (§4.8, rANS).

Por lo demás, el motor no comprueba los límites de los valores de `runEnd` ni
de `rawStart` frente a los bytes de una página; un lector independiente
debería hacerlo.

### 4.17 Lo que el código no determina

- Las conversiones f32 → f16 (las cajas de página antes de su paso hacia
  fuera, los valores del transfer CPCA, las medias de articulación de la capa
  K) usan el `f32tof16` del backend de GPU; el código no fija el modo de
  redondeo (round-to-nearest-even es el comportamiento habitual del
  hardware). La reconstrucción CPCA suma en f32 en el orden dado; no se
  controla si el compilador fusiona multiplicaciones y sumas, así que no se
  garantiza que una reimplementación en CPU coincida bit a bit con el
  transfer decodificado del motor.
- Ficheros de las versiones 1 y 2 escritos por codificadores anteriores: el
  lector actual decodifica todas las versiones con la asignación de carriles
  (carril `g / steps`) y el orden de codificación de planos (la palabra de
  cola primero) de arriba. `docs/decisions.md` describe la asignación de
  carriles del primer codificador como `g % 32`; si existen ficheros de las
  versiones 1 a 2 escritos antes del cambio con esa disposición, y cómo se
  leerían, no se puede determinar con el código actual.
- La disposición en bits de un código de celda (el entrelazado Morton de las
  coordenadas de la celda) la define el constructor de LOD, no este formato;
  un lector solo necesita que los códigos sean ascendentes dentro de un nivel
  y que el código de un padre sea `cell >> 3`.
- No se especifica el significado como texto (la codificación) de los bytes
  de la extensión de fuente.

## 5. `.athv`: un vídeo de gaussianas

Lo escribe `athenea video-encode`; lo lee `AthvReader`
(`modules/video/include/athenea/video/Athv.h`, `modules/video/src/Athv.cpp`).
Los splats de un frame se ponen en los planos de luma (y opcionalmente de
croma) de una imagen que toma un codificador HEVC hardware; lo que el códec no
puede llevar exacto viaja junto a ella en el mismo registro. El fichero es una
cabecera y después un registro por imagen, cada uno completo en sí mismo, así
que un emisor puede cortarlo en paquetes y un receptor volver a juntar los
registros sin el fichero.

### 5.1 Cabecera: 128 bytes (versión 2)

| Offset | Tamaño | Tipo | Campo | Significado |
|---|---|---|---|---|
| 0 | 4 | u32 | magic | `ATHV` (0x56485441) |
| 4 | 4 | u32 | version | 2 (la 1 se lee, §5.4) |
| 8 | 4 | u32 | codec | FourCC, `hvc1` (0x31637668) |
| 12 | 4 | u32 | bits | bits por muestra: 8 o 10 |
| 16 | 4 | u32 | restPerColour | 0, 3, 8, 15 |
| 20 | 4 | u32 | chunkSplats | splats por chunk: un múltiplo de 32768 |
| 24 | 4 | u32 | columns | teselas por fila del mosaico |
| 28 | 4 | u32 | maxSplats | el máximo de splats que lleva una imagen: para lo que se dimensionan sus teselas |
| 32 | 4 | u32 | flags | abajo |
| 36 | 4 | u32 | rendition | el id de este fichero entre las rendiciones de los mismos frames |
| 40 | 8 | f64 | rate | frames por segundo |
| 48 | 4 | u32 | rateNumerator | el ritmo como fracción |
| 52 | 4 | u32 | rateDenominator | |
| 56 | 4 | u32 | sourceFrames | frames de origen en el fichero: la longitud del bucle |
| 60 | 4 | f32 | quality | la calidad constante del codificador; 0: por tasa de bits |
| 64 | 4 | u32 | positionBlock | 0: posiciones sobre el rango del frame; si no, splats por caja (una potencia de dos, de 64 a chunkSplats) |
| 68 | 4 | u32 | positionHighBits | los bits de la posición más allá de `bits`, de 0 a `bits` |
| 72 | 4 | f32 | positionTarget | > 0: precisión adaptativa, metros |
| 76 | 4 | u32 | geometryInterval | capa de geometría: frames de clave a clave; 0 cada frame una clave |
| 80 | 4 | u32 | geometryBits | bits 0–7 bits de posición (10–16), 8–15 bits de escala logarítmica (6–10), 16–23 el orden de predicción (1, 2, 3), 24–31 bits de rotación (5–10; 0 se lee como 10); 0: los valores por defecto 16, 10, 1, 10 |
| 84 | 4 | u32 | geometryFold | con el bit 7 de los flags: los pliegues, bits 0–7 posición, 8–15 escala logarítmica, 16–23 rotación |
| 88 | 4 | f32 | volumeDensityScale | con el bit 8 de los flags |
| 92 | 36 | — | reserved | cero |

**Flags.**

| Bit | Valor | Significado |
|---|---|---|
| 0 | 1 | las partes altas de la posición son planos de la imagen, no datos laterales |
| 1 | 2 | la fuente se ajustó para dibujarse con antialiasing (la compensación de Mip-Splatting) |
| 2 | 4 | colores en el croma: YCoCg, Co y Cg en el segundo plano de la imagen |
| 3 | 8 | orden estable: el índice i es la misma gaussiana en todos los frames |
| 4 | 16 | la capa de geometría: posición, escala y rotación exactas junto al vídeo |
| 5 | 32 | la capa de geometría refresca bloques de 256 por turnos entre claves |
| 7 | 128 | geometría en el vídeo: posición, escala y rotación como planos de luma |
| 8 | 256 | un medio: los splats son gaussianas de extinción (§5.9) |

El escritor escribe la cabecera, los registros, y otra vez la cabecera al
final con `maxSplats` y `sourceFrames`.

### 5.2 Registro

Los registros siguen a la cabecera uno tras otro; cada longitud es un
múltiplo de 4. Los primeros 28 bytes son los mismos en todos los mensajes que
lleva un stream (§5.11), así que un relé enruta con ellos sin leer más.

| Offset | Tamaño | Tipo | Campo | Significado |
|---|---|---|---|---|
| 0 | 4 | u32 | magic | `ATVF` (0x46565441) |
| 4 | 2 | u16 | type | 0: una imagen |
| 6 | 2 | u16 | rendition | la de la cabecera |
| 8 | 4 | u32 | frame | el frame de presentación: en un fichero, el frame de origen |
| 12 | 8 | u64 | ptpTime | ns; 0 en un fichero, el momento de envío en un stream |
| 20 | 4 | u32 | bytes | el registro entero |
| 24 | 2 | u16 | version | 2 |
| 26 | 2 | u16 | flags | bit 0 una imagen clave (un IRAP de HEVC); bit 1 una clave de geometría (todas las gaussianas absolutas) |
| 28 | 4 | u32 | index | el índice de la imagen en el fichero |
| 32 | 4 | u32 | source | el frame de origen del que es una porción |
| 36 | 4 | u32 | slice | cuál de las imágenes de ese frame |
| 40 | 4 | u32 | slices | las imágenes del frame |
| 44 | 4 | u32 | firstSplat | dónde empiezan sus splats en el frame (rango Morton) |
| 48 | 4 | u32 | splats | los splats de la imagen |
| 52 | 4 | u32 | sourceSplats | los splats del frame |
| 56 | 8 | f64 | time | el time code de la fuente, segundos |
| 64 | 16 | char[16] | timecode | `HH:MM:SS:FF`, terminado en NUL; una etiqueta |
| 80 | 4 | u32 | rangeCount | floats en los rangos (dos por atributo; como mucho 4096) |
| 84 | 4 | u32 | highBytes | los bytes de las partes altas, descomprimidos (0: ninguno) |
| 88 | 4 | u32 | highStored | sus bytes tal como se guardan |
| 92 | 4 | u32 | parameterSets | 0, o 3 en una imagen clave (VPS, SPS, PPS; como mucho 16) |
| 96 | 4 | u32 | unitBytes | los bytes de la unidad de acceso |
| 100 | 4 | u32 | geometryStored | los bytes de la capa de geometría tal como se guardan (0: ninguno) |
| 104 | | f32 × rangeCount | ranges | |
| | | bytes | high | `highStored` bytes, un stream zlib (RFC 1950); rellenado a 4 |
| | | bytes | geometry | `geometryStored` bytes, un stream zlib; rellenado a 4 |
| | | | parameter sets | cada uno una longitud u32 y sus bytes, rellenado a 4 |
| | | bytes | unit | la unidad de acceso: unidades NAL, cada una tras una longitud big endian de 4 bytes; rellenado a 4 |

Las partes altas se descomprimen en exactamente `highBytes`; la capa de
geometría en `⌈splats / chunkSplats⌉ · chunkBytes`, donde `chunkBytes` es
`16 · chunkSplats`, más `chunkSplats / 256 · 32` con orden de predicción 3.
Un registro cuyas longitudes pasan de su final, o cuyos datos no se
descomprimen en esos tamaños, se rechaza. Una imagen nunca nombra un frame de
origen más allá del siguiente al último visto: un lector agrupa las porciones
por `source` a medida que recorre el fichero.

Un decodificador toma la unidad de acceso como HEVC con prefijos de longitud
al estilo de `hvcC`; un stream Annex B son los conjuntos de parámetros y las
unidades NAL, cada uno tras un código de inicio `00 00 00 01` (`annexB` en
`Athv.h`).

### 5.3 Imágenes

Una imagen son los splats de un frame, o una porción de ellos, dispuestos
como una sola imagen de luma (el *mosaico*). Con `chroma`, tiene además un
segundo plano NV12. Cada atributo de cada splat es una muestra en una tesela
del mosaico. Esta sección cubre qué significan las muestras. Los campos de
cabecera y de registro que se nombran abajo (`bits`, `restPerColour`,
`chunkSplats`, `columns`, `maxSplats`, los flags, `positionBlock`,
`positionHighBits`, `positionTarget`, los campos de geometría,
`geometryFold`, `volumeDensityScale`, y los `splats`, `firstSplat`, `source`
y `ranges` del registro) son los del contenedor.

Todos los valores de varios bytes son little endian. Salvo que se diga otra
cosa, la aritmética es sin signo de 32 bits y `>>` sobre un valor con signo
es aritmético.

#### Orden de los splats

Fuente: `shaders/athenea/scene/video_planes_encode.slang` (`videoPlanesMorton`), `shaders/athenea/common/morton.slang`.

Dentro de un frame, los splats se numeran por su *rango Morton* `r`. El
codificador calcula una clave de 30 bits para cada splat sobre los rangos de
posición (los atributos 0–2 de los rangos en uso):

```
u      = saturate((p - lo) / max(hi - lo, 1e-20))       por eje
q      = min(uint(u * 1024), 1023)                       por eje
key    = spread(q.x) << 2 | spread(q.y) << 1 | spread(q.z)
```

Después ordena los splats por radix sobre los 30 bits de la clave. Una imagen
contiene los rangos Morton `firstSplat .. firstSplat + splats - 1` de su
frame. Dentro de la imagen, el splat `i = r - firstSplat` es el rango local
de la imagen, y el decodificador lo escribe en la gaussiana `firstSplat + i`
del frame. El decodificador no necesita la clave.

El flag `stableOrder` (`--order stable`) significa que una sola ordenación,
tomada en el primer frame, se reutiliza en todos los frames. El índice `r` es
entonces la misma gaussiana en todo el fichero. Los frames predichos de la
capa de geometría dependen de esto.

#### Constantes de la disposición

Fuente: `modules/video/include/athenea/video/Planes.h` (`PlaneLayout`), `modules/video/src/Planes.cpp` (`PlaneLayout::make`).

| Constante | Valor |
|---|---|
| anchura de tesela | 512 muestras |
| bloque | 64 × 64 muestras = 4096 splats |
| bloques por fila de tesela | 8 |
| banda (el chunk más pequeño) | 8 bloques = 32768 splats = 64 filas de una tesela |
| `chunkSplats` | un múltiplo de 32768; `video-encode` escribe siempre 32768 |
| imagen más grande | anchura ≤ 16384, altura ≤ 16384, anchura × altura ≤ 8192² |

#### Tamaño del mosaico

Fuente: `modules/video/src/Planes.cpp` (`PlaneLayout::make`).

`planes` es el número de planos de luma, dado en «Planos por modo». Un
decodificador construye la disposición de cada imagen a partir de los
`splats` del registro y de los `maxSplats`, `chunkSplats` y `columns` de la
cabecera:

```
chunks      = ceil(splats / chunkSplats)
tileChunks  = max(chunks, ceil(maxSplats / chunkSplats))
tileHeight  = tileChunks * (chunkSplats / 512)          filas; un múltiplo de 64
columns     = header.columns
width       = columns * 512
height      = ceil(planes / columns) * tileHeight
```

Como las teselas se dimensionan para `maxSplats`, todas las imágenes de un
stream tienen las mismas dimensiones. No se aplica ningún otro redondeo. La
altura es siempre un múltiplo de 64, así que es par, como exige NV12.

El codificador elige `columns`. Para cada `c` en `1 .. planes`, calcula
`w = 512 c` y `h = ceil(planes / c) × tileHeight`. Descarta toda `c` con `w`
o `h` por encima del límite de lado (16384, o el menor `--max-picture`) o con
`w × h` por encima de 8192². De las restantes, se queda con la `c` de menor
`max(w, h)`, y en caso de empate la de menor `w × h`. `--columns` fuerza un
valor.

#### De splat a muestra

Fuente: `shaders/athenea/scene/video_planes_common.slang` (`pixelOf`, `planeAndSplatOf`).

El plano `p` es la tesela en la columna `p mod columns` y la fila
`p / columns`, así que las teselas llenan el mosaico por filas. La muestra
del splat `i` en el plano `p` está en:

```
block = i >> 12                  j = i & 4095
x = (p mod columns) * 512 + (block mod 8) * 64 + compact2(j)
y = (p / columns) * tileHeight + (block / 8) * 64 + compact2(j >> 1)
compact2(v): los bits 0, 2, 4, 6, 8, 10 de v empaquetados en los bits 0..5
```

Dentro de un bloque de 64 × 64, los splats siguen el orden Z, con x tomando
los bits pares de `j` e y los impares. Los bloques van de ocho en ocho por
fila de tesela, así que el chunk `c` de una imagen (los splats
`c·chunkSplats ..`) son las filas `c·chunkSplats/512 ..` de cada tesela. Los
splats `4m .. 4m+3` ocupan el quad de 2 × 2 cuya muestra superior izquierda
está en (x, y) pares.

Valores de relleno:

- Una muestra de un plano real cuyo índice `i ≥ splats` repite el splat
  `splats − 1`.
- Una tesela cuyo índice es `planes` o más (las teselas sobrantes de la
  última fila) es gris medio: `1 << (bits − 1)`.

#### Almacenamiento de las muestras

Fuente: `modules/video/src/Planes.cpp` (`setLayout`), `video_planes_encode.slang` (`videoPlanesEncode`), `video_planes_decode.slang` (`sampleAt`).

| `bits` | Almacenamiento de luma | Valor de muestra `s` |
|---|---|---|
| 8 | un byte por muestra (HEVC Main) | `s` = el byte |
| 10 | una palabra de 16 bits por muestra, P010: el valor en los 10 bits altos | `s` = word >> 6 |

Con `chroma` (solo 8 bits), el segundo plano es el de NV12: `width/2 ×
height/2` pares de bytes `(Cb, Cr)`, con las filas a `width` bytes de
distancia. El par `(cx, cy)` cubre las muestras de luma `(2cx .. 2cx+1, 2cy
.. 2cy+1)`, que son cuatro splats consecutivos `4m .. 4m+3` de una tesela. Los
pares bajo teselas que no son un plano Y de color, y los pares cuyo primer
splat `4m ≥ splats`, son `(128, 128)`.

### 5.4 Atributos

Fuente: `shaders/athenea/scene/video_planes_common.slang` (`plainAttributeOf`, `attributeOf`), `shaders/athenea/common/packing.slang`.

Un splat tiene `attributes = 14 + 3 × restPerColour` atributos escalares
(`kAttributesBeforeRest = 14`), donde `restPerColour` es uno de 0, 3, 8 o 15.
Los valores son los del almacén empaquetado del motor, leídos como sigue.

| Índice | Atributo | Valor tal como se envía |
|---|---|---|
| 0, 1, 2 | posición x, y, z | float32, en el espacio de la nube |
| 3 | opacidad | lineal, tal como se guarda (un logit de 3DGS ya ha pasado por la sigmoide); normalmente 0..1 |
| 4, 5, 6 | escala x, y, z | el logaritmo natural de la desviación típica (el `ln scale` f16 del almacén), limitado a [−20, 20] |
| 7, 8, 9, 10 | rotación x, y, z, w | el cuaternio del almacén decodificado (`decodeQuaternion`) y negado cuando `w < 0`, así que `w ≥ 0`; no se renormaliza |
| 11, 12, 13 | color base r, g, b | el color base f16 del almacén (véase abajo) |
| 14 + k | coeficiente de resto k | `k = basis × 3 + colour` (color 0 r, 1 g, 2 b; base 0 .. restPerColour−1 en el orden de 3DGS, grado 1 primero) |

Qué significan los valores guardados:

- **Color base.** Para una fuente SH es `0.5 + C0 · f_dc`, con
  `C0 = 0.28209479177387814`. Para otras fuentes es el color de la fuente tal
  como lo decodificó el cargador (`splat_encoding.slang`, `decodeColour`). El
  formato no declara una función de transferencia. El color base no se limita
  y puede salirse de 0..1.
- **Coeficientes de resto.** Están en el orden rgb por base del almacén, no en
  el orden por color de un PLY de 3DGS. Los coeficientes más allá de la
  cuenta propia de la fuente son 0.
- **Palabra de rotación.** La rotación del almacén es de las tres menores: el
  índice de la componente omitida (0 x, 1 y, 2 z, 3 w) en los bits 0–1. Las
  otras tres, en orden x, y, z, w sin la omitida, están en los bits 22, 12 y
  2, diez bits cada una, como `round((c / √½ · 0.5 + 0.5) · 1023)`.

Con `chroma`, los atributos de color se envían como YCoCg en vez de RGB. La
cantidad de color `q` es el color base cuando `q = 0` y la base armónica
`q − 1` en otro caso. Sus tres atributos `11 + 3q`, `12 + 3q` y `13 + 3q`
guardan Y, Co y Cg del (r, g, b) de esa cantidad:

```
Co = R − B,   t = B + Co/2,   Cg = G − t,   Y = t + Cg/2          (float32, sin redondeo)
inversa:  t = Y − Cg/2,  G = Cg + t,  B = t − Co/2,  R = B + Co
```

#### Planos por modo

Fuente: `video_planes_common.slang` (`lowPlane`, `attributePlane`, `planeAttribute`), `Planes.h` (`PlaneLayout::planes`).

Sea `C = 3 + 3 × restPerColour` sin `chroma`, o `C = 1 + restPerColour` con
él, el número de planos de color.

**Simple** (sin capa de geometría, sin geometría en el vídeo):

| Planos | Contenido |
|---|---|
| 0–2 | partes altas de la posición x, y, z (solo con `hiInVideo`) |
| 0–2, o 3–5 con `hiInVideo` | partes bajas de la posición x, y, z |
| las 8 siguientes | atributos 3–10: opacidad, escala logarítmica x y z, rotación x y z w |
| sin `chroma`: las `3 + 3·restPerColour` siguientes | atributos 11, 12, … en orden de índice |
| con `chroma`: las `1 + restPerColour` siguientes | la Y de la cantidad q (atributo `11 + 3q`); su Co y su Cg están en el croma bajo esa tesela |

`planes = (hiInVideo ? 6 : 3) + 8 + C`. Sin `chroma` esto es igual a
`(hiInVideo ? 6 : 3) + attributes − 3`.

**Capa de geometría** (bit 4 de los flags): los planos son la opacidad (plano
0), y después los planos de color como arriba. `planes = 1 + C`. La
posición, la escala y la rotación viajan en la capa de geometría.

**Geometría en el vídeo** (bit 7 de los flags): véase «Geometría en el
vídeo». `planes = 9 + 2F + 1 + C`, donde `F` es el número de atributos de
geometría plegados.

### 5.5 Cuantización

Fuente: `video_planes_common.slang` (`quantise`, `dequantise`, `splitPosition`, `joinPosition`), `video_planes_decode.slang`.

Todo atributo se cuantiza sobre un rango `[lo, hi]` con `L` niveles:

```
quantise(v, lo, hi, L)   = hi > lo ? min(round(saturate((v − lo) / (hi − lo)) · L), L) : 0
dequantise(q, lo, hi, L) = hi > lo ? lo + (hi − lo) · (min(q, L) / L) : lo         (float32)
```

`round` es el del shader: al más cercano, con los empates resueltos como los
resuelva el `round` del destino de GPU. El código no lo fija.

#### Rangos

Fuente: `video_planes_encode.slang` (`videoPlanesRangePartial`, `videoPlanesRangeFinish`, `videoPlanesRangeWiden`), `apps/athenea/src/CmdVideo.cpp`.

Los `ranges` de un registro guardan dos float32 por atributo:
`ranges[2a] = lo` (el mínimo) y `ranges[2a+1] = hi` (el máximo) del atributo
`a`. Hay `2 × attributes` floats.

- **Base.** Los valores son los de la tabla de atributos, tras el límite de
  la escala logarítmica, el `w ≥ 0` de la rotación, y YCoCg con `chroma`. Con
  geometría en el vídeo, los atributos 7–10 son las componentes de la
  rotación canónica.
- **Sin splats.** Un frame sin splats da `(0, 0)`.
- **Grupos de rangos.** Por defecto (`--ranges group`) los rangos son la
  unión sobre un grupo de rangos: `--range-group` frames, o el intervalo
  entre claves cuando no está puesto. Todos los registros del grupo repiten
  los mismos rangos. `--ranges frame` usa los de cada frame.
- **Capa de geometría.** Una capa de geometría enviada como diferencias
  (`geometryInterval` ≠ 1) exige un solo grupo de rangos para toda la
  secuencia.

Todo atributo que es un plano se decodifica como
`dequantise(s, lo_a, hi_a, L)`, con `L = 2^bits − 1` (255 o 1023). Las
posiciones son la única excepción.

#### Posiciones en planos

Fuente: `video_planes_encode.slang` (`positionOf`, `sampleOf`, `videoPlanesHigh`), `video_planes_decode.slang` (`videoPlanesDecode`, `highAt`, `positionRangeAt`, `highBitsAt`).

Esta sección se aplica en el modo simple. Cada eje de una posición se
cuantiza a `bits + H` bits:

```
Lp   = 2^(bits + H) − 1
Q    = quantise(p_axis, lo, hi, Lp)
high = Q >> bits
low  = Q & (2^bits − 1)
lowSample = (high odd) ? (2^bits − 1) − low : low        la parte baja se pliega (en espejo) en las partes altas impares
```

El decodificador reconstruye el eje:

```
Q = (high << bits) | ((high odd) ? (2^bits − 1) − min(low, 2^bits − 1) : min(low, 2^bits − 1))
p_axis = dequantise(Q, lo, hi, Lp)
```

De dónde viene cada entrada:

| Modo | `H` | `[lo, hi]` | parte alta |
|---|---|---|---|
| `hiInVideo` | `bits` | el rango del eje en el registro | planos 0–2 (muestras) |
| lateral, `positionBlock = 0` | `positionHighBits` (0 .. `bits`) | el rango del eje en el registro | carga lateral |
| lateral, `positionBlock > 0` | `positionHighBits` | la caja del bloque (abajo) | carga lateral |
| lateral, `positionBlock > 0`, `positionTarget > 0` | el byte propio del bloque (adaptativo) | la caja del bloque | carga lateral |

Con `positionHighBits = 0`, no se guardan partes altas, `high = 0`, y el
plano bajo guarda `Q` sin plegar. `hiInVideo` exige `positionBlock = 0` y
`positionHighBits = bits`.

**Bloques de posición.** Cuando `positionBlock > 0`, su valor es una potencia
de dos de 64 a 32768. El bloque `b` de un chunk cubre los splats de ese chunk
`b·positionBlock .. (b+1)·positionBlock − 1`, un tramo del orden Morton. Su
caja, por eje, es el menor `lo` f16 y `hi` f16 que encierran las posiciones
del tramo:

- `lo = halfBelow(min)`: el f16 más cercano al mínimo, bajado un ulp si queda
  por encima.
- `hi = halfAbove(max)`: del mismo modo, subido si queda por debajo.

Las posiciones del bloque se cuantizan sobre la caja, leída como float32. Un
bloque sin splats tiene la caja `(0, 0)`. Con `--hold-boxes`, la caja es la
unión sobre los frames del grupo de rangos en vez de la caja propia del
frame. Se guarda del mismo modo.

**Precisión adaptativa.** Cuando `positionTarget > 0`, el bloque `b` lleva su
propia cuenta de bits altos `H_b`, calculada así:

```
widest = máximo sobre los tres ejes de (f16 hi − f16 lo)
H_b    = widest > positionTarget ? min(ceil(log2(widest / positionTarget)), positionHighBits) : 0
```

El almacenamiento de las partes altas no cambia con `H_b`. Cada entrada
conserva su tamaño, y los bits sin usar son cero.

#### Muestras de croma

Fuente: `video_planes_encode.slang` (`videoPlanesChroma`, `chromaWeightOf`), `video_planes_decode.slang` (`colourAt`).

Para la cantidad de color `q`, el plano Y es por splat. El par de croma de un
quad guarda la media ponderada de los Co y Cg de los cuatro splats. Los
splats `≥ splats` quedan fuera de la media.

```
w_s = 1                                                   (chromaWeight 0, el valor por defecto)
w_s = max(opacity, 1e-4) · exp(ls_x + ls_y + ls_z − min(ls))   (opción del codificador --chroma-weight area)
Cb  = quantise(Σ w_s Co_s / Σ w_s, lo_{12+3q}, hi_{12+3q}, 255)
Cr  = quantise(Σ w_s Cg_s / Σ w_s, lo_{13+3q}, hi_{13+3q}, 255)
```

El peso del croma es una elección del codificador y no queda registrado en el
fichero. El decodificador toma, para cada splat, la Y de su propia muestra, y
Co y Cg del par en `(x/2, y/2)` de esa muestra. Descuantiza los tres sobre
sus rangos y los convierte a RGB con la YCoCg inversa.

#### Splat decodificado

Fuente: `video_planes_decode.slang` (`videoPlanesDecode`, `writeRest`).

El decodificador escribe estos valores en todos los modos:

| Campo | Valor |
|---|---|
| opacidad | `clamp(dequantised attribute 3, 0, 5)` |
| escala logarítmica | tal como se decodifica, guardada como f16, limitada a ±65000 |
| rotación (modo simple) | `q = (a7, a8, a9, a10) / |q|` (`(0,0,0,1)` cuando `|q| ≤ 1e-8`), recodificada en las tres menores |
| colores | tal como se decodifican, f16 |
| coeficientes de resto más allá de `restPerColour` | 0 |

### 5.6 Datos laterales

Cada registro lleva:

- `ranges`;
- la carga *high* comprimida con deflate (campos del contenedor `highRaw` /
  `highStored`);
- la carga *geometry* comprimida con deflate (`geometryStored`).

Las dos cargas son streams zlib, escritos con `compress2` en el nivel 9.

#### Carga high

Fuente: `Planes.h` (`blockBytes`, `bitsBytes`, `highChunkBytes`), `video_planes_encode.slang` (`videoPlanesBlocks`, `videoPlanesBlockBits`, `videoPlanesHigh`), `video_planes_decode.slang` (`highAt`).

La carga descomprimida son `ceil(splats / chunkSplats)` chunks de
`highChunkBytes` cada uno. Los chunks son los de la propia imagen, y el chunk
`c` cubre los splats de la imagen `c·chunkSplats ..`. Sea
`nb = chunkSplats / positionBlock` y `sb = 1` a 8 bits, `2` a 10 bits. La
carga está vacía con `hiInVideo` y sin `positionBlock`, con la capa de
geometría, y con geometría en el vídeo.

| Offset en el chunk | Bytes | Presente cuando | Contenido |
|---|---|---|---|
| 0 | `12 · nb` | `positionBlock > 0` | las cajas: para el bloque `b` y el eje `a`, el u32 en la palabra `3b + a` es `f16 lo \| f16 hi << 16` |
| `12 · nb` | `ceil(nb / 4) · 4` | `positionBlock > 0` y `positionTarget > 0` | `H_b`: un byte por bloque, el bloque `b` en el byte `b`; padding a cero |
| tras lo anterior | `3 · chunkSplats · sb` | no `hiInVideo` y `positionHighBits > 0` | las partes altas, por eje: la entrada `e = axis · chunkSplats + (i mod chunkSplats)` es un u8 (8 bits) o un u16 (10 bits) en el byte `e · sb` |

```
highChunkBytes = (positionBlock ? 12·nb : 0)
               + (positionTarget > 0 && positionBlock ? ceil(nb/4)·4 : 0)
               + (!hiInVideo && positionHighBits ? 3·chunkSplats·sb : 0)
```

Las entradas de los splats `≥ splats` son 0.

A 10 bits una parte alta ocupa dos bytes. Los comentarios de `Planes.h` y de
`video_planes_common.slang` dicen "a byte an axis", pero el código guarda
dos bytes.

### 5.7 Capa de geometría

Fuente: `shaders/athenea/scene/video_geometry.slang`, `video_planes_common.slang` ("THE GEOMETRY LAYER"), `Planes.h` (`geometryChunkBytes`), `modules/video/src/Athv.cpp` (`AthvReader::read`).

Con el bit 4 de los flags, la posición, la escala y la rotación no viajan en
planos. Viajan exactas, en la carga de geometría del registro. Cada gaussiana
`j` (su índice en el frame, `firstSplat + i`) tiene un estado de cuatro
palabras u32:

| Palabra | Contenido |
|---|---|
| 0 | `px \| py << 16` |
| 1 | `pz \| lsx << 16` |
| 2 | `lsy \| lsz << 16` |
| 3 | palabra de rotación (la disposición de las tres menores del almacén) |

Las palabras se forman así:

```
pa  = quantise(p_a,  lo_a,     hi_a,     2^P − 1)     P = geometryPositionBits, 10..16
lsa = quantise(clamp(ln s_a, −20, 20), lo_{4+a}, hi_{4+a}, 2^S − 1)    S = geometryScaleBits, 6..10
rotación: la palabra del almacén con cada componente v de 10 bits redondeada a R bits (R = geometryRotationBits, 5..10):
    d = 10 − R;  v' = (d == 0) ? v : min((v + 2^(d−1)) >> d, 2^R − 1) << d
```

Los rangos son los del registro: atributos 0–2 y 4–6. El decodificador
escribe:

- la posición, a partir de `dequantise(pa, lo_a, hi_a, 2^P − 1)`;
- la escala logarítmica, a partir de `dequantise(lsa, lo_{4+a}, hi_{4+a},
  2^S − 1)`;
- la palabra 3, tomada sin cambios como la palabra de rotación del almacén.

La cabecera empaqueta `P | S << 8 | order << 16 | R << 24` en un u32. Cuando
ese u32 es 0, los valores por defecto son 16, 10, 1 y 10. Cuando R se lee
como 0, es 10.

#### Planos y chunks

La carga descomprimida son `ceil(splats / chunkSplats)` chunks. Cada chunk
tiene 16 planos de bytes de `chunkSplats` bytes:

```
chunkBytes = 16 · chunkSplats + (order == 3 ? (chunkSplats / 256) · 32 : 0)
byte del plano p para el splat i de la imagen:  chunk (i / chunkSplats) · chunkBytes + p · chunkSplats + (i mod chunkSplats)
```

| Plano | Byte |
|---|---|
| `2f`, `2f + 1` (f = 0..5) | byte bajo y alto del campo de 16 bits `f`: px, py, pz, lsx, lsy, lsz |
| 12, 13, 14, 15 | bytes 0, 1, 2, 3 del campo de rotación de 32 bits |

Los bytes de los splats `≥ splats` son 0. Un campo guarda o un valor absoluto
o un residuo, como se describe a continuación.

#### Claves y predicción (órdenes 1 y 2)

El decodificador toma estos parámetros de la cabecera y del registro:

- `interval`: el `geometryInterval` de la cabecera;
- `stagger`: el bit 5 de los flags;
- `order`: el `geometryOrder` de la cabecera;
- `frame`: el `source` del registro (desde 0).

```
geometryKey(j)   = interval ≤ 1                                  → true
                   stagger    → (j / 256) mod interval == frame mod interval
                   otherwise  → frame mod interval == 0
order(j)         = 0  if geometryKey(j, frame)
                   1  if frame == 0 or geometryKey(j, frame − 1)
                   min(order, 2) otherwise
```

Cuando `order(j) = 0`, los campos son el estado, absoluto. Si no, con `b` el
estado del frame anterior y `c` el estado del frame anterior a ese:

```
campo 16 bits f: pred = (order(j) ≥ 2) ? (2·b_f − c_f) & 0xFFFF : b_f
                 sent = zigzag16((now_f − pred) mod 2^16)
                 now_f = (pred + unzigzag16(sent)) & 0xFFFF
rotación:        pred = order(j) ≥ 2 ? (b & 3) | cada componente de 10 bits (2·b_k − c_k) & 0x3FF : b
                 sent = ((now ^ pred) & 3) << 30
                        | for k = 0..2: zigzagR(((now >> s_k) − (pred >> s_k)) mod 2^R) << 10k,
                          s_k = 22 − 10k + (10 − R)
                 decode: index = (pred ^ (sent >> 30)) & 3;
                         componente k = ((pred >> s_k) + unzigzagR((sent >> 10k) & (2^R − 1))) mod 2^R, en s_k
zigzag_n(d): d como un entero con signo s de n bits → ((s << 1) ^ (s >> 31)) & (2^n − 1);   unzigzag_n(z) = ((z >> 1) ^ −(z & 1)) & (2^n − 1)
```

Tras cada frame, el decodificador conserva tanto `b` (desplazado un lugar)
como el estado nuevo. El orden 2 necesita el frame anterior a ese.

**El flag `geometryKey` del registro** (bit 1 de los flags del prefijo) se
pone cuando `interval ≤ 1`, o cuando no hay `stagger` y
`frame mod interval == 0`. Con `stagger` e `interval > 1`, ningún registro lo
tiene, aunque algunos bloques de 256 sean claves en todos los frames.

**Reposición condicional** (`--geometry-hold`, órdenes 1 y 3) es una elección
del codificador: un campo dentro de su umbral se envía como residuo 0. El
decodificador no hace nada distinto.

#### Movimiento (orden 3)

Fuente: `video_geometry.slang` (`videoGeometryFit`, `motionPosition`, `motionSteps`, `motionApply`, `videoGeometryApply`).

Las gaussianas forman clústeres de 256 índices de frame consecutivos. El
clúster de la gaussiana `j` es `j / 256`, y dentro de un chunk es
`(i mod chunkSplats) / 256`. A los 16 planos de cada chunk les sigue el
movimiento de sus `chunkSplats / 256` clústeres: 8 palabras u32 (32 bytes)
por clúster.

| Palabra | Contenido |
|---|---|
| 0–2 | `c`: el centroide retenido x, y, z, int32, en cuartos de paso de posición (4 × las unidades de P bits) |
| 3–5 | `t`: el centroide del frame x, y, z, int32, cuartos de paso de posición |
| 6 | `qx` (int16, mitad baja) `\| qy << 16` (int16) |
| 7 | `qz` (int16, mitad baja) `\| qw << 16` (int16); 8192 = 1, `w ≥ 0` |

Cuando `geometryKey(j)` es verdadero, los cuatro campos son el estado,
absoluto. Es el valor que retendría un decodificador, no la fuente
recuantizada. Si no, sea `h` el estado retenido, y sean los campos enviados
codificados en zigzag así:

- posición y escala logarítmica: con signo de 16 bits, como `zigzag16`;
- rotación: tres enteros con signo de 10 bits `n0 | n1 << 10 | n2 << 20`,
  deshechos del zigzag como `int(z >> 1) ^ −int(z & 1)`.

El decodificador calcula entonces:

```
posición:  d = 4·h_p − c                                   (int32, h_p la posición retenida de P bits)
           if any |d_a| ≥ 2^16:  p4 = 4·h_p + (t − c)
           else, with (x, y, z, w) = (qx, qy, qz, qw):
             R = [[8192 − ((y·y + z·z) >> 12), (x·y − z·w) >> 12,  (x·z + y·w) >> 12],
                  [(x·y + z·w) >> 12,  8192 − ((x·x + z·z) >> 12), (y·z − x·w) >> 12],
                  [(x·z − y·w) >> 12,  (y·z + x·w) >> 12,  8192 − ((x·x + y·y) >> 12)]]
             p4 = t + ((R·d + 4096) >> 13)
           p^ = clamp((p4 + 2) >> 2, 0, 2^P − 1)
           p_a = (p^_a + unzigzag16(sent_a)) & 0xFFFF
esc. log.: ls_a = (h_ls_a + unzigzag16(sent_{3+a})) & 0xFFFF           (sin predicción)
rotación:  sigma_a = exp(dequantise(ls_a, lo_{4+a}, hi_{4+a}, 2^S − 1))
           posStep = ((hi_0 − lo_0) + (hi_1 − lo_1) + (hi_2 − lo_2)) / (3 · (2^P − 1))
           base    = 4 · 0.70710678 / (2^min(R,10) − 1)
           step_a  = max(base, 0.5 · posStep / max(|sigma_{a+1} − sigma_{a+2}|, 1e-12)),   0 if > 3.14159265   (índices mod 3)
           q_R = normalize((qx, qy, qz, qw) / 8192)
           q   = normalize(q_R ⊗ decodeQuaternion(h_rot)) ⊗ exp(n ⊙ step)
                 exp(v) = (v/|v| · sin(|v|/2), cos(|v|/2)), la identidad cuando |v| < 1e-12;  ⊗ el producto de Hamilton (x, y, z, w)
           palabra de rotación del estado = encodeQuaternion(q)
```

Los rangos que se usan son los floats 0–5 y 8–13 del registro (atributos 0–2
y 4–6). El paso de rotación es aritmética en float. El codificador ejecuta
este mismo kernel compilado para obtener lo que retiene un decodificador, así
que otra implementación coincide con él solo en la medida en que coincidan
sus resultados en float.

#### Lo que el código deja sin determinar en la capa

- **Estado inicial.** En el frame 0, una gaussiana que no es clave (stagger,
  u orden 3) se predice desde un estado que el codificador supone que es el
  último frame del bucle. El codificador lo prepara codificando la cola del
  bucle, o en el orden 3 el bucle entero dos veces. Un decodificador que
  arranca en frío no tiene un estado definido para esas gaussianas hasta su
  primera clave. El buffer de estado del decodificador se crea sin contenido
  inicial.
- **Registros de clúster sin usar.** `videoGeometryFit` nunca escribe los
  registros de clúster del orden 3 de los clústeres sin gaussianas en la
  imagen. Sus 32 bytes son lo que tuviera el buffer del codificador.

### 5.8 Geometría en el vídeo

Fuente: `video_planes_common.slang` ("GEOMETRY IN THE VIDEO"), `video_planes_encode.slang` (`sampleOf`, `valueOf`), `video_planes_decode.slang` (`geometryAt`, `videoPlanesDecode`).

Con el bit 7 de los flags (solo 8 bits, sin capa de geometría, sin partes
altas), la posición, la escala logarítmica y la rotación son planos de luma.
Solo los rangos viajan junto al vídeo.

`geometryFold` (offset 84 de la cabecera) guarda tres pliegues `K`:

| Bits | Pliegue de |
|---|---|
| 0–7 | posición (atributos 0–2) |
| 8–15 | escala logarítmica (atributos 4–6) |
| 16–23 | rotación (atributos 7–9) |

`K = 0` significa solo planos gruesos. Los nueve atributos de geometría,
`g = 0..8`, son los atributos 0, 1, 2, 4, 5, 6, 7, 8, 9.

**Rotación canónica.** La rotación es el cuaternio del almacén, normalizado
y después hecho canónico. De `q`, `q·i = (w, z, −y, −x)`,
`q·j = (−z, w, x, −y)` y `q·k = (y, −x, w, −z)`, se elige el de mayor `|w|`;
en caso de empate se queda el primero en ese orden. Se niega si su `w < 0`.
Los atributos 7–9 son su x, y, z. El rango del atributo 10 se escribe pero no
se usa.

| Planos | Contenido |
|---|---|
| 0–8 | plano grueso del atributo de geometría `g` |
| 9 .. 9 + 2F − 1 | planos finos: para cada atributo plegado en orden de `g`, la onda A y después la onda B |
| 9 + 2F | opacidad |
| después | planos de color (simples o `chroma`, como en el modo simple) |

`F` es el número de atributos plegados: 3 por cada pliegue distinto de cero.
Para un atributo con rango `[lo, hi]`:

```
K = 0:   coarse = quantise(v, lo, hi, 255);           v = dequantise(coarse, lo, hi, 255)
K > 0:   N = floor(65025 / K)
         U = quantise(v, lo, hi, N)
         coarse = round(U · 255 / N)
         A = wave(U),  B = wave(U + 128),  wave(u) = m ≤ 255 ? m : 510 − m,  m = u mod 510
decode:  useA  = 64 ≤ A ≤ 191
         shift = useA ? 0 : 128;   f = min(useA ? A : B, 255)
         e     = coarse · N / 255 + shift                                 (float)
         j0    = floor(e / 510)
         U'    = el candidato entre  j·510 + f  y  j·510 + 510 − f,  j = j0−1 .. j0+1,
                 más cercano a e (en un empate gana el primero encontrado, en el orden j ascendente, a antes que b)
         U     = clamp(U' − shift, 0, N);   v = dequantise(U, lo, hi, N)
```

La w de la rotación se reconstruye como `sqrt(max(0, 1 − x² − y² − z²))`. El
decodificador normaliza `(x, y, z, w)` (con un mínimo de `1e-8` en la
longitud) y la guarda en las tres menores.

### 5.9 Volúmenes

Fuente: `modules/video/include/athenea/video/Athv.h` (`volume`), `shaders/athenea/technique/volume_splat_ply.slang`, `shaders/athenea/volume/vdb_splats.slang` (`vsImportSplats`), `shaders/athenea/scene/splat_decode.slang`.

Con el bit 8 de los flags (`video-encode --volume S`), los splats son las
gaussianas de extinción de un medio, leídas de un PLY de 3DGS escrito con la
disposición de `volume_splat_ply`. El `volumeDensityScale` de la cabecera
(offset 88) es `S`: la extinción por unidad de longitud para una sigma de 1.
Nada más cambia en la disposición de la imagen. Las magnitudes van en
atributos ordinarios.

Los `f_rest_0` y `f_rest_1` del PLY, en orden por color, son el rojo de la
base 0 y de la base 1. El cargador los escribe en las `k = 0` y `k = 3` del
almacén.

| Atributo | Contenido |
|---|---|
| 14 (`k = 0`, rojo, primera base) | ln sigma, el pico de la extinción en unidades de densidad; al menos −9.5 en una gaussiana viva (mínimo −10 + 0.5) |
| 17 (`k = 3`, rojo, segunda base) | la temperatura |
| 11–13 | la radiancia hacia el ojo, referida a pantalla: `srgb(x / (1 + x))` por canal, x = exposición × L |
| 3 | para un visor simple: `clamp(1 − exp(−sigma · S · √(2π) · d), 1/254, 0.999)`, d = exp(media de las tres escalas logarítmicas) |
| otros coeficientes de resto | 0 |

Un reproductor dibuja cada gaussiana con el alfa `1 − exp(−τ)`, donde τ es su
profundidad óptica a lo largo del rayo: sigma × `S` por unidad de longitud.

- **Huecos muertos.** Un hueco muerto (una mota) está en el origen con ln
  sigma −10, escalas logarítmicas `ln 1e-6`, rotación identidad y logit de
  opacidad −20. El importador trata como muerto todo splat con
  `ln sigma ≤ −9.75`.
- **Coeficientes necesarios.** La temperatura necesita `restPerColour ≥ 3`
  (`--sh-degree` ≥ 1). El importador lee ln sigma cuando
  `restPerColour ≥ 1` y la temperatura cuando `restPerColour ≥ 2`.
- **Sin determinar.** El código no dice cómo debe tratar un lector un
  fichero de volumen con `restPerColour = 0`. Tampoco dice cómo tratar un
  fichero de volumen con `chroma`. En ese caso, los coeficientes rojos de la
  base 0 y de la base 1 se reconstruyen a partir de YCoCg, cuyos Co y Cg
  comparten cuatro splats. `video-encode` no rechaza ninguna de las dos
  combinaciones.

### 5.10 Versión 1

Una cabecera de 64 bytes: los offsets 0 a 47 como los de la versión 2, con
el offset 36 sin usar (leído como rendición 0) y sin fracción, frames de
origen, calidad, bloques ni geometría; `positionHighBits` es igual a `bits`.
Su registro empieza con 48 bytes: magic, bytes (u32), index, splats, flags
(u32, bit 0 una clave), source, firstSplat, sourceSplats (u32 cada uno),
ptpTime (u64), time (f64); después la etiqueta de time code y las mismas seis
cuentas u32 y datos que en la versión 2, ignorada la cuenta de geometría. El
frame de un registro de la versión 1 es su origen; sus porciones son
`⌈sourceSplats / splats⌉` y su porción `firstSplat / splats`.

### 5.11 Como stream

`scripts/examples/athv_stream.py` envía un `.athv` por rendición como
mensajes, cada uno una longitud u32 (little endian) y esa cantidad de bytes.
Los tres tipos comparten el prefijo de 28 bytes de §5.2.

| Magic | Tipo | Rendición | Tras el prefijo |
|---|---|---|---|
| `ATVF` | 0 | la del fichero | el registro, byte a byte, salvo `frame` (sigue contando entre bucles; el mismo para el mismo instante en todas las rendiciones) y `ptpTime` (`CLOCK_REALTIME` al enviarlo) |
| `ATVM` | 1 | 0xFFFF | una longitud u32 y esa cantidad de bytes de JSON: el manifiesto |
| `ATVA` | 2 | 0xFFFF | una longitud u32 y JSON: datos auxiliares |

Un manifiesto y un mensaje auxiliar llevan `bytes` = su propia longitud,
versión 2 y flags 0. El manifiesto se envía primero y después una vez por
segundo, antes de las imágenes de ese frame; los datos auxiliares justo
después de cada manifiesto y otra vez cuando cambian. Las imágenes del frame
k de todas las rendiciones van antes que cualquiera del frame k + 1.

**El manifiesto** es `{"version": 1, "fps": {"num", "den"}, "frames",
"renditions": [...]}`, cada rendición los campos de la cabecera por su nombre
(`id`, `bits`, `restPerColour`, `chunkSplats`, `columns`, `maxSplats`,
`hiInVideo`, `antialiased`, `positionBlock`, `positionHighBits`,
`chromaPlanes`, `stableOrder`, `geometryLossless`, `geometryKeyInterval`,
`geometryStagger`, `geometryPositionBits`, `geometryScaleBits`,
`geometryOrder`, `geometryRotationBits`, `geometryVideo`, `geometryFold`,
`quality`) y lo que usa un reproductor para dimensionarse: `gaussians` (la
mediana por frame), `maxGaussians`, `shDegree`, `picturesPerFrame`,
`pictureWidth`, `pictureHeight`, `meanBytesPerFrame`, `meanMbps`,
`fullRange` (true), y, de la primera imagen clave, `codec` (`hvc1.…`),
`codecAnnexB` (`hev1.…`), `profile`, `tier`, `level` y `hvcC` (base64). Los
campos son los de `describe` de `scripts/examples/athv_read.py`.

**Los datos auxiliares** son cualquier objeto JSON; el bloque de entorno con
el que ilumina un reproductor es `{"environment": {"url", "mime", "format":
"latlong", "toWorld": 3 × 3 rows, "intensity", "exposure", "background"}}`.

## 6. `.aths`: una escena neural

Una escena neural es un directorio, `<nombre>.aths/`, que escribe y lee
`athenea neural-scene` en la rama `neural`, donde está el código que se cita
abajo (§2.3.4 del `docs/operations.md` de esa rama dice qué hace con él cada
subcomando). Guarda una escena de gaussianas en piezas, con el momento en que
cada gaussiana llegó, se fue y fue refinada, de modo que un directorio
responde a tres preguntas: qué es la escena ahora, qué era en cualquier
momento anterior y qué dibujar de ella desde un ojo dado. El manifiesto lo
leen y escriben `modules/io/src/NeuralScene.cpp`, el registro de un render lo
dispone `shaders/athenea/neural/scene_render.slang` (`render/Discovery.h`), y
las capas junto a una pieza las lee `apps/athenea/src/CmdNeuralScene.cpp`.

### 6.1 Disposición

```
<nombre>.aths/
  scene.json                      el manifiesto (§6.2)
  pieces/<id>.athg                las gaussianas de una pieza como las dejó un render (§6.3)
  pieces/<id>.usda|.usdc|.ply|…   las gaussianas de una pieza convertidas o traídas (add, discover)
  pieces/<id>.gone                sellos: la mirada que encontró ida cada gaussiana (§6.4)
  pieces/<id>.refined             sellos: la mirada que puso gaussianas más finas sobre cada una
  pieces/<id>.seen                sellos: la mirada que vio cada una por primera vez
  pieces/<id>.prims               ids: el prim del que se hizo cada gaussiana (§6.5)
  pieces/<id>.athn                la apariencia neural de una pieza, una vez ajustada (§6.7)
  <nube de la pieza>.l<snapshot>  la luz dicha de nuevo en ese snapshot (§6.6)
  <nube de la pieza>.g.<grupo>.<n>  la parte de un grupo de luces (§6.6)
```

- **Rutas.** Toda ruta del manifiesto es relativa al directorio, con `/` como
  separador, así que una escena se mueve como se mueve un directorio. Un
  fichero de capa no se nombra en el manifiesto: se encuentra por su nombre
  junto a la nube de su pieza.
- **No se sustituye nada.** Una mirada añade piezas y sellos; nada se quita
  del disco. Una gaussiana nunca se reordena dentro de su pieza, así que un
  fichero de sellos, uno de ids, una capa y la nube de la pieza van índice a
  índice.
- **Ids de pieza.** Una pieza de `shoot` es
  `<cámara>_t<tiempo>_f<huella>_s<snapshot>` (`tiempo` impreso como `%g`,
  `huella` el lado de sus celdas en píxeles, `snapshot` su lugar en
  `snapshots` contando desde uno). Cualquier otro id es libre; un lector toma
  los ids del manifiesto y no analiza ninguno.
- **Escritura.** A diferencia de los demás formatos de aquí, `scene.json` y
  los ficheros binarios se escriben en su sitio (truncados y reescritos), no
  con un nombre `.partial`: un lector no debe leer una escena que un escritor
  está escribiendo.

### 6.2 `scene.json`

JSON en UTF-8, un objeto. Una clave ausente toma el valor por defecto de
abajo; una clave que un lector no conoce se ignora.

| Clave | Tipo | Por defecto | Valor |
|---|---|---|---|
| `format` | cadena | obligatoria | `"athenea-neural-scene"`; cualquier otra cosa no es una escena neural |
| `version` | número | obligatoria | `2`. La versión `1` se lee tal cual (la 2 añadió `origin.snapshot`, `origin.footprint`, `samples`, `footprint` y `taken` de un snapshot, y `frames`); cualquier otra se rechaza |
| `name` | cadena | `""` | el nombre de la escena |
| `upAxis` | cadena | `"Y"` | `"Y"` o `"Z"` |
| `metersPerUnit` | número | `1` | metros por unidad de escena |
| `antialiased` | bool | `true` | `false`: se dibuja sin la compensación del filtro 2D de Mip-Splatting. Solo se escribe cuando es false |
| `spread` | número | `0` | la desviación típica, en píxeles del ojo que encontró cada uno, con que se hicieron sus discos. Solo se escribe cuando es mayor que 0 |
| `baseColour` | bool | `false` | `true`: el tinte (floats 23–25, §6.3) de una gaussiana opaca guarda el color base de su superficie, aparte de la luz con que se encontró. Solo se escribe cuando es true |
| `lightGroups` | array de cadenas | `[]` | los grupos de luces de los que la escena guarda una parte (§6.6) |
| `prims` | objeto | `{}` | ruta de prim → su id de Cryptomatte en ocho dígitos hexadecimales (escritos en minúscula). Una entrada cuyo valor no son ocho dígitos hexadecimales se salta |
| `source` | objeto | | `stage`: de qué se hicieron las piezas; `licence`: qué pide a quien la use. Las dos cadenas, `""` por defecto |
| `bounds` | objeto | | `min`, `max`: tres números cada uno, la caja que rodea las de todas las piezas. Se escribe, no se lee |
| `pieces` | array | `[]` | §6.2.1 |
| `lights` | array | `[]` | `name`; `kind`, `"distant"` (por defecto) o `"dome"`; `towards`, `radiance`, tres números cada uno. Bajo qué se hicieron las piezas: se registra, nada lo lee |
| `cameras` | array | `[]` | `name`; `eye`, `target`, `up`, tres números cada uno; `focal`, mm sobre una apertura horizontal de 24,576 mm (por defecto 35) |
| `snapshots` | array | `[]` | una mirada cada uno: `time` (el código de tiempo de la escena), `camera`, `pieces` (los ids que añadió); y de la mirada de un render `samples` (caminos por píxel), `footprint` (el lado de sus celdas en píxeles) y `taken` (ISO 8601). Un sello (§6.4) es el lugar de un snapshot aquí, contando desde uno |
| `frames` | array | `[]` | un frame renderizado cada uno: `time`, `camera`, `width`, `height`, `samples`, `firstSnapshot` y `lastSnapshot` (sus miradas, contando desde uno), `seconds`, `taken`. Solo se escribe cuando no está vacío |

#### 6.2.1 Una pieza

| Clave | Tipo | Por defecto | Valor |
|---|---|---|---|
| `id` | cadena | obligatoria | única en la escena |
| `cloud` | cadena | obligatoria | el fichero de sus gaussianas: un `.athg` (§6.3) para las de un render, o cualquier nube que lea athenea |
| `prim` | cadena | `""` | el prim del que vino |
| `gaussians` | número | `0` | cuántas tiene |
| `bounds` | objeto | | `min`, `max`: tres números cada uno, en el mundo; con lo que se recorta una vista |
| `transform` | 16 números | | de objeto a mundo, por filas |
| `revision` | número | `1` | sube cada vez que se escriben sus gaussianas, sus sellos o su apariencia |
| `movedAt`, `paintedAt` | número | `revision` | la revisión en que se movió por última vez, y en la que cambió por última vez de qué está hecha: un ajuste sube solo `paintedAt`, gaussianas o sellos nuevos suben las dos |
| `static` | bool | `true` | sus gaussianas no se mueven de un frame a otro |
| `origin` | objeto | | `kind`: `"converted"` (por defecto; de geometría, entera), `"discovered"` (lo que vio una cámara) o `"generated"`; `camera`, `time`: qué la encontró y cuándo; `snapshot`: el snapshot en que entró, contando desde uno (ausente: antes de ninguna mirada); `footprint`: el lado de sus celdas en píxeles (la de un render) |
| `gone`, `removed` | cadena, número | ausente | su fichero `.gone` y cuántas gaussianas sella |
| `refined`, `refinedCount` | cadena, número | ausente | su fichero `.refined` y cuántas sella |
| `seen`, `discovered` | cadena, número | ausente | su fichero `.seen` y cuántas han encontrado las miradas |
| `prims` | cadena | ausente | su fichero `.prims` |
| `lights` | array de números | `[]` | los snapshots en que un frame dijo de nuevo su luz, ascendentes: un fichero `.l<snapshot>` cada uno (§6.6) |
| `appearance` | objeto | ausente | `file` (su `.athn`), `lit`, `split` (bools), y `fit`: `steps`, `against` (`"harmonics"`), `relMSE`, `p99`, `max` (un entero), tal como los midió el dispositivo con cámaras reservadas |

Una pieza sin `id` o sin `cloud` hace ilegible el manifiesto.

### 6.3 `.athg`: las gaussianas de un render

Una cabecera de cuatro `u32`, y luego `count` registros de `floats` `f32`:

| Offset | Tipo | Valor |
|---|---|---|
| 0 | `u32` | magic `0x47485441` (`ATHG`) |
| 4 | `u32` | versión, `1` |
| 8 | `u32` | `count`, gaussianas |
| 12 | `u32` | `floats`, los floats de un registro; no 0 |

El mismo marco guarda las capas de §6.6, con `floats` = 4. El registro de
una pieza es de 30 floats; las escenas tomadas antes del 13 de noviembre de
2026 tienen 26 (sin padre) o 27 (la anchura de un padre, sin color) y se leen
tal cual.

| Floats | Campo |
|---|---|
| 0–2 | posición, en el espacio y la unidad de la escena |
| 3 | opacidad, como logit |
| 4–6 | las tres escalas, como logaritmos naturales: un disco en la superficie tan ancho como era su píxel a su distancia, y una décima de grueso |
| 7–10 | rotación, un cuaternión w x y z |
| 11–13 | color: el primer armónico esférico como lo guarda una captura (el color codificado para pantalla menos un medio, sobre Y₀₀); no hay armónicos superiores |
| 14–16 | la normal de sombreado de la superficie, en el mundo |
| 17 | metallic, 0–1 |
| 18 | roughness, 0–1 |
| 19 | transmisión: cuánto pasa a través, 0 por debajo de 0,02 |
| 20 | la fila del material en el frame que la hizo, más uno; 0 para lo que se vio a través de un vidrio |
| 21 | el índice de refracción |
| 22 | lámina: 1 para una lámina sin interior (o nada que desvíe), 0 para un sólido |
| 23–25 | tinte, r g b: lo que la superficie deja pasar en un cruce, antes de Fresnel (para un metal lo que refleja de frente, y con `baseColour` el color base de una superficie opaca) |
| 26 | la anchura de las gaussianas más gruesas sobre las que se puso (su padre), en unidades de escena; 0 donde no se puso sobre nada |
| 27–29 | el color del padre, codificado para pantalla, como se guarda un color base |

Un lector rechaza un magic incorrecto, una versión distinta de 1,
`floats` = 0 y un fichero más corto de lo que dice su cabecera; no mira más
allá del último registro.

### 6.4 Sellos: `.gone`, `.refined`, `.seen`

Un `u32` por gaussiana de la pieza, en su orden, y nada más: el fichero mide
exactamente `4 × gaussianas` bytes, y un lector rechaza cualquier otra
longitud. Un valor es el lugar de un snapshot en `snapshots`, contando desde
uno; 0 es ninguno.

| Sello | Dónde | Qué dice |
|---|---|---|
| encontrada | el `origin.snapshot` de la pieza | la mirada que hizo la pieza: todas sus gaussianas llegaron entonces |
| ida | `.gone` | el mundo cambió: una mirada posterior vio a través de donde estaba, no encontró nada, u otro color |
| refinada | `.refined` | una mirada más cercana o un objetivo más largo puso gaussianas más estrechas sobre ella. No se ha ido: es su padre, y de lejos sigue siendo la que se dibuja |
| vista | `.seen` | la mirada que la vio por primera vez (`look`) |

### 6.5 `.prims`

Dispuesto como un fichero de sellos, y guarda otra cosa: el id de
Cryptomatte del prim del que se hizo cada gaussiana -- el hash de la ruta del
prim, el id que `athenea stage` escribe en sus capas `CryptoObject` -- o 0
donde el frame no nombró ninguno (lo que se vio a través de un panel de
vidrio). `prims` del manifiesto dice qué ruta es un id. Una gaussiana toma el
id del píxel en el centro de su celda.

### 6.6 Capas de luz

Las dos son ficheros `.athg` (§6.3) de cuatro floats por registro, y un
registro nombra la gaussiana de la que es: el float 0 es el índice de la
gaussiana en la pieza, como `f32` -- exacto solo hasta 2²⁴, así que una pieza
de más gaussianas no puede llevar capas para las que pasan de ahí -- y los
floats 1–3 son un color. Un registro cuyo índice no es menor que el número de
gaussianas de la pieza se salta. Una capa cuyo `floats` no es 4 se rechaza, y
también una capa `.l` de más registros que gaussianas tiene la pieza.

- **La luz dicha de nuevo**, `<nube>.l<snapshot>`: se escribe cuando un
  frame de `shoot --relight` dijo la luz de gaussianas que no hizo. Los
  floats 1–3 sustituyen el color del registro (floats 11–13, como lo guarda
  una captura). Un lector pone los ficheros sobre la pieza en snapshot
  ascendente, hasta el snapshot pedido (todos para la escena tal como es
  ahora). `lights` de la pieza enumera los snapshots.
- **La parte de un grupo**, `<nube>.g.<grupo>.<n>`: la luz que manda ese
  grupo solo, rgb lineal. Los ficheros de un grupo se ponen en `n`
  ascendente (`shoot` escribe `n` = 1). Al dibujar, la capa de un grupo se
  suma al nivel del grupo. Un grupo llamado `dark` (lo escriben escenas
  tomadas antes de que los grupos se encontraran con la escena) es lo que
  brilla solo, y se resta una vez por cada grupo dibujado, al nivel de ese
  grupo, para que lo que brilla se cuente una vez.

### 6.7 `.athn`: la apariencia neural de una pieza

`modules/io/include/athenea/io/NeuralFile.h`. No nombra ninguna nube: es de
la nube que tiene al lado, gaussiana a gaussiana, y un lector comprueba el
número.

| Offset | Tipo | Valor |
|---|---|---|
| 0 | `u32` | magic `ATHN` |
| 4 | `u32` | versión, `1`; cualquier otra se rechaza |
| 8 | `u32` | gaussianas |
| 12 | `u32` | mitades por gaussiana (11) |
| 16 | `u32` | palabras por gaussiana, (mitades + 1) / 2 (6) |
| 20 | `u32` | pesos (1008; 2016 para dos redes) |
| 24 | `u32` | flags: bit 0 lit; bit 1 split, los pesos de dos redes, a la primera nunca se le dice la vista |
| 28 | `u32` × gaussianas × palabras | los latentes, dos `f16` por palabra, el bajo primero |
| … | `f32` × pesos | los pesos de la red |

### 6.8 Cómo se lee

Con S un snapshot -- el último, para la escena tal como es ahora; `--at` un
tiempo o `--snapshot` para como estaba:

| Lo pedido | Las gaussianas dibujadas |
|---|---|
| la escena en S | de las piezas encontradas en S, las que ni se han ido ni se han refinado en S |
| la escena en S para un ojo, por un umbral en píxeles (`render --lod`, `--budget`) | de las piezas encontradas en S, las que no se han ido en S cuya anchura del padre (float 26) proyecta por encima del umbral, o que no tienen padre, y cuya propia anchura proyecta por debajo, o que nada había refinado en S |
| un frame tal como se renderizó (`render --frame N`) | la escena en el `lastSnapshot` del frame, por su cámara a su tamaño |

«Encontrada en S» es `origin.snapshot` ≤ S (ausente cuenta como antes de
ninguna); «ida en S» es un valor de `.gone` en 1..S, y lo mismo para
refinada. La anchura de una gaussiana es su nivel de detalle: no se guarda
ningún árbol, y cada gaussiana se elige por su propio registro y sus dos
sellos.

### 6.9 Cómo se entrega

El directorio es como se guarda una escena. `neural-scene bake` escribe un
`.athc` (§3) de las gaussianas en pie en un snapshot, y `neural-scene export`
partes SPZ y un índice JSON para un visor web (`docs/operations.md` §8.2);
los dos pierden los sellos y el material.

## 7. Ficheros relacionados

**Un directorio de páginas** (`athenea flatten --range`, que leen `athenea
stream` y `athenea video-encode`): un fichero SPZ por página,
`f<frame>_c<page>.spz` (un `.ply` donde no hay `.spz`), y `stream.txt`, un
fichero de texto de líneas; `#` empieza un comentario.

```
rate <time codes por segundo>
chunks <el máximo de páginas por frame>
chunkSplats <splats por página>
degree 3
frame <índice> <time code> <páginas> <gaussianas> <bytes>     (uno por frame)
```

Un frame de menos páginas que `chunks` quita el resto del stream.

**Los formatos de la web.** sparkwebGPU, el reproductor web de athenea,
escribe un `.athc` versión 3 (`ATH3`: secciones, bloques y páginas, cada uno
con gzip o sin él, como palabras, planos de bytes o diferencias de planos de
bytes, con un transfer CPCA) y una escena web `.athz` (un zip que contiene
`scene.json` y sus nubes). Los dos son formatos de sparkwebGPU y se
especifican en su repositorio; no comparten nada con el `.athz` de athenea
(§4) salvo el nombre.

**Las páginas en vivo** (`ATVP`, `modules/lod/include/athenea/lod/DevicePages.h`)
y las páginas SPZ en staging (`LSPZ`) son cabeceras dentro del proceso sobre
bytes que un productor entrega al pool de streaming; ningún fichero las
guarda.
