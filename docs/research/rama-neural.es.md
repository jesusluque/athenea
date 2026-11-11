# Rama `neural` de athenea: implementación y características

Estado a 11 de noviembre de 2026. Rama `neural`, 36 commits sobre `release-0.5` (cf3ab5f1), solo local (sin push).
El registro completo de decisiones y medidas, en inglés, está en `docs/decisions.md` (secciones del 11 Nov); los
comandos, en `docs/operations.md` / `.es.md` §2.3.1, §2.3.2 y §8.2; la estructura, en `docs/development.md` §1.3.1.

Todas las cifras son del peón (captura 3DGS de 325 833 gaussianas, SH3) salvo que se diga otra cosa, medidas en un
Apple M5 Pro (Metal) y en una RTX 3070 Laptop (CUDA, WSL2). Ningún método publicado se ha ejecutado: las
comparaciones son contra el propio motor.

## 1. Qué es

Tres añadidos al rasterizador de splats de athenea, que no cambian un frame que no los pide:

1. **Trace por píxel**: cada píxel guarda las gaussianas de las que está hecho.
2. **Lift de máscaras**: una máscara hecha en unas cámaras se lleva a las gaussianas y se usa como key en cualquier otra.
3. **Apariencia neural**: el color de cada gaussiana sale de un latente y de una red compartida, en vez de sus SH;
   con una variante iluminada que responde a luces.

Todo el cálculo son kernels Slang en GPU, incluido el entrenamiento. No hay aritmética en CPU ni autodiff.

## 2. Trace por píxel

- **Qué guarda** (`RenderSettings::trace` → `RenderTargets::trace`): por píxel, las 4 gaussianas de más peso
  (slot, peso T·α, depth), más `rest` (el peso que no cupo en la lista) y `under` (lo que cerró el recorrido por
  detrás). Los tres suman la cobertura del píxel. 11 palabras por píxel, un plano por palabra.
- **Dónde**: `shaders/athenea/splat/splat_trace.slang`; cuatro entry points con trace en `splat_blend.slang`;
  `FrameStats::slotBases` convierte un slot en (instancia, gaussiana).
- **Comprobación**: `ReferenceRenderer` guarda la trace del recorrido exacto; el test `[trace]` exige color idéntico
  bit a bit con y sin trace y 0 diferencias con la referencia. Pasa en Metal y en CUDA.
- **Coste sobre el frame sin trace** (mínimo de 5 tandas de 100 frames):

| | Metal 1080p | Metal 4K | CUDA 1080p | CUDA 4K |
|---|---|---|---|---|
| Frame sin trace | 4,06 ms | 6,87 ms | 2,44 ms | 4,25 ms |
| Sobrecoste de la trace | +0,05 ms | +0,62 ms | +0,68 ms | +2,23 ms |

  El listón del plan (+1 ms a 4K) se cumple en Metal y no en la 3070. La primera versión costaba +2,0 ms (Metal) y
  +4,15 ms (CUDA) a 4K.

## 3. Lift de máscaras y key sin croma

- **Cómo** (`render/SplatLift.h`, `splat_lift.slang`): por gaussiana se acumulan dos sumas, peso·máscara y peso, en
  enteros (Metal no tiene atómicos float); el cociente es cuánto está la gaussiana dentro de la máscara. Varias
  cámaras suman en las mismas gaussianas.
- **Dos formas de elevar**: por la trace (las 4 gaussianas del píxel) o durante el blend (`contributionMode` 3,
  todas las gaussianas del píxel). El matte en otra cámara se dibuja desde su trace, repartiendo `rest`.
- **Comando**: `athenea lift` (máscara pintada por cámara o rectángulo; `--hard`, `--by-trace`, `--turn N` para
  una vuelta de cámara). Imprime además el parpadeo del key.
- **Medido contra el Cryptomatte** (dos nubes de test interpenetradas): lo que limita el lift es lo que vieron las
  cámaras, no el top-4. Con 1 cámara, 1237 de 2400 gaussianas quedan sin opinión y 3886 píxeles de otra cámara se
  desvían más de 0,25; con 8 cámaras, 69 gaussianas y 247 píxeles.
- **Estabilidad temporal del key** (tres frames a medio grado, 1080p; la nube sin key da relMSE 0,000011 y peor
  píxel 23 code values):

| Lift | relMSE del key | Peor píxel |
|---|---|---|
| 4 cámaras, durante el blend | 0,000002 | 31 |
| Igual, `--hard` | 0,000006 | 69 |
| 4 cámaras, por la trace | 0,000020 | 255 |
| 1 cámara | 0,000151 | 255 |

  Para un key estable hacen falta el lift durante el blend y varias cámaras.

## 4. Apariencia neural (sin luz)

- **Modelo** (`shaders/athenea/neural/neural_appearance.slang`, tras Hahlbohm et al., arXiv 2609.05255): 8 features
  y un color base por gaussiana (11 números; el SH3 son 48); una red por nube, 41→16→16→6 con ReLU y sin biases,
  1008 pesos. Entrada: SH0–3 de la dirección de vista y sin/cos de las features. Salida: tanh sumada al color base.
- **Dónde corre**: una vez por gaussiana visible, tras su proyección (`SplatInstance::neural`, kernel
  `neuralShade`); el blend no cambia. Los pesos se leen una vez por grupo de 64 gaussianas.
- **Calidad** (8 cámaras no usadas en el fit, contra el SH3 de la propia nube):

| Apariencia | Números por gaussiana | relMSE | p99 (code values) | Peor píxel |
|---|---|---|---|---|
| SH grado 1 | 12 | 0,00783 | 35,5 | 83 |
| SH grado 2 | 27 | 0,00188 | 16,8 | 77 |
| Latentes + red, 5000 pasos | 11 | 0,00033 | 7,0 | 112 |

- **Coste del frame** (`athenea bench`): igual que el SH3.

| | Metal 1080p | Metal 4K | CUDA 1080p | CUDA 4K |
|---|---|---|---|---|
| SH3 | 3,95 ms | 7,02 ms | 2,60 ms | 4,58 ms |
| Red por gaussiana | 4,09 ms | 7,09 ms | 2,48 ms | 4,97 ms |

- **Parpadeo al mover la cámara**: 3 a 6 veces el del SH3 (relMSE 0,000018–0,000045 frente a 0,000005–0,000007;
  p99 2–3 frente a 1 code value).
- **En disco**: sidecar `.athn` (`io/NeuralFile.h`), latentes como halves, 24 bytes por gaussiana (el SH3 son 180 en
  un PLY). El del peón ocupa 7,8 MB junto a un PLY de 80,8 MB. `athenea render --neural` contra el render SH3:
  PSNR 42,7 dB, SSIM 0,994.

## 5. Fit en GPU

- **Un paso** (`render/NeuralFit.h`, `neural_fit.slang`): frame de referencia y frame por latentes; error relativo
  por píxel (el relMSE del motor); el blend lleva el error de cada píxel al color de sus gaussianas
  (`contributionMode` 4); backward de la red una vez por gaussiana, escrito a mano; el gradiente de cada peso lo
  suma un hilo leyendo (determinista, sin atómicos float); Adam.
- **Tiempo**: 39 ms por paso en Metal, 60 ms en CUDA; 5000 pasos del peón en 195 s (Metal).
- **Comando**: `athenea neural-fit` (`--steps`, rates, `--rate-decay`, `--save`/`--load`, `-o`); mide siempre en
  cámaras que el fit no usa.
- **Autodiff de Slang descartado**: en Slang 2026.14.1 `bwd_diff` da gradientes incorrectos con bucles anidados de
  cuentas distintas (shader-slang/slang#13000, arreglado en v2026.20) y cuesta 71 veces más (505 frente a 7,1 ms).

## 6. Apariencia neural iluminada

- **Modelo** (`NeuralCloud::lit`): la red recibe además la dirección de la luz en el sistema propio de la gaussiana
  (su normal girada hacia la cámara) y da la respuesta del color base a una unidad de luz:
  `color = radiancia · base · (reached · directa + rebotada)`. `reached` es cuánta luz llega a la gaussiana y lo
  aporta el motor; la segunda salida (`rebotada`) no se multiplica por ella.
- **Varias luces**: hasta 8 por frame, una evaluación por luz, sumadas (la respuesta es lineal en la luz).
- **De dónde sale `reached`**: de un shadow ray por gaussiana, o del shadow map de la nube
  (`technique::SplatShadowMap`, sin rayos), que entra por `SplatLights::visibilityFactors`.
- **Contra el relight analítico del motor** (peón, cámaras y soles no vistos, 960×540):

| A la red se le dice | relMSE | p99 | Frame |
|---|---|---|---|
| Sin sombras en la referencia | 0,0010 | 19 | 7,3 ms (motor: 7,3 ms) |
| Sombras; solo la dirección | 0,0025 | 34 | 7,4 ms (motor con rayos: 32,3 ms) |
| Sombras; shadow map de 1024 | 0,00068–0,00071 | 14–15 | 7,4 + 4,1 ms (motor: 32,5 ms) |
| Sombras; un rayo por gaussiana | 0,00052 | 10,6 | 32,6 ms (motor: 32,4 ms) |

  Bajo tres soles de colores a la vez (ajustada con un sol blanco por frame): relMSE 0,00052–0,00054.
- **Contra bakes path-traced** (`--pictures`: peón de arcilla sobre suelo rojo, 100 mil gaussianas, la misma malla
  horneada con `mesh2splat` bajo 24–48 soles, 3 rebotes; soles y cámaras no vistos):

| A la red se le dice | relMSE | p99 |
|---|---|---|
| Nada aprendido | 3,4 | 205 |
| Solo la dirección | 0,030 | 88 |
| Shadow map, solo multiplicando | 0,0078 | 80 |
| Shadow map + salida de luz rebotada (24 soles) | 0,0045 | 38 |
| Igual, 48 soles, 8000 pasos, rates a un décimo | 0,0033 | 34 |

  El error que queda está en el borde de las sombras y en zonas finas que se dan sombra a sí mismas: es el shadow
  map, no la red. Descartado como causa: número de soles, ruido del bake y memorización.
- **CUDA**: en ese WSL no hay trazado de rayos, así que `--shadows` se rechaza; el fit iluminado sin sombras da lo
  mismo que en Metal (relMSE 0,0010).

## 7. Decoder por píxel (solo cronometrado)

`RenderSettings::neuralFeatures` (8, 16 o 32 canales): el blend suma features y una red 16+N→32→32→3 corre por
píxel (`neural_pixel.slang`). No está entrenada; se mide su coste.

| El frame se dibuja con | Metal 1080p | Metal 4K | CUDA 1080p | CUDA 4K |
|---|---|---|---|---|
| Red por gaussiana | 4,09 ms | 7,09 ms | 2,48 ms | 4,97 ms |
| 8 canales + red por píxel | 6,27 ms | 15,57 ms | 3,17 ms | 10,34 ms |
| 16 canales | 6,95 ms | 18,16 ms | 4,90 ms | 13,75 ms |
| 32 canales | 9,19 ms | 26,38 ms | 8,01 ms | 24,75 ms |

Por píxel es entre 1,3 y 5 veces más lento que por gaussiana; por eso la rama decodifica por gaussiana.

## 8. Comandos y ficheros nuevos

| Comando | Qué hace |
|---|---|
| `athenea lift` | máscara en unas cámaras → key en otra; `--turn` para una vuelta de cámara |
| `athenea neural-fit` | ajusta la apariencia neural; `--relight`, `--shadows`, `--reached ray\|map`, `--pictures`, `--turn`, `--save`/`--load` |
| `athenea render --neural f.athn` | dibuja una nube con su apariencia neural |
| `athenea bench --trace`, `--neural`, `--features N` | cronometra la trace y las dos formas de decode |

Código: `modules/render` (`SplatLift`, `NeuralAppearance`, `NeuralFit`, cambios aditivos en `TileRasterizer` y
`ReferenceRenderer`), `modules/io/NeuralFile`, `shaders/athenea/neural/`, `shaders/athenea/splat/splat_trace.slang`,
`splat_lift*.slang`, `apps/athenea/src/CmdLift.cpp`, `CmdNeuralFit.cpp`. Tests: `[trace]`, `[lift]`, `[neural]`,
`[fit]`, `[relight]`, `[lights]`, `[file]` en `athenea_render_tests`.

## 9. Lo que no está hecho

- Transporte de luz aprendido entre gaussianas (fase 5 del plan): sin empezar.
- Domo o LED wall como luz; luces que no estén en el infinito.
- Integración en USD, `athenea view`, LOD y decimate: la apariencia vive en un sidecar que casa con la nube solo por
  orden y número de gaussianas.
- Benchmarks públicos (Mip-NeRF 360, NVOS, SPIn-NeRF) y comparación directa con otros métodos.
- Medidas en una L4 y en CUDA con OptiX.
- El decoder por píxel no se entrena.
- Un color base que arranque del material: una conversión sin bake no conserva las mismas gaussianas que una con bake.

## 10. De dónde vienen las ideas

Ningún código es copiado. Ideas tomadas de la literatura (conocidas por el mapa del estado del arte y por los
resúmenes de athenea-research; las referencias no se han verificado leyendo los artículos en esta sesión):

- Latente por gaussiana + red compartida: Hahlbohm et al., arXiv 2609.05255.
- Lift de máscaras por pesos del blend: FlashSplat (arXiv 2409.08270), LUDVIG (arXiv 2410.14462).
- Dirección de la luz y transmitancia como entradas del decoder: RNG (CVPR 2025), Stochastic Ray Tracing (arXiv 2603.23637).

Decisiones propias de esta rama: la luz en el sistema de la gaussiana, la segunda salida para luz rebotada, el
backprop a mano en kernels, la suma del gradiente por lectura, el formato `.athn` y las medidas de coste y parpadeo.
