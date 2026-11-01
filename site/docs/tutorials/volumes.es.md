# Volúmenes: de un VDB a gaussianas volumétricas

athenea convierte un volumen de niebla de OpenVDB (humo, fuego, una
explosión, una nube) en unas decenas de miles de gaussianas 3D que
representan el propio medio, no una superficie. Su extinción es una suma de
gaussianas, μ_t(x) = Σ σ_i G_i(x), cada G_i de pico 1; los grids de
temperatura y de color, si el fichero los trae, van en cada gaussiana como
medias ponderadas por la densidad. Esas mismas gaussianas las dibuja el
raster como un medio participativo, vuelven a un grid para el path tracer y
llegan a un navegador como vídeo.

Esta página sigue una simulación pública por toda la ruta: la **explosión de
gasolina** de los VDB gratuitos de EmberGen de JangaFX (CC0, 131 frames,
`density`, `temperature` y `flames` en half float, hasta 3,2 millones de
vóxeles activos por frame), que se descarga en
[jangafx.com](https://jangafx.com/software/embergen/download/free-vdb-animations).

## Qué hace la conversión

`athenea vdb2splat` ajusta la mezcla en la GPU, cada paso un kernel de
compute:

1. **Corte.** Las hojas de la densidad se cortan en bloques (como mucho
   `--max-block` vóxeles de lado, 64 por defecto), una gaussiana cada uno,
   colocada en los momentos del bloque. El primer corte se queda con
   `--init-fraction` del presupuesto (0,6).
2. **EM.** `--em-iterations` pasos (40) llevan cada gaussiana a los momentos
   de la densidad de la que responde.
3. **Adam.** `--iterations` pasos (100) sobre el centro, el factor de la
   precisión y el pico, frente a la densidad del grid en sus vóxeles activos.
   Cada `--densify-every` pasos (20), los slots libres del presupuesto van
   donde el residuo es mayor: la mitad parten en dos, a lo largo de su eje
   más largo, las gaussianas que responden de más error, y el resto nacen en
   los peores vóxeles; una gaussiana por debajo de `--prune-mass` del total
   desaparece.
4. **Atributos.** Con la geometría fija, `--attribute-iterations` pasos de
   Jacobi (20) ajustan la temperatura y el color como medias ponderadas por
   la densidad.

Un grid mayor que el dispositivo se ajusta brick a brick (`--tile-dir`): el
plan sale solo de la estructura del árbol, cada brick se lee con la carga
diferida de OpenVDB y una segunda pasada (`--tile-refine`) cierra las
costuras.

## Qué guarda cada gaussiana

| | Bytes | Qué |
|---|---|---|
| En el stage | — | un `UsdVolParticleField3DGaussianSplat` con `primvars:athenea:splat:volume = true` y, por gaussiana, `primvars:athenea:volume:sigma` (half), `…:albedo` (color3h, desde Cd) y `…:temperature` (half), junto a las posiciones, orientaciones y escalas del schema |
| `--quantise half` | 24 | el centro en 16 bits por componente sobre la caja de la nube, ln σ en 16 bits, el factor de la precisión en halves, Cd en rgb8 y la temperatura en 8 bits |
| `--quantise byte` | 20 | lo mismo, con el factor en bytes sobre sus rangos |

`radiance:base` lleva una opacidad para un visor estándar, 1 − exp(−τ) por el
centro de la gaussiana; el raster de athenea no la lee y dibuja el medio.

## Un frame

```sh
GAS=embergen_gasoline_explosion/Gasoline_Explosion_01
athenea vdb2splat $GAS/embergen_gasoline_explosion_a_60.vdb -n 50000 \
    --density-scale 0.03 -o gas60_gs.usdc --back gas60_back.vdb --measure
```

`--back` escribe la mezcla evaluada en los vóxeles activos de la propia
entrada: un `.vdb` que el path tracer dibuja como cualquier volumen.
`--measure` imprime el error L2 relativo de ese grid frente a la entrada, su
mayor diferencia y la transmitancia en forma cerrada de las gaussianas frente
a ratio tracking por el original, a lo largo de rayos aleatorios.

En un Apple M5 Pro, los frames de la gasolina se ajustan así:

| Gaussianas | Error L2 medio de la densidad | Segundos por frame |
|---|---|---|
| 50.000 | 0,096 | 1,8 |
| 20.000 | 0,132 | 1,6 |
| 10.000 | 0,159 | 1,2 |

La simulación pública más pesada que se ha probado, la explosión aérea de
EmberGen, con hasta 24 millones de vóxeles activos por frame, ajusta 50.000
gaussianas en 12 s por frame, con un error medio de 0,161.

## Dibujar el medio

El raster dibuja un medio de gaussianas como un medio participativo, ordenado
y mezclado con todos los demás splats: el alpha de cada gaussiana en un píxel
es 1 − exp(−τ), con τ la profundidad óptica exacta del rayo del píxel a
través de ella. Su color es la radiancia que cada frame hornea para ella:

- **emisión**: un cuerpo negro a la temperatura de la gaussiana, en kelvin
  `temperatureScale × T + temperatureOffset`, por `emissionScale`;
- **scattering simple** de cada luz, con la transmitancia leída del shadow
  map de la nube en la gaussiana (el medio proyecta en el mapa y lo lee, así
  que sombrea mallas y nubes y ellas lo sombrean a él);
- **el domo**, por la transmitancia media de las direcciones del domo en el
  mapa;
- **scattering múltiple** en `octaves` (cuatro por defecto).

El aspecto se controla con unos pocos primvars en el field:

```usda
def "Fire" ( prepend references = @gas60_gs.usdc@</World/Splats> )
{
    float primvars:athenea:densityScale = 0.03
    color3f primvars:athenea:albedo = (0.8, 0.8, 0.8)
    float primvars:athenea:volume:emissionScale = 40
    float primvars:athenea:volume:temperatureScale = 2400
    float primvars:athenea:volume:temperatureOffset = 600
}
```

con una `DistantLight` y una `DomeLight` al lado, y el stage dibujado con
`athenea stage`. Frente a un path trace del grid original con la misma key
light y el mismo domo (1.024 caminos, cuatro rebotes, sin fuego), el raster
del ajuste de 50.000 gaussianas del frame 60 queda a relMSE 0,0171 (29,6 dB,
SSIM 0,977), y con el fuego dibuja en 9,3 ms por frame.

## Una secuencia

```sh
athenea vdb2splat "$GAS/embergen_gasoline_explosion_a_#.vdb" --sequence \
    --independent --frames 4 99 -n 50000 --device-mb 8192 \
    --density-scale 0.03 -o "gas/f_####.usdc"
```

Una tira de `#` nombra los frames. Hay dos formas de ajustarlos:

- **`--independent`**: cada frame, un ajuste completo propio. Es lo que pide
  un fichero de EmberGen: no trae velocidad, y una bola de fuego deja atrás a
  las gaussianas que vienen del frame anterior.
- **Frames en caliente** (por defecto): cada frame parte de las gaussianas del
  frame anterior en sus propios slots, arrastradas por el grid `vel` del
  fichero (el centro movido por el flujo y la covarianza deformada por
  F = I + Δt ∇u conservando su volumen; donde el flujo cizalla demasiado,
  solo el centro), y luego `--warm-iterations` pasos. Los slots son
  estables, así que la gaussiana *k* es la misma en todos los frames, y eso
  es lo que permite al stream predecir su movimiento. El grid `velocity`
  escalonado de Mantaflow se lee con `--velocity velocity --velocity-units
  mantaflow --fps 24`.

Por cada frame, una secuencia imprime las gaussianas vivas, el error de la
densidad, el tiempo del ajuste y el flicker: cuánto se equivoca el cambio de
frame a frame del grid devuelto frente al del original.

## Hasta un navegador

Tres pasos llevan los frames ajustados a un stream que dibuja el
[player web](https://atheneastream.lucab.co.uk/):

```sh
# 1. Hornear la luz de cada frame en un PLY 3DGS (radiancia, ln σ, temperatura)
ATHENEA_VOLUME_PLY=gas/b_0060.ply athenea stage gas/s.usda \
    --eye 0 0 500 --target 0 0 0 --size 32x32 -o bake.exr
# 2. Codificar los frames horneados como un rung de un ladder
athenea video-encode gas/b_*.ply --each-a-frame -o gas.athv --rendition 0 \
    --keyint 30 --bits 8 --rate 30 --sh-degree 1 --quality 0.7 --volume 0.03 \
    --order frame --range-group 30 --geometry --geometry-key 1 \
    --geometry-position-bits 11 --geometry-scale-bits 6 --geometry-rotation-bits 6
# 3. Decodificarlo y juzgarlo frente a los grids originales
athenea vdb2splat "$GAS/embergen_gasoline_explosion_a_#.vdb" --sequence \
    --frames 4 99 --from-athv gas.athv
```

`ATHENEA_VOLUME_PLY` escribe cada medio que dibuja un frame como un PLY con la
luz que ese frame horneó para él (el paso 1 se ejecuta una vez por frame,
sobre `s.usda`, con el aspecto de arriba referenciando `f_####.usdc`).
`--volume` marca el `.athv` como medio y guarda la escala de densidad, para
que el player dibuje cada splat con el alpha 1 − exp(−τ) de su propia
profundidad óptica. Posición, escala y rotación viajan exactas junto al vídeo
(`--geometry`); la opacidad y el color, dentro de él.

Las explosiones publicadas, cada una un ladder de tres rungs:

| Simulación | Frames | Rungs | Mbit/s a 30 fps | Error L2 decodificado, 50k |
|---|---|---|---|---|
| Explosión de gasolina (EmberGen) | 96 | 50k · 20k · 10k | 118 · 48 · 25 | 0,129 |
| Explosión en el aire (EmberGen) | 96 | 50k · 20k · 10k | 121 · 50 · 26 | 0,129 |
| Explosión aérea (EmberGen) | 60 | 50k · 20k · 10k | 116 · 48 · 25 | 0,178 |
| Explosión en el suelo (EmberGen) | 96 | 50k · 20k · 10k | 113 · 47 · 25 | 0,156 |
| Columna de humo (Blender Mantaflow, en caliente) | 90 | 20k · 10k | 67 · 33 | 0,036 (20k) |
| Bola de fuego (Blender Mantaflow, en caliente) | 60 | 50k · 20k · 10k | 169 · 63 · 30 | 0,050 |

Un loop se funde con su primer frame cuando el player recibe `fade=N`: la
extinción de los últimos N frames baja linealmente hasta cero.

## Dónde están los detalles

- [`docs/operations.md` §2.8.3](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#283-athenea-vdb2splat--a-volume-into-volumetric-gaussians-and-back): todas las opciones de `vdb2splat`, los bricks y los primvars.
- [`docs/operations.md` §2.7.2](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#272-athenea-video-encode--splat-frames-as-hardware-video): `video-encode`, con `--volume`, `--slots`, `--cells` y la capa de geometría.
- `docs/operations.md` §4 (authoring): un medio de gaussianas y sus primvars; §8.2: el formato `.athv` y su flag de volumen.
- `docs/decisions.md`, secciones «A volume into volumetric gaussians, and back», «A volume sequence as video planes» y «A medium on the web player»: el diseño y todas las medidas.
