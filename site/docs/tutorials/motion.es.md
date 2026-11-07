# Cómo se mueven las gaussianas

Una nube de athenea se mueve de una de cinco maneras. Cuál, depende de dónde
venga el movimiento (un esqueleto, un transform de USD, la velocidad de un
fluido) y de dónde se dibuje la nube (el motor de escritorio o un navegador al
final de un stream). Esta página nombra cada una, dice dónde se elige y qué
cuesta, y acaba con el motion blur, que comparten todas.

| Movimiento | De dónde viene | Dónde se elige | Lo dibuja |
|---|---|---|---|
| [Skinning en el dispositivo](#skinning-en-el-dispositivo) | un esqueleto UsdSkel | `mesh2splat --skinned`, `athenea visibility` | el motor, en cada frame |
| [Un esqueleto en el navegador](#un-esqueleto-en-el-navegador) | el mismo esqueleto, enviado una vez | la ruta Spark WebGPU | el navegador |
| [La geometría de un stream](#la-geometria-de-un-stream) | frames ya posados o ajustados | `video-encode --geometry-predict` | el player de `.athv` |
| [Un volumen arrastrado por su flujo](#un-volumen-arrastrado-por-su-flujo) | el grid `vel` de la simulación | `vdb2splat --sequence` | el motor o el player de `.athv` |
| [Transforms de USD](#transforms-de-usd) | `xformOp`s, con time samples | el stage | el motor |

## Skinning en el dispositivo

`athenea mesh2splat --skinned` construye las gaussianas en la bind pose y da
a cada una los joints que la llevan, de modo que el Skeleton al que está
ligada la nube la deforma en tiempo de render, en la GPU, igual que a la
malla de la que salió. Cada gaussiana guarda además cómo cambian sus pesos a
lo ancho de ella (`jointWeightGradients`, doce bytes por gaussiana): eso es
lo que la estira en un pliegue como se estira su triángulo, en vez de
moverse rígida con la mezcla de sus joints.

```sh
athenea mesh2splat bird.usda --skinned --resolution 1100 -o bird_gs.usdc
athenea visibility bird_gs.usdc --skeleton-stage bird.usda --skeleton-prim /World/Skel
```

Con `--skinned` se rechaza el bake de luz, porque la luz horneada en una
pose está mal en todas las demás. En su lugar se conserva el bake de
transfer, como dos lóbulos zonales en el marco propio de cada gaussiana, que
giran con ella. `athenea visibility` hornea lo que la nube proyecta sobre sí
misma, por partes, para que un ala sombree el cuerpo en cualquier pose sin
trazar un rayo (`SplatVisibility`). El gorrión (de 5 a 9 millones de
gaussianas, 609 joints) se mueve así.

## Un esqueleto en el navegador

La otra forma de enseñar una nube con skinning en la web es enviarla una
vez, con su esqueleto y su transfer, y animarla y reiluminarla en el
navegador. Es la ruta de [Spark WebGPU](https://sparkwebgpu.lucab.co.uk/sparrow):
las gaussianas viajan en un contenedor `.athc`, se les aplica el skinning en
el navegador y las ilumina el HDR que elija quien mira. Una vez cargada la
nube, enviarla no cuesta nada más; el trabajo de posarla lo hace el
navegador.

## La geometría de un stream

Una nube posada o ajustada frame a frame se puede enviar como vídeo
(`athenea video-encode`, el formato `.athv`). El color y la opacidad viajan
en los planos del vídeo; la posición, la escala y la rotación, junto a él y
exactas, en una capa de geometría (`--geometry`). Entre keys de geometría
(`--geometry-key N`), cada gaussiana envía solo su diferencia con lo que
predice el decodificador, y la predicción es el movimiento:

| `--geometry-predict` | Predicción | Sirve para |
|---|---|---|
| `1` | el frame anterior | cualquier cosa con slots estables |
| `2` | el frame anterior con su movimiento prolongado (2b − c) | movimiento suave y sostenido |
| `3` | el movimiento rígido de cada tira de 256 gaussianas, ajustado a partir de lo que tiene el decodificador (32 bytes por tira), con la rotación enviada como un giro en el marco propio de la gaussiana | una nube con skinning: sus tiras se mueven como piezas rígidas |

La predicción necesita `--order stable`: la gaussiana *k* tiene que ser la
misma en todos los frames. `--geometry-stagger` refresca bloques de 256
gaussianas por turnos en vez de keys completos, y así mantiene plano el
bitrate. `--geometry-hold P,S,R` conserva el valor que ya tiene el
decodificador cuando el nuevo está a menos de esos pasos de él.

En el gorrión, el orden 3 envía un tercio menos de bytes que el orden 1 con
la misma imagen (PSNR 34,15 frente a 34,17 dB con 150.000 gaussianas, el
flicker sin cambios):

| Rung | Orden 1, Mbit/s a 30 fps | Orden 3 |
|---|---|---|
| 150k | 148,9 | 100,5 |
| 75k | 76,6 | 57,4 |
| 40k | 44,9 | 34,4 |
| 20k | 25,9 | 20,0 |
| 10k | 21,7 | 16,9 |

El [player web](https://atheneastream.lucab.co.uk/) decodifica los tres
órdenes. Los volúmenes simulados se quedan en el orden 1: sus gaussianas no
se mueven en tiras rígidas y el orden 3 no ahorra nada con ellas.

## Un volumen arrastrado por su flujo

Una secuencia de frames `.vdb` que trae un grid de velocidad se ajusta con
slots estables: cada frame parte de las gaussianas del anterior, arrastradas
por el flujo. El centro se mueve con la velocidad y el gradiente del flujo
deforma la covarianza (F = I + Δt ∇u, conservando su volumen; donde el flujo
cizalla demasiado, solo se arrastra el centro). Después, unos pocos pasos de
optimización corrigen el frame.

```sh
athenea vdb2splat "smoke_####.vdb" --sequence --frames 1 120 -n 20000 \
    --velocity velocity --velocity-units mantaflow --fps 24 -o "smoke/f_####.usdc"
```

`--velocity` nombra el grid (`vel` por defecto). Un grid escalonado (MAC),
como el de Mantaflow, se promedia a los centros de los vóxeles, y
`--velocity-units mantaflow` lo lee en las unidades de Mantaflow. Como los
slots son estables, el stream puede predecirlos (`--geometry-predict 1`, con
`--slots` para fijar el número). Un fichero sin velocidad, como los de
EmberGen, se ajusta frame a frame (`--independent`). Más en
[Volúmenes](volumes.md).

## Transforms de USD

Una nube es un prim de USD, así que sus `xformOp`s la mueven como mueven una
malla, con time samples o sin ellos. No se convierte nada: las gaussianas se
quedan donde están en el espacio propio de la nube y el transform se aplica
al dibujar el frame.

## Motion blur

Bajo un shutter (el de la cámara, `athenea:shutter` en los render settings, o
`athenea stage --shutter OPEN:CLOSE` para una cámara hecha con `--eye`), el
raster desenfoca cada gaussiana en una sola pasada. Una gaussiana arrastrada
a lo largo de un segmento d durante el shutter es esa gaussiana convolucionada
con el segmento, así que su covarianza 2D en pantalla gana un término de
rango uno, S' = S + d dᵀ / 12, y su pico baja para conservar la energía. El
desplazamiento sale del transform de la nube y de la cámara durante el
shutter (una sola diferencia, `V_close M_close − V_open M_open`) y, en una
nube con skinning, de lo que el esqueleto movió cada gaussiana. El gorrión,
con 9,5 millones de gaussianas, se dibuja en 70 ms quieto y en 164 ms con un
shutter de 180 grados (900 × 675).

El path tracer desenfoca las mallas cortando el shutter en buckets (de 1 a 8,
`--motion-buckets`); una nube la sigue dibujando en el instante del frame.

## Dónde están los detalles

- [`docs/operations.md` §2.8](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#28-athenea-mesh2splat--a-model-into-gaussians) (`--skinned`, `--range`, `--keep-order`), §2.9 (`athenea visibility`), [§2.7.2](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#272-athenea-video-encode--splat-frames-as-hardware-video) (la capa de geometría), [§2.8.3](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#283-athenea-vdb2splat--a-volume-into-volumetric-gaussians-and-back) (secuencias y velocidad).
- `docs/decisions.md`: «The whole Jacobian of a skinned gaussian», «A splat under a shutter: one rank-one term, not five passes», «Motion blur: the shutter in buckets»; `docs/operations.md` §8.2, el formato `.athv`.
