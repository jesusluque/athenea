# Gaussian splats en Blender

Este tutorial lleva una escena de Blender de mallas a Gaussian splats con el
add-on de Athenea: el cubo por defecto sobre un suelo bajo un HDRI,
convertido con light transfer y shadow catcher, dibujado por el render
engine **Athenea** en el viewport y con F12, y exportado como un fichero de
splats estándar. Las capturas son de Blender 5.3 alpha en un Mac (Metal).

## Qué necesita

<figure markdown>
  ![Blender 5.3 con el add-on de athenea: el viewport dibujado por Athenea bajo el atardecer de Venecia, y su panel en las propiedades de render.](../assets/blender-addon-venice.jpg)
  <figcaption>Blender 5.3 con el add-on de athenea: el viewport dibujado por Athenea bajo el atardecer de Venecia, y su panel en las propiedades de render.</figcaption>
</figure>

- La development build de Blender 5.3. El add-on rechaza un Blender cuyo USD
  no es aquel contra el que se compiló hdAthenea, y dice cuál quiere.
- El paquete del add-on, `athenea_hydra-<versión>.zip`, que hace
  `scripts/package-blender-addon.sh` a partir de un build
  `macos-arm64-blender`. Se instala con *Edit > Preferences > Get Extensions
  > Install from Disk*.
- Un `.blend` guardado: la conversión escribe sus ficheros a su lado, en
  `//splats/`.

## 1. La escena y el render engine

Vale cualquier escena con mallas, una cámara y un world. Aquí: el cubo por
defecto (Principled BSDF naranja), un plano de 12 m debajo como suelo, y el
world una *Environment Texture* con el `forest.exr` de Blender. En *Render
Properties*, **Render Engine** a **Athenea**.

El world llega a Athenea como dome light cuando es una *Environment
Texture*; un cielo procedural aún no se traduce. Con *Film > Transparent*
activado, el world ilumina la escena y no se dibuja detrás, como en Cycles.

## 2. El panel de conversión

Selecciona las mallas que se convierten (el cubo) y abre *Sidebar (N) >
Athenea > Gaussian Splats*.

![La escena y el panel de Athenea antes de convertir](../assets/blender/1_before.png)

| Ajuste | Qué hace |
|---|---|
| Quality | *Preview*: 256 celdas de lado, como mucho un millón de gaussianas, 64 paths por gaussiana. *Final*: cada malla a su propia densidad, 512 celdas, como mucho 8 millones, 256 paths. |
| Shadow catcher | convierte también la sombra que la selección proyecta sobre el suelo que tiene debajo, que encuentra sola (la malla plana más grande en su base). |
| Result | *Relit cloud*: la nube tal como se convirtió, reiluminada en vivo por Athenea con el world y las luces de la escena. *Blender splats*: aplanada con el world actual en un fichero de splats estándar e importada como la nube de puntos Gaussian-splat propia de Blender, que dibuja cualquier engine, iluminada una vez. |
| Directory | dónde se escriben las mallas exportadas y las nubes (`//splats/` por defecto; en rojo mientras no existe). |
| Hide the meshes | oculta las mallas convertidas, y el suelo bajo un catcher, cuando entran las nubes. |
| Overrides | el presupuesto de gaussianas, el entorno que ve el bake, vidrio fino y sólido por material. La conversión los decide sola; un override es solo para cuando se equivoca. |

Pulsa **Convert to Gaussian Splats**. El panel muestra el progreso y Esc la
detiene. El cubo tardó aquí 35 s en un M5 Pro en *Preview*.

## 3. El resultado en el viewport

La nube y su catcher entran como Empties en una colección `Cube_splats` que
referencian los ficheros `.usdc`; el cubo y el suelo quedan ocultos. Pon el
viewport en shading *Rendered*: Athenea dibuja las gaussianas, reiluminadas
con el world, con la sombra del catcher en el suelo.

![El cubo convertido en el viewport, dibujado por Athenea](../assets/blender/2_converted.png)

## 4. El render final

F12 renderiza el encuadre de la cámara con lo que diga *Render Properties >
Athenea*:

| Ajuste | Qué hace |
|---|---|
| Final render | *Raster* (las gaussianas rasterizadas, como en el viewport) o *Path traced*. |
| Paths | paths por píxel en un render path traced (256). |
| Denoise | OIDN sobre el frame path traced. |
| Bounced light | la mitad indirecta del transfer (activado). |

Los pases son Combined, Depth, Normal y DiffCol.

![F12 con el raster](../assets/blender/3_f12.png)

**Compare with Cycles**, en el mismo panel, renderiza el encuadre con Cycles
a partir de las mallas originales (el suelo como shadow catcher de Cycles) y
con Athenea a partir de las nubes, y mide la diferencia: relMSE y el p99 en
8 bits en el panel, y las dos imágenes y un heatmap en Blender. En esta
escena: relMSE 0.036, p99 45.

## 5. Exportar

*File > Export > Gaussian Splats (Athenea: .spz, .ply, .glb)*, o **Export
Gaussian Splats** en el panel, escribe las nubes de la escena aplanadas con
el world actual en un fichero estándar que leen otros visores: SPZ (1.65 MB
el cubo), PLY o glTF.

## Para seguir

- El manual de operaciones, sección 4.1.1 *Dentro de Blender*: cada ajuste,
  variable y pase.
- [De un asset USD a una nube con bake](usd-to-cloud.md): la misma conversión
  desde la línea de comandos.
