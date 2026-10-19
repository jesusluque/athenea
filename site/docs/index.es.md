<p class="kicker">athenea</p>

# Un baker de relight para 3D Gaussian splatting

athenea convierte assets USD de producción en Gaussian splats y hace el bake, en cada Gaussian, del transporte de luz que le da relight al dibujar. El resultado se juzga contra un path trace de la malla original con la misma cámara y la misma luz. El baker se apoya en un renderer de USD e Hydra en el que cada paso que toca datos de la escena es un kernel de cómputo en la GPU.

Estas páginas describen la versión de escritorio de la rama pública `main`: cómo se compila y cómo un asset USD se convierte en una nube con relight.

## Un motor de render para Blender

<figure markdown>
  ![El peón en Blender 5.3, dibujado por athenea bajo el HDRI del atardecer de Venecia](assets/blender-addon-venice.jpg)
  <figcaption>Blender 5.3 con el add-on de athenea: las Gaussians del peón reiluminadas por el HDRI del World (atardecer de Venecia, Poly Haven, CC0) en el viewport. Render Engine: Athenea.</figcaption>
</figure>

El add-on hace de athenea el motor de render de Blender: el viewport y F12 dibujan las Gaussians reiluminadas por el cielo del World, convierte las mallas en Gaussians reiluminables (mesh2splat con el transfer del path tracer), añade un shadow catcher para el suelo, exporta a SPZ, PLY, glTF y USD, y compara con Cycles desde el panel. Para Blender 5.3 en macOS, Linux y Windows: ver [Gaussian splats en Blender](tutorials/blender.md).

<p align="center"><img src="assets/pawn-venice.jpg" width="440" alt="F12 del peón bajo el atardecer de Venecia, dibujado por athenea en Blender"></p>

## El flujo

| Paso | Comando | Qué produce |
|---|---|---|
| 1. Conversión y bake | `athenea mesh2splat` | un `UsdVolParticleField3DGaussianSplat`: un Gaussian por celda, con su material y, con `--transfer`, cuánto le llega de un cielo desde cada dirección |
| 2. Relight y render | `athenea stage`, `athenea view` | la nube dibujada bajo las luces del stage, por raster o por path trace, con el mismo delegate de Hydra que carga un DCC |
| 3. Referencia | `athenea stage --technique rt` | las mallas originales con path trace desde la misma cámara |
| 4. Juicio | `athenea compare` | el error relativo HDR y la distribución de códigos de 8 bits entre las dos imágenes, medidos en la GPU |

<figure markdown>
  ![El peón del Open Chess Set como Gaussians con relight y como mallas con path trace](assets/pawn-gs-pt.png)
  <figcaption>El peón del Open Chess Set de la ASWF (CC BY 4.0): a la izquierda, su nube de Gaussians con relight al dibujar; a la derecha, las mallas con path trace. Misma cámara y luz. Véanse los <a href="tutorials/usd-to-cloud/">tutoriales</a> y los <a href="credits/">créditos</a>.</figcaption>
</figure>

## En movimiento

Cada clip abre el vídeo completo.

| | |
|:---:|:---:|
| [![La luz del path trace, en tiempo real](assets/clip-relight.webp)](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/relight-reel.mp4) | [![El Corvette entero en un iPhone](assets/clip-iphone.webp)](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/corvette-iphone.mp4) |
| La luz del path trace, en tiempo real | El Corvette entero en un iPhone |
| [![El cielo gira, el mismo fichero lo sigue](assets/clip-corvette-sky.webp)](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/corvette-sky-turns.mp4) | [![Un gorrión con skinning en vuelo](assets/clip-sparrow.webp)](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/sparrow-flight.mp4) |
| El cielo gira, el mismo fichero lo sigue | Un gorrión con skinning en vuelo |

## Por dónde empezar

- [Instalación de escritorio](install.md): requisitos, los presets de CMake y el entorno de ejecución en macOS (Apple Silicon, Metal) y Linux (x86_64, CUDA).
- [De un asset USD a una nube con bake](tutorials/usd-to-cloud.md): un peón del Open Chess Set de la ASWF, convertido con un bake de transfer.
- [Render y comparación con el path trace](tutorials/render-compare.md): el mismo peón dibujado como Gaussians y como mallas, y medido.
- [Preparar un stage USD](tutorials/authoring-usd.md): rutas del plugin, materiales, luces, render settings y los schemas que lee el motor.
- [Nubes grandes y niveles de detalle](tutorials/large-clouds.md): niveles, decimación y streaming dentro de un presupuesto de GPU.

Las tablas completas de opciones están en [`docs/operations.es.md`](https://github.com/jesusluque/athenea/blob/main/docs/operations.es.md), que estas páginas siguen.
