<p class="kicker">athenea</p>

# Un baker de relight para 3D Gaussian splatting

athenea convierte assets USD de producción en Gaussian splats y hace el bake, en cada Gaussian, del transporte de luz que le da relight al dibujar. El resultado se juzga contra un path trace de la malla original con la misma cámara y la misma luz. El baker se apoya en un renderer de USD e Hydra en el que cada paso que toca datos de la escena es un kernel de cómputo en la GPU.

Estas páginas describen la versión de escritorio de la rama pública `main`: cómo se compila y cómo un asset USD se convierte en una nube con relight.

## El flujo

| Paso | Comando | Qué produce |
|---|---|---|
| 1. Conversión y bake | `athenea mesh2splat` | un `UsdVolParticleField3DGaussianSplat`: un Gaussian por celda, con su material y, con `--transfer`, cuánto le llega de un cielo desde cada dirección |
| 2. Relight y render | `athenea stage`, `athenea view` | la nube dibujada bajo las luces de la escena, por raster o por path trace, con el mismo delegate de Hydra que carga un DCC |
| 3. Referencia | `athenea stage --technique rt` | las mallas originales con path trace desde la misma cámara |
| 4. Juicio | `athenea compare` | el error relativo HDR y la distribución de códigos de 8 bits entre las dos imágenes, medidos en la GPU |

<figure markdown>
  ![El gorrión: malla y nube de Gaussians](assets/sparrow-mesh-cloud.png)
  <figcaption>El gorrión del repositorio: la malla original y su nube de Gaussians.</figcaption>
</figure>

## Por dónde empezar

- [Instalación de escritorio](install.md): requisitos, los presets de CMake y el entorno de ejecución en macOS (Apple Silicon, Metal) y Linux (x86_64, CUDA).
- [De un asset USD a una nube con bake](tutorials/usd-to-cloud.md): un peón del Open Chess Set de la ASWF, convertido con un bake de transfer.
- [Render y comparación con el path trace](tutorials/render-compare.md): el mismo peón dibujado como Gaussians y como mallas, y medido.
- [Preparar una escena USD](tutorials/authoring-usd.md): rutas del plugin, materiales, luces, render settings y los schemas que lee el motor.
- [Nubes grandes y niveles de detalle](tutorials/large-clouds.md): niveles, decimación y streaming dentro de un presupuesto de GPU.

Las tablas completas de opciones están en [`docs/operations.es.md`](https://github.com/jesusluque/athenea/blob/main/docs/operations.es.md), que estas páginas siguen.
