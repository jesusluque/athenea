# Render y comparación con el path trace

La nube del [tutorial anterior](usd-to-cloud.md) se dibuja bajo el domo y el sol de la escena, y las mallas originales se renderizan con path trace desde la misma cámara y con la misma luz. Después `athenea compare` mide la primera imagen frente a la segunda en la GPU. Las cifras de esta página salen de una ejecución en un Apple M5 Pro (Metal) con la rama pública `main`.

## La nube con relight

```sh
athenea stage gs.usda --camera /World/Camera -o gs.exr
```

`athenea stage` dibuja con el mismo delegate de Hydra que carga un DCC. Con la técnica por defecto, `--technique raster`, los splats de la nube se rasterizan; el transfer de cada Gaussian se combina con el cielo que pone el frame (el `DomeLight`) y con el sol. El frame es de 1920 × 1080 por defecto y tardó **2 s**, compilación de shaders incluida. La salida es un EXR RGBA de media precisión con un canal `Z`.

## La referencia

```sh
athenea stage mesh.usda --camera /World/Camera --technique rt \
    --path-total 1024 --denoise -o pt.exr
```

`--technique rt` renderiza las mallas con path trace y sus materiales; `--path-total 1024` dibuja hasta que cada píxel tiene 1024 caminos, y `--denoise` pasa Open Image Denoise al llegar a ellos. Este frame tardó **155 s**.

<figure markdown>
  ![El peón como Gaussians con relight (izquierda) y como mallas con path trace (derecha)](../assets/pawn-gs-pt.png)
  <figcaption>Izquierda: la nube, con relight al dibujar. Derecha: las mallas, con path trace de 1024 caminos por píxel. Misma cámara, domo y sol.</figcaption>
</figure>

## La medida

```sh
athenea compare gs.exr pt.exr
```

```text
image     mean 0.736384 0.794351 0.921122 1   max 12.2266 8.60938 3.81445 1   (2073600 pixels)
reference mean 0.732241 0.790268 0.916906 1   max 11.2812 10.9688 10.1328 1   (2073600 pixels)
hdr       relMSE 0.043379   p99 relative 2.181   max relative 279.2
8-bit     p99 60   max 225   over 2: 164522 of 2073600 pixels
```

| Línea | Qué dice |
|---|---|
| `image`, `reference` | la media y el valor más alto de cada canal del frame de la nube y del path trace |
| `hdr` | el error relativo HDR de la primera frente a la segunda: `relMSE`, su percentil 99 y su valor más alto |
| `8-bit` | la diferencia en códigos sRGB de 8 bits: el percentil 99, la mayor y los píxeles que difieren en más de 2 |

La mayor parte de este frame es el cielo detrás del peón, así que las medias sobre el frame entero quedan cerca. `--window X0 Y0 X1 Y1` limita las medias a una región (filas contadas desde abajo); las diferencias son siempre sobre la imagen entera. `--heatmap` escribe la imagen del efecto Measure con dónde difieren las dos:

```sh
athenea compare gs.exr pt.exr --window 700 140 1220 680        # el cuerpo
athenea compare gs.exr pt.exr --heatmap rel.exr --show relative --gain 4
```

## Lectura de la comparación

El cuerpo de mármol, sus vetas, el anillo dorado y los brillos están donde los tiene el path trace. La cabeza de vidrio es la diferencia visible: dibujada como Gaussians se ve más clara y menos saturada que el vidrio sólido con path trace. Un sólido que transmite por completo se convierte con la cobertura que le da `--glass-opacity` (0,6 por defecto), y en `main` sus Gaussians no refractan la escena de detrás como lo hace el vidrio trazado.

Los mismos cuatro pasos sirven para cualquier asset: una capa con la cámara y las luces, las mallas y la nube referenciadas cada una bajo ella, una conversión con la celda de la cámara, y los dos frames medidos uno frente al otro región a región.

## Referencia

[operations.es.md](https://github.com/jesusluque/athenea/blob/main/docs/operations.es.md): §2.5 `athenea stage`, §2.11 `athenea compare` y §3.2, render de una escena.
