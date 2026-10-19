# Render y comparación con el path trace

!!! tip "El peón de ejemplo"
    Los ficheros que compara este tutorial, listos para usar: [athenea-example-pawn.zip](https://github.com/jesusluque/athenea/releases/download/v0.3.0/athenea-example-pawn.zip). `pawn_gs.usda` es la nube y `pawn_mesh.usda` las mallas, con la misma cámara (`/World/Cam`) y el mismo cielo; a 1024 × 1024 con una referencia de 1024 caminos miden PSNR 35,7 dB.

!!! example "En vivo"
    El mismo peón, reiluminado en el navegador: [sparkwebgpu.lucab.co.uk/pawn.html](https://sparkwebgpu.lucab.co.uk/pawn.html).

La nube del [tutorial anterior](usd-to-cloud.md) se dibuja bajo el domo y el sol del stage, y las mallas originales se renderizan con path trace desde la misma cámara y con la misma luz. Después `athenea compare` mide la primera imagen frente a la segunda en la GPU. Las cifras de esta página salen de una ejecución en un Apple M5 Pro (Metal) con la rama pública `main`.

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
  ![El peón como Gaussians con relight (izquierda) y como mallas con path trace (derecha)](../assets/pawn-gs-pt.jpg)
  <figcaption>Izquierda: la nube, con relight al dibujar. Derecha: las mallas, con path trace de 1024 caminos por píxel y denoise. Misma cámara y mismo cielo (el taller), 1024 × 1024: la escena del <a href="https://github.com/jesusluque/athenea/releases/download/v0.3.0/athenea-example-pawn.zip">paquete de ejemplo</a>.</figcaption>
</figure>

## La medida

```sh
athenea compare gs.exr pt.exr
```

```text
image     mean 0.283718 0.288465 0.266441 1   max 7.66797 8.38281 6.72266 1   (1048576 pixels)
reference mean 0.283444 0.288235 0.266429 1   max 8.14844 11.3281 7.64453 1   (1048576 pixels)
hdr       relMSE 0.00378711   p99 relative 0.3856   max relative 117.4
8-bit     p99 18   max 128   over 2: 223380 of 1048576 pixels
display   PSNR 35.73 dB   SSIM 0.9700
```

| Línea | Qué dice |
|---|---|
| `image`, `reference` | la media y el valor más alto de cada canal del frame de la nube y del path trace |
| `hdr` | el error relativo HDR de la primera frente a la segunda: `relMSE`, su percentil 99 y su valor más alto |
| `8-bit` | la diferencia en códigos sRGB de 8 bits: el percentil 99, la mayor y los píxeles que difieren en más de 2 |
| `display` | el PSNR y el SSIM de las dos imágenes como se ven en pantalla |

La mayor parte de este frame es el taller detrás del peón, así que las medias sobre el frame entero quedan cerca. `--window X0 Y0 X1 Y1` limita las medias a una región (filas contadas desde abajo); las diferencias son siempre sobre la imagen entera. `--heatmap` escribe la imagen del efecto Measure con dónde difieren las dos:

```sh
athenea compare gs.exr pt.exr --window 230 150 470 470        # el cuerpo
athenea compare gs.exr pt.exr --heatmap rel.exr --show relative --gain 4
```

## Lectura de la comparación

El cuerpo de mármol, sus vetas, el anillo dorado, los brillos y la cabeza de vidrio, con el taller refractado e invertido dentro, están donde los tiene el path trace: relMSE 0,0038, PSNR 35,7 dB y SSIM 0,97 sobre el frame entero. Lo que queda está en los bordes y en los brillos más pequeños (percentil 99 de 18 códigos de 8 bits), donde una Gaussian cubre una región y el path trace resuelve el píxel.

Los mismos cuatro pasos sirven para cualquier asset: una capa con la cámara y las luces, las mallas y la nube referenciadas cada una bajo ella, una conversión con la celda de la cámara, y los dos frames medidos uno frente al otro región a región.

## Referencia

[operations.es.md](https://github.com/jesusluque/athenea/blob/main/docs/operations.es.md): §2.5 `athenea stage`, §2.11 `athenea compare` y §3.2, render de una escena.
