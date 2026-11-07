
## Varias vistas con una sola carga del stage (7 nov)

`athenea stage` dibujaba una cámara por proceso, así que un turntable pagaba
la carga del stage en cada frame: el USD, sus texturas, la cúpula, la
compilación de los kernels y las estructuras de aceleración. splat2mesh lo
midió en el enano (80 000 caras, cinco texturas de 4096², una cúpula HDRI):
45,8 s por frame de 1080² en raster, 107 s path traced a 96 caminos por píxel,
casi todo carga. `--cameras FICHERO` y `--turntable N` dibujan todas las
vistas desde un mismo `StageRenderer` y escriben un EXR por vista.

**Medido** en el enano, M5 Pro, build release:

| | tres vistas, 540², rt, 32 caminos, con denoise | 24 vistas, 1080², raster |
|---|---|---|
| un proceso por vista | 29,4 s (9,8 s por vista) | 45,8 s por vista (la cifra de splat2mesh) |
| un solo proceso | 16,1 s: primera vista 5,9 s, después 3,4 s | 18,5 s en total: primera vista 3,0 s, después 0,56 s de mediana |

Una vista dibujada así es la que dibuja un proceso propio: el mismo ojo por
`--eye` y por `--cameras` da la misma imagen raster, y la primera vista path
traced es idéntica; las siguientes solo difieren por la semilla de los caminos
(media 4 × 10⁻⁴), porque la semilla sigue corriendo entre los frames de un
proceso.
