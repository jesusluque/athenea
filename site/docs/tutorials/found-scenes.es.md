# Una escena encontrada al renderizar

!!! example "En vivo"
    Escenas hechas así, mostradas según crecieron frame a frame: [neural.lucab.co.uk](https://neural.lucab.co.uk).

Un path tracer suele dejar imágenes. `athenea neural-scene shoot` deja la escena: cada frame que renderiza añade las gaussianas que ese frame encontró, cada una con su material, su normal y la luz bajo la que se encontró, puestas sobre las gaussianas más gruesas que había dejado una mirada más lejana. Lo que queda al final se dibuja desde cualquier vista, se corta a un nivel de detalle, se reilumina por grupos de luces y se exporta para un navegador.

## El recorrido

Un fichero de vistas, una por línea: un nombre, el ojo y lo que mira, y opcionalmente el vector up.

```text
k00 150.0 0.0 175.0 1150.0 0.0 80.0 0 0 1
k01 172.7 4.3 174.7 1150.0 0.0 80.0 0 0 1
k02 195.4 8.5 174.3 1150.0 0.0 80.0 0 0 1
```

Lo que un frame ya tiene no se vuelve a encontrar: una vista posterior añade solo lo que ve nuevo o ve más fino.

## El shoot

```sh
athenea neural-scene shoot sponza_knight.usda knight.aths --views walk.txt \
    --focal 24 --size 960x540 --samples 32 --pass-samples 8 --bounces 3 \
    --denoise --ray-margin 2e-5 --inside-turned 0.1
```

`knight.aths` es un directorio: un manifiesto (`scene.json`) y una pieza por cada mirada, con sus gaussianas en un `.athg`. El Sponza de Intel con el caballo del Open Chess Set de la ASWF en vidrio, 60 vistas a 960 × 540 en un Apple M5 Pro, son **2 362 750 gaussianas en pie tras 414,5 s** de path tracing. `--inside-turned 0.1` hace que una gaussiana dentro de una superficie sea tan larga como su píxel sobre esa superficie, que es lo que evita que un suelo visto en rasante salga rayado a trazos en un nivel de detalle grueso; está apagado por defecto porque las vistas apartadas del recorrido miden entre 0,03 y 0,36 dB menos con él.

Un shoot sobre una escena que ya tiene piezas añade a ella. `--light-group NOMBRE` con `--lights-on` hace el shoot solo de la parte de esas luces sobre las mismas gaussianas, y así la escena se puede reiluminar por grupos después.

## Dibujarla

```sh
athenea neural-scene render knight.aths --eye 834 66 162 --target 1150 0 80 \
    --focal 24 --size 1920x1080 --glass -o view.exr
```

La vista no tiene por qué ser una del recorrido. `--glass` sombrea lo que se encontró como vidrio o como espejo según el ojo que mira ahora, no como la imagen con la que se encontró. `--budget 1000000` dibuja el nivel de detalle que cabe en esas gaussianas. Cuatro vistas apartadas del recorrido miden de 31,7 a 34,2 dB contra los frames del path tracer de las mismas vistas.

## Para la web

```sh
athenea neural-scene export knight.aths --by-frame --history \
    --levels 1000000,500000,250000 -o web/scenes/sponza_knight
```

Esto escribe partes SPZ, una por frame renderizado, y un índice `sponza_knight.json`; los niveles salieron de 913 434, 495 348 y 248 286 gaussianas. El player es una página, [`scripts/examples/neural-scene-viewer/index.html`](https://github.com/jesusluque/athenea/tree/main/scripts/examples/neural-scene-viewer), servida junto al directorio `scenes/`, a la que su dirección le dice qué mostrar: `?scene=sponza_knight`, `&detail=auto`, `&ride=1` para ir montado en la cámara del render. Dibuja primero el nivel más grueso y el detalle detrás, y el nivel que un dispositivo aguanta: un millón de gaussianas en un móvil.

## Dónde leer más

- Todas las opciones de `shoot`, `render` y `export`, con su valor por defecto y su unidad: [operations §2.3.4](https://github.com/jesusluque/athenea/blob/main/docs/operations.es.md).
- Lo que dice la dirección del player: [operations §2.3.5](https://github.com/jesusluque/athenea/blob/main/docs/operations.es.md).
- El directorio `.aths` byte a byte: [formats §6](https://github.com/jesusluque/athenea/blob/main/docs/formats.es.md).
- Por qué está hecho como está, con las medidas: [`docs/decisions.md`](https://github.com/jesusluque/athenea/blob/main/docs/decisions.md).
