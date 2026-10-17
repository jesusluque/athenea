# De un asset USD a una nube con bake

Este tutorial convierte el peón del Open Chess Set de la ASWF en una nube de Gaussians con un bake de transfer: para cada Gaussian, cuánto le llega de un cielo desde cada dirección. Los comandos, los logs y las cifras de esta página salen de una ejecución en un Apple M5 Pro (Metal) con la rama pública `main`. El [siguiente tutorial](render-compare.md) hace el relight de la nube y la compara con un path trace de las mallas.

## El asset

El Open Chess Set es un asset USD de producción con materiales MaterialX (Autodesk Standard Surface), con licencia CC BY 4.0 de la Academy Software Foundation ([créditos](../credits.md)). Del repositorio `usd-wg/assets` solo hace falta su carpeta, unos 37 MB:

```sh
git clone --depth 1 --filter=blob:none --sparse https://github.com/usd-wg/assets.git aswf-assets
git -C aswf-assets sparse-checkout set full_assets/OpenChessSet
```

El peón es `full_assets/OpenChessSet/assets/Pawn/Pawn.usd`: un `Xform` con dos mallas, `Geom_Body/Render` (mármol) y `Geom_Top/Render` (una cabeza de vidrio sobre un anillo dorado). El stage está en metros con Y hacia arriba; el peón mide unos 4,2 cm de ancho y 8,1 cm de alto.

## Tres capas

La conversión y la comparación comparten una cámara y una luz, así que viven en una capa propia, `shot.usda`. Dos capas la incluyen como sublayer: `mesh.usda` referencia las mallas del asset y `gs.usda` referencia la nube que escribe la conversión.

```usda title="shot.usda"
#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "World"
{
    def Camera "Camera"
    {
        matrix4d xformOp:transform = ( (0.792624, 0.000000, -0.609711, 0.000000), (-0.091876, 0.988581, -0.119438, 0.000000), (0.602749, 0.150687, 0.783573, 0.000000), (0.200000, 0.095000, 0.260000, 1.000000) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
        float focalLength = 50
        float horizontalAperture = 24.576
        float verticalAperture = 13.824
        float2 clippingRange = (0.01, 100)
    }

    def DomeLight "Sky"
    {
        color3f inputs:color = (0.8, 0.86, 1.0)
        float inputs:intensity = 1
    }

    def DistantLight "Sun"
    {
        float inputs:angle = 1.0
        color3f inputs:color = (1.0, 0.95, 0.88)
        float inputs:intensity = 3
        bool inputs:normalize = 1
        float3 xformOp:rotateXYZ = (-45, 35, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }
}
```

La cámara está en (0,20, 0,095, 0,26) m y mira al centro del peón, (0, 0,045, 0). `inputs:normalize = 1` hace que la intensidad del sol sea la iluminancia sobre una superficie orientada hacia él, en lux. Sin él, UsdLux lee la intensidad como la radiancia del disco del sol, y un sol de intensidad 3 con un ángulo de 1° no ilumina casi nada ([operations §3.2](https://github.com/jesusluque/athenea/blob/main/docs/operations.es.md)).

```usda title="mesh.usda"
#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Y"
    subLayers = [@./shot.usda@]
)

over "World"
{
    def "Pawn" (
        references = @../aswf-assets/full_assets/OpenChessSet/assets/Pawn/Pawn.usd@
    )
    {
    }
}
```

`gs.usda` es el mismo fichero con `references = @./pawn_gs.usdc@`.

!!! note "Ficheros"
    Las tres capas y los cuatro pasos de estos tutoriales: [shot.usda](files/shot.usda), [mesh.usda](files/mesh.usda), [gs.usda](files/gs.usda) y [run.sh](files/run.sh). `run.sh` espera `aswf-assets` en la carpeta superior.


## La conversión y el bake

```sh
athenea mesh2splat mesh.usda --prim /World/Pawn \
    --cell-from-camera /World/Camera --camera-pixels 1024 \
    --transfer --bake-samples 128 -o pawn_gs.usdc
```

| Opción | Qué hace aquí |
|---|---|
| `--prim /World/Pawn` | convierte solo las mallas del peón y las que cuelgan de él |
| `--cell-from-camera /World/Camera` | la celda de cada malla es lo que cubre un píxel de esa cámara donde la caja de la malla está más cerca de ella |
| `--camera-pixels 1024` | píxeles a lo ancho de la apertura horizontal de la cámara para esa regla |
| `--transfer` | hace el bake, para cada Gaussian, de cuánto de un entorno llega desde cada dirección, en lugar de la luz que llegó |
| `--bake-samples 128` | caminos que toma primero cada Gaussian; un transfer añade `--bake-extra` (128 por defecto) de media donde la primera pasada tuvo más ruido |

El log indica la celda, los materiales de cada malla, las Gaussians por malla y el bake:

```text
mesh2splat: the cell is a pixel of /World/Camera, 1024 across
mesh2splat: 4 of 4 textures decoded
mesh2splat: /World/Pawn/Geom_Top/Render uses /World/Pawn/Materials/M_Pawn_Top_B (colour 1.00 1.00 1.00, albedo '', metallic 0.00, roughness 1.00, transmission 1.000, opacity 1.000)
mesh2splat: /World/Pawn/Geom_Top/Render -> 55031 splats of 55031 wanted (2208 triangles, cell 0.0001503)
mesh2splat: /World/Pawn/Geom_Body/Render -> 271234 splats of 271234 wanted (42892 triangles, cell 0.000144)
mesh2splat: transfer baked for 326233 of 326265 gaussians (256 paths each, 3 bounces, with the indirect half) in 98133 ms
mesh2splat: wrote pawn_gs.usdc (326265 splats)
```

| Resultado | Valor |
|---|---|
| Gaussians | 326 265 (55 031 en la cabeza de vidrio, 271 234 en el cuerpo) |
| Celda | 0,15 mm en la cabeza, 0,14 mm en el cuerpo |
| Bake | 256 caminos por Gaussian, 3 rebotes, mitades directa e indirecta, 98 s |
| Comando completo | 106 s |
| `pawn_gs.usdc` | 70 MB |

## Qué contiene la nube

La salida es un `UsdVolParticleField3DGaussianSplat` bajo `/World/Pawn/Splats`: posiciones, orientaciones, escalas, opacidades y colores, el material (metallic, roughness, transmisión), las normales de shading y el transfer. El transfer no lleva color ni cielo: es geometría, así que el mismo fichero es correcto bajo cualquier entorno que ponga el frame. La mitad directa son nueve floats por Gaussian; la indirecta, la luz que llegó tras rebotar en la escena, añade veintisiete, y `--no-indirect` la deja fuera.

La cabeza de vidrio es un sólido que transmite por completo (`transmission 1.000`). Un sólido que transmite por completo conserva la cobertura que da `--glass-opacity` (0,6 por defecto).

## Qué cambia el resultado

| Opción | Efecto |
|---|---|
| `--camera-pixels` | la celda: el doble de píxeles, la mitad de celda y unas cuatro veces las Gaussians |
| `--bake-samples`, `--bake-extra` | el grano del bake; los caminos extra van donde la primera pasada tuvo más ruido para lo que cuestan sus caminos |
| `--bake-filter` (3 por defecto) | iteraciones à-trous del bundle `SplatBakeFilter` sobre la luz del bake, entre Gaussians vecinas de un prim; 0 no filtra nada |
| `--bake-bounces` (3 por defecto) | rebotes tras el primer impacto |
| `--no-indirect` | conserva solo la mitad directa del transfer |
| `--no-bake` | lleva el material para hacer relight en lugar de hacer el bake |

Todas las opciones de la conversión están en [operations §2.8](https://github.com/jesusluque/athenea/blob/main/docs/operations.es.md) y las decisiones que las explican, en el §3.1 del mismo documento.
