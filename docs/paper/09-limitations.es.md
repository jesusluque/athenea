<!-- Copyright (c) 2026 jesus luque. -->

# 9. Limitaciones y trabajo futuro

Las limitaciones de abajo se reúnen de las listas de las propias secciones (§2.16, §3.12, §4.12, §5.13, §6.11, §7.13)
y se agrupan por dónde actúan, primero el bake con relight y al final la aplicación a móvil. Cada una se enuncia con la
cifra que la mide, cuando la hay.

## 9.1 Defectos que quedan en el Corvette

Esto es lo que el coche de 1280 px con relight aún muestra frente al path trace.

| defecto | medido | causa | remedio propuesto | § |
|---|---|---|---|---|
| faros oscuros | faros oscuros frente al path trace | el bake del transfer no lleva la emisión de las lámparas | hacer el bake de la emisión de las lámparas y de su luz sobre el coche | 3.12, 4.12 |
| tira cromada del parachoques delantero | un 38 % más clara que el path trace, con el pico más bajo | un tubo de unos pocos píxeles de ancho: cada Gaussian lee su propio espejo con una sola normal | — (abierto) | 3.12, 4.12 |
| reflejo del suelo cercano en la llanta | 0.72 del path trace | — (no analizado más) | — (abierto) | 4.12 |
| reflejos abiertos de la puerta | 1.13 del path trace; el 68 % del exceso donde el espejo da contra el coche | cells de 16 × 16 de unos 11° dejan pasar cielo detrás de un retrovisor o un pilar | cells de 32 × 32 (+96 B por Gaussian) lo reducirían más o menos a la mitad | 4.10.4, 4.12 |
| brillos del capó | a 2.0 px de su posición en el path trace | el giro por píxel extrapola el shape operator del triángulo central | una curvatura ajustada sobre el footprint con muestras del mesh ($p_{90}$ 9.8° frente a 46°) lo reduciría más o menos a la mitad | 4.10.6, 4.12 |
| labio inferior del parachoques | irregular | el footprint de una cell por Gaussian en un borde enrollado | — (abierto) | 3.12 |
| huecos de la grille en los cuts gruesos | 2.24 de las leaves a 640 px, 1.04 a 1280 px | merges de tiras de menos de un píxel; el hueco es una propiedad de la proyección | leaves más finas (hecho a 1280 px); ninguna medida por nodo independiente de la vista lo ve | 6.5, 6.11 |
| molduras finas lejanas | trazos gruesos en todos los cuts | las cotas de error no ven este error | — (abierto) | 6.11 |

## 9.2 La representación y el transfer

- **Campo de rango uno.** El acoplamiento del reflected field al cielo pone en un reflejo el color de un suelo
  iluminado, no su forma, bajo un cielo iluminado desde un lado; con grado 3 resuelve un horizonte a unos veinte grados
  (correlación 0.56 con grado 3, 0.63 con grado 5, 0.85 con un mapa octaédrico de 32 × 32, en simulación) (§4.12).
- **Luces dentro de la escena.** Se extrae un sol por dome. Las lámparas dentro de una escena se sombrean con bits que
  dicen qué direcciones salen de la escena, no cuáles llegan a la lámpara (§4.12).
- **Una normal, un footprint.** Una Gaussian tiene una normal y el footprint de una cell; las tiras finas curvadas y
  los bordes enrollados son donde se nota (§9.1). La correspondencia de coverage supone el solapamiento $2\pi\sigma^2$
  de la lattice de cells, que no tienen los triángulos menores que una cell ni las Gaussians sujetas por bordes
  (§3.12).
- **Vidrio.** Un vidrio fino tintado se convierte como sólido, y una nube conserva un único índice de refracción salvo
  que lleve capas (§3.12). Las Gaussians de lámina fina se suben a un suelo de opacidad de 0.1 por Gaussian, así que
  hacer merge de menos láminas y más grandes hace variar la transmisión de cell a cell; queda pendiente un suelo por
  píxel (§6.11).
- **Tamaño en el device.** La memoria del device limita una pieza a unos 5 M leaves para el merge tree; el
  tope de 4.5 M hizo que las piezas más grandes se convirtieran a una resolución de cámara más gruesa (Bottom a
  3392 px, Part_3 y Front_Bumper_001 a 4480 px; §3.11, §6.9).
- **Instanciado.** Los point instancers anidados solo se expanden en el nivel más interno, y el movimiento creado como
  velocidades se lee en la muestra (§3.12).

## 9.3 Assets animados

- Cuatro influencias por Gaussian truncan las puntas de las alas del gorrión (el 27 % de los vértices del ala llevan
  más); la proporción de Gaussians truncadas no se ha medido (§5.3, §5.13).
- No hay skinning por cuaterniones duales para las nubes, ni retargeting, ni mezcla de clips; de varios meshes con
  skinning con distintas transformaciones de prim solo se conserva la del primero (§5.13).
- La ruta trazada posa una vez por frame, así que no desenfoca una nube con skinning, y las instancias de LoD no llevan
  movimiento (§5.13).
- La visibilidad por parte deja fuera el autosombreado de la propia parte, supone partes rígidas que ocluyen de forma
  independiente, y no se ha hecho su bake por encima de unos 3 M Gaussians por parte; los transfers zonales conservan
  solo la direct half y el autosombreado de la bind pose (§4.12, §5.8, §5.13).
- El error del ajuste zonal en el gorrión, y una medida geométrica de la coverage en una flexión cerrada, no se han
  medido (§5.13).

## 9.4 El motor

- Los splats todavía no están dentro de los rayos secundarios del path tracer: con meshes, los splats se componen por
  profundidad sobre las superficies del path trace y proyectan sombras sobre ellas, pero un rayo de bounce no los ve
  (§2.16).
- El blur de movimiento y de lente del rasterizer son Gaussians ajustadas por momentos, hasta un 38 % demasiado
  picudas para estelas largas (§2.16).
- Las Gaussians de unas centésimas de píxel desaparecen del raster; el LOD es el remedio (§2.16).
- La visibilidad de meshes por raster paga unos 8 ms de tiempo de host por cada 1800 draws; la compilación de
  materiales es síncrona; la TLAS se reconstruye cada frame; la distorsión de lente más allá de $k_1, k_2$ no se aplica
  (§2.16).
- La ruta trazada lee el transfer de una nube girada frente a un cielo que se queda en el sistema del mundo (§4.12).
- **Hosts.** En Nuke un nodo OpenFX da una capa por nodo, porque Nuke solo acepta de él sus propios planos; el manifest
  del Cryptomatte se queda en el plugin, porque OpenFX no lleva metadatos, así que un matte se elige por id; la cámara
  son knobs enlazados y no la cámara del propio host; las Gaussians no llevan normales ni vectores de movimiento como
  capas. El add-on de Blender está construido sobre un motor anterior y solo funciona en macOS (§2.14).

## 9.5 LOD, streaming y el player

- **Ponderación de la oclusión.** El codebook compartido se reparte por entradas usadas, no por la frecuencia con que
  una vista descarta un nodo; ponderar por una órbita de vistas en el bake debería llevar el cull por encima del 84 %
  del ideal (§6.11).
- **Sin portar.** La regla de caras traseras de closed shells y el cut por error viven en `athenea-pieces`; el
  `CutSelector` del motor principal no ha cambiado (§6.11).
- **Residencia.** Las pages tienen unas 4× el cut porque los nodos necesarios y los descartados por la vista comparten
  cada page; el player separa la traversal de la carga útil en lugar de encoger las pages (§6.8, §6.11).
- **Opacidad de los merges en el player.** El player dibuja la masa de cada merge, mientras que el rasterizer de athenea
  conserva la coverage por mínimos cuadrados; el ajuste de coverage es la mejor opción salvo en la grille (§6.4).
- **El cut en el player.** La traversal compacta no guarda el tamaño de LoD de un merge, así que un cut combinado de
  tamaño y error no tiene con qué protegerse, y el cut por error solo no ganó al cut por tamaño en todas las vistas
  (§7.13).

## 9.6 Huecos en la validación

- **Un único asset de producción.** Las cifras a nivel de coche son de un asset, el Corvette, con el gorrión y el peón
  como controles animado y rígido. La generalidad a otros assets no se ha medido.
- **Un único entorno para el coche.** Las comparaciones a nivel de coche frente al path trace son bajo un cielo
  (`autoshop_01_4k`, §1.6). La fidelidad del relight bajo otros cielos y luces locales no se ha medido a la escala del
  coche.
- **El close-up level a 5120 px frente al path trace** está medido sobre el coche entero con una cámara y un cut (0.25 px:
  24.00 dB en el frame, 18.31 dB en el coche, §6.9) y en el logo de cerca en el player (§7.12.3), no a lo largo de una
  órbita cercana.
- **v8.2b en el móvil.** El conjunto de datos final está publicado y medido frente al path trace en el player
  (§7.12.3); su descarga y sus tiempos de frame en el iPhone real no, ni la base de 512 clusters allí (§7.2.2).
- **Los tiempos en device real** existen solo para las rampas de estrés y dos pruebas en el device (§7.13);
  la velocidad del móvil emulado no es la del móvil.

## 9.7 Trabajo futuro

En el orden que sugieren las medidas:

1. Hacer el bake de la emisión de las lámparas en el transfer, para que los faros y su luz sobre el coche coincidan con
   el path trace.
2. Ajustar la curvatura sobre el footprint de cada Gaussian con muestras del mesh, para reducir a la mitad el
   desplazamiento de los brillos del capó y estabilizar el giro por píxel.
3. Pasar a cells de open directions de 32 × 32 donde un coat nítido refleja el propio coche (la clase abierta de
   la puerta).
4. Tratar los tubos de cromo finos y los labios enrollados (la tira y el labio inferior del parachoques), el reflejo
   del suelo cercano en la llanta y las molduras finas lejanas.
5. Medir el coche con relight frente al path trace bajo entornos variados y luces locales, y el close-up level a
   5120 px a lo largo de una órbita cercana del coche entero.
6. Para los assets animados: más de cuatro influencias, oclusión entre partes en poses distintas de la bind pose, y
   bakes de visibilidad por encima de 3 M Gaussians por parte.
7. Ponderar el codebook de oclusión por una órbita de vistas, y portar el cut por error y la regla de closed shells
   al `CutSelector` del motor principal.
8. En la aplicación a móvil, medir v8.2b en el iPhone real, de cerca y de lejos, con la base de 512 clusters.
9. Dibujar los canales de §4.11 para muchos assets, cielos y cámaras, alineados con los frames con relight, y aprender
   de ellos a predecir los canales del transporte o la luz con relight a partir de entradas que no los llevan -- un
   adaptador aprendido sobre un modelo de imagen que maneje la luz de una captura o de un render como lo haría el
   bake.
