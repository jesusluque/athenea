<!-- Copyright (c) 2026 jesus luque. -->

# 9. Limitaciones y trabajo futuro

Las limitaciones de abajo se reúnen de las listas de las propias secciones (§2.16, §3.12, §4.11, §5.13, §6.11, §7.13)
y se agrupan por dónde actúan. Cada una se enuncia con la cifra que la mide, cuando la hay.

## 9.1 Defectos que quedan en el Corvette

Esto es lo que el coche de 1280 px aún muestra frente al path trace, los defectos que quien mira el móvil ve primero.

| defecto | medido | causa | remedio propuesto | § |
|---|---|---|---|---|
| faros oscuros | faros oscuros frente al path trace | el horneado del transfer no lleva la emisión de las lámparas | hornear la emisión de las lámparas y su luz sobre el coche | 3.12, 4.11 |
| tira cromada del parachoques delantero | un 38 % más clara que el path trace, con el pico más bajo | un tubo de unos pocos píxeles de ancho: cada gaussiana lee su propio espejo con una sola normal | — (abierto) | 3.12, 4.11 |
| reflejo del suelo cercano en la llanta | 0.72 del path trace | — (no analizado más) | — (abierto) | 4.11 |
| reflejos abiertos de la puerta | 1.13 del path trace; el 68 % del exceso donde el espejo da contra el coche | celdas de 16 × 16 de unos 11° dejan pasar cielo detrás de un retrovisor o un pilar | celdas de 32 × 32 (+96 B por gaussiana) lo reducirían más o menos a la mitad | 4.10.4, 4.11 |
| labio inferior del parachoques | irregular | la huella de una celda por gaussiana en un borde enrollado | — (abierto) | 3.12 |
| brillos del capó | a 2.0 px de su posición en el path trace | el giro por píxel extrapola el operador de forma del triángulo central | una curvatura ajustada sobre la huella con muestras de la malla ($p_{90}$ 9.8° frente a 46°) lo reduciría más o menos a la mitad | 4.10.6, 4.11 |
| huecos de la rejilla en los cortes gruesos | 2.24 de las hojas a 640 px, 1.04 a 1280 px | fusiones de tiras de menos de un píxel; el hueco es una propiedad de la proyección | hojas más finas (hecho a 1280 px); ninguna medida por nodo independiente de la vista lo ve | 6.5, 6.11 |
| molduras finas lejanas | trazos gruesos en todos los cortes | las cotas de error no ven este error | — (abierto) | 6.11 |

## 9.2 La representación y el transfer

- **Una normal, una huella.** Una gaussiana tiene una normal y la huella de una celda; las tiras finas curvadas y los
  bordes enrollados son donde se nota (§9.1). La correspondencia de cobertura supone el solapamiento $2\pi\sigma^2$ de
  la retícula de celdas, que no tienen los triángulos menores que una celda ni las gaussianas sujetas por bordes
  (§3.12).
- **Vidrio.** Un vidrio fino tintado se convierte como sólido, y una nube conserva un único índice de refracción salvo
  que lleve capas (§3.12). Las gaussianas de lámina fina se suben a un suelo de opacidad de 0.1 por gaussiana, así que
  fusionar menos láminas y más grandes hace variar la transmisión de celda a celda; queda pendiente un suelo por píxel
  (§6.11).
- **Campo de rango uno.** El acoplamiento del campo reflejado al cielo pone en un reflejo el color de un suelo
  iluminado, no su forma, bajo un cielo iluminado desde un lado; con grado 3 resuelve un horizonte a unos veinte grados
  (correlación 0.56 con grado 3, 0.63 con grado 5, 0.85 con un mapa octaédrico de 32 × 32, en simulación) (§4.11).
- **Luces dentro de la escena.** Se extrae un sol por cúpula. Las lámparas dentro de una escena se sombrean con bits
  que dicen qué direcciones salen de la escena, no cuáles llegan a la lámpara (§4.11).
- **Tamaño en el dispositivo.** La memoria del dispositivo limita una pieza a unos 5 M hojas para el árbol de fusión;
  el tope de 4.5 M hizo que las piezas más grandes se convirtieran a una resolución de cámara más gruesa (Bottom a
  3392 px, Part_3 y Front_Bumper_001 a 4480 px; §3.11, §6.9).
- **Instanciado.** Los point instancers anidados solo se expanden en el nivel más interno, y el movimiento creado como
  velocidades se lee en la muestra (§3.12).

## 9.3 Assets animados

- Cuatro influencias por gaussiana truncan las puntas de las alas del gorrión (el 27 % de los vértices del ala llevan
  más); la proporción de gaussianas truncadas no se ha medido (§5.3, §5.13).
- No hay skinning por cuaterniones duales para las nubes, ni retargeting, ni mezcla de clips; de varias mallas con
  skinning con distintas transformaciones de prim solo se conserva la de la primera (§5.13).
- La ruta trazada posa una vez por frame, así que no desenfoca una nube con skinning, y las instancias de LoD no llevan
  movimiento (§5.13).
- La visibilidad por parte deja fuera el autosombreado de la propia parte, supone partes rígidas que ocluyen de forma
  independiente, y no se ha horneado por encima de unos 3 M gaussianas por parte; los transfers zonales conservan solo
  la mitad directa y el autosombreado de la pose de bind (§4.11, §5.8, §5.13).
- El error del ajuste zonal en el gorrión, y una medida geométrica de la cobertura en una flexión cerrada, no se han
  medido (§5.13).

## 9.4 Niveles de detalle, streaming y el reproductor

- **Residencia.** Las páginas tienen unas 4× el corte porque los nodos necesarios y los descartados por la vista
  comparten cada página; el móvil separa el recorrido de la carga útil en lugar de encoger las páginas (§6.8, §6.11).
- **Ponderación de la oclusión.** El codebook compartido se reparte por entradas usadas, no por la frecuencia con que
  una vista descarta un nodo; ponderar por una órbita de vistas en el horneado debería llevar el cull por encima del
  84 % del ideal (§6.11).
- **Opacidad de las fusiones en el móvil.** El reproductor dibuja la masa de cada fusión, mientras que el rasterizador
  de athenea conserva la cobertura por mínimos cuadrados; el ajuste de cobertura es la mejor opción salvo en la rejilla
  (§6.4).
- **El corte en el móvil.** El recorrido compacto no guarda el tamaño de LoD de una fusión, así que un corte combinado
  de tamaño y error no tiene con qué protegerse, y el corte por error solo no ganó al corte por tamaño en todas las
  vistas (§7.13).
- **Sin portar.** La regla de caras traseras de cáscaras cerradas y el corte por error viven en `athenea-pieces`; el
  `CutSelector` del motor principal no ha cambiado (§6.11).

## 9.5 El motor

- Los splats todavía no están dentro de los rayos secundarios del path tracer: con mallas, los splats se componen por
  profundidad sobre las superficies trazadas y proyectan sombras sobre ellas, pero un rayo de rebote no los ve (§2.16).
- El desenfoque de movimiento y de lente del rasterizador son gaussianas ajustadas por momentos, hasta un 38 % demasiado
  picudas para estelas largas (§2.16).
- Las gaussianas de unas centésimas de píxel desaparecen del raster; los niveles de detalle son el remedio (§2.16).
- La visibilidad de mallas por raster paga unos 8 ms de tiempo de host por cada 1800 draws; la compilación de
  materiales es síncrona; la TLAS se reconstruye cada frame; la distorsión de lente más allá de $k_1, k_2$ no se aplica
  (§2.16).

## 9.6 Huecos en la validación

- **v8 en el móvil.** Los árboles y la oclusión compartida del nivel de primer plano a 5120 px están escritos (§6.9),
  pero su conversión para el reproductor estaba en curso cuando se escribió esto. Su calidad frente al path trace, su
  descarga y su comportamiento en el iPhone real están sin medir (§7.13). En particular, lo blando de cerca que lo
  motivó (§7.12.1) solo se ha mostrado resuelto en una prueba del logo y la rejilla (§3.11), no en el coche entero en el
  dispositivo.
- **La base de 512 clusters** no se ha medido en el móvil real (§7.2.2).
- **Los tiempos en dispositivo real** existen solo para las rampas de estrés y las dos pruebas del usuario (§7.13); la
  velocidad del móvil emulado no es la del móvil.
- **Un único asset de producción.** Las cifras a nivel de coche son de un asset, el Corvette, con el gorrión y el peón
  como controles animado y rígido. La generalidad a otros assets no se ha medido.
- **Máquina compartida.** Los tiempos de horneado y de árboles se tomaron en una máquina compartida con otros trabajos
  de GPU y no son medidas de rendimiento (§3.11, §8.9).

## 9.7 Trabajo futuro

En el orden que sugieren las medidas:

1. Medir v8 frente al path trace y en el iPhone real, de cerca y de lejos, con la base de 512 clusters.
2. Hornear la emisión de las lámparas en el transfer, para que los faros y su luz sobre el coche coincidan con el path
   trace.
3. Ajustar la curvatura sobre la huella de cada gaussiana con muestras de la malla, para reducir a la mitad el
   desplazamiento de los brillos del capó y estabilizar el giro por píxel.
4. Pasar a celdas de direcciones abiertas de 32 × 32 donde un coat nítido refleja el propio coche (la clase abierta de
   la puerta).
5. Tratar los tubos de cromo finos y los labios enrollados (la tira y el labio inferior del parachoques), el reflejo
   del suelo cercano en la llanta y las molduras finas lejanas.
6. Ponderar el codebook de oclusión por una órbita de vistas, y portar el corte por error y la regla de cáscaras
   cerradas al `CutSelector` del motor principal.
7. Para los assets animados: más de cuatro influencias, oclusión entre partes en poses distintas de la de bind, y
   horneados de visibilidad por encima de 3 M gaussianas por parte.
