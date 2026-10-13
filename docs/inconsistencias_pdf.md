# Inconsistencias documentadas del PDF

## Diagonal B55 de la matriz de susceptancias

En una matriz `Ba` mostrada posteriormente en el PDF, la entrada diagonal
correspondiente a la barra 5 aparece como `5`. Según la topología del caso,
la barra 5 está conectada por las líneas `1-5` y `3-5`, ambas con `x = 0.2`
pu, por lo que `B55 = 1/0.2 + 1/0.2 = 10`. Esto también coincide con la
formulación previa `x15^-1 + x35^-1`.

Se trata de una errata del documento. El código construye `Bbus`
algorítmicamente a partir de las líneas cargadas desde los CSV y no copia la
matriz numérica del PDF.

## Límite superior de la línea 2-4

En una restricción específica del desarrollo MILP aparece `P23_max` como
límite superior de la línea `2-4`. Conceptualmente debe utilizarse `P24_max`.
En este dataset ambos valores son 100 MW, por lo que la errata no altera el
resultado numérico. El código obtiene siempre la capacidad del objeto
correspondiente al corredor `2-4`; no copia la expresión literal del PDF.
