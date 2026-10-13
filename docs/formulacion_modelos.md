# Formulación de los modelos

Todas las potencias y capacidades internas están en pu; los reportes de
flujos se convierten a MW usando `base_mva`. Los ángulos están en radianes y
las susceptancias son positivas, `b = 1/x`.

## Flujo DC

Para cada rama orientada `i -> j`:

`P_ij = b_ij (theta_i - theta_j)`.

La matriz nodal se construye acumulando `Bii += b`, `Bjj += b`,
`Bij -= b` y `Bji -= b`. El balance es `P_spec = PG - PL = Bbus theta`,
con la barra de referencia fijada a cero.

## LP interactivo

Cada corredor candidato tiene variables no negativas `PD_plus` y `PD_minus`.
El objetivo minimiza `sum(cost_k (PD_plus_k + PD_minus_k))`. Las líneas
existentes obedecen el flujo DC y sus límites; el flujo ficticio neto de cada
corredor entra en los balances nodales según su orientación. Después de cada
solución se instala una línea en el corredor con mayor
`PD_plus + PD_minus`.

## MILP Big-M

`w_ijk` selecciona una configuración de `k` líneas nuevas en el corredor.
Solo una configuración puede activarse. Las variables `DeltaTheta_ijk`
se enlazan con el ángulo mediante Big-M configurable. El flujo candidato es
`k * b_ij * DeltaTheta_ijk`, limitado por `k * Pmax_ij * w_ijk`.
El objetivo es el costo de inversión.

## MINLP bilineal

`n_ij` es entero entre cero y `max_lines`. El flujo candidato satisface la
igualdad no convexa `P_candidate_ij = n_ij b_ij
(theta_i - theta_j)`, sin Big-M ni binarización. Gurobi se ejecuta con
`NonConvex = 2`.

En MILP y MINLP la línea existente `3-5` y el candidato `3-5` son
contribuciones distintas. Las expansiones se reconstruyen y se validan
independientemente mediante el motor DC.

La expansión `4-2-0-1` solo es una referencia de validación y no se usa como
restricción ni como valor inicial obligatorio.
