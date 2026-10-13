# Garver 6-bus transmission expansion planning

Proyecto de expansión de transmisión para el caso Garver de seis barras.
Incluye carga validada de datos CSV, flujo DC, LP interactivo
Villasana-Garver, MILP con Big-M, MINLP bilineal, validación cruzada y
reportes CSV.

## Métodos

- **Flujo DC:** infraestructura eléctrica común.
- **LP interactivo:** usa capacidad ficticia y agrega líneas iterativamente.
- **MILP:** selecciona configuraciones discretas con variables binarias y
  enlaces Big-M.
- **MINLP:** usa el producto bilineal entre número de líneas y diferencia
  angular.

Los tres optimizadores reconstruyen la red y se validan posteriormente con
`solve_dc_power_flow`. La referencia esperada `4-2-0-1` no está codificada en
las restricciones.

## Requisitos e instalación

Se requiere Python 3.10 o superior, NumPy, pandas, pytest y una instalación
de `gurobipy`. MILP y MINLP requieren una licencia Gurobi válida.

```powershell
pip install -r requirements.txt
```

## Ejecución

Ejecutar desde la carpeta del proyecto:

```powershell
python run_lp.py
python run_milp.py
python run_minlp.py
python run_all.py
python -m pytest tests -q
```

Los scripts localizan `data/` y `results/` relativos a su propio archivo.

## Resultados

Los archivos generados en `results/` contienen iteraciones y solución del LP,
planes y flujos del MILP y MINLP, y una comparación consolidada. Los flujos
se reportan en MW; los modelos usan pu internamente.

## Datos e inconsistencias del PDF

Los datos están en `data/buses.csv`, `data/existing_lines.csv` y
`data/candidate_corridors.csv`. Las erratas documentadas sobre `B55` y el
límite de la línea `2-4` están en
[`docs/inconsistencias_pdf.md`](docs/inconsistencias_pdf.md).

La formulación compacta está en
[`docs/formulacion_modelos.md`](docs/formulacion_modelos.md).

## Escalabilidad

La topología, incidencia, Bbus, islas y balances se generan a partir de los
CSV y no asumen seis barras ni numeración consecutiva. El tamaño práctico de
MILP/MINLP depende del número de corredores, configuraciones y licencia de
Gurobi.
