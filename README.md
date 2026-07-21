# Arboles de Dependencia, Informacion Mutua y Redes Bayesianas

## Prediccion de Abandono Universitario

Proyecto universitario para la asignatura de Lenguajes de Programacion (UNAP, 2026).

### Dataset

Predict Students' Dropout and Academic Success (Realinho et al., 2021)
- UCI Machine Learning Repository, Dataset ID 697
- 4,424 estudiantes de educacion superior en Portugal
- 37 variables originales, 9 seleccionadas para el analisis

### Metodologia

1. Seleccion de 9 variables predictoras + Target
2. Split por `Displaced` (desplazado de residencia)
3. Calculo de entropia individual (Shannon)
4. Matriz de Informacion Mutua entre todos los pares
5. Arbol de Expansion Maxima (MST) con Kruskal y Prim
6. Analisis de sensibilidad rVP (Rango de Variacion Permitido)
7. Red Bayesiana via algoritmo de Chow-Liu

### Archivos

| Archivo | Descripcion |
|---------|-------------|
| `resolucion_dropout.py` | Script completo del analisis |
| `informe_dropout.tex` | Informe LaTeX completo |
| `Calculos_Dropout.xlsx` | Calculos paso a paso (entropia, MI, MST, CPTs) |
| `data.csv` | Dataset original |
| `arbol_*.png` | Visualizaciones de los arboles MST |
| `red_bayesiana.png` | Red Bayesiana (Chow-Liu) |
| `matriz_MI_*.csv` | Matrices de Informacion Mutua |

### Ejecucion

```bash
pip install pandas numpy networkx matplotlib scikit-learn
python resolucion_dropout.py
```

### Variables

| Variable | Descripcion |
|----------|-------------|
| x1 | Nota de admision |
| x2 | Edad al ingreso |
| x3 | Pagos al dia |
| x4 | Becado |
| x5 | Aprobados 1er semestre |
| x6 | Nota media 1er semestre |
| x7 | Aprobados 2do semestre |
| x8 | Nota media 2do semestre |
| x9 | Estado civil |
| Target | Dropout / Enrolled / Graduate |
| Split | Displaced (Si / No) |

### Resultados principales

| Metrica | No Desplazados | Desplazados |
|---------|---------------|-------------|
| Filas | 1,998 | 2,426 |
| Peso MST | 2.3421 | 2.2759 |
| Variables criticas rVP | x9, x3, x2, Target | |

La red Bayesiana aprendida (Chow-Liu) muestra que `Target` influye directamente sobre `x7` (aprobados 2do semestre), `x3` (pagos al dia), `x4` (becado) y `x2` (edad).

### Fuente

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). Predict Students' Dropout and Academic Success [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
