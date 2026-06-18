# Casos bi-Maxwellian de anisotropía

Este documento detalla los nueve ejecutables bi-Maxwellian listos para
compilar. El catálogo corto está en `src/SIMULACIONES.md`.

## Configuración común

| Parámetro | Valor |
|---|---:|
| Configuración PSC | `PscConfig1vbecSingle<dim_yz>` |
| Campo de fondo | `B0 = 0.05` en dirección `z` |
| `vA/c` | 0.05 |
| `mi/me` | 200 |
| `lambda0` | 20 |
| Densidad inicial | 1.0 |
| Partículas por celda | 2000 |
| Grilla | `1408 x 1408` |
| Dominio | `30 d_i x 30 d_i` |
| MPI patches | `64 x 16 = 1024` |
| Fronteras | Periódicas |
| Checkpoint | cada 7500 pasos en build ADIOS2 |

El campo paralelo es `z`; por tanto:

```text
T_parallel = T_z
T_perp = (T_x + T_y) / 2
A = T_perp / T_parallel
```

## Mirror

Iones con exceso de temperatura perpendicular. Electrones isotrópicos.

| Ejecutable | Régimen | beta_i_parallel | A_i | beta_e_parallel | A_e |
|---|---|---:|---:|---:|---:|
| `psc_M_S_bM` | Strong | 5.0 | 3.0 | 1.0 | 1.0 |
| `psc_M_M_bM` | Moderate | 5.0 | 2.0 | 1.0 | 1.0 |
| `psc_M_W_bM` | Weak | 6.0 | 1.5 | 1.0 | 1.0 |

Criterio simplificado:

```text
beta_i_parallel * (A_i - 1) > 1
```

## Firehose

Iones con exceso de temperatura paralela. Electrones isotrópicos.

| Ejecutable | Régimen | beta_i_parallel | A_i | beta_e_parallel | A_e |
|---|---|---:|---:|---:|---:|
| `psc_F_S_bM` | Strong | 10.0 | 0.1 | 1.0 | 1.0 |
| `psc_F_M_bM` | Moderate | 6.0 | 0.3 | 1.0 | 1.0 |
| `psc_F_W_bM` | Weak | 3.0 | 0.6 | 1.0 | 1.0 |

Criterio simplificado:

```text
beta_i_parallel * (1 - A_i) > 2
```

## Whistler

Electrones con exceso de temperatura perpendicular. Iones isotrópicos.

| Ejecutable | Régimen | beta_i_parallel | A_i | beta_e_parallel | A_e |
|---|---|---:|---:|---:|---:|
| `psc_W_S_bM` | Strong | 1.0 | 1.0 | 0.5 | 3.0 |
| `psc_W_M_bM` | Moderate | 1.0 | 1.0 | 0.5 | 2.0 |
| `psc_W_W_bM` | Weak | 1.0 | 1.0 | 0.5 | 1.5 |

Criterio práctico usado en análisis:

```text
A_e > 1 + 0.21 / beta_e_parallel^0.6
```

## Salidas

Los casos escriben campos, momentos y partículas con nombres propios:

```text
pfd.<step>_p<rank>.h5
pfd_moments.<step>_p<rank>.h5
prt_M_S_bM.<step>.h5
```

En build ADIOS2 también se escriben:

```text
checkpoint_<step>.bp/
```

La región de partículas guardada es el 20% central de cada dirección resuelta,
para controlar el tamaño de salida.
