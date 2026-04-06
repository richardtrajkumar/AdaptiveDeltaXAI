# Datasets

This folder contains sample data for AdaptiveDeltaXAI validation.

## Files

| File | Rows | Description |
|------|------|-------------|
| `sample_dc_temperature.csv` | 100 | First 100 timesteps of synthetic DC temperature stream |
| `sample_dc_power.csv` | 100 | First 100 timesteps of synthetic DC power stream |

The full 720-step (2-hour) datasets are generated programmatically using `code/generators.py`.

---

## DC Temperature schema

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `temp_0` .. `temp_7` | float | °C | Temperature reading per sensor |
| `timestamp` | datetime | — | ISO 8601, 10-second intervals |
| `anomaly` | int | 0/1 | 1 during injected cooling failure bursts |

**Normal range:** 35–47°C (CPU temperatures in air-cooled rack)
**Anomaly range:** 55–70°C (cooling failure / airflow blockage)
**Sampling:** 10 seconds
**Duration:** 2 hours (720 steps at full resolution)
**Anomaly rate:** ~6.7% (4 bursts × ~120s each)

### Physical model

```
temp_i(t) = baseline_i                          sensor-specific offset, U(33, 37)°C
          + 2·sin(2πt/86400 + φᵢ)              24-hour diurnal cycle
          + 1.5·sin(2πt/600  + φᵢ)             10-minute CRAC cooling cycle
          + 0.15·temp_{i-1}(t)                  left-neighbour thermal coupling
          + N(0, 0.3²)                          sensor measurement noise
          + anomaly_burst(t)                    exponential rise/fall spike
```

### Thermal coupling

Sensors are arranged in a rack row. Each sensor i inherits 15% of its left neighbour's temperature. This produces realistic stratification — sensors at the end of the row are systematically warmer.

### Anomaly model

Each burst uses an exponential rise/fall with thermal time constant τ = 3 steps (30 seconds):

```
rise: temp[t] += peak × (1 - exp(-Δt / τ))
fall: temp[t] += peak × exp(-Δt / τ)
```

---

## DC Power schema

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `power_0` .. `power_7` | float | Watts | Server power consumption |
| `timestamp` | datetime | — | ISO 8601, 10-second intervals |
| `anomaly` | int | 0/1 | 1 during injected power surge bursts |

**Normal range:** 100–300W (1U server idle to full load)
**Anomaly range:** 300–750W (runaway process / PSU fault)
**Clipped:** [50W, 900W] (physical hardware limits)
**Sampling:** 10 seconds
**Duration:** 2 hours (720 steps)
**Anomaly rate:** ~5%

### Physical model

```
power_i(t) = baseline_i                         server idle power, U(180, 220)W
           + 50·sin(2πt/1800 + φᵢ)             30-minute batch job cycle
           + PDU_coupling(t)                    shared PDU fluctuation (servers i//2 share PDU)
           + N(0, 4²)                           measurement noise
           × U(1.5, 2.5)  [anomaly only]        power surge multiplier
```

### PDU coupling

Servers are grouped by PDU: servers 0+1 share PDU_0, servers 2+3 share PDU_1, etc. Servers within the same PDU share a 20W-amplitude sinusoidal noise component, creating realistic within-PDU correlations that AdaptiveDeltaXAI's group attribution correctly captures.

---

## Generating the full dataset

```python
from code.generators import generate_dc_temperature, generate_dc_power

df_temp  = generate_dc_temperature(n_sensors=8, duration_min=120, seed=42)
df_power = generate_dc_power(n_servers=8, duration_min=120, seed=42)

# Shape: (720, 10) — 8 sensor columns + timestamp + anomaly
print(df_temp.shape, df_temp.anomaly.mean())
```

---

## Reproducing the published results

The Colab notebook `notebooks/Delta_XAI_Colab_Validation.ipynb` regenerates both datasets from scratch using the same seeds (42 for temperature, 52 for power) and runs all evaluation sections end-to-end.
