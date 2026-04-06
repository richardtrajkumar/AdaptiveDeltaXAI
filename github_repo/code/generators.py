"""
generators.py
=============
Synthetic Datacenter Telemetry Generators

Two generators for validating AdaptiveDeltaXAI on realistic DC telemetry:

  generate_dc_temperature() — 8-sensor rack temperature stream
  generate_dc_power()       — 8-server power consumption stream

Both include physically-motivated coupling, periodic components,
noise, and injected anomaly bursts with configurable intensity.
"""

import numpy as np
import pandas as pd
from typing import Optional


def generate_dc_temperature(
    n_sensors:        int   = 8,
    duration_min:     int   = 120,
    freq_s:           int   = 10,
    n_anomaly_bursts: int   = 4,
    burst_len_s:      int   = 120,
    burst_mag_low:    float = 8.0,
    burst_mag_high:   float = 20.0,
    thermal_tau:      int   = 3,
    seed:             int   = 42,
) -> pd.DataFrame:
    """
    Synthetic DC temperature telemetry with realistic thermal physics.

    Physical model per sensor i:
        temp_i(t) = baseline_i
                  + 2·sin(2πt/86400 + φᵢ)        24-hour diurnal
                  + 1.5·sin(2πt/600  + φᵢ)        10-min CRAC cycle
                  + 0.15·temp_{i-1}(t)             thermal coupling
                  + N(0, 0.3²)                     measurement noise
                  + anomaly_burst(t)               cooling failure spike

    Anomaly model: exponential rise to peak, then exponential fall.
        temp_anomaly(Δt) = spike_mag × (1 - exp(-Δt / τ))    rise
        temp_anomaly(Δt) = peak × exp(-Δt / τ)               fall

    Args:
        n_sensors:        Number of temperature sensors
        duration_min:     Total stream duration in minutes
        freq_s:           Sampling interval in seconds
        n_anomaly_bursts: Number of anomaly burst windows to inject
        burst_len_s:      Duration of each burst in seconds
        burst_mag_low:    Minimum spike magnitude (°C above baseline)
        burst_mag_high:   Maximum spike magnitude (°C above baseline)
        thermal_tau:      Thermal time constant (steps). 3 steps = 30s.
        seed:             Random seed for reproducibility

    Returns:
        pd.DataFrame with columns:
            temp_0 .. temp_{n-1}  float  — sensor readings in °C
            timestamp             datetime
            anomaly               int    — 1 during burst windows, else 0

    Realistic ranges:
        Normal:  35–47°C (CPU temperatures in air-cooled rack)
        Anomaly: 55–70°C (cooling failure / airflow blockage)
    """
    rng   = np.random.RandomState(seed)
    n     = int(duration_min * 60 / freq_s)
    t_sec = np.arange(n) * freq_s
    data  = {}
    labels = np.zeros(n, dtype=int)

    # Inject anomaly burst windows (avoid first and last 10% of stream)
    margin       = n // 10
    burst_starts = rng.choice(np.arange(margin, n - margin - burst_len_s // freq_s),
                               n_anomaly_bursts, replace=False)
    burst_len    = burst_len_s // freq_s

    for bs in burst_starts:
        be = min(bs + burst_len, n)
        labels[bs:be] = 1

    for i in range(n_sensors):
        # Periodic components
        diurnal  = 2.0  * np.sin(2 * np.pi * t_sec / 86400 + i * 0.3)
        cooling  = 1.5  * np.sin(2 * np.pi * t_sec / 600   + i * 0.5)
        baseline = 35.0 + rng.uniform(-2, 2)
        noise    = rng.randn(n) * 0.3

        temp = baseline + diurnal + cooling + noise

        # Thermal coupling from left-neighbour sensor
        if i > 0:
            temp += 0.15 * data[f"temp_{i-1}"]

        # Anomaly injection with exponential rise/fall
        spike_mags = rng.uniform(burst_mag_low, burst_mag_high, n_anomaly_bursts)
        for j, bs in enumerate(burst_starts):
            be      = min(bs + burst_len, n)
            peak_   = spike_mags[j]
            # Rise phase
            for dt in range(be - bs):
                temp[bs + dt] += peak_ * (1 - np.exp(-dt / max(thermal_tau, 1)))
            # Fall phase (after burst ends)
            for dt in range(1, burst_len + 1):
                idx = be - 1 + dt
                if idx >= n:
                    break
                temp[idx] += peak_ * np.exp(-dt / max(thermal_tau, 1))

        data[f"temp_{i}"] = temp

    df = pd.DataFrame(data)
    df["timestamp"] = pd.date_range("2025-01-15 08:00:00", periods=n,
                                     freq=f"{freq_s}s")
    df["anomaly"]   = labels
    return df


def generate_dc_power(
    n_servers:        int   = 8,
    duration_min:     int   = 120,
    freq_s:           int   = 10,
    n_anomaly_bursts: int   = 4,
    burst_len_s:      int   = 90,
    burst_mult_low:   float = 1.5,
    burst_mult_high:  float = 2.5,
    seed:             int   = 42,
) -> pd.DataFrame:
    """
    Synthetic DC power telemetry with PDU coupling and workload cycles.

    Physical model per server i (in PDU group i//2):
        power_i(t) = baseline_i
                   + 50·sin(2πt/1800 + φᵢ)         30-min batch job cycle
                   + PDU_shared(t)                   shared PDU fluctuation
                   + N(0, 4²)                        measurement noise

    Anomaly model: multiplicative power surge.
        power_anomaly(t) = power(t) × U(1.5, 2.5)   runaway process / PSU fault

    PDU grouping: servers 0+1 → PDU_0, servers 2+3 → PDU_1, etc.
    Servers within the same PDU share a common 20W-amplitude noise signal.

    Args:
        n_servers:        Number of servers
        duration_min:     Stream duration in minutes
        freq_s:           Sampling interval in seconds
        n_anomaly_bursts: Number of anomaly burst windows
        burst_len_s:      Duration of each burst in seconds
        burst_mult_low:   Minimum surge multiplier (×)
        burst_mult_high:  Maximum surge multiplier (×)
        seed:             Random seed

    Returns:
        pd.DataFrame with columns:
            power_0 .. power_{n-1}  float  — server power in Watts
            timestamp               datetime
            anomaly                 int    — 1 during burst windows

    Realistic ranges:
        Normal:  100–300W (1U server idle-to-full-load)
        Anomaly: 300–750W (runaway process / PSU fault)
        Clipped: [50W, 900W] (physical hardware limits)
    """
    rng    = np.random.RandomState(seed + 10)
    n      = int(duration_min * 60 / freq_s)
    t_sec  = np.arange(n) * freq_s
    data   = {}
    labels = np.zeros(n, dtype=int)

    margin       = n // 10
    burst_starts = rng.choice(np.arange(margin, n - margin - burst_len_s // freq_s),
                               n_anomaly_bursts, replace=False)
    burst_len    = burst_len_s // freq_s

    for bs in burst_starts:
        be = min(bs + burst_len, n)
        labels[bs:be] = 1

    # PDU-level shared signal (one per PDU group)
    n_pdus = max(n_servers // 2, 1)
    pdu_signals = {
        p: 20 * np.sin(2 * np.pi * t_sec / 1800 + p * 1.1) + rng.randn(n) * 3
        for p in range(n_pdus)
    }

    for i in range(n_servers):
        pdu_id   = i // 2
        workload = 50 * np.sin(2 * np.pi * t_sec / 1800 + i * 0.8)
        baseline = 200 + rng.uniform(-20, 20)
        noise    = rng.randn(n) * 4
        power    = baseline + workload + pdu_signals[pdu_id] + noise

        # Anomaly: multiplicative surge per burst
        for bs in burst_starts:
            be         = min(bs + burst_len, n)
            surge_mult = rng.uniform(burst_mult_low, burst_mult_high)
            power[bs:be] *= surge_mult

        data[f"power_{i}"] = np.clip(power, 50, 900)

    df = pd.DataFrame(data)
    df["timestamp"] = pd.date_range("2025-01-15 08:00:00", periods=n,
                                     freq=f"{freq_s}s")
    df["anomaly"]   = labels
    return df


def generate_sample_csvs(
    output_dir: str = "datasets",
    n_rows: int = 100,
    seed: int = 42,
):
    """Generate small sample CSVs for the repository datasets/ folder."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    df_t = generate_dc_temperature(n_sensors=8, duration_min=20,
                                    freq_s=10, seed=seed)
    df_p = generate_dc_power(n_servers=8, duration_min=20,
                              freq_s=10, seed=seed)

    df_t.head(n_rows).to_csv(f"{output_dir}/sample_dc_temperature.csv", index=False)
    df_p.head(n_rows).to_csv(f"{output_dir}/sample_dc_power.csv",       index=False)
    print(f"Saved {n_rows}-row samples to {output_dir}/")


if __name__ == "__main__":
    generate_sample_csvs()
    df_t = generate_dc_temperature()
    df_p = generate_dc_power()
    print(f"Temperature: {df_t.shape}  anomaly_rate={df_t.anomaly.mean():.1%}")
    print(f"Power:       {df_p.shape}  anomaly_rate={df_p.anomaly.mean():.1%}")
