#!/usr/bin/env python3
"""Tier-2 synthetic telemetry generator (seed 42).

Extracted verbatim from cells 1, 2, 4 and 6 of AdaptiveDeltaXAI_Tier2_DatasetGenerator.ipynb;
the only change is that the output directory is a command-line argument instead of /content/.
Running it reproduces the three SHA-256 values recorded in results/manifests/run_manifest_v1_9.json.
"""
import argparse
_p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_p.add_argument("--out", default="./tier2_out", help="output directory")
ARGS = _p.parse_args()

# ---- notebook cell 1 ----
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, time, json, warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
print('Imports OK')
print('Working directory:', os.getcwd())


# ---- notebook cell 2 ----
T2_STEPS      = 17_280
T2_SAMPLE_S   = 10
T2_HOURS      = 48
T2_WINDOW     = 20

TEMP_NAMES = ['Inlet_Temp','CPU1_Temp','CPU2_Temp','DIMM_A1_Temp',
               'DIMM_B1_Temp','PCH_Temp','Exhaust_Temp','Fan1_norm']
TEMP_BASE  = [22.0, 52.0, 50.0, 42.0, 41.0, 45.0, 38.0, 45.0]
TEMP_STD   = [ 0.3,  1.2,  1.1,  0.8,  0.8,  0.6,  0.5,  0.6]

POWER_NAMES = ['PS1_Input_W','PS2_Input_W','PS1_Output_W']
POWER_BASE  = [320.0, 315.0, 290.0]
POWER_STD   = [ 12.0,  11.0,  10.0]

N_CLEAN   = 60
N_OVERLAP = 6
N_DRIFT   = 6
BURST_LEN = 12
DRIFT_LEN = 75
TAU       = 3

DIURNAL_AMP  = 8.0
CRAC_AMP     = 1.5
CRAC_PERIOD  = 600
BATCH_PERIOD = 1800
FAN_LAG      = 9
VM_INTERVAL  = 120
VM_DC        = 2.0
VM_DW        = 8.0

print('Constants OK --', T2_STEPS, 'steps,', T2_HOURS, 'hours')


# ---- notebook cell 4 ----
t0 = time.time()
T  = T2_STEPS
t_sec = np.arange(T) * T2_SAMPLE_S

# Periodic base components
diurnal = DIURNAL_AMP * np.sin(2*np.pi*t_sec/86400 - np.pi/2)
crac    = CRAC_AMP    * np.sin(2*np.pi*t_sec/CRAC_PERIOD + 0.3)
batch   =               np.sin(2*np.pi*t_sec/BATCH_PERIOD + 0.8)

vm_sig = np.zeros(T)
for vs in range(VM_INTERVAL, T, VM_INTERVAL + np.random.randint(-15,15)):
    vm_sig[vs:min(vs+3,T)] = 1.0

# Temperature D=8
st = np.zeros((T,8), dtype=np.float64)
st[:,0] = TEMP_BASE[0]+diurnal+crac+vm_sig*VM_DC*0.3+np.random.normal(0,TEMP_STD[0],T)
st[:,1] = TEMP_BASE[1]+0.4*diurnal+15.0*batch+vm_sig*VM_DC+np.random.normal(0,TEMP_STD[1],T)
st[:,2] = TEMP_BASE[2]+0.35*diurnal+13.5*batch+vm_sig*VM_DC*0.9+np.random.normal(0,TEMP_STD[2],T)
st[:,3] = TEMP_BASE[3]+0.2*diurnal+8.0*batch+vm_sig*VM_DC*0.5+np.random.normal(0,TEMP_STD[3],T)
st[:,4] = TEMP_BASE[4]+0.2*diurnal+7.5*batch+vm_sig*VM_DC*0.45+np.random.normal(0,TEMP_STD[4],T)
st[:,5] = TEMP_BASE[5]+0.1*diurnal+5.0*batch+np.random.normal(0,TEMP_STD[5],T)
st[:,6] = TEMP_BASE[6]+0.5*diurnal+10.0*batch+vm_sig*VM_DC*0.4+np.random.normal(0,TEMP_STD[6],T)
cpu_proxy = 15.0*batch+0.4*diurnal
fan_lagged = np.roll(cpu_proxy,FAN_LAG); fan_lagged[:FAN_LAG]=cpu_proxy[0]
st[:,7] = TEMP_BASE[7]+fan_lagged+vm_sig*2.0+np.random.normal(0,TEMP_STD[7],T)
clips = [(-5,75),(-5,95),(-5,95),(-5,85),(-5,85),(-5,75),(-5,70),(30,90)]
for i,(lo,hi) in enumerate(clips): st[:,i]=np.clip(st[:,i],lo,hi)
stream_temp = st.astype(np.float32)

# Anomaly labels (shared across both streams)
labels = np.zeros(T, dtype=np.int32)
anomaly_meta = []

def free_slot(length, min_gap=30):
    for _ in range(2000):
        s = np.random.randint(T2_WINDOW+20, T-length-30)
        if labels[max(0,s-min_gap):s+length+min_gap].sum()==0:
            return s
    return None

def mark(s, length, atype):
    e = min(s+length,T)
    labels[s:e] = 1
    anomaly_meta.append({'type':atype,'start':int(s),'end':int(e)})

# Clean exponential bursts
for _ in range(N_CLEAN):
    s = free_slot(BURST_LEN*2)
    if s is None: continue
    peak = np.random.uniform(8,20)
    for dt in range(BURST_LEN):
        if s+dt>=T: break
        rise = peak*(1-np.exp(-dt/TAU))
        for si in [1,2,6]: st[s+dt,si]+=rise
        st[s+dt,7]+=rise*0.4*(1-np.exp(-max(0,dt-FAN_LAG)/TAU))
    for dt in range(1,BURST_LEN+1):
        if s+BURST_LEN+dt-1>=T: break
        fall=peak*np.exp(-dt/TAU)
        for si in [1,2,6]: st[s+BURST_LEN+dt-1,si]+=fall
    mark(s,BURST_LEN*2,'clean_burst')

# Overlapping bursts
for _ in range(N_OVERLAP):
    s = free_slot(BURST_LEN+8)
    if s is None: continue
    s2 = s+BURST_LEN//2
    for dt in range(BURST_LEN):
        if s+dt  <T: st[s+dt,  1]+=15*(1-np.exp(-dt/TAU))
        if s2+dt <T: st[s2+dt, 2]+=12*(1-np.exp(-dt/TAU))
    mark(s,BURST_LEN+8,'overlap_burst')

# Gradual drift
for _ in range(N_DRIFT):
    s = free_slot(DRIFT_LEN)
    if s is None: continue
    peak=np.random.uniform(10,16); half=DRIFT_LEN//2
    for dt in range(DRIFT_LEN):
        if s+dt>=T: break
        frac=dt/half if dt<half else 1.0-(dt-half)/half
        st[s+dt,1]+=peak*frac; st[s+dt,2]+=peak*frac*0.8; st[s+dt,6]+=peak*frac*0.5
    mark(s,DRIFT_LEN,'gradual_drift')

stream_temp = st.astype(np.float32)

# Power D=3
vm_sig2=np.zeros(T)
for vs in range(VM_INTERVAL,T,VM_INTERVAL+np.random.randint(-15,15)):
    vm_sig2[vs:min(vs+3,T)]=1.0
sp=np.zeros((T,3),dtype=np.float64)
sp[:,0]=POWER_BASE[0]+80.0*batch+vm_sig2*VM_DW+3.0*np.sin(2*np.pi*t_sec/3600)+np.random.normal(0,POWER_STD[0],T)
sp[:,1]=POWER_BASE[1]+78.0*batch+vm_sig2*VM_DW*0.9+3.0*np.sin(2*np.pi*t_sec/3600+0.5)+np.random.normal(0,POWER_STD[1],T)
sp[:,2]=POWER_BASE[2]+72.0*batch+vm_sig2*VM_DW*0.85+3.0*np.sin(2*np.pi*t_sec/3600+1.0)+np.random.normal(0,POWER_STD[2],T)
in_b=False; bs2=0
for idx in range(T):
    if labels[idx]==1 and not in_b:  bs2=idx; in_b=True
    elif labels[idx]==0 and in_b:
        sp[bs2:idx]*=np.random.uniform(1.5,2.5); in_b=False
if in_b: sp[bs2:]*=np.random.uniform(1.5,2.5)
stream_power = np.clip(sp,50,900).astype(np.float32)

anom_counts={k:sum(1 for m in anomaly_meta if m['type']==k)
             for k in ['clean_burst','overlap_burst','gradual_drift']}
print(f'Temperature: {stream_temp.shape}  Power: {stream_power.shape}')
print(f'Labels:      {labels.shape}  anomaly rate: {labels.mean()*100:.1f}%')
print(f'Events:      {anom_counts}')
print(f'Generated in {time.time()-t0:.1f}s')


# ---- notebook cell 6 (SAVE_DIR now comes from --out) ----
SAVE_DIR = ARGS.out
os.makedirs(SAVE_DIR, exist_ok=True)

def save(name, data):
    path = os.path.join(SAVE_DIR, name)
    if name.endswith('.npy'):
        if isinstance(data, np.ndarray):
            np.save(path, data)
        else:
            np.save(path, np.array(data, dtype=object), allow_pickle=True)
    elif name.endswith('.csv'):
        data.to_csv(path, index=False)
    elif name.endswith('.json'):
        with open(path,'w') as f: json.dump(data, f, indent=2)
    kb = os.path.getsize(path)//1024
    print(f'  Saved  {name:<42} {kb:>5} KB  ->  {path}')
    return path

ts = pd.date_range('2026-01-06', periods=T2_STEPS, freq='10s')

# Temperature CSV
df_temp = pd.DataFrame(stream_temp, columns=TEMP_NAMES)
df_temp.insert(0,'timestamp',ts)
df_temp['label'] = labels
save('tier2_temperature_48h.csv', df_temp)

# Power CSV
df_power = pd.DataFrame(stream_power, columns=POWER_NAMES)
df_power.insert(0,'timestamp',ts)
df_power['label'] = labels
save('tier2_power_48h.csv', df_power)

# NumPy arrays
save('tier2_temperature_stream.npy', stream_temp)
save('tier2_power_stream.npy',        stream_power)
save('tier2_labels.npy',              labels)
save('tier2_anomaly_meta.npy',        anomaly_meta)

# Metadata JSON
meta_dict = {
    'duration_hours': T2_HOURS,
    'total_steps': int(T2_STEPS),
    'sample_interval_s': T2_SAMPLE_S,
    'temperature_sensors': TEMP_NAMES,
    'power_sensors': POWER_NAMES,
    'anomaly_rate_pct': float(labels.mean()*100),
    'anomaly_counts': {k:int(v) for k,v in anom_counts.items()},
    'physics': {'tau_steps':int(TAU),'tau_s':int(TAU*T2_SAMPLE_S),
                'diurnal_amp_C':DIURNAL_AMP,'crac_amp_C':CRAC_AMP,
                'crac_period_s':CRAC_PERIOD,'batch_period_s':BATCH_PERIOD,
                'fan_lag_steps':FAN_LAG},
    'v3_compatible': True,
    'anomaly_events': anomaly_meta
}
save('tier2_metadata.json', meta_dict)

# Verify every file exists
print()
FILE_LIST = [
    'tier2_temperature_48h.csv',
    'tier2_power_48h.csv',
    'tier2_temperature_stream.npy',
    'tier2_power_stream.npy',
    'tier2_labels.npy',
    'tier2_anomaly_meta.npy',
    'tier2_metadata.json',
]
all_ok = True
for f in FILE_LIST:
    path = os.path.join(SAVE_DIR, f)
    ok = os.path.exists(path)
    if not ok: all_ok = False
    print('  OK  ' if ok else '  MISSING  ', f)
print()
print('ALL FILES SAVED:', all_ok)

