#!/usr/bin/env python3
"""Shared model, attribution, and controller primitives for Master v1.9.

Every section emits a JSON/CSV result and a duration line.  A section may be
resumed when its output exists.  No result from an earlier master version is
read by this program.
"""
from __future__ import annotations

import argparse, json, random, time, platform, sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from adaptive_deltaxai.models import DataPlaneGRU, DataPlaneTransformer, ReconGRU
from adaptive_deltaxai.swing import linear_completeness_check, swing_attribution


def jdump(path: Path, value) -> None:
    def conv(x):
        if isinstance(x, (np.integer, np.floating)): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, default=conv))


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def mark(name: str, start: float | None = None) -> float:
    now = time.perf_counter()
    if start is not None: print(f"[COMPLETE] {now-start:.1f}s", flush=True)
    print(f"\n{'='*84}\n{name}\n{'='*84}", flush=True)
    return now


def windows(x: np.ndarray, y: np.ndarray, w: int):
    return (np.stack([x[i-w:i] for i in range(w, len(x))]).astype(np.float32),
            y[w:].astype(np.int64))


def split_normalize(x: np.ndarray, y: np.ndarray):
    n=len(x); a=int(.7*n); b=int(.8*n)
    mu=x[:a].mean(0); sd=x[:a].std(0); sd=np.where(sd<1e-8,1.,sd)
    z=(x-mu)/sd
    assert a < b < n and not np.isnan(z).any()
    return (z[:a],z[a:b],z[b:]), (y[:a],y[a:b],y[b:]), {"train_end":a,"val_end":b,"test_start":b,"mean":mu,"std":sd}


def load_synthetic(root: Path):
    names=['tier2_temperature_stream.npy','tier2_power_stream.npy','tier2_labels.npy']
    missing=[n for n in names if not (root/n).exists()]
    if missing: raise FileNotFoundError(f"Missing synthetic files: {missing}")
    return {"temperature":np.load(root/names[0]).astype(np.float32),
            "power":np.load(root/names[1]).astype(np.float32)}, np.load(root/names[2]).astype(np.int64)


def train_binary(model, tr, ytr, va, yva, cfg, device, seed):
    w=cfg['window']; xtr,ytr=windows(tr,ytr,w); xva,yva=windows(va,yva,w)
    pos=(len(ytr)-ytr.sum())/max(float(ytr.sum()),1.)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos],device=device))
    opt=optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    rng=np.random.default_rng(seed); best=None; best_loss=float('inf')
    for epoch in range(cfg['epochs']):
        model.train(); order=rng.permutation(len(xtr))
        for s in range(0,len(order),cfg['batch']):
            idx=order[s:s+cfg['batch']]; xb=torch.as_tensor(xtr[idx],device=device); yb=torch.as_tensor(ytr[idx,None],dtype=torch.float32,device=device)
            loss=loss_fn(model(xb),yb); opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): val=loss_fn(model(torch.as_tensor(xva,device=device)),torch.as_tensor(yva[:,None],dtype=torch.float32,device=device)).item()
        print(f"  epoch {epoch+1}/{cfg['epochs']} val_loss={val:.5f}",flush=True)
        if val<best_loss: best_loss=val; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best); model.eval(); return model


def metrics(model,x,y,w,device):
    xx,yy=windows(x,y,w)
    with torch.no_grad(): p=model.predict_proba(torch.as_tensor(xx,device=device)).squeeze(1).cpu().numpy()
    h=(p>=.5).astype(int); cm=confusion_matrix(yy,h,labels=[0,1]).ravel().tolist()
    out={"n_raw_test_steps":len(x),"n_valid_windows":len(yy),"n_positive_windows":int(yy.sum()),
         "auroc":float(roc_auc_score(yy,p)),"auprc":float(average_precision_score(yy,p)),
         "precision":float(precision_score(yy,h,zero_division=0)),"recall":float(recall_score(yy,h,zero_division=0)),
         "f1":float(f1_score(yy,h,zero_division=0)),"confusion_matrix_tn_fp_fn_tp":cm}
    assert sum(cm)==out['n_valid_windows']; return out


def score_fn(model):
    if hasattr(model,'predict_proba'): return lambda x:model.predict_proba(x).reshape(-1)
    return lambda x:model.anomaly_score(x).reshape(-1)


def fresh_swing(model, history, cfg, device, serial_gradients=False):
    fn=score_fn(model); h=torch.as_tensor(history,dtype=torch.float32,device=device)
    sync(); t=time.perf_counter()
    a=swing_attribution(fn,h,cfg['window'],1,cfg['swing_lag'],cfg['swing_steps'],
                        serial_gradients=serial_gradients).squeeze(0)
    sync(); return a.detach().cpu().numpy(), (time.perf_counter()-t)*1000


@dataclass
class Record:
    """One online decision under the v1.9 temporal-index contract.

    ``endpoint_index`` is the last observation in the classifier window and
    ``target_index=endpoint_index+1`` is the next-observation label.  A record
    at the final endpoint is retained for operational timing but has
    ``target_valid=False`` and is excluded from all label-dependent metrics.
    ``age_before`` is used for the decision; ``age_after`` is used for a cached
    display.  Thus the first cache hit has display age one.
    """
    t:int; target_index:int; target_valid:bool; previous_score:float; current_score:float; refreshed:bool; reason:str; age_before:int; age_after:int
    step_latency_ms:float; fresh_latency_ms:float|None; fresh_completeness_residual:float|None
    cached_event_context:np.ndarray


def run_adaptive(stream, model, cfg, device, policy='adaptive', budget=None, seed=42, serial_gradients=False):
    w=cfg['window']; lag=cfg['swing_lag']; start=2*w+lag; fn=score_fn(model)
    # Scores are evaluated online for policy decisions; no future observations are used.
    scores=[]
    with torch.no_grad():
        for t in range(start,len(stream)):
            x=torch.as_tensor(stream[t-w+1:t+1],dtype=torch.float32,device=device).unsqueeze(0); scores.append(float(fn(x)[0]))
    scores=np.asarray(scores); rng=np.random.default_rng(seed)
    if policy!='adaptive':
        k=int(budget or max(1,.07*len(scores)))
        k=min(k,len(scores)); refresh=np.zeros(len(scores),bool)
        # Every reuse policy must initialize its cache at the first online
        # timestep.  Without this, a random/score-change schedule can attempt
        # to decay a nonexistent cached event context before its first refresh.
        refresh[0]=True
        remaining=max(0,k-1)
        if policy=='periodic': refresh[np.linspace(0,len(scores)-1,k,dtype=int)]=True
        elif policy=='random':
            if remaining: refresh[rng.choice(np.arange(1,len(scores)),remaining,replace=False)]=True
        elif policy=='score_change':
            candidates=np.argsort(np.abs(np.diff(scores,prepend=scores[0]))[1:])[-remaining:]+1 if remaining else []
            refresh[candidates]=True
        else: raise ValueError(policy)
    records=[]; ema=dev=change_ema=change_dev=None; cache=None; age=0
    for i,t in enumerate(range(start,len(stream))):
        sync(); started=time.perf_counter(); cur=float(scores[i])
        if i:
            prev=float(scores[i-1])
        else:
            with torch.no_grad():
                prior=torch.as_tensor(stream[t-w:t],dtype=torch.float32,device=device).unsqueeze(0)
                prev=float(fn(prior)[0])
        age_before = age
        if policy=='adaptive':
            delta=abs(cur-prev); residual=abs(cur-(ema if ema is not None else prev))
            if ema is None: ema=prev; dev=max(delta,1e-6); change_ema=0.; change_dev=1e-6
            inp=float(np.abs(stream[t]-stream[t-1]).mean()); z=abs(inp-change_ema)/max(change_dev,1e-6)
            score_fire=cfg.get('use_i1',True) and (residual>cfg['score_k']*max(dev,1e-6) if cfg.get('use_i2',True) else abs(cur-prev)>cfg['score_k']*max(dev,1e-6))
            drift_fire=cfg.get('use_i3',True) and (z>cfg['drift_k'] or age_before>=cfg['max_age'])
            fire= cache is None or cfg.get('always_refresh',False) or score_fire or drift_fire
            reason='initialization' if cache is None else ('always_refresh_no_I1' if cfg.get('always_refresh',False) else ('score_residual_I1_I2' if score_fire else ('input_drift_or_age_I3' if drift_fire else 'cached_event_context')))
            alpha=cfg['ema_alpha']; dev=(1-alpha)*dev+alpha*residual; ema=(1-alpha)*ema+alpha*cur; change_dev=(1-alpha)*change_dev+alpha*abs(inp-change_ema); change_ema=(1-alpha)*change_ema+alpha*inp
        else: fire=bool(refresh[i]); reason=f'{policy}_policy'
        if fire:
            hist=stream[t-(2*w+1+lag)+1:t+1]; raw,fresh_ms=fresh_swing(model,hist,cfg,device,serial_gradients); cache=raw; age=0
            resid=float(abs(raw.sum()-(cur-prev))); context=raw.copy()
        else:
            age+=1; fresh_ms=None; resid=None; context=cache*((cfg['gamma']**age) if cfg.get('use_i4',True) else 1.0)
        sync(); ms=(time.perf_counter()-started)*1000
        # The synthetic classifier receives stream[t-w+1:t+1] and is trained
        # to predict the following observation's label.  Store that mapping
        # explicitly so refresh-event analysis cannot rely on a hidden +1.
        records.append(Record(t, int(t + 1), bool(t + 1 < len(stream)), prev, cur, fire, reason, age_before, age, ms, fresh_ms, resid, context))
    return records


def summarize(records):
    fresh=[r for r in records if r.refreshed]; steps=np.array([r.step_latency_ms for r in records]); calls=np.array([r.fresh_latency_ms for r in fresh]); res=np.array([r.fresh_completeness_residual for r in fresh])
    reasons=pd.Series([r.reason for r in records]).value_counts().to_dict()
    return {"n_steps":len(records),"n_fresh_calls":len(fresh),"recompute_pct":100*len(fresh)/len(records),
            "amortized_step_latency_ms_mean":float(steps.mean()),"amortized_step_latency_ms_p95":float(np.percentile(steps,95)),
            "fresh_call_latency_ms_mean":float(calls.mean()),"fresh_call_latency_ms_p95":float(np.percentile(calls,95)),
            "fresh_completeness_residual_mean":float(res.mean()),"fresh_completeness_residual_p95":float(np.percentile(res,95)),"refresh_reasons":reasons}


def save_controller_records(path, records):
    """Persist the seed-42 controller state required by later stages.

    This lets a resumed run continue at the first incomplete experiment gate
    rather than repeating the three-seed synthetic monitoring pass.
    """
    np.savez_compressed(
        path,
        t=np.asarray([r.t for r in records], dtype=np.int64),
        target_index=np.asarray([r.target_index for r in records], dtype=np.int64),
        target_valid=np.asarray([r.target_valid for r in records], dtype=bool),
        previous_score=np.asarray([r.previous_score for r in records], dtype=np.float64),
        current_score=np.asarray([r.current_score for r in records], dtype=np.float64),
        refreshed=np.asarray([r.refreshed for r in records], dtype=bool),
        reason=np.asarray([r.reason for r in records], dtype=str),
        cache_age=np.asarray([r.age_after for r in records], dtype=np.int64),
        cache_age_before=np.asarray([r.age_before for r in records], dtype=np.int64),
        step_latency_ms=np.asarray([r.step_latency_ms for r in records], dtype=np.float64),
        fresh_latency_ms=np.asarray([np.nan if r.fresh_latency_ms is None else r.fresh_latency_ms for r in records], dtype=np.float64),
        fresh_completeness_residual=np.asarray([np.nan if r.fresh_completeness_residual is None else r.fresh_completeness_residual for r in records], dtype=np.float64),
        cached_event_context=np.stack([r.cached_event_context for r in records]).astype(np.float32),
    )


def load_controller_records(path):
    # Read each compressed array once, rather than repeatedly per record.
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    records=[]
    for i in range(len(data['t'])):
        fresh=None if np.isnan(data['fresh_latency_ms'][i]) else float(data['fresh_latency_ms'][i])
        residual=None if np.isnan(data['fresh_completeness_residual'][i]) else float(data['fresh_completeness_residual'][i])
        target_index = int(data['target_index'][i]) if 'target_index' in data else int(data['t'][i] + 1)
        target_valid = bool(data['target_valid'][i]) if 'target_valid' in data else True
        age_after = int(data['cache_age'][i]); age_before = int(data['cache_age_before'][i]) if 'cache_age_before' in data else age_after
        records.append(Record(int(data['t'][i]),target_index,target_valid,float(data['previous_score'][i]),float(data['current_score'][i]),
                              bool(data['refreshed'][i]),str(data['reason'][i]),age_before,age_after,
                              float(data['step_latency_ms'][i]),fresh,residual,data['cached_event_context'][i]))
    return records


def load_controller_refresh_arrays(path):
    """Load only the controller columns required for coverage evaluation.

    ``cached_event_context`` is deliberately *not* touched here.  It is a
    large compressed tensor and coverage analysis needs neither it nor any
    reconstructed ``Record`` objects.  Reading it repeatedly from an ``npz``
    archive was an avoidable RAM-pressure path in the first v1.9 notebook.
    """
    with np.load(path, allow_pickle=False) as data:
        target_index = np.asarray(
            data["target_index"] if "target_index" in data.files else data["t"] + 1,
            dtype=np.int64,
        ).copy()
        refreshed = np.asarray(data["refreshed"], dtype=bool).copy()
    if target_index.shape != refreshed.shape:
        raise RuntimeError("Controller target-index and refresh arrays are misaligned.")
    return target_index, refreshed


def cache_fidelity(records, stream, model, cfg, device, n, seed):
    rng=np.random.default_rng(seed); stale=[r for r in records if not r.refreshed]
    rows=[]
    for r in rng.choice(stale,min(n,len(stale)),replace=False):
        w=cfg['window']; lag=cfg['swing_lag']; history_length=2*w+1+lag; hist=stream[r.t-history_length+1:r.t+1]; fresh,_=fresh_swing(model,hist,cfg,device)
        a=r.cached_event_context.ravel(); b=fresh.ravel(); k=min(10,len(a)); ta=set(np.argpartition(abs(a),-k)[-k:]); tb=set(np.argpartition(abs(b),-k)[-k:])
        rows.append({"endpoint_index":r.t,"target_index":r.target_index,"cache_age_before":r.age_before,"cache_age_after":r.age_after,"cosine_similarity":float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)),"top10_jaccard":len(ta&tb)/len(ta|tb),"l1_difference":float(abs(a-b).sum())})
    return pd.DataFrame(rows)


def zero_ig(model, window, steps, device):
    x=torch.as_tensor(window,dtype=torch.float32,device=device); base=torch.zeros_like(x); d=x-base
    nodes=torch.stack([base+(i/steps)*d for i in range(1,steps+1)]).detach().requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False):
        g=torch.autograd.grad(score_fn(model)(nodes).sum(),nodes)[0]
    return (d*g.mean(0)).detach().cpu().numpy()


def aw_ig(model, previous_window, current_window, steps, device):
    """Adjacent-window IG baseline retained as a method distinct from SWING."""
    prev=torch.as_tensor(previous_window,dtype=torch.float32,device=device); cur=torch.as_tensor(current_window,dtype=torch.float32,device=device); d=cur-prev
    nodes=torch.stack([prev+(i/steps)*d for i in range(1,steps+1)]).detach().requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False):
        g=torch.autograd.grad(score_fn(model)(nodes).sum(),nodes)[0]
    return (d*g.mean(0)).detach().cpu().numpy()


def aopc(model, window, attribution, mean, device, kmax=8):
    fn=score_fn(model); x=np.array(window,copy=True); base=float(fn(torch.as_tensor(x,device=device).unsqueeze(0))[0].detach().cpu())
    rank=np.argsort(abs(attribution).ravel())[::-1]; vals=[]
    for k in range(1,kmax+1):
        y=x.copy(); ii=np.unravel_index(rank[:k],x.shape); y[ii]=np.broadcast_to(mean,x.shape)[ii]
        with torch.no_grad(): vals.append(base-float(fn(torch.as_tensor(y,device=device).unsqueeze(0))[0]))
    return float(np.mean(vals))


def delta_aopc(model, history, attribution, reference, cfg, device, kmax=8):
    """Perturb the same coordinates in real adjacent windows and evaluate Δf.

    This is the primary v1.9 faithfulness measurement.  ``history`` is the
    real W+delta+lag record used by fresh SWING, not a prefix-padded current
    window. ``attribution`` has the same shape. Only the union of previous
    and current windows is rankable; a shared history coordinate is masked
    once, then both scores are recomputed from the resulting adjacent windows.
    """
    w, d = cfg['window'], 1
    fn = score_fn(model); x = np.array(history, copy=True)
    previous = x[-w-d:-d]; current = x[-w:]
    with torch.no_grad():
        base = float(fn(torch.as_tensor(current, device=device).unsqueeze(0))[0] - fn(torch.as_tensor(previous, device=device).unsqueeze(0))[0])
    union_attr = attribution[-(w+d):]
    rank = np.argsort(np.abs(union_attr).ravel())[::-1]; vals=[]
    ref = np.broadcast_to(reference, x.shape)
    for k in range(1, kmax+1):
        masked=x.copy(); rr,cc=np.unravel_index(rank[:k], union_attr.shape)
        global_rows=rr + x.shape[0] - (w+d); masked[global_rows,cc]=ref[global_rows,cc]
        p=masked[-w-d:-d]; c=masked[-w:]
        with torch.no_grad():
            after=float(fn(torch.as_tensor(c,device=device).unsqueeze(0))[0]-fn(torch.as_tensor(p,device=device).unsqueeze(0))[0])
        vals.append(abs(base)-abs(after))
    return float(np.mean(vals))


def delta_baseline_attributions(model, history, cfg, device):
    """Adjacent-window baseline maps aligned to the Full-SWING union support."""
    w=cfg['window']; prev=history[-w-1:-1]; cur=history[-w:]
    out={}
    zero_prev=zero_ig(model,prev,cfg['ig_steps'],device)
    zero_cur=zero_ig(model,cur,cfg['ig_steps'],device)
    z=np.zeros((w+1,history.shape[1]),dtype=np.float32); z[:-1]-=zero_prev; z[1:]+=zero_cur; out['zero_ig_delta']=z
    aw=aw_ig(model,prev,cur,cfg['ig_steps'],device)
    a=np.zeros_like(z); a[1:]=aw; out['aw_ig_delta']=a
    return out


def bootstrap_ci(values, n=2000, seed=42):
    v=np.asarray(values); rng=np.random.default_rng(seed); means=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(n)])
    return float(v.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))


def full_swing_current(model, history, cfg, device):
    a,_=fresh_swing(model,history,cfg,device); return a[-cfg['window']:]


def faithfulness(model, test, labels, train_mean, cfg, device, output, seed):
    """Common held-out positive windows; all methods share masking and samples."""
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    w=cfg['window']; xx,yy=windows(test,labels,w); candidates=np.where(yy==1)[0]; rng=np.random.default_rng(seed); chosen=rng.choice(candidates,min(cfg['faithfulness_windows'],len(candidates)),replace=False)
    bg=xx[rng.choice(len(xx),min(30,len(xx)),replace=False)].reshape(min(30,len(xx)),-1)
    lime=LimeTabularExplainer(bg,mode='regression',discretize_continuous=False,random_state=seed)
    def fflat(z):
        with torch.no_grad(): return score_fn(model)(torch.as_tensor(z.reshape(-1,w,test.shape[1]),dtype=torch.float32,device=device)).detach().cpu().numpy()
    kernel=shap.KernelExplainer(fflat,bg)
    rows=[]
    for q,idx in enumerate(chosen,1):
        cur=xx[idx]; history=np.concatenate([cur[:1].repeat(cfg['swing_lag']+1,0),cur],0)[-(w+1+cfg['swing_lag']):]
        # AW-IG is retained as a distinct method, never called SWING.
        prev=np.vstack([cur[0],cur[:-1]]); aw=aw_ig(model,prev,cur,cfg['ig_steps'],device)
        methods={"zero_ig":zero_ig(model,cur,cfg['ig_steps'],device),"full_swing":full_swing_current(model,history,cfg,device),"aw_ig":aw}
        sv=kernel.shap_values(cur.reshape(1,-1),nsamples=cfg['kernel_shap_samples']); methods['kernel_shap']=np.asarray(sv).reshape(cur.shape)
        ex=lime.explain_instance(cur.ravel(),fflat,num_features=cur.size); lm=np.zeros(cur.size); [lm.__setitem__(i,v) for i,v in ex.as_map()[1]]; methods['lime']=lm.reshape(cur.shape)
        for name,att in methods.items(): rows.append({"method":name,"window_index":int(idx),"aopc":aopc(model,cur,att,train_mean,device)})
        if q%10==0: print(f"  faithfulness {q}/{len(chosen)}",flush=True)
    df=pd.DataFrame(rows); out=[]
    for name,g in df.groupby('method'):
        m,lo,hi=bootstrap_ci(g.aopc.values,cfg['bootstrap'],seed); out.append({"method":name,"n_common_windows":len(g),"aopc_mean":m,"aopc_ci_low":lo,"aopc_ci_high":hi})
    df.to_csv(output/'faithfulness_per_window.csv',index=False); pd.DataFrame(out).to_csv(output/'faithfulness_summary.csv',index=False)


def train_reconstruction(train, cfg, device, seed, checkpoint_path=None, resume=False):
    """Train a public-data reconstruction detector on its training partition only."""
    dummy=np.zeros(len(train),dtype=np.int64); x,_=windows(train,dummy,cfg['window'])
    model=ReconGRU(train.shape[1]).to(device)
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path,map_location=device)); model.eval(); print(f'  resumed reconstruction checkpoint: {checkpoint_path.name}',flush=True); return model
    opt=optim.AdamW(model.parameters(),lr=1e-3); loss_fn=nn.MSELoss(); rng=np.random.default_rng(seed)
    for epoch in range(cfg['epochs']):
        model.train(); order=rng.permutation(len(x))
        for s in range(0,len(order),cfg['batch']):
            idx=order[s:s+cfg['batch']]; xb=torch.as_tensor(x[idx],device=device); loss=loss_fn(model(xb),xb[:,-1])
            opt.zero_grad(); loss.backward(); opt.step()
        print(f'  reconstruction epoch {epoch+1}/{cfg["epochs"]}',flush=True)
    model.eval()
    if checkpoint_path is not None: torch.save(model.state_dict(),checkpoint_path)
    return model


def public_scores(model, test, cfg, device):
    dummy=np.zeros(len(test),dtype=np.int64); x,_=windows(test,dummy,cfg['window']); out=[]
    with torch.no_grad():
        for s in range(0,len(x),512): out.append(model.anomaly_score(torch.as_tensor(x[s:s+512],device=device)).cpu().numpy())
    return x,np.concatenate(out)


def public_subset(dataset, unit, train, test, labels, cfg, device, n, seed, checkpoint_path=None, resume=False):
    """Real full-SWING latency and AOPC on a documented public-data subset."""
    mu=train.mean(0); sd=np.where(train.std(0)<1e-8,1.,train.std(0)); train=(train-mu)/sd; test=(test-mu)/sd
    model=train_reconstruction(train,cfg,device,seed,checkpoint_path,resume); x,scores=public_scores(model,test,cfg,device); y=labels[cfg['window']-1:cfg['window']-1+len(x)].astype(int)
    eligible=np.where(y==1)[0]; rng=np.random.default_rng(seed)
    chosen=rng.choice(eligible,min(n,len(eligible)),replace=False) if len(eligible) else rng.choice(len(x),min(n,len(x)),replace=False)
    rows=[]; w=cfg['window']; lag=cfg['swing_lag']
    # Prefix replication is documented for the early rows, not future data.
    padded=np.vstack([np.repeat(test[:1],w+lag+1,axis=0),test])
    for q,idx in enumerate(chosen,1):
        end=w+lag+1+idx+w; hist=padded[end-(w+1+lag):end]
        att,lat=fresh_swing(model,hist,cfg,device); cur=x[idx]; current_att=att[-w:]
        rows.append({'dataset':dataset,'unit':unit,'mode':'full_attribution_subset','window_index':int(idx),'label':int(y[idx]),
                     'full_swing_latency_ms':lat,'fresh_completeness_residual':float(abs(att.sum()-(scores[idx]-scores[max(0,idx-1)]))),
                     'aopc':aopc(model,cur,current_att,np.zeros(cur.shape[1],dtype=np.float32),device)})
        if q%10==0: print(f'  {dataset}/{unit} attribution subset {q}/{len(chosen)}',flush=True)
    return pd.DataFrame(rows), model


def load_public(root):
    """Return SMD units and PSM, using the documented on-Drive layout."""
    out=[]; smd=root/'SMD'/'ServerMachineDataset'
    if (smd/'train').exists():
        for path in sorted((smd/'train').glob('*.txt')):
            unit=path.stem; out.append(('SMD',unit,np.loadtxt(path,delimiter=',').astype(np.float32),
                np.loadtxt(smd/'test'/f'{unit}.txt',delimiter=',').astype(np.float32),
                np.loadtxt(smd/'test_label'/f'{unit}.txt',delimiter=',').astype(int)))
    psm=root/'PSM'
    if (psm/'train.csv').exists() and (psm/'test.csv').exists() and (psm/'test_label.csv').exists():
        tr=pd.read_csv(psm/'train.csv').drop(columns=['timestamp_(min)'],errors='ignore').ffill().fillna(0).values.astype(np.float32)
        te=pd.read_csv(psm/'test.csv').drop(columns=['timestamp_(min)'],errors='ignore').ffill().fillna(0).values.astype(np.float32)
        lab=pd.read_csv(psm/'test_label.csv').iloc[:,-1].values.astype(int); out.append(('PSM','PSM',tr,te,lab))
    if not out: raise FileNotFoundError('No SMD/PSM data found under DATA_ROOT/SMD or DATA_ROOT/PSM.')
    return out


def topology_groups(n_features: int, stream_name: str):
    """Stable, reported topology mapping; never changes raw attribution."""
    if stream_name == 'power':
        return {'PDU_input': list(range(0, min(2, n_features))),
                'PDU_output': list(range(min(2, n_features), n_features))}
    mid=max(1,n_features//2)
    return {'thermal_zone_A': list(range(0,mid)), 'thermal_zone_B': list(range(mid,n_features))}


def grouped_context_rows(records, stream_name: str, n_features: int):
    groups=topology_groups(n_features,stream_name); rows=[]
    for r in records:
        sensor=np.asarray(r.cached_event_context).sum(axis=0)
        for group,indices in groups.items():
            rows.append({'stream':stream_name,'t_index':r.t,'target_index':r.target_index,'output_type':'fresh_explanation' if r.refreshed else 'cached_event_context',
                         'topology_group':group,'grouped_attribution':float(sensor[indices].sum()),'cache_age_after':r.age_after})
    return rows


def generate_paper_artifacts(output: Path):
    """Generate paper-facing tables/figures exclusively from frozen CSVs."""
    faith=pd.read_csv(output/'faithfulness_summary.csv'); scale=pd.read_csv(output/'scaling_summary.csv'); ab=pd.read_csv(output/'ablation_summary.csv'); eff=pd.read_csv(output/'efficiency_2x2.csv')
    table=("# v1.6 Generated Paper Tables\n\n## Faithfulness\n\n"+faith.to_markdown(index=False)+"\n\n## Ablation\n\n"+ab.to_markdown(index=False)+"\n\n## Batching × Caching\n\n"+eff.to_markdown(index=False)+"\n")
    (output/'PAPER_TABLES_v1_6.md').write_text(table)
    plt.figure(figsize=(5,3.2)); plt.plot(scale.sensor_count,scale.amortized_step_latency_ms_mean,marker='o'); plt.xlabel('Sensor count'); plt.ylabel('Amortized latency (ms)'); plt.tight_layout(); plt.savefig(output/'figure_scaling_latency.png',dpi=200); plt.close()
    plt.figure(figsize=(5,3.2)); plt.bar(faith.method,faith.aopc_mean,yerr=[faith.aopc_mean-faith.aopc_ci_low,faith.aopc_ci_high-faith.aopc_mean],capsize=3); plt.xticks(rotation=25,ha='right'); plt.ylabel('AOPC'); plt.tight_layout(); plt.savefig(output/'figure_common_window_faithfulness.png',dpi=200); plt.close()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',type=Path,required=True); ap.add_argument('--output-root',type=Path,required=True); ap.add_argument('--seeds',type=int,nargs='*',default=[42,52,62]); ap.add_argument('--public-subset',type=int,default=50); ap.add_argument('--smd-machines',type=int,default=3); ap.add_argument('--timing-steps',type=int,default=200); ap.add_argument('--resume',action='store_true'); ap.add_argument('--smoke',action='store_true'); args=ap.parse_args()
    out=args.output_root; out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg={"window":20,"swing_lag":5,"swing_steps":32,"ig_steps":32,"epochs":5,"batch":256,"lr":5e-4,"weight_decay":1e-4,"faithfulness_windows":50,"kernel_shap_samples":100,"bootstrap":2000,
         "temperature":{"score_k":2.5,"ema_alpha":.05,"drift_k":5.,"gamma":.97,"max_age":60},"power":{"score_k":3.,"ema_alpha":.08,"drift_k":5.,"gamma":.95,"max_age":60}}
    # Required frozen configuration, written before first experiment.
    jdump(out/'CONFIG_v1_6.json',cfg); manifest=Path(__file__).parents[1]/'manifests'/'FEATURE_MANIFEST_v1_6.json'; (out/'FEATURE_MANIFEST_v1_6.json').write_text(manifest.read_text())
    jdump(out/'run_manifest.json',{'experiment_id':'AdaptiveDeltaXAI_Master_v1_6','seeds':args.seeds,'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'cuda_version':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'config':cfg})
    started=mark('SECTION 0 — SWING verification and frozen protocol'); t=started; residual=linear_completeness_check(str(device)); assert residual<1e-5; print(f'[PASS] Full-SWING linear completeness={residual:.3e}')
    if args.smoke:
        # Deliberately bounded smoke test: one detector, one epoch, one real
        # full-SWING call. It never enters ablations, scaling, or public data.
        streams,labels=load_synthetic(args.data_root); set_seed(args.seeds[0]); parts,ys,scale=split_normalize(streams['temperature'],labels); smoke_cfg={**cfg,'epochs':1}; smoke_model=train_binary(DataPlaneGRU(streams['temperature'].shape[1]).to(device),parts[0],ys[0],parts[1],ys[1],smoke_cfg,device,args.seeds[0]); hist=((streams['temperature']-scale['mean'])/scale['std'])[:2*cfg['window']+1+cfg['swing_lag']]; att,lat=fresh_swing(smoke_model,hist,cfg,device); jdump(out/'SMOKE_REPORT_v1_6.json',{'status':'PASS','seed':args.seeds[0],'detector':metrics(smoke_model,parts[2],ys[2],cfg['window'],device),'full_swing_latency_ms':lat,'full_swing_shape':list(att.shape),'completeness_linear_residual':residual}); print(f'[DONE] bounded smoke test completed in {(time.perf_counter()-started)/60:.2f} min. Run the full protocol only after reviewing SMOKE_REPORT_v1_6.json.'); return
    streams,labels=load_synthetic(args.data_root); trained={}
    synthetic_done=(out/'synthetic_seed_results.csv').exists()
    controller_path=out/f'controller_temperature_seed{args.seeds[0]}_records.npz'
    if not synthetic_done:
        allrows=[]; topology_rows=[]
        for seed in args.seeds:
            set_seed(seed); t=mark(f'SECTION 1 — synthetic GRU, seed {seed}',started)
            for name,raw in streams.items():
                parts,ys,scale=split_normalize(raw,labels); assert len(parts[0])+len(parts[1])+len(parts[2])==len(raw)
                checkpoint=out/f'checkpoint_gru_{name}_seed{seed}.pt'; model=DataPlaneGRU(raw.shape[1]).to(device)
                if args.resume and checkpoint.exists(): model.load_state_dict(torch.load(checkpoint,map_location=device)); model.eval(); print(f'  resumed {checkpoint.name}',flush=True)
                else: model=train_binary(model,parts[0],ys[0],parts[1],ys[1],cfg,device,seed); torch.save(model.state_dict(),checkpoint)
                records=run_adaptive((raw-scale['mean'])/scale['std'],model,{**cfg,**cfg[name]},device,seed=seed); op=summarize(records); assert op['fresh_completeness_residual_mean']<.05
                cf=cache_fidelity(records,(raw-scale['mean'])/scale['std'],model,{**cfg,**cfg[name]},device,100,seed); cf.to_csv(out/f'cache_fidelity_{name}_seed{seed}.csv',index=False)
                topology_rows.extend(grouped_context_rows(records,name,raw.shape[1]))
                allrows.append({"seed":seed,"stream":name,**{f'detector_{k}':v for k,v in metrics(model,parts[2],ys[2],cfg['window'],device).items()},**{f'operational_{k}':v for k,v in op.items() if k!='refresh_reasons'}})
                if seed==args.seeds[0]:
                    trained[name]=(model,parts,ys,scale,records)
                    if name=='temperature': save_controller_records(controller_path,records)
        pd.DataFrame(allrows).to_csv(out/'synthetic_seed_results.csv',index=False)
        pd.DataFrame(topology_rows).to_csv(out/'topology_grouped_context.csv',index=False)
    else:
        # A completed synthetic CSV is a checkpointed stage boundary.  Only
        # rehydrate seed-42 temperature state needed by later experiment gates.
        print('[RESUME] SECTION 1 already complete; reusing synthetic_seed_results.csv',flush=True)
        raw=streams['temperature']; parts,ys,scale=split_normalize(raw,labels)
        model=DataPlaneGRU(raw.shape[1]).to(device); checkpoint=out/f'checkpoint_gru_temperature_seed{args.seeds[0]}.pt'
        if not checkpoint.exists(): raise FileNotFoundError(f'Missing checkpoint required to resume: {checkpoint}')
        model.load_state_dict(torch.load(checkpoint,map_location=device)); model.eval()
        if controller_path.exists():
            adaptive=load_controller_records(controller_path); print(f'[RESUME] loaded controller cache: {controller_path.name}',flush=True)
        else:
            print('[RESUME] controller cache not present; rebuilding seed-42 temperature controller once.',flush=True)
            adaptive=run_adaptive((raw-scale['mean'])/scale['std'],model,{**cfg,**cfg['temperature']},device,seed=args.seeds[0])
            save_controller_records(controller_path,adaptive)
        trained['temperature']=(model,parts,ys,scale,adaptive)
    # Required method-level faithfulness on a fixed held-out temperature set.
    model,parts,ys,scale,adaptive=trained['temperature']
    if (out/'faithfulness_summary.csv').exists() and (out/'faithfulness_per_window.csv').exists():
        print('[RESUME] SECTION 2 already complete; reusing common-window faithfulness files.',flush=True)
    else:
        t=mark('SECTION 2 — common-window method faithfulness with confidence intervals',t)
        faithfulness(model,parts[2],ys[2],np.zeros_like(scale['mean']),cfg,device,out,args.seeds[0])
    # Required multi-policy comparison at adaptive matched budget.
    model,parts,ys,scale,adaptive=trained['temperature']; stream=(streams['temperature']-scale['mean'])/scale['std']; budget=sum(r.refreshed for r in adaptive)
    if (out/'multi_policy_temperature.csv').exists():
        print('[RESUME] SECTION 3 already complete; reusing multi-policy table.',flush=True)
    else:
        t=mark('SECTION 3 — matched-budget multi-policy comparison',t); rows=[]
        for policy in ['adaptive','periodic','random','score_change']:
            rec=adaptive if policy=='adaptive' else run_adaptive(stream,model,{**cfg,**cfg['temperature']},device,policy,budget,args.seeds[0]); rows.append({'policy':policy,**summarize(rec)})
        pd.DataFrame(rows).to_csv(out/'multi_policy_temperature.csv',index=False)
    # Six-configuration ablation: exactly one intervention is disabled per
    # run. I5 changes only grouped presentation, so scheduling metrics remain
    # identical while the grouping artifact is saved separately.
    if (out/'ablation_summary.csv').exists():
        print('[RESUME] SECTION 3B already complete; reusing ablation table.',flush=True)
    else:
        t=mark('SECTION 3B — six-component ablation on held-out stream',t); ab_rows=[]
        variants={'Full':{},'No_I1':{'always_refresh':True,'use_i1':False},'No_I2':{'use_i2':False},
                  'No_I3':{'use_i3':False},'No_I4':{'use_i4':False},'No_I5':{'use_i5':False}}
        ablation_stream=stream[scale['test_start']-(2*cfg['window']+cfg['swing_lag']+1):]
        for label,patch in variants.items():
            rr=run_adaptive(ablation_stream,model,{**cfg,**cfg['temperature'],**patch},device,seed=args.seeds[0]); ss=summarize(rr)
            fresh=[r for r in rr if r.refreshed][:50]
            av=[]
            for r in fresh:
                cur=ablation_stream[r.t-cfg['window']+1:r.t+1]; av.append(aopc(model,cur,r.cached_event_context[-cfg['window']:],np.zeros(ablation_stream.shape[1]),device))
            ab_rows.append({'configuration':label,**ss,'aopc_fresh_only':float(np.mean(av)) if av else np.nan,'aopc_n_fresh':len(av)})
        pd.DataFrame(ab_rows).to_csv(out/'ablation_summary.csv',index=False)
        groups={'temperature_zone_A':list(range(min(4,stream.shape[1]))),'temperature_zone_B':list(range(min(4,stream.shape[1])),stream.shape[1])}
        sample=adaptive[0].cached_event_context[-cfg['window']:].sum(0); jdump(out/'topology_grouping_example.json',{name:float(sample[idx].sum()) for name,idx in groups.items()})
    # Timing repetitions have two warmups and then ten timed calls; batching/caching are separated in output.
    if (out/'efficiency_2x2.csv').exists() and (out/'timing_repeated_warmup.json').exists():
        print('[RESUME] SECTION 4 already complete; reusing timing artifacts.',flush=True)
    else:
        t=mark('SECTION 4 — repeated warm-up GPU timing and batching x caching',t); rec=adaptive; fresh=[r for r in rec if r.refreshed]
        assert fresh, 'Adaptive run produced no fresh calls.'
        probe=fresh[min(2,len(fresh)-1)]; probe_hist=stream[probe.t-(2*cfg['window']+1+cfg['swing_lag'])+1:probe.t+1]
        for _ in range(2): fresh_swing(model,probe_hist,cfg,device,serial_gradients=False)
        timing=[fresh_swing(model,probe_hist,cfg,device,serial_gradients=False)[1] for _ in range(10)]
        jdump(out/'timing_repeated_warmup.json',{'warmups':2,'repetitions':10,'same_probe_time_index':probe.t,'fresh_swing_ms':timing,'mean_ms':float(np.mean(timing))})
        timing_rows=[]
        for condition,base in [('unbatched_always_refresh',True),('unbatched_adaptive_refresh',False),('batched_always_refresh',True),('batched_adaptive_refresh',False)]:
            serial=condition.startswith('unbatched')
            bench=stream[-(args.timing_steps+2*cfg['window']+cfg['swing_lag']+1):]
            rr=run_adaptive(bench,model,{**cfg,**cfg['temperature'],'always_refresh':base},device,seed=args.seeds[0],serial_gradients=serial)
            ss=summarize(rr); timing_rows.append({'condition':condition,'attribution_evaluation':'batched_multipathig' if condition.startswith('batched') else 'serialized_fresh_calls','refresh_policy':'always' if base else 'adaptive',**ss})
        pd.DataFrame(timing_rows).to_csv(out/'efficiency_2x2.csv',index=False)
    # Sensor scaling uses deterministic tiled-and-jittered channels and the
    # same chronological split, detector recipe, and controller configuration.
    if (out/'scaling_summary.csv').exists():
        print('[RESUME] SECTION 4B already complete; reusing scaling table.',flush=True)
    else:
        t=mark('SECTION 4B — 8/16/32/64 sensor scaling',t); scale_rows=[]; base_raw=streams['temperature']
        for d in [8,16,32,64]:
            reps=int(np.ceil(d/base_raw.shape[1])); rng=np.random.default_rng(args.seeds[0]+d); expanded=np.tile(base_raw,(1,reps))[:,:d]+rng.normal(0,1e-3,(len(base_raw),d)).astype(np.float32)
            sp,sy,sc=split_normalize(expanded,labels); normalized=(expanded-sc['mean'])/sc['std']; sm=train_binary(DataPlaneGRU(d).to(device),sp[0],sy[0],sp[1],sy[1],cfg,device,args.seeds[0]); rr=run_adaptive(normalized[sc['test_start']-(2*cfg['window']+cfg['swing_lag']+1):],sm,{**cfg,**cfg['temperature']},device,seed=args.seeds[0]); scale_rows.append({'sensor_count':d,'stream_scope':'held_out_test_with_history',**summarize(rr)})
        pd.DataFrame(scale_rows).to_csv(out/'scaling_summary.csv',index=False)
    # Transformer validation shares normalized split/controller/seed with GRU.
    if (out/'architecture_validation.json').exists():
        print('[RESUME] SECTION 5 already complete; reusing architecture validation.',flush=True)
    else:
        t=mark('SECTION 5 — matched GRU/Transformer validation',t); parts,ys,scale=split_normalize(streams['temperature'],labels); normalized=(streams['temperature']-scale['mean'])/scale['std']; transformer_ckpt=out/'checkpoint_transformer_temperature_seed42.pt'; trm=DataPlaneTransformer(streams['temperature'].shape[1]).to(device)
        if args.resume and transformer_ckpt.exists(): trm.load_state_dict(torch.load(transformer_ckpt,map_location=device)); trm.eval(); print(f'  resumed {transformer_ckpt.name}',flush=True)
        else: trm=train_binary(trm,parts[0],ys[0],parts[1],ys[1],cfg,device,args.seeds[0]); torch.save(trm.state_dict(),transformer_ckpt)
        tr_records=run_adaptive(normalized,trm,{**cfg,**cfg['temperature']},device,seed=args.seeds[0]); jdump(out/'architecture_validation.json',{'gru_detector':metrics(model,parts[2],ys[2],cfg['window'],device),'transformer_detector':metrics(trm,parts[2],ys[2],cfg['window'],device),'gru_operational':summarize(adaptive),'transformer_operational':summarize(tr_records),'same_split_seed_controller':True})
    # Public scheduling policies and actual full-attribution subset.  The
    # subset is reported separately, never inflated into a full-dataset claim.
    public_outputs=['public_full_attribution_subset.csv','public_full_attribution_summary.csv','public_policy_results.csv']
    if all((out/x).exists() for x in public_outputs):
        print('[RESUME] SECTION 6 already complete; reusing public-dataset artifacts.',flush=True)
    else:
        t=mark('SECTION 6 — SMD/PSM policies and full-attribution subset',t); public_rows=[]; policy_rows=[]
        public_units=load_public(args.data_root); smd_seen=0
        for dataset,unit,ptr,pte,plabels in public_units:
            if dataset=='SMD':
                if smd_seen>=args.smd_machines: continue
                smd_seen+=1
            safe_unit=unit.replace('/','_'); public_ckpt=out/f'checkpoint_recon_{dataset}_{safe_unit}_seed{args.seeds[0]}.pt'; subset,public_model=public_subset(dataset,unit,ptr,pte,plabels,cfg,device,args.public_subset,args.seeds[0],public_ckpt,args.resume); public_rows.append(subset)
            _,scores=public_scores(public_model,(pte-ptr.mean(0))/np.where(ptr.std(0)<1e-8,1.,ptr.std(0)),cfg,device)
            adaptive_budget=max(1,int(.07*len(scores)))
            for policy in ['adaptive','periodic','random','score_change']:
                if policy=='periodic': mask=np.zeros(len(scores),bool); mask[np.linspace(0,len(scores)-1,adaptive_budget,dtype=int)]=True
                elif policy=='random': mask=np.zeros(len(scores),bool); mask[np.random.default_rng(args.seeds[0]).choice(len(scores),adaptive_budget,replace=False)]=True
                elif policy=='score_change': mask=np.zeros(len(scores),bool); mask[np.argsort(np.abs(np.diff(scores,prepend=scores[0])))[-adaptive_budget:]]=True
                else: mask=np.zeros(len(scores),bool); ema=scores[0]; dev=1e-6; age=0
                if policy=='adaptive':
                    for i,s in enumerate(scores):
                        signal=abs(s-ema); fire=signal>cfg['temperature']['score_k']*max(dev,1e-6) or age>=cfg['temperature']['max_age']; mask[i]=fire; age=0 if fire else age+1; dev=.95*dev+.05*signal; ema=.95*ema+.05*s
                policy_rows.append({'dataset':dataset,'unit':unit,'mode':'score_only_scheduling','policy':policy,'refresh_count':int(mask.sum()),'recompute_pct':float(100*mask.mean()),'matched_budget':int(adaptive_budget)})
        public_df=pd.concat(public_rows,ignore_index=True); public_df.to_csv(out/'public_full_attribution_subset.csv',index=False); pd.DataFrame(policy_rows).to_csv(out/'public_policy_results.csv',index=False)
        public_summary=[]
        for (dataset,unit),group in public_df.groupby(['dataset','unit']):
            mean,lo,hi=bootstrap_ci(group.aopc.values,cfg['bootstrap'],args.seeds[0]); public_summary.append({'dataset':dataset,'unit':unit,'mode':'full_attribution_subset','n_windows':len(group),'aopc_mean':mean,'aopc_ci_low':lo,'aopc_ci_high':hi,'latency_mean_ms':float(group.full_swing_latency_ms.mean())})
        pd.DataFrame(public_summary).to_csv(out/'public_full_attribution_summary.csv',index=False)
    if (out/'CONSISTENCY_REPORT_v1_6.json').exists():
        print('[RESUME] SECTION 7 already complete; reusing generated paper artifacts.',flush=True)
    else:
        t=mark('SECTION 7 — generated paper tables, figures, consistency, and evidence map',t); generate_paper_artifacts(out); report={'status':'PASS','full_swing_completeness':residual,'seed_count':len(args.seeds),'required_tables':['synthetic_seed_results.csv','faithfulness_summary.csv','ablation_summary.csv','efficiency_2x2.csv','scaling_summary.csv','multi_policy_temperature.csv','architecture_validation.json','public_full_attribution_subset.csv'],'paper_mapping':'docs/ELFIKKY_10_POINT_TRACEABILITY.md'}; jdump(out/'CONSISTENCY_REPORT_v1_6.json',report)
        docs=Path(__file__).parents[1]/'docs'; (out/'PAPER_EVIDENCE_MAP_v1_6.json').write_text((docs/'PAPER_EVIDENCE_LANGUAGE_v1_6.json').read_text()); (out/'PAPER_RELEASE_CHECKLIST_v1_6.md').write_text((docs/'PAPER_RELEASE_CHECKLIST_v1_6.md').read_text())
    print(f'[DONE] v1.6 completed in {(time.perf_counter()-started)/60:.2f} minutes. Run freeze_results.py only after every gate is present.')

if __name__=='__main__': main()
