"""Interactive, self-contained HTML view of X sentiment vs price over time (no CDN, opens offline).

Two stacked panels on one time axis (price on its own axis, sentiment on its own),
a hover crosshair with a tooltip, range presets and the live n+0..n+5 forecast band.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--s1:#2a78d6;--pos:#2a78d6;--neg:#e34948;--band:rgba(42,120,214,.16);--border:rgba(11,11,11,.10)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--pos:#3987e5;--neg:#e66767;
--band:rgba(57,135,229,.22);--border:rgba(255,255,255,.10)}}
:root[data-theme="dark"]{--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--axis:#383835;--s1:#3987e5;--pos:#3987e5;--neg:#e66767;--band:rgba(57,135,229,.22);--border:rgba(255,255,255,.10)}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:14px system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--ink2);font-size:13px;margin:0 0 14px}
.demo{background:#fdf3ee;color:#7a2e10;border-left:3px solid #eb6834;padding:8px 12px;margin-bottom:12px;font-size:13px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:10px 12px}
.tile .k{color:var(--ink2);font-size:12px}.tile .v{font-size:20px;font-weight:600;margin-top:2px}
.bar{display:flex;gap:6px;margin:0 0 8px;flex-wrap:wrap}
.bar button{background:var(--surface);color:var(--ink2);border:1px solid var(--border);border-radius:7px;padding:5px 10px;font:inherit;font-size:12px;cursor:pointer}
.bar button[aria-pressed="true"]{color:var(--ink);font-weight:600;border-color:var(--ink2)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:8px 8px 4px;position:relative}
svg{display:block;width:100%;height:auto}
.lbl{fill:var(--muted);font-size:11px}.ttl{fill:var(--ink);font-size:12px;font-weight:600}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--border);border-radius:8px;
padding:8px 10px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.12);display:none;min-width:170px}
.tip b{display:block;margin-bottom:4px}.tip div{display:flex;justify-content:space-between;gap:14px;color:var(--ink2)}
.tip span{color:var(--ink);font-variant-numeric:tabular-nums}
.note{color:var(--muted);font-size:12px;margin-top:10px}
table{border-collapse:collapse;margin-top:14px;font-size:13px;width:100%;background:var(--surface);border-radius:10px;overflow:hidden}
td,th{padding:6px 10px;border-bottom:1px solid var(--grid);text-align:right;font-variant-numeric:tabular-nums}td:first-child,th:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}
</style></head><body><div class="wrap">
__DEMO__
<h1>__TITLE__</h1><p class="sub">__SUB__</p>
<div class="tiles" id="tiles"></div>
<div class="bar" id="ranges"></div>
<div class="card" id="card"><svg id="chart" viewBox="0 0 1060 560" role="img" aria-label="Price and X sentiment over time"></svg><div class="tip" id="tip"></div></div>
<p class="note">Top: daily close (UTC) with the model's n+0…n+5 forecast and 80% band. Bottom: engagement-weighted X sentiment (bars, blue = positive, red = negative) and its 7-day average (line). Hover for values.</p>
<table id="fc"></table>
</div>
<script>
const D=__DATA__;const F=__FORECAST__;const T=__TILES__;
const $=s=>document.querySelector(s);const svg=$("#chart"),tip=$("#tip");const NS="http://www.w3.org/2000/svg";
const W=1060,L=62,R=16,P1=[16,300],P2=[340,520];
const fmt=v=>v==null?"–":v.toLocaleString(undefined,{maximumFractionDigits:0});
const fs=v=>v==null?"–":(v>=0?"+":"")+v.toFixed(3);
$("#tiles").innerHTML=T.map(t=>`<div class="tile"><div class="k">${t[0]}</div><div class="v">${t[1]}</div></div>`).join("");
const ranges=[["30D",30],["90D",90],["180D",180],["All",1e9]];let cur=180;
$("#ranges").innerHTML=ranges.map(r=>`<button data-d="${r[1]}" aria-pressed="${r[1]===cur}">${r[0]}</button>`).join("");
$("#ranges").onclick=e=>{const b=e.target.closest("button");if(!b)return;cur=+b.dataset.d;
 document.querySelectorAll("#ranges button").forEach(x=>x.setAttribute("aria-pressed",x===b));draw();};
function el(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);(p||svg).appendChild(e);return e;}
function ticks(a,b,n){const s=(b-a)/n,m=Math.pow(10,Math.floor(Math.log10(s))),st=[1,2,2.5,5,10].map(x=>x*m).find(x=>x>=s);
 const o=[];for(let v=Math.ceil(a/st)*st;v<=b+1e-9;v+=st)o.push(v);return o;}
function draw(){
 svg.innerHTML="";const last=D[D.length-1].t;const t0=last-cur*864e5;const d=D.filter(r=>r.t>=t0);
 const tmax=F.length?F[F.length-1].t:last;const tmin=d[0].t;
 const x=t=>L+(t-tmin)/(tmax-tmin||1)*(W-L-R);
 let lo=Math.min(...d.map(r=>r.c),...F.map(f=>f.lo)),hi=Math.max(...d.map(r=>r.c),...F.map(f=>f.hi));const pad=(hi-lo)*.06;lo-=pad;hi+=pad;
 const y1=v=>P1[1]-(v-lo)/(hi-lo)*(P1[1]-P1[0]);
 const sv=d.map(r=>r.s).filter(v=>v!=null);const sm=Math.max(.05,...sv.map(Math.abs))*1.1;
 const y2=v=>(P2[0]+P2[1])/2-v/sm*(P2[1]-P2[0])/2;
 ticks(lo,hi,5).forEach(v=>{el("line",{x1:L,x2:W-R,y1:y1(v),y2:y1(v),stroke:"var(--grid)"});el("text",{x:L-8,y:y1(v)+4,"text-anchor":"end",class:"lbl"}).textContent=fmt(v);});
 [-sm/1.1,0,sm/1.1].forEach(v=>{el("line",{x1:L,x2:W-R,y1:y2(v),y2:y2(v),stroke:v===0?"var(--axis)":"var(--grid)"});el("text",{x:L-8,y:y2(v)+4,"text-anchor":"end",class:"lbl"}).textContent=v.toFixed(2);});
 el("text",{x:L,y:P1[0]-2,class:"ttl"}).textContent="Close (USD)";el("text",{x:L,y:P2[0]-8,class:"ttl"}).textContent="X sentiment (engagement-weighted)";
 const months=[];
 if(cur<=30){for(let t=tmin+7*864e5;t<=tmax;t+=7*864e5)months.push(t);}
 else{const dt=new Date(tmin);dt.setUTCDate(1);dt.setUTCMonth(dt.getUTCMonth()+1);
  while(dt.getTime()<=tmax){months.push(dt.getTime());dt.setUTCMonth(dt.getUTCMonth()+1);}}
 months.forEach(t=>{el("line",{x1:x(t),x2:x(t),y1:P1[0],y2:P2[1],stroke:"var(--grid)"});
  el("text",{x:x(t),y:P2[1]+16,"text-anchor":"middle",class:"lbl"}).textContent=new Date(t).toLocaleDateString(undefined,cur<=30?{month:"short",day:"numeric",timeZone:"UTC"}:{month:"short",year:"2-digit",timeZone:"UTC"});});
 if(F.length){const lp=d[d.length-1];const pts=[{t:lp.t,lo:lp.c,hi:lp.c,p:lp.c},...F];
  el("path",{d:"M"+pts.map(f=>x(f.t)+","+y1(f.hi)).join("L")+"L"+pts.slice().reverse().map(f=>x(f.t)+","+y1(f.lo)).join("L")+"Z",fill:"var(--band)"});
  el("path",{d:"M"+pts.map(f=>x(f.t)+","+y1(f.p)).join("L"),fill:"none",stroke:"var(--s1)","stroke-width":2,"stroke-dasharray":"5 4"});}
 el("path",{d:"M"+d.map(r=>x(r.t)+","+y1(r.c)).join("L"),fill:"none",stroke:"var(--s1)","stroke-width":2,"stroke-linejoin":"round"});
 const bw=Math.max(1,(W-L-R)/((tmax-tmin)/864e5)*.7);
 d.forEach(r=>{if(r.s==null)return;const y=y2(r.s),z=y2(0);el("rect",{x:x(r.t)-bw/2,y:Math.min(y,z),width:bw,height:Math.max(1,Math.abs(y-z)),fill:r.s>=0?"var(--pos)":"var(--neg)",opacity:.55,rx:Math.min(2,bw/3)});});
 const ma=d.filter(r=>r.m!=null);if(ma.length)el("path",{d:"M"+ma.map(r=>x(r.t)+","+y2(r.m)).join("L"),fill:"none",stroke:"var(--ink)","stroke-width":1.8});
 const cross=el("line",{y1:P1[0],y2:P2[1],stroke:"var(--ink2)","stroke-width":1,visibility:"hidden"});
 const dot=el("circle",{r:4.5,fill:"var(--s1)",stroke:"var(--surface)","stroke-width":2,visibility:"hidden"});
 const hit=el("rect",{x:L,y:P1[0],width:W-L-R,height:P2[1]-P1[0],fill:"transparent"});
 const all=[...d.map(r=>({...r,kind:"obs"})),...F.map(f=>({...f,kind:"fc"}))];
 hit.addEventListener("pointermove",ev=>{const b=svg.getBoundingClientRect();const sx=(ev.clientX-b.left)*W/b.width;
  let best=all[0];for(const r of all)if(Math.abs(x(r.t)-sx)<Math.abs(x(best.t)-sx))best=r;
  cross.setAttribute("x1",x(best.t));cross.setAttribute("x2",x(best.t));cross.setAttribute("visibility","visible");
  const yv=best.kind==="obs"?best.c:best.p;dot.setAttribute("cx",x(best.t));dot.setAttribute("cy",y1(yv));dot.setAttribute("visibility","visible");
  const ds=new Date(best.t).toISOString().slice(0,10);
  tip.innerHTML=best.kind==="obs"?`<b>${ds}</b><div>Close<span>$${fmt(best.c)}</span></div><div>Sentiment<span>${fs(best.s)}</span></div><div>7-day avg<span>${fs(best.m)}</span></div><div>Posts<span>${fmt(best.n)}</span></div><div>Bullish share<span>${best.b==null?"–":(best.b*100).toFixed(0)+"%"}</span></div>`
   :`<b>${ds} · ${best.h} forecast</b><div>Forecast<span>$${fmt(best.p)}</span></div><div>80% band<span>$${fmt(best.lo)} – $${fmt(best.hi)}</span></div>`;
  tip.style.display="block";const px=x(best.t)*b.width/W;tip.style.left=Math.min(px+14,b.width-tip.offsetWidth-8)+"px";tip.style.top="24px";});
 hit.addEventListener("pointerleave",()=>{tip.style.display="none";cross.setAttribute("visibility","hidden");dot.setAttribute("visibility","hidden");});
}
draw();
if(F.length){$("#fc").innerHTML="<tr><th>Horizon</th><th>Date</th><th>Forecast</th><th>80% low</th><th>80% high</th></tr>"+
 F.map(f=>`<tr><td>${f.h}</td><td>${new Date(f.t).toISOString().slice(0,10)}</td><td>$${fmt(f.p)}</td><td>$${fmt(f.lo)}</td><td>$${fmt(f.hi)}</td></tr>`).join("");}
</script></body></html>"""


def _num(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def build_dashboard(daily: pd.DataFrame, live: pd.DataFrame, path, asset: str, demo: bool = False,
                    tiles: list | None = None) -> None:
    d = daily.sort_values("date_utc").copy()
    s = d["sent_ew"].where(d["has_x"].astype(bool))
    d["ma"] = s.rolling(7, min_periods=3).mean()
    data = [{"t": int(pd.Timestamp(r.date_utc).timestamp() * 1000), "c": _num(r.close),
             "s": _num(sv), "m": _num(r.ma), "n": _num(r.n_posts), "b": _num(r.bull_share)}
            for r, sv in zip(d.itertuples(), s)]
    fc = [{"t": int(pd.Timestamp(r.target_date).timestamp() * 1000), "h": r.horizon,
           "p": float(r.pred_close), "lo": float(r.lo80), "hi": float(r.hi80)} for r in live.itertuples()]
    title = f"{asset}: X sentiment vs price"
    sub = f"Daily UTC bars {d.date_utc.iloc[0]} → {d.date_utc.iloc[-1]} · forecasts from the walk-forward ensemble"
    html = (TEMPLATE.replace("__TITLE__", title).replace("__SUB__", sub)
            .replace("__DEMO__", '<div class="demo">DEMO — synthetic data used to test the pipeline. '
                     'Not real market data.</div>' if demo else "")
            .replace("__DATA__", json.dumps(data)).replace("__FORECAST__", json.dumps(fc))
            .replace("__TILES__", json.dumps(tiles or [])))
    path.write_text(html, encoding="utf-8")
