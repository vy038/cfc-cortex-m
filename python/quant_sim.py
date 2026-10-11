# Fake-quantization drift check for the int8-weight idea. Run from repo root: python3 python/quant_sim.py
# Weights are the untrained seed-42 init, so this measures numerical drift vs float, not task accuracy.
import json,re,numpy as np
H=open('weights/generated/weights.h').read()
W={m[0]:np.array([float(v.strip().rstrip('f')) for v in m[1].split(',')],np.float64)
   for m in re.findall(r'static const float (CFC_\w+)\[\d+\] = \{([^}]*)\}',H)}
bw=W['CFC_BACKBONE_W'].reshape(128,38)
mats={k:W[f'CFC_{k}_W'].reshape(32,128) for k in('FF1','FF2','TIME_A','TIME_B')}
g=json.load(open('weights/generated/golden.json'))
def q8(w,bits=8):  # per-output-row symmetric
    qm=2**(bits-1)-1; s=np.abs(w).max(1,keepdims=True)/qm; return np.round(w/s).clip(-qm,qm)*s
def qa(x,bits=16,r=4.0):  # activation fake-quant, fixed range
    qm=2**(bits-1)-1; return np.round(x/r*qm).clip(-qm,qm)*r/qm
sig=lambda z:1/(1+np.exp(-z))
def run(seq,wq,aq):
    BW=wq(bw); M={k:wq(v) for k,v in mats.items()}; h=np.zeros(32); out=[]
    for x,ts in zip(seq['inputs'],seq['ts']):
        c=np.concatenate([x,h]); c=aq(c)
        b=1.7159*np.tanh(0.666*(BW@c+W['CFC_BACKBONE_B'])); b=aq(b)
        f1=np.tanh(M['FF1']@b+W['CFC_FF1_B']); f2=np.tanh(M['FF2']@b+W['CFC_FF2_B'])
        ta=M['TIME_A']@b+W['CFC_TIME_A_B']; tb=M['TIME_B']@b+W['CFC_TIME_B_B']
        t=sig(ta*ts+tb); h=f1*(1-t)+t*f2; out.append(h)
    return np.array(out)
I=lambda x:x
for name,wq,aq in [('float',I,I),('w8 a16',q8,qa),('w8 a8',q8,lambda x:qa(x,8)),('w16 a16',lambda w:q8(w,16),qa)]:
    e=[];d=[]
    for s in g['sequences']:
        o=run(s,wq,aq); ref=np.array(s['hidden_states']); e.append(np.abs(o-ref).max()); d.append(np.abs(o-ref)[-1].max())
    print(f'{name:8s} max_abs_err per seq:',' '.join(f'{v:.1e}' for v in e),'| hidden range',np.abs(ref).max().round(3))
