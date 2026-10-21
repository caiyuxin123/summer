from __future__ import annotations
import numpy as np, torch
from torch import nn
from .state_utils import padded_area_target,padded_to_profile,clip_candidate,cat_meta

def node_month_features(xm,gen):
    xm=np.asarray(xm,float)
    if xm.ndim==1: xm=xm[None,:]
    meta=cat_meta(gen); by={1:[],2:[],3:[]}
    for c,(a,*_) in enumerate(meta): by[a].append(c)
    K=max(map(len,by.values())); out=np.zeros((len(xm),3,1+K+3),np.float32)
    for i in range(len(xm)):
        for a in (1,2,3):
            out[i,a-1,0]=xm[i,a-1]
            for s,c in enumerate(by[a]): out[i,a-1,1+s]=xm[i,3+c]
            # two directed flow shares incident from this area + Direct-M
            areas=[1,2,3]; flowpos=25; pairs=[]
            for aa in areas:
                for bb in areas:
                    if aa!=bb: pairs.append((aa,bb))
            vals=[xm[i,flowpos+j] for j,p in enumerate(pairs) if p[0]==a]
            out[i,a-1,1+K:1+K+2]=vals[:2]
            out[i,a-1,-1]=xm[i,31+a-1]
    return out

class PIGNNNet(nn.Module):
    def __init__(self,in_dim,hidden,out_dim):
        super().__init__(); self.lin=nn.Linear(in_dim,hidden); self.gru=nn.GRU(hidden,hidden,batch_first=True); self.out=nn.Linear(hidden,out_dim)
    def forward(self,x,A):
        # x B,T,N,F. one graph message pass then independent temporal GRU per node
        x=torch.einsum('ij,btjf->btif',A,x)
        h=torch.relu(self.lin(x)); B,T,N,H=h.shape
        z=h.permute(0,2,1,3).reshape(B*N,T,H)
        z,_=self.gru(z); y=self.out(z).reshape(B,N,T,-1).permute(0,2,1,3)
        return y

class PIGNNState:
    def __init__(self,hidden=16,lr=1e-3,epochs=120,weight_decay=1e-4,seed=0):
        self.hidden=int(hidden);self.lr=float(lr);self.epochs=int(epochs);self.wd=float(weight_decay);self.seed=int(seed)
    def fit(self,xm,xh,yshape,graph,gen):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        Y,mask,_=padded_area_target(yshape,gen); M=node_month_features(xm,gen)
        X=np.concatenate([np.asarray(xh,np.float32),np.repeat(M[:,None,:,:],168,axis=1)],axis=-1)
        self.model=PIGNNNet(X.shape[-1],self.hidden,Y.shape[-1]); opt=torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=self.wd)
        xt=torch.tensor(X); yt=torch.tensor(Y); A=torch.tensor(np.asarray(graph,np.float32)); maskt=torch.tensor(mask)[None,None,:,:]
        self.model.train()
        for _ in range(self.epochs):
            opt.zero_grad(); pred=self.model(xt,A); loss=((pred-yt)**2*maskt).sum()/maskt.sum()/len(Y)/168; loss.backward(); opt.step()
        self.gen=gen; self.graph=np.asarray(graph,np.float32); self.mask=mask
        return self
    def predict(self,xm,xh,active=None):
        M=node_month_features(xm,self.gen); X=np.concatenate([np.asarray(xh,np.float32),np.repeat(M[:,None,:,:],168,axis=1)],axis=-1)
        self.model.eval()
        with torch.no_grad(): p=self.model(torch.tensor(X),torch.tensor(self.graph)).numpy()
        S=padded_to_profile(p,self.gen); out=[]
        for i,s in enumerate(S): out.append(clip_candidate(s,None if active is None else active[i]))
        return np.stack(out)
    @property
    def complexity(self): return sum(p.numel() for p in self.model.parameters())
