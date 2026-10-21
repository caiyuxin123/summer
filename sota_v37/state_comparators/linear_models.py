from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from .state_utils import descriptor,history_baseline_profile,clip_candidate

class PCAStateForecast:
    def __init__(self,n_components=16,alpha=0.1,residual=False):
        self.n_components=int(n_components); self.alpha=float(alpha); self.residual=bool(residual)
    def fit(self,xm,xh,yshape,gen):
        X=descriptor(xm,xh); Y=np.asarray(yshape,float)
        if self.residual:
            self.base_train=history_baseline_profile(xm,xh,gen)
            T=(Y-self.base_train).reshape(len(Y),-1)
        else: T=Y.reshape(len(Y),-1)
        k=min(self.n_components,len(Y)-1,T.shape[1])
        self.pca=PCA(n_components=k,svd_solver='full',random_state=0).fit(T)
        Z=self.pca.transform(T)
        self.reg=Ridge(alpha=self.alpha,fit_intercept=True).fit(X,Z)
        self.gen=gen
        return self
    def predict(self,xm,xh,active=None):
        X=descriptor(xm,xh); Z=self.reg.predict(X); T=self.pca.inverse_transform(Z).reshape(len(X),168,-1)
        if self.residual: T=T+history_baseline_profile(xm,xh,self.gen)
        out=[]
        for i,t in enumerate(T):
            act=None if active is None else np.asarray(active[i],bool)
            out.append(clip_candidate(t,act))
        return np.stack(out)
    @property
    def complexity(self):
        return int(np.prod(self.reg.coef_.shape)+np.prod(self.pca.components_.shape))

class FourierDSE:
    def __init__(self,harmonics=4,alpha=0.1): self.h=int(harmonics); self.alpha=float(alpha)
    def _encode(self,Y):
        F=np.fft.rfft(np.asarray(Y,float),axis=1)/Y.shape[1]
        C=F[:,:self.h+1,:]
        return np.concatenate([C.real,C.imag],axis=1).reshape(len(Y),-1)
    def _decode(self,Z,nch):
        n=len(Z); z=Z.reshape(n,2*(self.h+1),nch); C=z[:,:self.h+1]+1j*z[:,self.h+1:]
        F=np.zeros((n,85,nch),dtype=np.complex128); F[:,:self.h+1,:]=C*168.0
        return np.fft.irfft(F,n=168,axis=1)
    def fit(self,xm,xh,yshape,gen):
        X=descriptor(xm,xh); self.nch=yshape.shape[2]; Z=self._encode(yshape)
        self.reg=Ridge(alpha=self.alpha).fit(X,Z); self.gen=gen; return self
    def predict(self,xm,xh,active=None):
        Z=self.reg.predict(descriptor(xm,xh)); T=self._decode(Z,self.nch)
        out=[]
        for i,t in enumerate(T): out.append(clip_candidate(t,None if active is None else active[i]))
        return np.stack(out)
    @property
    def complexity(self): return int(np.prod(self.reg.coef_.shape))
