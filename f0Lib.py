import numpy as np
from scipy import signal, fftpack



def pickpeak(spec, npicks = 2, rdiff = 5):

    mrows = spec.shape[0]
    ncols=1

    good = np.where(np.isfinite(spec))[0]
    rmin = min(spec[good]) - 1
    
    bad = np.where(np.isinf(spec))[0]#find(isinf(spec))
    
    if len(bad) > 0:
        spec[bad] = np.ones.ones(bad.shape) * rmin;
    
    
    #% ---- find a peak, zero out the data around the peak, and repeat
    
    val =  np.ones([npicks, ncols]) * np.NAN ;
    loc =  np.zeros([npicks, ncols]) ;
    loc[:,:] = -1
    
    for k in range(ncols):#=1:ncols
        dx = np.diff(np.concatenate((rmin.reshape(-1),spec[:],rmin.reshape(-1))))    # for a local peak at either end
        
        lp = (dx[:-1] >= 0) & (dx[1:] <= 0)#np.where(dx[:mrows] >= 0 and dx[1:] <= 0)[0]
        lp = np.where(lp == True)[0]

        vp = spec[lp]                       # peak values
    
        for p in range(npicks):#1:npicks
            l = vp.argmax()
            v = vp[l]
            #[v,l] = max(vp)

            val[p,k] = v
            loc[p,k] = lp[l]   # save value and location
    
            ind = np.where(abs(lp[l]-lp) > rdiff)[0] # find peaks which are far away
    
            if len(ind) == 0:
                break                           # no more local peaks to pick

            vp  = vp[ind]                     # shrink peak value array
            lp  = lp[ind]                    # shrink peak location array

    
    return loc,val

def transcost(F1,F2,voic_unv_cost,jump_cost):
    
    f1f2cost = np.zeros((len(F1), len(F2)))
    
    for n in range(len(F1)):#=1:len(F1)
        for m in range(len(F2)):#=1:length(F2)
            f1 = F1[n]
            f2 = F2[m]
            if (f1==0) and (f2==0):
                f1f2cost[n,m] = 0
                  
            elif (f1!=0) and (f2!=0):
                    
                f1f2cost[n,m] = jump_cost*abs(np.log2(f1/f2))

            else: 
                f1f2cost[n,m] = voic_unv_cost

    return f1f2cost

def getF0(y,
          Fs,
          time_step = 0.01, #Time step in secs.
          min_pitch = 75,#Min. pitch in Hz.
          max_pitch = 600,#Max. pitch in Hz.
          max_no_cand = 15,#Max. number of candidates
          sil_thresh = 0.03,#Silence threshold
          voicing_thresh = 0.45,#Voicing Threshold
          octave_cost = 0.01,#Octave Cost
          jump_cost = 0.35,#Jump Cost
          voic_unv_cost = 0.2):#Voice-Unvoice Cost 
    
    #y=y(:)';


    
    # Preprocess entire signal. Low pass filter with cutoff of 4kHz
    if Fs > 8000:
        wn = float(8000)/Fs
        N = 10
        b,a = signal.butter(N,wn);
        x = signal.lfilter(b,a,y);
    else:
        x = y;
        
    gap = max(abs(x));

    win_length = np.ceil(3.0*Fs/min_pitch).astype('int')   # Window length large enough to accommodate 3 periods
    win_step = np.round(time_step*Fs).astype('int')
    L = len(x)
    #x = x.transpose()
    
    wflag = 0;
    if (L < win_length):
        #warning('Segment too small. Replicating segment to improve accuracy');
        wflag = 1
        multfactor = np.ceil(win_length/L)
        x = np.kron(np.ones(1,multfactor),x)
        L = len(x)

    
    #% Find norm. autocorrelation of Gaussian window
    t = np.linspace(0, win_length - 1, win_length)
    
    w = (np.exp(-12.0*(((t+1) * 1.0/win_length)-0.5)**2)-np.exp(-12.0))/(1-np.exp(-12.0));
    
    w = np.concatenate((w, np.zeros(np.ceil(win_length/2.0).astype('int'), dtype = 'int')))
    
    l2 = int(2 ** np.ceil(np.log2(len(w))))
    W = fftpack.fft(w,l2)
    WW = np.abs(W)**2
    rw = np.real(fftpack.ifft(WW))
    rw = rw/np.max(rw)
    
    # Process short-time
    delta2 = np.floor(Fs * 1.0/min_pitch).astype('int')
    delta1 = np.ceil(Fs * 1.0/max_pitch).astype('int') - 1
    delta_diff = np.round(delta1/2.0).astype('int') + 1
    
    kmax = np.floor(1+(L-win_length) * 1.0/win_step).astype('int')
    
    #print('kmax is %f', kmax);
    C = list()
    C_father = list()

    T_ind = np.zeros((kmax, 2))



    RR = np.zeros([1000, 1])
    TT = np.zeros([1000, 1])
    TT[:,:] = -1
    for k in range(kmax):
        
        seg = x[k*win_step: k * win_step + win_length]
        
        
        if (max(abs(seg)) <1e-20): #% This is a rare case when the signal is a numerical zero throughout
            C[0,k].F = [0]
            C[0,k].R = [voicing_thresh]
            T_ind[k,:] = np.linspace((k-1)*win_step, (k-1)*win_step+win_length, win_length, dtype = 'int')#[(k-1)*win_step+1 (k-1)*win_step+win_length];
            continue;

        T_ind[k,:] = [k*win_step, k * win_step + win_length]
        
        #np.linspace((k-1)*win_step, (k-1)*win_step+win_length, win_length, dtype = 'int')# [(k-1)*win_step+1 (k-1)*win_step+win_length];

        lap = max(abs(seg));
        seg = seg - seg.mean()
        seg = np.concatenate((seg, \
                              np.zeros(np.ceil(win_length/2.0).astype('int'))))

        seg = seg * w;
        X_seg = fftpack.fft(seg,l2);
        XX_seg = abs(X_seg) ** 2;

        ra = np.real(fftpack.ifft(XX_seg));
        ra = ra/max(ra);
        ran = ra / rw;
        ranflpd = np.concatenate((ran[l2/2:], ran[0:l2/2]))
    
        ##% Find "roughly" the locations and values of maxima by peakpicking
        taus,r = pickpeak(ran[delta1:delta2],max_no_cand,delta_diff);
    
        zero_ind = np.where(taus >= 0)[0]
        if len(zero_ind) > 0:
            taus = taus[zero_ind]
            r = r[zero_ind]


        neg_ind = np.where(r >= 0)[0]
        if len(neg_ind) > 0:
            r = r[neg_ind]
            taus = taus[neg_ind]
      
        taus = taus+delta1#;     % Add the offset that pickpeak skips at the beginning
    
        
        for ck in range(len(taus)):#1:length(taus)  % Interpolate each of the candidates
        

            N=10#;  % Try this first; it's faster

            T = taus[ck,0]
            startl = T - N
            x1 = np.linspace(startl, T-1, N)#[startl:T-1]; 
            y1 = ranflpd[(0.5*l2-1+(x1)).astype('int')]#ranflpd[0.5*l2-1+(startl:T-1)]
            
            endr = T+N     
            x2 = np.linspace(T+1, endr, N)#[T+1:endr]; 
            y2 = ran[(T+1).astype('int'):(endr+1).astype('int')]
            xx = np.concatenate((x1,T.reshape(1),x2))
            yy = np.concatenate((y1,ran[int(T)].reshape(1),y2))
            
            testgrid=np.linspace(T, T+2, 201)#=[T-1:0.01:T+1];
            
            from scipy import interpolate
            
            tck=interpolate.splrep(xx + 1,yy,s=0)
            interpran=interpolate.splev(testgrid,tck,der=0)
            #interpran=interpolate.splrep(xx,yy,testgrid)
            maxind = interpran.argmax()
            RR[ck, 0] = interpran[maxind]
            #[RR[ck,0],maxind] = max(interpran);
            
            TT[ck,0] = testgrid[maxind];
            if (TT[ck,0] > delta2): 
                
                TT[ck,0] = delta2#; end %Ensure that limits are respected 
            if (TT[ck,0] < delta1): 
                TT[ck,0] = delta1#; end %after interpolation
       
        
        validInd = np.where(TT > -1) [0]
    
        Rvcd = RR[validInd] - octave_cost*(np.log2(min_pitch*TT[validInd]))#;   % Strength of voiced candidates
        Ro = voicing_thresh + max(0,2-(lap*(1+voicing_thresh)/(sil_thresh*gap)))
        
        C.append({'F': np.concatenate((np.zeros((1,1)), Fs / TT[validInd])),
                  'R': np.concatenate((np.array(Ro).reshape(1,1), Rvcd))})
        
        #print (C[k]['F'], C[k]['R'])
        #C[0,k]['F'] = np.concatenate((0,Fs/TT))#[0; Fs./TT];


    Delta = []
    Psi = []
    Delta.append( -C[0]['R'])
    Psi.append( 0*C[0]['R'])
    
    for k in range(1, kmax):
        A = transcost(C[k-1]['F'], \
                      C[k]['F'], \
                      voic_unv_cost, \
                      jump_cost)
         
        minarg = (Delta[k-1] * np.ones((1,A.shape[1]))+A).argmin(0)
        minval = (Delta[k-1] * np.ones((1,A.shape[1]))+A)[minarg,:]
        minval = minval[0,:]
        minval = minval.reshape([-1, 1])
        #[minval,minarg] = min(Delta[k-1] * np.ones((1,A.shape[1]))+A,[],1)
        #Delta[0,k] = minval
        #Psi[1,k] = minarg
        Delta.append(minval - C[k]['R'])
        Psi.append(minarg)
        #print Psi[k]
        
    

    
    mstar = np.zeros(kmax)
    
    mstar[kmax - 1] = (Delta[kmax - 1]).argmin()
    valstar = Delta[kmax - 1][int(mstar[kmax - 1])]

    F0= np.zeros(kmax)
    strength= np.zeros(kmax)
    
    
    #mstar
    F0[kmax - 1] = C[kmax - 1]['F'][int(mstar[kmax-1])]#C[kmax].F[mstar[kmax]]
    strength[kmax - 1] = C[kmax - 1]['R'][int(mstar[kmax-1])]
    
    for kk in range(1, kmax):  # =kmax-1:-1:1
        mstar[kmax-kk - 1] = Psi[kmax-kk][int(mstar[kmax-kk])]
        #print ((kmax-kk - 1), mstar[kmax-kk - 1])
        #print mstar[kmax-kk - 1]
        F0[kmax - kk - 1] = C[kmax - kk - 1]['F'][int(mstar[kmax - kk - 1])]
        strength[kmax - kk - 1] = C[kmax-kk - 1]['R'][int(mstar[kmax - kk - 1])]




    return F0, strength, T_ind, wflag




