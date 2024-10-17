import numpy as np
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey
from scipy.special import roots_legendre
from scipy.integrate import simpson


from get_fnm import get_fnm, get_gnm

from velocileptors.LPT.cleft_fftw import CLEFT
from velocileptors.Utils.spherical_bessel_transform import SphericalBesselTransform
from velocileptors.Utils.loginterp import loginterp


class ZeldovichBispectrum(CLEFT):
    '''
    Class based on self_fftw to compute density shape correlations.
    This is a placeholder for now.
    '''

    def __init__(self, *args, nmax=5, kout_min=-4, kout_max=1, kout_N=2048,\
                              ngauss_theta=16, ngauss_phi=16,\
                              kint_N = 512, kint_min=-3, kint_max=0, **kw):
        '''
        If beyond_gauss = True computes the third and fourth moments, otherwise
        default is to enable calculation of P(k), v(k) and sigma(k).
        
        Other keywords the same as the self_fftw class. Go look there!
        '''
        
        # Set up the configuration space quantities
        CLEFT.__init__(self, *args, **kw)
        
        self.nmax = nmax
        
        # where to store Gamma(p)
        self.kout_min = kout_min
        self.kout_max = kout_max
        self.kout_N = kout_N
        self.kout = np.logspace(kout_min, kout_max, kout_N)
        
        # Integration grid for episilon convolution
        self.kint_N = kint_N
        self.kint_min = kint_min
        self.kint_max = kint_max
        self.kint = np.logspace(kint_min, kint_max, kint_N)

        self.ngauss_theta = ngauss_theta
        self.ngauss_phi = ngauss_phi
        
        thetas, ws_theta = roots_legendre(ngauss_theta)
        phis, ws_phi = roots_legendre(ngauss_phi)

        self.thetas = 0.5 * np.pi * (thetas + 1)
        self.ws_theta = 0.5 * np.pi * ws_theta
        
        self.phis = np.pi * (phis + 1)
        self.ws_phi = np.pi * ws_phi
        
        self.Ps, self.Thetas, self.Phis = [arr.flatten() for arr in np.meshgrid( self.kint, self.thetas, self.phis , indexing='ij')]
        self.jac = (self.Ps**2 * np.sin(self.Thetas)).reshape( (self.kint_N, self.ngauss_theta, self.ngauss_phi) )

        self.Pz = self.Ps * np.cos(self.Thetas)
        self.Px = self.Ps * np.sin(self.Thetas) * np.cos(self.Phis)
        self.Py = self.Ps * np.sin(self.Thetas) * np.sin(self.Phis)

        self.Pvecs = np.vstack((self.Px,self.Py,self.Pz))
        
        self.sphb = SphericalBesselTransform(self.qint, L=10, low_ring=True)
        
        # power spectrum interpolation in case of linear theory/one-loop
        #self.pfunc = interp1d(self.k, self.p, kind='cubic', bounds_error=False, fill_value=0)
        self.pfunc = loginterp(self.k, self.p) # use this to avoid low k issues in squeezed limit, where epsilon*k is below interpolation range
        
    
    def set_vec_dict(self, vec_dict):
        self.vec_dict = vec_dict
    
    def compute_Gammas(self, order=None):

        vec_dict = self.vec_dict
        self.Gamma_dict = {}
        self.Gamma_minus_dict = {}
    
        for i, j in [(1,2), (1,3), (2,3)]:
    
            Gammas = np.zeros( (self.nmax,self.nmax,self.kout_N) )
            Gammas_minus = np.zeros( (self.nmax,self.nmax,self.kout_N) )
    
            kivec, kjvec = vec_dict[i], vec_dict[j]
            ki, kj = np.linalg.norm(kivec), np.linalg.norm(kjvec)
            kidotkj = np.dot(kivec,kjvec)
    
            expon = np.exp(0.5*kidotkj*(self.XYlin))
            beta = 0.5 * ki * kj * self.Ylin
            window = tukey(self.N, alpha=0.2)

            for n in range(self.nmax):
                # if we want to isolate the piece that's PL^order make sure beta^n * expon^pow ~ PL^order
                if order is not None:
                    pow = order - n
                    if pow >= 0:
                        intfac1 = (pow >= 0) / np.math.factorial(pow) * 4*np.pi * (0.5*kidotkj*(self.XYlin))**pow
                    else:
                        intfac1 = 0
                    
                    pow = order - n + 1
                    if pow >= 0:
                        intfac2 = (pow >= 0) / np.math.factorial(pow) * 4*np.pi * (0.5*kidotkj*(self.XYlin))**pow
                    else:
                        intfac2 = 0
                        
                else:
                    intfac1 = 4*np.pi * expon
                    intfac2 = intfac1
                    
                for m in range(int(n/2)+1):
                    # Gamma
                    _integrand = intfac1 * (beta)**n * self.qint**(2*m-n)
        
                    if n == 0:
                        _integrand = _integrand - _integrand[-1]
            
                    _integrand *= window
        
                    ktemps, res_fft = self.sphb.sph(n, _integrand)
                    Gammas[n,m,:] = interp1d(ktemps, res_fft, kind='cubic', bounds_error=False, fill_value=0)(self.kout)
        
                    # Gamma
                    _integrand = intfac2 * (beta)**(n-1) * self.qint**(2*m-n) * (n>0) * (m>0)
        
                    if n == 0:
                        _integrand = _integrand - _integrand[-1]
        
                    _integrand *= window
        
                    ktemps, res_fft = self.sphb.sph(n, _integrand)
                    Gammas_minus[n,m,:] = interp1d(ktemps, res_fft, kind='cubic', bounds_error=False, fill_value=0)(self.kout)
            
            self.Gamma_dict[(i,j)] = Gammas
            self.Gamma_minus_dict[(i,j)] = Gammas_minus
    
        return self.Gamma_dict, self.Gamma_minus_dict


    def Epsilon(self, pair, pvec, linear=False):
    
        vec_dict=self.vec_dict
        
        # if doing oneloop calculation return the linear theory finite piece
        if linear:
            if pvec.shape == (3,):
                pvec = pvec[:, None]
            ps = np.sum(pvec**2, axis=0)**0.5
            phats = pvec / ps[None, :]
    
            plins = self.pfunc(ps)
            angular_fac = np.sum( vec_dict[pair[0]][:,None] * phats, axis=0) * np.sum(vec_dict[pair[1]][:,None] * phats, axis=0)
    
            return - angular_fac / ps**2 * plins
    
        # if not... do the full thing
        Gamma_dict=self.Gamma_dict
        Gamma_minus_dict=self.Gamma_minus_dict
        
        eps = 0
        k1vec = vec_dict[pair[0]]
        k2vec = vec_dict[pair[1]]
    
        k1dotk2 = np.dot(k1vec, k2vec)
        k1 = np.linalg.norm(k1vec)
        k2 = np.linalg.norm(k2vec)
        p  = np.linalg.norm(pvec, axis=0)
    
        a = k1dotk2/k1/k2
        b = np.dot(k2vec, pvec)/k2/p
        c = np.dot(k1vec, pvec)/k1/p
    
        for n in range(self.nmax):
    
            fnms = get_fnm(n,a,b,c)
            gnms = get_gnm(n-1,a,b,c)
        

            for m in range(int(n/2)+1):

                eps += (-1)**(n+m) / np.math.factorial(n) * p**(2*m-n) * \
                        (fnms[m] * np.interp(p, self.kout, Gamma_dict[pair][n,m,:], left=0, right=0)\
                        + n*gnms[m-1] * np.interp(p, self.kout, Gamma_minus_dict[pair][n,m,:], left=0, right=0) )

        return eps


    def Btree(self, vdict):
    
        self.set_vec_dict(vdict)
    
        # Compute the delta function piece
        Binf =   self.Epsilon( (2,3), vdict[3], linear=True) * self.Epsilon( (1,2), vdict[1], linear=True) + \
                 self.Epsilon( (1,3), vdict[3], linear=True) * self.Epsilon( (1,2), vdict[2], linear=True) + \
                 self.Epsilon( (1,3), vdict[1], linear=True) * self.Epsilon( (2,3), vdict[2], linear=True)
                 
        # Compute the  finite piece
        Bfin = 0
        
        return Binf, Bfin

    def B1loop(self, vdict):
        
        self.set_vec_dict(vdict)
        
        # Compute the delta function piece:
        
        ## Tree Level
        
        Binf =   self.Epsilon( (2,3), vdict[3], linear=True) * self.Epsilon( (1,2), vdict[1], linear=True) + \
                 self.Epsilon( (1,3), vdict[3], linear=True) * self.Epsilon( (1,2), vdict[2], linear=True) + \
                 self.Epsilon( (1,3), vdict[1], linear=True) * self.Epsilon( (2,3), vdict[2], linear=True)

        ## Loop Level
        self.compute_Gammas(order=2)
        
        Binf += ( self.Epsilon( (2,3), vdict[3], linear=False) * self.Epsilon( (1,2), vdict[1], linear=True) +\
                  self.Epsilon( (2,3), vdict[3], linear=True)  * self.Epsilon( (1,2), vdict[1], linear=False) ) +\
                ( self.Epsilon( (1,3), vdict[3], linear=False) * self.Epsilon( (1,2), vdict[2], linear=True) +\
                  self.Epsilon( (1,3), vdict[3], linear=True)  * self.Epsilon( (1,2), vdict[2], linear=False) ) +\
                ( self.Epsilon( (1,3), vdict[1], linear=False) * self.Epsilon( (2,3), vdict[2], linear=True) +\
                  self.Epsilon( (1,3), vdict[1], linear=True) * self.Epsilon( (2,3), vdict[2], linear=False) )
                  
        # Compute the finite piece
        _int = self._integrand_ir_safe(oneloop=True)
        _int = _int.reshape( (self.kint_N, self.ngauss_theta, self.ngauss_phi) ) * self.jac
        Bfin = np.trapz(np.sum(np.sum(_int * self.ws_theta[None,:,None] * self.ws_phi[None,None,:],axis=1),axis=1),x=self.kint) / (2*np.pi)**3
        
        return Binf, Bfin
                  
        
        
    

    def Bzel(self, vdict):
        
        self.set_vec_dict(vdict)
        self.compute_Gammas()
    
        # Compute the delta function piece
        Binf =   self.Epsilon( (2,3), vdict[3]) * self.Epsilon( (1,2), vdict[1]) + \
                 self.Epsilon( (1,3), vdict[3]) * self.Epsilon( (1,2), vdict[2]) + \
                 self.Epsilon( (1,3), vdict[1]) * self.Epsilon( (2,3), vdict[2])
                 
        # Compute the  finite piece
        _int = self._integrand_ir_safe()
        _int = _int.reshape( (self.kint_N, self.ngauss_theta, self.ngauss_phi) ) * self.jac
        #Bfin = np.trapz(np.sum(np.sum(_int * self.ws_theta[None,:,None] * self.ws_phi[None,None,:],axis=1),axis=1),x=self.kint) / (2*np.pi)**3
        Bfin = simpson(np.sum(np.sum(_int * self.ws_theta[None,:,None] * self.ws_phi[None,None,:],axis=1),axis=1),x=self.kint) / (2*np.pi)**3
        
        return Binf, Bfin


    def _integrand_ir_safe(self, oneloop=False):
    
        pvecs = self.Pvecs
        vec_dict = self.vec_dict
        
        mod_p = np.sum(pvecs**2, axis=0)
        mod_k1mp = np.sum( (vec_dict[1][:,None] - pvecs)**2, axis=0 )
        mod_k2pp = np.sum( (vec_dict[2][:,None] + pvecs)**2, axis=0 )
        mod_k2mp = np.sum( (vec_dict[2][:,None] - pvecs)**2, axis=0 )
        mod_k3pp = np.sum( (vec_dict[3][:,None] + pvecs)**2, axis=0 )
        
        _ret =  self.Epsilon( (1,2), pvecs, linear=oneloop) *\
                (\
                self.Epsilon( (1,3), vec_dict[1][:,None] - pvecs, linear=oneloop) *\
                self.Epsilon( (2,3), vec_dict[2][:,None] + pvecs, linear=oneloop) *\
                (mod_p < mod_k1mp) * (mod_p < mod_k2pp) -\
                self.Epsilon( (1,3), vec_dict[1][:,None], linear=oneloop) *\
                self.Epsilon( (2,3), vec_dict[2][:,None], linear=oneloop)\
                )
    
        _ret +=  self.Epsilon( (1,3), pvecs, linear=oneloop) *\
                 (\
                 self.Epsilon( (1,2), vec_dict[1][:,None] - pvecs, linear=oneloop) *\
                 self.Epsilon( (2,3), -vec_dict[3][:,None] - pvecs, linear=oneloop) *\
                 (mod_p < mod_k1mp) * (mod_p < mod_k3pp) -\
                 self.Epsilon( (1,2), vec_dict[1][:,None], linear=oneloop) *\
                 self.Epsilon( (2,3), vec_dict[3][:,None], linear=oneloop)
                 )
            
        _ret += self.Epsilon( (2,3), pvecs, linear=oneloop) *\
                (\
                self.Epsilon( (1,2), vec_dict[2][:,None] - pvecs, linear=oneloop) *\
                self.Epsilon( (1,3), vec_dict[3][:,None] + pvecs, linear=oneloop) *\
                (mod_p < mod_k2mp) * (mod_p < mod_k3pp) -\
                self.Epsilon( (1,2), vec_dict[2][:,None], linear=oneloop) *\
                self.Epsilon( (1,3), vec_dict[3][:,None], linear=oneloop)
                )
        
        return _ret
    
    
    ###
    
    # Some extra functions implementing the non-IR safe version
    # For the sake of comparison.
    
    ###
    
    def B1loop_naive(self, vdict):
        
        self.set_vec_dict(vdict)
        
        Sigma2 = 0.5*self.Xlin[-1]
        k1dotk2 = np.dot(vdict[1],vdict[2])
        k1dotk3 = np.dot(vdict[1],vdict[3])
        k2dotk3 = np.dot(vdict[2],vdict[3])
        
        # Compute the delta function piece:
        
        ## Tree Level
        
        Binf =   (1 + k1dotk3*Sigma2) * self.Epsilon( (2,3), vdict[3], linear=True) * self.Epsilon( (1,2), vdict[1], linear=True) + \
                 (1 + k2dotk3*Sigma2) * self.Epsilon( (1,3), vdict[3], linear=True) * self.Epsilon( (1,2), vdict[2], linear=True) + \
                 (1 + k1dotk2*Sigma2) * self.Epsilon( (1,3), vdict[1], linear=True) * self.Epsilon( (2,3), vdict[2], linear=True)
                 
        ## Loop Level
        self.compute_Gammas(order=2)
        
        Binf += ( self.Epsilon( (2,3), vdict[3], linear=False) * self.Epsilon( (1,2), vdict[1], linear=True) +\
                  self.Epsilon( (2,3), vdict[3], linear=True)  * self.Epsilon( (1,2), vdict[1], linear=False) ) +\
                ( self.Epsilon( (1,3), vdict[3], linear=False) * self.Epsilon( (1,2), vdict[2], linear=True) +\
                  self.Epsilon( (1,3), vdict[3], linear=True)  * self.Epsilon( (1,2), vdict[2], linear=False) ) +\
                ( self.Epsilon( (1,3), vdict[1], linear=False) * self.Epsilon( (2,3), vdict[2], linear=True) +\
                  self.Epsilon( (1,3), vdict[1], linear=True) * self.Epsilon( (2,3), vdict[2], linear=False) )
                  
        # Compute the finite piece
        _int = self._integrand_naive(oneloop=True)
        _int = _int.reshape( (self.kint_N, self.ngauss_theta, self.ngauss_phi) ) * self.jac
        Bfin = np.trapz(np.sum(np.sum(_int * self.ws_theta[None,:,None] * self.ws_phi[None,None,:],axis=1),axis=1),x=self.kint) / (2*np.pi)**3
        
        return Binf, Bfin
    
    def Bzel_naive(self, vdict):
        
        self.set_vec_dict(vdict)
        self.compute_Gammas()
        
        Sigma2 = 0.5*self.Xlin[-1]
        k1dotk2 = np.dot(vdict[1],vdict[2])
        k1dotk3 = np.dot(vdict[1],vdict[3])
        k2dotk3 = np.dot(vdict[2],vdict[3])
        
        # Compute the delta function piece
        Binf =   np.exp(k1dotk3*Sigma2) * self.Epsilon( (2,3), vdict[3]) * self.Epsilon( (1,2), vdict[1]) + \
                 np.exp(k2dotk3*Sigma2) * self.Epsilon( (1,3), vdict[3]) * self.Epsilon( (1,2), vdict[2]) + \
                 np.exp(k1dotk2*Sigma2) * self.Epsilon( (1,3), vdict[1]) * self.Epsilon( (2,3), vdict[2])
                 
        # Compute the  finite piece
        _int = self._integrand_naive()
        _int = _int.reshape( (self.kint_N, self.ngauss_theta, self.ngauss_phi) ) * self.jac
        Bfin = np.trapz(np.sum(np.sum(_int * self.ws_theta[None,:,None] * self.ws_phi[None,None,:],axis=1),axis=1),x=self.kint) / (2*np.pi)**3
        
        return Binf, Bfin

    
    def _integrand_naive(self, oneloop=False):
    
        pvecs = self.Pvecs
        vec_dict = self.vec_dict
        
        mod_p = np.sum(pvecs**2, axis=0)
        mod_k1mp = np.sum( (vec_dict[1][:,None] - pvecs)**2, axis=0 )
        mod_k2pp = np.sum( (vec_dict[2][:,None] + pvecs)**2, axis=0 )
        mod_k2mp = np.sum( (vec_dict[2][:,None] - pvecs)**2, axis=0 )
        mod_k3pp = np.sum( (vec_dict[3][:,None] + pvecs)**2, axis=0 )
        
        _ret =  self.Epsilon( (1,2), pvecs, linear=oneloop) *\
                (\
                self.Epsilon( (1,3), vec_dict[1][:,None] - pvecs, linear=oneloop) *\
                self.Epsilon( (2,3), vec_dict[2][:,None] + pvecs, linear=oneloop) *\
                (mod_p < mod_k1mp) * (mod_p < mod_k2pp) \
                )
    
        _ret +=  self.Epsilon( (1,3), pvecs, linear=oneloop) *\
                 (\
                 self.Epsilon( (1,2), vec_dict[1][:,None] - pvecs, linear=oneloop) *\
                 self.Epsilon( (2,3), -vec_dict[3][:,None] - pvecs, linear=oneloop) *\
                 (mod_p < mod_k1mp) * (mod_p < mod_k3pp) \
                 )
            
        _ret += self.Epsilon( (2,3), pvecs, linear=oneloop) *\
                (\
                self.Epsilon( (1,2), vec_dict[2][:,None] - pvecs, linear=oneloop) *\
                self.Epsilon( (1,3), vec_dict[3][:,None] + pvecs, linear=oneloop) *\
                (mod_p < mod_k2mp) * (mod_p < mod_k3pp) \
                )
        
        return _ret
