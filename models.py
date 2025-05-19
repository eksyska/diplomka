import numpy as np


# H = (1 - ksi)/N H1 - ksi/(N(N-1)) W^2

def kron_d (i, j):
    return int(i == j)

def encode (n, l):
    i = n*(n+1)/2 + (l+n)/2
    return int(i)

def decode (i):
    n = np.floor((-1+np.sqrt(1+8*i))/2)
    
    k = int(i-n*(n+1)/2)
    l = int(-n + 2*k)

    return np.array([n, l])

#test model
class well_2d:
    def __init__(self, num_states, a, alpha, M=1, h=1):
        self.num_states = num_states
        self.a = a
        self.b = a*alpha
        self.M = M
        self.h = h

        self.max_energy = 1.1 * num_states / (self.a * self.b * self.M / (2 * np.pi * self.h**2))
        self.max_n1 = np.sqrt( self.max_energy * 2*M/(np.pi*h)**2 * self.a)
        self.max_n2 = np.sqrt( self.max_energy * 2*M/(np.pi*h)**2 * self.b)
    
    def e_2d (self, n1, n2):
        return (np.pi**2 * self.h**2)/(2*self.M) * ( (n1/self.a)**2 + (n2/self.b)**2 )

    def compute_spectrum(self):
        n1 = np.arange(1, self.max_n1 + 1, 1, dtype=int)
        n2 = np.arange(1, self.max_n2 + 1, 1, dtype=int)
        spectrum = []

        for _ in n1:
            spectrum.extend(self.e_2d(_, n2))

        spectrum = np.array(spectrum)
        spectrum = spectrum[spectrum <= self.max_energy]
        spectrum = np.sort(spectrum)
        return spectrum

class u3:
    def __init__(self, N, E0=0, ksi=0, alpha=0, beta=0):
        self.N = N
        self.E0 = E0
        self.eps = 1-ksi
        self.alpha = alpha
        self.beta = beta
        self.A = -ksi/(N+1)

        self.i_max = encode(self.N, self.N)
        self.H_ij = np.zeros([self.i_max+1, self.i_max+1], dtype=complex)
        self._state_vec_0 = np.zeros([self.i_max], dtype=complex)

    @property
    def state_vec_0(self):
        return self._state_vec_0
    
    @state_vec_0.setter
    def state_vec(self, state_vec):
        self._state_vec_0 = state_vec

    def H1_el(self, n, l):
        el = self.E0 + self.eps*n + self.alpha*n*(n+1) + self.beta*l**2
        return el
    
    def W_el_diag(self, n1, l):
        el = self.A*( (self.N - n1)*(n1 + 2) + (self.N - n1 + 1)*n1 + l**2 )
        return el
    
    def W_el_offdiag(self, n1, l):
        el = - self.A*np.sqrt( (self.N - n1 + 2)*(self.N - n1 + 1)*(n1 + l)*(n1 - l) )
        return el
        
    def H_ij_fill(self):
        for i in range(self.i_max + 1):
            for j in range(self.i_max + 1):
                n_i, l_i = decode(i)
                n_j, l_j = decode(j)
                if n_i == n_j and l_i == l_j:
                    self.H_ij[i, j] = self.H1_el(n_i, l_i) + self.W_el_diag(n_i, l_i)
                elif n_i == n_j + 2:
                    self.H_ij[i, j] = self.W_el_offdiag(n_i, l_i)
                elif n_i == n_j - 2:
                    self.H_ij[i, j] = self.W_el_offdiag(n_j, l_j) 
        print(self.H_ij)
    
    



