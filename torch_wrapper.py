
import torch
import numpy as np

class TorchWrapper:
    """
    A wrapper around PyTorch to provide a NumPy/CuPy-compatible API for pPXF.
    This handles differences in function signatures (e.g. axis vs dim) and 
    implements missing functionality (e.g. polynomials).
    """
    def __init__(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        self.fft = self.FFTModule(self)
        self.linalg = self.LinalgModule(self)
        self.polynomial = self.PolynomialModule(self)
        self.random = self.RandomModule(self)

    @property
    def pi(self):
        return torch.pi

    def asarray(self, a, dtype=None):
        target_dtype = self._map_dtype(dtype)
        
        # Infer dtype from input if not specified
        if target_dtype is None and not isinstance(a, torch.Tensor):
             if hasattr(a, 'dtype'):
                 if a.dtype == np.float64:
                     target_dtype = torch.float64
                 elif a.dtype == np.float32:
                     target_dtype = torch.float32
                 elif a.dtype == np.complex128:
                     target_dtype = torch.complex128
                 elif a.dtype == np.complex64:
                     target_dtype = torch.complex64
                 elif a.dtype == int or a.dtype == np.int64:
                     target_dtype = torch.long
        
        # Apply device constraints (MPS does not support float64/complex128)
        target_dtype = self._apply_device_constraints(target_dtype)
        
        if isinstance(a, torch.Tensor):
            t = a.to(device=self.device, dtype=target_dtype)
        else:
            t = torch.as_tensor(a, device=self.device, dtype=target_dtype)
        return t

    def asnumpy(self, a):
        if isinstance(a, torch.Tensor):
            if a.device.type != 'cpu':
                a = a.cpu()
            return a.numpy()
        return np.asarray(a)

    def zeros(self, shape, dtype=float):
        # Default to float (float64) to match dict(numpy)
        t_dtype = self._get_dtype(dtype)
        return torch.zeros(shape, device=self.device, dtype=t_dtype)

    def ones(self, shape, dtype=float):
        t_dtype = self._get_dtype(dtype)
        return torch.ones(shape, device=self.device, dtype=t_dtype)

    def empty(self, shape, dtype=float):
        t_dtype = self._get_dtype(dtype)
        return torch.empty(shape, device=self.device, dtype=t_dtype)
    
    def _map_dtype(self, dtype):
        if dtype is None: return None
        if dtype == int: return torch.long
        if dtype == float: return torch.float64 
        if dtype == complex: return torch.complex128
        return dtype

    def _get_dtype(self, dtype):
        t_dtype = self._map_dtype(dtype)
        return self._apply_device_constraints(t_dtype)

    def _apply_device_constraints(self, dtype):
        if self.device.type == 'mps':
            if dtype == torch.float64:
                return torch.float32
            if dtype == torch.complex128:
                return torch.complex64
        return dtype
    
    def arange(self, *args, **kwargs):
        # Default to int64 for integer args, but if float args, torch.arange might default to float32.
        # NumPy behavior: if args are ints, return int/int64. If oats, return float64.
        # For simplicity and given pPXF usage (often for indices or float ranges), 
        # let's rely on torch's inference but ensuring floats are float64 if device supports it.
        # However, torch.arange(float) -> float32 by default.
        # We should check args.
        
        dtype = kwargs.get('dtype')
        if dtype is None:
             # Check if any arg is float
             is_float = any(isinstance(a, float) for a in args)
             if is_float:
                 dtype = float 
        
        if dtype is not None:
             kwargs['dtype'] = self._get_dtype(dtype)
             
        return torch.arange(*args, device=self.device, **kwargs)

    def linspace(self, start, end, steps, **kwargs):
        # Default to float64
        dtype = kwargs.get('dtype', float)
        kwargs['dtype'] = self._get_dtype(dtype)
        return torch.linspace(start, end, steps, device=self.device, **kwargs)

    def array(self, object, dtype=None, copy=True, order='K', subok=False, ndmin=0):
        # Limited implementation of np.array
        return self.asarray(object, dtype=dtype)
    
    def append(self, arr, values, axis=None):
        arr = self.asarray(arr)
        values = self.asarray(values)
        if axis is None:
            return torch.cat((arr.flatten(), values.flatten()))
        return torch.cat((arr, values), dim=axis)

    def concatenate(self, arrays, axis=0):
        tensors = [self.asarray(a) for a in arrays]
        return torch.cat(tensors, dim=axis)

    def hstack(self, tup):
        tup = [self.asarray(t) for t in tup]
        return torch.hstack(tup)
    
    def vstack(self, tup):
        tup = [self.asarray(t) for t in tup]
        return torch.vstack(tup)
    
    def tile(self, A, reps):
        return torch.tile(self.asarray(A), reps)
    
    def flatnonzero(self, a):
        return torch.nonzero(self.asarray(a).flatten(), as_tuple=False).flatten()

    def diff(self, a, n=1, axis=-1):
        return torch.diff(self.asarray(a), n=n, dim=axis)
    
    def unique(self, ar, return_index=False, return_inverse=False, return_counts=False, axis=None):
        # Minimal implementation for pPXF usage
        sorted = True # pPXF assumes sorted? np.unique returns sorted.
        return torch.unique(self.asarray(ar), sorted=sorted, return_inverse=return_inverse, return_counts=return_counts, dim=axis)

    def atleast_1d(self, *arys):
        res = []
        for ary in arys:
            ary = self.asarray(ary)
            if ary.ndim == 0:
                res.append(ary.unsqueeze(0))
            else:
                res.append(ary)
        if len(res) == 1:
            return res[0]
        return res

    def issubdtype(self, dtype1, dtype2):
        return np.issubdtype(dtype1, dtype2) # Fallback to numpy for types

    def isscalar(self, num):
        return np.isscalar(num) or (isinstance(num, torch.Tensor) and num.numel() == 1)

    def abs(self, x): 
        x = self.asarray(x)
        return torch.abs(x)
    def log(self, x): 
        x = self.asarray(x)
        return torch.log(x)
    def exp(self, x): 
        x = self.asarray(x)
        return torch.exp(x)
    def sqrt(self, x): 
        x = self.asarray(x)
        return torch.sqrt(x)
    def sin(self, x): 
        x = self.asarray(x)
        return torch.sin(x)
    def cos(self, x): 
        x = self.asarray(x)
        return torch.cos(x)
    def tan(self, x): 
        x = self.asarray(x)
        return torch.tan(x)
    def mean(self, a, axis=None, keepdims=False, **kwargs):
        a = self.asarray(a)
        if axis is None:
            return torch.mean(a, **kwargs)
        return torch.mean(a, dim=axis, keepdim=keepdims, **kwargs)
    def sum(self, a, axis=None, **kwargs):
        a = self.asarray(a)
        if axis is None:
            return torch.sum(a, **kwargs)
        return torch.sum(a, dim=axis, **kwargs)
    def prod(self, a, axis=None, **kwargs):
        a = self.asarray(a)
        if axis is None:
            return torch.prod(a, **kwargs)
        return torch.prod(a, dim=axis, **kwargs)
    
    def min(self, a, axis=None, **kwargs):
        a = self.asarray(a)
        if axis is None:
            return torch.min(a)
        # torch.min returns (values, indices) when axis is specified
        return torch.min(a, dim=axis, **kwargs).values
    
    def max(self, a, axis=None, **kwargs):
        a = self.asarray(a)
        if axis is None:
            return torch.max(a)
        return torch.max(a, dim=axis, **kwargs).values
    
    def clip(self, a, a_min, a_max):
        a = self.asarray(a)
        return torch.clamp(a, a_min, a_max)

    def copy(self, a):
        a = self.asarray(a)
        return a.clone()

    def array_equal(self, a1, a2):
        if hasattr(a1, 'shape') and hasattr(a2, 'shape'):
             if a1.shape != a2.shape:
                 return False
        a1 = self.asarray(a1)
        a2 = self.asarray(a2)
        return torch.equal(a1, a2)

    def all(self, a, axis=None, out=None, keepdims=False):
        a = self.asarray(a)
        if axis is None:
             return torch.all(a)
        return torch.all(a, dim=axis, keepdim=keepdims)
    
    def any(self, a, axis=None):
        a = self.asarray(a)
        if axis is None:
            return torch.any(a)
        return torch.any(a, dim=axis)

    def isfinite(self, x):
        x = self.asarray(x)
        return torch.isfinite(x)
    
    def isnan(self, x):
        x = self.asarray(x)
        return torch.isnan(x)

    def conj(self, x):
        x = self.asarray(x)
        return torch.conj(x)
    
    def real(self, x):
        x = self.asarray(x)
        return torch.real(x)
    
    def imag(self, x):
        x = self.asarray(x)
        return torch.imag(x)

    def eye(self, N, M=None, k=0, dtype=None):
        # torch.eye doesn't support k? it does not.
        # But pPXF uses eye mostly for simple identity matrices (k=0).
        if k != 0:
            raise NotImplementedError("TorchWrapper.eye: k!=0 not implemented")
        return torch.eye(N, M, device=self.device, dtype=self._map_dtype(dtype))

    def _map_dtype(self, dtype):
        if dtype is None: return None
        if dtype == int: return torch.long
        if dtype == float: return torch.float32 
        if dtype == complex: return torch.complex64
        return dtype

    class FFTModule:
        def __init__(self, wrapper):
            self.wrapper = wrapper
        
        def rfft(self, a, n=None, axis=-1, norm=None):
            a = self.wrapper.asarray(a)
            return torch.fft.rfft(a, n=n, dim=axis, norm=norm)

        def irfft(self, a, n=None, axis=-1, norm=None):
            a = self.wrapper.asarray(a)
            return torch.fft.irfft(a, n=n, dim=axis, norm=norm)

    class LinalgModule:
        def __init__(self, wrapper):
            self.wrapper = wrapper
        
        def cholesky(self, a, lower=False):
            a = self.wrapper.asarray(a)
            if not lower:
                raise NotImplementedError("TorchWrapper.linalg.cholesky: lower=False not implemented")
            return torch.linalg.cholesky(a)

        def solve_triangular(self, a, b, lower=False):
            a = self.wrapper.asarray(a)
            b = self.wrapper.asarray(b)
            return torch.linalg.solve_triangular(a, b, upper=not lower)

        def solve(self, a, b):
            a = self.wrapper.asarray(a)
            b = self.wrapper.asarray(b)
            return torch.linalg.solve(a, b)
        
        def lstsq(self, a, b):
             a = self.wrapper.asarray(a)
             b = self.wrapper.asarray(b)
             return torch.linalg.lstsq(a, b)
        
        def norm(self, x, ord=None, axis=None, keepdims=False):
            x = self.wrapper.asarray(x)
            return torch.linalg.norm(x, ord=ord, dim=axis, keepdim=keepdims)

    class PolynomialModule:
        def __init__(self, wrapper):
            self.wrapper = wrapper
            self.legendre = self.LegendreModule(wrapper)
        
        class LegendreModule:
            def __init__(self, wrapper):
                self.wrapper = wrapper
            
                
            
            def legval(self, x, c):
                x = self.wrapper.asarray(x)
                c = self.wrapper.asarray(c)
                
                # If c is 1D, shape is (N,)
                dims = x.ndim
                if c.ndim == 1:
                    c = c.view(-1, *([1]*dims))
                
                nd = c.shape[0]
                c0 = c[-2]
                c1 = c[-1]
                
                # Iterative evaluation (Horner / Clenshaw like)
                # But here straightforward evaluation since pPXF typically uses low order.
                # Actually, standard legval evaluates sum(c[i]*L_i(x)).
                # Let's use the recurrence relation:
                # P_n+1(x) = ((2n+1)xP_n(x) - nP_n-1(x))/(n+1)
                
                # We need to accumulate c[0]*P0 + c[1]*P1 + ...
                
                p0 = torch.ones_like(x)
                if nd == 1:
                    return c[0] * p0

                p1 = x
                res = c[0] * p0 + c[1] * p1
                
                for i in range(2, nd):
                     # P_i = ((2i-1)x P_{i-1} - (i-1)P_{i-2}) / i
                     p_next = ((2*i - 1) * x * p1 - (i - 1) * p0) / i
                     res += c[i] * p_next
                     p0 = p1
                     p1 = p_next
                
                return res

            def legvander(self, x, deg):
                 x = self.wrapper.asarray(x)
                 if x.ndim != 1:
                     raise ValueError("x must be 1D for legvander")
                 
                 mat = self.wrapper.empty((x.shape[0], deg + 1), dtype=x.dtype)
                 
                 mat[:, 0] = 1
                 if deg > 0:
                     mat[:, 1] = x
                 
                 for i in range(2, deg + 1):
                     mat[:, i] = ((2*i - 1) * x * mat[:, i-1] - (i - 1) * mat[:, i-2]) / i
                 
                 return mat
        
        class HermiteModule:
            def __init__(self, wrapper):
                self.wrapper = wrapper
            
            def hermval(self, x, c):
                # Evaluate Hermite series at x with coefficients c.
                # H_0(x) = 1, H_1(x) = 2x
                # H_n+1(x) = 2xH_n(x) - 2nH_n-1(x)
                
                x = self.wrapper.asarray(x)
                c = self.wrapper.asarray(c)
                
                dims = x.ndim
                if c.ndim == 1:
                    c = c.view(-1, *([1]*dims))
                
                nd = c.shape[0]
                
                h0 = torch.ones_like(x)
                if nd == 1:
                    return c[0] * h0
                
                h1 = 2 * x
                res = c[0] * h0 + c[1] * h1
                
                for i in range(2, nd):
                    # H_i = 2xH_{i-1} - 2(i-1)H_{i-2}
                    h_next = 2 * x * h1 - 2 * (i - 1) * h0
                    res += c[i] * h_next
                    h0 = h1
                    h1 = h_next
                    
                return res
        
        def __init__(self, wrapper):
            self.wrapper = wrapper
            self.legendre = self.LegendreModule(wrapper)
            self.hermite = self.HermiteModule(wrapper)
            
    class RandomModule:
         def __init__(self, wrapper):
            self.wrapper = wrapper
         def rand(self, *size):
             return torch.rand(*size, device=self.wrapper.device)
         def randn(self, *size):
             return torch.randn(*size, device=self.wrapper.device)

