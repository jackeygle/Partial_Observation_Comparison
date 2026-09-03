from functools import cached_property, wraps
from math import log, pi, prod

import torch


class MissingInputError(AttributeError):
	def __init__(self, name=None, obj=None, *args):
		msg = f"input '{name}' missing from '{obj}'"
		super().__init__(msg, *args)


class MathRelation:
	def __init__(self, other=None, /, **kwargs):
		if other is not None:
			kwargs = dict(**vars(other), **kwargs)  # str only, disallow repeats
		cls = type(self)
		for key, value in kwargs.items():
			if hasattr(cls, key) and isinstance(getattr(cls, key), cached_property):
				setattr(self, key, value)
			else:
				raise AttributeError(f"kwarg '{key}' is not a cached_property of class {cls}")
	
	def __call__(self, **kwargs):
		return type(self)(self, **kwargs)
	
	def eager(self, *, include=None, exclude=()):
		""" Fill caches with computable results.
			Note: "missing" inputs will be assumed as defaults.
		"""
		cls = type(self)
		if include is None:
			include = [
				key for key, value in vars(cls).items()
				if not key.startswith('_')
				if isinstance(value, cached_property)
			]
		cached = []
		for key in include:
			if key not in exclude:
				try:
					getattr(self, key)
				except MissingInputError:
					pass
				else:
					cached.append(key)
		return cached


class GaussCond(MathRelation):
	""" Gaussian Conditioning
		Maths:
			if
				yd ~ N(md, Sdd)
				yq ~ N(mq, Sqq)
				V[yq,yd] = Sqd
			then
				yp  = yq|yd ~ N(mp, Spp)
			where
				mp  = mq  + Sqd * Sdd^-1 * (yd-md)
				Spp = Sqq - Sqd * Sdd^-1 * Sdq
		
		Naming conventions:
			_d  data, observation
			_q  prior, query
			_p  posterior, same shape as _q
			n_  number of dimensions, for calculating shape
			y_  random variable or observed value
			m_  mu, mean
			S__ Sigma, covariance
			L__ (Lower) Cholesky decomposition of S__
			s_  sigma^2, diagonal of S__
			l_  log probability/likelihood
		
		Attributes:
			yd  [..., d, 1] input
			md  [..., d, 1] optional input, default zero
			Sdd [..., d, d] input
			Ldd [..., d, d] alternate input, Cholesky of Sdd
			ld  [...]       optional input, default zero
			mq  [..., q, 1] optional input, default zero
			Sqq [..., q, q] input
			sq  [..., q]    alternate input, diagonal of Sqq
			lq  [...]       log probability, log P(yq) = log P(yq|yd) P(yd)
			Sqd [..., q, d] input
			a   [..., d, 1]
			v   [..., q, d]
			mp  [..., q, 1] posterior mean
			Spp [..., q, q] posterior covariance
			sp  [..., q]    posterior standard deviation
			lp  [...]       log marginal likelihood, log P(yp) = log P(yq|yd)
	"""
	
	# functions
	@staticmethod
	def chol_cov(A):
		""" based on MATLAB's `cholcov` """
		# tol = torch.finfo(A.dtype).eps * A.abs().max() * 10
		L = torch.linalg.cholesky(A)
		return L
	
	@staticmethod
	def solve_tri(L, b, *, left=True):
		# return torch.triangular_solve(b, L, upper=False).solution  # deprecated
		return torch.linalg.solve_triangular(L, b, upper=False, left=left)
	
	@staticmethod
	def solve_chol(L, b):
		return torch.cholesky_solve(b, L, upper=False)
	
	
	# shape
	def _get_n(self, **dims):
		# todo: check they all match/broadcast
		for attr, dim in dims.items():
			if attr in vars(self):
				n = getattr(self, attr).shape[dim]
				if n == 1: continue  # lazy broadcasting
				return n
		raise MissingInputError(dims.keys(), self)
	
	@property
	def nd(self): return self._get_n(yd=-2, md=-2, Sdd=-2, Ldd=-2, Sqd=-1)
	@property
	def nq(self): return self._get_n(mq=-2, Sqq=-2, sq=-1, Sqd=-2)  # todo: include _p
	@property
	def np(self): return self.nq
	
	
	# data d
	@cached_property  # [..., d, 1]
	def yd(self): raise MissingInputError('yd', self)
	
	@cached_property  # [..., d, 1]
	def md(self): return torch.zeros(1, 1)
	
	@cached_property  # [..., d, d]
	def Sdd(self):
		if 'Ldd' in vars(self):
			return self.Ldd @ self.Ldd.mT
		raise MissingInputError('Ldd', self) from MissingInputError('Sdd', self)
	
	@cached_property  # [..., d, d]
	def Ldd(self):
		return self.chol_cov(self.Sdd)
	
	@cached_property  # [...]
	def ld(self): raise MissingInputError('ld', self)
	
	# query q
	@cached_property  # [..., q, 1]
	def mq(self): return torch.zeros(1, 1)
	
	@cached_property  # [..., q, q]
	def Sqq(self): raise MissingInputError('Sqq', self)
	
	@cached_property  # [..., q]
	def sq(self):
		return self.Sqq.diagonal(0,-1,-2)
	
	@cached_property  # [...]
	def lq(self):
		return self.lp + self.ld
	
	# query-data qd
	@cached_property  # [..., q, d]
	def Sqd(self): raise MissingInputError('Sqd', self)
	
	# working
	@cached_property  # [..., d, 1]
	def a(self):
		return self.solve_chol(self.Ldd, self.yd - self.md)
	
	@cached_property  # [..., q, d]
	def v(self):
		return self.solve_tri(self.Ldd, self.Sqd.mT).mT
	
	# posterior p
	@cached_property  # [..., q, 1]
	def mp(self):
		return self.mq + self.Sqd @ self.a
	
	@cached_property  # [..., q, q]
	def Spp(self):
		return self.Sqq - self.v @ self.v.mT
	
	@cached_property  # [..., q]
	def sp(self):
		return self.sq - self.v.square().sum(-1)
	
	@cached_property  # [...]
	def lp(self):
		nd = self.yd.shape[-2]
		return -1/2 * ((self.yd - self.md) * self.a)[...,0].sum(-1) - self.Ldd.diagonal(0,-1,-2).log().sum(-1) - nd/2*log(2*pi)


class GaussProcess(GaussCond):
	""" Gaussian Process
		Gaussian Conditioning with covariances from a kernel.
			See GaussCond.
		
		Maths:
			mu and Sigma specified by functions
				mi  = m(xi)
				Sij = K(xi, xj')
			except
				Sdd = K(xd, xd') + Snn
			because observations yd are corrupted by zero-mean Gaussian noise
		
		Naming conventions:
			_n  noise of data yd, same shape as _d
			x_  location of random variable y_
			D   location (x_) dimension
			C   random variable (y_) dimension (channels)
		
		Attributes:
			xd  [..., d/C, D]       input
			xq  [..., q/C, D]       input
			Snn [..., d, d]         optional input, default zero
			sn  [..., d]            alternate input, diagonal of Snn
			m     (xi) -> mi        optional input, default zero function
			md_f  (xd) -> md        alternate input
			mq_f  (xq) -> mq        alternate input
			K     (xi, xj') -> Kij  input
			Kdd_f (xd, xd') -> Kdd  alternate input
			Kqd_f (xq, xd') -> Kqd  alternate input
			Kqq_f (xq, xq') -> Kqq  alternate input
	"""
	
	# location x
	@cached_property  # [..., d/C, D]
	def xd(self): raise MissingInputError('xd', self)
	
	@cached_property  # [..., d/C, D]
	def xq(self): raise MissingInputError('xq', self)
	
	# noise n
	@cached_property  # [..., d, d]
	def Snn(self):
		if 'sn' in vars(self):
			return self.sn.diag_embed()
		return torch.zeros(1, 1)
	
	@cached_property  # [..., d]
	def sn(self):
		if 'Snn' in vars(self):
			return self.Snn.diagonal(0,-1,-2)
		raise MissingInputError('Snn', self) from MissingInputError('sn', self)
	
	# mean function
	@cached_property  # (xi) -> mi
	def m(self): return lambda x: torch.zeros(1, 1)
	
	@cached_property  # (xd) -> md
	def md_f(self): return self.m
	
	@cached_property  # (xq) -> mq
	def mq_f(self): return self.m
	
	# kernel, covariance function
	@cached_property  # (xi, xj') -> Kij
	def K(self): raise MissingInputError('K', self)
	
	@cached_property  # (xd, xd') -> Kdd
	def Kdd_f(self): return self.K
	
	@cached_property  # (xq, xd') -> Kqd
	def Kqd_f(self): return self.K
	
	@cached_property  # (xq, xq') -> Kqq
	def Kqq_f(self): return self.K
	
	# calculate mean m
	@cached_property  # [..., d, 1]
	def md(self):
		return self.md_f(self.xd)
	
	@cached_property  # [..., q, 1]
	def mq(self):
		return self.mq_f(self.xq)
	
	# calculate covariance S
	@cached_property  # [..., d, d]
	def Sdd(self):
		return self.Kdd_ + self.Snn
	
	@cached_property  # [..., d, d]
	def Kdd_(self):
		return self.Kdd_f(self.xd[...,:,None,:], self.xd[...,None,:,:])
	
	@cached_property  # [..., q, d]
	def Sqd(self):
		return self.Kqd_f(self.xq[...,:,None,:], self.xd[...,None,:,:])
	
	@cached_property  # [..., q, q]
	def Sqq(self):
		return self.Kqq_f(self.xq[...,:,None,:], self.xq[...,None,:,:])
	
	@cached_property  # [..., q]
	def sq(self):
		if 'Sqq' in vars(self):
			return super().sq
		return (
			self.Kqq_f(self.xq[...,:,None,:], self.xq[...,:,None,:])  # [..., q, C]
			.unflatten(-2, (self.xq.shape[-2],-1))  # [..., q/C, C, C]
			.diagonal(0,-1,-2)  # [..., q/C, C]
			.flatten(-2,-1)  # [..., q]
		)


class GridGaussProcess(GaussProcess):
	""" Gaussian Process for Gridded Data
		
		xd is a complete meshgrid from 0 to N-1 for each dimension of the grid data.
		Set grid_data with data, or set grid_shape directly for pre-computing xd.
	"""
	@cached_property  # [*grid_shape, C]
	def grid_data(self): raise MissingInputError('grid_data', self)
	
	@cached_property
	def grid_shape(self):
		return self.grid_data.shape[:-1]
	
	@property
	def _xd_grid_shape(self): return self.grid_shape
	
	@cached_property  # [..., d/C, D]
	def xd(self):
		xd = torch.stack(
			torch.meshgrid(
				*(torch.arange(s) for s in self._xd_grid_shape),
				indexing='ij',
			),
			dim=-1,  # along D
		)
		xd = xd.double()  # todo: do I need this?
		xd = xd.reshape(prod(self._xd_grid_shape), -1)
		return xd
	
	@cached_property  # [..., d, 1]
	def yd(self):
		return self.grid_data.reshape(-1, 1)



class IntGridGaussProcess(GridGaussProcess):
	""" Gaussian Process for Gridded Data with an Integral Kernel
		
		xd is a complete meshgrid from 0 to N
		Data kernels are assumed (analytically) integrated, and will be diff-ed.
	"""
	@property
	def _xd_grid_shape(self):
		return torch.Size(s+1 for s in self.grid_data.shape[:-1])
	
	# disable insufficiently vague variables
	@property
	def K(self): raise AttributeError("A single K is not sufficient for an IntGridGaussProcess")
	@property
	def m(self): raise AttributeError("A single m is not sufficient for an IntGridGaussProcess")
	
	# new properties
	@cached_property
	def Kidid_f(self): raise MissingInputError('Kidid_f', self)
	@cached_property
	def Kqid_f(self): raise MissingInputError('Kqid_f', self)
	
	# helper functions
	@staticmethod
	def _diff_kernel(K, diff_left=(), diff_right=()):
		for dim, sizes in ((-2, diff_left), (-1, diff_right)):
			if sizes:  # don't waste time if empty or false
				ndim = len(sizes)
				K = K.unflatten(dim, (*sizes, -1))
				for i in range(dim-ndim, dim):
					K = K.diff(dim=i)
				K = K.flatten(dim-ndim, dim)  # this has 2 off-by-one errors which cancel out
		return K
	
	def _diff_wrapper(self, kernel, diff_left=False, diff_right=False):
		if diff_left  is True: diff_left  = self._xd_grid_shape
		if diff_right is True: diff_right = self._xd_grid_shape
		
		@wraps(kernel)
		def wrapper(*args):
			K = kernel(*args)
			K = self._diff_kernel(K, diff_left, diff_right)
			return K
		return wrapper
	
	
	@cached_property
	def Kdd_f(self):
		return self._diff_wrapper(self.Kidid_f, diff_left=True, diff_right=True)
	
	@cached_property
	def Kqd_f(self):
		return self._diff_wrapper(self.Kqid_f, diff_right=True)
