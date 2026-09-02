from functools import cached_property
from typing import Iterable

import torch
from torch.fft import fft, ifft
from gpytorch.lazy import CatLazyTensor, LazyTensor, cat as LazyCat
from gpytorch.utils.toeplitz import toeplitz_matmul


# patch gpytorch
# I pray this section remains small
LazyTensor.__neg__ = lambda self: self.mul(-1)
LazyTensor.__rsub__ = lambda self, other: -self + other


def MatCat(rows: Iterable[Iterable[LazyTensor]])->CatLazyTensor:
	return LazyCat([LazyCat(row, dim=-1) for row in rows], dim=-2)


def LazyStack(tensors: Iterable[LazyTensor], dim=0)->CatLazyTensor:
	assert dim == 0
	tensors = [tensor.expand(1, *tensor.shape) for tensor in tensors]
	return LazyCat(tensors, dim=dim)


class NonSymToeplitzLazyTensor(LazyTensor):
	def _check_args(self, rrow_column):
		if rrow_column.shape[-1] % 2 != 1:
			return "the length of the rrow_column vector should be odd"
	
	def __init__(self, *args, column=None, row=None, rrow_column=None):
		"""
		Args:
			Either:
			:attr: `rrow_column` (Tensor)
				A 1D Tensor of length `2n-1`, representing the reversed first row
				concatenated to the first column. The middle elements, being identical, are included once.
			OR
			:attr: `column` (Tensor)
				A 1D Tensor of length `n`.
			:attr: `row` (Tensor)
				Corresponding `row` vectors of the same size as `column`.
				The first entry of `row` must match the first entry of `column`.
		"""
		if   len(args) == 0: pass
		elif len(args) == 1 and rrow_column is None: rrow_column, = args
		elif len(args) == 2 and column is row is None: column, row = args
		else: raise TypeError(f"expected at most 2 positional arguments, got {len(args)}")
		
		if rrow_column is None:
			rrow = row[..., 1:].flip(dims=(-1,))
			rrow_column = torch.cat((rrow, column), dim=-1)
		else:
			assert column is row is None
		
		super(NonSymToeplitzLazyTensor, self).__init__(rrow_column)
		self.rrow_column = rrow_column
	
	@property
	def _n(self): return (self.rrow_column.shape[-1] + 1)//2
	
	@property
	def column(self): return self.rrow_column[..., self._n-1:]
	
	@property
	def row(self): return self.rrow_column[..., :self._n].flip(dims=(-1,))
	
	def _expand_batch(self, batch_shape):
		return self.__class__(self.rrow_column.expand(*batch_shape, self.rrow_column.shape[-1]))
	
	def _get_indices(self, row_index, col_index, *batch_indices):
		row_index = row_index.remainder(self._n)
		col_index = col_index.remainder(self._n)
		toeplitz_indices = (row_index - col_index + self._n-1).long()
		return self.rrow_column[(*batch_indices, toeplitz_indices)]
	
	@cached_property
	def _fft(self):
		return fft(self.rrow_column)
	
	def _matmul(self, rhs):
		# output = toeplitz_matmul(self.column, self.row, rhs)
		
		if rhs.ndimension() == 1:
			rhs = rhs.unsqueeze(-1)
		rhs = rhs.mT
		rhs_padded = torch.cat((rhs, torch.zeros(*rhs.shape[:-1], self._n-1)), dim=-1)
		fft_M = fft(rhs_padded)
		fft_c = self._fft
		fft_product = fft_M * fft_c
		output = ifft(fft_product).real.mT
		output = output[..., self._n-1:, :]
		
		return output
	
	def _t_matmul(self, rhs):
		return self._transpose_nonbatch()._matmul(rhs)
	
	def _quad_form_derivative(self, left_vecs, right_vecs):
		raise NotImplementedError("This is the code for a (Symmetric) ToeplitzLazyTensor")
		if left_vecs.ndimension() == 1:
			left_vecs = left_vecs.unsqueeze(1)
			right_vecs = right_vecs.unsqueeze(1)
		
		res = toeplitz_derivative_quadratic_form(left_vecs, right_vecs)
		
		# Collapse any expanded broadcast dimensions
		if res.dim() > self.column.dim():
			res = res.view(-1, *self.column.shape).sum(0)
		
		return (res,)
	
	def _size(self):
		return torch.Size((*self.rrow_column.shape[:-1], self._n, self._n))
	
	def _transpose_nonbatch(self):
		return NonSymToeplitzLazyTensor(self.rrow_column.flip(dims=(-1,)))
	
	def add_jitter(self, jitter_val=1e-3):
		jitter = torch.zeros_like(self.rrow_column)
		jitter.narrow(-1, self._n-1, 1).fill_(jitter_val)
		return NonSymToeplitzLazyTensor(self.rrow_column.add(jitter))
	
	def diag(self):
		"""
		Gets the diagonal of the Toeplitz matrix wrapped by this object.
		"""
		diag_term = self.rrow_column[..., self._n-1]
		if self.rrow_column.ndimension() > 1:
			diag_term = diag_term.unsqueeze(-1)
		return diag_term.expand(*self.batch_shape[:-1], self._n)
