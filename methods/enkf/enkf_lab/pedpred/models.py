from abc import ABC, abstractmethod
from functools import lru_cache, wraps
import inspect
from math import ceil, floor
from typing import Any, Optional, Sequence, TypeVar, Union

import torch
from torch import Tensor
from torch.nn import Conv2d, ConvTranspose2d, LeakyReLU, Module, Sequential, Sigmoid, Tanh
from torch.nn.modules.conv import _ConvNd

from .grid import GridData
from .tools import Exporter, seq_to_seq

export = Exporter()


class AbstractRNN(Module, ABC):
	@abstractmethod
	def forward(self, input: Tensor, hidden: Optional[Any]=None) -> (Tensor, Any):
		"""A unified interface for RNN blocks.
			Forward method maps (input, hidden_in) to (output, hidden_out).
			Always 2 inputs and 2 outputs,
			indicating which data is passed depth-wise and sequence-wise respectively.
		"""
AbstractRNN.register(torch.nn.RNNBase)  # includes torch.nn.{RNN,LSTM,GRU}
# torch.nn.{RNN,LSTM,GRU}Cell do not follow this convention. Wrapped:
@AbstractRNN.register
class RNNCell(torch.nn.RNNCell):
	def forward(self, input, hidden=None):
		hidden = super().forward(input, hidden)
		return hidden, hidden
@AbstractRNN.register
class LSTMCell(torch.nn.LSTMCell):
	def forward(self, input, hidden=None):
		hidden, cell = super().forward(input, hidden)
		return hidden, (hidden, cell)
@AbstractRNN.register
class GRUCell(torch.nn.GRUCell):
	def forward(self, input, hidden=None):
		hidden = super().forward(input, hidden)
		return hidden, hidden


@seq_to_seq
def pad_same(k,*,i=0,s=1,d=1,o=None):
	assert s==1 or i
	p = ceil(  ((i-1)*(s-1) + d*(k-1))/2 )
	assert (o or i) == floor( ((i-1) + 2*p - d*(k-1))/s )+1
	return p


Conv = TypeVar('Conv')

@AbstractRNN.register
class ConvGRUCell(Module):
	def __init__(self,
		ConvNd: Union[_ConvNd, Module],
		in_channels: int, out_channels: int,
		*,
		kernel_size=None, in_kernel_size=None, hidden_kernel_size=None,
		activation=Tanh(), gate_activation=Sigmoid(),
	):
		super().__init__()
		self.activation = activation
		self.gate_activation = gate_activation
		
		Ci = in_channels
		Ch = out_channels
		
		assert kernel_size is not None or (in_kernel_size is not None and hidden_kernel_size is not None)
		if kernel_size is not None:
			Ki = kernel_size
			Kh = kernel_size
		if in_kernel_size is not None:
			Ki = in_kernel_size
		if hidden_kernel_size is not None:
			Kh = hidden_kernel_size
		
		Cg = 3*Ch  # there are 3 gates in a GRU
		self.conv_i = ConvNd(Ci, Cg, Ki, padding=pad_same(Ki)) if in_channels else None
		self.conv_h = ConvNd(Ch, Cg, Kh, padding=pad_same(Kh))
	
	
	def forward(self, input: Tensor, hidden: Optional[Tensor]=None) -> (Tensor, Tensor):
		# shape:
		#   input:  (B,Ci,H,W)
		#   hidden: (B,Ch,H,W)
		# todo: use neat named tensors (refine_names, align_to)
		
		assert (input is not None) or (hidden is not None)
		
		if input is not None:
			ci = self.conv_i(input)
		else:
			d = hidden.ndim - 2
			ci = torch.zeros(1,3,*(1,)*d, device=hidden.device, dtype=hidden.dtype)
		
		if hidden is not None:
			ch = self.conv_h(hidden)
		else:
			d = input.ndim - 2
			ch = self.conv_h.bias.reshape(1,-1,*(1,)*d)  # just bias, but nasty reshapes
			hidden = 0
		
		# GRU
		ri,zi,ni = ci.chunk(3, dim=1)
		rh,zh,nh = ch.chunk(3, dim=1)
		r = self.gate_activation(ri +   rh)  # reset
		z = self.gate_activation(zi +   zh)  # update
		n = self.activation     (ni + r*nh)  # new
		hidden_out = (1-z)*n + z*hidden
		output = hidden_out  # to be compatible with more complex rnn frameworks
		
		return output, hidden_out


class Conv1dGRU(ConvGRUCell):
	def __init__(self, *args, **kwargs):
		super().__init__(torch.nn.Conv1d, *args, **kwargs)

class Conv2dGRU(ConvGRUCell):
	def __init__(self, *args, **kwargs):
		super().__init__(torch.nn.Conv2d, *args, **kwargs)

class Conv3dGRU(ConvGRUCell):
	def __init__(self, *args, **kwargs):
		super().__init__(torch.nn.Conv3d, *args, **kwargs)


@lru_cache
class InfIterable:
	def __init__(self, value): self.value = value
	def __iter__(self): return self
	def __reversed__(self): return self
	def __next__(self): return self.value
	def __getitem__(self, item): return self.value
	def __contains__(self, item): return item == self.value
	def __eq__(self, other): return self.value == other.value
Nones = InfIterable(None)


@AbstractRNN.register
class SequentialRNNCell(Sequential):
	"""A sequence of (possibly RNN) blocks.
		Has an input->output stream, and a hidden_in->hidden_out stream.
		Designed to be executed iteratively (hidden_out->hidden_in), see SequentialRNN.
		RNN blocks are assumed to have 2 inputs and 2 outputs:
			forward(input, hidden_in) -> (output, hidden_out)
	"""
	def __init__(self, *args):
		super().__init__(*args)
		self.is_rnn = [self.__is_rnn(module) for module in self]
	
	@classmethod
	def __is_rnn(cls, module: Module):
		# return 2 != len(inspect.signature(module.forward).parameters)
		return isinstance(module, AbstractRNN)
	
	def forward(self, input: Tensor, hidden: Tensor=None):
		output = input
		if hidden is None:
			hidden = Nones
		hidden_iter = iter(hidden)
		hidden_out = []
		for module, is_rnn in zip(self, self.is_rnn):
			if is_rnn:
				h_in = next(hidden_iter)
				output, h_out = module(output, h_in)
				hidden_out.append(h_out)
			else:
				output = module(output)
		assert next(hidden_iter, None) is None  # none remain
		return output, hidden_out


class SequentialRNN(SequentialRNNCell):
	"""Iterates over a SequentialRNNCell for each in input."""
	def forward(self, input, hidden=None):
		output = []
		for i in input:
			o, hidden = super().forward(i, hidden)
			if o is not None: output.append(o)
		if output:
			output = torch.stack(output)
		return output, hidden


class Encoder(SequentialRNN):
	"""A SequentialRNN, but with no output (just hidden_out).
		forward(input, hidden) -> hidden
	"""
	def forward(self, input, hidden=None):
		_, hidden = super().forward(input, hidden)
		return hidden


class Forecaster(SequentialRNN):
	"""A SequentialRNN, but with no input (just hidden_in).
		forward(hidden) -> (output, hidden)
	"""
	def __init__(self, *args, horizon=None):
		super().__init__(*args)
		self.horizon = horizon
	
	def forward(self, hidden, *, horizon=None):
		horizon = horizon or self.horizon
		output, hidden = super().forward([None]*horizon, hidden)
		return output, hidden


class EncoderForcaster(Module):
	def __init__(self, encoder, forecaster, horizon=None):
		super().__init__()
		self.encoder = encoder if isinstance(encoder, Encoder) else Encoder(*encoder)
		self.forecaster = forecaster if isinstance(forecaster, Forecaster) else Forecaster(*forecaster)
		self.forecaster.horizon = horizon
	
	def forward(self, input, hidden=None, *, horizon=None):
		hidden = self.encoder(input, hidden)
		hidden = list(reversed(hidden))  # todo: a neater way of reversing, in the forecaster?
		output, _ = self.forecaster(hidden, horizon=horizon)
		return output


@export
class PrecipitationNowcasting(EncoderForcaster):
	def __init__(self, *args, **kwargs):
		rnn_kwargs = dict(in_kernel_size=3, hidden_kernel_size=5, activation=LeakyReLU(0.2))
		
		encoder = SequentialRNN(
			Conv2d   (   1, prev:=       16 , kernel_size=3, stride=1, padding=1), LeakyReLU(0.2),
			Conv2dGRU(prev, prev:=(hid0:=64), **rnn_kwargs),
			Conv2d   (prev, prev:=       96 , kernel_size=3, stride=2, padding=1), LeakyReLU(0.2),
			Conv2dGRU(prev, prev:=(hid1:=96), **rnn_kwargs),
			Conv2d   (prev, prev:=       96 , kernel_size=3, stride=2, padding=1), LeakyReLU(0.2),
			Conv2dGRU(prev, prev:=(hid2:=96), **rnn_kwargs),
		)
		
		forecaster = SequentialRNN(
			Conv2dGRU      (None, prev:=hid2, **rnn_kwargs),
			ConvTranspose2d(prev, prev:=  96, kernel_size=4, stride=2, padding=1), LeakyReLU(0.2),
			Conv2dGRU      (prev, prev:=hid1, **rnn_kwargs),
			ConvTranspose2d(prev, prev:=  96, kernel_size=4, stride=2, padding=1), LeakyReLU(0.2),
			Conv2dGRU      (prev, prev:=hid0, **rnn_kwargs),
			ConvTranspose2d(prev, prev:=  16, kernel_size=3, stride=1, padding=1), LeakyReLU(0.2),
			Conv2d         (prev, prev:=  16, kernel_size=3, stride=1, padding=1), LeakyReLU(0.2),
			Conv2d         (prev, prev:=   1, kernel_size=1, stride=1, padding=0),
		)
		
		super().__init__(encoder, forecaster, *args, **kwargs)


@export
class PedPred(EncoderForcaster):
	def __init__(self, *args, **kwargs):
		# channel size, density + vel_mean + vel_var
		C = 1 + 2 + 1
		
		rnn_kwargs = dict(in_kernel_size=3, hidden_kernel_size=5, activation=LeakyReLU(0.2))
		
		encoder = Encoder(
			Conv2d   (   C, prev:=       16 , kernel_size=3, stride=1, padding=1), LeakyReLU(0.2),
			Conv2dGRU(prev, prev:=(hid0:=64), **rnn_kwargs),
			Conv2d   (prev, prev:=       96 , kernel_size=3, stride=2, padding=1), LeakyReLU(0.2),
			Conv2dGRU(prev, prev:=(hid1:=96), **rnn_kwargs),
			Conv2d   (prev, prev:=       96 , kernel_size=3, stride=2, padding=1), LeakyReLU(0.2),
			Conv2dGRU(prev, prev:=(hid2:=96), **rnn_kwargs),
		)
		
		forecaster = Forecaster(
			Conv2dGRU      (None, prev:=hid2, **rnn_kwargs),
			ConvTranspose2d(prev, prev:=  96, kernel_size=4, stride=2, padding=1), LeakyReLU(0.2),
			Conv2dGRU      (prev, prev:=hid1, **rnn_kwargs),
			ConvTranspose2d(prev, prev:=  96, kernel_size=4, stride=2, padding=1), LeakyReLU(0.2),
			Conv2dGRU      (prev, prev:=hid0, **rnn_kwargs),
			ConvTranspose2d(prev, prev:=  16, kernel_size=3, stride=1, padding=1), LeakyReLU(0.2),
			Conv2d         (prev, prev:=  16, kernel_size=3, stride=1, padding=1), LeakyReLU(0.2),
			Conv2d         (prev, prev:=   C, kernel_size=1, stride=1, padding=0),
		)
		
		super().__init__(encoder, forecaster, *args, **kwargs)
	
	
	def forward(self, input, hidden=None, *, horizon=None):
		""" Do some special interpretation of the channel dimension. """
		input = GridData(input).as_tensor('density','vel_mean','vel_var')
		input = input.transpose(0,1)  # [B,T,C,H,W] -> [T,B,C,H,W]
		output = super().forward(input, hidden, horizon=horizon)
		# output = GridData.from_tensor(output,('logdensity','vel_mean','vel_logvar'))
		
		logdensity, vel_mean, vel_logvar = output.split((1,2,1), dim=-3)
		# logdensity = logdensity.clamp(max=10)
		# vel_mean = vel_mean.clamp(min=-100, max=+100)
		# vel_logvar = vel_logvar.clamp(max=10)
		output = GridData(logdensity=logdensity, vel_mean=vel_mean, vel_logvar=vel_logvar)
		output = GridData(output.transpose(1,0))  # [B,T,C,H,W] <- [T,B,C,H,W]
		
		return output


class PedPred2(EncoderForcaster):
	def __init__(self, *args, **kwargs):
		# channel size, density + vel_mean + vel_var
		C = 1 + 2 + 1

		rnn_kwargs = dict(in_kernel_size=3, hidden_kernel_size=5, activation=LeakyReLU(0.2))

		encoder = Encoder(
			Conv2d(C, prev := 16, kernel_size=3, stride=1, padding=1), LeakyReLU(0.2),
			# Conv2dGRU(prev, prev := (hid0 := 64), **rnn_kwargs),
			Conv2d(prev, prev := 64, kernel_size=3, stride=2, padding=1), LeakyReLU(0.2),
			# Conv2dGRU(prev, prev := (hid1 := 96), **rnn_kwargs),
			Conv2d(prev, prev := 96, kernel_size=3, stride=2, padding=1), LeakyReLU(0.2),
			Conv2dGRU(prev, prev := (hid2 := 96), **rnn_kwargs),
		)

		forecaster = Forecaster(
			Conv2dGRU(None, prev := hid2, **rnn_kwargs),
			ConvTranspose2d(prev, prev := 96, kernel_size=4, stride=2, padding=1), LeakyReLU(0.2),
			# Conv2dGRU(prev, prev := hid1, **rnn_kwargs),
			ConvTranspose2d(prev, prev := 96, kernel_size=4, stride=2, padding=1), LeakyReLU(0.2),
			# Conv2dGRU(prev, prev := hid0, **rnn_kwargs),
			ConvTranspose2d(prev, prev := 16, kernel_size=3, stride=1, padding=1), LeakyReLU(0.2),
			Conv2d(prev, prev := 16, kernel_size=3, stride=1, padding=1), LeakyReLU(0.2),
			Conv2d(prev, prev := C, kernel_size=1, stride=1, padding=0),
		)

		super().__init__(encoder, forecaster, *args, **kwargs)

	def forward(self, input, hidden=None, *, horizon=None):
		""" Do some special interpretation of the channel dimension. """
		input = GridData(input).as_tensor('density', 'vel_mean', 'vel_var')
		input = input.transpose(0, 1)  # [B,T,C,H,W] -> [T,B,C,H,W]
		output = super().forward(input, hidden, horizon=horizon)
		# output = GridData.from_tensor(output,('logdensity','vel_mean','vel_logvar'))

		logdensity, vel_mean, vel_logvar = output.split((1, 2, 1), dim=-3)
		# logdensity = logdensity.clamp(max=10)
		# vel_mean = vel_mean.clamp(min=-100, max=+100)
		# vel_logvar = vel_logvar.clamp(max=10)
		output = GridData(logdensity=logdensity, vel_mean=vel_mean, vel_logvar=vel_logvar)
		output = GridData(output.transpose(1, 0))  # [B,T,C,H,W] <- [T,B,C,H,W]

		return output


class PedPred3(EncoderForcaster):
	def __init__(self, *args, **kwargs):
		# channel size, density + vel_mean + vel_var
		C = 1 + 2 + 1

		rnn_kwargs = dict(in_kernel_size=3, hidden_kernel_size=5, activation=LeakyReLU(0.2))

		encoder = Encoder(
			Conv2d(C, prev := 16, kernel_size=3, stride=1, padding=1), LeakyReLU(0.2),
			# Conv2dGRU(prev, prev := (hid0 := 64), **rnn_kwargs),
			Conv2d(prev, prev := 32, kernel_size=3, stride=2, padding=1), LeakyReLU(0.2),
			# Conv2dGRU(prev, prev := (hid1 := 96), **rnn_kwargs),s
			Conv2d(prev, prev := 64, kernel_size=3, stride=2, padding=1), LeakyReLU(0.2),
			Conv2dGRU(prev, prev := (hid2 := 64), **rnn_kwargs),
		)

		forecaster = Forecaster(
			Conv2dGRU(None, prev := hid2, **rnn_kwargs),
			ConvTranspose2d(prev, prev := 64, kernel_size=4, stride=2, padding=1), LeakyReLU(0.2),
			# Conv2dGRU(prev, prev := hid1, **rnn_kwargs),
			ConvTranspose2d(prev, prev := 64, kernel_size=4, stride=2, padding=1), LeakyReLU(0.2),
			# Conv2dGRU(prev, prev := hid0, **rnn_kwargs),
			ConvTranspose2d(prev, prev := 16, kernel_size=3, stride=1, padding=1), LeakyReLU(0.2),
			Conv2d(prev, prev := 16, kernel_size=3, stride=1, padding=1), LeakyReLU(0.2),
			Conv2d(prev, prev := C, kernel_size=1, stride=1, padding=0),
		)

		super().__init__(encoder, forecaster, *args, **kwargs)

	def forward(self, input, hidden=None, *, horizon=None):
		""" Do some special interpretation of the channel dimension. """
		input = GridData(input).as_tensor('density', 'vel_mean', 'vel_var')
		input = input.transpose(0, 1)  # [B,T,C,H,W] -> [T,B,C,H,W]
		output = super().forward(input, hidden, horizon=horizon)
		# output = GridData.from_tensor(output,('logdensity','vel_mean','vel_logvar'))

		logdensity, vel_mean, vel_logvar = output.split((1, 2, 1), dim=-3)
		# logdensity = logdensity.clamp(max=10)
		# vel_mean = vel_mean.clamp(min=-100, max=+100)
		# vel_logvar = vel_logvar.clamp(max=10)
		output = GridData(logdensity=logdensity, vel_mean=vel_mean, vel_logvar=vel_logvar)
		output = GridData(output.transpose(1, 0))  # [B,T,C,H,W] <- [T,B,C,H,W]
		# print(output.shape)

		return output