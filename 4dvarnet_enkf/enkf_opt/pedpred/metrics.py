import torch

from .grid import GridData


class Metrics(dict):
	def __class_getitem__(cls, item):
		return lambda pred, target: cls(pred, target)[item]
	
	def __init__(self, pred, target):
		super().__init__()
		self['prediction'] = GridData(pred)
		self['target'] = GridData(target)
	
	def __missing__(self, key: str):
		value = self.calculate(key)
		self[key] = value
		return value
	
	images = {  # to plot with add_image, or add_histogram
		# 'total square error',
			'square error density',
		# 	'square error vel_est',
		# 	'square error vel_unc',
		'total weighted square error',
			'weighted square error vel_est',
			'weighted square error vel_unc',
		# old (for backward comparison)
		# 	'square error vel_mean',
		# 	'square error vel_std',
			'weighted square error vel_mean',
			'weighted square error vel_std',
		# 'total KL',
		# 	'KL density',
		# 	'KL velocity',
		# 	'KL vel_est',
		# 	'KL vel_unc',
		# 'total weighted KL',
		# 	'weighted KL velocity',
		# 		'weighted KL vel_est',
		# 		'weighted KL vel_unc',
		# 'total NLLL',
			'NLLL density',
		# 	'NLLL velocity',
		# 		'NLLL vel_est',
		# 		'NLLL vel_unc',
		'total weighted NLLL',
			'weighted NLLL velocity',
				'weighted NLLL vel_est',
				'weighted NLLL vel_unc',
	}
	scalars = {
		*{'mean '+ metric for metric in images},
		# 'error count',  # not sure how to use this across multiple samples. Just look at 'mean square error density'.
	}
	
	def calculate(self, key: str):
		prefix,_,remainder = key.partition(' ')
		
		if prefix in {'prediction','target'}:
			# special keywords
			if remainder == 'vel_est': return self[prefix+_+'vel_mean']  # velocity ESTimate
			if remainder == 'vel_unc': return self[prefix+_+'vel_std']   # velocity UNCertainty
			if remainder == 'opt_spd': return (self['square'+_+prefix+_+'vel_mean'] + self[prefix+_+'vel_var']).sqrt()
			if remainder == 'min_cost_per_dist': return self[prefix+_+'density'] * self[prefix+_+'vel_var'] / self[prefix+_+'opt_spd']
			# get values from GridData
			return getattr(self[prefix], remainder)
		
		# point-wise operations (note: abs and square reduce channel dim to 1)
		if prefix == 'error':       return self['prediction'+_+remainder] - self['target'+_+remainder]
		if prefix == 'abs':         return self[remainder].abs().sum(dim=-3, keepdim=True)
		if prefix == 'square':      return self[remainder].square().sum(dim=-3, keepdim=True)
		if prefix == 'root':        return self[remainder].sqrt()
		if prefix == 'weighted':
			if 'density' in remainder:  return self[remainder]  # don't weight density
			else:                       return self['target density'] * self[remainder]
		
		# reduction operations
		# todo: provide non-time dimension reductions, to see prediction degrade into future
		if prefix == 'mean':    return self[remainder].mean()
		if prefix == 'sum':     return self[remainder].sum()
		
		# combination across channels
		# note: 'total' must prefix 'weighted' if you don't want to weight density
		if prefix == 'total':
			return self[remainder+_+'density'] + self[remainder+_+'vel_est'] + self[remainder+_+'vel_unc']
		
		# KL Divergence things
		if prefix == 'KL':
			if remainder == 'density':
				value = self['target density'] * -self['error logdensity']  +  self['error density']
				value[self['target density'] == 0] = 0  # limit of (x log x) as x->0 is 0
				return value
			if remainder == 'vel_est': return 0.5 * self['square error vel_mean'] / self['prediction vel_var']
			if remainder == 'vel_unc':  return self['target vel_var'] / self['prediction vel_var']  -  1  +  self['error vel_logvar']
			if remainder == 'velocity': return self[prefix+_+'vel_est'] + self[prefix+_+'vel_unc']
		
		# NLLL: Negative Log Likelihood Loss
		# Not true NLL (aka cross-entropy), not true KL; additive constants (wrt prediction) discarded
		if prefix == 'NLLL':
			if remainder == 'density': return self['prediction density'] - self['target density'] * self['prediction logdensity']
			if remainder == 'vel_est': return 0.5 * self['square error vel_mean'] / self['prediction vel_var']  # exactly as KL
			if remainder == 'vel_unc': return self['target vel_var'] / self['prediction vel_var'] + self['prediction vel_logvar']
			if remainder == 'velocity': return self[prefix+_+'vel_est'] + self[prefix+_+'vel_unc']
		
		# calculate metric for each time slice. It's not awfully clever, use it carefully.
		if prefix == 'timewise':
			next_op,_,remainder = remainder.partition(' ')
			tensor: torch.Tensor = self[remainder]
			dims = set(range(tensor.ndim)) - {1}
			if next_op == 'mean': return tensor.mean(dim=dims)
			if next_op == 'min': return tensor.min(dim=dims).values
			if next_op == 'max': return tensor.max(dim=dims).values
		
		raise KeyError(key)
