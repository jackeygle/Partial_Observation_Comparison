from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import inspect
from signal import SIGINT, signal, strsignal
import sys
from sys import stderr
from typing import Sequence
from weakref import WeakValueDictionary

from tqdm import tqdm as std_tqdm
from tqdm.contrib import DummyTqdmFile


# TODO: separate this out into multiple files


class Exporter:
	def __init__(self):
		global __all__
		__all__ = []
	
	def __call__(self, o):
		global __all__
		__all__.append(o.__name__)
		return o


# todo: make this a class or something, and tidy: it's too messy and too verbose
__doing_indent = 0
@contextmanager
def doing(doing_str="Doing", done_str="Done", fail_str="Failed"):
	global __doing_indent
	print('\t'*__doing_indent+f"{doing_str}...")
	__doing_indent += 1
	try:
		yield
	except Exception:
		__doing_indent -= 1
		print('\t'*__doing_indent+f"...{fail_str}!", file=stderr)
		raise
	else:
		__doing_indent -= 1
		print('\t'*__doing_indent+f"...{done_str}.")


def seq_to_seq(func):
	""" A decorator that will broadcast a function across sequence (tuple, list) inputs. """
	@wraps(func)
	def wrapper(*args, **kwargs):
		callargs = inspect.getcallargs(func, *args, **kwargs)
		lengths = {k: len(v) if isinstance(v, Sequence) else 0 for k,v in callargs.items()}
		max_len = max(lengths.values())
		assert all(l == 0 or l == max_len for l in lengths.values())
		if max_len:
			return tuple(
				func(**{
					k: v[i] if lengths[k] else v
					for k,v in callargs.items()
				})
				for i in range(max_len)
			)
		else:
			return func(**callargs)
	return wrapper


class frozendict(dict):
	def __hash__(self):
		return hash(frozenset(
			(k, frozendict(v) if isinstance(v,dict) else v)
			for k,v in self.items()
		))
	
	@property
	def __frozen_error(self):
		raise AttributeError("frozendict is immutable")
	clear = __frozen_error
	pop = __frozen_error
	popitem = __frozen_error
	setdefault = __frozen_error
	update = __frozen_error
	__delitem__ = __frozen_error
	__setitem__ = __frozen_error


def weakref_cache(func):
	""" This doesn't work because not all objects (builtins) are weak-reference-able. """
	cache = WeakValueDictionary()
	@wraps(func)
	def wrapper(*args, **kwargs):
		callargs = frozendict(inspect.getcallargs(func, *args, **kwargs))
		if callargs not in cache:
			cache[callargs] = func(**callargs)
		return cache[callargs]
	return wrapper


def rzip(*args):
	""" Reversed zip.
		
		Note: this is not the same as
		>>> reversed(zip(a, b, ...))
		if a and b are different lengths.
	"""
	for arg in args: print(arg)
	return zip(*(reversed(arg) for arg in args))


class CatchSignal:
	def __init__(self, signal=SIGINT):
		self.signal = signal
		self.count = 0
		self.prev_handler = None
	
	def handler(self, signum, frame):
		assert signum == self.signal
		self.count += 1
		print(f"Caught signal {strsignal(signum)} ({self.count} time{'' if self.count==1 else 's'})", file=stderr)
		if self.count >= 10:
			print(f"My patience is exhausted.", file=stderr)
			self.prev_handler(signum, frame)
	
	def __enter__(self):
		self.prev_handler = signal(self.signal, self.handler)
		return self
	
	def __exit__(self, exc_type, exc_val, exc_tb):
		signal(self.signal, self.prev_handler)
	
	def __bool__(self):
		return self.count > 0


class decorator_dict(dict):
	"""A dictionary providing a handy decorator method to add functions."""
	def add(self, key):
		assert key not in self
		
		def decorator(func):
			self[key] = func
			return func
		return decorator


def ddict():
	return defaultdict(ddict)


class tqdm(std_tqdm):
	orig_out_err = None
	
	def __init__(self, *args, postfix_value=False, **kwargs):
		if self.orig_out_err is None:
			tqdm.orig_out_err = sys.stdout, sys.stderr
			sys.stdout, sys.stderr = map(DummyTqdmFile, tqdm.orig_out_err)
		
		super().__init__(*args, leave=None, file=self.orig_out_err[1], **kwargs)
		self.postfix_value = postfix_value
	
	def __iter__(self):
		if self.postfix_value:
			for obj in std_tqdm.__iter__(self):
				self.set_postfix_str(obj, refresh=False)
				yield obj
		else:
			yield from std_tqdm.__iter__(self)
