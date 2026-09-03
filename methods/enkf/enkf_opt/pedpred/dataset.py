# Standard Library
import csv
from math import inf, nan
from math import pi, cos, sin
from numbers import Number
from pathlib import Path
from typing import Callable

# PyPI
import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

# Local
from .grid import Grid, GridData
from .tools import doing


class SeqDataset(Dataset):
	""" Sequence dataset to (input, target) dataset. """
	# TODO: this is nice and general, put it in tools.
	
	def __init__(self, dataset: Dataset, input_len: int, target_len: int, step='input_len'):
		self.dataset = dataset
		self.input_len = input_len
		self.target_len = target_len
		self.step = getattr(self, step) if isinstance(step, str) else step
	
	@property
	def seq_len(self): return self.input_len + self.target_len
	
	def __getitem__(self, item):
		if isinstance(item, slice):
			return (self[i] for i in range(*item.indices(len(self))))
		
		start = item * self.step
		stop = start + self.seq_len
		seq = self.dataset[start : stop]
		# input, target = seq.split([self.input_len, self.target_len])  # difficult with GridData
		input = seq[:self.input_len]
		target = seq[self.input_len:]
		return input, target
	
	def __len__(self):
		return (len(self.dataset) - self.seq_len)//self.step + 1


class GridFromH5Dataset(Dataset):
	""" GridData Dataset from H5 file.
		
		The H5 file should have 3 fileds: "position", "velocity", and "index".
		"position" and "velocity" are [N,2].
		"index" is [M], where each row is (time (s), start, stop, count).
			- Time is given in seconds after epoch (python style).
			- Start and stop give indices into position and velocity, listing the pedestrians at that time.
			- Count is the number of times each pedestrian is repeated (over time) between start and stop. Zero for this index.
		
		The data will be re-indexed at the given period, and cached into the H5 file for next time.
	"""
	def __init__(self, h5_filename: str,
		grid: Grid, period: Number,
		*,
		random_rotation=True,
		kernel=None, kernel_scale=1,
		attach_point_data=False,
		attach_point_time=False,
		normalise_period=True,
	):
		h5 = h5py.File(h5_filename, 'r')
		
		period = float(period)
		index_name = f'index_{period}s'
		if period == 0: index_name = 'index'
		
		if index_name not in h5:
			with doing(f"Building index for {h5_filename} with a period of {period}s"):
				h5.close()
				with h5py.File(h5_filename, 'r+') as h5w:
					index = h5w.create_dataset_like(index_name, h5w['index'], shape=(0,), maxshape=(None,), chunks=True)
					start_time, start, stop, count = h5w['index'][0]
					end_time = start_time + period
					for frame_time, frame_start, frame_stop, frame_count in h5w['index']:
						if frame_time >= end_time:
							index.resize(index.shape[0]+1,0)
							index[-1] = (end_time, start, stop, count)
							end_time = end_time + period
							start = frame_start
							count = 0
						stop = frame_stop
						count += frame_count
					# Last datapoint is possibly incomplete, don't add
					if period is inf:  # unless probably intended, todo fix last minute hacks
						index.resize(index.shape[0]+1,0)
						index[-1] = (end_time, start, stop, count)
				h5 = h5py.File(h5_filename, 'r')
		
		if attach_point_time and 'time' not in h5:
			with doing(f"Adding time data for {h5_filename}"):
				h5.close()
				with h5py.File(h5_filename, 'r+') as h5w:
					times = h5w.create_dataset('time', shape=(h5w['position'].shape[0],), dtype=h5w['index'].dtype[0], fillvalue=nan, chunks=True)
					for frame_time, frame_start, frame_stop, frame_count in h5w['index']:
						times[frame_start:frame_stop] = frame_time
				h5 = h5py.File(h5_filename, 'r')
		
		self.h5 = h5
		self.index = h5[index_name]
		self.grid = grid
		self.period = period
		self.random_rotation = random_rotation
		self.kernel = kernel
		self.kernel_scale = kernel_scale
		self.attach_point_data = attach_point_data
		self.attach_point_time = attach_point_time
		self.normalise_period = self.period if (normalise_period is True) else normalise_period
	
	
	def __getitem__(self, item):
		if isinstance(item, tuple):
			item, grid = item
		else:
			grid = self.grid.rotate_around_centre(torch.rand(())*2*pi) if self.random_rotation else self.grid
		
		if isinstance(item, slice):
			return GridData.stack(tuple(self[i,grid] for i in range(*item.indices(len(self)))))
		
		end_time, start, stop, count = self.index[item]
		pos = torch.as_tensor(self.h5['position'][start:stop])
		vel = torch.as_tensor(self.h5['velocity'][start:stop])
		
		griddata = grid.points2grid(pos, vel,
			kernel=self.kernel, kernel_scale=self.kernel_scale,
			normalise_count=count, normalise_resolution=True, normalise_period=self.normalise_period,
			attach_point_data=self.attach_point_data,
		)
		
		if self.attach_point_time:
			times = torch.as_tensor(self.h5['time'][start:stop])
			zero_time = end_time - (item+1) * self.period
			times = times - zero_time
			if self.normalise_period:
				times = times/self.normalise_period
			griddata.point_time = np.empty((), dtype=object)
			griddata.point_time[()] = times
		
		return griddata
	
	
	def __len__(self) -> Number:
		return len(self.index)


def split_h5(src: str, **splits):
	""" Generate multiple h5 files split from a larger sequence. """
	assert sum(v for v in splits.values()) <= 1.0
	src = Path(src)
	with h5py.File(src, 'r') as hs:
		length = hs['index'].shape[0]
		
		cumfrac = 0.0
		stop = 0
		for name, fraction in splits.items():
			cumfrac += fraction
			start = stop
			stop = int(cumfrac * length)  # round
			filename = src.with_name(f'{src.stem}_{name}').with_suffix(src.suffix)
			with h5py.File(filename, 'w') as hi:
				hi['position'] = h5py.ExternalLink(src.name, "position")
				hi['velocity'] = h5py.ExternalLink(src.name, "velocity")
				hi['index'] = hs['index'][start:stop]


_doc_atc_dataset = """
	- csv reader to point data
	- point data to grid data
	- SeqDataset: grid data to sequences (combine with above?)
	- some kind of collection dataset that points to a set of files
	
	OR
	
	- convert each csv to h5
	- GridFromH5Dataset
	- SeqDataset
	- some kind of collection dataset that points to a set of files
"""


def csv2h5(csv_filename: str):
	csv_filename= Path(csv_filename)
	h5_filename = csv_filename.with_suffix('.h5')
	
	with open(csv_filename) as csv_file:
		reader = csv.reader(csv_file)
		
		with h5py.File(h5_filename, 'w-', track_order=True) as h:
			# NOTE: time is in python format (ie float64 seconds)
			index = h.create_dataset('index', (0,), dtype=[('time','f8'),('start','i'),('stop','i'),('count','i')], maxshape=(None,), chunks=True)
			position = h.create_dataset('position', (0,2), dtype='f4', chunks=True, maxshape=(None,2), fillvalue=nan)
			velocity = h.create_dataset('velocity', (0,2), dtype='f4', chunks=True, maxshape=(None,2), fillvalue=nan)
			
			last_time = nan
			start = 0
			stop = 0
			for row in reader:
				if 0 == reader.line_num % 1e3: print(f"{reader.line_num:12n}")
				
				# ATC format
				time, pid, pos_x_mm, pos_y_mm, pos_z_mm, spd_mm, ang, face = row
				time, pos_x_mm, pos_y_mm, spd_mm, ang = (float(x) for x in (time, pos_x_mm, pos_y_mm, spd_mm, ang))
				pos = (pos_x_mm/1000, pos_y_mm/1000)
				spd = spd_mm/1000
				vel = (spd * cos(ang), spd * sin(ang))
				n = 1  # this is just one pedestrian
				
				# this populates the index record ending with the PREVIOUS row
				if time > last_time:
					index.resize(index.shape[0]+1, 0)
					index[-1] = (time, start, stop, 1)
					start = stop
				last_time = time
				stop += n
				
				for x,d in zip((pos, vel), (position, velocity)):
					d.resize(stop, 0)
					d[stop-n:stop] = x
				
			# add last start-stop to index
			index.resize(index.shape[0]+1, 0)
			index[-1] = (time, start, stop, 1)


class FileListDataset(ConcatDataset):
	def __init__(self, list_file: str, cls: Callable[[Path], Dataset]):
		list_file = Path(list_file)
		print(list_file)
		dir = list_file.parent
		with open(list_file) as f:
			lines = f.readlines()
		valid_lines = [line.split()[0] for line in lines if line.strip()]
		print(valid_lines)
		super().__init__([cls(dir / file) for file in valid_lines])
			# super().__init__([cls(dir/file) for file in (line.split()[0] for line in f.readlines())])


if __name__ == '__main__':
	csv2h5('./data/ATC/atc-20121128_small.csv')
	csv2h5('./data/ATC/atc-20121205_small.csv')
