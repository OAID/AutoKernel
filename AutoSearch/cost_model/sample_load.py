import struct
import sys
import os
from os.path import getsize
import NetworkSize as nws
import torch

class PipelineSample:
	pass

class Sample:
	pass

def load_from_file(filename):

	size = getsize(filename)
	assert size%4==0
	floats_read = int(size/4)
	num_features = floats_read - 3
	features_per_stage = nws.head2_w + (nws.head1_w + 1) * nws.head1_h
	assert num_features%features_per_stage==0
	num_stages = int(num_features / features_per_stage)

	with open(filename,"rb") as f:
		out = f.read(size-2*4)
		scratch = struct.unpack(str(floats_read-2)+'f',out)
		out = f.read(2*4)
		id = struct.unpack('2I', out)
	f.close()

	runtime = scratch[num_features]
	pipeline_id = id[0]
	schedule_id = id[1]


	# loading pipeline features
	pipeline_features = torch.empty(nws.head1_w, nws.head1_h, num_stages)
	for i in range(num_stages):
		for x in range(nws.head1_w):
			for y in range(nws.head1_h):
				f = scratch[i * features_per_stage + (x + 1) * 7 + y + nws.head2_w]
				pipeline_features[x, y, i] = f
	print('------------------')
	print('Pipeline feature:')
	print(pipeline_features)
	print(pipeline_features.shape)

	# loading schedule features
	schedule_features = torch.empty(nws.head2_w, num_stages)
	for i in range(num_stages):
		for x in range(nws.head2_w):
			f = scratch[i * features_per_stage + x]
			schedule_features[x, i] = f
	print('------------------')
	print('schedule feature:')
	print(schedule_features)
	print(schedule_features.shape)
	print('------------------')
	return num_stages, pipeline_features, schedule_features, runtime


if __name__ == "__main__":
	load_from_file("./matmul_batch_0000_sample_0000.sample")
	# a = getsize("./matmul_batch_0000_sample_0001.sample")
	# print(a)
	# b = 981-3
	# c = 39+41*7
	# print(b/c)