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

def load_a_sample(filename):

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

	runtime = torch.tensor(scratch[num_features])
	pipeline_id = id[0]
	schedule_id = id[1]


	# loading pipeline features
	pipeline_features = torch.empty(num_stages, nws.head1_h, nws.head1_w)
	for i in range(num_stages):
		for x in range(nws.head1_w):
			for y in range(nws.head1_h):
				f = scratch[i * features_per_stage + (x + 1) * 7 + y + nws.head2_w]
				pipeline_features[i, y, x] = f

	# loading schedule features
	schedule_features = torch.empty(num_stages, nws.head2_w)
	for i in range(num_stages):
		for x in range(nws.head2_w):
			f = scratch[i * features_per_stage + x]
			schedule_features[i, x] = f

	return pipeline_features, schedule_features, runtime

def load_from_dir(path):
	samples_path = []
	for dirpath, dirnames, filenames in os.walk(path):
		for file in filenames :  
			if file.endswith('.sample'):
				samples_path.append(os.path.join(dirpath, file))

	assert len(samples_path)>0

	pipeline_features, schedule_features, runtime = load_a_sample(samples_path[0])
	if len(samples_path) == 1:
		pipeline_features = torch.unsqueeze(pipeline_features, 0)
		schedule_features = torch.unsqueeze(schedule_features, 0)
		runtime = torch.unsqueeze(runtime, 0)
	else:
		# pipeline_features = torch.stack([pipeline_features] + [load_a_sample(sample)[0] for sample in samples_path[1:]], dim=3)
		# schedule_features = torch.stack([schedule_features] + [load_a_sample(sample)[1] for sample in samples_path[1:]], dim=2)
		pipeline_features = torch.stack([pipeline_features] + [load_a_sample(sample)[0] for sample in samples_path[1:]])
		schedule_features = torch.stack([schedule_features] + [load_a_sample(sample)[1] for sample in samples_path[1:]])
		runtime = torch.stack([runtime] + [load_a_sample(sample)[2] for sample in samples_path[1:]])

	return pipeline_features, schedule_features, runtime	



if __name__ == "__main__":
	print(load_from_dir('./samples'))
