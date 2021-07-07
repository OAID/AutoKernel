import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import NetworkSize as nws
from torch.utils.data import Dataset, DataLoader
from sample_load import load_from_dir
from Weights import Weights

class CSVDataset(Dataset):
	# load the dataset
	def __init__(self, path):
		# store the inputs and outputs
		data = load_from_dir(path)
		self.Xp = data[0]
		self.Xs = data[1]
		self.y = data[2]
	
	# number of rows in the dataset
	def __len__(self):
		return len(self.y)
	
	# get a row at an index
	def __getitem__(self, idx):
		return [self.Xp[idx], self.Xs[idx], self.y[idx]]

def inverse_sigmoid(tensor):
	return torch.log(tensor/(1-tensor))

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(40*7, 8)
		self.fc2 = nn.Linear(39, 24)
		self.fc3 = nn.Linear(32, 32)
	
	def network(self, pipeline_features, schedule_features):
		xs = torch.log(schedule_features + 1)
		xp = pipeline_features.flatten(-2, -1)
		self.fc1.weight = nn.Parameter(torch.sigmoid(self.fc1.weight))
		xp = self.fc1(xp)
		self.fc1.weight = nn.Parameter(inverse_sigmoid(self.fc1.weight))
		xs = F.relu(self.fc2(xs))
		conv1_stack = torch.cat((xp, xs), -1)
		conv1_stage2 = self.fc3(conv1_stack)
		conv1_relu = F.relu(conv1_stage2)
		return conv1_relu, conv1_stage2
	
	def forward(self, data, num_cores):

		pipeline_features = data[0]
		schedule_features = data[1]
		conv1_relu = self.network(pipeline_features, schedule_features)[0]

		# That's the end of the neural network. Now we will use these
		# coefficients with a bunch of hand-designed terms.

		# Unpack all of the schedule features. We don't use all of
		# them, but it's easier to avoid bugs if we just unpack them
		# all in the same order as Featurization.h
		idx = 0
		num_realizations = schedule_features[:, :, idx]
		idx+=1
		num_productions = schedule_features[:, :, idx]
		idx+=1
		points_computed_per_realization = schedule_features[:, :, idx]
		idx+=1
		points_computed_per_production = schedule_features[:, :, idx]
		idx+=1
		points_computed_total = schedule_features[:, :, idx]
		idx+=1
		points_computed_minimum = schedule_features[:, :, idx]
		idx+=1
		innermost_loop_extent = schedule_features[:, :, idx]
		idx+=1
		innermost_pure_loop_extent = schedule_features[:, :, idx]
		idx+=1
		unrolled_loop_extent = schedule_features[:, :, idx]
		idx+=1
		inner_parallelism = schedule_features[:, :, idx]
		idx+=1
		outer_parallelism = schedule_features[:, :, idx]
		idx+=1
		bytes_at_realization = schedule_features[:, :, idx]
		idx+=1
		bytes_at_production = schedule_features[:, :, idx]
		idx+=1
		bytes_at_root = schedule_features[:, :, idx]
		idx+=1
		innermost_bytes_at_realization = schedule_features[:, :, idx]
		idx+=1
		innermost_bytes_at_production = schedule_features[:, :, idx]
		idx+=1
		innermost_bytes_at_root = schedule_features[:, :, idx]
		idx+=1
		inlined_calls = schedule_features[:, :, idx]
		idx+=1
		unique_bytes_read_per_realization = schedule_features[:, :, idx]
		idx+=1
		unique_lines_read_per_realization = schedule_features[:, :, idx]
		idx+=1
		allocation_bytes_read_per_realization = schedule_features[:, :, idx]
		idx+=1
		working_set = schedule_features[:, :, idx]
		idx+=1
		vector_size = schedule_features[:, :, idx]
		idx+=1
		native_vector_size = schedule_features[:, :, idx]
		idx+=1
		num_vectors = schedule_features[:, :, idx]
		idx+=1
		num_scalars = schedule_features[:, :, idx]
		idx+=1
		scalar_loads_per_vector = schedule_features[:, :, idx]
		idx+=1
		vector_loads_per_vector = schedule_features[:, :, idx]
		idx+=1
		scalar_loads_per_scalar = schedule_features[:, :, idx]
		idx+=1
		bytes_at_task = schedule_features[:, :, idx]
		idx+=1
		innermost_bytes_at_task = schedule_features[:, :, idx]
		idx+=1
		unique_bytes_read_per_vector = schedule_features[:, :, idx]
		idx+=1
		unique_lines_read_per_vector = schedule_features[:, :, idx]
		idx+=1
		unique_bytes_read_per_task = schedule_features[:, :, idx]
		idx+=1
		unique_lines_read_per_task = schedule_features[:, :, idx]
		idx+=1
		working_set_at_task = schedule_features[:, :, idx]
		idx+=1
		working_set_at_production = schedule_features[:, :, idx]
		idx+=1
		working_set_at_realization = schedule_features[:, :, idx]
		idx+=1
		working_set_at_root = schedule_features[:, :, idx]
		idx+=1


		# Count up the number of things computed, applying a
		# different cost of vectors and scalars, and a different cost
		# depending on whether we were inlined
		choiselist = [vector_size * num_vectors * conv1_relu[:, :, 0] + num_scalars * conv1_relu[:, :, 1],
					vector_size * num_vectors * conv1_relu[:, :, 2] + num_scalars * conv1_relu[:, :, 3]]
		compute_cost = torch.where(inlined_calls == 0, choiselist[0], choiselist[1])

		# Round up these costs according to how neatly we're using
		# our cores.
		num_tasks = torch.maximum(inner_parallelism * outer_parallelism, torch.ones(inner_parallelism.shape))
		tasks_per_core = num_tasks / num_cores
		idle_core_wastage = torch.ceil(tasks_per_core) / torch.maximum(tasks_per_core, torch.ones(tasks_per_core.shape))
		compute_cost = compute_cost * idle_core_wastage

		# Next comes a long list of plausible terms to capture the cost of loads.
		load_cost = (num_realizations * unique_lines_read_per_realization * conv1_relu[:, :, 4] +
					num_realizations * unique_bytes_read_per_realization * conv1_relu[:, :, 6] +
					num_vectors * scalar_loads_per_vector * conv1_relu[:, :, 7] +
					num_scalars * scalar_loads_per_scalar * conv1_relu[:, :, 8] +
					num_vectors * vector_loads_per_vector * conv1_relu[:, :, 9] +
					num_scalars * unique_bytes_read_per_vector * conv1_relu[:, :, 10] +
					num_vectors * unique_bytes_read_per_vector * conv1_relu[:, :, 11] +
					num_scalars * unique_lines_read_per_vector * conv1_relu[:, :, 12] +
					num_vectors * unique_lines_read_per_vector * conv1_relu[:, :, 13] +
					num_tasks * unique_bytes_read_per_task * conv1_relu[:, :, 14] +
					num_tasks * unique_lines_read_per_task * conv1_relu[:, :, 15])
		
		# Next we have the cost of stores.
		lines_written_per_realization = inner_parallelism * (bytes_at_task / torch.maximum(innermost_bytes_at_task, torch.ones(innermost_bytes_at_task.shape)))

		# Use separate coefficients for things with internal
		# parallelism, because for stages with internal parallelism,
		# most values of the values being stored will be consumed on
		# another core, so they will get punted out to L3 no matter
		# how small. Also use a separate term for the final stage, as
		# we never pay the cost of loading from it.
		alpha = conv1_relu[:, :, 18].clone()
		alpha[:, 0] = conv1_relu[:, 0, 17]
		choiselist = [conv1_relu[:, :, 16], alpha]
		alpha = torch.where(inner_parallelism > 1, choiselist[0], choiselist[1])

		beta = conv1_relu[:, :, 21].clone()
		beta[:, 0] = conv1_relu[:, 0, 20]
		choiselist = [conv1_relu[:, :, 19], beta]
		beta = torch.where(inner_parallelism > 1, choiselist[0], choiselist[1])

		store_cost = num_realizations * (lines_written_per_realization * alpha + 
										bytes_at_realization * beta)

		# Now account for false sharing of cache lines. The
		# probability of a store hitting a cache line also hit by
		# another core is inversely proportional to
		# innermost_bytes_at_task, and the cost is paid on every
		# store.
		choiselist = [conv1_relu[:, :, 22] * (num_vectors + num_scalars) / torch.maximum(torch.ones(innermost_bytes_at_task.shape), innermost_bytes_at_task), torch.zeros(store_cost.shape)]
		cost_of_false_sharing = torch.where(inner_parallelism > 1, choiselist[0], choiselist[1])

		store_cost = store_cost + cost_of_false_sharing

		# Now add a term for false sharing of pages. The maximum
		# number of threads that could all fault on the same page at
		# the same time is:
		max_threads_hitting_same_page_fault = torch.minimum(inner_parallelism, 4096 / torch.maximum(torch.ones(innermost_bytes_at_task.shape), innermost_bytes_at_task))

		# The total number of page faults is proportionate to the number of bytes allocated
		num_page_faults = bytes_at_production

		# And page faults are serviced serially, so the total CPU time gets multiplied by the thread count again!
		cost_of_page_faults = (num_page_faults * max_threads_hitting_same_page_fault *
							inner_parallelism * outer_parallelism * conv1_relu[:, :, 23])

		store_cost = store_cost + cost_of_page_faults

		# Malloc is not free, so add a cost per allocation.
		cost_of_malloc = conv1_relu[:, :, 24] * num_realizations

		# A cost for launching a parallel task...
		choiselist = [conv1_relu[:, :, 25], torch.zeros(conv1_relu[:, :, 25].shape)]
		cost_of_parallel_launches = num_productions * torch.where(inner_parallelism > 1, choiselist[0], choiselist[1])

		# ... and an overhead per task.
		cost_of_parallel_tasks = num_productions * (inner_parallelism - 1) * conv1_relu[:, :, 26]

		cost_of_parallelism = cost_of_parallel_tasks + cost_of_parallel_launches
		# Make it easier for the model to penalize working sets that
		# start to fall out of cache by giving it a term that gets
		# multiplied by the working set.
		cost_of_working_set = working_set * conv1_relu[:, :, 27]

		# FIXME: For our best set of trained weights, store_cost was
		# accidentally in the list below twice, so we double it here
		# in order to not have to retrain.
		store_cost = store_cost * 2

		cost = (compute_cost +
				store_cost +
				load_cost +
				cost_of_malloc +
				cost_of_parallelism +
				cost_of_working_set)

		for i in range(32):
			cost = cost + 0.0 * conv1_relu[:, :, i]
		
		# Change units so that network weights are in a human-readable range.
		runtime_per_stage = cost * 1e-9

		# Sum across the stages.
		prediction_output = torch.sum(runtime_per_stage, axis = 1)
		# print('Prediction result :', prediction_output)


		return prediction_output

	def loss_func(self, data, prediction_output):
        # The tail end of the reverse-mode pipeline

        # We believe the coefficients on all the various
        # components of cost should be positive, even before the
        # relu, and even before schedule-specific features are
        # taken into account. The network shouldn't be telling us
        # that things would be cheaper if we would do more
        # mallocs, or compute more values, or launch more
        # parallel tasks. So we add a regularization term. This
        # helps dead relus get unstuck.

		conv1_stage2 = self.network(data[0], data[1])[1]
		regularize = torch.sum(-torch.minimum(conv1_stage2,torch.zeros(conv1_stage2.shape)),axis = (1,2))

        # Our loss will be L2 on relative throughput.

        # Get the reference runtime.
		# reference = torch.tensor(reference)
		true_runtime = data[2]
		scale = 1.0 / torch.max(true_runtime)

        # Compute the relatvie ture runtime and the relative predicted runtime
		p1 = prediction_output * scale
		r1 = scale * true_runtime

        # Inbert them to get relative throughput, and compute L2 loss.
		delta = torch.pow(1.0 / torch.maximum(p1, 1e-10*torch.ones(p1.shape)) - 1.0 / r1, 2)

        # Add the regulization with a small weight.
		err = delta + 1e-5 * regularize

        # Sum the errors over the batch.
		loss = torch.sum(err)

		return loss



data = CSVDataset("./samples2")
dataloader = DataLoader(data, batch_size=4, shuffle=False)

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-5)
for epoch in range(700):
# for epoch in range(1):
	for d in dataloader:
		optimizer.zero_grad()
		cost = net.forward(d, 32)
		loss = net.loss_func(d, cost)
		loss.backward()
		optimizer.step()
		if epoch % 25 == 0:
			print("epoch {:>3d}: loss = {:>8.3f} cost = {}".format(epoch, loss, cost))

# 权重后处理和导出
w1 = net.fc1.weight
w11 = w1.reshape((8, 40, 7))
w12 = w11.permute((2,1,0))
print(w12.shape)
net.fc1.weight = nn.Parameter(w12)

w2 = net.fc2.weight
w22 = w2.permute((1,0))
print(w22.shape)
net.fc2.weight = nn.Parameter(w22)

w3 = net.fc3.weight
w33 = w3.permute((1,0))
print(w33.shape)
net.fc3.weight = nn.Parameter(w33)

weights = Weights(nws.Network)
weights.load_from_model(net)
weights.save_to_json('./fist_test.json')
weights.save_to_file('./first_test.weights')
