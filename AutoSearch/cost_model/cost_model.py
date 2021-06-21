import numpy as np
import torch
import torch.optim as optim
import NetworkSize as nws

def cost_model(num_stages,        # Number of pipline stages
               batch_size,        # Batch size.
               num_cores,         # Number of cores on the target machine.
               pipeline_features, # Algorithm-specific features
               schedule_features, # Schedule-specific features
               weights,           # Network weights
               # Extra inputs for training mode
               learning_rate,     # 
               timestep,          # Need by ADAM
               reference,         # The index of the fastest schedule in the batch
               true_runtime,
               training):     # The true runtimes obtained by benchmarking

    def activation(e):
        zero = torch.zeros(e.shape)
        return torch.maximum(e, zero)
    
    # Network define
    # 这里看起来可能有些复杂，原因是weights的data是nhwc排布的，需要调整一下维度
    def extra_weights(name):
        shape = weights[name + '_extent']
        values = weights[name + '_data']
        tensor_values = torch.tensor(values)
        shape_list = [i for i in shape]
        shape_list.reverse()
        tensor2 = tensor_values.reshape(shape_list)
        permute_list = list(range(len(shape_list)))
        permute_list.reverse()
        tensor = tensor2.permute(permute_list)
        tensor.requires_grad = True
        return tensor
    
    # Expand a dimension and repeat data
    def expand_dims(obj, idx, size):
        dims_list = [i for i in obj.shape]
        dims_list.insert(0, size)
        permute_list = list(range(1, len(dims_list)))
        permute_list.insert(idx, 0)
        return obj.expand(dims_list).permute(permute_list)
    
    def backprop(loss, params, learning_rate):
        loss.backward()
        print('------------')
        print('example of gradien (head2_bias):')
        print(head2_bias.grad)
        optimizer = optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-5)
        optimizer.step()
    
    head1_filter = extra_weights('head1_filter')
    head1_bias = extra_weights('head1_bias')
    head2_filter = extra_weights('head2_filter')
    head2_bias = extra_weights('head2_bias')
    conv1_filter = extra_weights('conv1_filter')
    conv1_bias = extra_weights('conv1_bias')

    # 目前只测试一个batch所以这样拓展，但实际上要根据不同的batch拼接出第三个维度,这需要之后根据外部的函数进行修改
    schedule_features = schedule_features.expand(batch_size, nws.head2_w, num_stages)
    normalized_schedule_features = torch.log(schedule_features + 1)

    # Force the weights of the algorithm embedding layer to be positive and bounded.
    squashed_head1_filter = torch.sigmoid(head1_filter)

    # Explicitly broadcast the weights across the batch
    W = pipeline_features.shape[-1]
    squashed_head1_filter_broadcast = expand_dims(squashed_head1_filter, 1, W)

    # The conv layer that embeds the algorithm-specific features.
    head1_conv_init = expand_dims(head1_bias, 1, W)
    head1_conv = head1_conv_init.clone()
    C,W,S,N = squashed_head1_filter_broadcast.shape
    for c in range(C):
        for w in range(W):
            for x in range(S):
                for y in range(N):
                    head1_conv[c, w] += squashed_head1_filter_broadcast[c, w, x, y] * pipeline_features[x, y, w]

    # No point in a relu - the inputs and weights are positive

    # The conv layer that embeds the schedule-specific features.
    C = head2_bias.shape[0]
    N,R,W = normalized_schedule_features.shape
    head2_conv_init1 = expand_dims(head2_bias,1, W)
    head2_conv_init2 = expand_dims(head2_conv_init1,2, N)
    head2_conv = head2_conv_init2.clone()
    for c in range(C):
        for w in range(W):
            for n in range(N):
                for r_head2 in range(R):
                    head2_conv[c, w, n] += head2_filter[c, r_head2] * normalized_schedule_features[n, r_head2, w]
    
    head2_relu = activation(head2_conv)

    # The conv layer that computes coefficients, split into two
    # stages. First we consumer the algorithm ambedding.
    C,W = head1_conv.shape
    R = nws.head1_channels
    conv1_stage1_init = expand_dims(conv1_bias, 1, W)
    conv1_stage1 = conv1_stage1_init.clone()
    for c in range(C):
        for w in range(W):
            for r1_stage1 in range(R):
                conv1_stage1[c, w] += conv1_filter[c, r1_stage1] * head1_conv[r1_stage1, w]
    
    # Then we consume the schedule embedding.
    R, W, N = head2_relu.shape
    conv1_stage2_init = expand_dims(conv1_stage1, 2, N)
    conv1_stage2 = conv1_stage2_init.clone()
    for c in range(C):
        for w in range(W):
            for n in range(N):
                for r1_stage2 in range(R):
                    conv1_stage2[c, w, n] += conv1_filter[c, head1_filter.shape[0] + r1_stage2] * head2_relu[r1_stage2, w, n]
    
    conv1_relu = activation(conv1_stage2)

    # That's the end of the neural network. Now we will use these
    # coefficients with a bunch of hand-designed terms.

    # Unpack all of the schedule features. We don't use all of
    # them, but it's easier to avoid bugs if we just unpack them
    # all in the same order as Featurization.h
    idx = 0
    num_realizations = schedule_features[:, idx, :].T
    idx+=1
    num_productions = schedule_features[:, idx, :].T
    idx+=1
    points_computed_per_realization = schedule_features[:, idx, :].T
    idx+=1
    points_computed_per_production = schedule_features[:, idx, :].T
    idx+=1
    points_computed_total = schedule_features[:, idx, :].T
    idx+=1
    points_computed_minimum = schedule_features[:, idx, :].T
    idx+=1
    innermost_loop_extent = schedule_features[:, idx, :].T
    idx+=1
    innermost_pure_loop_extent = schedule_features[:, idx, :].T
    idx+=1
    unrolled_loop_extent = schedule_features[:, idx, :].T
    idx+=1
    inner_parallelism = schedule_features[:, idx, :].T
    idx+=1
    outer_parallelism = schedule_features[:, idx, :].T
    idx+=1
    bytes_at_realization = schedule_features[:, idx, :].T
    idx+=1
    bytes_at_production = schedule_features[:, idx, :].T
    idx+=1
    bytes_at_root = schedule_features[:, idx, :].T
    idx+=1
    innermost_bytes_at_realization = schedule_features[:, idx, :].T
    idx+=1
    innermost_bytes_at_production = schedule_features[:, idx, :].T
    idx+=1
    innermost_bytes_at_root = schedule_features[:, idx, :].T
    idx+=1
    inlined_calls = schedule_features[:, idx, :].T
    idx+=1
    unique_bytes_read_per_realization = schedule_features[:, idx, :].T
    idx+=1
    unique_lines_read_per_realization = schedule_features[:, idx, :].T
    idx+=1
    allocation_bytes_read_per_realization = schedule_features[:, idx, :].T
    idx+=1
    working_set = schedule_features[:, idx, :].T
    idx+=1
    vector_size = schedule_features[:, idx, :].T
    idx+=1
    native_vector_size = schedule_features[:, idx, :].T
    idx+=1
    num_vectors = schedule_features[:, idx, :].T
    idx+=1
    num_scalars = schedule_features[:, idx, :].T
    idx+=1
    scalar_loads_per_vector = schedule_features[:, idx, :].T
    idx+=1
    vector_loads_per_vector = schedule_features[:, idx, :].T
    idx+=1
    scalar_loads_per_scalar = schedule_features[:, idx, :].T
    idx+=1
    bytes_at_task = schedule_features[:, idx, :].T
    idx+=1
    innermost_bytes_at_task = schedule_features[:, idx, :].T
    idx+=1
    unique_bytes_read_per_vector = schedule_features[:, idx, :].T
    idx+=1
    unique_lines_read_per_vector = schedule_features[:, idx, :].T
    idx+=1
    unique_bytes_read_per_task = schedule_features[:, idx, :].T
    idx+=1
    unique_lines_read_per_task = schedule_features[:, idx, :].T
    idx+=1
    working_set_at_task = schedule_features[:, idx, :].T
    idx+=1
    working_set_at_production = schedule_features[:, idx, :].T
    idx+=1
    working_set_at_realization = schedule_features[:, idx, :].T
    idx+=1
    working_set_at_root = schedule_features[:, idx, :].T
    idx+=1
    assert idx == head2_filter.shape[-1]

    # Count up the number of things computed, applying a
    # different cost of vectors and scalars, and a different cost
    # depending on whether we were inlined
    choiselist = [vector_size * num_vectors * conv1_relu[0,:,:] + num_scalars * conv1_relu[1,:,:],
                  vector_size * num_vectors * conv1_relu[2,:,:] + num_scalars * conv1_relu[3,:,:]]
    compute_cost = torch.where(inlined_calls == 0, choiselist[0], choiselist[1])

    # Round up these costs according to how neatly we're using
    # our cores.
    num_tasks = torch.maximum(inner_parallelism * outer_parallelism, torch.ones(inner_parallelism.shape))
    tasks_per_core = num_tasks / num_cores
    idle_core_wastage = torch.ceil(tasks_per_core) / torch.maximum(tasks_per_core, torch.ones(tasks_per_core.shape))
    compute_cost = compute_cost * idle_core_wastage

    # Next comes a long list of plausible terms to capture the cost of loads.
    load_cost = (num_realizations * unique_lines_read_per_realization * conv1_relu[5, :, :] +
                 num_realizations * unique_bytes_read_per_realization * conv1_relu[6, :, :] +
                 num_vectors * scalar_loads_per_vector * conv1_relu[7, :, :] +
                 num_scalars * scalar_loads_per_scalar * conv1_relu[8, :, :] +
                 num_vectors * vector_loads_per_vector * conv1_relu[9, :, :] +
                 num_scalars * unique_bytes_read_per_vector * conv1_relu[10, :, :] +
                 num_vectors * unique_bytes_read_per_vector * conv1_relu[11, :, :] +
                 num_scalars * unique_lines_read_per_vector * conv1_relu[12, :, :] +
                 num_vectors * unique_lines_read_per_vector * conv1_relu[13, :, :] +
                 num_tasks * unique_bytes_read_per_task * conv1_relu[14, :, :] +
                 num_tasks * unique_lines_read_per_task * conv1_relu[15, :, :])
    
    # Next we have the cost of stores.
    lines_written_per_realization = inner_parallelism * (bytes_at_task / torch.maximum(innermost_bytes_at_task, torch.ones(innermost_bytes_at_task.shape)))

    # Use separate coefficients for things with internal
    # parallelism, because for stages with internal parallelism,
    # most values of the values being stored will be consumed on
    # another core, so they will get punted out to L3 no matter
    # how small. Also use a separate term for the final stage, as
    # we never pay the cost of loading from it.
    alpha = conv1_relu[18,:,:].clone()
    alpha[0,:] = conv1_relu[17,0,:]
    choiselist = [conv1_relu[16,:,:], alpha]
    alpha = torch.where(inner_parallelism > 1, choiselist[0], choiselist[1])

    beta = conv1_relu[21,:,:].clone()
    beta[0,:] = conv1_relu[20,0,:]
    choiselist = [conv1_relu[19,:,:], beta]
    beta = torch.where(inner_parallelism > 1, choiselist[0], choiselist[1])

    store_cost = num_realizations * (lines_written_per_realization * alpha + 
                                     bytes_at_realization * beta)

    # Now account for false sharing of cache lines. The
    # probability of a store hitting a cache line also hit by
    # another core is inversely proportional to
    # innermost_bytes_at_task, and the cost is paid on every
    # store.
    choiselist = [conv1_relu[22,:,:] * (num_vectors + num_scalars) / torch.maximum(torch.ones(innermost_bytes_at_task.shape), innermost_bytes_at_task), torch.zeros(store_cost.shape)]
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
                           inner_parallelism * outer_parallelism * conv1_relu[23, :, :])

    store_cost = store_cost + cost_of_page_faults

    # Malloc is not free, so add a cost per allocation.
    cost_of_malloc = conv1_relu[24, :, :] * num_realizations

    # A cost for launching a parallel task...
    choiselist = [conv1_relu[25,:,:], torch.zeros(conv1_relu[25,:,:].shape)]
    cost_of_parallel_launches = num_productions * torch.where(inner_parallelism > 1, choiselist[0], choiselist[1])

    # ... and an overhead per task.
    cost_of_parallel_tasks = num_productions * (inner_parallelism - 1) * conv1_relu[26, :, :]

    cost_of_parallelism = cost_of_parallel_tasks + cost_of_parallel_launches
    # Make it easier for the model to penalize working sets that
    # start to fall out of cache by giving it a term that gets
    # multiplied by the working set.
    cost_of_working_set = working_set * conv1_relu[27, :, :]

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
        cost = cost + 0.0 * conv1_relu[i, :, :]
    
    # Change units so that network weights are in a human-readable range.
    runtime_per_stage = cost * 1e-9

    # Sum across the stages.
    prediction_output = torch.sum(runtime_per_stage, axis = 0)
    print('Prediction result :', prediction_output)
    print('True runtime is :', true_runtime)

    if not training:
        loss_output = 0.0
    else:
        # The tail end of the reverse-mode pipeline

        # We believe the coefficients on all the various
        # components of cost should be positive, even before the
        # relu, and even before schedule-specific features are
        # taken into account. The network shouldn't be telling us
        # that things would be cheaper if we would do more
        # mallocs, or compute more values, or launch more
        # parallel tasks. So we add a regularization term. This
        # helps dead relus get unstuck.

        regularize = torch.sum(-torch.minimum(conv1_stage2,torch.zeros(conv1_stage2.shape)),axis = (0,1))

        # Our loss will be L2 on relative throughput.

        # Get the reference runtime.
        reference = torch.tensor(reference)
        true_runtime = torch.tensor(true_runtime)
        n2 = reference.clamp(0, batch_size-1)
        scale = 1.0 / true_runtime[n2]

        # Compute the relatvie ture runtime and the relative predicted runtime
        p1 = prediction_output * scale
        r1 = scale * true_runtime

        # Inbert them to get relative throughput, and compute L2 loss.
        delta = torch.pow(1.0 / torch.maximum(p1, 1e-10*torch.ones(p1.shape)) - 1.0 / r1, 2)

        # Add the regulization with a small weight.
        err = delta + 1e-5 * regularize

        # Sum the errors over the batch.
        loss = torch.sum(err)
        print('L2 loss: ', loss)

        loss_output = loss.clone()

        # Backprop
        params = [head1_filter, head1_bias, head2_filter, head2_bias, conv1_filter, conv1_bias]
        backprop(loss, params, learning_rate)
        
    return prediction_output, loss_output
   
