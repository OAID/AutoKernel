from DefaultCostModel import CostModel
from cost_model import cost_model
from sample_load import *
import NetworkSize as nws

Network = nws.Network
weights_in_path = './test.weights'
weights_out_path = './test.json'
costmodel = CostModel(Network, weights_in_path, weights_out_path)
weights = costmodel.weights.weights


num_stages, pipeline_features, schedule_features, runtime = load_from_file("./matmul_batch_0000_sample_0000.sample")

prediction, loss = cost_model(num_stages, 1, 32, pipeline_features, schedule_features, weights, 0.0001, 0, 0, [runtime], True)
