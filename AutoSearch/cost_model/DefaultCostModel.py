from Weights import Weights

class CostModel:
    
    def __init__(self, Network, weights_in_path, weights_out_path):
        self.weights_in_path = weights_in_path
        self.weights_out_path = weights_out_path
        self.weights = Weights(Network)        
        self.weights.load_from_file(self.weights_in_path)
    
    def set_pipeline_features():
        pass
    
    def enqueue():
        pass
    
    def backprop():
        pass

    def evaluate_costs():
        if (cursor == 0 and not schedule_feat_queue.data()): return None

        assert pipeline_feat_queue.data()
        assert schedule_feat_queue.data()

        result = cost_model(num_stages,
                            cursor,
                            num_cores,
                            pipeline_feat_queue,
                            schedule_feat_queue,
                            weights)

    def load_weights(self):
        self.weights.load_from_file(self.weights_in_path)

    def save_weights(self):
        self.weights.save_to_file(self.weights_out_path)


if __name__ == '__main__':
    Network = ['head1_filter', 'head1_bias', 'head2_filter', 'head2_bias', 'conv1_filter', 'conv1_bias']
    weights_in_path = './test.weights'
    weights_out_path = './test.json'
    costmodel = CostModel(Network, weights_in_path, weights_out_path)
    weights = costmodel.weights
    weights.save_to_json(costmodel.weights_out_path)
    print(weights)
