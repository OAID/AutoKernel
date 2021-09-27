def fuse_two_nodes_impl(graph, op1, op2):
    def fuse_nodes(u, v):
        node_u = graph.node_dict[u]
        node_v = graph.node_dict[v]

        # 1. remove connected_tensor (no need to implt)

        # 2. move v'inputs to node_u
        num_inputs = len(node_v.input)
        for idx in range(1, num_inputs):
            tensor = node_v.input[idx]
            tensor.consumer.remove(node_v.name)
            tensor.consumer.append(node_u.name)
            node_u.input.append(tensor)

        # 3. move v's outputs to node_u
        for t in node_v.output:
            t.producer.remove(node_v.name)
            t.producer.append(node_u.name)
        node_u.output = node_v.output

        # 4. new node
        new_op_type = node_u.op_type + "_" + node_v.op_type + "_fused"
        node_u.op_type = new_op_type

        # 5.remove node
        del graph.node_dict[v]

    name_list = []
    for u in graph.node_dict:
        node_u = graph.node_dict[u]
        if node_u.op_type == op1:
            # u only one output, in this case:
            if len(node_u.output) == 1:
                out_tensor = node_u.output[0]
                # u.children=[v]
                for v in out_tensor.consumer:
                    # print("node_v",v)
                    if graph.node_dict[v].op_type == op2:
                        name_list.append((u, v))
    for (u, v) in name_list:
        fuse_nodes(u, v)
    return graph


class Pass(object):
    def __init__(self, pass_type=None, input_name=[], output_name=[]):
        self.input_name = input_name
        self.output_name = output_name
        self.pass_type = pass_type
        if (pass_type == 'fuse_two_nodes' and len(output_name) == 0):
            self.output_name = ["_".join(input_name) + '_fused']
        self.input_name = input_name
        if len(output_name) >= 1:
            self.output_name = output_name
        self.name = pass_type + '@' + "@".join(
            self.input_name) + "@" + "@".join(self.output_name)
        self.name = pass_type + '@' + "@".join(
            self.input_name) + "@" + "@".join(self.output_name)

    def __repr__(self):
        op = "pass_type: %s\n" % self.pass_type
        input = "inputs: [" + ", ".join(self.input_name) + "]\n"
        output = "outputs: [" + ", ".join(self.output_name) + "]\n"
        return op + input + output

    def __str__(self):
        return self.__repr__()

    def set_impl(self, func):
        self.impl = func


class Pass_Manager(object):
    def __init__(self):
        self.pass_func_dict = {}
        self.pass_pattern_dict = {}

    def __setitem__(self, key, value):
        self.pass_pattern_dict[key] = value

    def register(self, key):
        return lambda func: self.__setitem__(key, func)

    def __getitem__(self, key):
        return self.pass_pattern_dict[key]

    def __contains__(self, key):
        return key in self.pass_pattern_dict

    def keys(self):
        return self.pass_pattern_dict.keys()

    def add(self, pass_type, input_name=[], output_name=[]):
        pass_i = Pass(pass_type, input_name, output_name)
        if pass_type == 'fuse_two_nodes':
            func = self.pass_pattern_dict[pass_type](input_name[0],
                                                     input_name[1])
            pass_i.set_impl(func)
        else:

            func = self.pass_pattern_dict[pass_type]()
            pass_i.set_impl(func)
        self.pass_func_dict[pass_i.name] = pass_i

    def get_seq_pass_list(self):
        dependency = dict()
        for u in self.pass_func_dict:
            dependency[u] = []
            pass_u = self.pass_func_dict[u]
            for t in pass_u.output_name:
                for v in self.pass_func_dict:
                    if v != u:
                        if t in self.pass_func_dict[v].input_name:
                            dependency[u].append(v)
        indegree = dict()
        for u in self.pass_func_dict:
            indegree[u] = 0
        for u in self.pass_func_dict:
            for v in dependency[u]:
                indegree[v] += 1

        no_depend_list = []
        for u in indegree:
            if indegree[u] == 0:
                no_depend_list.append(u)

        pass_seq_list = []
        while len(no_depend_list) > 0:
            u = no_depend_list.pop(0)
            pass_seq_list.append(u)
            # remove edges
            for v in dependency[u]:
                indegree[v] -= 1
                if indegree[v] == 0:
                    no_depend_list.append(v)
        assert (len(pass_seq_list) == len(self.pass_func_dict))
        return pass_seq_list

    def run_all_pass(self, graph):
        pass_seq_list = self.get_seq_pass_list()
        for i in pass_seq_list:
            opt_pass = pass_m.pass_func_dict[i]
            pass_func = opt_pass.impl
            graph = pass_func(graph)
        return graph


pass_m = Pass_Manager()


@pass_m.register("fuse_two_nodes")
def fuse_two_nodes(op1, op2):
    return lambda graph: fuse_two_nodes_impl(graph, op1, op2)


@pass_m.register("flatten_add_param_shape")
def flatten_add_param_shape():
    def flatten_add_param_shape_impl(graph):
        for u in graph.node_dict:
            node_u = graph.node_dict[u]
            if '_Add_fused' in node_u.op_type:
                inp_2 = node_u.input[2]
                size = 1
                for i in inp_2.dims:
                    size *= i
                inp_2.dims = [size]
        return graph

    return flatten_add_param_shape_impl


@pass_m.register("remove_reshape")
def remove_reshape():
    def remove_reshape_impl(graph):
        name_list = []
        for u in graph.node_dict:
            node_u = graph.node_dict[u]
            if node_u.op_type == 'Reshape':
                inp_0 = node_u.input[0]
                inp_1 = node_u.input[1]
                out_0 = node_u.output[0]
                # assert all param from weights
                flag = 1
                for t in node_u.input:
                    if '_param' not in t.producer:
                        flag = 0
                if (flag == 1):
                    # out_tensor.producer = param
                    tensor = node_u.output[0]
                    tensor.producer = ['_param']
                    # set value, dims
                    out_0.dims = inp_1.value
                    out_0.value = inp_0.value
                else:
                    children = node_u.get_children()
                    for v in children:
                        node_v = graph.node_dict[v]
                        for idx, t in enumerate(node_v.input):
                            if t.name == out_0.name:
                                node_v.input[idx] = inp_0
                                inp_0.dims = inp_1.value
                                inp_0.reshaped = 1
                                inp_0.consumer.remove(node_u.name)
                                inp_0.consumer.append(node_v.name)
                # remove node
                name_list.append(u)

        for u in name_list:
            del graph.node_dict[u]
        return graph

    return remove_reshape_impl


pass_m.add('fuse_two_nodes', ['MatMul', 'Add'])
pass_m.add('fuse_two_nodes', ['Conv', 'Add'])
pass_m.add('flatten_add_param_shape')
# pass_m.add('fuse_two_nodes',['Conv_Add_fused','Relu'])
pass_m.add('remove_reshape')

# print(pass_m.get_seq_pass_list())
