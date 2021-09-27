import numpy as np
import cv2
import onnx
import onnxruntime
import torch
from torch import nn
import array
from op_codegen import cg

class Attr(object):
    def __init__(self, ksize=0, stride=0, pad=0, c_in=0, c_out=0):
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
        self.c_in = c_in
        self.c_out = c_out


class Tensor(object):
    # consumer
    # producer
    def __init__(self, name, dims=None, value=None):
        self.name = name
        self._dims = dims
        self.dims_before_reshape = None
        self.reshaped = 0
        self.idx = -1
        if value == None:
            self._value = np.zeros(self._dims)
        else:
            # assert(self._dims==value.shape)
            self._value = value
        self.producer = []
        self.consumer = []

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, x):
        self._value = x

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, dims):
        self._dims = dims

    def get_size(self):
        s = 1
        if self._dims is not None:
            for i in self._dims:
                s *= i
        return s

    def get_dim_str(self):
        if self._dims == None:
            return ""
        str_dim = [str(i) for i in self._dims]
        return ','.join(str_dim)

    def __repr__(self):
        return "tensor: {}[{}] ({})".format(self.name, self.get_size(),
                                            self.get_dim_str())

    def __str__(self):
        return self.__repr__()

    def dump(self):
        return self.__repr__()


class Node(object):
    def __init__(self,
                 name,
                 op_type=None,
                 input=None,
                 output=None,
                 attr=Attr()):
        self.name = name
        self.op_type = op_type
        self.input = []
        if input is not None:
            self.input += input
        self.output = []
        if output is not None:
            self.output += output
        self.attr = attr

    def __repr__(self):
        # input="inputs: ["+", ".join(self.input)+"]\n"
        # output="outputs: ["+", ".join(self.output)+"]\n"
        return "name:{}\top_type:{}".format(self.name, self.op_type)

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def onnxattr_to_dict(attr):
        ret = dict()
        for i in attr:
            ret[i.name] = i.ints
        return ret

    def get_parents(self):
        parents = set()
        for t in self.input:
            parents.update(t.producer)
        if '_param' in parents:
            parents.remove('_param')
        return parents

    def get_children(self):
        child = set()
        for t in self.output:
            child.update(t.consumer)
        return child

    def dump_output(self, out_dir="data"):
        dims = self.output[0].dims
        dims_str = '_'.join([str(i) for i in dims])
        import array
        data = array.array('f', self.output[0].value.flatten())
        fname = '{}/{}_{}.bin'.format(out_dir, self.name, dims_str)
        with open(fname,'wb') as fp:
            data.tofile(fp)
        print('.... {}'.format(fname))

    def infershape(self):
        t = self.output[0]
        dims = self.input[0].dims

        if 'Conv' in self.op_type:
            n = self.input[0].dims[0]
            out_h = (self.input[0].dims[2] + 2 * self.attr.pad -
                     self.attr.ksize) / self.attr.stride + 1
            out_w = (self.input[0].dims[3] + 2 * self.attr.pad -
                     self.attr.ksize) / self.attr.stride + 1
            dims = [n, self.attr.c_out, int(out_h), int(out_w)]
        if 'MaxPool' in self.op_type:
            out_h = (self.input[0].dims[2] + 2 * self.attr.pad -
                     self.attr.ksize) / self.attr.stride + 1
            out_w = (self.input[0].dims[3] + 2 * self.attr.pad -
                     self.attr.ksize) / self.attr.stride + 1
            dims = [
                self.input[0].dims[0], self.input[0].dims[1],
                int(out_h),
                int(out_w)
            ]
        if 'MatMul' in self.op_type:
            dims = [self.input[0].dims[0], self.input[1].dims[1]]
        if t.reshaped == 0:
            t.dims = dims
        else:
            t.dims_before_reshape = dims

    def codegen(self, halide=1):
        inputs = ', '.join(
            ['float* inp{}'.format(i) for i in range(len(self.input))])
        outputs = ', '.join(
            ['float* out{}'.format(i) for i in range(len(self.output))])

        halide_name = "halide_op"
        others_head = ""
        others = ""
        if 'Conv' in self.op_type:
            halide_name = 'halide_conv'
            others = ", param->stride, param->pad"
        if 'MaxPool' in self.op_type:
            halide_name = 'halide_maxpool'
            others = ", param->ksize, param->stride"
        if 'MatMul' in self.op_type:
            halide_name = 'halide_matmul'
        if 'Relu' in self.op_type:
            halide_name = 'halide_relu'
        head = 'void {}({}, {}, Param* param)'.format(self.op_type, outputs,
                                                      inputs)
        body = ""
        if halide == 1:
            out = '    struct halide_buffer_t {};\n'.format(', '.join(
                ['b_out{}'.format(i) for i in range(len(self.output))]))
            inp = '    struct halide_buffer_t {};\n'.format(', '.join(
                ['b_inp{}'.format(i) for i in range(len(self.input))]))
            body = out + inp
            for i in range(len(self.output)):
                num_dim = len(self.output[i].dims)
                if self.output[i].reshaped == 1:
                    num_dim = len(self.output[i].dims_before_reshape)
                body += '    set_data(&b_out{},param->out{}_dims,{},out{});\n'.format(
                    i, i, len(self.output[i].dims), i)
            for i in range(len(self.input)):
                body += '    set_data(&b_inp{},param->inp{}_dims,{},inp{});\n'.format(
                    i, i, len(self.input[i].dims), i)
            outs = ', '.join(
                ['&b_out{}'.format(i) for i in range(len(self.output))])
            inps = ', '.join(
                ['&b_inp{}'.format(i) for i in range(len(self.input))])

            body += '    {}({}{}, {});\n'.format(halide_name, inps, others,
                                                 outs)
            for i in range(len(self.output)):
                body += '    free(b_out{}.dim);\n'.format(i)
            for i in range(len(self.input)):
                body += '    free(b_inp{}.dim);\n'.format(i)
        code = head + "{\n" + body + "}\n"
        return code


class Graph(object):
    def __init__(self, name, model_path, input_dims, debug = 0):
        self.name = name
        self.load_onnx_model(model_path, input_dims, debug)
        self.onnx_op_flag = 0

    # model
    def load_onnx_model(self, model_path, input_dims, debug=0):
        def get_weight_data(w):
            '''
            in onnx
            float32 data type is 1
            int64 data type is 7
            '''
            if w.data_type == 1:
                return list(w.float_data)
            if w.data_type == 7:
                return list(w.int64_data)

        session = onnxruntime.InferenceSession(model_path, None)
        input_name = session.get_inputs()[0].name
        self.input_name = input_name
        output_name = session.get_outputs()[0].name
        self.output_name = output_name

        onnx_model = onnx.load(model_path)
        graph = onnx_model.graph
        nodes = graph.node

        weight_data = onnx_model.graph.initializer
        w_dict = {}
        for idx, weight in enumerate(weight_data):
            w_dict[weight.name] = idx

        node_dict = {}
        tensor_list_name = set()
        tensor_dict = {}
        for n in nodes:
            tensor_list_name.update(n.input)
            tensor_list_name.update(n.output)
        for t_name in tensor_list_name:
            if t_name in w_dict:
                w = weight_data[w_dict[t_name]]
                tensor_dict[t_name] = Tensor(t_name, w.dims,
                                             get_weight_data(w))
                tensor_dict[t_name].producer.append('_param')
            else:
                tensor_dict[t_name] = Tensor(t_name)

        for n in nodes:
            # node attr/param
            onnxattr = Node.onnxattr_to_dict(n.attribute)
            attr = Attr()
            if 'kernel_shape' in onnxattr:
                attr.ksize = onnxattr['kernel_shape'][0]
            if 'strides' in onnxattr:
                attr.stride = onnxattr['strides'][0]
            if 'pads' in onnxattr:
                attr.pad = onnxattr['pads'][0]
            elif 'auto_pad' in onnxattr:
                attr.pad = int(attr.ksize // 2)
            if n.op_type == 'Conv':
                w = weight_data[w_dict[n.input[1]]]
                attr.c_in = w.dims[1]
                attr.c_out = w.dims[0]
            inputs = []
            outputs = []
            for t in n.input:
                tensor_dict[t].consumer.append(n.name)
                inputs.append(tensor_dict[t])
                if t == input_name:
                    self.input_tensor = tensor_dict[t]
                    tensor_dict[t].dims = input_dims

            for t in n.output:
                tensor_dict[t].producer.append(n.name)
                outputs.append(tensor_dict[t])
                if t == output_name:
                    self.output_tensor = tensor_dict[t]

            node = Node(n.name, n.op_type, inputs, outputs, attr)
            node_dict[n.name] = node

        self.node_dict = node_dict
        if debug:
            for tname in tensor_dict:
                t = tensor_dict[tname]
                print(t, "producer: ", t.producer, "\tconsumer: ", t.consumer)

    def __repr__(self):
        ret = ''
        for name in self.node_dict:
            n = self.node_dict[name]
            # input="inputs: ["+", ".join(n.input)+"]\t"
            # output="outputs: ["+", ".join(n.output)+"]\n"
            # line = name+op+input+output
            ret += (str(n) + "\n")
        return ret

    def __str__(self):
        return self.__repr__()

    def topologic_sort(self):
        #by indegree_dict
        # no depend list (indegree==0)
        indegree = dict()
        no_depend_list = []
        for u in self.node_dict:
            node_u = self.node_dict[u]
            parents = node_u.get_parents()
            indegree[u] = len(parents)
            if indegree[u] == 0:
                no_depend_list.append(u)

        seq_nodes = []
        while len(no_depend_list) > 0:
            u = no_depend_list.pop(0)
            seq_nodes.append(u)
            node_u = self.node_dict[u]
            # remove edges
            for v in node_u.get_children():
                indegree[v] -= 1
                if indegree[v] == 0:
                    no_depend_list.append(v)
        self.seq_nodes = seq_nodes

    def infershape(self):
        self.topologic_sort()
        for n in self.seq_nodes:
            node = self.node_dict[n]
            node.infershape()

    def get_all_tensors(self, debug=0):
        tensor_dict = {}
        tensor_map = {}
        tensor_idx = 0
        for n in self.seq_nodes:
            node = self.node_dict[n]
            for t in node.input:
                if t.name not in tensor_dict:
                    tensor_dict[t.name] = tensor_idx
                    tensor_map['{}'.format(tensor_idx)] = (t)
                    t.idx = tensor_idx
                    tensor_idx += 1
            for t in node.output:
                if t.name not in tensor_dict:
                    tensor_dict[t.name] = tensor_idx
                    tensor_map['{}'.format(tensor_idx)] = (t)
                    t.idx = tensor_idx
                    tensor_idx += 1
        if debug == 1:
            for i in tensor_map:
                print(i, tensor_map[i])
        return tensor_map

    def dump_ir(self):
        self.topologic_sort()
        tensor_dict = {}
        tensor_idx = 0
        ret = ""

        for n in self.seq_nodes:
            node = self.node_dict[n]
            line = ""
            for t in node.input:
                if t.name not in tensor_dict:
                    tensor_dict[t.name] = tensor_idx
                    tensor_idx += 1
                    t_string = "%" + str(
                        tensor_dict[t.name]) + " = " + t.name + "\n"
                    line += t_string
            for t in node.output:
                if t.name not in tensor_dict:
                    tensor_dict[t.name] = tensor_idx
                    tensor_idx += 1
            out_string = ['%' + str(tensor_dict[t.name]) for t in node.output]
            inp_string = ['%' + str(tensor_dict[t.name]) for t in node.input]
            node_string = ','.join(
                out_string) + " = " + node.op_type + "(" + ",".join(
                    inp_string) + ")\n"
            line += node_string
            ret += line
        print(ret)

    def feed_input_data(self, input_data, input_dim):
        x = self.input_tensor
        x.dims = input_dim
        x.value = input_data

    def get_output_data(self):
        return self.output_tensor.value

    def onnx_op_codegen(self):
        with open('data/reg_str', 'r') as f:
            head = f.read()
        with open('generated_op.py', 'w') as f:
            f.write(head)
            gen_list = []
            for n in self.node_dict:
                node = self.node_dict[n]
                if node.op_type not in gen_list:
                    string = cg(node)
                    f.write(string)
                    gen_list.append(node.op_type)

    def run(self,debug=0):
        if self.onnx_op_flag==0:
            self.onnx_op_codegen()
        from generated_op import op_reg

        self.topologic_sort()
        if (debug ==1):
            print("dumping output tensor of each nodes")
        for name in self.seq_nodes:
            node = self.node_dict[name]
            exec_func = op_reg[node.op_type]
            exec_func(node)
            if (debug == 1):
                node.dump_output()

    def gen_main_cpp(self, fname):
        tensor_map = self.get_all_tensors()

        f = open(fname, 'w')

        with open('./data/main_head', 'r') as f0:
            head = f0.read()
        f.write(head)

        codegen_op_impl = ""
        visited = []
        for name in self.seq_nodes:
            node = self.node_dict[name]
            if node.op_type not in visited:
                codegen_op_impl += node.codegen()
                visited.append(node.op_type)
        f.write(codegen_op_impl)
        main_head = '''int main(int argc, char** argv){
    if(argc<3){printf("exe <model> <inp_data>\\n");return 0;}
    char* weight_name=argv[1];
    char* input_data_file=argv[2];\n'''
        f.write(main_head)

        # data
        code_data = '\n    //data\n'
        for idx in tensor_map:
            t = tensor_map[idx]
            code_data += '    float* _{}= (float*)malloc(sizeof(float)*{}); //{}\n'.format(
                idx, t.get_size(), t.name)
        f.write(code_data)

        # load_weight
        f.write("\n    //load_weight\n")
        f.write(
            '    FILE* fp = fopen(weight_name, "rb");\n    if (!fp) printf("data can not be open");\n'
        )
        weight_name = "data/" + self.name + ".weights"
        w_fp = open(weight_name, 'wb')
        code_load_weight = ''
        for idx in tensor_map:
            t = tensor_map[idx]
            if 'Parameter' in t.name:
                w = array.array('f', t.value)
                w.tofile(w_fp)
                with open('data/{}'.format(t.name), 'wb') as fp:
                    w.tofile(fp)
                    print(t.get_size(), t.name)
                code_load_weight += '    fread(_{}, sizeof(float), {}, fp);\n'.format(
                    idx, t.get_size())
        f.write(code_load_weight)
        f.write("    fclose(fp);\n")
        w_fp.close()

        # read input
        f.write("\n    //read input data\n")
        for idx in tensor_map:
            t = tensor_map[idx]
            if t.name == self.input_name:
                f.write(
                    '    read_float_data(_{},{},input_data_file);\n'.format(
                        idx, t.get_size()))

        # shape
        f.write("\n    //data shape\n")
        for idx in tensor_map:
            t = tensor_map[idx]
            str_dims = ','.join([str(i) for i in t.dims[::-1]])
            f.write("    int s_{}[{}]={{{}}};\n".format(
                idx, len(t.dims), str_dims))

        # param
        f.write("\n    //param\n")
        len_node = len(self.seq_nodes)
        for i in range(len_node):
            f.write("    Param param_{};\n".format(i))

        param_line = ''
        # code_inference
        code_inference = '\n    //code_inference\n'
        for n_idx, n in enumerate(self.seq_nodes):
            node = self.node_dict[n]
            out_string = ['_{}'.format(i.idx) for i in node.output]
            inp_string = ['_{}'.format(i.idx) for i in node.input]
            for t_idx, t in enumerate(node.input):
                param_line += '    param_{}.inp{}_dims=s_{};\n'.format(
                    n_idx, t_idx, t.idx)
            for t_idx, t in enumerate(node.output):
                if t.reshaped == 0:
                    param_line += '    param_{}.out{}_dims=s_{};\n'.format(
                        n_idx, t_idx, t.idx)
                else:
                    str_dims = ','.join(
                        [str(i) for i in t.dims_before_reshape[::-1]])
                    param_line += '    int reshape_{}[{}]={{{}}};\n'.format(
                        t.idx, len(t.dims_before_reshape), str_dims)
                    param_line += '    param_{}.out{}_dims=reshape_{};\n'.format(
                        n_idx, t_idx, t.idx)
            if 'Conv' in node.op_type:
                param_line += '    param_{}.stride={};param_{}.pad={};//conv\n'.format(
                    n_idx, node.attr.stride, n_idx, node.attr.pad)
            if 'MaxPool' in node.op_type:
                param_line += '    param_{}.ksize={};param_{}.stride={};//maxpool\n'.format(
                    n_idx, node.attr.ksize, n_idx, node.attr.stride)

            code_inference += "    {}({},{},&param_{});\n".format(
                node.op_type, ','.join(out_string), ",".join(inp_string),
                n_idx)

        f.write(param_line)

        f.write(code_inference)

        f.write("\n    //print output data[:10]\n")
        size = min(10, self.output_tensor.get_size())
        f.write("    p(_{},{});\n".format(self.output_tensor.idx, size))

        f.write("\n    //free data\n")
        for idx in tensor_map:
            f.write("    free(_{});\n".format(idx))

        f.write("    return 0;\n}\n")
        print("done")
