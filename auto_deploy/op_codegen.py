
def get_conv_body(idx,start_idx,tmp,flag):
    input_0 = 'inp_0_tensor'
    if start_idx!=0:
        input_0='tmp_{}'.format(tmp-1)
    param_name = 'param_{}'.format(idx)
    if flag==0:
        line = '''    {} = node.attr
    conv_{} = nn.Conv2d({}.c_in, {}.c_out, {}.ksize, {}.stride, {}.pad,1,1,False)
    conv_{}.weight.data = inp_{}_tensor
    tmp_{} = conv_{}({})
'''.format(param_name,idx,param_name,param_name,param_name,param_name,param_name,idx,start_idx+1,tmp,idx,input_0)
    if flag==1:
        line = '''    {} = node.attr
    conv_{} = nn.Conv2d({}.c_in, {}.c_out, {}.ksize, {}.stride, {}.pad,1,1,True)
    conv_{}.weight.data = inp_{}_tensor
    conv_{}.bias.data = inp_{}_tensor
    tmp_{} = conv_{}({})
'''.format(param_name,idx,param_name,param_name,param_name,param_name,param_name,
        idx,start_idx+1,
        idx,start_idx+2,
        tmp,idx,input_0)
    return line
def get_maxpool_body(idx,start_idx,tmp):
    input_0 = 'inp_0_tensor'
    if start_idx!=0:
        input_0='tmp_{}'.format(tmp-1)
    param_name = 'param_{}'.format(idx)
    line = '''    {} = node.attr
    maxpool_{} = nn.MaxPool2d({}.ksize, {}.stride, {}.pad)
    tmp_{} = maxpool_{}({})
'''.format(param_name,idx,param_name,param_name,param_name,tmp,idx,input_0)
    return line
def get_relu_body(idx,start_idx,tmp):
    input_0 = 'inp_0_tensor'
    if start_idx !=0:
        input_0 = 'tmp_{}'.format(tmp-1)
    line = '''    relu_{} = nn.ReLU()
    tmp_{} = relu_{}({})
'''.format(idx,tmp,idx,input_0)
    return line
def get_matmul_body(idx,start_idx,tmp):
    input_0 = 'inp_0_tensor'
    input_1 = 'inp_1_tensor'
    if start_idx!=0:
        input_0 = 'tmp_{}'.format(tmp-1)
        input_1 = 'inp_{}_tensor'.format(start_idx)
    line = '''    tmp_{} = torch.matmul({},{})\n'''.format(tmp,input_0,input_1)
    return line
def get_add_body(idx,start_idx,tmp):
    input_0 = 'inp_0_tensor'
    input_1 = 'inp_1_tensor'
    if start_idx!=0:
        input_0 = 'tmp_{}'.format(tmp-1)
        input_1 = 'inp_{}_tensor'.format(start_idx)
    line = '''    tmp_{} = torch.add({},{})\n'''.format(tmp,input_0,input_1)
    return line

def codegen_node(node):

    # func parameter node
    op_type = node.op_type
    op_list = op_type.split("_")
    op_list = [i for i in op_list if i!='fused']
    num_op = len(op_list)

    num_input = len(node.input)
    num_output = len(node.output)

    ############### head
    head = "@op_reg.register(\"{}\")\ndef run_{}(node):\n".format(op_type,op_type)

    ############### input
    input_declare = ""
    for i in range(num_input):
        input_declare+="    inp_{} = node.input[{}]\n".format(i,i)
    input_tensor = ""
    for i in range(num_input):
        input_tensor += "    inp_{}_tensor = torch.tensor(np.array(inp_{}.value,dtype=np.float32).reshape(inp_{}.dims))\n".format(i,i,i)
    input = input_declare + input_tensor

    ############## body 
    start_idx = 0
    body_idx = 0
    body =""
    for body_idx in range(num_op):
        op = op_list[body_idx]
        if op=="Conv":
            flag = 0
            if op_list[body_idx+1]=='Add':
                flag = 1
            body += get_conv_body(body_idx,start_idx,body_idx,flag)
            start_idx+=2
        if op=="MaxPool":
            body += get_maxpool_body(body_idx,start_idx,body_idx)
            start_idx+=1
        if op=="Relu":
            body += get_relu_body(body_idx,start_idx,body_idx)
        if op=="MatMul":
            body += get_matmul_body(body_idx,start_idx,body_idx)
            start_idx+=2
        if op=="Add":
            if op_list[body_idx-1]=='Conv':
                continue
            body += get_add_body(body_idx,start_idx,body_idx)
            start_idx+=2
        body_idx+=1
    ############## output
    output = ""
    assert(num_output==1)
    i=0
    line = "    out = np.array(tmp_{})\n".format(body_idx-1)
    if 'Conv' in op_type:
        line = "    out = tmp_{}.detach().numpy()\n".format(body_idx-1)
    line1 = "    out_{} = node.output[{}]\n".format(i,i)
    line2 = "    out_{}.value=out\n".format(i)
    line3 = "    if out_{}.reshaped==0:\n".format(i)
    line4 = "        out_{}.dims=out.shape\n".format(i)
    output=(line+line1+line2+line3+line4)

    op_string = head + input + body + output
    # print(op_string)
    return op_string


cg = codegen_node