import struct
import json

# Read weights
def get_dict(file_name,Network):
    Weights = {}
    with open(file_name,"rb") as f: 

        # Signature
        out = f.read(1*4)
        Weights['Signature'] = struct.unpack('1I',out)

        # Versions
        out = f.read(1*4)
        Weights['Pipeline_version'] = struct.unpack('1I',out)
        out = f.read(1*4)
        Weights['Schedule_version'] = struct.unpack('1I',out)

        # Buffer count
        out = f.read(1*4)
        Weights['Buffer_count'] = struct.unpack('1I',out)

        # Each buffer
        for name in Network:
            out = f.read(1*4)
            Weights[name + '_dimension'] = struct.unpack('1I',out)

            dimension = Weights[name + '_dimension'][0]
            out = f.read(dimension*4)
            Weights[name + '_extent'] = struct.unpack(str(dimension)+'I',out)

            elements = 1
            for i in Weights[name + '_extent']:
                elements *= i
            out=f.read(elements*4)
            Weights[name + '_data'] =struct.unpack(str(elements)+'f',out)
    f.close()
    return Weights

# Write weights dict into json
def dict_to_json(Weights_dict, output):
    Weights_json = json.dumps(Weights_dict,sort_keys=False, indent=4, separators=(',', ': '))
    with open(output,'w') as f:
        f.write(Weights_json)
    f.close()

# Load weights dict from json
def load_json(file_name):
    with open(file_name,'r') as f:
        Weights_json = f.read()
        Weights_dict = json.loads(Weights_json)
    f.close()
    return Weights_dict

# pack weights dict in binary
def to_binary(file_name, Weights_dict):
    with open(file_name, 'wb') as f:
        for v in Weights_dict.values():
            fmt = ''
            for i in v:
                if type(i).__name__ == 'int':
                    fmt += 'I'
                elif type(i).__name__ == 'float':
                    fmt += 'f'
            s = struct.pack(fmt,*v)
            f.write(s)
    f.close()

if __name__ == '__main__':
    file_name = "baseline.weights"
    Network = ['head1_filter', 'head1_bias', 'head2_filter', 'head2_bias', 'conv1_filter', 'conv1_bias']
    file_out = "weights.json"
    Weights_dict = get_dict(file_name, Network)
    dict_to_json(Weights_dict, file_out)
    Weights_dict = load_json(file_out)
    to_binary('test.weights', Weights_dict)