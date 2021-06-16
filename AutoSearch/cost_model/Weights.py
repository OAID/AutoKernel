import struct
import json

class Weights:

    def __init__(self, Network):
        self.network = Network
        self.weights = {}

    def load_from_file(self, filename):
        with open(filename,"rb") as f: 

            # Signature
            out = f.read(1*4)
            self.weights['Signature'] = struct.unpack('1I',out)

            # Versions
            out = f.read(1*4)
            self.weights['Pipeline_version'] = struct.unpack('1I',out)
            out = f.read(1*4)
            self.weights['Schedule_version'] = struct.unpack('1I',out)

            # Buffer count
            out = f.read(1*4)
            self.weights['Buffer_count'] = struct.unpack('1I',out)

            # Each buffer
            for name in self.network:
                out = f.read(1*4)
                self.weights[name + '_dimension'] = struct.unpack('1I',out)

                dimension = self.weights[name + '_dimension'][0]
                out = f.read(dimension*4)
                self.weights[name + '_extent'] = struct.unpack(str(dimension)+'I',out)

                elements = 1
                for i in self.weights[name + '_extent']:
                    elements *= i
                out=f.read(elements*4)
                self.weights[name + '_data'] =struct.unpack(str(elements)+'f',out)
        f.close()

    def save_to_json(self, filename):
        Weights_json = json.dumps(self.weights,sort_keys=False, indent=4, separators=(',', ': '))
        with open(filename,'w') as f:
            f.write(Weights_json)
        f.close()

    def load_from_json(self, filename):
        with open(filename,'r') as f:
            Weights_json = f.read()
            self.weights = json.loads(Weights_json)
        f.close()

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            for v in self.weights.values():
                fmt = ''
                for i in v:
                    if type(i).__name__ == 'int':
                        fmt += 'I'
                    elif type(i).__name__ == 'float':
                        fmt += 'f'
                s = struct.pack(fmt,*v)
                f.write(s)
        f.close()




if __name__ == "__main__" :
    Network = ['head1_filter', 'head1_bias', 'head2_filter', 'head2_bias', 'conv1_filter', 'conv1_bias']
    weights = Weights(Network)
    weights.load_from_file("./baseline.weights")
    weights.save_to_json("./weights.json")
    weights.load_from_json("./weights.json")
    weights.save_to_file("./test.weights")