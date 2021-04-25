"""
tools for autoschedulers and compile for different platform
"""

import argparse
import platform
import os
import time
from collections import namedtuple
import string

HALIDE_HOME = os.environ['HALIDE_HOME']
CURRENT_DIR = os.getcwd()
DEMO_NAME = None
def run_cmd(cmd,display=False):
    '''
    run command in terminal
    '''
    if display:
        print(cmd)
    os.system(cmd)

def insert_line(file_dir, content):
    '''
    insert a line in front of file
    '''
    with open(file_dir, 'r+') as f:
        origin = f.read()        
        f.seek(0, 0)
        f.write(content+'\n'+origin)

def reset():
    '''
    delete some temp files
    '''
    run_cmd('rm -rf %s/samples' % CURRENT_DIR)
    run_cmd('rm -rf %s/temp' % CURRENT_DIR)
    run_cmd('mkdir %s/samples' % CURRENT_DIR)

def find_generator_name(filename):
    import re
    for line in open(filename, "r"):
        if line.startswith('HALIDE_REGISTER_GENERATOR'):
            line = re.sub(r'[( )]','',line.strip()).split(',')[1]
            return line
        else:
            pass
    return 'demo'

def get_shape(cfg):
    from shape_config import shape_dict,args_dict
    #demo_name = find_generator_name(cfg.gen)
    shapes = shape_dict[DEMO_NAME]
    args = args_dict[DEMO_NAME]
    return shapes,args
def generate_compile_file(cfg):
    '''
    compile the file by config
    '''
    global DEMO_NAME
    run_cmd('cp %s %s/samples/demo_gen.cpp' % (cfg.gen, CURRENT_DIR))
    run_cmd('cp %s/template/gen.cpp %s/samples/gen.cpp' % (CURRENT_DIR, CURRENT_DIR))
    run_cmd('cp %s/template/demo_run.cpp %s/samples/demo_run.cpp' % (CURRENT_DIR, CURRENT_DIR))
    run_cmd('g++ %s/samples/demo_gen.cpp %s/samples/gen.cpp -g -I %s/include -I %s/../src/ -L %s/bin -lHalide -ldl -lpthread -std=c++11 -fno-rtti -o demo_gen' 
    % (CURRENT_DIR, CURRENT_DIR, HALIDE_HOME, CURRENT_DIR, HALIDE_HOME))
    #demo_name = find_generator_name(cfg.gen)
    if not cfg.autotune:
        # No autotune
        run_cmd('LD_LIBRARY_PATH=%s/bin %s/demo_gen -g %s -e static_library,c_header,assembly,object,registration -o . target='% (HALIDE_HOME, CURRENT_DIR, DEMO_NAME)
        +cfg.target)
    else:
        data_mode = ''
        if cfg.datatransform:
            # set HL_USE_DATA_TRANSFORM True will do data transform
            data_mode +='HL_USE_DATA_TRANSFORM=True '
        if 'cuda' not in cfg.target:
            if "opencl" in cfg.target:
                # use li2018 for opencl gpu autotune
                run_cmd(data_mode+'''LD_LIBRARY_PATH=%s/bin %s/demo_gen -g %s -f %s -e static_library,assembly,h,schedule,registration -p %s/../build/src/li2018/libautoschedule_li2018.so -s Li2018 target='''%
                (HALIDE_HOME, CURRENT_DIR, DEMO_NAME, DEMO_NAME, CURRENT_DIR)+cfg.target+
                ''' auto_schedule=true machine_params=32,16777216,40 -o .''')
            elif "x86"  in cfg.target:
                # In x86 and cpu, using the autotune.sh in adams2019 will get a better result but time-cost.
                os.environ['HL_TUNE_NUM_BATCHES'] = str(cfg.num_tune_loops)
                os.environ['HL_TUNE_BATCHES_SIZE'] = str(cfg.batch_size)
                run_cmd('mkdir temp')
                run_cmd(data_mode+'''bash ../src/adams2019/autotune_loop.sh ./demo_gen %s %s ../src/adams2019/baseline.weights \
                ../build/src/adams2019/ %s ./temp > \
                ./temp/compile_log.txt''' % (DEMO_NAME,cfg.target+'-avx-avx2-f16c-fma-sse41', HALIDE_HOME ))
                best_file_dir = None
                name = None
                for line in open('./temp/best.%s.benchmark.txt' % DEMO_NAME, "r"):
                    if line.startswith('Benchmark for demo produces best case'):
                        line = line.split('of')[1].split('sec')[0]
                        cost = float(line)
                        break
                    elif line.startswith('Best runtime is'):
                        best_file_dir = os.path.split(line.split('file')[1])[0]
                        name = os.path.split(line.split('file')[1])[1].split('.')[0]
                        line = line.split('is')[1].split('msec')[0]
                        cost = float(line) / 1e3
                    else:
                        pass
                run_cmd('cp %s/%s.a ./samples/' % (best_file_dir,name))
                run_cmd('cp %s/%s.s ./samples/' % (best_file_dir,name))
                run_cmd('cp %s/%s.schedule.h ./samples/' % (best_file_dir,name))
                run_cmd('cp %s/%s.registration.cpp ./samples/' % (best_file_dir,name))
                run_cmd('cp %s/%s.h ./samples/' % (best_file_dir,name))
                run_cmd('rm -rf ./temp')
                run_cmd('rm %s/demo_gen' % CURRENT_DIR)
                DEMO_NAME = name
                return
            else:
                # if not x86, we could not use the adams2019's autotune.sh to search the best schedule
                run_cmd(data_mode+'''LD_LIBRARY_PATH=%s/bin %s/demo_gen -g %s -f %s -e static_library,assembly,h,schedule,registration -p %s/../build/src/adams2019/libautoschedule_adams2019.so -s Adams2019 target='''%
             (HALIDE_HOME, CURRENT_DIR, DEMO_NAME, DEMO_NAME, CURRENT_DIR)+cfg.target+
            ''' auto_schedule=true machine_params=32,16777216,40 -o .''')
        else:
            # use sioutas2020 for gpu autotune CUDA
            run_cmd(data_mode+'''LD_LIBRARY_PATH=%s/bin %s/demo_gen -g %s -f %s -e static_library,assembly,h,schedule,registration -p %s/../build/src/sioutas2020/libautoschedule_sioutas20.so -s Sioutas20 target='''%
             (HALIDE_HOME, CURRENT_DIR, DEMO_NAME, DEMO_NAME, CURRENT_DIR)+cfg.target+
            ''' auto_schedule=true machine_params=32,16777216,40 -o .''')
    run_cmd('mv %s/%s.* %s/samples/' % (CURRENT_DIR, DEMO_NAME, CURRENT_DIR))
    run_cmd('rm %s/demo_gen' % CURRENT_DIR)
def compute_cost_time(input_shape,cfg):
    '''
    Calculate the execution time by modifying the shape and other information in demorun.cpp File.
    The benchmark of halide is the key function
    '''
    target = cfg.target
    #demo_name = find_generator_name(cfg.gen)
    num_io = len(input_shape)
    input_name = ["_"+str(name) for name in range(num_io)]
    input_define = "#define INPUT_TEMPLATE Buffer<float> "
    init_define = "#define INIT_INPUT "
    args_define = "#define DEMO_ARGS "
    output_define = "#define OUTPUT "+input_name[-1]
    include_name = "#include \"{}.h\"\n".format(DEMO_NAME)
    func_define = "#define FUNC {}\n".format(DEMO_NAME)
    sample_define = "#define SAMPLES {}\n".format(cfg.compute_samples)
    iterator_define = "#define ITERATORS {}\n".format(cfg.num_iterators)
    for i in range(num_io):
        if i!=0:
            input_define +=','
            args_define +=','
        if i!=num_io-1:
            init_define +='init({});'.format(input_name[i])
        input_define +=input_name[i]
        input_define +='('
        for j in range(len(input_shape[i])):
            if j!=0:
                input_define +=','
            input_define += str(input_shape[i][j])
        input_define +=')'
        args_define += input_name[i]
    insert_line('%s/samples/demo_run.cpp' % CURRENT_DIR, init_define)
    insert_line('%s/samples/demo_run.cpp' % CURRENT_DIR,sample_define )
    insert_line('%s/samples/demo_run.cpp' % CURRENT_DIR,iterator_define )
    insert_line('%s/samples/demo_run.cpp' % CURRENT_DIR,input_define)
    insert_line('%s/samples/demo_run.cpp' % CURRENT_DIR,args_define)
    insert_line('%s/samples/demo_run.cpp' % CURRENT_DIR,output_define)
    insert_line('%s/samples/demo_run.cpp' % CURRENT_DIR,func_define)
    insert_line('%s/samples/demo_run.cpp' % CURRENT_DIR,include_name)
    if "arm" not in target:
        # if platform is x86, we will excute the demo_run right now and get the result.
        #run_cmd('c++ -std=c++11 -I %s/src/runtime -I %s/tools ./RunGenMain.cpp ./samples/*.registration.cpp ./samples/*.a -o demo_run -DHALIDE_NO_PNG -DHALIDE_NO_JPEG -ldl -lpthread'
                 #% (HALIDE_HOME,HALIDE_HOME))
        run_cmd('g++ %s/samples/demo_run.cpp %s/samples/%s.a -I %s/halide-build/inclue -I %s/tools -ldl -lpthread -o demo_run'
        % (CURRENT_DIR, CURRENT_DIR, DEMO_NAME, HALIDE_HOME, HALIDE_HOME))
        print("begin compute time for {}".format(input_shape))
        #HL_NUM_THREADS=32 timeout -k 60s 60s ./temp/batch_1_0/0/bench --estimate_all --benchmarks=all
        #run_cmd('HL_NUM_THREADS=32 timeout -k 60s 60s ./demo_run --estimate_all --benchmarks=all')
        run_cmd('%s/demo_run' % CURRENT_DIR)
        run_cmd('mv %s/demo_run %s/samples/demo_run' % (CURRENT_DIR, CURRENT_DIR))
    else:
        # if platform is arm, we will not excute the demo_run, but generate one which could run on arm.
        #run_cmd('aarch64-linux-gnu-g++ -std=c++11 -I %s/src/runtime -I %s/tools ./RunGenMain.cpp ./samples/*.registration.cpp ./samples/*.a -o demo_run -DHALIDE_NO_PNG -DHALIDE_NO_JPEG -ldl -lpthread'
                 #% (HALIDE_HOME,HALIDE_HOME))
        run_cmd('aarch64-linux-gnu-g++-8 %s/samples/demo_run.cpp %s/samples/%s.s -I %s/src/runtime -I %s/tools -ldl -lpthread -o demo_run'
        % (CURRENT_DIR, CURRENT_DIR, DEMO_NAME, HALIDE_HOME, HALIDE_HOME))
        run_cmd('mv %s/demo_run %s/samples/demo_run' % (CURRENT_DIR, CURRENT_DIR))

def compile_file(cfg):
    global DEMO_NAME
    reset()
    DEMO_NAME = find_generator_name(cfg.gen)
    if cfg.compute_time:
        shapes, args = get_shape(cfg)
        for shape,arg in zip(shapes,args):
            os.environ['HL_APP_ARGS'] = ', '.join(map(str, arg))
            new_name = os.environ['HL_APP_ARGS'].replace(' ','').replace(',','_')
            generate_compile_file(cfg)
            compute_cost_time(shape,cfg)
            changeFileName('%s/samples/' % CURRENT_DIR, '%s.'% DEMO_NAME,'demo_{}.'.format(new_name))
            changeFileName('%s/samples/' % CURRENT_DIR, 'demo_run','demo_{}_run'.format(new_name))
    else:
        _,args = get_shape(cfg)
        if len(args)>0:
            for arg in args:
                os.environ['HL_APP_ARGS'] = ', '.join(map(str, arg))
                new_name = os.environ['HL_APP_ARGS'].replace(' ','').replace(',','_')
                print("begin compile {}".format(new_name ))
                generate_compile_file(cfg)
                changeFileName('%s/samples/' % CURRENT_DIR, '%s.' % DEMO_NAME,'demo_{}.'.format(new_name))
        else:
            print("begin compile {}".format(DEMO_NAME))
            generate_compile_file(cfg)
    print("compile finished , you can find the file in {}".format(CURRENT_DIR+'/samples/'))
    
def changeFileName(path,src,tar):
    '''
    change the file in <path> from src to tar
    '''
    f_list = os.listdir(path)
    for i in f_list:
        if src in i:
            old_name = i
            new_name = old_name.replace(src,tar)
            run_cmd('cp {}/samples/{} {}/samples/{}'.format(CURRENT_DIR,i,CURRENT_DIR,new_name))

if __name__ == "__main__":
    # cfg = load_config('config.yaml')
    # cfg = parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen',help='generate operator',required=True)
    parser.add_argument('--target', default='x86-64-linux', help='the compile feature such as:x86-64-linux')
    parser.add_argument('-compute_time',default=False,action="store_true")
    parser.add_argument('-autotune',default=False,action="store_true")
    parser.add_argument('-datatransform', default=False,action="store_true")
    parser.add_argument('--num_tune_loops', type=int,default=2)
    parser.add_argument('--batch_size', type=int,default=2)
    parser.add_argument('--compute_samples',type=int,default=3)
    parser.add_argument('--num_iterators',type=int,default=50)
    cfg = parser.parse_args()
    if not os.path.exists('../build/src'):
        print("ERROR:AutoSearch must be build first. Please execute the following command:")
        print("cd .. && mkdir build && cd build ")
        print("cmake ..")
        print("make -j16")
    else:    
        compile_file(cfg)

    