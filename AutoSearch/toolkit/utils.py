
from tools import run_cmd
import os
import time
HALIDE_HOME = os.environ['HALIDE_HOME']
CURRENT_DIR = os.getcwd()
def create_generator(gen,demo_name):
    global CURRENT_DIR,HALIDE_HOME
    run_cmd('g++ %s %s/template/gen.cpp -g -I %s/include -I %s/../src/ -L %s/bin -lHalide -ldl -lpthread -std=c++11 -fno-rtti -o %s_gen' 
    % (gen, CURRENT_DIR, HALIDE_HOME, CURRENT_DIR, HALIDE_HOME,demo_name))
    run_cmd('if [ ! -d samples  ];then mkdir samples; fi')
    run_cmd('mv %s_gen ./samples/%s_gen' % (demo_name,demo_name))
def create_schedule(weight_dir,excute,target,demo_name,output_dir,fname,seed):
    run_cmd('mkdir -p {}'.format(output_dir))
    run_cmd('''HL_SEED={} HL_WEIGHTS_DIR={} HL_RANDOM_DROPOUT=100 HL_BEAM_SIZE=32 \
            HL_MACHINE_PARAMS=32,24000000,40 timeout -k 600s 600s {} -g {} -f {} -o {} \
            -e stmt,assembly,static_library,c_header,registration,schedule,featurization\
             target={}-disable_llvm_loop_opt \
              auto_schedule=true -p ../build/src/adams2019/libautoschedule_adams2019.so \
              -s Adams2019'''.format(seed,weight_dir,excute,demo_name,fname,output_dir,target),True)
def read_bench_float(dir):
    import re
    fr = open(dir,'r')
    line = fr.readline()
    temp=re.findall(r'-?\d+\.?\d*e?-?\d*?', line)
    return temp[0]
def benchmark(schedule_dir,demo_name,seed,fname):
    global HALIDE_HOME,CURRENT_DIR
    run_cmd('''c++ -std=c++11 -I {}/halide-build/include -I {}/tools RunGenMain.cpp \
    {}/*.registration.cpp {}/*.a -o {}/bench -DHALIDE_NO_PNG \
    -DHALIDE_NO_JPEG -ldl -lpthread'''.format(HALIDE_HOME,HALIDE_HOME,schedule_dir,schedule_dir,schedule_dir),False)
    time.sleep(1)
    run_cmd('''HL_NUM_THREADS=32 timeout -k 60s 60s {}/bench --estimate_all --benchmarks=all | tee {}/bench.txt'''.format(schedule_dir,schedule_dir))
# ../build/src/adams2019//featurization_to_sample ./temp/batch_2_0/0/blur_batch_0002_sample_0000.featurization 0.000108463 0 00020000 ./temp/batch_2_0/0/blur_batch_0002_sample_0000.sample
    cost = read_bench_float('{}/bench.txt'.format(schedule_dir))
    cost = eval(cost)/1000.0
    run_cmd('''../build/src/adams2019/featurization_to_sample {}/{}.featurization {} 0 {} {}/{}.sample'''.format(schedule_dir,fname,cost,seed,schedule_dir,fname))
def retrain_cost_model(weight_dir,weight_out,demo_name,epochs=1):
    run_cmd('''find samples/ -name '*.sample' | ../build/src/adams2019/retrain_cost_model --epochs={} \
            --rates="0.0001" --num_cores=32 --initial_weights={} --weights_out={} \
            --best_benchmark= samples/best.{}.benchmark.txt \
            --best_schedule = samples/best.{}.schedule.h'''.format(epochs,weight_dir,weight_out,demo_name,demo_name),True)
    
if __name__=='__main__':
    run_cmd('rm -rf ./samples')
    create_generator('../generator/batch_matmul.cpp','batch_matmul')

    os.environ['HL_APP_ARGS']='1, 512, 512, 512'

    weight_dir = 'samples/updated.weights'
    run_cmd('cp ../src/adams2019/baseline.weights {}'.format(weight_dir))

    for iters in range(2):
        DIR = 'samples/batch_{}'.format(iters)
        run_cmd('mkdir -p {}'.format(DIR))
        for batch in range(2):
            output_dir = DIR+'/{}'.format(batch)
            seed=iters*10000+batch
            fname = 'matmul_batch_%04d_sample_%04d' % (iters,batch)
            create_schedule(weight_dir,'./samples/batch_matmul_gen','x86-64-linux-avx-avx2-f16c-fma-sse41','matmul',output_dir,fname,seed)

        for batch in range(2):
            output_dir = DIR+'/{}'.format(batch)
            seed=iters*10000+batch
            fname = 'matmul_batch_%04d_sample_%04d' % (iters,batch)
            benchmark(output_dir,'matmul',seed,fname)
        print("begin retrain")
        retrain_cost_model(weight_dir,"new.weights",'matmul',epochs=2)          
        print("end retrain")