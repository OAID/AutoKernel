from CostModel import costmodel
from tools import run_cmd
import os
import time
HALIDE_HOME = os.environ['HALIDE_HOME']
CURRENT_DIR = os.getcwd()
def create_generator(gen,demo_name,samples_dir):
    global CURRENT_DIR,HALIDE_HOME
    run_cmd('g++ %s %s/template/gen.cpp -g -I %s/include -I %s/../src/ -L %s/bin -lHalide -ldl -lpthread -std=c++11 -fno-rtti -o %s_gen' 
    % (gen, CURRENT_DIR, HALIDE_HOME, CURRENT_DIR, HALIDE_HOME,demo_name))
    run_cmd('if [ ! -d {}  ];then mkdir {}; fi'.format(samples_dir,samples_dir))
    run_cmd('mv %s_gen %s/%s_gen' % (demo_name,samples_dir,demo_name))
def create_schedule(batch,weight_dir,excute,target,demo_name,output_dir,fname,hl_seed,seed):
    run_cmd('mkdir -p {}'.format(output_dir))
    if batch == 0:
        run_cmd('''HL_SEED={} HL_WEIGHTS_DIR={} HL_RANDOM_DROPOUT=100 HL_BEAM_SIZE=20 \
                HL_MACHINE_PARAMS=32,24000000,40 timeout -k 600s 600s {} -g {} -f {} -o {} \
                -e stmt,assembly,static_library,c_header,registration,schedule,featurization\
                target={}-disable_llvm_loop_opt \
                auto_schedule=true max_stages=6 seed={} -p ../build/src/adams2019/libautoschedule_adams2019.so \
                -s Adams2019 2> {}/compile_log_stderr.txt > {}/compile_log_stdout.txt'''.format(hl_seed,weight_dir,excute,demo_name,fname,output_dir,target,seed,output_dir,output_dir),True)
    else:
        # run_cmd('''HL_SEED={} HL_WEIGHTS_DIR={} HL_RANDOM_DROPOUT=80 HL_BEAM_SIZE=1 \
        run_cmd('''HL_SEED={} HL_WEIGHTS_DIR={} HL_RANDOM_DROPOUT=10 HL_BEAM_SIZE=5 \
                HL_MACHINE_PARAMS=32,24000000,40 timeout -k 600s 600s {} -g {} -f {} -o {} \
                -e stmt,assembly,static_library,c_header,registration,schedule,featurization\
                target={}-disable_llvm_loop_opt \
                auto_schedule=true max_stages=6 seed={} -p ../build/src/adams2019/libautoschedule_adams2019.so \
                -s Adams2019 2> {}/compile_log_stderr.txt > {}/compile_log_stdout.txt'''.format(hl_seed,weight_dir,excute,demo_name,fname,output_dir,target,seed,output_dir,output_dir),True)

def create_schedule_arm(batch,weight_dir,excute,target,demo_name,output_dir,fname,hl_seed,seed):
    run_cmd('mkdir -p {}'.format(output_dir))
    run_cmd('''{} -r runtime target={} -o {}'''.format(excute,target,output_dir))
    run_cmd('''HL_SEED={} HL_WEIGHTS_DIR={} HL_RANDOM_DROPOUT=100 HL_BEAM_SIZE=20 \
            HL_MACHINE_PARAMS=32,24000000,40 timeout -k 600s 600s {} -g {} -f {} -o {} \
            target={}-no_runtime \
            auto_schedule=true max_stages=6 seed={} -p ../build/src/adams2019/libautoschedule_adams2019.so \
            -s Adams2019 2> {}/compile_log_stderr.txt > {}/compile_log_stdout.txt'''.format(hl_seed,weight_dir,excute,demo_name,fname,output_dir,target,seed,output_dir,output_dir),True)

def read_bench_float(dir):
    import re
    fr = open(dir,'r')
    line = fr.readline()
    temp=re.findall(r'-?\d+\.?\d*e?-?\d*?', line)
    return temp[0]
def benchmark(schedule_dir,demo_name,hl_seed,fname):
    global HALIDE_HOME,CURRENT_DIR
    run_cmd('''c++ -std=c++11 -I {}/halide-build/include -I {}/tools RunGenMain.cpp \
    {}/*.registration.cpp {}/*.a -o {}/bench -DHALIDE_NO_PNG \
    -DHALIDE_NO_JPEG -ldl -lpthread'''.format(HALIDE_HOME,HALIDE_HOME,schedule_dir,schedule_dir,schedule_dir),False)
    time.sleep(1)
    run_cmd('''HL_NUM_THREADS=32 timeout -k 60s 60s {}/bench --estimate_all --benchmarks=all | tee {}/bench.txt'''.format(schedule_dir,schedule_dir))
# ../build/src/adams2019//featurization_to_sample ./temp/batch_2_0/0/blur_batch_0002_sample_0000.featurization 0.000108463 0 00020000 ./temp/batch_2_0/0/blur_batch_0002_sample_0000.sample
    cost = read_bench_float('{}/bench.txt'.format(schedule_dir))
    cost = eval(cost)/1000.0
    run_cmd('''../build/src/adams2019/featurization_to_sample {}/{}.featurization {} 0 {} {}/{}.sample'''.format(schedule_dir,fname,cost,hl_seed,schedule_dir,fname))
# def retrain_cost_model(weight_dir,weight_out,demo_name,epochs=1):
#     run_cmd('''find samples/ -name '*.sample' | ../build/src/adams2019/retrain_cost_model --epochs={} \
#             --rates="0.0001" --num_cores=32 --initial_weights={} --weights_out={} \
#             --best_benchmark= samples/best.{}.benchmark.txt \
#             --best_schedule = samples/best.{}.schedule.h'''.format(epochs,weight_dir,weight_out,demo_name,demo_name),True)
# def retrain_cost_model():
#     run_cmd(''' cd ./cost_model && python3 cost_model.py''')

def samples_generator(gen_path, 
                      demo_name, 
                      batch_num, 
                      batch_size = 16,
                      samples_dir='./default_samples', 
                      weight_path='./random_init.weights'):
    create_generator(gen_path, demo_name, samples_dir)
    generator = '{}/{}_gen'.format(samples_dir,demo_name)

    weight_dir = samples_dir+'/updated.weights'
    package_dir = os.path.split(os.path.realpath(__file__))[0]
    run_cmd('cp {}/baseline.weights {}'.format(package_dir,weight_dir))

    # find the starting index
    start_idx = -1
    dirs = os.listdir(samples_dir)
    for dir_name in dirs:
        if dir_name.startswith('batch_', 0, 6):
            idx = int(dir_name[6::]) 
            if idx > start_idx:
                start_idx = idx
    
    for iters in range(start_idx+1,start_idx+1+batch_num):
        try: # 使用try和expect可以在编译或搜索超时时跳到下一个batch
            run_cmd('cp {} {}'.format(weight_path, weight_dir))
            DIR = samples_dir+'/batch_{}'.format(iters)
            run_cmd('mkdir -p {}'.format(DIR))
            for batch in range(batch_size):
                print('Compiling batch_' + str(iters) + '_sample_' + str(batch))
                output_dir = DIR+'/{}'.format(batch)
                hl_seed=iters*10000+batch
                fname = '{}_{}_sample_{}'.format(demo_name,iters,batch)
                create_schedule(batch,weight_dir, generator,'x86-64-linux-avx-avx2-f16c-fma-sse41',demo_name,output_dir,fname,hl_seed,iters)
                # create_schedule(batch,weight_dir,'./samples/random_pipeline_gen','arm-64-linux','random_pipeline',output_dir,fname,hl_seed,iters)

            for batch in range(batch_size):
                print('Benchmarking batch_' + str(iters) + '_sample_' + str(batch))
                output_dir = DIR+'/{}'.format(batch)
                hl_seed=iters*10000+batch
                fname = '{}_{}_sample_{}'.format(demo_name,iters,batch)
                benchmark(output_dir,demo_name,hl_seed,fname)
        except:
            continue

def samples_train(samples_dir, 
                  weight_path='./random_init.weights', 
                  start_idx=0,
                  end_idx=-1,
                  learning_rate=0.001, 
                  train_iters=200):

    if end_idx == -1:
        # find the ending index
        dirs = os.listdir(samples_dir)
        for dir_name in dirs:
            if dir_name.startswith('batch_', 0, 6):
                idx = int(dir_name[6::]) 
                if idx > end_idx:
                    end_idx = idx
    else:
        assert end_idx>=start_idx
    
    for iters in range(start_idx,end_idx+1):
        DIR = samples_dir+'/batch_{}'.format(iters)
        print("begin retrain batch_{}".format(iters))
        costmodel.train_cost_model(DIR,weight_path,learning_rate = learning_rate, train_iters=train_iters)
        print("end retrain")
    
def retrain_cost_model(gen_path, 
                       demo_name, 
                       batch_num, 
                       batch_size = 16,
                       samples_dir='./default_samples', 
                       weight_path='./random_init.weights', 
                       learning_rate=0.001, 
                       train_iters=200):
    create_generator(gen_path, demo_name, samples_dir)
    generator = '{}/{}_gen'.format(samples_dir,demo_name)

    weight_dir = samples_dir+'/updated.weights'
    package_dir = os.path.split(os.path.realpath(__file__))[0]
    run_cmd('cp {}/baseline.weights {}'.format(package_dir,weight_dir))

    # find the starting index
    start_idx = -1
    dirs = os.listdir(samples_dir)
    for dir_name in dirs:
        if dir_name.startswith('batch_', 0, 6):
            idx = int(dir_name[6::]) 
            if idx > start_idx:
                start_idx = idx
    
    for iters in range(start_idx+1,start_idx+1+batch_num):
        try: # 使用try和expect可以在编译或搜索超时时跳到下一个batch
            run_cmd('cp {} {}'.format(weight_path, weight_dir))
            DIR = samples_dir+'/batch_{}'.format(iters)
            run_cmd('mkdir -p {}'.format(DIR))
            for batch in range(batch_size):
                print('Compiling batch_' + str(iters) + '_sample_' + str(batch))
                output_dir = DIR+'/{}'.format(batch)
                hl_seed=iters*10000+batch
                fname = '{}_{}_sample_{}'.format(demo_name,iters,batch)
                create_schedule(batch,weight_dir, generator,'x86-64-linux-avx-avx2-f16c-fma-sse41',demo_name,output_dir,fname,hl_seed,iters)
                # create_schedule(batch,weight_dir,'./samples/random_pipeline_gen','arm-64-linux','random_pipeline',output_dir,fname,hl_seed,iters)

            for batch in range(batch_size):
                print('Benchmarking batch_' + str(iters) + '_sample_' + str(batch))
                output_dir = DIR+'/{}'.format(batch)
                hl_seed=iters*10000+batch
                fname = '{}_{}_sample_{}'.format(demo_name,iters,batch)
                benchmark(output_dir,demo_name,hl_seed,fname)
        except:
            continue

        try:
            print("begin retrain")
            costmodel.train_cost_model(DIR,weight_path,learning_rate = learning_rate, train_iters=train_iters)
            print("end retrain")
        except:
            continue