from subprocess import Popen

param_name = '--c'
config_files = ['configs/dt_panda_pick_and_place_dense_tf.yaml', 'configs/dt_panda_pick_and_place_sparse_tf.yaml', 'configs/dt_panda_push_dense_tf.yaml', 'configs/dt_panda_push_sparse_tf.yaml', 'configs/dt_panda_reach_sparse_tf.yaml']


for file in config_files:
    Process = Popen('sbatch run_rl.sh %s %s' % (str(param_name), str(file),), shell=True)