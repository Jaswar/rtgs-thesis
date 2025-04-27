import os
import random
import time
import argparse


TEMPLATE_CONFIG = '''
gaussian_dim: 4
time_duration: [0.0, 1.0]
num_pts: 10_000
num_pts_ratio: 1.0
rot_4d: True
force_sh_3d: False
batch_size: 2
exhaust_test: True

ModelParams:
  sh_degree: 3
  source_path: <source_path>
  model_path: <model_path>
  images: "images"
  resolution: 1
  white_background: False
  data_device: "cuda"
  eval: False
  extension: ".png"
  num_extra_pts: 0
  loaded_pth: ""
  frame_ratio: 1
  dataloader: False

PipelineParams:
  convert_SHs_python: False
  compute_cov3D_python: False
  debug: False
  env_map_res: <env_map_res>
  env_optimize_until: <env_optimize_until>
  env_optimize_from: 0
  eval_shfs_4d: True

OptimizationParams:
  iterations: <iterations>
  position_lr_init: 0.00016
  position_t_lr_init: -1.0
  position_lr_final: 0.0000016
  position_lr_delay_mult: 0.01
  position_lr_max_steps: <position_lr_max_steps>
  feature_lr: 0.0025
  opacity_lr: 0.05
  scaling_lr: 0.005
  rotation_lr: 0.001
  percent_dense: 0.01
  lambda_dssim: 0.2
  thresh_opa_prune: 0.005
  densification_interval: <densification_interval>
  opacity_reset_interval: 30000000  # no reset
  densify_from_iter: 500
  densify_until_iter: <densify_until_iter>
  densify_grad_threshold: 0.0002
  densify_grad_t_threshold: 0.0002 / 40 # 想办法用上
  densify_until_num_points: -1
  final_prune_from_iter: -1
  sh_increase_interval: 1000
  lambda_opa_mask: 0.0
  lambda_rigid: <lambda_rigid>
  lambda_motion: 0.0
'''

SAMPLING_SETTINGS = {
    'env_map_res': [0, 500],
    'env_optimize_until': [1000000000, 5000],
    'iterations': [20000, 30000],
    'position_lr_max_steps': [15000, 30000],
    'densification_interval': [200, 100],
    'densify_until_iter': [10000, 15000],
    'lambda_rigid': [0.0, 1.0]
}


def sample_config():
    config = {}
    for key, values in SAMPLING_SETTINGS.items():
        config[key] = random.choice(values)
    return config


def write_config(config, path, source_path, model_path):
    new_config = TEMPLATE_CONFIG
    for key, value in config.items():
        new_config = new_config.replace(f'<{key}>', str(value))
    new_config = new_config.replace('<source_path>', source_path)
    new_config = new_config.replace('<model_path>', model_path)
    with open(path, 'w') as f:
        f.write(new_config)


def execute_in_env(command, env):
    return os.system(f'/bin/bash -c \"source ~/anaconda3/etc/profile.d/conda.sh && conda activate {env} && {command} \"')


def run_experiment(config_path):
    ret_val = execute_in_env(f'python train.py --config {config_path}', '4d_gaussian_splatting')
    if ret_val != 0:
        raise ValueError('Execution failed')


def main(data_path, model_path, timeout=5 * 60 * 60):
    configs_path = './configs/ego_exo/random_configs'
    os.makedirs(configs_path, exist_ok=True)
    start_time = time.time()
    config_index = 0
    while time.time() - start_time < timeout:
        config = sample_config()
        config_path = os.path.join(configs_path, f'config_{config_index}.yaml')
        model_path_ = os.path.join(model_path, f'model_{config_index}')
        write_config(config, config_path, data_path, model_path_)
        run_experiment(config_path)
        config_index += 1
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()
    data_path = args.data_path
    model_path = args.model_path
    main(data_path, model_path)