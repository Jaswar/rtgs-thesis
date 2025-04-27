import yaml
import os

def main(configs_path, datasets):
    combined_params = {}
    for dataset in datasets:
        for config_file in os.listdir(os.path.join(configs_path, dataset)):
            if not config_file.endswith('.yaml'):
                continue
            with open(os.path.join(configs_path, dataset, config_file), 'r') as f:
                config = yaml.safe_load(f)
            for param in config:
                if 'Params' not in param:
                    continue
                if param not in combined_params:
                    combined_params[param] = {}
                for sub_param in config[param]:
                    if sub_param not in combined_params[param]:
                        combined_params[param][sub_param] = set()
                    combined_params[param][sub_param].add(config[param][sub_param])
    for param in combined_params:
        print(f'{param}:')
        for sub_param in combined_params[param]:
            if sub_param not in ['source_path', 'model_path'] and len(combined_params[param][sub_param]) > 1:
                print(f'\t{sub_param}: {list(combined_params[param][sub_param])}')

if __name__ == '__main__':
    configs_path = './configs'
    datasets = ['dnerf', 'dynerf']
    main(configs_path, datasets)