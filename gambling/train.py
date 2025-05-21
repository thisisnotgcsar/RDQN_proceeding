from datetime import datetime
import json
from torch.utils.tensorboard import SummaryWriter


def pretty_json(hp):
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))

def get_robustq_params_dicts(vars):
    simulator_params = {
        'seed': vars['seed'],
        'env_batch_size': vars['env_batch_size'],
    }

    model_params = {
        'epsilon': vars['epsilon'],
        'delta' : vars['delta'],
        'norm_ord': vars['norm_ord'],
        'lamda_init': vars['lamda_init'],
        'lamda_max_iter': vars['lamda_max_iter'],
        'lamda_step_size': vars['lamda_step_size'],
        'lamda_gamma': vars['lamda_gamma'],
        'lamda_lr': vars['lamda_lr'],
        'n_inner': vars['n_inner'],
        'n_outer': vars['n_outer'],
        'robustq_lr': vars['robustq_lr'],
        'architecture': vars['architecture'],
        'discount': vars['discount'],
        'buffer_max_length': vars['buffer_max_length'],
        'eps_greedy': vars['eps_greedy'],
        'train_steps': vars['train_steps'],
        'clone_steps': vars['clone_steps'],
        'n_batches': vars['n_batches'],
        'n_epochs': vars['n_epochs'],
        'agent_batch_size': vars['agent_batch_size'],
    }

    return simulator_params, model_params

def start_writer(simulator_params, model_params, model_name='RobustQ'):
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    writer = SummaryWriter(f'./runs/{model_name}_{now}')
    model_params['checkpoint_path'] = f'{writer.log_dir}/checkpoint.pt'
    writer.add_text('Simulator parameters', pretty_json(simulator_params))
    writer.add_text('Model parameters', pretty_json(model_params))
    writer.flush()
    return writer