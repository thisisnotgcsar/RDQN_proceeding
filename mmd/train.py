import os
import random
from datetime import datetime
import json
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def pretty_json(hp):
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))

def get_params_dicts(vars):
    data_params = {
        'batch_size': vars['batch_size'],
        'sample_len': vars['sample_len'],
        'seed': vars['seed'],
        'stride': vars['stride'],
        'start_date': vars['start_date'],
        'end_date': vars['end_date'],
        'lead_lag': vars['lead_lag'],
        'lags': vars['lags'],
    }

    model_params = {
        'static_kernel_type': vars['static_kernel_type'], # linear, rbf, rq
        'n_levels': vars['n_levels'], # truncated signature kernel levels
        'hidden_size': vars['hidden_size'],
        'activation': vars['activation'], # pytorch activation function name
        'n_lstm_layers': vars['n_lstm_layers'],
        'noise_dim': vars['noise_dim'],
        'seq_dim': vars['seq_dim'],
        'conditional': vars['conditional'],
        'ma': vars['ma'] if 'ma' in vars else False,
        'ma_p': vars['ma_p'] if 'ma_p' in vars else vars['garch_p']
    }

    if model_params['conditional'] == True:
        model_params['hist_len'] = vars['hist_len']

    train_params = {
        'epochs': vars['epochs'],
        'start_lr': vars['start_lr'],
        'lr_factor': vars['lr_factor'],
        'patience': vars['patience'],
        'early_stopping': vars['early_stopping'],
        'kernel_sigma': vars['kernel_sigma'],
        'num_losses': vars['num_losses'],
    }

    return data_params, model_params, train_params

def get_robustq_params_dicts(vars):
    simulator_params = {
        'seed': vars['seed'],
        'batch_size': vars['batch_size'],
        'cal_start_date': vars['cal_start_date'],
        'cal_end_date': vars['cal_end_date'],
        'state_len': vars['state_len'],
        'other_state_vars': vars['other_state_vars'],
        'burn_in': vars['burn_in'],
        'int_rate': vars['int_rate'] if 'int_rate' in vars else 0.0,
        'trans_cost': vars['trans_cost'] if 'trans_cost' in vars else 5e-4,
        'action_values': list(vars['action_values'].tolist()),
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
        'obs_dim': vars['obs_dim'],
        'discount': vars['discount'],
        'num_actions': vars['num_actions'],
        'buffer_max_length': vars['buffer_max_length'],
        'eps_greedy': vars['eps_greedy'],
        'train_steps': vars['train_steps'],
        'clone_steps': vars['clone_steps'],
        'n_batches': vars['n_batches'],
        'n_epochs': vars['n_epochs'],
        'agent_batch_size': vars['agent_batch_size'],
        'n_episodes': vars['n_episodes'],
        'eval_batch_size': vars['eval_batch_size'],
        'eval_seed': vars['eval_seed'],
    }
    if 'nu_dist' in vars:
        model_params['nu_dist'] = vars['nu_dist']
        model_params['nu_scale'] = vars['nu_scale']
        if vars['nu_dist'] == 't': simulator_params['nu_df'] = vars['nu_df']

    return simulator_params, model_params

def start_writer(simulator_params, model_params, model_name='RobustQ'):
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    writer = SummaryWriter(f'./runs/{model_name}_{now}')
    model_params['checkpoint_path'] = f'{writer.log_dir}/checkpoint.pt'
    writer.add_text('Simulator parameters', pretty_json(simulator_params))
    writer.add_text('Model parameters', pretty_json(model_params))
    writer.flush()
    return writer

def get_hparams(path):
    hparam_type = {}
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    tags = event_acc.Tags()["tensors"]
    for tag in tags:
        name = tag.split('/')[0]
        event_list = event_acc.Tensors(tag)
        param_str = str(event_list[0].tensor_proto.string_val[0])
        param_str = param_str.replace('\\n', '')
        param_str = param_str.replace('\\t', '')
        param_str = param_str.replace('\'', '')
        param_str = param_str.replace('\\', '')
        param_str = param_str.replace('b{', '{')
        if param_str.startswith('{'):
            params = json.loads(param_str)
            hparam_type[name] = params
    return hparam_type

def get_params_from_events(path):
    if not path.endswith('/'):
        path += '/'
    events_files = []
    for file in os.listdir(path):
        if file.startswith('events'):
            events_files.append(file)
    events_files.sort()
    events_name = events_files[0]
    params = get_hparams(path + events_name)

    return params

def train_robustdqn(agent, env, writer, simulator_params, model_params, checkpoint_path=None):
    n_episodes = model_params['n_episodes']
    batch_size = simulator_params['batch_size']
    eps_greedy_schedule = model_params['eps_greedy_schedule'] if 'eps_greedy_schedule' in model_params else {}
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        agent.steps = checkpoint['agent_steps']
        agent.q_updates = checkpoint['agent_q_updates']
        agent.q.load_state_dict(checkpoint['agent_state_dict'])
        agent.target_q.load_state_dict(checkpoint['agent_target_q'])
        agent.optimizer.load_state_dict(checkpoint['agent_optimizer'])
        agent.buffer = checkpoint['agent_buffer']
        agent.rng.bit_generator.state = checkpoint['numpy_rng_state']
        agent.epsilon = checkpoint['agent_eps_greedy']
        random.setstate(checkpoint['random_state'])
        torch.set_rng_state(checkpoint['torch_rng_state'])
        start_episode = checkpoint['episode'] + 1
    else:
        start_episode = 0

    for episode in tqdm(range(start_episode, n_episodes)):
        if str(episode) in eps_greedy_schedule:
            agent.epsilon = eps_greedy_schedule[str(episode)]
        cum_rewards = torch.zeros(batch_size, 1)
        obs, _ = env.reset()
        act_idx = agent.agent_start(obs)
        done = torch.tensor([False] * batch_size)
        while not done.any():
            obs, rewards, done, truncated, info = env.step(act_idx) # step will convert action index to action value
            # NOTE: info contains dt used to calc cash interest in rewards and is of shape (batch_size)
            cum_rewards += rewards
            if done.any():
                agent.agent_end(rewards, obs, info)
            else:
                act_idx = agent.agent_step(rewards, obs, info)

        if writer is not None:
            writer.add_scalar('mean_cum_rewards', cum_rewards.mean(), episode+1)
            torch.save({'episode': episode,
                'agent_steps': agent.steps,
                'agent_q_updates': agent.q_updates,
                'agent_state_dict': agent.q.state_dict(),
                'agent_target_q': agent.target_q.state_dict(),
                'agent_optimizer': agent.optimizer.state_dict(),
                'agent_buffer': agent.buffer,
                'agent_eps_greedy': agent.epsilon,
                'random_state': random.getstate(),
                'numpy_rng_state': agent.rng.bit_generator.state,
                'torch_rng_state': torch.get_rng_state(),
                }, f'./{writer.log_dir}/checkpoint.pt')
        print(f'Episode {episode+1} mean of summed rewards: {cum_rewards.mean():.3f}')
    return agent