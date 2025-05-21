from typing import Optional, Union
import numpy as np
import pandas_market_calendars as mcal
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

DATATYPE = torch.float32

def load_generator(generator, path, device=torch.device('cpu')):
    weights = torch.load(path + 'generator.pt', map_location=device, weights_only=True)
    del_keys = []
    for key in weights.keys():
        if key.startswith('mean_net') or key.startswith('var_net'):
            del_keys.append(key)
    for key in del_keys:
        del weights[key]
    generator.load_state_dict(weights)
    return generator

class GenLSTM(nn.Module):
    '''
    LSTM-based generator model for generating sequences.

    Args:
        noise_dim (int): Dimension of the noise vector.
        seq_dim (int): Dimension of the time series (e.g., number of stocks).
        seq_len (int): Length of the time series including the historical portion (if any).
        hidden_size (int, optional): Size of the hidden state of the LSTM. Defaults to 64.
        n_lstm_layers (int, optional): Number of LSTM layers. Defaults to 1.
        activation (str, optional): Activation function for the LSTM. Defaults to 'Tanh'.

    Methods:
        _condition_lstm: Condition the LSTM with historical data and noise.
        _generate_sequence: Generate the sequence using the LSTM.
        forward: Forward pass of the generator model.
    '''

    def __init__(self, noise_dim, seq_dim, seq_len, hidden_size=64, n_lstm_layers=1, activation='Tanh'):
        super().__init__()
        self.seq_dim = seq_dim
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers

        activation = getattr(nn, activation)
        self.rnn = nn.LSTM(input_size=seq_dim+noise_dim+1, hidden_size=hidden_size, num_layers=n_lstm_layers, batch_first=True, bidirectional=False)
        self.output_net = nn.Linear(hidden_size, seq_dim)

    def _condition_lstm(self, noise, diff_x, dts):
        batch_size = noise.shape[0] # noise shape: batch_size, seq_len, noise_dim
        h = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size, requires_grad=False, device=noise.device)
        c = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size, requires_grad=False, device=noise.device)

        input = torch.cat([diff_x, noise, dts], dim=-1)
        _, (h, c) = self.rnn(input, (h, c))
        return h, c

    def _expand_outer(self, x, n_outer, cell_state=False):
        x = x.expand(n_outer, -1, -1, -1)
        x = x.permute(1, 0, 2, 3)
        x = x.reshape(x.shape[0], -1, x.shape[3]) if cell_state else x.reshape(-1, x.shape[2], x.shape[3])
        return x

    def _generate_sequence(self, last_return, noise, dts, h, c):
        '''
        Generate the sequence
        Inputs:
        - last_return: the last return of shape (batch_size, 1, seq_dim)
        - noise: the noise of shape (batch_size, seq_len-1, noise_dim) or (batch_size, n_outer, seq_len-1, noise_dim)
        - dts: the time difference of shape (batch_size, seq_len-1, 1)
        - h: the hidden state of the LSTM of shape (n_lstm_layers, batch_size, hidden_size)
        - c: the cell state of the LSTM of shape (n_lstm_layers, batch_size, hidden_size)
        Outputs:
        - generated_seq: the generated sequence of shape (batch_size, seq_len, seq_dim)
        '''

        if noise.ndim == 4:
            batch_size = noise.shape[0]
            n_outer = noise.shape[1]
            dts = self._expand_outer(dts, n_outer) # (batch_size*n_outer, dts.shape[1], 1)
            last_return = self._expand_outer(last_return, n_outer) # (batch_size*n_outer, 1, seq_dim)
            noise = noise.reshape(-1, noise.shape[2], noise.shape[-1]) # (batch_size*n_outer, noise.shape[2], noise_dim)
            h = self._expand_outer(h, n_outer, cell_state=True) # (n_lstm_layers, batch_size*n_outer, hidden_size)
            c = self._expand_outer(c, n_outer, cell_state=True) # (n_lstm_layers, batch_size*n_outer, hidden_size)
            # print(f'last_return.shape: {last_return.shape}, noise.shape: {noise.shape}, dts.shape: {dts.shape}, h.shape: {h.shape}, c.shape: {c.shape}')
        else:
            n_outer = None

        gen_seq = []
        for i in range(noise.shape[1]): # iterate over the remaining time steps
            input = torch.cat([last_return, noise[:,i:i+1,:], dts[:,i:i+1,:]], dim=-1) # (batch_size, 1, seq_dim+noise_dim+1)
            output, (h, c) = self.rnn(input, (h, c))
            last_return = self.output_net(output)
            gen_seq.append(last_return)
        generated_seq = torch.cat(gen_seq, dim=1) if len(gen_seq) > 1 else gen_seq[0]
        if n_outer is None:
            return generated_seq, h, c # h and c are required for the next step
        else:
            return generated_seq.reshape(batch_size, n_outer, -1) # (batch_size, n_outer, seq_dim) for robust Q learning

    def forward(self, noise, dts, h=None, c=None, last_return=None, hist_x=None, hist_noise=None):
        '''
        Generate the sequence either with or without historical data
        Let n be the number of log returns to generate
        If hist_x and hist_noise are provided, the LSTM is conditioned on the historical data first
        hist_x is the historical log price sequence of shape (batch_size, hist_len, seq_dim)
        hist_noise is the historical noise vector of shape (batch_size, hist_len-1, noise_dim)
        Inputs:
        - noise: the noise vector of shape (batch_size, n, noise_dim) or (batch_size, n_outer, n, noise_dim)
        - dts: the time difference vector of shape (batch_size, n, 1)
        - h: the hidden state of the LSTM of shape (n_lstm_layers, batch_size, hidden_size)
        - c: the cell state of the LSTM of shape (n_lstm_layers, batch_size, hidden_size)
        - last_return: the last return of shape (batch_size, 1, seq_dim)
        - hist_x: the historical data of shape (batch_size, hist_len, seq_dim)
        - hist_noise: the historical noise vector of shape (batch_size, hist_len-1, noise_dim)
        Outputs:
        - output_seq: the generated sequence of log returns (NOTE: not log prices) with shape (batch_size, n, seq_dim)
        - h: the hidden state of the LSTM of shape (n_lstm_layers, batch_size, hidden_size)
        - c: the cell state of the LSTM of shape (n_lstm_layers, batch_size, hidden_size)
        '''

        batch_size = noise.shape[0]
        if hist_x is None:
            if h is None and c is None and last_return is None:
                last_return = torch.zeros(noise.shape[0], 1, self.seq_dim, device=noise.device, requires_grad=False)
                h = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size, requires_grad=False, device=noise.device)
                c = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size, requires_grad=False, device=noise.device)
            elif h is None or c is None or last_return is None:
                raise ValueError('Hidden state, cell state and last return must be both provided together')
        else:
            if not (h is None and c is None and last_return is None):
                raise ValueError('Historical data and (hidden state, cell state and last return) cannot be provided together')
            hist_len = hist_x.shape[1]
            diff_x = hist_x.diff(dim=1)
            diff_x = torch.cat([torch.zeros(diff_x.shape[0], 1, self.seq_dim, device=noise.device, requires_grad=False), diff_x], dim=1)
            last_return = diff_x[:,-1:,:] # (batch_size, 1, seq_dim)
            diff_x = diff_x[:,:-1,:] # (batch_size, hist_len-1, seq_dim)
            hist_dts = dts[:,:hist_len-1,:] # (batch_size, hist_len-1, 1)
            dts = dts[:,hist_len-1:,:] # (batch_size, seq_len-hist_len, 1)
            if hist_noise is None:
                hist_noise = noise[:,:hist_len-1,:] # (batch_size, hist_len-1, noise_dim)
                noise = noise[:,hist_len-1:,:] # (batch_size, seq_len-hist_len, noise_dim)
            h, c = self._condition_lstm(hist_noise, diff_x, hist_dts)

        return self._generate_sequence(last_return, noise, dts, h, c)

class MMDSimulator(gym.Env):
    '''
    Simulator for the MMD environment
    Inputs:
    - generator: the neural network generator trained with the signature kernel MMD
    - ma_model_params: the parameters for the moving average model
    - trading_calendar: the trading calendar to use
    - start_date: the start date of the simulation
    - end_date: the end date of the simulation
    - state_len: the length of the state
    - burn_in: the number of periods to burn in
    - int_rate: the interest rate for continuous compounding
    - trans_cost: the transaction cost and a percentage of the transaction amount
    - batch_size: the batch size to use where obs is of shape and rewards (batch_size, obs_size/1)
    - device: the device to use
    - logging: whether to log the episode data for plotting function
    '''

    def __init__(self,
                 generator: nn.Module,
                 ma_model_params: dict,
                 trading_calendar: str,
                 start_date: str,
                 end_date: str,
                 state_len: int,
                 burn_in: int,
                 int_rate: float=0.0,
                 trans_cost: float=5e-4,
                 batch_size: int=32,
                 action_space: gym.Space=None,
                 action_values: torch.tensor=None,
                 device: str='cpu',
                 logging: bool=False):

        assert trans_cost >= 0, 'Transaction cost must be non-negative'

        self.generator = generator
        self.generator.to(device)
        self.noise_dim = generator.noise_dim
        self.seq_dim = generator.seq_dim
        self.ma_model_params = ma_model_params
        self.bias = torch.tensor(ma_model_params['omega'], dtype=DATATYPE)
        self.lags = torch.tensor(ma_model_params.values[1:], dtype=DATATYPE).flip(0).unsqueeze(-1) # flip to match the order of seq
        self.ma_p = len(self.lags)
        self.trading_calendar = trading_calendar
        self.calendar = mcal.get_calendar(self.trading_calendar)
        self.schedule = self.calendar.schedule(start_date=start_date, end_date=end_date)
        self.state_len = state_len
        self.burn_in = burn_in
        self.int_rate = int_rate
        self.trans_cost = trans_cost
        self.batch_size = batch_size
        self.num_envs = batch_size
        self.device = device

        # set up timeline
        t = np.zeros(len(self.schedule))
        t[1:] = (self.schedule.index.to_series().diff()[1:].dt.days / 365).values.cumsum()
        self.t = torch.tensor(t, dtype=DATATYPE, device=self.device, requires_grad=False)
        self.dts = self.t.diff(dim=0)
        self.total_steps = len(self.dts)
        self.action_steps = self.total_steps - (self.burn_in + self.state_len)

        # hd5f database for storing all episodes
        self.logging = logging

        low = np.ones(self.state_len * self.seq_dim + 1 + self.seq_dim) * -1e10
        high = np.ones(self.state_len * self.seq_dim + 1 + self.seq_dim) * 1e10
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # action space
        if action_space is None:
            print('Using default action space of Box(low=-1., high=1., shape=(seq_dim,))')
            self.action_space = spaces.Box(low=np.ones(self.seq_dim) * -1., high=np.ones(self.seq_dim) * 1., dtype=np.float32)
        elif type(action_space) == spaces.Discrete:
            if action_values is None:
                raise ValueError('Action values must be provided for Discrete action space')
            else:
                self.action_space = action_space
                self.action_values = action_values
        elif type(action_space) == spaces.Box:
            self.action_space = action_space
        else:
            raise ValueError(f'Action space must be Discrete or Box, got {type(action_space)}')

        super().__init__()

    def reset(self, seed: Optional[int]=None):
        '''
        Generate the initial state
        Returns:
        - seq: the initial state of shape (batch_size, state_len, seq_dim)
        - info: the info dictionary which is empty
        '''
        # initialise agent state
        self.position = torch.zeros((self.batch_size, self.seq_dim), dtype=DATATYPE, requires_grad=False)
        self.log_wealth = torch.zeros((self.batch_size, 1), dtype=DATATYPE, requires_grad=False)

        # initialise simulation state
        self.curr_step = self.burn_in + self.state_len
        self.ma_noise = self.generate_ma_noise(self.batch_size, self.total_steps)
        noise = self.ma_noise[:,:self.curr_step,:] # (batch_size, burn_in+state_len, noise_dim)
        # NOTE: self.dts[curr_step] is the time difference from current time step to the next time step
        dts = self.dts[:self.curr_step].expand(self.batch_size, -1).unsqueeze(-1).to(self.device) # (batch_size, burn_in+state_len, 1)

        with torch.no_grad():
            seq, self.h, self.c = self.generator(noise, dts)
        self.seq = seq[:,self.burn_in:] # (batch_size, sample_len-burn_in) - only keep the sequence after burn_in

        next_state = self.get_state()

        # initialise episode data for tensorboard logging
        if self.logging:
            self.episode_actions = []
            self.episode_rewards = []
            self.episode_log_returns = [next_state[:,:self.state_len*self.seq_dim]]

        return next_state, {}

    def step(self, action: Union[torch.tensor, np.ndarray]):
        '''
        Generate the state with internal state update
        Inputs:
        - action: the action of shape (batch_size, seq_dim)
                    signifies the portfolio weight if the action space is Box
                    signifies the action index if the action space is Discrete
        Returns:
        - next_state: the next state of shape (batch_size, state_len, seq_dim)
        - reward: the reward of shape (batch_size, 1)
        - done: whether the episode is done
        - truncated: whether the episode is truncated
        - info: the info dictionary
        '''

        action = self.check_action(action)

        noise = self.ma_noise[:,self.curr_step:self.curr_step+1,:] # (batch_size, 1, noise_dim)
        dts = self.dts[self.curr_step:self.curr_step+1].expand(self.batch_size, -1).unsqueeze(-1).to(self.device) # (batch_size, 1, 1)
        last_return = self.seq[:,-1:,:]

        # generate the next state
        with torch.no_grad():
            generated_seq, self.h, self.c = self.generator(noise, dts, h=self.h, c=self.c, last_return=last_return)
        self.seq[:,:-1,:] = self.seq[:,1:,:].clone()
        self.seq[:,-1:,:] = generated_seq

        log_return = self.seq[:,-1:,:]

        reward, interest, transation_cost = self.get_reward(action, log_return) # NOTE: self.positions are updated
        self.log_wealth = self.log_wealth + reward
        info = (interest, transation_cost) # NOTE: for agent to calc reward when sampling from nu

        next_state = self.get_state()

        if next_state.isnan().any():
            print(f'next_state: {next_state}')
            raise ValueError('Next state contains NaN values')

        self.curr_step += 1 # NOTE: increment after next state and reward are generated
        done = (self.curr_step == self.total_steps)
        done = torch.tensor(done, dtype=torch.bool, device=self.device).repeat(self.batch_size, 1)

        if self.logging:
            self.episode_actions.append(action)
            self.episode_rewards.append(reward)
            self.episode_log_returns.append(next_state[:,(self.state_len-1)*self.seq_dim:self.state_len*self.seq_dim])

        return next_state, reward, done, False, info

    def check_action(self, action):
        '''
        Check the action and convert it to the correct format
        If the action space is Discrete, convert the action index to the action values
        '''
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=DATATYPE, device=self.device, requires_grad=False)
        elif not isinstance(action, torch.Tensor):
            action = torch.tensor([action], dtype=DATATYPE, device=self.device, requires_grad=False)
        if action.ndim == 1:
            if action.shape[0] == self.seq_dim:
                action = action.unsqueeze(0)
            elif action.shape[0] == self.batch_size:
                action = action.unsqueeze(-1)

        if type(self.action_space) == spaces.Discrete:
            action = action.to(torch.int32)
            action = self.action_values[action]

        return action

    def get_state(self):
        dt = self.dts[self.curr_step-1].repeat(self.batch_size, 1) # (batch_size, 1)
        next_state = torch.cat([self.seq.reshape(self.batch_size, -1), self.log_wealth, self.position, dt], dim=1)

        if self.batch_size == 1:
            next_state = next_state.reshape(-1)

        return next_state

    def get_reward(self, action: torch.tensor, log_return: torch.tensor):
        '''
        Gets the reward for the current state and action based on the log_return
        Action is the new position weights at the time step before the new state is generated
        '''
        non_cash_weight_deltas = action - self.position # (batch_size, seq_dim)
        cash_weight = 1. - action.sum(dim=1, keepdim=True) # (batch_size, 1)
        interest_return = torch.exp(self.int_rate * self.dts[self.curr_step]) - 1. # scalar
        non_cash_return = log_return[:,0,:].exp() - 1. # (batch_size, seq_dim)
        self.position = action # update position if not simulation
        transation_cost_return = self.trans_cost * torch.abs(non_cash_weight_deltas.sum(dim=-1, keepdim=True))

        portfolio_simple_return = 1 + cash_weight * interest_return + (action * non_cash_return).sum(dim=-1, keepdims=True) - transation_cost_return
        reward = torch.log(portfolio_simple_return)
        if self.batch_size == 1:
            reward = reward.squeeze()

        return reward, interest_return.repeat(self.batch_size), transation_cost_return.squeeze()

    def generate_ma_noise(self, batch_size:int, length:int):
        '''
        Generate moving average (MA) noise for the generator with a random standard normal seeding sequence
        Inputs:
        - batch_size: the batch size of seeding sequence if seq not provided else n_outer for each sample in seq.shape[0]
        - length: the length of the sequence
        Returns:
        - noise: the MA noise of shape (batch_size, length, noise_dim)
        '''
        seq = torch.randn(batch_size, self.noise_dim, self.ma_p, dtype=DATATYPE)
        noise = []
        for _ in range(length):
            sigma = (seq**2 @ self.lags.expand(batch_size, -1, 1) + self.bias).sqrt()
            noise.append(sigma * torch.randn_like(sigma)) # (batch_size, noise_dim, 1)
            seq = seq.roll(-1, dims=2)
            seq[:,:,-1:] = noise[-1]
        noise = torch.cat(noise, dim=2) # (batch_size, noise_dim, length)
        noise = noise.permute(0, 2, 1) # (batch_size, length, noise_dim)
        return noise

    def generate_ma_noise_outer(self, seq:torch.Tensor, n_outer:int):
        seq = seq.permute(0, 2, 1) # (batch_size, noise_dim, ma_p)
        sigma = (seq**2 @ self.lags.expand(seq.shape[0], -1, 1) + self.bias).sqrt() # (batch_size, noise_dim, 1)
        noise = sigma * torch.randn((sigma.shape[0], sigma.shape[1], n_outer)) # (batch_size, noise_dim, n_outer)
        noise = noise.permute(0, 2, 1) # (batch_size, n_outer, noise_dim)
        noise = noise.unsqueeze(2) # (batch_size, n_outer, 1, noise_dim)
        return noise

    def print_metrics(self):
        env_log_returns = torch.cat(self.episode_log_returns, dim=1)
        env_final_log_wealth = env_log_returns.sum(dim=1)
        env_final_wealths = env_final_log_wealth.exp()
        env_mean_final_log_wealth = env_final_log_wealth.mean(dim=0).item()
        env_mean_final_wealth = env_final_wealths.mean(dim=0).item()

        env_log_return_mean = env_log_returns.mean(dim=1)
        env_return_vol = env_log_returns.std(dim=1) * np.sqrt(252) # annualize
        env_mean_return_vol = env_return_vol.mean().item()
        env_sharpes = env_log_return_mean / env_return_vol * 252 # annualize
        env_mean_sharpe = env_sharpes.mean(dim=0).item()
        env_downside_dev = torch.minimum(env_log_returns, torch.zeros_like(env_log_returns)).std(dim=1) * np.sqrt(252) # annualize
        env_mean_downside_dev = env_downside_dev.mean().item()
        env_sortino = env_log_return_mean / env_downside_dev * 252 # annualize
        env_mean_sortino = env_sortino.mean().item()

        rewards = torch.cat(self.episode_rewards, dim=1)
        eval_final_log_wealth = rewards.sum(dim=1)
        eval_final_wealths = eval_final_log_wealth.exp()
        eval_mean_final_log_wealth = eval_final_log_wealth.mean(dim=0).item()
        eval_mean_final_wealth = eval_final_wealths.mean(dim=0).item()

        rewards_mean = rewards.mean(dim=1)
        eval_return_vol = rewards.std(dim=1) * np.sqrt(252) # annualize
        eval_mean_return_vol = eval_return_vol.mean().item()
        eval_sharpes = rewards_mean / eval_return_vol * 252 # annualize
        eval_mean_sharpe = eval_sharpes.mean(dim=0).item()
        eval_downside_dev = torch.minimum(rewards, torch.zeros_like(rewards)).std(dim=1) * np.sqrt(252)
        eval_mean_downside_dev = eval_downside_dev.mean().item()
        eval_sortino = rewards_mean / eval_downside_dev * 252 # annualize
        eval_mean_sortino = eval_sortino.mean().item()

        print(f'Mean final log wealth eval vs env: {eval_mean_final_log_wealth} vs {env_mean_final_log_wealth}')
        print(f'Mean final wealth eval vs env: {eval_mean_final_wealth} vs {env_mean_final_wealth}')
        print(f'Mean return vol eval vs env: {eval_mean_return_vol} vs {env_mean_return_vol}')
        print(f'Mean sharpes eval vs env: {eval_mean_sharpe} vs {env_mean_sharpe}')
        print(f'Mean downside dev eval vs env: {eval_mean_downside_dev} vs {env_mean_downside_dev}')
        print(f'Mean sortino eval vs env: {eval_mean_sortino} vs {env_mean_sortino}')

    def seed(self, seed=None):
        print('seed')
        pass

    def close(self):
        print('close')

    def env_is_wrapped(self, wrapper_class):
        print('env_is_wrapped')
        return False

    def get_attr(self, attr_name, indices=None):
        attr = getattr(self, attr_name)
        if attr is None:
            raise AttributeError(f'{attr_name} is not defined in the environment')
        else:
            return attr

    def set_attr(self, attr_name, value, indices=None):
        print(f'set_attr: {attr_name} = {value}')
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, **kwargs):
        print('env_method')
        return getattr(self, method_name)(*args, **kwargs)