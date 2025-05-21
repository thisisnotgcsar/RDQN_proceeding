from typing import Tuple, Optional
import numpy as np
import torch
import gym
from gym.spaces import Discrete, Box

DATATYPE = torch.float32

def softplus(x:np.ndarray) -> np.ndarray:
        idx = x < 50
        x[idx] = np.log(1 + np.exp(x[idx]))
        return x

class Env(gym.Env):
    '''
    Gambling on the unit square environment.
    '''
    def __init__(self,
                 batch_size:int,
                 alpha:float=1.,
                 beta:float=1.,
                 length:Optional[int]=None,
                 rng:Optional[np.random.Generator]=None,
                 neg_reward_factor:float=1.,
                 seed:int=None):
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.length = length
        self.neg_reward_factor = neg_reward_factor
        self.rng = np.random.default_rng(seed) if rng is None else rng
        self.action_values = torch.tensor([-1, 0, 1], dtype=DATATYPE)
        self.action_space = Discrete(3)
        self.observation_space = Box(low=0, high=1, shape=(1,))
        self.state = None

    def reward(self,
               state:torch.Tensor|np.ndarray,
               action:torch.Tensor|np.ndarray,
               next_state:torch.Tensor|np.ndarray) -> torch.Tensor:
        next_state = torch.tensor(next_state, dtype=DATATYPE) if not isinstance(next_state, torch.Tensor) else next_state
        state = torch.tensor(state, dtype=DATATYPE) if not isinstance(state, torch.Tensor) else state
        action = torch.tensor(action, dtype=DATATYPE) if not isinstance(action, torch.Tensor) else action

        if next_state.ndim == 3:
            state = state.unsqueeze(-1)
            action = action.unsqueeze(-1)

        reward = action * (next_state - state)
        reward[reward < 0.] = reward[reward < 0.] * self.neg_reward_factor

        return reward

    def step(self, action:torch.Tensor|np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, bool, dict]:
        '''
        Action is a value and not an index.
        '''
        action = self.check_action(action)
        if self.state is None:
            raise ValueError("Reset the environment before taking any actions")
        if action.ndim > 2 or action.shape[0] != self.batch_size or action.shape[1] != 1:
            raise ValueError("Action must be of shape (batch_size, 1), got shape", action.shape)

        alphas = softplus(self.alpha - action.squeeze() * self.state.squeeze()) * (action.squeeze() != 0) + self.alpha * (action.squeeze() == 0)
        betas = softplus(self.beta + action.squeeze() * (1 - self.state.squeeze())) * (action.squeeze() != 0) + self.beta * (action.squeeze() == 0)
        next_state = torch.tensor(self.rng.beta(a=alphas, b=betas, size=self.batch_size), dtype=DATATYPE, requires_grad=False).unsqueeze(-1)

        rewards = self.reward(self.state, action, next_state)
        rewards = torch.tensor(rewards, dtype=DATATYPE, requires_grad=False) if not isinstance(rewards, torch.Tensor) else rewards
        self.state = next_state #NOTE: must be after reward calculation

        if self.length is not None:
            self.steps += 1
            if self.steps == self.length:
                return next_state, rewards, True, {}
        return next_state, rewards, False, {}

    def reset(self) -> torch.Tensor:
        if self.length is not None:
            self.steps = 0
        self.state = torch.tensor(self.rng.beta(a=self.alpha, b=self.beta, size=self.batch_size), dtype=DATATYPE, requires_grad=False).unsqueeze(-1)
        return self.state

    def check_action(self, action):
        '''
        Check the action and convert it to the correct format
        If the action space is Discrete, convert the action index to the action values
        '''
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=DATATYPE, requires_grad=False)
        elif not isinstance(action, torch.Tensor):
            action = torch.tensor([action], dtype=DATATYPE, requires_grad=False)

        if action.shape[0] != self.batch_size:
            raise ValueError("Action first dimension must be equal to batch size, got", action.shape[0])

        if action.ndim == 1:
            if action.shape[0] == 1:
                action = action.unsqueeze(0)
            elif action.shape[0] == self.batch_size:
                action = action.unsqueeze(-1)

        if type(self.action_space) == Discrete:
            action = action.to(torch.int32)
            action = self.action_values[action]

        return action

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed:int):
        self.rng = np.random.default_rng(seed)