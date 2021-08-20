import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.action_masks = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.action_masks[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var=256):
        super(ActorCritic, self).__init__()

        # actor
        self.obs_layer = nn.Linear(state_dim, n_latent_var)
        self.actor_layer = nn.Sequential(
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim)
        )

        #self.action_layer_weight = nn.Parameter(torch.ones(1,state_dim))
        #self.action_layer_bias = nn.Parameter(torch.zeros(1, state_dim))
        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self):
        raise NotImplementedError

    def hidden_state(self, state):
        hidden_state = self.obs_layer(state)
        return hidden_state

    def act(self, state, memory, action_mask):
        # self.action_layer_weight * state + self.action_layer_bias #B, N
        hidden_state = self.hidden_state(state)
        logits = self.actor_layer(hidden_state)
        inf_mask = torch.clamp(torch.log(action_mask.float()),
                               min=torch.finfo(torch.float32).min)
        logits = logits + inf_mask
        action_probs = F.softmax(logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.action_masks.append(deepcopy(action_mask))
        memory.logprobs.append(dist.log_prob(action))

        return action.detach()

    def evaluate(self, state, action, action_mask):
        hidden_state = self.hidden_state(state)
        logits = self.actor_layer(hidden_state)
        inf_mask = torch.clamp(torch.log(action_mask.float()),
                               min=torch.finfo(torch.float32).min)
        logits = logits + inf_mask
        action_probs = F.softmax(logits, dim=-1)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(hidden_state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr, betas, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        # discounted_reward = 0
        # for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
        #     if is_terminal:
        #         discounted_reward = 0
        #     discounted_reward = reward + (self.gamma * discounted_reward)
        #     rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        #rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = memory.rewards[0].repeat(len(memory.actions))
        rewards = rewards / (rewards.std() + 1e-5)
        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.cat(memory.states, dim=0).detach()
        old_actions = torch.cat(memory.actions, dim=0).detach()
        old_logprobs = torch.cat(memory.logprobs, dim=0).detach()
        old_actionmask = torch.cat(memory.action_masks, dim=0).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions, old_actionmask)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


def hard_sample(logits, dim=-1):
    y_soft = F.softmax(logits, dim=-1)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret, index.squeeze(1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var=256):
        super().__init__()
        # actor
        self.obs_layer = nn.Linear(state_dim, n_latent_var)
        self.actor_layer = nn.Sequential(
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim)
        )

    def forward(self, state, action_mask):
        hidden_state = self.obs_layer(state)
        logits = self.actor_layer(hidden_state)
        inf_mask = torch.clamp(torch.log(action_mask.float()),
                               min=torch.finfo(torch.float32).min)
        logits = logits + inf_mask
        train_mask, actions = hard_sample(logits)
        return train_mask, actions


class StraightThrough:
    def __init__(self, state_dim, action_dim, lr, betas):
        self.lr = lr
        self.betas = betas
        self.policy = Actor(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, betas=betas)

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()


def main():
    pass


if __name__ == '__main__':
    main()
