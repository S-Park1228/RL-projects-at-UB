# Reference: https://github.com/Git-123-Hub/maddpg-pettingzoo-pytorch

import argparse
import os

from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from pettingzoo.mpe import simple_adversary_v2, simple_spread_v2, simple_tag_v2

from MADDPG import MADDPG



def make_env(env_name, ep_len = 25):
    
    """
    Create a MPE environment and get observation and action dimension of each agent in this environment.
    """
    
    new_env = None
    if env_name == 'simple_adversary_v2':
        new_env = simple_adversary_v2.parallel_env(max_cycles = ep_len)
    if env_name == 'simple_spread_v2':
        new_env = simple_spread_v2.parallel_env(max_cycles = ep_len)
    if env_name == 'simple_tag_v2':
        new_env = simple_tag_v2.parallel_env(max_cycles = ep_len)

    new_env.reset() # It needs to be executed before using the new_env's attributes.
    _dim_info = {}
    for agent_id in new_env.agents:
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).n)

    return new_env, _dim_info

# Polyak update
def soft_update(tau, network, target_network):
    for params, target_params in zip(network.parameters(), target_network.parameters()):
        target_params.data.copy_(tau * params.data + (1.0 - tau) * target_params.data)

if __name__ == '__main__':
    # for logging
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type = str, default = 'simple_adversary_v2', help = 'name of the env',
                        choices=['simple_adversary_v2', 'simple_spread_v2', 'simple_tag_v2'])
    parser.add_argument('--num_episodes', type = int, default = 30000, help = 'total number of episodes during training procedure')
    parser.add_argument('--episode_length', type = int, default = 25, help = 'steps per episode')
    parser.add_argument('--learn_interval', type = int, default = 100, help = 'steps interval between learning time')
    parser.add_argument('--sampling_after', type = int, default = 5e4, help = 'random steps before accumulating sufficiently large replay buffer data')
    parser.add_argument('--tau', type = float, default = 0.02, help = 'soft update parameter')
    parser.add_argument('--gamma', type = float, default = 0.95, help = 'discount factor (gamma)')
    parser.add_argument('--capacity', type = int, default = int(1e6), help = 'capacity of replay buffer')
    parser.add_argument('--batch_size', type = int, default = 1024, help = 'batch-size of replay buffer')
    parser.add_argument('--actor_lr', type = float, default = 0.01, help = 'learning rate of actor')
    parser.add_argument('--critic_lr', type = float, default = 0.01, help = 'learning rate of critic')
    args, _ = parser.parse_known_args()

    # Create folders to save results.
    env_dir = os.path.join(os.getcwd(), 'results', args.env_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    result_dir = os.path.join(env_dir, f'{total_files + 1}')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Create a MPE environment.
    env, dim_info = make_env(args.env_name, args.episode_length)
    
    # Instantiate the MADDPG.
    maddpg = MADDPG(dim_info, args.capacity, args.batch_size, args.actor_lr, args.critic_lr, result_dir)

    counter = 0  # global counter
    print_every = 100 # Print results every 100 episodes.
    
    # reward of each episode for each agent
    episode_rewards = {agent_id: np.zeros(args.num_episodes) for agent_id in env.agents}
    episode_rewards_window = {agent_id: deque(maxlen = 100) for agent_id in env.agents}
    episode_rewards_ma = {agent_id: np.zeros(args.num_episodes) for agent_id in env.agents}
    
    # training
    for ep in range(args.num_episodes):
        obs = env.reset()
        agents = env.agents
        agent_rewards = {agent_id: 0 for agent_id in agents}  # agent rewards of the current episode
        
        while env.agents: # env.agents: ['adversary_0', 'agent_0', 'agent_1']
            counter += 1
            if counter < args.sampling_after:
                actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
            else:
                actions = maddpg.get_actions(obs)

            next_obs, rewards, dones, info = env.step(actions)
            
            
            # Push experience and update reward.
            maddpg.push(obs, actions, rewards, next_obs, dones)
            

            for agent_id in agents:                
                agent_rewards[agent_id] += rewards[agent_id]

            if counter >= args.sampling_after and counter % args.learn_interval == 0:  # Update every few steps.
                # Update networks and target networks.
                maddpg.update(args.batch_size, args.gamma)
                for agent_id in agents:
                    soft_update(args.tau, maddpg.agents[agent_id].actor, maddpg.agents[agent_id].target_actor)
                    soft_update(args.tau, maddpg.agents[agent_id].critic, maddpg.agents[agent_id].target_critic)

            obs = next_obs

        # Record rewards with respect to agent and episode and print the current status.
        for agent_id, r in agent_rewards.items():
            episode_rewards[agent_id][ep] = r
            episode_rewards_window[agent_id].append(r)
            episode_rewards_ma[agent_id][ep] = np.mean(episode_rewards_window[agent_id])
        
        if (ep + 1) % print_every == 0:
            ep_reward = 0 # Initialize the total reward for each episode.
            message = f'Episode: {ep + 1}\t|'
            for agent_id, r in agent_rewards.items():
                message += f'{agent_id}: {r:3.2f}\t|'
                ep_reward += r
            message += f'Episode Reward: {ep_reward:3.2f}'
            print(message)

    maddpg.save(episode_rewards)  # Save the model (actor-network parameters of all agents) and its training performance.

    # training finishes, plot reward
    plt.figure()
    x = range(1, args.num_episodes + 1)
    for agent_id, rewards in episode_rewards.items():
        plt.plot(x, rewards, label = agent_id)
        plt.plot(x, episode_rewards_ma[agent_id], label = agent_id + ' moving average')
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Training result of MADDPG for {args.env_name}')
    plt.savefig(os.path.join(result_dir, title))