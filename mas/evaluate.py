import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from MADDPG import MADDPG
from main import get_env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type = str, default = 'simple_adversary_v2', help = 'name of the env',
                        choices = ['simple_adversary_v2', 'simple_spread_v2', 'simple_tag_v2'])
#     parser.add_argument('--folder', type = str, help = 'name of the folder where model is saved')
    parser.add_argument('--num-episodes', type = int, default = 10, help = 'total episode num during evaluation')
    parser.add_argument('--episode-length', type = int, default = 50, help = 'steps per episode')

    args, _ = parser.parse_known_args()

    model_dir = os.path.join(os.getcwd(), 'results', args.env_name)#, args.folder)
    if os.path.exists(model_dir):
        gif_dir = os.path.join(model_dir, 'gif')
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    gif_num = len([file for file in os.listdir(gif_dir)])  # current number of gif

    env, dim_info = make_env(args.env_name, args.episode_length)
    
    maddpg = MADDPG.load(dim_info, os.path.join(model_dir, f'{total_files + 1}', 'maddpg_actor_for_mpe.pt'))

    num_agents = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent: np.zeros(args.num_episodes) for agent in env.agents}
    for ep in range(args.num_episodes):
        obs = env.reset()
        agent_rewards = {agent: 0 for agent in env.agents}  # agent reward of the current episode
        frame_list = []  # used to save gif
        while env.agents:  # interact with the env for an episode
            actions = maddpg.get_actions(obs)
            nobs, rewards, dones, info = env.step(actions)
            frame_list.append(Image.fromarray(env.render(mode = 'rgb_array')))
            obs = nobs

            for agent_id, reward in rewards.items():  # update reward
                agent_rewards[agent_id] += reward
        env.close()
        
        # Record rewards with respect to agent and episode and print the current status.
        message = f'Episode: {ep + 1}\t|'
        for agent_id, r in agent_rewards.items():
            episode_rewards[agent_id][ep] = r
            message += f' {agent_id}: {r:3.2f}'
        print(message)

        # Save gif.
        frame_list[0].save(os.path.join(gif_dir, f'out{gif_num + ep + 1}.gif'),
                           save_all = True, append_images = frame_list[1:], duration = 1, loop = 0)

    # Evaluation performance plot
    plt.figure(figsize = (10, 6))
    x = range(1, args.num_episodes + 1)
    for agent_id, rewards in episode_rewards.items():
        plt.plot(x, rewards, label = agent_id)
    plt.legend(loc = 'lower right', fontsize = 10)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    total_files = len([file for file in os.listdir(model_dir)])
    title = f'Evaluation result of MADDPG for {args.env_name} {total_files - 3}'
    plt.title(title)
    plt.savefig(os.path.join(model_dir, title))