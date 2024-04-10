import subprocess
import inspect
import time
from statistics import mean, stdev
import matplotlib.pyplot as plt

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
# from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
# from CybORG.Agents.SimpleAgents.BlueLoadAgent import BlueLoadAgent
# from CybORG.Agents.SimpleAgents.BlueReactAgent import BlueReactRemoveAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper
from CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG.Agents.MainAgent import MainAgent

MAX_EPS = 100
agent_name = 'Blue'


def wrap(env):
    return ChallengeWrapper(env=env, agent_name='Blue')

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    commit_hash = get_git_revision_hash()
    # # ask for a name
    # name = input('Name: ')
    # # ask for a team
    # team = input("Team: ")
    # # ask for a name for the agent
    # name_of_agent = input("Name of technique: ")

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    # Change this line to load your agent
    agent = MainAgent()

    # print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

    # file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.txt'
    # print(f'Saving evaluation results to {file_name}')
    # with open(file_name, 'a+') as data:
    #     data.write(f'CybORG v{cyborg_version}, {scenario}, Commit Hash: {commit_hash}\n')
    #     data.write(f'author: {name}, team: {team}, technique: {name_of_agent}\n')
    #     data.write(f"wrappers: {wrap_line}\n")

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

    all_rewards = []
    for i in range(100):
        all_rewards.append([])

    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    for num_steps in [100]:
        # for red_agent in [B_lineAgent, RedMeanderAgent, SleepAgent]:
        red_agent = RedMeanderAgent
        red2 = B_lineAgent
        cyborg = CybORG(path, 'sim', agents={'Red': red_agent, 'Red2': red2})
        wrapped_cyborg = wrap(cyborg)

        observation = wrapped_cyborg.reset()
        # observation = cyborg.reset().observation

        action_space = wrapped_cyborg.get_action_space(agent_name)
        # action_space = cyborg.get_action_space(agent_name)
        total_reward = []
        actions = []
        for i in range(MAX_EPS):
            r = []
            a = []
            # cyborg.env.env.tracker.render()
            for j in range(num_steps):
                action = agent.get_action(observation, action_space)
                # action = 0
                observation, rew, done, info = wrapped_cyborg.step(action)
                # result = cyborg.step(agent_name, action)
                r.append(rew)
                all_rewards[j].append(rew)
                # r.append(result.reward)
                a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))
            # agent.end_episode()
            total_reward.append(sum(r))
            actions.append(a)
            # observation = cyborg.reset().observation
            observation = wrapped_cyborg.reset()
        print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
            # with open(file_name, 'a+') as data:
            #     data.write(f'steps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
            #     for act, sum_rew in zip(actions, total_reward):
            #         data.write(f'actions: {act}, total reward: {sum_rew}\n')
    
    # Compute the average reward and st dev for each step
    # Note: This is NOT cumulative
    means = []
    stdevs = []
    for i in range(len(all_rewards)):
        means.append(mean(all_rewards[i]))
        stdevs.append(stdev(all_rewards[i]))

    cum_means = [means[0]]
    for i in range(1, len(means)):
        cum_means.append(cum_means[i - 1] + means[i])

    print("Average reward per step: ", means)
    print("Cumulative average reward: ", cum_means)
    print("Standard deviation per step: ", stdevs)

    xs = [x for x in range(len(cum_means))]
    plt.plot(xs, cum_means)
    plt.xlabel("Time step")
    plt.ylabel("Cumulative average reward")
    plt.show()
    plt.close()