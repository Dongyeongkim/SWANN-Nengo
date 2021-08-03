import ray
import gym
import time
import NEAT
import math
import nengo
import nengo_dl
import numpy as np

ray.init(webui_host='127.0.0.1',num_gpus=1)
env = gym.make('CartPole-v0').env

def average(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t - 100):(t + 1)].mean()
    return running_avg
@ray.remote(max_calls=1)
def sim(n):
    tau = 0.01
    sco_var_env = []
    weight_list = [-2,-1,1,2]
    for i in weight_list:
        envI = EnvironmentInterface(env)
        state_dimensions = envI.state_dim
        n_actions = envI.n_actions
        node = n[0]
        connection = n[1]
        model = nengo.Network()
        with model:
            sensor_nodes = nengo.Node(envI.sensor)
            sensing_neuron = nengo.Ensemble(n_neurons=state_dimensions, dimensions=state_dimensions)
            action_neurons = nengo.Ensemble(n_neurons=n_actions, dimensions=n_actions)
            step_node = nengo.Node(envI.step, size_in=n_actions)
            nengo.Connection(action_neurons, step_node, synapse=tau,solver=nengo.solvers.LstsqL2(weights=True))
            nengo.Connection(sensor_nodes, sensing_neuron.neurons,solver=nengo.solvers.LstsqL2(weights=True))
            middle_neurons = {}; middle_neurons.clear() 
            node = list(set(node))
            for f in node:
                if f < envI.state_dim:
                    pass
                elif f < envI.n_actions:
                    pass
                else:
                    middle_neurons[f] = nengo.Ensemble(1, dimensions=1)
            for x, k in enumerate(connection):
                if k[0] < envI.state_dim:
                    if k[1] < envI.n_actions:
                        nengo.Connection(sensing_neuron[k[0]], action_neurons[k[1]], synapse=tau,solver=nengo.solvers.LstsqL2(weights=True))
                    elif envI.n_actions <= k[1] < envI.state_dim:
                        nengo.Connection(sensing_neuron[k[0]], sensing_neuron[k[1]], synapse=tau,solver=nengo.solvers.LstsqL2(weights=True))
                    else:
                        nengo.Connection(sensing_neuron[k[0]], middle_neurons[k[1]], synapse=tau,solver=nengo.solvers.LstsqL2(weights=True))
                else:
                    if k[1] < envI.n_actions:
                        nengo.Connection(middle_neurons[k[0]], action_neurons[k[1]], synapse=tau,solver=nengo.solvers.LstsqL2(weights=True))
                    elif envI.n_actions <= k[1] < envI.state_dim:
                        nengo.Connection(middle_neurons[k[0]], sensing_neuron[k[1]], synapse=tau,solver=nengo.solvers.LstsqL2(weights=True))
                    else:
                        nengo.Connection(middle_neurons[k[0]], middle_neurons[k[1]], synapse=tau,solver=nengo.solvers.LstsqL2(weights=True))
        try:
            with nengo_dl.Simulator(model,device="/gpu:0",progress_bar=False) as sim:
                sim.run(20.0)
                for x in range(len(connection)):
                    print(sim.data[conn+str(x)].weights)
                avg_score_list = average(np.array(envI.reward_arr))
                sco_var_env.append(np.sum(avg_score_list)/len(avg_score_list)-len(n[1]))

        except nengo.exceptions.BuildError:
            sco_var_env.append(0)
    return sum(sco_var_env)/len(sco_var_env)


class EnvironmentInterface(object):
    def __init__(self, env, stepSize=1):
        self.env = env
        self.n_actions = env.action_space.n
        self.state_dim = env.observation_space.shape[0]
        self.t = 0
        self.stepsize = stepSize
        self.output = np.zeros(self.n_actions)
        self.state = env.reset()
        self.reward = 0
        self.current_action = 0
        self.reward_arr = []
        self.totalReward = 0

    def take_action(self, action):
        self.state, self.reward, self.done, _ = env.step(action)
        self.totalReward += self.reward
        if self.done:
            self.reward = -2
            self.totalReward += self.reward
            self.reward_arr.append(self.totalReward)
            self.state = env.reset()
            self.totalReward = 0

    def get_reward(self, t):
        return self.reward

    def sensor(self, t):
        return self.state

    def step(self, t, x):
        if int(t * 1000) % self.stepsize == 0:
            self.current_action = np.argmax(x)  # np.argmax(self.output)#
            self.take_action(self.current_action)

Gen = int(input('Generation_number'))
for i in range(Gen):
    if i == 0:
        gene_list = NEAT.generate_first_generation(192, 4, 2).copy()
        translated = NEAT.translate_gene_into_nengo_param(gene_list)
    else:
        gene_list = NEAT.crossover(gene_list, score_list, 8)
        gene_list = NEAT.mutate(gene_list, 0.25, 0.25, 0.1, env.observation_space.shape[0], env.action_space.n)
        translated = NEAT.translate_gene_into_nengo_param(gene_list)
    score_list = []
    score_list = [sim.remote(i) for i in translated]
    score_list = ray.get(score_list)
    sum_score = sum(score_list)
    f = open('reward/' + str(time.strftime('%c', time.localtime(time.time()))) + '.txt', 'w')
    f.write(str(score_list))
    f.close()
