import time
import gym
import nengo
import nengo_ocl
import numpy as np
import NEAT

env = gym.make('CartPole-v0').env


class EnvironmentInterface(object):
    def __init__(self, env, stepSize=5):
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


tau = 0.01

fast_tau = 0
slow_tau = 0.01



from gym import wrappers
from datetime import datetime

# Video Capturing Mechanism
filename = "test"
is_monitor = False
# env.close()
if is_monitor:
    # filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)
    env.reset()

Gen = int(input('Generation_number'))
prob_list = []
for i in range(Gen):
    score_list = []
    if i == 0:
        gene_list = NEAT.generate_first_generation(192, 4, 2).copy()
        print(gene_list)
        translated = NEAT.translate_gene_into_nengo_param(gene_list)
    else:
        gene_list = NEAT.crossover(gene_list, prob_list)
        print(gene_list)
        gene_list = NEAT.mutate(gene_list, 0.25, 0.25, 0.5)
        translated = NEAT.translate_gene_into_nengo_param(gene_list)
        print(translated)
        score_list = []
        prob_list = []

    for n in translated:
        envI = EnvironmentInterface(env)
        state_dimensions = envI.state_dim
        n_actions = envI.n_actions
        node = n[0]
        connection = n[1]
        model = nengo.Network()
        with model:
            sensor_nodes = nengo.Node(envI.sensor)
            sensing_neuron = nengo.Ensemble(n_neurons=envI.state_dim, dimensions=envI.state_dim)
            action_neurons = nengo.Ensemble(n_neurons=envI.n_actions, dimensions=envI.n_actions)
            step_node = nengo.Node(envI.step, size_in=2)
            nengo.Connection(action_neurons, step_node, synapse=fast_tau)
            nengo.Connection(sensor_nodes, sensing_neuron.neurons)
            middle_neurons = {}
            node = list(set(node)); print(node)
            for f in node:
                if f < envI.state_dim:
                    pass
                elif f < envI.n_actions:
                    pass
                else:
                    middle_neurons[f] = nengo.Ensemble(1, dimensions=1)
            for k in connection:
                if k[0] < envI.state_dim:
                    if k[1] < envI.n_actions:
                        nengo.Connection(sensing_neuron.neurons[k[0]], action_neurons.neurons[k[1]], synapse=fast_tau)
                    elif envI.n_actions <= k[1] < envI.state_dim:
                        nengo.Connection(sensing_neuron.neurons[k[0]],sensing_neuron.neurons[k[1]], synapse=tau)
                    else:
                        nengo.Connection(sensing_neuron.neurons[k[0]], middle_neurons[k[1]], synapse=tau)
                else:
                    if k[1] < envI.n_actions:
                        nengo.Connection(middle_neurons[k[0]], action_neurons.neurons[k[1]], synapse=tau)
                    elif envI.n_actions <= k[1] < envI.state_dim:
                        nengo.Connection(middle_neurons[k[0]], sensing_neuron.neurons[k[1]], synapse=tau)
                    else:
                        nengo.Connection(middle_neurons[k[0]], middle_neurons[k[1]], synapse=tau)
        simulator = nengo_ocl.Simulator(model)
        with nengo_ocl.Simulator(model) as sim:
            sim.run(3.0)
        print("Reward:" + str(sum(envI.reward_arr)))
        score_list.append(sum(envI.reward_arr))
        print(score_list)
    sum_score = sum(score_list)
    for z in score_list:
        prob_list.append(z / sum_score)




