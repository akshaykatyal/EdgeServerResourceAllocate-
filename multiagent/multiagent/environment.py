#importing of the all the necessary libraries.
import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
import math
from multiagent.multi_discrete import MultiDiscrete
#number of task at the lower biund of group G1
l_a = 10
#upper bound for the number of tasks for group 1
l_b = 30
#Number of tasks/ms for the edge server
l_a_r = 100
#Number of tasks in q size
l_b_r = 200
remote_delay_max = 8 #ms
size_map = 1000/2000
Qlength = 200 #for the server
fc=2800 #frequency of cell is 2.8ghz
Bandwidth=60 #60mhz bandwidth
gaussianpower=-95 # gaussianmoise is -95dbm


# environment for all agents in the multiagent resourece allocattion environment
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, arglist, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        self.n = len(world.policy_agents)
        # resource allocate scenario callback
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.Q_type = arglist.Q_type
        #getting the reward from the user
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0
        #defining maximum number of step for the environment
        self.max_step = 23
        # configure spaces
        self.action_space = []
        #the observation space
        self.observation_space = []
        for agent in self.agents:
            #defining total action space for agents
            total_action_space = []
            #vehicle agent action space
            u_action_space = spaces.Discrete(agent.action.dim_a)
            #total action space available for agaent
            total_action_space.append(u_action_space)
            # total action space
            if len(total_action_space) > 1:
                # the action spaces for the vehicles are discrete
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            # observation space for the environment
            obs_dim = len(observation_callback(agent, self.world, step=0))
            self.observation_space.append(spaces.Box(low=0, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
    #function for the agent taking steps in environment
    def step(self, action_n, step_num, learning_type ='maddpg'):
        #obseration space for the steps for vehicle agents
        obs_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # setting actions for agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step(action_n)
        for agent in self.agents:
            #getting observation for the agents in environment
            obs = self._get_obs(agent, step_num)
            obs_n.append(obs)
            #getting the done for the steps agent take
            done_n.append(self._get_done(agent))
            #getting information for each step
            info_n['n'].append(self._get_info(agent))
        reward, changed_number, changed_group_num, delay_in_group, reward_real= self._get_reward(learning_type)
        if self.shared_reward and np.size(reward)==1:
            reward_n = [reward] * self.n
        else:
            #if no reward is received by the agent
            reward_n = reward
        obser=obs_n
        #this returns the observation, rewards, changed groups, delays
        return obser, reward_n, done_n, info_n, changed_number, changed_group_num, delay_in_group, reward_real

    #function to reset the environment
    def reset(self, train_step, step, arglist, IF_test=False, TEST_V = None):
        # reset environment world
        self.reset_callback(self.world, train_step, step, arglist, IF_test, TEST_V)
        obs_n = []
        self.agents = self.world.policy_agents
        #function for q delay
        Q_delay = np.zeros(self.world.agent_num)
        for agent in self.agents:

            Q_delay[agent.id] = self.get_latency_one_agent(agent, self.world, agent.service_rate)
            agent.state.Q_delay = Q_delay[agent.id]
            obs = self._get_obs(agent, step)
            #appending the number of observations
            obs_n.append(obs)
        self.world.last_delay = self.get_latency_group(self.world, Q_delay, self.world.all_con_group)
        #function returns the number of observations
        return obs_n
    #fucntion to get delay in a group
    def get_latency_group(self, world, Q_delay, agent):
        delay_in_group = np.zeros([world.region_W, world.region_H])
        tdelay_group=self.Transmission_delay(world)
        for x in range(world.region_W):
            for y in range(world.region_H):
                if agent[x,y]>0:
                    i = int(agent[x,y])-1
                    #getting the delay in the group
                    delay_in_group[x,y] = Q_delay[i] + world.group_delay[i,x,y] + tdelay_group
        return delay_in_group

    #function getting latency of one agent

    def get_latency_one_agent(self, agent, world, service_rate):
        #latency of agent based on Small group

        for i in enumerate(world.agents):
            pdelay=self.Process_delay(service_rate,world)
        agent_load_vector = agent.state.c_load * agent.state.v_manage
        load_all_cross = np.sum(agent_load_vector)
        #getting the value of q delay
        if self.Q_type == "inf":
            Q_delay = self.delay_Queue_inf(load_all_cross, service_rate, agent.state.fix_load) +pdelay
        else:
            Q_delay = self.delay_Queue(load_all_cross, service_rate, Qlength, agent.state.fix_load) +pdelay
        #getting latency due to delays of the agents
        return Q_delay


    # function to get q delay
    def delay_Queue_inf(self, load_server, service_rate, fix_load_server=0):

        lamda = load_server + fix_load_server
        mu = service_rate
        K = Qlength
        rho = lamda / mu
        if mu > lamda:
            delay = min(40, 1 / (mu - lamda))
        else:
            delay = 40
        #returns the delay q for simulation
        return delay

    # function to get the value of q delay
    def delay_Queue(self, load_server, service_rate, Qlength, fix_load_server=0):

        lamda = load_server + fix_load_server
        mu = service_rate
        K = Qlength
        rho = lamda / mu
        Qlen = 200
        if np.size(rho) > 1:
            delay = []
            for i, r in enumerate(rho):
                if lamda[i] == 0:
                    d = 0
                else:
                    if r == 1:
                        d = Qlen / mu + (K - 1) / (2 * lamda[i])
                    else:
                        d = Qlen / mu + (pow(r, 2) + K * pow(r, K + 2) - K * pow(r, K + 1) - pow(r, K + 2)) / (
                                lamda[i] * (1 - r) * (1 - pow(r, K)))
                delay.append(d)
        else:
            if lamda == 0:
                d = 0
            else:
                if rho == 1:
                    d = Qlen / mu + (K - 1) / (2 * lamda)
                else:
                    d = (pow(rho, 2) + K * pow(rho, K + 2) - K * pow(rho, K + 1) - pow(rho, K + 2)) / (
                            lamda * (1 - rho) * (1 - pow(rho, K)))
            delay = d
        return delay+1/service_rate


    # get info used for the results
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation of a particular agent in the environment
    def _get_obs(self, agent, step):
        if self.observation_callback is None:
            return np.zeros(0)
        #returns the observation for agent
        return self.observation_callback(agent, self.world, step)

    # get dones for a agent in a environment
    def _get_done(self, agent):
        if self.done_callback is None:
            return 0
        #returns the number of dones for a agent
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, learning_type):
        if self.reward_callback is None:
            return 0.0
        #returns the reward for the type of lerning algorithm
        return self.reward_callback(self.world, learning_type)
    # simple_controllr -> reward()

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        #getting the probability of action from agent
        agent.action.p_ctl = action

    #fucntion for getting the data rate
    def datarate(self):
        path_loss=32.4+20*math.log10(fc)+31.9*math.log10(size_map)#this is the path loss as mentioned in the paper
        path=path_loss/10
        Power=100 #packets/ms, The transmit power of the server
        datarate=Bandwidth*math.log2(1+ (Power * pow(10,-path)/gaussianpower))
        return datarate
    # #function to ge the process delay
    # def process_delay(self, world):
    #     mu=1000
    #     pro=[]
    #     for agent in world.agents:
    #         load_agent = []
    #         all_group_agent = agent.group
    #         for i in range(len(all_group_agent)):
    #             load = world.load_group[all_group_agent[i]]
    #             load_agent.append(load)
    #         #for i in mu:
    #         #  if mu[i] == 0:
    #         #   pro_delay = 0
    #         # else:
    #         for agent in load_agent:
    #             pro_delay = agent/mu
    #     return pro_delay

    #
    #
    #function for getting the processing delay
    def Process_delay(self,servicerate, world):
        mu=servicerate
        for agent in world.agents:
            load_agent = []
            all_group_agent = agent.group
            for i in range(len(all_group_agent)):
                load = world.load_group[all_group_agent[i]]
                load_agent.append(load)
            for a in load_agent:
                prodelay=a/mu
        return prodelay
    #function for gettig transmission delay
    # def Transmssion_delay(self,world):
    #     rate=100
    #     for agent in world.agents:
    #         load_agent = []
    #         all_group_agent = agent.group
    #         for i in range(len(all_group_agent)):
    #             load = world.load_group[all_group_agent[i]]
    #             load_agent.append(load)
    #         for a in load_agent:
    #             tdelay=a/rate
    #     return tdelay
    #function to get the transmission delay
    def Transmission_delay(self,world):
        load_c =np.sum(world.load_group)
        rate=self.datarate()
        tdelay= load_c/rate
        return tdelay
