#import of all the necessary libraries
import numpy as np
from multiagent.core import World,Agent, Vehicle
from multiagent.scenario import BaseScenario
import json
import networkx as nx

import math
#size of the map of the environment
size_map = 1000/2000
# tasks/ms, this is the lower bound of the task arriving rate
l_a = 10
#upper bound of task arrivig rate for group 1
l_b = 30
#number of task by the server 100 tasks/ms
l_a_r = 100
#tasks in the q size
l_b_r = 200# packets of Q size
#max delay
remote_delay_max = 8
#qlength=200
Qlength = 200
#frequency of the cell is 2.8ghz
fc=2800 #frequency of cell is 2.8ghz
Bandwidth=60 #60mhz bandwidth
gaussianpower=-95 # gaussianmoise is -95dbm
#transmission delay for teh edge server
#Tdelay=l_a/l_a_r

#creating a class for scenario
class Scenario(BaseScenario):
    #function to make world for the scenario environment
    def make_world(self, arglist, groupnum = None):
        world = World()
        world.collaborative = True
        # adding agenets to world
        world.agents = [Agent() for i in range(world.agent_num )]
        #random position of the agents of the agents
        position = (np.array([(38, 29),(40, 70),(56, 34),(72.4, 73.3)])/(100/world.region_W))
        for i, agent in enumerate(world.agents):
            agent.id = i
            agent.name = 'id of agent %d' % i
            # position of agents
            agent.pos = tuple(position[i])
        #set the edge serve rate in the world
        world.set_server_rate()
        #set the crossover between the groups in the edge server environment
        world.get_crossover()
        #set the action dimentions for agent in world
        world.set_action_dim()
        #getting the coordinates of the edge
        world.set_coord(arglist.Coordinates_Edge_dir)

        #getting delays in the world
        self.get_delay_group(world)
        # get delay of cross group between agents, when vehcle cross the groups
        self.get_cross_group_delay(world)

        # make initial conditions
        self.reset_world(world, -1, 0, arglist)
        for agent in world.agents:
            agent.state.v_manage = agent.distance_manage
        #handling handover
        self.Handover = arglist.Handover
        #observation of queue in the edge server
        self.Que_obs = arglist.Que_obs
        #group traafic nmber of vehicles
        self.Group_traffic = arglist.Group_traffic
        #no of steps
        self.step_num = arglist.step_num
        #observ
        self.Step_observation = arglist.Step_observation
        self.reward_maddpg = arglist.reward_maddpg
        return world

        #Function to reset the environment world
    def reset_world(self, world, train_step, step, arglist, IF_test=False, TEST_V=None):
        # load of each block in height and width of the region
        if train_step > - 1:
            # getting the vehcle location from the bus dataset
            self.get_location_vehicle(world, train_step, step, arglist.vehicle_data_dir)
        else:
            #else generate vehicle function is used for getting the position
            self.generate_vehicle(world)

        self.Q_type = arglist.Q_type
        #service rate for the taks in the server
        self.service_rate =[]
        for agent in world.agents:
            #this is appending of the service rate for the edge server tasks
            self.service_rate.append(agent.service_rate)
        for agent in world.agents:
            load_agent = []
            all_group_agent = agent.group
            for i in range(len(all_group_agent)):
                load = world.load_group[all_group_agent[i]]
                load_agent.append(load)
            #get load of agent in all its coverage groups
            agent.state.load = load_agent

            # get load of cross group
            list_value = world.mul_group_c.get(agent.id)[0]
            agent_value_num = len(list_value)
            agent_load =[]
            for i in range(agent_value_num):
                load = world.load_group[list_value[i]]
                agent_load.append(load)
            agent.state.c_load = agent_load

            diff_agent_load = []
            # get load of fix group covered by the server
            diff_value = sorted(list(set(agent.group).difference(set(list_value))))
            for i in range(len(diff_value)):
                load = world.load_group[diff_value[i]]
                diff_agent_load.append(load)

            agent.state.fix_load = sum(diff_agent_load)
        # getting latency from the agent
        self.propagation_distance_based_delay(world)
        # get delay of each group based on centalised controller
        world.centralized_Delay = self.get_delay_centralized(world)

    #function for propagation distance based delay for the eedge server resource allocation
    def propagation_distance_based_delay(self, world):
        Q_delay_fix= self.get_latency_server_fix(world, np.array(self.service_rate))

        delay_in_group_fix = self.get_latency_group(world, Q_delay_fix, world.distance_assign_matrix)
        delay_agent_fix = np.zeros(world.agent_num)
        for i, agent in  enumerate(world.agents):
            load_agent = world.agent_group_cover[i,:,:]*world.load_group
            if np.sum(load_agent) <=0:
                delay_agent_fix[i] =0
            else:
                delay_agent_fix[i]=np.sum(load_agent*delay_in_group_fix)/np.sum(load_agent)
        #the agent and group delay in the server
        world.delay_fix = delay_agent_fix
        world.delay_in_group_fix = delay_in_group_fix


    # now we get the centralised value of the delay for the edge server
    def get_delay_centralized(self,world):
        load_c =np.sum(world.load_group)
        # tdelay=self.Transmssion_delay(world)
        if self.Q_type == "inf":
            Q_delay_one_agent = self.delay_Queue_inf(load_c, np.sum(np.array(self.service_rate)), fix_load_server=0)
            #  pdelay_one_agent=self.process_delay(world)
            tdelay_one_agent=self.Transmission_delay(world)
        else:
            Q_delay_one_agent = self.delay_Queue(load_c, np.sum(np.array(self.service_rate)), world.agent_num*Qlength, fix_load_server=0)
            #            pdelay_one_agent=self.process_delay(world)
            tdelay_one_agent=self.Transmission_delay(world)
        n = len(world.vehicles)/1200
        delay_group_centralzed_agent = max(2,min(remote_delay_max, n*remote_delay_max)) + Q_delay_one_agent + +tdelay_one_agent+ 1.3 #  pdelay_one_agent

        return delay_group_centralzed_agent

    def reward(self, world, learning_type = 'maddpg'):
        #TRANSMISSION DELAY
        tdelay_group=self.Transmission_delay(world)
        #number of vehicles changed in the server
        changed_num_vehicle = np.zeros(world.agent_num)
        changed_group_num = np.zeros(world.agent_num)
        Q_delay=np.zeros(world.agent_num)
        hand_over = np.zeros(world.agent_num)
        #GETTING Q DELAY
        for i, agent in  enumerate(world.agents):
            Q_delay[i]= self.get_latency_one_agent(agent, world, np.array(self.service_rate[i]))

            a = world.all_con_group.copy()

            b = world.last_all_con_group.copy()
            #agent changing the groups
            a[a != agent.id+1]=0
            b[b != agent.id+1]=0
            changed_group = a - b
            changed_group[ changed_group<0 ]=0
            changed_group[ changed_group!=0]=1
            all_num = a*world.vehicle_num
            #sumo of the changed group
            changed_group_num[i] = sum(sum(changed_group))
            changed_num_vehicle[i] = sum(sum(changed_group*world.vehicle_num))

            x = np.argwhere(changed_group ==1)

            # considered the change group and handover cost
            if np.size(x)==0:
                hand_over[i]=0
            else:
                hand_over[i] = changed_num_vehicle[i]*0.05

        delay_in_group = self.get_latency_group(world, Q_delay, world.all_con_group)
        Changed_percentage_all = np.sum(changed_num_vehicle)/sum(sum(world.vehicle_num))
        delay_agent = np.zeros(world.agent_num)
        #pdelay calculate
        for i, agent in  enumerate(world.agents):
            load_agent = world.agent_group_cover[i,:,:]*world.load_group
            pdelay=self.Process_delay(np.array(self.service_rate[i]),world)
            if np.sum(load_agent) <=0:
                delay_agent[i] =0
            elif self.Handover:
                delay_agent[i]=np.max(world.agent_group_cover[i,:,:]*delay_in_group)
            else:
                delay_agent[i]=np.max(world.agent_group_cover[i,:,:]*delay_in_group)

        if self.Handover:
            reward_real =  (- np.max(delay_in_group) - np.sum(changed_group_num)/world.cross_group_num)/30
        #
        else:
            reward_real =  - np.max(delay_in_group)/30
        if learning_type == 'maddpg':
            #adding of the delay
            reward = - delay_agent/30 + tdelay_group/30 + pdelay/30

        elif self.reward_maddpg:
            reward = reward_real
        else:
            reward =  - delay_agent/30

        world.delay_ma = delay_agent
        #returning the reard of the agent
        return (reward, Changed_percentage_all,
                np.sum(changed_group_num)/world.cross_group_num, delay_in_group, reward_real)

    #getting the latency in the group for the edge servers
    def get_latency_group(self, world, Q_delay, agent):
        delay_in_group = np.zeros([world.region_W, world.region_H])
        tdelay_group=self.Transmssion_delay(world)
        for x in range(world.region_W):
            for y in range(world.region_H):
                if agent[x,y]>0:
                    i = int(agent[x,y])-1
                    delay_in_group[x,y] = Q_delay[i] + world.group_delay[i,x,y] + tdelay_group
        return delay_in_group

    def get_delay_all(self, world):
        latency_p=np.zeros([world.agent_num, world.num_v])
        for j, vehicle in enumerate(world.vehicles):
            dists = [np.sqrt(np.sum(np.square(world.router_pos[node] - vehicle.pos))) for node in list(world.topo_matrix.node)]
            min_dists = dists.index( min(dists))
            a1 = np.array(world.path_delay_p)
            latency_p[:, j] = a1[:, min_dists]
            return latency_p


    def get_latency_one_agent(self, agent, world, service_rate):
        #  latency of agent based on Small group
        pdelay_one_agent=self.process_delay(world)
        #  tdelay_one_agent=self.Transmission_delay(world)
        agent_load_vector = agent.state.c_load * agent.state.v_manage
        load_all_cross = np.sum(agent_load_vector)
        #getting the value of q delay
        if self.Q_type == "inf":
            Q_delay = self.delay_Queue_inf(load_all_cross, service_rate, agent.state.fix_load) + pdelay_one_agent
        else:
            Q_delay = self.delay_Queue(load_all_cross, service_rate, Qlength, agent.state.fix_load) + pdelay_one_agent
        return Q_delay



    def get_latency_server_fix(self, world, service_rate):
        # latency of agent based on Small groups
        Q_delay = np.zeros(world.agent_num)
        # pdelay=self.process_delay(world)

        for i, agent in  enumerate(world.agents):
            agent_load_vector = agent.state.c_load * agent.distance_manage
            load_all_cross = np.sum(agent_load_vector)
            if self.Q_type == "inf":
                Q_delay[i] = self.delay_Queue_inf(load_all_cross, service_rate[i], agent.state.fix_load)
            # tdelay=self.Transmission_delay(world)#+pdelay
            else:
                Q_delay[i] = self.delay_Queue(load_all_cross, service_rate[i], Qlength, agent.state.fix_load)
            # tdelay=self.Transmission_delay(world)#+pdelay
        #getting the Q delay as server has a queue
        return Q_delay #+tdelay #+ pdelay

    #function to get q delay
    def delay_Queue_inf(self, load_server, service_rate, fix_load_server=0):

        lamda = load_server+ fix_load_server
        mu = service_rate
        K = Qlength
        rho = lamda/mu
        if mu > lamda:
            delay = min(35, 1/(mu-lamda))
        else:
            delay = 40
        return delay

    #function to get the value of q delay
    def delay_Queue(self, load_server, service_rate, Qlength, fix_load_server=0):

        lamda = load_server+ fix_load_server
        mu = service_rate
        K = Qlength
        rho = lamda/mu
        Qlen=200
        if np.size(rho) >1:
            delay =[]
            for i, r in enumerate(rho):
                if lamda[i] == 0:
                    d = 0
                else:
                    if r == 1:
                        d = Qlen/mu+ (K-1)/(2*lamda[i])
                    else:
                        d = Qlen/mu + (pow(r,2)+K*pow(r, K+2)-K*pow(r,K+1)-pow(r,K+2))/(lamda[i]*(1-r)*(1-pow(r,K)))
                delay.append(d)
        else:
            if lamda == 0:
                d = 0
            else:
                if rho == 1:
                    d = Qlen/mu+ (K-1)/(2*lamda)
                else:
                    d = (pow(rho,2)+K*pow(rho, K+2)-K*pow(rho,K+1)-pow(rho,K+2))/(lamda*(1-rho)*(1-pow(rho,K)))
            delay=d
        return delay

    #function for the observation of the server, world and the agent steps
    def observation(self, agent, world, step):
        load_group = list(map(lambda x:x/10, agent.state.c_load))
        obs = np.concatenate(([agent.state.fix_load/10], load_group), axis=0)
        if self.Step_observation:
            obs = np.concatenate((obs, [step/self.step_num]), axis=0)
        if self.Handover:
            obs =np.concatenate((obs, agent.state.v_manage*agent.state.c_load/10), axis=0)
        if self.Que_obs:
            obs =np.concatenate((obs,[agent.state.Q_delay/40] ), axis=0)
        return obs

    #generation of load for the servers by vehicle for a given range
    def generate_vehicle(self, world):
        # number of vehicles
        world.num_v = np.random.randint(50, high=120)
        # add vehicles
        world.vehicles = [Vehicle() for i in range(world.num_v)]#
        #vehicle location is in the coverage of agents
        select_group = np.random.randint(0,high=len(world.all_group), size = world.num_v)
        self.load = np.random.uniform(10, 20, size=world.num_v)
        for i, vehicle in enumerate(world.vehicles):
            x = world.all_group[select_group[i]][0]
            y = world.all_group[select_group[i]][1]
            #getting the position coordinates of vehicle
            vehicle.pos = (float(x+np.random.rand(1)),float(y+np.random.rand(1)))
            vehicle.load = self.load[i]/1000

    # get vehicle location from dataset
    def get_location_vehicle(self, world, train_step, step, dic):
        # episode by episode data also the data of dataset
        #group by group traffic
        k = train_step%self.Group_traffic+1
        if  k <10:
            name = '0'+str(k)
        else:
            name = str(k)
        with open(dic+name+'_min_10km.json','r') as f:
            diction = json.load(fp=f)
        with open(dic+'edge.json','r') as f:
            edge_dic = json.load(fp=f)

        #getting the locaton of the vehicles
        location = diction[str(step)]
        # number of vehicles are returned
        world.num_v = len(location)
        # adding vehicles to server
        world.vehicles = [Vehicle() for i in range(world.num_v)]
        #load for group 1 in the edge server
        self.load = (np.array(location)[:,2]-10)*(l_b-l_a)/10+l_a

        world.load_group = np.zeros([world.region_W, world.region_H] )
        world.vehicle_num = np.zeros([world.region_W, world.region_H] )
        #getting the vehicles from the file and loading in the group for the tasks
        for j, vehicle in enumerate(world.vehicles):
            x = location[j][0]/(100/world.region_W)
            y = location[j][1]/(100/world.region_H)
            vehicle.pos = (x,y)
            vehicle.load = self.load[j]/1000
            world.load_group[int(x),int(y)] += vehicle.load
            world.vehicle_num[int(x),int(y)]+=1


        #load on the edge servers based on the rate of the server
        for u in range(len(edge_dic)):
            x = edge_dic[u][0]/(100/world.region_W)
            y= edge_dic[u][1]/(100/world.region_H)
            n = 1
            load = n*((edge_dic[u][2+step]-50)*(l_b_r-l_a_r)/(100-50)+l_a_r)
            edge_load =min(max(l_a_r,load),l_b_r)
            world.load_group[int(x),int(y)] += edge_load/1000

    # function t create test vehicles for the edge server
    def test_vehicle_data(self, world,test_num):
        vehicles = []
        for k in range(test_num):
            num_v = np.random.randint(50, high=120)
            select_group = np.random.randint(0, high=len(world.all_group), size = num_v)
            load = np.random.uniform(0, 5, size = num_v)
            vehicle = []
            for i in range(num_v):
                x = world.all_group[select_group[i]][0]
                y = world.all_group[select_group[i]][1]
                pos_load = [float(x+np.random.rand(1)),float(y+np.random.rand(1)),load[i]]
                vehicle.append(pos_load)
            vehicles.append(vehicle)
        return vehicles

    #this function is used to get the group based delay
    def get_delay_group(self, world):
        world.group_delay = np.zeros([world.agent_num, world.region_W, world.region_H])
        for i, agent in enumerate(world.agents):
            for x in range(world.region_W):
                for y in range(world.region_H):
                    #gettingig llinear algerbra norm
                    if  np.linalg.norm([x,y] - np.array(agent.pos))> 1.5*agent.r :
                        #in not in coverage of agent, delay set to be inf
                        delay=1000000
                    else:
                        #getting linear algebra norm
                        dis =  np.linalg.norm([x,y] - np.array(agent.pos))/world.region_W*10
                        delay = (0.1 +dis*0.01/3)*2 + 1
                    world.group_delay[i, x, y] = delay

    #get delay in vehicles crossing groups between edge server
    def get_cross_group_delay(self, world):

        world.distance_assign_matrix = world.all_con_group.copy()
        for i, agent in enumerate(world.agents):
            a = np.array(agent.cross_group)
            agent.c_group_pro_delay = world.group_delay[i,a[:,0],a[:,1]]
            # assign of the fix assignment based on the vehicle distance
            dis_assi = np.zeros(len(agent.cross_group))
            tmp = world.group_delay[:,a[:,0],a[:,1]]
            agent_in_chardge = np.argmin(tmp, axis=0)
            dis_assi[np.where(agent_in_chardge==i)]=1
            agent.distance_manage = dis_assi

        for x in range(world.region_W):
            for y in range(world.region_H):
                id_ = np.argmin( world.group_delay[:,x,y], axis=0)
                if world.distance_assign_matrix[x,y] !=0:
                    world.distance_assign_matrix[x,y] = id_+1
                LB  = (x,y)
                for agent in world.agents:
                    distance = np.linalg.norm(np.array(agent.pos) - LB)
                    if distance <= agent.r:
                        tmp = world.group_delay[:,x,y]
                        world.agent_group_cover[agent.id,x,y] = 1

    #Function for getting the data rate based on the parameters
    def datarate(self):
        path_loss=32.4+20*math.log10(fc)+31.9*math.log10(size_map)#this is the path loss as mentioned in the paper
        path=path_loss/10
        Power=100 #packets/ms, The transmit power of the server
        datarate=Bandwidth*math.log2(1+ (Power * pow(10,-path)/gaussianpower))
        return datarate
    # #function to ge the process delay
    # # def process_delay(self, world):
    # #     mu=1000
    # #     pro=[]
    # #     for agent in world.agents:
    # #         load_agent = []
    # #         all_group_agent = agent.group
    # #         for i in range(len(all_group_agent)):
    # #             load = world.load_group[all_group_agent[i]]
    # #             load_agent.append(load)
    # #         #for i in mu:
    # #         #  if mu[i] == 0:
    # #         #   pro_delay = 0
    # #         # else:
    # #         for agent in load_agent:
    # #             pro_delay = agent/mu
    # #     return pro_delay
    #
    # #    #
    # function for getting the process delay
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
    # #function for getting transmission delay
    # # def Transmssion_delay(self,world):
    # #     rate=100
    # #     for agent in world.agents:
    # #         load_agent = []
    # #         all_group_agent = agent.group
    # #         for i in range(len(all_group_agent)):
    # #             load = world.load_group[all_group_agent[i]]
    # #             load_agent.append(load)
    # #         for a in load_agent:
    # #             tdelay=a/rate
    # #     return tdelay
    def Transmission_delay(self,world):
        load_c =np.sum(world.load_group)
        rate=self.datarate()
        tdelay= load_c/rate
        return tdelay
    #function for getting the total delay in the resource allocation environment
    def Totaldelay(self, world):
        propgation=self.propagation_distance_based_delay(world)
        trans=self.Transmission_delay(world)
        qdelay = np.zeros(world.agent_num)
        for i, agent in enumerate(world.agents):
            agent_load_vector = agent.state.c_load * agent.distance_manage
            load_all_cross = np.sum(agent_load_vector)
            pdelay=self.Process_delay(np.array(self.service_rate[i]),world)
            qdelay[i]= self.delay_Queue(load_all_cross, np.array(self.service_rate[i]), Qlength, agent.state.fix_load)
        totaldelay=propgation+max(trans,qdelay)+pdelay
        return totaldelay




