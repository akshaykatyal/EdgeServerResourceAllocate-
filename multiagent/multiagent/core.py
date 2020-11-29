#importing of all the required libraries
import numpy as np
import networkx as nx

#the size map for the environment
size_map = 1000/2000

# physical/external base state of all entites of the environment
#where the entities are vehicles and they have a position and velocity
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication of the edge server with entities
        self.communication = None
        # controlled vehicles by the server
        self.v_manage = None
        # the un-changed part of load
        self.fix_load = 0
        # the load of crossover grps
        self.c_load = None
        # load of agent's groups
        self.load =None
        # average delay of agents and its controlled groups
        self.avg_delay = 0
        # Queue delay
        self.Q_delay = 0
        #tranmission delay
        self.Tdelay=0
        #processing delay
        self.ProDelay=0
        #length of the queue
        self.Qlength=200


# action of the agent
#this is for the probability of action
class Action(object):
    def __init__(self):
        # probaility of control
        self.p_ctl = None
        # action dimensionality
        self.dim_a = 0
        # real action of agent
        self.v_manage = None

# properties and state of physical world entity(vehicle)
#in the case of resource allocation in edge server
class Entity(object):
    def __init__(self):
        # vehicle ID for the entity
        self.id = 0
        # name of the entity
        self.name = ''
        # the entity vehicles are moving
        self.movable = True
        # speed of vehicle
        self.max_speed = None
        #accleration of vehicle
        self.accel = None
        # state of the entity(vehicle)
        self.state = EntityState()

# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()

#class for the vehicle entity
class Vehicle(Entity):
    def __init__(self):
        super(Vehicle, self).__init__()
        self.pos = None
        self.load = 0

# properties of agent entities
#this is the properties of edge server
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # location of agents
        self.pos = None
        # number of cross-over groups in the environment
        self.cross_over_num = None
        # the (x-y) group under control coverage
        self.group = []
        # coverage radius of the server
        self.r = int(4*size_map)
        # crossover group of control
        self.cross_group =[]
        #distance between vehicle and server
        self.distance =[]
        # service rate of agents is 10tasks/s
        self.service_rate = 10
        # the router connected
        self.router = 0
        # manage based on distance
        self.distance_manage = None

# properties of agent entities
class Agent_Local(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # location of agents
        self.pos = None
        # the (x-y) grpss under control coverage
        self.group = []
        # coverage radius
        self.r = int(0.5*size_map)
        # crossover group of control
        self.cross_group =[]
        self.distance =[]
        # service rate for thee dge server is 10 tasks/ms
        self.service_rate = 10
        # manage based on distance
        self.distance_manage = None

class Edgeserver(object):
    def __init__(self):
        super(Agent, self).__init__()
        self.pos = None
#Now considering multi agent environment and creaing the world for the
#environment
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.vehicles = []
        # communication channel dimensionality
        self.dim_c = 0
        #number of vehicles
        self.num_v = 50
        self.agent_num = 4
        self.agent_num_local = 2
        #creating a region with width and height 10*10
        self.region_W = int(size_map*10)
        self.region_H = int(size_map*10)

        self.centralized_Delay=0

        #all the group between all agents
        self.all_group = []
        # for each of SAs who in charge [region_W,region_H]
        self.group_management= None
        # all the cross over groups id
        self.cross_group = []
        # multi controlled group (agent->group ID)
        self.mul_group_c = {}
        # sending rate of the server 3*10^8m/s
        self.sending_rate = 300
        # all controled group
        self.all_con_group = np.zeros([self.region_W, self.region_H] )
        self.last_all_con_group=None

        # opti assingment based on progam latency
        self.delay_opt = 0
        self.delay_ma = np.zeros(self.agent_num)
        # the maddpg assignment (agents, vehicles) of all vehicles
        self.MA_v_c = None
        # last round of assignment
        self.MA_v_c_last = None
        # fix management
        self.Fix_v_c = None
        # main controller for the edger servers managing the veicles and servers
        self.v_c_possible = None

        # coverage of server
        self.edgeserve = 0.5
        #number of vehicle in each groups
        self.vehicle_num = None
        #getting edge server location using the dummy coordinates
        self.coordinates = None
        self.num_router = 0
        self.router_pos = None
        self.routing_path = None
        self.path_delay = None
        #small group delay
        self.group_delay=None


        self.load_group = np.zeros([self.region_W, self.region_H] )

        #        for atitude and longitude change
        self.offsize_lat = 22.9427
        self.offsize_lng = 43.3175
        self.zom = 10.256

        self.distance_assign_matrix = np.zeros([self.region_W, self.region_H] )

        self.agent_group_cover = np.zeros([self.agent_num, self.region_W, self.region_H] )

        self.delay_in_group_fix = None

    #function to get the coordinates for the rio bus dataset,
    #this is used to get the coordinates of the bus
    def set_coord(self, dic):
        coordinates = dic+'Rio_de_Janeiro_coordinates.matrix'
        #this gets the number of edge servers from the matrix
        self.coordinates = nx.Graph(np.loadtxt(coordinates))

        self.num_router = self.coordinates.number_of_nodes()

        po = []
        with open(dic+'Rio_de_Janeiro_coordinates1.txt', 'r') as f:
            for j in f.readlines():
                po.append( list(map(lambda x: float(x), j[:-1].split(' ')[1:3])))
        po_array = np.array(po)

        po_array[:,0] = (po_array[:,0]+self.offsize_lat)*self.zom*self.region_W
        po_array[:,1] =(po_array[:,1]+self.offsize_lng)*self.zom*self.region_H
        #setting of the position in the vehicles
        self.router_pos = po_array

    #fucntion to set the dimensions for the action
    def set_action_dim(self):
        for agent in self.agents:
            agent.action.dim_a = len(self.mul_group_c.get(agent.id)[0])


    # set edge server service rate of agents 1 tasks/ms
    def set_server_rate(self):
        for agent in self.agents:
            agent.service_rate= 10

    # managing of the croossover between the vehicles
    #between the edge servers
    def get_crossover(self):
        cross_group = []
        all_group = []
        self.cross_group_num = 0
        for x in range(self.region_H):
            for y in range(self.region_W):
                # the left and bottom point
                LB  = (x,y)
                _term = 0
                for agent in self.agents:
                    distance = np.linalg.norm(np.array(agent.pos) - LB)
                    if distance <= agent.r:
                        agent.group.append(LB)
                        agent.distance.append(distance)
                        _term += 1
                        if _term == 2:
                            cross_group.append(LB)
                            # all groups
                        if _term == 1:
                            all_group.append(LB)
        self.cross_group_num = len(cross_group)
        self.cross_a = cross_group
        self.all_group = all_group

        for agent in self.agents:
            listA = agent.group
            id_agent = list(set(listA).intersection(set(cross_group)))
            id_agent.sort()
            self.mul_group_c.setdefault(agent.id,[]).append(id_agent)
            agent.cross_group = id_agent.copy()
            if list(set(agent.cross_group).difference(set(self.cross_a))) != [] :
                print("error in agent cross groups")
                input()
            # set region fix group from 1:agent_num
            for value in agent.group:
                self.all_con_group[value[0],value[1]] = agent.id+1

    # multiple servers performing actions in the environments
    #getting the action perfromed by the each server
    def get_real_action(self, probality_action):

        agent_group_matrix = np.ones([self.agent_num, self.cross_group_num])*(-10)
        for i, agent in enumerate(self.agents):
            for value in self.mul_group_c[agent.id]:
                for k in range(len(value)):
                    p = self.cross_a.index(value[k])
                    if probality_action[i][k] in agent_group_matrix[:i,p]:
                        agent_group_matrix[i,p] = probality_action[i][k]+np.random.uniform(low=-0.1, high=0.1, size=1)
                        if agent_group_matrix[i,p] in agent_group_matrix[:i,p]:
                            agent_group_matrix[i,p] = probality_action[i][k]+np.random.uniform(low=0, high=0.1, size=1)
                    else:
                        agent_group_matrix[i,p] = probality_action[i][k]

        self.last_all_con_group = self.all_con_group.copy()
        agent_max = np.where(agent_group_matrix==np.max(agent_group_matrix, axis=0))
        te = np.array(agent_max)
        agent_max = te[:,te[1].argsort()][0]
        k =0
        # set group crossover between the number of agents present in the environment
        for value in te[1]:
            self.all_con_group[self.cross_a[value][0], self.cross_a[value][1]] = te[0][k]+1
            k+=1
        if len(agent_max) > len(self.cross_a):
            print(agent_max)
        #this is used to get the action for the steps performed
        return  agent_max


        # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents that are controllable by policy
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by the scripted world
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the envronment world upon agent action in state
    def step(self, action_n):
        # agents control the group (cross_group_region,1)
        real_action = self.get_real_action(action_n)
        if len(real_action) > len(self.cross_a):
            print("error")
            print(action_n)
            print(real_action)
        # update agent state
        for agent in self.agents:
            # global group id for the group of vehicles under the server
            group_id = np.argwhere(real_action==agent.id)
            # update agent.state.v_manage
            self.update_agent_state(agent, group_id)

    #this function update the state of the agent on the steps taken by the agent
    def update_agent_state(self, agent, real_action):
        new_v_manage = np.zeros(len(agent.state.v_manage))
        if real_action is not None and real_action.size>0:
            ra = np.hstack(real_action).tolist()
            if max(ra) >= len(self.cross_a):
                print("Action error")
            for i in ra:
                node = self.cross_a[i]
                te = agent.cross_group
                if node not in te:
                    print("Node error")
                index_ = te.index(node)
                new_v_manage[index_]=1
        #this is use to manage agaent state and action
        agent.state.v_manage = new_v_manage
        agent.action.v_manage = new_v_manage





