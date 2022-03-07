import gym
import numpy as np
import heapq
import collections
import math
import random


nevents= 0




class ShortestpathNetworkEnvironment(gym.Env):


    def __init__(self):

        self.networktype = '36x36.txt'
        self.ifcomplete = False

        self.number_of_nodes = 0
        self.number_of_edges = 0
        self.insertedinqueue = {}
        self.nopacketinqueue = {}
        self.queuestimatedtime = []
        self.priority_Queue = []
        self.numberoflinks = {}
        self.linkmap = collections.defaultdict(dict)
        self.deliverytime = 0.0
        self.deliveredpackets = 0
        self.transitions = 0
        self.present_event = theevent(0.0, 0)
        self.timebetweennode = 1.0
        self.timebetweenqueues = 1.0
        self.livepackets = 0
        self.limitonthequeue = 100
        self.send_fail = 0
        self.loadlevel = 1

        self.distanceofnodemap = []
        self.shortestpathofnodemap =  []
        self.insertions = 0
        self.queuesum = 0

        self.nevents= 0

     


    def environmentreset(self):
        self.number_of_nodes = 0
        self.number_of_edges = 0

        graph_file = open(self.networktype, "r")


        for line in graph_file:
            line_contents = line.split()

            if line_contents[0] == 'declaringnode':

                self.numberoflinks[self.number_of_nodes] = 0

                self.number_of_nodes = self.number_of_nodes + 1


            if line_contents[0] == 'declaringlinks':

                node1 = int(line_contents[1])
                node2 = int(line_contents[2])
                self.linkmap[node1][self.numberoflinks[node1]] = node2
                self.numberoflinks[node1] = self.numberoflinks[node1] + 1
                self.linkmap[node2][self.numberoflinks[node2]] = node1
                self.numberoflinks[node2] = self.numberoflinks[node2] + 1
                self.number_of_edges = self.number_of_edges + 1


        self.distanceofnodemap = np.zeros((self.number_of_nodes,self.number_of_nodes))
        self.shortestpathofnodemap =  np.zeros((self.number_of_nodes,self.number_of_nodes))
        changing = True

        for i in range(self.number_of_nodes):
            for j in range(self.number_of_nodes):
                if i == j:
                    self.distanceofnodemap[i][j] = 0
                else:

                    self.distanceofnodemap[i][j] = self.number_of_nodes + 1


                self.shortestpathofnodemap[i][j] = -1


        while changing:
            changing = False
            for i in range(self.number_of_nodes):
                for j in range(self.number_of_nodes):


                    if i != j:
                        for k in range(self.numberoflinks[i]):

                            if self.distanceofnodemap[i][j] > 1 + self.distanceofnodemap[self.linkmap[i][k]][j]:
                                self.distanceofnodemap[i][j] = 1 + self.distanceofnodemap[self.linkmap[i][k]][j]
                                self.shortestpathofnodemap[i][j] = k
                                changing = True


        self.ifcomplete = False
        self.queuestimatedtime = [self.timebetweennode]*self.number_of_nodes

        self.priority_Queue =[]
        self.deliverytime= 0.0

        self.insertedinqueue = [0.0]*self.number_of_nodes
        self.nopacketinqueue = [0]*self.number_of_nodes


        insertevent = theevent(0.0, 0)

        insertevent.source = INSERT

        if self.loadlevel == 1.0:
            insertevent.estimatedtime = -math.log(random.random())
        else:
            insertevent.estimatedtime = -math.log(1- random.random())*float(self.loadlevel)


        self.nevents= 1

        insertevent.timeinqueues = 0.0
        heapq.heappush(self.priority_Queue,((1.0, -self.nevents), insertevent))

        self.insertions += 1
        self.nevents+= 1


        present_event = heapq.heappop(self.priority_Queue)[1]

        current_time = present_event.estimatedtime


        while present_event.source == INSERT:
            if self.loadlevel == 1.0 or self.loadlevel == 0.0:
                present_event.estimatedtime += -math.log(1 - random.random())
            else:
                present_event.estimatedtime += -math.log(1 - random.random()) * float(self.loadlevel)

            present_event.timeinqueues = current_time

            heapq.heappush(self.priority_Queue, ((current_time + 1.0, -self.nevents), present_event))
            self.nevents+= 1
            present_event = self.makepacket(current_time)
            if present_event == EMPTY:
                present_event = heapq.heappop(self.priority_Queue)[1]

        if present_event == EMPTY:
            present_event = heapq.heappop(self.priority_Queue)[1]

        self.present_event =  present_event



        return((self.present_event.node, self.present_event.dest), (self.present_event.node, self.present_event.dest))

    def takeastep(self, action):


        present_event = self.present_event
        current_time = present_event.estimatedtime
        current_node = present_event.node

        time_in_queue = current_time - present_event.timeinqueues - self.timebetweennode


        if action < 0 or action not in self.linkmap[current_node]:
            next_node = current_node

        else:
            next_node = self.linkmap[current_node][action]

        if next_node == present_event.dest:
            reward = time_in_queue + self.timebetweennode

            self.deliveredpackets +=  1
            self.nopacketinqueue[current_node] -= 1
            self.deliverytime +=  current_time - present_event.birth + self.timebetweennode
            self.transitions += present_event.hops + 1

            self.livepackets -= 1

            self.present_event = self.newpacketgenerator()

            if self.present_event == EMPTY:
                return ((present_event.node, present_event.dest), (present_event.node, present_event.dest)), reward, self.ifcomplete, {}
            else:
                return ((present_event.node, present_event.dest), (self.present_event.node, self.present_event.dest)), reward, self.ifcomplete, {}

        else:

            if self.nopacketinqueue[next_node] >= self.limitonthequeue:
                 self.send_fail = self.send_fail + 1
                 next_node = current_node

            reward =  time_in_queue + self.timebetweennode

            present_event.node = next_node
            present_event.hops += 1
            next_time = max(self.insertedinqueue[next_node]+self.queuestimatedtime[next_node], current_time + self.timebetweennode)
            present_event.estimatedtime = next_time
            self.insertedinqueue[next_node] = next_time

            present_event.timeinqueues = current_time
            if type(present_event) == int:
                print("this is present_event:{}".format(present_event))
            heapq.heappush(self.priority_Queue,((current_time+1.0, -self.nevents), present_event))
            self.nevents+= 1

            self.nopacketinqueue[next_node] += 1
            self.nopacketinqueue[current_node] -= 1


            self.present_event = self.newpacketgenerator()

            if self.present_event == EMPTY:
                return ((present_event.node, present_event.dest), (present_event.node, present_event.dest)), reward, self.ifcomplete, {}
            else:
                return ((present_event.node, present_event.dest), (self.present_event.node, self.present_event.dest)), reward, self.ifcomplete, {}













    def makepacket(self, time):
        source = np.random.random_integers(0,self.number_of_nodes-1)
        dest = np.random.random_integers(0,self.number_of_nodes-1)


        while source == dest:
            dest = np.random.random_integers(0,self.number_of_nodes-1)


        if self.nopacketinqueue[source] > self.limitonthequeue - 1:
             self.queuesum += 1
             return(EMPTY)

        self.nopacketinqueue[source] = self.nopacketinqueue[source] + 1

        self.livepackets = self.livepackets + 1
        present_event = theevent(time, dest)
        present_event.source = present_event.node = source

        return present_event

    def newpacketgenerator(self):

        present_event =  heapq.heappop(self.priority_Queue)[1]
        current_time = present_event.estimatedtime


        while present_event.source == INSERT :
             if self.loadlevel == 1.0 or self.loadlevel == 0.0:
                 present_event.estimatedtime += -math.log(1 - random.random())
             else:
                 present_event.estimatedtime += -math.log(1- random.random())*float(self.loadlevel)

             present_event.timeinqueues = current_time

             heapq.heappush(self.priority_Queue,((current_time+1.0, -self.nevents), present_event))
             self.nevents+= 1
             present_event = self.makepacket(current_time)
             if present_event == EMPTY :
                 present_event =  heapq.heappop(self.priority_Queue)[1]


        if present_event == EMPTY :
            present_event =  heapq.heappop(self.priority_Queue)[1]
        return present_event




    def compute_best(self):

        changing = True

        for i in range(self.number_of_nodes):
            for j in  range(self.number_of_nodes):
                if i == j:
                    self.distanceofnodemap[i][j] = 0
                else:
                    self.distanceofnodemap[i][j] = self.number_of_nodes+1
                self.shortestpathofnodemap[i][j] = -1

        while changing:
            changing = False
            for i in range(self.number_of_nodes):
                for j in  range(self.number_of_nodes):

                    if i != j:
                      for k in range(self.numberoflinks[i]):
                        if  self.distanceofnodemap[i][j] >  1 + self.distanceofnodemap[self.linkmap[i][k]][j]:
                          self.distanceofnodemap[i][j] = 1 + self.distanceofnodemap[self.linkmap[i][k]][j]
                          self.shortestpathofnodemap[i][j] = k
                          changing = True

class theevent:
    def __init__(self, time,  dest):

        self.dest = dest
        self.source = NEWVALS
        self.node = NEWVALS
        self.birth = time
        self.hops = 0
        self.estimatedtime = time
        self.timeinqueues = time


INSERT = -1
NEWVALS = -4


EMPTY = EMPTY =  -1