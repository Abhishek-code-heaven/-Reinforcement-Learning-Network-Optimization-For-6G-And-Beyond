from fullechoenvironment import ReinforcementNetworkEnvironment
from extrashortestpathenvironment import ShortestpathNetworkEnvironment
import math
import matplotlib.pyplot as plt
import numpy as np

class FullEchoAgent(object): #this is our agent

    def __init__(self, number_of_nodes, num_of_actions, distanceofnodemap, numberoflinks):

        self.config = {
            "learningrate" : 0.7,
            "epsilon": 0.1,
            "discountfactor": 1}
        self.Qvalues = np.zeros((number_of_nodes,number_of_nodes,num_of_actions))

        for currentnode in range(number_of_nodes):
            for destinationnode in range(number_of_nodes):
                for actionsavailable in range(numberoflinks[currentnode]):
                    self.Qvalues[currentnode][destinationnode][actionsavailable] = distanceofnodemap[currentnode][destinationnode]




    def performaction(self, state, numberoflinks,  bestaction=False):
        currentnode = state[0]
        destinationnode = state[1]

        if bestaction is True:
            bestaction = self.Qvalues[currentnode][destinationnode][0]
            takeaction = 0
            for action in range(numberoflinks[currentnode]):
                if self.Qvalues[currentnode][destinationnode][action] < bestaction:  #+ eps:
                    bestaction = self.Qvalues[currentnode][destinationnode][action]
                    takeaction = action
        else:
            takeaction = int(np.random.choice((0.0, numberoflinks[currentnode])))

        return takeaction


    def dolearn(self, present_event, next_event, reward, action, ifcomplete, nlinks):

        currentnode = present_event[0]

        destinationnode = present_event[1]
        nextnode = next_event[0]
        futurevals = self.Qvalues[nextnode][destinationnode][0]

        for link in range(nlinks[nextnode]):
            if self.Qvalues[nextnode][destinationnode][link] < futurevals:
                futurevals = self.Qvalues[nextnode][destinationnode][link]


        self.Qvalues[currentnode][destinationnode][action] = self.Qvalues[currentnode][destinationnode][action] + (reward + self.config["discountfactor"]*futurevals - self.Qvalues[currentnode][destinationnode][action])* self.config["learningrate"]

def main(): #this is the experiment iterations
    shortest = [0.0]
    fullechoqlerning = [0.0]
    rewardsum = []
    loadlevel = 1.0
    for i in range(20):
        loadlevel = loadlevel + 1.0

        env = ReinforcementNetworkEnvironment()
        env1 = ShortestpathNetworkEnvironment()
        outputpair = env.environmentreset()
        env1.environmentreset()
        env.loadlevel = loadlevel
        env1.loadlevel = loadlevel
        agent = FullEchoAgent(env.number_of_nodes, env.number_of_edges, env.distanceofnodemap, env.numberoflinks)
        ifcomplete = False
        cumulativereward = bestreward = 0


        for interation in range(10001):
            if not ifcomplete:
                current_state = outputpair[1]
                currentnode = current_state[0]
                destinationnode = current_state[1]
                short_act = env1.shortestpathofnodemap[currentnode][destinationnode]
                pairds, rewardsd, donesd, _ = env1.takeastep(short_act)
                for takenaction in range(env.numberoflinks[currentnode]):
                    reward, nexttransitionstate = env.fullecho(takenaction)
                    agent.dolearn(current_state, nexttransitionstate, reward, takenaction, ifcomplete, env.numberoflinks)

                takenaction  = agent.performaction(current_state, env.numberoflinks)
                outputpair, reward, ifcomplete, _ = env.takeastep(takenaction)

                nexttransitionstate = outputpair[0]
                agent.dolearn(current_state, nexttransitionstate, reward, takenaction, ifcomplete, env.numberoflinks)
                cumulativereward = cumulativereward + reward


                if interation%10000 == 0:
                    try:
                        print("Shortest path Algorithm Delivery time",float(env1.deliverytime)/float(env.deliveredpackets))
                        shortest.append(float(env1.deliverytime)/float(env.deliveredpackets))
                    except:
                        pass
                    if env.deliveredpackets != 0:
                        print("Current loadlevel = ",i)
                        print("total interations = ",interation)
                        print("Average Delivery Time taken = ",float(env.deliverytime)/float(env.deliveredpackets))
                        print("Span of average route taken = ",float(env.transitions)/float(env.deliveredpackets))
                        print("Cumulative Reward = ",cumulativereward)
                        print()
                        print()
                        rewardsum.append(int(cumulativereward))
                        try:
                            fullechoqlerning.append(float(env.deliverytime)/float(env.deliveredpackets))
                        except:
                            pass
                    current_state = outputpair[1]

                    curretnode = current_state[0]


                    for takenaction in range(env.numberoflinks[curretnode]):
                        reward, nexttransitionstate = env.fullecho(takenaction)
                        agent.dolearn(current_state, nexttransitionstate, reward, takenaction, ifcomplete, env.numberoflinks)

                    takenaction  = agent.performaction(current_state, env.numberoflinks, True)

                    outputpair, reward, ifcomplete, _ = env.takeastep(takenaction)

                    nexttransitionstate = outputpair[0]
                    agent.dolearn(current_state, nexttransitionstate, reward, takenaction, ifcomplete, env.numberoflinks)
                    bestreward = bestreward +  reward
                    if env.deliveredpackets != 0:
                        print("Current loadlevel = ", i)
                        print("total interations = ", interation)
                        print("Average Delivery Time taken = ", float(env.deliverytime) / float(env.deliveredpackets))
                        print("Span of average route taken = ", float(env.transitions) / float(env.deliveredpackets))
                        print("Best Reward = ", bestreward)
                        print()
                        print()

    y = np.array(fullechoqlerning)
    x = np.arange(0,len(y))
    z = np.array(rewardsum)
    indexer2 = np.arange(0, len(z))
    yy = np.array(shortest)
    xx = np.arange(0, len(yy))
    if len(x) > len(xx):
        indexer = x
    else:
        indexer = xx
    print(indexer)
    print(type(indexer))

    plt.plot(indexer, y, label="Full Echo Q-Learning")
    plt.plot(indexer, yy, label="Shortest Path")
    plt.title("Full Echo Q-Learning VS Shortest Path Packet Average delivery Time")
    plt.xlabel("Increasing Load level per learning cycle")
    plt.ylabel("Time")
    plt.legend(loc="upper left")
    new_list = range(math.floor(min(indexer)), math.ceil(max(indexer)) + 1)
    plt.xticks(new_list)
    plt.show()
    plt.plot(indexer2, z, label="Minimizing Reward")
    plt.title("Agent learning increase per learning cycle")
    plt.xlabel("Number of Iterations of learning cycle")
    plt.ylabel("Minimizing Time as Reward")
    plt.legend(loc="upper right")
    new_list = range(math.floor(min(indexer2)), math.ceil(max(indexer2)) + 1)
    plt.xticks(new_list)
    plt.show()

if __name__ == '__main__':
    main()
