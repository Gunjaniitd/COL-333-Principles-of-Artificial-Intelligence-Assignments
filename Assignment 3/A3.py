import math
import numpy as np
from numpy import random
from matplotlib import pyplot as plt
import sys


class node:
    def __init__(self, walls, special):
        self.walls = set()
        for i in walls:
            self.walls.add(i)
        self.spec = special

###########################################
################ PART A ###################
###########################################

grid = [
	[node(["D", "L", "R"], "Y"), node(["L", "R"], ""), node(["L"], ""), node(["L"], ""), node(["U", "L"], "R")],
	[node(["D", "L"], ""), node(["L"], ""), node([], ""), node(["R"], ""), node(["U", "R"], "")],
	[node(["D", "R"], ""), node(["R"], ""), node([], ""), node(["L"], ""), node(["U", "L"], "")],
	[node(["D", "L"], "B"), node(["L"], ""), node([], ""), node([], ""), node(["U"], "")],
	[node(["D", "R"], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node(["U", "R"], "G")]
]

start = (0, 0)
end = (4, 4)
taxiLocation = (0, 4)

def getNextStates(i, j, action):
	ans = []
	state = grid[i][j]
	walls = state.walls

	if action == "u":

		if "U" in walls:
			ans.append([(i, j), 0.85])
		else:
			ans.append([(i, j + 1), 0.85])

		if "D" in walls:
			ans.append([(i, j), 0.05])
		else:
			ans.append([(i, j - 1), 0.05])

		if "L" in walls:
			ans.append([(i, j), 0.05])
		else:
			ans.append([(i - 1, j), 0.05])

		if "R" in walls:
			ans.append([(i, j), 0.05])
		else:
			ans.append([(i + 1, j), 0.05])

	if action == "d":

		if "U" in walls:
			ans.append([(i, j), 0.05])
		else:
			ans.append([(i, j + 1), 0.05])

		if "D" in walls:
			ans.append([(i, j), 0.85])
		else:
			ans.append([(i, j - 1), 0.85])

		if "L" in walls:
			ans.append([(i, j), 0.05])
		else:
			ans.append([(i - 1, j), 0.05])

		if "R" in walls:
			ans.append([(i, j), 0.05])
		else:
			ans.append([(i + 1, j), 0.05])

	if action == "l":

		if "U" in walls:
			ans.append([(i, j), 0.05])
		else:
			ans.append([(i, j + 1), 0.05])

		if "D" in walls:
			ans.append([(i, j), 0.05])
		else:
			ans.append([(i, j - 1), 0.05])

		if "L" in walls:
			ans.append([(i, j), 0.85])
		else:
			ans.append([(i - 1, j), 0.85])

		if "R" in walls:
			ans.append([(i, j), 0.05])
		else:
			ans.append([(i + 1, j), 0.05])

	if action == "r":

		if "U" in walls:
			ans.append([(i, j), 0.05])
		else:
			ans.append([(i, j + 1), 0.05])

		if "D" in walls:
			ans.append([(i, j), 0.05])
		else:
			ans.append([(i, j - 1), 0.05])

		if "L" in walls:
			ans.append([(i, j), 0.05])
		else:
			ans.append([(i - 1, j), 0.05])

		if "R" in walls:
			ans.append([(i, j), 0.85])
		else:
			ans.append([(i + 1, j), 0.85])

	return ans

def findQ(table, i, j, action, onboarded, discount):
	if action == "pickdown" and onboarded == False:
		return -10

	if action == "pickup" and onboarded == False:
		if (i, j) == start:
			return -1
		else:
			return -10

	if action == "pickdown" and onboarded == True:
		if (i, j) == end:
			return 20
		else:
			return -10

	if action == "pickup" and onboarded == True:
		return -10

	nextStates = getNextStates(i, j, action)

	q = 0
	for (m, n), p in nextStates:
		q += p * ( -1 + discount * table[m][n] )
	return q

def valueIteration(epsilon, discount):
	table1_old = [[0 for i in range(5)] for j in range(5)]
	table2_old = [[0 for i in range(5)] for j in range(5)]
	table1_new = [[0 for i in range(5)] for j in range(5)]
	table2_new = [[0 for i in range(5)] for j in range(5)]

	policy1 = [[0 for i in range(5)] for j in range(5)]
	policy2 = [[0 for i in range(5)] for j in range(5)]

	actions = ["pickup", "pickdown", "u", "d", "l", "r"]

	change = 0
	iteration = 0

	x, y = [], []

	while (iteration == 0 or change > epsilon):
		iteration += 1
		change = 0

		for i in range(5):
			for j in range(5):

				Max = float("-inf")
				bestAction = ""

				for action in actions:
					q = findQ(table1_old, i, j, action, False, discount)
					if Max < q:
						Max = q
						bestAction = action

				change = max(change, abs(Max - table1_old[i][j]))
				table1_new[i][j] = Max
				policy1[i][j] = bestAction

		for i in range(5):
			for j in range(5):
				
				Max = float("-inf")
				bestAction = ""

				for action in actions:
					q = findQ(table2_old, i, j, action, True, discount)
					if Max < q:
						Max = q
						bestAction = action

				change = max(change, abs(Max - table2_old[i][j]))
				table2_new[i][j] = Max
				policy2[i][j] = bestAction

		table1_old = table1_new
		table2_old = table2_new

		x.append(iteration)
		y.append(change)

	# print(table1_new)
	# print(table2_new)

	print(policy1)
	print(policy2)

	print("No of iterations required for convergence: " + str(iteration))

	return (x, y, policy1, policy2)


def findBestAction(i, j , stateUtilities, onboarded, discount):
	if onboarded == False and (i, j) == start:
		return "pickup"

	if onboarded == True and (i, j) == end:
		return "pickdown"

	bestAction = ""
	Max = float("-inf")

	actions = ["u", "d", "l", "r"]

	for action in actions:
		nextStates = getNextStates(i, j, action)

		temp = 0
		for (m,n), p in nextStates:
			temp += p * (-1 + discount * stateUtilities[m][n])

		if temp > (Max + 0.00001):
			Max = temp
			bestAction = action

	return bestAction

def policyEvaluation(policy, onboarded, discount, epsilon):
	stateUtilities = [[0 for i in range(5)] for j in range(5)]
	# print(discount, epsilon)
	change = 0
	iteration = 0

	while (iteration == 0 or change > epsilon):
		iteration += 1
		change = 0

		for i in range(5):
			for j in range(5):

				action = policy[i][j]
				prev = stateUtilities[i][j]

				if action == "pickdown" and onboarded == False:
					stateUtilities[i][j] = -10

				if action == "pickup" and onboarded == False:
					if (i, j) == start:
						stateUtilities[i][j] = -1
					else:
						stateUtilities[i][j] = -10

				if action == "pickdown" and onboarded == True:
					if (i, j) == end:
						# print("yes")
						stateUtilities[i][j] = 20
						# print(stateUtilities)
					else:
						stateUtilities[i][j] = -10

				if action == "pickup" and onboarded == True:
					stateUtilities[i][j] = -10

				change = max(change, abs(stateUtilities[i][j] - prev))

				if not (action == "pickup") and not (action == "pickdown"):
					nextStates = getNextStates(i, j, action)

					temp = 0
					for (m, n), p in nextStates:
						temp += p * ( -1 + discount * stateUtilities[m][n] )
					stateUtilities[i][j] = temp

					change = max(change, abs(stateUtilities[i][j] - prev))

	return stateUtilities

def policyIteration(epsilon, discount):
	policy1 = [["u" for i in range(5)] for j in range(5)]
	policy2 = [["u" for i in range(5)] for j in range(5)]

	iteration = 0

	while True:
		iteration += 1

		stateUtilities1 = policyEvaluation(policy1, False, discount, epsilon)
		stateUtilities2 = policyEvaluation(policy2, True, discount, epsilon)
		# print("yes")

		newPolicy1 = [[0 for i in range(5)] for j in range(5)]
		newPolicy2 = [[0 for i in range(5)] for j in range(5)]

		for i in range(5):
			for j in range(5):	
				newPolicy1[i][j] = findBestAction(i, j, stateUtilities1, False, discount)
				newPolicy2[i][j] = findBestAction(i, j, stateUtilities2, True, discount)

		if policy1 == newPolicy1 and policy2 == newPolicy2:
			policy1 = newPolicy1
			policy2 = newPolicy2
			break

		# print(policy1)
		# print(newPolicy1)

		policy1 = newPolicy1
		policy2 = newPolicy2

	# print(stateUtilities1)
	# print(stateUtilities2)
	print(policy1)
	print(policy2)

	print("No of iterations required for convergence: " + str(iteration))

	return policy1, policy2

def findLoss(epsilon, discount, bestPolicy1, bestPolicy2):
	policy1 = [["u" for i in range(5)] for j in range(5)]
	policy2 = [["u" for i in range(5)] for j in range(5)]

	iteration = 0

	x, y = [], []

	bestStateUtilities1 = policyEvaluation(bestPolicy1, False, discount, epsilon)
	bestStateUtilities2 = policyEvaluation(bestPolicy2, True, discount, epsilon)

	while True:
		iteration += 1

		stateUtilities1 = policyEvaluation(policy1, False, discount, epsilon)
		stateUtilities2 = policyEvaluation(policy2, True, discount, epsilon)

		Max = float("-inf")
		for i in range(5):
			for j in range(5):
				Max = max(Max, max(bestStateUtilities1[i][j] - stateUtilities1[i][j], bestStateUtilities2[i][j] - stateUtilities2[i][j]))

		x.append(iteration)
		y.append(Max)

		newPolicy1 = [[0 for i in range(5)] for j in range(5)]
		newPolicy2 = [[0 for i in range(5)] for j in range(5)]

		for i in range(5):
			for j in range(5):	
				newPolicy1[i][j] = findBestAction(i, j, stateUtilities1, False, discount)
				newPolicy2[i][j] = findBestAction(i, j, stateUtilities2, True, discount)

		if policy1 == newPolicy1 and policy2 == newPolicy2:
			policy1 = newPolicy1
			policy2 = newPolicy2
			break

		policy1 = newPolicy1
		policy2 = newPolicy2

	return x, y

def getNextStatesWithAction(i, j, action):
	ans = []
	state = grid[i][j]
	walls = state.walls

	if action == "u":

		if "U" in walls:
			ans.append([(i, j), 0.85, "u"])
		else:
			ans.append([(i, j + 1), 0.85, "u"])

		if "D" in walls:
			ans.append([(i, j), 0.05, "d"])
		else:
			ans.append([(i, j - 1), 0.05, "d"])

		if "L" in walls:
			ans.append([(i, j), 0.05, "l"])
		else:
			ans.append([(i - 1, j), 0.05, "l"])

		if "R" in walls:
			ans.append([(i, j), 0.05, "r"])
		else:
			ans.append([(i + 1, j), 0.05, "r"])

	if action == "d":

		if "D" in walls:
			ans.append([(i, j), 0.85, "d"])
		else:
			ans.append([(i, j - 1), 0.85, "d"])

		if "U" in walls:
			ans.append([(i, j), 0.05, "u"])
		else:
			ans.append([(i, j + 1), 0.05, "u"])

		if "L" in walls:
			ans.append([(i, j), 0.05, "l"])
		else:
			ans.append([(i - 1, j), 0.05, "l"])

		if "R" in walls:
			ans.append([(i, j), 0.05, "r"])
		else:
			ans.append([(i + 1, j), 0.05, "r"])

	if action == "l":

		if "L" in walls:
			ans.append([(i, j), 0.85, "l"])
		else:
			ans.append([(i - 1, j), 0.85, "l"])

		if "U" in walls:
			ans.append([(i, j), 0.05, "u"])
		else:
			ans.append([(i, j + 1), 0.05, "u"])

		if "D" in walls:
			ans.append([(i, j), 0.05, "d"])
		else:
			ans.append([(i, j - 1), 0.05, "d"])

		if "R" in walls:
			ans.append([(i, j), 0.05, "r"])
		else:
			ans.append([(i + 1, j), 0.05, "r"])

	if action == "r":

		if "R" in walls:
			ans.append([(i, j), 0.85, "r"])
		else:
			ans.append([(i + 1, j), 0.85, "r"])

		if "U" in walls:
			ans.append([(i, j), 0.05, "u"])
		else:
			ans.append([(i, j + 1), 0.05, "u"])

		if "D" in walls:
			ans.append([(i, j), 0.05, "d"])
		else:
			ans.append([(i, j - 1), 0.05, "d"])

		if "L" in walls:
			ans.append([(i, j), 0.05, "l"])
		else:
			ans.append([(i - 1, j), 0.05, "l"])

	return ans

def simulate(discount):
	x, y, p1, p2 = valueIteration(0.01, discount)

	curr = taxiLocation
	count = 0
	while (curr != start):
		i = curr[0]
		j = curr[1] 
		action = p1[i][j]
		nextStates = getNextStatesWithAction(i, j, action)

		rnum = random.rand()

		if rnum <= 0.85:
			print(i, j, nextStates[0][2])
			curr = nextStates[0][0]
		elif rnum <= 0.90:
			print(i, j, nextStates[1][2])
			curr = nextStates[1][0]
		elif rnum <= 0.95:
			print(i, j, nextStates[2][2])
			curr = nextStates[2][0]
		else:
			print(i, j, nextStates[3][2])
			curr = nextStates[3][0]

		count += 1
		if (count == 20):
			return

	print(curr[0], curr[1], "pickup")

	while (curr != end):
		i = curr[0]
		j = curr[1] 
		action = p2[i][j]
		nextStates = getNextStatesWithAction(i, j, action)

		rnum = random.rand()

		if rnum <= 0.85:
			print(i, j, nextStates[0][2])
			curr = nextStates[0][0]
		elif rnum <= 0.90:
			print(i, j, nextStates[1][2])
			curr = nextStates[1][0]
		elif rnum <= 0.95:
			print(i, j, nextStates[2][2])
			curr = nextStates[2][0]
		else:
			print(i, j, nextStates[3][2])
			curr = nextStates[3][0]

		count += 1
		if (count == 20):
			return

	print(curr[0], curr[1], "pickdown")




###########################################
################ PARTB ###################
###########################################



class QLearning:
    def __init__(self, start, finish, taxiLoc, gamma=0.99, alpha=0.25, epsilon=0.1):
        self.start = start
        self.finish = finish
        self.taxiLoc = taxiLoc
        self.passLoc = start
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.onTaxi = False
        self.over = False

        #Q Matrix
        self.QVal = [[[ 0 for i in range(6)] for j in range(25)] for k in range(29) ]
        #Policy
        self.policy = [[0 for i in range(25)] for j in range(29)]

        self.grid = [
                    [node(["D", "L", "R"], "Y"), node(["L", "R"], ""), node(["L"], ""), node(["L"], ""), node(["U", "L"], "R")],
                    [node(["D", "L"], ""), node(["L"], ""), node([], ""), node(["R"], ""), node(["U", "R"], "")],
                    [node(["D", "R"], ""), node(["R"], ""), node([], ""), node(["L"], ""), node(["U", "L"], "")],
                    [node(["D", "L"], "B"), node(["L"], ""), node([], ""), node([], ""), node(["U"], "")],
                    [node(["D", "R"], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node(["U", "R"], "G")]
                    ]
        
        self.depots = {0:0 , 3:1, 20:2, 24:3}

    def getNext(self, action):
        currState = self.taxiLoc
        reward = 0
        if(action < 4):
            X = currState%5
            Y = currState//5
            Node = self.grid[X][Y]
            #u:0 ; d:1; l:2 ; r:3 ; pUp:4 ; pDn:5 
            if(action == 0):
                if("U" in Node.walls):
                    return currState, -1
                else:
                    return currState+5, -1
            elif(action == 1):
                if("D" in Node.walls):
                    return currState, -1
                else:
                    return currState-5, -1
            elif(action == 2):
                if("L" in Node.walls):
                    return currState, -1
                else:
                    return currState-1, -1
            elif(action == 3):
                if("R" in Node.walls):
                    return currState, -1
                else:
                    return currState+1, -1
            
        else:
            if(action == 4):
                if(self.onTaxi):
                    reward = -10
                else:
                    if(self.taxiLoc == self.passLoc):
                        reward = -1
                        self.onTaxi = True
                    else:
                        reward = -10
            elif(action == 5):
                if(self.onTaxi):
                    if(self.taxiLoc == self.finish):
                        reward = 20
                        self.over = True
                        self.onTaxi = False
                    else:
                        reward = -10
                        self.passLoc = self.taxiLoc
                        self.onTaxi = False
                else:
                    reward = -10
            return currState, reward
                

    def QL(self, QVal, policy):
        self.QVal = QVal
        self.policy = policy
        if(self.over):
            return
        passLoc = self.passLoc
        onTaxi = self.onTaxi
        state = self.start
        count = 0
        while(count < 500 and not self.over):
            count += 1
            Rval = np.random.random()
            if(Rval > self.epsilon):    #exploitation using policy
                desiredAction = self.policy[state][self.taxiLoc]
            else:
                desiredAction = np.random.randint(0,6)
                #u:0 ; d:1; l:2 ; r:3 ; pUp:4 ; pDn:5 
            if(desiredAction > 3):
                action = desiredAction
            else:
                #stochastic
                randVal = np.random.random()
                if(randVal < 0.8):
                    action = desiredAction
                else:
                    action = np.random.randint(0,4)

            nextTaxiLoc, reward = self.getNext(action)
            nextState = state
            if(state > 24 and self.onTaxi == False):
                nextState = self.passLoc
            if(state <= 24 and self.onTaxi):
                # self.depots = {0:0 , 3:1, 20:2, 24:3}
                nextState = 25 + self.depots[self.finish]
            
            #Update QVal and policy
            
            old_qVal = self.QVal[state][self.taxiLoc][action]
            nextBest_qVal = self.QVal[nextState][nextTaxiLoc][self.policy[nextState][nextTaxiLoc]]

            if(reward < 20):
                self.QVal[state][self.taxiLoc][action] = (1-self.alpha)*old_qVal + self.alpha*(reward + self.gamma*nextBest_qVal)
            else:
                self.QVal[state][self.taxiLoc][action] = (1-self.alpha)*old_qVal + self.alpha*(reward)
            
            curr_qVal = self.QVal[state][self.taxiLoc][action]
            oldBest_qVal = self.QVal[state][self.taxiLoc][self.policy[state][self.taxiLoc]]

            if(oldBest_qVal < curr_qVal):
                self.policy[state][self.taxiLoc] = action
            
            state = nextState
            self.taxiLoc = nextTaxiLoc
        
        return self.QVal, self.policy

    def QL_decay(self, QVal, policy):
        self.QVal = QVal
        self.policy = policy
        if(self.over):
            return
        passLoc = self.passLoc
        onTaxi = self.onTaxi
        state = self.start
        count = 0
        while(count < 500 and not self.over):
            count += 1
            Rval = np.random.random()
            if(Rval > self.epsilon):    #exploitation using policy
                desiredAction = self.policy[state][self.taxiLoc]
            else:
                desiredAction = np.random.randint(0,6)
                #u:0 ; d:1; l:2 ; r:3 ; pUp:4 ; pDn:5 
            if(desiredAction > 3):
                action = desiredAction
            else:
                #stochastic
                randVal = np.random.random()
                if(randVal < 0.8):
                    action = desiredAction
                else:
                    action = np.random.randint(0,4)

            nextTaxiLoc, reward = self.getNext(action)
            nextState = state
            if(state > 24 and self.onTaxi == False):
                nextState = self.passLoc
            if(state <= 24 and self.onTaxi):
                # self.depots = {0:0 , 3:1, 20:2, 24:3}
                nextState = 25 + self.depots[self.finish]
            
            #Update QVal and policy
            
            old_qVal = self.QVal[state][self.taxiLoc][action]
            nextBest_qVal = self.QVal[nextState][nextTaxiLoc][self.policy[nextState][nextTaxiLoc]]

            if(reward < 20):
                self.QVal[state][self.taxiLoc][action] = (1-self.alpha)*old_qVal + self.alpha*(reward + self.gamma*nextBest_qVal)
            else:
                self.QVal[state][self.taxiLoc][action] = (1-self.alpha)*old_qVal + self.alpha*(reward)
            
            curr_qVal = self.QVal[state][self.taxiLoc][action]
            oldBest_qVal = self.QVal[state][self.taxiLoc][self.policy[state][self.taxiLoc]]

            if(oldBest_qVal < curr_qVal):
                self.policy[state][self.taxiLoc] = action
            
            state = nextState
            self.taxiLoc = nextTaxiLoc
            self.epsilon = self.epsilon/(count+1)
        
        return self.QVal, self.policy

    def QL_SARSA(self, QVal, policy):
        self.QVal = QVal
        self.policy = policy
        if(self.over):
            return
        passLoc = self.passLoc
        onTaxi = self.onTaxi
        state = self.start
        prevState = None
        prevTaxiLoc = None
        prevAction = None
        prevReward = 0
        count = 0
        while(count < 500 and not self.over):
            count += 1
            Rval = np.random.random()
            if(Rval > self.epsilon):    #exploitation using policy
                desiredAction = self.policy[state][self.taxiLoc]
            else:
                desiredAction = np.random.randint(0,6)
                #u:0 ; d:1; l:2 ; r:3 ; pUp:4 ; pDn:5 
            if(desiredAction > 3):
                action = desiredAction
            else:
                #stochastic
                randVal = np.random.random()
                if(randVal < 0.8):
                    action = desiredAction
                else:
                    action = np.random.randint(0,4)

            nextTaxiLoc, reward = self.getNext(action)
            

            
            nextState = state
            if(state > 24 and self.onTaxi == False):
                nextState = self.passLoc
            if(state <= 24 and self.onTaxi):
                # self.depots = {0:0 , 3:1, 20:2, 24:3}
                nextState = 25 + self.depots[self.finish]
            
            #Update QVal and policy
            if(prevState != None):
                old_qVal = self.QVal[prevState][prevTaxiLoc][prevAction]
                nextBest_qVal = self.QVal[state][self.taxiLoc][action]

                if(prevReward < 20):
                    self.QVal[prevState][prevTaxiLoc][prevAction] = (1-self.alpha)*old_qVal + self.alpha*(prevReward + self.gamma*nextBest_qVal)
                else:
                    self.QVal[prevState][prevTaxiLoc][prevAction] = (1-self.alpha)*old_qVal + self.alpha*(prevReward)
                
                curr_qVal = self.QVal[prevState][prevTaxiLoc][prevAction]
                oldBest_qVal = self.QVal[prevState][prevTaxiLoc][self.policy[prevState][prevTaxiLoc]]

                if(oldBest_qVal < curr_qVal):
                    self.policy[prevState][prevTaxiLoc] = prevAction
               
            prevState = state
            prevTaxiLoc = self.taxiLoc
            prevAction = action
            prevReward = reward
            state = nextState
            self.taxiLoc = nextTaxiLoc
                
        if(self.over):
            old_qVal = self.QVal[prevState][prevTaxiLoc][prevAction]
            bestAction = self.policy[state][self.taxiLoc]
            nextBest_qVal = self.QVal[state][self.taxiLoc][bestAction]

            if(prevReward < 20):
                self.QVal[prevState][prevTaxiLoc][prevAction] = (1-self.alpha)*old_qVal + self.alpha*(prevReward + self.gamma*nextBest_qVal)
            else:
                self.QVal[prevState][prevTaxiLoc][prevAction] = (1-self.alpha)*old_qVal + self.alpha*(prevReward)
            
            curr_qVal = self.QVal[prevState][prevTaxiLoc][prevAction]
            oldBest_qVal = self.QVal[prevState][prevTaxiLoc][self.policy[prevState][prevTaxiLoc]]

            if(oldBest_qVal < curr_qVal):
                self.policy[prevState][prevTaxiLoc] = prevAction
        
        return self.QVal, self.policy

    def QL_SARSA_decay(self, QVal, policy):
        self.QVal = QVal
        self.policy = policy
        if(self.over):
            return
        passLoc = self.passLoc
        onTaxi = self.onTaxi
        state = self.start
        prevState = None
        prevTaxiLoc = None
        prevAction = None
        prevReward = 0
        count = 0
        while(count < 500 and not self.over):
            count += 1
            Rval = np.random.random()
            if(Rval > self.epsilon):    #exploitation using policy
                desiredAction = self.policy[state][self.taxiLoc]
            else:
                desiredAction = np.random.randint(0,6)
                #u:0 ; d:1; l:2 ; r:3 ; pUp:4 ; pDn:5 
            if(desiredAction > 3):
                action = desiredAction
            else:
                #stochastic
                randVal = np.random.random()
                if(randVal < 0.8):
                    action = desiredAction
                else:
                    action = np.random.randint(0,4)

            nextTaxiLoc, reward = self.getNext(action)
            

            
            nextState = state
            if(state > 24 and self.onTaxi == False):
                nextState = self.passLoc
            if(state <= 24 and self.onTaxi):
                # self.depots = {0:0 , 3:1, 20:2, 24:3}
                nextState = 25 + self.depots[self.finish]
            self.epsilon = self.epsilon/(count+1)
            #Update QVal and policy
            if(prevState != None):
                old_qVal = self.QVal[prevState][prevTaxiLoc][prevAction]
                nextBest_qVal = self.QVal[state][self.taxiLoc][action]

                if(prevReward < 20):
                    self.QVal[prevState][prevTaxiLoc][prevAction] = (1-self.alpha)*old_qVal + self.alpha*(prevReward + self.gamma*nextBest_qVal)
                else:
                    self.QVal[prevState][prevTaxiLoc][prevAction] = (1-self.alpha)*old_qVal + self.alpha*(prevReward)
                
                curr_qVal = self.QVal[prevState][prevTaxiLoc][prevAction]
                oldBest_qVal = self.QVal[prevState][prevTaxiLoc][self.policy[prevState][prevTaxiLoc]]

                if(oldBest_qVal < curr_qVal):
                    self.policy[prevState][prevTaxiLoc] = prevAction
               
            prevState = state
            prevTaxiLoc = self.taxiLoc
            prevAction = action
            prevReward = reward
            state = nextState
            self.taxiLoc = nextTaxiLoc
                
        if(self.over):
            old_qVal = self.QVal[prevState][prevTaxiLoc][prevAction]
            bestAction = self.policy[state][self.taxiLoc]
            nextBest_qVal = self.QVal[state][self.taxiLoc][bestAction]

            if(prevReward < 20):
                self.QVal[prevState][prevTaxiLoc][prevAction] = (1-self.alpha)*old_qVal + self.alpha*(prevReward + self.gamma*nextBest_qVal)
            else:
                self.QVal[prevState][prevTaxiLoc][prevAction] = (1-self.alpha)*old_qVal + self.alpha*(prevReward)
            
            curr_qVal = self.QVal[prevState][prevTaxiLoc][prevAction]
            oldBest_qVal = self.QVal[prevState][prevTaxiLoc][self.policy[prevState][prevTaxiLoc]]

            if(oldBest_qVal < curr_qVal):
                self.policy[prevState][prevTaxiLoc] = prevAction
        
        return self.QVal, self.policy
    
    
    
def QIteration(episode, algorithm):
    #Q Matrix
    QVal = [[[ 0 for i in range(6)] for j in range(25)] for k in range(29) ]
    #Policy
    policy = [[0 for i in range(25)] for j in range(29)]
    for i in range(episode):

        finishLoc = np.random.randint(0,4)
        reverseDepot = [0, 3, 20, 24]
        taxi = np.random.randint(0,25)
        passL = reverseDepot[finishLoc]
        while(passL == reverseDepot[finishLoc]):
            passL = np.random.randint(0,25)
        
        taxiDomain = QLearning(passL, reverseDepot[finishLoc], taxi, 0.99, 0.25, 0.1)
        if(algorithm == "QL"):
            QVal, policy = taxiDomain.QL(QVal, policy)
        elif(algorithm == "QL_decay"):
            QVal, policy = taxiDomain.QL_decay(QVal, policy)
        elif(algorithm == "QL_SARSA"):
            QVal, policy = taxiDomain.QL_SARSA(QVal, policy)
        else:
            QVal, policy = taxiDomain.QL_SARSA_decay(QVal, policy)
    
    print("first: ", taxiDomain.policy[-4])
    print("secod: ", taxiDomain.policy[-3])
    print("third: ", taxiDomain.policy[-2])
    print("fourth: ", taxiDomain.policy[-1])

    return QVal, policy

# QIteration(20000)

##############################################
    ############TESTING################
##############################################


def plotB3(rewardTest, episode):
    X_test = [1, 2, 3, 4, 5]
    plt.plot(X_test, rewardTest, marker = 'o')
    plt.ylabel("Reward")
    plt.xlabel("Episode number")
    plt.title("QLearning")
    plt.savefig("part_B3_" + str(episode) + ".png")
    plt.clf

    return 

def plotB4(TotalList, episode, paramList, isEpsilon):
    X_test = []
    for i in range(episode):
        if(i%200 == 0):
            X_test.append(i)
    for rewardTest in TotalList:
        plt.plot(X_test, rewardTest)
    plt.legend(paramList)
    plt.ylabel("Avg Reward")
    plt.xlabel("No. of Episodes")
    
    if(isEpsilon):
        plt.title("QLearning_epsilon")
        plt.savefig("partB4_epsilon.png")
    else:
        plt.title("QLearning_alpha")
        plt.savefig("partB4_alpha.png")
    plt.clf

    return 

def plot(rewardTest, episode, algorithm):
    print("Last_Reward: ", rewardTest[-1])
    X_test = []
    for i in range(episode):
        if(i%200 == 0):
            X_test.append(i)
    plt.plot(X_test, rewardTest)
    plt.ylabel("Avg Reward")
    plt.xlabel("No. of Episodes")
    plt.title("QLearning_" + algorithm)
    plt.savefig("QLearning_"+ algorithm +".png")
    plt.clf

    return 

def getNextTest(taxiLoc, action, onTaxi, passLoc, finish, over):
        grid = [
                [node(["D", "L", "R"], "Y"), node(["L", "R"], ""), node(["L"], ""), node(["L"], ""), node(["U", "L"], "R")],
                [node(["D", "L"], ""), node(["L"], ""), node([], ""), node(["R"], ""), node(["U", "R"], "")],
                [node(["D", "R"], ""), node(["R"], ""), node([], ""), node(["L"], ""), node(["U", "L"], "")],
                [node(["D", "L"], "B"), node(["L"], ""), node([], ""), node([], ""), node(["U"], "")],
                [node(["D", "R"], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node(["U", "R"], "G")]
                ]
        currState = taxiLoc
        reward = 0
        if(action < 4):
            X = currState%5
            Y = currState//5
            Node = grid[X][Y]
            #u:0 ; d:1; l:2 ; r:3 ; pUp:4 ; pDn:5 
            if(action == 0):
                if("U" in Node.walls):
                    return currState, -1, passLoc, onTaxi, over
                else:
                    return currState+5, -1, passLoc, onTaxi, over
            elif(action == 1):
                if("D" in Node.walls):
                    return currState, -1, passLoc, onTaxi, over
                else:
                    return currState-5, -1, passLoc, onTaxi, over
            elif(action == 2):
                if("L" in Node.walls):
                    return currState, -1, passLoc, onTaxi, over
                else:
                    return currState-1, -1, passLoc, onTaxi, over
            elif(action == 3):
                if("R" in Node.walls):
                    return currState, -1, passLoc, onTaxi, over
                else:
                    return currState+1, -1, passLoc, onTaxi, over
            
        else:
            if(action == 4):
                if(onTaxi):
                    reward = -10
                else:
                    if(taxiLoc == passLoc):
                        reward = -1
                        onTaxi = True
                    else:
                        reward = -10
            elif(action == 5):
                if(onTaxi):
                    if(taxiLoc == finish):
                        reward = 20
                        over = True
                        onTaxi = False
                    else:
                        reward = -10
                        passLoc = taxiLoc
                        onTaxi = False
                else:
                    reward = -10
            return currState, reward, passLoc, onTaxi, over

def QIterationTest(episode, algorithm):
    #Q Matrix
    QVal = [[[ 0 for i in range(6)] for j in range(25)] for k in range(29) ]
    #Policy
    policy = [[0 for i in range(25)] for j in range(29)]
    rewardTest = []
    for i in range(episode):
        finishLoc = np.random.randint(0,4)
        reverseDepot = [0, 3, 20, 24]
        taxi = np.random.randint(0,25)
        passL = reverseDepot[finishLoc]
        while(passL == reverseDepot[finishLoc]):
            passL = np.random.randint(0,25)
        
        taxiDomain = QLearning(passL, reverseDepot[finishLoc], taxi, 0.99, 0.25, 0.1)

        if(algorithm == "QL"):
            QVal, policy = taxiDomain.QL(QVal, policy)
        elif(algorithm == "QL_decay"):
            QVal, policy = taxiDomain.QL_decay(QVal, policy)
        elif(algorithm == "QL_SARSA"):
            QVal, policy = taxiDomain.QL_SARSA(QVal, policy)
        else:
            QVal, policy = taxiDomain.QL_SARSA_decay(QVal, policy)


        if(i%200 == 0):
            avgRewardTest = 0
            gamma = 0.99
            for j in range(10):
                onTaxi = False
                over = False
                finishLoc = np.random.randint(0,4)
                reverseDepot = [0, 3, 20, 24]
                depots = {0:0 , 3:1, 20:2, 24:3}
                taxi = np.random.randint(0,25)
                passL = reverseDepot[finishLoc]
                while(passL == reverseDepot[finishLoc]):
                    passL = np.random.randint(0,25)
                count = 0 
                rewardPerEpisode = 0
                finishBlock = reverseDepot[finishLoc]
                state = passL
                passLoc = passL
                

                while(count < 500 and not over):
                    testAction = policy[state][taxi]
                    taxi, tempReward, passLoc, onTaxi, over = getNextTest(taxi, testAction, onTaxi, passLoc, finishBlock, over)
                    
                    rewardPerEpisode += (gamma**count)*tempReward

                    if(state > 24 and onTaxi == False):
                        state = passLoc
                    if(state <= 24 and onTaxi):
                        # self.depots = {0:0 , 3:1, 20:2, 24:3}
                        state = 25 + depots[finishBlock]
                    count += 1
                avgRewardTest += rewardPerEpisode
            avgRewardTest = avgRewardTest/10
            rewardTest.append(avgRewardTest)
    plot(rewardTest, episode, algorithm)    

    print("first: ", taxiDomain.policy[-4])
    print("secod: ", taxiDomain.policy[-3])
    print("third: ", taxiDomain.policy[-2])
    print("fourth: ", taxiDomain.policy[-1])

def partB3(episode):
    QVal, policy = QIteration(episode, "QL")
    rewardTest = []
    for i in range(5):
        gamma = 0.99
        onTaxi = False
        over = False
        finishLoc = 1 #Fixed
        reverseDepot = [0, 3, 20, 24]
        depots = {0:0 , 3:1, 20:2, 24:3}
        taxi = np.random.randint(0,25)
        passL = reverseDepot[finishLoc]
        while(passL == reverseDepot[finishLoc]):
            passL = np.random.randint(0,25)
        count = 0 
        rewardPerEpisode = 0
        finishBlock = reverseDepot[finishLoc]
        state = passL
        passLoc = passL
        while(count < 500 and not over):
            testAction = policy[state][taxi]
            taxi, tempReward, passLoc, onTaxi, over = getNextTest(taxi, testAction, onTaxi, passLoc, finishBlock, over)
            
            rewardPerEpisode += (gamma**count)*tempReward

            if(state > 24 and onTaxi == False):
                state = passLoc
            if(state <= 24 and onTaxi):
                # self.depots = {0:0 , 3:1, 20:2, 24:3}
                state = 25 + depots[finishBlock]
            count += 1
        rewardTest.append(rewardPerEpisode)
    plotB3(rewardTest, episode)


def partB4(episode, isEpsilon):
    epsilonList = [0, 0.05, 0.1, 0.5, 0.9]
    alphaList = [0.1, 0.2, 0.3, 0.4, 0.5]
    if(isEpsilon):
        TotalReward = []
        for eps in epsilonList:
            #Q Matrix
            QVal = [[[ 0 for i in range(6)] for j in range(25)] for k in range(29) ]
            #Policy
            policy = [[0 for i in range(25)] for j in range(29)]
            rewardTest = []
            for i in range(episode):
                finishLoc = np.random.randint(0,4)
                reverseDepot = [0, 3, 20, 24]
                taxi = np.random.randint(0,25)
                passL = reverseDepot[finishLoc]
                while(passL == reverseDepot[finishLoc]):
                    passL = np.random.randint(0,25)
                
                taxiDomain = QLearning(passL, reverseDepot[finishLoc], taxi, 0.99, 0.1, eps)

                QVal, policy = taxiDomain.QL(QVal, policy)
                if(i%200 == 0):
                    gamma = 0.99

                    onTaxi = False
                    over = False
                    finishLoc = np.random.randint(0,4)
                    reverseDepot = [0, 3, 20, 24]
                    depots = {0:0 , 3:1, 20:2, 24:3}
                    taxi = np.random.randint(0,25)
                    passL = reverseDepot[finishLoc]
                    while(passL == reverseDepot[finishLoc]):
                        passL = np.random.randint(0,25)
                    count = 0 
                    rewardPerEpisode = 0
                    finishBlock = reverseDepot[finishLoc]
                    state = passL
                    passLoc = passL
                    

                    while(count < 500 and not over):
                        testAction = policy[state][taxi]
                        taxi, tempReward, passLoc, onTaxi, over = getNextTest(taxi, testAction, onTaxi, passLoc, finishBlock, over)
                        
                        rewardPerEpisode += (gamma**count)*tempReward

                        if(state > 24 and onTaxi == False):
                            state = passLoc
                        if(state <= 24 and onTaxi):
                            # self.depots = {0:0 , 3:1, 20:2, 24:3}
                            state = 25 + depots[finishBlock]
                        count += 1
                    rewardTest.append(rewardPerEpisode) 
            TotalReward.append(rewardTest)
            print("first: ", taxiDomain.policy[-4])
            print("secod: ", taxiDomain.policy[-3])
            print("third: ", taxiDomain.policy[-2])
            print("fourth: ", taxiDomain.policy[-1])
        plotB4(TotalReward, episode, epsilonList, isEpsilon)
    else:
        TotalReward = []
        for alpha in alphaList:
            #Q Matrix
            QVal = [[[ 0 for i in range(6)] for j in range(25)] for k in range(29) ]
            #Policy
            policy = [[0 for i in range(25)] for j in range(29)]
            rewardTest = []
            for i in range(episode):
                finishLoc = np.random.randint(0,4)
                reverseDepot = [0, 3, 20, 24]
                taxi = np.random.randint(0,25)
                passL = reverseDepot[finishLoc]
                while(passL == reverseDepot[finishLoc]):
                    passL = np.random.randint(0,25)
                
                taxiDomain = QLearning(passL, reverseDepot[finishLoc], taxi, 0.99, alpha, 0.1)

                QVal, policy = taxiDomain.QL(QVal, policy)
                if(i%200 == 0):
                    gamma = 0.99

                    onTaxi = False
                    over = False
                    finishLoc = np.random.randint(0,4)
                    reverseDepot = [0, 3, 20, 24]
                    depots = {0:0 , 3:1, 20:2, 24:3}
                    taxi = np.random.randint(0,25)
                    passL = reverseDepot[finishLoc]
                    while(passL == reverseDepot[finishLoc]):
                        passL = np.random.randint(0,25)
                    count = 0 
                    rewardPerEpisode = 0
                    finishBlock = reverseDepot[finishLoc]
                    state = passL
                    passLoc = passL
                    

                    while(count < 500 and not over):
                        testAction = policy[state][taxi]
                        taxi, tempReward, passLoc, onTaxi, over = getNextTest(taxi, testAction, onTaxi, passLoc, finishBlock, over)
                        
                        rewardPerEpisode += (gamma**count)*tempReward

                        if(state > 24 and onTaxi == False):
                            state = passLoc
                        if(state <= 24 and onTaxi):
                            # self.depots = {0:0 , 3:1, 20:2, 24:3}
                            state = 25 + depots[finishBlock]
                        count += 1
                    rewardTest.append(rewardPerEpisode) 
            TotalReward.append(rewardTest)
            print("first: ", taxiDomain.policy[-4])
            print("secod: ", taxiDomain.policy[-3])
            print("third: ", taxiDomain.policy[-2])
            print("fourth: ", taxiDomain.policy[-1])
        plotB4(TotalReward, episode, alphaList, isEpsilon)     


#############################################
########### PART 5: 10 X 10 GRID ############
#############################################


class QLearning2:
    def __init__(self, start, finish, taxiLoc, gamma=0.99, alpha=0.25, epsilon=0.1):
        self.start = start
        self.finish = finish
        self.taxiLoc = taxiLoc
        self.passLoc = start
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.onTaxi = False
        self.over = False

        self.QVal = [[[ 0 for i in range(6)] for j in range(100)] for k in range(108) ]
        self.policy = [[0 for i in range(100)] for j in range(108)]

        self.grid = [
                    [node(["D", "L", "R"], ""), node(["L", "R"], "Y"), node(["L", "R"], ""), node(["L", "R"], ""), node(["L"], ""), node(["L"], ""), node(["L"], ""), node(["L"], ""), node(["L"], ""), node(["L", "U"], "R")],
                    [node(["D", "L"], ""), node(["L"], ""), node(["L"], ""), node(["L"], ""), node([], ""), node([], ""), node([], ""), node([], ""), node([], ""), node(["U"], "")],
                    [node(["D"], ""), node([], ""), node([], ""), node([], ""), node([], ""), node([], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node(["U", "R"], "")],
                    [node(["D", "R"], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node([], ""), node([], ""), node(["L"], ""), node(["L"], ""), node(["L"], ""), node(["U", "L"], "")],
                    [node(["D", "L"], ""), node(["L"], ""), node(["L"], ""), node(["L"], ""), node([], ""), node([], ""), node([], ""), node([], ""), node([], ""), node(["U"], "")],
                    [node(["D"], ""), node([], ""), node([], ""), node([], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node([""], ""), node(["U"], "")],
                    [node(["D"], ""), node([], ""), node([], ""), node([], ""), node(["L"], ""), node(["L"], ""), node(["L"], ""), node(["L"], ""), node([""], ""), node(["U"], "")],
                    [node(["D", "R"], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node([""], ""), node([""], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node(["U", "R"], "")],
                    [node(["D", "L"], ""), node(["L"], ""), node(["L"], ""), node(["L"], ""), node([""], ""), node([""], ""), node(["L"], ""), node(["L"], ""), node(["L"], ""), node(["U", "L"], "")],
                    [node(["D", "R"], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node(["U", "R"], "")]
                    ]
        
        self.depots = {4:0 , 9:1, 10:2, 56:3, 63:4, 90:5, 95:6, 98:7}

    def getNext(self, action):
        currState = self.taxiLoc
        reward = 0
        if(action < 4):
            X = currState%10
            Y = currState//10
            Node = self.grid[X][Y]
            #u:0 ; d:1; l:2 ; r:3 ; pUp:4 ; pDn:5 
            if(action == 0):
                if("U" in Node.walls):
                    return currState, -1
                else:
                    return currState+10, -1
            elif(action == 1):
                if("D" in Node.walls):
                    return currState, -1
                else:
                    return currState-10, -1
            elif(action == 2):
                if("L" in Node.walls):
                    return currState, -1
                else:
                    return currState-1, -1
            elif(action == 3):
                if("R" in Node.walls):
                    return currState, -1
                else:
                    return currState+1, -1
            
        else:
            if(action == 4):
                if(self.onTaxi):
                    reward = -10
                else:
                    if(self.taxiLoc == self.passLoc):
                        reward = -1
                        self.onTaxi = True
                    else:
                        reward = -10
            elif(action == 5):
                if(self.onTaxi):
                    if(self.taxiLoc == self.finish):
                        reward = 20
                        self.over = True
                        self.onTaxi = False
                    else:
                        reward = -10
                        self.passLoc = self.taxiLoc
                        self.onTaxi = False
                else:
                    reward = -10
            return currState, reward
                

    def QL(self, QVal, policy):
        self.QVal = QVal
        self.policy = policy
        if(self.over):
            return
        passLoc = self.passLoc
        onTaxi = self.onTaxi
        state = self.start
        count = 0
        while(count < 10000 and not self.over):
            count += 1
            Rval = np.random.random()
            if(Rval > self.epsilon):
                desiredAction = self.policy[state][self.taxiLoc]
            else:
                desiredAction = np.random.randint(0,6)
            if(desiredAction > 3):
                action = desiredAction
            else:
                #stochastic
                randVal = np.random.random()
                if(randVal < 0.8):
                    action = desiredAction
                else:
                    action = np.random.randint(0,4)

            nextTaxiLoc, reward = self.getNext(action)
            nextState = state
            if(state > 99 and self.onTaxi == False):
                nextState = self.passLoc
            if(state <= 99 and self.onTaxi):
                # self.depots = {0:0 , 3:1, 20:2, 24:3}
                nextState = 100 + self.depots[self.finish]
            
            #Update QVal and policy
            
            old_qVal = self.QVal[state][self.taxiLoc][action]
            nextBest_qVal = self.QVal[nextState][nextTaxiLoc][self.policy[nextState][nextTaxiLoc]]

            if(reward < 20):
                self.QVal[state][self.taxiLoc][action] = (1-self.alpha)*old_qVal + self.alpha*(reward + self.gamma*nextBest_qVal)
            else:
                self.QVal[state][self.taxiLoc][action] = (1-self.alpha)*old_qVal + self.alpha*(reward)
            
            curr_qVal = self.QVal[state][self.taxiLoc][action]
            oldBest_qVal = self.QVal[state][self.taxiLoc][self.policy[state][self.taxiLoc]]

            if(oldBest_qVal < curr_qVal):
                self.policy[state][self.taxiLoc] = action
            
            state = nextState
            self.taxiLoc = nextTaxiLoc
        
        return self.QVal, self.policy
    
def QIterationBigMaze(episode):
    QVal = [[[ 0 for i in range(6)] for j in range(100)] for k in range(108) ]
    policy = [[0 for i in range(100)] for j in range(108)]
    for i in range(episode):

        finishLoc = np.random.randint(0,8)
        reverseDepot = [4, 9, 10, 56, 63, 90, 95, 98]
        taxi = np.random.randint(0, 100)
        passL = reverseDepot[finishLoc]
        while(passL == reverseDepot[finishLoc]):
            passL = np.random.randint(0, 100)
        
        taxiDomain = QLearning2(passL, reverseDepot[finishLoc], taxi, 0.99, 0.25, 0.1)

        QVal, policy = taxiDomain.QL(QVal, policy)
    
    print("eighth: ", taxiDomain.policy[-8])
    print("seventh: ", taxiDomain.policy[-7])
    print("sixth: ", taxiDomain.policy[-6])
    print("fifth: ", taxiDomain.policy[-5])
    print("first: ", taxiDomain.policy[-4])
    print("secod: ", taxiDomain.policy[-3])
    print("third: ", taxiDomain.policy[-2])
    print("fourth: ", taxiDomain.policy[-1])

    return QVal, policy


def getNextTestBig(taxiLoc, action, onTaxi, passLoc, finish, over):
    grid = [
            [node(["D", "L", "R"], ""), node(["L", "R"], "Y"), node(["L", "R"], ""), node(["L", "R"], ""), node(["L"], ""), node(["L"], ""), node(["L"], ""), node(["L"], ""), node(["L"], ""), node(["L", "U"], "R")],
            [node(["D", "L"], ""), node(["L"], ""), node(["L"], ""), node(["L"], ""), node([], ""), node([], ""), node([], ""), node([], ""), node([], ""), node(["U"], "")],
            [node(["D"], ""), node([], ""), node([], ""), node([], ""), node([], ""), node([], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node(["U", "R"], "")],
            [node(["D", "R"], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node([], ""), node([], ""), node(["L"], ""), node(["L"], ""), node(["L"], ""), node(["U", "L"], "")],
            [node(["D", "L"], ""), node(["L"], ""), node(["L"], ""), node(["L"], ""), node([], ""), node([], ""), node([], ""), node([], ""), node([], ""), node(["U"], "")],
            [node(["D"], ""), node([], ""), node([], ""), node([], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node([""], ""), node(["U"], "")],
            [node(["D"], ""), node([], ""), node([], ""), node([], ""), node(["L"], ""), node(["L"], ""), node(["L"], ""), node(["L"], ""), node([""], ""), node(["U"], "")],
            [node(["D", "R"], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node([""], ""), node([""], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node(["U", "R"], "")],
            [node(["D", "L"], ""), node(["L"], ""), node(["L"], ""), node(["L"], ""), node([""], ""), node([""], ""), node(["L"], ""), node(["L"], ""), node(["L"], ""), node(["U", "L"], "")],
            [node(["D", "R"], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node(["R"], ""), node(["U", "R"], "")]
            ]
    currState = taxiLoc
    reward = 0
    if(action < 4):
        X = currState%10
        Y = currState//10
        Node = grid[X][Y]
        #u:0 ; d:1; l:2 ; r:3 ; pUp:4 ; pDn:5 
        if(action == 0):
            if("U" in Node.walls):
                return currState, -1, passLoc, onTaxi, over
            else:
                return currState+10, -1, passLoc, onTaxi, over
        elif(action == 1):
            if("D" in Node.walls):
                return currState, -1, passLoc, onTaxi, over
            else:
                return currState-10, -1, passLoc, onTaxi, over
        elif(action == 2):
            if("L" in Node.walls):
                return currState, -1, passLoc, onTaxi, over
            else:
                return currState-1, -1, passLoc, onTaxi, over
        elif(action == 3):
            if("R" in Node.walls):
                return currState, -1, passLoc, onTaxi, over
            else:
                return currState+1, -1, passLoc, onTaxi, over
        
    else:
        if(action == 4):
            if(onTaxi):
                reward = -10
            else:
                if(taxiLoc == passLoc):
                    reward = -1
                    onTaxi = True
                else:
                    reward = -10
        elif(action == 5):
            if(onTaxi):
                if(taxiLoc == finish):
                    reward = 20
                    over = True
                    onTaxi = False
                else:
                    reward = -10
                    passLoc = taxiLoc
                    onTaxi = False
            else:
                reward = -10
        return currState, reward, passLoc, onTaxi, over



def QIterationBigMazeTest(episode):
    #Q Matrix
    QVal = [[[ 0 for i in range(6)] for j in range(100)] for k in range(108) ]
    policy = [[0 for i in range(100)] for j in range(108)]
    rewardTest = []
    for i in range(episode):
        finishLoc = np.random.randint(0,8)
        reverseDepot = [4, 9, 10, 56, 63, 90, 95, 98]
        taxi = np.random.randint(0, 100)
        passL = reverseDepot[finishLoc]
        while(passL == reverseDepot[finishLoc]):
            passL = np.random.randint(0, 100)
        
        taxiDomain = QLearning2(passL, reverseDepot[finishLoc], taxi, 0.99, 0.25, 0.1)
        
        QVal, policy = taxiDomain.QL(QVal, policy)


        if(i%200 == 0):
            avgRewardTest = 0
            gamma = 0.99
            for j in range(5):
                onTaxi = False
                over = False
                finishLoc = np.random.randint(0,8)
                reverseDepot = [4, 9, 10, 56, 63, 90, 95, 98]
                depots = {4:0 , 9:1, 10:2, 56:3, 63:4, 90:5, 95:6, 98:7}
                taxi = np.random.randint(0,100)
                passL = reverseDepot[finishLoc]
                while(passL == reverseDepot[finishLoc]):
                    passL = np.random.randint(0,100)
                count = 0 
                rewardPerEpisode = 0
                finishBlock = reverseDepot[finishLoc]
                state = passL
                passLoc = passL
                

                while(count < 10000 and not over):
                    testAction = policy[state][taxi]
                    taxi, tempReward, passLoc, onTaxi, over = getNextTestBig(taxi, testAction, onTaxi, passLoc, finishBlock, over)
                    
                    rewardPerEpisode += (gamma**count)*tempReward

                    if(state > 99 and onTaxi == False):
                        state = passLoc
                    if(state <= 99 and onTaxi):
                        # self.depots = {0:0 , 3:1, 20:2, 24:3}
                        state = 100 + depots[finishBlock]
                    count += 1
                avgRewardTest += rewardPerEpisode
            avgRewardTest = avgRewardTest/5
            rewardTest.append(avgRewardTest)
    plot(rewardTest, episode, "BigMatrix")    

    print("first: ", taxiDomain.policy[-4])
    print("secod: ", taxiDomain.policy[-3])
    print("third: ", taxiDomain.policy[-2])
    print("fourth: ", taxiDomain.policy[-1])



#############################################
################## MAIN #####################
#############################################



if __name__ == '__main__':
    part = sys.argv[1]
    episode = 200
    # global start, end, taxiLocation
    dic = {"R" : (0, 4), "G" : (4, 4), "B" : (3, 0), "Y" : (0, 0)}
    if (part == "A2a"):
        start = dic[sys.argv[2]]
        end = dic[sys.argv[3]]
        x = int(sys.argv[4])
        y = int(sys.argv[5])
        e = float(sys.argv[6]) 
        taxiLocation = (x, y)

        valueIteration(e, 0.9)

    if (part == "A2b"):
        discountList = [0.1, 0.5, 0.8, 0.99]

        for discount in discountList:
            print("Running for discount value of: " + str(discount) + "....")
            x, y, p1, p2 = valueIteration(0.01, discount)
            plt.plot(x, y, marker = 'o')

        plt.title("Max-norm versus no of iterations")
        plt.ylabel("Max-norm")
        plt.xlabel("No of Iterations")
        plt.legend(discountList)
        plt.savefig("plot1.png")
        plt.close()

    if (part == "A2c"):
        simulate(0.1)
        simulate(0.99)

    if (part == "A3b"):
        discountList = [0.01, 0.1, 0.5, 0.8, 0.99]
        for discount in discountList:
            print("Running for discount value of: " + str(discount) + "....")
            p1, p2 = policyIteration(0.001, discount)
            x,y = findLoss(0.001, discount, p1, p2)
            plt.plot(x, y, marker = 'o')

        plt.title("Policy Loss versus No of Iterations")
        plt.ylabel("Policy Loss")
        plt.xlabel("No of Iterations")
        plt.legend(discountList)
        plt.savefig("plot2.png")
        plt.close()



    if(part == "B1"):
        subpart = sys.argv[2]
        if(subpart == "a"):
            QIteration(episode, "QL")
        elif(subpart == "b"):
            QIteration(episode, "QL_decay")
        elif(subpart == "c"):
            QIteration(episode, "QL_SARSA")
        else:
            QIteration(episode, "QL_SARSA_decay")
    if(part == "B2"):
        subpart = sys.argv[2]
        if(subpart == "a"):
            QIterationTest(episode, "QL")
        elif(subpart == "b"):
            QIterationTest(episode, "QL_decay")
        elif(subpart == "c"):
            QIterationTest(episode, "QL_SARSA")
        else:
            QIterationTest(episode, "QL_SARSA_decay")
    if(part == "B3"):
        partB3(episode)
    if(part == "B4"):
        partB4(episode, True)
        partB4(episode, False)
    if(part == "B5"):
        QIterationBigMazeTest(episode)

                



                    


