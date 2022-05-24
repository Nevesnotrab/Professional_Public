from mimetypes import init
import numpy as np
import lhsmdu
import random

class GeneticAlgorithm():
    def DummyConstraintFunction(self, x):
        return -1

    def __init__(self):
        self.objFunc = None
        self.consFunc = self.DummyConstraintFunction
        self.bndsArr = []
        self.halfpop = 10
        self.maxit = 100
        self.tol = 1e-6
        self.mutationChance = 0.005
        self.mutationScale = 0.05
        self.verbose = 0
        self.initalized = 0
        self.totPop = 0
        self.numVars = 0

    def Help(self):
        print("This is a genetic algorithm class.")
        print("objFunc is the objective function. It should take a single vector 'x' as an argument.")
        print("consFunc is the constraint function. If you do not require a constraint, do not specify a function. A dummy function has been created and assigned for you.")
        print("bndsArr is the bounds array. It should be specified such that bnds(n,:) has dimensions (1, 2) where bnds(n,0) is the lower bound for x[n] and bnds(n,1) is the upper bound for x[n].")
        print("halfpop is half the population size. It should be an integer. Half the population is specified so that the user can't make the mistake of setting an odd number as the population size.")
        print("maxit is the maximum number of iterations. Useful for limiting the computational time.")
        print("tol is the tolerance. If the best member of the population two iterations in a row does not improve by more than the tolerance, the simulation ends.")
        print("mutationChance is the chance of a mutation occurring in a new population member.")
        print("mutationScale is how much a mutation changes the value. If you are working with small values, you likely want a small mutation scale. It is random whether the mutation scale is added or subtracted.")
        print("verbose makes the simulation output what is happening each iteration. For simple simulations it will likely go quickly and be unreadable.")
        print("initialized should not be messed with. It's an internal check for the simulation.")
        print("numVars should not be messed with. It's used by the simulation to keep track of the dimensionality of the simulation. It enables x to be nth dimensional.")
        print("")
        print("Help is the help function. You probably called it if you're seeing this.")
        print("Run runs the simulation. First it runs Initialize then it runs ErrorCheck to make sure everything is set up properly.")
        print("Initialize should not be messed with. It sets up the necessary variables for the simulation.")
        print("ErrorCheck should not be messed with. It checks that variables and functions have been assigned properly.")

    def Initialize(self):
        #Initializes the total population totPop and the dimensionality of x numVars.
        self.totPop = self.halfpop * 2
        self.numVars = len(self.bndsArr)
        self.initalized = 1

    def ErrorCheck(self):
        #Ensures the objective function is callable
        if(not callable(self.objFunc)):
            print("Error, objective function not specified correctly.")
            return 1
        #Ensures the constraint function is callable
        elif(not callable(self.consFunc)):
            print("Error, constraint function not specified correctly. Simply return -1 if no constraint is necessary. If you're seeing this message then you messed with something you should not have.")
            return 1
        #Ensures the total population is an integer
        elif(not isinstance(self.totPop, int)):
            print("Error, population size specified as non-integer.")
            return 1
        #Ensures the maximum number of iterations is an integer
        elif(not isinstance(self.maxit, int)):
            print("Error, maximum iterations specified as non-integer.")
            return 1
        #Ensures the tolerance is specified as a float or an int
        elif((not isinstance(self.tol, float)) and (not isinstance(self.tol, int))):
            print("Error, tolerance not specified as a number.")
            return 1
        #Ensures the bounds array is the correct size
        elif(np.shape(self.bndsArr) != (self.numVars, 2)):
            print("Error, bounds array is not correct size. Should be of size (x , 2)")
            return 1
        #Ensures the dimensionality of x is an integer and greater than 0
        elif(not isinstance(self.numVars, int) or self.numVars <= 0):
            print("Error, bounds array specified incorrectly.")
            return 1
        elif(not self.initalized):
            print("Error, the run is not initialized.")
            return 1
        else:
            return 0

    def Run(self):
        self.Initialize()
        check = self.ErrorCheck()

        if(check):
            print("Something went wrong when error checking. See output log for details.")
        else:
            print("Initialization success.")
            print("Creating initial population with LHS...")
            
            #Initializes an LHS array of size (#vars, total population)
            LHSarr = lhsmdu.sample(self.numVars, self.totPop)
            
            #Initializes an initial population of zeros of size
                #(total population, #vars)
            InitPop = np.zeros((self.totPop, self.numVars))
            
            
            #Reassigns each population member with the LHS sample to between the
                #lower bounds and upper bounds
            for i in range(0, self.totPop):              #i is population member index
                for j in range(0, self.numVars):    #j is variable number
                    InitPop[i,j] = LHSarr[j,i] * (self.bndsArr[j,1] - self.bndsArr[j,0]) + self.bndsArr[j,0]
            #The results have the form (Population Member Index, Variable)
            
            if(self.verbose == 1):
                print("Initial population established.\n")
                print("Testing initial population...")
            
            currentEval = np.zeros(self.totPop)
            for i in range(0, self.totPop):
                currentEval[i] = self.objFunc(InitPop[i,:])
            
            minIndex = np.argmin(currentEval)
            minVal   = min(currentEval)
            print("--------------------------------------------------------------")
            print("Initial Population Evaluation")
            print("The best objective function evaluation is ", minVal)
            print("It ocurrs at population member ", minIndex)
            print("--------------------------------------------------------------")
            
            prevGenObj = minVal
            
            for k in range(0, self.maxit):
                
                if(self.verbose == 1):
                    print("Creating contest population.")
                
                #Creates a copy of the initial population for the contest population 
                ContestPopulation = np.copy(InitPop)
                
                #Creates an array that will be populated with indices for the contest
                ContestIndexArray = np.zeros((self.totPop, 2), dtype = int)
                
                #Creates an array of available contest indices
                AvailableContestIndices = np.linspace(0, self.totPop - 1, self.totPop, dtype = int)
                
                #Assigns available contest indices randomly to the contest index array.
                    #Each population member is entered into the contest twice,
                    #so during the breeding phase the population size is maintained.
                for i in range(0, self.halfpop):
                    for j in range(0, 2):
                        select = random.randrange(0, len(AvailableContestIndices))
                        ContestIndexArray[i,j] = AvailableContestIndices[select]
                        AvailableContestIndices = np.delete(AvailableContestIndices, select)
                        
                #Reestablishes the available contest indices for the second pairing
                AvailableContestIndices = np.linspace(0, self.totPop - 1, self.totPop, dtype = int)
                
                for i in range(0, self.halfpop):
                    for j in range(0, 2):
                        select = random.randrange(0, len(AvailableContestIndices))
                        ContestIndexArray[i + self.halfpop,j] = AvailableContestIndices[select]
                        AvailableContestIndices = np.delete(AvailableContestIndices, select)
                
                EvaluatedConstraints = np.zeros(self.totPop)
                EvaluatedObjectives  = np.zeros(self.totPop)
                
                for i in range(0, self.totPop):
                    EvaluatedConstraints[i] = self.consFunc(ContestPopulation[i,:])
                    EvaluatedObjectives[i] = self.objFunc(ContestPopulation[i,:])
                
                
                ContestWinnerArr = np.zeros(self.totPop, dtype = int)
                
                if(self.verbose == 1):
                    print(ContestIndexArray)
                
                for i in range(0, self.totPop):
                    constraint1 = EvaluatedConstraints[ContestIndexArray[i,0]]
                    constraint2 = EvaluatedConstraints[ContestIndexArray[i,1]]
                    if((constraint1 > 0) and (constraint2 > 0)):
                        if(constraint1 < constraint2):
                            ContestWinnerArr[i] = ContestIndexArray[i,0]
                        else:
                            ContestWinnerArr[i] = ContestIndexArray[i,1]
                    elif((constraint1 < 0) and (constraint2 > 0)):
                        ContestWinnerArr[i] = ContestIndexArray[i,0]
                    elif((constraint1 > 0) and (constraint2 < 0)):
                        ContestWinnerArr[i] = ContestIndexArray[i,1]
                    else:
                        evaluate1 = EvaluatedObjectives[ContestIndexArray[i,0]]
                        evaluate2 = EvaluatedObjectives[ContestIndexArray[i,1]]
                        if(evaluate1 < evaluate2):
                            ContestWinnerArr[i] = ContestIndexArray[i,0]
                        else:
                            ContestWinnerArr[i] = ContestIndexArray[i,1]
                if(self.verbose == 1):            
                    print("Contest finished.")
                    print("Winner indices: ", ContestWinnerArr)
                
                    print("Creating breeding pairs.")
                
                AvailableBreeders = np.copy(ContestWinnerArr)
                BreedingIndices   = np.zeros((self.halfpop, 2), dtype = int)
                
                for i in range(0, self.halfpop):
                    for j in range(0, 2):
                        select = random.randrange(0, len(AvailableBreeders))
                        BreedingIndices[i,j] = AvailableBreeders[select]
                        AvailableBreeders = np.delete(AvailableBreeders, select)
                
                #Breed
                for i in range(0, self.halfpop):
                    InitPop[i] = 0.5 * ContestPopulation[BreedingIndices[i,0]] + \
                        0.5 * ContestPopulation[BreedingIndices[i,1]]
                    InitPop[i + self.halfpop] = 2 * ContestPopulation[BreedingIndices[i,1]] - \
                        ContestPopulation[BreedingIndices[i,0]]

                #Mutate
                for i in range(0, self.totPop):
                    for j in range(0, self.numVars):
                        mutate = random.random()
                        if(mutate < self.mutationChance):
                            if(self.verbose == 1):
                                print("Mutation.")
                            mutationAmount = self.mutationScale * (2. * random.random() - 1.)
                            InitPop[i,j] += mutationAmount

                #Ensure values are within bounds
                for i in range(0, self.totPop):
                    for j in range(0, self.numVars):
                        if(InitPop[i,j] < self.bndsArr[j,0]):
                            InitPop[i,j] = self.bndsArr[j,0]
                        elif(InitPop[i,j] > self.bndsArr[j,1]):
                            InitPop[i,j] = self.bndsArr[j,1]
            
                #Evaluate change in best value
                    
                currentEval = np.zeros(self.totPop)
                for i in range(0, self.totPop):
                    currentEval[i] = self.objFunc(InitPop[i,:])
                
                minIndex = np.argmin(currentEval)
                minVal   = min(currentEval)
                print("--------------------------------------------------------------")
                print("Iteration ", k + 1)
                print("The best objective function evaluation is ", minVal)
                print("It ocurrs at population member ", minIndex)
                print("--------------------------------------------------------------")
                
                currGenObj = minVal
                
                if(abs(prevGenObj - currGenObj) <= self.tol):
                    print("Tolerance reached. Aborting algorithm.")
                    print("Best results found at ", InitPop[minIndex])
                    print("The value was: ", minVal)
                    return 0
                else:
                    prevGenObj = currGenObj
                
            print("Maximum iterations reached.")
            print("Best results found at: ", InitPop[minIndex])
            print("The value was: ", minVal)

def RunTestAlgorithm():

    def ObjectiveFunction(x):
        x1 = x[0]
        x2 = x[1]
        return x1 ** 2. + x2 ** 2.

    bnds = np.zeros((2, 2))
    bnds[0,:] = (-3., 9.)
    bnds[1,:] = (-1., 11.)

    testAlg = GeneticAlgorithm()
    testAlg.objFunc = ObjectiveFunction
    testAlg.bndsArr = bnds
    testAlg.halfpop = 50
    testAlg.tol = 1e-10
    testAlg.Run()