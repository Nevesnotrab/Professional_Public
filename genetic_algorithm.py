import numpy as np
import lhsmdu
import random

def ObjectiveFunction(x):
    x1 = x[0]
    x2 = x[1]
    return x1 ** 2. + x2 ** 2.

def ConstraintFunction(x):
    return -1

def GeneticAlgorithm(objFunc, consFunc, bndsArr, halfpop = 10, maxit = 100, tol = 1e-6, mutationChance = 0.005, mutationScale = 0.05, verbose = 0):
    #objFunc is the objective function
    #consFunc is the constraing function
        #The constraint function must be of the form g(x) < 0, where x may be
        #a vector
    #bndsArr is the bounds array
        #bndsArr has the form (# vars, 2)
        #For example    [[1, 2]
        #                [3, 4]]
        #Means 2 variables. x1 is bounded by 1, 2, and x2 is bounded by 3, 4
    #halfpop is half the population size. Half the population is used to
        #ensure parity
    #maxit is maximum iterations
    #tol is tolerance per iteration. From one iteration to the next, if the change
        #of the objective function is less than the tolerance, the algorithm
        #will automatically terminate.
    #mutationChance is the chance of a mutation occurring. It defaults to 0.5%
    #mutationScale is the scale of the mutation, scale * (-1, 1)
    
    #Establishes the total population
    totPop = halfpop * 2
    
    #Establishes the number of variables
    numVars = len(bndsArr)
    
    #Ensures the objective function is callable
    if(not callable(objFunc)):
        print("Error, objective function not specified correctly.")
        return 0
    
    #Ensures the constraint function is callable
    elif(not callable(consFunc)):
        print("Error, constraint function is not specified correctly.")
        return 0
    #Ensures the total population is an integer
    elif(not isinstance(totPop, int)):
        print("Error, population size specified as non-integer.")
        return 0
    
    #Ensures the maximum iterations is an integer
    elif(not isinstance(maxit, int)):
        print("Error, maximum iterations specified as non-integer.")
        return 0
    
    #Ensures the tolerance is an integer or decimal
    elif((not isinstance(tol, float)) and (not isinstance(tol, int))):
        print("Error, tolerance not specified as a number.")
    
    #Ensures the bounds array is the proper shape of (#vars, 2)
    elif(np.shape(bndsArr) != (numVars, 2)):
        print("Error, bounds array is not correct size. Should be of size (x , 2)")
        return 0
    
    #If all the above goes through, then it will execute the Genetic Algorithm
    else:
        print("Initialization success.")
        print("Creating initial population with LHS...")
        
        #Initializes an LHS array of size (#vars, total population)
        LHSarr = lhsmdu.sample(numVars, totPop)
        
        #Initializes an initial population of zeros of size
            #(total population, #vars)
        InitPop = np.zeros((totPop, numVars))
        
        
        #Reassigns each population member with the LHS sample to between the
            #lower bounds and upper bounds
        for i in range(0, totPop):              #i is population member index
            for j in range(0, numVars):    #j is variable number
                InitPop[i,j] = LHSarr[j,i] * (bndsArr[j,1] - bndsArr[j,0]) + bndsArr[j,0]
        #The results have the form (Population Member Index, Variable)
        
        if(verbose == 1):
            print("Initial population established.\n")
            print("Testing initial population...")
        
        currentEval = np.zeros(totPop)
        for i in range(0, totPop):
            currentEval[i] = ObjectiveFunction(InitPop[i,:])
        
        minIndex = np.argmin(currentEval)
        minVal   = min(currentEval)
        print("--------------------------------------------------------------")
        print("Initial Population Evaluation")
        print("The best objective function evaluation is ", minVal)
        print("It ocurrs at population member ", minIndex)
        print("--------------------------------------------------------------")
        
        prevGenObj = minVal
        
        for k in range(0, maxit):
            
            if(verbose == 1):
                print("Creating contest population.")
            
            #Creates a copy of the initial population for the contest population 
            ContestPopulation = np.copy(InitPop)
            
            #Creates an array that will be populated with indices for the contest
            ContestIndexArray = np.zeros((totPop, 2), dtype = int)
            
            #Creates an array of available contest indices
            AvailableContestIndices = np.linspace(0, totPop - 1, totPop, dtype = int)
            
            #Assigns available contest indices randomly to the contest index array.
                #Each population member is entered into the contest twice,
                #so during the breeding phase the population size is maintained.
            for i in range(0, halfpop):
                for j in range(0, 2):
                    select = random.randrange(0, len(AvailableContestIndices))
                    ContestIndexArray[i,j] = AvailableContestIndices[select]
                    AvailableContestIndices = np.delete(AvailableContestIndices, select)
                    
            #Reestablishes the available contest indices for the second pairing
            AvailableContestIndices = np.linspace(0, totPop - 1, totPop, dtype = int)
            
            for i in range(0, halfpop):
                for j in range(0, 2):
                    select = random.randrange(0, len(AvailableContestIndices))
                    ContestIndexArray[i + halfpop,j] = AvailableContestIndices[select]
                    AvailableContestIndices = np.delete(AvailableContestIndices, select)
            
            EvaluatedConstraints = np.zeros(totPop)
            EvaluatedObjectives  = np.zeros(totPop)
            
            for i in range(0, totPop):
                EvaluatedConstraints[i] = consFunc(ContestPopulation[i,:])
                EvaluatedObjectives[i] = objFunc(ContestPopulation[i,:])
            
            
            ContestWinnerArr = np.zeros(totPop, dtype = int)
            
            if(verbose == 1):
                print(ContestIndexArray)
            
            for i in range(0, totPop):
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
            if(verbose == 1):            
                print("Contest finished.")
                print("Winner indices: ", ContestWinnerArr)
            
                print("Creating breeding pairs.")
            
            AvailableBreeders = np.copy(ContestWinnerArr)
            BreedingIndices   = np.zeros((halfpop, 2), dtype = int)
            
            for i in range(0, halfpop):
                for j in range(0, 2):
                    select = random.randrange(0, len(AvailableBreeders))
                    BreedingIndices[i,j] = AvailableBreeders[select]
                    AvailableBreeders = np.delete(AvailableBreeders, select)
            
            #Breed
            for i in range(0, halfpop):
                InitPop[i] = 0.5 * ContestPopulation[BreedingIndices[i,0]] + \
                    0.5 * ContestPopulation[BreedingIndices[i,1]]
                InitPop[i + halfpop] = 2 * ContestPopulation[BreedingIndices[i,1]] - \
                    ContestPopulation[BreedingIndices[i,0]]
            
            for i in range(0, totPop):
                mutate = random.random()
                if(mutate < mutationChance):
                    if(verbose == 1):
                        print("Mutation.")
                    mutationAmount = mutationScale * (2. * random.random() - 1.)
                    InitPop[i] += mutationAmount
        
            #Evaluate change in best value
                
            currentEval = np.zeros(totPop)
            for i in range(0, totPop):
                currentEval[i] = ObjectiveFunction(InitPop[i,:])
            
            minIndex = np.argmin(currentEval)
            minVal   = min(currentEval)
            print("--------------------------------------------------------------")
            print("Iteration ", k + 1)
            print("The best objective function evaluation is ", minVal)
            print("It ocurrs at population member ", minIndex)
            print("--------------------------------------------------------------")
            
            currGenObj = minVal
            
            if(abs(prevGenObj - currGenObj) <= tol):
                print("Tolerance reached. Aborting algorithm.")
                print("Best results found at ", InitPop[minIndex])
                print("The value was: ", minVal)
                return 0
            else:
                prevGenObj = currGenObj
            
        print("Maximum iterations reached.")
    return 0
        
        
    

bnds = np.zeros((2, 2))
bnds[0,:] = (-3., 9.)
bnds[1,:] = (-1., 11.)

GeneticAlgorithm(ObjectiveFunction, ConstraintFunction, bnds, halfpop = 50, tol = 1e-10, verbose = 0)