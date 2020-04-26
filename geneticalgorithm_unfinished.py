import numpy as np
import lhsmdu
import random
from collections.abc import Iterable

def ObjectiveFunction(x):
    x1 = x[0]
    x2 = x[1]
    return x1 ** 2. + x2 ** 2.

def ConstraintFunction(x):
    return -1

def GeneticAlgorithm(objFunc, consFunc, bndsArr, halfpop = 10, maxit = 100, tol = 1e-6, verbose = 0):
    #objFunc is the objective function
    #consFunc is the constraint function
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
        
        #Initializes an initial population of zeros of sizeanaconda navigator terminal
            #(total population, #vars)
        InitPop = np.zeros((totPop, numVars))
        
        
        #Reassigns each population member with the LHS sample to between the
            #lower bounds and upper bounds
        for i in range(0, totPop):              #i is population member index
            for j in range(0, numVars):    #j is variable number
                InitPop[i,j] = LHSarr[j,i] * (bndsArr[j,1] - bndsArr[j,0]) + bndsArr[j,0]
        #The results have the form (Population Member Index, Variable)
        
        print("Initial population established.\n")
        print("Testing initial population...")
        
        currentEval = np.zeros(totPop)
        for i in range(0, totPop):
            currentEval[i] = objFunc(InitPop[i,:])
        
        minIndex = np.argmin(currentEval)
        minVal   = min(currentEval)
        
        print("The best objective function evaluation is ", minVal)
        print("It ocurrs at population member", minIndex)
        prevGenObj = minVal
        
        
        for i in range(0, maxit):
            
            
            print("Creating contest population.")
            
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
            
            for i in range(0, totPop):
                if(ContestIndexArray[i,0] == ContestIndexArray[i,1]):
                    print("Something went wrong with the contest assignments.")
            
            #Finish the contest
            print("Creating breeding pool by comparing contestants.")
            BreedingPopIndexArr = np.zeros(totPop, dtype = int)
            
            for j in range(0, totPop):
                indexContestantOne = ContestIndexArray[j,0]
                indexContestantTwo = ContestIndexArray[j,1]
                constraintContestantOne = consFunc(InitPop[indexContestantOne,:])
                constraintContestantTwo = consFunc(InitPop[indexContestantTwo,:])
                
                constraintViolationContestantOne = 0.
                constraintViolationContestantTwo = 0.
                
                #If the constraint function is given as multiple functions
                    #that return an array
                if((isinstance(constraintContestantOne, Iterable)) and (isinstance(constraintContestantTwo, Iterable))):
                    for k in range(0, len(constraintContestantOne)):
                        if(constraintContestantOne[k] > 0.):
                            constraintViolationContestantOne += constraintContestantOne[k]
                        if(constraintContestantTwo[k] > 0.):
                            constraintViolationContestantTwo += constraintContestantTwo[k]
                #If the constraint function just returns a scalar
                else:
                    if(constraintContestantOne > 0.):
                        constraintViolationContestantOne += constraintContestantOne
                    if(constraintContestantTwo > 0.):
                        constraintViolationContestantTwo += constraintContestantTwo
                
                #Both violate constraint, choose less infeasible
                if((constraintViolationContestantOne > 0) and (constraintViolationContestantTwo > 0)):
                    if(constraintViolationContestantOne <= constraintViolationContestantTwo):
                        BreedingPopIndexArr[j] = indexContestantOne
                    else:
                        BreedingPopIndexArr[j] = indexContestantTwo
                        
                #One violates constraint, choose the one that doesn't
                elif((constraintViolationContestantOne > 0) and (constraintViolationContestantTwo <= 0)):
                    BreedingPopIndexArr[j] = indexContestantTwo
                elif((constraintViolationContestantOne <= 0) and (constraintViolationContestantTwo > 0)):
                    BreedingPopIndexArr[j] = indexContestantOne
                
                #Neither violates constraint, choose the minimum objective
                else:
                    objectiveContestantOne = objFunc(InitPop[indexContestantOne,:])
                    objectiveContestantTwo = objFunc(InitPop[indexContestantTwo,:])
                    
                    if(objectiveContestantOne <= objectiveContestantTwo):
                        BreedingPopIndexArr[j] = indexContestantOne
                    else:
                        BreedingPopIndexArr[j] = indexContestantTwo
                        
            print("Contest over. Breeding population established:")    
            print(BreedingPopIndexArr)
            BreedingPop = np.zeros((totPop, numVars))
            
            for j in range(0, totPop):
                BreedingPop[j,:] = InitPop[BreedingPopIndexArr[j],:]
            print(BreedingPop)
            
            
            
            #Breed
            
            
            
            #Mutate
            
            
            #Evaluate
            for j in range(0, totPop):
                currentEval[j] = objFunc(ChildPop[j,:])
            
            #Finds index of best member of child population
            minIndex = np.argmin(currentEval)
            #Finds value of best evaluation of the child population
            currGenObj = min(currentEval)
            
            #Terminate or update
            if((prevGenObj - currGenObj) ** 2. <= tol):
                print("Tolerance reached. Aborting algorithm.")
                print("Best results found at ", ChildPop[minIndex,:])
                print("With a value of ", currGenObj)
                return 0
            else:
                #Updates objective evaluation for the next generation
                prevGenObj = currGenObj
                #All the children become the Initial Population of the next
                    #generation.
                for j in range(0, totPop):
                    InitPop[j,:] = ChildPop[j,:]
                
            
        print("Maximum iterations reached.")
    return 0
        
        
    

bnds = np.zeros((2, 2))
bnds[0,:] = (-3., 9.)
bnds[1,:] = (-1., 11.)

GeneticAlgorithm(ObjectiveFunction, ConstraintFunction, bnds, maxit = 1, verbose = 1)