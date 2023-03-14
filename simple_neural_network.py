"""
SNN is a Simple Neural Network machine that performs the Matrix Multiplication for you.
SNN is essentially a glorified matrix multiplication macro.
SNN connects every node in one layer to every node in the next layer.
SNN was created because traditional "neural networks" also implement the optimization/ML/etc. processes and are very complicated. I wanted a way to use a neural network without all the bells and whistles.
SNN *DOES NOT* implement optimization, ML algorithms, etc.
SNN *DOES NOT* implement post processing, automatic hyperparameter adjustment, etc.
SNN *DOES NOT* analyze or alter your outputs. E.g. if you need one of the output nodes to be binary, you need to implement your own Sigmoid function on that node
SNN *DOES NOT* use your GPU/cuda cores for processing
SNN *DOES NOT* automatically assemble your optimization algorithm weights into the form SNN requires.
SNN is not optimized.
SNN is not "smart".
"""

import numpy as np

class SNN():

    def __init__(self):
        self.num_hidden_layers  = 0
        self.num_inputs  = 0
        self.num_outputs = 0
        self.num_weight_layers = 0
        self.hidden_layer_sizes = np.array([], dtype=int)
        self.hidden_layer_nodes = []
        self.complete_neural_network_array = []
        self.SetNumHiddenLayersCheck = False
        self.SetNumInputsCheck = False
        self.SetNumOutputsCheck = False
        self.SetHiddenLayerSizesCheck = False
        self.EstablishNumberWeightLayersCheck = False
        self.SNNAssembledCheck = False

    def PrintInfo(self):
        print(f"Number of inputs: {self.num_inputs}")
        print(f"Number of hidden layers: {self.num_hidden_layers}")
        print(f"Hidden layer sizes: {self.hidden_layer_sizes}")
        print(f"Number of outputs: {self.num_outputs}")
        print(f"Number of weight layers: {self.num_weight_layers}")

    def PrintChecks(self):
        print(f"Set number of inputs: {self.SetNumInputsCheck}")
        print(f"Set number of hidden layers: {self.SetNumHiddenLayersCheck}")
        print(f"Set hidden layer sizes: {self.SetHiddenLayerSizesCheck}")
        print(f"Set number of outputs: {self.SetNumOutputsCheck}")
        print(f"Set number of weight layers: {self.EstablishNumberWeightLayersCheck}")
        print(f"Simple Neural Network Assembled: {self.SNNAssembledCheck}")

    def SetNumHiddenLayers(self, num_layers):
        if num_layers < 0:
            print("Too few layers.")
        else:
            self.num_hidden_layers = num_layers
            self.SetNumHiddenLayersCheck = True

    def SetNumInputs(self, num_inputs):
        if num_inputs < 1:
            print("Too few inputs.")
        else:
            self.num_inputs = num_inputs
            self.SetNumInputsCheck = True

    def SetNumOutputs(self, num_outputs):
        if num_outputs < 1:
            print("Too few outputs.")
        else:
            self.num_outputs = num_outputs
            self.SetNumOutputsCheck = True

    def SetHiddenLayerSizes(self, x):
        if self.SetNumHiddenLayersCheck:
            if isinstance(x, np.ndarray):
                if x.shape[0] == x.size:
                    if x.size == self.num_hidden_layers:
                        for i in range(0,len(x)):
                            self.hidden_layer_sizes = np.append(self.hidden_layer_sizes, x[i])
                            self.hidden_layer_nodes.append(np.zeros(self.hidden_layer_sizes[i]))
                        self.SetHiddenLayerSizesCheck = True
                    else:
                        print("The number of hidden layers should match the number of hidden layer sizes provided.")
                else:
                    print("The Hidden Layer sizes array should be 1-D")
            else:
                print("The Hidden Layer sizes should be passed as a numpy array with the following format: [layer1size, layer2size, ...]")
        else:
            print("Set the number of hidden layers, first.")

    def EstablishNumberWeightLayers(self):
        if self.SetNumHiddenLayersCheck:
            self.num_weight_layers = self.num_hidden_layers + 1
            self.EstablishNumberWeightLayersCheck = True
        else:
            print("Set the number of hidden layers, first.")

    def SNNAssemble(self):
        if (self.SetNumHiddenLayersCheck and \
            self.SetNumInputsCheck and \
            self.SetNumOutputsCheck and \
            self.SetHiddenLayerSizesCheck and \
            self.EstablishNumberWeightLayersCheck):

            self.SNNAssembledCheck = True
            print("Good to go")

    def Run(self, input_array, weight_array, just_output_nodes = True):
        if self.SNNAssembledCheck:
            if isinstance(input_array, np.ndarray):
                if input_array.shape[0] == self.num_inputs:
                    self.complete_neural_network_array.append(input_array)
                    for i in range(0, len(self.hidden_layer_nodes)):
                        self.complete_neural_network_array.append(self.hidden_layer_nodes[i])
                    self.complete_neural_network_array.append(np.zeros(self.num_outputs))
                else:
                    print("The input array must be the same length as the number of inputs you provided and it must be 1-D")
                    return
            else:
                print("The input array should be passed as a numpy array with the following format: [node0, node1, node2, ...]")
                return

            if isinstance(weight_array, list):
                if self.num_weight_layers == len(weight_array):
                    for i in range(0, len(weight_array)):
                        if isinstance(weight_array[i], np.ndarray):
                            pass
                        else:
                            print("Each individual weight array must be a numpy array.")
                            return
                else:
                    print("There are not enough weight arrays for the number of specified hidden layers.")
                    return
            else:
                print("The weight array needs to be a list of numpy weight arrays for each hidden layer.")
                return

            try:
                for i in range(0, len(self.complete_neural_network_array) - 1):
                    self.complete_neural_network_array[i+1] = np.matmul(weight_array[i], self.complete_neural_network_array[i])

                if(just_output_nodes):
                    return self.complete_neural_network_array[-1]
                else:
                    return self.complete_neural_network_array

            except:
                print("There is probably a dimension mismatch, so matrix multiplication did not happen.")
                print("There could also be some other, unknown bug, in which case: good luck.")
                return
        else:
            print("Run the SNNAssemble() function to verify that the SNN is assembled properly.")

#test = SNN()
#test.SetNumInputs()
#test.SetNumHiddenLayers()
#test.SetNumOutputs()
#test.SetHiddenLayerSizes()
#test.EstablishNumberWeightLayers()
#test.PrintHiddenLayerNodes()
#test.SNNAssemble()
#test.PrintChecks()

"""
----------Example----------
"""
import time

test_first_layer_number = 10
test_second_layer_number = 5
test_third_layer_number = 20
test_fourth_layer_number = 2

FirstLayer = np.random.rand(test_first_layer_number) #Creates first layer of the size specified by test_first_layer_number

Warray = [] #Blank weight array

Warray.append(np.random.rand(test_second_layer_number, test_first_layer_number)) #Creates a weight array with the proper dimensions (layer 1 -> layer 2) and appends it to the overall weight array
Warray.append(np.random.rand(test_third_layer_number, test_second_layer_number)) #Creates a weight array with the proper dimensions (layer 2 -> layer 3) and appends it to the overall weight array
Warray.append(np.random.rand(test_fourth_layer_number, test_third_layer_number)) #Creates a weight array with the proper dimensions (layer 3 -> layer 4) and appends it to the overall weight array

hidden_layer_sizes = np.array([test_second_layer_number, test_third_layer_number]) #Creates the array of hidden layer sizes based off the sizes of the hidden layers (layers 2 and 3)

test_result = SNN()                                         #Initializes the Simple Neural Network
test_result.SetNumInputs(test_first_layer_number)           #Sets the number of input nodes
test_result.SetNumHiddenLayers(len(hidden_layer_sizes))     #Sets the number of hidden layers based on the length of the hidden layer sizes array
test_result.SetHiddenLayerSizes(hidden_layer_sizes)         #Sets the hidden layer sizes
test_result.SetNumOutputs(test_fourth_layer_number)         #Sets the number of output nodes
test_result.EstablishNumberWeightLayers()                   #Instructs the SNN to calculate the number of weight layers
test_result.PrintInfo()                                     #Prints the basic stats of the SNN
test_result.PrintChecks()                                   #Prints the results of the various checks the SNN performs before running
test_result.SNNAssemble()                                   #Verifies the various checks the SNN performs before running (and will not run without checking)

start = time.clock()
test_result = test_result.Run(FirstLayer, Warray)           #Runs a single instance of the SNN based on the input nodes and weights
end = time.clock()
print(test_result)                                          #Prints the output nodes
print("It took: ", end - start, " seconds to run.")         #Prints how long the Run function took to run (this example should be pretty fast).

""""""