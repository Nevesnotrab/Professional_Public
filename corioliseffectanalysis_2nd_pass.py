#let's try a complex differential equation
import numpy as np
import math
import matplotlib.pyplot as plt


def cot(x):
    return 1. / math.tan(x)

def csc(x):
    return 1. / math.sin(x)

#horizontal and vertical nodes for the problem
class Analysis:
    
    def __init__(self):
        self.rows          = 20
        self.cols          = 20
        self.r             = 6371071.03
        self.arc           = 0.0
        self.node_length   = 1000.0
        self.cmap          = None
        self.rho           = 1000.0
        self.mu            = .001
        self.dt            = 120.
        self.T             = 86400.

    class Cell:
        
        def __init__subclass__(self):
            self.theta  = 0.0
            self.phi    = 0.0
            self.vtheta = 0.0
            self.vphi   = 0.0
            self.dvthetadt       = 0.0
            self.dvphidt         = 0.0
            self.dvthetadtheta   = 0.0
            self.d2vthetadtheta2 = 0.06
            self.dvthetadphi     = 0.0
            self.d2vthetadphi2   = 0.0
            self.dvphidphi       = 0.0
            self.dvphidtheta     = 0.0
            self.d2vphidphi2     = 0.0
            
    def Calc_dvthetadtheta(self, row, col):
        if(row == 0):
            self.cmap[row,col].dvthetadtheta = (self.cmap[row + 1, col].vtheta - self.cmap[row, col].vtheta) / self.node_length
        elif(row == self.rows - 1):
            self.cmap[row,col].dvthetadtheta = (self.cmap[row, col].vtheta - self.cmap[row - 1, col].vtheta) / self.node_length
        else:
            self.cmap[row,col].dvthetadtheta = (self.cmap[row + 1, col].vtheta - self.cmap[row - 1, col].vtheta) / (2. * self.node_length)
    
    def Calc_dvthetadphi(self, row, col):
        if(col == 0):
            self.cmap[row,col].dvthetadphi   = (self.cmap[row, col + 1].vtheta - self.cmap[row, col].vtheta) / self.node_length
        elif(col == self.cols - 1):
            self.cmap[row,col].dvthetadphi   = (self.cmap[row, col].vtheta - self.cmap[row, col - 1].vtheta) / self.node_length
        else:
            self.cmap[row,col].dvthetadphi   = (self.cmap[row, col + 1].vtheta - self.cmap[row, col - 1].vtheta) / (2. * self.node_length)
    
    def Calc_dvphidphi(self, row, col):
        if(col == 0):
            self.cmap[row,col].dvphidphi     = (self.cmap[row, col + 1].vphi - self.cmap[row, col].vphi) / self.node_length
        elif(col == self.cols - 1):
            self.cmap[row,col].dvphidphi     = (self.cmap[row, col].vphi - self.cmap[row, col - 1].vphi) / self.node_length
        else:
            self.cmap[row,col].dvphidphi     = (self.cmap[row, col + 1].vphi - self.cmap[row, col - 1].vphi) / (2. * self.node_length)
        
    def Calc_dvphidtheta(self, row, col):
        if(row == 0):
            self.cmap[row,col].dvphidtheta   = (self.cmap[row + 1, col].vphi - self.cmap[row, col].vphi) / self.node_length
        elif(row == self.rows - 1):
            self.cmap[row,col].dvphidtheta   = (self.cmap[row, col].vphi - self.cmap[row - 1, col].vphi) / self.node_length
        else:
            self.cmap[row,col].dvphidtheta   = (self.cmap[row + 1, col].vphi - self.cmap[row - 1, col].vtheta) / (2. * self.node_length)
        
    def Calc_d2vthetadtheta2(self, row, col):
        if(row == 0):
            self.cmap[row,col].d2vthetadtheta2 = (self.cmap[row + 2, col].vtheta - 2. * self.cmap[row + 1, col].vtheta + self.cmap[row, col].vtheta) / (self.node_length * self.node_length)
        elif(row == self.rows - 1):
            self.cmap[row,col].d2vthetadtheta2 = (self.cmap[row, col].vtheta - 2. * self.cmap[row - 1, col].vtheta + self.cmap[row - 2, col].vtheta) / (self.node_length * self.node_length)
        else:
            self.cmap[row,col].d2vthetadtheta2 = (self.cmap[row + 1, col].vtheta - 2. * self.cmap[row, col].vtheta + self.cmap[row - 1, col].vtheta) / (self.node_length * self.node_length)

    def Calc_d2vthetadphi2(self, row, col):
        if(col == 0):
            self.cmap[row,col].d2vthetadphi2   = (self.cmap[row, col + 2].vtheta - 2. * self.cmap[row, col + 1].vtheta + self.cmap[row, col].vtheta) / (self.node_length * self.node_length)
        elif(col == self.cols - 1):
            self.cmap[row,col].d2vthetadphi2   = (self.cmap[row, col].vtheta - 2. * self.cmap[row, col - 1].vtheta + self.cmap[row, col - 2].vtheta) / (self.node_length * self.node_length)
        else:
            self.cmap[row,col].d2vthetadphi2   = (self.cmap[row, col + 1].vtheta - 2. * self.cmap[row, col].vtheta + self.cmap[row, col - 1].vtheta) / (self.node_length * self.node_length)

    def Calc_d2vphidtheta2(self, row, col):
        if(row == 0):
            self.cmap[row,col].d2vphidtheta2   = (self.cmap[row + 2, col].vphi - 2. * self.cmap[row + 1, col].vphi + self.cmap[row, col].vphi) / (self.node_length * self.node_length)
        elif(row == self.rows - 1):
            self.cmap[row,col].d2vphidtheta2   = (self.cmap[row, col].vphi - 2. * self.cmap[row - 1, col].vphi + self.cmap[row - 2, col].vphi) / (self.node_length * self.node_length)
        else:
            self.cmap[row,col].d2vphidtheta2   = (self.cmap[row + 1, col].vphi - 2. * self.cmap[row, col].vphi + self.cmap[row - 1, col].vphi) / (self.node_length * self.node_length)
        
    def Calc_d2vphidphi2(self, row, col):
        if(col == 0):
            self.cmap[row,col].d2vphidphi2     = (self.cmap[row, col + 2].vphi - 2. * self.cmap[row, col + 1].vphi + self.cmap[row, col].vphi) / (self.node_length * self.node_length)
        elif(col == self.cols - 1):
            self.cmap[row,col].d2vphidphi2     = (self.cmap[row, col].vphi - 2. * self.cmap[row, col - 1].vphi + self.cmap[row, col - 2].vphi) / (self.node_length * self.node_length)
        else:
            self.cmap[row,col].d2vphidphi2     = (self.cmap[row, col + 1].vphi - 2. * self.cmap[row, col].vphi + self.cmap[row, col - 1].vphi) / (self.node_length * self.node_length)
            
    def Update_dvthetadt(self, row, col):
        self.cmap[row,col].dvthetadt = (self.mu/self.rho) * ((1./self.r/self.r)*(self.cmap[row,col].d2vthetadtheta2 + self.cmap[row,col].dvthetadtheta * cot(self.cmap[row,col].theta) - self.cmap[row,col].vtheta * csc(self.cmap[row,col].theta) * csc(self.cmap[row,col].theta)) + (1./(self.r * self.r * math.sin(self.cmap[row,col].theta) * math.sin(self.cmap[row,col].theta))) * self.cmap[row,col].d2vthetadphi2 - ((2. * cot(self.cmap[row,col].theta))/(self.r * self.r * math.sin(self.cmap[row,col].theta))) * self.cmap[row,col].dvphidphi) - (self.cmap[row,col].vtheta/self.r) * self.cmap[row,col].dvthetadtheta - (self.cmap[row,col].vphi / self.r / math.sin(self.cmap[row,col].theta)) * self.cmap[row,col].dvthetadphi + self.cmap[row,col].vphi * self.cmap[row,col].vphi * cot(self.cmap[row,col].theta) / self.r
        
    def Update_dvphidt(self, row, col):
        self.cmap[row,col].dvphidt = (self.mu / self.rho) * ((1./self.r/self.r)*(self.cmap[row,col].d2vphidtheta2 + self.cmap[row,col].dvphidtheta * cot(self.cmap[row,col].theta) - self.cmap[row,col].vphi * csc(self.cmap[row,col].theta) * csc(self.cmap[row,col].theta)) + (1. / self.r / self.r / math.sin(self.cmap[row,col].theta) / math.sin(self.cmap[row,col].theta)) * self.cmap[row,col].d2vphidphi2 + (2. * cot(self.cmap[row,col].theta) / self.r / self.r / math.sin(self.cmap[row,col].theta)) * self.cmap[row,col].dvthetadphi) - (self.cmap[row,col].vtheta / self.r) * self.cmap[row,col].dvphidtheta - (self.cmap[row,col].vphi / self.r / math.sin(self.cmap[row,col].theta)) * self.cmap[row,col].dvphidphi - self.cmap[row,col].vtheta * self.cmap[row,col].vphi * cot(self.cmap[row,col].theta) / self.r
        
            
            
    
    def Setup(self):
        self.cmap = np.ndarray(shape = (self.rows, self.cols), dtype = np.object)
        
        self.arc = self.node_length / self.r
        
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                self.cmap[i,j] = self.Cell()
                
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                self.cmap[i,j].phi = self.arc * j
                self.cmap[i,j].theta = self.arc * i + math.pi / 2.
                
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                self.cmap[i,j].vphi   = 2. * math.pi * self.r * math.sin(self.cmap[i,j].theta) / self.T
                self.cmap[i,j].vtheta = 0.
                
    def NewtonianStep(self):
        #Update all cells
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                #Update delta space values
                self.Calc_dvthetadtheta(i,j)
                self.Calc_dvthetadphi(i,j)
                self.Calc_dvphidphi(i,j)
                self.Calc_dvphidtheta(i,j)
                self.Calc_d2vthetadtheta2(i,j)
                self.Calc_d2vthetadphi2(i,j)
                self.Calc_d2vphidtheta2(i,j)
                self.Calc_d2vphidphi2(i,j)

        for i in range(0, self.rows):
            for j in range(0, self.cols):
                #Update delta time values
                self.Update_dvthetadt(i,j)
                self.Update_dvphidt(i,j)
                
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                #Execute the Newtonian Step
                self.cmap[i,j].vtheta = self.cmap[i,j].vtheta + self.cmap[i,j].dvthetadt * self.dt
                self.cmap[i,j].vphi   = self.cmap[i,j].vphi   + self.cmap[i,j].dvphidt   * self.dt
                
                
        

Test = Analysis()
Test.Setup()

for i in range(0, Test.rows):
        for j in range(0, Test.cols):
            plt.figure(0)
            plt.quiver(j, -i, Test.cmap[i,j].vphi, Test.cmap[i,j].vtheta)
       
print("Initial conditions:")
for i in range(0, Test.rows):
    for j in range(0, Test.cols):
        print(Test.cmap[i,j].vtheta, Test.cmap[i,j].vphi)

for k in range(1, 20):
    Test.NewtonianStep()
    for i in range(0, Test.rows):
        for j in range(0, Test.cols):
            plt.figure(k)
            #plt.quiver(j, -i, 0., Test.cmap[i,j].vtheta)
            plt.quiver(j, -i, Test.cmap[i,j].vphi, Test.cmap[i,j].vtheta)

print("Final conditions:")            
for i in range(0, Test.rows):
    for j in range(0, Test.cols):
        print(Test.cmap[i,j].vtheta, Test.cmap[i,j].vphi)