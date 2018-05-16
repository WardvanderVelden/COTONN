from BinaryEncoderDecoder import BinaryEncoderDecoder

import math
import random

# Dataset class which will contain the data for nn training and functions to read controllers into the specific
class DataSet:
    def __init__(self):
        self.x = []
        self.y = []
        
        self.x_bounds = [] # boundary elements of x 
        self.y_bounds = [] # boundary elements of y
        
        self.x_eta = [] # etas of x
        self.y_eta = [] # etas of y
        
        self.x_dim = 1 # dimension of elements in x
        self.y_dim = 1 # dimension of elements in y
        
        self.size = 0 # amount of elements in dataset
        
    # Getters
    def getX(self, i): return self.x[i]
    def getY(self, i): return self.y[i]
    def getSize(self): return self.size
    
    def getPair(self, i): return [self.x[i], self.y[i]]
    def getPairByX(self, x):
        for i in range(self.size):
            if(self.x[i] == x):
                return [self.x[i], self.y[i]]
        return None
    
    def getLowestX(self): return min(int(x) for x in self.x)
    def getHighestX(self): return max(int(x) for x in self.x)
    def getLowestY(self): return min(int(y) for y in self.y)
    def getHighestY(self): return max(int(y) for y in self.y)
    
    def getXBounds(self): return self.x_bounds
    def getYBounds(self): return self.y_bounds
    
    def getXDim(self): return self.x_dim
    def getYDim(self): return self.y_dim
    
    def getXEta(self): return self.x_eta
    def getYEta(self): return self.y_eta
    
    # Add pair to dataset
    def add(self, x, y):
        self.x.append(x)
        self.y.append(y)
        self.size = len(self.x)
        
    # Read dataset from controller
    def readSetFromController(self, controller):
        for i in range(controller.getSize()):
            pair = controller.getPairFromIndex(i)
            self.x.append(pair[0])
            self.y.append(pair[1])
            
        self.size = len(self.x)
        
        self.x_bounds = [self.getLowestX(), self.getHighestX()]
        self.y_bounds = [self.getHighestY(), self.getHighestY()]
        
        self.x_eta = controller.getStateSpaceEtas()
        self.y_eta = controller.getInputSpaceEtas()
        
        print("Dataset size: " + str(self.size))
            
    # Read pseudo random subset from controller
    def readSubsetFromController(self, controller, percentage):
        size = controller.getSize()
        ids = []
        new_size = round(size*percentage)
        
        # add highest and lowest controller to make sure the set had the same input and output format
        h_s, l_s = controller.getHighestState(), controller.getLowestState()
        ids.append(controller.getIndexOfState(l_s))
        ids.append(controller.getIndexOfState(h_s))
        
        # get random ids 
#        for i in range(new_size - 2):
#            while True:
#                r = math.floor(random.random()*size)
#                if r not in ids:
#                    ids.append(r)
#                    break
        random_ids = random.sample(range(0, size), (new_size - 2))
        ids += random_ids
        
        # fill dataset
        for i in range(new_size):
            pair = controller.getPairFromIndex(ids[i])
            self.x.append(pair[0])
            self.y.append(pair[1])
            
        self.size = len(self.x)
                    
        self.x_bounds = [self.getLowestX(), self.getHighestX()]
        self.y_bounds = [self.getHighestY(), self.getHighestY()]
        
        self.x_eta = controller.getStateSpaceEtas()
        self.y_eta = controller.getInputSpaceEtas()
        
        print("Dataset size: " + str(self.size))

        
    # Shuffle data
    def shuffle(self):
        pairs = []
        for i in range(self.size):
            pairs.append([self.x[i], self.y[i]])
            
        random.shuffle(pairs)
        n_x, n_y = [], []
        for i in range(self.size):
            n_x.append(pairs[i][0])
            n_y.append(pairs[i][1])
            
        self.x = n_x
        self.y = n_y
    
    
    # Get a batch from the data set
    def getBatch(self, size, i):
        x_batch = []
        y_batch = []
        for i in range(i, i + size):
            x_batch.append(self.x[i%self.size])
            y_batch.append(self.y[i%self.size])
        return [x_batch, y_batch]
            
    
    # Format dataset to binary inputs and outputs
    def formatToBinary(self):
        bed = BinaryEncoderDecoder()
        
        new_x = []
        new_y = []
        
        n_x = len(bed.sntob(self.x_bounds[1])) # Input bit length
        n_y = len(bed.sntob(self.y_bounds[1])) # Output bit length
        
        for i in range(self.size):
            new_x.append(bed.ntoba(self.x[i],n_x))
            new_y.append(bed.ntoba(self.y[i],n_y))
            
        # set x and y to converted x and y
        self.x = new_x
        self.y = new_y
        
        # set binary upper and lower bounds
        self.x_bounds = [bed.ntoba(self.x_bounds[0], n_x), bed.ntoba(self.x_bounds[1], n_x)]
        self.y_bounds = [bed.ntoba(self.y_bounds[0], n_y), bed.ntoba(self.y_bounds[1], n_y)]
        
        # set the dimension of x and y (elements per element of x and y)
        self.x_dim = n_x
        self.y_dim = n_y
        
        # set etas of x and y (0.5 for binary)
        self.x_eta = []
        self.y_eta = []
        for i in range(n_x):
            self.x_eta.append(0.5)
        for i in range(n_y):
            self.y_eta.append(0.5)
        
        self.size = len(self.x)