from BinaryEncoderDecoder import BinaryEncoderDecoder
from Utilities import Utilities

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
        
        self.binary_format = False
        self.vector_format = False
        self.id_format = True
        
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
        
        utils = Utilities()
        print("Dataset size: " + str(self.size) + " - " + utils.formatBytes(self.size*2*4) + " (int32)")
            
        
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
        
        utils = Utilities()
        print("Dataset size: " + str(self.size) + " - " + utils.formatBytes(self.size*2*4) + " (int32)")

        
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
        
        self.binary_format = True
        self.id_format = False
        
    
    # Format the dataset to compressed state and compressed input vectors
    def formatToVector(self, controller):
        new_x = []
        new_y = []
        
        ss_dim = controller.getStateSpaceDim()
        ss_ll = controller.getStateSpaceLowerLeft()
        ss_ur = controller.getStateSpaceUpperRight()
        ss_ngp = controller.getStateSpaceNGP()

        is_dim = controller.getInputSpaceDim()
        is_ll = controller.getInputSpaceLowerLeft()
        is_ur = controller.getInputSpaceUpperRight()
        is_ngp = controller.getInputSpaceNGP()
        
        self.x_bounds = [ss_ll, ss_ur]
        self.y_bounds = [is_ll, is_ur]
        
        self.x_dim = ss_dim
        self.y_dim = is_dim
        
        for i in range(self.size):
            # get pair
            pair = controller.getVectorPairFromIndex(i) # make it dependent on the dataset instead of the controller
            state_vector = pair[0]
            input_vector = pair[1]
            
            # compress pair using boundaries
            for j in range(ss_dim):
                x = state_vector[j]
                state_vector[j] = (x - ss_ll[j])/(ss_ur[j] - ss_ll[j])
                if(state_vector[j] < 0 or state_vector[j] > 1):
                    print("State vector is out of range")
                    print(str(state_vector[j]))
                    print(str(x))
                    print(str(x - ss_ll[j]) + " " + str(ss_ur[j] - ss_ll[j]))
                    return
                
            for j in range(is_dim):
                u  = input_vector[j]
                input_vector[j] = (u - is_ll[j])/(is_ur[j] - is_ll[j])
                if(input_vector[j] < 0 or input_vector[j] > 1):
                    print("Input vector is out of range")
                    print(str(input_vector[j]))
                    print(str(u))
                    print(str(u - is_ll[j]) + " " + str(is_ur[j] - is_ll[j]))
                    return
            
            # append 
            new_x.append(state_vector)
            new_y.append(input_vector)
            
        self.x = new_x
        self.y = new_y
        
        # etas
        self.x_eta = []
        self.y_eta = []
        for i in range(ss_dim):
            self.x_eta.append(1/ss_ngp[i])
            #print("ss_eta" + str(i) + ": " + str(self.x_eta[i]))
            
        for i in range(is_dim):
            self.y_eta.append(1/is_ngp[i])
            #print("is_eta" + str(i) + ": " + str(self.y_eta[i]))
            
        self.vector_format = True
        self.id_format = False
                     