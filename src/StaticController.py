import math

# PlainController class which will hold the controller that can then be accessed in order to read training
# data for the neural network
class StaticController:
    def __init__(self):
        self.state_space_dim = 0
        self.state_space_etas = []
        self.state_space_lower_left = []
        self.state_space_upper_right = []
        self.state_space_ngp = None
        self.state_space_ipd = None

        self.input_space_dim = 0
        self.input_space_etas = []
        self.input_space_lower_left = []
        self.input_space_upper_right = []
        self.input_space_ngp = None
        self.input_space_ipd = None

        self.states = []
        self.inputs = []
        
        self.state_size = 0
        self.input_size = 0
        
    # Getters
    def getStateSpaceDim(self): return self.state_space_dim
    def getStateSpaceEtas(self): return self.state_space_etas
    def getStateSpaceLowerLeft(self): return self.state_space_lower_left
    def getStateSpaceUpperRight(self): return self.state_space_upper_right
    def getStateSpaceNGP(self): return self.state_space_ngp
    def getStateSpaceIPD(self): return self.state_space_ipd
    
    def getInputSpaceDim(self): return self.input_space_dim
    def getInputSpaceEtas(self): return self.input_space_etas
    def getInputSpaceLowerLeft(self): return self.input_space_lower_left
    def getInputSpaceUpperRight(self): return self.input_space_upper_right
    def getInputSpaceNGP(self): return self.input_space_ngp
    def getInputSpaceIPD(self): return self.input_space_ipd
    
    def getState(self, id): return self.states[id]
    def getInput(self, id): return self.inputs[id]
    
    
    # Get the input id and state id for a given state id
    def getPairFromState(self, state):
        for i in range(self.state_size):
            if(self.states[i] == state):
                return [self.states[i], self.inputs[i]]
        return None
    
    
    # Get the input id and state id for a given state index
    def getPairFromIndex(self, id):
        if(id >= 0 and id < self.state_size):
            return [self.states[id], self.inputs[id]]
        return None
        
    
    # Get the input id corresponding to a given state id
    def getInputFromState(self, state):
        for i in range(self.state_size):
            if(self.states[i]  == state):
                return self.inputs[i]
        print("ID does not correspond to a state in the winning domain.")
        return None
    
    
    # Get vector pair from index
    def getVectorPairFromIndex(self, id):
        if(id >= 0 and id < self.state_size):
            state_id = self.states[id]
            input_id = self.inputs[id]                        
            
        return [self.getStateVector(state_id), self.getInputVector(input_id)]
    
    # Get lowest state id contained in the controller
    def getLowestState(self):
        return min(int(s) for s in self.states)
    
    
    # Get highest state id contained in the controller
    def getHighestState(self):
        return max(int(s) for s in self.states)
    
    
    # Get lowest input id contained in the controller
    def getLowestInput(self):
        return min(int(i) for i in self.inputs)
    
    
    # Get highest input id contained in the controller
    def getHighestInput(self):
        return max(int(i) for i in self.inputs)
    
    
    # Get the size of the controller
    def getSize(self):
        return self.state_size
    
    
    # Get index of value
    def getIndexOfState(self, state):
        return self.states.index(state)
    
    
    # Get state vector
    def getStateVector(self, state_id):
        i = self.state_space_dim - 1
        x = [0.0]*self.state_space_dim
        
        while (i>0):
            num = math.floor(state_id/self.state_space_ipd[i])
            state_id = state_id % self.state_space_ipd[i]
            x[i] = self.state_space_lower_left[i] + num*self.state_space_etas[i]
            
            i -= 1
            
        x[0] = self.state_space_lower_left[0] + state_id*self.state_space_etas[0];
        return x
    
    
    # Get state id from the state vector # NOT CORRECT
    def getIdFromStateVector(self, state_vector):
        ss_eta = self.getStateSpaceEtas()
        ss_ll = self.getStateSpaceLowerLeft()
        ss_dim = self.getStateSpaceDim()
        ss_ipd = self.getStateSpaceIPD()
        
        id = 0
        for i in range(ss_dim):
            d_i = state_vector[i] - ss_ll[i]
            id = id + math.floor((d_i + ss_eta[i]/2.0)/ss_eta[i])*ss_ipd[i]
            
        return id
        
        
    # Get input vector
    def getInputVector(self, input_id):
        i = self.input_space_dim - 1
        u = [0.0]*self.input_space_dim
        
        while (i>0):
            num = math.floor(input_id/self.input_space_ipd[i])
            input_id = input_id % self.input_space_ipd[i]
            u[i] = self.input_space_lower_left[i] + num*self.input_space_etas[i]
            
            i -= 1

        u[0] = self.input_space_lower_left[0] + input_id*self.input_space_etas[0];  
        return u
    
        
    # Setters
    # State space
    def setStateSpaceDim(self, value): 
        self.state_space_dim = int(value)
        
    def setStateSpaceEtas(self, value):
        for i in range(len(value)):
            self.state_space_etas.append(float(value[i]))
            
    def setStateSpaceLowerLeft(self, value):
        for i in range(len(value)):
            self.state_space_lower_left.append(float(value[i]))
            
    def setStateSpaceUpperRight(self, value):
        for i in range(len(value)):
            self.state_space_upper_right.append(float(value[i]))
    
    # Input space
    def setInputSpaceDim(self, value): 
        self.input_space_dim = int(value)
        
    def setInputSpaceEtas(self, value):
        for i in range(len(value)):
            self.input_space_etas.append(float(value[i]))
            
    def setInputSpaceLowerLeft(self, value):
        for i in range(len(value)):
            self.input_space_lower_left.append(float(value[i]))
            
    def setInputSpaceUpperRight(self, value): 
        for i in range(len(value)):
            self.input_space_upper_right.append(float(value[i]))

    # Set pair
    def setStateInput(self, s, i):
        self.states.append(int(s))
        self.inputs.append(int(i))
    
    # Set size
    def setSize(self):
        self.state_size = len(self.states)
        self.input_size = len(self.inputs)
        
        
    # Calculate the state and input space helpers 
    def calculateSpaceHelpers(self):
        self.state_space_ngp = [1]*self.state_space_dim
        self.state_space_ipd = [1]*self.state_space_dim
        
        self.input_space_ngp = [1]*self.input_space_dim
        self.input_space_ipd = [1]*self.input_space_dim
        
        for i in range(self.state_space_dim):
            self.state_space_ngp[i] = int((self.state_space_upper_right[i] - self.state_space_lower_left[i])/self.state_space_etas[i] + 1)
            if(i != 0):
                self.state_space_ipd[i] = int(self.state_space_ipd[i-1]*self.state_space_ngp[i-1])

        for i in range(self.input_space_dim):
            self.input_space_ngp[i] = int((self.input_space_upper_right[i] - self.input_space_lower_left[i])/self.input_space_etas[i] + 1)
            if(i != 0):
                self.input_space_ipd[i] = int(self.input_space_ipd[i-1]*self.input_space_ngp[i-1])