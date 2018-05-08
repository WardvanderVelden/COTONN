
# PlainController class which will hold the controller that can then be accessed in order to read training
# data for the neural network
class StaticController:
    def __init__(self):
        self.state_space_dim = None
        self.state_space_etas = None
        self.state_space_bounds = None

        self.input_space_dim = None
        self.input_space_etas = None
        self.input_space_bounds = None

        self.states = []
        self.inputs = []
        
        self.state_size = 0
        self.input_size = 0
        
    # Getters
    def getStateSpaceDim(self): return self.state_space_dim
    def getStateSpaceEtas(self): return self.state_space_etas
    def getStateSpaceBounds(self): return self.state_space_bounds
    def getInputSpaceDim(self): return self.input_space_dim
    def getInputSpaceEtas(self): return self.input_space_etas
    def getInputSpaceBounds(self): return self.input_space_bounds
    
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
        
    # Setters
    def setStateSpaceDim(self, value): self.state_space_dim = value
    def setStateSpaceEtas(self, value): self.state_space_etas = value
    def setStateSpaceBounds(self, value): self.state_space_bounds = value
    def setInputSpaceDim(self, value): self.input_space_dim = value
    def setInputSpaceEtas(self, value): self.input_space_etas = value
    def setInputSpaceBounds(self, value): self.input_space_bounds = value

    def setStateInput(self, s, i):
        self.states.append(int(s))
        self.inputs.append(int(i))
        
    def setSize(self):
        self.state_size = len(self.states)
        self.input_size = len(self.inputs)
        
    
    
    
            

    





