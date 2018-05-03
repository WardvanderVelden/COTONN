
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
        
    # Default getters and setters
    def getStateSpaceDim(self): return self.state_space_dim
    def getStateSpaceEtas(self): return self.state_space_etas
    def getStateSpaceBounds(self): return self.state_space_bounds
    def getInputSpaceDim(self): return self.input_space_dim
    def getInputSpaceEtas(self): return self.input_space_etas
    def getInputSpaceBounds(self): return self.input_space_bounds
    
    def getState(self, id): return self.states[id]
    def getInput(self, id): return self.inputs[id]

    def setStateSpaceDim(self, value): self.state_space_dim = value
    def setStateSpaceEtas(self, value): self.state_space_etas = value
    def setStateSpaceBounds(self, value): self.state_space_bounds = value
    def setInputSpaceDim(self, value): self.input_space_dim = value
    def setInputSpaceEtas(self, value): self.input_space_etas = value
    def setInputSpaceBounds(self, value): self.input_space_bounds = value

    def setStateInput(self, s, i):
        self.states.append(int(s))
        self.inputs.append(int(i))
        
    # Get the size of the controller
    def size(self):
        if(len(self.states) == len(self.inputs)):
            return len(self.states)
        print("Controller states length and inputs length seem to be deviating.");
        return 0
    
    # Functions to acces the information imbedded in the static controller
    def getInputFromID(self, id):
        for i in range(self.size()):
            if(self.getState(i) == id):
                return self.getInput(i)
        print("ID does not correspond to a state in the winning domain.")
        return None
    
    # Get highest state id contained in the controller
    def getHighestStateID(self):
        highest = 0
        for i in range(self.size()):
            state = self.getState(i)
            if(state > highest):
                highest = state
        return highest
    
    # Get highest input id contained in the controller
    def getHighestInputID(self):
        highest = 0
        for i in range(self.size()):
            input = self.getInput(i)
            if(input > highest):
                highest = input
        return highest
            

    





