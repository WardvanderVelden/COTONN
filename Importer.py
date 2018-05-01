from PlainController import PlainController

# Importer class which will be responsible for reading in controller files and formatting them 
# such that they can be read into PlainControllers or BddControllers and used as neural network
# training data
class Importer:
    def __init__(self):
        self.filename = None


    # Read a plain controller into a format with which we can work in python
    def readPlainController(self, filename):
        self.filename = filename

        file = open(filename + ".scs",'r')

        if(file == None):
            print("Unable to open " + filename + ".scs")
            
        raw_str = file.read()
            
        con = PlainController()
        
        print(raw_str[:400])
        
        #Find locations of the dimensions, etas and borders of the controller
        dim_left_find = "MEMBER:DIM"
        dim_left = raw_str.index(dim_left_find)+len(dim_left_find)
        
        dim_right_find = "#VECTOR:ETA"
        dim_right = raw_str.index(dim_right_find)
        
        con.state_space_dim = raw_str[dim_left:dim_right]
        
        print("This is the dimension of the controller:", con.state_space_dim)
        
    
        etas_find = raw_str.index("#VECTOR:LOWER_LEFT")
        etas_left_1 = dim_right+len("#VECTOR:ETA#BEGIN:2 ")
                                 
        etas_right_2 = etas_find-len("#END ")
        etas_left_2 = int((etas_left_1+etas_right_2)/2)
        etas_right_1 = etas_left_2
                                   
        etas_left = raw_str[etas_left_1:etas_left_2]
        etas_right = raw_str[etas_right_1:etas_right_2]
        con.state_space_etas = [float(etas_left), float(etas_right)]
        
        print("These are the etas of the controller:", con.state_space_etas)
        
        bounds_find = raw_str.index("#VECTOR:UPPER_RIGHT")
        bounds_lower_left_left = etas_find+len("#VECTOR:LOWER_LEFT#BEGIN:2 ")
        bounds_lower_left_right = bounds_find - len("#END ")                                     
        bounds_upper_right_left = bounds_find+ len("#VECTOR:UPPER_RIGHT#BEGIN:2 ")
        bounds_upper_right_right = raw_str.index("#SCOTS:INPUT_SPACE")-len("#END ")
                                                                      
        bounds_lower_left_1 = raw_str[bounds_lower_left_left:int((bounds_lower_left_left+bounds_lower_left_right)/2)] 
        bounds_lower_left_2 = raw_str[int((bounds_lower_left_left+bounds_lower_left_right)/2):bounds_lower_left_right]
 
        bounds_lower_left = [float(bounds_lower_left_1),float(bounds_lower_left_2)]
        
        bounds_upper_right_1 = raw_str[bounds_upper_right_left:int((bounds_upper_right_right+bounds_upper_right_left)/2)]
        bounds_upper_right_2 = raw_str[int((bounds_upper_right_right+bounds_upper_right_left)/2):bounds_upper_right_right]
        
        bounds_upper_right = [float(bounds_upper_right_1), float(bounds_upper_right_2)]
                                                                        
        con.state_space_borders = [bounds_lower_left,bounds_upper_right]
        
        print("These are the borders of the controller:", con.state_space_borders)
        

        return con
    

    # Read a bdd controller
    def readBddController(self, filename):
        self.filename = filename

        bdd = open(filename + ".bdd",'r')
        support = open(filename + ".scs", 'r')

        if(bdd == None):
            print("Unable to open " + filename + ".bdd")

        if(support == None):
            print("Unable to open " + filename + ".scs")
            
imp = Importer()
imp.readPlainController("controllers/dcdc/controller")
