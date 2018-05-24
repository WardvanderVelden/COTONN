import math
from time import gmtime, strftime

# Utility class with some formatting functions
class Utilities:
    def formatTime(self, time):
        h = math.floor(time / 3600)
        m = math.floor(time / 60) % 60
        s = time - h*3600 - m*60
        
        return str(h)+" hrs "+str(m)+" mins "+str(float("{0:.2f}".format(s)))+" secs"
    
    def formatBytes(self, b):
        byte_str = str(b) + " bytes"
        
        if(b >= 1e3 and b < 1e6):
            byte_str = str(float("{0:.4f}".format(b/1e3))) + " kB"
            #byte_str = str(b/1e3) + " kB"
            
        if(b >= 1e6 and b < 1e9):
            byte_str = str(float("{0:.4f}".format(b/1e6))) + " MB"
            #byte_str = str(b/1e6) + " MB"
            
        if(b >= 1e9 and b < 1e12):
            byte_str = str(float("{0:.4f}".format(b/1e9))) + " GB"
            #byte_str = str(b/1e9) + " GB"
            
        return byte_str
    
    def getTimestamp(self):
        return strftime("%y%m%d%H%M%S", gmtime())