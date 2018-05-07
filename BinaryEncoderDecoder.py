# Encoder Decoder class which will contain utility functions to encode and decode ordinary numbers into binary numbers
class BinaryEncoderDecoder:
    # Converts a ordinary number to a binary number with length binary numbers. 
    # The most significant bit is on the left, least significant bit on the right
    def ntob(self, number, length):
        conversion_str = "{0:0" + str(length) + "b}"
        binary =  conversion_str.format(number)
        if (len(binary) > length):
            a = len(binary)-length
            binary = binary[a:]
        return binary
    
    
    # Converts a binary number (represented as a string) to a ordinay redix-10 number
    def bton(self, binary):
        return int(binary, 2)
    
    
    # Converts a number to a binary number in the smaller amount of digets required
    def sntob(self, number):
        b = bin(number)
        return b[2:]
    
    
    # Converts a number into a binary array
    def ntoba(self, number, length):
        a = []
        conversion_str = "{0:0" + str(length) + "b}"
        binary =  conversion_str.format(number)
        if (len(binary) > length):
            a = len(binary)-length
            binary = binary[a:]
        
        for i in range(length):
            a.append(int(binary[i]))
            
        return a
    
    # Converts a binary array into a number
    def baton(self, array):
        binary = ""
        for i in range(len(array)):
            binary += str(array[i])
        return int(binary, 2)
    
    
    # Converts a float array binary estimation to binary array
    def etoba(self, x):
        a = []
        for i in range(len(x)):
            tmp = round(x[i])
            a.append(int(tmp))
        return a
    
    
    # Converts an estimation to number
    def eton(self, x):
        return self.baton(self.etoba(x))
        
        