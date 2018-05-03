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
        