import random

size = round(1000 + (-1000 + random.random()*2000))

file = open("controller.txt",'w')
for s in range(size):
    u = round(random.random()*64)
    file.write(str(s) + " " + str(u) + "\n")
    
file.close()
    