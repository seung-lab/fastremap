import fastremap
import numpy as np 

x = np.ones((1024,1024,512), dtype=np.bool)
x = fastremap.asfortranarray(x)
print(x)
print(x.flags)
print(x.strides)





# @profile
# def run():
#   x = np.ones( (512,512,512), dtype=np.uint32, order='C')
#   x += 1
#   print(x.strides, x.flags)
#   y = np.asfortranarray(x)
#   print(x.strides, x.flags)

#   print("done.")

# run()