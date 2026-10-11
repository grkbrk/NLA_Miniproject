import numpy as np
# n1 = 2
# n2 = 2
# L = np.meshgrid(np.linspace(-1, 1, n1), np.linspace(-1, 1, n2))

# print(L.flatten())

# # Source - https://stackoverflow.com/a/63693272
# # Posted by Serenity Rising
# # Retrieved 2026-10-10, License - CC BY-SA 4.0

# answer = np.stack([x for x in np.ndindex(2,2)]).reshape(2,2,2)
# print(answer.flatten())

n1 =  3
n2 = 3
n = n1*n2

be1 = [-1, 1]                          #Bound x coord
be2 = [-1, 1]                          #Bound y coord
e1 = np.linspace(be1[0],be1[1],n1)     #x array (equidistant)
e2 = np.linspace(be2[0],be2[1],n2)     #y array (equidistant)
e1vv, e2vv = np.meshgrid(e1,e2,indexing='xy')         #2D coordinates of x,y. vx= all x val, vy= all y val

e1v = e1vv.flatten()  #Flattened x coordinates
e2v = e2vv.flatten()  #Flattened y coordinates

v = np.concatenate([e1v, e2v], axis=0).flatten()

print("v: ",v)
