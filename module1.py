
import re
import numpy as np
import os
def  find_similar():
    fp=open("D:/data/movie/ratings_final.txt","rb")
    r=fp.readlines()
    fp.close()


    a=np.zeros((6036,4006))


    #print(a)

    sum=[]


    z=[]

    for i in r:
        index=re.findall("\d+",str(i),re.S)
        #b.append(int(index[1]))
        z.append(int(index[0])) 
        # print(int(index[0]),int(index[1]))
        a[int(index[0])][int(index[1])]=int(index[2])

    print(a)


    def cosine_distance(a, b):
          if a.shape != b.shape:
              raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
          if a.ndim==1:
              a_norm = np.linalg.norm(a)
              b_norm = np.linalg.norm(b)
          elif a.ndim==2:
              a_norm = np.linalg.norm(a, axis=1, keepdims=True)
              b_norm = np.linalg.norm(b, axis=1, keepdims=True)
          else:
             raise RuntimeError("array dimensions {} not right".format(a.ndim))
          similiarity = np.dot(a, b.T)/(a_norm * b_norm) 
          dist = 1. - similiarity
          return dist

    print(cosine_distance(a,a))
    a=cosine_distance(a,a)
    print(a.shape)
    similarity_dict = {}
    p=0
    for i in a:
        #print(i)
        b=i.copy()
       # print(len(i))
        i.sort()
        #print(np.where(b==i[2]))
        #print(np.where(b==i[2])[0][0])
        similarity_dict[str(p)]=np.where(b==i[2])[0][0]
        p=p+1
    
  
    return similarity_dict
   