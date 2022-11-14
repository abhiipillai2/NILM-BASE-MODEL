import numpy as np

# A=np.array([1,2,3,4,5,6,7,8,9,10])

# B = np.reshape(A,[5,3])

# print(B)


def sequesnceGenerator(arr,n):
    i=0
    arr1 = []
    temp = []
    while(i < len(arr)):
        if(i%n == 0 and i != 0):
            arr1.append(temp)
            temp = [arr[i]]
        else:
            temp.append(arr[i])
        i+=1
    #arr1.append(temp)
    return arr1

arr = [1,2,3,4,5,6,7,8,9,10]
A=np.array(sequesnceGenerator(arr,2))

print(A)