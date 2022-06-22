import numpy as np 
import matplotlib.pyplot as plt 

def Error (X, Y, t, B):
    
    N = len(X)
    total_error = 0.0
    for i in range(N):
        total_error = total_error + 0.5*((t*X[i] + B) - Y[i])**2
    
    print ("Total error is :",  total_error)
    print ("MSE error is :",  (total_error/N))
    return 

def BGD(X, Y, t, B, alpha):
 
    dl_dt = 0.0 
    dl_dB = 0.0
        
    N = len(X)
   
    for i in range(N):
        
        dl_dB += ((t*X[i] + B) - Y[i]) 
        dl_dt += ((t*X[i] - B) - Y[i])* X[i]
        
    t = t - (1/N) * dl_dt * alpha
    B = B - (1/N) * dl_dB * alpha
    
    
    return float(t), float(B)

def Regression(X, Y, t, B, alpha, itera):
    
    for i in range(itera):
        t, B = BGD(X, Y, t, B, alpha)            
    return float(t), float(B)

alpha = 0.00001
                   
a = []
b = []
count = 0
with open(r"C:\Users\chouc\Desktop\ece\Stage\dm\data_lab1.txt","r") as data:
    for line in data:
        count = count +1
        coords = line.split()
        a.append(float(coords[0]))
        b.append(float(coords[1]))       
plt.figure()
plt.subplot(3,1,1)
plt.scatter(a, b) 

x = []
y = []
j = count*0.7
j = int(j)
itera = 10000000
for i in range (j) :
    x.append(float(a[i]))
    y.append(float(b[i]))
    

plt.subplot(3,1,2)
plt.scatter(x, y)
t,B = Regression(x,y,0,0, alpha, itera)
Y= []
Y = [element * t + B for element in  x] 
plt.plot(x, Y)

plt.subplot(3,1,3)

x_new = []
y_new = []

for i in range (j,count) :
    x_new.append(float(a[i]))
    y_new.append(float(b[i]))

plt.scatter(x_new, y_new)
t_new,B_new = Regression(x,y,0,0, alpha, itera)
Y_new= []
Y_new = [element * t_new + B_new for element in  x_new] 
plt.plot(x_new, Y_new)

plt.show()
print ("Errors for the training set:")
Error (x, y, t, B)
print ("Errors for the test set:")
Error (x_new, y_new, t, B)

         