# importing libraries

import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
#pip install matplotlib-label-lines
from labellines import labelLines

#data preprocessing
data = pd.read_csv(r"../data/04_cricket_1999to2011.csv")
data.drop(data[(data['Innings'] == 2)].index, inplace=True)
df= data.copy(deep= True)
df.drop(data[(df['Wickets.in.Hand'] == 0)].index, inplace=True)
df5 = df.groupby(['Wickets.in.Hand','Over'])

# final data list[wicket, over remaining, run remaining]
final_data =[]
for name,group in df5:
    final_data.append([name[0],50-name[1], group.mean(axis = 0)['Runs.Remaining']])

# initializing the parameters
x = [30,10,20,30,40,50,60,70,80,90,100]    #initial guesses x[0] = L other Z(1) to Z(10)

# defining loss function
def loss(x,final_data):
    L = x[0]
    z = []
    los = []
    for i in range(len(final_data)):

        z0 = x[final_data[i][0]]
        z.append(z0 *(1-np.exp(-L*final_data[i][1]/z0)))
        los.append(final_data[i][2] - z[i])
    return los

params = leastsq(loss, x,final_data)[0]

los = np.array(loss(params, final_data))
avg_sq_loss = np.square(los).mean()

# plotting
L0 = params[0]
plt.figure(figsize = (15,10))
xx = np.linspace(0.1,50,501,endpoint = True)
for i in range(10,0,-1):
    z0_w = params[i]
    output = z0_w *(1-np.exp(-L0*xx/z0_w))
    plt.plot((50-xx),output,label = str(i))
plt.title('avg runs scored vs overs used for N wickets in hand',fontsize = 18)
plt.xlabel('Overs Used',fontsize = 18)
plt.ylabel('Average Runs Scored',fontsize = 18)
plt.legend()
plt.show()


L0 = params[0]
plt.figure(figsize = (15,10))
xx = np.linspace(0.1,50,501,endpoint = True)
for i in range(10,0,-1):
    z0_w = params[i]
    output = z0_w *(1-np.exp(-L0*xx/z0_w))
    plt.plot((50-xx),output,label = str(i))
plt.title('avg runs scored vs overs used for N wickets in hand',fontsize = 18)
plt.xlabel('Overs Used',fontsize = 18)
plt.ylabel('Average Runs Scored',fontsize = 18)
labelLines(plt.gca().get_lines(), zorder=2.5)
plt.show()

#final parameters
print(params)   # l,z(1),.....,z(10) 
print(avg_sq_loss)

