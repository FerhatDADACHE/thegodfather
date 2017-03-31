import classifier
import numpy as np
repdata="/home/synophride/S4/Mini_projet/Projet/public_data/"
train_data="crime_train.data"
labelname="crime_train.solution"

def watclass (s):
        lst=s.split(" ")
        for i in range(6):
            string=lst[i]
            i=(float(i))
            if(string=="1"):
                return i
            

            
xa = open(repdata+train_data).read()
X = np.fromstring(xa, sep=" ")

ya = open(repdata+labelname).readlines()
y = np.array(int)
j=0
for i in ya:
    y=np.append(y, watclass(i))
    j=j+1

cl = classifier.Classifier()

cl.fit_linear(X, y)

