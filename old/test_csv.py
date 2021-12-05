import cv2
import numpy as np
import os

def rle(inarray):
    """ run length encoding. Partial credit to R rle function. 
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)                # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        return(z, p, ia[i])

def rlencode_mask(mask):
    l,s,v = rle(mask.flatten()) # length, starts, values
    l,s = l[v], s[v]
    encoded = ' '.join([' '.join(map(str, e)) for e in zip(s, l)])
    if not encoded:
        encoded = '0 0'
    return encoded



predicted_mask = cv2.imread('dataset/example/00_net.png', cv2.IMREAD_UNCHANGED)
predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_RGB2GRAY)
class_name = 'net'
class_id = 0
i = 0
encoded = f'{class_name}_{i},' + rlencode_mask(predicted_mask != class_id) + f'\n'

print(encoded[:1000])


predicted_mask = cv2.threshold(predicted_mask, 0, 255, cv2.THRESH_BINARY)[1]


encoded = f'{class_name}_{i},' + rlencode_mask(predicted_mask != 0) + f'\n'

print(encoded[:1000])

lines = ['Type_Id,Mask\n']
classes = ['wood', 'metall', 'net', 'plastic']
for i in range(32):
    for j in range(4):
        class_name = classes[j]
        f = f'{i:02}_{class_name}.png'
        print(f)
        predicted_mask = cv2.imread('dataset/example/{}'.format(f), cv2.IMREAD_UNCHANGED)
        if predicted_mask is None:
            encoded = f'{class_name}_{i},' + '0 0' + f'\n'
        else:          
            predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_RGB2GRAY)
            predicted_mask = cv2.threshold(predicted_mask, 0, 255, cv2.THRESH_BINARY)[1]
            encoded = f'{class_name}_{i},' + rlencode_mask(predicted_mask != 0) + f'\n'
        lines.append(encoded)
with open('baseline_solution.csv', 'w') as f:
    f.writelines(lines)


   
##files = os.listdir('dataset/example/')
##print(files)
##
##lines = ['Type_Id,Mask\n']
##
##for f in files:
##    if any(x in f for x in ['net', 'wood', 'metal', 'plastic']):
##        print(f)
##        i = int(f.split('_')[0])
##        class_name = f.split('_')[1].split('.')[0]
##        predicted_mask = cv2.imread('dataset/example/{}'.format(f), cv2.IMREAD_UNCHANGED)
##        predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_RGB2GRAY)
##        predicted_mask = cv2.threshold(predicted_mask, 0, 255, cv2.THRESH_BINARY)[1]
##        
##        encoded = f'{class_name}_{i},' + rlencode_mask(predicted_mask != 0) + f'\n'
##        lines.append(encoded)
##
##with open('baseline_solution.csv', 'w') as f:
##    f.writelines(lines)
