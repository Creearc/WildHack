import cv2
import numpy as np
import os

color_arr = [(0,0,255),(255,124,0),(255,255,0),(255,0,0)]

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

def resize(img, w=None, h=None):
    sh, sw = img.shape[:2]
    if not (w is None):
        k = sw / w
    elif not (h is None):
        k = sh / h
    w, h = int(sw // k), int(sh // k)
    return cv2.resize(img, (w, h))

lines = ['Type_Id,Mask\n']
classes = ['wood', 'metall', 'net', 'plastic']
kernel = np.ones((5,5),np.uint8)

for i in range(21,32):
    #f = f'odm_orthophoto (1).png'
    f = f'{i:02}_image.JPG'
    pic = cv2.imread('D:/hackathon/WildHack/test_dataset/pr/{}'.format(f), cv2.IMREAD_UNCHANGED)
    if pic is None:
        break
    else:
        out = pic.copy()
    for j in range(4):
        classname = classes[j]
        #f = f'odm_orthophoto (1)_mask_{j}.png'
        f = f'{i:02}_image_mask_{j}.png'
        print(f)
        predicted_mask = cv2.imread('pr1/{}'.format(f), cv2.IMREAD_UNCHANGED)
        
        if predicted_mask is None:
            encoded = f'{classname}_{i},' + '0 0' + f'\n'
        else:
            #predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_RGB2GRAY)
            #predicted_mask = cv2.erode(predicted_mask,kernel,iterations = 1)
            predicted_mask = cv2.dilate(predicted_mask,kernel,iterations = 4)
            predicted_mask = cv2.threshold(predicted_mask, 0, 255, cv2.THRESH_BINARY)[1]
            predicted_masks_new = np.zeros((predicted_mask.shape[0],predicted_mask.shape[1],3), np.uint8)
            
            #predicted_masks_new[:,:,0] = predicted_mask.copy()
            
            #predicted_masks_new = np.where((predicted_masks_new > 200), 200, predicted_masks_new)
            for c in range(3):
                predicted_masks_new[:,:,c] = np.where(predicted_mask != 0, color_arr[j][c], 0)

            #out = cv2.add(predicted_masks_new, out)
            out = np.where((predicted_masks_new == (0,0,0)), out, predicted_masks_new)
                        
            #encoded = f'{classname}_{i},' + rlencode_mask(predicted_mask != 0) + f'\n'
            if j == 2:
                encoded = f'{classname}_{i},' + '0 0' + f'\n'#class_id) + f'\n'
            else:
                encoded = f'{classname}_{i},' + rlencode_mask(predicted_mask != 0) + f'\n'#class_id) + f'\n'
        
        lines.append(encoded)
    #cv2.imshow(f'im_{i}', resize(out, h=720))
    cv2.imwrite(f'masked_output_images/out_{i}.png', out)
    #cv2.waitKey(0)
with open('nikita_solution_fresh_new_privat.csv', 'w') as f:
    f.writelines(lines)
