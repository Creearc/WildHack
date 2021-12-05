import cv2
import numpy as np


pr_masks = [np.zeros((out.shape[0], out.shape[1],1), np.uint8) for i in range(4)]
part_scale = 512
step = 256

for i in range(0, out.shape[0] - part_scale, step):
    for j in range(0, out.shape[1] - part_scale, step):
        img = out[i : i + part_scale,
                  j : j + part_scale].copy()

        image = torch.from_numpy(img).cuda().float()
        batch = FastBaseTransform()(image.unsqueeze(0))
        preds = net(batch)

        img_numpy, pred_masks = prep_display(preds, image, None, None, undo_transform=False)
        #print(save_path.split('.')[0] + str(time.time()) + '.'  + save_path.split('.')[1])
        #cv2.imwrite(save_path.split('.')[0] + str(time.time()) + '.' + save_path.split('.')[1], img_numpy)
        for k in range(4):
            part = pr_masks[k][i : i + part_scale,
                               j : j + part_scale]
            pr_masks[k][i : i + part_scale,
                        j : j + part_scale] = np.where(part == 0, pred_masks[k], part)
            #cv2.imshow(str(k), pred_masks[k])
            #print(k, pr_masks[k].any()!=0)
            #cv2.imshow(str(k), resize(pr_masks[k], h=720))
        #cv2.waitKey(0)
