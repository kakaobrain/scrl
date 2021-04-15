import random

import torch

    
class SCRLBoxGenerator(object):
    """To generate spatially matched box pairs in the two randomly augmented views.
    Ref: https://github.com/kakaobrain/scrl/issues/2
    """
    def __init__(self, input_size, min_size, num_patches_per_image, box_jittering, 
                 box_jittering_ratio, iou_threshold, grid_based_box_gen):
        self.input_size = input_size
        self.min_size = min_size
        self.num_patches_per_image = num_patches_per_image
        self.box_jittering = box_jittering
        self.box_jittering_ratio = box_jittering_ratio
        self.iou_threshold = iou_threshold
        self.grid_based_box_gen = grid_based_box_gen
        
    @classmethod
    def init_from_config(cls, cfg):
        return cls(
            input_size=cfg.augment.input_size,
            min_size=cfg.network.scrl.min_size,
            num_patches_per_image=cfg.network.scrl.num_patches_per_image,
            box_jittering=cfg.network.scrl.box_jittering,
            box_jittering_ratio=cfg.network.scrl.jittering_ratio,
            iou_threshold=cfg.network.scrl.iou_threshold,
            grid_based_box_gen=cfg.network.scrl.grid_based_box_gen,
        )

    def generate(self, transf):
        # spatial consistency matching requires the transform information of images
        assert len(transf) != 0

        # order of the batch_trans: y, x, h, w, flipped
        spatial_boxes1 = []
        spatial_boxes2 = []
        src_boxes1 = []
        src_boxes2 = []
        n_samples = transf[0].size(0)

        for batch_idx, (t1, t2) in enumerate(zip(transf[0], transf[1])):
            # find the intersection of two augmented images
            int_l = max(t1[1], t2[1]).item()
            int_r = min(t1[1] + t1[3], t2[1] + t2[3]).item()
            int_t = max(t1[0], t2[0]).item()
            int_b = min(t1[0] + t1[2], t2[0] + t2[2]).item()

            # not exist the int area
            if int_l >= int_r or int_t >= int_b:
                continue

            scale_w1 = self.input_size / t1[3].item()
            scale_h1 = self.input_size / t1[2].item()
            scale_w2 = self.input_size / t2[3].item()
            scale_h2 = self.input_size / t2[2].item()

            scale_wmin = min(scale_w1, scale_w2)
            scale_hmin = min(scale_h1, scale_h2)
            scale_wmin_inv = 1 / scale_wmin
            scale_hmin_inv = 1 / scale_hmin

            int_w_scaled = round((int_r - int_l) * scale_wmin)
            int_h_scaled = round((int_b - int_t) * scale_hmin)

            if self.min_size >= int_w_scaled or self.min_size >= int_h_scaled:
                continue
            
            div_w_range = int_w_scaled / self.min_size
            div_h_range = int_h_scaled / self.min_size
            for i in range(self.num_patches_per_image):
                for _ in range(50):  # try 50 times untill IoU condition meets
                    if self.grid_based_box_gen:
                        # grid-level box generation
                        div_w = random.randint(1, int(div_w_range))
                        div_h = random.randint(1, int(div_h_range))
                        grid_x = random.randint(0, div_w)
                        grid_y = random.randint(0, div_h)

                        grid_w = int_w_scaled / div_w
                        grid_h = int_h_scaled / div_h
                        box_w = random.uniform(self.min_size, grid_w)
                        box_h = random.uniform(self.min_size, grid_h)
                        box_x = random.uniform(0, grid_w - box_w) + (grid_x * grid_w)
                        box_y = random.uniform(0, grid_h - box_h) + (grid_y * grid_h)
                    else:
                        # random box generation
                        box_w = random.uniform(self.min_size, int_w_scaled)
                        box_h = random.uniform(self.min_size, int_h_scaled)
                        box_x = random.uniform(0, int_w_scaled - box_w)
                        box_y = random.uniform(0, int_h_scaled - box_h)

                    box1_l = box_x * scale_wmin_inv * scale_w1 + \
                        (int_l - t1[1].item()) * scale_w1
                    box1_r = box1_l + box_w * scale_wmin_inv * scale_w1
                    box1_t = box_y * scale_hmin_inv * scale_h1 + \
                        (int_t - t1[0].item()) * scale_h1
                    box1_b = box1_t + box_h * scale_hmin_inv * scale_h1

                    box2_l = box_x * scale_wmin_inv * scale_w2 + \
                        (int_l - t2[1].item()) * scale_w2
                    box2_r = box2_l + box_w * scale_wmin_inv * scale_w2
                    box2_t = box_y * scale_hmin_inv * scale_h2 + \
                        (int_t - t2[0].item()) * scale_h2
                    box2_b = box2_t + box_h * scale_hmin_inv * scale_h2

                    if t1[4]:
                        box1_l = self.input_size - box1_r
                        box1_r = self.input_size - box1_l

                    if t2[4]:
                        box2_l = self.input_size - box2_r
                        box2_r = self.input_size - box2_l

                    if self.box_jittering:
                        src_box2 = [batch_idx + n_samples, 
                                    box2_l, box2_t, box2_r, box2_b]
                        box2_t, box2_l, box2_b, box2_r = jitter_box(
                            box2_t, box2_l, box2_b, box2_r,
                            self.box_jittering_ratio, self.input_size
                        )
                        if t1[4]^t2[4]:
                            src_box1 = [
                                batch_idx, box1_l-(box2_r-src_box2[3])/scale_w2*scale_w1,
                                box1_t+(box2_t-src_box2[2])/scale_h2*scale_h1,
                                box1_r-(box2_l-src_box2[1])/scale_w2*scale_w1,
                                box1_b+(box2_b-src_box2[4])/scale_h2*scale_h1,
                            ]
                        else:
                            src_box1 = [
                                batch_idx, box1_l+(box2_l-src_box2[1])/scale_w2*scale_w1,
                                box1_t+(box2_t-src_box2[2])/scale_h2*scale_h1,
                                box1_r+(box2_r-src_box2[3])/scale_w2*scale_w1,
                                box1_b+(box2_b-src_box2[4])/scale_h2*scale_h1,
                            ]
                    else:
                        src_box1 = [batch_idx, box1_l, box1_t, box1_r, box1_b]
                        src_box2 = [batch_idx + n_samples, box2_l, box2_t, box2_r, box2_b]

                    if i == 0 or self.iou_threshold == 1.0:
                        break

                    # reject patches if the generated patch overlaps more than 
                    # the specific IoU threshold
                    max_iou = 0.
                    for box in spatial_boxes1[-i:]:
                        iou = bbox_iou([box1_l, box1_t, box1_r, box1_b],
                                        [box[1], box[2], box[3], box[4]])
                        max_iou = max(max_iou, iou)
                        if max_iou > self.iou_threshold:
                            break

                    if max_iou < self.iou_threshold:
                        break

                # append a spatial box
                spatial_box1 = [batch_idx, box1_l, box1_t, box1_r, box1_b]
                spatial_boxes1.append(clip_box(spatial_box1, self.input_size))
                src_boxes1.append(clip_box(src_box1, self.input_size))

                # note that batch index for view2 is re-calibrated
                spatial_box2 = [batch_idx + n_samples, box2_l, box2_t, box2_r, box2_b]
                spatial_boxes2.append(clip_box(spatial_box2, self.input_size))
                src_boxes2.append(clip_box(src_box2, self.input_size))

        spatial_boxes = [torch.tensor(spatial_boxes1),
                        torch.tensor(spatial_boxes2)]

        return spatial_boxes


def bbox_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def jitter_box(box_t, box_l, box_b, box_r, box_jittering_ratio, input_size):
    box_w = box_r - box_l
    box_h = box_b - box_t

    jitter = [random.uniform(1. - box_jittering_ratio, 
                             1. + box_jittering_ratio)
              for _ in range(4)]

    box_l = float(box_l + box_w * (jitter[0] - 1))
    box_t = float(box_t + box_h * (jitter[1] - 1))
    box_r = float(box_l + box_w * jitter[2])
    box_b = float(box_t + box_h * jitter[3])

    return box_t, box_l, box_b, box_r


def clip_box(box_with_inds, input_size):
    box_with_inds[1] = float(max(0, box_with_inds[1]))
    box_with_inds[2] = float(max(0, box_with_inds[2]))
    box_with_inds[3] = float(min(input_size, box_with_inds[3]))
    box_with_inds[4] = float(min(input_size, box_with_inds[4]))

    return box_with_inds
