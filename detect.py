import os
import cv2
import csv
from model.fcos import FCOSDetector
import torch
from torchvision import transforms
import numpy as np
from dataset.VOC_dataset import VOCDataset
import time
import matplotlib.patches as patches
import  matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from PIL import  Image

def preprocess_img(image,input_ksize):
    '''
    resize image and bboxes 
    Returns
    image_paded: input_ksize  
    bboxes: [None,4]
    '''
    min_side, max_side    = input_ksize
    h,  w, _  = image.shape

    smallest_side = min(w,h)
    largest_side=max(w,h)
    scale=min_side/smallest_side
    if largest_side*scale>max_side:
        scale=max_side/largest_side
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))
    '''
    pad_w=pad_h=0
    if nw%32:
        pad_w=32-nw%32
    if nh%32:
        pad_h=32-nh%32
    
    image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
    image_paded[:nh, :nw, :] = image_resized
    return image_paded
    '''
    return image_resized
    
def convertSyncBNtoBN(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features,
                                               module.eps, module.momentum,
                                               module.affine,
                                               module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
    for name, child in module.named_children():
        module_output.add_module(name,convertSyncBNtoBN(child))
    del module
    return module_output

def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result

# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def detect_image(image_path, model_path, class_list):

    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    class Config():
        #backbone
        pretrained=False
        freeze_stage_1=True
        freeze_bn=True

        #fpn
        fpn_out_channels=256
        use_p5=True
        
        #head
        class_num=5
        use_GN_head=True
        prior=0.01
        add_centerness=True
        cnt_on_reg=False

        #training
        strides=[8,16,32,64,128]
        limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]

        #inference
        score_threshold=0.3
        nms_iou_threshold=0.4
        max_detection_boxes_num=300

    model=FCOSDetector(mode="inference",config=Config)
    # model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # print("INFO===>success convert BN to SyncBN")
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    # model=convertSyncBNtoBN(model)
    # print("INFO===>success convert SyncBN to BN")
    model=model.eval()
    print("===>success loading model")

    for img_name in os.listdir(image_path):

        image = cv2.imread(os.path.join(image_path, img_name))
        if image is None:
            continue

        img=preprocess_img(image,[922,1640])
        image=transforms.ToTensor()(img)
        image=transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225],inplace=True)(image)
        

        start_t=time.time()
        with torch.no_grad():
            out=model(image.unsqueeze_(dim=0))
        end_t=time.time()
        cost_t=1000*(end_t-start_t)
        print("===>success processing img, cost time %.2f ms"%cost_t)

        scores,classes,boxes=out

        boxes=boxes[0].cpu().numpy().tolist()
        classes=classes[0].cpu().numpy().tolist()
        scores=scores[0].cpu().numpy().tolist()
        print(boxes)
        print(classes)
        print(scores)
        
        for i,box in enumerate(boxes):
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            label_name = labels[int(classes[i])-1]
            score = scores[i]
            caption = '{} {:.3f}'.format(label_name, score)    
            draw_caption(img, (x1, y1, x2, y2), caption)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)

            cv2.imshow('detections', img)
            cv2.waitKey(0)

if __name__=="__main__":
    
    detect_image("./test_images", "./checkpoint/model_100.pth", "./class4_classlist.csv")
