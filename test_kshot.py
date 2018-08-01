import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
import numpy as np
import os
import math
import argparse
import random
import cv2

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 1)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 5)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 1)
parser.add_argument("-e","--episode",type = int, default= 50000)
# parser.add_argument("-t","--test_episode", type = int, default = 1000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=1)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
parser.add_argument("-d","--display_query_num",type=int,default=5)
parser.add_argument("-t","--test_class",type=str,default='vending')
parser.add_argument("-modelf","--feature_encoder_model",type=str,default='models/feature_encoder_299999_1_way_5shot.pkl')
parser.add_argument("-modelr","--relation_network_model",type=str,default='models/relation_network_299999_1_way_5shot.pkl')
args = parser.parse_args()


# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
# TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
DISPLAY_QUERY = args.display_query_num
TEST_CLASS = args.test_class
FEATURE_MODEL = args.feature_encoder_model
RELATION_MODEL = args.relation_network_model

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        features = list(models.vgg16_bn(pretrained=True).features)
        self.layer1 = nn.Sequential(
                        nn.Conv2d(4,64,kernel_size=3,padding=1)
                        )
        self.features = nn.ModuleList(features)[1:]#.eval()
        # print (nn.Sequential(*list(models.vgg16_bn(pretrained=True).children())[0]))
        # self.features = nn.ModuleList(features).eval()

    def forward(self,x):
        results = []
        x = self.layer1(x)
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {4, 11, 21, 31, 41}:
                results.append(x)

        return x, results

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1024,512,kernel_size=3,padding=1),
                        nn.BatchNorm2d(512, momentum=1, affine=True),
                        nn.ReLU()
                        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(512,512,kernel_size=3,padding=1),
                        nn.BatchNorm2d(512, momentum=1, affine=True),
                        nn.ReLU()
                        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv1 = nn.Sequential(
                        nn.Conv2d(1024,512,kernel_size=3,padding=1),
                        nn.BatchNorm2d(512, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(512,512,kernel_size=3,padding=1),
                        nn.BatchNorm2d(512, momentum=1, affine=True),
                        nn.ReLU()
                        ) # 14 x 14
        self.double_conv2 = nn.Sequential(
                        nn.Conv2d(1024,256,kernel_size=3,padding=1),
                        nn.BatchNorm2d(256, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(256,256,kernel_size=3,padding=1),
                        nn.BatchNorm2d(256, momentum=1, affine=True),
                        nn.ReLU()
                        ) # 28 x 28
        self.double_conv3 = nn.Sequential(
                        nn.Conv2d(512,128,kernel_size=3,padding=1),
                        nn.BatchNorm2d(128, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(128,128,kernel_size=3,padding=1),
                        nn.BatchNorm2d(128, momentum=1, affine=True),
                        nn.ReLU()
                        ) # 56 x 56
        self.double_conv4 = nn.Sequential(
                        nn.Conv2d(256,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU()
                        ) # 112 x 112
        self.double_conv5 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(64,1,kernel_size=1,padding=0),
                        ) # 256 x 256

        # self.fc1 = nn.Sequential(
        #                 nn.Conv2d(128,1,kernel_size=1,padding=0)
        #                 )
        # self.upsample16 = nn.Upsample(scale_factor=32, mode='bilinear')

    def forward(self,x,concat_features):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.upsample(out) #block 1
        out = torch.cat((out, concat_features[-1]), dim=1)
        out = self.double_conv1(out)
        out = self.upsample(out) #block 2
        out = torch.cat((out, concat_features[-2]), dim=1)
        out = self.double_conv2(out)
        out = self.upsample(out) #block 3
        out = torch.cat((out, concat_features[-3]), dim=1)
        out = self.double_conv3(out)
        out = self.upsample(out) #block 4
        out = torch.cat((out, concat_features[-4]), dim=1)
        out = self.double_conv4(out)
        out = self.upsample(out) #block 5
        out = torch.cat((out, concat_features[-5]), dim=1)
        out = self.double_conv5(out)

        out = F.sigmoid(out)
        return out

def get_oneshot_batch():  #shuffle in query_images not done
    classes = list(range(1,21))
    chosen_classes = random.sample(classes, CLASS_NUM)
    support_images = np.zeros((CLASS_NUM,3,224,224), dtype=np.float32)
    support_labels = np.zeros((CLASS_NUM,CLASS_NUM,224,224), dtype=np.float32)
    query_images = np.zeros((CLASS_NUM*BATCH_NUM_PER_CLASS,3,224,224), dtype=np.float32)
    query_labels = np.zeros((CLASS_NUM*BATCH_NUM_PER_CLASS,CLASS_NUM,224,224), dtype=np.float32)
    class_cnt = 0
    for i in chosen_classes:
        imgnames = os.listdir('./fewshot/label/%s' % i)
        indexs = list(range(0,len(imgnames)))
        chosen_index = random.sample(indexs, SAMPLE_NUM_PER_CLASS + BATCH_NUM_PER_CLASS)
        j = 0
        for k in chosen_index:
            # process image
            image = cv2.imread('./fewshot/image/%s' % imgnames[k].replace('.png', '.jpg'))
            image = image[:,:,::-1] # bgr to rgb
            image = image / 255.0
            image = np.transpose(image, (2,0,1))
            # labels
            label = cv2.imread('./fewshot/label/%s/%s' % (i, imgnames[k]))[:,:,0]
            if j == 0:
                support_images[class_cnt] = image
                support_labels[class_cnt][class_cnt] = label
            else:
                query_images[class_cnt*BATCH_NUM_PER_CLASS+j-1] = image
                query_labels[class_cnt*BATCH_NUM_PER_CLASS+j-1][class_cnt] = label
            j += 1

        class_cnt += 1
    support_images_tensor = torch.from_numpy(support_images)
    support_labels_tensor = torch.from_numpy(support_labels)
    query_images_tensor = torch.from_numpy(query_images)
    query_labels_tensor = torch.from_numpy(query_labels)
    return support_images_tensor, support_labels_tensor, query_images_tensor, query_labels_tensor, chosen_classes

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors

    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0,0,0], [128,0,0], [0,128,0], [128,128,0],
                      [0,0,128], [128,0,128], [0,128,128], [128,128,128],
                      [64,0,0], [192,0,0], [64,128,0], [192,128,0],
                      [64,0,128], [192,0,128], [64,128,128], [192,128,128],
                      [0, 64,0], [128, 64, 0], [0,192,0], [128,192,0],
                      [0,64,128]])


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes

    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.

    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def decode_segmap(label_mask, plot=False):
    """Decode segmentation class labels into a color image

    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.

    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    label_colours = get_pascal_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, 21):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r #/ 255.0
    rgb[:, :, 1] = g #/ 255.0
    rgb[:, :, 2] = b #/ 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb




def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    # metatrain_character_folders,metatest_character_folders = tg.omniglot_character_folders()

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork()

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    if os.path.exists(FEATURE_MODEL):
        feature_encoder.load_state_dict(torch.load(FEATURE_MODEL))
        print("load feature encoder success")
    else:
        print('Can not load feature encoder: %s' % FEATURE_MODEL)
        stop
    if os.path.exists(RELATION_MODEL):
        relation_network.load_state_dict(torch.load(RELATION_MODEL))
        print("load relation network success")
    else:
        print('Can not load relation network: %s' % RELATION_MODEL)
        stop

    if not os.path.exists('tmpresult'):
        os.makedirs('tmpresult')


    # Step 3: build graph
    print("Testing...")

    imgnames = os.listdir('./fewshot/testimage/%s' % str(TEST_CLASS))
    print ('%s testing images in class %s' % (len(imgnames), TEST_CLASS))
    stick = np.zeros((224*4,224*5,3), dtype=np.uint8)
    support_image = np.zeros((SAMPLE_NUM_PER_CLASS, 3, 224, 224), dtype=np.float32)
    support_label = np.zeros((SAMPLE_NUM_PER_CLASS, 1, 224, 224), dtype=np.float32)
    supp_demo = np.zeros((224, 224*5,3), dtype=np.uint8)
    supplabel_demo = np.zeros((224, 224*5,3), dtype=np.uint8)

    for i, imgname in enumerate(imgnames):
        if i >= 5:
            break
        print ('Testing images %s / %s ' % (i, len(imgnames)))

        suppnames = os.listdir('./fewshot/support/%s/image' % str(TEST_CLASS))
        for j, suppname in enumerate(suppnames):
        #support image
            if j>= 5:
                break
            image = cv2.imread('./fewshot/support/%s/image/%s' % (str(TEST_CLASS), suppnames[j])).astype(np.float32)
            demo2 = image.copy()
            supp_demo[:,j*224:(j+1)*224,:] = demo2
            image = image[:,:,::-1] # bgr to rgb
            image = image / 255.0
            image = np.transpose(image, (2,0,1))
            image = image[np.newaxis,:]
            support_image[j] = image

            label = cv2.imread('./fewshot/support/%s/label/%s' % (str(TEST_CLASS), suppnames[j][:-4]+'.png')).astype(np.float32)
            demo_label = label.copy() * 255
            supplabel_demo[:,j*224:(j+1)*224,:] = demo_label
            label = label[:,:,0:1]
            label = np.transpose(label, (2,0,1))
            label = label[np.newaxis,:]
            support_label[j] = label

        image = np.concatenate((support_image,support_label), axis=1)

        #test image
        testimg = cv2.imread('./fewshot/testimage/%s/%s' % (str(TEST_CLASS),imgnames[i])).astype(np.float32)
        demo1 = testimg.copy()
        testimg = testimg[:,:,::-1] # bgr to rgb
        testimg = testimg / 255.0
        testimg = np.transpose(testimg, (2,0,1))
        testimg = testimg[np.newaxis,:]
        zeros = np.zeros((1,1,224,224), dtype=np.float32)
        testimg = np.concatenate((testimg, zeros), axis=1)

        #forward
        samples = torch.from_numpy(image)
        batches = torch.from_numpy(testimg)
        sample_features, _ = feature_encoder(Variable(samples).cuda(GPU))
        sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,512,7,7)
        sample_features = torch.sum(sample_features,1).squeeze(1) # 1*512*7*7
        batch_features, ft_list = feature_encoder(Variable(batches).cuda(GPU))
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
        batch_features_ext = torch.transpose(batch_features_ext,0,1)
        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,1024,7,7)
        output = relation_network(relation_pairs,ft_list).view(-1,CLASS_NUM,224,224)
        #get prediction
        pred = output.data.cpu().numpy()[0][0]
        pred_demo = (pred.copy() * 255).astype(np.uint8)
        pred_demo = cv2.cvtColor(pred_demo, cv2.COLOR_GRAY2RGB)
        pred[pred<=0.5] = 0
        pred[pred>0.5] = 1
        pred = (pred*255).astype(np.uint8)
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)

        #get stick image
        # stick[0:224,i*224:(i+1)*224,:] = demo2
        # stick[224:224*2,i*224:(i+1)*224,:] = demo_label
        stick[224*2:224*3,i*224:(i+1)*224,:] = demo1
        stick[224*3:224*4,i*224:(i+1)*224,:] = pred

    stick[0:224,:,:] = supp_demo
    stick[224:224*2,:,:] = supplabel_demo
    cv2.imwrite('./tmpresult/%s_test.png' % str(TEST_CLASS), stick)

if __name__ == '__main__':
    main()
