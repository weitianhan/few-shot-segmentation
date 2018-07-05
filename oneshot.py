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
parser.add_argument("-s","--sample_num_per_class",type = int, default = 1)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 9)
parser.add_argument("-e","--episode",type = int, default= 500000)
parser.add_argument("-t","--test_episode", type = int, default = 1000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
parser.add_argument("-d","--display_query_num",type=int,default=5)
args = parser.parse_args()


# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = 1
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
DISPLAY_QUERY = args.display_query_num

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        return out # 64

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1024,256,kernel_size=3,padding=1),
                        nn.BatchNorm2d(256, momentum=1, affine=True),
                        nn.ReLU()
                        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(256,128,kernel_size=3,padding=1),
                        nn.BatchNorm2d(128, momentum=1, affine=True),
                        nn.ReLU()
                        )
        self.fc1 = nn.Sequential(
                        nn.Conv2d(128,1,kernel_size=1,padding=0)
                        )
        self.upsample16 = nn.Upsample(scale_factor=32, mode='bilinear')

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc1(out)
        out = self.upsample16(out)
        out = F.sigmoid(out)
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

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

    #read pre-trained network here
    vgg16 = models.vgg16_bn(pretrained=True)
    feature_encoder = nn.Sequential(*list(vgg16.children())[0])
    relation_network = RelationNetwork(FEATURE_DIM,RELATION_DIM)

    # feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=EPISODE//10,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=EPISODE//10,gamma=0.5)

    if os.path.exists(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str("./models/omniglot_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/omniglot_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load relation network success")

    # Step 3: build graph
    print("Training...")

    last_accuracy = 0.0

    for episode in range(EPISODE):

        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training
        # degrees = random.choice([0,90,180,270])
        # task = tg.OmniglotTask(metatrain_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        # sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False,rotation=degrees)
        # batch_dataloader = tg.get_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True,rotation=degrees)


        # sample datas
        # samples,sample_labels = sample_dataloader.__iter__().next()
        # batches,batch_labels = batch_dataloader.__iter__().next()
        samples, sample_labels, batches, batch_labels, chosen_classes = get_oneshot_batch()
        # print (samples.size(), sample_labels.size(), batches.size())
        # print (type(samples), type(sample_labels))
        # stop

        # calculate features
        sample_features = feature_encoder(Variable(samples).cuda(GPU)) # 5x64*5*5
        batch_features = feature_encoder(Variable(batches).cuda(GPU)) # 20x64*5*5
        # print (sample_features.size(), batch_features.size())

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        batch_features_ext = torch.transpose(batch_features_ext,0,1)
        # print (sample_features_ext.size(), batch_features_ext.size())



        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,1024,7,7)
        output = relation_network(relation_pairs).view(-1,CLASS_NUM,224,224)
        # print (output.size())
        # stop
        # print (relation_pairs.size())
        # stop

        mse = nn.MSELoss().cuda(GPU)
        # one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1)).cuda(GPU)
        loss = mse(output,Variable(batch_labels).cuda(GPU))


        # training

        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(),0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(),0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()

        if (episode+1)%10 == 0:
                print("episode:",episode+1,"loss",loss.cpu().data.numpy())

        if not os.path.exists('result'):
            os.makedirs('result')

        # training result visualization
        if (episode+1)%1000 == 0:
            support_output = np.zeros((224*2, 224*CLASS_NUM, 3), dtype=np.uint8)
            query_output = np.zeros((224*3, 224*DISPLAY_QUERY, 3), dtype=np.uint8)
            extra = np.zeros((224*3, 224*CLASS_NUM, 3), dtype=np.uint8)
            chosen_query = random.sample(list(range(0,BATCH_NUM_PER_CLASS)), DISPLAY_QUERY)

            for i in range(CLASS_NUM):
                supp_img = (np.transpose(samples.numpy()[i],(1,2,0))*255).astype(np.uint8)[:,:,::-1]
                support_output[0:224,i*224:(i+1)*224,:] = supp_img
                supp_label = sample_labels.numpy()[i][i]
                supp_label[supp_label!=0] = chosen_classes[i]
                supp_label = decode_segmap(supp_label)
                support_output[224:224*2, i*224:(i+1)*224,:] = supp_label

                for cnt, x in enumerate(chosen_query):
                    query_img = (np.transpose(batches.numpy()[x],(1,2,0))*255).astype(np.uint8)[:,:,::-1]
                    query_output[0:224,cnt*224:(cnt+1)*224,:] = query_img
                    query_label = batch_labels.numpy()[x][0] #only apply to one-way setting
                    query_label[query_label!=0] = chosen_classes[i]
                    query_label = decode_segmap(query_label)
                    query_output[224:224*2, cnt*224:(cnt+1)*224,:] = query_label

                    query_pred = output.detach().cpu().numpy()[x][0]
                    query_pred = (query_pred*255).astype(np.uint8)
                    result = np.zeros((224,224,3), dtype=np.uint8)
                    result[:,:,0] = query_pred
                    result[:,:,1] = query_pred
                    result[:,:,2] = query_pred
                    query_output[224*2:224*3, cnt*224:(cnt+1)*224,:] = result
            extra = query_output.copy()
            for i in range(CLASS_NUM):
                for cnt, x in enumerate(chosen_query):
                    extra_label = batch_labels.numpy()[x][0]
                    extra_label[extra_label!=0] = 255
                    result1 = np.zeros((224,224,3), dtype=np.uint8)
                    result1[:,:,0] = extra_label
                    result1[:,:,1] = extra_label
                    result1[:,:,2] = extra_label
                    extra[224*2:224*3, cnt*224:(cnt+1)*224,:] = result1
            cv2.imwrite('result/%s_query.png' % episode, query_output)
            cv2.imwrite('result/%s_show.png' % episode, extra)
            cv2.imwrite('result/%s_support.png' % episode, support_output)

        #save models
        if (episode+1) % 100000 == 0:
            torch.save(feature_encoder.state_dict(),str("./models/feature_encoder_" + str(episode) + '_' + str(CLASS_NUM) +"_way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
            torch.save(relation_network.state_dict(),str("./models/relation_network_"+ str(episode) + '_' + str(CLASS_NUM) +"_way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
            print("save networks for episode:",episode)

        # if (episode+1)%5000 == 0:
        #
        #     # test
        #     print("Testing...")
        #     total_rewards = 0
        #
        #     for i in range(TEST_EPISODE):
        #         degrees = random.choice([0,90,180,270])
        #         # task = tg.OmniglotTask(metatest_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,SAMPLE_NUM_PER_CLASS,)
        #         # sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False,rotation=degrees)
        #         # test_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="test",shuffle=True,rotation=degrees)
        #
        #         sample_images,sample_labels = sample_dataloader.__iter__().next()
        #         test_images,test_labels = test_dataloader.__iter__().next()
        #
        #         # calculate features
        #         sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64
        #         test_features = feature_encoder(Variable(test_images).cuda(GPU)) # 20x64
        #
        #         # calculate relations
        #         # each batch sample link to every samples to calculate relations
        #         # to form a 100x128 matrix for relation network
        #         sample_features_ext = sample_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        #         test_features_ext = test_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        #         test_features_ext = torch.transpose(test_features_ext,0,1)
        #
        #         relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,5,5)
        #         relations = relation_network(relation_pairs).view(-1,CLASS_NUM)
        #
        #
        #         _,predict_labels = torch.max(relations.data,1)
        #
        #         rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(CLASS_NUM)]
        #
        #         total_rewards += np.sum(rewards)
        #
        #     test_accuracy = total_rewards/1.0/CLASS_NUM/TEST_EPISODE
        #
        #     print("test accuracy:",test_accuracy)
        #
        #     if test_accuracy > last_accuracy:
        #
        #         # save networks
        #         torch.save(feature_encoder.state_dict(),str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
        #         torch.save(relation_network.state_dict(),str("./models/omniglot_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
        #
        #         print("save networks for episode:",episode)
        #
        #         last_accuracy = test_accuracy


if __name__ == '__main__':
    main()
