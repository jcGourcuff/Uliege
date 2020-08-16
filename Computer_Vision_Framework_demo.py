import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torchvision
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from dataManager.load import DataSets
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from utils.display import Graphs
from nilm.imageGen_and_labelling import LabeledImagesMaker


"""""""""
This script puts in place the whole computer vision framework.
However, due to heavy computational requirements for computing features with resnet50, the testing is actually done on google colab gpus.
"""""""""

columns_drops = {'1': ['oven_1', 'kitchen_outlets_1',
                       'washer_dryer_2', 'electric_heat_1', 'stove_1']}

classes_modes = {'all': ['residuals', 'oven', 'refrigerator', 'dishwasher', 'kitchen_outlets', 'lighting', 'washer_dryer', 'microwave', 'bathroom_gfi'],
                 'minimal': ['refrigerator', 'dishwasher', 'lighting', 'washer_dryer'],
                 'minimal_plus_kitchen': ['refrigerator', 'dishwasher', 'lighting', 'washer_dryer', 'kitchen_outlets'],
                 'hybrid': ['oven', 'refrigerator', 'dishwasher', 'kitchen_outlets', 'lighting', 'washer_dryer', 'microwave', 'bathroom_gfi']}
classes = classes_modes['minimal']
lab_encoder = OneHotEncoder(sparse=False)
lab_encoder.fit([[name] for name in classes])

indexes_to_keep = {'2': [[0, 20010]], '3': [[0, 17500], [51000, 60000]], '4': [
    [5800, 22500], [51600, 69089]], '5': [], '6': [[4800, 16100], [25100, 27000], [28200, 33453]]}

Window_size = 1440
y_lim = 6000


class BoxNet(nn.Module):
    """
    Implements the NN that outputs the proba at time t that the corresponding app is on.
    """

    def __init__(self, output_size):
        super(BoxNet, self).__init__()
        self.output_size = output_size

        self.res_1 = nn.Sequential(nn.Dropout(inplace=False),
                                   nn.Linear(2048, 2048),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm1d(2048))

        self.res_2 = nn.Sequential(nn.Dropout(inplace=False),
                                   nn.Linear(2048, 2048),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm1d(2048))

        self.res_3 = nn.Sequential(nn.Dropout(inplace=False),
                                   nn.Linear(2048, 2048),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm1d(2048))

        self.res_4 = nn.Sequential(nn.Dropout(inplace=False),
                                   nn.Linear(2048, 2048),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm1d(2048))

        self.res_5 = nn.Sequential(nn.Dropout(inplace=False),
                                   nn.Linear(2048, 2048),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm1d(2048))

        self.fc = nn.Sequential(nn.Dropout(inplace=False),
                                nn.Linear(2048, self.output_size))

    def forward(self, x):
        x = x + self.res_1(x)
        x = x + self.res_2(x)
        x = x + self.res_3(x)
        x = x + self.res_4(x)
        x = x + self.res_5(x)
        x = self.fc(x)
        return x


feature_extractor = models.resnet50(pretrained=True)
feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])
for param in feature_extractor.parameters():
    param.requires_grad = False
feature_extractor.eval()

box_classif = BoxNet(len(classes))
box_classif.load_state_dict(torch.load("boxnet_state_dict.pt"))
for param in box_classif.parameters():
    param.requires_grad = False
box_classif.eval()

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_image = transforms.Compose([transforms.ToTensor(), normalize])


def make_inputs(serie, low_index):
    im_list, lab_list, _ = LabeledImagesMaker.make_labeled_images(
        serie, low_index)
    im_list = [x[1] for x in im_list]
    lab_list = [x[1] for x in lab_list]
    for k, lab in enumerate(lab_list):
        if lab not in classes:
            lab_list[k] = 'unknown'
    return im_list, lab_list


def import_test_df(num):
    if num == 5:
        return 0
    else:
        df = DataSets.REDD(num)
        df = df.resample('1min').fillna('ffill')[1:]
        df = pd.concat([df.iloc[x[0]:x[1]]
                        for x in indexes_to_keep[str(num)]], axis=0)
        return df


def preconv_feats(im_list, transform, verbose=False):
    batch = []
    for k, im in enumerate(im_list):
        if verbose:
            print("                       ", end='\r')
            print(" processing image {} out of {}".format(
                k + 1, len(im_list)), end='\r')
        img = np.flip(im, axis=0)
        img = Image.fromarray(img[-y_lim:, :, :]).convert("RGB")
        img = torch.as_tensor(transform(img))[:, -4850:, :]
        batch.extend((feature_extractor(img.unsqueeze(0))).data.numpy())
        del img
    batch = [torch.from_numpy(f).type(torch.float).reshape(-1) for f in batch]
    return batch


def predict_on_image(im_list, confidence_tresh=.6, verbose=False):
    F = nn.Sigmoid()
    preds = F(box_classif(torch.stack(im_list)))
    predict = []
    for pred in preds:
        if pred.max() > confidence_tresh:
            predict.append(classes[torch.argmax(pred)])
        else:
            predict.append('unknown')
    return predict


data_test = import_test_df(2)
data_test.head()
sample = data_test[["refrigerator_1", "dishwasher_1",
                    "microwave_1", "lighting_1", "washer_dryer_1"]][4500:5940]
Graphs.draw(sample.sum(axis=1))

images, labels = make_inputs(sample, 0)

features = preconv_feats(images, transform_image, verbose=True)
predictions = predict_on_image(features)

predictions


aggregated = sample.sum(axis=1)

image = LabeledImagesMaker.make_img(aggregated)
u_boxes = LabeledImagesMaker.make_boxes(aggregated)
boxed = LabeledImagesMaker.plot_unlabeled_on_image(
    image, u_boxes, color=[0, 0, 0])

gt_indexs = np.where(np.array(labels) != 'unknown')[0]
labeled_gt = LabeledImagesMaker.plot_boxes_with_labels(
    image, u_boxes[gt_indexs], np.array(labels)[gt_indexs])

predic_indexs = np.where(np.array(predictions) != 'unknown')[0]
labeled_predic = LabeledImagesMaker.plot_boxes_with_labels(
    image, u_boxes[predic_indexs], np.array(predictions)[predic_indexs])
