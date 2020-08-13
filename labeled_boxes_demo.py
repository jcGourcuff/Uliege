import numpy as np
import pandas as pd
from utils.display import Graphs
from dataManager.load import DataSets
from nilm.filters import Filters
from utils.compute import Metrics
from nilm.imageGen_and_labelling import LabeledImagesMaker

# loads demo data. 5 appliances, timestamp in secondes sampled every ~3 secondes.
df = DataSets.REDD(1, demo=True)
df.head()


# resample the data to have minute index
sampled = df.resample('1min').mean().fillna(method='ffill')
sampled.head()

# select a day of data and 3 appliances
test = sampled[:24 * 60][['refrigerator', 'lighting', 'dishwasher']]

# visualize
Graphs.aggregate(test)
Graphs.decompose_aggregate(test)

# compute aggregated signal and the associated image
aggregated = test.sum(axis=1)
image = LabeledImagesMaker.make_img(aggregated)

# Fit the aggregated signal with boxes and plot them on image
# Note : to see better the result, change the y_lim and Cons_Mx parameters of imageGen_and_labelling.py to 3000 for instance.
u_boxes = LabeledImagesMaker.make_boxes(aggregated)
new_img = LabeledImagesMaker.plot_unlabeled_on_image(image, u_boxes)

# Computes boxes and their label for training
_, u_boxes, labels = LabeledImagesMaker.compute_boxes_and_labels(
    test, 0, compute_image=False, demo=True)
new_img = LabeledImagesMaker.plot_boxes_with_labels(image, u_boxes, labels)

# Finally if we want to obtain the images that feed the computer vision model.
imgs, labels, _ = LabeledImagesMaker.make_labeled_images(test, 0, demo=True)

# We cn visualize on entry
print(labels[0])
LabeledImagesMaker.plot(imgs[0][1])
