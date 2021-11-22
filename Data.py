
###
## This file loads all our project data for reference.
###

import pickle
with open("data/images_l.pkl", 'rb') as f: labeled_images = pickle.load(f)
with open("data/labels_l.pkl", 'rb') as f: labels = pickle.load(f)
with open("data/images_ul.pkl", 'rb') as f: unlabeled_images = pickle.load(f)
with open("data/images_test.pkl", 'rb') as f: images_test = pickle.load(f)