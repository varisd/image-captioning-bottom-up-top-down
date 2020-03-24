"""
Reads in a tsv file with pre-trained bottom up attention features and
stores it in HDF5 format.  Also store {image_id: feature_idx}
 as a pickle file.
Hierarchy of HDF5 file:
{ 'image_features': num_images x num_boxes x 2048 array of features
  'image_bb': num_images x num_boxes x 4 array of bounding boxes }
"""
from __future__ import print_function

import os
import io
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import csv
import h5py
import cPickle
import numpy as np
import utils
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

## ORIGINAL ##
#FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
#infile = 'trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv'
#train_data_file = 'train36.hdf5'
#val_data_file = 'val36.hdf5'
#train_indices_file = 'train36_imgid2idx.pkl'
#val_indices_file = 'val36_imgid2idx.pkl'
#train_ids_file = 'train_ids.pkl'
#val_ids_file = 'val_ids.pkl'

#feature_length = 2048
#num_fixed_boxes = 36

feature_length = 2048
num_fixed_boxes = 100


def parse_args():
    parser = argparse.ArgumentParser(description="TODO")

    parser.add_argument("--infile", help="File containing the features")
    parser.add_argument("--outdir", help="Output directory")
    parser.add_argument(
        "--fields", nargs="+",
        default=['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features', 'classes', 'attributes'],
        help="Names of the fields to extract")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    train_data_file = os.path.join(args.outdir, 'train100.hdf5')
    val_data_file = os.path.join(args.outdir, 'val100.hdf5')
    train_indices_file = os.path.join(args.outdir, 'train100_imgid2idx.pkl')
    val_indices_file = os.path.join(args.outdir, 'val100_imgid2idx.pkl')
    train_ids_file = os.path.join(args.outdir, 'train_ids.pkl')
    val_ids_file = os.path.join(args.outdir, 'val_ids.pkl')

    h_train = h5py.File(train_data_file, "w")
    h_val = h5py.File(val_data_file, "w")


    if os.path.exists(train_ids_file) and os.path.exists(val_ids_file):
        train_imgids = cPickle.load(open(train_ids_file))
        val_imgids = cPickle.load(open(val_ids_file))
    else:
        train_imgids = utils.load_imageid('data/train2014')
        val_imgids = utils.load_imageid('data/val2014')
        cPickle.dump(train_imgids, open(train_ids_file, 'wb'), protocol=2)
        cPickle.dump(val_imgids, open(val_ids_file, 'wb'), protocol=2)

    train_indices = {}
    val_indices = {}

    train_img_features = h_train.create_dataset(
        'image_features', (len(train_imgids), num_fixed_boxes, feature_length), 'f')
    train_img_bb = h_train.create_dataset(
        'image_bb', (len(train_imgids), num_fixed_boxes, 4), 'f')
    train_spatial_img_features = h_train.create_dataset(
        'spatial_features', (len(train_imgids), num_fixed_boxes, 6), 'f')
    train_classes = h_train.create_dataset(
        'classes', (len(train_imgids), num_fixed_boxes), 'f')
    train_attributes = h_train.create_dataset(
        'attributes', (len(train_imgids), num_fixed_boxes), 'f')

    val_img_bb = h_val.create_dataset(
        'image_bb', (len(val_imgids), num_fixed_boxes, 4), 'f')
    val_img_features = h_val.create_dataset(
        'image_features', (len(val_imgids), num_fixed_boxes, feature_length), 'f')
    val_spatial_img_features = h_val.create_dataset(
        'spatial_features', (len(val_imgids), num_fixed_boxes, 6), 'f')
    val_classes = h_val.create_dataset(
        'classes', (len(val_imgids), num_fixed_boxes), 'f')
    val_attributes = h_val.create_dataset(
        'attributes', (len(val_imgids), num_fixed_boxes), 'f')

    train_counter = 0
    val_counter = 0

    print("reading tsv...")
    with open(args.infile, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=args.fields)
        for item in tqdm(reader):
            item['num_boxes'] = int(item['num_boxes'])
            image_id = int(item['image_id'])
            image_w = float(item['image_w'])
            image_h = float(item['image_h'])
            bboxes = np.frombuffer(
                base64.decodestring(item['boxes']),
                dtype=np.float32).reshape((item['num_boxes'], -1))
            classes = np.frombuffer(
                base64.decodestring(item['classes']),
                dtype=np.int64).reshape((item['num_boxes']))
            attributes = np.frombuffer(
                base64.decodestring(item['attributes']),
                dtype=np.int64).reshape((item['num_boxes']))

            box_width = bboxes[:, 2] - bboxes[:, 0]
            box_height = bboxes[:, 3] - bboxes[:, 1]
            scaled_width = box_width / image_w
            scaled_height = box_height / image_h
            scaled_x = bboxes[:, 0] / image_w
            scaled_y = bboxes[:, 1] / image_h

            box_width = box_width[..., np.newaxis]
            box_height = box_height[..., np.newaxis]
            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]

            spatial_features = np.concatenate(
                (scaled_x,
                 scaled_y,
                 scaled_x + scaled_width,
                 scaled_y + scaled_height,
                 scaled_width,
                 scaled_height),
                axis=1)

            if image_id in train_imgids:
                train_imgids.remove(image_id)
                train_indices[image_id] = train_counter
                train_img_bb[train_counter, :item['num_boxes'], :] = bboxes
                train_img_features[train_counter, :item['num_boxes'], :] = np.frombuffer(
                    base64.decodestring(item['features']),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                train_spatial_img_features[train_counter, :item['num_boxes'], :] = spatial_features
                train_classes[train_counter, :item['num_boxes']] = classes
                train_attributes[train_counter, :item['num_boxes']] = attributes
                train_counter += 1
            elif image_id in val_imgids:
                val_imgids.remove(image_id)
                val_indices[image_id] = val_counter
                val_img_bb[val_counter, :item['num_boxes'], :] = bboxes
                val_img_features[val_counter, :item['num_boxes'], :] = np.frombuffer(
                    base64.decodestring(item['features']),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                val_spatial_img_features[val_counter, :item['num_boxes'], :] = spatial_features
                val_classes[val_counter, :item['num_boxes']] = classes
                val_attributes[val_counter, :item['num_boxes']] = attributes
                val_counter += 1
            else:
                assert False, 'Unknown image id: %d' % image_id

    if len(train_imgids) != 0:
        print('Warning: train_image_ids is not empty')

    if len(val_imgids) != 0:
        print('Warning: val_image_ids is not empty')

    cPickle.dump(train_indices, open(train_indices_file, 'wb'))
    cPickle.dump(val_indices, open(val_indices_file, 'wb'))
    h_train.close()
    h_val.close()
    print("done!")
