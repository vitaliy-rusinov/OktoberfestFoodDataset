# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np

def parse_rec(filename):
  """ Parse a PASCAL VOC xml file """
  tree = ET.parse(filename)
  objects = []
  for obj in tree.findall('object'):
    obj_struct = {}
    obj_struct['name'] = obj.find('name').text
    obj_struct['pose'] = obj.find('pose').text
    obj_struct['truncated'] = int(obj.find('truncated').text)
    obj_struct['difficult'] = int(obj.find('difficult').text)
    bbox = obj.find('bndbox')
    obj_struct['bbox'] = [int(bbox.find('xmin').text),
                          int(bbox.find('ymin').text),
                          int(bbox.find('xmax').text),
                          int(bbox.find('ymax').text)]
    objects.append(obj_struct)

  return objects


def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap

def voc_get_gt(cachedir, annopath, imagesetfile):

  if not os.path.isdir(cachedir):
    os.mkdir(cachedir)
  cachefile = os.path.join(cachedir, '%s_annots.pkl' % imagesetfile)
  # read list of images
  with open(imagesetfile, 'r') as f:
    lines = f.readlines()
  imagenames = [x.strip() for x in lines]

  if not os.path.isfile(cachefile):
    # load annotations
    recs = {}
    for i, imagename in enumerate(imagenames):
      recs[imagename] = parse_rec(annopath.format(imagename))
      if i % 100 == 0:
        print('Reading annotation for {:d}/{:d}'.format(
          i + 1, len(imagenames)))
    # save
    print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'wb') as f:
      pickle.dump(recs, f)
  else:
    # load
    with open(cachefile, 'rb') as f:
      try:
        recs = pickle.load(f)
      except:
        recs = pickle.load(f, encoding='bytes')
  gt=dict()
  for imagename in imagenames:
    for obj in recs[imagename]:
      imgdict=gt.get(imagename,dict())
      imgdict[obj['name']]=imgdict.get(obj['name'], 0)+1
      gt[imagename]=imgdict
  return gt


def evaluate_helper(gt,category_names,n_s_c,threshold=0.2, eval_per_class=False,  merged=False,
             confusion_matrix=False ):

  image_ids,confidence,classes=n_s_c

  # sort by score
  sorted_ind = np.argsort(-confidence)
  sorted_scores = np.sort(-confidence)
  image_ids = [image_ids[x] for x in sorted_ind]
  classes = [classes[x] for x in sorted_ind]

  all_images=list(set(gt.keys()))
  all_images.sort()
  img_id_dict=dict()
  for ind,el in enumerate(all_images):
    img_id_dict[el]=ind

  n=list()
  for img in image_ids:
    n.append(img_id_dict[img])
  classes_dict=dict()
  for ind,cat in enumerate(category_names):
    classes_dict[cat]=ind

  s=-sorted_scores
  c=list()
  for cls in classes:
    c.append(classes_dict[cls])

  new_gt=np.zeros((len(gt.keys()),len(category_names)))
  for key,val in gt.items():
    img_id=img_id_dict[key]
    for cls_key, cls_val in val.items():
      new_gt[img_id][classes_dict[cls_key]]=cls_val

  gt=new_gt


  tp, fp, fn = 0, 0, np.sum(gt)

  num_classes = len(category_names)  # np.max(c)+1
  cm = np.zeros((num_classes, num_classes))

  idiot_points = 0

  if eval_per_class:

    rpg_list, arpg_list = [], []
    print("Evaluation for %d classes" % num_classes)
    for class_i in range(num_classes):
      rpg = [[0., 1.]]  # [recall, precision]
      tp, fp, fn = 0, 0, np.sum(gt[:, class_i])
      if fn == 0:
        print('Class %d, has no ground truth available!' % class_i)
      else:
        for i in range(len(n)):
          if c[i] == class_i:
            if gt[n[i]][c[i]] > 0:  # ground_truth of detected item in image i > 0
              tp += 1
              fn -= 1
              gt[n[i], c[i]] -= 1
            else:
              fp += 1
            rpg.append([tp / (tp + fn), tp / (tp + fp)])
      rpg += [[rpg[-1][0] + 1e-8, 0], [1., 0.]]
      rpg = np.array(rpg)
      rpg[:, 1] = [np.max(rpg[i:, 1]) for i in range(len(rpg))]
      arpg = np.trapz(y=rpg[:, 1], x=rpg[:, 0])
      rpg_list.append(rpg)
      arpg_list.append(arpg)
    # return rpg_list, arpg_list
  else:
    rpg = np.empty((len(image_ids) + 2, 4))  # [recall, precision]
    for i in range(len(n)):
      if c[i] >= num_classes:
        idiot_points += 1
      # print("c[i] = %d"%c[i])
      if c[i] < num_classes and gt[n[i]][c[i]] > 0:
        tp += 1
        fn -= 1
        gt[n[i], c[i]] -= 1
      else:
        fp += 1
      rpg[i] = tp / (tp + fn), tp / (tp + fp), s[i],c[i]

    rpg[-2:, :] = [rpg[-3, 0] + 1e-8, 0, 0, None], [1., 0., 0, None]
    rpg[:, 1] = [np.max(rpg[i:, 1]) for i in range(len(rpg))]
    arpg = np.trapz(y=rpg[:, 1], x=rpg[:, 0])
    # return rpg, arpg


  if idiot_points > 0:
    print(
      "%d items were categorized incorrectly" % idiot_points)
  if eval_per_class:
    return rpg_list, arpg_list
  else:
    return rpg, arpg


def voc_gt_eval_accumulate(detpath,
             gt,
             imagesetfile,
             classname):
  detfile = detpath.format(classname)
  with open(detfile, 'r') as f:
    lines = f.readlines()

  splitlines = [x.strip().split(' ') for x in lines]
  image_ids = [x[0] for x in splitlines]
  confidence = np.array([float(x[1]) for x in splitlines])
  class_names=[classname for num in image_ids]

  return image_ids, list(confidence), class_names



def voc_eval_total(gt,image_ids, confidence,class_names, categories, perClass=False):
  return evaluate_helper(gt,categories,(image_ids,np.asarray(confidence),class_names ), eval_per_class=perClass )

def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False,
             use_diff=False):
  """rec, prec, ap = voc_eval(detpath,
                              annopath,
                              imagesetfile,
                              classname,
                              [ovthresh],
                              [use_07_metric])

  Top level function that does the PASCAL VOC evaluation.

  detpath: Path to detections
      detpath.format(classname) should produce the detection results file.
  annopath: Path to annotations
      annopath.format(imagename) should be the xml annotations file.
  imagesetfile: Text file containing the list of images, one image per line.
  classname: Category name (duh)
  cachedir: Directory for caching the annotations
  [ovthresh]: Overlap threshold (default = 0.5)
  [use_07_metric]: Whether to use VOC07's 11 point AP computation
      (default False)
  """
  # assumes detections are in detpath.format(classname)
  # assumes annotations are in annopath.format(imagename)
  # assumes imagesetfile is a text file with each line an image name
  # cachedir caches the annotations in a pickle file

  # first load gt
  if not os.path.isdir(cachedir):
    os.mkdir(cachedir)
  cachefile = os.path.join(cachedir, '%s_annots.pkl' % imagesetfile)
  # read list of images
  with open(imagesetfile, 'r') as f:
    lines = f.readlines()
  imagenames = [x.strip() for x in lines]

  if not os.path.isfile(cachefile):
    # load annotations
    recs = {}
    for i, imagename in enumerate(imagenames):
      recs[imagename] = parse_rec(annopath.format(imagename))
      if i % 100 == 0:
        print('Reading annotation for {:d}/{:d}'.format(
          i + 1, len(imagenames)))
    # save
    print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'wb') as f:
      pickle.dump(recs, f)
  else:
    # load
    with open(cachefile, 'rb') as f:
      try:
        recs = pickle.load(f)
      except:
        recs = pickle.load(f, encoding='bytes')

  # extract gt objects for this class
  class_recs = {}
  npos = 0
  for imagename in imagenames:
    R = [obj for obj in recs[imagename] if obj['name'] == classname]
    bbox = np.array([x['bbox'] for x in R])
    if use_diff:
      difficult = np.array([False for x in R]).astype(np.bool)
    else:
      difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    det = [False] * len(R)
    npos = npos + sum(~difficult)
    class_recs[imagename] = {'bbox': bbox,
                             'difficult': difficult,
                             'det': det}

  # read dets
  detfile = detpath.format(classname)
  with open(detfile, 'r') as f:
    lines = f.readlines()

  splitlines = [x.strip().split(' ') for x in lines]
  image_ids = [x[0] for x in splitlines]
  confidence = np.array([float(x[1]) for x in splitlines])
  BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

  nd = len(image_ids)
  tp = np.zeros(nd)
  fp = np.zeros(nd)

  if BB.shape[0] > 0:
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    for d in range(nd):
      R = class_recs[image_ids[d]]
      bb = BB[d, :].astype(float)
      ovmax = -np.inf
      BBGT = R['bbox'].astype(float)

      if BBGT.size > 0:
        # compute overlaps
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

      if ovmax > ovthresh:
        if not R['difficult'][jmax]:
          if not R['det'][jmax]:
            tp[d] = 1.
            R['det'][jmax] = 1
          else:
            fp[d] = 1.
      else:
        fp[d] = 1.

  # compute precision recall
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = tp / float(npos)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  ap = voc_ap(rec, prec, use_07_metric)

  return rec, prec, ap
