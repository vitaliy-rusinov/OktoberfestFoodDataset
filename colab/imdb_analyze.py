import sys
from os.path import dirname, join
import numpy as np
from pathlib import Path

if(len(sys.argv)!=2):
    print("please specify the dataset to display statistics")

#change it to path to your rootdir
PY_FASTER_RCNN_ROOTDIR='/content/pytorch-faster-rcnn/'
PY_FASTER_RCNN_LIBDIR=PY_FASTER_RCNN_ROOTDIR + 'tools/'

this_dir = dirname(__file__)
if __name__ == '__main__':
    sys.path.insert(0, PY_FASTER_RCNN_ROOTDIR)
    sys.path.insert(0, PY_FASTER_RCNN_LIBDIR)
    import _init_paths
    import os
    os.chdir(PY_FASTER_RCNN_ROOTDIR)

nonempty=set()

from datasets.factory import get_imdb

WARNING_AMOUNT_OF_BOXES=50

imdb=get_imdb(sys.argv[1])

class Image:

    def __init__(self):
        pass

def get_statistics(imdb):
    images=[]
    for image_id, entry in enumerate(imdb.gt_roidb()):
        lstboxes=[(entry['boxes']) ]
        lstclasses=[(entry['gt_classes']) ]
        for id,boxes in enumerate(lstboxes):
            image=Image()
            image.id=image_id
            image.bbox_dims=list()
            image.aspect_ratios=list()

            for item in boxes:
                image.bbox_dims.append((item[2]-item[0], item[3]-item[1]))
                image.aspect_ratios.append(float((item[2]-item[0]) / float(item[3]-item[1])))
            image.classes=lstclasses[id]
            image.boxes=boxes
            images.append(image)
    return images



filenames=list()
num_images = len(imdb.image_index)
for i in range(num_images):
    filenames.append(imdb.image_path_at(i))
#print(filenames)
file_names = set()
timestamps=list()
for fname in filenames:
    file_names.add(Path(fname).name)
    timestamps.append(int(Path(fname).name.split('_')[0]))


day_images=list()
night_images=list()

import datetime

for timestamp in timestamps:
    date = datetime.datetime.fromtimestamp(int(timestamp / 1000))
    if (date.hour > 7 and date.hour < 20):
        day_images.append(timestamp)
    if (not (date.hour > 7 and date.hour < 22)):
        night_images.append(timestamp)

stats = get_statistics(imdb)
warnings=[]
clsstats=dict()
clsimgstats=dict()
clsarea=dict()

total_area=int
total_non_zero_images=0
for im in stats:
    for cls in im.classes:
        cur_cnt=clsstats.get(cls,0)
        clsstats[cls]=cur_cnt+1
    classes_set=set(im.classes)
    for cls in classes_set:
        cur_cnt = clsimgstats.get(cls, 0)
        clsimgstats[cls] = cur_cnt + 1

    #print ('image size: %s x %s  -> %s x %s'%(im.width,im.height, int(im.dest_width), int(im.dest_height)))
    if len(im.boxes) > WARNING_AMOUNT_OF_BOXES :
        warnings.append("Warning: more than %d boxes: %s"%(WARNING_AMOUNT_OF_BOXES, str(im.id)))
    elif len(im.boxes)==0:
        warnings.append("Warning: zero boxes: %s" % str(im.id))

    if(len(im.boxes)>0):
        total_non_zero_images=total_non_zero_images+1
        nonempty.add(Path(imdb.image_path_at(im.id)).name.split('.')[0])
    for box_idx in range(0,len(im.boxes)):
        dest_x1, dest_y1, dest_x2, dest_y2 = im.boxes[box_idx]
        aspect_ratio=im.aspect_ratios[box_idx]
        width = (dest_x2-dest_x1)
        height = (dest_y2-dest_y1)
        newlist=clsarea.get(int(im.classes[box_idx]), list())
        newlist.append(width * height)
        clsarea[im.classes[box_idx]]=newlist
        if (width <= 0 or height <= 0):
            warnings.append("ERROR: height or width is 0 for "+im.filename)
        if (width*height < 10):
            warnings.append("Warning: contains small box: "+im.filename)
        #print("bbox: ")
        #print( "    width       : %d"%(dest_x2-dest_x1))
        #print( "    height       : %d" % (dest_y2 - dest_y1))
        #print( "    aspect ratio : %.2f"%aspect_ratio)

for msg in warnings:
    print ( 'warning:',msg)

flatten = lambda l: [item for sublist in l for item in sublist]

dims= flatten([im.bbox_dims for im in stats])

aspect_ratios = flatten([im.aspect_ratios for im in stats])
bbox_widths = [w for w,h in dims ]
bbox_heights = [h for w,h in dims]

cls_total_area=0

total_area=0
area_num=0
cls_area_mean=dict()

for ind in clsarea.keys():
    val=clsarea[ind]
    cls_total_area=0
    for area in val:
        total_area=total_area+area
        area_num=area_num+1
        cls_total_area=cls_total_area+area
    cls_area_mean[ind]=cls_total_area/len(val)
import math



for ind,val in enumerate(imdb._classes):
    if(not cls_area_mean.get(ind,None) is None):
        print('class mean area: '+imdb._classes[ind] + ' ' + str(math.sqrt(cls_area_mean[ind])))

print('mean area: %.2f' % (math.sqrt(total_area/area_num)))

print ('min bbox width: %.2f '%(min(bbox_widths)))
print ('mean bbox width: %.2f '%(np.mean(bbox_widths)))
print ('max bbox width: %.2f '%(max(bbox_widths)))

print ('min bbox height: %.2f '%(min(bbox_heights)))
print ('mean bbox height: %.2f '%(np.mean(bbox_heights)))
print ('max bbox height: %.2f '%(max(bbox_heights)))

print ('min aspect ratio: %.2f '%(min(aspect_ratios)))
print ('mean aspect ratio: %.2f '%(np.mean(aspect_ratios)))
print ('max aspect ratio: %.2f '%(max(aspect_ratios)))

print('number of images per class:\n')
total_img=0
lst_imgstats=list(sorted(clsimgstats))

for cls in lst_imgstats:
    print(imdb._classes[cls]+': ' + str(clsimgstats[cls]))

print('total images: ' + str(len(imdb.gt_roidb())))
print('total annotated images: ' + str(total_non_zero_images))
print('day images: ' + str(len(day_images)))
print('night images: ' + str(len(night_images)))


print('number of annotations per class:\n')
lst_stats=list(sorted(clsstats))
total_ann=0
for cls in lst_stats:
    print(imdb._classes[cls]+': ' + str(clsstats[cls]))
    total_ann=total_ann+int(clsstats[cls])
print('total boxes: ' + str(total_ann))
