
# From Python
import sys
import cv2
import numpy as np
import os
from sys import platform
from dotmap import DotMap
from shapely import affinity
from shapely.geometry import Polygon
#sys.path.append('/Users/Tilman/Documents/Programme/Python/forschungspraktikum/openpose/python');
from openpose import pyopenpose as op
from icc.triangles import *
from icc.bisection import *
from icc.misc import *
from icc.kmeans import *

SHOW_WINDOW = False
SAVE_FILE = not SHOW_WINDOW
TRIANGLES = False
BODY_LINES = True
BISEC_VECTORS = True
BISEC_CONES = True
GLOBAL_LINE = True
INPAINT_AND_KMEANS = True
DRAW_FIRST_CONE_LAYER_BRIGTHER = True
BISEC_CONE_ANGLE = 50
CORRECTION_ANGLE = 23
OVERLAY_ALPHA = 0.2
COLORED_CANVAS = True
BISEC_SKIP_LOWER_LEVELS = False
DISPLAY_RASTER_ELEMENTS = 500
MAX_WIDTH = 1500 #px
KMEANS_AREA_MASK_THRESHOLD = 0.06 #0.08 #max 1.0 (percent of pixels), cur 0.09,  sonntag mittag 0.05,   #smaller threshold -> more colors, higher threshold -> less colors
KMEANS_K = 7 #10
#KMEANS_K = 6 #10
OUT_DIR = os.environ['OUT_DIR'] if 'OUT_DIR' in os.environ else './images/out/asana_task/art_structure_inf/eccv2020_coco/'
#IN_DIR = "images/first_email/"     # images from first email
IN_DIR = os.environ['IN_DIR'] if 'IN_DIR' in os.environ else "./images/eccv2020_coco/"    # images from imdahl

print("in dir",IN_DIR, "out dir", OUT_DIR)

if os.path.dirname(__file__):
    print("__file__",__file__)
    os.chdir(os.path.dirname(__file__)) #make sure our curdir is the dir of the script so all relative paths will work

SKIP_OPENPOSE = False
OPENPOSE_DEMO_KEYPOINTS = np.array([[[4.7613028e+02, 3.3695804e+02, 9.0203685e-01],[5.3667474e+02, 3.8633786e+02, 6.6615295e-01],[5.1645105e+02, 3.8405157e+02, 5.1514143e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[5.5459924e+02, 3.8859457e+02, 6.4240879e-01],[5.6353766e+02, 4.7384988e+02, 1.8810490e-01],[5.3886292e+02, 5.2543573e+02, 9.0144195e-02],[5.4566248e+02, 5.3215259e+02, 3.6083767e-01],[5.2768524e+02, 5.3213129e+02, 3.1196830e-01],[5.4556714e+02, 6.3534674e+02, 1.8182488e-01],[5.8149310e+02, 7.2958716e+02, 1.3625422e-01],[5.6579541e+02, 5.3216382e+02, 3.6866242e-01],[5.8822272e+02, 6.2862476e+02, 1.7708556e-01],[6.0843213e+02, 7.2955762e+02, 2.2736737e-01],[4.7597812e+02, 3.2798129e+02, 5.7176876e-01],[4.8729745e+02, 3.3027243e+02, 9.1296065e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[5.2090784e+02, 3.3472034e+02, 7.7942842e-01],[5.7928674e+02, 7.5646222e+02, 2.0351715e-01],[5.9049512e+02, 7.5648248e+02, 2.0819387e-01],[6.2183606e+02, 7.3853394e+02, 1.7312977e-01],[5.8145673e+02, 7.5420642e+02, 1.2660497e-01],[5.7701074e+02, 7.5417773e+02, 1.2881383e-01],[5.8374255e+02, 7.3627380e+02, 9.4869599e-02]]
                                    ,[[6.4435681e+02, 3.6383255e+02, 8.9096022e-01],[6.6903070e+02, 3.9760306e+02, 8.7681645e-01],[6.4430103e+02, 3.9525812e+02, 7.9584122e-01],[6.3310535e+02, 4.5589160e+02, 3.7108111e-01],[5.9046979e+02, 4.2451276e+02, 4.0277350e-01],[6.9366602e+02, 4.0197583e+02, 8.9528430e-01],[6.8247137e+02, 4.6042902e+02, 5.5132395e-01],[6.0616620e+02, 4.3569894e+02, 3.4303352e-01],[6.5551196e+02, 5.1196445e+02, 2.9572365e-01],[6.3529651e+02, 5.0747903e+02, 2.8629595e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[6.7573169e+02, 5.1421967e+02, 3.0180413e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[6.4206000e+02, 3.5276721e+02, 7.2430253e-01],[6.5327673e+02, 3.5271103e+02, 9.4265050e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[6.7577380e+02, 3.5269864e+02, 8.9672232e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00]]
                                    ,[[7.2723553e+02, 4.0875150e+02, 8.3982950e-01],[7.6091986e+02, 4.6032086e+02, 5.0676465e-01],[7.3178253e+02, 4.5359366e+02, 3.5797939e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[7.8784674e+02, 4.6483188e+02, 5.6356871e-01],[7.6320721e+02, 5.6802844e+02, 3.7939239e-01],[7.2953772e+02, 5.4564911e+02, 1.5424372e-01],[7.6546356e+02, 6.1964557e+02, 1.7308682e-01],[7.3854327e+02, 6.1513757e+02, 1.5351829e-01],[7.3855487e+02, 7.3405249e+02, 5.6986582e-02],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[7.8789227e+02, 6.2636108e+02, 1.8666090e-01],[7.9010718e+02, 7.5197815e+02, 9.0752751e-02],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[7.2722571e+02, 3.9980579e+02, 4.9854943e-01],[7.4074554e+02, 4.0420221e+02, 8.2562774e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[7.6537799e+02, 4.0880304e+02, 6.8228495e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00]]
                                    ,[[2.6297342e+02, 3.4823679e+02, 9.1535652e-01],[2.1584425e+02, 3.8410617e+02, 4.2777365e-01],[2.0466562e+02, 3.8629623e+02, 6.5148002e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[2.2483388e+02, 3.7963403e+02, 2.8349286e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[2.1584836e+02, 5.5681036e+02, 7.1318626e-02],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[2.5625162e+02, 3.3922253e+02, 8.9375269e-01],[2.6528430e+02, 3.3701016e+02, 1.3707811e-01],[2.2490630e+02, 3.4151849e+02, 8.1041366e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00]]
                                    ,[[3.1685654e+02, 3.3244104e+02, 6.7855740e-01],[2.9669766e+02, 3.7735825e+02, 3.5962355e-01],[2.6300262e+02, 3.8186972e+02, 4.8755571e-01],[2.8984323e+02, 5.1421375e+02, 1.4892229e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[3.3024896e+02, 3.7068640e+02, 3.1298172e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[3.0774149e+02, 3.2796570e+02, 6.2570477e-01],[3.1678952e+02, 3.1911349e+02, 2.6238269e-01],[2.8093988e+02, 3.3702823e+02, 4.3097427e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00]]
                                    ,[[4.0661322e+02, 3.3243243e+02, 7.2449613e-01],[3.5496320e+02, 3.7965060e+02, 3.2941282e-01],[2.0466562e+02, 3.8629623e+02, 6.5148002e-01],[2.4725473e+02, 4.3794165e+02, 1.0388593e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[3.9527917e+02, 3.7732455e+02, 1.9104436e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[3.9753812e+02, 3.2572601e+02, 7.9601538e-01],[4.1098145e+02, 3.2347913e+02, 4.7544584e-01],[3.5937631e+02, 3.2570648e+02, 6.8124008e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00]]
                                    ,[[1.1046178e+02, 3.3481174e+02, 8.0748719e-01],[5.2089359e+01, 3.7064417e+02, 1.8357244e-01],[6.7774979e+01, 3.6842288e+02, 4.0538907e-01],[1.3283961e+02, 3.8408841e+02, 2.2997330e-01],[1.7771373e+02, 3.4149902e+02, 2.7701011e-01],[3.4127533e+01, 3.6390732e+02, 1.1019738e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[9.9306007e+01, 3.3019724e+02, 9.2014235e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[7.8990585e+01, 3.3474988e+02, 8.4556317e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00]]
                                    ,[[7.0929346e+02, 3.5261667e+02, 3.9232758e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[7.8120416e+02, 4.0420923e+02, 5.1513046e-02],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[7.0712549e+02, 3.4816431e+02, 1.5942883e-01],[7.1832990e+02, 3.4808749e+02, 3.8954309e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[7.4299701e+02, 3.5270523e+02, 2.7498546e-01],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],[0.0000000e+00, 0.0000000e+00, 0.0000000e+00]]])

## misc functions
def overlay_two_image_v2(image, overlay, ignore_color=[0,0,0], alpha=0.1):
    ignore_color = np.asarray(ignore_color)
    mask = (overlay==ignore_color).all(-1,keepdims=True)
    out = np.where(mask,image,(image * (1-alpha) + overlay * alpha).astype(image.dtype))
    return out
## misc functions end


params = dict()
#params["model_folder"] = "/Users/Tilman/Documents/Programme/Python/forschungspraktikum/openpose/models/"
params["model_folder"] = os.environ['OPENPOSE_MODELS']
#https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#pose-output-format-body_25

# Starting OpenPose
if not SKIP_OPENPOSE:
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

images = [os.path.join(os.getcwd(), IN_DIR, f) for f in os.listdir(IN_DIR)] #make path absolute so os.chdir has no side effects
images.sort()

os.chdir(OUT_DIR) #save images in this dir
#for img_name in images:
# for img_name, img_bdcn in list(zip(images, images_bdcn)):
# for img_name in images[0:1]:
# print(images)

#filter out hidden files
images = list(filter(lambda e: os.path.basename(e)[0]!='.',images))
#filter images if we want to inspect single image:
# images = list(filter(lambda e: "Franziskus-Giotto1" in e,images))

for img_name in images: #beweinung
# for img_name, img_bdcn in (list(zip(images, images_bdcn))[15:16] if SKIP_OPENPOSE else list(zip(images, images_bdcn))): #jesus
# for img_name, img_bdcn in list(zip(images, images_bdcn))[6:7]: #fusswaschung
    # Process Image
    print("calculating: "+img_name)
    img = cv2.imread(img_name)


    #resize to max width
    scale_percent = MAX_WIDTH/len(img[0]) # percent of original size
    if(scale_percent < 1.0):
        width = int(img.shape[1] * scale_percent);
        height = int(img.shape[0] * scale_percent);
        dim = (width, height);
        # resize image
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA);
        print("img scaled to:",len(img),len(img[0]))
    kmeans_copy = img.copy();
    intermediate_output_canvas = img.copy();

    output_canvas = np.array([[[255,255,255]]*len(img[0])]*len(img),np.uint8)
    max_lw = len(img);
    esz = max_lw / DISPLAY_RASTER_ELEMENTS
    if SKIP_OPENPOSE: #skip openpose for debugging, we use already calculated poses then, only works on image nr. 15
        datum = DotMap()
        datum.poseKeypoints = OPENPOSE_DEMO_KEYPOINTS
        print("Skipping OPENPOSE")
    else:
        datum = op.Datum()
        datum.cvInputData = img
        opWrapper.emplaceAndPop([datum])
        img = datum.cvOutputData
    
    if INPAINT_AND_KMEANS:
        # general idea, people are placed in foreground 
        # -> inpaint around people to replace color invormation from people with color information from direct environment
        # -> kmeans the result to dramatically reduce the amount of colors 
        # -> check witch colors are now on the position of the people. The colors participating more than 8% are considered as foreground and replaced by color with the most participation in foreground
        # -> output this mask if color information with details is needed.
        # -> if binary mask is needed we further do some morph filtering to get away small details and blobs
        target = kmeans_copy;
        #remove cracks and prepare for kmeans
        # target = cv2.GaussianBlur(target,(5,5),0)       #unsharpen all
        # target = cv2.medianBlur(target,33)              #remove cracks with with 33 pixel
        # target = cv2.bilateralFilter(target,33,90,90)   #bring back edges and remove textures
        # target = cv2.bilateralFilter(target,13,90,90)   #bring back edges and remove textures
        # target = cv2.medianBlur(target,53)              #remove big color patches
        # target = cv2.bilateralFilter(target,13,90,90)   #bring back edges and remove textures
        #target = cv2.medianBlur(target,11)              #smoothen cracks
        md = int(esz*4)+1 if int(esz*4)%2==0 else int(esz*4)
        bl1 = int(esz*20)
        bl2 = int(esz*40)
        target = cv2.medianBlur(target,md if md<14 else 13) #smoothen cracks
        target = cv2.bilateralFilter(target,bl1 if bl1<31 else 31 ,bl2 if bl2<80 else 80,bl2 if bl2<80 else 80)        #remove cracks
        # target = cv2.medianBlur(target,) #smoothen cracks
        # target = cv2.bilateralFilter(target,50,70,70)        #remove cracks
        print("filter vals: med",int(esz*5)+1 if int(esz*5)%2==0 else int(esz*5), "bil1", int(esz*25), "bil2,3", int(esz*40))
        cv2.imwrite(os.path.basename(img_name)+'_inkm_step1_crackremoval.jpg',target)

        fposes = np.array([np.array([line[:2] for line in pose if line[2] > 0]) for pose in datum.poseKeypoints]) #filtered poses without zero lines
        mask = np.zeros((len(img),len(img[0]),1), np.uint8)
        kmout_mask = np.zeros((len(img),len(img[0]),1), np.uint8)
        for pose in fposes: #remove bodys
            convexhull = Polygon(pose).convex_hull
            #inpainting
            # sconvexhull = affinity.scale(convexhull, xfact=1.5, yfact=1.7, origin=convexhull.centroid)
            sconvexhull = affinity.scale(convexhull, xfact=1.7, yfact=1.4, origin=convexhull.centroid)
            cv2.drawContours(mask, [polyToArr(sconvexhull)], 0, 255, int(15*esz))
            cv2.drawContours(mask, [polyToArr(sconvexhull)], 0, 255, -1)

            #kmeans check
            sconvexhull = affinity.scale(convexhull, xfact=1, yfact=0.7, origin=convexhull.centroid)
            cv2.drawContours(kmout_mask, [polyToArr(sconvexhull)], 0, 255, int(7*esz))
            cv2.drawContours(kmout_mask, [polyToArr(sconvexhull)], 0, 255, -1)
        cv2.rectangle(mask, (0,0), (len(img[0]),len(img)), 255, int(40*esz)) #remove frames
        #shift kmeans mask pixels downwards 40px
        # shift = -40
        # for i in range(kmout_mask.shape[0] -1, kmout_mask.shape[0] - shift, -1):
        #     kmout_mask = np.roll(kmout_mask, -1, axis=0)
        #     kmout_mask[-1, :] = 0
        kmout_mask = cv2.warpAffine(kmout_mask, np.float32([ [1,0,0], [0,1,30] ]), (kmout_mask.shape[:2][1], kmout_mask.shape[:2][0]))   

        
        cv2.imwrite(os.path.basename(img_name)+'_inkm_step2_inpaintmask.jpg',mask)
        cv2.imwrite(os.path.basename(img_name)+'_inkm_step2_kmeansresmask.jpg',kmout_mask)
        inpainted = cv2.inpaint(target, mask, 3, cv2.INPAINT_TELEA)
        # inpainted = cv2.medianBlur(inpainted,int(esz*5)+1 if int(esz*5)%2==0 else int(esz*5))
        # inpainted = cv2.bilateralFilter(inpainted,30,40,40)
        cv2.imwrite(os.path.basename(img_name)+'_inkm_step3_inpainted.jpg',inpainted)
       
        #to kmeans
        kmeans_output = imgKmeans(inpainted, KMEANS_K)
        cv2.imwrite(os.path.basename(img_name)+'_inkm_step4_kmeansres.jpg',kmeans_output)

        # cv2.imshow(img_name+"inpainted", inpainted)
        # cv2.imshow(img_name+"mask", kmout_mask)
        kmout_mask=cv2.cvtColor(kmout_mask,cv2.COLOR_GRAY2BGR) #change mask to a 3 channel image 
        km_mask_out=cv2.subtract(kmout_mask,kmeans_output)        #subtract mask from kmeans result
        km_mask_out=cv2.subtract(kmout_mask,km_mask_out)

        #idea: count how many pixels the white mask has, count how many pixels each color from kmeans result appears in mask -> area > 80% is foreground
        kmeans_colors, kmeans_counts =   np.unique(kmeans_output.reshape(-1, kmeans_output.shape[-1]),axis=0,return_counts=True)
        km_mask_colors, km_mask_counts = np.unique(km_mask_out.reshape(-1, km_mask_out.shape[-1]),axis=0,return_counts=True)
        mask_colors, mask_counts =       np.unique(kmout_mask.reshape(-1, kmout_mask.shape[-1]),axis=0,return_counts=True)

        white_pixels = mask_counts[(np.argwhere(mask_colors>0))[0,0]]
        threshold_count = int(KMEANS_AREA_MASK_THRESHOLD*white_pixels) #select counts from white color and generate threshold with it.
        print("white_pixels",white_pixels,"threshold_count",threshold_count)
        
        kmeans_colors = kmeans_colors[kmeans_counts.argsort()][::-1] # sort kmeans_counts by count
        kmeans_counts = kmeans_counts[kmeans_counts.argsort()][::-1] # sort kmeans_counts by count
        
        km_mask_colors = km_mask_colors[km_mask_counts.argsort()][::-1] # sort mask colors by count
        km_mask_counts = km_mask_counts[km_mask_counts.argsort()][::-1] # sort mask counts by count
        print("km_mask_colors",km_mask_colors,"km_mask_counts",km_mask_counts)
        filtered_km_mask_FG_colors = km_mask_colors[np.argwhere(km_mask_counts>=threshold_count)]
        filtered_km_mask_FG_colors = filtered_km_mask_FG_colors[np.sum(filtered_km_mask_FG_colors, axis=2)>0] #filter out black from mask
        
        filtered_km_mask_BG_colors = km_mask_colors[np.argwhere(km_mask_counts<threshold_count)]
        filtered_km_mask_BG_colors = filtered_km_mask_BG_colors[np.sum(filtered_km_mask_BG_colors, axis=2)>0] #filter out black from mask
        

        if(len(filtered_km_mask_FG_colors)>0): #we can not do anything if we have no foreground colors

            cv2.putText(km_mask_out, "Foreground colors:", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))
            
            cv2.rectangle(kmeans_output, (0,0), (235,30), (0,0,0), -1)
            cv2.putText(kmeans_output, "Foreground color:", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))
            cv2.rectangle(kmeans_output, (220,15), (230,25), (int(filtered_km_mask_FG_colors[0][0]),int(filtered_km_mask_FG_colors[0][1]),int(filtered_km_mask_FG_colors[0][2])), -1)
            cv2.rectangle(kmeans_output, (220,15), (230,25), (255,255,255), 1)

            cv2.rectangle(km_mask_out, (0,0), (235,30), (0,0,0), -1)
            cv2.putText(km_mask_out, "Foreground colors:", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))
            cv2.rectangle(km_mask_out, (220,15), (230,25), (255,255,255), 1)

            # cv2.imwrite(os.path.basename(img_bdcn)+'_km_mask_out_before.jpg',kmeans_output)
            binary_output = kmeans_output.copy()
            offset=220;
            for color in filtered_km_mask_FG_colors:
                #print colors on screen
                print("filtered_km_mask_FG_colors",color)
                x = offset
                y = 15
                offset += 15
                #replace colors
                # binary_output[((binary_output[:,:,0] == color[0]) & (binary_output[:,:,1] == color[1]) & (binary_output[:,:,2] == color[2]))]=filtered_km_mask_FG_colors[0] #search in kmeans image for specific color and replace all by first color
                binary_output[((binary_output[:,:,0] == color[0]) & (binary_output[:,:,1] == color[1]) & (binary_output[:,:,2] == color[2]))]=[255,255,255] #search in binary_output image for specific color and replace all by first color
                kmeans_output[((kmeans_output[:,:,0] == color[0]) & (kmeans_output[:,:,1] == color[1]) & (kmeans_output[:,:,2] == color[2]))]=filtered_km_mask_FG_colors[0] #search in kmeans_output image for specific color and replace all by first color
                # km_mask_out[((km_mask_out[:,:,0] == color[0]) & (km_mask_out[:,:,1] == color[1]) & (km_mask_out[:,:,2] == color[2]))]=filtered_km_mask_FG_colors[0] #search in binary_output image for specific color and replace all by first color
                cv2.rectangle(km_mask_out, (x,y), (x+10,y+10), (int(color[0]),int(color[1]),int(color[2])), -1)
            for color in filtered_km_mask_BG_colors:
                print("filtered_km_mask_BG_colors",color)
                binary_output[((binary_output[:,:,0] == color[0]) & (binary_output[:,:,1] == color[1]) & (binary_output[:,:,2] == color[2]))]=[0,0,0] #search in binary_output image for specific color and replace all by first color
                # binary_output[((binary_output[:,:,0] == color[0]) & (binary_output[:,:,1] == color[1]) & (binary_output[:,:,2] == color[2]))]=filtered_km_mask_BG_colors[0] #search in binary_output image for specific color and replace all by first color
                # km_mask_out[((km_mask_out[:,:,0] == color[0]) & (km_mask_out[:,:,1] == color[1]) & (km_mask_out[:,:,2] == color[2]))]=filtered_km_mask_BG_colors[0] #search in binary_output image for specific color and replace all by first color
            colors_only_BG = kmeans_colors[np.invert(np.isin(kmeans_colors, km_mask_colors).all(axis=1))]    #colors in kmean_out wich do not appear in any mask
            for color in colors_only_BG: 
                print("colors_only_BG",color)
                binary_output[((binary_output[:,:,0] == color[0]) & (binary_output[:,:,1] == color[1]) & (binary_output[:,:,2] == color[2]))]=[0,0,0] #search in binary_output image for specific color and replace all by first color
                # kmeans_output[((kmeans_output[:,:,0] == color[0]) & (kmeans_output[:,:,1] == color[1]) & (kmeans_output[:,:,2] == color[2]))]=colors_only_BG[0] #search in kmeans_output image for specific color and replace all by first color
                # km_mask_out[((km_mask_out[:,:,0] == color[0]) & (km_mask_out[:,:,1] == color[1]) & (km_mask_out[:,:,2] == color[2]))]=colors_only_BG[0] #search in binary_output image for specific color and replace all by first color

            cv2.imwrite(os.path.basename(img_name)+'_inkm_step5_kmeans_masked.jpg',km_mask_out)
            cv2.imwrite(os.path.basename(img_name)+'_inkm_step5_kmean_colorreplaced.jpg',kmeans_output)
            cv2.imwrite(os.path.basename(img_name)+'_inkm_step5_kmean_binarization.jpg',binary_output)
            


            # cv2.imwrite(os.path.basename(img_bdcn)+'_kmeans_output_after.jpg',kmeans_output)
            # cv2.imwrite(os.path.basename(img_bdcn)+'_km_mask_out.jpg',km_mask_out)
            # cv2.namedWindow("binary_output", cv2.WINDOW_NORMAL)

            # cv2.imshow("binary_output", binary_output)
            # cv2.waitKey(0)
            #apply erosure/dilation morphing filters
            # kernel = np.ones((5,5),np.uint8)
            
            #reprint text because medianBlur destroys it
            # cv2.imshow("binary_output", binary_output)
            # cv2.waitKey(0)


            if COLORED_CANVAS:
                #remove small kmeans fragments
                kmeans_output = cv2.medianBlur(kmeans_output,7)
                cv2.rectangle(kmeans_output, (0,0), (235,30), (0,0,0), -1)
                cv2.putText(kmeans_output, "Foreground color:", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))
                cv2.rectangle(kmeans_output, (220,15), (230,25), (int(filtered_km_mask_FG_colors[0][0]),int(filtered_km_mask_FG_colors[0][1]),int(filtered_km_mask_FG_colors[0][2])), -1)
                cv2.rectangle(kmeans_output, (220,15), (230,25), (255,255,255), 1)
                
                cv2.imwrite(os.path.basename(img_name)+'_inkm_step6_kmean_fragementsremoved.jpg',kmeans_output)
                output_canvas = kmeans_output
            else:
                # try:
                #     #medianblur randomly crashes. we can also continue without
                #     binary_output = cv2.medianBlur(binary_output,int(esz*60)+1 if int(esz*60)%2==0 else int(esz*60))
                # finally:
                #     pass
                
                kernel = np.array([
                    [0,0,0.5,0,0],
                    [0,1,1,1,0],
                    [0.5,1,1,1,0.5],
                    [0,1,1,1,0],
                    [0,0,0.5,0,0]], dtype=np.uint8)

                binary_output = cv2.dilate(binary_output,kernel,iterations = 10)
                binary_output = cv2.erode(binary_output,kernel,iterations = 10)
                # cv2.imshow("binary_output", binary_output)
                # cv2.waitKey(0)
                # cv2.imshow("binary_output", binary_output)
                # cv2.waitKey(0)
                cv2.imwrite(os.path.basename(img_name)+'_inkm_step6_1_kmean_morphclose.jpg',binary_output)
                # binary_output = cv2.medianBlur(binary_output,int(esz*60)+1 if int(esz*60)%2==0 else int(esz*60))
                # cv2.imwrite(os.path.basename(img_name)+'_inkm_step6_2_kmean_morpblurred.jpg',binary_output)
                # cv2.imshow("binary_output", binary_output)
                # cv2.waitKey(0)
                binary_output = cv2.erode(binary_output,kernel,iterations = 10)
                binary_output = cv2.dilate(binary_output,kernel,iterations = 10)
                # cv2.imshow("binary_output", binary_output)
                # cv2.waitKey(0)
                cv2.imwrite(os.path.basename(img_name)+'_inkm_step6_2_kmean_morpopen.jpg',binary_output)
                # cv2.imshow("binary_output", binary_output)
                # cv2.waitKey(0)

                # binary_output = 
                # cv2.imshow(img_name+"kmeans_masked", km_mask_out)
                #replace colors
                # binary_output = cv2.medianBlur(binary_output,53)
                # binary_output = cv2.bilateralFilter(binary_output,30,60,60)
                # kmeans_bgfg = imgKmeans(binary_output, 3)
                # cv2.namedWindow("kmeans_bgfg", cv2.WINDOW_NORMAL)
                # cv2.imshow("kmeans_bgfg", kmeans_bgfg)
                cv2.rectangle(binary_output, (0,0), (235,30), (0,0,0), -1)
                cv2.putText(binary_output, "Foreground color:", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))
                cv2.rectangle(binary_output, (220,15), (230,25), (int(filtered_km_mask_FG_colors[0][0]),int(filtered_km_mask_FG_colors[0][1]),int(filtered_km_mask_FG_colors[0][2])), -1)
                cv2.rectangle(binary_output, (220,15), (230,25), (255,255,255), 1)
                output_canvas = binary_output
        else:
            print("WARNING: skipped fg/bg calc")


        


    if TRIANGLES or BODY_LINES:
        triangles = [poseToTriangle(pose) for pose in datum.poseKeypoints]
        for triangle in triangles:
            if triangle is not None:
                if TRIANGLES:
                    cv2.drawContours(img, [triangle], 0, 255, -1)
                    cv2.drawContours(output_canvas, [triangle], 0, 255, -1)
                if BODY_LINES:
                    linePoints = triangleToBodyLine(triangle)
                    cv2.line(img, trp(linePoints[0]), trp(linePoints[1]), (0,255,0), int(6*esz))
                    cv2.line(output_canvas, trp(linePoints[0]), trp(linePoints[1]), (0,255,0), int(6*esz))

    if BISEC_CONES or GLOBAL_LINE:
        print("BISEC_CONES",BISEC_CONES);
        bisecCones = [poseToBisectCone(pose, max_lw, BISEC_CONE_ANGLE, CORRECTION_ANGLE) for pose in datum.poseKeypoints] #use max(img.height,width) as cone length
        bisecCones = [v for v in bisecCones if v] #remove None values
        #print([polyToArr(bisecCone) for bisecCone in bisecCones])
        intersections = coneIntersections(bisecCones)
        if(len(intersections)>0):
            maxlevel = max(map(lambda t: len(t), intersections.keys()))
            for combi in intersections:
                is_not_last_level = len(combi) < maxlevel
                if is_not_last_level and BISEC_SKIP_LOWER_LEVELS:
                    continue;
                else:
                    overlay = np.zeros((len(img),len(img[0]),3), np.uint8)
                    color = min(((len(combi)-1)*100,255))
                    alpha = OVERLAY_ALPHA
                    if DRAW_FIRST_CONE_LAYER_BRIGTHER and len(combi) == 1:
                        cv2.drawContours(overlay, [polyToArr(intersections[combi])], 0, (0,255,0), -1)
                        img = overlay_two_image_v2(img, overlay, [0,0,0], (0.25))
                        intermediate_output_canvas = overlay_two_image_v2(intermediate_output_canvas, overlay, [0,0,0], (alpha if is_not_last_level else 0.6))
                    if BISEC_CONES:
                        cv2.drawContours(overlay, [polyToArr(intersections[combi])], 0, (color,0,(0 if is_not_last_level else 255)), -1)
                        intermediate_output_canvas = overlay_two_image_v2(img, overlay, [0,0,0], (alpha if is_not_last_level else 0.6))
                    img = overlay_two_image_v2(img, overlay, [0,0,0], (alpha if is_not_last_level else 0.6))
                    intermediate_output_canvas = overlay_two_image_v2(intermediate_output_canvas, overlay, [0,0,0], (alpha if is_not_last_level else 0.6))


                    # if(len(combi)==1 or not is_not_last_level):
                    #     black_debug = np.array([[[0,0,0]]*len(img[0])]*len(img),np.uint8)
                    #     cv2.drawContours(black_debug, [polyToArr(intersections[combi])], 0, (255,255,255), -1)
                    #     cv2.namedWindow(str(combi), cv2.WINDOW_NORMAL)
                    #     cv2.imshow(str(combi), black_debug)

                    if not is_not_last_level and GLOBAL_LINE: #draw centroid of last polygon
                        xy = (int(intersections[combi].centroid.x),int(intersections[combi].centroid.y))
                        global_angle = getGlobalLineAngle(datum.poseKeypoints, CORRECTION_ANGLE)
                        print("global_angle",np.rad2deg(global_angle))
                        dist = max_lw
                        d = (int(dist * np.cos(global_angle)), int(dist * np.sin(global_angle))) #with origin zero
                        d_l = (int(-dist * np.cos(global_angle)), int(-dist * np.sin(global_angle))) #with origin zero
                        # draw line with global gaze angle (special mean of all gaze angles) and through center of last intersection
                        cv2.line(img, xy, (xy[0]+d[0],xy[1]-d[1]), (0,255,255), int(10*esz))
                        cv2.line(output_canvas, xy, (xy[0]+d[0],xy[1]-d[1]), (0,255,255), int(10*esz))
                        cv2.line(intermediate_output_canvas, xy, (xy[0]+d[0],xy[1]-d[1]), (0,255,255), int(10*esz))
                        cv2.line(img, xy, (xy[0]+d_l[0],xy[1]-d_l[1]), (0,255,255), int(10*esz))
                        cv2.line(output_canvas, xy, (xy[0]+d_l[0],xy[1]-d_l[1]), (0,255,255), int(10*esz))
                        cv2.line(intermediate_output_canvas, xy, (xy[0]+d_l[0],xy[1]-d_l[1]), (0,255,255), int(10*esz))
                        cv2.circle(img, xy, int(13*esz), (255,255,0), -1)
                        cv2.circle(output_canvas, xy, int(13*esz), (255,255,0), -1)
                        cv2.circle(intermediate_output_canvas, xy, int(13*esz), (255,255,0), -1)


                        # put markers on image border if centroid is outside of the image
                        if(xy[0]<0):
                            if(xy[1]<0):
                                # draw in top left: (0,0)
                                cv2.arrowedLine(img, (int(46*esz),int(46*esz)), (0,0), (255,255,0), int(5*esz))
                                cv2.arrowedLine(output_canvas, (int(46*esz),int(46*esz)), (0,0), (255,255,0), int(5*esz))
                                cv2.arrowedLine(intermediate_output_canvas, (int(46*esz),int(46*esz)), (0,0), (255,255,0), int(5*esz))
                            if(xy[1]>len(img[0])):
                                # draw in bottom left (0,len(img[0]))
                                cv2.arrowedLine(img, (int(46*esz),len(img[0])-int(46*esz)), (0,len(img)), (255,255,0), int(5*esz))
                                cv2.arrowedLine(output_canvas, (int(46*esz),len(img[0])-int(46*esz)), (0,len(img)), (255,255,0), int(5*esz))
                                cv2.arrowedLine(intermediate_output_canvas, (int(46*esz),len(img[0])-int(46*esz)), (0,len(img)), (255,255,0), int(5*esz))
                            else:
                                # draw on (0,xy[1])
                                cv2.arrowedLine(img, (int(46*esz),xy[1]), (0,xy[1]), (255,255,0), int(5*esz))
                                cv2.arrowedLine(output_canvas, (int(46*esz),xy[1]), (0,xy[1]), (255,255,0), int(5*esz))
                                cv2.arrowedLine(intermediate_output_canvas, (int(46*esz),xy[1]), (0,xy[1]), (255,255,0), int(5*esz))
                        elif (xy[0]>len(img[0])):
                            if(xy[1]<0):
                                # draw in top right: (len(img),0)
                                cv2.arrowedLine(img, (len(img[0])-int(46*esz),int(46*esz)), (len(img[0]),0), (255,255,0), int(5*esz))
                                cv2.arrowedLine(output_canvas, (len(img[0])-int(46*esz),int(46*esz)), (len(img[0]),0), (255,255,0), int(5*esz))
                                cv2.arrowedLine(intermediate_output_canvas, (len(img[0])-int(46*esz),int(46*esz)), (len(img[0]),0), (255,255,0), int(5*esz))
                            if(xy[1]>len(img[0])):
                                # draw in bottom right (len(img),len(img[0]))
                                cv2.arrowedLine(img, (len(img[0])-int(46*esz),len(img[0])-int(46*esz)), (len(img[0]),len(img)), (255,255,0), int(5*esz))
                                cv2.arrowedLine(output_canvas, (len(img[0])-int(46*esz),len(img[0])-int(46*esz)), (len(img[0]),len(img)), (255,255,0), int(5*esz))
                                cv2.arrowedLine(intermediate_output_canvas, (len(img[0])-int(46*esz),len(img[0])-int(46*esz)), (len(img[0]),len(img)), (255,255,0), int(5*esz))
                            else:
                                # draw on (len(img),xy[1])
                                cv2.arrowedLine(img, (len(img[0])-int(46*esz),xy[1]), (len(img[0]),xy[1]), (255,255,0), int(5*esz))
                                cv2.arrowedLine(output_canvas, (len(img[0])-int(46*esz),xy[1]), (len(img[0]),xy[1]), (255,255,0), int(5*esz))
                                cv2.arrowedLine(intermediate_output_canvas, (len(img[0])-int(46*esz),xy[1]), (len(img[0]),xy[1]), (255,255,0), int(5*esz))

                        # maybe even simpler as i thought, just put marker on (len(img[0], xy[1])
                        # if(xy[0]>len(img[0])): #outside of canvas, to the right
                        #     l = (len(img[0])-xy[0])/np.cos(global_angle)
                        #     y_new = int(xy[1]+l*np.sin(global_angle))
                        #     cv2.circle(img, (len(img[0]),y_new), int(13*esz), (255,235,50), -1)
                        #     cv2.circle(output_canvas, (len(img[0]),y_new), int(13*esz), (255,235,50), -1)
                        #     cv2.circle(intermediate_output_canvas, (len(img[0]),y_new), int(13*esz), (255,235,50), -1)
        else:
            print("WARNING-------------------WARNING: no intersections there")
    
    cv2.waitKey(0)

    if BISEC_VECTORS:
        bisecVectors = [poseToBisectVector(pose, CORRECTION_ANGLE) for pose in datum.poseKeypoints]
        for bisecVector in bisecVectors:
            if bisecVector is not None:
                cv2.arrowedLine(img, trp(bisecVector[1]), trp(bisecVector[0]), (0,0,255), int(4*esz))
                cv2.arrowedLine(intermediate_output_canvas, trp(bisecVector[1]), trp(bisecVector[0]), (0,0,255), int(4*esz))

    if BISEC_CONES or BISEC_VECTORS:
        cv2.imwrite(os.path.basename(img_name)+'_bcones_io.jpg',intermediate_output_canvas)
                
    if SAVE_FILE:
        #cv2.imwrite(os.path.basename(img_name),img)
        if COLORED_CANVAS:
            cv2.imwrite(os.path.basename(img_name)+'_final_colored_canvas.jpg',output_canvas)
            print("saved _final_colored_canvas")
        else:
            cv2.imwrite(os.path.basename(img_name)+'_final_binary_canvas.jpg',output_canvas)
    if SHOW_WINDOW:
        cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
        cv2.imshow(img_name, img)
        cv2.namedWindow(img_name+"canvas", cv2.WINDOW_NORMAL)
        cv2.imshow(img_name+"canvas", output_canvas)
        cv2.waitKey(0)
if SHOW_WINDOW:
    cv2.waitKey(0)
    cv2.destroyAllWindows()