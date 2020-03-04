
# From Python
import sys
import cv2
import numpy as np
import os
from sys import platform
from shapely import affinity
from shapely.geometry import Polygon
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
MAX_WIDTH = 1500 #in px
KMEANS_AREA_MASK_THRESHOLD = 0.06
KMEANS_K = 7
OUT_DIR = os.environ['OUT_DIR'] if 'OUT_DIR' in os.environ else './images/out/'
IN_DIR = os.environ['IN_DIR'] if 'IN_DIR' in os.environ else "./images/in/"    # images from imdahl

print("in dir",IN_DIR, "out dir", OUT_DIR)

if os.path.dirname(__file__):
    print("__file__",__file__)
    os.chdir(os.path.dirname(__file__)) #make sure our curdir is the dir of the script so all relative paths will work

params = dict()
params["model_folder"] = os.environ['OPENPOSE_MODELS']

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

images = [os.path.join(os.getcwd(), IN_DIR, f) for f in os.listdir(IN_DIR)] #make path absolute so os.chdir has no side effects
images.sort()
os.chdir(OUT_DIR) #save images in this dir


#filter out hidden files
images = list(filter(lambda e: os.path.basename(e)[0]!='.',images))

#filter images if we want to inspect single image:
# images = list(filter(lambda e: "Franziskus-Giotto1" in e,images))

for img_name in images:
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
        md = int(esz*4)+1 if int(esz*4)%2==0 else int(esz*4)
        bl1 = int(esz*20)
        bl2 = int(esz*40)
        target = cv2.medianBlur(target,md if md<14 else 13) #smoothen cracks
        target = cv2.bilateralFilter(target,bl1 if bl1<31 else 31 ,bl2 if bl2<80 else 80,bl2 if bl2<80 else 80)        #remove cracks
        print("filter vals: med",int(esz*5)+1 if int(esz*5)%2==0 else int(esz*5), "bil1", int(esz*25), "bil2,3", int(esz*40))
        cv2.imwrite(os.path.basename(img_name)+'_inkm_step1_crackremoval.jpg',target)

        fposes = np.array([np.array([line[:2] for line in pose if line[2] > 0]) for pose in datum.poseKeypoints]) #filtered poses without zero lines
        mask = np.zeros((len(img),len(img[0]),1), np.uint8)
        kmout_mask = np.zeros((len(img),len(img[0]),1), np.uint8)
        for pose in fposes: #remove bodys
            convexhull = Polygon(pose).convex_hull
            #inpainting
            sconvexhull = affinity.scale(convexhull, xfact=1.7, yfact=1.4, origin=convexhull.centroid)
            cv2.drawContours(mask, [polyToArr(sconvexhull)], 0, 255, int(15*esz))
            cv2.drawContours(mask, [polyToArr(sconvexhull)], 0, 255, -1)

            #kmeans check
            sconvexhull = affinity.scale(convexhull, xfact=1, yfact=0.7, origin=convexhull.centroid)
            cv2.drawContours(kmout_mask, [polyToArr(sconvexhull)], 0, 255, int(7*esz))
            cv2.drawContours(kmout_mask, [polyToArr(sconvexhull)], 0, 255, -1)
        cv2.rectangle(mask, (0,0), (len(img[0]),len(img)), 255, int(40*esz)) #remove frames
        #shift kmeans mask pixels downwards 40px
        kmout_mask = cv2.warpAffine(kmout_mask, np.float32([ [1,0,0], [0,1,30] ]), (kmout_mask.shape[:2][1], kmout_mask.shape[:2][0]))   

        
        #inpaint the image
        cv2.imwrite(os.path.basename(img_name)+'_inkm_step2_inpaintmask.jpg',mask)
        cv2.imwrite(os.path.basename(img_name)+'_inkm_step2_kmeansresmask.jpg',kmout_mask)
        inpainted = cv2.inpaint(target, mask, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(os.path.basename(img_name)+'_inkm_step3_inpainted.jpg',inpainted)
       
        #to kmeans
        kmeans_output = imgKmeans(inpainted, KMEANS_K)
        cv2.imwrite(os.path.basename(img_name)+'_inkm_step4_kmeansres.jpg',kmeans_output)
        kmout_mask=cv2.cvtColor(kmout_mask,cv2.COLOR_GRAY2BGR) #change mask to a 3 channel image 
        km_mask_out=cv2.subtract(kmout_mask,kmeans_output)     #subtract mask from kmeans result
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

            binary_output = kmeans_output.copy()

            cv2.putText(km_mask_out, "Foreground colors:", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))
            
            cv2.rectangle(kmeans_output, (0,0), (235,30), (0,0,0), -1)
            cv2.putText(kmeans_output, "Foreground color:", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))
            cv2.rectangle(kmeans_output, (220,15), (230,25), (int(filtered_km_mask_FG_colors[0][0]),int(filtered_km_mask_FG_colors[0][1]),int(filtered_km_mask_FG_colors[0][2])), -1)
            cv2.rectangle(kmeans_output, (220,15), (230,25), (255,255,255), 1)

            cv2.rectangle(km_mask_out, (0,0), (235,30), (0,0,0), -1)
            cv2.putText(km_mask_out, "Foreground colors:", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))
            cv2.rectangle(km_mask_out, (220,15), (230,25), (255,255,255), 1)

            offset=220;
            for color in filtered_km_mask_FG_colors:
                #print colors on screen
                print("filtered_km_mask_FG_colors",color)
                x = offset
                y = 15
                offset += 15
                #replace colors
                binary_output[((binary_output[:,:,0] == color[0]) & (binary_output[:,:,1] == color[1]) & (binary_output[:,:,2] == color[2]))]=[255,255,255] #search in binary_output image for specific color and replace all by first color
                kmeans_output[((kmeans_output[:,:,0] == color[0]) & (kmeans_output[:,:,1] == color[1]) & (kmeans_output[:,:,2] == color[2]))]=filtered_km_mask_FG_colors[0] #search in kmeans_output image for specific color and replace all by first color
                cv2.rectangle(km_mask_out, (x,y), (x+10,y+10), (int(color[0]),int(color[1]),int(color[2])), -1)
            for color in filtered_km_mask_BG_colors:
                print("filtered_km_mask_BG_colors",color)
                binary_output[((binary_output[:,:,0] == color[0]) & (binary_output[:,:,1] == color[1]) & (binary_output[:,:,2] == color[2]))]=[0,0,0] #search in binary_output image for specific color and replace all by first color
            colors_only_BG = kmeans_colors[np.invert(np.isin(kmeans_colors, km_mask_colors).all(axis=1))]    #colors in kmean_out wich do not appear in any mask
            for color in colors_only_BG: 
                print("colors_only_BG",color)
                binary_output[((binary_output[:,:,0] == color[0]) & (binary_output[:,:,1] == color[1]) & (binary_output[:,:,2] == color[2]))]=[0,0,0] #search in binary_output image for specific color and replace all by first color

            cv2.imwrite(os.path.basename(img_name)+'_inkm_step5_kmeans_masked.jpg',km_mask_out)
            cv2.imwrite(os.path.basename(img_name)+'_inkm_step5_kmean_colorreplaced.jpg',kmeans_output)
            cv2.imwrite(os.path.basename(img_name)+'_inkm_step5_kmean_binarization.jpg',binary_output)


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
                
                kernel = np.array([
                    [0,0,0.5,0,0],
                    [0,1,1,1,0],
                    [0.5,1,1,1,0.5],
                    [0,1,1,1,0],
                    [0,0,0.5,0,0]], dtype=np.uint8)

                binary_output = cv2.dilate(binary_output,kernel,iterations = 10)
                binary_output = cv2.erode(binary_output,kernel,iterations = 10)
                cv2.imwrite(os.path.basename(img_name)+'_inkm_step6_1_kmean_morphclose.jpg',binary_output)
                binary_output = cv2.erode(binary_output,kernel,iterations = 10)
                binary_output = cv2.dilate(binary_output,kernel,iterations = 10)

                cv2.imwrite(os.path.basename(img_name)+'_inkm_step6_2_kmean_morpopen.jpg',binary_output)
                cv2.rectangle(binary_output, (0,0), (235,30), (0,0,0), -1)
                cv2.putText(binary_output, "Foreground color:", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))
                cv2.rectangle(binary_output, (220,15), (230,25), (int(filtered_km_mask_FG_colors[0][0]),int(filtered_km_mask_FG_colors[0][1]),int(filtered_km_mask_FG_colors[0][2])), -1)
                cv2.rectangle(binary_output, (220,15), (230,25), (255,255,255), 1)

                
                cv2.rectangle(binary_output, (0,0), (235,30), (0,0,0), -1)
                cv2.putText(binary_output, "Foreground color:", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))
                cv2.rectangle(binary_output, (220,15), (230,25), (255,255,255), -1)
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