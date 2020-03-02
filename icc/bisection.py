import numpy as np
import numpy.linalg as la
from shapely.geometry import Polygon
import itertools

def polyToArr(poly):
    if poly is not None and poly.exterior is not None:
        return np.array(poly.exterior.coords.xy).transpose().astype(int)
    else:
        return None

def getAngle(a,b,c, CORRECTION_ANGLE):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle
def getAngleGroundNormed(a,b,c, CORRECTION_ANGLE):
    if(a[0]-b[0]<0):
        b_plane_point = np.array([b[0]-50,b[1]])
    else:
        b_plane_point = np.array([b[0]+50,b[1]])
    vector_angle = getAngle(a,b,c, CORRECTION_ANGLE)
    ground_angle = getAngle(a,b,b_plane_point, CORRECTION_ANGLE)
    normed_angle = vector_angle/2 - ground_angle
    if(a[0]-b[0]<0):
        return (normed_angle+np.deg2rad(180-CORRECTION_ANGLE))
    else:
        return (np.deg2rad(360+CORRECTION_ANGLE)-normed_angle)

def angleMapper(pose, CORRECTION_ANGLE):
    angle = getAngleGroundNormed(*pose[[0,1,8]][:,:2], CORRECTION_ANGLE)
     #map all angles to one direction, so the mean does not get influences by left/right direction
    if(angle > np.deg2rad(180)):
        mapped = angle - np.deg2rad(180)
        return mapped - np.deg2rad(180) if mapped > np.deg2rad(90) else mapped
    else:
        mapped = angle
        return mapped - np.deg2rad(180) if mapped > np.deg2rad(90) else mapped

def getGlobalLineAngle(poses, CORRECTION_ANGLE):
    return np.mean([angleMapper(pose, CORRECTION_ANGLE) for pose in poses if not 0.0 in pose[[0,1,8]][:,2:]])

def getBisecPoint(a,b,c, CORRECTION_ANGLE):
    angle = getAngleGroundNormed(a,b,c, CORRECTION_ANGLE)
    dist = la.norm(a-b)*2
    d = (int(dist * np.cos(angle)), int(dist * np.sin(angle))) #with origin zero
    out = (b[0]+d[0],b[1]-d[1])
    return out #with origin b

def getBisecCone(a,b,c, length, BISEC_CONE_ANGLE, CORRECTION_ANGLE):
    angle = getAngleGroundNormed(a,b,c, CORRECTION_ANGLE)
    conePoint1 = (int(length * np.cos(angle-(BISEC_CONE_ANGLE/2))), int(length * np.sin(angle-(BISEC_CONE_ANGLE/2)))) #with origin zero
    conePoint2 = (int(length * np.cos(angle+(BISEC_CONE_ANGLE/2))), int(length * np.sin(angle+(BISEC_CONE_ANGLE/2)))) #with origin zero
    return ((b[0]+conePoint1[0],b[1]-conePoint1[1]),(b[0]+conePoint2[0],b[1]-conePoint2[1]))

def poseToBisectVector(pose, CORRECTION_ANGLE):
    points = pose[[0,1,8]]
    if(0.0 in points[:,2:]): #if one point has confidence zero, we can not generate the vector
        return None
    a,b,c = points[:,:2] # cut of confidence score so we have normal coordinate points
    bisecPoint = getBisecPoint(a,b,c, CORRECTION_ANGLE)
    return np.array([bisecPoint,b])

def poseToBisectCone(pose, length, angle, CORRECTION_ANGLE):
    width = np.deg2rad(angle)
    points = pose[[0,1,8]]
    if(0.0 in points[:,2:]): #if one point has confidence zero, we can not generate the vector
        return None
    a,b,c = points[:,:2] # cut of confidence score so we have normal coordinate points
    conePoint1, conePoint2 = getBisecCone(a,b,c,length,width, CORRECTION_ANGLE)
    return Polygon([conePoint1, conePoint2, b])

def coneIntersections(bisecCones):
    out = {}
    for r in range(1,len(bisecCones)+1):
        pc = list(itertools.combinations(range(0,len(bisecCones)),r))
        for combi in pc:
            intersect = bisecCones[combi[0]]
            for i in combi[1:]:
                intersect = intersect.intersection(bisecCones[i])
            if not intersect.is_empty:
                out[combi] = intersect
    return out