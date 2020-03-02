import numpy as np

def reduceToTrianglePoint(indices,pose):
    points = pose[indices] #filter out poses wich are not part of the points points indices
    if len(points) <= 0:
        return None #no relevant indices exist
    pointsFiltered = points[np.where(points[:,2] > 0.1)[0]] #filter out points bellow openpose confidence value of 0.1
    if len(pointsFiltered) <= 0 or pointsFiltered is None:
        bestPoint = points[np.argsort(points[:,2])[::-1]][0] #points with best openpose confidence
        if bestPoint[2]>0.0:
            return bestPoint
        else:
            return None
        return None #else return point with best openpose confidence
    else:
        return pointsFiltered[0] #take first point if filterd has values

def poseToTriangle(pose):
    triangleTopSortedIndices = [1,0,2,5,15,16,17,18]
    triangleLeftSortedIndices = [11,24,13,22,10,9]
    triangleRightSortedIndices = [14,21,19,20,13,12]
    top = reduceToTrianglePoint(triangleTopSortedIndices,pose)
    left = reduceToTrianglePoint(triangleLeftSortedIndices,pose)
    right = reduceToTrianglePoint(triangleRightSortedIndices,pose)

    if top is not None and left is not None and right is not None:
        retval = np.array([(top[0],top[1]),(left[0],left[1]),(right[0],right[1])])
        return np.array(retval).astype(int)
    else:
        return None

def triangleToBodyLine(triangle):
    bottom_middle_point = (triangle[1]+triangle[2])/2
    return (triangle[0],bottom_middle_point)
