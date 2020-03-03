import numpy as np

def trp(point): #trim array to Point
    return (int(point[0]),int(point[1]))

def overlay_two_image_v2(image, overlay, ignore_color=[0,0,0], alpha=0.1):
    ignore_color = np.asarray(ignore_color)
    mask = (overlay==ignore_color).all(-1,keepdims=True)
    out = np.where(mask,image,(image * (1-alpha) + overlay * alpha).astype(image.dtype))
    return out