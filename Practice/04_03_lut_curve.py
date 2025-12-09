
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
# https://rawpedia.rawtherapee.com/Exposure#Parametric

import cv2
from scipy import interpolate
import numpy as np
from matplotlib import pyplot as plt


def get_diagram_as_image(fig_in):
    fig_in.canvas.draw()
    data = np.array(fig_in.canvas.renderer.buffer_rgba())
    data_bgr = cv2.cvtColor(data, cv2.COLOR_RGBA2BGR)

    return data_bgr


def create_curve_func(points, requested_kind='automatic'):
    """Return a function derived from control points."""
    global curve_kind

    curve_kind = 'none'

    if points is None:
        return None

    num_points = len(points)
    if num_points < 2:
        return None

    xs, ys = zip(*points)

    ax.scatter(xs, ys, color='r')

    if requested_kind == 'automatic':
        if num_points < 3:
            kind = 'linear'
        elif num_points < 4:
            kind = 'quadratic'
        else:
            kind = 'cubic'
    else:
        kind = requested_kind

    curve_kind = kind

    return interpolate.interp1d(xs, ys, kind, bounds_error=False)


im = cv2.imread('SeaCliffBridge_3_800.jpg', cv2.IMREAD_COLOR)
cv2.imshow('Original', im)

# Diagram settings
fig = plt.figure(figsize=(4, 4), dpi=100)
ax = fig.add_subplot(111)
ax.set_title('Tone curve LUT diagram')
plt.xlabel('Original intensity values')
plt.ylabel('Result of point operation')
plt.xlim([0, 255])
plt.ylim([0, 255])
plt.tight_layout()

# point_list = [(0, 0), (186, 216), (255, 255)]
# point_list = [(0, 0), (50, 30), (80, 70), (140, 160), (200, 220), (255, 255)]
point_list = [(0, 0), (70, 40), (128, 128), (186, 216), (255, 255)]
# point_list = [(0, 0), (25, 40), (80, 120), (170, 170), (255, 255)]

curve_kind = 'none'
curve = create_curve_func(point_list, 'automatic')
# curve = create_curve_func(point_list, 'linear')
orig_value = np.arange(0, 256, 1)
new_value = np.clip(curve(orig_value), 0, 255)
new_value_uint8 = np.uint8(new_value)

result = cv2.LUT(im, new_value_uint8)
cv2.imshow('Result', result)

# Show LUT diagram
ax.plot(orig_value, orig_value, 'g--', label='Original')
ax.plot(orig_value, new_value, 'r-', label='Tone curve (' + curve_kind + ')')
plt.legend()

lut_diag = get_diagram_as_image(fig)
cv2.imshow('lut_plot', lut_diag)

cv2.waitKey(0)
cv2.destroyAllWindows()
