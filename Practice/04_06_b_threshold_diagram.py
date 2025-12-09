
import numpy as np
import cv2
from matplotlib import pyplot as plt

thresh_value = 80
thresh_max_val = 200


def get_diagram_as_image(fig_in):
    fig_in.canvas.draw()
    data = np.array(fig_in.canvas.renderer.buffer_rgba())
    data_bgr = cv2.cvtColor(data, cv2.COLOR_RGBA2BGR)

    return data_bgr


def plot_thresh_result(res, title_text, fname=None):
    ax.clear()
    ax.set_title(title_text)
    plt.xlim([0, 255])
    plt.ylim([-10, 265])
    # plt.axis('equal')
    # plt.xlabel('Original intensity value')
    # plt.ylabel('After thresholding/clipping')
    # plt.scatter(125, 0, c='r')
    plt.plot(x, x, 'g--', label='Original')
    plt.axvline(x=thresh_value, c='b', label='thresh')
    plt.plot(x, max_val_values, 'y--', linewidth=1, label='maxval')
    plt.plot(res, c='r', linewidth=3, label='Result')
    plt.legend()
    res_lut_diag = get_diagram_as_image(fig)
    cv2.imshow('lut_result', res_lut_diag)
    if fname is not None:
        cv2.imwrite(fname, res_lut_diag)
    cv2.waitKey(0)


# Base points [0, 255]
x = np.arange(0, 256, 1, np.uint8)
max_val_values = np.arange(0, 256, 1, np.uint8)
max_val_values.fill(thresh_max_val)

# Plot settings
fig = plt.figure(figsize=(4, 4), dpi=100)
# fig = plt.figure()
ax = fig.add_subplot(111)
fig.canvas.manager.set_window_title('cv2.threshold() LUT plots')
plt.xlim([0, 255])
plt.ylim([-10, 265])

_, res1 = cv2.threshold(x, thresh_value, thresh_max_val, cv2.THRESH_BINARY)
plot_thresh_result(res1, 'cv2.threshold() THRESH_BINARY', '03_05_b_thresh_binary.png')

_, res2 = cv2.threshold(x, thresh_value, thresh_max_val, cv2.THRESH_BINARY_INV)
plot_thresh_result(res2, 'cv2.threshold() THRESH_BINARY_INV', '03_05_b_thresh_binary_inv.png')

_, res3 = cv2.threshold(x, thresh_value, thresh_max_val, cv2.THRESH_TRUNC)
plot_thresh_result(res3, 'cv2.threshold() THRESH_BINARY_TRUNC', '03_05_b_thresh_trunc.png')

_, res4 = cv2.threshold(x, thresh_value, thresh_max_val, cv2.THRESH_TOZERO)
plot_thresh_result(res4, 'cv2.threshold() THRESH_TOZERO', '03_05_b_thresh_tozero.png')

_, res5 = cv2.threshold(x, thresh_value, thresh_max_val, cv2.THRESH_TOZERO_INV)
plot_thresh_result(res5, 'cv2.threshold() THRESH_TOZERO_INV', '03_05_b_thresh_tozero_inv.png')
