
import cv2
import numpy as np
import math

filter_radius = 40
filter_type = 0  # low-pass by default; 1: high-pass


def display_freq_log_magnitude(dft_in, wnd_title='Frequency magnitude'):
    # Displaying log normalized magnitude of the frequency spectrum
    im_spectrum_magnitude = np.log(1.0 + cv2.magnitude(dft_in[:, :, 0], dft_in[:, :, 1]))
    im_magnitude_norm = cv2.normalize(im_spectrum_magnitude, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    cv2.imshow(wnd_title, im_magnitude_norm)


def do_fft_shift(dft_in):
    dft_np_shift = np.fft.fftshift(dft_in)
    return cv2.merge([dft_np_shift[:, :, 1], dft_np_shift[:, :, 0]])


def do_ifft_shift(dft_in):
    dft_np_shift = np.fft.ifftshift(dft_in)
    return cv2.merge([dft_np_shift[:, :, 1], dft_np_shift[:, :, 0]])


def on_radius_change(pos):
    global filter_radius

    filter_radius = pos
    do_ideal_filtering()


def on_type_change(pos):
    global filter_type

    filter_type = pos
    do_ideal_filtering()


def do_ideal_filtering():
    # Create ideal low pass filter (as a centered circle)
    circle_img = np.zeros(dft_shift.shape[:2], np.float32)
    cv2.circle(circle_img, (int(circle_img.shape[1] / 2), int(circle_img.shape[0] / 2)), filter_radius, 1, cv2.FILLED)
    if filter_type == 1:
        circle_img = 1 - circle_img
    cv2.imshow('filter', circle_img)

    # The frequency filter must be complex
    ideal_filter = np.ndarray(dft_shift.shape, np.float32)
    ideal_filter[:, :, 0] = circle_img  # real part
    ideal_filter[:, :, 1] = 0  # imaginary part
    # print(dft_shift.shape)
    # print(ideal_filter.shape)

    # Apply the frequency filter and display the spectrum
    filtered = cv2.mulSpectrums(dft_shift, ideal_filter, 0)
    display_freq_log_magnitude(filtered, 'filtered log magnitude')

    # Inverse DFT
    filtered_shift = do_ifft_shift(filtered)
    idft_cplx = cv2.idft(filtered_shift, flags=cv2.DFT_SCALE)

    # Check whether imaginary is (near) zero
    # idft_imag = idft_cplx[:, :, 1]
    # print('idft imaginary dtype min max', idft_imag.dtype, np.min(idft_imag), np.max(idft_imag))

    # Take the real part of the result
    idft = idft_cplx[:, :, 0]
    # print('idft dtype min max', idft.dtype, np.min(idft), np.max(idft))
    if filter_type == 0:
        filtered_img = np.uint8(np.clip(idft, 0, 1) * 255)[:orig_height, :orig_width]
    else:
        filtered_img = cv2.normalize(idft, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)[:orig_height, :orig_width]
    cv2.imshow('filtered_img', filtered_img)
    # cv2.waitKey(1)


# Read input image and convert to float32
# im_in = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_GRAYSCALE)
im_in = cv2.imread('OpenCV-logo.png', cv2.IMREAD_GRAYSCALE)
orig_width = im_in.shape[1]
orig_height = im_in.shape[0]

print('Proposed padding of the original image sizes:')
padded_width = cv2.getOptimalDFTSize(im_in.shape[1])
padded_height = cv2.getOptimalDFTSize(im_in.shape[0])
print('Width:', im_in.shape[1], '->', cv2.getOptimalDFTSize(im_in.shape[1]))
print('Height:', im_in.shape[0], '->', cv2.getOptimalDFTSize(im_in.shape[0]))
im = np.zeros((padded_height, padded_width), np.uint8)
im[:im_in.shape[0], :im_in.shape[1]] = im_in

img_float = np.float32(im) / 255
cv2.imshow('img_float', img_float)

# Image DFT + shift
dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = do_fft_shift(dft)

# Display image frequency magnitude
display_freq_log_magnitude(dft_shift, 'dft log magnitude')

max_radius = int(math.sqrt((padded_width ** 2 + padded_height ** 2)) / 2)
cv2.createTrackbar('Radius', 'img_float', filter_radius, max_radius, on_radius_change)
cv2.createTrackbar('Type', 'img_float', 0, 1, on_type_change)
do_ideal_filtering()

cv2.waitKey(0)
cv2.destroyAllWindows()
