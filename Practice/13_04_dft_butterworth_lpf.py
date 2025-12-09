import cv2
import numpy as np
import math

filter_radius = 40


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


# Create a single channel butterworth low-pass filter
# with radius d, order n, and of given matrix shape
def create_butterworth_lowpass_filter(shape, d, n):
    filter_res = np.zeros(shape[:2], np.float32)
    center_x = filter_res.shape[1] / 2
    center_y = filter_res.shape[0] / 2
    for y in range(0, filter_res.shape[0]):
        for x in range(0, filter_res.shape[1]):
            radius = math.sqrt(math.pow(x - center_x, 2) + math.pow(y - center_y, 2))
            filter_res[y, x] = 1.0 / (1 + math.pow(radius / d, (2 * n)))

    return filter_res


# Read input image and convert to float32
im = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('OpenCV-logo.png', cv2.IMREAD_GRAYSCALE)
img_float = np.float32(im) / 255
cv2.imshow('img_float', img_float)

# Image DFT + shift
dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = do_fft_shift(dft)

# Display image frequency magnitude
display_freq_log_magnitude(dft_shift, 'dft log magnitude')
cv2.waitKey(0)

# Create complex Butterworth low-pass frequency filter
filter_img = create_butterworth_lowpass_filter(dft_shift.shape[:2], filter_radius, 2)
freq_filter = np.zeros(dft_shift.shape, np.float32)
freq_filter[:, :, 0] = filter_img  # real part
freq_filter[:, :, 1] = 0           # imaginary part
cv2.imshow('Butterworth low-pass', filter_img)

# Apply the frequency filter and display the spectrum
filtered = cv2.mulSpectrums(dft_shift, freq_filter, 0)
display_freq_log_magnitude(filtered, 'filtered log magnitude')
cv2.waitKey(0)

# Inverse DFT
filtered_shift = do_ifft_shift(filtered)
idft_cplx = cv2.idft(filtered_shift, flags=cv2.DFT_SCALE)

# Take the real part of the result
idft = idft_cplx[:, :, 0]
print('idft dtype min max', idft.dtype, np.min(idft), np.max(idft))
filtered_img = np.uint8(np.clip(idft, 0, 1) * 255)
cv2.imshow('filtered_img', filtered_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
