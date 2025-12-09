import cv2
import numpy as np


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


# Read input image and convert to float32
im = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_GRAYSCALE)
# im = cv2.imread('OpenCV-logo.png', cv2.IMREAD_GRAYSCALE)

im_float = np.float32(im) / 255.0
cv2.imshow('img_float', im_float)

# Image DFT + shift
dft = cv2.dft(im_float, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = do_fft_shift(dft)

# Display image frequency magnitude
display_freq_log_magnitude(dft_shift, 'dft log magnitude')
cv2.waitKey(0)

# Inverse DFT
filtered_shift = do_ifft_shift(dft_shift)
idft_cplx = cv2.idft(filtered_shift, flags=cv2.DFT_SCALE)

# Take the real part of the result
idft = idft_cplx[:, :, 0]
print('idft dtype min max', idft.dtype, np.min(idft), np.max(idft))
im_inverse_dft = np.uint8(np.clip(idft, 0, 1) * 255)
# filtered_img = cv2.normalize(idft, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
cv2.imshow('im_inverse_dft', im_inverse_dft)

cv2.waitKey(0)
cv2.destroyAllWindows()
