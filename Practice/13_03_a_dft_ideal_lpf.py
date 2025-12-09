import cv2
import numpy as np

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


# Read input image and convert to float32
img_in = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('OpenCV-logo.png', cv2.IMREAD_GRAYSCALE)
orig_width = img_in.shape[1]
orig_height = img_in.shape[0]

print('Proposed padding of the original image sizes:')
padded_width = cv2.getOptimalDFTSize(img_in.shape[1])
padded_height = cv2.getOptimalDFTSize(img_in.shape[0])
print('Width:', img_in.shape[1], '->', cv2.getOptimalDFTSize(img_in.shape[1]))
print('Height:', img_in.shape[0], '->', cv2.getOptimalDFTSize(img_in.shape[0]))
im = np.zeros((padded_height, padded_width), np.uint8)
im[:img_in.shape[0], :img_in.shape[1]] = img_in

img_float = np.float32(im) / 255
cv2.imshow('img_float', img_float)

# Image DFT + shift
dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = do_fft_shift(dft)

# Display image frequency magnitude
display_freq_log_magnitude(dft_shift, 'dft log magnitude')

# Create ideal low pass filter (as a centered circle)
circle_img = np.zeros(dft_shift.shape[:2], np.float32)
cv2.circle(circle_img, (int(circle_img.shape[1] / 2), int(circle_img.shape[0] / 2)), filter_radius, 1, cv2.FILLED)
cv2.imshow('circle', circle_img)

# The frequency filter must be complex
ideal_filter = np.ndarray(dft_shift.shape, np.float32)
ideal_filter[:, :, 0] = circle_img  # real part
ideal_filter[:, :, 1] = 0           # imaginary part
print(dft_shift.shape)
print(ideal_filter.shape)

# Apply the frequency filter and display the spectrum
filtered = cv2.mulSpectrums(dft_shift, ideal_filter, 0)
display_freq_log_magnitude(filtered, 'filtered log magnitude')
cv2.waitKey(0)

# Inverse DFT
filtered_shift = do_ifft_shift(filtered)
idft_cplx = cv2.idft(filtered_shift, flags=cv2.DFT_SCALE)

# Check whether imaginary is (near) zero
idft_imag = idft_cplx[:, :, 1]
print('idft imaginary dtype min max', idft_imag.dtype, np.min(idft_imag), np.max(idft_imag))

# Take the real part of the result
idft = idft_cplx[:, :, 0]
print('idft dtype min max', idft.dtype, np.min(idft), np.max(idft))
filtered_img = np.uint8(np.clip(idft, 0, 1) * 255)[:orig_height, :orig_width]
# filtered_img = cv2.normalize(idft, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
cv2.imshow('filtered_img', filtered_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
