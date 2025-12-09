
import cv2
import numpy as np

button_pressed = False


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


def inverse_dft(dft_in):
    global orig_width, orig_height

    # Inverse DFT
    filtered_shift = do_ifft_shift(dft_in)
    idft_cplx = cv2.idft(filtered_shift, flags=cv2.DFT_SCALE)

    # Check whether imaginary is (near) zero
    idft_imag = idft_cplx[:, :, 1]
    print('idft imaginary dtype min max:', idft_imag.dtype, np.min(idft_imag), np.max(idft_imag))

    # Take the real part of the result
    idft = idft_cplx[:, :, 0]
    print('idft dtype min max:', idft.dtype, np.min(idft), np.max(idft))
    filtered_img = np.uint8(np.clip(idft, 0, 1) * 255)[:orig_height, :orig_width]
    cv2.imshow('filtered_img', filtered_img)


def modify_dft(x, y, keep_modified=False):
    global dft_shift, max_coef

    print(x, y)
    # Modify frequency
    dft_shift_copy = dft_shift.copy()
    print('Modify frequency')
    w = dft_shift.shape[1]
    h = dft_shift.shape[0]
    print('Before:', dft_shift[y, x], dft_shift[h - y, w - x])
    dft_shift_copy[y, x, 1] = np.sign(dft_shift_copy[y, x, 1]) * max_coef
    if y != h - y and x != w - x:  # if not the center position
        # Assure complex conjugate symmetry
        dft_shift_copy[h - y, w - x, 1] = -1.0 * dft_shift_copy[y, x, 1]
    print('After:', dft_shift_copy[y, x], dft_shift_copy[h - y, w - x])
    display_freq_log_magnitude(dft_shift_copy, 'dft log magnitude')
    inverse_dft(dft_shift_copy)
    if keep_modified:
        dft_shift = dft_shift_copy


# Read input image and convert to float32
img_in = cv2.imread('GolyoAlszik_rs.jpg', cv2.IMREAD_GRAYSCALE)
# img_in = cv2.imread('OpenCV-logo.png', cv2.IMREAD_GRAYSCALE)
# img_in = np.zeros((256, 256), np.uint8)
# img_in[110:146, 124:132] = 255
cv2.imshow('img_in', img_in)
orig_width = img_in.shape[1]
orig_height = img_in.shape[0]

print('Proposed padding of the original image sizes:')
padded_width = cv2.getOptimalDFTSize(img_in.shape[1])
padded_height = cv2.getOptimalDFTSize(img_in.shape[0])
print('Width:', img_in.shape[1], '->', cv2.getOptimalDFTSize(img_in.shape[1]))
print('Height:', img_in.shape[0], '->', cv2.getOptimalDFTSize(img_in.shape[0]))
im = np.zeros((padded_height, padded_width), np.uint8)
im[:img_in.shape[0], :img_in.shape[1]] = img_in

# Image DFT + shift
img_float = np.float32(im) / 255
print('img_float sum:', np.sum(img_float))
dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
print('Center:', dft[0, 0])
dft_shift = do_fft_shift(dft)
# max_coef = np.max(dft_shift) / 2
max_coef = 18000

# Display image frequency magnitude
display_freq_log_magnitude(dft_shift, 'dft log magnitude')
inverse_dft(dft_shift)
x = 277
y = 185
modify_dft(x, y, True)

cv2.waitKey(0)
cv2.destroyAllWindows()
