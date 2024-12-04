import cv2
import numpy as np

def load_image(fname, mode='RGB', return_orig=False):
    img = np.array(Image.open(fname).convert(mode))
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img


def hconcat_resize_min(im_list, interpolation=cv2.INTER_LANCZOS4):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def drawMask(points, image, device='cuda'):
    points = np.array(points)
    cv2.fillPoly(image, [points], color=(255, 255, 255))
    return image


def drawDilateMask(points, image, kernel_size=5, device='cuda'):
    points = np.array(points)
    cv2.fillPoly(image, [points], color=(255, 255, 255))
    cv2.dilate(image, (kernel_size, kernel_size), iterations=6)
    return image


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    # return Image.fromarray(img)
    if img.shape[-1] == 4:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
    elif img.shape[-1] == 3:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError(f"Invalid number of channels: {img.shape[-1]}")


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return np.asarray(img)
    if img.mode == 'RGBA':
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    elif img.mode == 'RGB':
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Invalid mode: {img.mode}")


def style_detect(img, boundary_pts, color_thres=40):
    # for human part ONLY
    bg_color = False
    row, col, channel = img.shape
    alpha_mask = np.zeros((row, col), dtype=np.uint8)

    xy1 = (max(int(boundary_pts[0]), 0), max(int(boundary_pts[1]), 0))
    xy2 = (min(int(boundary_pts[2]), col-1), max(int(boundary_pts[3]), 0))
    xy3 = (min(int(boundary_pts[4]), col-1), min(int(boundary_pts[5]), row-1))
    xy4 = (max(int(boundary_pts[6]), 0), min(int(boundary_pts[7]), row-1))
    boundary = np.array([xy1, xy2, xy3, xy4])
    # obtain boundary size info
    center, wh, orient = cv2.minAreaRect(boundary)

    cv2.fillPoly(alpha_mask, [boundary], color=(255,))
    # count transparent area
    trans_arr = np.where(alpha_mask == 0)[0]
    print(trans_arr.shape)

    # create binary
    cropped_region = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA) # BGR2RGBA?
    cropped_region[:, :, 3] = alpha_mask
    #image_file = image_file.convert('LA') # converts to grayscale w/ alpha

    img_gray = Image.fromarray(cropped_region).convert('LA') #img_gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
    img_gray = np.array(img_gray).astype(np.uint8) # gray, cv2.cvtColor no need
    thres, img_bin = cv2.threshold(img_gray[:, :, 0], 100, 255, cv2.THRESH_OTSU)

    #plt.axis("on")
    ##plt.imshow(cv2.cvtColor(cropped_region, cv2.COLOR_RGBA2BGRA))
    #tmp_ = img_bin.copy()
    #tmp_[img_gray[:, :, -1] == 0] = 200
    #plt.imshow(cv2.cvtColor(tmp_, cv2.COLOR_RGBA2BGRA))
    #plt.title(f"blending with alpha channel {img.shape} {img_bin.shape} {trans_arr.shape}")
    #plt.show()
    #del tmp_

    img_bin[img_gray[:, :, -1] == 0] = 2 # non-transparent area
    img_bin[img_bin == 255] = 1

    #arr = np.sort(img_bin.sum(axis=0)) # sum by row
    #one_sum = arr[len(arr) - 1]
    #zero_sum = img_bin.shape[0] - arr[0] - trans_arr.shape[0]
    one_sum = len(np.where(img_bin == 1)[0])
    zero_sum = row*col - one_sum - len(trans_arr)
    print(f"one_sum: {one_sum}, zero_sum: {zero_sum}, redundance {one_sum * 0.8}")
    if zero_sum < one_sum * 0.8:
        img_bin = (1 - img_bin)

    # get component colors
    var_bg = np.where(img_bin == 0)
    colors = cropped_region[:,:,:3][var_bg[0], var_bg[1]]
    background_color = np.mean(colors, axis=0)
    # to reduce the overall computations, this ver only support blue text background search
    #if abs(background_color[2] - 50) < 10 and abs(background_color[1]-110) < 15 and abs(background_color[0]-160) < 20:

    n_iter = 1
    kernel = 5 #max(int(wh[0] / 10), int(wh[1] / 10))
    #print(f'kernel check it out {kernel}')
    bound_ext0 = cv2.dilate(alpha_mask, (kernel, kernel), iterations=n_iter)
    var_edge_0 = np.where((bound_ext0 == 255) & (alpha_mask == 0))
    edge_0_colors = img[:,:,:3][var_edge_0[0], var_edge_0[1]]
    edge_0_mean = np.mean(edge_0_colors, axis=0)
    dist_0 = np.sqrt(np.sum((edge_0_mean - background_color)**2))

    # color distance
    while dist_0 < color_thres and n_iter < 3:
        n_iter += 1
        bound_ext1 = cv2.dilate(alpha_mask, (kernel, kernel), iterations=n_iter)
        var_edge_0 = np.where((bound_ext1 == 255) & (bound_ext0 == 0))
        edge_0_colors = img[:,:,:3][var_edge_0[0], var_edge_0[1]]
        edge_0_mean = np.mean(edge_0_colors, axis=0)
        dist_0 = np.sqrt(np.sum((edge_0_mean - background_color)**2))
        bound_ext0 = bound_ext1 # renew
        #print(f"find the edge {dist_0} {n_iter}")
        bg_color = True
    if n_iter * kernel > min(int(wh[0]), int(wh[1])):
        bg_color = False
        n_iter = 0

    edge_0_black = np.sqrt(np.sum(edge_0_colors**2)) < 10
    if np.any(edge_0_black == True):
        n_iter += 2

    var_fg = np.where(img_bin == 1)
    colors = cropped_region[:,:,:3][var_fg[0], var_fg[1]]
    foreground_color = np.mean(colors, axis=0)

    colors = {'background': background_color,
              'foreground': foreground_color
    }

    return colors, bg_color, int(round(n_iter*kernel))
