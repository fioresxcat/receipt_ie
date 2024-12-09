import os
import json
import pdb
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect



def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped



class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, img):
        """Return updated labels and image with added border."""
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = self.new_shape
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border

        return img
    


class CornerDetector:
    def __init__(self, model_path, input_shape=(640, 640)):
        self.ort_sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        self.letterbox = LetterBox(input_shape, auto=False, stride=32)
        self.input_shape = input_shape
    

    def predict(self, images, batch=1):
        warped_images, list_points = [], []
        for raw_image in images:
            orig_img = raw_image.copy()
            h, w, _ = orig_img.shape
            resize_img = cv2.resize(raw_image, (self.input_shape[0], self.input_shape[0]), interpolation = cv2.INTER_LINEAR)
            image = np.array([np.transpose(resize_img, (2, 0, 1))]).astype(np.float32)

            outputs = self.ort_sess.run(None, {'input': image})[0]
            output = np.clip(outputs[0], a_min = 0, a_max = 1.0) 
            list_pts = [int(float(p) * w) if i % 2 == 0 else int(float(p) * h)for i, p in enumerate(output)]
            pts = [(list_pts[i], list_pts[i + 1]) for i in range(0, 4 * 2, 2)]
            pts = np.array(pts)
            # cut invoice from original image
            cut_image = four_point_transform(orig_img, pts)
            warped_images.append(cut_image)
            list_points.append(pts)
        return warped_images, list_points



def main():
    model = CornerDetector(model_path='utils/corner_detect.onnx', input_shape=(512, 512))
    dir = 'raw_data/adtima_data'

    save_paths, images, im_names = [], [], []
    for ip in Path(dir).rglob('*'):
        if not ip.suffix in ['.jpg', '.jpeg', '.png']:
            continue
        im = cv2.imread(str(ip))
        images.append(im)
        save_dir = str(ip.parent).replace('/raw_images', '/corner_jsons')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, ip.stem+'.json')
        save_paths.append(Path(save_path))
        im_names.append(ip.name)
        print(f'done append {ip}')
    
    print('Predicting ...')
    warped_images, list_points = model.predict(images)

    print('Saving ...')
    for pts, save_path, im_name in zip(list_points, save_paths, im_names):
        json_data = {
            'verion': '5.2.1',
            'flags': {},
            'shapes': [],
            'imagePath': f'../raw_images/{im_name}',
            'imageData': None,
            'imageHeight': 512,
            'imageWidth': 512
        }
        shape = {
            'points': pts.tolist(),
            'shape_type': 'polygon',
            'flags': {},
            'label': 'invoice'
        }
        json_data['shapes'].append(shape)
        with open(save_path, 'w') as f:
            json.dump(json_data, f)
        print(f'done {save_path}')



def warp_image_dir():
    im_dir = 'raw_data/adtima_data/aeon_glam/raw_images'
    json_dir = 'raw_data/adtima_data/aeon_glam/corner_jsons'
    out_dir = 'raw_data/adtima_data/aeon_glam/warped_images'
    os.makedirs(out_dir, exist_ok=True)

    for ip in Path(im_dir).rglob('*'):
        jp = os.path.join(json_dir, ip.stem+'.json')
        im = cv2.imread(str(ip))
        with open(jp) as f:
            data = json.load(f)
        for index, shape in enumerate(data['shapes']):
            pts = shape['points']
            pts = np.array(pts)
            cut_image = four_point_transform(im, pts)
            new_name = f'{ip.stem}_{index}.jpg'
            cv2.imwrite(os.path.join(out_dir, new_name), cut_image)
        print(f'done {ip}')


def nothing():
    for jp in Path('raw_data').rglob('*.json'):
        os.remove(jp)


if __name__ == '__main__':
    pass
    # nothing()
    # main()
    warp_image_dir()