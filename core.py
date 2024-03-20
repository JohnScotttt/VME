import argparse
import os
import shutil
import sys
import warnings

import cv2
import numpy as np
import paddleseg
from paddleseg.cvlibs import manager
from paddleseg.utils import get_sys_env, logger
from tqdm import tqdm

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..'))

manager.BACKBONES._components_dict.clear()
manager.TRANSFORMS._components_dict.clear()

import ppmatting
from ppmatting.core import predict
from ppmatting.utils import Config, MatBuilder, get_image_list

warnings.filterwarnings("ignore")


class VideoReader:
    def __init__(self, input_video):
        self.cap = cv2.VideoCapture(input_video)
        if not self.cap.isOpened():
            raise Exception(f"Cannot open {input_video}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_number = -1

    def read(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_number += 1
            return ret, frame
        else:
            return None, None

    def save_all(self, save_dir, save_name="frame", save_type="png"):
        os.makedirs(save_dir, exist_ok=True)
        while True:
            ret, frame = self.read()
            if ret:
                save_path = os.path.join(save_dir, save_name + "_" + str(self.frame_number) + "." + save_type)
                cv2.imwrite(save_path, frame)
            else:
                break
        self.release()
        print(f"All frames have been saved to {save_dir}")

    def release(self):
        self.cap.release()


class VideoWriter:
    def __init__(self, output_dir, output_name, fps):
        self.fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
        self.output_video = os.path.join(output_dir, output_name + ".mp4")
        self.fps = fps

    def add_frame(self, frame: np.ndarray):
        if not hasattr(self, "video"):
            height, width, layers = frame.shape
            self.video = cv2.VideoWriter(self.output_video, self.fourcc, self.fps, (width, height))
        self.video.write(frame)

    def add_all(self, frames_path: str):
        frames_list = os.listdir(frames_path)
        frames_list = sorted(frames_list, key=lambda x: int(x.split("_")[1].split(".")[0]))
        for frame in frames_list:
            frame_path = os.path.join(frames_path, frame)
            frame = cv2.imread(frame_path)
            self.add_frame(frame)
        self.release()

    def release(self):
        self.video.release()
        print(f"Video has been saved to {self.output_video}")


class Capture:
    def __init__(self, capture_type, capture_range, capture_numbers, fps):
        if capture_type == "time":
            begin_frame = int(capture_range[0] * fps)
            end_frame = int(capture_range[1] * fps)
            interval = (end_frame - begin_frame) // capture_numbers
            self.capture_list = list(range(begin_frame, end_frame, interval))
        elif capture_type == "frame":
            interval = (capture_range[1] - capture_range[0]) // capture_numbers
            self.capture_list = list(range(capture_range[0], capture_range[1], interval))

    def check(self, frame_number):
        if len(self.capture_list) > 0 and self.capture_list[0] == frame_number:
            self.capture_list.pop(0)
            return True
        else:
            return False


def watermark(image, watermark_numbers):
    def draw_rectangle(event, x, y, flags, param):
        nonlocal marked, flag, ix, iy, xyxy
        if event == cv2.EVENT_LBUTTONDOWN:
            if flag == 0:
                flag += 1
                ix, iy = x, y
                cv2.circle(scaled_image, (x, y), 2, (255, 255, 255), -1)
            else:
                flag = 0
                cv2.rectangle(scaled_image, (ix, iy), (x, y), (255, 255, 255), 2)
                xyxy.append([int(min(ix / scale, x / scale)),
                             int(min(iy / scale, y / scale)),
                             int(max(ix / scale, x / scale)),
                             int(max(iy / scale, y / scale))])
                marked += 1

    marked = 0
    flag = 0
    ix, iy = 0, 0
    xyxy = []

    target_width, target_height = 1600, 900
    original_width, original_height = image.shape[1], image.shape[0]
    scale = min(target_width / original_width, target_height / original_height)
    scaled_width, scaled_height = int(original_width * scale), int(original_height * scale)
    scaled_image = cv2.resize(image, (scaled_width, scaled_height))
    mask = np.zeros((original_height, original_width), dtype=np.uint8)

    if watermark_numbers > 0:
        cv2.namedWindow('image')
        cv2.moveWindow('image', 50, 50)
        cv2.setMouseCallback('image', draw_rectangle)
        while 1:
            cv2.imshow('image', scaled_image)
            if (cv2.waitKey(20) & 0xFF == 13) or marked == watermark_numbers:
                cv2.setMouseCallback('image', lambda *args: None)
                break
        cv2.imshow('image', scaled_image)
        cv2.waitKey(300)
        cv2.destroyAllWindows()

        for xy in xyxy:
            mask[xy[1]:xy[3], xy[0]:xy[2]] = 255
    return mask


def analyzer(img1, img2, img1_alpha, img2_alpha, mask):
    def rm_bad_kp(kp, mask, alpha):
        new_kp = []
        for i in kp:
            x, y = int(i.pt[0]), int(i.pt[1])
            if mask[y, x] != 255 and alpha[y, x, 3] == 0:
                new_kp.append(i)
        return new_kp

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1 = sift.detect(gray1, None)
    kp2 = sift.detect(gray2, None)
    kp1 = rm_bad_kp(kp1, mask, img1_alpha)
    kp2 = rm_bad_kp(kp2, mask, img2_alpha)
    kp1, des1 = sift.compute(gray1, kp1)
    kp2, des2 = sift.compute(gray2, kp2)
    matcher = cv2.FlannBasedMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
    return homography


def matting(image_path, save_dir, device="gpu"):
    cfg = Config("ppmattingv2-stdc1-human_512.yml")
    model_path = "ppmattingv2-stdc1-human_512.pdparams"
    builder = MatBuilder(cfg)

    paddleseg.utils.show_env_info()
    paddleseg.utils.show_cfg_info(cfg)
    paddleseg.utils.set_device(device)

    model = builder.model
    transforms = ppmatting.transforms.Compose(builder.val_transforms)

    image_list, image_dir = get_image_list(image_path)
    logger.info('Number of predict images = {}'.format(len(image_list)))

    predict(
        model,
        model_path=model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        trimap_list=None,
        save_dir=save_dir,
        fg_estimate=True)


def overlay(f_img, b_img):  # return bgra

    def img_float32(img):
        return img.copy() if img.dtype != 'uint8' else (img / 255.).astype('float32')

    if b_img[0][0].size == 3:
        b_img = cv2.cvtColor(b_img, cv2.COLOR_BGR2BGRA)
    if f_img[0][0].size == 3:
        f_img = cv2.cvtColor(f_img, cv2.COLOR_BGR2BGRA)
    f_img, b_img = img_float32(f_img), img_float32(b_img)
    (fb, fg, fr, fa), (bb, bg, br, ba) = cv2.split(f_img), cv2.split(b_img)
    color_f, color_b = cv2.merge((fb, fg, fr)), cv2.merge((bb, bg, br))
    alpha_f, alpha_b = np.expand_dims(fa, axis=-1), np.expand_dims(ba, axis=-1)

    color_f[fa == 0] = [0, 0, 0]
    color_b[ba == 0] = [0, 0, 0]

    a = fa + ba * (1 - fa)
    a[a == 0] = np.NaN
    color_over = (color_f * alpha_f + color_b * alpha_b * (1 - alpha_f)) / np.expand_dims(a, axis=-1)
    color_over = np.clip(color_over, 0, 1)

    result_float32 = np.append(color_over, np.expand_dims(a, axis=-1), axis=-1)
    return (result_float32 * 255).astype('uint8')


def run(input_video: str,
        device: str,
        output_dir: str,
        output_type: str,
        output_name: str,
        capture_type: str,
        capture_range: list,
        capture_numbers: int,
        watermark_numbers: int):
    temp_dir = os.path.join(output_dir, "temp")
    temp_original = os.path.join(temp_dir, "original")
    temp_matting = os.path.join(temp_dir, "matting")
    temp_composite = os.path.join(temp_dir, "composite")
    output_path = os.path.join(output_dir, output_name)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    os.makedirs(temp_original)
    os.makedirs(temp_matting)
    os.makedirs(temp_composite)

    vr = VideoReader(input_video)
    vr.save_all(temp_original)
    capture = Capture(capture_type, capture_range, capture_numbers, vr.fps)

    matting(temp_original, temp_matting, device)

    original_list = os.listdir(temp_original)
    original_list = sorted(original_list, key=lambda x: int(x.split("_")[1].split(".")[0]))
    img1 = None
    img2 = None
    img1_alpha = None
    img2_alpha = None
    mid_img = None
    for i, original in enumerate(tqdm(original_list)):
        img1 = img2
        img2 = cv2.imread(os.path.join(temp_original, original))
        img1_alpha = img2_alpha
        img2_alpha = cv2.imread(os.path.join(temp_matting, original.split(".")[0] + "_rgba.png"), -1)
        if i == 0:
            mask = watermark(img2, watermark_numbers)
            mid_img = np.zeros_like(img2_alpha)
            continue
        homography = analyzer(img1, img2, img1_alpha, img2_alpha, mask)
        mid_img = cv2.warpPerspective(mid_img, homography, (img2.shape[1], img2.shape[0]), mid_img,
                                      cv2.WARP_INVERSE_MAP)
        output = overlay(mid_img, img2)
        cv2.imwrite(os.path.join(temp_composite, f"frame_{i}.png"), output)
        if capture.check(i):
            mid_img = overlay(img2_alpha, mid_img)

    if output_type == "video":
        vw = VideoWriter(output_dir, output_name, vr.fps)
        vw.add_all(temp_composite)
    elif output_type == "image":
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
        shutil.copytree(temp_composite, output_path)
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_type", type=str, choices=["video", "image"], default="video")
    parser.add_argument("--output_name", type=str, default="demo")
    parser.add_argument("--capture_type", type=str, choices=["time", "frame"], required=True)
    parser.add_argument("--capture_range", type=int, nargs=2, required=True)
    parser.add_argument("--capture_numbers", type=int, default=5)
    parser.add_argument("--watermark_numbers", type=int, default=0)
    args = parser.parse_args()

    try:
        run(args.input_video,
            args.device,
            args.output_dir,
            args.output_type,
            args.output_name,
            args.capture_type,
            args.capture_range,
            args.capture_numbers,
            args.watermark_numbers)
    except Exception as e:
        print(e)
        shutil.rmtree(os.path.join(args.output_dir, 'temp'))
