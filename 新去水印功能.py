import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import os
from tqdm import tqdm
import time
import logging
from datetime import datetime, timedelta

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            logging.info(f"创建路径: {directory}")
        except OSError as error:
            logging.info(f"创建路径失败 {directory}: {error}")
            raise


def detect_watermark_adaptive(frame, roi):
    # 提取ROI区域进行水印检测
    roi_frame = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 创建水印遮罩
    mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
    mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = binary_frame

    return mask
def get_position(position,video_clip):
    width, height = video_clip.size

    if position =='左上':
        x=5
        y=5
        width =320
        height =200
        return (x,y,width,height)
    elif position =='左下':
        x=5
        y=height-220
        width =320
        height =200
        return (x,y,width,height)
    elif position =='右上':
        x= width-330
        y=5
        width =320
        height =200
        return (x,y,width,height)
    elif position == '右下':
        x = width-330
        y= height-220
        width =320
        height =200
        return (x,y,width,height)
    else:
        return None

def generate_watermark_mask(video_clip,position, num_frames=10, min_frame_count=7):
    total_frames = int(video_clip.duration * video_clip.fps)
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    frames = [video_clip.get_frame(idx / video_clip.fps) for idx in frame_indices]

    r_original = get_position(position,video_clip)
    if r_original is None:
        return None

    masks = [detect_watermark_adaptive(frame, r_original) for frame in frames]

    final_mask = sum((mask == 255).astype(np.uint8) for mask in masks)
    # 根据像素点在至少min_frame_count张以上的帧中的出现来生成最终的遮罩
    final_mask = np.where(final_mask >= min_frame_count, 255, 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(final_mask, kernel)


def process_video(video,video_clip, output_path, apply_mask_func):
    try:
        total_frames = int(video_clip.duration * video_clip.fps)
        progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frames")

        def process_frame(frame):
            result = apply_mask_func(frame)
            progress_bar.update(1000)
            return result

        processed_video = video_clip.fl_image(process_frame, apply_to=["each"])
        processed_video.write_videofile(f"{output_path}.mp4", codec="libx264")
        output_path =f'{output_path}.mp4'
        return output_path
    except Exception as e:
        logging.info(f'去水印失败：{e}')
        return video
def video_main(video,output_dir,position):
    try:
        # 获取今天的关闭时间：23:59
        today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        start_time = time.time()
        if not os.path.exists(video):
            logging.error(f"视频 {video} 无法打开")
            download_flag = -1

        #确定视频输出路径
        ensure_directory_exists(output_dir)
        watermark_mask = None

        video_clip = VideoFileClip(video)
        if watermark_mask is None:
            watermark_mask = generate_watermark_mask(video_clip,position)
            if watermark_mask is None:
                raise Exception("未获取水印位置")
        #制作遮罩
        mask_func = lambda frame: cv2.inpaint(frame, watermark_mask, 3, cv2.INPAINT_NS)
        video_name = os.path.basename(video)
        output_video_path = os.path.join(output_dir, os.path.splitext(video_name)[0])

        out_path = process_video(video,video_clip, output_video_path, mask_func)


        logging.info(f"process_flage状态已更新为1")

        logging.info(f"视频路径 {video_name}")
        end_time = time.time()
        logging.info(f"处理时间： {end_time - start_time} 秒")

    except Exception as e:
        logging.info(f'处理视频失败{e}')

if __name__ == '__main__':
    video =''
    output_dir = ''
    position = ''
    video_main(video,output_dir, position)