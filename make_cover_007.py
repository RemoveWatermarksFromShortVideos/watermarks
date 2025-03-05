import cv2
import os
import numpy as np
import subprocess
from collections import Counter
import time
from io import BytesIO
'''逐帧处理
'''
FFMPEG_PATH = r'D:\ffmpeg-7.0.1-full_build\bin\ffmpeg.exe'  # FFmpeg的路径


def extract_frames_from_video(video_path, max_frames=10):
    """从视频中提取多帧保存到内存中"""
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = total_frames // (max_frames + 1)  # 间隔地选择帧

    frames = []
    for i in range(1, max_frames + 1):
        frame_pos = frame_interval * i
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = video_capture.read()
        if not ret:
            break  # 如果没有更多帧，退出循环
        frames.append(frame)

    video_capture.release()
    return frames

def load_watermark_images(watermark_dir):
    """加载所有水印图片并返回文件名和图像"""
    watermark_images = []
    for file_name in os.listdir(watermark_dir):
        watermark_path = os.path.join(watermark_dir, file_name)
        watermark_img = cv2.imread(watermark_path)

        if watermark_img is not None:  # 确保图像成功加载
            watermark_img_gray = cv2.cvtColor(watermark_img, cv2.COLOR_BGR2GRAY)
            watermark_images.append((file_name, watermark_img_gray))  # 返回文件名和灰度图像
        else:
            print(f"警告: 无法加载图像 {file_name}")

    return watermark_images
def detect_watermark_in_frame(watermark_images, frame_img, sift, bf):
    """检测水印是否存在于视频帧中"""
    for file_name,watermark_img in watermark_images:
        keypoints1, descriptors1 = sift.detectAndCompute(watermark_img, None)
        keypoints2, descriptors2 = sift.detectAndCompute(frame_img, None)

        if descriptors1 is not None and descriptors2 is not None:
            # 使用 KNN 匹配描述符，并设定 k=2 进行两邻近匹配
            knn_matches = bf.knnMatch(descriptors1, descriptors2, k=2)
            # 通过 Lowe's ratio test 筛选出较好的匹配
            good_matches = []
            for m, n in knn_matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if len(good_matches) > 10:
                src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None:
                    h, w = watermark_img.shape[:2]
                    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    dst_corners = cv2.perspectiveTransform(corners, M)

                    x_min, y_min = np.int32(dst_corners.min(axis=0).ravel())
                    x_max, y_max = np.int32(dst_corners.max(axis=0).ravel())
                    return (file_name,x_min, y_min, x_max - x_min, y_max - y_min)  # 返回水印坐标和大小

    return None

def apply_watermark_to_video(input_video_path, watermark_images, output_video_path, x, y):
    """使用FFmpeg在视频的每一帧上应用水印"""
    # 打开视频文件
    video_capture = cv2.VideoCapture(input_video_path)
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用合适的编码器
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_index = 0  # 用于跟踪当前帧
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # 如果没有读取到帧，则退出循环

        # 读取水印图像，确保在访问 watermark_images 时使用当前帧的索引
        if frame_index < len(watermark_images):
            watermark_image_stream = watermark_images[frame_index]
            watermark_image = cv2.imdecode(np.frombuffer(watermark_image_stream.getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)
            if watermark_image is None:
                print("警告: 水印图像解码失败，跳过该帧。")
                continue

            # 判断 x_offset 是否超过视频宽度
            if x-250< 0 and y>10:
                x_offset = 0
                y_offset = y-10
            else:
                x_offset = x-250
                y_offset = y-10


            # 防止水印超出边界
            # 如果水印图像宽度超出边界，缩小水印宽度
            if x_offset + watermark_image.shape[1] > width:
                new_width = width - x_offset
                watermark_image = cv2.resize(watermark_image, (new_width, watermark_image.shape[0]))

            # 如果水印图像高度超出边界，缩小水印高度
            if y_offset + watermark_image.shape[0] > height:
                new_height = height - y_offset
                watermark_image = cv2.resize(watermark_image, (watermark_image.shape[1], new_height))

            #创建帧的副本用于叠加水印
            overlay_frame = frame.copy()
            # 叠加水印
            overlay_frame[y_offset:y_offset + watermark_image.shape[0], x_offset:x_offset + watermark_image.shape[1]] = watermark_image

            # 写入处理后的帧
            video_writer.write(overlay_frame)
        else:
            print("警告: 水印图像不足，可能没有提取到所有帧的水印。")

        frame_index += 1  # 更新帧索引

    # 释放资源
    video_capture.release()
    video_writer.release()
    print(f"水印已应用，视频保存到 {output_video_path}")
def get_most_common_position(positions):
    """统计最常见的水印坐标"""
    if not positions:
        return None
    x_coords = [pos[0] for pos in positions]

    # 初始化最常见的宽度和高度
    most_common_width = None
    most_common_height = None
    # 找出 x 和 y 位置的最常值
    most_common_x = Counter(x_coords).most_common(1)[0][0]

    # 获取与最常见的 x 和 y 值对应的宽度和高度
    for pos in positions:
        if pos[0] == most_common_x:
            most_common_y = pos[1]
            most_common_width = pos[2]
            most_common_height = pos[3]
            break

    return most_common_x, most_common_y, most_common_width, most_common_height


def watermark_video_picture(input_path,x,y,mark_height):
    """从视频中提取每一帧的水印区域图像"""
    video_capture = cv2.VideoCapture(input_path)
    watermark_images = []  # 存储每帧提取的水印区域
    # 设置水印区域大小和位置调整
    if mark_height>70:
        width, height = 414,mark_height
    else:
        width, height = 414,70


    # 定义水印区域的目标宽度和高度
    #判断水印位置
    ret, frame = video_capture.read()
    if x <= 250 and y<=frame.shape[0]-200:#左上角
        y += 100
    elif x<=250 and y>=frame.shape[0]-200:#左下角
        x+=100
        y-=mark_height
    elif x >= 250 and y<=frame.shape[0]-200:#右上角
        x = x - 250
        y += 80
    else:#右下角
        x = x - 250
        y -=100

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # 如果没有读取到帧，则退出循环
        # 确保水印区域不超出帧的边界
        # if y + height > frame.shape[0] or x + width > frame.shape[1]:
        #     print(f"警告: 水印区域超出帧边界，位置: ({x}, {y})，帧大小: {frame.shape[1]}x{frame.shape[0]}")
        #     continue  # 跳过这帧，继续下一个帧

        # 截取水印区域并检查其大小
        cropped_image = frame[y:y + height, x:x + width]
        if cropped_image.size == 0:  # 检查截取的区域是否为空
            print("警告: 截取的水印区域为空，跳过该帧。")
            continue

        # 将提取的水印图像存储为内存中的JPEG格式
        _, buffer = cv2.imencode('.jpg', cropped_image)
        image_stream = BytesIO(buffer)
        watermark_images.append(image_stream)

    video_capture.release()
    return watermark_images
def video_main(input_video_path, watermark_dir, output_video_path):
    try:
        # 提取视频帧
        start_time = time.time()
        frames = extract_frames_from_video(input_video_path)

        # 加载水印图片
        watermark_images = load_watermark_images(watermark_dir)

        # 设置SIFT检测器和暴力匹配器
        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # 遍历视频帧进行水印检测
        watermark_positions = []
        for frame_index, frame_img in enumerate(frames):
            frame_gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
            watermark_position = detect_watermark_in_frame(watermark_images, frame_gray, sift, bf)

            if watermark_position:
                file_name,x, y, width, height = watermark_position
                print(f"水印{file_name}在帧 {frame_index} 中找到，坐标: ({x}, {y}), 尺寸: {width}x{height}")
                watermark_positions.append((x, y, width, height))

        # 确定最常见的水印位置
        common_position = get_most_common_position(watermark_positions)

        if common_position:
            x, y, mark_width, mark_height= common_position
            print(f"应用最常见的水印位置: ({x}, {y}),大小：({mark_width},{mark_height})")
            # 将BytesIO流中的数据转换为numpy数组
            watermark_images = watermark_video_picture(input_video_path,x, y,mark_height)
            # 应用水印到视频
            apply_watermark_to_video(input_video_path, watermark_images, output_video_path, x, y)
        else:
            print("未找到水印位置。")
        end_time = time.time()
        print(f"处理完成，耗时: {end_time - start_time} 秒")
    except Exception as e:
        print(f"处理视频时发生错误: {e}")


input_path = r'D:\video_cut_splict\watermark_blurry\video_material/test.mp4'  # 视频目录
watermark_dir = r'D:\video_cut_splict\watermark_blurry/Water_mark'  # 水印图片目录
output_video_path = r'D:\video_cut_splict\watermark_blurry\output_video\test_B.mp4'  # 遮挡水印后的视频

video_main(input_path, watermark_dir,output_video_path)
