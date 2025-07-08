import cv2
from imgaug import augmenters as iaa
import os

# sometimes = lambda aug: iaa.Sometimes(0.5, aug)   # 所有情况的 50% 中应用给定的增强器

# 创建所有增强器的列表
augmenters_list = [
    iaa.imgcorruptlike.MotionBlur(severity=(1, 2)),  # 运动模糊
    # iaa.Clouds(),  # 云雾
    iaa.imgcorruptlike.Fog(severity=1),  # 多雾/霜
    iaa.imgcorruptlike.Snow(severity=2),  # 下雨、大雪
    iaa.Rain(drop_size=(0.10, 0.15), speed=(0.1, 0.2)),  # 雨
    iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.03)), # 雪点
    # iaa.FastSnowyLandscape(lightness_threshold=(100, 255),lightness_multiplier=(1.5, 2.0)), # 雪地
    # iaa.imgcorruptlike.Spatter(severity=5),  # 溅 123水滴、45泥
    iaa.imgaug.augmenters.contrast.LinearContrast((0.5, 2.0), per_channel=0.5),  # 对比度变为原来的一半或者二倍
    iaa.imgcorruptlike.Brightness(severity=(1, 2)),  # 亮度增加
    iaa.imgcorruptlike.Saturate(severity=(1, 3)),  # 色彩饱和度
    iaa.Multiply((0.3, 0.5)), # 整体变暗-slight
    iaa.Multiply((0.2, 0.3)), # 整体变暗-smooth
    iaa.Multiply((0.1, 0.2)), # 整体变暗-severe
]

titles_list = [
    'motion blur',
    # 'cloud',
    'fog',
    'snow',
    'rain',
    'snowflakes',
    # 'snowland',
    # 'spatter',
    'contrast',
    'brightness',
    'saturate',
    'dark_slight',
    'dark_smooth',
    'dark_severe',
]

# 修改图片文件相关路径
# base_path = "/home4/xinhai/new_dataset/JACKAL/train"
# base_path = "/home4/TerraX-xinhai/new_dataset/RSCD_4/train"
base_path = "/home4/lihongze/our_data_new/train"
base_savedpath = '/home4/xuchuan/imgaug_results/our_data_new_imgaug_results'
os.makedirs(base_savedpath, exist_ok=True)

# 首先遍历所有增强器
for aug_index, augmenter in enumerate(augmenters_list):
    print(f'\n开始应用增强方式 {titles_list[aug_index]}...')
    current_aug_path = os.path.join(base_savedpath, titles_list[aug_index])
    os.makedirs(current_aug_path, exist_ok=True)
    
    # 对每个增强器创建一个序列
    seq = iaa.Sequential([augmenter])
    
    # 然后遍历所有子文件夹
    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir)
        
        # 跳过非目录文件
        if not os.path.isdir(subdir_path):
            continue
            
        # 为每个子文件夹创建对应的输出目录
        output_subdir = os.path.join(current_aug_path, subdir)
        os.makedirs(output_subdir, exist_ok=True)
        
        # 处理当前子文件夹中的图片
        imglist = []
        filelist = os.listdir(subdir_path)
        
        # 遍历当前文件夹，把所有的图片保存在imglist中
        for item in filelist:
            img_path = os.path.join(subdir_path, item)
            img = cv2.imread(img_path)
            if img is not None:  # 确保图片正确读取
                imglist.append(img)
        
        print(f'处理文件夹 {subdir} 中的图片...')
        
        # 对图片进行增强
        images_aug = seq.augment_images(imglist)
        
        # 保存增强后的图片
        for img_index in range(len(images_aug)):
            new_filename = filelist[img_index]
            if new_filename.endswith('.jpg'):
                output_path = os.path.join(output_subdir, new_filename)
                cv2.imwrite(output_path, images_aug[img_index])
                print(f'已保存: {output_path}')
        
        print(f'完成文件夹 {subdir} 的 {titles_list[aug_index]} 增强处理')
    
    print(f'完成所有文件夹的 {titles_list[aug_index]} 增强处理\n')