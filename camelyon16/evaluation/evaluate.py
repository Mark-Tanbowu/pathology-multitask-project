"""
用于 CAMELYON16 挑战赛中癌转移检测的评估代码。
"""

import multiresolutionimageinterface as mir
import pandas as ps
import numpy as np
import scipy.ndimage
import skimage.measure
import matplotlib.pyplot as plt
import bisect
import argparse
import os

#----------------------------------------------------------------------------------------------------

def load_detections(detection_path, mask_path, level):
    """
    从检测 CSV 中读取 (概率, X 坐标, Y 坐标) 的元组列表。

    读取后会把像素坐标调整到与 mask 评估层级对应的 WSI 层级。

    Args:
        detection_path (str): 检测 CSV 路径。
        mask_path (str): mask 图像路径。
        level (int): mask 的评估层级。

    Returns:
        list: (概率, 行, 列) 的检测元组列表。
    """

    # 从检测 CSV 中读取 (概率, y 坐标, x 坐标) 元组列表。注意 numpy 是 (行, 列) 顺序。
    #
    detection_table = ps.read_csv(detection_path)
    detection_items = [(detection_row['p'], int(detection_row['y']), int(detection_row['x'])) for _, detection_row in detection_table.iterrows()]

    # 读取 mask 在评估层级的下采样倍率。
    #
    mask_image = mir.MultiResolutionImageReader().open(mask_path)
    level_downsampling = mask_image.getLevelDownsample(level=level)
    mask_image.close()

    # 将 WSI level 0 的预测坐标缩放到与 mask 评估层级一致的坐标系。
    #
    detection_items = [(detection[0], round(detection[1] / level_downsampling), round(detection[2] / level_downsampling)) for detection in detection_items]

    return detection_items

#----------------------------------------------------------------------------------------------------

def compute_evaluation_mask(mask_path, level, include_itcs):
    """
    计算评估用 mask，并得到包含 ITC（Isolated Tumor Cells）的区域标签集合。

    评估 mask 为标签图，计算步骤：
        1. 读取指定层级的 mask。
        2. 选出肿瘤区域。
        3. 按“若干个肿瘤细胞”距离进行膨胀。
        4. 标注连通区域。

    Args:
        mask_path (str): 要读取的 mask TIF 文件。
        level (int): 处理层级。
        include_itcs (bool): 是否在评估中包含 ITC。

    Returns:
        (np.ndarray, set): 评估 mask 与 ITC 区域标签集合。
    """

    # 固定常量。
    #
    tumor_label_value = 2     # TIF mask 中肿瘤标签值为 2。
    dilation_distance = 75.0  # 75 um 约等于 5 个肿瘤细胞

    # 在指定层级读取 mask 到内存。
    #
    mask_image = mir.MultiResolutionImageReader().open(mask_path)
    level_width, level_height = mask_image.getLevelDimensions(level=level)
    image_array = mask_image.getUCharPatch(startX=0, startY=0, width=level_width, height=level_height, level=level)
    image_array = image_array.squeeze()

    # 计算到肿瘤边界的距离。
    #
    image_negative_array = np.not_equal(image_array, tumor_label_value)
    image_distance_array = scipy.ndimage.distance_transform_edt(input=image_negative_array)

    # 在肿瘤区域周围扩展若干细胞的距离。
    #
    image_spacing = mask_image.getSpacing()[0]
    image_downsampling = mask_image.getLevelDownsample(level=level)
    image_level_spacing = image_spacing * image_downsampling
    distance_threshold_pixels = dilation_distance / (image_level_spacing * 2.0)
    image_binary_array = np.less(image_distance_array, distance_threshold_pixels)

    mask_image.close()

    # 对膨胀后的肿瘤 mask 填洞并标注连通区域。
    #
    image_filled_array = scipy.ndimage.morphology.binary_fill_holes(input=image_binary_array)
    image_evaluation_mask = skimage.measure.label(input=image_filled_array, connectivity=2)

    # 收集 ITC 区域标签。若最长直径 < 200 um，则视为 ITC。
    #
    if include_itcs:
        itc_labels = set()
    else:
        itc_size_threshold = (200.0 + dilation_distance) / image_level_spacing
        region_properties = skimage.measure.regionprops(label_image=image_evaluation_mask)
        itc_labels = set(label_index + 1 for label_index in range(len(region_properties)) if region_properties[label_index].major_axis_length < itc_size_threshold)

    return image_evaluation_mask, itc_labels

#----------------------------------------------------------------------------------------------------

def compute_probabilities(detection_items, evaluation_mask, itc_labels):
    """
    生成该图像的 TP/FP 统计信息。

    Args:
        detection_items (list): (概率, 行, 列) 的检测列表。
        evaluation_mask (np.ndarray, None): 评估 mask。
        itc_labels (set): ITC 标签集合。

    Returns:
        (list, list, int): FP 概率列表、TP 概率列表，以及非 ITC 肿瘤数。
    """

    # 判断是否有肿瘤：是否存在 mask。
    #
    if evaluation_mask is not None:
        # 初始化结果。
        #
        max_label = evaluation_mask.max()
        fp_probs = []
        tp_probs = [0.0] * (max_label + 1)

        # 判断检测是否命中肿瘤区域或正常组织，并丢弃 ITC 的结果。
        #
        for detection in detection_items:
            hit_label = evaluation_mask[detection[1:]]

            if hit_label == 0:
                fp_probs.append(detection[0])

            elif hit_label not in itc_labels:
                if tp_probs[hit_label] < detection[0]:
                    tp_probs[hit_label] = detection[0]

        # 统计肿瘤区域数量。
        #
        number_of_tumors = max_label - len(itc_labels)

    else:
        # 初始化结果。
        #
        fp_probs = []
        tp_probs = [0.0]

        # 该切片不含肿瘤，所有检测都记为 FP。
        #
        for detection in detection_items:
            fp_probs.append(detection[0])

        # 肿瘤数量为 0。
        #
        number_of_tumors = 0

    # 去掉第一个未使用的占位值。
    #
    tp_probs = tp_probs[1:]

    return fp_probs, tp_probs, number_of_tumors

#----------------------------------------------------------------------------------------------------

def compute_froc(froc_data):
    """
    生成绘制 FROC 曲线所需的数据。

    Args:
        froc_data (dict): 包含每张图像的 TP、FP 和肿瘤数量。

    Returns:
        (list, list): 各阈值下的平均 FP/图像，以及总体敏感度列表。
    """

    # 汇总所有图像的结果。
    #
    aggregated_fps = [prob for froc_item in froc_data.values() for prob in froc_item['fp']]
    aggregated_tps = [prob for froc_item in froc_data.values() for prob in froc_item['tp']]
    all_probs = sorted(set(aggregated_fps + aggregated_tps) - {0.0})
    image_count = len(froc_data)
    total_tumor_count = sum(froc_item['count'] for froc_item in froc_data.values())

    # 随阈值递增统计数量。
    #
    aggregated_fps = np.asarray(a=aggregated_fps, dtype=np.float64)
    aggregated_tps = np.asarray(a=aggregated_tps, dtype=np.float64)

    total_fps = []
    total_tps = []
    for threshold in all_probs:
        total_fps.append(np.greater_equal(aggregated_fps, threshold).sum())
        total_tps.append(np.greater_equal(aggregated_tps, threshold).sum())

    total_fps.append(0)
    total_tps.append(0)

    # 计算最终指标。
    #
    total_fps = [count / image_count for count in total_fps]
    total_sensitivity = [count / total_tumor_count for count in total_tps]

    return total_fps, total_sensitivity

#----------------------------------------------------------------------------------------------------

def compute_score(average_fps, sensitivities):
    """
    计算挑战赛的第二个指标：在 6 个固定 FP/WSI 下的平均敏感度
    （1/4、1/2、1、2、4、8 FP/WSI）。

    Args:
        average_fps (list): 不同阈值下的平均 FP/图像列表。
        sensitivities (list): 不同阈值下的总体敏感度列表。

    Returns:
        float: 计算得到的分数。
    """

    average_fps_r = list(reversed(average_fps))
    sensitivities_r = list(reversed(sensitivities))

    threshold_count = len(sensitivities_r)
    target_fp_items = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    target_sum = sum(sensitivities_r[min(bisect.bisect_left(average_fps_r, target_fp), threshold_count - 1)] for target_fp in target_fp_items)

    return target_sum / len(target_fp_items)

#----------------------------------------------------------------------------------------------------

def save_results(result_file_path, average_fps, sensitivities):
    """
    保存结果。

    Args:
        result_file_path (str): 结果文件路径。
        average_fps (list): 不同阈值下的平均 FP/图像列表。
        sensitivities (list): 不同阈值下的总体敏感度列表。
    """

    result_df = ps.DataFrame.from_dict(data={'Average FP Counts': average_fps, 'Overall Sensitivities': sensitivities}, dtype=np.float64)
    result_df.to_csv(path_or_buf=result_file_path, columns=['Average FP Counts', 'Overall Sensitivities'], index=False)

#----------------------------------------------------------------------------------------------------

def plot_froc(average_fps, sensitivities):
    """
    绘制 FROC 曲线。

    Args:
        average_fps (list): 不同阈值下的平均 FP/图像列表。
        sensitivities (list): 不同阈值下的总体敏感度列表。
    """

    plt.xlabel('Average Number of False Positives')
    plt.ylabel('Metastasis Detection Sensitivity')
    plt.title('Free Response Receiver Operating Characteristic Curve')
    plt.plot(average_fps, sensitivities, linestyle='-', color='black')

    plt.show()

#----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    解析命令行参数。

    Returns:
        (str, str, str, str, bool, bool): 解析后的参数：reference CSV 路径、mask 目录路径、
            detections 目录路径、结果 CSV 路径、是否包含 ITC、是否绘制曲线。
    """

    # 配置参数解析器。
    #
    argument_parser = argparse.ArgumentParser(description='Compute FROC on the CAMELYON16 test set.')

    argument_parser.add_argument('-r', '--reference',  required=True,  type=str, default='./reference.csv', help='reference CSV file path')
    argument_parser.add_argument('-m', '--masks',      required=True,  type=str, default='../masks',        help='reference mask folder path')
    argument_parser.add_argument('-d', '--detections', required=True,  type=str,                            help='detection file folder path')
    argument_parser.add_argument('-o', '--result',     required=False, type=str, default=None,              help='result table file path')
    argument_parser.add_argument('-t', '--itc',        action='store_true',                                 help='include ITCs in calculation')
    argument_parser.add_argument('-p', '--plot',       action='store_true',                                 help='plot curve')

    # 解析参数。
    #
    arguments = vars(argument_parser.parse_args())
    parsed_reference_path = arguments['reference']
    parsed_masks_path = arguments['masks']
    parsed_detections_path = arguments['detections']
    parsed_result_path = arguments['result']
    parsed_itc_flag = arguments['itc']
    parsed_plot_flag = arguments['plot']

    # 打印参数。
    #
    print(argument_parser.description)
    print('Reference path: {path}'.format(path=parsed_reference_path))
    print('Masks path: {path}'.format(path=parsed_masks_path))
    print('Detections path: {path}'.format(path=parsed_detections_path))
    print('Result path: {path}'.format(path=parsed_result_path))
    print('Include ITCs: {flag}'.format(flag=parsed_itc_flag))
    print('Plot curve: {flag}'.format(flag=parsed_plot_flag))
    print('')

    return parsed_reference_path, parsed_masks_path, parsed_detections_path, parsed_result_path, parsed_itc_flag, parsed_plot_flag

#----------------------------------------------------------------------------------------------------

def main():
    """入口函数：计算并可选绘制 FROC 曲线。"""

    # 固定常量。
    #
    evaluation_mask_level = 5

    # 读取命令行参数。
    #
    reference_file_path, masks_folder_path, detections_folder_path, result_file_path, include_itcs, plot_curve = collect_arguments()

    # CAMELYON16 官方评测默认不包含 ITC。
    #
    if include_itcs:
        print('Warning: FROC 计算中包含 ITC，但 CAMELYON16 官方评测不包含 ITC。')
        print('')

    # 处理每张测试图像。
    #
    froc_data = dict()

    reference_table = ps.read_csv(reference_file_path)
    for _, reference_row in reference_table.iterrows():
        image_name, _ = os.path.splitext(reference_row['image'])
        if image_name.startswith('test'):
            mask_path = os.path.join(masks_folder_path, '{image}_mask.tif'.format(image=image_name))
            detection_path = os.path.join(detections_folder_path, '{image}.csv'.format(image=image_name))
            print('Processing: {path}'.format(path=detection_path))

            # 加载检测结果与评估 mask。
            #
            detection_items = load_detections(detection_path=detection_path, mask_path=mask_path, level=evaluation_mask_level)
            evaluation_mask, itc_labels = compute_evaluation_mask(mask_path=mask_path, level=evaluation_mask_level, include_itcs=include_itcs) if reference_row['type'] == 'tumor' else (None, set())

            # 计算 FP/TP 概率。
            #
            fp_probs, tp_probs, number_of_tumors = compute_probabilities(detection_items=detection_items, evaluation_mask=evaluation_mask, itc_labels=itc_labels)
            froc_data[reference_row['image']] = {'fp': fp_probs, 'tp': tp_probs, 'count': number_of_tumors}

    # 计算 FROC。
    #
    average_fps, sensitivities = compute_froc(froc_data=froc_data)

    # 计算得分。
    #
    challenge_score = compute_score(average_fps=average_fps, sensitivities=sensitivities)

    print('')
    print('Score: {score}'.format(score=challenge_score))

    # 保存结果。
    #
    if result_file_path:
        save_results(result_file_path=result_file_path, average_fps=average_fps, sensitivities=sensitivities)

    # 绘制 FROC 曲线。
    #
    if plot_curve:
        plot_froc(average_fps=average_fps, sensitivities=sensitivities)

#----------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    main()
