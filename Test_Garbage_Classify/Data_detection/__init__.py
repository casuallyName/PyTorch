# python
# -*- coding:utf-8 -*-

"""
垃圾分类数据分布分析
"""

# 整体数据探测
import os
from os import walk
from glob import glob

base_path = '../data/'
data_path = os.path.join(base_path, 'train_data')

# walk 迭代器
# for (dirpath, dirnames, filenames) in walk(data_path):
#     if len(filenames) > 0:
#         print('*' * 60)
#         print("Directory path: ", dirpath)
#         print("total examples: ", len(filenames))
#         print("File name Example: ", filenames[:5])


garbage_classify_rule = {
    "0": "其他垃圾/一次性快餐盒",
    "1": "其他垃圾/污损塑料",
    "2": "其他垃圾/烟蒂",
    "3": "其他垃圾/牙签",
    "4": "其他垃圾/破碎花盆及碟碗",
    "5": "其他垃圾/竹筷",
    "6": "厨余垃圾/剩饭剩菜",
    "7": "厨余垃圾/大骨头",
    "8": "厨余垃圾/水果果皮",
    "9": "厨余垃圾/水果果肉",
    "10": "厨余垃圾/茶叶渣",
    "11": "厨余垃圾/菜叶菜根",
    "12": "厨余垃圾/蛋壳",
    "13": "厨余垃圾/鱼骨",
    "14": "可回收物/充电宝",
    "15": "可回收物/包",
    "16": "可回收物/化妆品瓶",
    "17": "可回收物/塑料玩具",
    "18": "可回收物/塑料碗盆",
    "19": "可回收物/塑料衣架",
    "20": "可回收物/快递纸袋",
    "21": "可回收物/插头电线",
    "22": "可回收物/旧衣服",
    "23": "可回收物/易拉罐",
    "24": "可回收物/枕头",
    "25": "可回收物/毛绒玩具",
    "26": "可回收物/洗发水瓶",
    "27": "可回收物/玻璃杯",
    "28": "可回收物/皮鞋",
    "29": "可回收物/砧板",
    "30": "可回收物/纸板箱",
    "31": "可回收物/调料瓶",
    "32": "可回收物/酒瓶",
    "33": "可回收物/金属食品罐",
    "34": "可回收物/锅",
    "35": "可回收物/食用油桶",
    "36": "可回收物/饮料瓶",
    "37": "有害垃圾/干电池",
    "38": "有害垃圾/软膏",
    "39": "有害垃圾/过期药物"
}

garbage_classify_index = {"0": "其他垃圾", "1": "厨余垃圾", "2": "可回收物", "3": "有害垃圾"}
garbage_index_classify = {"其他垃圾": "0", "厨余垃圾": "1", "可回收物": "2", "有害垃圾": "3"}

data_list = []
rank1_garbage_classify_rule = {}
for k, v in garbage_classify_rule.items():
    rank1_k = v.split('/')[0]
    rank1_v = k
    data_list.append([rank1_k, int(garbage_index_classify[rank1_k]), int(rank1_v)])

# 获取一级分类label 对应的原始数据label
rank_k_v_dict = {}
for data in data_list:
    k = data[2]  # 原标签
    v = data[1]  # 新标签
    rank_k_v_dict[k] = v
# print(rank_k_v_dict)


def get_img_info():
    data_path_txt = os.path.join(data_path, '*.txt')
    txt_file_list = glob(data_path_txt)

    # 存储txt 文件
    img_path_txt = 'img.txt'
    img_path_list = []
    img_label_dict = dict()  # <标签，次数>
    img_name2label_dict = {}
    for file_path in txt_file_list:
        with open(file_path, 'r') as f:
            line = f.readline()

        line = line.strip()
        img_name = line.split(',')[0]
        img_label = line.split(',')[1]
        img_label = int(img_label.strip())
        # 图片路径＋标签
        img_name_path = os.path.join(base_path, 'train_data/{}'.format(img_name))
        img_path_list.append(
            {'img_name_path': img_name_path,
             'img_label': img_label})
    return img_path_list


# print('img_path_list = ', get_img_info()[:10])


# 对img_path_list 的img_label 进行修改为一级分类的标签
img_path_list = []
img_label_dict = {}
for img_info in get_img_info():
    img_label = img_info['img_label']  # 修正前的标签
    img_label = rank_k_v_dict[img_label]
    img_info.update({'img_label': img_label})  # 修正后的标签

    # 图片路径＋标签
    img_path_list.append(img_info)

    # 统计每个标签出现次数
    img_label = int(img_label)
    img_label_count = img_label_dict.get(img_label, 0)
    if img_label_count:
        img_label_dict[img_label] = img_label_count + 1
    else:
        img_label_dict[img_label] = 1

# print('img_path_list = ', img_path_list[:3])
# print('img_label_dict = ', img_label_dict)

img_label_dict = dict(sorted(img_label_dict.items()))
print(img_label_dict)
print(garbage_classify_index)
print([garbage_classify_index[str(k)] for k in img_label_dict.keys()])
print(list(img_label_dict.values()))