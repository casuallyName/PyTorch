import json
"""
Demo


先测试一下


测试

"""


class Data_detection():
      


    garbage_classify_index = {"0": "其他垃圾", "1": "厨余垃圾", "2": "可回收物", "3": "有害垃圾"}
    garbage_index_classify = {"其他垃圾": "0", "厨余垃圾": "1", "可回收物": "2", "有害垃圾": "3"}

    # 写入json
    def store(self, path, data):
        with open(path, 'w') as fw:
            json.dump(data, fw)


    # 加载json
    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data


# print(Data_detection.load(
D = Data_detection()
path = '../data/garbage_classify_rule.json'# ))
data = D.load(path)
for key,value in data.items():
    print(key,D.garbage_index_classify[value.split('/')[0]])

print()
#
#
