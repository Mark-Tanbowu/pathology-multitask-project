# 数据准备
## 数据处理顺序
检测重复样本-fix-check_size-mask_checker-transforms-camelyon_datasets
（是否存在check and mask 产生功能重复的问题？）
## 脚本作用

### 11.5涉及到的一些改进点以及问题
1. 按照max进行归一化有一定的小风险 如果再运行之前统一为255之后此处直接按照255或可节省算力 而其他情况可以另外立一个分支
2. 预处理较为朴素 或可做stain标准化和强度标准化