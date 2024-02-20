
# 获得文件夹下的所有文件名
import glob
pin = '文件夹路径'
pathname_expansion = pin + '*.obj'
filenames = glob.glob(pathname_expansion)
filenames.sort()    # 排序


# 加载 npz 查看
import numpy as np
path = r'datasets_processed\coseg_from_meshcnn\coseg_aliens\train_1_not_changed_1500.npz'
data = np.load(path)
data.close()    # 释放内存
