import glob
import os

import trimesh

# 处理coseg_chair
dataset_name = 'coseg'
dataset = 'coseg'
subfolder = 'coseg_chairs'

p_out_sub = 'coseg'
p_in2add = 'coseg/coseg_chairs'
p_ext = 'coseg_chairs'

# 处理后的数据存放的文件夹
p_out = 'datasets_processed/coseg_from_meshcnn/coseg_chairs'
path_in = 'datasets_raw/from_meshcnn/coseg/coseg_chairs/'
pin = 'datasets_raw/from_meshcnn/coseg/coseg_chairs//test/'

# 创建输出文件夹
if not os.path.isdir(p_out):
    os.makedirs(p_out)


pathname_expansion = pin + '*.obj'
filenames = glob.glob(pathname_expansion)  # 要处理的模型路径列表，测试集4个，训练集16个
filenames.sort()    # 先排序


part = 'test'
file = filenames[0]
fn_prefix = part + '_'
out_fn = p_out + '/' + fn_prefix + os.path.split(file)[1].split('.')[0]

# load_mesh
model_fn = file
mesh_ = trimesh.load_mesh(model_fn, process=False)