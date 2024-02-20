import glob
import os
import sys

import numpy as np
import open3d
import trimesh
from easydict import EasyDict
from tqdm import tqdm

import utils

# Labels for all datasets
# -----------------------
sigg17_part_labels = ['---', 'head', 'hand', 'lower-arm', 'upper-arm', 'body', 'upper-lag', 'lower-leg', 'foot']
sigg17_shape2label = {v: k for k, v in enumerate(sigg17_part_labels)}

model_net_labels = [
    'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet',
    'wardrobe', 'bookshelf', 'laptop', 'door', 'lamp', 'person', 'curtain', 'piano', 'airplane', 'cup',
    'cone', 'tent', 'radio', 'stool', 'range_hood', 'car', 'sink', 'guitar', 'tv_stand', 'stairs',
    'mantel', 'bench', 'plant', 'bottle', 'bowl', 'flower_pot', 'keyboard', 'vase', 'xbox', 'glass_box'
]
model_net_shape2label = {v: k for k, v in enumerate(model_net_labels)}

cubes_labels = [
    'apple', 'bat', 'bell', 'brick', 'camel',
    'car', 'carriage', 'chopper', 'elephant', 'fork',
    'guitar', 'hammer', 'heart', 'horseshoe', 'key',
    'lmfish', 'octopus', 'shoe', 'spoon', 'tree',
    'turtle', 'watch'
]
cubes_shape2label = {v: k for k, v in enumerate(cubes_labels)}

shrec11_labels = [
    'armadillo', 'man', 'centaur', 'dinosaur', 'dog2',
    'ants', 'rabbit', 'dog1', 'snake', 'bird2',
    'shark', 'dino_ske', 'laptop', 'santa', 'flamingo',
    'horse', 'hand', 'lamp', 'two_balls', 'gorilla',
    'alien', 'octopus', 'cat', 'woman', 'spiders',
    'camel', 'pliers', 'myScissor', 'glasses', 'bird1'
]
shrec11_shape2label = {v: k for k, v in enumerate(shrec11_labels)}

coseg_labels = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c',
]
coseg_shape2label = {v: k for k, v in enumerate(coseg_labels)}


def calc_mesh_area(mesh):
    """计算mesh的面积"""
    t_mesh = trimesh.Trimesh(vertices=mesh['vertices'], faces=mesh['faces'], process=False)
    mesh['area_faces'] = t_mesh.area_faces
    mesh['area_vertices'] = np.zeros((mesh['vertices'].shape[0]))
    for f_index, f in enumerate(mesh['faces']):
        for v in f:
            mesh['area_vertices'][v] += mesh['area_faces'][f_index] / f.size


def prepare_edges_and_kdtree(mesh):
    """obj文件里只有顶点和面，通过顶点和面得到边"""
    vertices = mesh['vertices']
    faces = mesh['faces']
    mesh['edges'] = [set() for _ in range(vertices.shape[0])]
    for i in range(faces.shape[0]):
        for v in faces[i]:  # 一个面三个点
            mesh['edges'][v] |= set(faces[i])
    for i in range(vertices.shape[0]):
        if i in mesh['edges'][i]:
            mesh['edges'][i].remove(i)
        mesh['edges'][i] = list(mesh['edges'][i])
    max_vertex_degree = np.max([len(e) for e in mesh['edges']])
    for i in range(vertices.shape[0]):
        if len(mesh['edges'][i]) < max_vertex_degree:
            mesh['edges'][i] += [-1] * (max_vertex_degree - len(mesh['edges'][i]))
    mesh['edges'] = np.array(mesh['edges'], dtype=np.int32)

    mesh['kdtree_query'] = []
    t_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    n_nbrs = min(10, vertices.shape[0] - 2)
    for n in range(vertices.shape[0]):
        d, i_nbrs = t_mesh.kdtree.query(vertices[n], n_nbrs)
        i_nbrs_cleared = [inbr for inbr in i_nbrs if inbr != n and inbr < vertices.shape[0]]
        if len(i_nbrs_cleared) > n_nbrs - 1:
            i_nbrs_cleared = i_nbrs_cleared[:n_nbrs - 1]
        mesh['kdtree_query'].append(np.array(i_nbrs_cleared, dtype=np.int32))
    mesh['kdtree_query'] = np.array(mesh['kdtree_query'])
    assert mesh['kdtree_query'].shape[1] == (n_nbrs - 1), 'Number of kdtree_query is wrong: ' + str(
        mesh['kdtree_query'].shape[1])


def add_fields_and_dump_model(mesh_data, fileds_needed, out_fn, dataset_name, dump_model=True):
    # mesh_data 就是字典
    # fileds_needed 字典需要的7个键
    # out_fn 输出路径
    # dataset_name = 'coseg'
    m = {}   # 存放模型的字典
    for k, v in mesh_data.items():
        if k in fileds_needed:     # 先把前面搞好的4个写进去，顶点，面，labels(单值标签)，label
            m[k] = v
    for field in fileds_needed:    # 初始化其他键的值
        if field not in m.keys():
            if field == 'labels':  # 前面没有给单值标签，这里就初始化为0
                m[field] = np.zeros((0,))
            if field == 'dataset_name':    # 就是所属数据集
                m[field] = dataset_name
            if field == 'walk_cache':      # 游走的缓存
                m[field] = np.zeros((0,))
            if field == 'kdtree_query' or field == 'edges':  # 两个是一个东西，用kdtree表示边
                prepare_edges_and_kdtree(m)

    if dump_model:
        np.savez(out_fn, **m)    # 一个obj变成一个npz，神经网络就用这个

    return m


def get_labels(dataset_name, mesh, file, fn2labels_map=None):  # 这个mes是字典
    v_labels_fuzzy = np.zeros((0,))
    if dataset_name.startswith('coseg') or dataset_name == 'human_seg_from_meshcnn':
        # labels_fn = '/'.join(file.split('/')[:-2]) + '/seg/' + file.split('/')[-1].split('.')[-2] + '.eseg'
        # labels_fn = 'datasets_raw/from_meshcnn/coseg/coseg_aliens/seg/train\\1.eseg'
        labels_fn = '/'.join(file.split('/')[:-2]) + '/seg/' + file.split('/')[-1].split('.')[-2].split('\\')[-1] + '.eseg'
        # labels_fn = 'datasets_raw/from_meshcnn/coseg/coseg_aliens/seg/1.eseg'
        e_labels = np.loadtxt(labels_fn)  # 单值标签,2250个，应该是边数
        v_labels = [[] for _ in range(mesh['vertices'].shape[0])]  # 初始化，全是空的，这个mesh是字典
        faces = mesh['faces']

        fuzzy_labels_fn = '/'.join(file.split('/')[:-2]) + '/sseg/' + file.split('/')[-1].split('.')[-2] + '.seseg'
        # fuzzy_labels_fn = 'datasets_raw/from_meshcnn/coseg/coseg_aliens/sseg/train\\1.seseg'
        fuzzy_labels_fn = '/'.join(file.split('/')[:-2]) + '/sseg/' + file.split('/')[-1].split('.')[-2].split('\\')[-1] + '.seseg'
        # fuzzy_labels_fn = 'datasets_raw/from_meshcnn/coseg/coseg_aliens/sseg/1.seseg'
        seseg_labels = np.loadtxt(fuzzy_labels_fn)  # 多值标签
        v_labels_fuzzy = np.zeros((mesh['vertices'].shape[0], seseg_labels.shape[1]))

        edge2key = dict()  # 边就是键，边标签文件里面对应的索引就是值，(8, 246) 这条边就是对应 0号边
        edges = []  # 记录所有边的列表
        edges_count = 0
        # 只提供了边的标签，要用这个得到顶点的标签
        # 面里面是顶点的索引 [246 8 118]
        for face_id, face in enumerate(faces):
            faces_edges = []
            for i in range(3):
                cur_edge = (face[i], face[(i + 1) % 3])
                faces_edges.append(cur_edge)  # 一个面的三条边  [(246, 8), (8, 118), (118, 246)]
            for idx, edge in enumerate(faces_edges):
                edge = tuple(sorted(list(edge)))  # 调整顺序，并且变成元组  (8, 246)
                faces_edges[idx] = edge           # 调整好的顺序写回去
                if edge not in edge2key:  # 所有的边都要过一次，也就是一次
                    # 如果可以work的话，也就是说seseg文件里的标签也是按照这个逻辑写的
                    v_labels_fuzzy[edge[0]] += seseg_labels[edges_count]  # 一条边的第一个顶点，多值标签
                    v_labels_fuzzy[edge[1]] += seseg_labels[edges_count]  # 一条边的第二个顶点，都加上去，最后应该要归一化

                    edge2key[edge] = edges_count   # (8, 246) 这条边就是对应 0号边
                    edges.append(list(edge))
                    v_labels[edge[0]].append(e_labels[edges_count])  # 对应的单值标签
                    v_labels[edge[1]].append(e_labels[edges_count])  # 都加进来，后面还要看谁出现次数最多
                    edges_count += 1
        # 非 0 标签的数量不能超过3
        assert np.max(np.sum(v_labels_fuzzy != 0, axis=1)) <= 3, 'Number of non-zero labels must not acceeds 3!'

        vertex_labels = []
        for l in v_labels:
            l2add = np.argmax(np.bincount(l))  # 就是获得出现次数做多的那个数
            vertex_labels.append(l2add)  # 写标签
        vertex_labels = np.array(vertex_labels)
        model_label = np.zeros((0,))

        return model_label, vertex_labels, v_labels_fuzzy  # 模型标签(分类任务)，顶点的单值标签，顶点的多值标签
    else:
        tmp = file.split('/')[-1]
        model_name = '_'.join(tmp.split('_')[:-1])
        if dataset_name.lower().startswith('modelnet'):
            model_label = model_net_shape2label[model_name]
        elif dataset_name.lower().startswith('cubes'):
            model_label = cubes_shape2label[model_name]
        elif dataset_name.lower().startswith('shrec11'):
            model_name = file.split('/')[-3]
            if fn2labels_map is None:
                model_label = shrec11_shape2label[model_name]
            else:
                file_index = int(file.split('.')[-2].split('T')[-1])
                model_label = fn2labels_map[file_index]
        else:
            raise Exception('Cannot find labels for the dataset')
        vertex_labels = np.zeros((0,))
        return model_label, vertex_labels, v_labels_fuzzy


def remesh(mesh_orig, target_n_faces, add_labels=False, labels_orig=None):
    """就是下采样，但是没看到标签变"""
    # mesh_orig 就是加载的 mesh 模型的复制
    # target_n_faces = inf
    # add_labels = 'coseg'
    # labels_orig 顶点的单值标签
    labels = labels_orig
    if target_n_faces < np.asarray(mesh_orig.triangles).shape[0]:  # 小于 1500 就是要下采样，设置的是 inf
        mesh = mesh_orig.simplify_quadric_decimation(target_n_faces)  # 边折叠？反正把面数降下来
        str_to_add = '_simplified_to_' + str(target_n_faces)    # 简化到了多少个面
        mesh = mesh.remove_unreferenced_vertices()              # 清理没用的顶点
        if add_labels and labels_orig.size:   # 没实现，不管他
            pass
            # labels = fix_labels_by_dist(np.asarray(mesh.vertices), np.asarray(mesh_orig.vertices), labels_orig)
    else:
        mesh = mesh_orig   # 面数足够少了
        str_to_add = '_not_changed_' + str(np.asarray(mesh_orig.triangles).shape[0])

    return mesh, labels, str_to_add   # 模型，单值标签，表示模型有没有下少采样的附加信息


def load_mesh(model_fn, classification=True):
    # To load and clean up mesh - "remove vertices that share position"
    if classification:
        mesh_ = trimesh.load_mesh(model_fn, process=True)
        mesh_.remove_duplicate_faces()   # 去除重复的
    else:
        mesh_ = trimesh.load_mesh(model_fn, process=False)   # 加载三角形mesh
    mesh = open3d.geometry.TriangleMesh()                          # 实例化一个 mesh
    mesh.vertices = open3d.utility.Vector3dVector(mesh_.vertices)  # 把顶点写进来
    mesh.triangles = open3d.utility.Vector3iVector(mesh_.faces)    # 把面写进来

    return mesh


def create_tmp_dataset(model_fn, p_out, n_target_faces):
    fileds_needed = ['vertices', 'faces', 'edge_features', 'edges_map', 'edges', 'kdtree_query',
                     'label', 'labels', 'dataset_name']
    if not os.path.isdir(p_out):
        os.makedirs(p_out)
    mesh_orig = load_mesh(model_fn)
    mesh, labels, str_to_add = remesh(mesh_orig, n_target_faces)
    labels = np.zeros((np.asarray(mesh.vertices).shape[0],), dtype=np.int16)  # 对顶点标注？
    mesh_data = EasyDict(
        {'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': 0, 'labels': labels})
    out_fn = p_out + '/tmp'
    add_fields_and_dump_model(mesh_data, fileds_needed, out_fn, 'tmp')


def prepare_directory(dataset_name, pathname_expansion=None, p_out=None, n_target_faces=None, add_labels=True,
                      size_limit=np.inf, fn_prefix='', verbose=True, classification=True):
    # dataset_name = 'coseg'
    # pathname_expansion = 'datasets_raw/from_meshcnn/coseg/coseg_aliens//train/*.obj'
    # p_out = 'datasets_processed/coseg_from_meshcnn/coseg_aliens'
    # n_target_faces = [inf]
    # add_labels = 'coseg'
    # size_limit = np.inf
    # fn_prefix = 'train_'
    # verbose = True
    # classification = false
    fileds_needed = ['vertices', 'faces', 'edges',
                     'label', 'labels', 'dataset_name', 'labels_fuzzy']  # mesh字典的键，7个

    if not os.path.isdir(p_out):
        os.makedirs(p_out)

    filenames = glob.glob(pathname_expansion)  # 要处理的模型路径列表，训练集下的169个模型
    filenames.sort()                           # 先排序
    if len(filenames) > size_limit:
        filenames = filenames[:size_limit]
    for file in tqdm(filenames, disable=1 - verbose):   # diasble 是否输出详细信息
        out_fn = p_out + '/' + fn_prefix + os.path.split(file)[1].split('.')[0]
        # 'datasets_processed/coseg_from_meshcnn/coseg_aliens/train_1'
        mesh = load_mesh(file, classification=classification)
        mesh_orig = mesh
        mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles)})
        if add_labels:  # 是否加上标签
            if type(add_labels) is list:
                fn2labels_map = add_labels
            else:
                fn2labels_map = None
            label, labels_orig, v_labels_fuzzy = get_labels(dataset_name, mesh_data, file, fn2labels_map=fn2labels_map)
        else:
            label = np.zeros((0,))
        for this_target_n_faces in n_target_faces:  # 多个就是要下采样
            mesh, labels, str_to_add = remesh(mesh_orig, this_target_n_faces, add_labels=add_labels,
                                              labels_orig=labels_orig)
            mesh_data = EasyDict(
                {'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': label,
                 'labels': labels})
            mesh_data['labels_fuzzy'] = v_labels_fuzzy
            out_fc_full = out_fn + str_to_add  # 最后的npz输出路径
            add_fields_and_dump_model(mesh_data, fileds_needed, out_fc_full, dataset_name)


# ------------------------------------------------------- #

def prepare_modelnet40():
    n_target_faces = [1000, 2000, 4000]
    labels2use = model_net_labels
    for i, name in tqdm(enumerate(labels2use)):
        for part in ['test', 'train']:
            pin = 'datasets_raw/ModelNet40/' + name + '/' + part + '/'
            p_out = 'datasets_processed/modelnet40/'
            prepare_directory('modelnet40', pathname_expansion=pin + '*.off',
                              p_out=p_out, add_labels='modelnet', n_target_faces=n_target_faces,
                              fn_prefix=part + '_', verbose=False)


def prepare_cubes(labels2use=cubes_labels,
                  path_in='datasets_raw/from_meshcnn/cubes/',
                  p_out='datasets_processed/cubes'):
    dataset_name = 'cubes'
    if not os.path.isdir(p_out):
        os.makedirs(p_out)

    for i, name in enumerate(labels2use):
        print('-->>>', name)
        for part in ['test', 'train']:
            pin = path_in + name + '/' + part + '/'
            prepare_directory(dataset_name, pathname_expansion=pin + '*.obj',
                              p_out=p_out, add_labels=dataset_name, fn_prefix=part + '_', n_target_faces=[np.inf],
                              classification=False)


def prepare_seg_from_meshcnn(dataset, subfolder=None):   # coseg coseg_aliens
    if dataset == 'human_body':
        dataset_name = 'human_seg_from_meshcnn'
        p_in2add = 'human_seg'
        p_out_sub = p_in2add
        p_ext = ''
    elif dataset == 'coseg':
        p_out_sub = dataset_name = 'coseg'
        p_in2add = dataset_name + '/' + subfolder   # 'coseg/coseg_aliens'
        p_ext = subfolder   # 'coseg_aliens'

    path_in = 'datasets_raw/from_meshcnn/' + p_in2add + '/'   # 'datasets_raw/from_meshcnn/coseg/coseg_aliens/'
    p_out = 'datasets_processed/' + p_out_sub + '_from_meshcnn/' + p_ext  # 'datasets_processed/coseg_from_meshcnn/coseg_aliens'

    for part in ['test', 'train']:
        pin = path_in + '/' + part + '/'  # 'datasets_raw/from_meshcnn/coseg/coseg_aliens//train/'
        # pathname_expansion = 'datasets_raw/from_meshcnn/coseg/coseg_aliens//train/*.obj'
        # fn_prefix = 'train_'
        prepare_directory(dataset_name, pathname_expansion=pin + '*.obj',
                          p_out=p_out, add_labels=dataset_name, fn_prefix=part + '_', n_target_faces=[np.inf],
                          classification=False)


# ------------------------------------------------------- #


def prepare_one_dataset(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == 'modelnet40' or dataset_name == 'modelnet':
        prepare_modelnet40()

    if dataset_name == 'shrec11':
        print('To do later')

    if dataset_name == 'cubes':
        prepare_cubes()

    # Semantic Segmentations
    if dataset_name == 'human_seg':
        prepare_seg_from_meshcnn('human_body')

    if dataset_name == 'coseg':
        prepare_seg_from_meshcnn('coseg', 'coseg_aliens')
        # prepare_seg_from_meshcnn('coseg', 'coseg_chairs')
        # prepare_seg_from_meshcnn('coseg', 'coseg_vases')


if __name__ == '__main__':
    utils.config_gpu(False)
    np.random.seed(1)

    if len(sys.argv) != 2:
        print('Use: python dataset_prepare.py <dataset name>')
        print('For example: python dataset_prepare.py cubes')
        print('Another example: python dataset_prepare.py all')
    else:
        dataset_name = sys.argv[1]
        if dataset_name == 'all':
            for dataset_name in ['cubes', 'human_seg', 'coseg', 'modelnet40']:
                prepare_one_dataset(dataset_name)
        else:
            prepare_one_dataset(dataset_name)
