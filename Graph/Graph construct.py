import joblib

# t_img_fea = joblib.load('./luad/luad_256_features/np/TCGA-4B-A93V-01Z-00-DX1.C263DC1C-298D-47ED-AAF8-128043828530.pkl')
# a = t_img_fea
# print(t_img_fea)


import os

#获取图像特征的集合
def create_pkl_dict(folder_path, save_file_path):
    pkl_dict = {}

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(folder_path, file_name)

            # 读取 pkl 文件内容
            with open(file_path, 'rb') as f:
                file_content = joblib.load(f)

            # 将文件名和文件内容添加到字典中
            pkl_dict[file_name[:12]] = file_content
    # 保存字典为 pkl 文件
    # with open(save_file_path, 'wb') as f:
    #     joblib.dump(pkl_dict, f)

    return pkl_dict

# 文件夹路径
folder_path_256 = '../kimaidatas/blca/256/np'
folder_path_512 = '../kimaidatas/blca/512/np'
folder_path_1024 = '../kimaidatas/blca/1024/np'
# 保存文件的路径
save_file_path = '../datas/blca/t_img_fea.pkl'
# 创建字典
pkl_dict_256 = create_pkl_dict(folder_path_256, save_file_path)
pkl_dict_512 = create_pkl_dict(folder_path_512, save_file_path)
pkl_dict_1024 = create_pkl_dict(folder_path_1024, save_file_path)
############################################
###构图

##读取图像特征
# t_img_fea = joblib.load('./luad/t_img_fea.pkl')
# t_img_fea = joblib.load('./HGCN/LUSC/lusc_data.pkl')
t_img_fea_256 = pkl_dict_256
t_img_fea_512 = pkl_dict_512
t_img_fea_1024 = pkl_dict_1024
# print(t_img_fea)
# 获取病人的名称
t_patients_fea = joblib.load('./blca/blca_patients.pkl')
patients = t_patients_fea

##获取病人的生存和时间
t_sur_and_time_fea = joblib.load('./blca/blca_sur_and_time.pkl')
sur_and_time = t_sur_and_time_fea

####RNA
t_rna_fea = joblib.load('./blca/blca_rna.pkl')

#####CNV
t_cnv_fea = joblib.load('./blca/blca_cnv.pkl')

###MUT
t_mut_fea = joblib.load('./blca/blca_mut.pkl')


def get_edge_index_image(id, t_img_fea):
    start = []
    end = []
    if id in t_img_fea:
        patch_id = {}
        i=0
        for x in t_img_fea[id]:
            patch_id[x.split('.')[0]] = i
            i+=1
    #     print(patch_id)
        for x in patch_id:
    #         print(x)
            i = int(x.split('_')[0])
            j = int(x.split('_')[1])#.split('-')[1])
            # j = int(x.split('.')[0].split('-')[1])
            if str(i)+'_'+str(j+1) in patch_id:
                start.append(patch_id[str(i)+'_'+str(j)])
                end.append(patch_id[str(i)+'_'+str(j+1)])
            if str(i)+'_'+str(j-1) in patch_id:
                start.append(patch_id[str(i)+'_'+str(j)])
                end.append(patch_id[str(i)+'_'+str(j-1)])
            if str(i+1)+'_'+str(j) in patch_id:
                start.append(patch_id[str(i)+'_'+str(j)])
                end.append(patch_id[str(i+1)+'_'+str(j)])
            if str(i-1)+'_'+str(j) in patch_id:
                start.append(patch_id[str(i)+'_'+str(j)])
                end.append(patch_id[str(i-1)+'_'+str(j)])
            if str(i+1)+'_'+str(j+1) in patch_id:
                start.append(patch_id[str(i)+'_'+str(j)])
                end.append(patch_id[str(i+1)+'_'+str(j+1)])
            if str(i-1)+'_'+str(j+1) in patch_id:
                start.append(patch_id[str(i)+'_'+str(j)])
                end.append(patch_id[str(i-1)+'_'+str(j+1)])
            if str(i+1)+'_'+str(j-1) in patch_id:
                start.append(patch_id[str(i)+'_'+str(j)])
                end.append(patch_id[str(i+1)+'_'+str(j-1)])
            if str(i-1)+'_'+str(j-1) in patch_id:
                start.append(patch_id[str(i)+'_'+str(j)])
                end.append(patch_id[str(i-1)+'_'+str(j-1)])

    return [start,end]


import torch
from torch_geometric.data import Data
feature_img_256 = {}
feature_img_512 = {}
feature_img_1024 = {}
feature_rna = {}
feature_mut = {}
feature_cnv = {}
data_type = {}
for x in patients:
    f_img_256 = []
    f_img_512 = []
    f_img_1024 = []
    f_rna = []
    f_mut = []
    f_cnv = []
    t_type = []
    if x in t_img_fea_256:
        for z in t_img_fea_256[x]:
            f_img_256.append(t_img_fea_256[x][z])
        # t_type.append('img256')
    if x in t_img_fea_512:
        for z in t_img_fea_512[x]:
            f_img_512.append(t_img_fea_512[x][z])
        # t_type.append('img512')
    if x in t_img_fea_1024:
        for z in t_img_fea_1024[x]:
            f_img_1024.append(t_img_fea_1024[x][z])
        # t_type.append('img1024')
    t_type.append('img')
    if x in t_rna_fea:
        # for r in t_rna_fea[x]:
        f_rna.append([float(k) for k in t_rna_fea[x]])
        # t_type.append('rna')
    t_type.append('rna')
    if x in t_mut_fea:
        # for r in t_mut_fea[x]:
        f_mut.append([float(k) for k in t_mut_fea[x]])
        # t_type.append('mut')
    if x in t_cnv_fea:
        # for r in t_cnv_fea[x]:
        f_cnv.append([float(k) for k in t_cnv_fea[x]])
        # t_type.append('cnv')
    t_type.append('mut_cnv')
    data_type[x]=t_type
    feature_img_256[x] = f_img_256
    feature_img_512[x] = f_img_512
    feature_img_1024[x] = f_img_1024
    feature_rna[x] = f_rna
    feature_mut[x] = f_mut
    feature_cnv[x] = f_cnv



patient_sur_type = {}
for x in patients:
    patient_sur_type[x] = sur_and_time[x][0]

time = []
patient_and_time = {}
for x in patients:
    time.append(sur_and_time[x][-1])
    patient_and_time[x] = sur_and_time[x][-1]
#测试
all_data = {}
for id in data_type:
    print(id)
    node_img_256=torch.tensor(feature_img_256[id], dtype=torch.float)
    node_img_512 = torch.tensor(feature_img_512[id], dtype=torch.float)
    node_img_1024 = torch.tensor(feature_img_1024[id], dtype=torch.float)
    rna_s=torch.tensor(feature_rna[id], dtype=torch.float)
    mut_s = torch.tensor(feature_mut[id], dtype=torch.float)
    cnv_s = torch.tensor(feature_cnv[id], dtype=torch.float)
    edge_index_image_256 = torch.tensor(get_edge_index_image(id, t_img_fea_256), dtype=torch.long)
    edge_index_image_512 = torch.tensor(get_edge_index_image(id, t_img_fea_512), dtype=torch.long)
    edge_index_image_1024 = torch.tensor(get_edge_index_image(id, t_img_fea_1024), dtype=torch.long)
    # edge_index_rna = torch.tensor(get_edge_index_rna(id),dtype=torch.long)
    # edge_index_cli = torch.tensor(get_edge_index_cli(id),dtype=torch.long)
    sur_type=torch.tensor([patient_sur_type[id]])
    data_id = id
    t_data_type = data_type[id]
    data=Data(x_img_256=node_img_256, x_img_512=node_img_512, x_img_1024=node_img_1024,edge_index_image_256=edge_index_image_256,edge_index_image_512=edge_index_image_512,edge_index_image_1024=edge_index_image_1024, rna=rna_s, cnv=cnv_s, mut=mut_s, data_id=id, sur_type=sur_type, data_type=t_data_type)
    all_data[id] = data
    print(data)
    # a = edge_index_image
    # print(edge_index_image)

b = all_data
print(b)
joblib.dump(all_data,'./blca/all_data.pkl')