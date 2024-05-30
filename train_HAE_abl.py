import os
import yaml
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import copy
import torch
import joblib
import random
import json
import math
import sys
import argparse
import numpy as np
import torch.nn as nn
import time as sys_time
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.model_selection import KFold
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index as ci
from sklearn.model_selection import StratifiedKFold
from models.models_hae_abl import fusion_model_mae

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def adjust_learning_rate(optimizer, lr, epoch, lr_step=20, lr_gamma=0.5):
    lr = lr * (lr_gamma ** (epoch // lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_val_ci(pre_time,patient_and_time,patient_sur_type):
    ordered_time, ordered_pred_time, ordered_observed=[],[],[]
    for x in pre_time:
        ordered_time.append(patient_and_time[x])
        ordered_pred_time.append(pre_time[x]*-1)
        ordered_observed.append(patient_sur_type[x])
#     print(len(ordered_time), len(ordered_pred_time), len(ordered_observed))
    return ci(ordered_time, ordered_pred_time, ordered_observed)

def get_all_ci(pre_time,patient_and_time,patient_sur_type):
    ordered_time, ordered_pred_time, ordered_observed=[],[],[]
    for x in patient_and_time:
        ordered_time.append(patient_and_time[x])
        ordered_pred_time.append(pre_time[x]*-1)
        ordered_observed.append(patient_sur_type[x])
#     print(ci(ordered_time, ordered_pred_time, ordered_observed))
    return ci(ordered_time, ordered_pred_time, ordered_observed)


def generate_mask(num=3):
    mask_num = num-1
#     mask_num = random.randint(1,2)
    mask = np.hstack([
        np.zeros(num-mask_num,dtype=bool),
        np.ones(mask_num,dtype=bool),
    ])
    np.random.shuffle(mask)
    mask = np.expand_dims(mask,0)
    mask = np.expand_dims(mask,0)
    return mask


def get_patients_information(patients, sur_and_time):
    patient_sur_type = {}
    for x in patients:
        patient_sur_type[x] = sur_and_time[x][0]

    time = []
    patient_and_time = {}
    for x in patients:
        time.append(sur_and_time[x][-1])
        patient_and_time[x] = sur_and_time[x][-1]

    kf_label = []
    for x in patients:
        kf_label.append(patient_sur_type[x])

    return patient_sur_type, patient_and_time, kf_label
def prediction(all_data, v_model, val_id, patient_and_time, patient_sur_type, args):
    v_model.eval()

    lbl_pred_all = None
    status_all = []
    survtime_all = []
    val_pre_time = {}
    val_pre_time_img = {}
    val_pre_time_rna = {}
    val_pre_time_mut_cnv = {}
    iter = 0
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            for i_batch, id in enumerate(val_id):

                graph = all_data[id].to(device)
                if args.train_use_type != None:
                    use_type_eopch = args.train_use_type
                else:
                    use_type_eopch = graph.data_type
                out_pre, out_fea, out_att, _ = v_model(graph, args.train_use_type, use_type_eopch, mix=args.mix, hae = args.hae)
                lbl_pred = out_pre[0]

                survtime_all.append(patient_and_time[id])
                status_all.append(patient_sur_type[id])

                val_pre_time[id] = lbl_pred.cpu().detach().numpy()[0]

                if iter == 0 or lbl_pred_all == None:
                    lbl_pred_all = lbl_pred
                else:
                    lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])

                iter += 1

                if 'img' in use_type_eopch:
                    val_pre_time_img[id] = out_pre[1][use_type_eopch.index('img')].cpu().detach().numpy()
                if 'rna' in use_type_eopch:
                    val_pre_time_rna[id] = out_pre[1][use_type_eopch.index('rna')].cpu().detach().numpy()
                if 'mut_cnv' in use_type_eopch:
                    val_pre_time_mut_cnv[id] = out_pre[1][use_type_eopch.index('mut_cnv')].cpu().detach().numpy()

    survtime_all = np.asarray(survtime_all)
    status_all = np.asarray(status_all)
    #     print(lbl_pred_all,survtime_all,status_all)
    loss_surv = _neg_partial_log(lbl_pred_all, survtime_all, status_all)
    loss = loss_surv

    val_ci_ = get_val_ci(val_pre_time, patient_and_time, patient_sur_type)
    val_ci_img_ = 0
    val_ci_rna_ = 0
    val_ci_mut_cnv_ = 0

    if 'img' in args.train_use_type:
        val_ci_img_ = get_val_ci(val_pre_time_img, patient_and_time, patient_sur_type)
    if 'rna' in args.train_use_type:
        val_ci_rna_ = get_val_ci(val_pre_time_rna, patient_and_time, patient_sur_type)
    if 'mut_cnv' in args.train_use_type:
        val_ci_mut_cnv_ = get_val_ci(val_pre_time_mut_cnv, patient_and_time, patient_sur_type)
    return loss.item(), val_ci_, val_ci_img_, val_ci_rna_, val_ci_mut_cnv_


def _neg_partial_log(prediction, T, E):
    current_batch_len = len(prediction)
    R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_matrix_train[i, j] = T[j] >= T[i]

    train_R = torch.FloatTensor(R_matrix_train)
    train_R = train_R.cuda()

    train_ystatus = torch.tensor(np.array(E), dtype=torch.float).to(device)

    theta = prediction.reshape(-1)

    exp_theta = torch.exp(theta)
    loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)

    return loss_nn


def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True



def train_a_epoch(model, train_data, all_data, patient_and_time, patient_sur_type, batch_size, optimizer, epoch,
                  format_of_coxloss, args):
    model.train()

    lbl_pred_each = None
    lbl_pred_img_each = None
    lbl_pred_rna_each = None
    lbl_pred_cli_each = None
    batch_feature = None

    survtime_all = []
    status_all = []
    survtime_img = []
    status_img = []
    survtime_rna = []
    status_rna = []
    survtime_cli = []
    status_cli = []

    iter = 0
    loss_nn_all = []
    train_pre_time = {}
    train_pre_time_img = {}
    train_pre_time_rna = {}
    train_pre_time_cli = {}

    all_loss = 0.0
    mes_loss_of_mae = nn.MSELoss()

    mse_loss_of_mae = 0.0
    loss_surv = 0.0
    all_loss_surv = 0.0
    img_loss_surv = 0.0
    rna_loss_surv = 0.0
    cli_loss_surv = 0.0

    scaler = torch.cuda.amp.GradScaler()
    with torch.cuda.amp.autocast():
        for i_batch, id in enumerate(train_data):

            iter += 1
            num_of_model = len(all_data[id].data_type)
            mask = generate_mask(num=len(args.train_use_type))
            mask1 = generate_mask(num=len(args.train_use_type))
            mask2 = generate_mask(num=len(args.train_use_type))

            if len(args.train_use_type) == 1:
                assert args.format_of_coxloss == 'one' and args.add_mse_loss_of_mae == False
                if args.train_use_type[0] in all_data[id].data_type:
                    graph = all_data[id].to(device)
                    with torch.cuda.amp.autocast():
                        out_pre, out_fea, out_att, fea_dict = model(graph, args.train_use_type, args.train_use_type, mix=args.mix, hae=args.hae)
                        lbl_pred = out_pre[0]
                        use_type_eopch = args.train_use_type
                        num_of_model = 1
            else:
                if args.train_use_type != None:
                    use_type_eopch = args.train_use_type
                    num_of_model = len(use_type_eopch)
                else:
                    use_type_eopch = all_data[id].data_type
                # with torch.cuda.amp.autocast():
                graph = all_data[id].to(device)
                with torch.cuda.amp.autocast():
                    out_pre, out_fea, out_att, fea_dict = model(graph, use_type_eopch, use_type_eopch, mask, mask1, mask2, mix=args.mix, hae=args.hae)
                    lbl_pred = out_pre[0]

            if len(args.train_use_type) == 1 and args.train_use_type[0] not in all_data[id].data_type:
                pass
            else:
                train_pre_time[id] = lbl_pred.cpu().detach().numpy()

                if args.add_mse_loss_of_mae:
                    mse_loss_of_mae += args.mse_loss_of_mae_factor * mes_loss_of_mae(input=fea_dict['mae_out'][mask[0]],
                                                                                     target=fea_dict['mae_labels'][mask[0]])

                survtime_all.append(patient_and_time[id])
                status_all.append(patient_sur_type[id])
                if iter == 0 or lbl_pred_each == None:
                    lbl_pred_each = lbl_pred
                else:
                    lbl_pred_each = torch.cat([lbl_pred_each, lbl_pred])

                if 'img' in use_type_eopch and len(args.train_use_type) != 1:
                    train_pre_time_img[id] = out_pre[1][use_type_eopch.index('img')].cpu().detach().numpy()
                    survtime_img.append(patient_and_time[id])
                    status_img.append(patient_sur_type[id])
                    if lbl_pred_img_each == None:
                        lbl_pred_img_each = out_pre[1][use_type_eopch.index('img')]
                    else:
                        lbl_pred_img_each = torch.cat([lbl_pred_img_each, out_pre[1][use_type_eopch.index('img')]])
                if 'rna' in use_type_eopch and len(args.train_use_type) != 1:
                    train_pre_time_rna[id] = out_pre[1][use_type_eopch.index('rna')].cpu().detach().numpy()
                    survtime_rna.append(patient_and_time[id])
                    status_rna.append(patient_sur_type[id])
                    if lbl_pred_rna_each == None:
                        lbl_pred_rna_each = out_pre[1][use_type_eopch.index('rna')]
                    else:
                        lbl_pred_rna_each = torch.cat([lbl_pred_rna_each, out_pre[1][use_type_eopch.index('rna')]])
                if 'cli' in use_type_eopch and len(args.train_use_type) != 1:
                    train_pre_time_cli[id] = out_pre[1][use_type_eopch.index('cli')].cpu().detach().numpy()
                    survtime_cli.append(patient_and_time[id])
                    status_cli.append(patient_sur_type[id])
                    if lbl_pred_cli_each == None:
                        lbl_pred_cli_each = out_pre[1][use_type_eopch.index('cli')]
                    else:
                        lbl_pred_cli_each = torch.cat([lbl_pred_cli_each, out_pre[1][use_type_eopch.index('cli')]])

            if iter % batch_size == 0 or i_batch == len(train_data) - 1:

                survtime_all = np.asarray(survtime_all)
                status_all = np.asarray(status_all)

                if np.max(status_all) == 0:
                    lbl_pred_each = None
                    lbl_pred_img_each = None
                    lbl_pred_rna_each = None
                    lbl_pred_cli_each = None
                    batch_feature = None
                    con_loss_label = None
                    con_time_label = None
                    survtime_all = []
                    status_all = []
                    survtime_img = []
                    status_img = []
                    survtime_rna = []
                    status_rna = []
                    survtime_cli = []
                    status_cli = []
                    iter = 0
                    mse_loss_of_mae = 0.0
                    loss_surv = 0.0
                    all_loss_surv = 0.0
                    img_loss_surv = 0.0
                    rna_loss_surv = 0.0
                    cli_loss_surv = 0.0
                    continue

                optimizer.zero_grad()

                if format_of_coxloss == 'one':
                    all_loss_surv = _neg_partial_log(lbl_pred_each, survtime_all, status_all)
                    loss_surv = args.all_cox_loss_factor * all_loss_surv
                elif format_of_coxloss == 'multi':
                    if lbl_pred_img_each != None:
                        img_loss_surv = args.img_cox_loss_factor * _neg_partial_log(lbl_pred_img_each, survtime_img,
                                                                                    status_img)
                        loss_surv += img_loss_surv

                    if lbl_pred_rna_each != None:
                        rna_loss_surv = args.rna_cox_loss_factor * _neg_partial_log(lbl_pred_rna_each, survtime_rna,
                                                                                    status_rna)
                        loss_surv += rna_loss_surv

                    if lbl_pred_cli_each != None:
                        cli_loss_surv = args.cli_cox_loss_factor * _neg_partial_log(lbl_pred_cli_each, survtime_cli,
                                                                                    status_cli)
                        loss_surv += cli_loss_surv
                else:
                    raise ("Wrong format_of_coxloss")

                loss = loss_surv

                if args.add_mse_loss_of_mae:
                    mse_loss_of_mae /= iter
                    loss += mse_loss_of_mae

                all_loss += loss.item()
                # loss.backward()
                scaler.scale(loss).backward()
                if epoch == 0:
                    print('*', end='')
                else:
                    # optimizer.step()
                    scaler.step(optimizer)
                    scaler.update()

                torch.cuda.empty_cache()
                lbl_pred_each = None
                lbl_pred_img_each = None
                lbl_pred_rna_each = None
                lbl_pred_cli_each = None
                batch_feature = None
                con_loss_label = None
                con_time_label = None
                survtime_all = []
                status_all = []
                survtime_img = []
                status_img = []
                survtime_rna = []
                status_rna = []
                survtime_cli = []
                status_cli = []
                loss_nn_all.append(loss.data.item())
                con_loss = 0.0
                mse_loss = 0.0
                mse_loss_of_mae = 0.0
                kl_loss = 0.0
                loss_surv = 0.0
                all_loss_surv = 0.0
                img_loss_surv = 0.0
                rna_loss_surv = 0.0
                cli_loss_surv = 0.0
                iter = 0

    t_train_ci_img = 0
    t_train_ci_rna = 0
    t_train_ci_cli = 0
    all_loss = all_loss / len(train_data) * batch_size
    t_train_ci = get_val_ci(train_pre_time, patient_and_time, patient_sur_type)
    if len(args.train_use_type) != 1:
        if 'img' in args.train_use_type:
            t_train_ci_img = get_val_ci(train_pre_time_img, patient_and_time, patient_sur_type)
        if 'rna' in args.train_use_type:
            t_train_ci_rna = get_val_ci(train_pre_time_rna, patient_and_time, patient_sur_type)
        if 'cli' in args.train_use_type:
            t_train_ci_cli = get_val_ci(train_pre_time_cli, patient_and_time, patient_sur_type)

    return all_loss, t_train_ci, t_train_ci_img, t_train_ci_rna, t_train_ci_cli


def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)

    return dict

def main(args):
    global model_mae, all_data
    path = args.config
    Dict = read_yaml(path)
    start_seed = args.start_seed
    cancer_type = args.cancer_type
    repeat_num = args.repeat_num
    drop_out_ratio = args.drop_out_ratio
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    details = args.details
    fusion_model = args.fusion_model
    format_of_coxloss = args.format_of_coxloss
    if_adjust_lr = args.if_adjust_lr

    label = "{} {} lr_{} {}_coxloss".format(cancer_type, details, lr, format_of_coxloss)

    if args.add_mse_loss_of_mae:
        label = label + " {}xmae_loss".format(args.mse_loss_of_mae_factor)

    if args.img_cox_loss_factor != 1:
        label = label + " img_ft_{}".format(args.img_cox_loss_factor)
    if args.rna_cox_loss_factor != 1:
        label = label + " rna_ft_{}".format(args.rna_cox_loss_factor)
    if args.mut_cnv_cox_loss_factor != 1:
        label = label + " mut_cnv_ft_{}".format(args.mut_cnv_cox_loss_factor)
    if args.mix:
        label = label + " mix"
    if args.train_use_type != None:
        label = label + ' use_'
        for x in args.train_use_type:
            label = label + x

    print(label)

    if cancer_type == 'lihc':
        patients = joblib.load('../HGCN/LIHC/lihc_patients.pkl')
        sur_and_time = joblib.load('../HGCN/LIHC/lihc_sur_and_time.pkl')
        all_data = joblib.load('../HGCN/LIHC/lihc_data.pkl')
        seed_fit_split = joblib.load('../HGCN/LIHC/lihc_split.pkl')
    elif cancer_type == 'lusc':
        patients = joblib.load('../HGCN/LUSC/lusc_patients.pkl')
        sur_and_time = joblib.load('../HGCN/LUSC/lusc_sur_and_time.pkl')
        all_data = joblib.load('../HGCN/LUSC/lusc_data.pkl')
        seed_fit_split = joblib.load('../HGCN/LUSC/lusc_split.pkl')
    elif cancer_type == 'esca':
        patients = joblib.load('../HGCN/ESCA/esca_patients.pkl')
        sur_and_time = joblib.load('../HGCN/ESCA/esca_sur_and_time_154.pkl')
        all_data = joblib.load('../HGCN/ESCA/esca_data.pkl')
        seed_fit_split = joblib.load('../HGCN/ESCA/esca_split.pkl')
    elif cancer_type == 'luad':
        patients = joblib.load('../datas/luad/luad_patients.pkl')
        sur_and_time = joblib.load('../datas/luad/luad_sur_and_time.pkl')
        all_data = joblib.load('../datas/luad/luad_data.pkl')
        seed_fit_split = joblib.load('../datas/luad/luad_split.pkl')
    elif cancer_type == 'brca':
        patients = joblib.load('../datas/brca/brca_patients.pkl')
        sur_and_time = joblib.load('../datas/brca/brca_sur_and_time.pkl')
        all_data = joblib.load('../datas/brca/brca_data.pkl')
        seed_fit_split = joblib.load('../datas/brca/brca_split.pkl')
    elif cancer_type == 'ucec':
        patients = joblib.load('../datas/ucec/ucec_patients.pkl')
        sur_and_time = joblib.load('../datas/ucec/ucec_sur_and_time.pkl')
        all_data = joblib.load('../datas/ucec/ucec_data.pkl')
        seed_fit_split = joblib.load('../datas/ucec/ucec_split.pkl')
    elif cancer_type == 'blca':
        patients = joblib.load('../datas/blca/blca_patients.pkl')
        sur_and_time = joblib.load('../datas/blca/blca_sur_and_time.pkl')
        all_data = joblib.load('../datas/blca/blca_data.pkl')
        seed_fit_split = joblib.load('../datas/blca/blca_split.pkl')

    patient_sur_type, patient_and_time, kf_label = get_patients_information(patients, sur_and_time)

    all_seed_patients = []

    all_fold_test_ci = []
    all_fold_test_ci_mut_cnv_3 = []

    all_all_ci = []
    all_gnn_time = []
    all_each_model_time = []
    all_fold_each_model_ci = []
    ##

    all_epoch_val_loss = []
    all_epoch_test_loss = []

    all_epoch_test_img_ci = []
    all_epoch_test_rna_ci = []
    all_epoch_test_mut_cnv_ci = []

    all_epoch_train_ci = []
    all_epoch_val_ci = []
    all_epoch_test_ci = []

    repeat = -1
    for seed in range(start_seed, start_seed + repeat_num):
        repeat += 1
        setup_seed(0)

        seed_patients = []
        gnn_feature = {}
        one_test_feature = {}
        gnn_time = {}
        each_model_time = {'img': {}, 'rna': {}, 'mut_cnv': {}, 'imgrna': {}, 'imgmut_cnv': {}, 'rnamut_cnv': {}}

        val_gnn_time = {}
        test_fold_ci = []

        val_fold_ci = []
        test_each_model_ci = {'img': [], 'rna': [], 'mut_cnv': [], 'imgrna': [], 'imgmut_cnv': [], 'rnamut_cnv': []}

        train_fold_ci = []
        fold_att_1 = {}
        fold_att_2 = {}

        epoch_train_loss = []
        epoch_val_loss = []
        epoch_test_loss = []
        epoch_train_ci = []
        epoch_val_ci = []
        epoch_test_ci = []

        n_fold = 0

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_index, test_index in kf.split(patients, kf_label):
            fold_patients = []
            n_fold += 1
            print('fold: ', n_fold)

            if fusion_model == 'fusion_model':
                model=fusion_model_mae(dict=Dict, #n_hidden=args.n_hidden,
                                         out_classes=args.out_classes,
                                         dropout=drop_out_ratio,
                                         train_type_num=len(args.train_use_type)).to(device)
                # print(model)

            # model = model()
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)

            if args.if_fit_split:
                train_data = seed_fit_split[n_fold - 1][0]
                val_data = seed_fit_split[n_fold - 1][1]
                test_data = seed_fit_split[n_fold - 1][2]
            else:
                t_train_data = np.array(patients)[train_index]
                t_l = []
                for x in t_train_data:
                    t_l.append(patient_sur_type[x])
                train_data, val_data, _, _ = train_test_split(t_train_data, t_train_data, test_size=0.25,
                                                              random_state=1, stratify=t_l)
                test_data = np.array(patients)[test_index]

            print(len(train_data), len(val_data), len(test_data))
            fold_patients.append(train_data)
            fold_patients.append(val_data)
            fold_patients.append(test_data)
            seed_patients.append(fold_patients)

            best_loss = 9999
            best_val_ci = 0
            tmp_train_ci = 0

            for epoch in range(epochs):

                if if_adjust_lr:
                    adjust_learning_rate(optimizer, lr, epoch, lr_step=20, lr_gamma=args.adjust_lr_ratio)
                # with torch.no_grad:
                all_loss, t_train_ci, t_train_ci_img, t_train_ci_rna, t_train_ci_mut_cnv = train_a_epoch(model, train_data, all_data,
                                                                                                     patient_and_time,
                                                                                                     patient_sur_type,
                                                                                                     batch_size,
                                                                                                     optimizer, epoch,
                                                                                                     format_of_coxloss,
                                                                                                     args)

                t_test_loss, test_ci, test_img_ci, test_rna_ci, test_mut_cnv_ci = prediction(all_data, model, test_data,
                                                                                         patient_and_time,
                                                                                         patient_sur_type, args)
                v_loss, val_ci, val_img_ci, val_rna_ci, val_mut_cnv_ci = prediction(all_data, model, val_data,
                                                                                patient_and_time, patient_sur_type,
                                                                                args)

                if val_ci >= best_val_ci and epoch > 1:
                    best_val_ci = val_ci
                    tmp_train_ci = t_train_ci
                    print(val_ci)
                    t_model = copy.deepcopy(model)

                print(
                    "epoch：{:2d}，train_loos：{:.4f},train_ci：{:.4f},val_loos：{:.4f},val_ci：{:.4f},test_loos：{:.4f},test_ci：{:.5f}".format(
                        epoch, all_loss, t_train_ci, v_loss, val_ci, t_test_loss, test_ci))

            t_model.eval()

            t_test_loss, test_ci, _, _, _ = prediction(all_data, t_model, test_data, patient_and_time, patient_sur_type,
                                                       args)

            test_fold_ci.append(test_ci)
            val_fold_ci.append(best_val_ci)
            train_fold_ci.append(tmp_train_ci)

            one_model_res = [{}, {}, {}]
            two_model_res = [{}, {}, {}]
            fold_fusion_test_ci = {}
            with torch.no_grad():
                for id in test_data:
                    data = all_data[id]
                    data.to(device)
                    (one_x, multi_x), fea, (att_1, att_2), _ = t_model(data, args.train_use_type, args.train_use_type, mix=args.mix, hae = args.hae)
                    gnn_time[id] = one_x.cpu().detach().numpy()[0]
                    fold_fusion_test_ci[id] = one_x.cpu().detach().numpy()[0]
                    print(data.sur_type.cpu().detach().numpy()[0], one_x.cpu().detach().numpy()[0],
                          patient_and_time[id])
                    one_test_feature[id] = {}
                    for i, type_name in enumerate(['img', 'rna', 'mut_cnv']):
                        if type_name in data.data_type:
                            (one_, _), one_fea, (_, _), _ = t_model(data, args.train_use_type, use_type=[type_name],
                                                                    mix=args.mix)
                            one_model_res[i][id] = one_.cpu().detach().numpy()[0]
                            each_model_time[type_name][id] = one_.cpu().detach().numpy()[0]

                    for i, two_type_name in enumerate([['img', 'rna'], ['img', 'mut_cnv'], ['rna', 'mut_cnv']]):
                        (one_, two_), one_fea, (_, _), _ = t_model(data, args.train_use_type, use_type=two_type_name,
                                                                   mix=args.mix)
                        two_model_res[i][id] = one_.cpu().detach().numpy()[0]
                        cat_name = two_type_name[0] + two_type_name[1]
                        each_model_time[cat_name][id] = one_.cpu().detach().numpy()[0]

                    del data
            for i, type_name in enumerate(['img', 'rna', 'mut_cnv']):
                t_ci = get_val_ci(one_model_res[i], patient_and_time, patient_sur_type)
                test_each_model_ci[type_name].append(t_ci)
                print(len(one_model_res[i]), ' ', type_name, ' mut_cnv:', t_ci)

            for i, type_name in enumerate([['img', 'rna'], ['img', 'mut_cnv'], ['rna', 'mut_cnv']]):
                t_ci = get_val_ci(two_model_res[i], patient_and_time, patient_sur_type)
                cat_name = type_name[0] + type_name[1]
                test_each_model_ci[cat_name].append(t_ci)
                print(len(two_model_res[i]), ' ', cat_name, ' ci:', t_ci)

            test_ci = get_val_ci(fold_fusion_test_ci, patient_and_time, patient_sur_type)
            print('all ci:', test_ci)

            torch.save(t_model.state_dict(),'../results/UCEC/our/' + sys_time.strftime('%Y-%m-%d') + label + '_' + str(seed) + '_' + str( n_fold) + '.pth')
            del model, train_data, test_data, t_model

        print('seed: ', seed)
        print('test fold ci:')
        for x in test_fold_ci:
            print(x)

        print('all ci:')
        print(get_all_ci(gnn_time, patient_and_time, patient_sur_type))

        print('val fold ci:')
        for x in val_fold_ci:
            print(x)

        all_fold_test_ci.append(test_fold_ci)
        all_fold_each_model_ci.append(test_each_model_ci)
        all_all_ci.append(get_all_ci(gnn_time, patient_and_time, patient_sur_type))
        all_gnn_time.append(gnn_time)
        all_each_model_time.append(each_model_time)

    print('summary :')
    print(label)

    print('fusion test fold ci')
    for i, x in enumerate(all_fold_test_ci):
        print(x)

    for i, type_name in enumerate(['img', 'rna', 'mut_cnv', 'imgrna', 'imgmut_cnv', 'rnamut_cnv']):

        print(type_name, ' ci:')
        for fold_ in all_fold_each_model_ci:
            print(fold_[type_name])

    joblib.dump(all_gnn_time, '../results/UCEC/our/' + sys_time.strftime('%Y-%m-%d-%H-%M') + label + '.pkl')
    joblib.dump(all_each_model_time, '../results/UCEC/our/' + sys_time.strftime('%Y-%m-%d-%H-%M') + label + '.pkl')
    joblib.dump(all_all_ci, '../results/UCEC/our/' + 'all' + sys_time.strftime('%Y-%m-%d-%H-%M') + label + '.pkl')


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='E:\\multi-mode\\Configs\\BLCA.yml', help='yml path')
    parser.add_argument('-seed', type=int, help='random seed of the run', default=611)
    parser.add_argument("--cancer_type", type=str, default="blca", help="Cancer type")
    parser.add_argument("--img_cox_loss_factor", type=float, default=5, help="img_cox_loss_factor")
    parser.add_argument("--rna_cox_loss_factor", type=float, default=1, help="rna_cox_loss_factor")
    parser.add_argument("--mut_cnv_cox_loss_factor", type=float, default=5, help="mut_cnv_cox_loss_factor")
    parser.add_argument("--train_use_type", type=list, default=['img', 'rna', 'mut_cnv'], #, 'mut_cnv'
                        help='train_use_type,Please keep the relative order of img, rna, mut_cnv')
    parser.add_argument("--format_of_coxloss", type=str, default="multi", help="format_of_coxloss:multi,one")
    parser.add_argument("--add_mse_loss_of_mae", action='store_true', default=True, help="add_mse_loss_of_mae")
    parser.add_argument("--mse_loss_of_mae_factor", type=float, default=5, help="mae_loss_factor")
    parser.add_argument("--start_seed", type=int, default=0, help="start_seed")
    parser.add_argument("--repeat_num", type=int, default=5, help="Number of repetitions of the experiment")
    parser.add_argument("--fusion_model", type=str, default="fusion_model", help="")
    parser.add_argument("--drop_out_ratio", type=float, default=0.2, help="Drop_out_ratio")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate of model training")
    parser.add_argument("--epochs", type=int, default=50, help="Cycle times of model training")
    parser.add_argument("--batch_size", type=int, default=10, help="Data volume of model training once")
    parser.add_argument("--n_hidden", type=int, default=512, help="Model middle dimension")
    parser.add_argument("--out_classes", type=int, default=500, help="Model out dimension")
    parser.add_argument("--mix", action='store_true', default=True, help="mix mae")
    parser.add_argument("--if_adjust_lr", action='store_true', default=True, help="if_adjust_lr")
    parser.add_argument("--adjust_lr_ratio", type=float, default=0.5, help="adjust_lr_ratio")
    parser.add_argument("--if_fit_split", action='store_true', default=False, help="fixed division/random division")
    parser.add_argument("--details", type=str, default='', help="Experimental details")
    parser.add_argument("--hae", type=str, default='CNN', help="Experimental details")


    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        args = get_params()
        main(args)
    except Exception as exception:
        raise
