
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import math
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from HIGSyn_GDSC2 import HIGSyn, DrugGNN, CellMLP, SSI, HgnnEncoder
import glob
from drug_util import GraphDataset, collate, drug_feature_extract, smile_to_graph
from utils import metrics_graph, set_seed_all, get_ECFP_with_counts
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
import warnings
warnings.filterwarnings("ignore")
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')


# --train+test
def train_with_epoch_accumulation(drug_fea_set, cline_fea_set, synergy_adj, index, label, batch_size, epoch, total_epochs):
    loss_train = 0
    true_ls, pre_ls = [], []
    scaler = torch.cuda.amp.GradScaler()  # 初始化混合精度缩放器
    optimizer.zero_grad()

    num_batches = math.ceil(len(index) / batch_size)  # 计算总批次数

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(index))
        batch_index = index[start_idx:end_idx]
        batch_label = label[start_idx:end_idx]
        neg_ids = batch_index[batch_label == 0]
        pos_ids = batch_index[batch_label == 1]

        # 前向传播
        with torch.cuda.amp.autocast():  # 混合精度训练
            pred, rec_drug, rec_cline, contrastive_loss = model(
                drug_fea_set.x, drug_fea_set.edge_index, drug_fea_set.batch, cline_fea_set, synergy_adj,
                batch_index[:, 0], batch_index[:, 1], batch_index[:, 2], neg_ids, pos_ids)

            loss = loss_func(pred, batch_label)
            loss_rec_1 = loss_func(rec_drug, drug_sim_mat)
            loss_rec_2 = loss_func(rec_cline, cline_sim_mat)
            loss = loss + 0.5 * loss_rec_1 + 0.5 * loss_rec_2 + 0.5 * contrastive_loss
            loss = loss / num_batches  # 将损失按总批次数平均分摊

        # 累积梯度
        scaler.scale(loss).backward()

        # 累积损失
        loss_train += loss.item()
        true_ls += batch_label.cpu().detach().numpy().tolist()
        pre_ls += torch.sigmoid(pred).cpu().detach().numpy().tolist()

        # 每个 batch 计算评估指标
        auc_train, aupr_train, f1_train, acc_train, recall_train, pre_train, kappa_train, bacc_train = metrics_graph(
            true_ls, pre_ls
        )

        # 打印当前 batch 进度
        progress = (batch_idx + 1) / num_batches * 100
        print(f'Epoch {epoch}/{total_epochs}, Batch {batch_idx + 1}/{num_batches} ({progress:.2f}%): '
              f'Loss: {loss_train:.6f}, AUC: {auc_train:.6f}, PR AUC: {aupr_train:.6f}, '
              f'F1: {f1_train:.6f}, ACC: {acc_train:.6f}, Recall: {recall_train:.6f}, '
              f'Precision: {pre_train:.6f}, Kappa: {kappa_train:.6f}, BACC: {bacc_train:.6f}')

    # 一个 epoch 结束后更新梯度
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()  # 清空累积的梯度

    return [auc_train, aupr_train, f1_train, acc_train, recall_train, pre_train, kappa_train, bacc_train], loss_train

def test(drug_fea_set, cline_fea_set, synergy_adj, index, label, batch_size):
    model.eval()
    loss_test = 0
    true_ls, pre_ls = [], []

    num_batches = (len(index) + batch_size - 1) // batch_size  # 向上取整计算 batch 数量

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(index))
            batch_index = index[start_idx:end_idx]
            batch_label = label[start_idx:end_idx]
            neg_ids = batch_index[batch_label == 0]
            pos_ids = batch_index[batch_label == 1]

            with torch.cuda.amp.autocast():  # 混合精度
                pred, rec_drug, rec_cline, contrastive_loss = model(
                    drug_fea_set.x, drug_fea_set.edge_index, drug_fea_set.batch, cline_fea_set, synergy_adj,
                    batch_index[:, 0], batch_index[:, 1], batch_index[:, 2], neg_ids, pos_ids)  # pred shape: (batch_size,)

                # 计算损失
                loss = loss_func(pred, batch_label)
                loss_rec_1 = loss_func(rec_drug, drug_sim_mat)
                loss_rec_2 = loss_func(rec_cline, cline_sim_mat)
                loss = loss + 0.5 * loss_rec_1 + 0.5 * loss_rec_2 + 0.5 * contrastive_loss
                loss = loss / num_batches  # 将损失按总批次数平均分摊

            loss_test += loss.item()  # 累计损失
            true_ls += batch_label.cpu().detach().numpy().tolist()

            # 将 logits 转换为概率，用于评估指标
            probabilities = torch.sigmoid(pred).cpu().detach().numpy()
            pre_ls += probabilities.tolist()

            # 打印进度
            progress = (batch_idx + 1) / num_batches * 100
            print(f"Test Progress: {progress:.2f}% (Batch {batch_idx + 1}/{num_batches})")

        # 计算评估指标
        auc_test, aupr_test, f1_test, acc_test, recall_test, pre_test, kappa_test, bacc_test = metrics_graph(
            true_ls, pre_ls
        )

        # 返回测试评估指标、平均损失值和所有预测的概率
        return [auc_test, aupr_test, f1_test, acc_test, recall_test, pre_test, kappa_test, bacc_test], loss_test / num_batches, pre_ls




if __name__ == '__main__':
    for fold in range(0, 5):
        path = f'CV5_GDSC2/fold_{fold}/'
        if not os.path.exists(path):
            os.makedirs(path)

        file = open(path + 'result.txt', 'w')
        final_metric = np.zeros(4)

        seed = 0
        epochs = 2000
        learning_rate = 0.0001

        drug = pd.read_csv('Data/GDSC2/GDSC2_cid_unique.csv')
        drug = drug.drop_duplicates()
        drug_data = pd.DataFrame()  # (4092, 3)
        drug_smiles_fea = []
        for tup in zip(drug['DrugName'], drug['smiles']):
            smile = tup[1]
            c_size, atom_features_list, edge_index = smile_to_graph(smile)
            drug_data[str(tup[0])] = [np.array(atom_features_list), edge_index]
            drug_smiles_fea.append(get_ECFP_with_counts(tup[1], radius=3, nBits=1024))
        drug_num = len(drug_data.keys())  # drug_num: 4092
        print(f'drug_num:{drug_num}')
        d_map = dict(zip(drug_data.keys(), range(drug_num)))
        drug_feature = drug_feature_extract(drug_data)  # len: 4092; get_atom_features, adjacency_list

        gene_data = pd.read_csv('Data/GDSC2/GDSC2_cellline(z_score).csv', sep=',', header=0, index_col=[0])
        cline_num = len(gene_data.index)
        print(f'cline_num:{cline_num}')
        c_map = dict(zip(gene_data.index, range(drug_num, drug_num + cline_num)))
        cline_feature = np.array(gene_data, dtype='float32')  # torch.Size([136, 954])

        synergy_load_train = pd.read_csv(f'Data/GDSC2/GDSC2_cv5_train_fold_{fold}.csv')
        synergy_load_validation = pd.read_csv(f'Data/GDSC2/GDSC2_cv5_valid_fold_{fold}.csv')
        synergy_load_test = pd.read_csv(f'Data/GDSC2/GDSC2_cv5_test_fold_{fold}.csv')

        batch_size = math.ceil(len(synergy_load_train) / 1)

        synergy_train = [[d_map[str(row[0])], d_map[str(row[1])], c_map[row[2]], float(row[3])] for index, row in
                         synergy_load_train.iterrows() if
                         (str(row[0]) in drug_data.keys() and str(row[1]) in drug_data.keys() and str(
                             row[2]) in gene_data.index)]  # len = 472131

        synergy_validation = [[d_map[str(row[0])], d_map[str(row[1])], c_map[row[2]], float(row[3])] for index, row in
                        synergy_load_validation.iterrows() if
                        (str(row[0]) in drug_data.keys() and str(row[1]) in drug_data.keys() and str(
                            row[2]) in gene_data.index)]  # len = 118033

        synergy_test = [[d_map[str(row[0])], d_map[str(row[1])], c_map[row[2]], float(row[3])] for index, row in
                        synergy_load_test.iterrows() if
                        (str(row[0]) in drug_data.keys() and str(row[1]) in drug_data.keys() and str(
                            row[2]) in gene_data.index)]  # len = 118033

        drug_smiles_fea = torch.tensor(drug_smiles_fea, dtype=torch.float32).to(device)
        cline_feature = torch.from_numpy(cline_feature).to(device)  # torch.Size([136, 954])

        drug_smiles_fea_norm = F.normalize(drug_smiles_fea, p=2, dim=1)
        drug_sim_mat = torch.mm(drug_smiles_fea_norm, drug_smiles_fea_norm.T)

        cline_fea_norm = F.normalize(cline_feature, p=2, dim=1)
        cline_sim_mat = torch.mm(cline_fea_norm, cline_fea_norm.T)

        drug_set = Data.DataLoader(dataset=GraphDataset(graphs_dict=drug_feature), collate_fn=collate,
                                   batch_size=len(drug_feature), shuffle=False)  # GraphDataset(4092)
        cline_set = Data.DataLoader(dataset=Data.TensorDataset(cline_feature),
                                    batch_size=len(cline_feature), shuffle=False)  # TensorDataset: 136

        for batch, (drug, cline) in enumerate(zip(drug_set, cline_set)):
            drug_set = drug
            cline_set = cline[0]

        synergy_train = np.array(synergy_train, dtype='int')  # (472131, 4)
        synergy_validation = np.array(synergy_validation, dtype='int')  # (118033, 4)
        synergy_test = np.array(synergy_test, dtype='int')  # (118033, 4)
        np.random.shuffle(synergy_train)
        np.random.shuffle(synergy_test)

        np.savetxt(path + 'test_y_true.txt', synergy_test[:, 3])
        label_test = torch.from_numpy(np.array(synergy_test[:, 3], dtype='float32')).to(device)
        index_test = torch.from_numpy(synergy_test).to(device)

        np.savetxt(path + 'val_y_true.txt', synergy_validation[:, 3])
        label_train = torch.from_numpy(np.array(synergy_train[:, 3], dtype='float32')).to(device)
        label_validation = torch.from_numpy(np.array(synergy_validation[:, 3], dtype='float32')).to(device)
        index_train = torch.from_numpy(synergy_train).to(device)  # torch.Size([472131, 4])
        index_validation = torch.from_numpy(synergy_validation).to(device)  # torch.Size([118033, 4])

        # -----construct hyper_synergy_graph_set
        edge_data = synergy_train[synergy_train[:, 3] == 1, 0:3]  # (148627, 3)
        synergy_edge = edge_data.reshape(1, -1)  # (1, 445881)
        index_num = np.expand_dims(np.arange(len(edge_data)), axis=-1)  # (148627, 1)
        synergy_num = np.concatenate((index_num, index_num, index_num), axis=1)  # (148627, 3)
        synergy_num = np.array(synergy_num).reshape(1, -1)  # (1, 445881)
        synergy_graph = np.concatenate((synergy_edge, synergy_num), axis=0)  # (2, 445881)
        synergy_graph = torch.from_numpy(synergy_graph).type(torch.LongTensor).to(device)

        # ---model_build
        model = HIGSyn(DrugGNN(dim_drug=78), CellMLP(dim_cellline=954), SSI(), HgnnEncoder()).to(device)

        pos_weight = torch.tensor([1.0], device=device)
        loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # ---run
        scaler = torch.cuda.amp.GradScaler()
        best_metric = [0, 0, 0, 0, 0, 0, 0, 0]
        best_epoch = 0
        no_improvement_count = 0  # To track how many epochs since the best AUC
        best_auc = 0

        for epoch in range(epochs):
            model.train()
            train_metric, train_loss = train_with_epoch_accumulation(drug_set, cline_set, synergy_graph, index_train,
                                                                     label_train, batch_size, epoch + 1, epochs)
            val_metric, val_loss, _ = test(drug_set, cline_set, synergy_graph, index_validation, label_validation,
                                           batch_size)
            print('Epoch: {:05d},'.format(epoch), 'loss_val: {:.6f},'.format(val_loss),
                  'AUC: {:.6f},'.format(val_metric[0]), 'AUPR: {:.6f},'.format(val_metric[1]),
                  'F1: {:.6f},'.format(val_metric[2]), 'ACC: {:.6f},'.format(val_metric[3]),
                  'Recall: {:.6f},'.format(val_metric[4]), 'Precision: {:.6f},'.format(val_metric[5]),
                  'Kappa: {:.6f},'.format(val_metric[6]), 'BACC: {:.6f},'.format(val_metric[7])
                  )

            torch.save(model.state_dict(), path + f'{epoch}.pth')

            # Save training loss
            with open(path + 'train_loss.txt', 'a') as train_loss_file:
                train_loss_file.write(f'{train_loss}\n')

            # Save validation AUC to a separate file
            with open(path + 'val_auc.txt', 'a') as val_auc_file:
                val_auc_file.write(f'{val_metric[0]}\n')  # Save AUC

            # Save validation ACC to a separate file
            with open(path + 'val_acc.txt', 'a') as val_acc_file:
                val_acc_file.write(f'{val_metric[3]}\n')  # Save ACC

            if val_metric[0] > best_metric[0]:
                best_metric = val_metric
                best_epoch = epoch
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            files = glob.glob(path + '*.pth')
            for f in files:
                epoch_nb = int(os.path.basename(f).split('.')[0])  # 获取文件的 epoch 编号
                if epoch_nb != best_epoch:  # 删除非最优模型文件
                    os.remove(f)
            # Early stopping if no improvement for 200 epochs
            if no_improvement_count >= 200:
                print(f"Early stopping triggered at epoch {epoch + 1} due to no improvement in AUC.")
                break
        files = glob.glob(path + '*.pth')
        for f in files:
            epoch_nb = int(os.path.basename(f).split('.')[0])
            if epoch_nb != best_epoch:
                os.remove(f)
        print('The best results on validation set, Epoch: {:05d},'.format(best_epoch),
              'AUC: {:.6f},'.format(best_metric[0]), 'AUPR: {:.6f},'.format(best_metric[1]),
              'F1: {:.6f},'.format(best_metric[2]), 'ACC: {:.6f},'.format(best_metric[3]),
              'Recall: {:.6f},'.format(best_metric[4]), 'Precision: {:.6f},'.format(best_metric[5]),
              'Kappa: {:.6f},'.format(best_metric[6]), 'BACC: {:.6f},'.format(best_metric[7])
              )


        model.load_state_dict(torch.load(path + f'{best_epoch}.pth'))
        val_metric, _, y_val_pred = test(drug_set, cline_set, synergy_graph, index_validation, label_validation,
                                         batch_size)
        test_metric, _, y_test_pred = test(drug_set, cline_set, synergy_graph, index_test, label_test, batch_size)
        np.savetxt(path + 'val_' + str(fold) + '_pred.txt', y_val_pred)
        np.savetxt(path + 'test_' + str(fold) + '_pred.txt', y_test_pred)
        file.write('val_metric:')
        for item in val_metric:
            file.write(str(item) + '\t')
        file.write('\ntest_metric:')
        for item in test_metric:
            file.write(str(item) + '\t')
        file.write('\n')

        results_final = pd.DataFrame(
            {'Accuracy': [test_metric[3]], 'AUC': [test_metric[0]], 'Precision': [test_metric[5]],
             'Recall': [test_metric[4]], 'F1': [test_metric[2]],
             'PR AUC': [test_metric[1]], 'Kappa': [test_metric[6]], 'Balanced Accuracy (BACC)': [test_metric[7]]})

        results_final.to_csv(path + 'Evaluation_results_fold_' + str(fold) + '.csv', index=False)

