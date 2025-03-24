import torch
import torch.nn as nn
from torch_geometric.nn import HypergraphConv, GCNConv, GINConv, global_max_pool, global_mean_pool, SAGPooling, JumpingKnowledge, global_add_pool
import sys
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

sys.path.append('..')

drug_num = 35
dropout_rate = 0.1
gnn_dim = 128
num_heads = 4

def contrastive_loss(pos_embeddings, neg_embeddings, margin=1.0):
    """
    对比损失函数
    pos_embeddings: 正样本嵌入表示，形状为 [batch_size, embedding_dim]
    neg_embeddings: 负样本嵌入表示，形状为 [batch_size, embedding_dim]
    margin: 对比损失的边界值
    """
    batch_size = pos_embeddings.size(0)

    # 随机采样负样本，确保数量与正样本一致
    neg_indices = torch.randint(0, neg_embeddings.size(0), (batch_size,), device=neg_embeddings.device)
    sampled_neg_embeddings = neg_embeddings[neg_indices]

    # 计算正样本和采样负样本的 L2 范数距离
    pos_distances = torch.norm(pos_embeddings, dim=1)
    neg_distances = torch.norm(sampled_neg_embeddings, dim=1)

    # 拉大正负样本距离，计算损失
    loss = torch.mean(F.relu(margin + pos_distances - neg_distances))
    return loss


class NodeLevelSAGPooling(nn.Module):
    def __init__(self, in_channels, ratio=1.0, nonlinearity='tanh'):
        super(NodeLevelSAGPooling, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio

        # GCN layer for computing node importance scores
        self.gnn = GCNConv(in_channels, 1)

        # Activation function for scores
        if isinstance(nonlinearity, str):
            self.nonlinearity = getattr(torch, nonlinearity)
        else:
            self.nonlinearity = nonlinearity

    def forward(self, x, edge_index, batch):
        # Compute node scores using GCN
        scores = self.gnn(x, edge_index).squeeze(-1)  # Shape: [num_nodes]
        scores = self.nonlinearity(scores)

        # Normalize scores using softmax (per subgraph)
        normalized_scores = torch.zeros_like(scores)
        for graph_idx in batch.unique():
            mask = batch == graph_idx
            normalized_scores[mask] = torch.softmax(scores[mask], dim=0)

        # Compute weighted features
        x_pooled = global_add_pool(x * normalized_scores.unsqueeze(-1), batch)  # Element-wise multiplication

        return x_pooled, normalized_scores


class DrugGNN(nn.Module):
    def __init__(self, dim_drug, gnn_dim=gnn_dim, dropout_rate=dropout_rate):
        super(DrugGNN, self).__init__()
        # Add normalization layer
        self.drug_normalizer = nn.BatchNorm1d(dim_drug)  # Suitable for global feature normalization

        # ------ drug_layer ------ #
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(dim_drug, gnn_dim * 2),
            nn.ReLU(),
            nn.Linear(gnn_dim * 2, gnn_dim)
        ))  # GCNConv, GraphSAGE, GAT, ChebNet, ResGCN, SAGEConv
        self.batch_conv1 = nn.BatchNorm1d(gnn_dim)  # BatchNorm1d after GINConv

        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(gnn_dim, gnn_dim * 2),
            nn.ReLU(),
            nn.Linear(gnn_dim * 2, gnn_dim)
        ))
        self.batch_conv2 = nn.BatchNorm1d(gnn_dim)

        self.conv3 = GINConv(nn.Sequential(
            nn.Linear(gnn_dim, gnn_dim * 2),
            nn.ReLU(),
            nn.Linear(gnn_dim * 2, gnn_dim)
        ))
        self.batch_conv3 = nn.BatchNorm1d(gnn_dim)

        # SAGPooling with pooling_ratio=1.0 to retain all nodes
        self.pool1 = NodeLevelSAGPooling(gnn_dim, ratio=1.0)
        self.pool2 = NodeLevelSAGPooling(gnn_dim, ratio=1.0)
        self.pool3 = NodeLevelSAGPooling(gnn_dim, ratio=1.0)

        self.jk = JumpingKnowledge(mode='cat', channels=gnn_dim, num_layers=3)  # cat/max/mean/lstm/attn
        self.pool4 = NodeLevelSAGPooling(gnn_dim * 3, ratio=1.0)

        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.ReLU()

    def forward(self, drug_feature, drug_adj, ibatch):
        drug_feature = self.drug_normalizer(drug_feature)

        # ------ Drug Training (GIN + SAGPooling) ------ #
        x_drug_local1 = self.conv1(drug_feature, drug_adj)
        x_drug_local1 = self.act(x_drug_local1)
        x_drug_local1 = self.batch_conv1(x_drug_local1)  # BatchNorm
        x_drug_local1 = self.dropout(x_drug_local1)

        x_drug_local2 = self.conv2(x_drug_local1, drug_adj)
        x_drug_local2 = self.act(x_drug_local2)
        x_drug_local2 = self.batch_conv2(x_drug_local2)  # BatchNorm
        x_drug_local2 = self.dropout(x_drug_local2)

        x_drug_local3 = self.conv3(x_drug_local2, drug_adj)
        x_drug_local3 = self.act(x_drug_local3)
        x_drug_local3 = self.batch_conv3(x_drug_local3)  # BatchNorm
        x_drug_local3 = self.dropout(x_drug_local3)

        # Node-level SAGPooling for Layer 1
        x_drug_global1, scores1 = self.pool1(x_drug_local1, drug_adj, ibatch)

        # Node-level SAGPooling for Layer 2
        x_drug_global2, scores2 = self.pool2(x_drug_local2, drug_adj, ibatch)

        # Node-level SAGPooling for Layer 3
        x_drug_global3, scores3 = self.pool3(x_drug_local3, drug_adj, ibatch)

        # Jumping Knowledge to aggregate global features
        x_drug_global = self.jk([x_drug_local1, x_drug_local2, x_drug_local3])
        x_drug_global, scores4 = self.pool4(x_drug_global, drug_adj, ibatch)

        # Concatenate features from both layers
        x_drug_local = torch.cat((x_drug_global1, x_drug_global2, x_drug_global3), dim=-1)  # Shape: [num_nodes, 2 * gnn_dim]

        return x_drug_local, x_drug_global


class CellMLP(nn.Module):
    def __init__(self, dim_cellline, output_dim=gnn_dim*3, dropout_rate=dropout_rate):
        super(CellMLP, self).__init__()
        self.cell_normalizer = nn.BatchNorm1d(dim_cellline)

        # ------ cell line_layer ------ #
        self.fc_cell1 = nn.Linear(dim_cellline, 512)  # 第一层全连接
        self.batch_cell1 = nn.BatchNorm1d(512)       # 第一层 BatchNorm

        self.fc_cell2 = nn.Linear(512, output_dim)   # 第二层全连接
        self.batch_cell2 = nn.BatchNorm1d(output_dim) # 第二层 BatchNorm

        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.ReLU()                          # 激活函数

    def forward(self, gexpr_data):
        gexpr_data = self.cell_normalizer(gexpr_data)

        # ----cellline_train
        x_cellline = self.fc_cell1(gexpr_data)
        x_cellline = self.batch_cell1(x_cellline)      # BatchNorm
        x_cellline = self.act(x_cellline)              # ReLU 激活
        x_cellline = self.dropout(x_cellline)

        x_cellline = self.fc_cell2(x_cellline)
        x_cellline = self.batch_cell2(x_cellline)      # BatchNorm
        x_cellline = self.act(x_cellline)              # ReLU 激活
        x_cellline = self.dropout(x_cellline)

        return x_cellline

class CrossCoAttention(nn.Module):
    def __init__(self, input_dim=gnn_dim * 3, heads=num_heads, dropout_rate=dropout_rate):
        super(CrossCoAttention, self).__init__()
        self.attn1 = nn.MultiheadAttention(embed_dim=input_dim, num_heads=heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim=input_dim, num_heads=heads, batch_first=True)

        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, input_dim)

        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(input_dim)  # After first cross-attention
        self.bn2 = nn.BatchNorm1d(input_dim)  # After second cross-attention
        self.bn3 = nn.BatchNorm1d(2)

        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.ReLU()

    def forward(self, A, B):
        """
        A: Drug feature matrix A (batch_size, seq_len, input_dim)
        B: Drug or Cell line feature matrix B (batch_size, seq_len, input_dim)
        """
        A = A.unsqueeze(1)
        B = B.unsqueeze(1)

        # Cross-attention (A to B)
        attn_output1, attn_weights1 = self.attn1(A, B, B)
        attn_output1 = attn_output1 + A  # Add residual connection
        attn_output1 = self.bn1(attn_output1.squeeze(1))  # Apply BatchNorm
        attn_output1 = self.linear1(attn_output1)
        attn_output1 = self.act(attn_output1)
        attn_output1 = self.dropout(attn_output1)  # Apply Dropout
        attn_output1 = attn_output1.unsqueeze(1)

        # Cross-attention (B to A)
        attn_output2, attn_weights2 = self.attn2(B, A, A)
        attn_output2 = attn_output2 + B  # Add residual connection
        attn_output2 = self.bn2(attn_output2.squeeze(1))  # Apply BatchNorm
        attn_output2 = self.linear2(attn_output2)
        attn_output2 = self.act(attn_output2)
        attn_output2 = self.dropout(attn_output2)  # Apply Dropout
        attn_output2 = attn_output2.unsqueeze(1)

        # Concatenate the two outputs
        combined = torch.cat([attn_output1, attn_output2], dim=1)
        combined = self.bn3(combined)  # Apply BatchNorm
        combined = self.dropout(combined)  # Apply Dropout

        return combined


class SSI(nn.Module):
    def __init__(self, input_dim=gnn_dim, heads=num_heads, dropout_rate=dropout_rate):
        super(SSI, self).__init__()
        self.cross_co_attention_d1d2 = CrossCoAttention(input_dim=input_dim * 3, heads=heads)
        self.cross_co_attention_cd1 = CrossCoAttention(input_dim=input_dim * 3, heads=heads)
        self.cross_co_attention_cd2 = CrossCoAttention(input_dim=input_dim * 3, heads=heads)

        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim * 3, num_heads=heads, batch_first=True)

        # Linear layers with BN and activation for dimensionality reduction
        self.linear1 = nn.Linear(input_dim * 3, input_dim * 3)
        self.bn1 = nn.BatchNorm1d(6)
        self.linear2 = nn.Linear(input_dim * 18, input_dim * 9)
        self.bn2 = nn.BatchNorm1d(input_dim * 18)

        # Activation and Dropout
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_drug1_local, x_drug2_local, x_cellline):
        """
        x_drug_local1, x_drug_local2: Feature matrices for drug1 and drug2 (batch_size, seq_len, input_dim)
        x_cellline: Cell-line feature matrix (batch_size, seq_len, input_dim)
        """
        # Step 1: Compute attention between drug1 and drug2 (cross-co-attention)
        d1d2 = self.cross_co_attention_d1d2(x_drug1_local, x_drug2_local)

        # Step 2: Compute attention between drug1 and cell-line (cross-co-attention)
        cd1 = self.cross_co_attention_cd1(x_drug1_local, x_cellline)

        # Step 3: Compute attention between drug2 and cell-line (cross-co-attention)
        cd2 = self.cross_co_attention_cd2(x_drug2_local, x_cellline)

        ssi = torch.cat((d1d2, cd1, cd2), dim=1)

        # Apply self-attention on the concatenated features
        self_attn_output, self_attn_weights = self.self_attn(ssi, ssi, ssi)

        # Residual connection after self-attention
        ssi = self_attn_output + ssi  # Add residual connection
        ssi = self.bn1(ssi)  # Apply BatchNorm
        ssi = self.linear1(ssi)
        ssi = self.act(ssi)
        ssi = self.dropout(ssi)  # Apply Dropout

        ssi = ssi.flatten(1)
        ssi = self.bn2(ssi)
        ssi = self.linear2(ssi)
        ssi = self.act(ssi)
        ssi = self.dropout(ssi)

        # Return the outputs
        return ssi


class HgnnEncoder(torch.nn.Module):
    def __init__(self, input_dim=gnn_dim, dropout_rate=dropout_rate):
        super(HgnnEncoder, self).__init__()
        self.conv1 = HypergraphConv(input_dim*3, 6*input_dim)
        self.batch1 = nn.BatchNorm1d(6*input_dim)

        self.conv2 = HypergraphConv(6*input_dim, 6*input_dim)
        self.batch2 = nn.BatchNorm1d(6*input_dim)

        self.conv3 = HypergraphConv(6*input_dim, 3*input_dim)
        self.batch3 = nn.BatchNorm1d(3*input_dim)

        self.conv4 = HypergraphConv(3*input_dim, 3*input_dim)
        self.batch4 = nn.BatchNorm1d(3*input_dim)  # Optional: Add BN for the final layer

        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.ReLU()

    def forward(self, x, edge):
        x_original = x

        # First HypergraphConv layer
        x = self.conv1(x, edge)
        x = self.batch1(x)  # Apply BatchNorm
        x = self.act(x)
        x = self.dropout(x)

        # Second HypergraphConv layer
        x = self.conv2(x, edge)
        x = self.batch2(x)  # Apply BatchNorm
        x = self.act(x)
        x = self.dropout(x)

        # Third HypergraphConv layer
        x = self.conv3(x, edge)
        x = self.batch3(x)  # Apply BatchNorm
        x = self.act(x)
        x = self.dropout(x)

        # Forth HypergraphConv layer
        x = self.conv4(x+x_original, edge)
        x = self.batch4(x)  # Apply BatchNorm
        x = self.act(x)
        x = self.dropout(x)

        return x


class Highway(nn.Module):
    def __init__(self, input_dim=gnn_dim * 18, num_layers=1, dropout_rate=dropout_rate):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.linears = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(input_dim) for _ in range(num_layers)])  # BatchNorm for each layer
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_layers)])  # Dropout for each layer

        # Gate bias initialization to encourage skipping early in training
        for gate in self.gates:
            nn.init.constant_(gate.bias, -1.0)

    def forward(self, x):
        for i in range(self.num_layers):
            # Compute gate and transform
            gate = torch.sigmoid(self.gates[i](x))  # Gate: controls how much to keep
            transform = F.relu(self.linears[i](x))  # Transform: apply non-linearity
            transform = self.bns[i](transform)  # Apply BatchNorm first
            transform = self.dropouts[i](transform)  # Apply Dropout next

            # Highway connection
            x = gate * transform + (1 - gate) * x  # Residual connection controlled by gate
        return x


class HIGSyn(torch.nn.Module):
    def __init__(self, DrugGNN, CellMLP, SSI, HgnnEncoder, output_dim=gnn_dim, dropout_rate=0.3):
        super(HyperSubNet, self).__init__()
        self.DrugGNN = DrugGNN
        self.CellMLP = CellMLP
        self.SSI = SSI
        self.HgnnEncoder = HgnnEncoder
        self.Highway = Highway(input_dim=output_dim * 18)

        # 定义两层 MLP
        self.bn0 = nn.BatchNorm1d(output_dim * 18)

        self.fc_fusion = nn.Linear(output_dim * 18, output_dim * 18)
        self.bn_fusion = nn.BatchNorm1d(output_dim * 18)

        self.fc1 = nn.Linear(output_dim * 18, output_dim * 8)
        self.bn1 = nn.BatchNorm1d(output_dim * 8)

        self.fc2 = nn.Linear(output_dim * 8, output_dim * 2)
        self.bn2 = nn.BatchNorm1d(output_dim * 2)

        # 输出层
        self.fc_out = nn.Linear(output_dim * 2, 1)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout for final output
        self.act = nn.ReLU()


    def forward(self, drug_feature, drug_adj, ibatch, gexpr_data, adj, druga_id, drugb_id, cellline_id, neg_ids, pos_ids):
        # Drug and Cell line embeddings
        drug_local_set, drug_embed_set = self.DrugGNN(drug_feature, drug_adj, ibatch)
        # print(f'drug_local_set: {drug_local_set.shape}')
        # print(f'drug_embed_set: {drug_embed_set.shape}')
        cellline_embed_set = self.CellMLP(gexpr_data)
        # print(f'cellline_embed_set: {cellline_embed_set.shape}')

        # Hypergraph encoding
        merge_embed = torch.cat((drug_embed_set, cellline_embed_set), 0)
        hypergraph_embed = self.HgnnEncoder(merge_embed, adj)

        drug_emb_hyper, cline_emb_hyper = hypergraph_embed[:drug_num], hypergraph_embed[drug_num:]

        d1d2c = torch.cat(
            (drug_emb_hyper[druga_id, :], drug_emb_hyper[drugb_id, :], cline_emb_hyper[(cellline_id - drug_num), :]), 1
        )

        # SSI embedding
        d1_local = drug_local_set[druga_id, :]
        d2_local = drug_local_set[drugb_id, :]
        cell_local = cellline_embed_set[(cellline_id - drug_num), :]
        #  Call SSI with adjusted inputs
        ssi = self.SSI(d1_local, d2_local, cell_local)
        # print(f'd1d2: {d1d2.shape}')

        # Concatenate embeddings
        concat_embed = torch.cat((d1d2c, ssi), dim=-1)
        # print(f'concat_embed: {concat_embed.shape}')
        concat_embed = self.bn0(concat_embed)
        concat_embed = self.dropout(self.act(self.bn_fusion(self.fc_fusion(concat_embed))))
        concat_embed = self.Highway(concat_embed)

        # Two-layer MLP
        x = self.dropout(self.act(self.bn1(self.fc1(concat_embed))))
        x = self.dropout(self.act(self.bn2(self.fc2(x))))

        # Final output layer with additional dropout
        x = self.dropout(x)
        pred = self.fc_out(x)

        # Squeeze the last dimension
        pred = pred.squeeze(1)

        drug_norm = F.normalize(drug_emb_hyper, p=2, dim=1)
        sim_drug = torch.mm(drug_norm, drug_norm.T)
        cline_norm = F.normalize(cline_emb_hyper, p=2, dim=1)
        sim_cline = torch.mm(cline_norm, cline_norm.T)

        # 正样本嵌入
        pos_emb = torch.cat(
            (drug_emb_hyper[pos_ids[:, 0], :], drug_emb_hyper[pos_ids[:, 1], :], cline_emb_hyper[(pos_ids[:, 2] - drug_num), :]),
            dim=-1
        )

        # 负样本嵌入
        neg_emb = torch.cat(
            (drug_emb_hyper[neg_ids[:, 0], :], drug_emb_hyper[neg_ids[:, 1], :], cline_emb_hyper[(neg_ids[:, 2] - drug_num), :]),
            dim=-1
        )

        # 计算对比损失
        contrastive_loss_value = contrastive_loss(pos_emb, neg_emb)

        return pred, sim_drug, sim_cline, contrastive_loss_value
