# 定义模型
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model_type, dropout_rate):
        super(GNN, self).__init__()
        self.dropout_rate = dropout_rate

        if model_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif model_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels)
            self.conv2 = GATConv(hidden_channels, out_channels)
        elif model_type == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        elif model_type == 'ChebNet':
            self.conv1 = ChebConv(in_channels, hidden_channels, K=2)
            self.conv2 = ChebConv(hidden_channels, out_channels, K=2)
        elif model_type == 'TransformerConv':
            self.conv1 = TransformerConv(in_channels, hidden_channels)
            self.conv2 = TransformerConv(hidden_channels, out_channels)
        else:
            raise NotImplementedError

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = torch.nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x