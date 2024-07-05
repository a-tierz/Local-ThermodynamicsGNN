"""model.py"""
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.utils import add_self_loops

# Multi Layer Perceptron (MLP) class
class MLP(torch.nn.Module):
    def __init__(self, layer_vec):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for k in range(len(layer_vec) - 1):
            layer = nn.Linear(layer_vec[k], layer_vec[k + 1])
            self.layers.append(layer)
            self.layers.append(nn.SiLU()) if k != len(layer_vec) - 2 else None

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Edge model
class EdgeModel(torch.nn.Module):
    def __init__(self, n_hidden, dim_hidden):
        super(EdgeModel, self).__init__()
        self.n_hidden = n_hidden
        self.dim_hidden = dim_hidden
        self.edge_mlp = MLP([3 * self.dim_hidden] + self.n_hidden * [self.dim_hidden] + [self.dim_hidden])

    def forward(self, src, dest, edge_attr):
        out = torch.cat([edge_attr, src, dest], dim=1)
        out = self.edge_mlp(out)
        return out


# Node model
class NodeModel(torch.nn.Module):
    def __init__(self, n_hidden, dim_hidden, dims):
        super(NodeModel, self).__init__()
        self.n_hidden = n_hidden
        self.dim_hidden = dim_hidden
        self.node_mlp = MLP(
            [2 * self.dim_hidden + dims['f']] + self.n_hidden * [self.dim_hidden] + [self.dim_hidden])

    def forward(self, x, dest, edge_attr, f=None):
        out = scatter_mean(edge_attr, dest, dim=0, dim_size=x.size(0))
        if f is not None:
            out = torch.cat([x, out, f], dim=1)
        else:
            out = torch.cat([x, out], dim=1)
        out = self.node_mlp(out)
        return out


# Modification of the original MetaLayer class
class MetaLayer(torch.nn.Module):
    def __init__(self, edge_model=None, node_model=None):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model

    def forward(self, x, edge_index, edge_attr, f=None):
        src, dest = edge_index
        edge_attr = self.edge_model(x[src], x[dest], edge_attr)
        x = self.node_model(x, dest, edge_attr, f)
        return x, edge_attr


class NodalGNN(pl.LightningModule):
    def __init__(self, dims, scaler_pu, scaler_s, dt_info, save_folder):
        super().__init__()
        n_hidden = dt_info['model']['n_hidden']
        dim_hidden = dt_info['model']['dim_hidden']
        self.project_name = dt_info['project_name']
        self.passes = dt_info['model']['passes']
        self.batch_size = dt_info['model']['batch_size']
        self.dtset_type = dt_info['dataset']['type']
        self.radius_connectivity = dt_info['dataset']['radius_connectivity']
        self.save_folder = save_folder
        self.data_dim = dt_info['dataset']['q_dim']
        self.dims = dims
        self.dim_z = self.dims['z']
        self.dim_q = self.dims['q']

        dim_node = self.dims['z'] - self.dims['q']
        dim_edge = self.dims['q'] + 1 #+ self.dims['q_0']
        self.state_variables = dt_info['dataset']['state_variables_out']

        # Encoder MLPs
        self.encoder_node = MLP([dim_node] + n_hidden * [dim_hidden] + [dim_hidden])
        self.encoder_edge = MLP([dim_edge] + n_hidden * [dim_hidden] + [dim_hidden])

        node_model = NodeModel(n_hidden, dim_hidden, self.dims)
        edge_model = EdgeModel(n_hidden, dim_hidden)
        self.GraphNet = \
            MetaLayer(node_model=node_model, edge_model=edge_model)

        self.decoder = MLP([dim_hidden] + n_hidden * [dim_hidden] + [self.dims['y']])


        self.scaler_pu = scaler_pu
        self.scaler_s = scaler_s
        self.dt = dt_info['dataset']['dt']
        self.noise_var = dt_info['model']['noise_var']
        self.lambda_d = dt_info['model']['lambda_d']
        self.lr = dt_info['model']['lr']
        self.miles = dt_info['model']['miles']
        self.gamma = dt_info['model']['gamma']
        self.criterion = torch.nn.functional.mse_loss

        # Rollout simulation
        self.rollout_freq = dt_info['model']['rollout_freq']
        self.error_message_pass = []


    def pass_thought_net(self, z_t0, y_gt, edge_index, q0, batch=None, mode='val'):
        self.batch_size = torch.max(batch) + 1
        z_norm = torch.from_numpy(self.scaler_pu.transform(z_t0.cpu())).float().to(self.device)
        s_gt = torch.from_numpy(self.scaler_s.transform(y_gt.cpu())).float().to(self.device)

        if mode == 'train':
            noise = self.noise_var * torch.randn_like(z_norm)
            z_norm = z_norm + noise

        q = z_norm[:, :self.dim_q]
        x = z_norm[:, self.dim_q:]


        # Edge attributes
        src, dest = edge_index
        u = q[src] - q[dest]
        u_norm = torch.norm(u, dim=1).reshape(-1, 1)
        edge_attr = torch.cat((u, u_norm), dim=1)

        '''Encode'''
        x = self.encoder_node(x)
        edge_attr = self.encoder_edge(edge_attr)

        '''Process'''
        for i in range(self.passes):
            x_res, edge_attr_res = self.GraphNet(x, edge_index, edge_attr)
            x += x_res
            edge_attr += edge_attr_res

        '''Decoder'''
        s_net = self.decoder(x)  #self.decoder_E(x).unsqueeze(-1)

        loss = self.criterion(s_net, s_gt)

        if mode != 'eval':
            self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

            if self.state_variables is not None:
                for i, variable in enumerate(self.state_variables):
                    loss_variable = self.criterion(s_net.reshape(s_gt.shape)[:, i], s_gt[:, i])
                    self.log(f"{mode}_loss_{variable}", loss_variable, prog_bar=True, on_step=False, on_epoch=True)
        torch.cuda.empty_cache()
        return s_net, loss

    def extrac_pass(self, batch, mode):
        # Extract data from DataGeometric
        if self.project_name == 'Beam_3D_s':
            z_t0, z_t1, edge_index, q0 = batch.x, batch.y, batch.edge_index, batch.q0
        elif self.project_name == 'Beam_2D':
            z_t0, z_t1, edge_index, n, f = batch.x, batch.y, batch.edge_index, batch.n, batch.f
        s_net, loss = self.pass_thought_net(z_t0, z_t1, edge_index,q0=q0,
                                               batch=batch.batch, mode=mode)
        return s_net, loss

    def training_step(self, batch, batch_idx, g=None):

        dzdt_net, loss = self.extrac_pass(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx, g=None):

        self.extrac_pass(batch, 'val')


    def predict_step(self, batch, batch_idx, g=None):

        s_net, loss = self.extrac_pass(batch, 'eval')
        s_net_denorm = torch.from_numpy(self.scaler_s.inverse_transform(s_net.detach().to('cpu'))).float().to(
            self.device)

        return s_net_denorm, batch.y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.miles, gamma=self.gamma),
            'monitor': 'val_loss'}

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
