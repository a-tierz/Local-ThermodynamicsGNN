"""model.py"""
import torch
import torch.nn as nn
import numpy as np
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
        if dims['f'] == 0:
            self.node_mlp = MLP(
                [2 * self.dim_hidden + dims['f']] + self.n_hidden * [self.dim_hidden] + [self.dim_hidden])
        else:
            self.node_mlp = MLP(
                [2 * self.dim_hidden + int(1 / 2 * self.dim_hidden)] + self.n_hidden * [self.dim_hidden] + [
                    self.dim_hidden])

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
    def __init__(self, dims, scaler, dt_info, save_folder):
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
        dim_node = self.dims['z'] + self.dims['n'] - self.dims['q']
        dim_edge = self.dims['q'] + self.dims['q_0'] + 1
        dim_f = self.dims['f']
        self.state_variables = dt_info['dataset']['state_variables']

        # Encoder MLPs
        self.encoder_node = MLP([dim_node] + n_hidden * [dim_hidden] + [dim_hidden])
        self.encoder_edge = MLP([dim_edge] + n_hidden * [dim_hidden] + [dim_hidden])
        if self.dims['f'] >0:
            self.encoder_f = MLP([dim_f] + n_hidden * [dim_hidden] + [int(dim_hidden/2)])

        node_model = NodeModel(n_hidden, dim_hidden, self.dims)
        edge_model = EdgeModel(n_hidden, dim_hidden)
        self.GraphNet = \
            MetaLayer(node_model=node_model, edge_model=edge_model)

        self.decoder_E = MLP([dim_hidden] + n_hidden * [dim_hidden] + [self.dim_z])
        self.decoder_S = MLP([dim_hidden] + n_hidden * [dim_hidden] + [self.dim_z])

        self.decoder_L = MLP([dim_hidden * 3] + n_hidden * [dim_hidden] * 2 + [
            int(self.dim_z * (self.dim_z + 1) / 2 - self.dim_z)])
        self.decoder_M = MLP(
            [dim_hidden * 3] + n_hidden * [dim_hidden] * 2 + [int(self.dim_z * (self.dim_z + 1) / 2)])

        self.ones = torch.ones(self.dim_z, self.dim_z)
        self.scaler, self.scaler_f  = scaler
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
        self.save_hyperparameters()

    def integrator(self, L, M, dEdz, dSdz):
        # GENERIC time integration and degeneration
        dzdt = torch.matmul(L, dEdz) + torch.matmul(M, dSdz) #+ f
        # dzdt = torch.sparse.mm(L, dEdz) + torch.sparse.mm(M, dSdz)
        deg_E = torch.matmul(M, dEdz)
        deg_S = torch.matmul(L, dSdz)
        return dzdt, deg_E, deg_S
        # return dzdt[:,:,0], deg_E[:,:,0], deg_S[:,:,0]


    def decoder(self, x, edge_attr, batch, src, dest):


        # Gradients
        dEdz = self.decoder_E(x).unsqueeze(-1)
        dSdz = self.decoder_S(x).unsqueeze(-1)

        l = self.decoder_L(torch.cat([edge_attr, x[src], x[dest]], dim=1))
        m = self.decoder_M(torch.cat([edge_attr, x[src], x[dest]], dim=1))

        loss_deg_E = 0
        loss_deg_S = 0
        cnt_n_node = 0
        cnt_edge_node = 0
        dzdt_net_b = []
        for batch_i in range(self.batch_size):
            '''Select the info of one simulation'''
            x_batch = x[batch == batch_i]
            x_batch_size = x_batch.size(0)
            # f_dec_batch = f_dec[batch == batch_i]
            # n_edges_batch = torch.repeat_interleave(n_edges[batch ==batch_i], self.dim_z)

            cnt_n_node += x_batch_size
            ini_n = cnt_n_node - x_batch_size

            src_batch = src[(src >= ini_n) & (src < cnt_n_node)]
            dest_batch = dest[(dest >= ini_n) & (dest < cnt_n_node)]
            src_batch = torch.subtract(src_batch, torch.min(src_batch))
            dest_batch = torch.subtract(dest_batch, torch.min(dest_batch))

            edge_batch_size = src_batch.size(0)
            cnt_edge_node += edge_batch_size
            ini_n = cnt_edge_node - edge_batch_size
            l_batch = l[ini_n:cnt_edge_node, :]
            m_batch = m[ini_n:cnt_edge_node, :]

            ############################
            '''Reparametrization'''
            L = torch.zeros(edge_batch_size, self.dim_z, self.dim_z, device=l.device, dtype=l.dtype)
            M = torch.zeros(edge_batch_size, self.dim_z, self.dim_z, device=m.device, dtype=m.dtype)

            L[:, torch.tril(self.ones, -1) == 1] = l_batch
            M[:, torch.tril(self.ones) == 1] = m_batch#*100

            Ledges = torch.subtract(L, torch.transpose(L, 1, 2))
            Ledges[src_batch<dest_batch, :, :] = - torch.transpose(Ledges[src_batch < dest_batch, :, :], 1, 2)

            Medges = M + torch.transpose(M, 1, 2)
            # self.ones = torch.ones(self.N_dim, self.N_dim)
            # M symmetric and positive semi-definite
            # Medges = torch.mul(M[src_batch, :, :], M[dest_batch, :, :])
            # Medges = M[src_batch, :, :] * M[dest_batch, :, :] * m_edge_batch
            # Medges = torch.bmm(Medges, torch.transpose(Medges, 1, 2))

            ############################
            N_dim = x_batch_size * self.dim_z

            ############################
            idx_row = []
            idx_col = []
            for i in range(self.dim_z):
                for j in range(self.dim_z):
                    if idx_col == []:
                        idx_row = (src_batch * self.dim_z)
                        idx_col = (dest_batch * self.dim_z)
                    else:
                        idx_row = torch.cat([idx_row, (src_batch * self.dim_z) + i], dim=0)
                        idx_col = torch.cat([idx_col, (dest_batch * self.dim_z) + j], dim=0)

            # inicio = time.time()
            # L_big = torch.zeros(N_dim, N_dim, device=l.device, dtype=l.dtype)
            # M_big = torch.zeros(N_dim, N_dim, device=m.device, dtype=m.dtype)
            # L_big[idx_row, idx_col] = torch.transpose(torch.transpose(Ledges, 1, 0), 2, 1).reshape(-1)
            # M_big[idx_row, idx_col] = torch.transpose(torch.transpose(Medges, 1, 0), 2, 1).reshape(-1)
            # print(f"Tiempo de ejecución de 0: {time.time() - inicio} segundos")

            i = [idx_row.tolist(), idx_col.tolist()]
            Ledges = torch.transpose(torch.transpose(Ledges, 1, 0), 2, 1).reshape(-1)
            Medges = torch.transpose(torch.transpose(Medges, 1, 0), 2, 1).reshape(-1)
            L_big = torch.sparse_coo_tensor(i, Ledges, (N_dim, N_dim))
            M_big = torch.sparse_coo_tensor(i, Medges, (N_dim, N_dim))
            #
            # M_big = torch.sparse.mm(M_big, M_big.T)

            ############################
            # for i in range(self.dim_z):
            # for i in range(self.dim_z):
            #     for j in range(self.dim_z):
            #         L_big[(src_batch * self.dim_z) + i, (dest_batch * self.dim_z) + j] = Ledges[:, i, j]
            #         M_big[(src_batch * self.dim_z) + i, (dest_batch * self.dim_z) + j] = Medges[:, i, j]
            #
            # M_big = torch.matmul(M_big, torch.transpose(M_big, 1, 0))/torch.max(M_big) #forzamos que la M sea SDP
            # L_big = torch.matmul(L_big, torch.transpose(L_big, 1, 0)) #forzamos que la M sea SDP
            # torch.allclose(-L_big, L_big.T)


            ########################
            dzdt_net, deg_E, deg_S = self.integrator(L_big, M_big,
                                                     dEdz[batch == batch_i, :].reshape(-1, 1),
                                                     dSdz[batch == batch_i, :].reshape(-1, 1))
                                                     # f_dec_batch.reshape(-1, 1))
            # del M_big, L_big, M, L, dEdz, dSdz
            # torch.cuda.empty_cache()
            dzdt_net = dzdt_net[:, 0] #/ n_edges_batch
            dzdt_net_b.append(dzdt_net.reshape((x_batch_size, self.dim_z)))
            # loss_deg_E += torch.norm(deg_E)
            loss_deg_E += (deg_E ** 2).mean()
            # loss_deg_S += torch.norm(deg_S)
            loss_deg_S += (deg_S ** 2).mean()

        dzdt_net = torch.cat(dzdt_net_b, dim=0)

        return dzdt_net, loss_deg_E, loss_deg_S


    def pass_thought_net(self, z_t0, z_t1, edge_index, n, f, g=None, batch=None, mode='val', plot_info = []):
        self.batch_size = torch.max(batch) + 1
        z_norm = torch.from_numpy(self.scaler.transform(z_t0.cpu())).float().to(self.device)
        z1_norm = torch.from_numpy(self.scaler.transform(z_t1.cpu())).float().to(self.device)
        if f is not None:
            f = torch.from_numpy(self.scaler_f.transform(f.cpu())).float().to(self.device)

        if mode == 'train':
            noise = self.noise_var * torch.randn_like(z_norm[n == 1])
            z_norm[n == 1] = z_norm[n == 1] + noise*z_norm[n == 1]
            noise = self.noise_var * torch.randn_like(z_norm[n == 2])
            z_norm[n == 2] = z_norm[n == 2] + noise*z_norm[n == 2]

        q = z_norm[:, :self.dim_q]
        v = z_norm[:, self.dim_q:]
        x = torch.cat((v, torch.reshape(n.type(torch.float32), (len(n), 1))), dim=1)

        # Edge attributes
        src, dest = edge_index
        u = q[src] - q[dest]
        u_norm = torch.norm(u, dim=1).reshape(-1, 1)
        edge_attr = torch.cat((u, u_norm), dim=1)

        '''Encode'''
        x = self.encoder_node(x)
        edge_attr = self.encoder_edge(edge_attr)
        if f is not None:
            f = self.encoder_f(f)

        '''Process'''
        for i in range(self.passes):
            if mode == 'eval':
                plot_info.append(torch.norm(x, dim=1).reshape(-1, 1).clone())
            x_res, edge_attr_res = self.GraphNet(x, edge_index, edge_attr, f=f)
            # if f is not None:
            #     f = f*0
            x += x_res
            edge_attr += edge_attr_res

        '''Decoder'''
        if self.project_name == 'Glass3D':
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
            n_glass = x[n == 0].shape[0]
            mask_fluid = ((edge_index >= n_glass)[0, :]) & ((edge_index >= n_glass)[1, :])
            edge_index = edge_index[:, mask_fluid]
            edge_index = edge_index - torch.min(edge_index)
            edge_attr = edge_attr[mask_fluid, :]
            x = x[n == 1]
            batch = batch[n == 1]
        else:
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr)

        dzdt_net, loss_deg_E, loss_deg_S = self.decoder(x, edge_attr, batch,  edge_index[0, :], edge_index[1, :])

        dzdt = (z1_norm - z_norm) / self.dt

        if self.project_name == 'Glass3D':
            # Cojemos las particulas del glass de gt y no las predecimos
            dzdt_net_b = dzdt.clone()
            dzdt_net_b[n == 1] = dzdt_net
            dzdt = dzdt[n == 1]
        else:
            dzdt_net_b = dzdt_net.reshape(dzdt.shape)

        loss_z = self.criterion(dzdt_net, dzdt)

        loss = self.lambda_d * loss_z + (loss_deg_E + loss_deg_S)

        if mode != 'eval':
            self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            if mode == 'val':
                self.log(f"{mode}_deg_E", loss_deg_E, prog_bar=False, on_step=False, on_epoch=True)
                self.log(f"{mode}_deg_S", loss_deg_S, prog_bar=False, on_step=False, on_epoch=True)

            if self.state_variables is not None:
                for i, variable in enumerate(self.state_variables):
                    loss_variable = self.criterion(dzdt_net.reshape(dzdt.shape)[:, i], dzdt[:, i])
                    self.log(f"{mode}_loss_{variable}", loss_variable, prog_bar=True, on_step=False, on_epoch=True)
        torch.cuda.empty_cache()
        return dzdt_net_b, loss, plot_info

    def extrac_pass(self, batch, mode):
        # Extract data from DataGeometric
        if self.project_name == 'Beam_3D':
            z_t0, z_t1, edge_index, n, f = batch.x, batch.y, batch.edge_index, batch.n[:,0], batch.f
            # z_t0, z_t1, edge_index, n, f = batch.x, batch.y, batch.edge_index, batch.n, batch.f
        elif self.project_name == 'Beam_2D':
            z_t0, z_t1, edge_index, n, f = batch.x, batch.y, batch.edge_index, batch.n, batch.f
        else:
            z_t0, z_t1, edge_index, n, f = batch.x, batch.y, batch.edge_index, batch.n, None

        dzdt_net, loss, plot_info = self.pass_thought_net(z_t0, z_t1, edge_index, n, f, g=None,
                                               batch=batch.batch, mode=mode)
        return dzdt_net, loss, plot_info

    def training_step(self, batch, batch_idx, g=None):

        dzdt_net, loss, _ = self.extrac_pass(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx, g=None):

        self.extrac_pass(batch, 'val')


    def predict_step(self, batch, batch_idx, g=None):

        dzdt_net, loss, plot_info = self.extrac_pass(batch, 'eval')
        z_norm = torch.from_numpy(self.scaler.transform(batch.x.cpu())).float().to(self.device)
        z1_net = z_norm + self.dt * dzdt_net
        z1_net_denorm = torch.from_numpy(self.scaler.inverse_transform(z1_net.detach().to('cpu'))).float().to(
            self.device)

        return z1_net_denorm, batch.y, plot_info

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.miles, gamma=self.gamma),
            'monitor': 'train_loss'}

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
