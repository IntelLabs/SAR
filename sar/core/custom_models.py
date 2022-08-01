import dgl.function as fn
import torch
import torch.nn as nn

def mean_udf(nodes):
    msg = nodes.mailbox['m']
    n_nonzeros = (msg.sum(2) != 0).sum(1)
    return {'h_N': msg.sum(1) / (n_nonzeros.unsqueeze(-1) + 1)}


class SageConvExt(nn.Module):
    """
    Populates missing node from neighbors using maxpool operation.
    Doesn't change the nodes with signal.
    """
    def __init__(self, in_feat, out_feat, update_func="cat"):
        super(SageConvExt, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.update_func = update_func
        if update_func == "cat":
            self.linear = nn.Linear(2 * in_feat, out_feat)
        elif update_func == "copy":
            self.linear = nn.Linear(in_feat, out_feat)
        else:
            self.linear = nn.Identity()

    def forward(self, g, h, idx):
        """Forward computation

        Parameters
        ----------
        g : Graph
            The input graph.
        h : Tensor
            The input node feature.
        idx: Tensor
            Indices of the nodes with signal.
        """
        with g.local_scope():
            if self.update_func == "I":
                return h
            g.ndata['h'] = h
            # update_all is a message passing API.
            if self.update_func == "cat":
                g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.mean('m', 'h_N'))
            else:
                g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.max('m', 'h_N'))
            
            h_N = g.ndata['h_N']
            if self.update_func == "cat":
                h_total = torch.cat([h, h_N], dim=1)
                return self.linear(h_total)
            else:
                h_N = self.linear(h_N)
                # h_N /= torch.linalg.norm(h_N, dim=1, keepdim=True)
                h_N[idx, :] = h[idx, :]
                return h_N