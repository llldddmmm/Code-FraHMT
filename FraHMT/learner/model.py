import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import dgl
import dgl.nn as dglnn
from dgl import function as fn
from functools import partial
import copy
import math
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss


class MyGRU(nn.GRU):
    def __init__(self, *args, **kwargs):
        kwargs['batch_first'] = True 
        super(MyGRU, self).__init__(*args, **kwargs)

    def forward(self, input, hx=None):
        if hx is None:
            hx = input.new_zeros((self.num_layers, input.size(0), self.hidden_size), requires_grad=False)
        origin_state= hx*0.2
        outputs = []

        for t in range(input.size(1)):
            if t == 0:
                modified_state = hx
            else:
                modified_state=torch.add(origin_state, hx*0.8)
            output, hx = super(MyGRU, self).forward(input[:, t, :].unsqueeze(1), modified_state)
            outputs.append(output.squeeze(1))
        output = torch.stack(outputs, dim=1)

        return output, hx

class Decoder(nn.Module):
    def __init__(self, embed_size, latent_size, hidden_size,
                 hidden_layers, dropout, output_size):
        super().__init__()
        self.embed_size = embed_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.dropout = dropout

        self.rnn = MyGRU(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.hidden_layers,
            dropout=self.dropout,
            batch_first=True)

        self.rnn2out = nn.Linear(
            in_features=hidden_size,
            out_features=output_size)

    def forward(self, embeddings, state, lengths):
        batch_size = embeddings.size(0)
        hidden, state = self.rnn(embeddings, state)
        state = state.view(self.hidden_layers, batch_size, self.hidden_size)
        output = self.rnn2out(hidden)
        return output, state

class Frag2Mol(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.input_size = vocab.get_size()
        self.embed_size = config.get('embed_size')
        self.hidden_size = config.get('hidden_size')
        self.hidden_layers = config.get('hidden_layers')
        self.latent_size = config.get('latent_size')
        self.dropout = config.get('dropout')
        self.use_gpu = config.get('use_gpu')

        embeddings = self.load_embeddings()
        self.embedder = nn.Embedding.from_pretrained(embeddings)

        self.latent2rnn = nn.Linear(
            in_features=self.latent_size,
            out_features=self.hidden_size)
        
        encoder_model_args={
		"atom_dim": 42,
		"bond_dim": 14,
		"pharm_dim": 194,
		"reac_dim": 34,
		"hid_dim": 300,
		"depth": 3,
		"act": "ReLU",
		"num_task": 1
        }
        sssmodel = PharmHGT(encoder_model_args)
        self.encoder=sssmodel
        
        self.decoder = Decoder(
            embed_size=self.embed_size,
            latent_size=self.latent_size,
            hidden_size=self.hidden_size,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            output_size=self.input_size)

    def forward(self, inputs, lengths, sss):
        batch_size = inputs.size(0)
        embeddings = self.embedder(inputs)
        z, mu, sigma = self.encoder(sss)
        state = self.latent2rnn(z)
        state = state.view(self.hidden_layers, batch_size, self.hidden_size)
        embeddings2 = F.dropout(embeddings, p=self.dropout, training=self.training)
        output, state = self.decoder(embeddings2, state, lengths)
        return output, mu, sigma

    def load_embeddings(self):
        filename = f'emb_{self.embed_size}.dat'
        path = self.config.path('config') / filename
        embeddings = np.loadtxt(path, delimiter=",")
        return torch.from_numpy(embeddings).float()


class Loss(nn.Module):
    def __init__(self, config, pad):
        super().__init__()
        self.config = config
        self.pad = pad

    def forward(self, output, target, mu, sigma, epoch):
        output = F.log_softmax(output, dim=1)

        # flatten all predictions and targets
        target = target.view(-1)
        output = output.view(-1, output.size(2))

        # create a mask filtering out all tokens that ARE NOT the padding token
        mask = (target > self.pad).float()

        # count how many tokens we have
        nb_tokens = int(torch.sum(mask).item())

        # pick the values for the label and zero out the rest with the mask
        output = output[range(output.size(0)), target] * mask

        # compute cross entropy loss which ignores all <PAD> tokens
        CE_loss = -torch.sum(output) / nb_tokens

        # compute KL Divergence
        KL_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        # alpha = (epoch + 1)/(self.config.get('num_epochs') + 1)
        # return alpha * CE_loss + (1-alpha) * KL_loss
        return CE_loss + KL_loss


def remove_nan_label(pred,truth):
    nan = torch.isnan(truth)
    truth = truth[~nan]
    pred = pred[~nan]

    return pred,truth

def roc_auc(pred,truth):
    return roc_auc_score(truth,pred)

def rmse(pred,truth):
    return nn.functional.mse_loss(pred,truth)**0.5

def mae(pred,truth):
    return mean_absolute_error(truth,pred)

func_dict={'relu':nn.ReLU(),
           'sigmoid':nn.Sigmoid(),
           'mse':nn.MSELoss(),
           'rmse':rmse,
           'mae':mae,
           'crossentropy':nn.CrossEntropyLoss(),
           'bce':nn.BCEWithLogitsLoss(),
           'auc':roc_auc,
           }

def get_func(fn_name):
    fn_name = fn_name.lower()
    return func_dict[fn_name]


def reverse_edge(tensor):
    n = tensor.size(0)
    assert n%2 ==0
    delta = torch.ones(n).type(torch.long)
    delta[torch.arange(1,n,2)] = -1
    return tensor[delta+torch.tensor(range(n))]

def del_reverse_message(edge,field):
    """for g.apply_edges"""
    return {'m': edge.src[field]-edge.data['rev_h']}

def add_attn(node,field,attn):
        feat = node.data[field].unsqueeze(1)
        return {field: (attn(feat,node.mailbox['m'],node.mailbox['m'])+feat).squeeze(1)}


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class Node_GRU(nn.Module):
    """GRU for graph readout. Implemented with dgl graph"""
    def __init__(self,hid_dim,bidirectional=True):
        super(Node_GRU,self).__init__()
        self.hid_dim = hid_dim
        if bidirectional:
            self.direction = 2
        else:
            self.direction = 1
        self.att_mix = MultiHeadedAttention(6,hid_dim)
        self.gru  = nn.GRU(hid_dim, hid_dim, batch_first=True, 
                           bidirectional=bidirectional)
    
    def split_batch(self, bg, ntype, field, device):
        hidden = bg.nodes[ntype].data[field]
        node_size = bg.batch_num_nodes(ntype)
        start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(node_size,0)[:-1]])
        max_num_node = max(node_size)
        hidden_lst = []
        for i  in range(bg.batch_size):
            start, size = start_index[i],node_size[i]
            assert size != 0, size
            cur_hidden = hidden.narrow(0, start, size)
            cur_hidden = torch.nn.ZeroPad2d((0,0,0,max_num_node-cur_hidden.shape[0]))(cur_hidden)
            hidden_lst.append(cur_hidden.unsqueeze(0))
        hidden_lst = torch.cat(hidden_lst, 0)
        return hidden_lst
        
    def forward(self,bg,suffix='h'):
        """
        bg: dgl.Graph (batch)
        hidden states of nodes are supposed to be in field 'h'.
        """
        self.suffix = suffix
        device = bg.device
        p_pharmj = self.split_batch(bg,'p',f'f_{suffix}',device)
        a_pharmj = self.split_batch(bg,'a',f'f_{suffix}',device)

        mask = (a_pharmj!=0).type(torch.float32).matmul((p_pharmj.transpose(-1,-2)!=0).type(torch.float32))==0
        h = self.att_mix(a_pharmj, p_pharmj, p_pharmj,mask) + a_pharmj

        hidden = h.max(1)[0].unsqueeze(0).repeat(self.direction,1,1)
        h, hidden = self.gru(h, hidden)
        graph_embed = []
        node_size = bg.batch_num_nodes('p')
        start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(node_size,0)[:-1]])
        for i  in range(bg.batch_size):
            start, size = start_index[i],node_size[i]
            graph_embed.append(h[i, :size].view(-1, self.direction*self.hid_dim).mean(0).unsqueeze(0))
        graph_embed = torch.cat(graph_embed, 0)

        return graph_embed

class HeteroGraphConvNet(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(HeteroGraphConvNet, self).__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            'b': dglnn.GraphConv(in_feats, hidden_feats),
            'r': dglnn.GraphConv(in_feats, hidden_feats),
            'j': dglnn.GraphConv(in_feats, hidden_feats),
            'j': dglnn.GraphConv(in_feats, hidden_feats)
        })
        self.conv2 = dglnn.HeteroGraphConv({
            'b': dglnn.GraphConv(hidden_feats, out_feats),
            'r': dglnn.GraphConv(hidden_feats, out_feats),
            'j': dglnn.GraphConv(hidden_feats, out_feats),
            'j': dglnn.GraphConv(hidden_feats, out_feats)
        })

    def forward(self, g, node_feats):
        node_feats = self.conv1(g, node_feats)
        node_feats = {k: torch.relu(v) for k, v in node_feats.items()}
        node_feats = self.conv2(g, node_feats)
        return node_feats

class PharmHGT(nn.Module):
    def __init__(self,args):
        super(PharmHGT,self).__init__()

        self.use_gpu=False
        hid_dim = args['hid_dim']
        self.act = get_func(args['act'])
        self.depth = args['depth']
        self.w_atom = nn.Linear(args['atom_dim'],hid_dim)
        self.w_bond = nn.Linear(args['bond_dim'],hid_dim)
        self.w_pharm = nn.Linear(args['pharm_dim'],hid_dim)
        self.w_reac = nn.Linear(args['reac_dim'],hid_dim)
        self.w_junc = nn.Linear(args['atom_dim'] + args['pharm_dim'],hid_dim)
        self.readout = Node_GRU(hid_dim)
        self.readout_attn = Node_GRU(hid_dim)
        self.initialize_weights()
        self.qliner=nn.Linear(600, 64)
        self.gcn2=nn.Linear(600,128)

        self.rnn2mean = nn.Linear(
            in_features=64 * 2,
            out_features=100)

        self.rnn2logv = nn.Linear(
            in_features=64 * 2,
            out_features=100)
        
        in_feats = 300
        hidden_feats = 100
        out_feats = 20
        self.GCNmodel = HeteroGraphConvNet(in_feats, hidden_feats, out_feats)

        self.pliner=nn.Linear(240,128)
        self.aliner=nn.Linear(760,128)

    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    def init_feature(self,bg):
        bg.nodes['a'].data['f'] = self.act(self.w_atom(bg.nodes['a'].data['f']))
        bg.edges[('a','b','a')].data['x'] = self.act(self.w_bond(bg.edges[('a','b','a')].data['x']))
        bg.nodes['p'].data['f'] = self.act(self.w_pharm(bg.nodes['p'].data['f']))
        bg.edges[('p','r','p')].data['x'] = self.act(self.w_reac(bg.edges[('p','r','p')].data['x']))
        bg.nodes['a'].data['f_junc'] = self.act(self.w_junc(bg.nodes['a'].data['f_junc']))
        bg.nodes['p'].data['f_junc'] = self.act(self.w_junc(bg.nodes['p'].data['f_junc']))

    def sample_normal(self, dim):
        z = torch.randn((2, dim, 100))
        return Variable(z).cuda() if self.use_gpu else Variable(z)        

    def split_batch(self, bg, ntype, field, device):
        hidden = bg.nodes[ntype].data[field]  
        node_size = bg.batch_num_nodes(ntype)  
        start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(node_size,0)[:-1]])
        if ntype == 'a':
            max_num_node = 38
        else:
            max_num_node = 12
        hidden_lst = []
        for i  in range(bg.batch_size):
            start, size = start_index[i],node_size[i]
            assert size != 0, size
            cur_hidden = hidden.narrow(0, start, size)
            cur_hidden = torch.nn.ZeroPad2d((0,0,0,max_num_node-cur_hidden.shape[0]))(cur_hidden)
            hidden_lst.append(cur_hidden.unsqueeze(0))
        hidden_lst = torch.cat(hidden_lst, 0)
        return hidden_lst

    def forward(self,bg):
        """
        Args:
            bg: a batch of graphs
        """
        self.init_feature(bg)

        output = self.GCNmodel(bg, {'a':bg.nodes['a'].data['f'] , 'p':bg.nodes['p'].data['f']})
        bg.nodes['a'].data['f_h']=output['a']
        bg.nodes['p'].data['f_h']=output['p']
        device = bg.device
        p_pharmj = self.split_batch(bg,'p',f'f_h',device)
        p_pharmj=p_pharmj.reshape(32, -1)
        a_pharmj = self.split_batch(bg,'a',f'f_h',device)
        a_pharmj=a_pharmj.reshape(32, -1)
        p_emb=self.pliner(p_pharmj)
        a_emb=self.aliner(a_pharmj)
        embed=torch.add(p_emb,a_emb)

        state=embed
        mean = self.rnn2mean(state)
        logv = self.rnn2logv(state)
        std = torch.exp(0.5 * logv)
        z = self.sample_normal(dim=32)
        z=z.cuda()
        latent_sample = z * std + mean
        return latent_sample, mean, std
        
