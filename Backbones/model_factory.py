import torch.nn.functional as F
from .gnns import GAT, GCN, GIN, GCN_dmsg

def get_model(dataset, args):
    n_classes = args.n_cls_per_task
    #print('n_classes', n_classes)
    if args.backbone == 'GAT':
        heads = ([args.GAT_args['heads']] * args.GAT_args['num_layers']) + [args.GAT_args['out_heads']]
        model = GAT(args, heads, F.elu)
    elif args.backbone == 'GCN':
        if args.method == 'dmsg':
            model = GCN_dmsg(args)
        else:
            model = GCN(args)
    elif args.backbone == 'GIN':
        model = GIN(args)
    elif args.backbone == 'SGC':          # PDGNN's backbone (OCGL): SGC propagation + MLP
        from .sgc import SGC_MLP
        model = SGC_MLP(args)
    return model
