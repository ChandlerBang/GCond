'''Configuration'''

def load_config(args):
    dataset = args.dataset
    if dataset in ['flickr']:
        args.nlayers = 2
        args.hidden = 256
        args.weight_decay = 5e-3
        args.dropout = 0.0

    if dataset in ['reddit']:
        args.nlayers = 2
        args.hidden = 256
        args.weight_decay = 0e-4
        args.dropout = 0

    if dataset in ['ogbn-arxiv']:
        args.hidden = 256
        args.weight_decay = 0
        args.dropout = 0

    return args


