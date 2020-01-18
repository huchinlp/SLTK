import argparse
from struct import pack

import torch

parser = argparse.ArgumentParser(description='Pack Pytorch model to NiuTensor')
parser.add_argument('-task', type=str, default='wnut17')
parser.add_argument('-src', help='pytorch model', type=str, default='wnut17/best-model.pt')
parser.add_argument('-tgt', help='niutensor model', type=str, default='wnut17/wnut17.model')
args = parser.parse_args()

model = torch.load(args.src, map_location='cpu')

tag2id = model['tag_dictionary'].idx2item

with open(args.task + '/' + args.task + '.tag.vocab', 'w') as f:
    f.write('{}\n'.format(len(tag2id)))
    for i, tag in enumerate(tag2id):
        tag = tag.decode()
        if tag == "":
            tag = "<>"
        f.write('{}\t{}\n'.format(tag, i))

model = model['state_dict']


def extract_transitions(trans):
    trans = trans[:, :-1]
    a = trans[:-2, :]
    b = trans[-1:, :]
    return torch.cat((a, b), 0)

def get_model_parameters(m):
    p = []
    for k in m:
        print(k)
        if 'weight' in k:
            p.append(m[k].t())
        else:
            p.append(m[k])
    return p

with torch.no_grad():
    params = get_model_parameters(model)

    # trim transitions in the CRF
    # params.insert(1, params[0][-2, :])
    # params.insert(2, params[0][:, -1])
    # params[0] = extract_transitions(params[0])

    params_number = pack("Q", len(params))
    params_size = pack("Q" * len(params), *[p.numel() for p in params])

    with open(args.tgt, 'wb') as tgt:
        # part 1: number of parameters
        tgt.write(params_number)

        # part 2: offsets of parameters
        tgt.write(params_size)

        # part 3: values of parameters
        for p in params:
            values = pack("f" * p.numel(), *(p.contiguous().view(-1).cpu().tolist()))
            tgt.write(values)
