import torch
import argparse
from struct import pack

parser = argparse.ArgumentParser(description='Pack Pytorch model to NiuTensor')
parser.add_argument('-task', type=str, default='wnut17')
parser.add_argument('-src', help='pytorch model', type=str, default='wnut17/best-model.pt')
parser.add_argument('-tgt', help='niutensor model', type=str, default='wnut17/sltk.bin')
args = parser.parse_args()

model = torch.load(args.src, map_location='cpu')['state_dict']

def extract_transitions(trans):
    trans = trans[:, :-1]
    a = trans[:-2, :]
    b = trans[-1:, :]
    return torch.cat((a, b), 0)

with torch.no_grad():
    params = list(model.values())

    # trim transitions in the CRF
    params.insert(1, params[0][-2, :])
    params.insert(2, params[0][:, -1])
    params[0] = extract_transitions(params[0])

    params_number = pack("Q", len(params))
    params_size = pack("Q" * len(params), *[p.numel() for p in params])

    with open(args.tgt, 'wb') as tgt:
        # part 1: number of parameters
        tgt.write(params_number)

        # part 2: offsets of parameters
        tgt.write(params_size)

        # part 3: values of parameters
        for p in params:
            values = pack("f" * p.numel(), *(p.view(-1).cpu().tolist()))
            tgt.write(values)