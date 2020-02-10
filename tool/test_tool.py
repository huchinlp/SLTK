import sys


def get_text(fin, fout):
    with open(fin, 'r', encoding='utf8') as fi, open(fout, 'w', encoding='utf8') as fout:
        for l in fi:
            if l != '\n':
                l = l.split()[0]+'\n'
            fout.write(l)


get_text(sys.argv[1], sys.argv[2])
