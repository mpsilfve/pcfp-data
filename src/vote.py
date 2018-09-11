from sys import argv
from collections import Counter

prefix = argv[1]
filecount=int(argv[2])

datas = []
for fn in ["%s.%u.res" % (prefix,i) for i in range(1,filecount+1)]:
    f = open(fn)
    datas.append([])
    paradigm = {}
    for line in f:
        line = line.strip('\n')
        if line == '':
            if paradigm != {}:
                datas[-1].append(paradigm)
                paradigm = {}
        else:
            wf, label = line.split('\t')
            paradigm[label] = wf
    if paradigm != {}:
        datas[-1].append(paradigm)
        
for paradigms in zip(*datas):
    for l in paradigms[0]:
        majority = Counter([p[l] for p in paradigms]).most_common()[0][0]
        print("%s\t%s" % (majority,l))
    print()
