from sys import argv, stdout

import dynet as dy
import numpy as np

EOS = "<EOS>"
UNK = "<UNK>"

LSTM_NUM_OF_LAYERS = 1
EMBEDDINGS_SIZE = 100
STATE_SIZE = 100

def readdata(fn):
    print("Read data from file %s" % fn)
    int2char = [UNK,EOS,'+']
    char2int = {UNK:0, EOS:1,'+':2}
    data = [[]]
    for line in open(fn):
        line = line.strip('\n')
        if not line:
            data.append([])
        else:
            wf, label = line.split('\t')
            if wf == '':
                continue
            label = ["TAG=" + l for l in label.split(',')]
            wf = [c for c in wf]
            for c in wf + label:
                if not c in char2int:
                    char2int[c] = len(char2int)
                    int2char.append(c)
            data[-1].append((wf,label))
    examples = []
    for i,d in enumerate(data):
#        tab = "TABLE:%u" % i
#        if not tab in char2int:
#            char2int[tab] = len(char2int)
        for form, label in d:
            examples.append((form + ['+'] + label,form))
    return examples, int2char, char2int

class LM:
    def __init__(self,int2char,char2int):
        self.int2char = int2char
        self.char2int = char2int
        self.VOCAB_SIZE = len(char2int)
        self.model = dy.Model()
        self.input_lookup = self.model.add_lookup_parameters((self.VOCAB_SIZE, EMBEDDINGS_SIZE))        
        self.lm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, 2*EMBEDDINGS_SIZE, STATE_SIZE, 
                                 self.model)
        self.w = self.model.add_parameters( (self.VOCAB_SIZE, STATE_SIZE))
        self.b = self.model.add_parameters( (self.VOCAB_SIZE))
        self.trainer = dy.AdamTrainer(self.model)

    def get_ll(self,c,s):
        c = self.char2int[c]
        w = dy.parameter(self.w)
        b = dy.parameter(self.b)
        out_vector = w * s + b
        probs = dy.softmax(out_vector)
        return -dy.log(dy.pick(probs, c))

    def get_prob(self,c,s):
        c = self.char2int[c]
        w = dy.parameter(self.w)
        b = dy.parameter(self.b)
        out_vector = w * s + b
        probs = dy.softmax(out_vector)
        return dy.pick(probs, c).value()

    def train(self,data,epochs):
        for epoch in range(epochs):
            totloss = 0
            for d in data:
                dy.renew_cg()
                tag = [c for c in d[0] if '=' in c]
                tag = dy.esum([self.input_lookup[self.char2int[t]] for t in tag])
                input = d[1]
                input_emb = [dy.concatenate([self.input_lookup[self.char2int[c]],
                                             tag]) for c in [EOS] + input]
                input_states = []
                s = self.lm.initial_state()
                for e in input_emb:
                    s = s.add_input(e)
                    input_states.append(s.output())
                loss = dy.esum([self.get_ll(c,s) for c,s in zip(input,input_states)])
                loss.backward()
                totloss += loss.value()
                self.trainer.update()
            print("EPOCH %u AVG LOSS: %.3f" % (epoch+1,totloss/len(data)))
            stdout.flush()
            
    def get_probs(self,data):
        res = []
        for d in data:
            dy.renew_cg()
            tag = [c for c in d[0] if '=' in c]
            tag = dy.esum([self.input_lookup[self.char2int[t]] for t in tag])
            input = d[1]
            input_emb = [dy.concatenate([self.input_lookup[self.char2int[c]],
                                         tag]) for c in [EOS] + input]
            input_states = []
            s = self.lm.initial_state()
            for e in input_emb:
                s = s.add_input(e)
                input_states.append(s.output())
            res.append([self.get_prob(c,s) for c,s in zip(input,input_states)])
        return res

def getprobs(fn,epochs):
    data, int2char, char2int = readdata(fn)
    lm = LM(int2char,char2int)
    lm.train(data,epochs)
    probs = lm.get_probs(data)
    return probs
    

if __name__=="__main__":
    data, int2char, char2int = readdata(argv[1])
    data,int2char,char2int = readdata(argv[1])
    wf,tag = argv[2].split('\t')
    wf = [c for c in wf]
    tag = ["TAG=%s" % t for t in tag.split(',')]
    lm = LM(int2char,char2int)
    lm.train(data,20)
    print([wf+['+']+tag,wf])
    print(lm.get_probs([[wf+tag,wf]]))

#    print(data[0])
#    print(lm.get_probs([data[0]]))
