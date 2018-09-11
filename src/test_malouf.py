from sys import stdout, argv

import dynet as dy
import random
import pickle

EOS = "<EOS>"

LSTM_NUM_OF_LAYERS = 1
EMBEDDINGS_SIZE = 132
STATE_SIZE = 100
ATTENTION_SIZE = 100

EPOCHS=30

char2int = None
int2char = None

def init():
    global model, generator, lookup, attention_w1,\
    decoder_w,decoder_b,output_lookup, VOCAB_SIZE
    VOCAB_SIZE = len(char2int)
    model = dy.Model()
    generator = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, 2*EMBEDDINGS_SIZE, STATE_SIZE, model)
    lookup = model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE))
    decoder_w = model.add_parameters( (VOCAB_SIZE, STATE_SIZE))
    decoder_b = model.add_parameters( (VOCAB_SIZE))
    
def readdata(fn):
    data = [[]]

    for line in open(fn):
        line = line.strip('\n')
        if not line:
            data.append([])
        else:
            wf, label = line.split('\t')
#            if not label in char2int:
#                char2int[label] = len(char2int)
#                int2char.append(label)
            wf = [c for c in wf]
#            for c in wf:
#                if not c in char2int:
#                    char2int[c] = len(char2int)
#                    int2char.append(c)
            data[-1].append((wf,label))
    data = [d for d in data if d != []]
    return data

def embed_sentence(sentence):
    sentence = [EOS] + list(sentence) + [EOS]
    sentence = [char2int[c] for c in sentence]

    global input_lookup

    return [input_lookup[char] for char in sentence]


def run_lstm(init_state, input_vecs):
    s = init_state

    out_vectors = []
    for vector in input_vecs:
        s = s.add_input(vector)
        out_vector = s.output()
        out_vectors.append(out_vector)
    return out_vectors


def encode_sentence(enc_fwd_lstm, enc_bwd_lstm, sentence):
    sentence_rev = list(reversed(sentence))

    fwd_vectors = run_lstm(enc_fwd_lstm.initial_state(), sentence)
    bwd_vectors = run_lstm(enc_bwd_lstm.initial_state(), sentence_rev)
    bwd_vectors = list(reversed(bwd_vectors))
    vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

    return vectors


def attend(input_mat, state, w1dt):
    global attention_w2
    global attention_v
    w2 = dy.parameter(attention_w2)
    v = dy.parameter(attention_v)

    # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
    # w1dt: (attdim x seqlen)
    # w2dt: (attdim x attdim)
    w2dt = w2*dy.concatenate(list(state.s()))
    # att_weights: (seqlen,) row vector
    unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
    att_weights = dy.softmax(unnormalized)
    # context: (encoder_state)
    context = input_mat * att_weights
    return context


def decode(embedded, wf):
    wf = list(wf) + [EOS]
    wf = [char2int[c] for c in wf]
    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    last_output_embeddings = lookup[char2int[EOS]]
    s = generator.initial_state().add_input(dy.concatenate([embedded,
                                                            last_output_embeddings]))
    loss = []
    for char in wf:
        # w1dt can be computed and cached once for the entire decoding phase
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector)
        last_output_embeddings = lookup[char]
        loss.append(-dy.log(dy.pick(probs, char)))
        s = s.add_input(dy.concatenate([embedded,last_output_embeddings]))
    loss = dy.esum(loss)
    return loss

def embed(tab,label):
    return dy.esum([lookup[tab],lookup[label]])

def generate(tab,label):
    embedded = embed(char2int[tab],char2int[label])
    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    last_output_embeddings = lookup[char2int[EOS]]
    s = generator.initial_state().add_input(dy.concatenate([embedded,
                                                            last_output_embeddings]))
    out = ''
    count_EOS = 1
    for i in range(30):
        if count_EOS == 2: break
        # w1dt can be computed and cached once for the entire decoding phase
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector).vec_value()
        next_char = probs.index(max(probs))
        last_output_embeddings = lookup[next_char]
        s = s.add_input(dy.concatenate([embedded,last_output_embeddings]))
        if int2char[next_char] == EOS:
            count_EOS += 1
            continue
        out += int2char[next_char]
    return out

def embed(tab,label):
    return dy.esum([lookup[tab],lookup[label]])

def get_loss(tab,label,wf):
    dy.renew_cg()
    embedded = embed(char2int[tab],char2int[label])
    return decode(embedded, wf)


def test(data):
    for i, d in enumerate(data):
        for ex in d:
            tab = "TABLE:%u" % i
            wf,label = ex
            if wf != []:
                wf = ''.join(wf)
            else:
                dy.renew_cg()
                wf = generate(tab,label)
            print("%s\t%s" % (wf,label))
        print("")

if __name__=='__main__':
    data = readdata(argv[1])
    int2char, char2int, VOCAB_SIZE = pickle.load(open("%s.obj.pkl" % argv[2],
                                                      "rb"))
    init()
    model.populate(argv[2])

    test(data)
