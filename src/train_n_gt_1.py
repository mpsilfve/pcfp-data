from sys import stdout, argv

import dynet as dy
import random
import pickle

EOS = "<EOS>"

int2char = [EOS,'+']
char2int = {EOS:0,'+':1}

LSTM_NUM_OF_LAYERS = 1
EMBEDDINGS_SIZE = 100
STATE_SIZE = 100
ATTENTION_SIZE = 100

EPOCHS=50
def init():
    global model, enc_fwd_lstm, enc_bwd_lstm, dec_lstm, input_lookup, attention_w1,\
    attention_w2,attention_v,decoder_w,decoder_b,output_lookup, VOCAB_SIZE
    VOCAB_SIZE = len(char2int)

    model = dy.Model()
    
    enc_fwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, 
                                  model)
    enc_bwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, 
                                  model)
    
    dec_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE*2+EMBEDDINGS_SIZE, 
                              STATE_SIZE, model)
    
    input_lookup = model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE))
    attention_w1 = model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*2))
    attention_w2 = model.add_parameters( (ATTENTION_SIZE, 
                                          STATE_SIZE*LSTM_NUM_OF_LAYERS*2))
    attention_v = model.add_parameters( (1, ATTENTION_SIZE))
    decoder_w = model.add_parameters( (VOCAB_SIZE, STATE_SIZE))
    decoder_b = model.add_parameters( (VOCAB_SIZE))
    output_lookup = model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE))

def readdata(fn):
    print(fn)
    data = [[]]
    for line in open(fn):
        line = line.strip('\n')
        if not line:
            data.append([])
        else:
            wf, label = line.split('\t')
            label = label.split(',')
            wf = [c for c in wf]
            for c in wf + label:
                if not c in char2int:
                    char2int[c] = len(char2int)
                    int2char.append(c)
            if wf == []:
                continue
            data[-1].append((wf,label))
    examples = []
    for d in data:
        if len(d) <= 1:
            continue
        for form1, label1 in d:
            for form2, label2 in d:                
                if label1 != label2:
                    examples.append((form1 + ['+'] + label1 + ['+'] + label2,
                                     form2))
    return examples

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


def decode(dec_lstm, vectors, output):
    output = [EOS] + list(output) + [EOS]
    output = [char2int[c] for c in output]

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(vectors)
    w1dt = None

    last_output_embeddings = output_lookup[char2int[EOS]]
    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE*2), last_output_embeddings]))
    loss = []

    for char in output:
        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(input_mat, s, w1dt), last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector)
        last_output_embeddings = output_lookup[char]
        loss.append(-dy.log(dy.pick(probs, char)))
    loss = dy.esum(loss)
    return loss


def generate(in_seq, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    embedded = embed_sentence(in_seq)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(encoded)
    w1dt = None

    last_output_embeddings = output_lookup[char2int[EOS]]
    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_output_embeddings]))

    out = ''
    count_EOS = 0
    for i in range(len(in_seq)*2):
        if count_EOS == 2: break
        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(input_mat, s, w1dt), last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector).vec_value()
        next_char = probs.index(max(probs))
        last_output_embeddings = output_lookup[next_char]
        if int2char[next_char] == EOS:
            count_EOS += 1
            continue

        out += int2char[next_char]
    return out


def get_loss(input_sentence, output_sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    dy.renew_cg()
    embedded = embed_sentence(input_sentence)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)
    return decode(dec_lstm, encoded, output_sentence)


def train(model, data):
    trainer = dy.SimpleSGDTrainer(model)
    for n in range(EPOCHS):
        totalloss = 0
        random.shuffle(data)
        for i, io in enumerate(data):
            if i > 10000:
                break
            stdout.write('EPOCH %u: ex %u of %u\r' % (n+1,i+1,len(data)))
            input,output = io
            loss = get_loss(input, output, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
            totalloss += loss.value()
            loss.backward()
            trainer.update()
        print()
        print(totalloss/len(data))
        for input, output in data[:10]:
            print('input:',''.join(input),
                  'sys:',generate(input, enc_fwd_lstm, enc_bwd_lstm, dec_lstm),
                  'gold:',''.join(output))


if __name__=='__main__':
    data = readdata(argv[1])
    init()
    train(model, data)
    model.save(argv[2])    
    pickle.dump((int2char,char2int,VOCAB_SIZE),open("%s.obj.pkl" % argv[2],"wb"))
