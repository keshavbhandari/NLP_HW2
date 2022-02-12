import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MyParameters(object):
    mode = 2
    time_name = 'example'

    if mode == 2:
        dim = 200
        threshold = 3
        max_epochs = 20
        bCorpus = 2  # 1 = PTB, 2 = Wiki2, 3 = Wiki103
        num_layers = 2

    vocab_size = 0
    epoch = 0
    gpu_mem = 0.90
    batch_size = 20
    time_steps = 30
    sliding_window = 30

def ReadCorpus(file_name, words, vocab, params, src):
    if src == 0:
        temp = dict()
        last = 0
        total_tokens = 0
        with open(file_name, "r") as f:
            for line in f:
                #                line = line.replace(" "+chr(8211)+" "," - ")
                tokens = line.replace("\n", " </s> ").split()
                total_tokens = total_tokens + len(tokens)

                if (total_tokens - last) > 10000000:
                    print(total_tokens)
                    last = total_tokens

                for t in tokens:
                    if t == '"':
                        t = '<quote>'
                    try:
                        elem = temp[t]
                    except:
                        elem = [0, 0]
                    elem[1] = elem[1] + 1
                    temp[t] = elem

        wNextID = 0
        words = dict()
        words['<unk>'] = [wNextID, 0]
        wNextID = wNextID + 1

        for t in temp:
            elem = temp[t]
            if elem[1] >= params.threshold:
                words[t] = [wNextID, elem[1]]
                wNextID = wNextID + 1

        vocab = list()
        vocab.append(' ')
        for w in words:
            vocab.append(' ')
        for w in words:
            elem = words[w]
            vocab[elem[0]] = w

    corpus = list()
    garbage = dict()

    last = 0
    total_tokens = 0
    with open(file_name, "r") as f:
        for line in f:
            #            line = line.replace(" "+chr(8211)+" "," - ")
            tokens = line.replace("\n", " </s> ").split()
            total_tokens = total_tokens + len(tokens)

            if (total_tokens - last) > 10000000:
                print(total_tokens)
                last = total_tokens

            for t in tokens:
                if t == '"':
                    t = '<quote>'
                try:
                    elem = words[t]
                except:
                    try:
                        g = garbage[t]
                    except:
                        g = 0
                    g = g + 1
                    garbage[t] = g
                    elem = words['<unk>']
                #                elem[1] = elem[1] + 1
                #                words[t] = elem
                corpus.append(elem[0])

    return corpus, words, vocab, garbage

class RNN_Language_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim,
                 hidden_dim, gru_layers, dropout):
        super(RNN_Language_Model, self).__init__()

        self.rnn = nn.RNN(embedding_dim, hidden_dim, gru_layers,
                          dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, vocab_size)
        self.hl_1 = nn.Linear(hidden_dim, hidden_dim)
        self.hl_2 = nn.Linear(hidden_dim, hidden_dim)

    def get_embedded(self, word_indexes):
        return self.fc1.weight.index_select(0, word_indexes)

    def forward(self, packed_sents):
        """ Takes a PackedSequence of sentences tokens that has T tokens
        belonging to vocabulary V. Outputs predicted log-probabilities
        for the token following the one that's input in a tensor shaped
        (T, |V|).
        """
        # print(packed_sents.data.shape)
        embedded_sents = nn.utils.rnn.PackedSequence(
            self.get_embedded(packed_sents.data), packed_sents.batch_sizes)
        # print(embedded_sents.data.shape)
        out_packed_sequence, _ = self.rnn(embedded_sents)
        # print(out_packed_sequence.data.shape)

        hl_1 = self.hl_1(out_packed_sequence.data)
        hl_1 = nn.Tanh()(hl_1)
        # print(hl_1.shape)
        hl_2 = self.hl_2(hl_1)
        hl_2 = nn.Tanh()(hl_2)
        # print(hl_2.shape)

        # out = self.fc1(out_packed_sequence.data)
        out = self.fc1(hl_2)
        # print(out.shape)
        return F.log_softmax(out, dim=1)


def batches(data, batch_size, time_steps, sliding_window):
    """ Yields batches of sentences from 'data', ordered on length. """
    # random.shuffle(data)
    batch = []
    counter = 0
    for i in range(1, len(data)+1, sliding_window+1):
        counter+=1
        sentences = data[i-1:i-1 + time_steps+1]
        batch.append(sentences)
        if counter > 0 and counter % batch_size == 0:
            tmp_batch = batch
            batch = []
            tmp_batch.sort(key=lambda l: len(l), reverse=True)
            yield tmp_batch

def step(model, sents, device):
    """ Performs a model inference for the given model and sentence batch.
    Returns the model otput, total loss and target outputs. """
    x = nn.utils.rnn.pack_sequence([torch.tensor(s[:-1]) for s in sents])
    y = nn.utils.rnn.pack_sequence([torch.tensor(s[1:]) for s in sents])
    if device.type == 'cuda':
        x, y = x.cuda(), y.cuda()
    out = model(x)
    loss = F.nll_loss(out, y.data)
    return out, loss, y

def train_epoch(data, model, optimizer, args, device):
    """ Trains a single epoch of the given model. """
    model.train()
    for batch_ind, sents in enumerate(batches(data, args.batch_size, args.time_steps, args.sliding_window)):
        model.zero_grad()
        out, loss, y = step(model, sents, device)
        loss.backward()
        optimizer.step()
        if batch_ind % 5 == 0:
            # Calculate perplexity.
            prob = out.exp()[
                torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
            perplexity = 2 ** prob.log2().neg().mean().item()
            print("\tBatch %d, loss %.3f, perplexity %.2f", batch_ind, loss.item(), perplexity)

params = MyParameters()
epochs = 1
train, words, vocab, train_g = ReadCorpus("wiki.train.txt", None, None, params, 0)
valid, words, vocab, valid_g = ReadCorpus("wiki.valid.txt", words, vocab, params, 1)
test, words, vocab, test_g = ReadCorpus("wiki.test.txt", words, vocab, params, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNN_Language_Model(vocab_size=len(vocab), embedding_dim = 100, hidden_dim = 100, gru_layers = 1, dropout = 0.1).to(device)
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

a = batches(train, 20, 30, 30)
for n, i in enumerate(a):
    print(len(i))
    if n > 3:
        break
#
# x = nn.utils.rnn.pack_sequence([torch.tensor(s[:-1]) for s in i])
# y = nn.utils.rnn.pack_sequence([torch.tensor(s[1:]) for s in i])
#
# b = step(model, i, device)

for epoch_ind in range(epochs):
    train_epoch(train, model, optimizer, params, device)




# A list of sentences, each being a list of tokens.
# sents = [[4, 545, 23, 1], [34, 84], [23, 6, 774]]
# # Embedding for 10k words with d=128
# emb = nn.Embedding(1000, 128)
# # When packing a sequence it has to be sorted on length.
# sents.sort(key=len, reverse=True)
# packed = nn.utils.rnn.pack_sequence(
#     [torch.tensor(s) for s in sents])
# embedded = nn.utils.rnn.PackedSequence(
#     emb(packed.data), packed.batch_sizes)
# # An LSTM
# lstm_layer = nn.LSTM(128, 128)
# output, states = lstm_layer(embedded)