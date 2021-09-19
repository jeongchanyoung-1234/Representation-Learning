import cupy as np

def save_params(model, word_to_id, id_to_word, pkl_file='cbow_params.pkl') :
    params = {}
    params['params'] = model.params
    params['emb_in'] = model.W_in
    params['emb_out'] = model.W_out
    params['word_to_id'] = word_to_id
    params['id_to_word'] = id_to_word
    with open('./drive/MyDrive/deeplearning/' + pkl_file, 'wb') as f :
        pickle.dump(params, f, -1)


def load_params(model, pkl_file='cbow_params.pkl') :
    with open('./drive/MyDrive/deeplearning/' + pkl_file, 'rb') as f :
        params = pickle.load(f, -1)

        ps = [p.astype('f') for p in params['params']]
        for i, param in enumerate(model.params) :
            param[...] = params[i]


def clip_grads(grads, max_norm) :
    total_norm = 0
    for grad in grads :
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1 :
        for grad in grads :
            grad *= rate


def get_norm(params, grads, norm_type=2.) :
    p_norm = 0
    for param in params :
        p_norm += (param ** norm_type).sum()
    p_norm **= (1. / norm_type)

    g_norm = 0
    for grad in grads :
        g_norm += (grad ** norm_type).sum()
    g_norm **= (1. / norm_type)

    if np.isnan(p_norm) or np.isinf(p_norm) :
        p_norm = 0.

    if np.isnan(g_norm) or np.isinf(g_norm) :
        g_norm = 0.

    return p_norm, g_norm


def softmax(x) :
    if x.ndim == 2 :
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1 :
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x


def sigmoid(x) :
    pos_indice = x >= 0
    neg_indice = x < 0

    new_x = np.zeros_like(x).astype('f')
    new_x[pos_indice] = 1 / (1 + np.exp(-x[pos_indice]))
    new_x[neg_indice] = np.exp(x[neg_indice]) / (1 + np.exp(x[neg_indice]))

    return new_x


def binary_cross_entropy(y_hat, y, eps=1e-7) :
    C = 1. if type(y) == int else len(y)
    return -(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)).sum() / C


def create_contexts_target(corpus, window_size=1, one_hot_encoding=True) :
    target = corpus[window_size :-window_size]
    contexts = []
    for idx in range(window_size, len(corpus) - window_size) :
        cs = []
        for t in range(-window_size, window_size + 1) :
            if t == 0 :
                continue
            cs.append(corpus[idx + t])

        contexts.append(cs)
    if one_hot_encoding :
        vocab_size = len(np.unique(corpus))
        contexts, target = np.eye(vocab_size)[np.array(contexts)], np.eye(vocab_size)[np.array(target)]
    return np.array(contexts), np.array(target)


def preprocess(text) :
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words :
        if word not in word_to_id :
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


def convert_one_hot(corpus, vocab_size) :
    N = corpus.shape[0]

    if corpus.ndim == 1 :
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus) :
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2 :
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus) :
            for idx_1, word_id in enumerate(word_ids) :
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot