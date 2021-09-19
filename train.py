import argparse

import ptb
from models import *
from module.functions import *
from module.optimizer import *
from module.trainer import SingleTrainer


def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='cbow', required=True)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--optimizer', type=str, default='adam')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--valid_batch_size', type=int, default=128)
    p.add_argument('--hidden_size', type=int, default=100)
    p.add_argument('--early_stopping', type=int, default=0)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--max_grad_norm', type=float, default=1e8)
    p.add_argument('--window_size', type=int, default=5)
    p.add_argument('--verbose', type=int, default=1)
    p.add_argument('--train_size', type=int, default=0)
    p.add_argument('--lr_decay', type=float, default=.25)
    p.add_argument('--warmup_epochs', type=int, default=0)
    p.add_argument('--shuffle', action='store_true')
    p.add_argument('--save_fn', type=str)

    config = p.parse_args()
    return config

def main(config):
    opts = {'sgd': SGD, 'momentum': Momentum, 'adam': Adam}
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_val, _, _ = ptb.load_data('valid')

    if config.train_size > 0:
      corpus = corpus[:config.train_size]
      corpus_val = corpus_val[:int(config.train_size * 0.2)]

    print('TRAIN SIZE', len(corpus))
    print('VALID SIZE', len(corpus_val))

    contexts, target = create_contexts_target(corpus, window_size=config.window_size, one_hot_encoding=False)
    vc, vt = create_contexts_target(corpus_val, window_size=config.window_size, one_hot_encoding=False)
    valid_data = vt, vc

    vocab_size = len(word_to_id)

    optimizer = opts[config.optimizer.lower()](lr=config.lr)
    if config.model.lower() == 'skipgram':
        model = Skipgram(vocab_size,
                         config.hidden_size,
                         config.window_size)
    elif config.model.lower() == 'cbow':
        model = CBOW(corpus,
                     vocab_size,
                     config.hidden_size,
                     config.window_size,
                     config.batch_size,
                     config.sample_size,
                     config.power,
                     config.exclude_target)
    else:
        raise NotImplementedError

    trainer = SingleTrainer(config, model, optimizer)
    trainer.train(target, contexts, valid_data=valid_data)
    save_params(trainer.model, word_to_id, id_to_word, config.save_fn)

if __name__ == '__main__':
    config = define_argparser()
    main(config)