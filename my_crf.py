import random
from collections import defaultdict as dd
from math import exp, log

BOS, EOS = '<BOS>', '<EOS>'
y2i = dd(lambda: len(y2i))
l2_coeff = 1
learning_rate = 10
features = {}

random.seed(111)

def print_ab(alpha, T, name):
    print(name)
    for t in range(T):
        print(f't={t}', [(k, v) for k, v in alpha.items() if k[1] == t])


def load_data(fpath):
    corpus = []
    V = set()
    Y = set()

    with open(fpath, 'r') as fin:
        words = [BOS]
        tags = []
        # tags.append(y2i[BOS])
        tags.append(BOS)
        for line in fin:
            if line == '\n':
                words.append(EOS)
                # tags.append(y2i[EOS])
                tags.append(EOS)
                corpus.append((words, tags))
                continue
            x, y = line.strip().split(' ', 1)
            words.append(x)
            # tags.append(y2i[y])
            y2i[y]
            tags.append(y)
    return corpus


# def calc_feat(x, i, y_prev, y_crnt):
def init_weight_and_features(corpus):
    """TODO: calc hash is slow"""
    w = dd(lambda: random.random())
    # w = dd(lambda: 0)
    feat = dd(lambda: 0)
    # for X, Y in corpus:
    #     for t, (x, y) in enumerate(zip(X, Y)):
    #         t += 1 # +1 offset
    #         y_prev = Y[t-1]
    #         keys = [('T', y_prev, y), ('E', y, x)]
    #         for k in keys:
    #             feat[k] = 1
    #             w[k] = 1
    return w, feat


def calc_gradient(x, y, w):
    alpha = {()}


def main():
    """main.
    """
    corpus = load_data('./train.data')
    w, features = init_weight_and_features(corpus)
    for X, Y in corpus:
        T = len(Y) - 1  # -1 for EOS

        # forward
        alpha = dd(lambda: 0)
        alpha[(BOS, 0)] = 1
        for t in range(1, T + 1):
            y, x = Y[t], X[t]
            if t == 1:
                y_pred_list = [BOS]
            else:
                y_pred_list = y2i.keys()

            for y_pred in y_pred_list:
                a_prev = alpha[(y_pred, t - 1)]
                val = exp(w[('T', y_pred, y)] + w[('E', y, x)])
                alpha[(y, t)] += val * a_prev

            if t == T:
                Z = alpha[(y, t)]
        print(f'Z = {Z:.3f}')

        # backward
        beta = dd(lambda: 0)
        beta[(EOS, T)] = 1
        for t in reversed(range(T)):
            y, x = Y[t], X[t]
            if t == T - 1:
                y_next_list = [EOS]
            else:
                y_next_list = y2i.keys()

            for y_next in y_next_list:
                b_next = beta[(y_next, t + 1)]
                val = exp(w[('T', y, y_next)] + w[('E', y, x)])
                beta[(y, t)] += val * b_next

        print_ab(alpha, T + 1, 'alpha')
        print_ab(beta, T + 1, 'beta')
        # P(y3,y2|x) = (1/Z)*exp(w・φ(y3, y2))*a(y2, 2)*b(y3, 3)
        t = 3
        P = 0
        x = X[t]
        print(f'calc P(y{t}, y{t-1}|x)')
        for y_prev in y2i:
            for y in y2i:
                val = exp(w[('T', y_prev, y)] + w[('E', y, x)])
                a = alpha[(y_prev, t - 1)]
                b = beta[(y, t)]
                this_p = val * a * b
                print(
                    f'P += exp(w・φ(y3={y}, y2={y_prev}) * a({y_prev}, {t-1}) * b({y}, {t})' + \
                            f'= {this_p:.2f}'
                )
                P += this_p
        print(f'P(y{t}, y{t-1}|x) = {P/Z} ({P:.3f}/{Z:.3f})')

    pass
    return

    for iter_num in range(1, 3):
        grad = dd(lambda: 0)
        regularized_likelihood = 0
        for k, v in w.items():
            grad[k] -= 2 * v * l2_coeff
            regularized_likelihood -= v * v * l2_coeff

        likelihood = 0
        for x, y in corpus:
            my_grad, my_lik = calc_gradient(x, y, w)
            for k, v in my_grad.items():
                grad[k] += v
            likelihood += my_lik

        l1 = sum([abs(k) for k in grad.values()])
        print(
            f'Iter {iter_num} likelihood: lik={likelihood},'\
                    'ref_lik={regularized_likelihood}, gradL1={l1}'
        )

        for k, v in grad.items():
            w[k] += v / l1 * learning_rate


if __name__ == '__main__':
    main()
