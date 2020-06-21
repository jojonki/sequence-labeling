import random
import sys
from collections import defaultdict as dd
from math import exp, log

N_ITERS = 10
BOS, EOS = '<BOS>', '<EOS>'
y2i = dd(lambda: len(y2i))
l2_coeff = 1
learning_rate = 1
features = {}

random.seed(1110)


def debug(X, Z, w, alpha, beta):
    """周辺確率を足して１になるかチェック
    P(y3,y2|x) = (1/Z)*exp(w・φ(y3, y2))*a(y2, 2)*b(y3, 3)
    """
    t = 3
    P = 0
    x = X[t]
    # print('y2i', dict(y2i))
    # print(f'calc P(y{t}, y{t-1}|x)')
    # for y_prev in [BOS]:
    for y_prev in y2i:
        for y in y2i:
            val = exp(w[('T', y_prev, y)] + w[('E', y, x)])
            a = alpha[(y_prev, t - 1)]
            b = beta[(y, t)]
            this_p = val * a * b
            # print(
            #     f'P += exp(w・φ(y{t-1}={y_prev}, y{t}={y}) * a({y_prev}, {t-1}) * b({y}, {t})' + \
            #             f' = {this_p/Z:.2f} ({this_p:.3f}/{Z:.3f})'
            # )
            P += this_p / Z
    epsilon = 0.0001
    assert 1.0 - epsilon <= P <= 1.0 + epsilon
    # print(f'P(y{t}, y{t-1}|x) = {P:.3f}')


def print_ab(alpha, T, name):
    print(name)
    for t in range(T):
        print(f't={t}', [(k, v) for k, v in alpha.items() if k[1] == t])


def print_w(w):
    for k, v in w.items():
        if k[0] == 'T' and v != 0.0:
            print(f'{k}: {v:.3f}')
    for k, v in w.items():
        if k[0] == 'E' and v != 0.0:
            print(f'{k}: {v:.3f}')


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
                words = [BOS]
                tags = []
                continue
            x, y = line.strip().split(' ', 1)
            words.append(x)
            # tags.append(y2i[y])
            y2i[y]
            tags.append(y)
    return corpus


def forward_backward(X, w, T):
    # forward
    alpha = dd(lambda: 0)
    alpha[(BOS, 0)] = 1
    for t in range(1, T + 1):
        # y, x = Y[t], X[t]
        x = X[t]
        if t == 1:
            y_prev_list = [BOS]
        else:
            y_prev_list = y2i.keys()
        if t == T:
            y_list = [EOS]
        else:
            y_list = y2i.keys()

        for y in y_list:
            for y_prev in y_prev_list:
                a_prev = alpha[(y_prev, t - 1)]
                val = exp(w[('T', y_prev, y)] + w[('E', y, x)])
                alpha[(y, t)] += val * a_prev

        if t == T:
            Z = alpha[(EOS, t)]
    print(f'Z = {Z:.3f}')

    # backward
    beta = dd(lambda: 0)
    beta[(EOS, T)] = 1
    for t in reversed(range(T)):
        # y, x = Y[t], X[t]
        x = X[t]
        if t == T - 1:
            y_next_list = [EOS]
        else:
            y_next_list = y2i.keys()

        if t == 0:
            y_list = [BOS]
        else:
            y_list = y2i.keys()

        for y in y_list:
            for y_next in y_next_list:
                b_next = beta[(y_next, t + 1)]
                val = exp(w[('T', y, y_next)] + w[('E', y, x)])
                beta[(y, t)] += val * b_next

    return Z, alpha, beta


def calc_grad(X, Y, w, T, Z, alpha, beta):
    grad = dd(lambda: 0)
    for t in range(1, T + 1):
        y_prev, y = Y[t - 1], Y[t]
        x = X[t]
        features = [('T', y_prev, y)]
        if t != T:
            features += [('E', y, x)]
        for ft in features:
            grad[ft] = 1

    likelihood = sum(grad[k] * w[k] for k in grad if k in w) - log(Z)
    print(f'likelihood = {likelihood:.3f}, P(y|x) = {exp(likelihood):.5f}')

    for t in range(1, T):
        if t == 1:
            y_prev_list = [BOS]
        else:
            y_prev_list = y2i.keys()
        if t == T - 1:
            y_list = [EOS]
        else:
            y_list = y2i.keys()

        for y_prev in y_prev_list:
            for y in y_list:
                # P(yt-1, yt|x) = (1/Z) * exp(w・φ(yt, yt-1)) * a(yt-1, t-1) * b(yt, t)
                val = exp(w[('T', y_prev, y)] + w[('E', y, x)])
                a = alpha[(y_prev, t - 1)]
                b = beta[(y, t)]
                p = val * a * b / Z

                features = [('T', y_prev, y)]
                if t != T:
                    features += [('E', y, x)]
                for ft in features:
                    grad[ft] -= p
    return grad


def main():
    corpus = load_data(sys.argv[1])

    w = dd(lambda: 0)
    feat = dd(lambda: 1)
    for iter_num in range(1, N_ITERS + 1):
        print(f'-------Iter {iter_num} --------')
        for X, Y in corpus:
            T = len(Y) - 1  # -1 for EOS
            Z, alpha, beta = forward_backward(X, w, T)
            debug(X, Z, w, alpha, beta)

            grad = calc_grad(X, Y, w, T, Z, alpha, beta)

            # print(f'iter={iter_num}, grad={dict(grad)}')
            print_w(w)
            for k, v in grad.items():
                w[k] += learning_rate * (v - l2_coeff * (v * v))

    # test
    X = [BOS, 'i', 'have', 'books', 'in', 'tokyo', EOS]
    T = len(X) - 1
    Z, alpha, beta = forward_backward(X, w, T)
    y_prev = BOS
    for t in range(1, T):
        x = X[t]
        y = None
        max_p = -float('inf')
        for y_cand in y2i:
            val = exp(w[('T', y_prev, y_cand)] + w[('E', y_cand, x)])
            a = alpha[(y_prev, t - 1)]
            b = beta[(y_cand, t)]
            p = val * a * b / Z
            if p >= max_p:
                max_p = p
                y = y_cand
        print(f't={t}: {x}: {y}')
        y_prev = y


if __name__ == '__main__':
    main()
