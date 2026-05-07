import os
import numpy as np
import tqdm

def p(x, sigma, N=10):
    _p = 0
    for i in tqdm.trange(-N, N + 1):
        _p += np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return _p

def grad(x, sigma, N=10):
    _p = 0
    for i in tqdm.trange(-N, N + 1):
        _p += (x + 2 * np.pi * i) / sigma ** 2 * np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return _p

X_MIN, X_N = 1e-5, 5000  # relative to pi
SIGMA_MIN, SIGMA_MAX, SIGMA_N = 3e-3, 2, 5000  # relative to pi

x = 10 ** np.linspace(np.log10(X_MIN), 0, X_N + 1) * np.pi
sigma = 10 ** np.linspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_N + 1) * np.pi
'''
if os.path.exists('./dataset/pdbbind/processed/torus/.p.npy'):
    _p = np.load('./dataset/pdbbind/processed/torus/.p.npy')
    _score = np.load('./dataset/pdbbind/processed/torus/.score.npy')
else:
    print("Precomputing and saving to cache torus distribution table")
    os.makedirs('./dataset/pdbbind/processed/torus', exist_ok=True)
    _p = p(x, sigma[:, None], N=100)
    np.save('./dataset/pdbbind/processed/torus/.p.npy', _p)

    _score = grad(x, sigma[:, None], N=100) / _p
    np.save('./dataset/pdbbind/processed/torus/.score.npy', _score)
'''
def get_p():
    if os.path.exists('./dataset/pdbbind/processed/torus/.p.npy'):
        _p = np.load('./dataset/pdbbind/processed/torus/.p.npy')
    else:
        print("Precomputing and saving to cache torus distribution table")
        os.makedirs('./dataset/pdbbind/processed/torus', exist_ok=True)
        _p = p(x, sigma[:, None], N=100)
        np.save('./dataset/pdbbind/processed/torus/.p.npy', _p)
    return _p

def get_score():
    if os.path.exists('./dataset/pdbbind/processed/torus/.score.npy'):
        _score = np.load('./dataset/pdbbind/processed/torus/.score.npy')
    else:
        print("Precomputing and saving to cache torus distribution table")
        os.makedirs('./dataset/pdbbind/processed/torus', exist_ok=True)
        _p = get_p()
        _score = grad(x, sigma[:, None], N=100) / _p
        np.save('./dataset/pdbbind/processed/torus/.score.npy', _score)
    return _score

def score(x, sigma):
    x = (x + np.pi) % (2 * np.pi) - np.pi
    sign = np.sign(x)
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    _score = get_score()
    return -sign * _score[sigma, x]

def sample(sigma):
    out = sigma * np.random.randn(*sigma.shape)
    out = (out + np.pi) % (2 * np.pi) - np.pi
    return out

def get_score_norm():
    score_norm_ = score(
        sample(sigma[None].repeat(10000, 0).flatten()),
        sigma[None].repeat(10000, 0).flatten()
    ).reshape(10000, -1)
    score_norm_ = (score_norm_ ** 2).mean(0)
    return score_norm_

def score_norm(sigma):
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    score_norm_ = get_score_norm()
    return score_norm_[sigma]
