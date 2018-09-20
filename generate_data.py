import numpy as np


def sample_gaussian_noise(n_samples=10000, seq_length=1000):
    return np.random.normal(0, 1, (n_samples, seq_length))

def sample_sin(n_samples=10000, x_vals=np.arange(0, 100, .1), max_offset=100, mul_range=[1, 2]):
    vectors = []
    for i in range(n_samples):
        offset = np.random.random() * max_offset
        mul = mul_range[0] + np.random.random() * (mul_range[1] - mul_range[0])
        sin = np.sin(offset + x_vals * mul) / 2 + .5
        vectors.append(sin)
    return np.array(vectors)


if __name__ == '__main__':
    noise = sample_gaussian_noise(n_samples=10)
    sins = sample_sin(n_samples=10)

    np.savetxt('data/noise.csv', noise, delimiter=',')
    np.savetxt('data/sins.csv', sins, delimiter=',')

