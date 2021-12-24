import torch
import hamiltorch as ht
import matplotlib.pyplot as plt
ht.set_random_seed(123)
device = torch.device('cpu')


def log_prob_func(params):
    mean = torch.tensor([1., 2., 3.])
    stddev = torch.tensor([0.5, 0.5, 0.5])
    return torch.distributions.Normal(mean, stddev).log_prob(params).sum()


num_samples = 40000
step_size = .3
num_steps_per_sample = 5  # this is L

ht.set_random_seed(123)
params_init = torch.zeros(3)
params_hmc = ht.sample(log_prob_func=log_prob_func, params_init=params_init,  num_samples=num_samples,
                       step_size=step_size, num_steps_per_sample=num_steps_per_sample)


a, b, c = 0, 0, 0

for i in params_hmc:
    a += i[0]
    b += i[1]
    c += i[2]
a, b, c = a/num_samples, b/num_samples, c/num_samples
