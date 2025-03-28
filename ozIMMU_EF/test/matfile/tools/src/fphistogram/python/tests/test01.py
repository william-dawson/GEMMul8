import numpy as np
import fphistogram as fphist

a = np.random.rand(100000)

for base in [2, 10]:
    print("base = ", base)
    fphist.print_histogram(a, base=base)
