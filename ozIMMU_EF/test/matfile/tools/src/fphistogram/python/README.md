# Python module

## Installation
```bash
git clone https://github.com/enp1s0/fphistogram
cd fphistogram/python
pip install .
```

## Examle
```python
import numpy as np
import fphistogram as fphist

a = np.random.rand(100000)
fphist.print_histogram(a)
```
