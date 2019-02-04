import numpy as np
import pandas as pd
from scipy import sparse


a = np.random.dirichlet(alpha=np.ones(5), size=1).flatten()
print(a)
