import numpy as np
import pandas as pd
from scipy import sparse


df = pd.DataFrame({'a': np.random.randint(0, 2, 5), 'b': np.random.randint(0, 2, 5), 'c': np.random.randint(0, 2, 5)})
csr = sparse.csr_matrix(df.values)
print(csr)
