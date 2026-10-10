
import sys

import numpy as np
import pandas as pd
import sklearn
import matplotlib
import seaborn

print("Python executable:", sys.executable)
print("Python version:", sys.version.split()[0])
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Seaborn version:", seaborn.__version__)

# Basic calculation test
values = np.array([1, 2, 3, 4, 5])
print("NumPy mean:", values.mean())

# Basic DataFrame test
df = pd.DataFrame({
    "review": [
        "The hotel was excellent",
        "The room was dirty"
    ],
    "label": [
        "POSITIVE",
        "NEGATIVE"
    ]
})

print("\nSample DataFrame:")
print(df)

print("\nSetup test passed!")
