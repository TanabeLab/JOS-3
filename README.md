# jos

jos is a pyhon package to simulate a human thermoregulation and thermal responses.

Please cite us if you use this package : Y. Takahashi, A. Nomoto, S. Yoda, R. Hisayama, M. Ogata, Y. Ozeki, S-i. Tanabe,Thermoregulation Model JOS-3 with New Open Source Code, Energy & Buildings (2020), doi: https://doi.org/10.1016/j.enbuild.2020.110575

# Requirement

* python 3
* numpy

# Installation

```bash
pip install jos ??
```

# Usage

```python

import pandas as pd
import jos

model = jos.JOS(height=1.7, weight=60, age=30)  # Builds a model

# Set the first condition
model.To = 28  # Operative temperature [oC]
model.RH = 40  # Relative humidity [%]
model.Va = 0.2  # Air velocity [m/s]
model.PAR = 1.2  # Physical activity ratio [-]
model.simulate(60)  # Exposre time = 60 [min]

# Set the next condition
model.To = 20  # Changes only operative temperature
model.simulate(60)  # Additional exposre time = 60 [min]

df = pd.DataFrame(model.dict_results())  # Make pandas.DataFrame
df.TskMean.plot()  # Show the graph of mean skin temp.
```

![result](https://raw.githubusercontent.com/yoshito-takahashi/jos-dev/tree/develop/example/ex_result.png)

# Author

* Yoshito Takahashi
* Former master student of Tanabe Laboratory, Waseda University
* takahashiyoshito64@gmail.com

# License
jos is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
