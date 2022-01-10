# JOS-3

JOS-3 is a model to simulate a human thermoregulation.

The model has been derived from 65MN https://doi.org/10.1016/S0378-7788(02)00014-2 and JOS-2 https://doi.org/10.1016/j.buildenv.2013.04.013 model.

Please cite us if you use this package : Y. Takahashi, A. Nomoto, S. Yoda, R. Hisayama, M. Ogata, Y. Ozeki, S. Tanabe,Thermoregulation Model JOS-3 with New Open Source Code, Energy & Buildings (2020), doi: https://doi.org/10.1016/j.enbuild.2020.110575

# Note

Our work is packed in [pythermalcomfort](https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort).

Please also check pythermalcomfort project: F. Tartarini, S. Schiavon, pythermalcomfort: A Python package for thermal comfort research, SoftwareX (2020), doi: https://doi.org/10.1016/j.softx.2020.100578 .


# Requirement

* python3
* numpy

# Installation

```bash
pip install jos3
```

or

```bash
pip install git+https://github.com/TanabeLab/JOS-3.git
```

If you have not installed numpy in your environment, do the following

```bash
pip install numpy
```

# Usage

```python
import jos3

model = jos3.JOS3(height=1.7, weight=60, age=30)  # Builds a model

# Set the first condition
model.To = 28  # Operative temperature [oC]
model.RH = 40  # Relative humidity [%]
model.Va = 0.2  # Air velocity [m/s]
model.PAR = 1.2  # Physical activity ratio [-]
model.simulate(60)  # Exposre time = 60 [min]

# Set the next condition
model.To = 20  # Changes only operative temperature
model.simulate(60)  # Additional exposre time = 60 [min]

# Show the results
import pandas as pd
df = pd.DataFrame(model.dict_results())  # Make pandas.DataFrame
df.TskMean.plot()  # Show the graph of mean skin temp.
```
![result](https://raw.githubusercontent.com/TanabeLab/JOS-3/master/example/ex_result.png)

```python
# Exporting the results as csv
model.to_csv(folder="C:/Users/takahashi/Desktop")

# Show the documentaion of the output parameters
model.show_outparam_docs()

# Check basal metabolic rate [W/m2] using Getters
model.BMR
```


# Author

* Yoshito Takahashi
* Finished master's degree at [Tanabe Laboratory, Waseda University](https://www.tanabe.arch.waseda.ac.jp/en/)
* takahashiyoshito64@gmail.com

# License

jos3 is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
