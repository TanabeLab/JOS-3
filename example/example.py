# -*- coding: utf-8 -*-
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
model.To = 20  # Change only operative temperature
model.simulate(60)  # Additional exposre time = 60 [min]

df = pd.DataFrame(model.dict_results())  # Make pandas.DataFrame
df.TskMean.plot()  # Show the graph of mean skin temp.