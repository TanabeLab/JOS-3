# -*- coding: utf-8 -*-
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

# Exporting the results as csv
model.to_csv()

# Show the documentaion of the output parameters
jos3.show_outparam_docs()

# Check basal metabolic rate [W/m2] using Getters
model.BMR