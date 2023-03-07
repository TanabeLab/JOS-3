# JOS-3

JOS-3 is a thermoregulation model to simulate human thermal physiology such as skin temperature, core temperature, sweating rate, and so on at 17 local body parts as well as the whole body.

This model was developed at [Tanabe Laboratory, Waseda University](https://www.tanabe.arch.waseda.ac.jp/en/) and was derived from 65MN (https://doi.org/10.1016/S0378-7788(02)00014-2) and JOS-2 model (https://doi.org/10.1016/j.buildenv.2013.04.013).

Please cite us if you use this package and describe which version you use : Y. Takahashi, A. Nomoto, S. Yoda, R. Hisayama, M. Ogata, Y. Ozeki, S. Tanabe,Thermoregulation Model JOS-3 with New Open Source Code, Energy & Buildings (2020), doi: https://doi.org/10.1016/j.enbuild.2020.110575

# Note

Please also check [pythermalcomfort](https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort) : F. Tartarini, S. Schiavon, pythermalcomfort: A Python package for thermal comfort research, SoftwareX (2020), doi: https://doi.org/10.1016/j.softx.2020.100578 .


# Requirement

* python3
* numpy

# Documentation
https://pythermalcomfort.readthedocs.io/

# Installation

```bash
pip install jos3
```


If you have not installed numpy in your environment, do the following.

```bash
pip install numpy
```

# Example 1

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
print(jos3.show_outparam_docs())

# Check basal metabolic rate [W/m2] using Getters
model.BMR
```

# Example 2

"""
1. Builds a model and set a body built
-------

As a first step, you need to build a model and set a body built that you want to simulate.

The following are the parameters for JOS3 class.

Parameters of body built
-------
* height : float, optional
    Body height [m]. The default is 1.72.

* weight : float, optional
    Body weight [kg]. The default is 74.43.

* fat : float, optional
** Fat percentage [%]. The default is 15.

* age : int, optional
    Age [years]. The default is 20.

* sex : str, optional
    Sex ("male" or "female"). The default is "male".

* ci : float, optional
    Cardiac index [L/min/m2]. The default is 2.6432.

* bmr_equation : str, optional
    Choose a BMR equation. The default is "harris-benedict". 
    To use the equation for Japanese, enter "japanese".

* bsa_equation : str, optional
    Choose a BSA equation.
    You can choose "dubois", "fujimoto", "kruazumi", "takahira".
    The default is "dubois".

* ex_output : None, list or "all", optional
    If you want to get extra output parameters, set the parameters as the list format like ["BFsk", "BFcr", "Tar"].
    If ex_output is "all", all parameters are output.
    The default is None, which outputs only important parameters such as local skin temperatures. 
"""
```python
import jos3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

model = jos3.JOS3(height=1.7,
                  weight=60,
                  fat=20,
                  age=30,
                  sex="male",
                  bmr_equation="japanese",
                  bsa_equation="fujimoto",
                  ex_output=None
                  )

"""
2. Set environmental conditions

Next, you need to set thermal environmental conditions that you want to simulate.

If you want to simulate non-uniform thermal environment, use numpy.ndarray and input the data separately to local bodies.
You can also input a clothing insulation value for each body part individually as well as for the whole body.

If you want to simulate transient thermal environment, 
alternate between entering environmental information and executing the simulate() method.
After the simulate() method is executed, the environment input values are inherited, 
so you only need to enter the input parameters that you want to change.

The following are the parameters for JOS3 class.

Setter & Getter
-------
Input parameters of environmental conditions are set as the Setter format.
If you set the different conditons in each body parts, set the list.
List input must be 17 lengths and means the input of "Head", "Neck", "Chest",
"Back", "Pelvis", "LShoulder", "LArm", "LHand", "RShoulder", "RArm",
"RHand", "LThigh", "LLeg", "LFoot", "RThigh", "RLeg" and "RFoot".

Ta : float or list
    Air temperature [oC].
Tr : float or list
    Mean radiant temperature [oC].
To : float or list
    Operative temperature [oC].
    This parameter can be input only when air temperature and mean radiant temperature are equal.
Va : float or list
    Air velocity [m/s].
RH : float or list
    Relative humidity [%].
Icl : float or list
    Clothing insulation [clo].
PAR : float
    Physical activity ratio [-].
    This equals the ratio of metabolic rate to basal metabolic rate.
    PAR of sitting quietly is 1.2.
    The default is 1.2.
posture : str
    Choose a posture from "standing", "sitting" or "lying". 
    This parameter affects convective and radiant heat transfer coefficients for local body parts
    The default is "standing".
bodytemp : numpy.ndarray (85,)
    All segment temperatures of JOS-3

Getter
-------
JOS3 has some useful getters to check the current parameters.

BSA : numpy.ndarray (17,)
    Body surface areas by local body segments [m2].
Rt : numpy.ndarray (17,)
    Dry heat resistances between the skin and ambience areas by local body segments [K.m2/W].
Ret : numpy.ndarray (17,)
    Wet (Evaporative) heat resistances between the skin and ambience areas by local body segments [Pa.m2/W].
Wet : numpy.ndarray (17,)
    Skin wettedness on local body segments [-].
WetMean : float
    Mean skin wettedness of the whole body [-].
TskMean : float
    Mean skin temperature of the whole body [oC].
Tsk : numpy.ndarray (17,)
    Skin temperatures by the local body segments [oC].
Tcr : numpy.ndarray (17,)
    Skin temperatures by the local body segments [oC].
Tcb : numpy.ndarray (1,)
    Core temperatures by the local body segments [oC].
Tar : numpy.ndarray (17,)
    Arterial temperatures by the local body segments [oC].
Tve : numpy.ndarray (17,)
    Vein temperatures by the local body segments [oC].
Tsve : numpy.ndarray (12,)
    Superfical vein temperatures by the local body segments [oC].
Tms : numpy.ndarray (2,)
    Muscle temperatures of Head and Pelvis [oC].
Tfat : numpy.ndarray (2,)
    Fat temperatures of Head and Pelvis  [oC].
BMR : float
    Basal metabolic rate [W/m2].
"""

# Set the first condition
model.Ta = 28  # Air temperature [oC]
model.Tr = 30  # Mean radiant temperature [oC]
model.RH = 40  # Relative humidity [%]
model.Va = 0.2  # Air velocity [m/s]
model.PAR = 1.2  # Physical activity ratio [-], assuming a sitting position
model.posture = 'sitting' # Posture [-], assuming a sitting position
model.Icl = np.array([ # Clothing insulation [clo]
    0.00, # Head
    0.00, # Neck
    1.14, # Chest
    0.84, # Back
    1.04, # Pelvis
    0.84, # Left-Shoulder
    0.42, # Left-Arm
    0.00, # Left-Hand
    0.84, # Right-Shoulder
    0.42, # Right-Arm
    0.00, # Right-Hand
    0.58, # Left-Thigh
    0.62, # Left-Leg
    0.82, # Left-Foot
    0.58, # Right-Thigh
    0.62, # Right-Leg
    0.82, # Right-Foot
])
# Execute JOS-3 model
model.simulate(times=30, # Number of loops of a simulation
               dtime=60, # Time delta [sec]. The default is 60.
               )  # Exposure time = 30 [loops] * 60 [sec] = 30 [min]

# Set the next condition (You only need to change the parameters that you want to change)
model.To = 20  # Change operative temperature
model.Va = np.array([ # Air velocity [m/s], assuming to use a desk fan
    0.2, # Head
    0.4, # Neck
    0.4, # Chest
    0.1, # Back
    0.1, # Pelvis
    0.4, # Left-Shoulder
    0.4, # Left-Arm
    0.4, # Left-Hand
    0.4, # Right-Shoulder
    0.4, # Right-Arm
    0.4, # Right-Hand
    0.1, # Left-Thigh
    0.1, # Left-Leg
    0.1, # Left-Foot
    0.1, # Right-Thigh
    0.1, # Right-Leg
    0.1, # Right-Foot
])
# Execute JOS-3 model
model.simulate(times=60, # Number of loops of a simulation
               dtime=60, # Time delta [sec]. The default is 60.
               ) # Additional exposure time = 60 [loops] * 60 [sec] = 60 [min]

# Set the next condition (You only need to change the parameters that you want to change)
model.Ta = 30  # Change air temperature [oC]
model.Tr = 35  # Change mean radiant temperature [oC]
# Execute JOS-3 model
model.simulate(times=30, # Number of loops of a simulation
               dtime=60, # Time delta [sec]. The default is 60.
               ) # Additional exposure time = 30 [loops] * 60 [sec] = 30 [min]

# Show the results
df = pd.DataFrame(model.dict_results())  # Make pandas.DataFrame
df.TskMean.plot()  # Plot time series of mean skin temperature.
plt.show('example2.png') # Show the plot

# Exporting the results as csv
model.to_csv('example2.csv')

# Show the documentaion of the output parameters
print(jos3.show_outparam_docs())
```

# Contact

* Yoshito Takahashi (takahashiyoshito64@gmail.com)
* 

# License

jos3 is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).