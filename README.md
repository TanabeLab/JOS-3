# Joint system thermoregulation model (JOS-3)

[Joint system thermoregulation model (JOS-3)](https://www.sciencedirect.com/science/article/pii/S0378778820333612)
is a numerical model to simulate human thermal physiology such as skin temperature, core temperature, 
sweating rate, and so on at 17 local body parts as well as the whole body.

This model was developed at [Shin-ichi Tanabe Laboratory, Waseda University](https://www.tanabe.arch.waseda.ac.jp/en/) 
and was derived from [65 Multi-Node model](https://doi.org/10.1016/S0378-7788(02)00014-2) 
and [JOS-2 model](https://doi.org/10.1016/j.buildenv.2013.04.013).

Please cite us if you use this package and describe which version you used: 
Y. Takahashi, A. Nomoto, S. Yoda, R. Hisayama, M. Ogata, Y. Ozeki, S. Tanabe,
Thermoregulation Model JOS-3 with New Open Source Code, Energy & Buildings (2020), 
doi: https://doi.org/10.1016/j.enbuild.2020.110575

# Note

Please also check [pythermalcomfort](https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort) :
F. Tartarini, S. Schiavon, pythermalcomfort: A Python package for thermal comfort research, SoftwareX (2020), 
doi: https://doi.org/10.1016/j.softx.2020.100578.

# Requirement

* python3
* numpy

# Documentation
https://pythermalcomfort.readthedocs.io/

# Installation

You can install the model with:
```bash
pip install jos3
```

If you have not installed numpy in your environment, please do so with:

```bash
pip install numpy
```

# Example

## Step 0: Import packages

```python
import jos3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Step 1: Build model and set body built

As a first step, you need to build a model and set a body built that you want to simulate.

### Parameters for JOS3 class:

* height (float, optional) : Body height [m]. The default is 1.72.
* weight (float, optional) : Body weight [kg]. The default is 74.43.
* fat (float, optional) : Fat percentage [%]. The default is 15.
* age (int, optional) : Age [years]. The default is 20.
* sex (str, optional) : Sex ("male" or "female"). The default is "male".
* ci (float, optional) : Cardiac index [L/min/m2]. The default is 2.6432.
* bmr_equation (str, optional) : BMR equation. The default is "harris-benedict". 
  * To use the equation for Japanese, type "japanese".
* bsa_equation (str, optional) : BSA equation. The default is "dubois".
  * You can choose "dubois", "fujimoto", "kruazumi", "takahira".
* ex_output (list/int, optional) : Extra output. The default is "None", 
  which outputs only important parameters such as local skin temperatures or core temperature. 
  * Set the parameters as the list format.
    (for example, if you want to see the data of ) ["BFsk", "BFcr", "Tar"]. 
  * If you want to see the all outputs, set ex_output to "all".

### Example code to built a model and set body buit

```python
model = jos3.JOS3(height=1.7,
                  weight=60,
                  fat=20,
                  age=30,
                  sex="male",
                  bmr_equation="japanese",
                  bsa_equation="fujimoto",
                  ex_output="all"
                  )
```

## Step 2: Set environmental conditions

Next, you need to set thermal environmental conditions that you want to simulate.

If you want to simulate non-uniform thermal environment, 
use numpy.ndarray (or list-like data) and input the data separately to local bodies.
You can also input a clothing insulation value for each body part individually as well as for the whole body.

If you want to simulate transient thermal environment, 
alternate between entering environmental information and executing the simulate() method.
After the simulate() method is executed, the environment input values are inherited, 
so you only need to enter the input parameters that you want to change.

### Environmental parameters

Input parameters of environmental conditions are set as the Setter format.

If you set the different conditions in each body parts, set them as a list-type object.

List-type input must be 17 lengths and means the input of "Head", "Neck", "Chest", 
"Back", "Pelvis", "Left-Shoulder", "Left-Arm", "Left-Hand", "Right-Shoulder", "Right-Arm",
"Right-Hand", "Left-Thigh", "Left-Leg", "Left-Foot", "Right-Thigh", "Right-Leg" and "Right-Foot".

* Ta (float or list) : Air temperature [oC].
* Tr (float or list) : Mean radiant temperature [oC].
* To (float or list) : Operative temperature [oC]. 
  This parameter can be input only when air temperature and mean radiant temperature are equal.
* Va (float or list) : Air velocity [m/s].
* RH (float or list) : Relative humidity [%].
* Icl (float or list) : Clothing insulation [clo].
  * [Reference for clothing insulation for the whole body](https://pythermalcomfort.readthedocs.io/en/latest/reference/pythermalcomfort.html#clothing-insulation-of-typical-ensembles-clo)
  * Reference for local clothing insulation: [A.Nomoto et al. (2019)](https://onlinelibrary.wiley.com/doi/full/10.1002/2475-8876.12124)
* PAR (float) Physical activity ratio [-]. The default is 1.2. 
  * This equals the ratio of metabolic rate to basal metabolic rate. 
  * PAR is for calculation metabolic rate considering personal characteristics such as gender or age.
  * If you want to input a specific value of metabolic rate like 58.2 W/m2, check the basal metabolic rate
    for the simulated people using Getter (there is an example at the bottom of this document), 
    and set PAR such that the metabolic rate is 58.2 W/m2. 
  * PAR of sitting quietly is 1.2.
* posture (str) : posture [-]. The default is "standing".
  * choose posture from "standing", "sitting" or "lying". 
  * This parameter affects convective and radiant heat transfer coefficients for local body parts

### Example code to simulate non-uniform and transient conditions
```python
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
```

## Step 3: How to output

As explained above, output parameters can be added arbitrarily by setting ex_output in list format when creating JOS objects.
The output parameters are suffixed with "Head," "Neck," "Chest," etc. for each body part.

### Defalt output parameters

* CO      : Cardiac output (the sum of the whole blood flow) [L/h]
* CycleTime: The counts of executing one cycle calculation [-]
* Met     : Total heat production of the whole body [W]
* ModTime : Simulation times [sec]
* RES     : Heat loss by the respiration [W]
* THLsk   : Heat loss from the skin of the body part [W]
* Tcr     : Core temperature of the body part [oC]
* Tsk     : Skin temperature of the body part [oC]
* TskMean : Mean skin temperature of the body [oC]
* Wet     : Local skin wettedness of the body part [-]
* WetMean : Mean skin wettedness of the body [-]
* Wle     : Weight loss rate by the evaporation and respiration of the whole body [g/sec]
* dt      : Time delta of the model [sec]

```python
# Show the results
df = pd.DataFrame(model.dict_results())  # Make pandas.DataFrame
df.TskMean.plot()  # Plot time series of mean skin temperature.
plt.show('example.png') # Show the plot
```

![result](https://raw.githubusercontent.com/TanabeLab/JOS-3/Akihisa_2023-03-07/example/example.png)


```python
# Exporte the results as csv
model.to_csv('example.csv')
```

### Extra output parameters

* Age     : Age [years]
* BFava_foot: AVA blood flow rate of one foot [L/h]
* BFava_hand: AVA blood flow rate of one hand [L/h]
* BFcr    : Core blood flow rate of the body part [L/h]
* BFfat   : Fat blood flow rate of the body part [L/h]
* BFms    : Muscle blood flow rate of the body part [L/h]
* BFsk    : Skin blood flow rate of the body part [L/h]
* BSA     : Body surface area of the body part [m2]
* Emax    : Maximum evaporative heat loss at the skin of th body part [W]
* Esk     : Evaporative heat loss at the skin of the body part [W]
* Esweat  : Evaporative heat loss at the skin by only sweating of the body part [W]
* Fat     : Body fat rate [%]
* Height  : Body heigh [m]
* Icl     : Clothing insulation value of the body part [clo]
* LHLsk   : Latent heat loss at the skin of the body part [W]
* Mbasecr : Core heat production by basal metaborism of th body part [W]
* Mbasefat: Fat heat production by basal metaborism of th body part [W]
* Mbasems : Muscle heat production by basal metaborism of th body part [W]
* Mbasesk : Skin heat production by basal metaborism of th body part [W]
* Mnst    : Core heat production by non-shivering of the body part [W]
* Mshiv   : Core or muscle heat production by shivering of th body part [W]
* Mwork   : Core or muscle heat production by work of the body part [W]
* Name    : Name of the model [-]
* PAR     : Physical activity ratio [-]
* Qcr     : Core total heat production of the body part [W]
* Qfat    : Fat total heat production of the body part [W]
* Qms     : Muscle total heat production of the body part [W]
* Qsk     : Skin total heat production of the body part [W]
* RESlh   : Latent heat loss by respiration of the body part [W]
* RESsh   : Sensible heat loss by respiration of the body part [W]
* RH      : Relative humidity of the body part [%]
* Ret     : Total evaporative heat resistance of the body part [m2.kPa/W]
* Rt      : Total heat resistance of the body part [m2.K/W]
* SHLsk   : Sensible heat loss at the skin of the body part [W]
* Setptcr : Set point skin temperatre of the body part [oC]
* Setptsk : Set point core temperatre of the body part [oC]
* Sex     : Male or female [-]
* Ta      : Air temperature of the body part [oC]
* Tar     : Arterial temperature of the body part [oC]
* Tcb     : Central blood temperature [oC]
* Tfat    : Fat temperature of the body part [oC]
* Tms     : Muscle temperature as the body part [oC]
* To      : Operative temperature of the body part [oC]
* Tr      : Mean radiant temperature of the body part [oC]
* Tsve    : Superfical vein temperature of the body part [oC]
* Tve     : Vein temperature of the body part [oC]
* Va      : Air velocity of the body part [m/s]
* Weight  : Body weight [kg]

### Example code to check the output parameters
```python
# Show the documentaion of the output parameters
print(jos3.show_outparam_docs())
```

## Getter
JOS3 has some useful getters to check the current parameters.

### Getter parameters

* BSA (numpy.ndarray (17,)) : Body surface areas by local body segments [m2]. 
* Rt (numpy.ndarray (17,)) : Dry heat resistances between the skin and ambience areas by local body segments [K.m2/W].
* Ret (numpy.ndarray (17,)) : Wet (Evaporative) heat resistances between the skin and ambience areas by local body segments [Pa.m2/W]. 
* Wet (numpy.ndarray (17,)) : Skin wettedness on local body segments [-]. 
* WetMean (float) : Mean skin wettedness of the whole body [-]. 
* TskMean (float) : Mean skin temperature of the whole body [oC]. 
* Tsk (numpy.ndarray (17,)) : Skin temperatures by the local body segments [oC]. 
* Tcr (numpy.ndarray (17,)) : Core temperatures by the local body segments [oC]. 
* Tcb (numpy.ndarray (1,)) : Central blood pool temperatures [oC]. 
* Tar (numpy.ndarray (17,)) : Arterial temperatures by the local body segments [oC]. 
* Tve (numpy.ndarray (17,)) : Vein temperatures by the local body segments [oC]. 
* Tsve (numpy.ndarray (12,)) : Superfical vein temperatures by the local body segments [oC]. 
* Tms (numpy.ndarray (2,)) : Muscle temperatures of Head and Pelvis [oC]. 
* Tfat (numpy.ndarray (2,)) : Fat temperatures of Head and Pelvis  [oC]. 
* BMR (float) : Basal metabolic rate [W/m2].

### Example code

```python
# Check basal metabolic rate [W/m2] using Getters
model.BMR
```

# Contact

* Yoshito Takahashi (takahashiyoshito64@gmail.com)
* Akihisa Nomoto (monyo323232@gmail.com)

# License

jos3 is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).