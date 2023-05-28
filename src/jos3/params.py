# -*- coding: utf-8 -*-

import textwrap

ALL_OUT_PARAMS = {
    'Age': {'ex_output': True,
         'meaning': 'Age',
         'suffix': None,
         'unit': 'years'},

    'BFava_foot': {'ex_output': True,
                   'meaning': 'AVA blood flow rate of one foot',
                   'suffix': None,
                   'unit': 'L/h'},

    'BFava_hand': {'ex_output': True,
                   'meaning': 'AVA blood flow rate of one hand',
                   'suffix': None,
                   'unit': 'L/h'},

    'BFcr': {'ex_output': True,
             'meaning': 'Core blood flow rate of the body part',
             'suffix': 'Body name',
             'unit': 'L/h'},

    'BFfat': {'ex_output': True,
              'meaning': 'Fat blood flow rate of the body part',
              'suffix': 'Body name',
              'unit': 'L/h'},

    'BFms': {'ex_output': True,
             'meaning': 'Muscle blood flow rate of the body part',
             'suffix': 'Body name',
             'unit': 'L/h'},

    'BFsk': {'ex_output': True,
             'meaning': 'Skin blood flow rate of the body part',
             'suffix': 'Body name',
             'unit': 'L/h'},

    'BSA': {'ex_output': True,
            'meaning': 'Body surface area of the body part',
            'suffix': 'Body name',
            'unit': 'm2'},

    'CO': {'ex_output': False,
           'meaning': 'Cardiac output (the sum of the whole blood flow)',
           'suffix': None,
           'unit': 'L/h'},

    'CycleTime': {'ex_output': False,
                  'meaning': 'The counts of executing one cycle calculation',
                  'suffix': None,
                  'unit': '-'},

    'Emax': {'ex_output': True,
             'meaning': 'Maximum evaporative heat loss at the skin of th body '
                        'part',
             'suffix': 'Body name',
             'unit': 'W'},

    'Esk': {'ex_output': True,
            'meaning': 'Evaporative heat loss at the skin of the body part',
            'suffix': 'Body name',
            'unit': 'W'},

    'Esweat': {'ex_output': True,
               'meaning': 'Evaporative heat loss at the skin by only sweating of '
                          'the body part',
               'suffix': 'Body name',
               'unit': 'W'},

    'Fat': {'ex_output': True,
            'meaning': 'Body fat rate',
            'suffix': None,
            'unit': '%'},

    'Height': {'ex_output': True,
               'meaning': 'Body heigh',
               'suffix': None,
               'unit': 'm'},

    'Icl': {'ex_output': True,
            'meaning': 'Clothing insulation value of the body part',
            'suffix': 'Body name',
            'unit': 'clo'},

    'LHLsk': {'ex_output': True,
              'meaning': 'Latent heat loss at the skin of the body part',
              'suffix': 'Body name',
              'unit': 'W'},

    'Mbasecr': {'ex_output': True,
                'meaning': 'Core heat production by basal metaborism of th body '
                           'part',
                'suffix': 'Body name',
                'unit': 'W'},

    'Mbasefat': {'ex_output': True,
                 'meaning': 'Fat heat production by basal metaborism of th body '
                            'part',
                 'suffix': 'Body name',
                 'unit': 'W'},

    'Mbasems': {'ex_output': True,
                'meaning': 'Muscle heat production by basal metaborism of th body '
                           'part',
                'suffix': 'Body name',
                'unit': 'W'},

    'Mbasesk': {'ex_output': True,
                'meaning': 'Skin heat production by basal metaborism of th body '
                           'part',
                'suffix': 'Body name',
                'unit': 'W'},

    'Met': {'ex_output': False,
            'meaning': 'Total heat production of the whole body',
            'suffix': None,
            'unit': 'W'},

    'Mnst': {'ex_output': True,
             'meaning': 'Core heat production by non-shivering of the body part',
             'suffix': 'Body name',
             'unit': 'W'},

    'ModTime': {'ex_output': False,
                'meaning': 'Simulation times',
                'suffix': None,
                'unit': 'sec'},

    'Mshiv': {'ex_output': True,
              'meaning': 'Core or muscle heat production by shivering of th body '
                         'part',
              'suffix': 'Body name',
              'unit': 'W'},

    'Mwork': {'ex_output': True,
              'meaning': 'Core or muscle heat production by work of the body part',
              'suffix': 'Body name',
              'unit': 'W'},

    'Name': {'ex_output': True,
             'meaning': 'Name of the model',
             'suffix': None,
             'unit': '-'},

    'PAR': {'ex_output': True,
            'meaning': 'Physical activity ratio',
            'suffix': None,
            'unit': '-'},

    'Qcr': {'ex_output': True,
            'meaning': 'Core total heat production of the body part',
            'suffix': 'Body name',
            'unit': 'W'},

    'Qfat': {'ex_output': True,
             'meaning': 'Fat total heat production of the body part',
             'suffix': 'Body name',
             'unit': 'W'},

    'Qms': {'ex_output': True,
            'meaning': 'Muscle total heat production of the body part',
            'suffix': 'Body name',
            'unit': 'W'},

    'Qsk': {'ex_output': True,
            'meaning': 'Skin total heat production of the body part',
            'suffix': 'Body name',
            'unit': 'W'},

    'RES': {'ex_output': False,
            'meaning': 'Heat loss by the respiration',
            'suffix': None,
            'unit': 'W'},

    'RESlh': {'ex_output': True,
              'meaning': 'Latent heat loss by respiration of the body part',
              'suffix': 'Body name',
              'unit': 'W'},
    'RESsh': {'ex_output': True,
              'meaning': 'Sensible heat loss by respiration of the body part',
              'suffix': 'Body name',
              'unit': 'W'},

    'RH': {'ex_output': True,
           'meaning': 'Relative humidity of the body part',
           'suffix': 'Body name',
           'unit': '%'},

    'Ret': {'ex_output': True,
            'meaning': 'Total evaporative heat resistance of the body part',
            'suffix': 'Body name',
            'unit': 'm2.kPa/W'},

    'Rt': {'ex_output': True,
           'meaning': 'Total heat resistance of the body part',
           'suffix': 'Body name',
           'unit': 'm2.K/W'},

    'SHLsk': {'ex_output': True,
              'meaning': 'Sensible heat loss at the skin of the body part',
              'suffix': 'Body name',
              'unit': 'W'},

    'Setptcr': {'ex_output': True,
                'meaning': 'Set point skin temperatre of the body part',
                'suffix': 'Body name',
                'unit': 'oC'},

    'Setptsk': {'ex_output': True,
                'meaning': 'Set point core temperatre of the body part',
                'suffix': 'Body name',
                'unit': 'oC'},

    'Sex': {'ex_output': True,
            'meaning': 'Male or female',
            'suffix': None,
            'unit': '-'},

    'THLsk': {'ex_output': False,
              'meaning': 'Heat loss from the skin of the body part',
              'suffix': 'Body name',
              'unit': 'W'},

    'Ta': {'ex_output': True,
           'meaning': 'Air temperature of the body part',
           'suffix': 'Body name',
           'unit': 'oC'},

    'Tar': {'ex_output': True,
            'meaning': 'Arterial temperature of the body part',
            'suffix': 'Body name',
            'unit': 'oC'},

    'Tcb': {'ex_output': True,
            'meaning': 'Central blood temperature',
            'suffix': None,
            'unit': 'oC'},

    'Tcr': {'ex_output': False,
            'meaning': 'Core temperature of the body part',
            'suffix': 'Body name',
            'unit': 'oC'},

    'Tfat': {'ex_output': True,
             'meaning': 'Fat temperature of the body part',
             'suffix': 'Body name',
             'unit': 'oC'},

    'Tms': {'ex_output': True,
            'meaning': 'Muscle temperature as the body part',
            'suffix': 'Body name',
            'unit': 'oC'},

    'To': {'ex_output': True,
           'meaning': 'Operative temperature of the body part',
           'suffix': 'Body name',
           'unit': 'oC'},

    'Tr': {'ex_output': True,
           'meaning': 'Mean radiant temperature of the body part',
           'suffix': 'Body name',
           'unit': 'oC'},

    'Tsk': {'ex_output': False,
            'meaning': 'Skin temperature of the body part',
            'suffix': 'Body name',
            'unit': 'oC'},

    'TskMean': {'ex_output': False,
                'meaning': 'Mean skin temperature of the body',
                'suffix': None,
                'unit': 'oC'},

    'Tsve': {'ex_output': True,
             'meaning': 'Superfical vein temperature of the body part',
             'suffix': 'Body name',
             'unit': 'oC'},

    'Tve': {'ex_output': True,
            'meaning': 'Vein temperature of the body part',
            'suffix': 'Body name',
            'unit': 'oC'},

    'Va': {'ex_output': True,
           'meaning': 'Air velocity of the body part',
           'suffix': 'Body name',
           'unit': 'm/s'},

    'Weight': {'ex_output': True,
               'meaning': 'Body weight',
               'suffix': None,
               'unit': 'kg'},

    'Wet': {'ex_output': False,
            'meaning': 'Local skin wettedness of the body part',
            'suffix': 'Body name',
            'unit': '-'},

    'WetMean': {'ex_output': False,
                'meaning': 'Mean skin wettedness of the body',
                'suffix': None,
                'unit': '-'},

    'Wle': {'ex_output': False,
            'meaning': 'Weight loss rate by the evaporation and respiration of '
                       'the whole body',
            'suffix': None,
            'unit': 'g/sec'},

    'dt': {'ex_output': False,
        'meaning': 'Time delta of the model',
        'suffix': None,
        'unit': 'sec'}}


def show_outparam_docs():
    """
    Show the documentation of the output parameters.

    Returns
    -------
    docstirng : str
        Text of the documentation of the output parameters

    """


    outparams = textwrap.dedent("""
    Output parameters
    -------
    """)

    exoutparams = textwrap.dedent("""
    Extra output parameters
    -------
    """)

    sortkeys = list(ALL_OUT_PARAMS.keys())
    sortkeys.sort()
    for key in sortkeys:
        value = ALL_OUT_PARAMS[key]

        line = "{}: {} [{}]".format(key.ljust(8), value["meaning"], value["unit"])

        if value["ex_output"]:
            exoutparams += line + "\n"
        else:
            outparams += line + "\n"

    docs = outparams + "\n" + exoutparams
    docs = textwrap.indent(docs.strip(), "    ")

    return docs

if __name__ == "__main__":
    show_outparam_docs()