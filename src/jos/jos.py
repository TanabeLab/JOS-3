# -*- coding: utf-8 -*-
import csv
import datetime as dt
import os

import numpy as np
# Import from relative path
try:
    from . import thermoregulation as threg
    from . import matrix
    from .matrix import NUM_NODES, INDEX, VINDEX, BODY_NAMES
    from .comfmod import preferred_temp
    from . import construction as cons
    from .construction import _BSAst
# Import from absolute path
# These codes are for debugging
except ImportError:
    from jos import thermoregulation as threg
    from jos import matrix
    from jos.matrix import NUM_NODES, INDEX, VINDEX, BODY_NAMES
    from jos.comfmod import preferred_temp
    from jos import construction as cons
    from jos.construction import _BSAst


class JOS():
    def __init__(
            self,
            height=1.72,
            weight=74.43,
            fat=15,
            age=20,
            sex="male",
            ci=2.59,
            bmr_equation="harris-benedict",
            bsa_equation="dubois",
            ex_output=None,
            ):
        """
        Parameters
        ----------
        height : float, optional
            Body height [m]. The default is 1.72.
        weight : float, optional
            Body weight [kg]. The default is 74.43.
        fat : float, optional
            Fat rte [%]. The default is 15.
        age : int, optional
            Age [years]. The default is 20.
        sex : str, optional
            Sex ("male" or "female"). The default is "male".
        ci : float, optional
            Cardiac index [L/min/m2]. The default is 2.6432.
        bmr_equation : str, optional
            Choose a BMR equation. The default is "harris-benedict".
        bsa_equation : str, optional
            Choose a BSA equation. The default is "dubois".
        ex_output : None, list or "all", optional
            Extra output parameters. If "all", all parameters are output.
            The default is None.

        Returns
        -------
        None.
        """


        self._height = height
        self._weight = weight
        self._fat = fat
        self._sex = sex
        self._age = age
        self._ci = ci
        self._bmr_equation = bmr_equation
        self._bsa_equation = bsa_equation
        self._ex_output = ex_output

        # Body surface area [m2]
        self._bsa_rate = cons.bsa_rate(height, weight, bsa_equation,)
        # Body surface area rate [-]
        self._bsa = cons.localbsa(height, weight, bsa_equation,)
        # Basal blood flow rate [-]
        self._bfb_rate = cons.bfb_rate(height, weight, bsa_equation, age, ci)
        # Thermal conductance [W/K]
        self._cdt = cons.conductance(height, weight, bsa_equation, fat,)
        # Thermal capacity [J/K]
        self._cap = cons.capacity(height, weight, bsa_equation, age, ci)

        # Set point temp [oC]
        self.setpt_cr = np.ones(17)*37  # core
        self.setpt_sk = np.ones(17)*34  # skin

        # Initial body temp [oC]
        self._bodytemp = np.ones(NUM_NODES) * 36

        # Default values of input condition
        self._ta = np.ones(17)*28.8
        self._tr = np.ones(17)*28.8
        self._rh = np.ones(17)*50
        self._va = np.ones(17)*0.1
        self._clo = np.zeros(17)
        self._iclo = np.ones(17) * 0.45
        self._par = 1.25 # Physical activity ratio
        self._posture = "standing"
        self._hc = None
        self._hr = None
        self.ex_q = np.zeros(NUM_NODES)
        self._t = dt.timedelta(0) # Elapsed time
        self._cycle = 0 # Cycle time
        self.model_name = "JOS3"
        self.options = {
                "nonshivering_thermogenesis": True,
                "cold_acclimated": False,
                "shivering_threshold": False,
                "limit_dshiv/dt": False,
                "bat_positive": False,
                "ava_zero": False,
                "shivering": False,}
        threg.PRE_SHIV = 0 # reset
        self._history = []
        self._t = dt.timedelta(0) # Elapsed time
        self._cycle = 0 # Cycle time

        # Reset setpoint temperature
        dictout = self.reset_setpt()
        self._history.append(dictout)  # Save the last model parameters


    def reset_setpt(self):
        """
        Reset setpoint temperature by steady state calculation.
        Be careful, input parameters (Ta, Tr, RH, Va, Icl, PAR) and body
        tempertures are also resetted.

        Returns
        -------
        Parameters of JOS-3 : dict
        """
        # Set operative temperature under PMV=0 environment
        # Metabolic rate at PAR = 1.25
        # 1 met = 58.15 W/m2
        met = self.BMR * 1.25 / 58.15 / self.BSA.sum() # [met]
        self.To = preferred_temp(met=met)
        self.RH = 50
        self.Va = 0.1
        self.Icl = 0
        self.PAR = 1.25

        # Steady-calculation
        self.options["ava_zero"] = True
        for t in range(10):
            dictout = self._run(dtime=60000, passive=True)

        # Set new setpoint temperatures
        self.setpt_cr = self.Tcr
        self.setpt_sk = self.Tsk
        self.options["ava_zero"] = False

        return dictout


    def simulate(self, times, dtime=60, output=True):
        """
        Execute JOS3 model.

        Parameters
        ----------
        times : int
            Number of loops of a simulation
        dtime : int or float, optional
            Time delta [sec]. The default is 60.
        output : bool, optional
            If you don't record paramters, set False. The default is True.

        Returns
        -------
        None.

        """
        for t in range(times):
            self._t += dt.timedelta(0, dtime)
            self._cycle += 1
            dictdata = self._run(dtime=dtime, output=output)
            if output:
                # self.history.append(dictdata)
                self._history.append(dictdata)


    def _run(self, dtime=60, passive=False, output=True):
        """
        Run a model for a once and get model parameters.

        Parameters
        ----------
        dtime : int or float, optional
            Time delta [sec]. The default is 60.
        passive : bool, optional
            If you run a passive model, set True. The default is False.
        output : bool, optional
            If you don't need paramters, set False. The default is True.

        Returns
        -------
        dictout : dictionary
            Output parameters.

        """
        tcr = self.Tcr
        tsk = self.Tsk

        # Convective and radiative heat transfer coefficient [W/K.m2]
        hc = threg.fixed_hc(threg.conv_coef(self._posture, self._va, self._ta, tsk,), self._va)
        hr = threg.fixed_hr(threg.rad_coef(self._posture,))
        # Manual setting
        if self._hc is not None:
            hc = self._hc
        if self._hr is not None:
            hr = self._hr

        # Operarive temp. [oC], heat and evaporative heat resistance [K/W], [kPa/W]
        to = threg.operative_temp(self._ta, self._tr, hc, hr,)
        r_t = threg.dry_r(hc, hr, self._clo)
        r_et = threg.wet_r(hc, self._clo, self._iclo)

        #------------------------------------------------------------------
        # Thermoregulation
        #------------------------------------------------------------------
        # Setpoint temperature of thermoregulation
        if passive:
            setpt_cr = tcr.copy()
            setpt_sk = tsk.copy()
        else:
            setpt_cr = self.setpt_cr.copy()
            setpt_sk = self.setpt_sk.copy()
        # Difference between setpoint and body temperatures
        err_cr = tcr - setpt_cr
        err_sk = tsk - setpt_sk

        # Skinwettedness [-], Esk, Emax, Esw [W]
        wet, e_sk, e_max, e_sweat = threg.evaporation(
                err_cr, err_sk, tsk,
                self._ta, self._rh, r_et, 
                self._height, self._weight, self._bsa_equation, self._age)

        # Skin blood flow, basal skin blood flow [L/h]
        bf_sk = threg.skin_bloodflow(err_cr, err_sk,
            self._height, self._weight, self._bsa_equation, self._age, self._ci)

        # Hand, Foot AVA blood flow [L/h]
        bf_ava_hand, bf_ava_foot = threg.ava_bloodflow(err_cr, err_sk,
            self._height, self._weight, self._bsa_equation, self._age, self._ci)
        if self.options["ava_zero"] and passive:
            bf_ava_hand = 0
            bf_ava_foot = 0

        # Thermogenesis by shivering [W]
        mshiv = threg.shivering(
                err_cr, err_sk, tcr, tsk,
                self._height, self._weight, self._bsa_equation, self._age, self._sex, dtime,
                self.options,)

        # Thermogenesis by non-shivering [W]
        if self.options["nonshivering_thermogenesis"]:
            mnst = threg.nonshivering(err_cr, err_sk,
                self._height, self._weight, self._bsa_equation, self._age, 
                self.options["cold_acclimated"], self.options["bat_positive"])
        else: # not consider NST
            mnst = np.zeros(17)

        #------------------------------------------------------------------
        # Thermogenesis
        #------------------------------------------------------------------
        # Basal thermogenesis [W]
        mbase = threg.local_mbase(
                self._height, self._weight, self._age, self._sex,
                self._bmr_equation,)
        mbase_all = sum([m.sum() for m in mbase])

        # Thermogenesis by work [W]
        mwork = threg.local_mwork(mbase_all, self._par)

        # Sum of thermogenesis in core, muscle, fat, skin [W]
        qcr, qms, qfat, qsk = threg.sum_m(mbase, mwork, mshiv, mnst,)
        qall = qcr.sum() + qms.sum() + qfat.sum() + qsk.sum()

        #------------------------------------------------------------------
        # Other
        #------------------------------------------------------------------
        # Blood flow in core, muscle, fat [L/h]
        bf_cr, bf_ms, bf_fat = threg.crmsfat_bloodflow(mwork, mshiv,
            self._height, self._weight, self._bsa_equation, self._age, self._ci)

        # Heat loss by respiratory
        p_a = threg.antoine(self._ta)*self._rh/100
        res_sh, res_lh = threg.resp_heatloss(self._ta[0], p_a[0], qall)

        # Sensible heat loss [W]
        shlsk = (tsk - to) / r_t * self._bsa

        # Cardiac output [L/h]
        co = threg.sum_bf(
                bf_cr, bf_ms, bf_fat, bf_sk, bf_ava_hand, bf_ava_foot)

        # Weight loss rate by evaporation [g/sec]
        wlesk = (e_sweat + 0.06*e_max) / 2418
        wleres = res_lh / 2418

        #------------------------------------------------------------------
        # Matrix
        #------------------------------------------------------------------
        # Matrix A
        # (83, 83,) ndarray
        bf_art, bf_vein = matrix.vessel_bloodflow(
                bf_cr, bf_ms, bf_fat, bf_sk, bf_ava_hand, bf_ava_foot
                )
        bf_local = matrix.localarr(
                bf_cr, bf_ms, bf_fat, bf_sk, bf_ava_hand, bf_ava_foot
                )
        bf_whole = matrix.wholebody(
                bf_art, bf_vein, bf_ava_hand, bf_ava_foot
                )
        arr_bf = np.zeros((NUM_NODES,NUM_NODES))
        arr_bf += bf_local
        arr_bf += bf_whole

        arr_bf /= self._cap.reshape((NUM_NODES,1)) # Change unit [W/K] to [/sec]
        arr_bf *= dtime # Change unit [/sec] to [-]

        arr_cdt = self._cdt.copy()
        arr_cdt /= self._cap.reshape((NUM_NODES,1)) # Change unit [W/K] to [/sec]
        arr_cdt *= dtime # Change unit [/sec] to [-]

        arrB = np.zeros(NUM_NODES)
        arrB[INDEX["skin"]] += 1/r_t*self._bsa
        arrB /= self._cap # Change unit [W/K] to [/sec]
        arrB *= dtime # Change unit [/sec] to [-]

        arrA_tria = -(arr_cdt + arr_bf)

        arrA_dia = arr_cdt + arr_bf
        arrA_dia = arrA_dia.sum(axis=1) + arrB
        arrA_dia = np.diag(arrA_dia)
        arrA_dia += np.eye(NUM_NODES)

        arrA = arrA_tria + arrA_dia
        arrA_inv = np.linalg.inv(arrA)

        # Matrix Q [W] / [J/K] * [sec] = [-]
        # Thermogensis
        arrQ = np.zeros(NUM_NODES)
        arrQ[INDEX["core"]] += qcr
        arrQ[INDEX["muscle"]] += qms[VINDEX["muscle"]]
        arrQ[INDEX["fat"]] += qfat[VINDEX["fat"]]
        arrQ[INDEX["skin"]] += qsk

        # Respiratory [W]
        arrQ[INDEX["core"][2]] -= res_sh + res_lh #Chest core

        # Sweating [W]
        arrQ[INDEX["skin"]] -= e_sk

        # Extra heat gain [W]
        arrQ += self.ex_q.copy()

        arrQ /= self._cap # Change unit [W]/[J/K] to [K/sec]
        arrQ *= dtime # Change unit [K/sec] to [K]

        # Boundary batrix [℃]
        arr_to = np.zeros(NUM_NODES)
        arr_to[INDEX["skin"]] += to

        # all
        arr = self._bodytemp + arrB * arr_to + arrQ

        #------------------------------------------------------------------
        # New body temp. [oC]
        #------------------------------------------------------------------
        self._bodytemp = np.dot(arrA_inv, arr)

        #------------------------------------------------------------------
        # Output paramters
        #------------------------------------------------------------------
        dictout = {}
        if output:  # Default output
            dictout["CycleTime"] = self._cycle
            dictout["ModTime"] = self._t
            dictout["dt"] = dtime
            dictout["TskMean"] = self.TskMean
            dictout["Tsk"] = self.Tsk
            dictout["Tcr"] = self.Tcr
            dictout["WetMean"] = np.average(wet, weights=_BSAst)
            dictout["Wet"] = wet
            dictout["Wle"] = (wlesk.sum() + wleres)
            dictout["CO"] = co
            dictout["Met"] = qall
            dictout["Met"] = qall
            dictout["RES"] = res_sh + res_lh
            dictout["THLsk"] = shlsk + e_sk


        detailout = {}
        if self._ex_output:
            detailout["Name"] = self.model_name
            detailout["Height"] = self._height
            detailout["Weight"] = self._weight
            detailout["BSA"] = self._bsa
            detailout["Fat"] = self._fat
            detailout["Sex"] = self._sex
            detailout["Age"] = self._age
            detailout["Setptcr"] = setpt_cr
            detailout["Setptcr"] = setpt_sk
            detailout["Tcb"] = self.Tcb
            detailout["Tar"] = self.Tar
            detailout["Tve"] = self.Tve
            detailout["Tsve"] = self.Tsve
            detailout["Tms"] = self.Tms
            detailout["Tfat"] = self.Tfat
            detailout["To"] = to
            detailout["Rt"] = r_t
            detailout["Ret"] = r_et
            detailout["Ta"] = self._ta.copy()
            detailout["Tr"] = self._tr.copy()
            detailout["RH"] = self._rh.copy()
            detailout["Va"] = self._va.copy()
            detailout["PAR"] = self._par
            detailout["Icl"] = self._clo.copy()
            detailout["Esk"] = e_sk
            detailout["Emax"] = e_max
            detailout["Esweat"] = e_sweat
            detailout["BFcr"] = bf_cr
            detailout["BFms"] = bf_ms[VINDEX["muscle"]]
            detailout["BFfat"] = bf_fat[VINDEX["fat"]]
            detailout["BFsk"] = bf_sk
            detailout["BFava_hand"] = bf_ava_hand
            detailout["BFava_foot"] = bf_ava_foot
            detailout["Mbasecr"] = mbase[0]
            detailout["Mbasems"] = mbase[1][VINDEX["muscle"]]
            detailout["Mbasefat"] = mbase[2][VINDEX["fat"]]
            detailout["Mbasesk"] = mbase[3]
            detailout["Mwork"] = mwork
            detailout["Mshiv"] = mshiv
            detailout["Mnst"] = mnst
            detailout["Qcr"] = qcr
            detailout["Qms"] = qms[VINDEX["muscle"]]
            detailout["Qfat"] = qfat[VINDEX["fat"]]
            detailout["Qsk"] = qsk
            dictout["SHLsk"] = shlsk
            dictout["LHLsk"] = e_sk
            dictout["RESsh"] = res_sh
            dictout["RESlh"] = res_lh


        if self._ex_output == "all":
            dictout.update(detailout)
        elif isinstance(self._ex_output, list):  # if ex_out type is list
            outkeys = detailout.keys()
            for key in self._ex_output:
                if key in outkeys:
                    dictout[key] = detailout[key]
        return dictout


    def dict_results(self):
        """
        Get results as pandas.DataFrame format.

        Returns
        -------
        pandas.DataFrame
        """
        if not self._history:
            print("The model has no data.")
            return None

        def check_word_contain(word, *args):
            """
            Check if word contains *args.
            """
            boolfilter = False
            for arg in args:
                if arg in word:
                    boolfilter = True
            return boolfilter

        # Set column titles
        # If the values are iter, add the body names as suffix wrods.
        # If the values are not iter and the single value data, convert it to iter.
        key2keys = {}  # Column keys
        for key, value in self._history[0].items():
            try:
                length = len(value)
                if isinstance(value, str):
                    keys = [key]  # str is iter. Convert to list without suffix
                elif check_word_contain(key, "sve", "sfv", "superficialvein"):
                    keys = [key+BODY_NAMES[i] for i in VINDEX["sfvein"]]
                elif check_word_contain(key, "ms", "muscle"):
                    keys = [key+BODY_NAMES[i] for i in VINDEX["muscle"]]
                elif check_word_contain(key, "fat"):
                    keys = [key+BODY_NAMES[i] for i in VINDEX["fat"]]
                elif length == 17:  # if data contains 17 values
                    keys = [key+bn for bn in BODY_NAMES]
                else:
                    keys = [key+BODY_NAMES[i] for i in range(length)]
            except TypeError:  # if the value is not iter.
                keys= [key]  # convert to iter
            key2keys.update({key: keys})
        
        data = []
        for i, dictout in enumerate(self._history):
            row = {}
            for key, value in dictout.items():
                keys = key2keys[key]
                if len(keys) == 1:
                    values = [value]  # make list if value is not iter
                else:
                    values = value
                row.update(dict(zip(keys, values)))
            data.append(row)
            
        outdict = dict(zip(data[0].keys(), [[] for i in range(len(data[0].keys()))]))
        for row in data:
            for k in data[0].keys():
                outdict[k].append(row[k])
        return outdict


    def to_csv(self, path=None, folder=None, unit=True):
        """
        Export results with "units" as csv format.
        If you want to know units of parametes, use this method.

        Parameters
        ----------
        path : str, optional
            Output path. If you don't use the default file name, set a name.
            The default is None.
        folder : str, optional
            Output folder. If you use the default file name, set a only folder path.
            The default is None.
        unit : bool, optional
            Writes unit in csv file. The default is True.
        """
        if path is None:
            nowtime = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            path = "{}_{}.csv".format(self.model_name, nowtime)
            if folder:
                os.makedirs(folder, exist_ok=True)
                path = folder + os.sep + path
        elif not ((path[-4:] == ".csv") or (path[-4:] == ".txt")):
            path += ".csv"
        dictout = self.dict_results()

        columns = [k for k in dictout.keys()]
        units = []
        for col in columns:
            if "RES" in col[:3]:
                units.append("[W]")
            elif "Setpt" in col[:5]:
                units.append("[oC]")
            elif "RH" in col[:2]:
                units.append("[%]")
            elif "Va" in col[:2]:
                units.append("[m/s]")
            elif "Met" in col[:3]:
                units.append("[W]")
            elif "HL" in col[1:3]:
                units.append("[W]")
            elif "CO" in col[:2]:
                units.append("[L/h]")
            elif "T" in col[:1]:
                units.append("[oC]")
            elif "BF" in col[:2]:
                units.append("[L/h]")
            elif "M" in col[:1]:
                units.append("[W]")
            elif "Q" in col[:1]:
                units.append("[W]")
            elif "R" in col[:1]:
                units.append("[K.m2/W]")
            elif "E" in col[:1]:
                units.append("[W]")
            elif "dt" == col:
                units.append("[sec]")
            else:
                units.append("")

        with open(path, "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(columns))
            writer.writerow(units)
            for i in range(len(dictout["CycleTime"])):
                row = []
                for k in columns:
                    row.append(dictout[k][i])
                writer.writerow(row)


    #--------------------------------------------------------------------------
    # Setter
    #--------------------------------------------------------------------------
    def set_ex_q(self, tissue, value):
        """
        Set extra heat gain by tissue name.

        Parameters
        ----------
        tissue : str
            Tissue name. "core", "skin", or "artery".... If you set value to
            Head muscle and other segment's core, set "all_muscle".
        value : int, float, array
            Heat gain [W]

        Returns
        -------
        (83,) np.ndarray
            Current extra heat gain of model.
        """
        self.ex_q[INDEX[tissue]] = value
        return self.ex_q


    #--------------------------------------------------------------------------
    # Setter & getter
    #--------------------------------------------------------------------------

    @property
    def Ta(self):
        return self._ta
    @Ta.setter
    def Ta(self, inp):
        self._ta = _to17array(inp)

    @property
    def Tr(self):
        return self._tr
    @Tr.setter
    def Tr(self, inp):
        self._tr = _to17array(inp)

    @property
    def To(self):
        hc = threg.fixed_hc(threg.conv_coef(self._posture, self._va, self._ta, self.Tsk,), self._va)
        hr = threg.fixed_hr(threg.rad_coef(self._posture,))
        to = threg.operative_temp(self._ta, self._tr, hc, hr,)
        return to
    @To.setter
    def To(self, inp):
        self._ta = _to17array(inp)
        self._tr = _to17array(inp)

    @property
    def RH(self):
        return self._rh
    @RH.setter
    def RH(self, inp):
        self._rh = _to17array(inp)

    @property
    def Va(self):
        return self._va
    @Va.setter
    def Va(self, inp):
        self._va = _to17array(inp)

    @property
    def posture(self):
        return self._posture
    @posture.setter
    def posture(self, inp):
        if inp == 0:
            self._posture = "standing"
        elif inp == 1:
            self._posture = "sitting"
        elif inp == 2:
            self._posture = "lying"
        elif type(inp) == str:
            if inp.lower() == "standing":
                self._posture = "standing"
            elif inp.lower() in ["sitting", "sedentary"]:
                self._posture = "sitting"
            elif inp.lower() in ["lying", "supine"]:
                self._posture = "lying"
        else:
            self._posture = "standing"
            print('posture must be 0="standing", 1="sitting" or 2="lying".')
            print('posture was set "standing".')

    @property
    def Icl(self):
        return self._clo
    @Icl.setter
    def Icl(self, inp):
        self._clo = _to17array(inp)

    @property
    def PAR(self):
        return self._par
    @PAR.setter
    def PAR(self, inp):
        self._par = inp

    @property
    def bodytemp(self):
        return self._bodytemp
    @bodytemp.setter
    def bodytemp(self, inp):
        self._bodytemp = inp.copy()

    #--------------------------------------------------------------------------
    # Getter
    #--------------------------------------------------------------------------

    @property
    def BSA(self):
        return self._bsa.copy()

    @property
    def Rt(self):
        hc = threg.fixed_hc(threg.conv_coef(self._posture, self._va, self._ta, self.Tsk,), self._va)
        hr = threg.fixed_hr(threg.rad_coef(self._posture,))
        return threg.dry_r(hc, hr, self._clo)

    @property
    def Ret(self):
        hc = threg.fixed_hc(threg.conv_coef(self._posture, self._va, self._ta, self.Tsk,), self._va)
        return threg.wet_r(hc, self._clo, self._iclo)

    @property
    def Wet(self):
        err_cr = self.Tcr - self.setpt_cr
        err_sk = self.Tsk - self.setpt_sk
        wet, *_ = threg.evaporation(err_cr, err_sk,
                self._ta, self._rh, self.Ret, self._bsa_rate, self._age)
        return wet
    
    @property
    def WetMean(self):
        wet = self.Wet
        return np.average(wet, weights=_BSAst)

    

    @property
    def TskMean(self):
        return np.average(self._bodytemp[INDEX["skin"]], weights=_BSAst)

    @property
    def Tsk(self):
        return self._bodytemp[INDEX["skin"]].copy()

    @property
    def Tcr(self):
        return self._bodytemp[INDEX["core"]].copy()

    @property
    def Tcb(self):
        return self._bodytemp[0].copy()

    @property
    def Tar(self):
        return self._bodytemp[INDEX["artery"]].copy()

    @property
    def Tve(self):
        return self._bodytemp[INDEX["vein"]].copy()

    @property
    def Tsve(self):
        return self._bodytemp[INDEX["sfvein"]].copy()

    @property
    def Tms(self):
        return self._bodytemp[INDEX["muscle"]].copy()

    @property
    def Tfat(self):
        return self._bodytemp[INDEX["fat"]].copy()

    @property
    def bodyname(self):
        body = [
                "Head", "Neck", "Chest", "Back", "Pelvis",
                "LShoulder", "LArm", "LHand",
                "RShoulder", "RArm", "RHand",
                "LThigh", "LLeg", "LHand",
                "RThigh", "RLeg", "RHand",]
        return body

    @property
    def result(self):
        return self.dict_results()

    @property
    def BMR(self):
        bmr = threg.basal_met(
                self._height, self._weight, self._age,
                self._sex, self._bmr_equation,)
        return bmr
    
def _to17array(inp):
    """
    Make ndarray (17,).

    Parameters
    ----------
    inp : int, float, ndarray, list
        Number you make as 17array.

    Returns
    -------
    ndarray
    """
    try:
        if len(inp) == 17:
            array = np.array(inp)
        else:
            first_item = inp[0]
            array = np.ones(17)*first_item
    except:
        array = np.ones(17)*inp
    return array.copy()


if __name__ == "__main__":
    mod = JOS()
    print("\nNeutral")
    print("TcrHead: {:.3f} [oC]".format(mod.Tcr[0]))
    print("TskMean: {:.3f} [oC]".format(mod.TskMean))

    mod = JOS()
    mod.PAR = 2
    mod.To = 40
    mod.RH = 70
    mod.Va = 2
    mod.Icl = 0.6
    mod.simulate(60)
    print("\nAfter Hot Exposure")
    print("TcrHead: {:.3f} [oC]".format(mod.Tcr[0]))
    print("TskMean: {:.3f} [oC]".format(mod.TskMean))

    mod = JOS()
    mod.PAR = 1.2
    mod.To = 10
    mod.RH = 20
    mod.Va = 3
    mod.Icl = 0.1
    mod.simulate(60)
    print("\nAfter Cold Exposure")
    print("TcrHead: {:.3f} [oC]".format(mod.Tcr[0]))
    print("TskMean: {:.3f} [oC]".format(mod.TskMean))

    # Measure calculation time
    import time
    stime = time.time()
    mod = JOS()
    mod.To = 30
    mod.simulate(60)
    mod.To = 20
    mod.simulate(60)
    mod.To = 40
    mod.simulate(60)
    mod.To = 10
    mod.simulate(60)
    etime = time.time()
    print("Default output")
    print("Calculation time {:.2f} [sec]".format(etime-stime))

    stime = time.time()
    mod = JOS(ex_output=["BFsk", "BFcr", "Emax"])
    mod.To = 30
    mod.simulate(60)
    mod.To = 20
    mod.simulate(60)
    mod.To = 40
    mod.simulate(60)
    mod.To = 10
    mod.simulate(60)
    etime = time.time()
    print("Extra output")
    print("Calculation time {:.2f} [sec]".format(etime-stime))

    stime = time.time()
    mod = JOS(ex_output="all")
    mod.To = 30
    mod.simulate(60)
    mod.To = 20
    mod.simulate(60)
    mod.To = 40
    mod.simulate(60)
    mod.To = 10
    mod.simulate(60)
    etime = time.time()
    print("All output")
    print("Calculation time {:.2f} [sec]".format(etime-stime))