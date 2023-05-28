# -*- coding: utf-8 -*-
import numpy as np
import math

# Import from relative path
try:
    from .matrix import NUM_NODES, IDICT, BODY_NAMES
    from . import construction as cons
# Import from absolute path
# These codes are for debugging
except ImportError:
    from jos3.matrix import NUM_NODES, IDICT, BODY_NAMES
    from jos3 import construction as cons


_BSAst = np.array([
        0.110, 0.029, 0.175, 0.161, 0.221,
        0.096, 0.063, 0.050, 0.096, 0.063, 0.050,
        0.209, 0.112, 0.056, 0.209, 0.112, 0.056,])


def conv_coef(posture="standing", va=0.1, ta=28.8, tsk=34.0,):
    """
    Calculate convective heat transfer coefficient (hc) [W/K.m2]

    Parameters
    ----------
    posture : str, optional
        Select posture from standing, sitting or lying.
        The default is "standing".
    va : float or iter, optional
        Air velocity [m/s]. If iter is input, its length should be 17.
        The default is 0.1.
    ta : float or iter, optional
        Air temperature [oC]. If iter is input, its length should be 17.
        The default is 28.8.
    tsk : float or iter, optional
        Skin temperature [oC]. If iter is input, its length should be 17.
        The default is 34.0.

    Returns
    -------
    hc : numpy.ndarray
        Convective heat transfer coefficient (hc) [W/K.m2].

    """
    # Natural convection
    if posture.lower() == "standing":
        # Ichihara et al., 1997, https://doi.org/10.3130/aija.62.45_5
        hc_natural = np.array([
                4.48, 4.48, 2.97, 2.91, 2.85,
                3.61, 3.55, 3.67, 3.61, 3.55, 3.67,
                2.80, 2.04, 2.04, 2.80, 2.04, 2.04,])
    elif posture.lower() in ["sitting", "sedentary"]:
        # Ichihara et al., 1997, https://doi.org/10.3130/aija.62.45_5
        hc_natural = np.array([
                4.75, 4.75, 3.12, 2.48, 1.84,
                3.76, 3.62, 2.06, 3.76, 3.62, 2.06,
                2.98, 2.98, 2.62, 2.98, 2.98, 2.62,])

    elif posture.lower() in ["lying", "supine"]:
        # Kurazumi et al., 2008, https://doi.org/10.20718/jjpa.13.1_17
        # The values are applied under cold environment.
        hc_a = np.array([
                1.105, 1.105, 1.211, 1.211, 1.211,
                0.913, 2.081, 2.178, 0.913, 2.081, 2.178,
                0.945, 0.385, 0.200, 0.945, 0.385, 0.200,])
        hc_b = np.array([
                0.345, 0.345, 0.046, 0.046, 0.046,
                0.373, 0.850, 0.297, 0.373, 0.850, 0.297,
                0.447, 0.580, 0.966, 0.447, 0.580, 0.966,])
        hc_natural = hc_a * (abs(ta - tsk) ** hc_b)

    # Forced convection
    # Ichihara et al., 1997, https://doi.org/10.3130/aija.62.45_5
    hc_a = np.array([
            15.0, 15.0, 11.0, 17.0, 13.0,
            17.0, 17.0, 20.0, 17.0, 17.0, 20.0,
            14.0, 15.8, 15.1, 14.0, 15.8, 15.1,])
    hc_b = np.array([
            0.62, 0.62, 0.67, 0.49, 0.60,
            0.59, 0.61, 0.60, 0.59, 0.61, 0.60,
            0.61, 0.74, 0.62, 0.61, 0.74, 0.62,])
    hc_forced = hc_a * (va ** hc_b)

    # Select natural or forced hc.
    # If local va is under 0.2 m/s, the hc valuse is natural.
    hc = np.where(va<0.2, hc_natural, hc_forced) # hc [W/K.m2)]

    return hc


def rad_coef(posture="standing"):
    """
    Calculate radiative heat transfer coefficient (hr) [W/K.m2]

    Parameters
    ----------
    posture : str, optional
        Select posture from standing, sitting or lying.
        The default is "standing".

    Returns
    -------
    hc : numpy.ndarray
        Radiative heat transfer coefficient (hr) [W/K.m2].

    """


    if posture.lower() == "standing":
        # Ichihara et al., 1997, https://doi.org/10.3130/aija.62.45_5
        hr = np.array([
                4.89, 4.89, 4.32, 4.09, 4.32,
                4.55, 4.43, 4.21, 4.55, 4.43, 4.21,
                4.77, 5.34, 6.14, 4.77, 5.34, 6.14,])
    elif posture.lower() in ["sitting", "sedentary"]:
        # Ichihara et al., 1997, https://doi.org/10.3130/aija.62.45_5
        hr = np.array([
                4.96, 4.96, 3.99, 4.64, 4.21,
                4.96, 4.21, 4.74, 4.96, 4.21, 4.74,
                4.10, 4.74, 6.36, 4.10, 4.74, 6.36,])
    elif posture.lower() in ["lying", "supine"]:
        # Kurazumi et al., 2008, https://doi.org/10.20718/jjpa.13.1_17
        hr = np.array([
                5.475, 5.475, 3.463, 3.463, 3.463,
                4.249, 4.835, 4.119, 4.249, 4.835, 4.119,
                4.440, 5.547, 6.085, 4.440, 5.547, 6.085,])
    return hr


def fixed_hc(hc, va):
    """
    Fixes hc values to fit tow-node-model's values.
    """
    mean_hc = np.average(hc, weights=_BSAst)
    mean_va = np.average(va, weights=_BSAst)
    mean_hc_whole = max(3, 8.600001*(mean_va**0.53))
    _fixed_hc = hc * mean_hc_whole/mean_hc
    return _fixed_hc


def fixed_hr(hr):
    """
    Fixes hr values to fit tow-node-model's values.
    """
    mean_hr = np.average(hr, weights=_BSAst)
    _fixed_hr = hr * 4.7/mean_hr
    return _fixed_hr

def operative_temp(ta, tr, hc, hr):
    to = (hc*ta + hr*tr) / (hc + hr)
    return to


def clo_area_factor(clo):
    fcl = np.where(clo<0.5, clo*0.2+1, clo*0.1+1.05)
    return fcl


def dry_r(hc, hr, clo, pt=101.33):
    """
    Calculate total sensible thermal resistance.

    Parameters
    ----------
    hc : float or array
        Convective heat transfer coefficient (hc) [W/K.m2].
    hr : float or array
        Radiative heat transfer coefficient (hr) [W/K.m2].
    clo : float or array
        Clothing insulation [clo].
    pt : float
        Local atmospheric pressure [kPa].
        Corrected hc (hcc) is calculated as follows:
            hcc = hc * ((pt / 101.33) ** 0.55)

    Returns
    -------
    rt : float or array
        Total sensible thermal resistance between skin and ambient.
    """
    fcl = clo_area_factor(clo)
    hcc = hc * ((pt / 101.33) ** 0.55)
    r_a = 1/(hc+hcc)
    r_cl = 0.155*clo
    r_t = r_a/fcl + r_cl
    return r_t


def wet_r(hc, clo, iclo=0.45, lewis_rate=16.5, pt=101.33):
    """
    Calculate total evaporative thermal resistance.

    Parameters
    ----------
    hc : float or array
        Convective heat transfer coefficient (hc) [W/K.m2].
    clo : float or array
        Clothing insulation [clo].
    iclo : float, or array, optional
        Clothin vapor permeation efficiency [-]. The default is 0.45.
    lewis_rate : float, optional
        Lewis rate [K/kPa]. The default is 16.5.
    pt : float
        Local atmospheric pressure [kPa].
        Corrected he (hec) is calculated as follows:
            hec = he * ((101.33 / pt) ** 0.45)

    Returns
    -------
    ret : float or array
        Total evaporative thermal resistance.

    """
    fcl = clo_area_factor(clo)
    r_cl = 0.155 * clo
    he = hc * lewis_rate
    hec = he * ((101.33 / pt) ** 0.45)
    r_ea = 1 / hec
    r_ecl = r_cl / (lewis_rate * iclo)
    r_et = r_ea / fcl + r_ecl
    return r_et


def heat_resistances(
        ta=np.ones(17)*28.8,
        tr=np.ones(17)*28.8,
        va=np.ones(17)*0.1,
        tsk=np.ones(17)*34,
        clo=np.zeros(17),
        posture="standing",
        iclo=np.ones(17)*0.45,
        options={},):

    hc = fixed_hc(conv_coef(posture, va, ta, tsk,))
    hr = fixed_hr(rad_coef(posture,))
    to = operative_temp(ta, tr, hc, hr,)
    fcl = clo_area_factor(clo,)
    r_t, r_a, r_cl = dry_r(hc, hr, clo)
    r_et, r_ea, r_ecl = wet_r(hc, clo, iclo)

    return to, r_t, r_et, r_a, r_cl, r_ea, r_ecl, fcl


def error_signals(err_cr=0, err_sk=0):
    """
    Calculate WRMS and CLDS signals of thermoregulation

    Parameters
    ----------
    err_cr, err_sk : float or array, optional
        Difference between setpoint and body temperatures.
        The default is 0.

    Returns
    -------
    wrms, clds : array
        WRMS and CLDS signals.
    """

    # SKINR
    receptor = np.array([
            0.0549, 0.0146, 0.1492, 0.1321, 0.2122,
            0.0227, 0.0117, 0.0923, 0.0227, 0.0117, 0.0923,
            0.0501, 0.0251, 0.0167, 0.0501, 0.0251, 0.0167,])

    # wrms signal
    wrm = np.maximum(err_sk, 0)
    wrm *= receptor
    wrms = wrm.sum()
    # clds signal
    cld = np.minimum(err_sk, 0)
    cld *= -receptor
    clds = cld.sum()

    return wrms, clds


# Antoine equation [kPa]
antoine = lambda x: math.e**(16.6536-(4030.183/(x+235)))
# Tetens equation [kPa]
tetens = lambda x: 0.61078*10**(7.5*x/(x+237.3))


def evaporation(err_cr, err_sk, tsk, ta, rh, ret,
                height=1.72, weight=74.43, equation="dubois", age=20):
    """
    Calculate evaporative heat loss.

    Parameters
    ----------
    err_cr, err_sk : array
        Difference between setpoint and body temperatures [oC].
    tsk : array
        Skin temperatures [oC].
    ta : array
        Air temperatures at local body segments [oC].
    rh : array
        Relative humidity at local body segments [%].
    ret : array
        Total evaporative thermal resistances [m2.K/W].
    height : float, optional
        Body height [m]. The default is 1.72.
    weight : float, optional
        Body weight [kg]. The default is 74.43.
    equation : str, optional
        The equation name (str) of bsa calculation. Choose a name from "dubois",
        "takahira", "fujimoto", or "kurazumi". The default is "dubois".
    age : float, optional
        Age [years]. The default is 20.

    Returns
    -------
    wet : array
        Local skin wettedness [-].
    e_sk : array
        Evaporative heat loss at the skin by sweating and diffuse [W].
    e_max : array
        Maximum evaporative heat loss at the skin [W].
    e_sweat : TYPE
        Evaporative heat loss at the skin by only sweating [W].

    """

    wrms, clds = error_signals(err_cr, err_sk,)  # Thermoregulation signals
    bsar = cons.bsa_rate(height, weight, equation,)  # BSA rate
    bsa = _BSAst * bsar  # BSA
    p_a = antoine(ta)*rh/100  # Saturated vapor pressure of ambient [kPa]
    p_sk_s = antoine(tsk)  # Saturated vapor pressure at the skin [kPa]

    e_max = (p_sk_s - p_a) / ret * bsa  # Maximum evaporative heat loss

    # SKINS
    skin_sweat = np.array([
            0.064, 0.017, 0.146, 0.129, 0.206,
            0.051, 0.026, 0.0155, 0.051, 0.026, 0.0155,
            0.073, 0.036, 0.0175, 0.073, 0.036, 0.0175,])

    sig_sweat = (371.2*err_cr[0]) + (33.64*(wrms-clds))
    sig_sweat = max(sig_sweat, 0)
    sig_sweat *= bsar

    # Signal decrement by aging
    if age < 60:
        sd_sweat = np.ones(17)
    else: #age >= 60
        sd_sweat = np.array([
                0.69, 0.69, 0.59, 0.52, 0.40,
                0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
                0.40, 0.40, 0.40, 0.40, 0.40, 0.40,])

    e_sweat = skin_sweat * sig_sweat * sd_sweat * 2**((err_sk)/10)
    wet = 0.06 + 0.94*(e_sweat/e_max)
    wet = np.minimum(wet, 1)  # Wettedness' upper limit
    e_sk = wet * e_max
    e_sweat = (wet - 0.06) / 0.94 * e_max  # Effective sweating
    return wet, e_sk, e_max, e_sweat


def skin_bloodflow(err_cr, err_sk,
        height=1.72, weight=74.43, equation="dubois", age=20, ci=2.59,):
    """
    Calculate skin blood flow rate (BFsk) [L/h].

    Parameters
    ----------
    err_cr, err_sk : array
        Difference between setpoint and body temperatures [oC].
    height : float, optional
        Body height [m]. The default is 1.72.
    weight : float, optional
        Body weight [kg]. The default is 74.43.
    equation : str, optional
        The equation name (str) of bsa calculation. Choose a name from "dubois",
        "takahira", "fujimoto", or "kurazumi". The default is "dubois".
    age : float, optional
        Age [years]. The default is 20.
    ci : float, optional
        Cardiac index [L/min/㎡]. The default is 2.59.

    Returns
    -------
    BFsk : array
        Skin blood flow rate [L/h].

    """

    wrms, clds = error_signals(err_cr, err_sk)

    # BFBsk
    bfb_sk = np.array([
            1.754, 0.325, 1.967, 1.475, 2.272,
            0.91, 0.508, 1.114, 0.91, 0.508, 1.114,
            1.456, 0.651, 0.934, 1.456, 0.651, 0.934,])
    # SKIND
    skin_dilat = np.array([
            0.0692, 0.0992, 0.0580, 0.0679, 0.0707,
            0.0400, 0.0373, 0.0632, 0.0400, 0.0373, 0.0632,
            0.0736, 0.0411, 0.0623, 0.0736, 0.0411, 0.0623,])
    # SKINC
    skin_stric = np.array([
            0.0213, 0.0213, 0.0638, 0.0638, 0.0638,
            0.0213, 0.0213, 0.1489, 0.0213, 0.0213, 0.1489,
            0.0213, 0.0213, 0.1489, 0.0213, 0.0213, 0.1489,])

    sig_dilat = (100.5*err_cr[0]) + (6.4*(wrms-clds))
    sig_stric = (-10.8*err_cr[0]) + (-10.8*(wrms-clds))
    sig_dilat = max(sig_dilat, 0)
    sig_stric = max(sig_stric, 0)

    # Signal decrement by aging
    if age < 60:
        sd_dilat = np.ones(17)
        sd_stric = np.ones(17)
    else: #age >= 60
        sd_dilat = np.array([
                0.91, 0.91, 0.47, 0.47, 0.31,
                0.47, 0.47, 0.47, 0.47, 0.47, 0.47,
                0.31, 0.31, 0.31, 0.31, 0.31, 0.31,
                ])
        sd_stric = np.ones(17)

    #皮膚血流量 [L/h]
    bf_sk = (1 + skin_dilat * sd_dilat * sig_dilat) / \
            (1 + skin_stric * sd_stric * sig_stric) * bfb_sk * 2**(err_sk/6)

    bfbr = cons.bfb_rate(height, weight, equation, age, ci,)
    bf_sk *= bfbr
    return bf_sk


def ava_bloodflow(err_cr, err_sk,
        height=1.72, weight=74.43, equation="dubois", age=20, ci=2.59,):
    """
    Calculate areteriovenous anastmoses (AVA) blood flow rate [L/h] based on
    Takemori's model, 1995.

    Parameters
    ----------
    err_cr, err_sk : array
        Difference between setpoint and body temperatures [oC].
    height : float, optional
        Body height [m]. The default is 1.72.
    weight : float, optional
        Body weight [kg]. The default is 74.43.
    equation : str, optional
        The equation name (str) of bsa calculation. Choose a name from "dubois",
        "takahira", "fujimoto", or "kurazumi". The default is "dubois".
    age : float, optional
        Age [years]. The default is 20.
    ci : float, optional
        Cardiac index [L/min/m2]. The default is 2.59.

    Returns
    -------
    BFava_hand, BFava_foot : array
        AVA blood flow rate at hand and foot [L/h].

    """
    # Cal. mean error body core temp.
    cap_bcr = [10.2975, 9.3935, 13.834]  # Thermal capacity at Chest, Back and Pelvis
    err_bcr = np.average(err_cr[2:5], weights=cap_bcr)

    # Cal. mean error skin temp.
    bsa = _BSAst
    err_msk = np.average(err_sk, weights=bsa)

    # Openbess of AVA [-]
    sig_ava_hand = 0.265 * (err_msk + 0.43) + 0.953 * (err_bcr + 0.1905) + 0.9126
    sig_ava_foot = 0.265 * (err_msk - 0.997) + 0.953 * (err_bcr + 0.0095) + 0.9126

    sig_ava_hand = min(sig_ava_hand, 1)
    sig_ava_hand = max(sig_ava_hand, 0)
    sig_ava_foot = min(sig_ava_foot, 1)
    sig_ava_foot = max(sig_ava_foot, 0)

    bfbr = bfbr = cons.bfb_rate(height, weight, equation, age, ci,)
    # AVA blood flow rate [L/h]
    bf_ava_hand = 1.71 * bfbr * sig_ava_hand  # Hand
    bf_ava_foot = 2.16 * bfbr * sig_ava_foot  # Foot
    return bf_ava_hand, bf_ava_foot


def basal_met(height=1.72, weight=74.43, age=20,
            sex="male", equation="harris-benedict"):
    """
    Calculate basal metabolic rate [W].

    Parameters
    ----------
    height : float, optional
        Body height [m]. The default is 1.72.
    weight : float, optional
        Body weight [kg]. The default is 74.43.
    age : float, optional
        Age [years]. The default is 20.
    sex : str, optional
        Choose male or female. The default is "male".
    equation : str, optional
        Choose harris-benedict or ganpule. The default is "harris-benedict".

    Returns
    -------
    BMR : float
        Basal metabolic rate [W].

    """

    if equation=="harris-benedict":
        if sex=="male":
            bmr = 88.362 + 13.397*weight + 500.3*height - 5.677*age
        else:
            bmr = 447.593 + 9.247*weight + 479.9*height - 4.330*age

    elif equation=="harris-benedict_origin":
        if sex=="male":
            bmr = 66.4730 + 13.7516*weight + 500.33*height - 6.7550*age
        else:
            bmr = 655.0955 + 9.5634*weight + 184.96*height - 4.6756*age

    elif equation=="japanese" or equation=="ganpule":
        # Ganpule et al., 2007, https://doi.org/10.1038/sj.ejcn.1602645
        if sex=="male":
            bmr = 0.0481*weight + 2.34*height - 0.0138*age - 0.4235
        else:
            bmr = 0.0481*weight + 2.34*height - 0.0138*age - 0.9708
        bmr *= 1000 / 4.186

    bmr *= 0.048  # [kcal/day] to [W]

    return bmr


def local_mbase(height=1.72, weight=74.43, age=20,
            sex="male", equation="harris-benedict"):
    """
    Calculate local basal metabolic rate [W].

    Parameters
    ----------
    height : float, optional
        Body height [m]. The default is 1.72.
    weight : float, optional
        Body weight [kg]. The default is 74.43.
    age : float, optional
        Age [years]. The default is 20.
    sex : str, optional
        Choose male or female. The default is "male".
    equation : str, optional
        Choose harris-benedict or ganpule. The default is "harris-benedict".

    Returns
    -------
    mbase : array
        Local basal metabolic rate (Mbase) [W].

    """

    mbase_all = basal_met(height, weight, age, sex, equation)
    # Distribution coefficient of basal metabolic rate
    mbf_cr = np.array([
            0.19551, 0.00324, 0.28689, 0.25677, 0.09509,
            0.01435, 0.00409, 0.00106, 0.01435, 0.00409, 0.00106,
            0.01557, 0.00422, 0.00250, 0.01557, 0.00422, 0.00250,])
    mbf_ms = np.array([
            0.00252, 0.0, 0.0, 0.0, 0.04804,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,])
    mbf_fat = np.array([
            0.00127, 0.0, 0.0, 0.0, 0.00950,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,])
    mbf_sk = np.array([
            0.00152, 0.00033, 0.00211, 0.00187, 0.00300,
            0.00059, 0.00031, 0.00059, 0.00059, 0.00031, 0.00059,
            0.00144, 0.00027, 0.00118, 0.00144, 0.00027, 0.00118,])

    mbase_cr = mbf_cr * mbase_all
    mbase_ms = mbf_ms * mbase_all
    mbase_fat = mbf_fat * mbase_all
    mbase_sk = mbf_sk * mbase_all
    return mbase_cr, mbase_ms, mbase_fat, mbase_sk


def local_mwork(bmr, par):
    """
    Calculate local metabolic rate by work [W]

    Parameters
    ----------
    bmr : float
        Basal metbolic rate [W].
    par : float
        Physical activity ratio [-].

    Returns
    -------
    Mwork : array
        Local metabolic rate by work [W].

    """
    mwork_all = (par-1) * bmr
    mwf = np.array([
            0, 0, 0.091, 0.08, 0.129,
            0.0262, 0.0139, 0.005, 0.0262, 0.0139, 0.005,
            0.2010, 0.0990, 0.005, 0.2010, 0.0990, 0.005])
    mwork = mwork_all * mwf
    return mwork


PRE_SHIV = 0
def shivering(err_cr, err_sk, tcr, tsk,
              height=1.72, weight=74.43, equation="dubois", age=20, sex="male", dtime=60,
              options={}):
    """
    Calculate local metabolic rate by shivering [W].

    Parameters
    ----------
    err_cr, err_sk : array
        Difference between setpoint and body temperatures [oC].
    tcr, tsk : array
        Core and skin temperatures [oC].
    height : float, optional
        Body height [m]. The default is 1.72.
    weight : float, optional
        Body weight [kg]. The default is 74.43.
    equation : str, optional
        The equation name (str) of bsa calculation. Choose a name from "dubois",
        "takahira", "fujimoto", or "kurazumi". The default is "dubois".
    age : float, optional
        Age [years]. The default is 20.
    sex : str, optional
        Choose male or female. The default is "male".
    dtime : float, optional
        Interval of analysis time. The default is 60.

    Returns
    -------
    Mshiv : array
        Local metabolic rate by shivering [W].

    """
    wrms, clds = error_signals(err_cr, err_sk,)
    shivf = np.array([
            0.0339, 0.0436, 0.27394, 0.24102, 0.38754,
            0.00243, 0.00137, 0.0002, 0.00243, 0.00137, 0.0002,
            0.0039, 0.00175, 0.00035, 0.0039, 0.00175, 0.00035,])
    sig_shiv = 24.36 * clds * (-err_cr[0])
    sig_shiv = max(sig_shiv, 0)

    if options:
        if options["shivering_threshold"]:
            # Asaka, 2016
            # Threshold of starting shivering
            tskm = np.average(tsk, weights=_BSAst) # Mean skin temp.
            if tskm < 31:
                thres = 36.6
            else:
                if sex == "male":
                    thres = -0.2436 * tskm + 44.10
                else: # sex == "female":
                    thres = -0.2250 * tskm + 43.05
            # Second threshold of starting shivering
            if thres < tcr[0]:
                sig_shiv = 0

    global PRE_SHIV  # Previous shivering thermogenesis [W]
    if options:
        if options["limit_dshiv/dt"]:
            # Asaka, 2016
            # dshiv < 0.0077 [W/s]
            dshiv = sig_shiv - PRE_SHIV
            if options["limit_dshiv/dt"] is True: # default is 0.0077 [W/s]
                limit_dshiv = 0.0077 * dtime
            else:
                limit_dshiv = options["limit_dshiv/dt"] * dtime
            if dshiv > limit_dshiv:
                sig_shiv = limit_dshiv + PRE_SHIV
            elif dshiv < -limit_dshiv:
                sig_shiv = -limit_dshiv + PRE_SHIV
        PRE_SHIV = sig_shiv

    # Signal sd_shiv by aging
    if age < 30:
        sd_shiv = np.ones(17)
    elif age < 40:
        sd_shiv = np.ones(17) * 0.97514
    elif age < 50:
        sd_shiv = np.ones(17) * 0.95028
    elif age < 60:
        sd_shiv = np.ones(17) * 0.92818
    elif age < 70:
        sd_shiv = np.ones(17) * 0.90055
    elif age < 80:
        sd_shiv = np.ones(17) * 0.86188
    else: #age >= 80
        sd_shiv = np.ones(17) * 0.82597

    bsar = cons.bsa_rate(height, weight, equation)
    mshiv = shivf * bsar * sd_shiv * sig_shiv
    return mshiv

shivering
def nonshivering(err_cr, err_sk,
             height=1.72, weight=74.43, equation="dubois", age=20,
             coldacclimation=False, batpositive=True,
             options={},):
    """
    Calculate local metabolic rate by non-shivering [W]

    Parameters
    ----------
    err_cr, err_sk : array
        Difference between setpoint and body temperatures [oC].
    height : float, optional
        Body height [m]. The default is 1.72.
    weight : float, optional
        Body weight [kg]. The default is 74.43.
    equation : str, optional
        The equation name (str) of bsa calculation. Choose a name from "dubois",
        "takahira", "fujimoto", or "kurazumi". The default is "dubois".
    age : float, optional
        Age [years]. The default is 20.
    coldacclimation : bool, optional
        Whether the subject acclimates cold enviroment or not.
        The default is False.
    batpositive : bool, optional
        Whether BAT ativity is positive or not.
        The default is True.

    Returns
    -------
    Mnst : array
        Local metabolic rate by non-shivering [W].

    """
    # NST (Non-Shivering Thermogenesis) model, Asaka, 2016
    wrms, clds = error_signals(err_cr, err_sk, )

    bmi = weight / height**2

    # BAT: brown adipose tissue [SUV]
    bat = 10**(-0.10502 * bmi + 2.7708)

    # age factor
    if age < 30:
        bat *= 1.61
    elif age < 40:
        bat *= 1.00
    else: # age >= 40
        bat *= 0.80

    if coldacclimation:
        bat += 3.46

    if not batpositive:
        # incidence age factor: T.Yoneshiro 2011
        if age < 30:
            bat *= 44/83
        elif age < 40:
            bat *= 15/38
        elif age < 50:
            bat *= 7/26
        elif age < 50:
            bat *= 1/8
        else: # age > 60
            bat *= 0

    # NST limit
    thres = ((1.80 * bat + 2.43) + 5.62)  # [W]

    sig_nst = 2.8 * clds  # [W]
    sig_nst = min(sig_nst, thres)

    mnstf = np.array([
            0.000, 0.190, 0.000, 0.190, 0.190,
            0.215, 0.000, 0.000, 0.215, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000,])
    bsar = cons.bsa_rate(height, weight, equation)
    mnst = bsar * mnstf * sig_nst
    return mnst


def sum_m(mbase, mwork, mshiv, mnst):
    qcr = mbase[0].copy()
    qms = mbase[1].copy()
    qfat = mbase[2].copy()
    qsk = mbase[3].copy()

    for i, bn in enumerate(BODY_NAMES):
        # If the segment has a muscle layer, muscle heat production increases by the activity.
        if not IDICT[bn]["muscle"] is None:
            qms[i] += mwork[i] + mshiv[i]
        # In other segments, core heat production increase, instead of muscle.
        else:
            qcr[i] += mwork[i] + mshiv[i]
    qcr += mnst  # Non-shivering thermogenesis occurs in core layers
    return qcr, qms, qfat, qsk


def crmsfat_bloodflow(mwork, mshiv,
        height=1.72, weight=74.43, equation="dubois", age=20, ci=2.59,):
    """
    Calculate core, muslce and fat blood flow rate [L/h].

    Parameters
    ----------
    mwork : array
        Metablic rate by work [W].
    mshiv : array
        Metablic rate by shivering [W].
    height : float, optional
        Body height [m]. The default is 1.72.
    weight : float, optional
        Body weight [kg]. The default is 74.43.
    equation : str, optional
        The equation name (str) of bsa calculation. Choose a name from "dubois",
        "takahira", "fujimoto", or "kurazumi". The default is "dubois".
    age : float, optional
        Age [years]. The default is 20.
    ci : float, optional
        Cardiac index [L/min/㎡]. The default is 2.59.

    Returns
    -------
    BFcr, BFms, BFfat : array
        Core, muslce and fat blood flow rate [L/h].

    """
    # Basal blood flow rate [L/h]
    # core, CBFB
    bfb_cr = np.array([
            35.251, 15.240, 89.214, 87.663, 18.686,
            1.808, 0.940, 0.217, 1.808, 0.940, 0.217,
            1.406, 0.164, 0.080, 1.406, 0.164, 0.080,])
    # muscle, MSBFB
    bfb_ms = np.array([
            0.682, 0.0, 0.0, 0.0, 12.614,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,])
    # fat, FTBFB
    bfb_fat = np.array([
            0.265, 0.0, 0.0, 0.0, 2.219,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,])

    bfbr = cons.bfb_rate(height, weight, equation, age, ci)
    bf_cr = bfb_cr * bfbr
    bf_ms = bfb_ms * bfbr
    bf_fat = bfb_fat * bfbr

    for i, bn in enumerate(BODY_NAMES):
        # If the segment has a muscle layer, muscle blood flow increases.
        if not IDICT[bn]["muscle"] is None:
            bf_ms[i] += (mwork[i] + mshiv[i])/1.163
        # In other segments, core blood flow increase, instead of muscle blood flow.
        else:
            bf_cr[i] += (mwork[i] + mshiv[i])/1.163
    return bf_cr, bf_ms, bf_fat


def sum_bf(bf_cr, bf_ms, bf_fat, bf_sk, bf_ava_hand, bf_ava_foot):
    co = 0
    co += bf_cr.sum()
    co += bf_ms.sum()
    co += bf_fat.sum()
    co += bf_sk.sum()
    co += 2*bf_ava_hand
    co += 2*bf_ava_foot
    return co


def resp_heatloss(t, p, met):
    res_sh = 0.0014 * met * (34 - t) #顕熱
    res_lh = 0.0173 * met * (5.87 - p) #潜熱
    return res_sh, res_lh


def get_lts(ta):
    return 2.418*1000