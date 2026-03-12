# File that contains the definitions of various quantities related to
# stellar feedback from massive stars
# author: Lachlan Lancaster

import numpy as np
from astropy import units as u
from astropy import constants as aconsts

#########################################################################################
################################ Length Scale Quantities ################################
#########################################################################################

def RSt(Q0, nbar, alphaB = 3.11e-13*(u.cm**3/u.s)):
    # Returns the Strömgren Radius in parsecs
    # Q0 : the ionizing photon rate
    # nbar : the number density of hydrgen in the background
    # alphaB : the case B recombination rate

    # start by making sure that the dimensions of the arguments are correct
    t1 = u.get_physical_type(Q0)=="frequency"
    t2 = u.get_physical_type(nbar)=="number density"
    t3 = u.get_physical_type(alphaB)=="volumetric flow rate"
    if not(t1):
        raise ValueError("Units of Q0 are incorrect")
    if not(t2):
        raise ValueError("Units of nbar are incorrect")
    if not(t3):
        raise ValueError("Units of alphaB are incorrect")
    
    r_st = (3*Q0/(4*np.pi*nbar**2*alphaB))**(1./3)
    
    return r_st.to("pc")

def Req_MD(pdotw, rhobar, ci = 10*u.km/u.s):
    # Returns the equilibration radius
    # for a momentum-driven solution
    # pdotw : wind momentum input rate
    # rhobar : background mass density
    # ci : ionized gas sound speed

    # start by making sure that the dimensions of the arguments are correct
    t1 = u.get_physical_type(pdotw)=="force"
    t2 = u.get_physical_type(rhobar)=="mass density"
    t3 = u.get_physical_type(ci)=="speed"
    if not(t1):
        raise ValueError("Units of pdotw are incorrect")
    if not(t2):
        raise ValueError("Units of rhobar are incorrect")
    if not(t3):
        raise ValueError("Units of ci are incorrect")

    r_eq = (pdotw/(4*np.pi*rhobar*ci**2))**(1./2)

    return r_eq.to("pc")

def Req_ED(Lwind, rhobar, ci = 10*u.km/u.s):
    # Returns the equilibration radius
    # for an energy-driven solution
    # Lwind : wind luminosity
    # rhobar : background mass density
    # ci : ionized gas sound speed

    # start by making sure that the dimensions of the arguments are correct
    t1 = u.get_physical_type(Lwind)=="power"
    t2 = u.get_physical_type(rhobar)=="mass density"
    t3 = u.get_physical_type(ci)=="speed"
    if not(t1):
        raise ValueError("Units of Lwind are incorrect")
    if not(t2):
        raise ValueError("Units of rhobar are incorrect")
    if not(t3):
        raise ValueError("Units of ci are incorrect")

    prefac = np.sqrt(7)/(22*np.pi)
    r_eq = (prefac*Lwind/(rhobar*(ci**3)))**(1./2)

    return r_eq.to("pc")

def Rch(Q0, nbar, pdotw, rhobar, ci = 10*u.km/u.s, alphaB = 3.11e-13*(u.cm**3/u.s)):
    # gives the characteristic radius at which the force from photo-ionized gas
    # is equal to that of the wind bubble, defined using the relationship 
    # Rch = Req^4 / RSt^3 and the above relations
    # no need to check for units as that is done above

    r_ch = Req_MD(pdotw,rhobar, ci=ci)**4 / RSt(Q0, nbar,alphaB=alphaB)**3
    return r_ch.to("pc")

def Rwshock(Mdotw, rhobar, Vwind):
    # gives the radius at which the inertia of the swept-up surrounding material is
    # comparable to the wind and the wind shocks
    # Mdotw : wind mass loss rate
    # rhobar : background mass density
    # Vwind : wind velocity

    # start by making sure that the dimensions of the arguments are correct
    t1 = u.get_physical_type(Mdotw*u.s)=="mass"
    t2 = u.get_physical_type(rhobar)=="mass density"
    t3 = u.get_physical_type(Vwind)=="speed"
    if not(t1):
        raise ValueError("Units of Mdotw are incorrect")
    if not(t2):
        raise ValueError("Units of rhobar are incorrect")
    if not(t3):
        raise ValueError("Units of Vwind are incorrect")

    r_wshock = (Mdotw/(4*np.pi*rhobar*Vwind))**(1./2)

    return r_wshock.to("pc")

def Rcl(Mcl, nbar, muH = 1.4):
    # gives the cloud radius
    # Mcl : cloud mass
    # nbar : background number density
    # muH : mean molecular weight

    # start by making sure that the dimensions of the arguments are correct
    t1 = u.get_physical_type(Mcl)=="mass"
    t2 = u.get_physical_type(nbar)=="number density"
    t3 = u.get_physical_type(muH)=="dimensionless"
    if not(t1):
        raise ValueError("Units of Mcl are incorrect")
    if not(t2):
        raise ValueError("Units of nbar are incorrect")
    if not(t3):
        raise ValueError("Units of muH are incorrect")

    r_cl = (3*Mcl/(4*np.pi*nbar*muH*aconsts.m_p))**(1./3)

    return r_cl.to("pc")

#########################################################################################
################################# Time Scale Quantities #################################
#########################################################################################

def Twshock(Mdotw, rhobar, Vwind):
    # gives the wind shock time
    # Mdotw : wind mass loss rate
    # rhobar : background mass density
    # Vwind : wind velocity

    # dimensions are checked in Rwshock
    t_wshock = Rwshock(Mdotw, rhobar, Vwind)/Vwind

    return t_wshock.to("Myr")

def Tcool(nbar, Lwind):
    # give the shell formation/shell cooling time for a
    # a Weaver-like Wind-blown bubble
    # nbar : the number density of hydrgen in the background
    # Lwind : the wind luminosity

    # Equation 8 from Mac Low & McCray (1988)
    t_cool = 2.3e-2*u.Myr
    t_cool *= (nbar/(u.cm**-3))**-0.71
    t_cool *= (Lwind/(1e38*u.erg/u.s))**0.29

    return t_cool


def Trec(nbar, alphaB=3.11e-13*(u.cm**3/u.s)):
    # gives the ionization-recombination time-scale in Myr
    # nbar : the number density of hydrgen in the background
    # alphaB : the case B recombination rate

    # start by making sure that the dimensions of the arguments are correct
    t1 = u.get_physical_type(nbar)=="number density"
    t2 = u.get_physical_type(alphaB)=="volumetric flow rate"
    if not(t1):
        raise ValueError("Units of nbar are incorrect")
    if not(t2):
        raise ValueError("Units of alphaB are incorrect")

    t_ion = (nbar*alphaB)**-1

    return t_ion.to("Myr")

def Tff(rhobar):
    # gives the free-fall timescale in Myr
    # rhobar : mass density

    t1 = u.get_physical_type(rhobar)=="mass density"
    if not(t1):
        raise ValueError("Units of rhobar are incorrect")
    
    t_ff = (3*np.pi/(32*aconsts.G*rhobar))**(1./2)

    return t_ff.to("Myr")

def Teq_MD(pdotw, rhobar, ci = 10*u.km/u.s):
    # gives the time it takes to reach Req
    # if following a momentum-driven solution
    # pdotw : wind momentum input rate
    # rhobar : background mass density
    # ci : ionized gas sound speed

    # start by making sure that the dimensions of the arguments are correct
    t1 = u.get_physical_type(pdotw)=="force"
    t2 = u.get_physical_type(rhobar)=="mass density"
    t3 = u.get_physical_type(ci)=="speed"
    if not(t1):
        raise ValueError("Units of pdotw are incorrect")
    if not(t2):
        raise ValueError("Units of rhobar are incorrect")
    if not(t3):
        raise ValueError("Units of ci are incorrect")

    t_eq = (((3*pdotw/(2*np.pi*rhobar))**(1./2)))/(6*ci**2)

    return t_eq.to("Myr")

def Teq_ED(Lwind, rhobar, ci = 10*u.km/u.s):
    # gives the time it takes to reach equilibrium
    # if following an energy-driven solution
    # Lwind : wind luminosity
    # rhobar : background mass density
    # ci : ionized gas sound speed

    # start by making sure that the dimensions of the arguments are correct
    t1 = u.get_physical_type(Lwind)=="power"
    t2 = u.get_physical_type(rhobar)=="mass density"
    t3 = u.get_physical_type(ci)=="speed"
    if not(t1):
        raise ValueError("Units of Lwind are incorrect")
    if not(t2):
        raise ValueError("Units of rhobar are incorrect")
    if not(t3):
        raise ValueError("Units of ci are incorrect")

    prefac = 0.2*(7**0.75)/((22*np.pi)**0.5)

    t_eq = prefac*(Lwind/(rhobar*(ci**5)))**(1./2)

    return t_eq.to("Myr")

def TSt_MD(Q0, nbar, pdotw, rhobar, ci = 10*u.km/u.s, alphaB = 3.11e-13*(u.cm**3/u.s)):
    # time at which an unimpeded momentum-driven wind bubble would reach
    # the Stromgren Radius parameters defined as above
    t1 = u.get_physical_type(pdotw)=="force"
    t2 = u.get_physical_type(rhobar)=="mass density"
    t3 = u.get_physical_type(ci)=="speed"
    if not(t1):
        raise ValueError("Units of pdotw are incorrect")
    if not(t2):
        raise ValueError("Units of rhobar are incorrect")
    if not(t3):
        raise ValueError("Units of ci are incorrect")
    
    r_st = RSt(Q0,nbar,alphaB=alphaB)
    r_eq = Req_MD(pdotw, rhobar, ci=ci)
    t_eq = Teq_MD(pdotw, rhobar, ci=ci)
    t_st = t_eq*(r_st/r_eq)**(2)

    return t_st.to("Myr")

def TSt_ED(Q0, nbar, Lwind, rhobar, ci = 10*u.km/u.s, alphaB = 3.11e-13*(u.cm**3/u.s)):
    # time at which an unimpeded energy-driven wind bubble would reach
    # the Stromgren Radius parameters defined as above
    t1 = u.get_physical_type(pdotw)=="force"
    t2 = u.get_physical_type(Lwind)=="power"
    t3 = u.get_physical_type(ci)=="speed"
    if not(t1):
        raise ValueError("Units of pdotw are incorrect")
    if not(t2):
        raise ValueError("Units of Lwind are incorrect")
    if not(t3):
        raise ValueError("Units of ci are incorrect")
    
    r_st = RSt(Q0,nbar,alphaB=alphaB)
    t_eq = Teq_ED(Lwind, rhobar, ci=ci)
    r_eq = Req_ED(Lwind, rhobar, ci=ci)    
    t_st = t_eq*(r_st/r_eq)**(5./3)
    return t_st.to("Myr")

def Tdion(Q0, nbar, ci = 10*u.km/u.s, alphaB = 3.11e-13*(u.cm**3/u.s)):
    # the dynamical expansion time of an ionized gas bubble

    # get the stromgren radius
    r_st = RSt(Q0,nbar,alphaB=alphaB)

    t1 = u.get_physical_type(ci)=="speed"
    if not(t1):
        raise ValueError("Units of ci are incorrect")
    
    t_di = np.sqrt(3)*r_st/(2*ci)

    return t_di.to("Myr")