# Models to the evolution of feedback bubbles
# stellar feedback from massive stars
# author: Lachlan Lancaster

import numpy as np
from astropy import units as u
from astropy import constants as ac
import quantities
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

#########################################################################################
########################### CLASSICAL BUBBLE EVOLUTION MODELS ###########################
#########################################################################################

class Bubble():
    def __init__(self, **kwargs):
        self._set_parmeters(**kwargs)
        self._check_parameter_units()

    def _set_parmeters(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if "rho0" not in self.__dict__:
            self.rho0 = 140*ac.m_p/(u.cm**3)

    def _check_parameter_units(self):
        t1 = u.get_physical_type(self.rho0)=="mass density"
        if not(t1):
            raise ValueError("Units of rho0 are incorrect")
    
    def _check_time_units(self, t):
        t1 = u.get_physical_type(t)=="time"
        if not t1:
            raise ValueError("Units of t are incorrect")

    def radius(self, t):
        return 0.0
    
    def velocity(self, t):
        return 0.0
    
    def momentum(self, t):
        return 0.0

    def pressure(self, t):
        return 0.0

class SedovTaylorBW(Bubble):
    # Sedov Taylor Solution for an instantaneous blast wave
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_parmeters(**kwargs)
        self._check_parameter_units()
        super()._check_parameter_units()

    def _set_parmeters(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if "E" not in self.__dict__:
            self.E = 1e51*u.erg
    
    def _check_parameter_units(self):
        t1 = u.get_physical_type(self.E)=="energy"
        if not(t1):
            raise ValueError("Units of E are incorrect")

    def radius(self, t):
        self._check_time_units(t)
        r_ST = 1.15167*(self.E*t**2/(self.rho0))**(1./5)
        return r_ST.to("pc")
    
    def velocity(self, t):
        self._check_time_units(t)
        v_ST = 0.4*self.radius(t)/t
        return v_ST.to("km/s")
    
    def momentum(self, t):
        self._check_time_units(t)
        pr_ST = 4*np.pi*self.rho0*self.radius(t)**3*self.velocity(t)/3
        return pr_ST.to("solMass*km/s")

class Spitzer(Bubble):
    # Spitzer solution for a photo-ionized gas bubble
    # includes the Hosokawa & Inutsuka (2006) correction
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_parmeters(**kwargs)
        self._check_parameter_units()
        super()._check_parameter_units()

        self.nbar = self.rho0/(self.muH*ac.m_p)
        self.RSt = quantities.RSt(self.Q0, self.nbar, alphaB=self.alphaB)
        self.tdio = quantities.Tdion(self.Q0, self.nbar, ci=self.ci, alphaB=self.alphaB)

    def _set_parmeters(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        if "Q0" not in self.__dict__:
            self.Q0 = 1e50/u.s
        if "ci" not in self.__dict__:
            self.ci = 10*u.km/u.s
        if "alphaB" not in self.__dict__:
            self.alphaB = 3.11e-13*(u.cm**3/u.s)
        if "muH" not in self.__dict__:
            self.muH = 1.4
    
    def _check_parameter_units(self):
        t1 = u.get_physical_type(self.Q0)=="frequency"
        t2 = u.get_physical_type(self.ci)=="speed"
        t3 = u.get_physical_type(self.alphaB)=="volumetric flow rate"
        t4 = u.get_physical_type(self.muH)=="dimensionless"
        if not(t1):
            raise ValueError("Units of Q0 are incorrect")
        if not(t2):
            raise ValueError("Units of ci are incorrect")
        if not(t3):
            raise ValueError("Units of alpha_B are incorrect")
        if not(t4):
            raise ValueError("Units of mu_H are incorrect")

    def rhoi(self, t):
        self._check_time_units(t)
        rhoi_sp = self.rho0*(1 + 7*t/(4*self.tdio))**(-3./2)
        return rhoi_sp.to("solMass/pc3")

    def radius(self, t):
        self._check_time_units(t)
        r_sp = self.RSt*(1 + 7*t/(4*self.tdio))**(4./7)
        return r_sp.to("pc")
    
    def velocity(self, t):
        self._check_time_units(t)
        v_sp = (self.RSt/self.tdio)*(1 + 7*t/(4*self.tdio))**(-3./7)
        return v_sp.to("km/s")
    
    def momentum(self, t, adj=True):
        self._check_time_units(t)
        prefac = 4*np.pi*self.rho0*self.RSt**4/(3*self.tdio)
        pr_sp = prefac*(1 + 7*t/(4*self.tdio))**(9./7)
        if adj:
            pr_sp *= (1 - (self.RSt/self.radius(t))**1.5)
        return pr_sp.to("solMass*km/s")
    
    def pressure(self, t):
        self._check_time_units(t)
        press_sp = self.rhoi(t)*self.ci**2
        return (press_sp/ac.k_B).to("K/cm3")

class EnergyDrivenWind(Bubble):
    # Weaver solution for a wind bubble
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_parmeters(**kwargs)
        self._check_parameter_units()
        super()._check_parameter_units()

    def _set_parmeters(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if "Lwind" not in self.__dict__:
            self.Lwind = 1e38*u.erg/u.s
    
    def _check_parameter_units(self):
        t1 = u.get_physical_type(self.Lwind)=="power"
        if not(t1):
            raise ValueError("Units of L_wind are incorrect")

    def radius(self, t):
        self._check_time_units(t)
        r_we = (125*self.Lwind*(t**3)/(154*np.pi*self.rho0))**(1./5)
        return r_we.to("pc")
    
    def velocity(self, t):
        self._check_time_units(t)
        v_we = 0.6*self.radius(t)/t
        return v_we.to("km/s")
    
    def momentum(self, t):
        self._check_time_units(t)
        pr_we = 4*np.pi*self.rho0*self.radius(t)**3*self.velocity(t)/3
        return pr_we.to("solMass*km/s")
    
    def pressure(self, t):
        self._check_time_units(t)
        press_we = (10./33)*self.Lwind*t/((4*np.pi/3)*self.radius(t)**3)
        return (press_we/ac.k_B).to("K/cm3")

class AdiabaticWind(Bubble):
    # Weaver solution Section 2 for an adiabatic wind bubble
    # assumes no radiative losses, even in the shell.
    # We don't treat conduction here either

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_parmeters(**kwargs)
        self._check_parameter_units()
        super()._check_parameter_units()

        self._ad_shell_solve()
        # fraction of the shell's outer radius at which the shell's inner radius lies
        # approximate 0.86, but determined here from the numerical solution
        self.xic = self.ad_shell_sol.t[-1]

    def _set_parmeters(self, **kwargs):
        # scaling paramter for dimensional analysis solution
        # given after Equation 13 of Weaver et al. (1977)
        self.alpha = 0.88
        for key, value in kwargs.items():
            setattr(self, key, value)

        if "Lwind" not in self.__dict__:
            self.Lwind = 1e38*u.erg/u.s
    
    def _check_parameter_units(self):
        t1 = u.get_physical_type(self.Lwind)=="power"
        if not(t1):
            raise ValueError("Units of L_wind are incorrect")

    def radius(self, t):
        self._check_time_units(t)
        r_we = self.alpha*(self.Lwind*(t**3)/self.rho0)**(1./5)
        return r_we.to("pc")
    
    def velocity(self, t):
        self._check_time_units(t)
        v_we = 0.6*self.radius(t)/t
        return v_we.to("km/s")
    
    def momentum(self, t):
        self._check_time_units(t)
        pr_we = 4*np.pi*self.rho0*self.radius(t)**3*self.velocity(t)/3
        return pr_we.to("solMass*km/s")
    
    def pressure(self, t):
        self._check_time_units(t)
        prefac = 5./(22*np.pi*(0.86*slef.alpha)**3)
        press_we = prefac*(self.Lwind**2 * self.rho0**3 / t**4)**(1./5)
        return (press_we/ac.k_B).to("K/cm3")
    
    #####################################################################################
    ########################## FUNCTIONS FOR INTERNAL STRUCTURE #########################
    #####################################################################################
    
    def _ad_shell_solve(self):
        """
        Solves the structure equation for the dimensionless parameters of the shell
        surrounding an adiabatic wind bubble following section 2 of Weaver et al. (1977).
        """

        kappa = -2./3
        gamma = 5./3

        def derivs(xi, ys):
            (U, G, P) = ys
            t1 = kappa*G*(U-xi)/P - 2*gamma/xi - 2*kappa/U
            t2 = gamma - (U-xi)**2 *G/P
            Up = U*(t1/t2)
            t1 = Up + 2*U/xi
            t2 = U-xi
            Gp = -G*(t1/t2)
            Pp = P*(gamma*Gp/G - 2*kappa/(U-xi))
            return (Up, Gp, Pp)

        # stop if density goes to 0
        def event_1(t, ys):
            (U, G, P) = ys
            return G
        event_1.terminal = True

        U0 = 2./(gamma + 1)
        G0 = (gamma + 1)/(gamma - 1)
        P0 = 2/(gamma + 1)
        if not(hasattr(self, "ad_shell_sol")):
            self.ad_shell_sol = solve_ivp(derivs, (1, 0.75), [U0, G0, P0],\
                                          events=[event_1], dense_output=True,\
                                          rtol=1e-12, atol = 1e-12)
        return

class MomentumDrivenWind(Bubble):
    # Momentum-driven bubble solution
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_parmeters(**kwargs)
        self._check_parameter_units()
        super()._check_parameter_units()

    def _set_parmeters(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if "pdotw" not in self.__dict__:
            self.pdotw = 1e5*u.Msun*u.km/u.s/u.Myr

    def _check_parameter_units(self):
        t1 = u.get_physical_type(self.pdotw)=="force"
        if not(t1):
            raise ValueError("Units of pdotw are incorrect")

    def radius(self, t):
        self._check_time_units(t)
        r_md = ((3*self.pdotw*t**2)/(2*np.pi*self.rho0))**(1./4)
        return r_md.to("pc")
    
    def velocity(self, t):
        self._check_time_units(t)
        v_md = 0.5*self.radius(t)/t
        return v_md.to("km/s")
    
    def momentum(self, t):
        self._check_time_units(t)
        pr_md = self.pdotw*t
        return pr_md.to("solMass*km/s")
    
    def pressure(self, t):
        self._check_time_units(t)
        press_md = self.pdotw/(4*np.pi*self.radius(t)**2)
        return (press_md/ac.k_B).to("K/cm3")


#########################################################################################
################################### CO-EVOLUTION MODELS #################################
#########################################################################################

class MD_CEM(Bubble):
    # Joint solution for the evolution of a photo-ionized gas bubble
    # and a momentum-driven wind bubble in force balance with one another
    # assumes that the bubbles are uncoupled and evolve independently
    # up until t_eq, the equilibration time
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_parmeters(**kwargs)
        self._check_parameter_units()
        super()._check_parameter_units()
        self._set_derived_parameters()

        # Separate Spitzer solution
        sp_dict = {"rho0": self.rho0, "Q0": self.Q0, "ci": self.ci,\
                   "alphaB": self.alphaB, "muH": self.muH}
        self.spitz_bubble = Spitzer(**sp_dict)
        # separate momentum-driven wind bubble
        md_dict = {"rho0": self.rho0, "pdotw": self.pdotw}
        self.wind_bubble = MomentumDrivenWind(**md_dict)
        # call ODE integrator to get the joint evolution solution
        self.joint_sol = self.joint_evol()

    def _set_parmeters(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if "Q0" not in self.__dict__:
            self.Q0 = 1e50/u.s
        if "pdotw" not in self.__dict__:
            self.pdotw = 1e5*u.Msun*u.km/u.s/u.Myr
        if "ci" not in self.__dict__:
            self.ci = 10*u.km/u.s
        if "alphaB" not in self.__dict__:
            self.alphaB = 3.11e-13*(u.cm**3/u.s)
        if "muH" not in self.__dict__:
            self.muH = 1.4

    def _check_parameter_units(self):
        # check that the units are correct
        t1 = u.get_physical_type(self.ci)=="speed"
        t2 = u.get_physical_type(self.Q0)=="frequency"
        t3 = u.get_physical_type(self.pdotw)=="force"
        t4 = u.get_physical_type(self.alphaB)=="volumetric flow rate"
        t5 = u.get_physical_type(self.muH)=="dimensionless"
        if not(t1):
            raise ValueError("Units of ci are incorrect")
        if not(t2):
            raise ValueError("Units of Q0 are incorrect")
        if not(t3):
            raise ValueError("Units of pdotw are incorrect")
        if not(t4):
            raise ValueError("Units of alpha_B are incorrect")
        if not(t5):
            raise ValueError("Units of mu_H are incorrect")

    def _set_derived_parameters(self):
        (ci, alphaB, muH) = (self.ci, self.alphaB, self.muH)
        (Q0, pdotw, rho0) = (self.Q0, self.pdotw, self.rho0)
        self.nbar = rho0/(muH*ac.m_p)
        self.RSt = quantities.RSt(Q0, self.nbar, alphaB=alphaB)
        self.teq = quantities.Teq_MD(pdotw, rho0, ci=ci)
        self.Req = quantities.Req_MD(pdotw, rho0, ci=ci)
        self.Rch = quantities.Rch(Q0, self.nbar, pdotw, rho0, ci=ci, alphaB=alphaB)
        self.tdio = quantities.Tdion(Q0, self.nbar, ci=ci, alphaB=alphaB)
        self.tff = quantities.Tff(rho0)
        self.pscl = ((4*np.pi/3)*rho0*(self.Req**4)/self.tdio).to("solMass*km/s")
        self.zeta = (self.Req/self.RSt).to(" ").value

        if self.zeta < 1:
            self.tswitch = self.teq
        else:
            tot = self._get_Tot()
            self.tswitch = min(self.teq.value, tot.value)*u.Myr

    def _get_Tot(self):
        # Returns the time at which the WBB overtakes the PIR
        # used as the switch-over time in the zeta > 1 case.
        # root found in dimensionless form, as in Equation C35
        # of Paper 1
        fac1 = (4.5**0.25)*np.sqrt(self.zeta)
        f  = lambda x: fac1*np.sqrt(x) - (1 + 1.75*x)**(4./7)
        # over-take time only matters if it is smaller than t_eq
        chi_eq = (self.teq/self.tdio).to(" ").value
        try:
            chi_ot = brentq(f, 0, chi_eq)
        except:
            chi_ot = chi_eq
        return (chi_ot*self.tdio).to(u.Myr)

    @staticmethod
    def _get_largest_real(roots):
        real_roots = np.real(roots[np.isreal(roots)])
        return np.max(real_roots)

    def get_xiw(self, xii):
        xiw = []
        for xi in xii:
            p = [self.zeta**-3, 1.0, 0., 0., -1*(xi**3)]
            roots = np.roots(p)
            xiw.append(self._get_largest_real(np.roots(p)))
        return np.array(xiw)

    def joint_evol(self):
        # Gives the solution for the joint dynamical evolution of
        # photo-ionized gas and a wind bubble
        # zeta : the Req/RSt ratio, free parameter of the model

        zeta = self.zeta

        if zeta < 1:
            xiw0 = 1
        else:
            xiw0 = self.wind_bubble.radius(self.tswitch)/self.Req
            xiw0 = xiw0.to(" ").value
        xii0 = xiw0*((1+(zeta**-3)*xiw0)**(1./3))
        momentum_tot = self.wind_bubble.momentum(self.tswitch)
        momentum_tot += self.spitz_bubble.momentum(self.tswitch)
        mass_tot = 4*np.pi*self.rho0*((self.Req*xii0)**3)/3
        psi0 = (((momentum_tot/mass_tot)*(self.tdio/self.Req)).to(" ")).value

        # pre-calculate the relationship between xii and xiw
        xii_range = np.linspace(xii0/2,100*xii0,1000)
        xiw_prec = self.get_xiw(xii_range)

        # defin the differential equations
        def derivs(chi,y):
            xii = y[0]
            psi = y[1]
            xiw = np.interp(xii,xii_range,xiw_prec)
            t1 = (2.25*(zeta**-2)*(1 + (zeta**-3)*xiw)**(2./3))/(xii**3)
            t2 = 3*(psi**2)/xii
            return (psi,t1-t2)

        # use solve_ivp to get solution
        return solve_ivp(derivs,[0,100],[xii0,psi0],dense_output=True)

    def radius(self, t):
        # Returns the radius of the ionized bubble at time t
        # t : the time
        self._check_time_units(t)
        ri = self.spitz_bubble.radius(t)*(t<self.tswitch)
        chi = ((t-self.tswitch)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        ri += solution[0]*self.Req*(t>self.tswitch)
        return ri.to("pc")

    def wind_radius(self, t):
        # Returns the radius of the wind bubble at time t
        # t : the time
        self._check_time_units(t)
        # up until tswitch the wind bubble follows the normal momentum-driven solution
        rw = self.wind_bubble.radius(t)*(t<self.tswitch)
        # afterwards it follows the joint evolution solution
        chi = ((t-self.tswitch)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        xiw = self.get_xiw(solution[0])
        rw += xiw*self.Req*(t>self.tswitch)
        return rw.to("pc")

    def velocity(self, t):
        # Returns the velocity of the ionized bubble at time t
        # t : the time
        self._check_time_units(t)
        # up until tswitch the ionized bubble follows the Spitzer solution
        vi = self.spitz_bubble.velocity(t)*(t<self.tswitch)
        chi = ((t-self.tswitch)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        vi += solution[1]*self.Req/self.tdio*(t>self.tswitch)
        return vi.to("km/s")
    
    def momentum(self, t):
        # returns the momentum carried by the joint bubble at time t
        # t : the time
        self._check_time_units(t)
        prefac = self.pscl
        chi = ((t-self.tswitch)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        pr = prefac*solution[1]*solution[0]**3*(t>self.tswitch)
        pr += self.spitz_bubble.momentum(t)*(t<self.tswitch)
        pr += self.wind_bubble.momentum(t)*(t<self.tswitch)
        return pr.to("solMass*km/s")

    def momentum_uncoupled(self, t):
        # returns the momentum carried by the joint bubble at time t
        # if the two constituent bubbles evolved independently
        # t : the time
        self._check_time_units(t)
        pr = self.spitz_bubble.momentum(t)
        pr += self.wind_bubble.momentum(t)
        return pr.to("solMass*km/s")
    
    def pressure(self, t):
        # returns the pressure of the wind bubble at time t
        # t : the time
        self._check_time_units(t)
        press = self.pdotw/(4*np.pi*self.wind_radius(t)**2)
        return (press/ac.k_B).to("K/cm3")
    
    def pressure_ionized(self, t):
        # returns the pressure of the ionized bubble at time t
        # t : the time
        self._check_time_units(t)
        press = self.spitz_bubble.pressure(t)*(t<self.tswitch)
        press += self.pressure(t)*(t>self.tswitch)
        return press

class ED_CEM(Bubble):
    # Joint solution for the evolution of a photo-ionized gas bubble
    # and a wind bubble in force balance with each other
    # assumes that the bubbles are uncoupled and evolve independently
    # up until t_eq, the equilibration time
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_parmeters(**kwargs)
        self._check_parameter_units()
        super()._check_parameter_units()
        self._set_derived_parameters()

        # Separate Spitzer solution
        sp_dict = {"rho0": self.rho0, "Q0": self.Q0, "ci": self.ci,\
                   "alphaB": self.alphaB, "muH": self.muH}
        self.spitz_bubble = Spitzer(**sp_dict)
        # separate energy-driven wind bubble
        w_dict = {"rho0": self.rho0, "Lwind": self.Lwind}
        self.wind_bubble = EnergyDrivenWind(**w_dict)
        # call ODE integrator to get the joint evolution solution
        self.joint_sol = self.joint_evol()

    def _set_parmeters(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if "Q0" not in self.__dict__:
            self.Q0 = 1e50/u.s
        if "Lwind" not in self.__dict__:
            self.Lwind = 1e38*u.erg/u.s
        if "ci" not in self.__dict__:
            self.ci = 10*u.km/u.s
        if "alphaB" not in self.__dict__:
            self.alphaB = 3.11e-13*(u.cm**3/u.s)
        if "muH" not in self.__dict__:
            self.muH = 1.4

    def _check_parameter_units(self):
        # check that the units are correct
        t1 = u.get_physical_type(self.ci)=="speed"
        t2 = u.get_physical_type(self.Q0)=="frequency"
        t3 = u.get_physical_type(self.Lwind)=="power"
        t4 = u.get_physical_type(self.alphaB)=="volumetric flow rate"
        t5 = u.get_physical_type(self.muH)=="dimensionless"
        if not(t1):
            raise ValueError("Units of ci are incorrect")
        if not(t2):
            raise ValueError("Units of Q0 are incorrect")
        if not(t3):
            raise ValueError("Units of L_wind are incorrect")
        if not(t4):
            raise ValueError("Units of alpha_B are incorrect")
        if not(t5):
            raise ValueError("Units of mu_H are incorrect")


    def _set_derived_parameters(self):
        (ci, alphaB, muH) = (self.ci, self.alphaB, self.muH)
        (Q0, Lwind, rho0) = (self.Q0, self.Lwind, self.rho0)
        self.nbar = rho0/(muH*ac.m_p)
        self.RSt = quantities.RSt(Q0, self.nbar, alphaB=alphaB)
        self.teq = quantities.Teq_ED(Lwind, rho0, ci=ci)
        self.Req = quantities.Req_ED(Lwind, rho0, ci=ci)
        self.tdio = quantities.Tdion(Q0, self.nbar, ci=ci, alphaB=alphaB)
        self.tff = quantities.Tff(rho0)
        self.pscl = ((4*np.pi/3)*rho0*(self.Req**4)/self.tdio).to("solMass*km/s")

        self.zeta = (self.Req/self.RSt).to(" ").value
        if self.zeta < 1:
            self.tswitch = self.teq
        else:
            tot = self._get_Tot()
            self.tswitch = min(self.teq.value, tot.value)*u.Myr

    def _get_Tot(self):
        # Returns the time at which the WBB overtakes the PIR
        # used as the seitch-over time in the zeta > 1 case.
        # root found in dimensionless form, as in Equation C35
        # of Paper 1
        fac1 = ((2.5*np.sqrt(3./7))**0.6)*(self.zeta**0.4)
        f  = lambda x: fac1*(x**0.6) - (1 + 1.75*x)**(4./7)
        # over-take time only matters if it is smaller than t_eq
        chi_eq = (self.teq/self.tdio).to(" ").value
        try:
            chi_ot = brentq(f, 0, chi_eq)
        except:
            chi_ot = chi_eq
        return (chi_ot*self.tdio).to(u.Myr)

    def joint_evol(self):
        # Gives the solution for the joint dynamical evolution of
        # photo-ionized gas and a wind bubble
        # zeta : the Req/RSt ratio, free parameter of the model

        zeta = self.zeta

        if zeta < 1:
            xiw0 = 1
        else:
            xiw0 = self.wind_bubble.radius(self.tswitch)/self.Req
        xii0 = xiw0*((1+(zeta**-3)*xiw0)**(1./3))
        Pfac = self.wind_bubble.pressure(self.tswitch)*ac.k_B/(self.rho0*self.ci**2)
        Pfac = Pfac.to(" ").value
        Et0 = (2./11)*np.sqrt(7./3)*zeta*(xiw0**3)*Pfac
        # get initial condition for derivative of xii -> Mi
        momentum_tot = self.wind_bubble.momentum(self.tswitch)
        momentum_tot += self.spitz_bubble.momentum(self.tswitch)
        mass_tot = 4*np.pi*self.rho0*((self.Req*xii0)**3)/3
        dxii_dchi0 = (((momentum_tot/mass_tot)*(self.tdio/self.Req)).to(" ")).value
        Mi0 = (2*zeta/np.sqrt(3))*dxii_dchi0

        # define the differential equations
        def derivs(chi,y):
            (xii,Mi,xiw,Et) = y
            Pt = (11./2)*np.sqrt(3/7)*Et*(xiw**-3)/zeta
            A = 2/(3*((xiw*zeta)**3)*(Pt**2))
            dlnxii_dchi = (np.sqrt(3)/(2*zeta))*Mi/xii

            dxii_dchi = dlnxii_dchi*xii
            dMi_dchi = (3*np.sqrt(3)/(2*zeta*xii))*(Pt - Mi**2)
            dlnxiw_dchi = (((xii/xiw)**3)*dlnxii_dchi + A/Et)/(1 + 5*A)
            dxiw_dchi = dlnxiw_dchi*xiw
            dlnEt_chi = 1./Et - 2*dlnxiw_dchi
            dEt_dchi = dlnEt_chi*Et
            return (dxii_dchi,dMi_dchi,dxiw_dchi,dEt_dchi)

        # use solve_ivp to get solution
        return solve_ivp(derivs,[0,100],[xii0,Mi0,xiw0,Et0],dense_output=True)

    def radius(self, t):
        # Returns the radius of the ionized bubble at time t
        # t : the time
        self._check_time_units(t)
        ri = self.spitz_bubble.radius(t)*(t<self.tswitch)
        chi = ((t-self.tswitch)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        ri += solution[0]*self.Req*(t>self.tswitch)
        return ri.to("pc")

    def wind_radius(self, t):
        # Returns the radius of the wind bubble at time t
        # t : the time
        self._check_time_units(t)        
        # up until tswitch the wind bubble follows the normal momentum-driven solution
        rw = self.wind_bubble.radius(t)*(t<self.tswitch)
        # afterwards it follows the joint evolution solution
        chi = ((t-self.tswitch)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        xiw = solution[2]
        rw += xiw*self.Req*(t>self.tswitch)
        return rw.to("pc")

    def velocity(self, t):
        # Returns the velocity of the ionized bubble at time t
        # t : the time
        self._check_time_units(t)
        # up until tswitch the ionized bubble follows the Spitzer solution
        vi = self.spitz_bubble.velocity(t)*(t<self.tswitch)
        chi = ((t-self.tswitch)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        vi += solution[1]*self.ci*(t>self.tswitch)
        return vi.to("km/s")
    
    def momentum(self, t):
        # returns the momentum carried by the joint bubble at time t
        # t : the time
        self._check_time_units(t)
        prefac = (4*np.pi/3)*self.Req**3*self.rho0*self.ci
        chi = ((t-self.tswitch)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        pr = prefac*solution[1]*solution[0]**3*(t>self.tswitch)
        pr += self.spitz_bubble.momentum(t)*(t<self.tswitch)
        pr += self.wind_bubble.momentum(t)*(t<self.tswitch)
        return pr.to("solMass*km/s")

    def momentum_uncoupled(self, t):
        # returns the momentum carried by the joint bubble at time t
        # if the two constituent bubbles evolved independently
        # t : the time
        self._check_time_units(t)
        pr = self.spitz_bubble.momentum(t)
        pr += self.wind_bubble.momentum(t)
        return pr.to("solMass*km/s")
    
    def pressure(self, t):
        # returns the pressure of the wind bubble at time t
        # t : the time
        self._check_time_units(t)
        press = self.wind_bubble.pressure(t)*(t<self.tswitch)
        chi = ((t-self.tswitch)/self.tdio).to(" ").value
        solution =  self.joint_sol.sol(chi)
        Pt = (11./2)*np.sqrt(3/7)*solution[3]*(solution[2]**-3)/self.zeta
        Pt = Pt*self.rho0*self.ci**2
        press += Pt*(t>self.tswitch)/ac.k_B
        return (press).to("K/cm3")
    
    def pressure_ionized(self, t):
        # returns the pressure of the ionized bubble at time t
        # t : the time
        self._check_time_units(t)
        press = self.spitz_bubble.pressure(t)*(t<self.tswitch)
        press += self.pressure(t)*(t>self.tswitch)
        return press
