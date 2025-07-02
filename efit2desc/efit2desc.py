
from omfit_classes import omfit_eqdsk
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from desc.profiles import SplineProfile, PowerSeriesProfile
from desc.equilibrium import Equilibrium
from desc.plotting import *
from desc.grid import QuadratureGrid
from desc.objectives import ForceBalance, ObjectiveFunction
from desc.geometry import FourierRZToroidalSurface, FourierRZCurve
from desc.grid import LinearGrid

#FIXME: these should be in init!!
from compare_to_diag_signals import compute_Bp_probe_signals_from_DESC
from get_coilset_for_shot import get_coilset_for_shot

def convert_EFIT_to_DESC(
    efitfile,
    current_or_iota="current",
    profile_type="power_series",
    L=24,
    M=24,
    profile_L=24,
    psiN_cutoff = 0.99,
    solve=True,
    plot=True,
    save=True,
    savefolder=".",
):
    """Read the EFIT file and generate a solved DESC equilibrium, as well as return the OMFITgeqdsk class object.

    This function:
    - Read the EFIT equilibrium information from the gfile using the ``omfit_eqdsk.OMFITgeqdsk`` class
    - Use the ``omfit_eqdsk.OMFITgeqdsk.addAuxQuantities()`` and ``omfit_eqdsk.OMFITgeqdsk.addFluxSurfaces()`` functions to 
     populate the object with flux-surface quantities such as the flux surface geometries and the safety factor (q = 1/iota) and toroidal current density
    - Find the last-closed-flux-surface based on the passed-in desired psiN_cutoff
     and parametrize this surface with a Fourier series based off of a geometric poloidal angle
    - integrates the q profile over the poloidal flux ``psi`` in order to get the toroidal flux ``psi_T(chi)``
      and uses it to define the DESC radial variable ``rho = sqrt(psi_T/psi_T(bdry))``
    - integrates the toroidal current density in order to the net toroidal current as a flux function
      and shifts the toroidal current such that at the magnetic axis, there is zero net enclosed toroidal current
    - fits the pressure, 1/q profile and the net toroidal current profiles as functions of ``rho``, with either a power series
    or a spline, in order to form the necessary inputs for a DESC fixed-boundary Equilibrium.
    - creates a DESC ``Equilibrium`` object using the LCFS, profiles, and net enclosed toroidal flux calculated from EFIT
    - Optionally, solves this equilibrium in DESC using by default the grid ``QuadratureGrid(L=eq.L_grid, M=eq.M, N=0)``
    - Optionally, plots the results against the EFIT profiles and the EFIT flux surfaces
    - returns the DESC ``Equilibrium`` object as well as the ``omfit_eqdsk.OMFITgeqdsk`` object

    NOTE: up-down asymmetry is assumed by default.

    Paramters
    ---------

    efitfile: str, 
        Path to eqdsk file
    current_or_iota: {"current", "iota"} 
        Whether to fix the iota or current profile
    profile_type: {"power_series", "spline"}, 
        What type of Profile to use for pressure and iota/current
    L,M : int, 
        Radial/poloidal spectral resolution to use for DESC equilibrium
    profile_L : int, 
        Radial resolution to use for the profile fits (if using a power series)
    psiN_cutoff : int, 
        Which normalized poloidal flux to cut the EFIT equilibrium off at and consider as the LCFS for the DESC Equilibrium
    solve : bool, 
        Whether or not to solve the DESC Equilibrium before returning. if False, will return an unsolved DESC Equilibrium object,
         which will not be in equilibrium and will not have the correct interior flux surfaces.
    plot : bool, 
        Whether or not to create the plots showing the initial and final flux surfaces and profiles compared to EFIT.
    save : bool, 
        Whether or not to save the Equilibrium and the plots (plots are saved only if ``plot=True``)
    savefolder : str, 
        What folder to save the Equilibrium and figures to (if save=True)

    Returns
    -------
    eq : desc.equilibrium.Equilibrium, the DESC ``Equilibrium`` object.
    efit : omfit_classes.omfit_eqdsk.OMFITgeqdsk, the ``OMFITgeqdsk`` object containing the read-in and post-processed EFIT data from the gfile.


    """

    efit = omfit_eqdsk.OMFITgeqdsk(efitfile)
    # run the methods of the OMFITgeqdsk class to get
    # aux and flux surface quantities for the EFIT
    efit.addAuxQuantities()
    efit.addFluxSurfaces(levels=list(np.linspace(0, psiN_cutoff, 129)))
    fluxsurf = efit["fluxSurfaces"]
    Jt = efit["AuxQuantities"]["Jt"]
    # this is the toroidal flux enclosed by the bdry, as calc by EFIT
    efit_Psi = efit["AuxQuantities"]["PHI"]  # [fluxsurfind]
    name = f"{current_or_iota}_{profile_type}_M_{M}_prof_L_{profile_L}_psimax_{psiN_cutoff}"  # _surfind_{fluxsurfind}"
    # this method obtains the iota, etc on the flux surfaces
    fluxsurf.surfAvg()

    # get bdry
    plt.figure()
    for k in range(0, len(fluxsurf["flux"]))[::-10]:
        plt.plot(fluxsurf["flux"][k]["R"], fluxsurf["flux"][k]["Z"])
    plt.axis("equal")

    # choose the LCFS as the bdry
    #TODO: use spectral condensation (ideally when implemented in DESC) to choose a better angle
    lastind = len(fluxsurf["flux"]) - 1
    Rbdry = fluxsurf["flux"][lastind]["R"]
    Zbdry = fluxsurf["flux"][lastind]["Z"]
    Raxis = np.mean(fluxsurf["flux"][0]["R"])
    Zaxis = np.mean(fluxsurf["flux"][0]["Z"])
    x1 = Zbdry - Zaxis
    x2 = Rbdry - Raxis
    thetas = np.arctan2(x1, x2)

    surface = FourierRZToroidalSurface.from_values(
        coords=np.vstack([Rbdry, np.zeros_like(Rbdry), Zbdry]).T,
        theta=thetas,
        sym=False,
        NFP=1,
        M=20,
        N=0,
    )
    plt.figure()
    for k in range(0, len(fluxsurf["flux"]))[::-10]:
        plt.plot(fluxsurf["flux"][k]["R"], fluxsurf["flux"][k]["Z"])
    plt.axis("equal")

    # choose the LCFS as the bdry
    lastind = len(fluxsurf["flux"]) - 1
    Rbdry = fluxsurf["flux"][lastind]["R"]
    Zbdry = fluxsurf["flux"][lastind]["Z"]
    Raxis = np.mean(fluxsurf["flux"][0]["R"])
    Zaxis = np.mean(fluxsurf["flux"][0]["Z"])
    x1 = Zbdry - Zaxis
    x2 = Rbdry - Raxis
    thetas = np.arctan2(x1, x2)

    surface = FourierRZToroidalSurface.from_values(
        coords=np.vstack([Rbdry, np.zeros_like(Rbdry), Zbdry]).T,
        theta=thetas,
        sym=False,
        NFP=1,
        M=20,
        N=0,
    )
    data_surf = surface.compute(["R", "Z"], grid=LinearGrid(M=50, rho=1.0))
    plt.plot(data_surf["R"], data_surf["Z"], "k--")
    plt.savefig(savefolder + "/" + f"initial_surfs_and_bdry_{efitfile}_{name}.png")

    efit_rho = efit["RHOVN"]
    # so if one were to integrate it over chi, we would get psi_T(chi)

    psi_T = integrate.cumtrapz(
        fluxsurf["avg"]["q"],
        abs(fluxsurf["geo"]["psi"] - np.max(fluxsurf["geo"]["psi"])),
    )

    psi_T = np.insert(psi_T, 0, 0) * 2 * np.pi * -1  # need this factor apparently
    efit_rho = np.sqrt(abs(psi_T / np.max(abs(psi_T))))

    current = integrate.cumtrapz(
        fluxsurf["avg"]["dip/dpsi"], fluxsurf["geo"]["psi"], initial=0
    )
    current_shifted = current - current[0]  # make current[0]=0 for spline fit
    current_spline = SplineProfile(
        knots=efit_rho, values=current_shifted, method="cubic2"
    )
    current_poly = PowerSeriesProfile.from_values(
        efit_rho, current, order=profile_L, sym="even"
    )
    current_poly.params[0] = 0.0  # make current[0]=0 for poly to enforce zero on-axis net toroidal current

    p = fluxsurf["avg"]["P"]
    p_spline = SplineProfile(knots=efit_rho, values=p)
    p_poly = PowerSeriesProfile.from_values(efit_rho, p, order=profile_L, sym="even")

    efit_iota = 1 / fluxsurf["avg"]["q"]

    i_spline = SplineProfile(knots=efit_rho, values=efit_iota)
    i_poly = PowerSeriesProfile.from_values(
        efit_rho, efit_iota, order=profile_L, sym="even"
    )

    # make axis initial guess from the Raxis, Zaxis earlier
    axis = FourierRZCurve(R_n=Raxis, Z_n=Zaxis, sym=False, modes_R=[0], modes_Z=[0])

    iprof = i_poly if profile_type == "power_series" else i_spline
    iprof = None if current_or_iota == "current" else iprof

    currprof = current_poly if profile_type == "power_series" else current_spline
    currprof = None if current_or_iota == "iota" else currprof

    pprof = p_poly if profile_type == "power_series" else p_spline

    efit_Psi = psi_T[-1]
    eq = Equilibrium(
        surface=surface,
        axis=axis,
        pressure=pprof,
        iota=iprof,
        current=currprof,
        sym=False,
        Psi=efit_Psi,
        M=M,
        L=L,
    )
    if solve:
        eq.solve(
            ftol=1e-12,
            maxiter=150,
            gtol=0,
            verbose=3,
            xtol=0,
            objective=ObjectiveFunction(
                ForceBalance(eq, grid=QuadratureGrid(L=eq.L_grid, M=eq.M, N=0))
            ),
        )
    if save:
        eq.save(savefolder + "/" + f"DESC_eq_{efitfile}_{name}.h5")

    if plot:
        plot_1d(eq,"iota", label="DESC")
        plt.plot(efit_rho, efit_iota, "--", label="EFIT")
        plt.legend()
        if save:    
            plt.savefig(savefolder + "/" + f"iota_comp_{efitfile}_{name}.png")

        plot_1d(eq, "current")
        plt.plot(efit_rho, current, "--", label="EFIT")
        plt.legend()
        if save:
            plt.savefig(savefolder + "/" + f"current_comp_{efitfile}_{name}.png")

        plt.figure()
        inds = np.arange(len(fluxsurf["flux"]))[::-10]
        rho_to_plot = efit_rho[inds]
        plot_surfaces(eq, figsize=(8, 8), theta=0, rho_lw=3, rho=rho_to_plot)
        is_labelled = False
        for k in inds:
            if not is_labelled:
                plt.plot(
                    fluxsurf["flux"][k]["R"],
                    fluxsurf["flux"][k]["Z"],
                    "k--",
                    label="EFIT",
                    lw=3,
                )
                is_labelled = True
            else:
                plt.plot(fluxsurf["flux"][k]["R"], fluxsurf["flux"][k]["Z"], "k--", lw=3)

        plt.axis("equal")
        plt.scatter(Raxis, Zaxis, marker="x", label="EFIT axis", c="k")
        desc_axis = eq.axis.compute(["R", "Z"])
        plt.scatter(desc_axis["R"][0], desc_axis["Z"][0], label="DESC Axis")
        plt.legend()
        if save:
            plt.savefig(savefolder + "/" + f"final_surfs_and_bdry_{efitfile}_{name}.png")

    return eq, efit

            
            
            
            
            
            
