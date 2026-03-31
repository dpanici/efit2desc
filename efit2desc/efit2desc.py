import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import mu_0
from scipy import integrate

from omfit_classes import omfit_eqdsk

from desc.profiles import SplineProfile, PowerSeriesProfile
from desc.equilibrium import Equilibrium
from desc.plotting import *
from desc.grid import QuadratureGrid
from desc.objectives import ForceBalance, ObjectiveFunction
from desc.geometry import FourierRZToroidalSurface, FourierRZCurve
from desc.grid import LinearGrid


def read_EFIT_and_get_fluxsurfs(efitfile, psiN_cutoff=1.0):

    efit = omfit_eqdsk.OMFITgeqdsk(efitfile)
    # run the methods of the OMFITgeqdsk class to get
    # aux and flux surface quantities for the EFIT
    efit.addAuxQuantities()
    efit.addFluxSurfaces(levels=list(np.linspace(0, psiN_cutoff, 129)))
    fluxsurf = efit["fluxSurfaces"]
    Jt = efit["AuxQuantities"]["Jt"]
    # this is the toroidal flux enclosed by the bdry, as calc by EFIT
    efit_Psi = efit["AuxQuantities"]["PHI"]  # [fluxsurfind]
    # this method obtains the iota, etc on the flux surfaces
    fluxsurf.surfAvg()
    return efit


def plot_eq_surfaces_against_efit(efitfile, desc_eq, levels=20):

    efit = read_EFIT_and_get_fluxsurfs(efitfile, 1.0)
    fluxsurf = efit["fluxSurfaces"]

    inds = np.arange(len(fluxsurf["flux"]))[::-10]
    efit_rho = efit["RHOVN"]
    rho_to_plot = efit_rho[inds]
    fig, ax = plot_surfaces(
        desc_eq,
        figsize=(8, 8),
        theta=0,
        rho_lw=3,
        rho=rho_to_plot[np.where(rho_to_plot <= 1.0)],
    )
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
            plt.plot(
                fluxsurf["flux"][k]["R"],
                fluxsurf["flux"][k]["Z"],
                "k--",
                lw=3,
                label="EFIT",
            )
    # also want to add a couple contours outside
    plt.contour(
        efit["AuxQuantities"]["R"],
        efit["AuxQuantities"]["Z"],
        efit["PSIRZ"] - np.max(efit["PSIRZ"]),
        colors=["k"],
        linewidths=[3],
        linestyles=["--"],
    )
    # plt.colorbar()
    plt.axis("equal")
    plt.axis("equal")
    # plt.scatter(Raxis, Zaxis, marker="x", label="EFIT axis", c="k")
    # desc_axis = desc_eq.axis.compute(["R", "Z"])
    # plt.scatter(desc_axis["R"][0], desc_axis["Z"][0], label="DESC Axis")
    plt.legend()
    return fig, ax


### taken from vmeclauncher.py truncateEFIT.py by Wingen ###
# find psi with q(psi) = m/n for all m in [n*qmin, n*qmax]
def scan_q(q, n=3):
    rhos = np.linspace(0, 1.0, 1000)
    qs = q(rhos)
    qmax = np.max(qs)
    qmin = np.min(qs)
    N = int(n * qmax) + 1

    psia = []
    qa = []

    for m in range(int(n * qmin) + 1, N):
        psi = bisec(lambda x: q(x) - float(m) / n, a=0, b=1)
        # print (m, psi, q(psi))
        psia.append(psi)
        qa.append(q(psi))

    # m = int(n * qmax) - 1
    # qbest = (m + 0.2) / n
    # psi = bisec(lambda x: q(x) - qbest, a=0, b=1)

    return psia, qa  # , psi


# ----------------------------------------------------------------------------------------
# find root through bisection
def bisec(funct, a=0, b=1.5):
    eps = 1e-14

    x = a
    f = funct(x)

    if f > 0:
        xo = a
        xu = b
    else:
        xo = b
        xu = a

    while abs(xo - xu) > eps:
        x = (xo + xu) / 2.0
        f = funct(x)
        if f > 0:
            xo = x
        else:
            xu = x

    return x


################################


def plot_eq_iota_against_efit(
    efitfile,
    desc_eq,
    psiN_cutoff=0.996,
    levels=20,
    show_rationals=False,
    max_n=6,
    method="cubic",
):

    efit = read_EFIT_and_get_fluxsurfs(efitfile, psiN_cutoff)
    fluxsurf = efit["fluxSurfaces"]
    # this is the toroidal flux enclosed by the bdry, as calc by EFIT

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

    efit_iota = 1 / fluxsurf["avg"]["q"]

    fig, ax = plt.subplots(dpi=1000)

    if show_rationals:
        qprof = SplineProfile(
            knots=abs(psi_T / np.max(abs(psi_T))),
            values=np.abs(fluxsurf["avg"]["q"]),
            method=method,
        )
        rhos = np.linspace(0, 1, 1000)
        ax.plot(rhos, qprof(rhos), "k--", label="EFIT |q| fit", lw=3)
        for n in range(1, max_n + 1):
            psi, q = scan_q(qprof, n)
            ax.scatter(psi, q, label="n = " + str(n), s=40)
        ax.set_xlabel("psi")
    else:
        fig, ax = plot_1d(desc_eq, "q", label="DESC", lw=6, ax=ax)
        ax.plot(efit_rho, fluxsurf["avg"]["q"], "k--", label="EFIT", lw=6)
    ax.legend()
    return fig, ax


def convert_EFIT_to_DESC(
    efitfile,
    current_or_iota="current",
    profile_type="power_series",
    L=24,
    M=24,
    profile_L=24,
    psiN_cutoff=0.99,
    solve=True,
    solve_options=None,
    plot=True,
    save=True,
    savefolder=".",
    poloidal_angle="polar",
):
    """Read the EFIT file and generate a solved DESC equilibrium.

    Also returns the OMFITgeqdsk class object.

    This function:

    - Reads the EFIT equilibrium from the gfile using ``omfit_eqdsk.OMFITgeqdsk``.
    - Calls ``addAuxQuantities()`` and ``addFluxSurfaces()`` to populate flux-surface
      quantities (geometry, safety factor q = 1/iota, toroidal current density).
    - Finds the LCFS at ``psiN_cutoff`` and parametrizes it with a Fourier series
      using the chosen poloidal angle.
    - Integrates the q profile over poloidal flux ``psi`` to obtain toroidal flux
      ``psi_T(psi)`` and defines the DESC radial variable
      ``rho = sqrt(psi_T / psi_T(bdry))``.
    - Integrates the toroidal current density to get net enclosed toroidal current
      as a flux function (zero at the magnetic axis by construction).
    - Fits pressure, iota = 1/q, and current profiles as functions of ``rho``
      using either a power series or spline.
    - Constructs a DESC ``Equilibrium`` with the LCFS, profiles, and total
      toroidal flux from EFIT.
    - Optionally solves the equilibrium, plots results against EFIT, and saves
      outputs to disk.

    NOTE: up-down asymmetry is assumed by default (``sym=False``).

    Parameters
    ----------
    efitfile : str
        Path to the eqdsk file.
    current_or_iota : {"current", "iota"}
        Whether to fix the current or iota profile in the DESC equilibrium.
    profile_type : {"power_series", "spline"}
        Profile type to use for pressure and iota/current.
    L : int
        Radial spectral resolution for the DESC equilibrium.
    M : int
        Poloidal spectral resolution for the DESC equilibrium.
    profile_L : int
        Radial order for power-series profile fits.
    psiN_cutoff : float
        Normalized poloidal flux value to treat as the LCFS.
    solve : bool
        If False, return the Equilibrium object without solving it. The
        returned equilibrium will not satisfy force balance and will not have
        correct interior flux surfaces.
    solve_options : dict, optional
        Keyword arguments forwarded to ``eq.solve()``. Keys not present are
        filled with defaults: ``ftol=1e-8``, ``gtol=0``, ``xtol=0``,
        ``maxiter=100``, ``verbose=3``, ``objective=ObjectiveFunction(
        ForceBalance(eq, grid=QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=0)))``.
    plot : bool
        Whether to produce comparison plots of flux surfaces and profiles
        against EFIT.
    save : bool
        Whether to save the Equilibrium and figures (figures only if
        ``plot=True``).
    savefolder : str
        Directory to write output files to.
    poloidal_angle : {"arclength", "polar"}
        Poloidal angle convention used when fitting the LCFS boundary.

    Returns
    -------
    eq : desc.equilibrium.Equilibrium
        The DESC ``Equilibrium`` object.
    efit : omfit_classes.omfit_eqdsk.OMFITgeqdsk
        The ``OMFITgeqdsk`` object with the read-in and post-processed EFIT
        data from the gfile.

    """
    if solve_options is None:
        solve_options = {}
    assert poloidal_angle in [
        "arclength",
        "polar",
    ], "poloidal_angle must be one of polar or arclength"
    efitname = os.path.basename(efitfile)
    name = f"{current_or_iota}_{profile_type}_M_{M}_prof_L_{profile_L}_psimax_{psiN_cutoff}"
    efit = read_EFIT_and_get_fluxsurfs(efitfile, psiN_cutoff)
    fluxsurf = efit["fluxSurfaces"]
    # this is the toroidal flux enclosed by the bdry, as calc by EFIT
    efit_Psi = efit["AuxQuantities"]["PHI"]

    # get bdry
    if plot:
        plt.figure()
        for k in range(0, len(fluxsurf["flux"]))[::-10]:
            plt.plot(fluxsurf["flux"][k]["R"], fluxsurf["flux"][k]["Z"])
        plt.axis("equal")

    # choose the LCFS as the bdry
    # TODO: use spectral condensation (ideally when implemented in DESC) to choose a better angle
    lastind = len(fluxsurf["flux"]) - 1
    Rbdry = fluxsurf["flux"][lastind]["R"]
    Zbdry = fluxsurf["flux"][lastind]["Z"]
    Raxis = np.mean(fluxsurf["flux"][0]["R"])
    Zaxis = np.mean(fluxsurf["flux"][0]["Z"])
    x1 = Zbdry - Zaxis
    x2 = Rbdry - Raxis
    # use arclength as the angle
    if poloidal_angle == "arclength":
        arclengths = np.sqrt(
            (Rbdry[1:] - Rbdry[0:-1]) ** 2 + (Zbdry[1:] - Zbdry[0:-1]) ** 2
        )
        arclengths = np.append(
            arclengths,
            np.sqrt((Rbdry[0] - Rbdry[-1]) ** 2 + (Zbdry[0] - Zbdry[-1]) ** 2),
        )
        theta_norm_arclength = integrate.cumulative_trapezoid(y=arclengths, initial=0)
        theta_norm_arclength = (
            theta_norm_arclength / np.max(theta_norm_arclength) * 2 * np.pi
        )
        thetas = theta_norm_arclength
    elif poloidal_angle == "polar":
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
    if plot:
        plt.figure()
        for k in range(0, len(fluxsurf["flux"]))[::-10]:
            plt.plot(fluxsurf["flux"][k]["R"], fluxsurf["flux"][k]["Z"])
        plt.axis("equal")
    if plot:
        plt.plot(data_surf["R"], data_surf["Z"], "k--")
        plt.savefig(savefolder + "/" + f"initial_surfs_and_bdry_{efitname}_{name}.png")

    efit_rho = efit["RHOVN"]
    # so if one were to integrate it over chi, we would get psi_T(chi)

    psi_T = integrate.cumtrapz(
        fluxsurf["avg"]["q"],
        abs(fluxsurf["geo"]["psi"] - np.max(fluxsurf["geo"]["psi"])),
    )

    psi_T = np.insert(psi_T, 0, 0) * 2 * np.pi * -1  # need this factor apparently
    efit_rho = np.sqrt(abs(psi_T / np.max(abs(psi_T))))

    # current[0] is always 0
    current = integrate.cumtrapz(
        fluxsurf["avg"]["dip/dpsi"], fluxsurf["geo"]["psi"], initial=0
    )
    current_spline = SplineProfile(knots=efit_rho, values=current, method="cubic2")
    current_poly = PowerSeriesProfile.from_values(
        efit_rho, current, order=profile_L, sym="even"
    )
    # make current.params[0]=0 strictly to enforce zero on-axis net toroidal current
    # can be nonzero due to small profile_L
    current_poly.params[0] = 0.0

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

    pprof = p_poly if profile_type == "power_series" else p_spline
    iprof = i_poly if profile_type == "power_series" else i_spline
    currprof = current_poly if profile_type == "power_series" else current_spline

    # assign only choosen profile
    iprof = None if current_or_iota == "current" else iprof
    currprof = None if current_or_iota == "iota" else currprof

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
        solve_options.setdefault("ftol", 1e-8)
        solve_options.setdefault("gtol", 0)
        solve_options.setdefault("xtol", 0)
        solve_options.setdefault("maxiter", 100)
        solve_options.setdefault("verbose", 3)
        solve_options.setdefault(
            "objective",
            ObjectiveFunction(
                ForceBalance(eq, grid=QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=0))
            ),
        )
        eq.solve(**solve_options)
    if save:
        eq.save(savefolder + "/" + f"DESC_eq_{efitname}_{name}.h5")

    if plot:
        plot_1d(eq, "iota", label="DESC", lw=3)
        plt.plot(efit_rho, efit_iota, "r--", label="EFIT", lw=3)
        plt.legend()
        if save:
            plt.savefig(savefolder + "/" + f"iota_comp_{efitname}_{name}.png")
        plot_1d(eq, "p", label="DESC", lw=3)
        plt.plot(efit_rho, p, "r--", label="EFIT", lw=3)
        plt.legend()
        if save:
            plt.savefig(savefolder + "/" + f"pressure_comp_{efitname}_{name}.png")

        plot_1d(eq, "current", label="DESC", lw=3)
        plt.plot(efit_rho, current, "r--", label="EFIT", lw=3)
        plt.legend()
        if save:
            plt.savefig(savefolder + "/" + f"current_comp_{efitname}_{name}.png")

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
                plt.plot(
                    fluxsurf["flux"][k]["R"], fluxsurf["flux"][k]["Z"], "k--", lw=3
                )

        plt.axis("equal")
        plt.scatter(Raxis, Zaxis, marker="x", label="EFIT axis", c="k")
        desc_axis = eq.axis.compute(["R", "Z"])
        plt.scatter(desc_axis["R"][0], desc_axis["Z"][0], label="DESC Axis")
        plt.legend()
        if save:
            plt.savefig(
                savefolder + "/" + f"final_surfs_and_bdry_{efitname}_{name}.png"
            )
        plot_surfaces(eq, figsize=(8, 8), rho_lw=3, rho=rho_to_plot)
        if save:
            plt.savefig(
                savefolder
                + "/"
                + f"final_surfs_with_sfl_theta_and_bdry_{efitname}_{name}.png"
            )

    return eq, efit


def compute_betap_li_shaf_integrals(eq, efit=None):
    """Given a DESC eq, compute some common volumetrics.

        NOTE: mainly follows the definitions given in
        Hirshman 1993, which are relations valid for both
        tokamaks and stellarators.
        These have been ~verified against VMEC (at least li, s1,s2,s3 and betai),
        but not yet against EFIT (S integrals especially are very different than s in
        EFIT trees...)

    Parameters
    ----------
    eq : Equilibrium, DESC eq object
    efit : EFIT OMFITclasses dict, if provided, will print li etc from EFIT as well

    Returns
    -------
    _type_
        _description_
    """

    vol_grid = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N, NFP=eq.NFP)
    vol_data = eq.compute(
        ["p", "B_theta", "sqrt(g)", "B_R", "B_Z", "V", "B_phi", "R", "<beta_pol>_vol"],
        grid=vol_grid,
    )

    lcfs_grid = LinearGrid(rho=1.0, M=eq.M_grid, N=eq.N, NFP=eq.NFP, sym=False)
    lcfs_data = eq.compute(
        [
            "p",
            "B_theta",
            "sqrt(g)",
            "B_R",
            "B_Z",
            "V",
            "R0",
            "|e_theta x e_zeta|",
            "S",
            "A",
            "G",
            "a",
            "current",
            "perimeter(z)",
        ],
        grid=lcfs_grid,
    )

    def vol_int(q):  # , grid,vol_data):
        return np.sum(vol_grid.weights * q * vol_data["sqrt(g)"])
        # return np.sum(grid.weights * q )

    def lcfs_int(q):  # ,grid, lcfs_data): # <q>_V from Hirshman 1993 paper def eq 8
        return (
            np.sum(lcfs_grid.weights * q * lcfs_data["|e_theta x e_zeta|"])
            * lcfs_data["V"]
            / lcfs_data["S"]
        )
        # return np.sum(grid.weights * q ) * lcfs_data["R0"] * 2 * np.pi
        # return np.sum(grid.weights * q ) * lcfs_data["V"]/lcfs_data["A"]

    Bsq_P_lcfs = lcfs_data["B_R"] ** 2 + lcfs_data["B_Z"] ** 2
    Bsq_P_vol = vol_data["B_R"] ** 2 + vol_data["B_Z"] ** 2

    I_TF = lcfs_data["G"][-1] / mu_0 * 2 * np.pi
    B_T_bracket = mu_0 * I_TF / 2 / np.pi / lcfs_data["R0"]
    B_T_hat_sq_vol = vol_data["B_phi"] ** 2 - B_T_bracket**2

    B_P_v = lcfs_int(Bsq_P_lcfs)

    musubi = -vol_int(B_T_hat_sq_vol) / B_P_v

    lsubi = vol_int(Bsq_P_vol) / B_P_v  # internal inductance li eqn 9c
    lsubR = vol_int(vol_data["B_R"] ** 2) / B_P_v  # eq 9d
    lsubZ = vol_int(vol_data["B_Z"] ** 2) / B_P_v  # eq 9d

    betai = 2 * mu_0 * vol_int(vol_data["p"]) / B_P_v  # poloidal beta, Hirshman eq 91

    L_i = vol_int(Bsq_P_vol) / mu_0 / lcfs_data["current"][-1] ** 2
    lsubi_current_normalized = 2 * L_i / mu_0 / lcfs_data["R0"]  # fro fusion wiki

    # following Hirhsman 1993
    A = betai + lsubi + musubi  # eq 11a
    B = betai + lsubi - musubi - lsubR * 2  # eq 11b
    C = betai - lsubi - musubi + 2 * lsubR  # eq11c

    # these assume alphabar/R = 1, alphabar_R = 1, betabar_Z = 1, see eq 14 and below of Hisrhman 1993
    sig_hat_R = A + B  # eq 10a alpha=R
    sig_hat_Z = C  # eq10b alpha=Z
    one_over_RT = vol_int(1 / vol_data["R"] * vol_data["p"]) / vol_int(
        vol_data["p"]
    )  # eq 12, pressure-weighted R
    sig_hat_R_alpha_1 = A * one_over_RT  # eq 10a but with alpha=1

    def S1(R_star):  # eq 14a
        return (
            sig_hat_R + sig_hat_Z - R_star * sig_hat_R_alpha_1
        )  # this last one should be assuming alpha=1...

    def S2(R_star):  # eq 14b
        return R_star * sig_hat_R_alpha_1  # this last one should be assuming alpha=1...

    S3 = C  # = sig_hat_Z(Z) eq 14c

    Rgeo = lcfs_data["R0"]
    Rlao = lcfs_data["V"] / 2 / np.pi / lcfs_data["A"]
    Rshaf = 1 / one_over_RT

    fgeo = Rshaf / Rgeo
    flao = Rshaf / Rlao

    # shafranov integrals for different choices of Rstar
    # for some reason, hirshman multiplies the RG and RL by fgeo=Rshaf/Rgeo and flao=Rshaf/Rgeo... dont ask me why
    s1 = S1(Rlao) / 2 / flao
    s2 = S2(Rlao) / 2 / flao
    s3 = S3 / 2
    print("#" * 10)
    print("DESC")
    print("#" * 10)
    print(f"s3 = {0.5*(betai-lsubi-musubi+2*lsubR)}")  # #S3/2, eq 14c
    print(f"{lsubi=}")
    print(f"{lsubi_current_normalized=}")
    print(f"{musubi=}")
    print(f"{betai=}")
    print(
        f"s1 = S1/2 = {S1(1/one_over_RT)/2}  (RT) , {S1(Rgeo)/2/fgeo}  (RG?)  {S1(Rlao)/2/flao}  (RL)"
    )
    print(
        f"s2 = S2/2 = {S2(1/one_over_RT)/2}  (RT) , {S2(Rgeo)/2/fgeo} (RG?)  {S2(Rlao)/2/flao} (RL)"
    )
    print(f"DESC poloidal beta calc: {vol_data['<beta_pol>_vol']}")
    # FIXME: does this actually work?
    Bpave = lcfs_data["current"][-1] * mu_0 / lcfs_data["perimeter(z)"][-1]
    beta_p_with_efit_formula = vol_int(
        lcfs_data["p"][-1] / Bpave**2 / 2 / mu_0 / vol_data["V"]
    )
    print(f"DESC poloidal beta w/efit formula: {beta_p_with_efit_formula}")

    # compute same defs as EFIT
    circum = lcfs_data["perimeter(z)"][-1]
    vol = lcfs_data["V"]
    # TODO: this is actually vac center??
    # https://github.com/gafusion/OMFIT-source/blob/ebfff46939e1e8a56ff6add2fd617ccbf80eee1f/omfit/omfit_classes/fluxSurface.py#L1746
    r_0 = lcfs_data["R0"]
    r_axis = r_0
    ip = lcfs_data["current"][-1]
    Bp2_vol = vol_int(Bsq_P_vol)
    li_from_definition = Bp2_vol / vol / mu_0 / mu_0 / ip / ip * circum * circum
    desc_li = {
        "li_from_definition": li_from_definition,
        "li_(1)_TLUCE": li_from_definition
        / circum
        / circum
        * 2
        * vol
        / r_0
        * 1,  # correction_factor, has to do w/ kappa
        "li_(2)_TLUCE": li_from_definition / circum / circum * 2 * vol / r_axis,
        "li_(3)_TLUCE": li_from_definition / circum / circum * 2 * vol / r_0,
        "li_(1)_EFIT": circum * circum * Bp2_vol / (vol * mu_0 * mu_0 * ip * ip),
        "li_(3)_IMAS": 2 * Bp2_vol / r_0 / ip / ip / mu_0 / mu_0,
    }

    if efit is not None:
        print("#" * 10)
        print("EFIT")
        print("#" * 10)
        print(f"EFIT Poloidal beta: {efit['fluxSurfaces']['avg']['beta_p'][-1]}")
        for key in desc_li.keys():
            line = f"{key}: DESC = {desc_li[key]:1.4f}"
            efit_line = (
                f"EFIT = {efit['fluxSurfaces']['info']['internal_inductance'][key]}"
            )
            print(f"{line}" + " " * len(line) + efit_line)

        # print(f"EFIT li: {efit["fluxSurfaces"]['info']['internal_inductance']}")

    return {
        "betai": betai,
        "<beta_pol>_vol": vol_data["<beta_pol>_vol"],
        "li": lsubi,
        "s1": s1,
        "s2": s2,
        "s3": s3,
    }
