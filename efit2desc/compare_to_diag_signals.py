from scipy.interpolate import interp1d
from scipy.constants import mu_0
from desc.coils import MixedCoilSet, CoilSet, FourierRZCoil, SplineXYZCoil
from desc.magnetic_fields import (
    ToroidalMagneticField,
    SumMagneticField,
    SplineMagneticField,
)
from desc.utils import dot
import numpy as np
from desc.io import load
import pandas as pd
import io
import jax
import matplotlib.pyplot as plt
from desc.equilibrium import Equilibrium

### Get coil currents from ptdata ###
try:
    from ptdata import fetch_ptdata, PtDataFetcher
except Exception as e:
    print("could not import ptdata, got exception")
    print(e)
from matplotlib.backends.backend_pdf import PdfPages
from .get_coilset_for_shot import get_coilset_for_shot

PCS_SYS_D3 = ":/fusion/projects/codes/pcs/data/ptdata:/fusion/projects/codes/pcs/data/ptdata/uncomp:"


def compute_Bp_probe_signals_from_DESC(
    coilset, coil_grid=None, eq=None, coords=None, angles=None, names=None
):
    """Given a DESC coilset and optionally an Equilibrium, compute Bp probe signals

    Parameters
    ----------
    coilset : CoilSet
        _description_
    coil_grid : Grid, optional
        _description_, by default None
    eq : Equilibrium, optional
        _description_, by default None
    coords : (n,3) array
        locations of the probes in R,phi,Z coordinates. If None, defaults to the
        locations stored in source code
    angles : (n,) array
        angles of the probes in the R,Z plane w.r.t the horizontal axis. If None, defaults to the
        angles stored in source code
    names : length-n list
        names of magnetic probes corresponding to the passed-in coords and angles.


    Returns
    -------
    coil_B_aligned_with_directions_Bp_probes : array
        array of the signal computed at each probe location/direction.

    Bp_probe_data : dict
        Dictionary of the probe ptnames, and the corresponding signal computed
        from DESC. This signal is comprised of both the response to the coil field
        and (if eq was passed in), the plasma-generated field, as computed by the
        virtual casing principle.

    """
    ## read in the .csv to get the positions of the poloidal probes
    ## and their directions
    # TODO: store these not in a csv or move the csv into efit2desc and in the init, have this
    #    csv be read and the data stored as a dataframe that I can just import from the module
    # TODO: assert that if any of names, coords, angles is not None, then they all must be passed in
    if names is None:
        data = pd.read_csv("d3d_coils_pickup_coils_only.csv", sep=",")
        data.dropna(
            inplace=True, subset=["Name"]
        )  # removes all rows with any NaN in them
        ptnames_Bp_probes = data["Name"].values
        # need to make "coords" array
        coords_Bp_probes = np.vstack(
            [data["R"].values, data["phi"].values, data["Z"].values]
        ).T
        dir_phis_Bp_probes = np.zeros_like(data["gam"].values)
        dir_Rs_Bp_probes = np.cos(data["gam"].values / 180 * np.pi)
        dir_Zs_Bp_probes = np.sin(data["gam"].values / 180 * np.pi)
        directions_Bp_probes = np.vstack(
            [dir_Rs_Bp_probes, dir_phis_Bp_probes, dir_Zs_Bp_probes]
        ).T
    else:
        data = {}
        ptnames_Bp_probes = names
        coords_Bp_probes = coords
        data["gam"] = np.atleast_1d(angles)  # check this is right
        dir_phis_Bp_probes = np.zeros_like(data["gam"])
        dir_Rs_Bp_probes = np.cos(data["gam"] / 180 * np.pi)
        dir_Zs_Bp_probes = np.sin(data["gam"] / 180 * np.pi)
        directions_Bp_probes = np.vstack(
            [dir_Rs_Bp_probes, dir_phis_Bp_probes, dir_Zs_Bp_probes]
        ).T

    # compute coil contribution to diagnostic probe response
    Bp_probe_data = {}
    coil_B_aligned_with_directions_Bp_probes = dot(
        coilset.compute_magnetic_field(coords_Bp_probes, source_grid=coil_grid),
        directions_Bp_probes,
    )

    # compute plasma contribution (optionally)
    if eq is not None:
        # FIXME: fill this in
        assert isinstance(eq, Equilibrium)
        from desc.objectives._reconstruction import PointBMeasurement

        obj = PointBMeasurement(
            eq,
            ToroidalMagneticField(0.0, 0.0),
            coords_Bp_probes,
            directions=directions_Bp_probes,
            coils_fixed=True,
        )
        obj.build()
        Bplasma_contrib = obj.compute(eq.params_dict)
        coil_B_aligned_with_directions_Bp_probes += Bplasma_contrib
    for name, measurement in zip(
        ptnames_Bp_probes, coil_B_aligned_with_directions_Bp_probes
    ):
        Bp_probe_data[name] = measurement
    return coil_B_aligned_with_directions_Bp_probes, Bp_probe_data


def compute_flux_loop_signals_from_DESC(
    coilset, coil_grid=None, flux_loop_grid=None, eq=None, coords=None, names=None
):
    """Given a DESC coilset and optionally an Equilibrium, compute flux loop signals

    Parameters
    ----------
    coilset : CoilSet
        _description_
    coil_grid : Grid, optional
        Grid to discretize coil filaments for biot-savart law
    coil_grid : Grid, optional
        Grid to discretize the flux loops with for computation of the
        flux through them via the A.dl loop integral

    eq : Equilibrium, optional
        _description_, by default None
    coords : (n,2) array
        locations of the flux loops in R,Z coordinates. If None, defaults to the
        locations stored in source code
    names : length-n list
        names of the flux loops corresponding to the passed-in coords.


    Returns
    -------
    flux_loop_signals : array
        array of the signal computed at each flux loop.

    flux_loop_data : dict
        Dictionary of the flux loop ptnames, and the corresponding signal computed
        from DESC. This signal is comprised of both the response to the coil field
        and (if eq was passed in), the plasma-generated field, as computed by the
        virtual casing principle.

    """
    ## read in the .csv to get the positions of the poloidal probes
    ## and their directions
    # TODO: store these not in a csv or move the csv into efit2desc and in the init, have this
    #    csv be read and the data stored as a dataframe that I can just import from the module
    # TODO: assert that if any of names, coords, angles is not None, then they all must be passed in
    if names is None:
        raise NotImplementedError("No Default flux loop coordinates yet")
    else:
        data = {}
        ptnames_flux_loops = names
        coords_flux_loops = np.atleast_2d(coords)

    flux_loops = CoilSet(
        [
            FourierRZCoil(current=0.0, R_n=r, Z_n=z, sym=False)
            for r, z in coords_flux_loops
        ],
        check_intersection=False,
    )
    # compute coil contribution to diagnostic probe response
    flux_loop_data = {}

    # compute plasma contribution (optionally)
    if eq is not None:
        # FIXME: fill this in
        assert isinstance(eq, Equilibrium)
        from desc.objectives._reconstruction import FluxLoop

        obj = FluxLoop(
            eq,
            coilset,
            flux_loops,
            flux_loop_grid=flux_loop_grid,
            field_grid=coil_grid,
            coils_fixed=True,
        )
        obj.build()
        flux_loop_signals = obj.compute(eq.params_dict)
    else:
        try:
            from desc.objectives._reconstruction import FluxLoop

            obj = FluxLoop(
                Equilibrium(),
                coilset,
                flux_loops,
                flux_loop_grid=flux_loop_grid,
                field_grid=coil_grid,
                coils_fixed=True,
                vacuum=True,
            )
            obj.build()
            flux_loop_signals = obj.compute(eq.params_dict)
        except Exception as e:
            print(e)
            # do integral manually
            flux_loop_signals = []
            flux_loops_pos_data = flux_loops.compute(["x", "x_s"], grid=flux_loop_grid)
            A = coilset.compute_magnetic_vector_potential(
                np.vstack([d["x"] for d in flux_loops_pos_data]), source_grid=coil_grid
            )
            for i in range(len(flux_loops)):
                print(i)
                this_A = A[
                    i * flux_loop_grid.num_nodes : (i + 1) * flux_loop_grid.num_nodes
                ]
                A_dot_dxds = np.sum(this_A * flux_loops_pos_data[i]["x_s"], axis=1)
                Psi = np.sum(flux_loop_grid.spacing[:, 2] * A_dot_dxds)
                flux_loop_signals.append(Psi)
            flux_loop_signals = np.asarray(flux_loop_signals)
    for name, measurement in zip(ptnames_flux_loops, flux_loop_signals):
        flux_loop_data[name] = measurement
    return flux_loop_signals, flux_loop_data


def compare_to_ptdata_diags_Bp_probes(
    shot,
    time,
    pdf_savename,
    coilset_in=None,
    coil_grid=None,
    eq=None,
    get_coilset_kwargs={},
):

    if coilset_in is None:
        coilset_in = get_coilset_for_shot(shot, time, **get_coilset_kwargs)
    coilset = coilset_in.copy()
    ptnames = [
        "BTI66M053"
    ]  # tor. field probe at R=2.414, Z=-0.271, phi=52.6, with 0.1 deg. inclination rel. to horizontal in the Phi plane

    data = pd.read_csv("d3d_coils_pickup_coils_only.csv", sep=",")
    data.dropna(inplace=True, subset=["Name"])  # removes all rows with any NaN in them
    ptnames_Bp_probes = data["Name"].values
    # need to make "coords" array
    coords_Bp_probes = np.vstack(
        [data["R"].values, data["phi"].values, data["Z"].values]
    ).T
    # as well as a "directions" array based off gam
    # for the poloidal probes (which these are), gam is the angle
    # from the horizontal in the R-Z plane
    # assuming gam=0 is pointing in the +R direction
    # and gam increases in the CCW direction in the R-Z plane
    # gam is in degrees
    dir_phis_Bp_probes = np.zeros_like(data["gam"].values)
    dir_Rs_Bp_probes = np.cos(data["gam"].values / 180 * np.pi)
    dir_Zs_Bp_probes = np.sin(data["gam"].values / 180 * np.pi)
    directions_Bp_probes = np.vstack(
        [dir_Rs_Bp_probes, dir_phis_Bp_probes, dir_Zs_Bp_probes]
    ).T

    ptnames += list(ptnames_Bp_probes)

    shot = round(shot)

    bad_names = []
    interpolable_ptnames = {}
    print(shot)
    for ptname in ptnames:
        # print(ptname)
        try:
            fetcher = PtDataFetcher(ptname, round(shot), sys_d3=PCS_SYS_D3)

            header = fetcher.header
            results = fetcher.fetch()
            interpolable_ptnames[ptname] = interp1d(
                x=results["times"], y=results["data"], kind="nearest"
            )
        except:
            print(f"Shot {shot} does not have ptname {ptname}")
            bad_names.append(ptname)
    bad_name_inds = np.asarray(
        [i for i, name in enumerate(ptnames_Bp_probes) if name not in bad_names]
    )
    probe_names_minus_bad_names = [
        name for name in ptnames_Bp_probes if name not in bad_names
    ]

    poloidal_probe_coords_minus_bad_ones = coords_Bp_probes[bad_name_inds, :]
    poloidal_probe_directions_minus_bad_ones = directions_Bp_probes[bad_name_inds, :]

    print("#" * 15)
    print(f"Shot {shot}")
    print("#" * 15)
    print(f"t = {time}")

    # poloidal probes
    Bps_measured = [
        interpolable_ptnames[name](time) for name in probe_names_minus_bad_names
    ]
    coil_B_aligned_with_directions_Bp_probes = dot(
        coilset.compute_magnetic_field(
            poloidal_probe_coords_minus_bad_ones, source_grid=coil_grid
        ),
        poloidal_probe_directions_minus_bad_ones,
    )

    # print(
    #     f"avg ratio of my B to measured B: {np.mean(coil_B_aligned_with_directions_Bp_probes / np.asarray(Bps_measured))}"
    # )
    # errs = np.asarray(coil_B_aligned_with_directions_Bp_probes) - np.asarray(
    #     Bps_measured
    # )
    # print(f"Average Abs error in poloidal probes: {np.mean(abs(errs))}")
    # print(f"Max Abs error in poloidal probes: {np.max(abs(errs))}")
    # print(f"Min Abs error in poloidal probes: {np.min(abs(errs))}")

    # print(
    #     f"Average rel abs error in poloidal probes: {np.mean(abs(errs/Bps_measured))}"
    # )
    # print(f"Max rel abs error in poloidal probes: {np.max(abs(errs/Bps_measured))}")
    # print(f"Min rel abs error in poloidal probes: {np.min(abs(errs/Bps_measured))}")
    if pdf_savename is not None:
        pdf_page = PdfPages(pdf_savename)
        plt.figure()
        plt.scatter(
            np.arange(len(coil_B_aligned_with_directions_Bp_probes)),
            coil_B_aligned_with_directions_Bp_probes,
            label="My Calculated",
        )
        plt.scatter(
            np.arange(len(coil_B_aligned_with_directions_Bp_probes)),
            Bps_measured,
            marker="x",
            label="Measured",
        )
        plt.xlabel("Probe Index")
        plt.ylabel("Field (T)")
        plt.legend()
        plt.title(f"Shot {int(shot)} Time = {time} ms")
        fig_mag = plt.gcf()
        pdf_page.savefig(fig_mag)
        plt.figure()
        plt.scatter(
            np.arange(len(coil_B_aligned_with_directions_Bp_probes)),
            coil_B_aligned_with_directions_Bp_probes,
            label="My Calculated",
        )
        plt.scatter(
            np.arange(len(coil_B_aligned_with_directions_Bp_probes)),
            Bps_measured,
            marker="x",
            label="Measured",
        )
        plt.yscale("symlog", linthresh=0.01)
        plt.xlabel("Probe Index")
        plt.ylabel("Field (T)")
        plt.legend()
        plt.title(f"Shot {int(shot)} Time = {time} ms")
        fig_mag_zoom = plt.gcf()
        pdf_page.savefig(fig_mag_zoom)
        plt.figure()
        plt.scatter(
            np.arange(len(coil_B_aligned_with_directions_Bp_probes)),
            coil_B_aligned_with_directions_Bp_probes,
            label="My Calculated",
        )
        plt.xlabel("Probe Index")
        plt.ylabel("Field (T)")
        plt.legend()
        plt.title(f"Shot {int(shot)} Time = {time} ms")
        fig_mag2 = plt.gcf()
        pdf_page.savefig(fig_mag2)
        plt.figure()
        plt.scatter(
            np.arange(len(coil_B_aligned_with_directions_Bp_probes)),
            coil_B_aligned_with_directions_Bp_probes / np.asarray(Bps_measured),
        )
        plt.xlabel("Probe Index")
        plt.ylabel("ratio of my B to measured B")
        plt.legend()
        plt.title(f"Shot {int(shot)} Time = {time} ms")
        fig_mag3 = plt.gcf()
        pdf_page.savefig(fig_mag3)

        # plot the currents for the coils as a bar chart
        plt.figure(figsize=(16, 8))
        if isinstance(coilset, SumMagneticField):
            coilset = coilset[0]
        if not isinstance(coilset, SplineMagneticField):
            # TODO: can just use the extcurs to get this I think as well...
            current_names = []
            currents = []
            for c in coilset:
                while hasattr(c, "len"):
                    c = c[0]
                current = c.current
                name = c.name
                if hasattr(current, "len"):
                    current = c.current[0]
                currents.append(c.current)
                current_names.append(name)
            try:
                plt.bar(current_names, currents)
                plt.ylabel("Current (A)")
                plt.xticks(rotation=30, ha="right")
                plt.title("currents in the coilset")

                fig_currents = plt.gcf()
                pdf_page.savefig(fig_currents)
            except Exception as e:
                print(e)
                print(currents)
                print(current_names)
        plt.figure()
        plt.quiver(
            poloidal_probe_coords_minus_bad_ones[:, 0],
            poloidal_probe_coords_minus_bad_ones[:, 2],
            poloidal_probe_directions_minus_bad_ones[:, 0],
            poloidal_probe_directions_minus_bad_ones[:, 2],
            label="Bp probes",
        )
        # ind_of_highest_measurement = np.argmax(np.abs(Bps_measured))
        # plt.quiver(
        #     poloidal_probe_coords_minus_bad_ones[ind_of_highest_measurement, 0],
        #     poloidal_probe_coords_minus_bad_ones[ind_of_highest_measurement, 2],
        #     poloidal_probe_directions_minus_bad_ones[ind_of_highest_measurement, 0],
        #     poloidal_probe_directions_minus_bad_ones[ind_of_highest_measurement, 2],
        #     color="red",
        #     label="Highest B measured",
        # )
        # plt.legend()

        plt.title("Probe locations")
        pdf_page.savefig(plt.gcf())

        pdf_page.close()
    return Bps_measured, probe_names_minus_bad_names


def compare_to_ptdata_diags_Br_probes(
    shot,
    time,
    pdf_savename,
    coilset_in=None,
    coil_grid=None,
    eq=None,
    get_coilset_kwargs={},
):

    if coilset_in is None:
        coilset_in = get_coilset_for_shot(shot, time, **get_coilset_kwargs)
    coilset = coilset_in.copy()
    ptnames = [
        "BTI66M053"
    ]  # tor. field probe at R=2.414, Z=-0.271, phi=52.6, with 0.1 deg. inclination rel. to horizontal in the Phi plane

    data = pd.read_csv("d3d_coils_pickup_coils_only.csv", sep=",")
    data.dropna(inplace=True, subset=["Name"])  # removes all rows with any NaN in them
    ptnames_Bp_probes = data["Name"].values
    # need to make "coords" array
    coords_Bp_probes = np.vstack(
        [data["R"].values, data["phi"].values, data["Z"].values]
    ).T
    # as well as a "directions" array based off gam
    # for the poloidal probes (which these are), gam is the angle
    # from the horizontal in the R-Z plane
    # assuming gam=0 is pointing in the +R direction
    # and gam increases in the CCW direction in the R-Z plane
    # gam is in degrees
    dir_phis_Bp_probes = np.zeros_like(data["gam"].values)
    dir_Rs_Bp_probes = np.cos(data["gam"].values / 180 * np.pi)
    dir_Zs_Bp_probes = np.sin(data["gam"].values / 180 * np.pi)
    directions_Bp_probes = np.vstack(
        [dir_Rs_Bp_probes, dir_phis_Bp_probes, dir_Zs_Bp_probes]
    ).T

    ptnames += list(ptnames_Bp_probes)

    shot = round(shot)

    bad_names = []
    interpolable_ptnames = {}
    print(shot)
    for ptname in ptnames:
        # print(ptname)
        try:
            fetcher = PtDataFetcher(ptname, round(shot), sys_d3=PCS_SYS_D3)

            header = fetcher.header
            results = fetcher.fetch()
            interpolable_ptnames[ptname] = interp1d(
                x=results["times"], y=results["data"], kind="nearest"
            )
        except:
            print(f"Shot {shot} does not have ptname {ptname}")
            bad_names.append(ptname)
    bad_name_inds = np.asarray(
        [i for i, name in enumerate(ptnames_Bp_probes) if name not in bad_names]
    )
    probe_names_minus_bad_names = [
        name for name in ptnames_Bp_probes if name not in bad_names
    ]

    poloidal_probe_coords_minus_bad_ones = coords_Bp_probes[bad_name_inds, :]
    poloidal_probe_directions_minus_bad_ones = directions_Bp_probes[bad_name_inds, :]

    print("#" * 15)
    print(f"Shot {shot}")
    print("#" * 15)
    print(f"t = {time}")

    # poloidal probes
    Bps_measured = [
        interpolable_ptnames[name](time) for name in probe_names_minus_bad_names
    ]
    coil_B_aligned_with_directions_Bp_probes = dot(
        coilset.compute_magnetic_field(
            poloidal_probe_coords_minus_bad_ones, source_grid=coil_grid
        ),
        poloidal_probe_directions_minus_bad_ones,
    )

    # print(
    #     f"avg ratio of my B to measured B: {np.mean(coil_B_aligned_with_directions_Bp_probes / np.asarray(Bps_measured))}"
    # )
    # errs = np.asarray(coil_B_aligned_with_directions_Bp_probes) - np.asarray(
    #     Bps_measured
    # )
    # print(f"Average Abs error in poloidal probes: {np.mean(abs(errs))}")
    # print(f"Max Abs error in poloidal probes: {np.max(abs(errs))}")
    # print(f"Min Abs error in poloidal probes: {np.min(abs(errs))}")

    # print(
    #     f"Average rel abs error in poloidal probes: {np.mean(abs(errs/Bps_measured))}"
    # )
    # print(f"Max rel abs error in poloidal probes: {np.max(abs(errs/Bps_measured))}")
    # print(f"Min rel abs error in poloidal probes: {np.min(abs(errs/Bps_measured))}")

    pdf_page = PdfPages(pdf_savename)
    plt.figure()
    plt.scatter(
        np.arange(len(coil_B_aligned_with_directions_Bp_probes)),
        coil_B_aligned_with_directions_Bp_probes,
        label="My Calculated",
    )
    plt.scatter(
        np.arange(len(coil_B_aligned_with_directions_Bp_probes)),
        Bps_measured,
        marker="x",
        label="Measured",
    )
    plt.xlabel("Probe Index")
    plt.ylabel("Field (T)")
    plt.legend()
    plt.title(f"Shot {int(shot)} Time = {time} ms")
    fig_mag = plt.gcf()
    pdf_page.savefig(fig_mag)
    plt.figure()
    plt.scatter(
        np.arange(len(coil_B_aligned_with_directions_Bp_probes)),
        coil_B_aligned_with_directions_Bp_probes,
        label="My Calculated",
    )
    plt.scatter(
        np.arange(len(coil_B_aligned_with_directions_Bp_probes)),
        Bps_measured,
        marker="x",
        label="Measured",
    )
    plt.yscale("symlog", linthresh=0.01)
    plt.xlabel("Probe Index")
    plt.ylabel("Field (T)")
    plt.legend()
    plt.title(f"Shot {int(shot)} Time = {time} ms")
    fig_mag_zoom = plt.gcf()
    pdf_page.savefig(fig_mag_zoom)
    plt.figure()
    plt.scatter(
        np.arange(len(coil_B_aligned_with_directions_Bp_probes)),
        coil_B_aligned_with_directions_Bp_probes,
        label="My Calculated",
    )
    plt.xlabel("Probe Index")
    plt.ylabel("Field (T)")
    plt.legend()
    plt.title(f"Shot {int(shot)} Time = {time} ms")
    fig_mag2 = plt.gcf()
    pdf_page.savefig(fig_mag2)
    plt.figure()
    plt.scatter(
        np.arange(len(coil_B_aligned_with_directions_Bp_probes)),
        coil_B_aligned_with_directions_Bp_probes / np.asarray(Bps_measured),
    )
    plt.xlabel("Probe Index")
    plt.ylabel("ratio of my B to measured B")
    plt.legend()
    plt.title(f"Shot {int(shot)} Time = {time} ms")
    fig_mag3 = plt.gcf()
    pdf_page.savefig(fig_mag3)

    # plot the currents for the coils as a bar chart
    plt.figure(figsize=(16, 8))
    if isinstance(coilset, SumMagneticField):
        coilset = coilset[0]
    if not isinstance(coilset, SplineMagneticField):
        # TODO: can just use the extcurs to get this I think as well...
        current_names = []
        currents = []
        for c in coilset:
            current = c.current
            name = c.name
            if hasattr(current, "len"):
                current = c.current[0]
            currents.append(c.current)
            current_names.append(name)

        plt.bar(
            current_names,
        )
        plt.ylabel("Current (A)")
        plt.xticks(rotation=30, ha="right")
        plt.title("currents in the coilset")

        fig_currents = plt.gcf()
        pdf_page.savefig(fig_currents)

    plt.figure()
    plt.quiver(
        poloidal_probe_coords_minus_bad_ones[:, 0],
        poloidal_probe_coords_minus_bad_ones[:, 2],
        poloidal_probe_directions_minus_bad_ones[:, 0],
        poloidal_probe_directions_minus_bad_ones[:, 2],
        label="Bp probes",
    )
    # ind_of_highest_measurement = np.argmax(np.abs(Bps_measured))
    # plt.quiver(
    #     poloidal_probe_coords_minus_bad_ones[ind_of_highest_measurement, 0],
    #     poloidal_probe_coords_minus_bad_ones[ind_of_highest_measurement, 2],
    #     poloidal_probe_directions_minus_bad_ones[ind_of_highest_measurement, 0],
    #     poloidal_probe_directions_minus_bad_ones[ind_of_highest_measurement, 2],
    #     color="red",
    #     label="Highest B measured",
    # )
    # plt.legend()

    plt.title("Probe locations")
    pdf_page.savefig(plt.gcf())

    pdf_page.close()
