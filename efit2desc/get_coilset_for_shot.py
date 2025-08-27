from scipy.interpolate import interp1d
from scipy.constants import mu_0
from desc.coils import MixedCoilSet, CoilSet, FourierRZCoil, SplineXYZCoil
from desc.magnetic_fields import ToroidalMagneticField, SplineMagneticField
from desc.utils import dot
import numpy as np
from desc.io import load
import io
import jax
import matplotlib.pyplot as plt
import warnings

### Get coil currents from ptdata ###
try:
    from ptdata import fetch_ptdata, PtDataFetcher
except Exception as e:
    print("could not import ptdata, got exception")
    print(e)
from matplotlib.backends.backend_pdf import PdfPages


def get_coilset_for_shot(
    shot,
    time,
    interp_method="nearest",
    use_mgrid_instead_of_coilset=False,
    mgrid_interp_method="linear",
    use_TF_for_Bcoil=False,
    base_coils_file=None,
):
    """_summary_

    Parameters
    ----------
    shot : int
        What shot to get coil currents for
    time : float
        at what time in the shot to get coil currents from
    interp_method : str, optional
        how to interpolate the coil currents to the desired time, by default "nearest"
        This is passed to the `kind` argument of ``scipy.interpolate.interp1d``
    use_mgrid_instead_of_coilset : bool, optional
        Whether to use the mgrid and make a SplineMagneticField instead of
        create a DESC coilset, by default False
    mgrid_interp_method : str, optional
        What method to interpolate the B field from the mgrid, by default "linear"
    use_TF_for_Bcoil : bool
        whether to use an idealized toroidal field for the TF coilset, in which case the
        BCOIL coil current will be set to zero.
        False by default.

    Returns
    -------
    field : CoilSet or SplineMagneticField
        the coilset with the currents for the given shot and time, as read in from
        PTDATA. If ``use_mgrid_instead_of_coilset``, then this is instead a
        ``SplineMagneticField`` based off of an mgrid file for the d3d coilset.

    """
    # first, get the ptnames for all of the coil currents we need to fetch
    ptnames = [
        "ECOILA",
        "ECOILB",
        "E567UP",
        "E567DN",
        "E89DN",
        "E89UP",
        "BCOIL",
        "IP",
    ]

    for i in range(1, 10):
        ptnames.append(f"F{i}A")
        ptnames.append(f"F{i}B")

    C_angs = [19, 79, 139, 199, 259, 319]
    for ang in C_angs:
        ptnames.append(f"C{ang}")

    I_angs = [30, 90, 150, 210, 270, 330]
    for ang in I_angs:
        ptnames.append(f"IU{ang}")
        ptnames.append(f"IL{ang}")

    def update_F_coils(t, coils, interpolable_ptnames):
        # first 18 coils in Andreas' arrays is are the Fcoils
        Fturns = np.array(
            [
                58,
                58,
                58,
                58,
                58,
                55,
                55,
                58,
                55,  # number of turns in Fcoils
                58,
                58,
                58,
                58,
                58,
                55,
                55,
                58,
                55,
            ]
        )
        for i in range(18):
            assert "f" in coils[i].name
            total_curr = interpolable_ptnames[coils[i].name[2:].upper().strip()](t)
            current = total_curr  # the current measured is per turn, and they actually have the number of turns given by Fturns
            for j in range(len(coils[i])):
                coils[i][j].current = current * np.sign(coils[i][j].current)

    """
    IU330 ,  ind = 21
    IU270 ,  ind = 22
    IU210 ,  ind = 23
    IU150 ,  ind = 24
    IU90 ,  ind = 25
    IU30 ,  ind = 26
    IL330 ,  ind = 27
    IL270 ,  ind = 28
    IL210 ,  ind = 29
    IL150 ,  ind = 30
    IL90 ,  ind = 31
    IL30 ,  ind = 32
        """

    def update_IL_coils(t, coils, interpolable_ptnames):
        for i in range(6):
            ind = i + 21
            coils[ind][0].current = interpolable_ptnames[coils[ind].name[3:]](
                t
            ) * np.sign(coils[ind][0].current)

    def update_Iu_coils(t, coils, interpolable_ptnames):
        for i in range(6):
            ind = i + 27
            coils[ind][0].current = interpolable_ptnames[coils[ind].name[3:]](
                t
            ) * np.sign(coils[ind][0].current)

    """
    CCoil79 ,  ind = 33
    CCoil139 ,  ind = 34
    CCoil199 ,  ind = 35
    """

    import re

    nums = [str(i) for i in range(10)]

    def update_C_coils(t, coils, interpolable_ptnames):
        for i in range(3):
            ind = i + 33

            for coil in coils[
                ind
            ]:  # go into the coilset so can set correct currents per coil
                angle_str = ""
                for c in coil.name[2:]:
                    angle_str += c if c in nums else ""
                coil.current = interpolable_ptnames[f"C{angle_str}"](t) * np.sign(
                    coil.current
                )

    def update_all_coil_currents(
        t, total_coils, interpolable_ptnames, use_TF_for_Bcoil=False
    ):
        # expects total_coils to be a MixedCoilSet
        # loaded from "coils.d3d_efbic_kp48_from_andreas"
        # Ecoils is total_coils[19] and [20]
        for c in total_coils[19]:  # eca
            c.current = interpolable_ptnames["ECOILA"](t) * np.sign(c.current)
        for c in total_coils[20]:
            c.current = interpolable_ptnames["ECOILB"](t) * np.sign(c.current)
        # Bcoil is total_coils[18] which is 24 separate coils
        for i in range(len(total_coils[18])):
            total_coils[18][i].current = (
                interpolable_ptnames["BCOIL"](t)
                * 6
                * np.sign(total_coils[18][i].current)
                if not use_TF_for_Bcoil
                else 0.0
            )
        # Fcoils are total_coils[0:18]
        update_F_coils(t, total_coils, interpolable_ptnames)
        # C coils are total_coils[33:36]
        update_C_coils(t, total_coils, interpolable_ptnames)
        # iU coils are total_coils[21:27]
        update_Iu_coils(t, total_coils, interpolable_ptnames)
        # iL coils are total_coils[27:32]
        update_IL_coils(t, total_coils, interpolable_ptnames)

    # coilsfile I got from Andreas Wingen
    coilset_name = base_coils_file  # "coils.d3d_efbic_kp48_from_andreas"

    # get interp1d representations of the currents
    interpolable_ptnames = {}
    PCS_SYS_D3 = ":/fusion/projects/codes/pcs/data/ptdata:/fusion/projects/codes/pcs/data/ptdata/uncomp:"
    for ptname in ptnames:
        # print(ptname)
        try:
            fetcher = PtDataFetcher(ptname, round(shot), sys_d3=PCS_SYS_D3)

            header = fetcher.header
            results = fetcher.fetch()
            interpolable_ptnames[ptname] = interp1d(
                x=results["times"], y=results["data"], kind=interp_method
            )
        except:
            print(f"Shot {shot} does not have ptname {ptname}")

    TF_ideal = ToroidalMagneticField(
        R0=1, B0=mu_0 * interpolable_ptnames["BCOIL"](time) * 144 / 2 / np.pi
    )
    if not use_mgrid_instead_of_coilset:
        if base_coils_file is None:
            base_coils_file = "coils.d3d_efbic_kp48_from_andreas"
            rename_coils = True
        else:
            rename_coils = False
            warnings.warn(
                "Make sure the coilset you load has the correct coil names", UserWarning
            )

        coilset_name = base_coils_file
        full_nominal_coils = MixedCoilSet.from_makegrid_coilfile(
            coilset_name, check_intersection=False, method="linear"
        )

        if rename_coils:
            # I rename his C coils as he has it setup so
            # 79 and 259 are in the same group
            # 139 and 319
            # 199 and 19
            # but we want to have them named acc. to what they individually are
            full_nominal_coils[33][1].name = "34 CCoil259"
            full_nominal_coils[34][1].name = "35 CCoil319"
            full_nominal_coils[35][1].name = "36 CCoil19"

        update_all_coil_currents(
            time, full_nominal_coils, interpolable_ptnames, use_TF_for_Bcoil
        )
        if use_TF_for_Bcoil:
            # TODO: check if sign is ok for this?
            full_nominal_coils = full_nominal_coils + TF_ideal
        return full_nominal_coils
    else:
        # mgrid created with xgrid on PPPL portal cluster, NR=300 NZ=300 Np=48
        # R range: []
        # Z range: []
        mgrid_file_name = "mgrid_d3d_efbic_kp48_nr300_nz300.nc"

        """
        Example extcur for shot 166439 for Andreas coilsfile
!-- F-coils --
  EXTCUR(01) = -3.3476183E+03  EXTCUR(02) = -2.2529044E+03
  EXTCUR(03) = -2.1845469E+03  EXTCUR(04) = -2.1381224E+03
  EXTCUR(05) = -3.4815377E+01  EXTCUR(06) = -2.0800159E+03
  EXTCUR(07) =  3.3240285E+02  EXTCUR(08) =  2.3166404E+03
  EXTCUR(09) =  2.7935908E+03  EXTCUR(10) =  1.6172593E+03
  EXTCUR(11) = -4.2975523E+03  EXTCUR(12) = -4.0290787E+03
  EXTCUR(13) = -4.5535883E+03  EXTCUR(14) = -6.2774472E+03
  EXTCUR(15) =  1.8125020E+03  EXTCUR(16) =  3.4835199E+03
  EXTCUR(17) = -1.5392389E+03  EXTCUR(18) = -6.0826442E+02
  !-- B-coils --
  EXTCUR(19) = -6.7412729E+05
  !-- E-coils --
  EXTCUR(20) = -2.1570434E+04  EXTCUR(21) = -2.1414466E+04
  !-- I-coils --
  EXTCUR(22) =  1.6158750E+03  EXTCUR(23) = -1.6404780E+03
  EXTCUR(24) =  1.6271340E+03  EXTCUR(25) = -1.6288020E+03
  EXTCUR(26) =  1.6279680E+03  EXTCUR(27) = -1.6325550E+03
  EXTCUR(28) =  1.6263000E+03  EXTCUR(29) = -1.6396440E+03
  EXTCUR(30) =  1.6258830E+03  EXTCUR(31) = -1.5387300E+03
  EXTCUR(32) =  1.6250490E+03  EXTCUR(33) = -1.6246320E+03
  !-- C-coils --
  EXTCUR(34) =  1.0540000E+03  EXTCUR(35) = -1.4100000E+02
  EXTCUR(36) = -1.2510000E+03

  
!----- Final comments ---------------------------------------------------
! mapcode boundary fraction is 0.9936
! file input.d3d.166439.04400_lcfs9936_fixed_ns97 generated by
! g2vmi for wingen on Fri Sep  3 18:17:50 2021
! Coil ordering for VMEC differs from that for EFIT
!      1:      f1a      2:     f1b
!      3:      f2a      4:     f2b
!      5:      f3a      6:     f3b
!      7:      f4a      8:     f4b
!      9:      f5a      10:    f5b
!      11:     f6a      12:    f6b
!      13:     f7a      14:    f7b
!      15:     f8a      16:    f8b
!      17:     f9a      18:    f9b
!      19:     B-coil current
!      20:     eca      21:    ecb
!      22:     iu330    23:    iu270
!      24:     iu210    25:    iu150
!      26:     iu90     27:    iu30
!      28:     il330    29:    il270
!      30:     il210    31:    il150
!      32:     il90     33:    il30
!      34:     ccoil79
!      35:     ccoil139
!      36:     ccoil199
!      37:     Current in the B-coil bus feed


        """

        all_coilnames_in_extcur_order = []
        for i in range(1, 10):
            all_coilnames_in_extcur_order.append(f"F{i}A")
            all_coilnames_in_extcur_order.append(f"F{i}B")
        all_coilnames_in_extcur_order.append("BCOIL")
        all_coilnames_in_extcur_order.append("ECOILA")
        all_coilnames_in_extcur_order.append("ECOILB")

        C_angs = [19, 79, 139, 199, 259, 319]
        for ang in C_angs:
            ptnames.append(f"C{ang}")

        I_angs = [330, 270, 210, 150, 90, 30]
        for ang in I_angs:
            ptnames.append(f"IU{ang}")
        for ang in I_angs:
            ptnames.append(f"IL{ang}")
        ptnames.append("C79")
        ptnames.append("C139")
        ptnames.append("C199")
        # our coils file does not have the error field so we ignore last one

        extcur = []
        for name in all_coilnames_in_extcur_order:
            # we don't need the np.sign() stuff here because the coilsfile
            # itself has the - signs in the right places
            extcur.append(interpolable_ptnames[name](time))
        if use_TF_for_Bcoil:
            # zero current for BCOIL if using idealized TF
            extcur[18] = 0.0

        field = SplineMagneticField.from_mgrid(
            mgrid_file_name, extcur, method=mgrid_interp_method
        )
        if use_TF_for_Bcoil:
            field = field + TF_ideal
        return field
