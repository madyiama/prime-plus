"""Keck WFS seeing is already on-axis; do not apply airmass twice.

Existing generic AO .ini files use zenith seeing by default. Keck
telemetry-generated .ini files must declare SeeingReference='line_of_sight'.
"""

import numpy as np
import pytest

from p3.aoSystem.aoSystem import aoSystem, r0_to_science_line_of_sight


def test_keck_n0120_wfs_r0_roundtrips_without_double_airmass():
    r0_tel = 0.11474041274562248
    wvl500 = 500e-9
    wvl_sci = 1.530e-6
    airmass = 1.3583588411686751
    seeing_tel = 0.976 * wvl500 / r0_tel * (180*3600/np.pi)
    inverse_r0 = 0.976 * wvl500 / seeing_tel * (180*3600/np.pi)
    assert inverse_r0 == pytest.approx(r0_tel)

    r0_los = r0_to_science_line_of_sight(
        inverse_r0, airmass, "line_of_sight"
    )
    assert r0_los == pytest.approx(r0_tel)
    r0_sci = r0_los*(wvl_sci/wvl500)**(6/5)
    assert r0_sci == pytest.approx(0.43911888960217066, rel=1e-12)
    # The previous Keck code used zenith scaling on a LOS value.
    previously = r0_to_science_line_of_sight(
        inverse_r0, airmass, "zenith"
    ) * (wvl_sci/wvl500)**(6/5)
    assert previously == pytest.approx(0.3654040960224877, rel=1e-12)


def test_existing_generic_ao_config_retains_zenith_default():
    airmass = 1.4
    r0_zenith = 0.16
    assert r0_to_science_line_of_sight(
        r0_zenith, airmass
    ) == pytest.approx(r0_zenith * airmass**(-3/5))


@pytest.mark.parametrize("reference", ["unknown", "LOS", "", None])
def test_seeing_reference_is_explicit_and_strict(reference):
    with pytest.raises(ValueError, match="SeeingReference"):
        r0_to_science_line_of_sight(0.12, 1.2, reference)


def test_ao_system_parses_seeing_reference_from_config_map():
    system = aoSystem.__new__(aoSystem)
    system.my_data_map = {"atmosphere": {
        "SeeingReference": "line_of_sight", "Seeing": 0.9
    }}
    assert system._atmosphere_seeing_reference() == "line_of_sight"
    del system.my_data_map["atmosphere"]["SeeingReference"]
    assert system._atmosphere_seeing_reference() == "zenith"
