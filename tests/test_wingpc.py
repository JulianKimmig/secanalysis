import pytest
from secanalysis.sec_formats.win_gpc import WinGPCData
import matplotlib.pyplot as plt


@pytest.fixture
def wingpc_data(request):
    with open("tests/win_gpc_sample.txt", "r") as f:
        lines = f.readlines()
    return lines


HEADERLINES = (0, 83)
RAWLINES = (83, 1887)
ELULINES = (1888, 3695)
MWDLINES = (3696, 4303)


def _prep_wgpc_lines(
    lines,
    with_header=True,
    with_elu=True,
    with_mwd=True,
    with_volume=True,
    with_time=True,
):
    if not with_time:
        # remove the time column: in raw lines drop everything after the last \t
        for i in range(*RAWLINES):
            if "\t" not in lines[i]:
                continue
            lines[i] = lines[i].split("\t")[:-1]
            lines[i] = "\t".join(lines[i]) + "\n"

    if not with_volume:
        # remove the volume column: in raw lines drop everything before the first \t
        for i in range(*RAWLINES):
            if "\t" not in lines[i]:
                continue
            lines[i] = lines[i].split("\t")[1:]
            lines[i] = "\t".join(lines[i])

    lineas_to_drop = []

    if not with_header:
        lineas_to_drop.extend(range(83))
    if not with_elu:
        lineas_to_drop.extend(range(1888, 3695))

    if not with_mwd:
        lineas_to_drop.extend(range(3696, 4303))

    # Drop the lines
    for i in sorted(set(lineas_to_drop), reverse=True):
        lines.pop(i)

    return lines


@pytest.mark.parametrize(
    "with_header", [True, False], ids=["with_header", "without_header"]
)
@pytest.mark.parametrize("with_elu", [True, False], ids=["with_elu", "without_elu"])
@pytest.mark.parametrize("with_mwd", [True, False], ids=["with_mwd", "without_mwd"])
@pytest.mark.parametrize(
    "with_volume", [True, False], ids=["with_volume", "without_volume"]
)
@pytest.mark.parametrize("with_time", [True, False], ids=["with_time", "without_time"])
def test_wingpc(
    wingpc_data, with_header, with_elu, with_mwd, with_volume, with_time, request
):
    wingpc_data = _prep_wgpc_lines(
        wingpc_data, with_header, with_elu, with_mwd, with_volume, with_time
    )

    if not with_time and not with_volume:
        with pytest.raises(ValueError):
            WinGPCData.from_string("".join(wingpc_data))
        return

    if not with_header and not with_volume:
        with pytest.raises(KeyError):
            WinGPCData.from_string("".join(wingpc_data))
        return
    gpc = WinGPCData.from_string("".join(wingpc_data))

    expected_raw_data_cols = [WinGPCData.DEFAULT_VOLUME_COLUMN, "RID"]
    assert len(expected_raw_data_cols) == len(gpc._raw_data.columns)
    assert all(col in gpc._raw_data.columns for col in expected_raw_data_cols), (
        f"Expected columns {expected_raw_data_cols} not found in raw data"
    )

    # if not with_header:
    #     gpc.set_calibration_function(
    #         lambda x: 2.608178e01
    #         + -2.957393e00 * x
    #         + 1.419570e-01 * x**2
    #         + -2.489230e-03 * x**3
    #     )
    # v = gpc.raw_volumes
    # s = gpc.raw_signals
    # y1 = gpc._calibration_function(v)
    # y2 = 2.608178e01 + -2.957393e00 * v + 1.419570e-01 * v**2 + -2.489230e-03 * v**3
    # np.testing.assert_allclose(y1, y2)
    # gy1 = -np.gradient(gpc._calibration_function(v), v)
    # gy2 = -(-2.957393e00 + 2 * 1.419570e-01 * v + 3 * -2.489230e-03 * v**2)

    # np.testing.assert_allclose(gy1, gy2, rtol=2e-2)

    # mass_range_target = 10**y2
    # np.testing.assert_allclose(gpc.mass_range, mass_range_target, rtol=1e-3)

    # Signal_norm_0_to_1 = (s - s.min(0)) / (s.max(0) - s.min(0))

    # mass_f = (Signal_norm_0_to_1.T / (mass_range_target * gy2)).T
    # wm, mass = gpc.calc_Wm()
    # np.testing.assert_allclose(wm, mass_f, rtol=5e-3)

    # detection
    gpc.autodetect_signal_boarders()

    # gpc.add_signal_boarder(mass=(8e3, 1e6))
    fig, _ = gpc.plot(
        # lower_volume_cutoff=10,
        # upper_volume_cutoff=20,
    )

    # Save the figure
    # test_id = request.node.name  # Gets the pytest-generated test name with param IDs
    # os.makedirs("test_outputs", exist_ok=True)
    # fig.savefig(f"test_outputs/{test_id}.png")
    plt.close(fig)  # Optional: close to free memory
