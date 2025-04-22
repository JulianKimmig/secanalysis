from charset_normalizer import from_path, from_bytes
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, List, Tuple
import pybaselines
from scipy.signal import savgol_filter, find_peaks
from ..auto_detect import estimate_baseline_regions


class SECDataBase(ABC):
    DEFAULT_VOLUME_COLUMN = "volume"
    """
    Base class for SEC data parsers.

    This class provides a common interface for parsing SEC data files.

    Attributes:
        volume: 1D numpy array of type float representing the volume data.
    """

    def __init__(self, raw_data: pd.DataFrame):
        if self.DEFAULT_VOLUME_COLUMN not in raw_data.columns:
            raise ValueError(
                f"Volume column '{self.DEFAULT_VOLUME_COLUMN}' not found in raw data."
            )
        if len(raw_data.columns) < 2:
            raise ValueError(
                "Raw data must contain at least 2 columns (volume and signal)."
            )
        self._raw_data = raw_data
        self._mass_range = None
        self._calibration_function = None
        self._calibration_function_params = None
        self.signal_boarders: List[Tuple[int, int, Optional[int]]] = []
        self._basline_corrected = None

    @property
    def raw_volumes(self) -> np.ndarray:
        """
        :return: The raw volumes
        """
        return self._raw_data[self.DEFAULT_VOLUME_COLUMN].values

    @property
    def raw_signals(self) -> np.ndarray:
        """
        :return: The raw signals
        """
        return self._raw_data.drop(columns=[self.DEFAULT_VOLUME_COLUMN]).values

    @property
    def signal_names(self) -> list:
        """
        :return: The signal names
        """
        cols = self._raw_data.columns
        cols = cols[cols != self.DEFAULT_VOLUME_COLUMN]
        return list(cols)

    @classmethod
    def from_file(cls, path: Path):
        enc = from_path(path).best().encoding
        with open(path, "r", encoding=enc) as f:
            content = f.read()

        return cls.from_string(content)

    @classmethod
    def from_bytes(cls, data: bytes):
        """
        :param data: The bytes to parse
        :return: An instance of the class
        """
        enc = from_bytes(data).best().encoding
        content = data.decode(enc)
        return cls.from_string(content)

    @classmethod
    @abstractmethod
    def from_string(cls, string: str):
        """
        :param string: The string to parse
        :return: An instance of the class
        """
        raise NotImplementedError("Subclasses must implement from_string()")

    def _plot_elu(
        self,
        fig: plt.Figure,
        total: int,
        current: int,
        lower_cutoff: int,
        upper_cutoff: int,
        plot_boarders: List[Tuple[int, int, Optional[int]]],
    ):
        raw_ax = fig.add_subplot(total * 100 + 10 + current)

        l1 = raw_ax.plot(
            self.raw_volumes,
            self.basline_corrected_signal,
            label=self.signal_names,
        )

        for b1, b2, p in plot_boarders:
            raw_ax.axvspan(
                self.raw_volumes[b1],
                self.raw_volumes[b2],
                color="red",
                alpha=0.3,
                # label="Signal Boarder",
            )
        raw_ax.set_xlabel("Volume")
        raw_ax.set_ylabel("Signal")
        raw_ax.set_title("Elugram")

        mass_ax = raw_ax.twinx()

        mass_ax._get_lines = raw_ax._get_lines
        mass_ax.set_ylabel("Mass")
        mass_ax.set_yscale("log")
        handel = l1
        if self.mass_range is not None:
            l2 = mass_ax.plot(
                self.raw_volumes,
                self.mass_range,
                label="mass",
            )
            handel += l2

        labs = [hand.get_label() for hand in handel]
        mass_ax.legend(handel, labs)
        raw_ax.grid()

        return raw_ax, mass_ax

    def _plot_elu2(
        self,
        fig: plt.Figure,
        total: int,
        current: int,
        lower_cutoff: int,
        upper_cutoff: int,
        plot_boarders: List[Tuple[int, int, Optional[int]]],
    ):
        raw_ax = fig.add_subplot(total * 100 + 10 + current)

        for b1, b2, p in plot_boarders:
            sig = self.basline_corrected_signal[b1:b2]
            sig = sig - sig.min()
            sig = sig / sig.max()
            raw_ax.plot(
                self.raw_volumes[b1:b2],
                sig,
            )

        raw_ax.set_xlabel("Volume")
        raw_ax.set_ylabel("Signal")
        raw_ax.set_title("Elugram")

        # mass_ax = raw_ax.twinx()

        # mass_ax._get_lines = raw_ax._get_lines
        # mass_ax.set_ylabel("Mass")
        # mass_ax.set_yscale("log")

        # l2 = mass_ax.plot(
        #     self.raw_volumes,
        #     self.mass_range,
        #     label="mass",
        # )

        # lns = l1 + l2
        # labs = [l.get_label() for l in lns]
        # mass_ax.legend(lns, labs)
        # raw_ax.grid()

        return raw_ax  # , mass_ax

    def _plot_raw(
        self,
        fig: plt.Figure,
        total: int,
        current: int,
        lower_cutoff: int,
        upper_cutoff: int,
        plot_boarders: List[Tuple[int, int, Optional[int]]],
    ):
        raw_ax = fig.add_subplot(total * 100 + 10 + current)

        raw_ax.plot(
            self.raw_volumes,
            self.raw_signals,
            label=self.signal_names,
        )

        for b1, b2, p in plot_boarders:
            raw_ax.axvspan(
                self.raw_volumes[b1],
                self.raw_volumes[b2],
                color="red",
                alpha=0.3,
                # label="Signal Boarder",
            )
        raw_ax.set_xlabel("Volume")
        raw_ax.set_ylabel("Signal")
        raw_ax.set_title("SEC Data")

        return raw_ax

    def _plot_mwd(
        self,
        fig: plt.Figure,
        total: int,
        current: int,
        lower_cutoff: int,
        upper_cutoff: int,
        plot_boarders: List[Tuple[int, int, Optional[int]]],
    ):
        raw_ax = fig.add_subplot(total * 100 + 10 + current)

        if self.mass_range is not None:
            max_height = -np.inf
            for i, (b1, b2, p) in enumerate(plot_boarders):
                m1, m2 = self.mass_range[[b2, b1]]
                wm_log, mass = self.calc_logMW(m1, m2)
                max_height = max(max_height, wm_log.max())

                # raise ValueError(f"Mn: {mn}, Mw: {mw}, Mz: {mz}, m1: {m1}, m2: {m2}")

                raw_ax.plot(
                    mass,
                    wm_log,
                    label=self.signal_names,
                )
                # for i, (b1, b2, p) in enumerate(plot_boarders):

                m1, m2 = self.mass_range[[b2, b1]]
                massparams = self.calc_mass_params(m1, m2)
                plt.vlines(
                    [mp for mp in massparams.values()],
                    0,
                    max_height,
                    color=raw_ax.lines[i].get_color(),
                    linestyle="--",
                )
                for i, (key, value) in enumerate(
                    sorted(
                        massparams.items(),
                        key=lambda x: wm_log[np.argmin(np.abs(mass - x[1]))],
                        reverse=True,
                    )
                ):
                    idx = np.argmin(np.abs(mass - value))
                    raw_ax.annotate(
                        f"{key}: {value.mean():.2e}",
                        xy=(value, wm_log[idx]),
                        # xytext=(value, (wm_log[idx] + wm_log.max()) / 2),
                        xytext=(2, -i),
                        #   color=_plot[0].get_color(),
                        fontsize=8,
                        # rotation=45,
                        arrowprops=dict(arrowstyle="->"),
                        textcoords="offset fontsize",
                        ha="left",
                    )

        # raise ValueError("test")
        # Set the x-axis to be logarithmic
        raw_ax.set_xscale("log")
        raw_ax.set_xlabel("Mass")
        raw_ax.legend()

        return raw_ax

    def plot(
        self,
        lower_mass_cutoff: float = None,
        upper_mass_cutoff: Optional[float] = None,
        lower_volume_cutoff: Optional[float] = None,
        upper_volume_cutoff: Optional[float] = None,
    ) -> plt.Figure:
        """
        Plot the signals.
        """
        numberplots = 1
        lower_cutoff = 0
        upper_cutoff = len(self.raw_volumes) - 1
        try:
            masses = self.mass_range  # [::-1]
            if lower_mass_cutoff is not None:
                upper_cutoff = np.argmin(np.abs(masses - lower_mass_cutoff))
            if upper_mass_cutoff is not None:
                lower_cutoff = np.argmin(np.abs(masses - upper_mass_cutoff))

            if lower_volume_cutoff is not None:
                lower_cutoff = max(
                    lower_cutoff,
                    np.argmin(np.abs(self.raw_volumes - lower_volume_cutoff)),
                )
            if upper_volume_cutoff is not None:
                upper_cutoff = min(
                    upper_cutoff,
                    np.argmin(np.abs(self.raw_volumes - upper_volume_cutoff)),
                )

            numberplots += 1
        except Exception:
            pass

        boarders = self.signal_boarders.copy()
        if not boarders:
            boarders = [(lower_cutoff, upper_cutoff, None)]

        plot_boarders = []
        for b1, b2, p in boarders:
            if p is None:
                if (b1 >= lower_cutoff) and (b2 <= upper_cutoff):
                    plot_boarders.append(
                        (b1, b2, np.argmax(self.raw_signals[b1:b2]) + b1)
                    )
            else:
                if p >= lower_cutoff and p <= upper_cutoff:
                    plot_boarders.append((b1, b2, p))

        tot = 4
        fig = plt.figure(
            figsize=(12, tot * 6),
            dpi=300,
        )

        axes = {
            "raw": self._plot_raw(
                fig,
                tot,
                1,
                lower_cutoff,
                upper_cutoff,
                plot_boarders,
            ),
            "elu": self._plot_elu(
                fig,
                tot,
                2,
                lower_cutoff,
                upper_cutoff,
                plot_boarders,
            ),
            "elu2": self._plot_elu2(
                fig,
                tot,
                3,
                lower_cutoff,
                upper_cutoff,
                plot_boarders,
            ),
            "mwd": self._plot_mwd(
                fig,
                tot,
                4,
                lower_cutoff,
                upper_cutoff,
                plot_boarders,
            ),
        }

        plt.tight_layout()

        return fig, axes

    def calc_mass_params(self, lower_mass_cutoff: float, upper_mass_cutoff: float):
        """
        Calculates Mn (Number average molecular mass),
        Mw (Weight average molecular mass)
        and Mz (Z average molecular mass)
        :param lower_mass_cutoff: Lower mass cutoff
        :param upper_mass_cutoff: Upper mass cutoff
        :return: Mn, Mw, Mz
        """
        wlog, mass = self.calc_logMW(lower_mass_cutoff, upper_mass_cutoff)
        logM = np.log10(mass[:, np.newaxis])  # log10(M)
        # --- make sure ∫wlog dlogM = 1 -----------------------------------
        area = np.trapezoid(wlog, logM, axis=0)  # ∫wlog dlogM
        wlog = wlog / area
        # --- moments in log‑space ---------------------------------------
        M = 10**logM
        m_neg1 = np.trapezoid(wlog * M**-1, logM, axis=0)  # ∫w / M  dM  (via logM)
        m0 = 1.0  # by construction
        m1 = np.trapezoid(wlog * M, logM, axis=0)  # ∫w·M      dM
        m2 = np.trapezoid(wlog * M**2, logM, axis=0)  # ∫w·M²     dM

        # --- averages ----------------------------------------------------
        Mn = m0 / m_neg1  # Mn = m0 / m(‑1)
        Mw = m1 / m0  # Mw = m1 / m0
        Mz = m2 / m1  # Mz = m2 / m1
        Mp = M[np.argmax(wlog)]  # peak molar mass (mode)

        return {
            "Mp": Mp,
            "Mn": Mn,
            "Mw": Mw,
            "Mz": Mz,
        }

    def set_calibration_function(self, fit, params=None):
        """
        Set the calibration function.

        :param fit: The calibration function
        """
        if self._calibration_function is not None:
            raise ValueError("Calibration function already set")
        self._calibration_function = fit
        self._calibration_function_params = params

    @property
    def mass_range(self) -> np.ndarray:
        """
        :return: The mass range
        """
        if self._mass_range is None and self._calibration_function:
            self._mass_range = 10 ** self._calibration_function(self.raw_volumes)

        return self._mass_range

    @property
    def Wm(
        self,
    ) -> np.ndarray:
        rw = self.raw_volumes  # ml
        masses = self.mass_range  # [::-1]
        rev_masses = False
        if np.all(np.diff(masses) < 0):
            masses = masses[::-1]
            rev_masses = True
        elif np.all(np.diff(masses) > 0):
            pass
        else:
            raise ValueError("Mass range is not monotonic. Check the data.")
        r = self._calibration_function(rw)  # ml^x
        if self._calibration_function(0) > self._calibration_function(1):
            sigma = -np.gradient(r, rw)
        else:
            sigma = np.gradient(r, rw)

        if rev_masses:
            sigma = sigma[::-1]
        sig_region = self.basline_corrected_signal
        if rev_masses:
            sig_region = sig_region[::-1]

        wm = (sig_region.T / (masses * sigma)).T
        return wm

    def calc_logMW(
        self,
        lower_mass_cutoff: Optional[float] = None,
        upper_mass_cutoff: Optional[float] = None,
    ) -> np.ndarray:
        wm = self.Wm
        mass = self.mass_range[::-1]

        lower_cutoff = 0
        upper_cutoff = len(wm)

        if upper_mass_cutoff is not None:
            upper_cutoff = np.argmin(np.abs(mass - upper_mass_cutoff))
        if lower_mass_cutoff is not None:
            lower_cutoff = np.argmin(np.abs(mass - lower_mass_cutoff))

        wm_log = (
            wm[lower_cutoff:upper_cutoff]
            * mass[lower_cutoff:upper_cutoff, np.newaxis]
            * np.log(10)
        )  # convert to dw/dlog10 M
        #  (multiply *before* any normalisation!)
        area = np.trapezoid(
            wm_log,
            np.log10(mass[lower_cutoff:upper_cutoff, np.newaxis]),
            axis=0,
        )  # should =1
        wm_log /= area

        return wm_log, mass[lower_cutoff:upper_cutoff]

    def add_signal_boarder(
        self,
        vol: Optional[Tuple[int, int, Optional[int]]] = None,
        mass: Optional[Tuple[int, int, Optional[int]]] = None,
        index: Optional[Tuple[int, int, Optional[int]]] = None,
    ):
        """
        Add a signal boarder to the plot.

        """
        # allow only one of the three parameters
        if vol is None and mass is None and index is None:
            raise ValueError("At least one of ml, mass or index must be provided.")

        not_nones = sum(x is not None for x in [vol, mass, index])

        if not_nones > 1:
            raise ValueError("Only one of ml, mass or index can be provided at a time.")
        if vol is not None:
            if len(vol) != 2:
                raise ValueError("ml must be a tuple of two values.")
            index = (
                np.argmin(np.abs(self.raw_volumes - vol[0])),
                np.argmin(np.abs(self.raw_volumes - vol[1])),
                np.argmin(np.abs(self.raw_volumes - vol[2])) if len(vol) > 2 else None,
            )
        elif mass is not None:
            if len(mass) != 2:
                raise ValueError("mass must be a tuple of two values.")
            index = (
                np.argmin(np.abs(self.mass_range - mass[1])),
                np.argmin(np.abs(self.mass_range - mass[0])),
                np.argmin(np.abs(self.mass_range - mass[2])) if len(mass) > 2 else None,
            )

        self.signal_boarders.append(index)

    @property
    def basline_corrected_signal(self):
        if self._basline_corrected is None:
            _min = self.raw_signals.min(0)
            _fac = self.raw_signals.max(0) - _min
            normalized_signals = (self.raw_signals - _min) / _fac

            # add signals together
            summed_signals = np.sum(normalized_signals, axis=1)

            vol = self.raw_volumes
            y_bl, smoothed_is_baseline_region = estimate_baseline_regions(
                vol,
                summed_signals,
                pre_flatted_y=None,
                window=None,
            )

            _data = summed_signals[smoothed_is_baseline_region]
            x_data = vol[smoothed_is_baseline_region]

            baseline, _ = pybaselines.smooth.snip(
                _data,
                x_data=x_data,
                filter_order=2,
            )

            baseline = np.interp(vol, x_data, baseline)

            baseline_corrected = (
                self.raw_signals - (baseline[:, np.newaxis] * _fac) + _min
            )
            self._basline_corrected = baseline_corrected
        return self._basline_corrected

    def autodetect_signal_boarders(self):
        """
        Automatically detect the signal boarders.
        """

        baseline_corrected = self.basline_corrected_signal.sum(1)
        baseline_corrected = baseline_corrected - baseline_corrected.min()
        baseline_corrected = baseline_corrected / baseline_corrected.max()
        vol = self.raw_volumes
        # smooth the baseline corrected signal
        med_xdiff = np.nanmedian(np.diff(vol))
        window = max(1, int(0.5 / med_xdiff))

        smoothed = savgol_filter(baseline_corrected, window, 4)
        smoothed = smoothed - np.nanmedian(smoothed)
        smoothed = smoothed / smoothed.max()

        # find the peaks

        # smoothed_std = np.nanstd(smoothed[smoothed_is_baseline_region])

        peaks, peak_data = find_peaks(
            smoothed,
            height=0.01,
            rel_height=0.5,
            width=4,
        )

        peak_edges = list(zip(peak_data["left_bases"], peak_data["right_bases"], peaks))
        # peak edges overlap. if overlapping set the min between the two
        merged_edges = []
        for left, right, peak in sorted(peak_edges, key=lambda x: x[0]):
            if not merged_edges:
                merged_edges.append([left, right, peak])
                continue

            prev_left, prev_right, prev_peak = merged_edges[-1]

            # If this peak starts before the previous one ends, extend the region
            if left <= prev_right and right <= prev_right:
                # peak is inside the previous peak
                continue

            if left <= prev_right:
                merged_edges[-1][1] = (
                    np.argmin(smoothed[prev_peak:prev_right]) + prev_peak
                )
                left = merged_edges[-1][1] + 1

            # if right <= prev_left:
            #     merged_edges[-1][0] = np.argmin(vol[prev_left:peak]) + prev_left
            #     right = merged_edges[-1][0] - 1

            merged_edges.append([left, right, peak])

        merged_edges = [
            [
                np.argmin(smoothed[left:peak]) + left,
                np.argmin(smoothed[peak:right]) + peak,
                peak,
            ]
            for left, right, peak in merged_edges
        ]

        # raise ValueError(merged_edges)
        for left, right, peak in merged_edges:
            self.add_signal_boarder(
                index=(left, right, peak),
            )
