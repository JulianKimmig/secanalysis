from io import StringIO
from .base import SECDataBase
from typing import List, Tuple
import pandas as pd
import numpy as np


class WinGPCData(SECDataBase):
    POSSIBLE_VOLUME_COLUMNS = [SECDataBase.DEFAULT_VOLUME_COLUMN] + [
        "Volume",
        "volume",
        "VOLUME",
        "Volume (mL)",
    ]
    POSSIBLE_TIME_COLUMNS = [
        "Time",
        "TIME",
        "Time (s)",
    ]

    RAW_BLOCK = "RAW"
    ELU_BLOCK = "ELU"
    MWD_BLOCK = "MWD"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._elu_data = None
        self._mwd_data = None
        self._mass_params = {}

    @classmethod
    def from_string(cls, string: str):
        lines = [line.lstrip(" ") + "\n" for line in string.splitlines(keepends=False)]
        data_dict, clmLines = cls.process_sec_file(lines)
        metadata_df = cls.extract_metadata(lines, clmLines[0])

        raw_data = data_dict[cls.RAW_BLOCK]

        volume_cols = [
            col for col in raw_data.columns if col in cls.POSSIBLE_VOLUME_COLUMNS
        ]
        if len(volume_cols) > 1:
            raise ValueError("Multiple volume columns found")

        time_cols = [
            col for col in raw_data.columns if col in cls.POSSIBLE_TIME_COLUMNS
        ]
        if len(time_cols) > 1:
            raise ValueError("Multiple time columns found")

        volume_col = volume_cols[0] if len(volume_cols) == 1 else None
        time_col = time_cols[0] if len(time_cols) == 1 else None
        if len(volume_cols) == 0:
            if len(time_cols) == 0:
                raise ValueError("No time or volume column found")
            time_cols = time_cols[0]
            time = raw_data[time_cols]
            # make sure time is either a wall clock time or a time in seconds
            # the format is either HH:MM:SS or HH:MM:SS:MS
            if time.dtypes == "object":
                if ":" in time.iloc[0]:
                    time = pd.to_datetime(time, format="%H:%M:%S:%f", errors="coerce")

            # if time is absolute time, convert to seconds
            if time.dtypes.name.startswith("datetime"):
                time = (time - time.min()).dt.total_seconds()

            flow = float(metadata_df["Flow"].split("ml")[0].strip())  # ml/min
            volume = time * flow / 60.0  # convert to ml
            volume_col = cls.DEFAULT_VOLUME_COLUMN
            raw_data[volume_col] = volume

        else:
            volume = np.array(raw_data[volume_col])

        if time_col:
            # drop it
            raw_data.drop(columns=[time_col], inplace=True)

        raw_data[volume_col] = volume
        # rename the volume column to "Volume"
        raw_data.rename(columns={volume_col: cls.DEFAULT_VOLUME_COLUMN}, inplace=True)

        gpc = cls(raw_data)
        try:
            fit, params = cls.make_fit(metadata_df)
            gpc.set_calibration_function(fit, params)
        except KeyError:
            pass

        gpc.elu_data = data_dict.get(cls.ELU_BLOCK)
        gpc.mwd_data = data_dict.get(cls.MWD_BLOCK)

        gpc.mass_params = {
            "Mn": metadata_df.get("Mn", None),
            "Mw": metadata_df.get("Mw", None),
            "Mz": metadata_df.get("Mz", None),
            "Mp": metadata_df.get("Mp", None),
        }

        return gpc

    @staticmethod
    def extract_metadata(lines: list, clmLine: int) -> pd.DataFrame:
        meta_lines = "".join(lines[:clmLine]).replace("\t", " ")
        metadata_dict = {}
        for line in meta_lines.split("\n"):
            key_val = line.split(":", 1)
            if len(key_val) == 2:
                key = key_val[0].strip()
                val = key_val[1].strip()
                if key != "" and val != "":
                    metadata_dict[key] = val
        return metadata_dict

    @staticmethod
    def process_sec_file(lines: List[str]) -> Tuple[dict, List[int]]:
        skipRows = []
        clmLines = []
        skipFooters = []
        blockKeys = []
        corrected_lines = []
        dfTotal = {}

        for lineNr, lineContext in enumerate(lines):
            if "\t\n" not in lineContext:
                lineContext = lineContext.replace("\n", "\t\n")
            if "start" in lineContext and len(lineContext.split(" ")) == 2:
                skipRows.append(lineNr)
                clmLines.append(lineNr + 1)
                blockKeys.append(lineContext.split("start")[0])
            elif "stop" in lineContext and len(lineContext.split(" ")) == 2:
                skipFooters.append(lineNr)
            corrected_lines.append(lineContext)
        for index, key in enumerate(blockKeys):
            lines = "".join(corrected_lines[clmLines[index] : skipFooters[index]])
            fileString = StringIO(lines)
            df = (
                pd.read_csv(
                    fileString,
                    sep="\t",
                    index_col=False,
                    thousands=",",
                    decimal=".",
                )
                .dropna(axis=1, how="all")
                .dropna(axis=0, how="all")
            )
            dfTotal[key] = df

            # for all string columns, remove leading and trailing spaces and try to interfer the type again
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].str.strip()
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except ValueError:
                        pass

        return dfTotal, clmLines

    @staticmethod
    def make_fit(metadata: pd.DataFrame):
        params = [
            float(metadata["Const."].strip()),
            float(metadata["Coef.2"].strip()),
        ]
        for i in range(3, 12):
            if "Coef." + str(i) in metadata:
                params.append(float(metadata["Coef." + str(i)].strip()))
            else:
                params.append(0.0)

        while params[-1] == 0.0:
            params.pop()
        params.reverse()

        fit = np.poly1d(
            params,
        )
        fit_grad = np.polyder(fit)
        min_vol = float(metadata["Vol. min"])
        max_vol = float(metadata["Vol. max"])

        grad_min = fit_grad(min_vol)
        grad_max = fit_grad(max_vol)

        lin_min = np.poly1d([grad_min, fit(min_vol) - grad_min * min_vol])
        lin_max = np.poly1d([grad_max, fit(max_vol) - grad_max * max_vol])

        def _calc_mass(vol):
            vol = np.atleast_1d(vol)
            res = fit(vol)
            res[vol < min_vol] = lin_min(vol[vol < min_vol])
            res[vol > max_vol] = lin_max(vol[vol > max_vol])
            return res

        return _calc_mass, params

    @property
    def elu_data(self):
        return self._elu_data

    @elu_data.setter
    def elu_data(self, value):
        self._elu_data = value
        if self._elu_data is None:
            return
        self._elu_data.rename(columns={"Volume": "Volume (mL)"}, inplace=True)
        self._elu_data["Volume (mL)"] = self._elu_data["Volume (mL)"].astype(float)

    @property
    def mwd_data(self):
        return self._mwd_data

    @mwd_data.setter
    def mwd_data(self, value):
        self._mwd_data = value

    @property
    def mass_params(self):
        return self._mass_params

    @mass_params.setter
    def mass_params(self, value):
        if "Mn" in value and value["Mn"] is not None:
            self._mass_params["Mn"] = float(value["Mn"].replace("g/mol", ""))
        if "Mw" in value and value["Mw"] is not None:
            self._mass_params["Mw"] = float(value["Mw"].replace("g/mol", ""))
        if "Mz" in value and value["Mz"] is not None:
            self._mass_params["Mz"] = float(value["Mz"].replace("g/mol", ""))
        if "Mp" in value and value["Mp"] is not None:
            self._mass_params["Mp"] = float(value["Mp"].replace("g/mol", ""))

    def _plot_elu(self, *args, **kwargs):
        raw_ax, mass_ax = super()._plot_elu(*args, **kwargs)
        if self.elu_data is not None:
            elu_volume = self.elu_data["Volume (mL)"]
            elu_signal = self.elu_data["RID"]
            elu_molar_mass = self.elu_data["Molar mass"]

            l1 = raw_ax.plot(
                elu_volume,
                elu_signal,
                label="WinGPC Elu",
            )

            l2 = mass_ax.plot(
                elu_volume,
                elu_molar_mass,
                label="WinGPC Mass",
            )

            # add to the legend of mass_ax

            handles = mass_ax.legend_.legend_handles
            handles.append(l1[0])
            handles.append(l2[0])
            labels = [hand.get_label() for hand in handles]
            mass_ax.legend(handles, labels)

        return raw_ax, mass_ax

    def _plot_mwd(self, *args, **kwargs):
        raw_ax = super()._plot_mwd(*args, **kwargs)
        if self.mwd_data is not None:
            molar_mass = self.mwd_data["Molar mass"]
            signal = self.mwd_data["RID"]
            # signal_area = np.trapezoid(signal, molar_mass)
            # signal = signal / signal_area

            l1 = raw_ax.plot(
                molar_mass,
                signal,
                label="WinGPC MWD",
            )

            raw_ax.vlines(
                [v for v in self.mass_params.values() if v is not None],
                0,
                signal.max(),
                color=l1[0].get_color(),
                linestyle="--",
            )

            # add to the legend of mass_ax
            handles = raw_ax.legend_.legend_handles
            handles.append(l1[0])
            labels = [hand.get_label() for hand in handles]
            raw_ax.legend(handles, labels)

        return raw_ax

    # def plot(
    #     self,
    #     lower_mass_cutoff=None,
    #     upper_mass_cutoff=None,
    #     lower_volume_cutoff=None,
    #     upper_volume_cutoff=None,
    # ):
    #     fig, axes = super().plot(
    #         lower_mass_cutoff,
    #         upper_mass_cutoff,
    #         lower_volume_cutoff,
    #         upper_volume_cutoff,
    #     )

    #     if self.elu_data is not None:
    #         elu_volume = self.elu_data["Volume (mL)"]
    #         elu_signal = self.elu_data["RID"]
    #         elu_molar_mass = self.elu_data["Molar mass"]
    #         min_volume = elu_volume.min()
    #         max_volume = elu_volume.max()

    #         rawindex_min = np.argmin(np.abs(self.raw_volumes - min_volume))
    #         rawindex_max = np.argmin(np.abs(self.raw_volumes - max_volume))

    #         # new subplot shift position of axes["mass"] from 212 ro 313
    #         # axes["mass"].set_position([0.1, 0.1, 0.8, 0.8])

    #         eluaxes = fig.add_subplot(312)
    #         eluaxes.plot(elu_volume, elu_signal, label="Elution")
    #         eluaxes.plot(
    #             self.raw_volumes[rawindex_min:rawindex_max],
    #             self.raw_signals[rawindex_min:rawindex_max]
    #             - np.median(self.raw_signals[rawindex_min:rawindex_max]),
    #             label="Raw signal",
    #         )
    #         eluaxes.set_xlabel("Volume (mL)")
    #         eluaxes.set_ylabel("Signal")
    #         eluaxes.set_title("Elution")
    #         eluaxes.legend()
    #         eluaxes.grid()
    #         # set the x axis limits to the same as the mass plot
    #         # add second y axis (twinx)
    #         elu2 = eluaxes.twinx()
    #         elu2.plot(
    #             elu_volume,
    #             elu_molar_mass,
    #             label="Molar mass",
    #         )

    #         massindex_min = np.argmax(~np.isnan(elu_molar_mass)) + rawindex_min
    #         massindex_max = rawindex_max - np.argmax(elu_molar_mass[::-1] == 0)

    #         elu2.plot(
    #             self.raw_volumes[massindex_min:massindex_max],
    #             self.mass_range[massindex_min:massindex_max],
    #             label="Mass range",
    #         )

    #         import matplotlib.pyplot as plt

    #         interpolated_mass = np.interp(
    #             elu_volume,
    #             self.raw_volumes,
    #             self.mass_range,
    #         )

    #         fig2 = plt.figure()
    #         _ax = fig2.add_subplot(111)
    #         _ax.plot(
    #             elu_volume,
    #             interpolated_mass,
    #             label="Interpolated mass range",
    #         )
    #         _ax.plot(
    #             elu_volume,
    #             elu_molar_mass,
    #             label="Molar mass",
    #         )

    #         _ax.set_xlabel("Volume (mL)")
    #         _ax.set_yscale("log")

    #         # performe curve fit that is the same as the one in the plot
    #         # o_params = self._calibration_function_params

    #         # def _calc_mass(x):
    #         #     return np.polyval(o_params, x)

    #         # from scipy.optimize import curve_fit

    #         # def _func(x, *params):
    #         #     return np.polyval(params, x)

    #         # popt, pcov = curve_fit(
    #         #     _func,
    #         #     elu_volume[~np.isnan(elu_molar_mass)],
    #         #     np.log(elu_molar_mass[~np.isnan(elu_molar_mass)]),
    #         #     p0=o_params,
    #         #     bounds=(
    #         #         np.array(o_params) - np.abs(np.array(o_params)) * 0.2,
    #         #         np.array(o_params) + np.abs(np.array(o_params)) * 0.2,
    #         #     ),
    #         # )

    #         # _ax.text(
    #         #     0.5,
    #         #     0.75,
    #         #     f"original params: {o_params}",
    #         # )

    #         # _ax.text(
    #         #     0.5,
    #         #     np.nanmax(elu_molar_mass) * 0.75,
    #         #     f"fitted params: {popt}",
    #         # )
    #         # # plot the fit
    #         # _ax.plot(
    #         #     elu_volume,
    #         #     np.exp(_func(elu_volume, *popt)),
    #         #     label="Fit",
    #         # )

    #         # _ax.plot(
    #         #     elu_volume,
    #         #     np.exp(_func(elu_volume, *o_params)),
    #         #     label="OFit",
    #         # )

    #         # _ax.plot(
    #         #     elu_volume[~np.isnan(elu_molar_mass)],
    #         #     elu_molar_mass[~np.isnan(elu_molar_mass)],
    #         #     "o",
    #         # )

    #         _ax.legend()

    #         # show fig2
    #         fig2.show()
    #         plt.show()

    #         axes["elu"] = eluaxes

    #     return fig, axes
