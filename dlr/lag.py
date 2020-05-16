import os

import numpy as np
import pandas as pd

import plot


class LagSearch:
    def __init__(self, wind_rot_df, segment_name, ref_sig, lagged_sig, dir_out, data_timewin_lag,
                 file_idx, segment_start, segment_end, filename):
        self.wind_rot_df = wind_rot_df
        self.segment_name = segment_name
        self.segment_start = segment_start
        self.segment_end = segment_end
        self.dir_out = dir_out
        self.w_rot_turb_col = ref_sig
        self.scalar_turb_col = lagged_sig
        self.data_timewin_lag = data_timewin_lag
        self.file_idx = file_idx
        self.filename = filename

        self.run()

    def run(self):
        self.cov_max_shift, self.cov_max, self.cov_max_timestamp, self.lag_search_df = \
            self.find_max_cov(df=self.wind_rot_df)

        self.save_cov_plot(x=self.lag_search_df['shift'],
                           y=self.lag_search_df['cov'],
                           z_color=self.lag_search_df['cov_abs'])

    def get(self):
        return self.cov_max_shift, self.cov_max, self.cov_max_timestamp

    def save_cov_plot(self, x, y, z_color):
        """Plot and save covariance plot for segment."""

        txt_info = \
            f"Segment name: {self.segment_name}\n" \
            f"Segment start: {self.segment_start}\n" \
            f"Segment end: {self.segment_end}\n" \
            f"File: {self.filename} - File date: {self.file_idx}\n" \
            f"Time lag search window: from {self.data_timewin_lag[0]} to {self.data_timewin_lag[1]} records\n" \
            f"Max absolute covariance {self.cov_max:.3f} found @ record {self.cov_max_shift}"

        fig = plot.make_scatter_cov(x=x, y=y, z_color=z_color,
                                    cov_max_shift=self.cov_max_shift,
                                    cov_max=self.cov_max,
                                    txt_info=txt_info)

        dir_out = self.dir_out / '2_plots_segment_covariances'
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out)
        outpath = dir_out / self.segment_name

        # Save
        print(f"Saving plot in PNG file: {outpath}.png ...")
        fig.savefig(f"{outpath}.png", format='png', bbox_inches='tight', facecolor='w',
                    transparent=True, dpi=150)

    def find_max_cov(self, df):
        """Find maximum absolute covariance between turbulent wind data
        and turbulent scalar data.
        """

        print("Searching maximum covariance ...")

        _df = df.copy()
        _df['index'] = _df.index

        lag_search_df = pd.DataFrame()
        lagwin_start = self.data_timewin_lag[0]
        lagwin_end = self.data_timewin_lag[1]
        lag_search_df['shift'] = range(lagwin_start, lagwin_end)  # Negative moves lagged values "upwards" in column
        lag_search_df['cov'] = np.nan
        lag_search_df['index'] = np.nan

        for ix, row in lag_search_df.iterrows():
            shift = int(row['shift'])
            # print(shift)

            try:
                if shift < 0:
                    index_shifted = str(_df['index'][-shift])  # Note the negative sign
                else:
                    index_shifted = pd.NaT
            except:
                print("S")

            scalar_data_shifted = _df[self.scalar_turb_col].shift(shift)
            cov = _df[self.w_rot_turb_col].cov(scalar_data_shifted)
            lag_search_df.loc[lag_search_df['shift'] == row['shift'], 'cov'] = cov
            lag_search_df.loc[lag_search_df['shift'] == row['shift'], 'index'] = index_shifted

        lag_search_df['cov_abs'] = lag_search_df['cov'].abs()
        cov_max_ix = lag_search_df['cov_abs'].idxmax()
        cov_max_shift = lag_search_df.iloc[cov_max_ix]['shift']
        cov_max = lag_search_df.iloc[cov_max_ix]['cov']
        cov_max_timestamp = lag_search_df.iloc[cov_max_ix]['index']

        return cov_max_shift, cov_max, cov_max_timestamp, lag_search_df
