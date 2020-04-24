import datetime as dt
import os
from pathlib import Path

import pandas as pd

import file
import plot
from lag import LagSearch
from wind import WindRotation


class DynamicLagRemover:
    u_col = 'u_ms-1'
    v_col = 'v_ms-1'
    w_col = 'w_ms-1'
    scalar_col = 'co2_ppb_qcl'
    dir_input = 'input'
    dir_output = 'output'
    filename_format_dt = '%Y%m%d%H%M'
    time_res = 0.05  # Measurement every 0.05s
    time_res_hz = 1 / time_res  # 1 measurement every 0.05s (20Hz)
    file_duration = 21600  # in seconds
    segment_duration = '30T'  # 30min segments
    segment_overhang = '1T'  # 1min
    testing = True

    def __init__(self):
        if self.testing:
            self.nrows = 10000
        else:
            self.nrows = None

        dir_root = Path(os.path.dirname(os.path.abspath(__file__)))
        self.dir_input = dir_root / self.dir_input
        self.dir_output = dir_root / self.dir_output
        self.lagtimes_df = pd.DataFrame(columns=['cov_max_shift', 'cov_max', 'cov_max_timestamp'])

        self.run()

    def run(self):

        filelist = file.search(dir=self.dir_input, pattern='*.csv')
        data_collection_df = pd.DataFrame()

        # FILE LOOP
        # =========
        for idx, filepath in enumerate(filelist):
            if idx > 0:  # for testing
                break

            print(f"\nFile #{idx}: {filepath}")

            filename = filepath.stem
            file_start_dt = dt.datetime.strptime(filename, self.filename_format_dt)

            # Read data file
            data_df = \
                file.read_data(filepath=filepath, nrows=self.nrows, time_res_hz=self.time_res_hz,
                               df_start_dt=file_start_dt,
                               file_duration=self.file_duration)  # nrows for testing

            if idx == 0:
                data_collection_df = data_df.copy()
            else:
                data_collection_df = data_collection_df.append(data_df)

            # SEGMENT LOOP
            # ------------
            # https://stackoverflow.com/questions/52426972/pandas-bin-dates-into-30-minute-intervals-and-calculate-averages
            counter_segment = -1
            data_df['index'] = pd.to_datetime(data_df.index)
            segment_grouped = data_df.groupby(pd.Grouper(key='index', freq=self.segment_duration))
            for segment_key, segment_df in segment_grouped:
                counter_segment += 1
                if counter_segment > 0:  # for testing
                    break
                segment_name = segment_df.index[0].strftime('%Y%m%d%H%M%S')
                print(f"Working on segment {segment_name} ...")

                wind_rot_df, w_rot_turb_col, scalar_turb_col = \
                    WindRotation(wind_df=segment_df[[self.u_col, self.v_col, self.w_col, self.scalar_col]],
                                 u_col=self.u_col, v_col=self.v_col,
                                 w_col=self.w_col, scalar_col=self.scalar_col).get()

                cov_max_shift, cov_max, cov_max_timestamp = \
                    LagSearch(wind_rot_df=wind_rot_df,
                              filename=segment_name,
                              ref_sig=w_rot_turb_col,
                              lagged_sig=scalar_turb_col,
                              dir_out=self.dir_output).get()

                # Collect results
                self.lagtimes_df.loc[segment_name, 'cov_max_shift'] = int(cov_max_shift)
                self.lagtimes_df.loc[segment_name, 'cov_max'] = float(cov_max)
                self.lagtimes_df.loc[segment_name, 'cov_max_timestamp'] = str(cov_max_timestamp)

        self.lagtimes_df['cov_max_shift'] = self.lagtimes_df['cov_max_shift'].astype(int)
        self.lagtimes_df['cov_max'] = self.lagtimes_df['cov_max'].astype(float)
        self.lagtimes_df['cov_max_timestamp'] = self.lagtimes_df['cov_max_timestamp'].astype(str)

        self.lagtimes_df.to_csv(self.dir_output / '_found_lag_times.csv')
        plot.found_lag_times(df=self.lagtimes_df, dir_output=self.dir_output)

if __name__ == "__main__":
    DynamicLagRemover()
