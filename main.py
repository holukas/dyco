# import altair as alt
import os
from pathlib import Path

import pandas as pd

import file
from lag import LagSearch
from wind import WindRotation


class DynamicLagRemover:
    u_col = 'u_ms-1'
    v_col = 'v_ms-1'
    w_col = 'w_ms-1'
    scalar_col = 'co2_ppb_qcl'
    dir_input = 'input'
    dir_output = 'output'
    time_res = 1 / 0.05  # 1 measurement every 0.05s (20Hz)
    file_duration = 21600  # in seconds
    segment_records = time_res * 60 * 30  # 30min segments
    segment_overhang = (segment_records * 1.1) - segment_records

    def __init__(self):
        dir_root = Path(os.path.dirname(os.path.abspath(__file__)))
        self.dir_input = dir_root / self.dir_input
        self.dir_output = dir_root / self.dir_output
        self.lagtimes_df = pd.DataFrame(columns=['cov_max_shift', 'cov_max'])

        self.run()

    def run(self):

        filelist = file.search(dir=self.dir_input, pattern='*.csv')

        # FILE LOOP
        # =========
        for idx, filepath in enumerate(filelist):
            if idx > 0:  # for testing
                break

            print(f"\nFile #{idx}: {filepath}")
            filename = filepath.stem

            # Read data file
            data_df = \
                file.read_data(filepath=filepath, time_res=self.time_res, nrows=150000)  # nrows for testing

            # SEGMENT LOOP
            # ------------
            num_segments = int(len(data_df) / self.segment_records)
            for segment_idx in range(0, num_segments):
                if segment_idx > 2:  # for testing
                    break

                segment_df, segment_name = self.make_segment(df=data_df, filename=filename, segment_idx=segment_idx)

                wind_rot_df, w_rot_turb_col, scalar_turb_col = \
                    WindRotation(wind_df=segment_df[[self.u_col, self.v_col, self.w_col, self.scalar_col]],
                                 u_col=self.u_col, v_col=self.v_col,
                                 w_col=self.w_col, scalar_col=self.scalar_col).get()

                cov_max_shift, cov_max = \
                    LagSearch(wind_rot_df=wind_rot_df,
                              filename=segment_name,
                              ref_sig=w_rot_turb_col,
                              lagged_sig=scalar_turb_col,
                              dir_out=self.dir_output).get()

                # Collect results
                self.lagtimes_df.loc[segment_name, 'cov_max_shift'] = cov_max_shift
                self.lagtimes_df.loc[segment_name, 'cov_max'] = cov_max

        self.lagtimes_df['filename'] = self.lagtimes_df.index
        self.lagtimes_df.reset_index(drop=True, inplace=True)
        self.lagtimes_df['file_idx'] = self.lagtimes_df.index

        print(f"Saving plot in HTML file: _summary.html ...")

        import altair as alt
        selection = alt.selection_interval(bind='scales')
        chart = alt.Chart(self.lagtimes_df).mark_circle(size=30).encode(
            x='file_idx',
            y='cov_max_shift',
            tooltip=['file_idx', 'cov_max_shift', 'filename']
        ).properties(
            width=1600,
            height=900,
            title='Found Lag Times'
        ).add_selection(
            selection
        )

        chart.configure_title(
            fontSize=20,
            font='Courier',
            anchor='start',
            color='gray'
        )

        outfile = self.dir_output / '_summary'
        chart.save(f"{outfile}.html")

    def make_segment(self, df, segment_idx, filename):
        segment_name = f"{filename}-{segment_idx}"
        start = int(segment_idx * self.segment_records)
        end = int(((segment_idx + 1) * self.segment_records) + self.segment_overhang)
        segment_df = df.iloc[start:end]

        print(f"{start}:{end}")
        print(segment_df)
        return segment_df, segment_name


if __name__ == "__main__":
    DynamicLagRemover()
