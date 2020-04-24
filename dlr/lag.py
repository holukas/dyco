import altair as alt
import numpy as np
import pandas as pd


class LagSearch:
    def __init__(self, wind_rot_df, filename, ref_sig, lagged_sig, dir_out):
        self.wind_rot_df = wind_rot_df
        self.filename = filename
        self.dir_out = dir_out
        self.w_rot_turb_col = ref_sig
        self.scalar_turb_col = lagged_sig

        self.run()

    def run(self):
        self.cov_max_shift, self.cov_max, self.cov_max_timestamp, self.lag_search_df =\
            self.find_max_cov(df=self.wind_rot_df)

        self.save_html(df=self.lag_search_df, outfile=self.filename,
                       x='shift', y='cov', z_color='cov_abs', timestamp='index')

    def get(self):
        return self.cov_max_shift, self.cov_max, self.cov_max_timestamp

    def save_html(self, df, x, y, z_color, outfile, timestamp):
        print(f"Saving plot in HTML file: {outfile}.html ...")

        selection = alt.selection_interval(bind='scales')
        chart = alt.Chart(df).mark_circle(size=20).encode(
            x=x,
            y=y,
            color=z_color,
            tooltip=[x, y, z_color, timestamp]
        ).properties(
            width=800,
            height=500,
            title=outfile
        ).add_selection(
            selection
        )

        chart.configure_title(
            fontSize=20,
            font='Courier',
            anchor='start',
            color='gray'
        )

        outfile = self.dir_out / outfile
        chart.save(f"{outfile}_cov.html")

    def find_max_cov(self, df):
        """Find maximum absolute covariance between turbulent wind data
        and turbulent scalar data.
        """

        print("Searching maximum covariance ...")

        _df = df.copy()
        _df['index'] = _df.index

        lag_search_df = pd.DataFrame()
        lag_search_df['shift'] = range(-300, 300)  # Negative moves lagged values "upwards" in column
        lag_search_df['cov'] = np.nan
        lag_search_df['index'] = np.nan

        for ix, row in lag_search_df.iterrows():
            # print(int(row['shift']))
            shift = int(row['shift'])

            if shift < 0:
                index_shifted = str(_df['index'][-shift])  # Note the negative sign
            else:
                index_shifted = pd.NaT

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
