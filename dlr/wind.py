import math

import numpy as np


class WindRotation:

    def __init__(self, wind_df, u_col, v_col, w_col, scalar_col):
        self.wind_rot_df = wind_df.copy()

        # Available data columns
        self.u_col = u_col
        self.v_col = v_col
        self.w_col = w_col
        self.scalar_col = scalar_col

        # New data columns
        self.u_temp_col = self.u_col + '_temp'
        self.v_temp_col = self.v_col + '_temp'
        self.w_temp_col = self.w_col + '_temp'
        self.u_rot_col = self.u_col + '_rot'
        self.v_rot_col = self.v_col + '_rot'
        self.w_rot_col = self.w_col + '_rot'
        self.w_rot_mean_col = self.w_col + '_rot_mean'
        self.w_rot_turb_col = self.w_col + '_rot_turb'
        self.scalar_mean_col = self.scalar_col + '_mean'
        self.scalar_turb_col = self.scalar_col + '_turb'

        self.run()

    def run(self):
        self.angle_r1, self.angle_r2 = self.rot_angles_from_mean_wind(u_mean=float(self.wind_rot_df[self.u_col].mean()),
                                                                      v_mean=float(self.wind_rot_df[self.v_col].mean()),
                                                                      w_mean=float(self.wind_rot_df[self.w_col].mean()))
        self.wind_rot_df = self.rotate_wind(wind_rot_df=self.wind_rot_df)
        self.wind_rot_df = self.calculate_turbulent_fluctuations(df=self.wind_rot_df)

        # lagsearch_df.plot.scatter(x='shift', y='cov')
        # plt.show()

        # self.wind_rot_df['record'] = self.wind_rot_df.index
        # source = self.wind_rot_df
        # chart=alt.Chart(source).mark_circle(size=20).encode(
        #     x='record',
        #     y=self.w_rot_col,
        #     color=self.u_rot_col,
        #     tooltip=[self.u_rot_col, self.v_rot_col, self.w_rot_col]
        # )

        # save(chart, 'test.pdf')

    def get(self):
        return self.wind_rot_df, self.w_rot_turb_col, self.scalar_turb_col

    def calculate_turbulent_fluctuations(self, df):
        print("Calculating turbulent fluctuations ...")
        df[self.w_rot_mean_col] = df[self.w_rot_col].mean()
        df[self.scalar_mean_col] = df[self.scalar_col].mean()
        df[self.w_rot_turb_col] = df[self.w_rot_col] - df[self.w_rot_mean_col]
        df[self.scalar_turb_col] = df[self.scalar_col] - df[self.scalar_mean_col]
        return df

    def rotate_wind(self, wind_rot_df):
        """
        Use rotation angles from mean wind to perform double rotation
        on high-resolution wind data
        """

        print("Rotating wind ...")

        df = wind_rot_df.copy()

        # Init new cols for new data
        df[self.u_rot_col] = np.nan
        df[self.v_rot_col] = np.nan
        df[self.w_rot_col] = np.nan
        df[self.u_temp_col] = np.nan
        df[self.v_temp_col] = np.nan
        df[self.w_temp_col] = np.nan

        # Measured wind components
        u = df[self.u_col]
        v = df[self.v_col]
        w = df[self.w_col]

        # Perform first rotation of coordinate system
        # Make v component zero --> mean of high-res v_temp_col becomes zero (or very close to)
        df[self.u_temp_col] = u * math.cos(self.angle_r1) + v * math.sin(self.angle_r1)
        df[self.v_temp_col] = -u * math.sin(self.angle_r1) + v * math.cos(self.angle_r1)
        df[self.w_temp_col] = w

        # Perform second rotation of coordinate system
        # Make w component zero --> mean of high-res w_rot_col becomes zero (or very close to)
        df[self.u_rot_col] = \
            df[self.u_temp_col] * math.cos(self.angle_r2) + df[self.w_temp_col] * math.sin(self.angle_r2)
        df[self.v_rot_col] = df[self.v_temp_col]
        df[self.w_rot_col] = \
            -df[self.u_temp_col] * math.sin(self.angle_r2) + df[self.w_temp_col] * math.cos(self.angle_r2)

        return df

    def rot_angles_from_mean_wind(self, u_mean, v_mean, w_mean):
        """
        Calculate rotation angles for double rotation from mean wind

        The rotation angles are calculated from mean wind, but are later
        applied sample-wise to the full high-resolution data (typically 20Hz
        for wind data).

        Note that rotation angles are given in radians.

        First rotation angle:
            thita = tan-1 (v_mean / u_mean)

        Second rotation angle:
            phi = tan-1 (w_temp / u_temp)

        """

        print("Calculating rotation angles ...")

        # First rotation angle, in radians
        angle_r1 = math.atan(v_mean / u_mean)

        # Perform first rotation of coordinate system for mean wind
        # Make v component of mean wind zero --> v_temp becomes zero
        u_temp = u_mean * math.cos(angle_r1) + v_mean * math.sin(angle_r1)
        v_temp = -u_mean * math.sin(angle_r1) + v_mean * math.cos(angle_r1)
        w_temp = w_mean

        # Second rotation angle, in radians
        angle_r2 = math.atan(w_temp / u_temp)

        # For calculating the rotation angles, it is not necessary to perform the second
        # rotation of the coordinate system for mean wind
        # Make v component zero, vm = 0
        # u_rot = u_temp * math.degrees(math.cos(angle_r2)) + w_temp * math.degrees(math.sin(angle_r2))
        # v_rot = v_temp
        # w_rot = -u_temp * math.degrees(math.sin(angle_r2)) + w_temp * math.degrees(math.cos(angle_r2))

        return angle_r1, angle_r2
