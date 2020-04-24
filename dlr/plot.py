import altair as alt


def prepare_df(df):
    df['filename'] = df.index
    df.reset_index(drop=True, inplace=True)
    df['file_idx'] = df.index
    df['cov_max_shift']=df['cov_max_shift'].astype(int)
    return df


def found_lag_times(df, dir_output):
    print(f"Saving plot in HTML file: _found_lag_times.html ...")

    _df = df.copy()
    _df = prepare_df(df=_df)

    selection = alt.selection_interval(bind='scales')
    chart = alt.Chart(_df).mark_circle(size=30).encode(
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

    outfile = dir_output / '_found_lag_times'
    chart.save(f"{outfile}.html")
