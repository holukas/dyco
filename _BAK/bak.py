# Was called with @dyco

def dyco(cls):
    """
    Wrapper function for DYCO processing chain

    Parameters
    ----------
    cls: class
        The class that is wrapped.

    Returns
    -------
    Wrapper

    """

    class ProcessingChain:
        """

        Wrapper class for executing the main script multiple times, each
        time with different settings

        The processing chain comprises 3 Phases and one finalization step:
            * Phase 1: First normalization to default lag
            * Phase 2: Second normalization to default lag
            * Phase 3: Correction for instantaneous lag
            * Finalize: Plots that summarize Phases 1-3

        """

        def __init__(self, **args):
            # PHASE 1 - First normalization to default lag
            # ============================================
            args['phase'] = 1
            self.run_phase_1_input_files = cls(**args)

            # PHASE 2 - Second normalization to default lag
            # =============================================
            args['phase'] = 2
            args = self._update_args(args=args,
                                     prev_phase=self.run_phase_1_input_files.phase,
                                     prev_phase_files=self.run_phase_1_input_files.phase_files,
                                     prev_outdir_files=self.run_phase_1_input_files.outdir,
                                     prev_last_iteration=self.run_phase_1_input_files.lgs_num_iter,
                                     prev_outdirs=self.run_phase_1_input_files.outdirs)
            self.run_phase_2_normalized_files = cls(**args)

            # PHASE 3 - Correction for instantaneous lag
            # ==========================================
            args['phase'] = 3
            args = self._update_args(args=args,
                                     prev_phase=self.run_phase_2_normalized_files.phase,
                                     prev_phase_files=self.run_phase_2_normalized_files.phase_files,
                                     prev_outdir_files=self.run_phase_2_normalized_files.outdir,
                                     prev_last_iteration=self.run_phase_2_normalized_files.lgs_num_iter,
                                     prev_outdirs=self.run_phase_2_normalized_files.outdirs)
            self.run_phase_3_finalize = cls(**args)

            # FINALIZE - Make some more plots summarizing Phases 1-3
            # ======================================================
            plot.SummaryPlots(instance_phase_1=self.run_phase_1_input_files,
                              instance_phase_2=self.run_phase_2_normalized_files,
                              instance_phase_3=self.run_phase_3_finalize)

        def _update_args(self, args, prev_phase, prev_phase_files, prev_outdir_files, prev_last_iteration,
                         prev_outdirs):
            """Update args for running Phases 2 and 3: use results from Phase 1 and 2, respectively"""
            if args['phase'] == 2:
                args['lgs_winsize'] = self._update_winsize(prev_phase=prev_phase,
                                                           prev_phase_files=prev_phase_files,
                                                           prev_last_iteration=prev_last_iteration,
                                                           prev_outdirs=prev_outdirs)
            else:
                args['lgs_winsize'] = 100  # Small window for instantaneous search in Phase 3
                args['lgs_num_iter'] = 1

            args['indir'] = Path(prev_outdir_files) / f"{prev_phase}-7_{prev_phase_files}_normalized"
            args['var_lagged'] = f"{args['var_lagged']}_DYCO"  # Use normalized signal
            filename, file_extension = os.path.splitext(args['fnm_pattern'])
            args['fnm_pattern'] = f"{filename}_DYCO{file_extension}"  # Search normalized files
            args[
                'fnm_date_format'] = f"{Path(args['fnm_date_format']).stem}_DYCO.csv"  # Parse file names of normalized files
            var_target = [var_target + '_DYCO' for var_target in args['var_target']]  # Use normalized target cols
            args['var_target'] = var_target
            return args

        def _update_winsize(self, prev_phase, prev_phase_files, prev_last_iteration, prev_outdirs):
            """
            Calculate the range of the lag search window from the last iteration and use
            it for lag search in normalized files

            During the last iteration, this window was detected as the *next* window for
            the *next* iteration, i.e. it was not yet used for lag detection.

            Returns
            -------
            Time window for lag search in Phase 2 iteration 1 and in Phase 3
            """
            filepath_last_iteration = \
                prev_outdirs[
                    f"{prev_phase}-3_{prev_phase_files}_time_lags_overview"] \
                / f'{prev_last_iteration}_segments_found_lag_times_after_iteration-{prev_last_iteration}.csv'

            segment_lagtimes_last_iteration_df = \
                files.read_segment_lagtimes_file(filepath=filepath_last_iteration)
            lgs_winsize = \
                [segment_lagtimes_last_iteration_df['lagsearch_next_start'].unique()[0],
                 segment_lagtimes_last_iteration_df['lagsearch_next_end'].unique()[0]]

            lgs_winsize_normalized = np.abs(lgs_winsize[0] - lgs_winsize[1])  # Range
            lgs_winsize_normalized = lgs_winsize_normalized / 2  # Normalized search window +/- around zero
            return lgs_winsize_normalized

    return ProcessingChain