from __future__ import annotations

import polars as pl
from sim_utils import collapse_abo
from sim_utils import PolarsFrame

__version__ = '0.0.0'


def _donor_outside_range(
    frame: PolarsFrame,
    don_col: str,
    can_min_col: str,
    can_max_col: str,
) -> PolarsFrame:
    '''

        Helper for screening out candidates when a
        donor value is outside the candidate's acceptable
        range.

    '''

    return (
        frame
        .with_columns(

            # A missing value for a min/max value is interpreted
            # as there is no min/max. Setting the value to something
            # unrealistically low/high ensures this behavior.

            pl.col(can_max_col)
            .fill_null(99999)
            .fill_nan(99999)
            .alias(f'tmp_{can_max_col}'),

            pl.col(can_min_col)
            .fill_null(-99999)
            .fill_nan(-99999)
            .alias(f'tmp_{can_min_col}'),
        )
        .filter(
            (
                (pl.col(don_col) >= pl.col(f'tmp_{can_min_col}')) &
                (pl.col(don_col) <= pl.col(f'tmp_{can_max_col}'))
            ) |
            pl.col(don_col).is_null() |
            pl.col(don_col).is_nan(),
        )
        .drop(f'tmp_{can_min_col}', f'tmp_{can_max_col}')
    )


def abo_compatibility(frame: PolarsFrame) -> PolarsFrame:
    '''

        Removes rows that do not include biologically
        compatible combinations.

    '''

    frame = (
        frame
        .with_columns(
            collapse_abo(pl.col('DON_ABO')).alias('don_abo_clean'),
            collapse_abo(pl.col('CAN_ABO')).alias('can_abo_clean'),
        )
        .filter(
            (
                pl.col('don_abo_clean') == pl.lit('O')
            ) |
            (
                pl.col('can_abo_clean').is_in(('AB', 'A1B', 'A2B'))
            ) |
            (
                pl.col('don_abo_clean').is_in(('A', 'A1', 'A2')) &
                pl.col('can_abo_clean').is_in(('A', 'A1', 'A2'))
            ) |
            (
                (pl.col('don_abo_clean') == pl.lit('B')) &
                (pl.col('can_abo_clean') == pl.lit('B'))
            ),
        )
        .drop(('don_abo_clean', 'can_abo_clean'))
    )

    return frame


def age(frame: PolarsFrame) -> PolarsFrame:
    '''

        Removes rows where the donors age is outside
        the candidates specified range.

    '''

    return _donor_outside_range(
        frame,
        'DON_AGE_IN_MONTHS',
        'CAN_MIN_AGE',
        'CAN_MAX_AGE',
    )


def distance(frame: PolarsFrame) -> PolarsFrame:
    '''

        Removes rows indicating the donor is too
        far away from the candidate.

    '''

    if 'distance_NM' not in frame.collect_schema():
        raise KeyError(
            'Missing column distance_NM, the distance in nautical '
            'miles between the transplant candidate transplant center '
            'and donor hospital. This is not contained in the SRTR '
            'standard analysis files (SAFs) and must be calculated by '
            'you :)',
        )

    return (
        frame
        .filter(
            pl.col('distance_NM').is_null() |
            pl.col('distance_NM').is_nan() |
            pl.col('CAN_MAX_MILE').is_null() |
            pl.col('CAN_MAX_MILE').is_nan() |
            (pl.col('distance_NM') <= pl.col('CAN_MAX_MILE')),
        )
    )


def hbc(frame: PolarsFrame) -> PolarsFrame:
    '''

        Removes rows when the donor is hbc anti-body positive
        and the candidate is unwilling to accept that.

    '''

    return (
        frame
        .filter(
            pl.col('DON_ANTI_HBC').is_null() |
            (pl.col('DON_ANTI_HBC') != pl.lit('P')) |
            (
                (pl.col('DON_ANTI_HBC') == pl.lit('P')) &
                (pl.col('CAN_ACPT_HBC_POS') == pl.lit('Y'))
            ),
        )
    )


def hcv(frame: PolarsFrame) -> PolarsFrame:
    '''

        Removes rows when the donor is anti-hcv positive
        and the candidate is unwilling to accept an
        anti-hcv positive donor.

    '''

    return (
        frame
        .filter(
            pl.col('DON_ANTI_HCV').is_null() |
            (pl.col('DON_ANTI_HCV') != pl.lit('P')) |
            (
                (pl.col('DON_ANTI_HCV') == pl.lit('P')) &
                (pl.col('CAN_ACPT_HCV_POS') == pl.lit('Y'))
            ),
        )
    )


def inactive(frame: PolarsFrame) -> PolarsFrame:
    '''

        Removes rows indicating the candidate is inactive.

    '''

    # Based on an inspection of all CANHX_STAT_CD values
    # in stathist safs from 2606.
    INACTIVE_STATUS_CODES = (
        1999.0,
        2999.0,
        3999.0,
        4999.0,
        5999.0,
        6999.0,
        7999.0,
        8999.0,
    )

    return (
        frame
        .remove(
            pl.col('CANHX_STAT_CD')
            .is_in(INACTIVE_STATUS_CODES),
        )
    )


def weight(frame: PolarsFrame) -> PolarsFrame:
    '''

        Removes rows where the donors weight is outside
        the candidates specified range.

    '''

    return _donor_outside_range(
        frame,
        'DON_WGT_KG',
        'CAN_MIN_WGT',
        'CAN_MAX_WGT',
    )
