from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import screening


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_abo_compatibility(polars_frame_function):

    frame = polars_frame_function(
        data=(
            # DON_ABO, CAN_ABO
            ('   ', 'A  '),
            ('   ', 'A1 '),
            ('   ', 'A1B'),
            ('   ', 'A2 '),
            ('   ', 'A2B'),
            ('   ', 'AB '),
            ('   ', 'B  '),
            ('   ', 'O  '),
            ('   ', 'UNK'),
            ('   ', 'Z  '),
            ('A  ', 'A  '),
            ('A  ', 'A1 '),
            ('A  ', 'A1B'),
            ('A  ', 'A2 '),
            ('A  ', 'A2B'),
            ('A  ', 'AB '),
            ('A  ', 'B  '),
            ('A  ', 'O  '),
            ('A  ', 'UNK'),
            ('A  ', 'Z  '),
            ('A1 ', 'A  '),
            ('A1 ', 'A1 '),
            ('A1 ', 'A1B'),
            ('A1 ', 'A2 '),
            ('A1 ', 'A2B'),
            ('A1 ', 'AB '),
            ('A1 ', 'B  '),
            ('A1 ', 'O  '),
            ('A1 ', 'UNK'),
            ('A1 ', 'Z  '),
            ('A1B', 'A  '),
            ('A1B', 'A1 '),
            ('A1B', 'A1B'),
            ('A1B', 'A2 '),
            ('A1B', 'A2B'),
            ('A1B', 'AB '),
            ('A1B', 'B  '),
            ('A1B', 'O  '),
            ('A1B', 'UNK'),
            ('A1B', 'Z  '),
            ('A2 ', 'A  '),
            ('A2 ', 'A1 '),
            ('A2 ', 'A1B'),
            ('A2 ', 'A2 '),
            ('A2 ', 'A2B'),
            ('A2 ', 'AB '),
            ('A2 ', 'B  '),
            ('A2 ', 'O  '),
            ('A2 ', 'UNK'),
            ('A2 ', 'Z  '),
            ('A2B', 'A  '),
            ('A2B', 'A1 '),
            ('A2B', 'A1B'),
            ('A2B', 'A2 '),
            ('A2B', 'A2B'),
            ('A2B', 'AB '),
            ('A2B', 'B  '),
            ('A2B', 'O  '),
            ('A2B', 'UNK'),
            ('A2B', 'Z  '),
            ('AB ', 'A  '),
            ('AB ', 'A1 '),
            ('AB ', 'A1B'),
            ('AB ', 'A2 '),
            ('AB ', 'A2B'),
            ('AB ', 'AB '),
            ('AB ', 'B  '),
            ('AB ', 'O  '),
            ('AB ', 'UNK'),
            ('AB ', 'Z  '),
            ('B  ', 'A  '),
            ('B  ', 'A1 '),
            ('B  ', 'A1B'),
            ('B  ', 'A2 '),
            ('B  ', 'A2B'),
            ('B  ', 'AB '),
            ('B  ', 'B  '),
            ('B  ', 'O  '),
            ('B  ', 'UNK'),
            ('B  ', 'Z  '),
            ('O  ', 'A  '),
            ('O  ', 'A1 '),
            ('O  ', 'A1B'),
            ('O  ', 'A2 '),
            ('O  ', 'A2B'),
            ('O  ', 'AB '),
            ('O  ', 'B  '),
            ('O  ', 'O  '),
            ('O  ', 'UNK'),
            ('O  ', 'Z  '),
            ('UNK', 'A  '),
            ('UNK', 'A1 '),
            ('UNK', 'A1B'),
            ('UNK', 'A2 '),
            ('UNK', 'A2B'),
            ('UNK', 'AB '),
            ('UNK', 'B  '),
            ('UNK', 'O  '),
            ('UNK', 'UNK'),
            ('UNK', 'Z  '),
        ),
        orient='row',
        schema={
            'DON_ABO': pl.String,
            'CAN_ABO': pl.String,
        },
    )

    expected = polars_frame_function(
        data=(
            # DON_ABO, CAN_ABO
            ('   ', 'A1B'),
            ('   ', 'A2B'),
            ('   ', 'AB '),
            ('A  ', 'A  '),
            ('A  ', 'A1 '),
            ('A  ', 'A1B'),
            ('A  ', 'A2 '),
            ('A  ', 'A2B'),
            ('A  ', 'AB '),
            ('A1 ', 'A  '),
            ('A1 ', 'A1 '),
            ('A1 ', 'A1B'),
            ('A1 ', 'A2 '),
            ('A1 ', 'A2B'),
            ('A1 ', 'AB '),
            ('A1B', 'A1B'),
            ('A1B', 'A2B'),
            ('A1B', 'AB '),
            ('A2 ', 'A  '),
            ('A2 ', 'A1 '),
            ('A2 ', 'A1B'),
            ('A2 ', 'A2 '),
            ('A2 ', 'A2B'),
            ('A2 ', 'AB '),
            ('A2B', 'A1B'),
            ('A2B', 'A2B'),
            ('A2B', 'AB '),
            ('AB ', 'A1B'),
            ('AB ', 'A2B'),
            ('AB ', 'AB '),
            ('B  ', 'A1B'),
            ('B  ', 'A2B'),
            ('B  ', 'AB '),
            ('B  ', 'B  '),
            ('O  ', 'A  '),
            ('O  ', 'A1 '),
            ('O  ', 'A1B'),
            ('O  ', 'A2 '),
            ('O  ', 'A2B'),
            ('O  ', 'AB '),
            ('O  ', 'B  '),
            ('O  ', 'O  '),
            ('O  ', 'UNK'),
            ('O  ', 'Z  '),
            ('UNK', 'A1B'),
            ('UNK', 'A2B'),
            ('UNK', 'AB '),
        ),
        orient='row',
        schema={
            'DON_ABO': pl.String,
            'CAN_ABO': pl.String,
        },
    )
    assert_frame_equal(
        screening.abo_compatibility(frame),
        expected,
    )


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_age(polars_frame_function):

    frame = polars_frame_function(
        data=(
            # DON_AGE_IN_MONTHS, CAN_MIN_AGE, CAN_MAX_AGE
            (None, None, None),
            (None, 20, None),
            (None, 20, 185),
            (None, None, 185),
            (50, 20, 185),
            (10, 20, 185),
            (200, 20, 185),
            (50, 20, None),
            (10, 20, None),
            (200, None, 185),
            (150, None, 185),
        ),
        schema={
            'DON_AGE_IN_MONTHS': pl.Float64,
            'CAN_MIN_AGE': pl.Float64,
            'CAN_MAX_AGE': pl.Float64,
        },
        orient='row',
    )
    expected = polars_frame_function(
        data=(
            (None, None, None),
            (None, 20, None),
            (None, 20, 185),
            (None, None, 185),
            (50, 20, 185),
            (50, 20, None),
            (150, None, 185),
        ),
        schema={
            'DON_AGE_IN_MONTHS': pl.Float64,
            'CAN_MIN_AGE': pl.Float64,
            'CAN_MAX_AGE': pl.Float64,
        },
        orient='row',
    )
    frame = screening.age(frame)
    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_distance_NM_missing(polars_frame_function):

    frame = polars_frame_function()
    with pytest.raises(KeyError) as excinfo:
        screening.distance(frame)

    msg, = excinfo.value.args
    expected = (
        'Missing column distance_NM, the distance in nautical '
        'miles between the transplant candidate transplant center '
        'and donor hospital. This is not contained in the SRTR '
        'standard analysis files (SAFs) and must be calculated by '
        'you :)'
    )
    assert msg == expected


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_distance(polars_frame_function):

    frame = polars_frame_function(
        data=(
            # distance_NM, CAN_MAX_MILE
            (None, None),
            (100, None),
            (None, 500),
            (100, 500),
            (1000, 500),
        ),
        schema={
            'distance_NM': pl.Float64,
            'CAN_MAX_MILE': pl.Float64,
        },
        orient='row',
    )
    expected = polars_frame_function(
        data=(
            # distance_NM, CAN_MAX_MILE
            (None, None),
            (100, None),
            (None, 500),
            (100, 500),
        ),
        schema={
            'distance_NM': pl.Float64,
            'CAN_MAX_MILE': pl.Float64,
        },
        orient='row',
    )
    frame = screening.distance(frame)
    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_hbc(polars_frame_function):

    frame = polars_frame_function(
        data=(
            # DON_ANTI_HBC, CAN_ACPT_HBC_POS
            ('P', 'Y'),
            ('P', 'N'),
            ('P', ''),
            ('P', 'anything else'),
            ('P', None),
            ('anything else', 'Y'),
            ('anything else', 'N'),
            ('anything else', ''),
            ('anything else', 'anything else'),
            ('anything else', None),
            (None, None),
        ),
        schema={
            'DON_ANTI_HBC': pl.String,
            'CAN_ACPT_HBC_POS': pl.String,
        },
        orient='row',
    )
    expected = polars_frame_function(
        data=(
            # DON_ANTI_HBC, CAN_ACPT_HBC_POS
            ('P', 'Y'),
            ('anything else', 'Y'),
            ('anything else', 'N'),
            ('anything else', ''),
            ('anything else', 'anything else'),
            ('anything else', None),
            (None, None),
        ),
        schema={
            'DON_ANTI_HBC': pl.String,
            'CAN_ACPT_HBC_POS': pl.String,
        },
        orient='row',
    )
    frame = screening.hbc(frame)
    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_hcv(polars_frame_function):

    frame = polars_frame_function(
        data=(
            # DON_ANTI_HCV, CAN_ACPT_HCV_POS
            ('P', 'Y'),
            ('P', 'N'),
            ('P', ''),
            ('P', 'anything else'),
            ('P', None),
            ('anything else', 'Y'),
            ('anything else', 'N'),
            ('anything else', ''),
            ('anything else', 'anything else'),
            ('anything else', None),
            (None, None),
        ),
        schema={
            'DON_ANTI_HCV': pl.String,
            'CAN_ACPT_HCV_POS': pl.String,
        },
        orient='row',
    )
    expected = polars_frame_function(
        data=(
            # DON_ANTI_HCV, CAN_ACPT_HCV_POS
            ('P', 'Y'),
            ('anything else', 'Y'),
            ('anything else', 'N'),
            ('anything else', ''),
            ('anything else', 'anything else'),
            ('anything else', None),
            (None, None),
        ),
        schema={
            'DON_ANTI_HCV': pl.String,
            'CAN_ACPT_HCV_POS': pl.String,
        },
        orient='row',
    )
    frame = screening.hcv(frame)
    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_inactive(polars_frame_function):

    # Based on all stathist safs from 2606
    status_codes = [
        0.0,
        1010.0,
        1020.0,
        1030.0,
        1090.0,
        1110.0,
        1120.0,
        1130.0,
        1140.0,
        1150.0,
        1160.0,
        1999.0,
        2010.0,
        2020.0,
        2030.0,
        2090.0,
        2110.0,
        2120.0,
        2130.0,
        2140.0,
        2150.0,
        2160.0,
        2999.0,
        3010.0,
        3020.0,
        3999.0,
        4010.0,
        4020.0,
        4050.0,
        4060.0,
        4998.0,
        4999.0,
        5010.0,
        5999.0,
        6002.0,
        6004.0,
        6010.0,
        6011.0,
        6012.0,
        6020.0,
        6030.0,
        6040.0,
        6188.0,
        6189.0,
        6190.0,
        6191.0,
        6192.0,
        6193.0,
        6194.0,
        6195.0,
        6196.0,
        6197.0,
        6198.0,
        6199.0,
        6200.0,
        6201.0,
        6202.0,
        6203.0,
        6204.0,
        6205.0,
        6206.0,
        6207.0,
        6208.0,
        6209.0,
        6210.0,
        6211.0,
        6212.0,
        6213.0,
        6214.0,
        6215.0,
        6216.0,
        6217.0,
        6218.0,
        6219.0,
        6220.0,
        6221.0,
        6222.0,
        6223.0,
        6224.0,
        6225.0,
        6226.0,
        6227.0,
        6228.0,
        6229.0,
        6230.0,
        6231.0,
        6232.0,
        6233.0,
        6234.0,
        6235.0,
        6236.0,
        6237.0,
        6238.0,
        6239.0,
        6240.0,
        6241.0,
        6242.0,
        6243.0,
        6244.0,
        6245.0,
        6246.0,
        6247.0,
        6248.0,
        6249.0,
        6250.0,
        6251.0,
        6252.0,
        6253.0,
        6254.0,
        6255.0,
        6256.0,
        6257.0,
        6258.0,
        6259.0,
        6260.0,
        6261.0,
        6262.0,
        6263.0,
        6264.0,
        6265.0,
        6266.0,
        6267.0,
        6268.0,
        6269.0,
        6270.0,
        6271.0,
        6272.0,
        6274.0,
        6275.0,
        6276.0,
        6277.0,
        6278.0,
        6279.0,
        6280.0,
        6281.0,
        6282.0,
        6285.0,
        6290.0,
        6299.0,
        6999.0,
        7010.0,
        7099.0,
        7999.0,
        8010.0,
        8999.0,
    ]
    active_status_codes = [
        0.0,
        1010.0,
        1020.0,
        1030.0,
        1090.0,
        1110.0,
        1120.0,
        1130.0,
        1140.0,
        1150.0,
        1160.0,
        2010.0,
        2020.0,
        2030.0,
        2090.0,
        2110.0,
        2120.0,
        2130.0,
        2140.0,
        2150.0,
        2160.0,
        3010.0,
        3020.0,
        4010.0,
        4020.0,
        4050.0,
        4060.0,
        4998.0,
        5010.0,
        6002.0,
        6004.0,
        6010.0,
        6011.0,
        6012.0,
        6020.0,
        6030.0,
        6040.0,
        6188.0,
        6189.0,
        6190.0,
        6191.0,
        6192.0,
        6193.0,
        6194.0,
        6195.0,
        6196.0,
        6197.0,
        6198.0,
        6199.0,
        6200.0,
        6201.0,
        6202.0,
        6203.0,
        6204.0,
        6205.0,
        6206.0,
        6207.0,
        6208.0,
        6209.0,
        6210.0,
        6211.0,
        6212.0,
        6213.0,
        6214.0,
        6215.0,
        6216.0,
        6217.0,
        6218.0,
        6219.0,
        6220.0,
        6221.0,
        6222.0,
        6223.0,
        6224.0,
        6225.0,
        6226.0,
        6227.0,
        6228.0,
        6229.0,
        6230.0,
        6231.0,
        6232.0,
        6233.0,
        6234.0,
        6235.0,
        6236.0,
        6237.0,
        6238.0,
        6239.0,
        6240.0,
        6241.0,
        6242.0,
        6243.0,
        6244.0,
        6245.0,
        6246.0,
        6247.0,
        6248.0,
        6249.0,
        6250.0,
        6251.0,
        6252.0,
        6253.0,
        6254.0,
        6255.0,
        6256.0,
        6257.0,
        6258.0,
        6259.0,
        6260.0,
        6261.0,
        6262.0,
        6263.0,
        6264.0,
        6265.0,
        6266.0,
        6267.0,
        6268.0,
        6269.0,
        6270.0,
        6271.0,
        6272.0,
        6274.0,
        6275.0,
        6276.0,
        6277.0,
        6278.0,
        6279.0,
        6280.0,
        6281.0,
        6282.0,
        6285.0,
        6290.0,
        6299.0,
        7010.0,
        7099.0,
        8010.0,
    ]

    frame = polars_frame_function(
        data=status_codes,
        schema={'CANHX_STAT_CD': pl.Float64},
    )
    expected = polars_frame_function(
        data=active_status_codes,
        schema={'CANHX_STAT_CD': pl.Float64},
    )
    frame = screening.inactive(frame)
    assert_frame_equal(frame, expected)


@pytest.mark.parametrize(
    'polars_frame_function',
    (
        pl.DataFrame,
        pl.LazyFrame,
    ),
)
def test_weight(polars_frame_function):

    frame = polars_frame_function(
        data=(
            # DON_WGT_KG, CAN_MIN_WGT, CAN_MAX_WGT
            (None, None, None),
            (None, 20, None),
            (None, 20, 185),
            (None, None, 185),
            (50, 20, 185),
            (10, 20, 185),
            (200, 20, 185),
            (50, 20, None),
            (10, 20, None),
            (200, None, 185),
            (150, None, 185),
        ),
        schema={
            'DON_WGT_KG': pl.Float64,
            'CAN_MIN_WGT': pl.Float64,
            'CAN_MAX_WGT': pl.Float64,
        },
        orient='row',
    )
    expected = polars_frame_function(
        data=(
            (None, None, None),
            (None, 20, None),
            (None, 20, 185),
            (None, None, 185),
            (50, 20, 185),
            (50, 20, None),
            (150, None, 185),
        ),
        schema={
            'DON_WGT_KG': pl.Float64,
            'CAN_MIN_WGT': pl.Float64,
            'CAN_MAX_WGT': pl.Float64,
        },
        orient='row',
    )
    frame = screening.weight(frame)
    assert_frame_equal(frame, expected)
