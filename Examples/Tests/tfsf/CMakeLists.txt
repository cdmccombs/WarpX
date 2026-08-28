# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_tfsf  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_tfsf  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1002400"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_tfsf  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_tfsf  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1001400"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_tfsf  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_tfsf  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000800"  # checksum
    OFF  # dependency
)
