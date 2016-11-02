clustered_mple_for_interval_censored_data
{
  file : clustered_mple_for_interval_censored_data.cpp
  type : CLASS
  dependencies
  {
    mple_for_interval_censored_data.o
    ../FileReaderUtils/read_file_utils.o
    ../FileReaderUtils/read_time_dep_interval_censored_data.o
    ../MapUtils/map_utils.o
    ../MathUtils/constants.o
    ../MathUtils/data_structures.o
    ../MathUtils/eq_solver.o
    ../MathUtils/gamma_fns.o
    ../MathUtils/gaussian_quadrature.o
    ../MathUtils/number_comparison.o
    ../MathUtils/statistics_utils.o
    ../StringUtils/string_utils.o
    ../TestUtils/test_utils.o
  }
}

clustered_mple_for_interval_censored_data_main
{
  file : clustered_mple_for_interval_censored_data_main.cpp
  type : MAIN
  dependencies
  {
    clustered_mple_for_interval_censored_data.o
    mple_for_interval_censored_data.o
    regression_utils.o
    ../FileReaderUtils/command_line_utils.o
    ../FileReaderUtils/read_input.o
    ../FileReaderUtils/read_time_dep_interval_censored_data.o
    ../FileReaderUtils/read_time_indep_interval_censored_data.o
    ../MathUtils/statistics_utils.o
  }
}

cox_compute_info_matrix_main
{
  file : cox_compute_info_matrix_main.cpp
  type : MAIN
  dependencies
  {
    cox_dual_regression.o
    ../FileReaderUtils/read_input.o
    ../MathUtils/data_structures.o
    ../StringUtils/string_utils.o
  }
}

cox_compute_k_m_estimator_main
{
  file : cox_compute_k_m_estimator_main.cpp
  type : MAIN
  dependencies
  {
    cox_dual_regression.o
    ../FileReaderUtils/read_input.o
    ../FileReaderUtils/read_table_with_header.o
    ../MathUtils/data_structures.o
    ../StringUtils/string_utils.o
  }
}

cox_dual_regression
{
  file : cox_dual_regression.cpp
  type : CLASS
  dependencies
  {
    regression_utils.o
    ../FileReaderUtils/read_file_utils.o
    ../FileReaderUtils/read_input.o
    ../FileReaderUtils/read_table_with_header.o
    ../MapUtils/map_utils.o
    ../MathUtils/constants.o
    ../MathUtils/data_structures.o
    ../MathUtils/gamma_fns.o
    ../MathUtils/number_comparison.o
    ../MathUtils/statistics_utils.o
  }
}

cox_dual_regression_main
{
  file : cox_dual_regression_main.cpp
  type : MAIN
  dependencies
  {
    cox_dual_regression.o
    regression_utils.o
    ../FileReaderUtils/command_line_utils.o
    ../FileReaderUtils/read_file_utils.o
    ../FileReaderUtils/read_input.o
    ../FileReaderUtils/read_table_with_header.o
    ../MapUtils/map_utils.o
    ../MathUtils/constants.o
    ../MathUtils/data_structures.o
    ../MathUtils/statistics_utils.o
    ../StringUtils/string_utils.o
  }
}

cox_regression
{
  file : cox_regression.cpp
  type : CLASS
  dependencies
  {
    linear_regression.o
    ../FileReaderUtils/read_table_with_header.o
    ../MathUtils/data_structures.o
    ../MathUtils/gamma_fns.o
    ../MathUtils/number_comparison.o
    ../MathUtils/statistics_utils.o
  }
}

cox_regression_main
{
  file : cox_regression_main.cpp
  type : MAIN
  dependencies
  {
    cox_regression.o
    regression_utils.o
    ../FileReaderUtils/command_line_utils.o
    ../FileReaderUtils/read_input.o
  }
}

cox_regression_test
{
  file : cox_regression_test.cpp
  type : MAIN
  dependencies
  {
    logistic_regression.o
    ../FileReaderUtils/read_table_with_header.o
  }
}

cox_regression_v2_api_main
{
  file : cox_regression_v2_api_main.cpp
  type : MAIN
  dependencies
  {
    cox_regression.o
    ../FileReaderUtils/read_table_with_header.o
    ../MathUtils/number_comparison.o
  }
}

linear_regression
{
  file : linear_regression.cpp
  type : CLASS
  dependencies
  {
    ../FileReaderUtils/read_table_with_header.o
    ../MathUtils/gamma_fns.o
    ../MathUtils/statistics_utils.o
  }
}

linear_regression_main
{
  file : linear_regression_main.cpp
  type : MAIN
  dependencies
  {
    linear_regression.o
    regression_utils.o
    ../FileReaderUtils/command_line_utils.o
    ../FileReaderUtils/read_input.o
  }
}

logistic_regression
{
  file : logistic_regression.cpp
  type : CLASS
  dependencies
  {
    linear_regression.o
    ../FileReaderUtils/read_table_with_header.o
    ../MathUtils/gamma_fns.o
    ../MathUtils/number_comparison.o
    ../MathUtils/statistics_utils.o
  }
}

logistic_regression_main
{
  file : logistic_regression_main.cpp
  type : MAIN
  dependencies
  {
    logistic_regression.o
    regression_utils.o
    ../FileReaderUtils/command_line_utils.o
    ../FileReaderUtils/read_input.o
  }
}

logistic_regression_test
{
  file : logistic_regression_test.cpp
  type : MAIN
  dependencies
  {
    logistic_regression.o
    ../FileReaderUtils/read_table_with_header.o
  }
}

mple_for_interval_censored_data
{
  file : mple_for_interval_censored_data.cpp
  type : CLASS
  dependencies
  {
    ../FileReaderUtils/read_file_utils.o
    ../FileReaderUtils/read_time_dep_interval_censored_data.o
    ../FileReaderUtils/read_time_indep_interval_censored_data.o
    ../MapUtils/map_utils.o
    ../MathUtils/constants.o
    ../MathUtils/data_structures.o
    ../MathUtils/eq_solver.o
    ../MathUtils/gamma_fns.o
    ../MathUtils/gaussian_quadrature.o
    ../MathUtils/number_comparison.o
    ../MathUtils/statistics_utils.o
    ../StringUtils/string_utils.o
    ../TestUtils/test_utils.o
  }
}

mple_for_interval_censored_data_main
{
  file : mple_for_interval_censored_data_main.cpp
  type : MAIN
  dependencies
  {
    mple_for_interval_censored_data.o
    regression_utils.o
    ../FileReaderUtils/command_line_utils.o
    ../FileReaderUtils/read_input.o
    ../FileReaderUtils/read_time_dep_interval_censored_data.o
    ../FileReaderUtils/read_time_indep_interval_censored_data.o
    ../MathUtils/statistics_utils.o
  }
}

multivariate_mple_for_interval_censored_data
{
  file : multivariate_mple_for_interval_censored_data.cpp
  type : CLASS
  dependencies
  {
    mple_for_interval_censored_data.o
    ../FileReaderUtils/read_file_utils.o
    ../FileReaderUtils/read_time_dep_interval_censored_data.o
    ../MapUtils/map_utils.o
    ../MathUtils/constants.o
    ../MathUtils/data_structures.o
    ../MathUtils/eq_solver.o
    ../MathUtils/gamma_fns.o
    ../MathUtils/gaussian_quadrature.o
    ../MathUtils/number_comparison.o
    ../MathUtils/statistics_utils.o
    ../StringUtils/string_utils.o
    ../TestUtils/test_utils.o
  }
}

multivariate_mple_for_interval_censored_data_main
{
  file : multivariate_mple_for_interval_censored_data_main.cpp
  type : MAIN
  dependencies
  {
    mple_for_interval_censored_data.o
    multivariate_mple_for_interval_censored_data.o
    regression_utils.o
    ../FileReaderUtils/command_line_utils.o
    ../FileReaderUtils/read_input.o
    ../FileReaderUtils/read_time_dep_interval_censored_data.o
    ../FileReaderUtils/read_time_indep_interval_censored_data.o
  }
}

regression_utils
{
  file : regression_utils.cpp
  type : CLASS
  dependencies
  {
    ../FileReaderUtils/read_file_utils.o
    ../FileReaderUtils/read_input.o
    ../MathUtils/data_structures.o
    ../MathUtils/eq_solver.o
    ../MathUtils/statistics_utils.o
  }
}
