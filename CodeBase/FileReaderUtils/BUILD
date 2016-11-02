command_line_utils
{
  file : command_line_utils.cpp
  type : CLASS
  dependencies
  {
    read_file_structures.h
    ../MapUtils/map_utils.o
    ../MathUtils/data_structures.o
    ../StringUtils/string_utils.o
  }
}

csv_utils
{
  file : csv_utils.cpp
  type : CLASS
  dependencies
  {
    read_file_structures.h
    read_file_utils.o
    ../StringUtils/string_utils.o
  }
}

for_ran_main
{
  file : for_ran_main.cpp
  type : MAIN
  dependencies
  {
    read_file_utils.o
    read_input.o
    read_table_with_header.o
    ../MathUtils/data_structures.o
    ../Regression/regression_utils.o
    ../StringUtils/string_utils.o
  }
}

plink_utils
{
  file : plink_utils.cpp
  type : CLASS
  dependencies
  {
    csv_utils.o
    vcf_utils.o
    ../MapUtils/map_utils.o
    ../StringUtils/string_utils.o
  }
}

plink_utils_test
{
  file : plink_utils_test.cpp
  type : MAIN
  dependencies
  {
    plink_utils.o
    ../StringUtils/string_utils.o
  }
}

read_file_structures
{
  file : read_file_structures.h
  type : HEADER
}

read_file_utils
{
  file : read_file_utils.cpp
  type : CLASS
  dependencies
  {
    command_line_utils.o
    read_file_structures.h
    ../MapUtils/map_utils.o
    ../MathUtils/data_structures.o
    ../MathUtils/eq_solver.o
    ../TestUtils/test_utils.o
  }
}

read_file_utils_test
{
  file : read_file_utils_test.cpp
  type : MAIN
  dependencies
  {
    read_file_utils.o
  }
}

read_input
{
  file : read_input.cpp
  type : CLASS
  dependencies
  {
    read_file_utils.o
    read_table_with_header.o
    ../MathUtils/constants.o
    ../MathUtils/data_structures.o
    ../MathUtils/number_comparison.o
    ../StringUtils/string_utils.o
    ../TestUtils/test_utils.o
  }
}

read_interval_censored_data_main
{
  file : read_interval_censored_data_main.cpp
  type : MAIN
  dependencies
  {
    command_line_utils.o
    read_input.o
    read_time_dep_interval_censored_data.o
    read_time_indep_interval_censored_data.o
    ../Regression/regression_utils.o
  }
}

read_table_with_header
{
  file : read_table_with_header.cpp
  type : CLASS
  dependencies
  {
    read_file_utils.o
    ../MapUtils/map_utils.o
    ../MathUtils/constants.o
    ../MathUtils/data_structures.o
    ../StringUtils/string_utils.o
  }
}

read_time_dep_interval_censored_data
{
  file : read_time_dep_interval_censored_data.cpp
  type : CLASS
  dependencies
  {
    read_file_utils.o
    read_table_with_header.o
    ../MapUtils/map_utils.o
    ../MathUtils/constants.o
    ../MathUtils/data_structures.o
    ../MathUtils/eq_solver.o
    ../MathUtils/statistics_utils.o
    ../StringUtils/string_utils.o
  }
}

read_time_indep_interval_censored_data
{
  file : read_time_indep_interval_censored_data.cpp
  type : CLASS
  dependencies
  {
    read_file_utils.o
    read_table_with_header.o
    read_time_dep_interval_censored_data.o
    ../MapUtils/map_utils.o
    ../MathUtils/constants.o
    ../MathUtils/data_structures.o
    ../MathUtils/eq_solver.o
    ../MathUtils/statistics_utils.o
    ../StringUtils/string_utils.o
  }
}

vcf_main
{
  file : vcf_main.cpp
  type : MAIN
}

vcf_utils
{
  file : vcf_utils.cpp
  type : CLASS
  dependencies
  {
    ../StringUtils/string_utils.o
  }
}
