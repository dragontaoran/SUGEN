cdf_fns
{
  file : cdf_fns.cpp
  type : CLASS
  dependencies
  {
    gamma_fns.o
  }
}

cdf_fns_main
{
  file : cdf_fns_main.cpp
  type : MAIN
  dependencies
  {
    cdf_fns.o
    ../StringUtils/string_utils.o
  }
}

constants
{
  file : constants.cpp
  type : CLASS
}

data_structures
{
  file : data_structures.cpp
  type : CLASS
  dependencies
  {
    constants.o
    ../StringUtils/string_utils.o
  }
}

eq_solver
{
  file : eq_solver.cpp
  type : CLASS
  dependencies
  {
    cdf_fns.o
    constants.o
    data_structures.o
    gamma_fns.o
    number_comparison.o
  }
}

eq_solver_main
{
  file : eq_solver_main.cpp
  type : MAIN
  dependencies
  {
    data_structures.o
    eq_solver.o
  }
}

gamma_fns
{
  file : gamma_fns.cpp
  type : CLASS
}

gamma_fns_test
{
  file : gamma_fns_test.cpp
  type : MAIN
  dependencies
  {
    gamma_fns.o
    ../StringUtils/string_utils.o
  }
}

gaussian_quadrature
{
  file : gaussian_quadrature.cpp
  type : CLASS
  dependencies
  {
    constants.o
  }
}

interval_tree
{
  file : interval_tree.h
  type : HEADER
}

interval_tree_test
{
  file : interval_tree_test.cpp
  type : MAIN
  dependencies
  {
    interval_tree.h
  }
}

number_comparison
{
  file : number_comparison.cpp
  type : CLASS
  dependencies
  {
    constants.o
  }
}

quad_orig
{
  file : quad_orig.cpp
  type : MAIN
}

quantile_fns
{
  file : quantile_fns.cpp
  type : CLASS
  dependencies
  {
    eq_solver.o
  }
}

quantile_fns_main
{
  file : quantile_fns_main.cpp
  type : MAIN
  dependencies
  {
    quantile_fns.o
    ../StringUtils/string_utils.o
  }
}

statistics_utils
{
  file : statistics_utils.cpp
  type : CLASS
  dependencies
  {
    constants.o
    gamma_fns.o
    number_comparison.o
    quantile_fns.o
    ../MapUtils/map_utils.o
    ../StringUtils/string_utils.o
  }
}

statistics_utils_test
{
  file : statistics_utils_test.cpp
  type : MAIN
  dependencies
  {
    statistics_utils.o
    ../FileReaderUtils/csv_utils.o
    ../StringUtils/string_utils.o
  }
}
