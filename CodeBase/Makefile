CC = g++
CFLAGS = -c -O4 -std=c++11 -I../eigen-eigen-07105f7124f9 -I./
OBJ_DIR = obj

all: command_line_utils read_file_utils read_input read_table_with_header map_utils data_structures \
eq_solver cdf_fns number_comparison constants gamma_fns string_utils test_utils

command_line_utils: ./FileReaderUtils/command_line_utils.cpp ./FileReaderUtils/command_line_utils.h \
./FileReaderUtils/read_file_structures.h \
map_utils data_structures string_utils
	$(CC) $(CFLAGS) ./FileReaderUtils/command_line_utils.cpp -o ./$(OBJ_DIR)/command_line_utils.o
	
read_file_utils: ./FileReaderUtils/read_file_utils.cpp ./FileReaderUtils/read_file_utils.h \
./FileReaderUtils/read_file_structures.h \
command_line_utils map_utils data_structures eq_solver test_utils
	${CC} ${CFLAGS} ./FileReaderUtils/read_file_utils.cpp -o ./$(OBJ_DIR)/read_file_utils.o
	
read_input: ./FileReaderUtils/read_input.cpp ./FileReaderUtils/read_input.h \
read_file_utils read_table_with_header constants data_structures number_comparison string_utils test_utils
	${CC} ${CFLAGS} ./FileReaderUtils/read_input.cpp -o ./$(OBJ_DIR)/read_input.o
	
read_table_with_header: ./FileReaderUtils/read_table_with_header.cpp ./FileReaderUtils/read_table_with_header.h \
read_file_utils map_utils constants data_structures string_utils
	${CC} ${CFLAGS} ./FileReaderUtils/read_table_with_header.cpp -o ./$(OBJ_DIR)/read_table_with_header.o
	
map_utils: ./MapUtils/map_utils.cpp ./MapUtils/map_utils.h
	$(CC) $(CFLAGS) ./MapUtils/map_utils.cpp -o ./$(OBJ_DIR)/map_utils.o
	
data_structures: ./MathUtils/data_structures.cpp ./MathUtils/data_structures.h \
string_utils constants
	$(CC) $(CFLAGS) ./MathUtils/data_structures.cpp -o ./$(OBJ_DIR)/data_structures.o
	
eq_solver: ./MathUtils/eq_solver.cpp ./MathUtils/eq_solver.h \
cdf_fns constants data_structures gamma_fns number_comparison
	$(CC) $(CFLAGS) ./MathUtils/eq_solver.cpp -o ./$(OBJ_DIR)/eq_solver.o
	
cdf_fns: ./MathUtils/cdf_fns.cpp ./MathUtils/cdf_fns.h \
gamma_fns
	$(CC) $(CFLAGS) ./MathUtils/cdf_fns.cpp -o ./$(OBJ_DIR)/cdf_fns.o
	
number_comparison: ./MathUtils/number_comparison.cpp ./MathUtils/number_comparison.h \
constants
	$(CC) $(CFLAGS) ./MathUtils/number_comparison.cpp -o ./$(OBJ_DIR)/number_comparison.o
	
constants: ./MathUtils/constants.cpp ./MathUtils/constants.h
	$(CC) $(CFLAGS) ./MathUtils/constants.cpp -o ./$(OBJ_DIR)/constants.o
	
gamma_fns: ./MathUtils/gamma_fns.cpp ./MathUtils/gamma_fns.h
	$(CC) $(CFLAGS) ./MathUtils/gamma_fns.cpp -o ./$(OBJ_DIR)/gamma_fns.o
	
string_utils: ./StringUtils/string_utils.cpp ./StringUtils/string_utils.h
	$(CC) $(CFLAGS) ./StringUtils/string_utils.cpp -o ./$(OBJ_DIR)/string_utils.o
	
test_utils: ./TestUtils/test_utils.cpp ./TestUtils/test_utils.h \
string_utils
	$(CC) $(CFLAGS) ./TestUtils/test_utils.cpp -o ./$(OBJ_DIR)/test_utils.o
	
clean:
	rm -rf ./$(OBJ_DIR)/*.o
