CC = g++
EXE = SUGEN
CFLAGS = -O4 -std=c++11 -I./libStatGen/include -I. -D__ZLIB_AVAILABLE__ -D_FILE_OFFSET_BITS=64 -D__STDC_LIMIT_MACROS -I./CodeBase -I./eigen-eigen-07105f7124f9
CODEBASE_OBJS = ./CodeBase/obj/command_line_utils.o ./CodeBase/obj/read_file_utils.o ./CodeBase/obj/read_input.o ./CodeBase/obj/read_table_with_header.o ./CodeBase/obj/map_utils.o ./CodeBase/obj/data_structures.o ./CodeBase/obj/eq_solver.o ./CodeBase/obj/cdf_fns.o ./CodeBase/obj/number_comparison.o ./CodeBase/obj/constants.o ./CodeBase/obj/gamma_fns.o ./CodeBase/obj/string_utils.o ./CodeBase/obj/test_utils.o
SRC_OBJS = ./src/obj/sugen_utils.o ./src/obj/main.o

all:
	(cd ./libStatGen; make)
	(cd ./CodeBase; make)
	(cd ./src; make)
	$(CC) $(CFlAGS) -o $(EXE) $(CODEBASE_OBJS) $(SRC_OBJS) ./libStatGen/libStatGen.a -lm -lz

clean:
	(cd ./libStatGen; make clean)
	(cd ./CodeBase; make clean)
	(cd ./src; make clean)
	rm $(EXE)
