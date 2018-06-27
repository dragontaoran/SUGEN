CC = g++
EXE = SUGEN
CFLAGS = -O4 -std=c++11 -I./libStatGen/include -I. -D__ZLIB_AVAILABLE__ -D_FILE_OFFSET_BITS=64 -D__STDC_LIMIT_MACROS -I./codeBase -I./eigen
CODEBASE_OBJS = ./codeBase/obj/command_line_utils.o ./codeBase/obj/read_file_utils.o ./codeBase/obj/read_input.o ./codeBase/obj/read_table_with_header.o ./codeBase/obj/map_utils.o ./codeBase/obj/data_structures.o ./codeBase/obj/eq_solver.o ./codeBase/obj/cdf_fns.o ./codeBase/obj/number_comparison.o ./codeBase/obj/constants.o ./codeBase/obj/gamma_fns.o ./codeBase/obj/string_utils.o ./codeBase/obj/test_utils.o
SRC_OBJS = ./src/obj/sugen_utils.o ./src/obj/main.o

all:
	(cd ./libStatGen; make)
	(cd ./codeBase; make)
	(cd ./src; make)
	$(CC) $(CFlAGS) -o $(EXE) $(CODEBASE_OBJS) $(SRC_OBJS) ./libStatGen/libStatGen.a -lm -lz

clean:
	(cd ./libStatGen; make clean)
	(cd ./codeBase; make clean)
	(cd ./src; make clean)
	rm $(EXE)
