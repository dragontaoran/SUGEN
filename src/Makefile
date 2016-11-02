CC = g++
CFLAGS = -c -O4 -std=c++11 -I../eigen-eigen-07105f7124f9 -I../libStatGen/include -I../CodeBase
OBJ_DIR = obj

all: ./$(OBJ_DIR)/main.o ./$(OBJ_DIR)/sugen_utils.o

./$(OBJ_DIR)/main.o: main.cpp ./$(OBJ_DIR)/sugen_utils.o
	$(CC) $(CFLAGS) main.cpp -o ./$(OBJ_DIR)/main.o

./$(OBJ_DIR)/sugen_utils.o: sugen_utils.cpp sugen_utils.h
	$(CC) $(CFLAGS) sugen_utils.cpp -o ./$(OBJ_DIR)/sugen_utils.o

clean:
	rm -rf ./$(OBJ_DIR)/*.o

