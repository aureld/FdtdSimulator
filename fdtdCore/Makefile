CC = gcc
CFLAGS = -W -Wall -g -I/usr/local/libtiff/include 
LDFLAGS = -L/usr/local/lib -ltiff 
OMP =  -fopenmp

SRC = $(wildcard *.c)
OBJS = $(SRC:.c=.o)
AOUT = fdtdcore
 
all : $(AOUT) 
 
fdtdcore : $(OBJS)
	$(CC) $(OMP) $(LDFLAGS) -o $@ $^
%.o : %.c
	$(CC) $(OMP) $(CFLAGS) -o $@ -c $<
clean :
	@rm *.o
	@rm -f results/*.tif
cleaner : clean
	@rm -f $(AOUT)
	@rm -f results/*.tif
