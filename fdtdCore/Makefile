CC = gcc
CFLAGS = -W -Wall -g -I/usr/local/libtiff/include
LDFLAGS = -L/usr/local/libtiff/lib -ltiff
 
SRC = $(wildcard *.c)
OBJS = $(SRC:.c=.o)
AOUT = fdtdcore
 
all : $(AOUT) 
 
fdtdcore : $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^
%.o : %.c
	$(CC) $(CFLAGS) -o $@ -c $<
clean :
	@rm *.o
	@rm results/*.tif
cleaner : clean
	@rm $(AOUT)
	@rm results/*.tif