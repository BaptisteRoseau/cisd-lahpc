CFLAGS= -Wall -I ../perf/
CLIBS=

%.o: %.c
	gcc -c $(CFLAGS) -o $^