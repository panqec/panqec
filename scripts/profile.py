import sys
import pstats

"""Read cProfile output file."""

p = pstats.Stats(sys.argv[1])
p.strip_dirs().sort_stats('tottime').print_stats(10)
