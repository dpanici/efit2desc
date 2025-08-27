import sys

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, "../efit2desc")  # insert path ti source so can import things

import numpy as np

print(np.__version__)
print(np.__file__)
print(sys.path)

from efit2desc import convert_EFIT_to_DESC

eq, efit = convert_EFIT_to_DESC("g200245.02200", L=16, M=16, psiN_cutoff=0.98)
