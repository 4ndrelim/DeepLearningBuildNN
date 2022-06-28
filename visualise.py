import sys

FILENAME = "my_model_results"

sys.path.append("./visualisation/network2/")
import compare_initialization as cmp_init
path = "././visualisation/network2/"
path = path + FILENAME
cmp_init.compare(path)
