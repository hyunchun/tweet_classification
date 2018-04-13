from __future__ import print_function
from __future__ import division
try:
    import json
except ImportError:
    import simplejson as json
import string
import sys

def main():
    outfilename = sys.argv[1]
    outFile = open("%s" %(outfilename), "w")

    file_number = int(sys.argv[2])
    file_count = 3
    while file_number > 0:
        file_number -= 1
        filename = sys.argv[file_count]
        inFile = open("%s" %(filename), "r")
        file_count += 1

        for line in inFile:
            outFile.write(line)

        inFile.close()
    outFile.close()
 # ---------------------- #
if __name__ == "__main__":
    main()
