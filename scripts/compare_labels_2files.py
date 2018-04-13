from __future__ import print_function
from __future__ import division
try:
    import json
except ImportError:
    import simplejson as json
import string
import sys

def main():
    train_filename1 = sys.argv[1]
    inFile1 = open("%s" %(train_filename1), "r")
    train_filename2 = sys.argv[2]
    inFile2 = open("%s" %(train_filename2), "r")

    # file 1
    file1_text = []
    file1_typelabel = []
    file2_typelabel = []
    file2_text = []
    file1_contentlabel = []
    file2_contentlabel = []

    for line in inFile1:
        tweet = json.dumps(line)
        tweet = json.loads(tweet)
        tweet = json.loads(tweet)
        file1_text.append(tweet["text"])
        file1_typelabel.append(int(tweet["type_label"]))
        file1_contentlabel.append(int(tweet["content_label"]))

    for line in inFile2:
        tweet = json.dumps(line)
        tweet = json.loads(tweet)
        tweet = json.loads(tweet)
        file2_text.append(tweet["text"])
        file2_typelabel.append(int(tweet["type_label"]))
        file2_contentlabel.append(int(tweet["content_label"]))
    
    type_mismatch = {}
    print("type mismatch")
    type_difference = 0
    content_difference = 0
    for i in range(0, len(file1_typelabel)):
        if file1_typelabel[i] != file2_typelabel[i]:
            print("text: ", file1_text[i])
            print("file 1 type: ", file1_typelabel[i], ", file 2 type: ", file2_typelabel[i])
            type_difference += 1

            type_tuple = (file1_typelabel[i], file2_typelabel[i])
            if type_tuple not in type_mismatch:
                type_mismatch[type_tuple] = 1
            else:
                type_mismatch[type_tuple] += 1
    print("content mismatch")
    for i in range(0, len(file1_contentlabel)):
        if file1_contentlabel[i] != file2_contentlabel[i]:
            print("text: ", file1_text[i])
            print("file 1 content: ", file1_contentlabel[i], ", file 2 content: ", file2_contentlabel[i])
            content_difference += 1
    
    print("type difference: ", type_difference)
    print(type_mismatch)
    print("content difference: ", content_difference)

    inFile1.close()
    inFile2.close()
    
 # ---------------------- #
if __name__ == "__main__":
    main()
