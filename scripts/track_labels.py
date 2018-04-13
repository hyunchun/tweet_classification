from __future__ import print_function
from __future__ import division
try:
    import json
except ImportError:
    import simplejson as json
import string
import sys

def main():
    train_filename = sys.argv[1]
    inFile = open("%s" %(train_filename), "r")
    
    # label dictionary
    label_dict = {}
    type_label_dict = {}
    content_label_dict = {}
    
    count = 0
    
    # track labels
    for line in inFile:
        count += 1
        tweet = json.dumps(line)
        tweet = json.loads(tweet)
        tweet = json.loads(tweet)
        try:
            type_label = tweet["type_label"]
            content_label = tweet["content_label"]
            label_tuple = (type_label, content_label)
            
            # add to combined label dictionary
            if label_tuple not in label_dict:
                label_dict[label_tuple] = 1
            else:
                label_dict[label_tuple] += 1
            
            # add to type label dictionary
            if type_label not in type_label_dict:
                type_label_dict[type_label] = 1
            else:
                type_label_dict[type_label] += 1

            # add to content 
            if content_label not in content_label_dict:
                content_label_dict[content_label] = 1
            else:
                content_label_dict[content_label] += 1
        
        except:
            break
    
    # print result
    print("%s lines read" %(count))
    for label_tuple in label_dict:
        print("%s, %s: %s" %(label_tuple[0], label_tuple[1], label_dict[label_tuple]))
    
    print("\ntype labels: ")
    for type_label in type_label_dict:
        print("%s: %d" %(type_label, type_label_dict[type_label]))
    
    print("\ncontent labels: ")
    for content_label in content_label_dict:
        print("%s: %d" %(content_label, content_label_dict[content_label]))
    
    inFile.close()

 # ---------------------- #
if __name__ == "__main__":
    main()
