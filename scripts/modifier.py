from __future__ import print_function
from __future__ import division

try:
    import json
except ImportError:
    import simplejson as json

import string
import sys
import os

def main():
    filename = sys.argv[1]
    
    inFile = open("%s" %(filename))
    
    # read in the file first
    tweet_list = []
    for line in inFile:
        tweet_list.append(line)

    # take in which label type that user wants to modify/edit
    modify_label = raw_input("modify content or type: ")
    
    if (modify_label != "content") and (modify_label != "type"):
        print("typed: ", modify_label, ", only content or type label")
        sys.exit(1)

    # take in which label type that user wants to search for 
    chosen_label = raw_input("search content or type: ")

    if (chosen_label != "content") and (chosen_label != "type"):
        print("typed: ", chosen_label, ", only content or type label")
        sys.exit(1)

    # takes in which lavel title that user wants to search for
    chosen_label_title = raw_input("label (number): ")

    line_count = 0
    for line in tweet_list:
        # read in json line (except: empty line, continue)
        tweet = json.dumps(line)
        tweet = json.loads(tweet)
        tweet = json.loads(tweet)
        
        try:
            type_label = tweet["type_label"]
            content_label = tweet["content_label"]
        except:
            continue

        # modify
        if chosen_label == "content":
            current_label = content_label
        elif chosen_label == "type":
            current_label = type_label
        
        # if current_label == q, quit
        if current_label == "q":
            print("Reached 'q' marker")
            break
        elif current_label == chosen_label_title:
            if modify_label == "content":
                print("screen name: ", tweet["screen_name"])
                print("text: ", tweet["text"].encode(encoding="utf-8"))
                print("type label: ", tweet["type_label"])

                new_content_label = raw_input("new content label: ")
                
                if (new_content_label == "next") or (new_content_label == "no"):
                    print("no change\n")
                elif (new_content_label == "quit") or (new_content_label == "q"):
                    print("ending modifying process...")
                    break
                else:
                    tweet["content_label"] = new_content_label
            elif modify_label == "type":
                print("screen name: ", tweet["screen_name"])
                print("text: ", tweet["text"].encode(encoding="utf-8"))
                print("content label: ", tweet["content_label"])
        
                new_type_label = raw_input("new type label: ")
                
                if (new_type_label == "next") or (new_type_label == "no"):
                    print("no change\n")
                elif (new_type_label == "quit") or (new_type_label == "q"):
                    print("ending modifying process...")
                    break
                else:
                    tweet["type_label"] = new_type_label
            
        tweet_list[line_count] = json.dumps(tweet) + "\n"
        line_count += 1
    
    # save to output
    print("saving...")
    os.remove(filename)
    with open(filename, "w") as f:
        for line in tweet_list:
            f.write(line)
    f.close()


 # ---------------------- #
if __name__ == "__main__":
    main()
