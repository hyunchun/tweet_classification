from __future__ import print_function
from __future__ import division
try:
    import json
except ImportError:
    import simplejson as json
import string
import sys

# ----- labeler function ----- #
def labeler_json():
    filename = sys.argv[1]
    inFile = open("%s" %(filename), "r")

    # type: 0: None, 1: News, 2: announcement-advertisement, 3: emergency, 4: quotes
    # content: 0: URL/mentions only, 1: environmental, 2: business, 3: sports, 4: technology/science, 5: entertainments, 6: personal, 7: politics, 8: lifestyle
    # lifestyle: lifestyle, location, religion, events, food, home improvement, outer life-related
    # personal: relationship, colloquial, quick comments, opinion, feeling

    type_label_list = []
    content_label_list = []
    
    # read in the file first
    tweet_list = []
    for line in inFile:
        tweet_list.append(line)
    
    # read in label until "q" label
    labeled_count = 0
    for line in tweet_list:
        tweet = json.loads(line.strip())
        try:
            type_label = tweet["type_label"]
            content_label = tweet["content_label"]
        
            print("count: ", labeled_count, ", type: ", type_label, ", content:", content_label)
        
            if (type_label == "q"):
                break
            else:
                type_label_list.append(type_label)
                content_label_list.append(content_label)
                labeled_count += 1
        except:
            print("no \"q\" label exist in the tweet set")
            break
    print("resuming labeling tweets at: ", labeled_count)
    
    # start labeling
    while (labeled_count < len(tweet_list)):
        print("\n", labeled_count)
        line = tweet_list[labeled_count]
        tweet = json.loads(line.strip())
        labeled_count += 1
       
       # read in previous labels until 'q' label 
        print("screen name: ", tweet['screen_name'])
        print("tweet: ", tweet['text'].encode(encoding="utf-8"))
        type_label = raw_input("type label: ")
        content_label = raw_input("content label: ")
            
        # show bio 
        if (type_label == "bio") or (content_label == "bio"):
            print("\nbio: ", tweet['bio'])
            print("screen name: ", tweet['screen_name'])
            print("tweet: ", tweet['text'].encode(encoding="utf-8"))
            type_label = raw_input("type label: ")
            content_label = raw_input("content label: ")
            
        # quit
        elif (type_label == "quit") or (content_label == "quit"):
            print("exiting labeling")
            type_label_list.append("q")
            content_label_list.append("q")
            
            break
        
        # add to list
        type_label_list.append(type_label)
        content_label_list.append(content_label)
             
    # close inFile
    inFile.close()
    
    # create an output file
    outFile = open("%s" %(filename+"_l"), "w")
    count = 0
    for line in tweet_list:
        tweet = json.loads(line.strip())
        try:
            tweet['type_label'] = type_label_list[count]
            tweet['content_label'] = content_label_list[count]
            outFile.write(json.dumps(tweet))
            outFile.write('\n')
            count += 1
        except:
            outFile.write(json.dumps(tweet))
            outFile.write("\n")
  
    # close outFile
    outFile.close()
    print("finished label. Total: ", count, " lines read")
        
# ----- main ----- #
def main():
    labeler_json()

# ---------------------- #
if __name__ == "__main__":
    main()
