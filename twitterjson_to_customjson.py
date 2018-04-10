#!/usr/bin/python

from io import open

import sys, getopt
import csv
import json
import datetime
from time import strptime

reload(sys)
sys.setdefaultencoding("utf-8")

def main(argv):
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    convert_json(input_file, output_file)

def convert_json(inputFile, outputFile):
    output = open(outputFile, "w")
    with open(inputFile, 'r', encoding='mac_roman') as original:
        for line in original:
            tweet = json.loads(line.strip())
            #print(tweet)
            #print(tweet['text'])
            row = {
                'tweet_id':tweet['id'],
                'date':convertDate(tweet['created_at']),
                'user_name':tweet['user']['name'],
                'screen_name':tweet['user']['screen_name'],
                'bio':tweet['user']['description'],
                'text':tweet['text']
            }
            output.write(unicode(json.dumps(row)))
            output.write(unicode('\n'))
    output.close()

def convertDate(originalFormat):
    original = originalFormat.split()
    year = str(original[5])
    month = str(strptime(original[1] , '%b').tm_mon)
    day = str(original[2])
    date = year + '-' + month + '-' + day
    return date

if __name__ == "__main__":
   main(sys.argv[1:])
