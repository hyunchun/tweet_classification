# Import the necessary package to process data in JSON format
try:
    import json
except ImportError:
    import simplejson as json

# Import the necessary methods from "twitter" library
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream



# Variables that contains the user credentials to access Twitter API 
ACCESS_TOKEN =
ACCESS_SECRET =
CONSUMER_KEY = 
CONSUMER_SECRET = 

oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

# Initiate the connection to Twitter Streaming API
twitter = Twitter(auth=oauth)
#twitter = twitter.statuses.filter(language="english")
#iterator = twitter_stream.statuses.filter(language="en")

# Get a sample of the public data following through Twitter
#iterator = twitter.statuses.user_timeline(screen_name=user_name) 

# Print each tweet in the stream to the screen 
# Here we set it to stop after getting 1000 tweets. 
# You don't have to set it to stop, but can continue running 
# the Twitter API to collect data for days or even longer. 
#name = raw_input("Please enter name: ")
#tweet_count = int(raw_input("Please enter status counts: "))

#iterator = twitter.statuses.user_timeline(screen_name=name, count=tweet_count, exclude_replies=False)
id_entered = int(raw_input("id: "))
#iterator = twitter.statuses.show(id=id_entered)

twitter_stream = TwitterStream(auth=oauth)
iterator = twitter_stream.statuses.show(id = id_entered)


#iterator = twitter.statuses.user_timeline(screen_name=name, count=tweet_count)

output_file = open("test.txt", "w")

#tweet_count = 1000
for tweet in iterator:
    #print(tweet)
    #tweet_count -= 1
    # Twitter Python Tool wraps the data returned by Twitter 
    # as a TwitterDictResponse object.
    # We convert it back to the JSON format to print/score
    output_file.write(json.dumps(tweet))
    print(json.dumps(tweet))    
    output_file.write("\n")
    #print json.dumps(tweet)
    # The command below will do pretty printing for JSON data, try it out
    # print json.dumps(tweet, indent=4)
       
    #if tweet_count <= 0:
    #    break

output_file.close()

