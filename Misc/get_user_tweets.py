def get_user_tweets(username = "", cnt = 200, ckey = "", csecret = "", akey = "", asecret = ""):
    #This function grabs tweets from a specific user and stores the tweets in a pandas dataframe
    #requires tweepy and pandas
    
    #authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(ckey, csecret)
    auth.set_access_token(akey, asecret)
    api = tweepy.API(auth)
    
    #make initial request for most recent tweets (200 is the maximum allowed count)
    tweets = api.user_timeline(screen_name = username, count = cnt)
    
    #initiate empty dataframe
    tweetData = pd.DataFrame(columns=['username', 'date', 'text'])
    #loop through the tweets and populate the dataframe
    for i in range(len(tweets)):
        tweetData.loc[i] = [username, tweets[i].created_at, tweets[i].text]
    
    #return the tweet dataframe
    return(tweetData)
