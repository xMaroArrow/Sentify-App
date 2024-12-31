import tweepy

# Twitter API credentials
api_key = "B8kZJ8o3RFMqJg3pQ3it9Xedd"
api_secret = "wVrJm5HtzKVRcfTpzN5ubdmeKobw2CJB4b6o5xGxPioX8EXTf7"
access_token = "716919102082846720-uyI91pJ8dLy68ngVieGviaoTWWfR9pu"
access_token_secret = "LejP9sVI4yta90zAAahDWCzAzEsBksgcHQCJeuhOT1Nu0"
Bearer = "AAAAAAAAAAAAAAAAAAAAAONjxwEAAAAAj74TclHPqXhKgRmuSlsIRJSXF9g%3DdsDgbh7xAa0apGZjGtkfFWYKWVIZO0Hd2Y1Hi9uqXobjSrzFw1"

# Authenticate with the API
auth = tweepy.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

def fetch_tweet_text(tweet_url):
    try:
        tweet_id = tweet_url.split("/")[-1]
        tweet = api.get_status(tweet_id)
        return tweet.text
    except Exception as e:
        print(f"Error fetching tweet: {e}")
        return None
