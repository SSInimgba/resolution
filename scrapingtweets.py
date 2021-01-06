import snscrape.modules.twitter as sntwitter
import pandas as pd


def tweetscraper(keyword, sample_size):
    """
    scrapes twitter data of a given keyword and timeframe and samplesize
    and returns a dataframe
    INPUT:
        keyword: "#lol since:2020-01-01 until:2020-01-03"
        sample_size: 500
    OUTPUT:
        df
    """

    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(keyword).get_items()):
        if i > sample_size:
            break
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.user.username, tweet.lang, tweet.sourceLabel])

    # Creating a dataframe from the tweets list above
    tweets_df = pd.DataFrame(tweets, columns=['datetime', 'tweet_id', 'text', 'username','lang', 'source'])
    return(tweets_df)
