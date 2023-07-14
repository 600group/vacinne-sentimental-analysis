
import twitter
from functools import partial
import sys
import time
from urllib.error import URLError
from http.client import BadStatusLine
from collections import OrderedDict
import csv





def getTwitterAPI():

    # the following code is based on the code from book <Mining the social web>
    # the function return the twitter API

    CONSUMER_KEY = '1tHO4NtWarr69yiDCsiLjKbwi'
    CONSUMER_SECRET = 'cO0n6ezKh4xdC6u0bHesEPuOK2RfxL8GwaTBJuUOhT1oYbPc6p'
    OAUTH_TOKEN = '1499766466832334849-GzQ4uru5zxJHBCOLYjOSJwZIgZ4efg'
    OAUTH_TOKEN_SECRET = 'JyckvQX3Ew3JSFmm3WyYcuAPM9G05ZDY7gD2dBKhXiexa'


    auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

    twitter_api = twitter.Twitter(auth=auth)

    return twitter_api


def make_twitter_request(twitter_api_func, max_errors=10, *args, **kw):

    # The following code is from the book <Mining the social web>

    # A nested helper function that handles common HTTPErrors. Return an updated
    # value for wait_period if the problem is a 500 level error. Block until the
    # rate limit is reset if it's a rate limiting issue (429 error). Returns None
    # for 401 and 404 errors, which requires special handling by the caller.
    def handle_twitter_http_error(e, wait_period=2, sleep_when_rate_limited=True):

        if wait_period > 3600:  # Seconds
            print('Too many retries. Quitting.', file=sys.stderr)
            raise e

        # See https://developer.twitter.com/en/docs/basics/response-codes
        # for common codes

        if e.e.code == 401:
            print('Encountered 401 Error (Not Authorized)', file=sys.stderr)
            return None
        elif e.e.code == 404:
            print('Encountered 404 Error (Not Found)', file=sys.stderr)
            return None
        elif e.e.code == 429:
            print('Encountered 429 Error (Rate Limit Exceeded)', file=sys.stderr)
            if sleep_when_rate_limited:
                print("Retrying in 15 minutes...ZzZ...", file=sys.stderr)
                sys.stderr.flush()
                time.sleep(60 * 15 + 5)
                print('...ZzZ...Awake now and trying again.', file=sys.stderr)
                return 2
            else:
                raise e  # Caller must handle the rate limiting issue
        elif e.e.code in (500, 502, 503, 504):
            print('Encountered {0} Error. Retrying in {1} seconds'.format(e.e.code, wait_period), file=sys.stderr)
            time.sleep(wait_period)
            wait_period *= 1.5
            return wait_period
        else:
            raise e

    # End of nested helper function

    wait_period = 2
    error_count = 0

    while True:
        try:
            return twitter_api_func(*args, **kw)
        except twitter.api.TwitterHTTPError as e:
            error_count = 0
            wait_period = handle_twitter_http_error(e, wait_period)
            if wait_period is None:
                return
        except URLError as e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            print("URLError encountered. Continuing.", file=sys.stderr)
            if error_count > max_errors:
                print("Too many consecutive errors...bailing out.", file=sys.stderr)
                raise
        except BadStatusLine as e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            print("BadStatusLine encountered. Continuing.", file=sys.stderr)
            if error_count > max_errors:
                print("Too many consecutive errors...bailing out.", file=sys.stderr)
                raise


if __name__ == '__main__':


    # get twitter_api object to process request
    twitter_api = getTwitterAPI()
    q = 'COVID-19 vaccine'
    twitter_stream = twitter.TwitterStream(auth=twitter_api.auth)
    stream = twitter_stream.statuses.filter(languages=["en"], track=q)
    counter = 0
    data_output = open('data.txt', 'a')
    for tweet in stream:
        if(counter ==10):
            break
        counter +=1
        print("**********")
        print(tweet['text'])
        data_output.write(tweet['text']+'\n')
        print("***********")
    data_output.close()
    print("End!")

    # with open('data.txt','a',encoding='UTF8') as f:
    #     writer = csv.writer(f)
    #
    #     for item in range(100):
    #         writer.writerow( str(item) )



else:
    print("This is not main function, the code cannot be executed")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

#test
