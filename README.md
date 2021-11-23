# How People Like Munich Bus Servce

How people are satisfied  with the bus service in Munich? We collected
**tweets** within 20 km of Munich, and measure the **sentiments** in the texts.
Find how people are satisfied, and
how people are not. Where is the street that has most unsatisfied
comments? What they are not satifised with?

## CAUTIONS

<b style="color:salmon">NOTE 1 </b> 

This is a **mock-up** dashboard. The content **does not reflect the
actual sentiments** of the tweets. This is because I **used up the
credits** to send the queries to the sentiment-analysis API by
MonkeyLearn during the code-development phase. Currently App is
working with a dummy sentiment-analysis function that returns random
values.

<b style="color:salmon">NOTE 2 </b> 

If you are using **safari** browser, and do not see anything on the
browser when you open the URL of the App, you should **update** your
safari to the newest version. I had this problem myself.


<b style="color:salmon">NOTE 3 </b> 

As you see below, the App uses a number of APIs. The **API tokens are
stored locally** in the script `config.py`, and not given in the
GitHub repo. You can still obtain the API keys yourself by creating
your own account at each service, and plug them in the code.

-----------------------------------------------------------------
## What the App does

1. Collect tweets that

   - contain the word 'bus', and 
   - are sent from 20 km from MarienPlatz. 

   Twitter API was used.

   - The majority of the tweets about a bus are from 'MVVticker'
     reporting delays, accidents, and coming back to the
     normality. Tweets by 'MVVticker' are removed.


2. Convert German texts to English, using DeepL API, for the sentiment
   analysis.


3. Sentiments and keywords are extracted from the texts. MonkeyLearn
   API was used.

   - The sentiments are weighted by the 'confidence' (that comes with
     the sentiment analysis), and are averaged to the overall
     sentiment. The neutral sentiments are removed (weighted to zero).


4. Use `plotly` (for the diagrams), `foilum` (map), and `wordcloud`
   (wordcloud) to create the visualizations.

5. Put the visualizations into a dynamic webpage using `streamlit`.

6. Deploy the App on `Heroku`. 

-----------------------------------------------------------------
## Summary of Tech Stack

| For                           | Tech used                   | 
|-------------------------------|-----------------------------|
|Coding                         | Python                      |
|Data source                    | Twitter                     |   
|Translation                    | DeepL                       |   
|Sentiment analysis             | MonkeyLearn                 |   
|Keyword extraction             | MonkeyLearn                 |   
|Visualization                  | plotly, folium, wordcloud   |
|Webpage                        | streamlit                   |   
|Deployment                     | Heroku                      |   

-----------------------------------------------------------------
## Conculsions

The purpose of the project is to see if I could answer three questions
below.

1. If people in Munich are satisfied with their bus service?
2. What people are satisfied / unsatisfied with?
3. Which part of the city, people are most unsatisfied?

There are a few difficulties to answer the questions. 

1. The number of accessible tweets are too small. We have about 10-30
   tweets per day on the bus, and not all of them are useful to
   extract clear sentiments. Much more (not clear how much at the
   moment) would be necessary to be statistically significant on the
   keyword collection (=wordcloud).
   

2. (almost) None of the tweets come with geo tags.

   In the map, there are two big circles near
      - Maxvorstadt
      - Pasing

  The locations only reflect that the tweets come from Munich, as the
  coordinate (11.57540E, 48.13714N; near Maxvorstadt) is automatically
  assigned when the tweet-authors open their geo-location as
  'Munich'. The granularity of the localization is not enough.

-----------------------------------------------------------------
## Actions to take

There are a few things that may change the situations above.

1. Switch to the commercial plan to get an access to the whole
   collection of the tweets, instead of the sampled collection.

2. Switch to the commercial plan to get an access to the historical
   tweets older than 7 days.

3. Perform the same analysis in the cities similar to Munich, and
   combine the results, to understand the overall characteristics in a
   city.

   - Nuremberg
   - Augsburg
   - Heidelberg
   - Berlin
   - London
   - Kyoto

4. Run a campaign to persuade people to tweet with their geo tags
   open.
   - synchronized with the night of museums / shopping?
   - continuous campaign?

5. Switch the data source to Google Analytics.
   (and go for a cloud solution in passing, including Google's NLP
   services to replace MonkeyLearn API)


6. Switch to commercial plans

   The App strictly uses free services only. The following are the
   restrictions, and what would be different in the commercial
   services.

   [Twitter]

   - The tweets that the App can search is a 'sample' (a part) of the
     whole tweets, not the entire collection. It is not clear how much
     is the fraction of this sample with respect to the whole tweets.

   - The tweets that the App can search are 7-days old or newer. The
     App cannot look for the historical records older than one week.
     The restriction is lifted in the commercial plan.

   - More than half f the collected tweets are truncated. It is not
     clear what are the criterions for the truncation. The commercial
     plan will provide the non-truncated tweets.

   - Geo tags are extremely rare in the tweets collected in the free
     plan. It is not clear how this would differ in the commercial
     plan.
   

   [MonkeyLearn]
   
   - The number of queries are restricted up to 1000 per month, i.e.
     about 30 per day. This is barely enough, or slightly less than
     the number of the tweets expected for a day. The restriction is
     lifted in the commercial plan.

   [DeepL]

   - The number of queries are restricted up to 500,000 characters
     per month. The restriction is lifted in the commercial plan,
     but the queries are charged by the amount. 


-----------------------------------------------------------------
# END
