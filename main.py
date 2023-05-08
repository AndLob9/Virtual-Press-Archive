###NEWS SCRAPPER###

!pip install openai --quiet
!pip install beautifulsoup4 --quiet
!pip install requests --quiet
!pip install lxml --quiet
!pip install requests_html --quiet
!pip install pygooglenews --upgrade --quiet
!pip install newspaper3k --quiet

import pandas as pd
from pygooglenews import GoogleNews
from newspaper import Article
from newspaper import Config
import nltk
nltk.download('punkt')

USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Geck/20100101 Firefox/78.0'
config = Config()
config.browser_user_agent = USER_AGENT
config.request_timeout = 10 

gn = GoogleNews(lang = 'pl', country = 'PL')

def get_titles(search):
    
    summary = []
    stories = []
    title_list= []
    text_list = []
    
    search = gn.search(search, when = '1y') #when: duration; 2y = 2 years, 6m = 6 months, etc. 
    newsitem = search['entries']
    
    titles = newsitem
    for i in titles:
        article_t= {'title': i.title,"published":i.published}
        title_list.append(article_t)

    for item in newsitem:
        story = item.link
        url = item.link
        article = Article(url, config=config)

        article.download()
        article.parse()
        authors = ", ".join(author for author in article.authors) 
        title = article.title
        date = article.publish_date
        text = article.text
        url = article.url

        article.nlp()
        keywords = article.keywords
        keywords.sort()
        print("\n")
        print(item.link)
        print(f"📌 Keywords: {keywords}")
        print(f"📰 Summary: {article.summary}")
        stories.append(story)
        summary.append(article.summary)
        text_list.append(article.text)

    return title_list, stories, summary, text_list
    
    df = pd.DataFrame(data).transpose()
df.columns = ['title', 'link', 'summary', 'text']
df['title'] = df['title'].apply(lambda x: x['title'])
df.shape

###TEXT SUMMARIZER###
import os
from time import time,sleep
import textwrap
import openai
import re

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

openai.api_key = 'PUT YOUR KEY HERE'

def save_file(content, filepath):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def gpt3_completion(prompt, engine='text-davinci-002', temp=0.6, top_p=1.0, tokens=2000, freq_pen=0.25, pres_pen=0.0, stop=['<<END>>']):
    max_retry = 5
    retry = 0
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            with open('/content/drive/MyDrive/Badania korpusowe/Projekt 2/gpt3_logs/gpt3_logs%s' % filename, 'w') as outfile:
                outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            return text
        
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


if __name__ == '__main__':
    alltext = open_file('/content/summary.txt')
    chunks = textwrap.wrap(alltext, 2000)
    result = list()
    count = 0
    for chunk in chunks:
        count = count + 1
        prompt = open_file('/content/drive/MyDrive/Badania korpusowe/Projekt 2/prompt.txt').replace('<<SUMMARY>>', chunk)
        prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
        summary = gpt3_completion(prompt)
        print('\n\n\n', count, 'of', len(chunks), ' - ', summary)
        result.append(summary)
    save_file('\n\n'.join(result), '/content/drive/MyDrive/Badania korpusowe/Projekt 2/output_%s.txt' % time())
    
###TOKEN CLASSIFICATION###
!pip install sentencepiece
!pip install transformers

from transformers import pipeline

model_path = "xlm-roberta-large-finetuned-conll03-english"

token_classifier = pipeline("token-classification", model=model_path, tokenizer=model_path)

def classify_reviews(text):
    classified_text = token_classifier(text)
    return classified_text

text_to_analize = open_file('/content/drive/MyDrive/Badania korpusowe/Projekt 2/output_1675179516.4668875.txt')

analized_txt = classify_reviews(text_to_analize)

def combine_entity_words(analized_txt):
    combined_entities = []
    entity_words = []
    entity = None
    for i, item in enumerate(analized_txt):
        if i == 0 or item['entity'] != analized_txt[i-1]['entity']:
            if entity_words:
                combined_entities.append((entity, "".join(entity_words)))
                entity_words = []
            entity = item['entity']
        entity_words.append(item['word'].replace("▁", " "))
    combined_entities.append((entity, "".join(entity_words)))
    return combined_entities

result = combine_entity_words(analized_txt)
print(result)

