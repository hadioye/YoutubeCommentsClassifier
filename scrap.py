import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langdetect import detect
from bs4 import BeautifulSoup
import requests
import pandas as pd
import csv
import time


def web_driver():
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(options=options)
    return driver

driver = web_driver()

channels = ['https://www.youtube.com/@CNN/videos' , 'https://www.youtube.com/@PiersMorganUncensored/videos', 'https://www.youtube.com/@jakepaul/videos' , 'https://www.youtube.com/@benshapiro/videos']
users = []
comments = []
links = []

for channel in channels:
    driver.get(channel)

    time.sleep(2)
    WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.ID, 'thumbnail')))

    thumbnail = driver.find_elements(By.XPATH, "//a[@id='thumbnail']")

    for t in thumbnail:
        link = t.get_attribute('href')

        if link not in links:
            links.append(link)

    #first entry is always null, so remove it
    links.pop(0) 


for link in links:
    if link != None:
        driver.get(link)

        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
        time.sleep(3) 

        for _ in range(2): #scroll down to comments
            driver.find_element(By.CSS_SELECTOR ,'body').send_keys(Keys.PAGE_DOWN)
            time.sleep(2)

        all_comments = driver.find_elements(By.CSS_SELECTOR, 'ytd-comment-renderer')
            
        for info in all_comments:
            try:
                author_element = info.find_element(By.CSS_SELECTOR, '#author-text')
                content_element = info.find_element(By.CSS_SELECTOR, '#content-text')
        
                # If both elements are found, extract text
                users.append(author_element.text)
                comments.append(content_element.text)
        
            except:
                print("Element not found for this info")
                continue 

dic = {"Users" : users , "Comments" : comments}

print(len(users))
print(len(comments))

print(users)
print(comments)

df = pd.DataFrame(dic)
df.to_csv("data.csv")


df = df[df['Comments'].str.strip().astype(bool)]

# Resetting the index after removing rows
df.reset_index(drop=True, inplace=True)

df['IsEnglish'] = df['Comments'].apply(detect_language)
df = df[df['IsEnglish']]  # Keep only rows where 'IsEnglish' column is True

df.drop(columns=['IsEnglish'], inplace=True)
df.to_csv("data.csv")

print(df.head())
