from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import time

# Setup driver (make sure chromedriver is installed)
driver = webdriver.Chrome()

base_url = "https://reviews.webmd.com/drugs/drugreview-lisinopril-prinivil-zestril-others"

all_reviews = []

for page in range(5):
    url = f"{base_url}?pageIndex={page}"
    print(f"Scraping page {page+1}...")

    driver.get(url)
    time.sleep(5)  # wait for JS to load

    reviews = driver.find_elements(By.CLASS_NAME, "review-details-holder")

    for r in reviews:
        try:
            review = r.find_element(By.CLASS_NAME, "description").text
        except:
            review = None

        try:
            condition = r.find_element(By.CLASS_NAME, "condition").text
        except:
            condition = None

        try:
            reviewer = r.find_element(By.CLASS_NAME, "reviewer-details").text
        except:
            reviewer = None

        all_reviews.append({
            "review": review,
            "condition": condition,
            "reviewer": reviewer
        })

driver.quit()

df = pd.DataFrame(all_reviews)
print(df.head())
df.to_csv("webmd_reviews.csv", index=False)