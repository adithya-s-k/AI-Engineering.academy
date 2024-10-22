import sys
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time

def scrape_youtube_channel(url):
    # Determine the video type based on the URL
    if url.split('/')[-1] == 'videos':
        video_type = "videos"
    else:
        video_type = "shorts"

    # Create a new instance of the Edge driver
    driver = webdriver.Edge()

    # Open the YouTube channel page
    driver.get(url)
    time.sleep(5)
    last_height = driver.execute_script("return document.documentElement.scrollHeight")

    # Scroll to the bottom of the page to load all videos
    while True:
        # Scroll down to the bottom
        print("Scrolling down...")
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)

        # Check if the page height has increased
        driver.implicitly_wait(3)
        time.sleep(3)  # Adjust the wait time as needed

        new_height = driver.execute_script("return document.documentElement.scrollHeight")
        if new_height == last_height:
            break  # End of page reached
        last_height = new_height

    # Get the page source after scrolling
    page_source = driver.page_source

    # Close the webdriver
    driver.quit()

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(page_source, 'html.parser')

    # Find all links with the specified attributes
    if video_type == "shorts":
        video_links = soup.find_all('a', {'id': 'thumbnail', 'class': 'yt-simple-endpoint inline-block style-scope ytd-thumbnail'})
    elif video_type == "videos":
        video_links = soup.find_all('a', {'id': 'thumbnail', 'class': 'yt-simple-endpoint style-scope ytd-playlist-thumbnail'})

    # Extract and print the href attributes
    channel = url.split('/')[-2]

    with open(f'{channel}-{video_type}.txt', 'w') as f:
        for link in video_links:
            href = link.get('href')
            if href:
                f.write(f"https://www.youtube.com{href}\n")

if __name__ == "__main__":
    # Check if a command-line argument is provided
    if len(sys.argv) < 2:
        print("Usage: python youtube_channel_scraper.py <YouTube channel URL> Example: https://www.youtube.com/@Fireship/videos or https://www.youtube.com/@Fireship/shorts")
        sys.exit(1)

    # URL of the YouTube channel
    url = sys.argv[1]

    # Call the main function with the provided URL
    scrape_youtube_channel(url)