from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')

# block images
prefs = {"profile.managed_default_content_settings.images": 2}
chrome_options.add_experimental_option("prefs", prefs)
chrome_options.add_argument('blink-settings=imagesEnabled=false')

chrome_options.add_argument(
    'user-agent= "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.89 Safari/537.36"')

driver = webdriver.Chrome(chrome_options=chrome_options)
driver.get("https://ssl.zc.qq.com/v3/index-chs.html")
driver.find_element_by_id('nickname').send_keys('name384383')
driver.find_element_by_id('password').send_keys("pass123457")
driver.find_element_by_id('phone').send_keys("123456789")
time.sleep(1)
driver.find_element_by_id('send-sms').click()
