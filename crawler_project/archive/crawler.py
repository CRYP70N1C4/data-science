from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import requests, re,json
from bs4 import BeautifulSoup

# need money for ip proxy
class Crawler():
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')

    # block images
    prefs = {"profile.managed_default_content_settings.images": 2}
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument('blink-settings=imagesEnabled=false')

    chrome_options.add_argument(
        'user-agent= "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.89 Safari/537.36"')

    headers = {
        'user-agent': "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.89 Safari/537.36"}

    def url2file(self, url, path):
        driver = webdriver.Chrome(chrome_options=self.chrome_options)
        driver.get(url)
        with open(path, "w", encoding='utf-8') as f:
            f.write(driver.page_source)

    def get_articles(self, id):
        if not id:
            return
        url = 'https://weixin.sogou.com/weixin?type=1&s_from=input&query=' + id
        html = requests.get(url, headers=self.headers).text
        soup = BeautifulSoup(html, 'lxml')
        ele = soup.select_one('[uigs="account_name_0"]')
        if ele:
            url = ele['href']
            html = requests.get(url, headers=self.headers).text
            print(html)
            result = re.search('msgList = (.*); +seajs.use', html.replace('\n',''))
            print(result.group(1))
            for e in json.loads(result.group(1))['list']:
                print(e)


Crawler().get_articles("顾子明")