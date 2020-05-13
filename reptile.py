import requests
from bs4 import BeautifulSoup
import bs4
import os
from time import sleep

url_list = []
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}


def url_all():
    for page in range(1, 14):
        url = 'https://www.metoffice.gov.uk/about-us/press-office/news/weather-and-climate?r82_r1_r5_r1:page=' + str(
            page)
        url_list.append(url)


def essay_url():  # 找到所有文章地址
    blog_urls = []
    for url in url_list:
        html = requests.get(url, headers=headers)
        html.encoding = html.apparent_encoding
        soup = BeautifulSoup(html.text, 'html.parser')
        for h2 in soup.find_all('h2'):
            try:
                blog_url = (h2('a')[0]['href'])
                blog_urls.append(blog_url)
            except:
                pass
    return blog_urls


def save_path():
    s_path = 'D:/blog/'
    if not os.path.isdir(s_path):
        os.mkdir(s_path)
    else:
        pass
    return s_path


def save_essay(blog_urls, s_path):  # 找到所有文章标题，文章内容。
    blogname = 0
    for url in blog_urls:
        url = 'https://www.metoffice.gov.uk' + url
        blog_html = requests.get(url, headers=headers)
        blog_html.encoding = blog_html.apparent_encoding
        soup = BeautifulSoup(blog_html.text, 'html.parser')
        try:
            for title in soup.find('p', {'class': 'article-tagline'}):
                try:
                    file = open(str(blogname) + '.txt', 'w',encoding='UTF-8')
                    file.close()
                except BaseException as a:
                    print("aa")
                    print(a)

            for p in soup.find('div', {'class': 'article-body'}).children:
                if isinstance(p, bs4.element.Tag):
                    try:
                        file = open(s_path + str(blogname) + '.txt', 'a',encoding='UTF-8')
                        file.write(p.text)
                        file.close()
                    except BaseException as f:
                        print("bb")
                        print(f)
        except BaseException as b:
            print("cc")
            print(b)
        blogname += 1
    print('---------------所有页面遍历完成----------------')


sleep(10)
url_all()
save_essay(essay_url(), save_path())
