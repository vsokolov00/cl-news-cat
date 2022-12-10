import bs4
import requests
from tqdm import tqdm
import pandas as pd
from urllib.parse import urlparse


def sub_path_to_url(domain, sub_path):
    return "https://" + urlparse(domain).netloc + sub_path

html_pages = ["https://ruralindiaonline.org/en/categories/"]
    
data = []
for link in html_pages:
    page = requests.get(link)
    soup_cat = bs4.BeautifulSoup(page.text, 'html.parser')
    categories = soup_cat.find("div", class_="story-grid")
    categories = categories.find_all("div", class_="grid")
    categories = [sub_path_to_url(link, c.find("a")['href']) for c in categories]

    for link in tqdm(categories, unit="category"):
        first_page = True
        page_to_get = link
        while True: 
            page = requests.get(page_to_get)
            soup = bs4.BeautifulSoup(page.text, 'html.parser')
            try:
                lis = soup.find("body").find("ul", class_="")
                articles = lis.find_all("li", class_="item")
            except AttributeError:
                break
            for article in articles:
                article_data = {}
                title_h3 = article.find("h3")
                title = title_h3.text.strip()
                href = title_h3.find("a")['href']
                en_article_path = sub_path_to_url(link, href)

                en_article = requests.get(en_article_path)
                sub_soup = bs4.BeautifulSoup(en_article.text, 'html.parser')
                topics = sub_soup.find_all("span", class_="tag-space")
                topics = ','.join([c.find("a").text.strip() for c in topics])

                try:
                    ul_lang = article.find("ul", class_="dropdown-menu")
                    langs = ul_lang.find_all("li")
                    translations = {k: v for k, v in [(c.find("a").text, sub_path_to_url(link, c.find("a")['href'])) for c in langs]}
                except AttributeError:
                    translations = {"English": en_article_path}

                article_data = {"title": title, "topics": topics}
            
                data.append(article_data | translations)
            
            nav = soup.find_all("li", class_="page-nav")
            if nav == []:
                break

            if len(nav) == 2:
                if not first_page:
                    break
                first_page = False
                next = nav[0].find("a")['href']
            elif len(nav) == 4:
                first_page = False
                next = nav[2].find("a")['href']
            page_to_get = page_to_get.split('?')[0] + next

    df = pd.DataFrame(data)
    df.to_csv(f"./{urlparse(link).netloc}.csv", sep=";", index=False)
    data = []
    break

        