from bs4 import BeautifulSoup
import requests
#%%
def extract_reviews(soup):
    pos_reviews = []
    neg_reviews = []
    review_blocks = soup.find('div', class_='sp-reviews')

    if review_blocks is not None:
        review_blocks = review_blocks.find_all('div', class_='sp-review')

        for review_block in review_blocks:

            rating = int(
                review_block.find('div', class_='sp-review-rating').find('div', class_='sp-review-rating-label').find(
                    'span', class_=
                    'sp-review-rating-value').text)
            body_content = review_block.find('div', class_='sp-review-content').find('div',
                                                                                     class_='sp-review-pros-cons-body').find(
                'div',
                class_='sp-review-body-content')

            if body_content is not None:
                review_text = body_content.text.strip()
                if rating == 4 or rating == 5:
                    pos_reviews.append(review_text)
                else:
                    neg_reviews.append(review_text)

    return pos_reviews, neg_reviews

def append_reviews_to_all(pos_reviews, neg_reviews):
    for pos_rev in pos_reviews:
        all_pos_reviews.append(pos_rev)
    for neg_rev in neg_reviews:
        all_neg_reviews.append(neg_rev)
#%%
catalog_page1_url = 'https://goods.ru/catalog/smartfony/'

def catalog_pages_urls(catalog_page1_url):
    page = requests.get(catalog_page1_url)
    soup = BeautifulSoup(page.text, 'html.parser')

    full_pagination_list = soup.find('ul', class_='pagination__list').find_all('li', class_='full-pagination__item')

    hrefs = []
    for full_pagination_item in full_pagination_list:
        atag = full_pagination_item.find('a', href=True)
        if atag is not None:
            hrefs.append('https:' + atag['href'])

    return hrefs

catalog_hrefs = catalog_pages_urls(catalog_page1_url)

additional_hrefs =['https://goods.ru/catalog/smartfony/', 'https://goods.ru/catalog/smartfony/page-2/',
                   'https://goods.ru/catalog/smartfony/page-3/']

for i, href in enumerate(additional_hrefs):
    catalog_hrefs.insert(i, href)
print('catalog links successfully initialized')
#%%
#will work if out of stock or no reviews
def reviews_urls_from_catalog_page(catalog_page_url):
    catalog_page_phones_reviews_urls = []

    catalog_page = requests.get(catalog_page_url)
    soup = BeautifulSoup(catalog_page.text, 'html.parser')

    catalog_page_product_ids = soup.find('div', id='goodsBlock')

    if catalog_page_product_ids is not None:
        catalog_page_product_ids = catalog_page_product_ids.find_all('div', class_='prod-item')

        for catalog_page_produc_id in catalog_page_product_ids:
            atag = catalog_page_produc_id.find('a', class_='card-prod--reviews', href=True)
            if atag is not None:
                catalog_page_phones_reviews_urls.append('https://goods.ru' + atag['href'])

    return catalog_page_phones_reviews_urls
#%%
print(catalog_hrefs)
#%%
print(reviews_urls_from_catalog_page(catalog_page1_url))

#%%
all_pos_reviews = []
all_neg_reviews = []

for i, catalog_page_url in enumerate(catalog_hrefs):
    reviews_catalog_page_urls = reviews_urls_from_catalog_page(catalog_page_url)
    for review_url in reviews_catalog_page_urls:
        page = requests.get(review_url)
        soup = BeautifulSoup(page.text, 'html.parser')
        pos_reviews, neg_reviews = extract_reviews(soup)
        append_reviews_to_all(pos_reviews, neg_reviews)
        print('page from catalog {} parsed'.format(i))
        print()

print('revies successfully obtained')
#%%
print(len(all_pos_reviews))
print(len(all_neg_reviews))

#%%

import pandas as pd

data = pd.DataFrame(all_pos_reviews)
print(data)

data.to_csv('data/positive_reviews.csv')
#%%
data = pd.DataFrame(all_neg_reviews)
data.to_csv('data/negative_revews.csv')

