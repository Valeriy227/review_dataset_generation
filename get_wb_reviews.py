import time
import requests
import json

session = requests.session()

start_headers = {
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3',
    'Connection': 'keep-alive',
    'Host': 'www.wildberries.ru',
    'Referer': 'https://www.wildberries.ru/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'no-cors',
    'Sec-Fetch-Site': 'same-site',
    'TE': 'trailers',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:95.0) Gecko/20100101 Firefox/95.0',
}


def get_keywords():
    categories = ['zhenshchinam', 'obuv', 'detyam', 'muzhchinam', 'dom-i-dacha', 'krasota', 'aksessuary',
                  'elektronika', 'igrushki', 'dom/mebel', 'aksessuary/tovary-dlya-vzroslyh', 'pitanie',
                  'bytovaya-tehnika', 'tovary-dlya-zhivotnyh', 'aksessuary/avtotovary', 'detyam/shkola',
                  'knigi', 'premium-store', 'yuvelirnye-ukrasheniya', 'dom-i-dacha/instrumenty', 'dachniy-sezon',
                  'dom-i-dacha/zdorove', 'knigi-i-diski/kantstovary']
    headers = start_headers
    headers['Host'] = 'www.wildberries.ru'
    headers['Origin'] = 'https://www.wildberries.ru'
    headers['Sec-Fetch-Mode'] = 'cors'
    headers['Sec-Fetch-Site'] = 'same-origin'
    headers['TE'] = 'trailers'

    all_data = {}
    for category in categories:
        headers['Referer'] = f'https://www.wildberries.ru/catalog/{category}'
        session.headers = headers
        link = f'https://www.wildberries.ru/webapi/catalogdata/{category}'
        resp = session.post(link)
        data = json.loads(resp.text)
        all_data[category] = []
        for item in data['value']['data']['model']['catalogMenu']:
            all_data[category].append(item['entity']['name'])
            print(item['entity']['name'])
    j = json.dumps(all_data, ensure_ascii=False).encode('utf8')
    with open('keywords.json', 'w') as file:
        file.write(j.decode('utf8'))
    return all_data


def parse_cookies(cookies):
    cookies = cookies.split("|split|")
    cookies = list(map(lambda x: x.split(), cookies))[:-1]
    return cookies


def get_reviews(imtid, vcode):
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3',
        'Connection': 'keep-alive',
        'DNT': '1',
        'Upgrade-Insecure-Requests': '1',
        'Host': 'feedbacks.wildberries.ru',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'TE': 'trailers',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:106.0) Gecko/20100101 Firefox/106.0',
    }
    session.headers = headers
    reviews = []
    reviews_link = f'https://feedbacks.wildberries.ru/api/v1/summary/full?imtId={imtid}&skip=0&take=10'
    try:
        resp = session.get(reviews_link)
    except Exception:
        return reviews
    raw_reviews = json.loads(resp.text)
    if 'feedbacks' in raw_reviews and raw_reviews['feedbacks'] is not None:
        raw_reviews = raw_reviews['feedbacks'][:100]
        for rr in raw_reviews:
            review = {
                'reviewerName': rr['wbUserDetails']['name'] if 'wbUserDetails' in rr and 'name' in rr[
                    'wbUserDetails'] else None,
                'text': rr['text'],
                'pros': rr['pros'] if 'pros' in rr else None,
                'cons': rr['cons'] if 'cons' in rr else None,
                'isObscene': rr['isObscene'],
                'matchingSize': rr['matchingSize'],
                'mark': rr['productValuation'],
                'color': rr['color'],
                'size': rr['size']
            }
            reviews.append(review)
    return reviews


def get_items_by_keyword():
    all_cards = []
    cards = []
    skipped, all_skipped, done, all_done = 0, 0, 0, 0

    start_link = 'https://search.wb.ru/exactmatch/ru/male/v4/search?appType=1&couponsGeo=12,3,18,15,21&curr=rub&' \
                 'dest=-1029256,-102269,-1282181,-950664&emp=0&lang=ru&locale=ru&pricemarginCoeff=1.0&query=[KEYWORD]&' \
                 'reg=1&regions=80,68,64,83,4,38,33,70,82,69,86,75,30,40,48,1,22,66,31,71&resultset=catalog&' \
                 'sort=popular&spp=25&sppFixGeo=4&suppressSpellcheck=false'

    headers = start_headers
    headers['Host'] = 'www.wildberries.ru'
    headers['Origin'] = 'https://www.wildberries.ru'
    headers['Referer'] = 'https://www.wildberries.ru/'
    headers['Sec-Fetch-Mode'] = 'cors'
    headers['Sec-Fetch-Site'] = 'cross-site'
    headers['TE'] = 'trailers'
    session.headers = headers

    keywords = json.loads(open('keywords.json', 'r').read())
    for key in keywords.keys():
        for kw in keywords[key]:
            print(kw)
            link = start_link.replace('[KEYWORD]', kw)
            resp = session.get(link)
            items_by_kw = json.loads(resp.text)

            card_start_link = 'https://basket-[NUM].wb.ru/vol[3VC]/part[5VC]/[FullVC]/info/ru/card.json'
            for item in items_by_kw['data']['products']:
                to_skip = False
                vc = str(item['id'])
                card_link = card_start_link.replace('[3VC]', vc[:3]).replace('[5VC]', vc[:5]).replace('[FullVC]', vc)
                nums = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
                for num in nums:
                    if num == '09':
                        to_skip = True
                    next_link = card_link.replace('[NUM]', num)
                    card_resp = session.get(next_link)
                    # print(card_resp)
                    if str(card_resp).find('200') != -1:
                        break
                if to_skip:
                    skipped += 1
                    continue
                done += 1
                # print(next_link)
                # print(card_resp)
                # print(card_resp.text)
                card = json.loads(card_resp.text)
                reviews = get_reviews(card['imt_id'], card['nm_id'])
                cards.append({'category': key,
                              'keyword': kw,
                              'kinds': card['kinds'] if 'kinds' in card else None,
                              'name': card['imt_name'] if 'imt_name' in card else card['subj_name'],
                              'brand': card['selling']['brand_name'] if 'selling' in card and 'brand_name' in card[
                                  'selling'] else None,
                              'description': card['description'] if 'description' in card else None,
                              'colors': card['nm_colors_names'] if 'nm_colors_names' in card else None,
                              'all_colors': card['colors'] if 'colors' in card else None,
                              'has_sizes': True if 'sizes_table' in card else False,
                              'reviews': reviews})
        print(f'done: {done}')
        print(f'skipped: {skipped}')
        j = json.dumps(cards, ensure_ascii=False).encode('utf8')
        with open(f'{done}_{key.replace("/", "&")}_cards.json', 'w') as file:
            file.write(j.decode('utf8'))
        all_cards += cards
        all_skipped += skipped
        all_done += done
        cards.clear()
        done = 0
        skipped = 0
    print(f'all done: {all_done}')
    print(f'all skipped: {all_skipped}')
    j = json.dumps(all_cards, ensure_ascii=False).encode('utf8')
    with open(f'{all_done}_all_cards.json', 'w') as file:
        file.write(j.decode('utf8'))


# get_keywords()
# get_items_by_keyword()

with open('27181_all_cards.json', 'r') as file:
    data = json.loads(file.read())
    reviews_count, description_count, both = 0, 0, 0
    for elem in data:
        x, y = False, False
        if len(elem['reviews']) != 0:
            x = True
            reviews_count += 1
        if elem['description'] is not None:
            y = True
            description_count += 1
        if x and y:
            both += 1
    total = len(data)
    print('{:.1f}% товаров имеют отзывы'.format(reviews_count / total * 100))
    print('{:.1f}% товаров имеют описание'.format(description_count / total * 100))
    print('{:.1f}% товаров имеют и описание и отзывы'.format(both / total * 100))
