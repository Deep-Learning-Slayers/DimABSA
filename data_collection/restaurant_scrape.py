import time
import random
import os
import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup

TARGET_PER_STAR = 225
MAX_PER_RESTAURANT_PER_STAR = 15 
OUTPUT_FILE = "data/data_restaurants.csv"

CHAIN_URLS = [
    "https://www.trustpilot.com/review/www.mcdonalds.com",
    "https://www.trustpilot.com/review/www.burgerking.com",
    "https://www.trustpilot.com/review/www.kfc.com",
    "https://www.trustpilot.com/review/www.subway.com",
    "https://www.trustpilot.com/review/www.pizzahut.com",
    "https://www.trustpilot.com/review/www.dominos.com",
    "https://www.trustpilot.com/review/www.popeyes.com",
    "https://www.trustpilot.com/review/www.tacobell.com",
    "https://www.trustpilot.com/review/www.wendys.com",
    "https://www.trustpilot.com/review/www.dunkindonuts.com",
    "https://www.trustpilot.com/review/chipotle.com",
    "https://www.trustpilot.com/review/www.papajohns.com",
    "https://www.trustpilot.com/review/www.fiveguys.com",
    "https://www.trustpilot.com/review/www.chick-fil-a.com",
    "https://www.trustpilot.com/review/www.pandaexpress.com"
]

def get_driver():
    options = uc.ChromeOptions()
    return uc.Chrome(options=options)

def get_chain_links():
    """Chain restoranları karıştırılmış sırayla döndür."""
    print("[1/2] Chain restoran listesi hazırlanıyor...")
    links = CHAIN_URLS.copy()
    random.shuffle(links)
    print(f"[BİLGİ] Toplam {len(links)} chain restoran hedeflendi.")
    return links

def scrape_balanced_reviews(driver, profile_links):
    print("[2/2] Yorumlar toplanıyor...")
    
    # Veri Deposu
    star_buffer = {i: [] for i in range(1, 6)}
    seen_texts = set()
    restaurant_star_count = {}  # Her restoran için yıldız başına sayaç

    for idx, link in enumerate(profile_links):
        if all(len(star_buffer[s]) >= TARGET_PER_STAR for s in star_buffer):
            print("\n✅ TÜM HEDEFLERE ULAŞILDI! Veri toplama bitiyor.")
            break

        # Link isminden restoran adını çıkar
        restaurant_name = link.split('/')[-1].replace('www.', '').replace('.com', '').replace('-', ' ').title()
        print(f"\n[{idx+1}/{len(profile_links)}] Geziliyor: {restaurant_name}")
        
        # Bu restoran için sayaçları başlat
        if link not in restaurant_star_count:
            restaurant_star_count[link] = {i: 0 for i in range(1, 6)}

        for target_star in range(1, 6):
            if len(star_buffer[target_star]) >= TARGET_PER_STAR:
                continue
            
            # Bu restorandan bu yıldız için limit doldu mu?
            if restaurant_star_count[link][target_star] >= MAX_PER_RESTAURANT_PER_STAR:
                continue

            url = f"{link}?stars={target_star}&sort=recency"
            
            driver.get(url)
            time.sleep(random.uniform(2, 4))
            
            soup = BeautifulSoup(driver.page_source, "html.parser")
            
            if "No reviews" in soup.get_text() or "There are no reviews" in soup.get_text():
                continue

            reviews = soup.find_all('article')
            added_count = 0
            
            for review in reviews:
                try:
                    star_img = review.find('img', alt=lambda x: x and "Rated" in x)
                    if not star_img: continue
                    rating = int(star_img['alt'].split()[1]) 
                    
                    content = review.find('p', attrs={'data-service-review-text-typography': True})
                    
                    if not content: continue
                    full_text = content.get_text(strip=True)
                    
                    if rating != target_star: continue
                    if len(full_text) < 20: continue
                    if len(full_text) > 1200: continue
                    if full_text in seen_texts: continue
                    if len(star_buffer[rating]) >= TARGET_PER_STAR: break
                    if restaurant_star_count[link][rating] >= MAX_PER_RESTAURANT_PER_STAR: break

                    seen_texts.add(full_text)

                    star_buffer[rating].append({
                        "review_text": full_text,
                        "star_rating": rating,
                        "domain": "restaurant",
                        "restaurant_name": restaurant_name
                    })
                    restaurant_star_count[link][rating] += 1
                    added_count += 1
                    
                except Exception:
                    continue
            
            # Durum Çubuğu
            status = " | ".join(f"{k}★:{len(v)}" for k, v in star_buffer.items())
            if added_count > 0:
                print(f"   -> {target_star} Yıldız: +{added_count} adet eklendi. [{status}]")

    return star_buffer

if __name__ == "__main__":
    driver = get_driver()
    
    try:
        chain_links = get_chain_links()
        
        data_map = scrape_balanced_reviews(driver, chain_links)
        
        flat_data = []
        for s in data_map:
            flat_data.extend(data_map[s])
            
        df = pd.DataFrame(flat_data)
        os.makedirs(os.path.dirname(OUTPUT_FILE) if os.path.dirname(OUTPUT_FILE) else ".", exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        
        print(f"\n[BAŞARILI] İşlem tamamlandı. Dosya: {OUTPUT_FILE}")
        if not df.empty:
            print("\nVeri Dağılımı:")
            print(df['star_rating'].value_counts().sort_index())
            print("\nRestoran Dağılımı:")
            print(df['restaurant_name'].value_counts())
            
    except Exception as e:
        print(f"Hata: {e}")
    finally:
        driver.quit()