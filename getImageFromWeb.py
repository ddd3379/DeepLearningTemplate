from icrawler.builtin import BingImageCrawler

save_folder = "data/fruits"
keyword = "フルーツ"
max_num = 10 # max = 1000

crawler = BingImageCrawler(storage={"root_dir": save_folder})
crawler.crawl(keyword=keyword, max_num=max_num)