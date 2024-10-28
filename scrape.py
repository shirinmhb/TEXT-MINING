from bs4 import BeautifulSoup
import requests

page = requests.get("https://en.wikipedia.org/wiki/Text_mining")
soup = BeautifulSoup(page.content, 'html.parser')
list(soup.children)
allPTags = soup.find_all('p')
textfile = open("TextMining.txt", "w")
for pTag in allPTags:
  textfile.write(pTag.get_text())
textfile.close


