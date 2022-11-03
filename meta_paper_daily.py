import os
import random
import requests
import re
from bs4 import BeautifulSoup
import datetime
import json

KEYS = ['source free', "object detection"]
papers = {}
DateNow = datetime.date.today()
DateNow = str(DateNow)
DateNow = DateNow.replace('-', '.')
# 参考连接 https://zhuanlan.zhihu.com/p/425670267
# 转换日期为标准格式 https://blog.csdn.net/weixin_43751840/article/details/89947528
def get_paper_from_arxiv(key):
    query_key = key.replace(" ", "+")
    url = f"https://arxiv.org/search/cs?query={query_key}&searchtype=title&abstracts=show&order=-announced_date_first&size=25"
    res = requests.get(url)
    content = BeautifulSoup(res.text, 'html.parser')
    papers[key] = {}
    # 正则匹配http
    pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')  # 匹配模式
    for arxiv_result in content.find(name="ol").find_all(name='li'):
        paper_url = arxiv_result.find(attrs={'class': 'list-title'}).a['href']
        abstract = arxiv_result.find(class_="abstract-full")  # 只返回当前节点的text不包括子节点
        code_url = re.search(pattern, abstract.text)
        code_url = code_url.group() if code_url else "-"
        if abstract.a: abstract.a.extract()  # 去除a标签里的额外文本
        abstract = abstract.text.strip()
        title = arxiv_result.find(class_='title')  # class是关键字所以要加下划线_
        title = title.text.strip()  # replace("<span class=\"search-hit mathjax\">Source</span>","").replace("</span>")text直接取出来了
        author = [au.strip() for au in arxiv_result.find(class_='authors').text.replace("\n", "").split(",")]
        #author = ",".join(author)
        author = author[0].replace("Authors:","") + "et.al" # 只取一作
        submit_date = [item.text.replace("\n", "").replace("  ", "") for item in arxiv_result.select("p.is-size-7")]
        time_format = datetime.datetime.strptime(submit_date[0].split(";")[0], 'Submitted %d %B, %Y')
        format_date = f"{time_format.year}-{time_format.month}-{time_format.day}"
        if len(submit_date) > 1:
            comments = submit_date[1]
            # 有时代码链接在commnets里
            if code_url == "-":
                code_url = re.search(pattern, comments)
                code_url = code_url.group() if code_url else "-"
            comments = comments.replace(";", ".").split(",")[0].split(".")[0].replace("Comments:", "").replace(
                "Accepted at ", "").replace("Accepted to ", "")
            if "pages" in comments:
                comments = "-"
        else:
            comments = "-"
        #json_res.append(result)
        #f.write("|Publish Date|Title|Authors|PDF|Code|Comments|\n" + "|---|---|---|---|\n")
        code_url = f"[code]({code_url})|" if code_url != "-" else "-|"
        if title not in papers:
            # 会议相关折叠
            comments = f"<details><summary>other</summary>{comments}</details>" if comments != "-" else "-"
            papers[key][title] = f"|**{format_date}**|**{title}**|**{author}**|[paper]({paper_url})|" + code_url + f"{comments}|\n"


def get_paper_from_google(key):
    headers = ['Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36 Edg/86.0.622.61',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.183 Safari/537.36 Edg/86.0.622.63',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.80 Safari/537.36 Edg/86.0.622.43',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36 Edg/87.0.664.66',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.146 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36 Edg/88.0.705.63',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36 Edg/88.0.705.50',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4346.0 Safari/537.36 Edg/89.0.731.0',]

    url = f"https://scholar.google.com/scholar?as_vis=0&q=allintitle:+{key}&hl=zh-CN&scisbd=1&as_sdt=0,5"
    res = requests.get(url, headers=random.choice(headers))
    content = BeautifulSoup(res.text, 'html.parser')
    body = content.find(id="gs_res_ccl_mid")
    # 谷歌学术爬虫
    for div in body.find_all(class_="gs_r gs_or gs_scl"):
        main = div.find(id=div.attrs['data-cid'])
        pdf_url = main.attrs['href']
        titie = main.text.strip()
        author_and_comment = div.find(class_="gs_a").text.strip()


def json_to_md(data):
    with open("papers.json", "r") as f:
        content = f.read()
        if not content:
            data = {}
        else:
            data = json.loads(content)

    md_filename = "README.md"

    # clean README.md if daily already exist else create it
    with open(md_filename, "w+") as f:
        f.write("<details>\n")
        f.write("  <summary>Table of Contents</summary>\n")
        f.write("  <ol>\n")
        for keyword in data.keys():
            day_content = data[keyword]
            if not day_content:
                continue
            kw = keyword.replace(" ", "-")
            f.write(f"    <li><a href=#{kw}>{keyword}</a></li>\n")
        f.write("  </ol>\n")
        f.write("</details>\n\n")
        #pass

    # write data into README.md
    with open(md_filename, "a+") as f:

        f.write("## Updated on " + DateNow + "\n\n")

        for keyword in data.keys():
            day_content = data[keyword]
            if not day_content:
                continue
            # the head of each part
            f.write(f"## {keyword}\n\n")
            f.write("|Date|Title|Authors|PDF|Code|Comments|\n")
            # "|---|---|---|---|---|---|\n"
            f.write("|:------|:---------------------|:-------|:-|:-|:---|\n")
            # sort papers by date
            # day_content = sort_papers(day_content)

            for _, v in day_content.items():
                if v is not None:
                    f.write(v)
            f.write(f"\n")


if __name__ == "__main__":
    for key in KEYS:
        get_paper_from_arxiv(key)
    json.dump(papers, open("papers.json", "w"))
    json_to_md(papers)
