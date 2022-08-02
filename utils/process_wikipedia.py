import sqlite3
import urllib
from collections import Counter
from typing import Optional

from bs4 import BeautifulSoup

### Processing pages ####


def make_wikilinks_consistent(url):
    url = url.lower()
    unquote = urllib.parse.unquote(url)
    if "_" in unquote:
        unquote = unquote.replace("_", " ")
    if "#" in unquote:
        unquote = unquote.split("#")[0]
    quote = urllib.parse.quote(unquote)
    return quote


def make_wikipedia2wikidata_consisent(entity):
    quoted_entity = make_wikilinks_consistent(entity)
    underscored = urllib.parse.unquote(quoted_entity).replace(" ", "_")
    return underscored


def clean_page(page):

    entities = [
        x
        for x in page.findAll("a")
        if (x.has_attr("href")) and ("https://" not in x["href"] or "http://" not in x["href"])
    ]
    box_mentions = Counter([x.text for x in entities])
    box_entities = Counter(
        [make_wikilinks_consistent(x["href"]) for x in entities]
    )  # this is the issue, some of them have the URL lowercased, like states%20of%20germany (for States%20of%20Germany)

    mentions_dict = {x: [] for x in box_mentions}
    for e in entities:
        mentions_dict[e.text].append(
            make_wikilinks_consistent(e["href"])
        )  # and here is where we add the (from time to time) partially lowercased URL to the mention dictionary

    mentions_dict = {x: Counter(y) for x, y in mentions_dict.items()}

    return [box_mentions, box_entities, mentions_dict]


def get_sections(page):
    page = page.text.strip().split("\n")
    sections = {"Main": {"order": 1, "content": []}}
    dict_key = "Main"
    ct = 1
    for line in page:
        if not "Section::::" in line:
            sections[dict_key]["content"].append(line)
        else:
            ct += 1
            dict_key = line.replace("Section::::", "")[:-1]
            sections[dict_key] = {"order": ct, "content": []}

    sections = {x: y for x, y in sections.items() if len(y["content"]) > 0}
    return sections


# we open each file and extract the title, the entity frequencies and the sections
def process_doc(filename):
    content = open(filename).read()
    content = BeautifulSoup(content, "html.parser").findAll("doc")
    pages = []
    for page in content:
        title = page["title"]
        title = make_wikilinks_consistent(title)
        sections = {"title": title, "sections": get_sections(page)}
        r = [title] + clean_page(page) + [sections]
        pages.append([r])
    return pages


### Aggregating statistics pages ####


def fill_dicts(
    res, mentions_freq, entity_freq, mention_overall_dict, entity_inlink_dict, entity_outlink_dict
):

    title, box_mentions, box_entities, mentions_dict = res[0], res[1], res[2], res[3]
    # to make it percent encoded as the other references to the same entity
    mentions_freq += box_mentions
    entity_freq += box_entities
    entity_outlink_dict[title] = box_entities

    for k, v in box_entities.items():
        if k in entity_inlink_dict:
            entity_inlink_dict[k].append(title)
        else:
            entity_inlink_dict[k] = [title]

    for k, v in mentions_dict.items():
        if k in mention_overall_dict:
            mention_overall_dict[k] += v
        else:
            mention_overall_dict[k] = Counter()
            mention_overall_dict[k] += v
    return (
        mentions_freq,
        entity_freq,
        mention_overall_dict,
        entity_inlink_dict,
        entity_outlink_dict,
    )


def title_to_id(path_to_db: str, page_title: str) -> Optional[str]:
    """This function is adapted from https://github.com/jcklie/wikimapper
    Given a Wikipedia page title, returns the corresponding Wikidata ID.
    The page title is the last part of a Wikipedia url **unescaped** and spaces
    replaced by underscores , e.g. for `https://en.wikipedia.org/wiki/Fermat%27s_Last_Theorem`,
    the title would be `Fermat's_Last_Theorem`.
    Args:
        path_to_db: The path to the wikidata2wikipedia db
        page_title: The page title of the Wikipedia entry, e.g. `Manatee`.
    Returns:
        Optional[str]: If a mapping could be found for `wiki_page_title`, then return
                        it, else return `None`.
    """

    with sqlite3.connect(path_to_db) as conn:
        c = conn.cursor()
        c.execute("SELECT wikidata_id FROM mapping WHERE wikipedia_title=?", (page_title,))
        result = c.fetchone()

    if result is not None and result[0] is not None:
        return result[0]
    else:
        return None
