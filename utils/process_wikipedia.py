import sqlite3
import urllib
from typing import Optional


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


def title_to_id(page_title, path_to_db, lower=False) -> Optional[str]:
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
        if lower == True:
            c.execute(
                "SELECT wikidata_id FROM mapping WHERE lower_wikipedia_title=?",
                (page_title,),
            )
        else:
            c.execute(
                "SELECT wikidata_id FROM mapping WHERE wikipedia_title=?",
                (page_title,),
            )
        result = c.fetchone()

    if result is not None and result[0] is not None:
        return result[0]
    else:
        return None
