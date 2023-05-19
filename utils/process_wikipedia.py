import sqlite3
import urllib
from typing import Optional


def make_wikilinks_consistent(url: str) -> str:
    """
    Make the wiki links consistent by performing the following operations:

    #. Convert the URL to lowercase.
    #. Unquote the URL to decode any percent-encoded characters.
    #. Replace underscores with spaces if they exist in the unquoted URL.
    #. Remove any fragment identifier (text after the '#' symbol) if present.
    #. Quote the modified URL to encode any special characters.

    Arguments:
        url (str): The URL to make consistent.

    Returns:
        str: The modified and quoted URL.

    Example:
        >>> make_wikilinks_consistent("https://en.wikipedia.org/wiki/Python_(programming_language)#Overview")
        'https://en.wikipedia.org/wiki/Python (programming language)'
        >>> make_wikilinks_consistent("https://en.wikipedia.org/wiki/Data_science")
        'https://en.wikipedia.org/wiki/Data_science'
    """
    url = url.lower()
    unquote = urllib.parse.unquote(url)
    if "_" in unquote:
        unquote = unquote.replace("_", " ")
    if "#" in unquote:
        unquote = unquote.split("#")[0]
    quote = urllib.parse.quote(unquote)
    return quote


def make_wikipedia2wikidata_consisent(entity: str) -> str:
    """
    Make the Wikipedia entity consistent with Wikidata by performing the
    following operations:

    #. Make the wiki links consistent using the 'make_wikilinks_consistent'
       function.
    #. Unquote the modified and quoted URL to decode any percent-encoded
       characters.
    #. Replace spaces with underscores in the unquoted URL.

    Arguments:
        entity (str): The Wikipedia entity to make consistent.

    Returns:
        str: The modified and consistent Wikipedia entity in Wikidata format.

    Example:
        >>> make_wikipedia2wikidata_consistent("Python (programming language)")
        'Python_(programming_language)'
        >>> make_wikipedia2wikidata_consistent("Data science")
        'Data_science'
    """
    quoted_entity = make_wikilinks_consistent(entity)
    underscored = urllib.parse.unquote(quoted_entity).replace(" ", "_")
    return underscored


def title_to_id(
    page_title: str, path_to_db: str, lower: Optional[bool] = False
) -> Optional[str]:
    """
    Given a Wikipedia page title, returns the corresponding Wikidata ID.
    The page title is the last part of a Wikipedia url **unescaped** and spaces
    replaced by underscores , e.g. for `https://en.wikipedia.org/wiki/Fermat%27s_Last_Theorem`,
    the title would be `Fermat's_Last_Theorem`.

    Arguments:
        path_to_db: The path to the wikidata2wikipedia db
        page_title: The page title of the Wikipedia entry, e.g. ``Manatee``.

    Returns:
        str, optional:
            If a mapping could be found for ``wiki_page_title``, then returns
            the mapping, otherwise None.

    Note:
        This function is adapted from https://github.com/jcklie/wikimapper.
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
