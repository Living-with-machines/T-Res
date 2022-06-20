from REL.wikipedia import Wikipedia
from REL.wikipedia_yago_freq import WikipediaYagoFreq

wiki_version = "lwm_rel_filtered"
base_url = "/resources/wikipedia/rel_db/"
wikipedia = Wikipedia(base_url, wiki_version)

wiki_yago_freq = WikipediaYagoFreq(base_url, wiki_version, wikipedia)
wiki_yago_freq.compute_wiki()
wiki_yago_freq.compute_custom()
wiki_yago_freq.store()
