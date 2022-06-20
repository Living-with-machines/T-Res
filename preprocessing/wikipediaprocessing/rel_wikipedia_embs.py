from REL.db.generic import GenericLookup

wiki_version = "lwm_rel_filtered"
base_url = "/resources/wikipedia/rel_db/"

save_dir = "{}/{}/generated/".format(base_url, wiki_version)
db_name = "entity_word_embedding"
embedding_file = "./enwiki_w2v_model"

# Embedding load.
emb = GenericLookup(db_name, save_dir=save_dir, table_name='embeddings')
emb.load_word2emb(embedding_file, batch_size=5000, reset=True)