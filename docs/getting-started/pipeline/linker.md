# Linker

The Linker takes as input the set of candidates selected by the Ranker and disambiguates them, selecting the best matching entity depending on the approach selected for disambiguation.

We provide two different strategies for disambiguation:

-   `mostpopular`: Unsupervised method, which, given a set of candidates for a given mention, returns as a prediction the candidate that is most popular in terms of inlink structure in Wikipedia.

-   `reldisamb`: Given a set of candidates, this approach uses the [REL re-implementation](https://github.com/informagi/REL/) of the [ment-norm algorithm](https://github.com/lephong/mulrel-nel) proposed by Le and Titov (2018) and partially based on Ganea and Hofmann (2017), and adapts it. To know more:

        Van Hulst, Johannes M., Faegheh Hasibi, Koen Dercksen, Krisztian Balog, and
        Arjen P. de Vries. "Rel: An entity linker standing on the shoulders of giants."
        In Proceedings of the 43rd International ACM SIGIR Conference on Research and
        Development in Information Retrieval, pp. 2197-2200. 2020.

        Le, Phong, and Ivan Titov. "Improving Entity Linking by Modeling Latent Relations
        between Mentions." In Proceedings of the 56th Annual Meeting of the Association
        for Computational Linguistics (Volume 1: Long Papers), pp. 1595-1604. 2018.

        Ganea, Octavian-Eugen, and Thomas Hofmann. "Deep Joint Entity Disambiguation
        with Local Neural Attention." In Proceedings of the 2017 Conference on
        Empirical Methods in Natural Language Processing, pp. 2619-2629. 2017.

**TODO:** insert Linker class diagram here.

## 1. Instantiate the Linker

### Most Popular Linker

To use the Linker with the `mostpopular` approach, instantiate it as follows:

```python
from t_res.geoparser import linking

linker = linking.MostPopularLinker(
    resources_path="resources/"
)
```

Description of the parameters:

-   `resources_path`: path to the resources directory.

Note that `resources_path` should contain the path to the directory where the resources are stored.

When using the `mostpopular` linking approach, the resources folder should at least contain the following resources:

    T-Res/
      └── resources/
          └── wikidata/
              ├── entity2class.txt
              ├── mentions_to_wikidata.json
              └── wikidata_gazetteer.csv

### By Distance Linker

To use the Linker with the `bydistance` approach, instantiate it as follows:

```python
from t_res.geoparser import linking

linker = linking.ByDistanceLinker(
    resources_path="resources/"
)
```

Description of the parameters:

-   `resources_path`: path to the resources directory.

Note that `resources_path` should contain the path to the directory where the resources are stored.

When using the `bydistance` linking approach, the resources folder should at least contain the following resources:

    T-Res/
      └── resources/
          └── wikidata/
              ├── entity2class.txt
              ├── mentions_to_wikidata.json
              └── wikidata_gazetteer.csv

### REL Disambiguation Linker

To use the Linker with the `reldisamb` approach, instantiate it as follows:

```python
from t_res.geoparser import linking

with sqlite3.connect("resources/rel_db/embeddings_database.db") as conn:
    cursor = conn.cursor()
    linker = linking.RelDisambLinker(
        resources_path="resources/",
        rel_params={
            "model_path": "resources/models/disambiguation/",
            "data_path": "experiments/outputs/data/lwm/",
            "training_split": "originalsplit",
            "db_embeddings": cursor,
            "with_publication": True,
            "without_microtoponyms": True,
            "do_test": False,
            "default_publname": "London",
            "default_publwqid": "Q84",
        },
        overwrite_training=False,
    )
```

Description of the parameters:

-   `method`: name of the method, in this case `reldisamb`.
-   `resources_path`: path to the resources directory.
-   `overwrite_training`: whether to overwrite the training of the entity disambiguation model provided a model with the same path and name already exists.
-   `rel_params`: set of parameters specific to the `reldisamb` method:
    -   `model_path`: Path to the entity disambiguation model.
    -   `data_path`: Path to the dataset file `linking_df_split.tsv` used for training a model (see information about the dataset in the "[Resources and file structure](../../getting-started/resources.md)" page in the documentation).
    -   `training_split`: Column from the `linking_df_split.tsv` file that indicates which documents are used for training, development, and testing (see more information about this in the "[Resources and file structure](../../getting-started/resources.md)" page in the documentation).
    -   `db_embeddings`: cursor for the embeddings database (see more information about this in the "[Resources and file structure](../../getting-started/resources.md)" page in the documentation).
    -   `with_publication`: whether place of publication should be used as a feature when disambiguating (by adding it as an already disambiguated entity).
    -   `without_microtoponyms`: whether to filter out microtoponyms or not (i.e. filter out all entities that are not `LOC`).
    -   `do_test`: Whether to train an entity disambiguation model in test mode.
    -   `default_publname`: The default value for the place of publication of the texts. For example, "London". This will be the default publication place name, but you will be able to override it when using the Linker to do predictions. This will be ignored if `with_publication` is `False`.
    -   `default_publwqid`: The wikidata ID of the place of publication. For example, `Q84` for London. As in `default_publname`, you will be able to override it at inference time, and it will be ignored if `with_publication` is `False`.

In this way, an entity disambiguation model will be trained unless a model trained using the same characteristics already exists (i.e. same candidate ranker method, same `training_split` column name, and same values for `with_publication` and `without_microtoponyms`).

When using the `reldisamb` linking approach, the resources folder should at least contain the following resources:

    T-Res/
      └── resources/
          ├── wikidata/
          |   ├── entity2class.txt
          |   ├── mentions_to_wikidata.json
          |   └── wikidata_gazetteer.csv
          └── rel_db/
              └── embeddings_database.db

## 2. Load the resources

!!! title "Note"

    Note that this step is already taken care of if you use the `Pipeline`.

The following line of code loads the resources required by the Linker, regardless of the Linker method.

```python
linker.load()
```

## 3. Train an entity disambiguation model

!!! title "Note: Only the `RelDisambLinker` requires training"

    The training step is only possible if the `reldisamb` linking method is selected by instantiating a linker of type `RelDisambLinker`. The other linking methods (`mostpopular` and `bydistance`) are rule-based and therefore no model training is necessary.

!!! title "Note"

    Note that this step is already taken care of if you use the `Pipeline`.

The following line will train a REL model for entity disambiguation, given the arguments specified when instantiating the `RelDisambLinker`.

```python
linker.train_load_model()
```

Note that if the model already exists and `overwrite_training` is set to `False`, the training will be skipped, even if you call the `train_load_model()` method.

The resulting model will be stored in the location specified when instantiating the Linker (i.e. `resources/models/disambiguation/` in the example) in a new folder whose name combines information about the ranking and linking arguments used in training the method.

## 4. Link & disambiguate candidates to obtain predictions

This example demonstrates the use of the Linker to link toponym matches to entities in the knowledgebase. The Linker's `run` method requires that the input is a `CandidateMatches` instance, which can be obtained by executing the [`Ranker`](ranker.md) on a toponym `Mention`. Here we assume the `candidate_matches` variable is such an instance. The `run` method also takes two optional arguments relating to the place of publication of the text:
```python
mention_candidates = linker.run(candidate_matches, place_of_pub_wqid="Q84", place_of_pub="London")
print(mention_candidates)
```
The result is a `MentionCandidates` instance.

The final step is entity disambiguation, which produces a disambiguation score for each fo the candidate links in the knowledgebase. This step is also performed by the Linker, via the `disambiguate` method, which takes a list of `SentenceCandidates` instances:
```python
predictions = linker.disambiguate(sentence_candidates)
print(predictions)
```

&nbsp;
