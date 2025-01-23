from typing import List, Optional

from sentence_splitter import split_text_into_sentences

from . import ner, ranking, linking
from ..utils.dataclasses import SentenceContext, SentenceMentions, SentenceCandidates, Candidates, Predictions

class Pipeline:
    """
    The Pipeline class constitutes an end-to-end pipeline for toponym resolution 
    using natural language processing, including Named Entity Recognition (NER), 
    candidate ranking, and linking to geolocated entities in the Wikidata 
    knowledgebase.

    Arguments:
        ner (ner.Recogniser, optional): The NER (Named Entity
            Recogniser) object to use in the pipeline. If None, a default
            ``Recogniser`` will be instantiated. For the default settings, see
            Notes below.
        ranker (ranking.Ranker, optional): The ``Ranker`` object to use in
            the pipeline. If None, the default ``Ranker`` will be instantiated.
            For the default settings, see Notes below.
        linker (linking.Linker, optional): The ``Linker`` object to use in
            the pipeline. If None, the default ``Linker`` will be instantiated.
            For the default settings, see Notes below.
        resources_path (str, optional): The path to your resources directory.
        experiments_path (str, optional): The path to the experiments directory.

    Example:
        ```python
        # Instantiate the Pipeline object with a default setup
        pipeline = Pipeline()

        # Now you can use the pipeline for processing text or sentences
        text = "I visited Paris and New York City last summer."
        predictions = pipeline.run(text)
        print(predictions)

        # Access the results for each toponym mention in the text
        for toponym in predictions.candidates():
            print(toponym)
        ```

    Note:
        The default settings for the ``Recogniser``:

        ```python
        ner.PretrainedRecogniser(
            model="Livingwithmachines/toponym-19thC-en",
        )
        ```

        The default settings for the ``Ranker``:
        ```python
        ranking.PerfectMatchRanker(
            resources_path=resources_path,
        )
        ```

        The default settings for the ``Linker``:
        ```python
        linking.MostPopularLinker(
            resources_path=resources_path,
            experiments_path=experiments_path,
        )
        ```
    """

    def __init__(
        self,
        recogniser: Optional[ner.Recogniser] = None,
        ranker: Optional[ranking.Ranker] = None,
        linker: Optional[linking.Linker] = None,
        resources_path: Optional[str] = None,
        experiments_path: Optional[str] = "../experiments",
    ):
        """
        Instantiates a Pipeline object.
        """

        self.recogniser = recogniser
        self.ranker = ranker
        self.linker = linker

        # If ner is None, instantiate the default Recogniser.
        if not self.recogniser:
            self.recogniser = ner.PretrainedRecogniser(
                model_name="Livingwithmachines/toponym-19thC-en",
            )

        # If ranker is None, instantiate the default Ranker.
        if not self.ranker:
            if not resources_path:
                raise ValueError("[ERROR] Please specify path to resources directory.")
            self.ranker = ranking.PerfectMatchRanker(
                resources_path=resources_path,
            )

        # If linker is None, instantiate the default Linker.
        if not self.linker:
            if not resources_path:
                raise ValueError("[ERROR] Please specify path to resources directory.")
            self.linker = linking.MostPopularLinker(
                resources_path=resources_path,
                experiments_path=experiments_path,
            )

        self.recogniser.load()
        self.ranker.load()
        self.linker.load()

    # TODO: docstring
    def run(self, 
            text: str, 
            place_of_pub_wqid: Optional[str]=None,
            place_of_pub: Optional[str]=None, 
        ) -> Predictions:
        """
        Runs the end-to-end pipeline.

        Args:
            text (str): A block of text.
            place_of_pub_wqid (Optional[str]): The Wikidata ID of the
                place of publication of the text, if available.
            place_of_pub (Optional[str]): The place of publication of
                the text, if available.

        Returns:
            Toponyms identified in the text, linked to the Wikidata 
                knowledgebase.
        """

        mentions = self.run_text_recognition(text)
        candidates = self.run_candidate_selection(mentions, place_of_pub_wqid, place_of_pub)
        return self.run_disambiguation(candidates)

    ### Modular/stepwise methods:

    def run_text_recognition(self, text: str) -> List[SentenceMentions]:
        """
        Runs the named entity recognition step of the pipeline.
        
        Args:
            text (str): A block of text.

        Returns:
            A list of `SentenceMentions` instances, one for each sentence in 
                the text.
        """
        # Split the text into sentences.
        text = str(text)
        sentences = SentenceContext.from_text(text, language="en")
        return [self.recogniser.run(sentence.sentence) for sentence in sentences]
    
    def run_candidate_selection(self, 
            sentence_mentions: List[SentenceMentions], 
            place_of_pub_wqid: Optional[str]=None,
            place_of_pub: Optional[str]=None, 
        ) -> Candidates:
        """
        Runs the candidate selection step of the pipeline.
        
        Args:
            sentence_mentions (List[SentenceMentions]): A list of 
                `SentenceMentions` instances, as produced by the 
                `run_text_recognition` method. 
            place_of_pub_wqid (Optional[str]): The Wikidata ID of the
                place of publication of the text, if available.
            place_of_pub (Optional[str]): The place of publication of
                the text, if available.

        Returns:
            A `Candidates` instance containing toponym candidates.
        """
        sentence_candidates = list()
        for sms in sentence_mentions:
            matches = [self.ranker.run(mention) for mention in sms.mentions]
            candidates = [self.linker.run(m, place_of_pub_wqid, place_of_pub) for m in matches]
            sentence_candidates.append(SentenceCandidates(sms.sentence, candidates))
        return Candidates(sentence_candidates)

    def run_disambiguation(self, candidates: Candidates) -> Predictions:
        """
        Runs the entity disambiguation step of the pipeline.
        
        Args:
            candidates (Candidates): A `Candidates` instance, as produced by 
                the `run_candidate_selection` method. 

        Returns:
            A `Predictions` instance containing linked toponym predictions.
        """
        return self.linker.disambiguate(candidates.sentence_candidates)
