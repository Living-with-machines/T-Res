# T-Res data structures

T-Res defines a collection of interrelated data structures to manage the flow of information through the toponym resolution pipeline. These structures are implemented as [Python dataclasses](https://realpython.com/python-data-classes/). Understanding these dataclasses, and how to use them, will help you to understand the toponym resolution workflow and how to work with the results.

Here we describe the purpose of the most important dataclasses, what they represent and where they appear within the pipeline. 

We also provide class diagrams that capture the relationships between the different dataclasses. Further details, including complete lists of attributes for each dataclass, and relevant snippets of source code, can be found in the [reference section](../reference/utils/dataclasses.md) of this site.

## Mention

The `Mention` dataclass represents a toponym mention in a piece of text. In addition to the toponym mention itself, found in the `mention` attribute, it records its location (i.e. its character position and token offset within the text), together with its NER confidence score and NER label (e.g. `LOC` for location).

<figure markdown="1">
![Mention dataclass](../../assets/mention-dataclass.svg){ width="260" } 
</figure>

There is a predicate method named `is_microtoponym` which returns `True` if the mention refers to a toponym which is *not* a location. For instance, mentions having the NER label `BUILDING` are microtoponyms.

## SentenceMentions

The `SentenceMentions` dataclass represents a collection of toponym mentions within a sentence. It consists of a sentence of text, together with a list of `Mention` instances.

<figure markdown="1">
![SentenceMentions dataclass](../../assets/sentencementions-dataclass.svg){ width="260" } 
</figure>

The `is_empty` method returns `True` if there are no toponym mentions in the sentence. The `exclude_microtoponyms` method returns another `SentenceMentions` instance which is identical except all microtoponym mentions are omitted.

!!! title "Note"

    The output from the T-Res named entity recognition process is a list of `SentenceMentions` instances, one for each sentence in the text.

## MentionCandidates

The `MentionCandidates` dataclass represents a collection of candidate links for a given toponym mention. It consists of a `Mention` instance together with list of `CandidateLinks` instances. Each `CandidateLinks` object contains a candidate string match for the toponym and, for each candidate string match, a list of candidate links in the knowledgebase.

This dataclass also records the ranking and linking methods used to generate the candidates, and the place of publication information, if it was provided.

<figure markdown="1">
![MentionCandidates dataclass](../../assets/mentioncandidates-dataclass.svg){ width="260" } 
</figure>

## Candidates

The `Candidates` dataclass represents candidate matches for all toponym mentions in a block of text.

<figure markdown="1">
![Candidates dataclass](../../assets/candidates-dataclass.svg){ width="260" } 
</figure>

Internally, candidates are stored as a list of `SentenceCandidates` instances. This preserves the sentence structure of the text. 

If, however, this structure is not needed, the `candidates` method can be executed to obtain the list of candidates by toponoym mention.

!!! title "Note"

    The output from the T-Res candidate selection process is an instance of the `Candidates` dataclass.

## Predictions

The `Predictions` dataclass is a subclass of `Candidates`, and represents predicted matches (in the knowledgebase) for all toponym mentions in a block of text.

<figure markdown="1">
![Predictions dataclass](../../assets/predictions-dataclass.svg){ width="260" } 
</figure>

Details of the predicted toponym matches can be obtained from a `Predictions` instance using the methods `best_wqids`, `best_coords` and `best_disambiguation_scores` which return (respectively), for each toponym mention, the predicted Wikidata link (by its Wikidata QID), the predicted geographical coordinates of the toponym and the highest disambiguation score.

!!! tip

    To obtain a list of toponym matches from an instance of the `Predictions` dataclass, always use the `candidates` method, which returns a list of `MentionCandidates` objects, one per toponym mention.

!!! note

    The output from the T-Res end-to-end pipeline is an instance of the `Predictions` dataclass. 

&nbsp;
