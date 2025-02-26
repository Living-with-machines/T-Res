# T-Res data structures

TODO.

T-Res defines a collection of interrelated data structures to manage the flow of information through the toponym resolution pipeline. The structures are implemented as [Python dataclasses](https://realpython.com/python-data-classes/).

Here we describe the purpose of the most important dataclasses, what they represent and where they appears within the pipeline. We also provide class diagrams that capture the relationships between the different dataclasses. Further details, including complete lists of attributes for each dataclass, and relevant snippets of source code, can be found in the [reference section](../reference/utils/dataclasses.md) of this site.

## Mention

The `Mention` dataclass represents a toponym mention in a piece of text.

## SentenceMentions

The output from the T-Res candidate selection process is a list of instances of the `SentenceMentions` dataclass, one instance for each sentence in the text.

## Candidates

The output from the T-Res candidate selection process is an instance of the `Candidates` dataclass.

## CandidateLinks

## CandidateMatches

## Predictions

The output from the T-Res end-to-end pipeline is an instance of the `Predictions` dataclass. 


