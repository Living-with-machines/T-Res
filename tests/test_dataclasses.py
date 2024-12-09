from ast import literal_eval
from t_res.geoparser.dataclasses import SentenceContext, SentenceMentions, Sentence, Mention

def test_sentence_context():

    text = "A remarkable case of rattening has just occurred in the building trade at Shefiield, but also in Leeds. Not in London though. What is to be done?"
    result = SentenceContext.from_text(text)

    assert len(result) == 3
    assert result[0].sentence == "A remarkable case of rattening has just occurred in the building trade at Shefiield, but also in Leeds."
    assert result[0].preceding_sentence == None
    assert result[0].following_sentence == "Not in London though."

    assert result[1].sentence == "Not in London though."
    assert result[1].preceding_sentence == "A remarkable case of rattening has just occurred in the building trade at Shefiield, but also in Leeds."
    assert result[1].following_sentence == "What is to be done?"

    assert result[2].sentence == "What is to be done?"
    assert result[2].preceding_sentence == "Not in London though."
    assert result[2].following_sentence == None

def test_sentence_mentions():

    sentence = ', thence to Emery Down,crowing to Minesteed Manor ; he ther tacked back to Notherwood, and from thence back again to the Manor, where, after a brilliant run (Arnie hour and forty-five minutes, Reynold was compelled to succumb to his pursuers. '
    mentions_str = "[{'mention': 'Emery Down', 'start_offset': 3, 'end_offset': 4, 'start_char': 12, 'ner_score': 0.999, 'ner_label': 'LOC', 'entity_link': 'O'}, {'mention': 'Minesteed Manor', 'start_offset': 8, 'end_offset': 9, 'start_char': 34, 'ner_score': 0.999, 'ner_label': 'BUILDING', 'entity_link': 'O'}, {'mention': 'Notherwood', 'start_offset': 16, 'end_offset': 16, 'start_char': 75, 'ner_score': 0.999, 'ner_label': 'BUILDING', 'entity_link': 'O'}]"
    mentions = [Mention.from_dict(d) for d in literal_eval(mentions_str)]

    sentence_mentions = SentenceMentions(Sentence(sentence), mentions=mentions)
    assert sentence_mentions.len() == 3
    assert sentence_mentions.exclude_microtoponyms().len() == 1
    assert sentence_mentions.exclude_microtoponyms().mentions[0].ner_label == 'LOC'