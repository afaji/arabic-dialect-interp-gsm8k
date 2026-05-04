from geometric_complexity_scaling.scoring import (
    clean_short_answer,
    normalize_triviaqa_answer,
    score_triviaqa_answer,
    triviaqa_f1,
)
from geometric_complexity_scaling.tasks import is_correct, parse_answer, target_for_row


def trivia_target():
    return {
        "value": "York",
        "normalized_value": "york",
        "aliases": ["York, UK", "City of York"],
        "normalized_aliases": ["york uk", "city of york"],
    }


def test_triviaqa_normalization_handles_punctuation_articles_and_underscores():
    assert normalize_triviaqa_answer("The_City, of York!") == "city of york"
    assert normalize_triviaqa_answer("30's") == "30 s"


def test_triviaqa_open_ended_answer_can_score_correct():
    score = score_triviaqa_answer("Answer: York.", trivia_target())
    assert score["correct"] is True
    assert score["exact_match"] is True


def test_triviaqa_f1_allows_extra_words_without_accepting_unrelated_answers():
    assert score_triviaqa_answer("The answer is City of York.", trivia_target())["correct"] is True
    assert score_triviaqa_answer("London", trivia_target())["correct"] is False
    assert triviaqa_f1("City of York", "York") >= 0.5


def test_fact_target_extracts_full_triviaqa_answer_schema():
    row = {
        "answer": {
            "value": "Chicago Bears",
            "normalized_value": "chicago bears",
            "aliases": ["Chicago Bears football"],
            "normalized_aliases": ["chicago bears football"],
            "matched_wiki_entity_name": "Chicago Bears",
            "normalized_matched_wiki_entity_name": "chicago bears",
        }
    }
    target = target_for_row("fact", row)
    assert target["normalized_value"] == "chicago bears"
    assert is_correct("fact", "Answer: Chicago Bears", target)


def test_sentiment_and_math_correctness():
    assert parse_answer("sentiment", "Positive.") == "positive"
    assert is_correct("sentiment", "positive", "positive")
    assert parse_answer("math", "Final answer: 1,234") == "1234"
    assert is_correct("math", "1234", "1234")


def test_math_parsing_and_scoring_supports_arabic_indic_digits():
    target = target_for_row("math", {"answer": "#### ١٬٢٣٤"})
    assert target == "1234"
    assert parse_answer("math", "الإجابة النهائية: ١٬٢٣٤") == "1234"
    assert is_correct("math", "١٢٫٥", "12.5")


def test_clean_short_answer_strips_special_tokens_and_prefixes():
    assert clean_short_answer("<end_of_turn>\nFinal answer: York.") == "York."
