"""Guard against 'correcting' real cities into different covered cities."""
import pytest

from geoparse.phonetic.corrector import PhoneticCorrector


@pytest.fixture(scope="module")
def corrector():
    return PhoneticCorrector()


@pytest.mark.parametrize("city", ["Raipur", "Kanpur", "Jabalpur", "Udaipur", "Aligarh", "Mysuru"])
def test_known_out_of_coverage_cities_are_left_unchanged(corrector, city):
    assert corrector.correct_city(city) == (city, 0.0)


@pytest.mark.parametrize("query", ["Satna", "Hapur", "Raigarh"])
def test_first_letter_guard_blocks_unrelated_cities(corrector, query):
    assert corrector.correct_city(query)[0] == query


@pytest.mark.parametrize("query,expected", [
    ("Bnglr", "Bengaluru"), ("Jaipr", "Jaipur"), ("Hydrabad", "Hyderabad"),
    ("Lukcnow", "Lucknow"), ("Chandigrah", "Chandigarh"),
])
def test_real_misspellings_are_still_corrected(corrector, query, expected):
    assert corrector.correct_city(query)[0] == expected


def test_guard_can_be_disabled_for_comparison():
    assert PhoneticCorrector(guard_false_corrections=False).correct_city("Raipur")[0] == "Jaipur"
