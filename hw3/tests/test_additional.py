import math
from calculator.utils import can_use_calculator, use_calculator, extract_label

from pytest_utils.decorators import max_score


@max_score(5)
def test_use_calculator():
    assert use_calculator("") == ""
    assert (
        use_calculator("Q: How many mugs do I have? \nA: <<3+2+1>>")
        == "Q: How many mugs do I have? \nA: <<3+2+1>>6"
    )
