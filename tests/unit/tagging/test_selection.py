from langcheck.tagging.selection import ByFlags, BySlots, EpochChoice
from langcheck.tagging.tiers import Falloff, TierPolicy


def test_flags_are_judged_under_their_own_policy() -> None:
    policy = TierPolicy(growth=Falloff(join=0.1))
    choices: list[EpochChoice] = [BySlots(), ByFlags(policy)]

    assert choices[1] == ByFlags(TierPolicy(growth=Falloff(join=0.1)))
    assert choices[0] != choices[1]
