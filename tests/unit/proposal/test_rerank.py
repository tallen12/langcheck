from langcheck.proposal.rerank import RerankPolicy, Scored, reranked

GOOD = Scored("goes", -2.0, 0.9, 0)
LIKELY = Scored("gone", -1.0, 0.0, 0)
DELETE = Scored("", -0.5, 0.9, 1)


def test_the_tagger_and_the_penalty_change_the_order() -> None:
    assert reranked([LIKELY, GOOD], RerankPolicy(0.0, 0.0)) == [LIKELY, GOOD]
    assert reranked([LIKELY, GOOD], RerankPolicy(2.0, 0.0)) == [GOOD, LIKELY]
    assert reranked([GOOD, DELETE], RerankPolicy(1.0, 0.0))[0] == DELETE
    assert reranked([GOOD, DELETE], RerankPolicy(1.0, 2.0))[0] == GOOD
