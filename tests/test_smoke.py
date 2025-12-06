from memleak import MembershipTester


def test_run_attacks_smoke():
    train = ["hello world", "membership inference is important"]
    test = ["unseen sample"]
    tester = MembershipTester(model="distilbert-base-uncased", max_length=32, batch_size=1)
    report = tester.run_attacks(train, test)
    assert report.risk_score >= 0
    assert not report.summary.empty

