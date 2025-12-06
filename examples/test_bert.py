from memleak import MembershipTester


train = [
    "Privacy-preserving machine learning protects sensitive data.",
    "Neural networks can overfit small datasets.",
    "Membership inference attacks exploit confidence scores.",
]

test = [
    "This is a new example not seen during training.",
    "Regularization reduces overfitting.",
]

tester = MembershipTester(model="bert-base-uncased", max_length=96)
report = tester.run_attacks(train, test)
print(report.summary.head())

