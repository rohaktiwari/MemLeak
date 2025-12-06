from memleak import MembershipTester


train = [
    "The patient was diagnosed with a rare condition.",
    "Quantum computing is still in its infancy.",
    "Alice loves to bake sourdough bread on Sundays.",
]

test = [
    "This sentence was not in training.",
    "Large language models can memorize data.",
]

tester = MembershipTester(model="gpt2", max_length=64, batch_size=2)
report = tester.run_attacks(train, test)
print(report.summary.head())

