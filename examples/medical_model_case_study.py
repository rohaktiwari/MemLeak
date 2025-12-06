"""
Hypothetical case study: analyze leak risk for a clinical notes model.
"""
from memleak import MembershipTester


# In practice, point to a finetuned checkpoint on clinical notes.
MODEL_ID = "distilbert-base-uncased"

train = [
    "Patient reports chronic lower back pain for two months.",
    "Diagnosis: Type II diabetes with HbA1c of 8.1%.",
    "Prescription: amoxicillin 500mg twice daily.",
]

test = [
    "Patient presents with mild fever and headache.",
    "Follow-up visit scheduled in two weeks.",
    "No past medical history noted.",
]


def main():
    tester = MembershipTester(model=MODEL_ID, max_length=128)
    report = tester.run_attacks(train, test)
    print(f"Privacy risk score: {report.risk_score}")
    print(report.summary)
    for tip in report.recommendations():
        print(f"- {tip}")


if __name__ == "__main__":
    main()

