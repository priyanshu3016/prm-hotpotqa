from src.pipeline import MultiHopQAPipeline

# Dummy test data mimicking HotpotQA structure
test_questions = [
    {
        "question": "What nationality is the director of Inception?",
        "paragraphs": [
            "Inception is a 2010 science fiction film directed by Christopher Nolan.",
            "Christopher Nolan is a British-American filmmaker born in London.",
            "Leonardo DiCaprio is an American actor born in Los Angeles.",
            "The Dark Knight is a superhero film also directed by Christopher Nolan.",
            "Interstellar is a space film released in 2014.",
            "Tom Hardy is a British actor known for many roles.",
            "Warner Bros is an American film studio.",
            "Hans Zimmer composed the score for Inception.",
            "The film grossed over 800 million dollars worldwide.",
            "Ellen Page starred in Inception alongside DiCaprio.",
        ],
        "gold_answer": "British-American"
    },
    {
        "question": "What country is the headquarters of the company that makes iPhone?",
        "paragraphs": [
            "Apple Inc. is an American multinational technology company.",
            "Apple is headquartered in Cupertino, California, United States.",
            "The iPhone is a smartphone designed and marketed by Apple Inc.",
            "Samsung is a South Korean electronics company.",
            "Google is headquartered in Mountain View, California.",
            "Microsoft was founded by Bill Gates in Albuquerque.",
            "Tim Cook is the current CEO of Apple Inc.",
            "The App Store was launched by Apple in 2008.",
            "Android is a mobile operating system developed by Google.",
            "Nokia is a Finnish telecommunications company.",
        ],
        "gold_answer": "United States"
    },
]

# Run pipeline
pipeline = MultiHopQAPipeline(threshold=0.4)

for i, item in enumerate(test_questions):
    print(f"\n{'='*50}")
    print(f"Question {i+1}: {item['question']}")
    print(f"Gold answer: {item['gold_answer']}")
    
    result = pipeline.run(
        question=item["question"],
        paragraphs=item["paragraphs"],
    )
    
    print(f"Model answer: {result['answer']}")
    print(f"Sub-questions: {result['sub_questions']}")
    print(f"Kept paragraphs: {len(result['kept'])}")
    print(f"Pruned paragraphs: {len(result['pruned'])}")