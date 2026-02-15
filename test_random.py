from utils import generate_srt_content

segments = [
    {
        'start': 0.0,
        'end': 2.0,
        'words': [
            {'word': 'Hello', 'start': 0.0, 'end': 0.5},
            {'word': 'world', 'start': 0.5, 'end': 1.0},
            {'word': 'this', 'start': 1.0, 'end': 1.5},
            {'word': 'is', 'start': 1.5, 'end': 2.0}
        ]
    }
]

print("--- Both True ---")
srt = generate_srt_content(segments, use_roman=False, use_karaoke=True, random_base=True, random_highlight=True)
print(srt)
