import re

class RuleFilter:
    def __init__(self):
        self.hard_block_keywords = [
            "kill", "die", "hate", "stupid", "idiot"
        ]

        self.hard_allow_patterns = [
            r"^thank(s| you)?$",
            r"^ok$",
            r"^lol$"
        ]
         

    def apply(self, text: str) -> str:
        # Normalize text 
        # Reduce repeated characters to two
        # Remove special characters except Chinese characters
        # Reduce multiple spaces to single space
        cleaned_text = text.lower().strip()
        cleaned_text = re.sub(r"([a-z])\1{2,}", r"\1\1", cleaned_text)
        cleaned_text = re.sub(r"[^\w\s\u4e00-\u9fff]+", " ", cleaned_text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        # Further reduce repeated characters to one for additional check
        cleaned_text_no_repeats = re.sub(r"([a-z])\1{1,}", r"\1", cleaned_text)
    
        # Tokenize
        tokens = cleaned_text.split()
        tokens_no_repeats = cleaned_text_no_repeats.split()

        # Hard block
        for token in tokens:
            if token in self.hard_block_keywords:
                return "BLOCK"
            
        for token in tokens_no_repeats:
            if token in self.hard_block_keywords:
                return "BLOCK"  

        # Hard allow
        for pattern in self.hard_allow_patterns:
            if re.match(pattern, cleaned_text) or re.match(pattern, cleaned_text_no_repeats):
                return "ALLOW"

        return "PASS"
    

# Test few cases
if __name__ == "__main__":
    rf = RuleFilter()

    samples = [
        "I hattte you!",
        "thannnnks",
        "This is a cool skilllll",
        "Thanks, idiot!",
        "diet is important",
    ]

    for sample in samples:
        result = rf.apply(sample)
        print(sample, "->", result)
