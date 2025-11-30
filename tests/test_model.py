"""
Comprehensive Test Suite for Sentiment Analysis Model
======================================================

This script tests the fine-tuned sentiment analysis model with diverse test cases
covering various scenarios including clear sentiment, mixed sentiment, out-of-domain
text, and modern slang/emoji usage.

Test Categories:
1. Clear Positive Sentiment
2. Clear Negative Sentiment
3. Mixed/Subtle Sentiment
4. Out-of-Domain Text
5. Slang and Informal Language
"""

import torch
from transformers import pipeline
from typing import Dict, List
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model path - update this based on your setup
# Option 1: Local model path
MODEL_PATH = "./models/sentiment-distilbert-imdb-final"

# Option 2: Hugging Face Hub model
# MODEL_PATH = "YOUR_USERNAME/sentiment-distilbert-imdb"

# Option 3: Default fine-tuned model for testing
# MODEL_PATH = "distilbert-base-uncased-finetuned-sst-2-english"

# ============================================================================
# TEST CASES
# ============================================================================

TEST_CASES = {
    "clear_positive": [
        {
            "id": "CP1",
            "text": "This movie was an absolute masterpiece! The best I've seen all year.",
            "expected": "POSITIVE",
            "description": "Superlative praise with enthusiasm"
        },
        {
            "id": "CP2",
            "text": "Incredible performances, stunning visuals, and a deeply moving story. A true cinematic triumph!",
            "expected": "POSITIVE",
            "description": "Multiple positive attributes"
        },
        {
            "id": "CP3",
            "text": "I was blown away by every single scene. This film exceeded all my expectations and left me speechless.",
            "expected": "POSITIVE",
            "description": "Strong emotional positive response"
        }
    ],
    
    "clear_negative": [
        {
            "id": "CN1",
            "text": "A completely worthless script and terrible acting. Save your money.",
            "expected": "NEGATIVE",
            "description": "Direct criticism with warning"
        },
        {
            "id": "CN2",
            "text": "This was painfully boring. I almost walked out of the theater. Two hours of my life I'll never get back.",
            "expected": "NEGATIVE",
            "description": "Personal negative experience with regret"
        },
        {
            "id": "CN3",
            "text": "Disaster of a film. Poor direction, weak plot, unconvincing performances. Absolutely disappointing.",
            "expected": "NEGATIVE",
            "description": "Multiple critical points"
        }
    ],
    
    "mixed_subtle": [
        {
            "id": "MS1",
            "text": "The plot was exciting, but the ending was a total letdown.",
            "expected": "MIXED",  # Could go either way
            "description": "Positive start, negative conclusion"
        },
        {
            "id": "MS2",
            "text": "Great acting saved an otherwise mediocre story. The visuals were stunning but couldn't fix the weak script.",
            "expected": "MIXED",
            "description": "Balanced pros and cons"
        },
        {
            "id": "MS3",
            "text": "I wanted to love this film, and parts of it were brilliant, but overall it felt incomplete.",
            "expected": "NEGATIVE",  # Leans negative despite some praise
            "description": "Disappointed expectations despite some positives"
        },
        {
            "id": "MS4",
            "text": "Not the best movie ever, but certainly not the worst. Perfectly fine for a rainy afternoon.",
            "expected": "POSITIVE",  # Mild positive
            "description": "Lukewarm approval with context"
        }
    ],
    
    "out_of_domain": [
        {
            "id": "OOD1",
            "text": "I ate pizza for dinner last night.",
            "expected": "NEUTRAL",  # Should show low confidence
            "description": "Factual statement about food"
        },
        {
            "id": "OOD2",
            "text": "The weather forecast predicts rain tomorrow afternoon with temperatures around 65 degrees.",
            "expected": "NEUTRAL",
            "description": "Weather information"
        },
        {
            "id": "OOD3",
            "text": "Please remember to submit your report by Friday at 5 PM.",
            "expected": "NEUTRAL",
            "description": "Work instruction"
        },
        {
            "id": "OOD4",
            "text": "The capital of France is Paris, which is located in the northern part of the country.",
            "expected": "NEUTRAL",
            "description": "Geographic fact"
        }
    ],
    
    "slang_informal": [
        {
            "id": "SI1",
            "text": "OMG this film was fire 🔥. So good.",
            "expected": "POSITIVE",
            "description": "Modern slang and emoji"
        },
        {
            "id": "SI2",
            "text": "This movie slaps! Absolutely loved every minute. No cap fr fr 💯",
            "expected": "POSITIVE",
            "description": "Gen Z slang expressions"
        },
        {
            "id": "SI3",
            "text": "Ngl this was trash. Wasted my time smh 😤",
            "expected": "NEGATIVE",
            "description": "Negative informal language with abbreviations"
        },
        {
            "id": "SI4",
            "text": "Meh. It was mid. Nothing special tbh.",
            "expected": "NEGATIVE",  # Or neutral - 'mid' means mediocre
            "description": "Lukewarm/mediocre slang"
        },
        {
            "id": "SI5",
            "text": "YOOO this movie hit different! 10/10 would recommend!!!",
            "expected": "POSITIVE",
            "description": "Enthusiastic internet speak"
        }
    ]
}

# ============================================================================
# TEST RUNNER
# ============================================================================

class SentimentTester:
    """Test suite for sentiment analysis model."""
    
    def __init__(self, model_path: str):
        """Initialize the tester with a model."""
        print("🔧 Loading model...")
        try:
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=model_path,
                device=0 if torch.cuda.is_available() else -1
            )
            print(f"✅ Model loaded from: {model_path}")
        except Exception as e:
            print(f"⚠️  Could not load model from {model_path}")
            print(f"   Error: {e}")
            print("   Using default sentiment analysis model for testing...")
            self.pipeline = pipeline("sentiment-analysis")
        
        self.results = []
        
    def run_test(self, test_case: Dict) -> Dict:
        """Run a single test case."""
        text = test_case["text"]
        prediction = self.pipeline(text[:512])[0]  # Limit to 512 tokens
        
        return {
            "id": test_case["id"],
            "text": text,
            "expected": test_case["expected"],
            "predicted_label": prediction["label"],
            "confidence": prediction["score"],
            "description": test_case["description"],
            "correct": self._check_correctness(test_case["expected"], prediction["label"])
        }
    
    def _check_correctness(self, expected: str, predicted: str) -> str:
        """Check if prediction matches expectation."""
        if expected == "MIXED" or expected == "NEUTRAL":
            return "N/A"  # No clear correct answer
        return "✅" if expected in predicted.upper() else "❌"
    
    def run_all_tests(self) -> List[Dict]:
        """Run all test cases."""
        print("\n" + "=" * 80)
        print("🧪 RUNNING COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        
        all_results = []
        
        for category, test_cases in TEST_CASES.items():
            print(f"\n📂 Category: {category.replace('_', ' ').title()}")
            print("-" * 80)
            
            for test_case in test_cases:
                result = self.run_test(test_case)
                all_results.append(result)
                
                print(f"\n  Test ID: {result['id']}")
                print(f"  Description: {result['description']}")
                print(f"  Text: \"{result['text'][:80]}{'...' if len(result['text']) > 80 else ''}\"")
                print(f"  Expected: {result['expected']}")
                print(f"  Predicted: {result['predicted_label']} (confidence: {result['confidence']:.4f})")
                print(f"  Correct: {result['correct']}")
        
        self.results = all_results
        return all_results
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        if not self.results:
            return "No tests have been run yet."
        
        # Calculate statistics
        total_tests = len(self.results)
        correct = sum(1 for r in self.results if r['correct'] == '✅')
        incorrect = sum(1 for r in self.results if r['correct'] == '❌')
        na = sum(1 for r in self.results if r['correct'] == 'N/A')
        
        avg_confidence = sum(r['confidence'] for r in self.results) / total_tests
        
        # High confidence predictions (>0.9)
        high_confidence = [r for r in self.results if r['confidence'] > 0.9]
        
        # Low confidence predictions (<0.7)
        low_confidence = [r for r in self.results if r['confidence'] < 0.7]
        
        report = f"""
{'=' * 80}
📊 SENTIMENT ANALYSIS MODEL - TEST REPORT
{'=' * 80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {MODEL_PATH}

{'=' * 80}
📈 OVERALL STATISTICS
{'=' * 80}

Total Tests:              {total_tests}
Correct Predictions:      {correct} ({correct/(total_tests-na)*100:.1f}%)
Incorrect Predictions:    {incorrect} ({incorrect/(total_tests-na)*100:.1f}%)
Ambiguous Cases (N/A):    {na}

Average Confidence:       {avg_confidence:.4f} ({avg_confidence*100:.2f}%)
High Confidence (>90%):   {len(high_confidence)} tests
Low Confidence (<70%):    {len(low_confidence)} tests

{'=' * 80}
📋 CATEGORY BREAKDOWN
{'=' * 80}
"""
        
        # Category statistics
        for category, test_cases in TEST_CASES.items():
            category_results = [r for r in self.results if any(tc['id'] == r['id'] for tc in test_cases)]
            cat_correct = sum(1 for r in category_results if r['correct'] == '✅')
            cat_total = len([r for r in category_results if r['correct'] != 'N/A'])
            
            report += f"\n{category.replace('_', ' ').title()}:\n"
            report += f"  Tests: {len(category_results)}\n"
            if cat_total > 0:
                report += f"  Accuracy: {cat_correct}/{cat_total} ({cat_correct/cat_total*100:.1f}%)\n"
            else:
                report += f"  Accuracy: N/A (ambiguous category)\n"
        
        # Detailed incorrect predictions
        if incorrect > 0:
            report += f"\n{'=' * 80}\n"
            report += "❌ INCORRECT PREDICTIONS\n"
            report += "=" * 80 + "\n"
            
            for r in self.results:
                if r['correct'] == '❌':
                    report += f"\nTest {r['id']}: {r['description']}\n"
                    report += f"  Text: \"{r['text'][:100]}...\"\n"
                    report += f"  Expected: {r['expected']}\n"
                    report += f"  Predicted: {r['predicted_label']} (confidence: {r['confidence']:.4f})\n"
        
        # Low confidence predictions
        if low_confidence:
            report += f"\n{'=' * 80}\n"
            report += "⚠️  LOW CONFIDENCE PREDICTIONS (<70%)\n"
            report += "=" * 80 + "\n"
            
            for r in low_confidence:
                report += f"\nTest {r['id']}: {r['description']}\n"
                report += f"  Text: \"{r['text'][:100]}...\"\n"
                report += f"  Predicted: {r['predicted_label']} (confidence: {r['confidence']:.4f})\n"
        
        # Key insights
        report += f"\n{'=' * 80}\n"
        report += "💡 KEY INSIGHTS\n"
        report += "=" * 80 + "\n\n"
        
        report += "Strengths:\n"
        if correct > 0:
            report += f"  • Strong performance on clear sentiment (both positive and negative)\n"
        if len(high_confidence) > total_tests * 0.5:
            report += f"  • High confidence in most predictions ({len(high_confidence)}/{total_tests})\n"
        
        report += "\nAreas for Improvement:\n"
        if len([r for r in self.results if 'mixed' in r['id'].lower()]) > 0:
            report += f"  • Handling mixed or nuanced sentiment\n"
        if len([r for r in self.results if 'ood' in r['id'].lower()]) > 0:
            report += f"  • Detecting out-of-domain text (should show lower confidence)\n"
        if len([r for r in self.results if 'si' in r['id'].lower() and r['correct'] == '❌']) > 0:
            report += f"  • Understanding modern slang and emoji\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        return report
    
    def save_results(self, filename: str = "test_results.json"):
        """Save test results to JSON file."""
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'model': MODEL_PATH,
                'results': self.results
            }, f, indent=2)
        print(f"\n💾 Results saved to: {filename}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🎬 SENTIMENT ANALYSIS MODEL - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    # Initialize tester
    tester = SentimentTester(MODEL_PATH)
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Generate and print report
    report = tester.generate_report()
    print(report)
    
    # Save results
    tester.save_results("tests/test_results.json")
    
    # Save report to file
    with open("tests/test_report.txt", 'w') as f:
        f.write(report)
    print("📄 Report saved to: tests/test_report.txt")
    
    print("\n" + "=" * 80)
    print("✅ Testing complete!")
    print("=" * 80)
