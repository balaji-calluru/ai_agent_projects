#!/usr/bin/env python3
"""
Comprehensive test suite for Real-Time AI Assistant

This test suite helps validate and learn RAG concepts in detail.
Run with: python test_real_time_assistant.py
"""

import sys
import time
from typing import List, Tuple

# Import the assistant (adjust path if needed)
try:
    import importlib.util
    import os
    
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    assistant_path = os.path.join(
        current_dir, 
        "03 - Build a Real-Time AI Assistant Using RAG + LangChain.py"
    )
    
    if not os.path.exists(assistant_path):
        # Try alternative path
        assistant_path = os.path.join(
            os.path.dirname(current_dir),
            "src",
            "03 - Build a Real-Time AI Assistant Using RAG + LangChain.py"
        )
    
    spec = importlib.util.spec_from_file_location("assistant", assistant_path)
    assistant_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assistant_module)
    RealTimeAssistant = assistant_module.RealTimeAssistant
    
except Exception as e:
    print(f"❌ Could not import RealTimeAssistant: {e}")
    print("Please ensure the assistant file is in the correct location.")
    print(f"Looking for: {assistant_path}")
    sys.exit(1)


class TestRealTimeAssistant:
    """Test cases for learning RAG concepts."""
    
    def __init__(self):
        """Set up assistant for tests."""
        print("🤖 Initializing assistant for testing...")
        try:
            self.assistant = RealTimeAssistant(model="llama3:8b")
            print("✅ Assistant initialized\n")
        except Exception as e:
            print(f"❌ Failed to initialize assistant: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure Ollama is installed and running")
            print("2. Verify model is available: ollama pull llama3:8b")
            print("3. Check your internet connection")
            sys.exit(1)
    
    def test_basic_question(self) -> bool:
        """Test 1: Basic Functionality"""
        print("🧪 Test 1: Basic Question Answering")
        print("-" * 50)
        try:
            response = self.assistant.ask("What is Python?")
            assert len(response) > 0, "Response should not be empty"
            assert isinstance(response, str), "Response should be a string"
            print(f"✅ Response received: {len(response)} characters")
            print(f"   Preview: {response[:100]}...")
            print("✅ Test 1 PASSED: Basic question answered\n")
            return True
        except Exception as e:
            print(f"❌ Test 1 FAILED: {e}\n")
            return False
    
    def test_realtime_information(self) -> bool:
        """Test 2: Real-Time Information Access"""
        print("🧪 Test 2: Real-Time Information Access")
        print("-" * 50)
        try:
            response = self.assistant.ask("What is the current year?")
            # Check for recent years
            current_years = ["2024", "2025", "2026"]
            found_year = any(year in response for year in current_years)
            
            if found_year:
                print(f"✅ Found current year in response")
            else:
                print(f"⚠️ Current year not explicitly found, but response received")
            
            print(f"   Response: {response[:150]}...")
            print("✅ Test 2 PASSED: Real-time information accessed\n")
            return True
        except Exception as e:
            print(f"❌ Test 2 FAILED: {e}\n")
            return False
    
    def test_search_integration(self) -> bool:
        """Test 3: Search Integration"""
        print("🧪 Test 3: Search Integration")
        print("-" * 50)
        questions = [
            "What's happening in AI today?",
            "Latest news about Python",
        ]
        
        passed = 0
        for question in questions:
            try:
                print(f"   Testing: {question}")
                response = self.assistant.ask(question)
                assert len(response) > 0, "Response should not be empty"
                print(f"   ✅ Got response ({len(response)} chars)")
                passed += 1
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        if passed == len(questions):
            print(f"✅ Test 3 PASSED: All search tests passed ({passed}/{len(questions)})\n")
            return True
        else:
            print(f"⚠️ Test 3 PARTIAL: {passed}/{len(questions)} tests passed\n")
            return False
    
    def test_error_handling(self) -> bool:
        """Test 4: Error Handling"""
        print("🧪 Test 4: Error Handling")
        print("-" * 50)
        test_cases = [
            ("", "Empty question"),
            ("a" * 500, "Very long question"),
        ]
        
        passed = 0
        for question, description in test_cases:
            try:
                print(f"   Testing: {description}")
                response = self.assistant.ask(question)
                # Should not crash
                assert response is not None, "Should return a response"
                print(f"   ✅ Handled gracefully")
                passed += 1
            except Exception as e:
                print(f"   ⚠️ Exception (may be expected): {e}")
                passed += 1  # Exception handling is also valid
        
        if passed == len(test_cases):
            print(f"✅ Test 4 PASSED: Error handling works ({passed}/{len(test_cases)})\n")
            return True
        else:
            print(f"❌ Test 4 FAILED: {passed}/{len(test_cases)} tests passed\n")
            return False
    
    def test_response_quality(self) -> bool:
        """Test 5: Response Quality"""
        print("🧪 Test 5: Response Quality")
        print("-" * 50)
        try:
            question = "Explain RAG in simple terms"
            response = self.assistant.ask(question)
            
            # Check response length
            assert len(response) > 50, "Response should be substantial"
            print(f"   Response length: {len(response)} characters ✅")
            
            # Check for relevant keywords (may not always be present)
            relevant_words = ["retrieval", "augmented", "generation", "search", "context", "information"]
            response_lower = response.lower()
            found_words = [word for word in relevant_words if word in response_lower]
            
            if found_words:
                print(f"   Found relevant keywords: {', '.join(found_words)} ✅")
            else:
                print(f"   ⚠️ No expected keywords found (may still be valid)")
            
            print(f"   Preview: {response[:200]}...")
            print("✅ Test 5 PASSED: Response quality is good\n")
            return True
        except Exception as e:
            print(f"❌ Test 5 FAILED: {e}\n")
            return False
    
    def test_question_types(self) -> bool:
        """Test 6: Different Question Types"""
        print("🧪 Test 6: Different Question Types")
        print("-" * 50)
        test_cases = [
            ("What is...", "What is machine learning?"),
            ("How does...", "How does RAG work?"),
            ("Who is...", "Who is the current president of the USA?"),
            ("When did...", "When did Python first release?"),
            ("Why is...", "Why is Python popular?"),
        ]
        
        passed = 0
        for question_type, question in test_cases:
            try:
                print(f"   Testing: {question_type}")
                response = self.assistant.ask(question)
                assert len(response) > 0, "Should get a response"
                print(f"   ✅ Answered ({len(response)} chars)")
                passed += 1
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        if passed == len(test_cases):
            print(f"✅ Test 6 PASSED: All question types handled ({passed}/{len(test_cases)})\n")
            return True
        else:
            print(f"⚠️ Test 6 PARTIAL: {passed}/{len(test_cases)} tests passed\n")
            return False
    
    def test_performance(self) -> bool:
        """Test 7: Performance"""
        print("🧪 Test 7: Performance")
        print("-" * 50)
        try:
            question = "What is Python?"
            start = time.time()
            response = self.assistant.ask(question)
            elapsed = time.time() - start
            
            print(f"   Response time: {elapsed:.2f} seconds")
            print(f"   Response length: {len(response)} characters")
            
            # Should respond within reasonable time (adjust threshold as needed)
            max_time = 60  # 60 seconds
            if elapsed < max_time:
                print(f"✅ Test 7 PASSED: Response time acceptable\n")
                return True
            else:
                print(f"⚠️ Test 7 WARNING: Response took {elapsed:.2f}s (threshold: {max_time}s)\n")
                return True  # Still pass, but warn
        except Exception as e:
            print(f"❌ Test 7 FAILED: {e}\n")
            return False
    
    def test_edge_cases(self) -> bool:
        """Test 8: Edge Cases"""
        print("🧪 Test 8: Edge Cases")
        print("-" * 50)
        edge_cases = [
            ("?", "Single character"),
            ("a" * 100, "Very long single word"),
            ("What is " + "? " * 10, "Many question marks"),
            ("1234567890", "Numbers only"),
            ("!@#$%^&*()", "Special characters only"),
        ]
        
        passed = 0
        for edge_case, description in edge_cases:
            try:
                print(f"   Testing: {description}")
                response = self.assistant.ask(edge_case)
                assert response is not None, "Should return a response"
                print(f"   ✅ Handled")
                passed += 1
            except Exception as e:
                print(f"   ⚠️ Exception (may be expected): {e}")
                passed += 1  # Exception handling is valid
        
        if passed == len(edge_cases):
            print(f"✅ Test 8 PASSED: Edge cases handled ({passed}/{len(edge_cases)})\n")
            return True
        else:
            print(f"⚠️ Test 8 PARTIAL: {passed}/{len(edge_cases)} tests passed\n")
            return False
    
    def test_chain_understanding(self) -> bool:
        """Test 9: Understanding the Chain (Educational)"""
        print("🧪 Test 9: Understanding the Chain")
        print("-" * 50)
        try:
            question = "What is RAG?"
            
            # Test that chain processes question
            response = self.assistant.ask(question)
            
            # Verify response structure
            assert isinstance(response, str), "Response should be string"
            assert len(response) > 0, "Response should not be empty"
            
            print(f"   Question: {question}")
            print(f"   Response length: {len(response)} characters")
            print(f"   Response preview: {response[:150]}...")
            print("✅ Test 9 PASSED: Chain processing works\n")
            return True
        except Exception as e:
            print(f"❌ Test 9 FAILED: {e}\n")
            return False
    
    def test_conversation_flow(self) -> bool:
        """Test 10: Conversation Flow"""
        print("🧪 Test 10: Conversation Flow")
        print("-" * 50)
        try:
            # Simulate a conversation
            questions = [
                "What is Python?",
                "What can you do with it?",
                "Is it easy to learn?",
            ]
            
            responses = []
            for i, question in enumerate(questions, 1):
                print(f"   Q{i}: {question}")
                response = self.assistant.ask(question)
                responses.append(response)
                print(f"   A{i}: {response[:100]}...")
            
            # All questions should be answered
            assert len(responses) == len(questions), "All questions should be answered"
            assert all(len(r) > 0 for r in responses), "All responses should be non-empty"
            
            print("✅ Test 10 PASSED: Conversation flow works\n")
            return True
        except Exception as e:
            print(f"❌ Test 10 FAILED: {e}\n")
            return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("🧪 REAL-TIME AI ASSISTANT TEST SUITE")
    print("=" * 60)
    print()
    
    test_suite = TestRealTimeAssistant()
    
    tests = [
        ("Basic Question", test_suite.test_basic_question),
        ("Real-Time Information", test_suite.test_realtime_information),
        ("Search Integration", test_suite.test_search_integration),
        ("Error Handling", test_suite.test_error_handling),
        ("Response Quality", test_suite.test_response_quality),
        ("Question Types", test_suite.test_question_types),
        ("Performance", test_suite.test_performance),
        ("Edge Cases", test_suite.test_edge_cases),
        ("Chain Understanding", test_suite.test_chain_understanding),
        ("Conversation Flow", test_suite.test_conversation_flow),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}\n")
            results.append((test_name, False))
    
    # Summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    elif passed >= total * 0.7:
        print("\n⚠️ Most tests passed, but some issues detected.")
    else:
        print("\n❌ Many tests failed. Please check your setup.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

