class MockLLM:
    model = "mock"
    server = "local"

    def generate(self, messages):
        return "MOCK ANSWER: LLM integration placeholder."
