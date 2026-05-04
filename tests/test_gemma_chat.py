from geometric_complexity_scaling import modeling
from geometric_complexity_scaling.modeling import apply_gemma_chat_template


class FakeProcessor:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "formatted prompt"


def test_chat_template_disables_thinking():
    processor = FakeProcessor()
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    assert apply_gemma_chat_template(processor, messages) == "formatted prompt"
    assert processor.kwargs["tokenize"] is False
    assert processor.kwargs["add_generation_prompt"] is True
    assert processor.kwargs["enable_thinking"] is False


class FakeProcessorWithoutThinkingArg:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        if "enable_thinking" in kwargs:
            raise TypeError("unexpected keyword")
        self.messages = messages
        self.kwargs = kwargs
        return "formatted prompt"


def test_chat_template_falls_back_when_model_template_lacks_enable_thinking():
    processor = FakeProcessorWithoutThinkingArg()
    messages = [{"role": "user", "content": "hi"}]
    assert apply_gemma_chat_template(processor, messages) == "formatted prompt"
    assert processor.kwargs == {"tokenize": False, "add_generation_prompt": True}


def test_chat_template_falls_back_to_plain_prompt_without_template_support():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    prompt = apply_gemma_chat_template(object(), messages)
    assert prompt == "System: sys\n\nUser: hi\n\nAssistant:"


def test_cpu_device_map_is_handled_without_accelerate():
    assert modeling._manual_device_for_device_map("cpu").type == "cpu"
    assert modeling._should_pass_device_map("cpu") is False


def test_generated_token_slicing_contract():
    sequence = [10, 11, 12, 99, 100]
    input_len = 3
    assert sequence[input_len:] == [99, 100]
