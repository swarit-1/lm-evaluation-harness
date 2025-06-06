from lm_eval.base import Task, rf
import datasets

_CITATION = """
@misc{logianwilliam2024misleading,
    title={Misleading Context Prompts},
    author={Logan William},
    year={2024},
    howpublished={\\url{https://huggingface.co/datasets/logianwilliam/misleading_context_prompts}}
}
"""

def doc_to_text(doc):
    return doc["prompt"]

def doc_to_target(doc):
    return doc["correct_answer"]

def doc_to_choice(doc):
    return doc["choices"]

class MisleadingContext(Task):
    VERSION = 0
    DATASET_PATH = "logianwilliam/misleading_context_prompts"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return doc_to_text(doc)

    def doc_to_target(self, doc):
        return doc_to_target(doc)

    def doc_to_choice(self, doc):
        return doc_to_choice(doc)

    def fewshot_description(self):
        return ""
