from transformers import ViTForImageClassification, ResNetForImageClassification, VanForImageClassification
from transformers import SwinForImageClassification, BeitForImageClassification, ConvNextForImageClassification
from transformers import DeiTForImageClassification, Data2VecVisionForImageClassification, DeiTForImageClassificationWithTeacher

_CLASSIFIER = {}


def _add_classifier(classifier_fn):
    _CLASSIFIER[classifier_fn.__name__] = classifier_fn
    return classifier_fn

@_add_classifier
def resnetforimageclassification(model_name):
    return ResNetForImageClassification.from_pretrained(model_name)

@_add_classifier
def vitforimageclassification(model_name):
    return ViTForImageClassification.from_pretrained(model_name)

@_add_classifier
def vanforimageclassification(model_name):
    return VanForImageClassification.from_pretrained(model_name)

@_add_classifier
def swinforimageclassification(model_name):
    return SwinForImageClassification.from_pretrained(model_name)

@_add_classifier
def beitforimageclassification(model_name):
    return BeitForImageClassification.from_pretrained(model_name)

@_add_classifier
def convnextforimageclassification(model_name):
    return ConvNextForImageClassification.from_pretrained(model_name)

@_add_classifier
def deitforimageclassification(model_name):
    return DeiTForImageClassification.from_pretrained(model_name)

@_add_classifier
def data2vecvisionforimageclassification(model_name):
    return Data2VecVisionForImageClassification.from_pretrained(model_name)

@_add_classifier
def deitforimageclassificationwithteacher(model_name):
    return DeiTForImageClassificationWithTeacher.from_pretrained(model_name)

