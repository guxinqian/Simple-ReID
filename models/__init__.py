from models.Classifier import Classifier, NormalizedClassifier
from models.ResNet import ResNet50


def build_model(config, num_classes):
	# Build backbone
	print("Initializing model: {}".format(config.MODEL.NAME))
	if config.MODEL.NAME == 'resnet50':
		model = ResNet50(res4_stride=config.MODEL.RES4_STRIDE)
	else:
		raise KeyError("Invalid model: '{}'".format(config.MODEL.NAME))
	print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

	# Build classifier
	if config.LOSS.CLA_LOSS in ['crossentropy', 'crossentropylabelsmooth']:
		classifier = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_classes)
	else:
		classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_classes)

	return model, classifier    