from utils.registry import Registry


"""
Feature Extractor.
"""
# Backbone
BACKBONES = Registry()

# FPN
FPN_BODY = Registry()

"""
ROI Head.
"""
# Box Head
ROI_CLS_HEADS = Registry()
ROI_CLS_OUTPUTS = Registry()
ROI_BOX_HEADS = Registry()
ROI_BOX_OUTPUTS = Registry()

# OPLD Head
ROI_OPLD_HEADS = Registry()
ROI_OPLD_OUTPUTS = Registry()
