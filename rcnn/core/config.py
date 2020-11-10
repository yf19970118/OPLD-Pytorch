import os
import os.path as osp
import copy
import yaml
import numpy as np
from ast import literal_eval

from utils.collections import AttrDict

__C = AttrDict()
cfg = __C

# ---------------------------------------------------------------------------- #
# MISC options
# ---------------------------------------------------------------------------- #
# Device for training or testing
# E.g., 'cuda' for using GPU, 'cpu' for using CPU
__C.DEVICE = 'cuda'

# Number of GPUs to use (applies to both training and testing)
__C.NUM_GPUS = 1

# Pixel mean values (BGR order) as a list
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# Pixel std values (BGR order) as a list
__C.PIXEL_STDS = np.array([[[1.0, 1.0, 1.0]]])

# Clean up the generated files during model testing
__C.CLEAN_UP = True

# Directory for saving checkpoints and loggers
__C.CKPT = 'ckpts/rcnn/mscoco/e2e_faster_rcnn_R-50-FPN_1x'

# Display the log per iteration
__C.DISPLAY_ITER = 20

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# A very small number that's used many times
__C.EPS = 1e-14

# Convert image to BGR format (for Caffe2 models), in range 0-255
__C.TO_BGR255 = True


# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.MODEL = AttrDict()

# The type of model to use, TO REMOVE
# The string must match a function in the modeling.model_builder module
# (e.g., 'generalized_rcnn', ...)
__C.MODEL.TYPE = 'generalized_rcnn'

# FPN is enabled if True
__C.MODEL.FPN_ON = False

# RPN is enabled if True
# Default is True, if RPN_ON = False means that only training the backbone
__C.MODEL.RPN_ON = True

# The meaning of FASTER_RCNN depends on the context (training vs. inference):
# 1) During training, FASTER_ON = True means that end-to-end training will be
#    used to jointly train the RPN subnetwork and the Fast R-CNN subnetwork
#    (Faster R-CNN = RPN + Fast R-CNN).
# 2) During inference, FASTER_ON = True means that the model's RPN subnetwork
#    will be used to generate proposals rather than relying on precomputed
#    proposals. Note that FASTER_ON = True can be used at inference time even
#    if the Faster R-CNN model was trained with stagewise training (which
#    consists of alternating between RPN and Fast R-CNN training in a way that
#    finally leads to a single network).
__C.MODEL.FASTER_ON = False

# Indicates the model uses H_GRID R-CNN
__C.MODEL.OPLD_ON = False

# Indicates the model contains any kind of roihead (except Faster R-CNN)
__C.MODEL.HAS_ROIHEAD = False   # NEED UPDATE

# Indicates the model contains any kind of box detection (except Grid R-CNN)
__C.MODEL.HAS_BOX = False   # NEED UPDATE

# Type of batch normalizaiton, default: 'freeze'
# E.g., 'normal', 'freeze', 'sync', ...
__C.MODEL.BATCH_NORM = 'freeze'

# Number of classes in the dataset; must be set
# E.g., 81 for COCO (80 foreground + 1 background)
__C.MODEL.NUM_CLASSES = -1

# Swap model conv1 weight, for pet/rcnn we use BGR input channel (cv2), for pet/cls we use RGB channel,
# for caffe/caffe2 model using BGR channel. Thus if we use pet pretrain weights set 'True', else if use
# caffe or caffe2 weights set 'False'.
__C.MODEL.CONV1_RGB2BGR = True


# ---------------------------------------------------------------------------- #
# Solver options
# Note: all solver options are used exactly as specified; the implication is
# that if you switch from training on 1 GPU to N GPUs, you MUST adjust the
# solver configuration accordingly. We suggest using gradual warmup and the
# linear learning rate scaling rule as described in
# "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" Goyal et al.
# https://arxiv.org/abs/1706.02677
# ---------------------------------------------------------------------------- #
__C.SOLVER = AttrDict()

# Type of the optimizer
# E.g., 'SGD', 'RMSPROP', 'ADAM' ...
__C.SOLVER.OPTIMIZER = 'SGD'

# Base learning rate for the specified schedule
__C.SOLVER.BASE_LR = 0.001

# Maximum number of max iterations
__C.SOLVER.MAX_ITER = 90000

# Momentum to use with SGD
__C.SOLVER.MOMENTUM = 0.9

# L2 regularization hyperparameter
__C.SOLVER.WEIGHT_DECAY = 0.0005

# L2 regularization hyperparameter for GroupNorm's parameters
__C.SOLVER.WEIGHT_DECAY_GN = 0.0

# Whether to double the learning rate for bias
__C.SOLVER.BIAS_DOUBLE_LR = True

# Whether to have weight decay on bias as well
__C.SOLVER.BIAS_WEIGHT_DECAY = False

# Multiple learning rate for fine-tuning
# Random initial layer learning rate is LR_MULTIPLE * BASE_LR
__C.SOLVER.LR_MULTIPLE = 1.0  # TODO

# Warm up to SOLVER.BASE_LR over this number of SGD iterations
__C.SOLVER.WARM_UP_ITERS = 500

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARM_UP_FACTOR
__C.SOLVER.WARM_UP_FACTOR = 1.0 / 10.0

# WARM_UP_METHOD can be either 'CONSTANT' or 'LINEAR' (i.e., gradual)
__C.SOLVER.WARM_UP_METHOD = 'LINEAR'

# Schedule type (see functions in utils.lr_policy for options)
# E.g., 'POLY', 'STEP', 'COSINE', ...
__C.SOLVER.LR_POLICY = 'STEP'

# For 'POLY', the power in poly to drop LR
__C.SOLVER.LR_POW = 0.9

# For 'STEP', Non-uniform step iterations
__C.SOLVER.STEPS = [60000, 80000]

# For 'STEP', the current LR is multiplied by SOLVER.GAMMA at each step
__C.SOLVER.GAMMA = 0.1

# Suppress logging of changes to LR unless the relative change exceeds this
# threshold (prevents linear warm up from spamming the training log)
__C.SOLVER.LOG_LR_CHANGE_THRESHOLD = 1.1

# Snapshot (model checkpoint) period
__C.SOLVER.SNAPSHOT_ITERS = 10000


# -----------------------------------------------------------------------------
# DataLoader options
# -----------------------------------------------------------------------------
__C.DATALOADER = AttrDict()

# Type of training sampler, default: 'DistributedSampler'
# E.g., 'DistributedSampler', 'RepeatFactorTrainingSampler', ...
__C.DATALOADER.SAMPLER_TRAIN = "DistributedSampler"

# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
__C.DATALOADER.ASPECT_RATIO_GROUPING = True

# if True, the dataloader will filter out images that have no associated
# annotations at train time.
__C.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True  # TODO

# ---------------------------------------------------------------------------- #
# RepeatFactorTrainingSampler options
# ---------------------------------------------------------------------------- #
__C.DATALOADER.RFTSAMPLER = AttrDict()

# parameters for RepeatFactorTrainingSampler
# rep_times = max(MIN_REPEAT_TIMES, min(MAX_REPEAT_TIMES, math.pow((REPEAT_THRESHOLD / cat_freq),POW)))
__C.DATALOADER.RFTSAMPLER.REPEAT_THRESHOLD = 0.001
__C.DATALOADER.RFTSAMPLER.POW = 0.5
__C.DATALOADER.RFTSAMPLER.MAX_REPEAT_TIMES = 10000.0
__C.DATALOADER.RFTSAMPLER.MIN_REPEAT_TIMES = 1.0


# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()

# Initialize network with weights from this .pkl file
__C.TRAIN.WEIGHTS = ''

# Type of training data augmentation, default: 'none'
# E.g., 'none', 'random_crop', ...
__C.TRAIN.PREPROCESS_TYPE = 'none'

# Datasets to train on
# Available dataset list: datasets.dataset_catalog.DATASETS.keys()
# If multiple datasets are listed, the model is trained on their union
__C.TRAIN.DATASETS = ()

# Scales to use during training
# Each scale is the pixel size of an image's shortest side
# If multiple scales are listed, then one is selected uniformly at random for
# each training image (i.e., scale jitter data augmentation)
__C.TRAIN.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000

# Number of Python threads to use for the data loader during training
__C.TRAIN.LOADER_THREADS = 4

# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
__C.TRAIN.SIZE_DIVISIBILITY = 32

# Mini-batch size for training
# This is global, so if we have 8 GPUs and BATCH_SIZE = 16, each GPU will
# see 2 images per batch
__C.TRAIN.BATCH_SIZE = 16

# Freeze the backbone architecture during training if set to True
__C.TRAIN.FREEZE_CONV_BODY = False

# Training will resume from the latest snapshot (model checkpoint) found in the
# output directory
__C.TRAIN.AUTO_RESUME = True

# Image ColorJitter Augmentation
__C.TRAIN.BRIGHTNESS = 0.0
__C.TRAIN.CONTRAST = 0.0
__C.TRAIN.SATURATION = 0.0
__C.TRAIN.HUE = 0.0

# Left right mapping for flipping training
__C.TRAIN.LEFT_RIGHT = ()

# Save training metrics as json file
__C.TRAIN.SAVE_AS_JSON = False

# Use tensorboard to show training progress
__C.TRAIN.USE_TENSORBOARD = False

# DOTA category_id for change order
__C.TRAIN.CHANGE_CATEGORIES = [3, 4, 8, 9, 10, 11, 12, 14]

# ---------------------------------------------------------------------------- #
# Inference ('test') options
# ---------------------------------------------------------------------------- #
__C.TEST = AttrDict()

# Initialize network with weights from this .pkl file
__C.TEST.WEIGHTS = ''

# Number of Python threads to use for the data loader during testing
__C.TEST.LOADER_THREADS = 4

# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
__C.TEST.SIZE_DIVISIBILITY = 32

# Datasets to test on
# Available dataset list: datasets.dataset_catalog.DATASETS.keys()
# If multiple datasets are listed, testing is performed on each one sequentially
__C.TEST.DATASETS = ()

# Scale to use during testing (can NOT list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TEST.SCALE = 600

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Number of images in each GPU for testing
__C.TEST.IMS_PER_GPU = 1

# If True, force resize the image to [H, W].
__C.TEST.FORCE_TEST_SCALE = [-1, -1]

# ---------------------------------------------------------------------------- #
# Soft NMS (from the Soft NMS paper)
# ---------------------------------------------------------------------------- #
__C.TEST.SOFT_NMS = AttrDict()

# Use soft NMS instead of standard NMS if set to True
__C.TEST.SOFT_NMS.ENABLED = False

# See soft NMS paper for definition of these options
__C.TEST.SOFT_NMS.METHOD = 'linear'

# For the soft NMS overlap threshold, we simply use TEST.NMS
__C.TEST.SOFT_NMS.SIGMA = 0.5

# ---------------------------------------------------------------------------- #
# Bounding box voting (from the Multi-Region CNN paper)
# ---------------------------------------------------------------------------- #
__C.TEST.BBOX_VOTE = AttrDict()

# Use box voting if set to True
__C.TEST.BBOX_VOTE.ENABLED = False

# We use TEST.NMS threshold for the NMS step. VOTE_TH overlap threshold
# is used to select voting boxes (IoU >= VOTE_TH) for each box that survives NMS
__C.TEST.BBOX_VOTE.VOTE_TH = 0.8

# The method used to combine scores when doing bounding box voting
# Valid options include ('ID', 'AVG', 'IOU_AVG', 'GENERALIZED_AVG', 'QUASI_SUM')
__C.TEST.BBOX_VOTE.SCORING_METHOD = 'ID'

# Hyperparameter used by the scoring method (it has different meanings for
# different methods)
__C.TEST.BBOX_VOTE.SCORING_METHOD_BETA = 1.0

# ---------------------------------------------------------------------------- #
# Test-time augmentations for bounding box detection
# ---------------------------------------------------------------------------- #
__C.TEST.BBOX_AUG = AttrDict()

# Enable test-time augmentation for bounding box detection if True
__C.TEST.BBOX_AUG.ENABLED = False

# Horizontal flip at the original scale (id transform)
__C.TEST.BBOX_AUG.H_FLIP = False

# Each scale is the pixel size of an image's shortest side
__C.TEST.BBOX_AUG.SCALES = ()

# Max pixel size of the longer side
__C.TEST.BBOX_AUG.MAX_SIZE = 4000

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
__C.BACKBONE = AttrDict()

# The backbone conv body to use
__C.BACKBONE.CONV_BODY = 'resnet'

# The eps of batch_norm layer
__C.BACKBONE.BN_EPS = 1e-5

# ---------------------------------------------------------------------------- #
# ResNet options
# ---------------------------------------------------------------------------- #
__C.BACKBONE.RESNET = AttrDict()

# The number of layers in each block
# (2, 2, 2, 2) for resnet18 with basicblock
# (3, 4, 6, 3) for resnet34 with basicblock
# (3, 4, 6, 3) for resnet50
# (3, 4, 23, 3) for resnet101
# (3, 8, 36, 3) for resnet152
__C.BACKBONE.RESNET.LAYERS = (3, 4, 6, 3)

# Network stem width
__C.BACKBONE.RESNET.STEM_WIDTH = 64

# Network initial width
__C.BACKBONE.RESNET.WIDTH = 64

# Use bottleneck block, False for basicblock
__C.BACKBONE.RESNET.BOTTLENECK = True

# Place the stride 2 conv on the 3x3 filter.
# True for resnet-b
__C.BACKBONE.RESNET.STRIDE_3X3 = False

# Use a three (3 * 3) kernels head; False for (7 * 7) kernels head.
# True for resnet-c
__C.BACKBONE.RESNET.USE_3x3x3HEAD = False

# Use a (2 * 2) kernels avg_pooling layer in downsampling block.
# True for resnet-d
__C.BACKBONE.RESNET.AVG_DOWN = False

# Use SplAtConv2d in bottleneck.
# 2 for resnest
__C.BACKBONE.RESNET.RADIX = 1

# Use group normalization
__C.BACKBONE.RESNET.USE_GN = False

# Use attentive normalization
# when it is True means use an_bn (an with bn)
# when it is True and USE_GN is True means use an_gn (an with gn)
__C.BACKBONE.RESNET.USE_AN = False

# Use weight standardization
__C.BACKBONE.RESNET.USE_WS = False

# Use a aligned module in each block
__C.BACKBONE.RESNET.USE_ALIGN = False

# Type of context module in each block
# 'se' for se, 'gcb' for gcb
__C.BACKBONE.RESNET.STAGE_WITH_CONTEXT = ('none', 'none', 'none', 'none')

# Context module innerplanes ratio
__C.BACKBONE.RESNET.CTX_RATIO = 0.0625

# Type of 3x3 convolution layer in each block
# 'deform' for dcnv1, 'deformv2' for dcnv2
__C.BACKBONE.RESNET.STAGE_WITH_CONV = ('normal', 'normal', 'normal', 'normal')

# Apply dilation in stage "c5"
__C.BACKBONE.RESNET.C5_DILATION = 1

# Freeze model weights before and including which block.
# Choices: [0, 2, 3, 4, 5]. O means not fixed. First conv and bn are defaults to
# be fixed.
__C.BACKBONE.RESNET.FREEZE_AT = 2

# ---------------------------------------------------------------------------- #
# ResNeXt options
# ---------------------------------------------------------------------------- #
__C.BACKBONE.RESNEXT = AttrDict()

# The number of layers in each block
# (3, 4, 6, 3) for resnext50
# (3, 4, 23, 3) for resnext101
# (3, 8, 36, 3) for resnext152
__C.BACKBONE.RESNEXT.LAYERS = (3, 4, 6, 3)

# Cardinality (groups) of convolution layers
__C.BACKBONE.RESNEXT.C = 32

# Network initial width of each (conv) group
__C.BACKBONE.RESNEXT.WIDTH = 4

# Use a three (3 * 3) kernels head; False for (7 * 7) kernels head.
# True for resnext-c
__C.BACKBONE.RESNEXT.USE_3x3x3HEAD = False

# Use a (2 * 2) kernels avg_pooling layer in downsampling block.
# True for resnext-d
__C.BACKBONE.RESNEXT.AVG_DOWN = False

# Use group normalization
__C.BACKBONE.RESNEXT.USE_GN = False

# Use weight standardization
__C.BACKBONE.RESNEXT.USE_WS = False

# Use a aligned module in each block
__C.BACKBONE.RESNEXT.USE_ALIGN = False

# Type of context module in each block
# 'se' for se, 'gcb' for gcb
__C.BACKBONE.RESNEXT.STAGE_WITH_CONTEXT = ('none', 'none', 'none', 'none')

# Context module innerplanes ratio
__C.BACKBONE.RESNEXT.CTX_RATIO = 0.0625

# Type of 3x3 convolution layer in each block
# 'deform' for dcnv1, 'deformv2' for dcnv2
__C.BACKBONE.RESNEXT.STAGE_WITH_CONV = ('normal', 'normal', 'normal', 'normal')

# Apply dilation in stage "c5"
__C.BACKBONE.RESNEXT.C5_DILATION = 1

# Freeze model weights before and including which block.
# Choices: [0, 2, 3, 4, 5]. O means not fixed. First conv and bn are defaults to
# be fixed.
__C.BACKBONE.RESNEXT.FREEZE_AT = 2


# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
__C.FPN = AttrDict()

# The Body of FPN to use
# (e.g., "fpn", "hrfpn", "nasfpn", "deconvx")
__C.FPN.BODY = "fpn"

# Use C5 or P5 to generate P6
__C.FPN.USE_C5 = True

# Channel dimension of the FPN feature levels
__C.FPN.DIM = 256

# FPN may be used for just RPN, just object detection, or both
# E.g., "conv2"-like level
__C.FPN.LOWEST_BACKBONE_LVL = 2

# E.g., "conv5"-like level
__C.FPN.HIGHEST_BACKBONE_LVL = 5

# Use FPN for RoI transform for object detection if True
__C.FPN.MULTILEVEL_ROIS = True

# Hyperparameters for the RoI-to-FPN level mapping heuristic
__C.FPN.ROI_CANONICAL_SCALE = 224  # s0  # TODO

__C.FPN.ROI_CANONICAL_LEVEL = 4  # k0: where s0 maps to  # TODO

# Coarsest level of the FPN pyramid
__C.FPN.ROI_MAX_LEVEL = 5

# Finest level of the FPN pyramid
__C.FPN.ROI_MIN_LEVEL = 2

# Use FPN for RPN if True
__C.FPN.MULTILEVEL_RPN = True

# Coarsest level of the FPN pyramid
__C.FPN.RPN_MAX_LEVEL = 6

# Finest level of the FPN pyramid
__C.FPN.RPN_MIN_LEVEL = 2

# Use extra FPN levels, as done in the RetinaNet paper
__C.FPN.EXTRA_CONV_LEVELS = False

# Use FPN Lite (dwconv) to replace standard FPN
__C.FPN.USE_LITE = False

# Use BatchNorm in the FPN-specific layers (lateral, etc.)
__C.FPN.USE_BN = False

# Use GroupNorm in the FPN-specific layers (lateral, etc.)
__C.FPN.USE_GN = False

# Use Weight Standardization in the FPN-specific layers (lateral, etc.)
__C.FPN.USE_WS = False


# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
__C.RPN = AttrDict()

# Indicates the model's computation terminates with the production of RPN
# proposals (i.e., it outputs proposals ONLY, no actual object detections)
__C.RPN.RPN_ONLY = False

# Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
__C.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)

# Stride of the feature map that RPN is attached.
# For FPN, number of strides should match number of scales
__C.RPN.ANCHOR_STRIDE = (16,)

# RPN anchor aspect ratios
__C.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)

# Remove RPN anchors that go outside the image by RPN_BOUNDARY_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
__C.RPN.BOUNDARY_THRESH = -1

# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example)
__C.RPN.FG_IOU_THRESHOLD = 0.7

# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example)
__C.RPN.BG_IOU_THRESHOLD = 0.3

# Total number of RPN examples per image
__C.RPN.BATCH_SIZE_PER_IMAGE = 256

# Target fraction of foreground (positive) examples per RPN minibatch
__C.RPN.POSITIVE_FRACTION = 0.5

# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
__C.RPN.PRE_NMS_TOP_N_TRAIN = 12000

__C.RPN.PRE_NMS_TOP_N_TEST = 6000

# Number of top scoring RPN proposals to keep after applying NMS
__C.RPN.POST_NMS_TOP_N_TRAIN = 2000

__C.RPN.POST_NMS_TOP_N_TEST = 2000

# NMS threshold used on RPN proposals
__C.RPN.NMS_THRESH = 0.7

# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (a the scale used during training or inference)
__C.RPN.MIN_SIZE = 0

# Number of top scoring RPN proposals to keep after combining proposals from
# all FPN levels
__C.RPN.FPN_POST_NMS_TOP_N_TRAIN = 2000

__C.RPN.FPN_POST_NMS_TOP_N_TEST = 2000

# Apply the post NMS per batch (default) or per image during training
# (default is True to be consistent with Detectron, see Issue #672)
__C.RPN.FPN_POST_NMS_PER_BATCH = True

# Custom rpn head, empty to use default conv or separable conv
__C.RPN.RPN_HEAD = "SingleConvRPNHead"  # TODO

# The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
__C.RPN.SMOOTH_L1_BETA = 1.0 / 9

# Set 0 by default
__C.RPN.ANCHOR_OFFSET = 0.0

# ---------------------------------------------------------------------------- #
# Fast R-CNN options
# ---------------------------------------------------------------------------- #
__C.FAST_RCNN = AttrDict()

# The head of Fast R-CNN to use
# (e.g., "roi_2mlp_head", "roi_convx_head")
__C.FAST_RCNN.ROI_BOX_HEAD = "roi_2mlp_head"

# Output module of Fast R-CNN head
__C.FAST_RCNN.ROI_BOX_OUTPUT = "box_output"

# RoI transformation function (e.g., ROIPool or ROIAlign or ROIAlignV2)
__C.FAST_RCNN.ROI_XFORM_METHOD = 'ROIAlign'

# Loss type of Fast R-CNN head (e.g., cross_entropy or eql)
__C.FAST_RCNN.ROI_CLS_LOSS = "cross_entropy"

# Number of grid sampling points in ROIAlign (usually use 2)
# Only applies to ROIAlign
__C.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO = 2

# RoI transform output resolution
# Note: some models may have constraints on what they can use, e.g. they use
# pretrained FC layers like in VGG16, and will ignore this option
__C.FAST_RCNN.ROI_XFORM_RESOLUTION = (14, 14)

# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
__C.FAST_RCNN.FG_IOU_THRESHOLD = 0.5

# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
__C.FAST_RCNN.BG_IOU_THRESHOLD = 0.5

# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
__C.FAST_RCNN.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)

# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH
# E.g., a common configuration is: 512 * 2 * 8 = 8192
__C.FAST_RCNN.BATCH_SIZE_PER_IMAGE = 512

# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
__C.FAST_RCNN.POSITIVE_FRACTION = 0.25

# Use a class agnostic bounding box regressor instead of the default per-class
# regressor
__C.FAST_RCNN.CLS_AGNOSTIC_BBOX_REG = False

# Prior prob for the positives at the beginning of training. This is used to set
# the bias init for the logits layer
__C.FAST_RCNN.PRIOR_PROB = 0.0

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
__C.FAST_RCNN.SCORE_THRESH = 0.05

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.FAST_RCNN.NMS_TH = 0.5

# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
__C.FAST_RCNN.DETECTIONS_PER_IMG = 100

# The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
__C.FAST_RCNN.SMOOTH_L1_BETA = 1.0

# Classifier branch switch
__C.FAST_RCNN.CLS_ON = True

# Box regression branch switch
__C.FAST_RCNN.REG_ON = True

# ---------------------------------------------------------------------------- #
# Fast R-CNN mlp head options
# ---------------------------------------------------------------------------- #
__C.FAST_RCNN.MLP_HEAD = AttrDict()

# Hidden layer dimension when using an MLP for the RoI box head
__C.FAST_RCNN.MLP_HEAD.MLP_DIM = 1024

# Use BatchNorm in the Fast R-CNN mlp head
__C.FAST_RCNN.MLP_HEAD.USE_BN = False

# Use GroupNorm in the Fast R-CNN mlp head
__C.FAST_RCNN.MLP_HEAD.USE_GN = False

# Use Weight Standardization in the Fast R-CNN mlp head
__C.FAST_RCNN.MLP_HEAD.USE_WS = False

# ---------------------------------------------------------------------------- #
# Fast R-CNN convfc head options
# ---------------------------------------------------------------------------- #
__C.FAST_RCNN.CONVFC_HEAD = AttrDict()

# Dilation
__C.FAST_RCNN.CONVFC_HEAD.DILATION = 1

# Hidden Conv layer dimension when using Convs for the RoI box head
__C.FAST_RCNN.CONVFC_HEAD.CONV_DIM = 256

# Number of stacked Conv layers in the RoI box head
__C.FAST_RCNN.CONVFC_HEAD.NUM_STACKED_CONVS = 4

# Hidden layer dimension when using an MLP for the RoI box head
__C.FAST_RCNN.CONVFC_HEAD.MLP_DIM = 1024

# Use Fast R-CNN Lite (dwconv) to replace standard Fast R-CNN
__C.FAST_RCNN.CONVFC_HEAD.USE_LITE = False

# Use BatchNorm in the Fast R-CNN convfc head
__C.FAST_RCNN.CONVFC_HEAD.USE_BN = False

# Use GroupNorm in the Fast R-CNN convfc head
__C.FAST_RCNN.CONVFC_HEAD.USE_GN = False

# Use Weight Standardization in the Fast R-CNN convfc head
__C.FAST_RCNN.CONVFC_HEAD.USE_WS = False

# ---------------------------------------------------------------------------- #
# OPLD options
# ---------------------------------------------------------------------------- #
__C.OPLD = AttrDict()

# The head of H-Grid to use
__C.OPLD.ROI_OPLD_HEAD = "roi_opld_head"

# Output module of H-Grid head
__C.OPLD.ROI_OPLD_OUTPUT = "opld_output"

# Whether to use center post processing
__C.OPLD.USE_CPP = False

# RoI transformation function and associated options
__C.OPLD.ROI_XFORM_METHOD = 'ROIAlign'

# Number of grid sampling points in ROIAlign (usually use 2) Only applies to ROIAlign
__C.OPLD.ROI_XFORM_SAMPLING_RATIO = 2

# RoI transformation function (e.g., ROIPool or ROIAlign)
__C.OPLD.ROI_XFORM_RESOLUTION = (14, 14)

# Loss weight in Grid R-CNN
__C.OPLD.LOSS_WEIGHT = 15

# Make gt points as cruciform targets
__C.OPLD.POS_RADIUS = 2

# Number of grid in Grid R-CNN
__C.OPLD.NUM_POINTS = 4

# The size of feature after pooling in Grid R-CNN
__C.OPLD.ROI_FEAT_SIZE = 14

__C.OPLD.SCORE_WEIGHTS = (1, 0)

__C.OPLD.NMS_THRESH = 0.2

__C.OPLD.DETECTIONS_PER_IMG = 200

# ---------------------------------------------------------------------------- #
# OPLD head options
# ---------------------------------------------------------------------------- #
__C.OPLD.ROI_HEAD = AttrDict()

# Number of convolution in Grid R-CNN head
__C.OPLD.ROI_HEAD.NUM_CONVS = 8

# Number of Group Nrom's groups
__C.OPLD.ROI_HEAD.GN_GOURPS = 32

# Number of channels to represent one point in Grid R-CNN head
__C.OPLD.ROI_HEAD.POINT_FEAT_CHANNELS = 64

# Neighbor_points for each grid point to use feature fusion
__C.OPLD.ROI_HEAD.NEIGHBOR_POINTS = ((1, 3), (0, 2), (1, 3), (0, 2), (4,))


# ---------------------------------------------------------------------------- #
# Visualization options
# ---------------------------------------------------------------------------- #
__C.VIS = AttrDict()

# Dump detection visualizations
__C.VIS.ENABLED = False

# Score threshold for visualization
__C.VIS.VIS_TH = 0.9

# ---------------------------------------------------------------------------- #
# Show box options
# ---------------------------------------------------------------------------- #
__C.VIS.SHOW_QUAD_BOX = AttrDict()

# Visualizing detection bboxes
__C.VIS.SHOW_QUAD_BOX.ENABLED = False

__C.VIS.SHOW_QUAD_BOX.BORDER_THICK = 2

__C.VIS.SHOW_QUAD_BOX.CENTER_RADIUS = 2

# ---------------------------------------------------------------------------- #
# Show class options
# ---------------------------------------------------------------------------- #
__C.VIS.SHOW_CLASS = AttrDict()

# Visualizing detection classes
__C.VIS.SHOW_CLASS.ENABLED = True

# Default: gray
__C.VIS.SHOW_CLASS.COLOR = (218, 227, 218)

# Font scale of class string
__C.VIS.SHOW_CLASS.FONT_SCALE = 0.45


# ---------------------------------------------------------------------------- #
# Deprecated options
# If an option is removed from the code and you don't want to break existing
# yaml configs, you can add the full config key as a string to the set below.
# ---------------------------------------------------------------------------- #
_DEPCRECATED_KEYS = set()


# ---------------------------------------------------------------------------- #
# Renamed options
# If you rename a config option, record the mapping from the old name to the new
# name in the dictionary below. Optionally, if the type also changed, you can
# make the value a tuple that specifies first the renamed key and then
# instructions for how to edit the config file.
# ---------------------------------------------------------------------------- #
_RENAMED_KEYS = {
    'EXAMPLE.RENAMED.KEY': 'EXAMPLE.KEY',  # Dummy example to follow
    'PIXEL_MEAN': 'PIXEL_MEANS',
    'PIXEL_STD': 'PIXEL_STDS',
}


def assert_and_infer_cfg(make_immutable=True):
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg settings
    during script execution (which can lead to hard to debug errors or code
    that's harder to understand than is necessary).
    """
    if __C.MODEL.OPLD_ON:
        __C.MODEL.HAS_ROIHEAD = True
    if __C.MODEL.FASTER_ON:
        __C.MODEL.HAS_BOX = True

    if make_immutable:
        cfg.immutable(True)


def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)


def merge_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        if _key_is_deprecated(full_key):
            continue
        if _key_is_renamed(full_key):
            _raise_key_rename_error(full_key)
        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, str):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a


def _key_is_deprecated(full_key):
    if full_key in _DEPCRECATED_KEYS:
        return True
    return False


def _key_is_renamed(full_key):
    return full_key in _RENAMED_KEYS


def _raise_key_rename_error(full_key):
    new_key = _RENAMED_KEYS[full_key]
    if isinstance(new_key, tuple):
        msg = ' Note: ' + new_key[1]
        new_key = new_key[0]
    else:
        msg = ''
    raise KeyError(
        'Key {} was renamed to {}; please update your config.{}'.
            format(full_key, new_key, msg)
    )
