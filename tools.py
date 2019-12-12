import torch
import torch.nn as nn
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class BBoxTransform(nn.Module):

  def __init__(self, mean=None, std=None):
    super(BBoxTransform, self).__init__()
    if mean is None:
      self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
    else:
      self.mean = mean
    if std is None:
      self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
    else:
      self.std = std

  def forward(self, boxes, deltas):

    widths = boxes[:, :, 2] - boxes[:, :, 0]
    heights = boxes[:, :, 3] - boxes[:, :, 1]
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
    dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
    dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
    dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
    pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
    pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
    pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

    pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

    return pred_boxes


class ClipBoxes(nn.Module):

  def __init__(self, width=None, height=None):
    super(ClipBoxes, self).__init__()

  def forward(self, boxes, img):
    batch_size, num_channels, height, width = img.shape

    boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
    boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

    boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
    boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

    return boxes


def compute_overlap(a, b):
  area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

  iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
  ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

  iw = np.maximum(iw, 0)
  ih = np.maximum(ih, 0)

  ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

  ua = np.maximum(ua, np.finfo(float).eps)

  intersection = iw * ih

  return intersection / ua


def _compute_ap(recall, precision):
  mrec = np.concatenate(([0.], recall, [1.]))
  mpre = np.concatenate(([0.], precision, [0.]))

  for i in range(mpre.size - 1, 0, -1):
    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

  i = np.where(mrec[1:] != mrec[:-1])[0]

  ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


def _get_detections(dataset, retinanet, score_threshold=0.05, max_detections=100, save_path=None):
  all_detections = [[None for i in range(dataset.num_classes())] for j in range(len(dataset))]

  retinanet.eval()

  with torch.no_grad():

    for index in range(len(dataset)):
      data = dataset[index]
      scale = data['scale']

      scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
      scores = scores.cpu().numpy()
      labels = labels.cpu().numpy()
      boxes = boxes.cpu().numpy()

      boxes /= scale

      indices = np.where(scores > score_threshold)[0]
      if indices.shape[0] > 0:
        # select those scores
        scores = scores[indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes = boxes[indices[scores_sort], :]
        image_scores = scores[scores_sort]
        image_labels = labels[indices[scores_sort]]
        image_detections = np.concatenate(
          [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        for label in range(dataset.num_classes()):
          all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
      else:
        for label in range(dataset.num_classes()):
          all_detections[index][label] = np.zeros((0, 5))

      print('{}/{}'.format(index + 1, len(dataset)), end='\r')

  return all_detections


def _get_annotations(generator):
  all_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

  for i in range(len(generator)):
    # load the annotations
    annotations = generator.load_annotations(i)

    # copy detections to all_annotations
    for label in range(generator.num_classes()):
      all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

    print('{}/{}'.format(i + 1, len(generator)), end='\r')

  return all_annotations


def evaluate(
  generator,
  retinanet,
  iou_threshold=0.5,
  score_threshold=0.05,
  max_detections=100,
  save_path=None
):
  """ Evaluate a given dataset using a given retinanet.
  # Arguments
      generator       : The generator that represents the dataset to evaluate.
      retinanet           : The retinanet to evaluate.
      iou_threshold   : The threshold used to consider when a detection is positive or negative.
      score_threshold : The score confidence threshold to use for detections.
      max_detections  : The maximum number of detections to use per image.
      save_path       : The path to save images with visualized detections to.
  # Returns
      A dict mapping class names to mAP scores.
  """

  # gather all detections and annotations

  all_detections = _get_detections(generator, retinanet, score_threshold=score_threshold, max_detections=max_detections,
                                   save_path=save_path)
  all_annotations = _get_annotations(generator)

  average_precisions = {}

  for label in range(generator.num_classes()):
    false_positives = np.zeros((0,))
    true_positives = np.zeros((0,))
    scores = np.zeros((0,))
    num_annotations = 0.0

    for i in range(len(generator)):
      detections = all_detections[i][label]
      annotations = all_annotations[i][label]
      num_annotations += annotations.shape[0]
      detected_annotations = []

      for d in detections:
        scores = np.append(scores, d[4])

        if annotations.shape[0] == 0:
          false_positives = np.append(false_positives, 1)
          true_positives = np.append(true_positives, 0)
          continue

        overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
        assigned_annotation = np.argmax(overlaps, axis=1)
        max_overlap = overlaps[0, assigned_annotation]

        if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
          false_positives = np.append(false_positives, 0)
          true_positives = np.append(true_positives, 1)
          detected_annotations.append(assigned_annotation)
        else:
          false_positives = np.append(false_positives, 1)
          true_positives = np.append(true_positives, 0)

    # no annotations -> AP for this class is 0 (is this correct?)
    if num_annotations == 0:
      average_precisions[label] = 0, 0
      continue

    # sort by score
    indices = np.argsort(-scores)
    false_positives = false_positives[indices]
    true_positives = true_positives[indices]

    # compute false positives and true positives
    false_positives = np.cumsum(false_positives)
    true_positives = np.cumsum(true_positives)

    # compute recall and precision
    recall = true_positives / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

    # compute average precision
    average_precision = _compute_ap(recall, precision)
    average_precisions[label] = average_precision, num_annotations

  print('\nAP:')
  for label in range(generator.num_classes()):
    label_name = generator.label_to_name(label)
    print('{}: {}'.format(label_name, average_precisions[label][0]))

  return average_precisions

