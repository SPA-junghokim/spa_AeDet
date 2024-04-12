# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from torch.utils.data import Dataset
from mmdet3d.datasets.nuscenes_mono_dataset import NuScenesMonoDataset
from mmdet.core import eval_map, eval_recalls
#from .builder import DATASETS
from mmdet3d.datasets.pipelines import Compose
#from mmdet.datasets.coco import CocoDataset
#from .coco import CocoDataset
from .api_wrappers import COCO, COCOeval

CLASSES = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]

pipeline=[{'type': 'LoadImageFromFileMono3D'}, \
    {'type': 'LoadAnnotations3D', 'with_bbox': True, 'with_label': True, 'with_attr_label': True, \
        'with_bbox_3d': True, 'with_label_3d': True, 'with_bbox_depth': True}, \
            {'type': 'Normalize', 'mean': [103.53, 116.28, 123.675], 'std': [1.0, 1.0, 1.0], \
                'to_rgb': False}, {'type': 'Pad', 'size_divisor': 32}, \
                    {'type': 'DefaultFormatBundle3D', 'class_names': \
                        ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']}, {'type': 'Collect3D', \
                            'keys': ['img', 'gt_bboxes', 'gt_labels', 'attr_labels', 'gt_bboxes_3d', 'gt_labels_3d', 'centers2d', 'depths']}]


nusc_mono_dataset = NuScenesMonoDataset(
    data_root='data/nuScenes',
    ann_file = 'data/nuScenes/nuscenes_infos_train_mono3d.coco.json',
    #ann_file = 'data/nuScenes/nuscenes_infos_train.pkl',
    pipeline = pipeline,
    modality = {'use_lidar': False, 'use_camera': True, 'use_radar': False, 'use_map': False, 'use_external': False},
    classes = CLASSES,
    #img_prefix = 'data/nuScenes',
)

class CustomDataset(Dataset):
    """Custom dataset for detection.

    The annotation format is shown as follows. The `ann` field is optional for
    testing.

    .. code-block:: none

        [
            {
                'filename': 'a.jpg',
                'width': 1280,
                'height': 720,
                'ann': {
                    'bboxes': <np.ndarray> (n, 4) in (x1, y1, x2, y2) order.
                    'labels': <np.ndarray> (n, ),
                    'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                    'labels_ignore': <np.ndarray> (k, 4) (optional field)
                }
            },
            ...
        ]

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        data_root (str, optional): Data root for ``ann_file``,
            ``img_prefix``, ``seg_prefix``, ``proposal_file`` if specified.
        test_mode (bool, optional): If set True, annotation will not be loaded.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`, i.e., we never filter images
            during tests.
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk')):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.file_client = mmcv.FileClient(**file_client_args)
        self.CLASSES = self.get_classes(classes)
        
        #self.get_ann_info = CocoDataset.get_ann_info()

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(self.ann_file) as local_path:
                self.data_infos = self.load_annotations(local_path)
        else:
            warnings.warn(
                'The used MMCV version does not have get_local_path. '
                f'We treat the {self.ann_file} as local paths and it '
                'might cause errors if the path is not a local path. '
                'Please use MMCV>= 1.3.16 if you meet errors.')
            self.data_infos = self.load_annotations(self.ann_file)

        if self.proposal_file is not None:
            if hasattr(self.file_client, 'get_local_path'):
                with self.file_client.get_local_path(
                        self.proposal_file) as local_path:
                    self.proposals = self.load_proposals(local_path)
            else:
                warnings.warn(
                    'The used MMCV version does not have get_local_path. '
                    f'We treat the {self.ann_file} as local paths and it '
                    'might cause errors if the path is not a local path. '
                    'Please use MMCV>= 1.3.16 if you meet errors.')
                self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            #self.data_infos = [self.data_infos[i] for i in valid_inds] # chgd
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            #self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    # def load_annotations(self, ann_file):
    #     """Load annotation from annotation file."""
    #     ann_file = ann_file[14:] # chgd
    #     return mmcv.load(ann_file)
    
    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        ann_file = ann_file[14:]
        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def load_proposals(self, proposal_file):
        """Load proposal from proposal file."""
        return mmcv.load(proposal_file)

    # def get_ann_info(self, idx):
    #     """Get annotation by index.

    #     Args:
    #         idx (int): Index of data.

    #     Returns:
    #         dict: Annotation info of specified index.
    #     """
    #     # return self.data_infos[idx] # org
    #     return self.data_infos['annotations'][idx] # chgd
    
    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return nusc_mono_dataset._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.data_infos[idx]['ann']['labels'].astype(np.int).tolist()

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        if self.filter_empty_gt:
            warnings.warn(
                'CustomDataset does not support filtering empty gt images.')
        valid_inds = []
        
        for i, img_info in enumerate(self.data_infos): # chgd
            #breakpoint()
            #if min(img_info['width'], img_info['height']) >= min_size:
            valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx, cams):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        #idx =108919 # major chgd
        #breakpoint()
        
        pers_gt_boxes = []
        for idx_cam, cam in enumerate(cams):
            img_info = self.data_infos[idx] # org
            
            ann_info = self.get_ann_info(idx+idx_cam) # chgd
            #results = dict(img_info=img_info, ann_info=ann_info)
            # if self.proposals is not None:
            #     results['proposals'] = self.proposals[idx]
            #self.pre_pipeline(results)
            #results['img_info']['filename'] = results['img_info'].pop('file_name') # chgd
            #results['ann_info']['bboxes'] = results['ann_info'].pop('bbox') # chgd
            #results['ann_info']['labels'] = np.array([results['ann_info'].pop('category_id')]) # chgd
            #pers_gt_box = nusc_mono_dataset._parse_ann_info(results['img_info'], results['ann_info'])
            
            pers_gt_boxes.append(ann_info)

        # Post-process cam order
        pers_gt_boxes[0], pers_gt_boxes[2] = pers_gt_boxes[2], pers_gt_boxes[0] # front left, front
        pers_gt_boxes[1], pers_gt_boxes[2] = pers_gt_boxes[2], pers_gt_boxes[1]
        pers_gt_boxes[3], pers_gt_boxes[4] = pers_gt_boxes[4], pers_gt_boxes[3]
        
        return pers_gt_boxes
        
        #return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def get_cat2imgs(self):
        """Get a dict with class as key and img_ids as values, which will be
        used in :class:`ClassAwareSampler`.

        Returns:
            dict[list]: A dict of per-label image list,
            the item of the dict indicates a label index,
            corresponds to the image index that contains the label.
        """
        if self.CLASSES is None:
            raise ValueError('self.CLASSES can not be None')
        # sort the label index
        cat2imgs = {i: [] for i in range(len(self.CLASSES))}
        for i in range(len(self)):
            cat_ids = set(self.get_cat_ids(i))
            for cat in cat_ids:
                cat2imgs[cat].append(i)
        return cat2imgs

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results

    def __repr__(self):
        """Print the number of instance number."""
        dataset_type = 'Test' if self.test_mode else 'Train'
        result = (f'\n{self.__class__.__name__} {dataset_type} dataset '
                  f'with number of images {len(self)}, '
                  f'and instance counts: \n')
        if self.CLASSES is None:
            result += 'Category names are not provided. \n'
            return result
        instance_count = np.zeros(len(self.CLASSES) + 1).astype(int)
        # count the instance number in each image
        for idx in range(len(self)):
            label = self.get_ann_info(idx)['labels']
            unique, counts = np.unique(label, return_counts=True)
            if len(unique) > 0:
                # add the occurrence number to each class
                instance_count[unique] += counts
            else:
                # background is the last index
                instance_count[-1] += 1
        # create a table with category count
        table_data = [['category', 'count'] * 5]
        row_data = []
        for cls, count in enumerate(instance_count):
            if cls < len(self.CLASSES):
                row_data += [f'{cls} [{self.CLASSES[cls]}]', f'{count}']
            else:
                # add the background number
                row_data += ['-1 background', f'{count}']
            if len(row_data) == 10:
                table_data.append(row_data)
                row_data = []
        if len(row_data) >= 2:
            if row_data[-1] == '0':
                row_data = row_data[:-2]
            if len(row_data) >= 2:
                table_data.append([])
                table_data.append(row_data)

        table = AsciiTable(table_data)
        result += table.table
        return result
