import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import json
import numpy as np
from PIL import Image
import torchvision.transforms as T
from data.base_dataset import BaseDataset
from data.utils import make_white_background


class MyDataset(BaseDataset):
    """
    Dataset for training and testing on the PartNet-Mobility and ACD datasets (with our preprocessing).
    The GT graph is given.
    """

    def __init__(self, hparams, model_ids, mode="train", json_name="object.json"):
        self.hparams = hparams
        self.json_name = json_name
        self.model_ids = self._filter_models(model_ids)
        self.mode = mode
        self.map_cat = False

        self.no_GT = (
            True if self.hparams.get("test_no_GT", False) and self.hparams.get("test_pred_G", False)
            else False
        )
        self.pred_G = (
            False
            if mode in ["train", "val"]
            else self.hparams.get("test_pred_G", False)
        )

        if mode == 'test':
            if hparams.test_which == "acd":
                self.map_cat = True
                self.get_acd_mapping()
        
        self.files = self._cache_data()
        print(f"[INFO] {mode} dataset: {len(self)} data samples loaded.")

    def _cache_data_train(self):
        data_root = self.hparams.root
        # number of views per model and in total
        n_views_per_model = 17
        n_views = n_views_per_model * len(self.model_ids)
        # json files for each model
        json_files = []
        # mapping to the index of the corresponding model in json_files
        model_mappings = []
        # space for dinov2 patch features
        feats = np.empty((n_views, 256, 768), dtype=np.float16)
        # space for object masks on image patches
        obj_masks = np.empty((n_views, 256), dtype=bool)
        # input images (not required in training)
        imgs = None
        # load data for non-aug views
        i = 0  # index for views
        for j, model_id in enumerate(self.model_ids):
            # 3D data
            with open(os.path.join(data_root, model_id, self.json_name), "r") as f:
                json_file = json.load(f)
            json_files.append(json_file)
            # features for all views
            all_feats = np.load(
                os.path.join(data_root, model_id, "features/dinov2_patch_reg.npy")
            )  # (20, Np, 768)
            # convert to np.float16 to save memory, only used for 16-mixed precision training
            all_feats = all_feats.astype(np.float16)
            feats[i : i + n_views_per_model] = all_feats[:n_views_per_model]
            # object masks for all views
            all_obj_masks = np.load(
                os.path.join(data_root, model_id, "features/patch_obj_masks.npy")
            )  # (20, Np)
            obj_masks[i : i + n_views_per_model] = all_obj_masks[:n_views_per_model]
            # mapping to json file
            model_mappings += [j] * n_views_per_model
            # update index
            i += n_views_per_model

        return {
            "len": n_views,
            "gt_files": json_files,
            "features": feats,
            "obj_masks": obj_masks,
            "model_mappings": model_mappings,
            "imgs": imgs,
        }

    def _cache_data_non_train(self):
        # number of views per model and in total
        n_views_per_model = 2 if self.mode == "test" else 1
        n_views = n_views_per_model * len(self.model_ids)
        # json files for each model
        gt_files = []
        pred_files = []  # for predicted graphs
        # mapping to the index of the corresponding model in json_files
        model_mappings = []
        # space for dinov2 patch features
        feats = np.empty((n_views, 256, 768), dtype=np.float16)
        # space for input images
        imgs = np.empty((n_views, 128, 128, 3), dtype=np.uint8)
        # transformation for input images
        transform = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.Resize(128, interpolation=T.InterpolationMode.BICUBIC),
            ]
        )

        i = 0  # index for views
        for j, model_id in enumerate(self.model_ids):
            # 3D data
            
            if self.pred_G:
                tokens = model_id.split("/")
                fname = ""
                for token in tokens:
                    fname += token + "@"
                with open(os.path.join(self.hparams.G_dir, f"{fname}18.json"), "r") as f:
                    pred_graph_18 = json.load(f)
                with open(os.path.join(self.hparams.G_dir, f"{fname}19.json"), "r") as f:
                    pred_graph_19 = json.load(f)
                pred_files.append(pred_graph_18)
                pred_files.append(pred_graph_19)
 
            if not self.no_GT:
                with open(os.path.join(self.hparams.root, model_id, self.json_name), "r") as f:
                    gt_file = json.load(f)
                gt_files.append(gt_file)
            
            # features for all views
            all_feats = np.load(
                os.path.join(
                    self.hparams.root, model_id, "features/dinov2_patch_reg.npy"
                )
            )  # (20, Np, 768)
            # convert to np.float16 to save memory, only used for 16-mixed precision training
            all_feats = all_feats.astype(np.float16)
            feats[i : i + n_views_per_model] = all_feats[-n_views_per_model:]
            # mapping to json file
            model_mappings += [j] * n_views_per_model

            # loaf input images for val or test
            if self.mode == "val":
                with Image.open(
                    os.path.join(self.hparams.root, model_id, "imgs", "17.png")
                ) as img_:
                    img = np.asarray(
                        make_white_background(transform(img_)), dtype=np.uint8
                    )
                imgs[i] = img
            else:
                with Image.open(os.path.join(self.hparams.root, model_id, "imgs", "18.png")) as img1_:
                    img1 = np.asarray(make_white_background(transform(img1_)), dtype=np.uint8)
                with Image.open(os.path.join(self.hparams.root, model_id, "imgs", "19.png")) as img2_:
                    img2 = np.asarray(make_white_background(transform(img2_)), dtype=np.uint8)
                imgs[i] = img1
                imgs[i + 1] = img2
            # update index
            i += n_views_per_model

        return {
            "len": n_views,
            "gt_files": gt_files,
            "pred_files": pred_files,
            "features": feats,
            "model_mappings": model_mappings,
            "imgs": imgs,
        }

    def _cache_data(self):
        """
        Function to cache data from disk.
        """
        if self.mode == "train":
            return self._cache_data_train()
        else:
            return self._cache_data_non_train()

    def _get_item_train_val(self, index):
        model_i = self.files["model_mappings"][index]
        gt_file = self.files["gt_files"][model_i]
        data, cond = self._prepare_input_GT(
            file=gt_file, model_id=self.model_ids[model_i]
        )
        if self.mode == "val":
            # input image for visualization
            img = self.files["imgs"][index]
            cond["img"] = img
        else:
            # object masks on patches
            obj_mask = self.files["obj_masks"][index][None, ...].repeat(self.hparams.K * 5, axis=0)
            cond["img_obj_mask"] = obj_mask
        return data, cond

    def _get_item_test(self, index):
        model_i = self.files["model_mappings"][index]
        pred_file = (
            self.files["pred_files"][index]
            if self.pred_G
            else self.files["gt_files"][model_i]
        )
        gt_file = None if self.no_GT else self.files["gt_files"][model_i] 

        data, cond = self._prepare_input(self.model_ids[model_i], pred_file, gt_file)
        # input image for visualization
        img = self.files["imgs"][index]
        cond["img"] = img
        return data, cond

    def __getitem__(self, index):
        # input image features
        feat = self.files["features"][index]

        # prepare input, GT data and other axillary info
        if self.mode == "test":
            data, cond = self._get_item_test(index)
        else:
            data, cond = self._get_item_train_val(index)

        return data, cond, feat

    def __len__(self):
        return self.files["len"]
