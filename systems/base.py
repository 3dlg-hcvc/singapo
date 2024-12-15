import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import json
import numpy as np
import lightning.pytorch as pl
from metrics.iou_cdist import IoU_cDist
from utils.savermixins import SaverMixin
from utils.refs import sem_ref, joint_ref
from data.utils import convert_data_range, parse_tree
from utils.plot import viz_graph, make_grid, add_text
from utils.render import draw_boxes_axiss_anim, prepare_meshes

class BaseSystem(pl.LightningModule, SaverMixin):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)

    def setup(self, stage: str):
        # config the logger dir for images
        self.hparams.save_dir = os.path.join(self.hparams.exp_dir, 'output', stage) 
        os.makedirs(self.hparams.save_dir, exist_ok=True)
        

    # --------------------------------- visualization ---------------------------------
        
    def convert_json(self, x, c, idx, prefix=''):
        out = {"meta": {}, "diffuse_tree": []}
        
        n_nodes = c[f"{prefix}n_nodes"][idx].item()
        par = c[f"{prefix}parents"][idx].cpu().numpy().tolist()
        adj = c[f"{prefix}adj"][idx].cpu().numpy()
        np.fill_diagonal(adj, 0) # remove self-loop for the root node
        if f"{prefix}obj_cat" in c:
            out["meta"]["obj_cat"] = c[f"{prefix}obj_cat"][idx]

        # convert the data to original range
        data = convert_data_range(x.cpu().numpy())
        # parse the tree
        out["diffuse_tree"] = parse_tree(data, n_nodes, par, adj)
        return out

    def save_val_img(self, pred, gt, cond):
        B = pred.shape[0]
        pred_imgs, gt_imgs, gt_graphs_view = [], [], []
        for b in range(B):
            # convert to humnan readable format json
            pred_json = self.convert_json(pred[b], cond, b)
            gt_json = self.convert_json(gt[b], cond, b)
            # visualize bbox and axis
            pred_meshes = prepare_meshes(pred_json)
            bbox_0, bbox_1, axiss = (
                pred_meshes["bbox_0"],
                pred_meshes["bbox_1"],
                pred_meshes["axiss"],
            )
            pred_img = draw_boxes_axiss_anim(
                bbox_0, bbox_1, axiss, mode="graph", resolution=128
            )
            gt_meshes = prepare_meshes(gt_json)
            bbox_0, bbox_1, axiss = (
                gt_meshes["bbox_0"],
                gt_meshes["bbox_1"],
                gt_meshes["axiss"],
            )
            gt_img = draw_boxes_axiss_anim(
                bbox_0, bbox_1, axiss, mode="graph", resolution=128
            )
            # visualize graph
            gt_graph = viz_graph(gt_json, res=128)
            gt_graph = add_text(cond["name"][b], gt_graph)
            # GT views
            rgb_view = cond["img"][b].cpu().numpy()

            pred_imgs.append(pred_img)
            gt_imgs.append(gt_img)
            gt_graphs_view.append(rgb_view)
            gt_graphs_view.append(gt_graph)

        # save images for generated results
        epoch = str(self.current_epoch).zfill(5)
        pred_thumbnails = np.concatenate(pred_imgs, axis=1)  # concat batch in width

        # save images for ground truth
        gt_graph_imgs = np.concatenate(gt_graphs_view, axis=1)
        gt_thumbnails = np.concatenate(gt_imgs, axis=1)  # concat batch in width
        grid = np.concatenate([gt_graph_imgs, gt_thumbnails, pred_thumbnails], axis=0)
        self.save_rgb_image(f"{epoch}.png", grid)

    def save_test_step(self, pred, gt, cond, batch_idx, res=128):
        exp_name = self._get_exp_name()
        model_name = cond["name"][0].replace("/", '@')
        save_dir = f"{exp_name}/{str(batch_idx)}@{model_name}"

        # input image
        input_img = cond["img"][0].cpu().numpy()
        # GT recordings
        if not self.hparams.get('test_no_GT', False):
            gt_json = self.convert_json(gt[0], cond, 0, prefix='gt_')
            gt_graph = viz_graph(gt_json, res=256)
            gt_meshes = prepare_meshes(gt_json)
            bbox_0, bbox_1, axiss = (
                gt_meshes["bbox_0"],
                gt_meshes["bbox_1"],
                gt_meshes["axiss"],
            )
            gt_img = draw_boxes_axiss_anim(bbox_0, bbox_1, axiss, mode="graph", resolution=res)
        else:
            gt_graph = 255 * np.ones((res, res, 3), dtype=np.uint8)
            gt_img = 255 * np.ones((res, 2 * res, 3), dtype=np.uint8)
        gt_block = np.concatenate([input_img, gt_img], axis=1)

        # recordings for generated results
        img_blocks = []
        for b in range(pred.shape[0]):
            pred_json = self.convert_json(pred[b], cond, 0)
            # visualize bbox and axis
            pred_meshes = prepare_meshes(pred_json)
            bbox_0, bbox_1, axiss = (
                pred_meshes["bbox_0"],
                pred_meshes["bbox_1"],
                pred_meshes["axiss"],
            )
            pred_img = draw_boxes_axiss_anim(
                bbox_0, bbox_1, axiss, mode="graph", resolution=res
            )
            img_blocks.append(pred_img)
            self.save_json(f"{save_dir}/{b}/object.json", pred_json)
        # save images for generated results
        img_grid = make_grid(img_blocks, cols=5)
        # visualize the input graph
        input_graph = viz_graph(pred_json, res=256)

        # save images
        self.save_rgb_image(f"{save_dir}/gt_graph.png", gt_graph)
        self.save_rgb_image(f"{save_dir}/output.png", img_grid)
        self.save_rgb_image(f"{save_dir}/gt.png", gt_block)
        self.save_rgb_image(f"{save_dir}/input_graph.png", input_graph)

    def _save_html_end(self):
        exp_name = self._get_exp_name()
        save_dir = self.get_save_path(exp_name)
        cases = sorted(os.listdir(save_dir), key=lambda x: int(x.split("@")[0]))
        html_head = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Test Image Results</title>
            <style>
                table {
                    width: 100%;
                    border-collapse: collapse;
                }
                th, td {
                    border: 1px solid black;
                    padding: 8px;
                    text-align: left;
                }
                .separator {
                    border-top: 2px solid black;
                }
            </style>
        </head>
        <body>
            <table>

        """
        total = len(cases)
        each = 200
        n_pages = total // each + 1
        for p in range(n_pages):
            html_content = html_head
            for i in range(p * each, min((p + 1) * each, total)):
                case = cases[i]
                if self.hparams.get("test_no_GT", False):
                    aid_iou = rid_iou = aid_cdist = rid_cdist = aid_cd = rid_cd = aor = "N/A"              
                else:
                    with open(os.path.join(save_dir, case, "metrics.json"), "r") as f:
                        metrics = json.load(f)["avg"]
                    aid_iou = round(metrics["AS-IoU"], 4)
                    rid_iou = round(metrics["RS-IoU"], 4)
                    aid_cdist = round(metrics["AS-cDist"], 4)
                    rid_cdist = round(metrics["RS-cDist"], 4)
                    aid_cd = round(metrics["AS-CD"], 4)
                    rid_cd = round(metrics["RS-CD"], 4)
                    aor = metrics["AOR"]
                    if aor is not None:
                        aor = round(aor, 4)
                html_content += f"""
                    <tr>
                        <th>Object ID</th>
                        <th>Metrics (avg) </th>
                        <th>Input image + GT object + GT graph</th>
                        <th>Input graph </th>
                    </tr>
                    <tr>
                        <td rowspan="3">{case}</td>
                        <td> 
                        [AS-cDist] {aid_cdist}<br>
                        [RS-cDist] {rid_cdist}<br>
                        -----------------------<br>
                        [AS-IoU]  {aid_iou}<br>
                        [RS-IoU]  {rid_iou}<br>
                        -----------------------<br>
                        [RS-CD]   {rid_cd}<br> 
                        [AS-CD]   {aid_cd}<br>
                        -----------------------<br>
                        [AOR]     {aor}<br>
                        </td>
                        <td>
                            <img src="{exp_name}/{case}/gt.png" alt="GT Image" style="height: 128px; width: 3*128px;">
                            <img src="{exp_name}/{case}/gt_graph.png" alt="Graph Image" style="height: 128px; width: 3*128px;">
                        </td>
                        <td>
                            <img src="{exp_name}/{case}/input_graph.png" alt="Graph Image" style="height: 128px; width: 3*128px;">
                        </td>
                    </tr>
                    <tr><th colspan="3">Generated samples</th></tr>
                    <tr>
                        <td colspan="3"><img src="{exp_name}/{case}/output.png" alt="Generated Image" style="height: 3*128px; width: 10*128px;"></td>
                    </tr>
                    <tr class="separator"><td colspan="4"></td></tr>
                """
            html_content += """</table></body></html>"""
            outfile = self.get_save_path(f"{exp_name}_page_{p+1}.html")
            with open(outfile, "w") as file:
                file.write(html_content)

    def val_compute_metrics(self, pred, gt, cond):
        loss_dict = {}
        B = pred.shape[0]
        as_ious = 0.0
        rs_ious = 0.0
        as_cdists = 0.0
        rs_cdists = 0.0
        for b in range(B):
            gt_json = self.convert_json(gt[b], cond, b)
            pred_json = self.convert_json(pred[b], cond, b)
            scores = IoU_cDist(
                pred_json,
                gt_json,
                num_states=5,
                compare_handles=True,
                iou_include_base=False,
            )
            as_ious += scores['AS-IoU']
            rs_ious += scores['RS-IoU']
            as_cdists += scores['AS-cDist']
            rs_cdists += scores['RS-cDist']

        as_ious /= B
        rs_ious /= B
        as_cdists /= B
        rs_cdists /= B

        loss_dict['val/AS-IoU'] = as_ious
        loss_dict['val/RS-IoU'] = rs_ious
        loss_dict['val/AS-cDist'] = as_cdists
        loss_dict['val/RS-cDist'] = rs_cdists

        return loss_dict

    def _get_exp_name(self):
        which_ds = self.hparams.get("test_which", 'pm')
        is_pred_G = self.hparams.get("test_pred_G", False)
        is_label_free = self.hparams.get("test_label_free", False)
        guidance_scaler = self.hparams.get("guidance_scaler", 0)
        # config saving directory
        exp_postfix = f"_w={guidance_scaler}_{which_ds}"
        if is_pred_G:
            exp_postfix += "_pred_G"
        if is_label_free:
            exp_postfix += "_label_free"
        
        exp_name = "epoch_" + str(self.current_epoch).zfill(3) + exp_postfix
        return exp_name

        
