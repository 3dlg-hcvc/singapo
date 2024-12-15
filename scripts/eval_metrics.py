import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import argparse
from metrics.cd import CD
from metrics.aor import AOR
from metrics.iou_cdist import IoU_cDist

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def _parse_gt_model(src):
    '''
    Parse the name of the predicted model to match the ground truth model name.
    '''
    name_tokens = src.split("@")
    tgt = ''
    for i in range(1, len(name_tokens)):
        tgt += f"{name_tokens[i]}/"
    return tgt[:-1]

def _get_metrics_per_sample(pred_dir, gt_dir, pred_json_name, gt_json_name):
    # with open(f"{pred_dir}/metrics.json", 'r') as f:
    #     metrics = json.load(f)
    # load GT json
    with open(f"{gt_dir}/{gt_json_name}", "r") as f:
        gt = json.load(f)
    # load pred json
    with open(f"{pred_dir}/{pred_json_name}", 'r') as f:
        pred = json.load(f)
    # Chamfer Distance
    cd_scores = CD(pred, pred_dir, gt, gt_dir, include_base=True)
    # AOR
    aor_out = AOR(pred)
    aor = aor_out if aor_out != -1 else None
    # exclude base version
    scores = IoU_cDist(
        pred,
        gt,
        compare_handles=True,
        iou_include_base=True,
    )
    # save metrics
    metrics = {
        "AS-IoU": scores["AS-IoU"],
        "RS-IoU": scores["RS-IoU"],
        "AS-cDist": scores["AS-cDist"],
        "RS-cDist": scores["RS-cDist"],
        "AS-CD": cd_scores["AS-CD"],
        "RS-CD": cd_scores["RS-CD"],
        "AOR": aor,
    }
    # save_json(f"{pred_dir}/metrics.json", metrics)
    return metrics

def _get_metrics_per_input(src_dir, gt_dir, pred_json_name, gt_json_name):
    # if os.path.exists(f"{src_dir}/metrics.json"):
    #     return
    samples = os.listdir(src_dir)
    samples.sort()
    metrics = []
    for sample in samples:
        if os.path.isdir(f"{src_dir}/{sample}"):
            metrics.append(_get_metrics_per_sample(f"{src_dir}/{sample}", gt_dir, pred_json_name, gt_json_name))
    # average, best among samples, first sample
    as_iou, rs_iou, as_cdist, rs_cdist, as_cd, rs_cd, aor = 0, 0, 0, 0, 0, 0, 0
    bes_as_iou, bes_rs_iou, bes_as_cdist, bes_rs_cdist, bes_as_cd, bes_rs_cd, bes_aor = 99, 99, 99, 99, 99, 99, 99
    cnt_valid_aor = 0
    for m in metrics:
        as_iou += m["AS-IoU"]
        rs_iou += m["RS-IoU"]
        as_cdist += m["AS-cDist"]
        rs_cdist += m["RS-cDist"]
        as_cd += m["AS-CD"]
        rs_cd += m["RS-CD"]
        bes_as_iou = min(bes_as_iou, m["AS-IoU"])
        bes_rs_iou = min(bes_rs_iou, m["RS-IoU"])
        bes_as_cdist = min(bes_as_cdist, m["AS-cDist"])
        bes_rs_cdist = min(bes_rs_cdist, m["RS-cDist"])
        bes_as_cd = min(bes_as_cd, m["AS-CD"])
        bes_rs_cd = min(bes_rs_cd, m["RS-CD"])
        if m["AOR"] is not None:
            cnt_valid_aor += 1
            aor += m["AOR"]
            bes_aor = min(bes_aor, m["AOR"])
    N = len(metrics)
    as_iou /= N
    rs_iou /= N
    as_cdist /= N
    rs_cdist /= N
    as_cd /= N
    rs_cd /= N
    if cnt_valid_aor > 0:
        aor /= cnt_valid_aor
    else:
        aor = None
        bes_aor = None
    ii = 4
    res = {
        "avg": {
            "AS-IoU": as_iou,
            "RS-IoU": rs_iou,
            "AS-cDist": as_cdist,
            "RS-cDist": rs_cdist,
            "AS-CD": as_cd,
            "RS-CD": rs_cd,
            "AOR": aor,
        },
        "best": {
            "AS-IoU": bes_as_iou,
            "RS-IoU": bes_rs_iou,
            "AS-cDist": bes_as_cdist,
            "RS-cDist": bes_rs_cdist,
            "AS-CD": bes_as_cd,
            "RS-CD": bes_rs_cd,
            "AOR": bes_aor,
        },
        "first": {
            "AS-IoU": metrics[ii]["AS-IoU"],
            "RS-IoU": metrics[ii]["RS-IoU"],
            "AS-cDist": metrics[ii]["AS-cDist"],
            "RS-cDist": metrics[ii]["RS-cDist"],
            "AS-CD": metrics[ii]["AS-CD"],
            "RS-CD": metrics[ii]["RS-CD"],
            "AOR": metrics[ii]["AOR"],
        }
    }
    save_json(f"{src_dir}/metrics.json", res)
    

def eval_our_metrics(exp_dir, gt_root, pred_json_name='object.json', gt_json_name='object.json'):
    '''
    Script to evaluate the generated objects using our metrics and save the results in the metrics.json file.
    '''
    output_dirs = os.listdir(exp_dir)
    output_dirs.sort()
    print(f"---------Eval on our metrics---------")
    
    pred_dirs = []
    gt_dirs = []
    
    for pred_id in output_dirs:
        pred_dirs.append(f"{exp_dir}/{pred_id}")
        gt_dirs.append(f"{gt_root}/{_parse_gt_model(pred_id)}")
        _get_metrics_per_input(f"{exp_dir}/{pred_id}", f"{gt_root}/{_parse_gt_model(pred_id)}", pred_json_name, gt_json_name)
    
    all_metrics = []
    for dir in pred_dirs:
        with open(f"{dir}/metrics.json", 'r') as f:
            metrics = json.load(f)
        all_metrics.append(metrics)
    # average of the averages, bests, and firsts
    avg_as_iou, avg_rs_iou, avg_as_cdist, avg_rs_cdist, avg_as_cd, avg_rs_cd, avg_aor = 0, 0, 0, 0, 0, 0, 0
    avg_bes_as_iou, avg_bes_rs_iou, avg_bes_as_cdist, avg_bes_rs_cdist, avg_bes_as_cd, avg_bes_rs_cd, avg_bes_aor = 0, 0, 0, 0, 0, 0, 0
    avg_fir_as_iou, avg_fir_rs_iou, avg_fir_as_cdist, avg_fir_rs_cdist, avg_fir_as_cd, avg_fir_rs_cd, avg_fir_aor = 0, 0, 0, 0, 0, 0, 0
    cnt_aor_avg, cnt_aor_best, cnt_aor_first = 0, 0, 0
    N = len(all_metrics)
    for m in all_metrics:
        avg_as_iou += m["avg"]["AS-IoU"]
        avg_rs_iou += m["avg"]["RS-IoU"]
        avg_as_cdist += m["avg"]["AS-cDist"]
        avg_rs_cdist += m["avg"]["RS-cDist"]
        avg_as_cd += m["avg"]["AS-CD"]
        avg_rs_cd += m["avg"]["RS-CD"]
        if m["avg"]["AOR"] is not None:
            cnt_aor_avg += 1
            avg_aor += m["avg"]["AOR"]
        
        avg_bes_as_iou += m["best"]["AS-IoU"]
        avg_bes_rs_iou += m["best"]["RS-IoU"]
        avg_bes_as_cdist += m["best"]["AS-cDist"]
        avg_bes_rs_cdist += m["best"]["RS-cDist"]
        avg_bes_as_cd += m["best"]["AS-CD"]
        avg_bes_rs_cd += m["best"]["RS-CD"]
        if m["best"]["AOR"] is not None:
            cnt_aor_best += 1
            avg_bes_aor += m["best"]["AOR"]
        
        avg_fir_as_iou += m["first"]["AS-IoU"]
        avg_fir_rs_iou += m["first"]["RS-IoU"]
        avg_fir_as_cdist += m["first"]["AS-cDist"]
        avg_fir_rs_cdist += m["first"]["RS-cDist"]
        avg_fir_as_cd += m["first"]["AS-CD"]
        avg_fir_rs_cd += m["first"]["RS-CD"]
        if m["first"]["AOR"] is not None:    
            cnt_aor_first += 1
            avg_fir_aor += m["first"]["AOR"]
        
    avg_as_iou /= N
    avg_rs_iou /= N
    avg_as_cdist /= N
    avg_rs_cdist /= N
    avg_as_cd /= N
    avg_rs_cd /= N
    avg_aor /= cnt_aor_avg if cnt_aor_avg > 0 else None
    
    avg_bes_as_iou /= N
    avg_bes_rs_iou /= N
    avg_bes_as_cdist /= N
    avg_bes_rs_cdist /= N
    avg_bes_as_cd /= N
    avg_bes_rs_cd /= N
    avg_bes_aor /= cnt_aor_best if cnt_aor_best > 0 else None
    
    avg_fir_as_iou /= N
    avg_fir_rs_iou /= N
    avg_fir_as_cdist /= N
    avg_fir_rs_cdist /= N
    avg_fir_as_cd /= N
    avg_fir_rs_cd /= N
    avg_fir_aor /= cnt_aor_first if cnt_aor_first > 0 else None
    
    res = {
        "avg": { 
            "RS-IoU": round(avg_rs_iou, 4),
            "AS-IoU": round(avg_as_iou, 4),
            "RS-cDist": round(avg_rs_cdist, 4),
            "AS-cDist": round(avg_as_cdist, 4),
            "RS-CD": round(avg_rs_cd, 4),
            "AS-CD": round(avg_as_cd, 4),
            "AOR": avg_aor, 
        },
        "best": {
            "RS-IoU": round(avg_bes_rs_iou, 4),
            "AS-IoU": round(avg_bes_as_iou, 4),
            "RS-cDist": round(avg_bes_rs_cdist, 4),
            "AS-cDist": round(avg_bes_as_cdist, 4),
            "RS-CD": round(avg_bes_rs_cd, 4),
            "AS-CD": round(avg_bes_as_cd, 4),
            "AOR": avg_bes_aor, 
        },
        "first": {
            "RS-IoU": round(avg_fir_rs_iou, 4),
            "AS-IoU": round(avg_fir_as_iou, 4),
            "RS-cDist": round(avg_fir_rs_cdist, 4),
            "AS-cDist": round(avg_fir_as_cdist, 4),
            "RS-CD": round(avg_fir_rs_cd, 4),
            "AS-CD": round(avg_fir_as_cd, 4),
            "AOR": avg_fir_aor, 
        }  
    }

    save_json(f"{os.path.dirname(exp_dir)}/metrics_{os.path.basename(exp_dir)}.json", res)



if __name__ == '__main__':
    '''Script to evaluate the generated objects using our metrics and save the results in the json file at the previous level of the experiment directory.'''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', required=True, help='The directory of the experiment.')
    parser.add_argument('--gt_root', default='../data', help='The root directory of the ground truth data.')
    parser.add_argument('--pred_json_name', default='object.json', help='The name of the predicted json file.')
    parser.add_argument('--gt_json_name', default='object.json', help='The name of the ground truth json file.')
    args = parser.parse_args()
    
    print(f"Experiment directory: {args.exp_dir}")
    eval_our_metrics(args.exp_dir, args.gt_root, args.pred_json_name, args.gt_json_name)