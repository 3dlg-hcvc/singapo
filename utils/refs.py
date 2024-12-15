# reference of object categories
cat_ref = {
    "Table": 0,
    "Dishwasher": 1,
    "StorageFurniture": 2,
    "Refrigerator": 3,
    "WashingMachine": 4,
    "Microwave": 5,
    "Oven": 6,
}

# reference of semantic labels for each part
sem_ref = {
    "fwd": {
        "door": 0,
        "drawer": 1,
        "base": 2,
        "handle": 3,
        "wheel": 4,
        "knob": 5,
        "shelf": 6,
        "tray": 7,
    },
    "bwd": {
        0: "door",
        1: "drawer",
        2: "base",
        3: "handle",
        4: "wheel",
        5: "knob",
        6: "shelf",
        7: "tray",
    },
}

# reference of joint types for each part
joint_ref = {
    "fwd": {"fixed": 1, "revolute": 2, "prismatic": 3, "screw": 4, "continuous": 5},
    "bwd": {1: "fixed", 2: "revolute", 3: "prismatic", 4: "screw", 5: "continuous"},
}


import plotly.express as px

# pallette for joint type color
joint_color_ref = px.colors.qualitative.Set1
# pallette for graph node color
graph_color_ref = px.colors.qualitative.Bold + px.colors.qualitative.Prism
# pallette for semantic label color
semantic_color_ref = px.colors.qualitative.Vivid_r
# attention map visulaization color
attn_color_ref = px.colors.sequential.Viridis

from matplotlib.colors import LinearSegmentedColormap

cmap_attn = LinearSegmentedColormap.from_list("mycmap", attn_color_ref, N=256)
