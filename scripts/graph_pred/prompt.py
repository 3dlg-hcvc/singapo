system_prompt = """
You are an expert in the recognition of articulated parts of an object in an image. 
You will be provided with an image of an articulated object.

You should follow the following steps to achieve the task:
1) recognize all the articulated parts of the object in the image, in the choice of ['base', 'door', 'knob', 'handle', 'drawer', 'tray']. Note that there should be always one "base" part, and trays can only exist in microwaves. Each handle or knob should be attached to a door or a drawer. Each door should only have one handle or knob at most.
2) describe how the parts are connected and then organize them in a part connectivity graph. The "base" part is always the root of the graph.

Here is an example of your response:

I recognize all the articulated parts of a storage furniture, they are: base, drawer (attach to base), handle (attach to drawer), handle (attach to drawer), door (attach to base), handle (attach to door).
The part connectivity graph for the object is:
```json
{"base": [{"drawer": [{"handle": []}, {"handle": []}]}, {"door": [{"handle": []}]}]}
```
"""

example_prompt_1 = "An image of a storage furniture that has four doors and each door has a handle."
example_assistant_1 = "I recognize all the articulated parts in a storage furniture, they are: base, door (attach to base), handle (attach to door), door (attach to base), handle (attach to door), door (attach to base), handle (attach to door), door (attach to base), handle (attach to door).\n\nThe part connectivity graph for the object is:\n```json\n{\"base\": [{\"door\": [{\"handle\": []}]}, {\"door\": [{\"handle\": []}]}, {\"door\": [{\"handle\": []}]}, {\"door\": [{\"handle\": []}]}]}\n```",

example_prompt_2 = "An image of a table that has two drawers and each drawer has a handle."
example_assistant_2 = "I recognize all the articulated parts in a table, they are: base, drawer (attach to base), handle (attach to drawer), drawer (attach to base), handle (attach to drawer).\n\nThe part connectivity graph for the object is:\n```json\n{\"base\": [{\"drawer\": [{\"handle\": []}]}, {\"drawer\": [{\"handle\": []}]}]}\n```",

example_prompt_3 = "An image of a refrigerator that has two doors and each door has a handle."
example_assistant_3 = "I recognize all the articulated parts in a refrigerator, they are: base, door (attach to base), handle (attach to door),  door (attach to base), handle (attach to door).\n\nThe part connectivity graph for the object is:\n```json\n{\"base\": [{\"door\": [{\"handle\": []}]}, {\"door\": [{\"handle\": []}]}]}\n```",

example_prompt_4 = "An image of a oven that has two door and each door has a handle."
example_assistant_4 = "I recognize all the articulated parts in a table, they are: base, door (attach to base), handle (attach to door), door (attach to base), handle (attach to door).\n\nThe part connectivity graph for the object is:\n```json\n{\"base\": [{\"door\": [{\"handle\": []}]}, {\"door\": [{\"handle\": []}]}]}\n```",

example_prompt_5 = "An image of a microwave that has one door with a handle and two knobs for control."
example_assistant_5 = "I recognize all the articulated parts in a microwave, they are: base, door (attach to base), handle (attach to door).\n\nThe part connectivity graph for the object is:\n```json\n{\"base\": [{\"door\": [{\"handle\": []}]}, {\"knob\": []}, {\"knob\": []}]}\n```",

example_prompt_6 = "An image of a washer that has one door and one knob for control."
example_assistant_6 = "I recognize all the articulated parts in a washer, they are: base, door (attach to base), knob (attach to base).\n\nThe part connectivity graph for the object is:\n```json\n{\"base\": [{\"door\": []}, {\"knob\": []}]}\n```",


examples = [
    {
        'prompt': example_prompt_1,
        'assistant': example_assistant_1
    },
    {
        'prompt': example_prompt_2,
        'assistant': example_assistant_2
    },
    {
        'prompt': example_prompt_3,
        'assistant': example_assistant_3
    },
    {
        'prompt': example_prompt_4,
        'assistant': example_assistant_4
    },
    {
        'prompt': example_prompt_5,
        'assistant': example_assistant_5
    },
    {
        'prompt': example_prompt_6,
        'assistant': example_assistant_6
    },
]
    