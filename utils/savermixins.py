import os
import json
import torch
import imageio
import numpy as np

class SaverMixin():

    @property
    def save_dir(self):
        return self.hparams.save_dir
    
    def convert_format(self, data):
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, list):
            return [self.convert_format(d) for d in data]
        elif isinstance(data, dict):
            return {k: self.convert_format(v) for k, v in data.items()}
        else:
            raise TypeError('Data must be in type numpy.ndarray, torch.Tensor, list or dict, getting', type(data))
    
    def get_save_path(self, filename):
        save_path = os.path.join(self.save_dir, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return save_path
    
    def save_rgb_image(self, filename, img):
        imageio.imwrite(self.get_save_path(filename), img)
    
    def save_rgb_video(self, filename, stage='fit', filter=None):
        img_dir = os.path.join(self.logger.log_dir, 'images', stage)
      
        writer_graph = imageio.get_writer(os.path.join(img_dir, filename), fps=1)

        for file in sorted(os.listdir(img_dir)):
            if file.endswith('.png') and 'gt' not in file:
                if filter is not None:
                    if filter in file:
                        writer_graph.append_data(imageio.imread(os.path.join(img_dir, file)))
                else:
                    writer_graph.append_data(imageio.imread(os.path.join(img_dir, file)))

        writer_graph.close()


    
    def save_json(self, filename, data):
        save_path = self.get_save_path(filename)
        with open(save_path, 'w') as f:
            json.dump(data, f)
    
    