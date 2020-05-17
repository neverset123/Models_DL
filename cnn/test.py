import os
import numpy as np
from pathlib import Path
save_dir='./'
kernel_size=[2,2]
#Path(os.path.join(save_dir, 'accuracy_loss', 'train_loss')).mkdir(parents=True, exist_ok=True)
save_dir=os.path.join(save_dir, 'accuracy_loss', 'train_loss')
print(save_dir)
np.save(save_dir, kernel_size)