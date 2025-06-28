conda env: dino 


### Installation note
- make sure cuda driver is installed : nvidia-smi
- check nvcc --version match with cuda toolkit 
- cd usr/local/cuda 
- install cuda toolkit if not installed as setup will build package base on system cuda toolkit

- Follow below to avoid error
```bash
cd /content/GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn
sed -i 's/value.type()/value.scalar_type()/g' ms_deform_attn_cuda.cu
sed -i 's/value.scalar_type().is_cuda()/value.is_cuda()/g' ms_deform_attn_cuda.cu
```
- install only after cuda and edit MsDeformAttn is done 
!pip intall -r requirements.txt  
!pip install -q -e .  
!pip install -q roboflow dataclasses-json onemetric  


# Code changes
line 284 - prevent crash of scheduler as step mismatch 
```
steps_per_epoch = math.ceil(len(train_loader.dataset) / data_config.batch_size)
```
----

- add to examine batch with spike in loss
```python
def save_debug_batch(self, images, targets, captions, step):
    os.makedirs("debug_samples", exist_ok=True)
    for i, (img, target, cap) in enumerate(zip(images, targets, captions)):
        img_pil = F.to_pil_image(img.cpu())
        fig, ax = plt.subplots()
        ax.imshow(img_pil)
        for box, label in zip(target['boxes'], target['labels']):
            x0, y0, x1, y1 = box.cpu().numpy()
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       edgecolor='red', facecolor='none', linewidth=2))
            ax.text(x0, y0 - 5, str(label.item()), color='white', backgroundcolor='red')
        ax.set_title(f"Step {step} - Caption: {cap}")
        plt.axis('off')
        plt.savefig(f"debug_samples/step{step}_img{i}.png", bbox_inches='tight')
        plt.close()
```     

Token value : b7mvlOzf7a1MpohO7U589D1yThg_lFsB3hQshQ3q
Access Key ID: 10a967eb9d9bd4a502460fc6a7638cd7
Secret Access Key: c9803b24bd012d79f52882adecee1add7020640d8c99cfe9a03e27cae25ad872
https://21cccb913ff6d161fa6d058aee987513.r2.cloudflarestorage.com