# Test Time Adaptation through externally guided Region Proposal


If you want to test the latest version, please download the `TTA_project_SRtesting.ipynb` file that had been modified in AWS.

(Deprecated) [CoLAB Notebook (with CMA-ES implementation) and Entropy map testing](https://colab.research.google.com/drive/1jK08Hx10QWX3SFNeiSVORKEqDppa6hvy#scrollTo=DDSzKgRUFsKG)

**Testing pictures:** credits @ [Pexel.com](https://www.pexels.com/it-it/) (CC-0)

## To do:
### Visualization
- [ ] flux diagram

### connection issues
- [x] Connect AWS notebook to imagenet-a bucket dataset (you need to insert "imagenet-a" inside of the img_root of dataloader while running main() )
- [ ] Insert classes list based on the "ReadMe" inside the imagenet-a dataset
- [ ] Connect output probabilities to text

### CMA-ES improvement
- [ ] Once CMA-ES obtains optimal generation for bounding box, gaussian noise must be injected to move the bounding boxes or alternatively train forms of early stopping

### MEMO improvement


